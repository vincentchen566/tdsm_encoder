import time, functools, torch, os, random, utils, fnmatch, psutil, argparse, math
from shower_features import shower_features
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
#from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
#import torch.multiprocessing as mp
#import wandb

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__() # inherits from pytorch nn class
        # Randomly sampled weights initialisation. Fixed during optimisation i.e. not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # Time information incorporated via Gaussian random feature encoding
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """Fully connected layer that reshapes outputs to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """Dense nn layer output must have same dimensions as input data:
            For point clouds: [batchsize, (dummy)nhits, (dummy)features]
        """
        return self.dense(x)[..., None]

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        """Initialise an encoder block:
        For 'class token' input (initially random):
            - Attention layer
            - Linear layer
            - GeLU activation
            - Linear layer
        
        For input:
            - Add class token block output
            - GeLU activation
            - Dropout regularisation layer
            - Linear layer
            - Add to original input

        Args:
        embed_dim: length of embedding
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # batch_first=True because normally in NLP the batch dimension would be the second dimension
        # In everything(?) else it is the first dimension so this flag is set to true to match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.fc1_cls = nn.Linear(embed_dim, hidden)
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.act_dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        # Stash original embedded input
        residual = x.clone()

        # Multiheaded self-attention but replacing queries of all input examples with a single mean field approximator
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x_cls = self.act(self.fc1_cls(x_cls))
        x_cls = self.act_dropout(x_cls)
        x_cls = self.fc2(x_cls)

        # Add mean field approximation to input embedding (acts like a bias)
        x = x + x_cls.clone()
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        
        # Add to original input embedding
        x = x + residual
        
        return x

class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, dropout_gen, marginal_prob_std, **kwargs):
        '''Transformer encoder model
        Arguments:
        n_dim = number of features
        l_dim_gen = dimensionality to embed input
        hidden_gen = dimensionaliy of hidden layer
        num_layers_gen = number of encoder blocks
        heads_gen = number of parallel attention heads to use
        dropout_gen = regularising layer
        marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
        '''
        super().__init__()

        # Embedding: size of input (n_dim) features -> size of output (l_dim_gen)
        self.embed = nn.Linear(n_dim, l_dim_gen)
        
        # Seperate embedding for (time/incident energy) conditional inputs (small NN with fixed weights)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=64), nn.Linear(64, 64))
        # Boils embedding down to single value
        self.dense1 = Dense(64, 1)

        # Module list of encoder blocks
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=l_dim_gen,
                    num_heads=heads_gen,
                    hidden=hidden_gen,
                    dropout=dropout_gen,
                )
                for i in range(num_layers_gen)
            ]
        )
        
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(l_dim_gen, n_dim)

        # token simply has same dimension as input feature embedding
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)
        self.cls_token = nn.Parameter(torch.ones(1, 1, l_dim_gen), requires_grad=True)
        self.act = nn.GELU()

        # Swish activation function
        self.act_sig = lambda x: x * torch.sigmoid(x)
        # Standard deviation of SDE
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, e, mask=None):
        # Embed 4-vector input
        x = self.embed(x)
        
        # Add time embedding
        embed_t_ = self.act_sig( self.embed_t(t) )
        # Now need to get dimensions right
        x += self.dense1(embed_t_).clone()

        # Add incident particle energy embedding
        embed_e_ = self.embed_t(e)
        embed_e_ = self.act_sig(embed_e_)
        # Now need to get dimensions right
        x += self.dense1(embed_e_).clone()

        # 'class' token (mean field)
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        
        # Feed input embeddings into encoder block
        for layer in self.encoder:
            # Each encoder block takes previous blocks output as input
            x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
            # Should concatenate with the time embedding after each block?

        mean_ , std_ = self.marginal_prob_std(x,t)
        std_1d = std_[:, None, None]
        output = self.out(mean_) / std_1d
        return output 

def loss_fn(model, x, incident_energies, marginal_prob_std , eps=1e-5, device='cpu'):
    """The loss function for training score-based generative models
    Uses the weighted sum of Denoising Score matching objectives
    Denoising score matching
    - Perturbs data points with pre-defined noise distribution
    - Uses score matching objective to estimate the score of the perturbed data distribution
    - Perturbation avoids need to calculate trace of Jacobian of model output

    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model
        x: A mini-batch of training data
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel
        eps: A tolerance value for numerical stability
    """
    
    # Generate padding mask for 0 padded entries or equivalent value if energies have been transformed
    padding_mask = (x!=-20).type(torch.int)

    # Collapse hit 4-vector dimension so mask simply masks out entire hit
    padding_mask = torch.min(padding_mask,2)[0]

    # Tensor of randomised conditional variable 'time' steps
    random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps
    
    # Tensor of conditional variable incident energies 
    incident_energies = torch.squeeze(incident_energies,-1)
    
    # Matrix of noise
    z = torch.randn_like(x)
    z = z.to(device)
    
    # Sample from standard deviation of noise
    mean_, std_ = marginal_prob_std(x,random_t)
    #std_.to(device)
    
    # Add noise to input
    perturbed_x = x + z * std_[:, None, None]
    
    # Evaluate model
    model_output = model(perturbed_x, random_t, incident_energies, mask=padding_mask)
    
    losses = (model_output*std_[:,None,None] + z)**2
    
    # Collect loss
    batch_loss = torch.sum( losses, dim=(0,1,2))
    return batch_loss

def  pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_energies, sampled_hits, batch_size=1, snr=0.16, device='cuda', eps=1e-3):
    ''' Generate samples from score based models with Predictor-Corrector method
        Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the std of the perturbation kernel
        diffusion_coeff: A function that gives the diffusion coefficient 
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

        Returns:
            samples
    '''
    num_steps=100
    t = torch.ones(batch_size, device=device)
    gen_n_hits = int(sampled_hits.item())
    init_x = torch.randn(batch_size, gen_n_hits, 4, device=device)
    mean_,std_ =  marginal_prob_std(init_x,t)
    std_.to(device)
    init_x = init_x*std_[:,None,None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0]-time_steps[1]
    x = init_x
    with torch.no_grad():
         for time_step in time_steps:

            print(f"Sampler step: {time_step:.4f}")
            batch_time_step = torch.ones(batch_size, device=init_x.device) * time_step
            
            # matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            alpha = torch.ones_like(torch.tensor(time_step))
            
            #for _ in range(1):
            # Corrector step (Langevin MCMC)
            # First calculate Langevin step size using the predicted scores
            grad = score_model(x, batch_time_step, sampled_energies)
            #noise = np.prod(x.shape[1:])
            noise = torch.randn_like(x)
            
            # Vector norm (sqrt sum squares)
            # Take the mean value
            # of the vector norm (sqrt sum squares)
            # of the flattened scores for e,x,y,z
            flattened_scores = grad.reshape(grad.shape[0], -1)
            grad_norm = torch.linalg.norm( flattened_scores, dim=-1 ).mean()
            flattened_noise = noise.reshape(noise.shape[0],-1)
            noise_norm = torch.linalg.norm( flattened_noise, dim=-1 ).mean()
            langevin_step_size =  (snr * noise_norm / grad_norm)**2 * 2 * alpha
            
            # Implement iteration rule
            x_mean = x + langevin_step_size * grad
            x = x_mean + torch.sqrt(2 * langevin_step_size) * noise
        
            # Euler-Maruyama predictor step
            drift, diff = diffusion_coeff(x,batch_time_step)
            x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies) * step_size
            x = x_mean + torch.sqrt(diff**2 * step_size)[:, None, None] * torch.randn_like(x)
            
    # Do not include noise in last step
    return x_mean

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-o','--output',dest='output_path', help='Path to output directory', default='', type=str)
    argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that run: evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
    args = argparser.parse_args()
    workingdir = args.output_path
    switches_ = int('0b'+args.switches,2)
    switches_str = bin(int('0b'+args.switches,2))
    trigger = 0b0001
    print(f'switches trigger: {switches_str}')
    if switches_ & trigger:
        print('input_feature_plots = ON')
    if switches_>>1 & trigger:
        print('training_switch = ON')
    if switches_>>2 & trigger:
        print('sampling_switch = ON')
    if switches_>>3 & trigger:
        print('evaluation_plots_switch = ON')

    print('torch version: ', torch.__version__)

    #global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    
    # Useful when debugging gradient issues
    torch.autograd.set_detect_anomaly(True)

    print('Working directory: ' , workingdir)
    
    sigma = 25.0
    vesde = utils.VESDE(device=device)
    new_marginal_prob_std_fn = functools.partial(vesde.marginal_prob)
    new_diffusion_coeff_fn = functools.partial(vesde.sde)

    # List of training input files
    training_file_path = '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/datasets/'
    files_list_ = []
    for filename in os.listdir(training_file_path):
        #if fnmatch.fnmatch(filename, 'padded_dataset_2_1_graph_0.pt'):
        if fnmatch.fnmatch(filename, 'toy_model.pt'):
            files_list_.append(os.path.join(training_file_path,filename))

    if switches_ & trigger:
        files_list_ = files_list_[:1]
        for filename in files_list_:
            print('filename: ', filename)
            custom_data = utils.cloud_dataset(filename)#, transform=utils.rescale_energies(), transform_y=utils.rescale_conditional())
            output_directory = workingdir+'/feature_plots_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Load input data
            point_clouds_loader = DataLoader(custom_data,batch_size=1,shuffle=False)
            all_x = []
            all_y = []
            all_z = []
            all_e = []
            all_incident_e = []
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                all_e.append(shower_data[0][:,0])
                all_x.append(shower_data[0][:,1])
                all_y.append(shower_data[0][:,2])
                all_z.append(shower_data[0][:,3])
                all_incident_e.append(incident_energies.item())
                
            plot_e = np.concatenate(all_e)
            plot_e = np.ma.masked_where(plot_e==-20,plot_e)
            plot_x = np.concatenate(all_x)
            plot_x = np.ma.masked_where(plot_x==-20,plot_x)
            plot_y = np.concatenate(all_y)
            plot_y = np.ma.masked_where(plot_y==-20,plot_y)
            plot_z = np.concatenate(all_z)
            plot_z = np.ma.masked_where(plot_z==-20,plot_z)

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit energy')
            plt.hist(plot_e, 50, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_energies.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit x position')
            plt.hist(plot_x, 9, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_x.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit y position')
            plt.hist(plot_y, 16, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_y.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Hit z position')
            plt.hist(plot_z, 45, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_z.png')

            fig, ax = plt.subplots(ncols=1, figsize=(10,10))
            plt.title('')
            plt.ylabel('# entries')
            plt.xlabel('Incident energy')
            plt.hist(all_incident_e, 30, label='Geant4')
            plt.legend(loc='upper right')
            fig.savefig(output_directory+'hit_incident_e.png')

    #### Training ####
    if switches_>>1 & trigger:
        output_directory = workingdir+'/training_rescaled_energies_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        print('Output directory: ', output_directory)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        batch_size = 32
        lr = 0.0001
        n_epochs = 500
        '''if tracking_ == True:
            if sweep_ == True:
                run_ = wandb.init()
                print('Running sweep!')
                # note that we define values from `wandb.config` instead of defining hard values
                lr  =  wandb.config.lr
                batch_size = wandb.config.batch_size
                n_epochs = wandb.config.epochs
            else:
                run_ = wandb.init()
                print('Tracking!')
                lr  =  wandb.config.lr
                batch_size = wandb.config.batch_size
                n_epochs = wandb.config.epochs'''

        
        model=Gen(4, 200, 128, 3, 1, 0, marginal_prob_std=new_marginal_prob_std_fn)
        print('model: ', model)
        if torch.cuda.device_count() > 1:
            print(f'Lets us: {torch.cuda.device_count()} GPUs!')
            model = nn.DataParallel(model)

        #for para_ in model.parameters():
        #    print('model parameters: ', para_)

        # Optimiser needs to know model parameters for to optimise
        optimiser = Adam(model.parameters(),lr=lr)
        
        av_training_losses_per_epoch = []
        av_testing_losses_per_epoch = []
        for epoch in range(0,n_epochs):
            print(f"epoch: {epoch}")
            # Create/clear per epoch variables
            cumulative_epoch_loss = 0.
            cumulative_test_epoch_loss = 0.

            file_counter = 0
            n_training_showers = 0
            n_testing_showers = 0
            # Load files
            files_list_ = [files_list_[f] for f in range(0,1)]
            # For debugging purposes
            files_list_ = files_list_[:1]
            
            for filename in files_list_:
                file_counter+=1
                
                # Resident set size memory (non-swap physical memory process has used)
                #process = psutil.Process(os.getpid())
                #print('Memory usage of current process 0 [MB]: ', process.memory_info().rss/1000000)

                # Rescaling now done in padding package
                #custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), device=device)
                custom_data = utils.cloud_dataset(filename, device=device)
                print(f'{len(custom_data)} showers in file')
                
                train_size = int(0.9 * len(custom_data.data))
                test_size = len(custom_data.data) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])

                n_training_showers+=train_size
                n_testing_showers+=test_size
            
                # Load clouds for each epoch of data dataloaders length will be the number of batches
                shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                print(f'train_dataset: {len(train_dataset)} showers, batch size: {len(shower_loader_train.dataset)} showers')
                
                # Load a shower for training
                for i, (shower_data,incident_energies) in enumerate(shower_loader_train,0):
                    print(f'batch {i} of length {len(shower_data)}')
                    
                    # Move model to device and set dtype as same as data (note torch.double works on both CPU and GPU)
                    model.to(device, shower_data.dtype)
                    shower_data.to(device)
                    incident_energies.to(device)

                    if len(shower_data) < 1:
                        print('Very few hits in shower: ', len(shower_data))
                        continue
                    
                    loss = loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device)
                    # Average cloud loss in batch to backpropagate (could also use sum)
                    batch_loss_averaged = loss/len(shower_data)
                    cumulative_epoch_loss+=batch_loss_averaged.item()*batch_size
                    # Zero any gradients from previous steps
                    optimiser.zero_grad()
                    # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                    batch_loss_averaged.backward(retain_graph=True)
                    # Update value of x += -lr * x.grad
                    optimiser.step()
            
                # Testing on subset of file
                for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
                    with torch.no_grad():
                        test_loss = loss_fn(model, shower_data, incident_energies, new_marginal_prob_std_fn, device=device)
                        test_batch_loss_averaged = test_loss/len(shower_data)
                        cumulative_test_epoch_loss+=test_batch_loss_averaged.item()*batch_size
            
            #wandb.log({"training_loss": cumulative_epoch_loss/n_training_showers,
            #           "testing_loss": cumulative_test_epoch_loss/n_testing_showers})

            # Add the batch size just used to the total number of clouds
            av_training_losses_per_epoch.append(cumulative_epoch_loss/n_training_showers)
            av_testing_losses_per_epoch.append(cumulative_test_epoch_loss/n_testing_showers)
            print(f'End-of-epoch: average train loss = {av_training_losses_per_epoch}, average test loss = {av_testing_losses_per_epoch}')
            # Save checkpoint file after each epoch
            torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')

        av_training_losses_per_epoch = av_training_losses_per_epoch
        av_testing_losses_per_epoch = av_testing_losses_per_epoch
        print('Training losses : ', av_training_losses_per_epoch)
        print('Testing losses : ', av_testing_losses_per_epoch)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.plot(av_training_losses_per_epoch, label='training')
        plt.plot(av_testing_losses_per_epoch, label='testing')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(output_directory+'loss_v_epoch.png')
        #wandb.finish()
    
    #### Sampling ####
    if switches_>>2 & trigger:    
        output_directory = workingdir+'/sampling_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        sample_batch_size = 10
        model=Gen(4, 200, 128, 3, 1, 0, marginal_prob_std=new_marginal_prob_std_fn)
        load_name = os.path.join(workingdir,'training_rescaled_energies_20230410_1159_output/ckpt_tmp_499.pth')

        model.load_state_dict(torch.load(load_name, map_location=device))
        model.to(device)
        samples_ = []
        in_energies = []
        hits_lengths = []
        sampled_energies = []
        # Use clouds from a sample of random files to generate a distribution of # hits
        for idx_ in random.sample( range(0,len(files_list_)), 1):
            custom_data = utils.cloud_dataset(files_list_[idx_], device=device)
            point_clouds_loader = DataLoader(custom_data, batch_size=50, shuffle=True)
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                # Get nhits for examples in input file
                hits_lengths.append( [len(shower_data[0])] )
                # Get incident energies from input file
                sampled_energies.append( incident_energies.tolist() )
        
        # Stack
        e_h_comb = list(zip(sampled_energies, hits_lengths))
        print(f'{len(sampled_energies)} sampled_energies')
        for s_ in range(0,sample_batch_size):
            print(f'Generating point cloud: {s_}')
            # Generate random number to sample an example energy and nhits
            idx = np.random.randint(len(e_h_comb), size=1)[0]
            sampled_e_h_ = e_h_comb[idx]
            in_energies.append(sampled_e_h_[0])
            # Energies and hits to pass to sampler
            sampled_energies = torch.tensor(sampled_e_h_[0])
            sampled_hits = torch.tensor(sampled_e_h_[1])
            # Initiate sampler
            sampler = pc_sampler
            # Generate a sample of point clouds
            samples = sampler(model, new_marginal_prob_std_fn, new_diffusion_coeff_fn, sampled_energies, sampled_hits, 50, device=device)
            samples_.append(samples)
        print('samples_: ', samples_)
        print('np.squeeze(samples_): ', np.squeeze(samples_))
        torch.save([samples_,torch.tensor(in_energies)], output_directory+'generated_samples.pt')

    #### Evaluation plots ####
    if switches_>>3 & trigger:
        output_directory = workingdir+'/evaluation_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        batch_size=1
        # Initialise clouds with detector structure
        layer_list = []
        xbins_list = []
        ybins_list = []
        for idx_ in random.sample(range(0,len(files_list_)),1):
            # Load input data (just need example file for now)
            print(f'GEANT file: {files_list_[idx_]}')
            custom_data = utils.cloud_dataset(files_list_[idx_],device=device)
            point_clouds_loader = DataLoader(custom_data,batch_size=batch_size,shuffle=False)

            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                zlayers_ = shower_data[:,:,3].tolist()[0]
                xbins_ = shower_data[:,:,1].tolist()[0]
                ybins_ = shower_data[:,:,2].tolist()[0]
                for z in zlayers_:
                    layer_list.append( z )
                for x in xbins_:
                    xbins_list.append( x )
                for y in ybins_:
                    ybins_list.append( y )
        
        layer_set = set([z for z in layer_list])
        print('layer_set: ', layer_set)

        for idx_ in random.sample(range(0,len(files_list_)),1):
            total_shower_e_GEANT = []
            total_shower_nhits_GEANT = []
            incident_e_GEANT = []
            all_hit_e_GEANT = []
            all_hit_x_GEANT = []
            all_hit_y_GEANT = []
            all_hit_z_GEANT = []
            transformed_total_shower_e_GEANT = []
            transformed_incident_e_GEANT = []
            transformed_all_hit_e_GEANT = []
            transformed_all_hit_x_GEANT = []
            transformed_all_hit_y_GEANT = []
            transformed_all_hit_z_GEANT = []
            total_shower_e_per_layer_GEANT = {}
            total_shower_nhits_per_layer_GEANT = {}
            for layer_ in range(0,len(layer_set)):
                total_shower_e_per_layer_GEANT[layer_] = []
                total_shower_nhits_per_layer_GEANT[layer_] = []
            
            custom_data = utils.cloud_dataset(files_list_[idx_], device=device)
            point_clouds_loader = DataLoader(custom_data,batch_size=1,shuffle=False)
            
            # Load each cloud and calculate desired quantity
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
                #print(f'shower {i} \n {shower_data}\n')
                #print(f'shower {i} \n {shower_data[0]}')
                #print(f'incident_energies {incident_energies.shape}: ', incident_energies)
                if i>500:
                    continue
                incident_e_GEANT.append(incident_energies)
                all_hit_e_GEANT.append(shower_data[0][:,0])
                all_hit_x_GEANT.append(shower_data[0][:,1])
                all_hit_y_GEANT.append(shower_data[0][:,2])
                all_hit_z_GEANT.append(shower_data[0][:,3])
                total_shower_e_GEANT.append(sum(shower_data[0][:,0]))
                total_shower_nhits_GEANT.append(len(shower_data[0][:,0]))
                rescaler = utils.rescale_energies()
                shower_data_transformed = rescaler(shower_data[0],incident_energies)
                rescaler_cond = utils.rescale_conditional()
                incident_energies_transformed = rescaler_cond(incident_energies,custom_data.min_y,custom_data.max_y)
                transformed_incident_e_GEANT.append(incident_energies_transformed)
                transformed_all_hit_e_GEANT.append(shower_data_transformed[:,0].tolist())
                transformed_all_hit_x_GEANT.append(shower_data_transformed[:,1].tolist())
                transformed_all_hit_y_GEANT.append(shower_data_transformed[:,2].tolist())
                transformed_all_hit_z_GEANT.append(shower_data_transformed[:,3].tolist())
                transformed_total_shower_e_GEANT.append(sum(shower_data_transformed[:,0]))
                
                # append sum of each clouds deposited energy/hits per layer to lists
                '''for layer_ in cloud_features.layer_set:
                    input_layers_sum_e[layer_].append(cloud_features.layers_sum_e.get(layer_)[0])
                    input_layers_nhits[layer_].append(cloud_features.layers_nhits.get(layer_)[0])'''
        
        # Get averages for all clouds in dataset
        '''input_layer_e_averages = []
        input_layer_n_hits_averages = []
        for layer_ in cloud_features.layer_set:
            sum_e_for_layer_across_clouds = sum(input_layers_sum_e.get(layer_))
            sum_hits_for_layer_across_clouds = sum(input_layers_nhits.get(layer_))
            n_clouds = len(input_layers_sum_e.get(layer_))
            mean_e_for_layer = sum_e_for_layer_across_clouds/n_clouds
            mean_n_hits_for_layer = sum_hits_for_layer_across_clouds/n_clouds
            input_layer_e_averages.append( mean_e_for_layer )
            input_layer_n_hits_averages.append( mean_n_hits_for_layer )'''
        
        # Load generated image file
        test_ge_filename = 'sampling_20230411_1500_output/generated_samples.pt'
        
        custom_gendata = utils.cloud_dataset(test_ge_filename, device=device)
        gen_point_shower_loader = DataLoader(custom_gendata,batch_size=1,shuffle=False)

        total_shower_e_GEN = []
        total_shower_nhits_GEN = []
        incident_e_GEN = []
        all_hit_e_GEN = []
        all_hit_x_GEN = []
        all_hit_y_GEN = []
        all_hit_z_GEN = []
        untransformed_all_hit_e_GEN = []

        total_shower_e_per_layer_GEN = {}
        total_shower_nhits_per_layer_GEN = {}       
        
        for layer_ in range(0,len(layer_set)):
                total_shower_e_per_layer_GEN[layer_] = []
                total_shower_nhits_per_layer_GEN[layer_] = []
        print(f'Generated {len(gen_point_shower_loader)} showers')
        for i, (shower_data,incident_energies) in enumerate(gen_point_shower_loader,0):
            shower_data = torch.squeeze(shower_data)
            print('incident_energies: ', incident_energies)
            print(f'Batch {i} shape = {shower_data.shape} ')
            if i>450:
                    continue
            
            batch_hits_e = np.concatenate(shower_data[:,:,0].tolist())
            batch_hits_x = np.concatenate(shower_data[:,:,1].tolist())
            batch_hits_y = np.concatenate(shower_data[:,:,2].tolist())
            batch_hits_z = np.concatenate(shower_data[:,:,3].tolist())
            total_e_per_shower =[]
            total_hits_per_shower = []
            for shower in shower_data[:]:
                sum_e = sum(shower[:,0])
                len_hits = len(shower[:,0])
                total_e_per_shower.append(sum_e)
                total_hits_per_shower.append(len_hits)

            incident_e_GEN.append(incident_energies)
            all_hit_e_GEN.append(batch_hits_e)
            all_hit_x_GEN.append(batch_hits_x)
            all_hit_y_GEN.append(batch_hits_y)
            all_hit_z_GEN.append(batch_hits_z)
            total_shower_e_GEN.append(total_e_per_shower)
            total_shower_nhits_GEN.append(total_hits_per_shower)

            unscaler = utils.unscale_energies()
            incident_energies = torch.squeeze(incident_energies)
            shower_n = 0
            for shower_ in shower_data:
                shower_data_transformed = unscaler(shower_,incident_energies[shower_n])
                untransformed_all_hit_e_GEN.append(shower_data_transformed[:,0].tolist())
                shower_n+=1
            
            '''
            # append each clouds deposited energy per layer to lists
            for layer_ in gen_cloud_features.layer_set:
                gen_layers_sum_e[layer_].append(gen_cloud_features.layers_sum_e.get(layer_)[0])
                gen_layers_nhits[layer_].append(gen_cloud_features.layers_nhits.get(layer_)[0])
            '''

        '''gen_layer_e_averages = []
        gen_layer_n_hits_averages = []
        for layer_ in cloud_features.layer_set:
            sum_e_for_layer_across_clouds = sum(gen_layers_sum_e.get(layer_))
            sum_hits_for_layer_across_clouds = sum(gen_layers_nhits.get(layer_))
            n_clouds = len(gen_layers_sum_e.get(layer_))
            mean_e_for_layer = sum_e_for_layer_across_clouds/n_clouds
            mean_n_hits_for_layer = sum_hits_for_layer_across_clouds/n_clouds
            gen_layer_e_averages.append(mean_e_for_layer)
            gen_layer_n_hits_averages.append(mean_n_hits_for_layer)'''

        geant_hit_energy = np.concatenate(all_hit_e_GEANT)
        transformed_geant_hit_energy = np.concatenate(transformed_all_hit_e_GEANT)
        gen_hit_energy = np.concatenate(all_hit_e_GEN)
        untransformed_gen_hit_energy = np.concatenate(untransformed_all_hit_e_GEN)
        
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Hit energies')
        plt.ylabel('Entries')
        plt.xlabel('Hit Energy [GeV]')
        plt.hist(geant_hit_energy, 20,range=(0,4), label='Geant4', alpha=0.5)
        plt.hist(untransformed_gen_hit_energy, 20,range=(0,4), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_hit_GEANT.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Transformed hit energies')
        plt.ylabel('Entries')
        plt.xlabel('Hit Energy')
        plt.hist(transformed_geant_hit_energy, 20, range=(-14,-4), label='Geant4', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_hit_transformed_GEANT.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Hit energies')
        plt.ylabel('Entries')
        plt.xlabel('Hit Energy')
        plt.hist(gen_hit_energy, 20, range=(-14,-4), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_hit_GEN.png')
        
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Transformed hit energies')
        plt.ylabel('Entries')
        plt.xlabel('Hit Energy')
        plt.hist(transformed_geant_hit_energy, 20, range=(-14,-4), label='Geant4', alpha=0.5)
        plt.hist(gen_hit_energy, 20, range=(-14,-4), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_hit_transformed_overlay.png')

        geant_hit_x = np.concatenate(all_hit_x_GEANT)
        gen_hit_x = np.concatenate(all_hit_x_GEN)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Hit x')
        plt.ylabel('Entries')
        plt.xlabel('Hit x')
        plt.hist(geant_hit_x, 20, range=(-8,8), label='Geant4', alpha=0.5)
        plt.hist(gen_hit_x, 20, range=(-8,8), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_x_per_hit.png')

        geant_hit_y = np.concatenate(all_hit_y_GEANT)
        gen_hit_y = np.concatenate(all_hit_y_GEN)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Hit y')
        plt.ylabel('Entries')
        plt.xlabel('Hit y')
        plt.hist(geant_hit_y, 20, range=(-8,8), label='Geant4', alpha=0.5)
        plt.hist(gen_hit_y, 20, range=(-8,8), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_y_per_hit.png')

        geant_hit_z = np.concatenate(all_hit_z_GEANT)
        gen_hit_z = np.concatenate(all_hit_z_GEN)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Hit z')
        plt.ylabel('Entries')
        plt.xlabel('Hit z')
        plt.hist(geant_hit_z, 5, range=(-1,4), label='Geant4', alpha=0.5)
        plt.hist(gen_hit_z, 5, range=(-1,4), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_z_per_hit.png')

        geant_total_shower_e = torch.stack(total_shower_e_GEANT,dim=0)
        transformed_geant_total_shower_e = torch.stack(transformed_total_shower_e_GEANT,dim=0)
        transformed_gen_total_shower_e = np.concatenate(total_shower_e_GEN)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Total deposited energy per shower')
        plt.ylabel('Entries')
        plt.xlabel('Energy [GeV]')
        plt.yscale('log')
        plt.hist(geant_total_shower_e.tolist(), 10, label='Geant4', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_shower_GEANT.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Transformed total deposited energy per shower')
        plt.ylabel('Entries')
        plt.xlabel('Energy [GeV]')
        plt.hist(transformed_geant_total_shower_e.tolist(), 50, label='GEANT', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'transformed_deposited_energy_per_shower_GEANT.png')    

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Transformed total deposited energy per shower')
        plt.ylabel('Entries')
        plt.xlabel('Energy [GeV]')
        plt.hist(transformed_gen_total_shower_e, 50, label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_shower_GEN.png')    

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Transformed total deposited energy per shower')
        plt.ylabel('Entries')
        plt.xlabel('Energy')
        plt.hist(transformed_geant_total_shower_e.tolist(), 50, range=(-800,-400), label='GEANT', alpha=0.5)
        plt.hist(transformed_gen_total_shower_e, 50, range=(-800,-400), label='Gen', alpha=0.5)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_shower_overlay.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('# hits per shower (includes null hits)')
        plt.ylabel('Entries')
        plt.xlabel('# hits')
        plt.hist(total_shower_nhits_GEANT, 10, label='Geant4', alpha=0.6)
        plt.hist(np.concatenate(total_shower_nhits_GEN), 10, label='Gen', alpha=0.6)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'nhits_per_shower.png')


        '''fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Average deposited energy per layer')
        plt.ylabel('Energy [GeV]')
        plt.xlabel('Layer number')
        plt.plot([x/1000 for x in input_layer_e_averages], label='Geant4')
        plt.plot([x/1000 for x in gen_layer_e_averages], label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'avE_per_layer.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Average nhits per shower')
        plt.ylabel('# hits')
        plt.xlabel('Layer number')
        plt.plot(input_layer_n_hits_averages, label='Geant4')
        plt.plot(gen_layer_n_hits_averages, label='Gen')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'avHits_per_layer.png')    

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Total # hits (per shower) [Sanity check]')
        plt.ylabel('Entries')
        plt.xlabel('# hits')
        plt.hist(input_total_h_per_cloud, 10, range=(0,5000), label='Geant4', alpha=0.6)
        plt.hist(gen_total_h_per_cloud, 10, range=(0,5000), label='Gen', alpha=0.6)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'nhits_per_cloud.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('Incident Energies [Sanity check]')
        plt.ylabel('Entries')
        plt.xlabel('Energy [GeV]')
        plt.yscale('log')
        plt.hist([x/1000 for x in input_incident_e], 20, range=(0,1100), label='Geant4', alpha=0.6)
        plt.hist([x/1000 for x in gen_incident_e], 20, range=(0,1100), label='Gen', alpha=0.6)
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'incident_energies.png')'''


if __name__=='__main__':
    start = time.time()
    '''global tracking_
    global sweep_
    tracking_ = False
    sweep_ = False
    # Start sweep job.
    if tracking_:
            if sweep_:
                # Define sweep config
                sweep_configuration = {
                    'method': 'random',
                    'name': 'sweep',
                    'metric': {'goal': 'maximize', 'name': 'val_acc'},
                    'parameters': 
                    {
                        'batch_size': {'values': [50, 100, 150]},
                        'epochs': {'values': [5, 10, 15]},
                        'lr': {'max': 0.001, 'min': 0.00001},
                    }
                }
                sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
                wandb.agent(sweep_id, function=main, count=4)
            else:
                # start a new wandb run to track this script
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="trans_tdsm",
                    # track hyperparameters and run metadata
                    config={
                    "architecture": "encoder",
                    "dataset": "calochallenge_2",
                    "batch_size": 10,
                    "epochs": 10,
                    "lr": 0.0001,
                    }
                )
                main()
    else:
        main()'''
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))