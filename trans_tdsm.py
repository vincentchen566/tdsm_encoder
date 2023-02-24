import time, functools, torch, os, random, utils
from CloudFeatures import CloudFeatures
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as transforms
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

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
        super().__init__()

        # Embedding: size of input (n_dim) features -> size of output (l_dim_gen)
        self.embed = nn.Linear(n_dim, l_dim_gen)

        # Seperate time embedding (small NN with fixed weights)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=64), nn.Linear(64, 64))
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)
        self.act = nn.GELU()

        # Swish activation function
        self.act_sig = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, e, mask=None):
        # Embed poisitional input
        x = self.embed(x)
        
        # Add time embedding
        embed_t_ = self.act_sig( self.embed_t(t) )
        # Now need to get dimensions right
        x += self.dense1(embed_t_).clone()

        # Add energy embedding
        embed_e_ = self.act_sig( self.embed_t(e) )
        # Now need to get dimensions right
        x += self.dense1(embed_e_).clone()

        # 'class' token (mean field)
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        
        # Feed input embeddings into encoder block
        for layer in self.encoder:
            # Each encoder block takes previous blocks output as input
            x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
            # Should concatenate with the time embedding after each block?

        return self.out(x) / self.marginal_prob_std(t)[:, None, None]

"""Set up the SDE"""
def marginal_prob_std(t, sigma):
    """ 
    Choosing the SDE: 
        dx = sigma^t dw
        t in [0,1]
    Compute the standard deviation of: p_{0t}(x(t) | x(0))
        Args:
    t: A vector of time steps taken as random numbers sampled from uniform distribution [0,1)
    sigma: The sigma in our SDE which we set in the code
  
    Returns:
        The standard deviation.
    """    
    t = t.clone().detach()
    std = torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma)) 
    return std

def loss_fn(model, x, injection_energy, marginal_prob_std , eps=1e-5):
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
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    injection_energy = torch.squeeze(injection_energy,-1)
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None]
    model_output = model(perturbed_x, random_t, injection_energy)
    cloud_loss = torch.sum( (model_output*std + z)**2, dim=(1,2))
    return cloud_loss

def  pc_sampler(score_model, marginal_prob_std, diffusion_coeff, sampled_energies, batch_size=1, snr=0.16, device='cuda', eps=1e-3):
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
    num_steps=500
    t = torch.ones(batch_size, device=device)
    # Currently setting to some number of hits
    gen_n_hits = 289
    init_x = torch.randn(batch_size, gen_n_hits, 4, device=device) * marginal_prob_std(t)[:,None,None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0]-time_steps[1]
    x = init_x
    with torch.no_grad():
         for time_step in time_steps:
            print(f"Sampler step: {time_step:.4f}")
            batch_time_step = torch.ones(batch_size,device=device) * time_step
            
            # Sneaky bug fix (matrix multiplication in GaussianFourier projection doesnt like float64s)
            sampled_energies = sampled_energies.to(torch.float32)
            
            # Corrector step (Langevin MCMC)
            # First calculate Langevin step size
            grad = score_model(x, batch_time_step, sampled_energies)
            grad_norm = torch.norm( grad.reshape(grad.shape[0], -1), dim=-1 ).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            
            # Implement iteration rule
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Euler-Maruyama predictor step
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None] * torch.randn_like(x)

    # Do not include noise in last step
    return x_mean

def diffusion_coeff(t, sigma=25.0):
    """Compute the diffusion coefficient of our SDE
    Args:
        t: A vector of time steps
        sigma: from the SDE
    Returns:
    Vector of diffusion coefficients
    """
    return torch.tensor(sigma**t, device=device)

def main():
    
    print('torch version: ', torch.__version__)
    workingdir = os.path.abspath('.')
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    # Useful when debugging gradient issues
    #torch.autograd.set_detect_anomaly(True)

    training_switch = 0
    testing_switch = 1
    plotting_switch = 0

    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    filename = '/eos/user/t/tihsu/SWAN_projects/homepage/datasets/graph/dataset_2_1_graph_0.pt'
    loaded_file = torch.load(filename)
    point_clouds = loaded_file[0]
    print(f'Loading {len(point_clouds)} point clouds from file {filename}')
    energies = loaded_file[1]
    custom_data = utils.cloud_dataset(point_clouds, energies)
    load_n_clouds = 1

    if training_switch:
        lr = 1e-4
        n_epochs = 30
        av_losses_per_epoch = []
        output_directory = workingdir+'/training_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Size of the last dimension of the input must match the input to the embedding layer
        # First arg = number of features
        model=Gen(4, 20, 128, 3, 1, 0, marginal_prob_std=marginal_prob_std_fn)
        #for para_ in model.parameters():
        #    print('model parameters: ', para_)

        # Optimiser needs to know model parameters for to optimise
        optimiser = Adam(model.parameters(),lr=lr)
        
        batch_size = 200
        for epoch in range(0,n_epochs):
            # Load clouds for each epoch of data dataloaders length will be the number of batches
            #point_clouds_loader = DataLoader(point_clouds,batch_size=load_n_clouds,shuffle=True)
            #energies_loader = DataLoader(energies,batch_size=load_n_clouds,shuffle=True)
            point_clouds_loader = DataLoader(custom_data,batch_size=load_n_clouds,shuffle=False)
            print(f"epoch: {epoch}")
            cumulative_epoch_loss = 0.
            cloud_batch_losses = []
            cloud_counter = 0
            batch_counter = 0
            # Load a cloud
            for i, (cloud_data,injection_energy) in enumerate(point_clouds_loader,0):
                #if batch_counter>5:
                #    print(f'Done 5 batches, move to next epoch')
                #    break
                if i%100 == 0: print(f"Cloud: {i}")
                cloud_counter+=1
                
                if len(cloud_data.x) < 1:
                    print('Very few points in cloud: ', cloud_data.x)
                    continue
                # Adds batch dimension to front of data (currently making batches of 1 cloud manually)
                input_data = torch.unsqueeze(cloud_data.x, 0)

                # Calculate loss for for individual clouds
                cloud_loss = loss_fn(model, input_data, injection_energy, marginal_prob_std_fn)
                
                # Collect losses of clouds in batch
                cloud_batch_losses.append( cloud_loss )
                
                # If # clouds reaches batch_size
                if i%batch_size == 0 and i>0:
                    batch_counter+=1
                    print(f'Batch: {batch_counter} (cloud: {i})')
                    # Average cloud loss in batch to backpropagate (could also use sum)
                    cloud_batch_loss_average = sum(cloud_batch_losses)/len(cloud_batch_losses)
                    print(f'Batch loss average: ', cloud_batch_loss_average.item())
                    # Zero any gradients from previous steps
                    optimiser.zero_grad()
                    # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                    cloud_batch_loss_average.backward(retain_graph=True)
                    # add the batch mean loss * size of batch to cumulative loss
                    cumulative_epoch_loss+=cloud_batch_loss_average.item()*batch_size
                    # Update value of x += -lr * x.grad
                    optimiser.step()
                    # Ensure batch losses list is cleared
                    cloud_batch_losses.clear()
            
            # Add the batch size just used to the total number of clouds
            av_losses_per_epoch.append(cumulative_epoch_loss/cloud_counter)
            # Save checkpoint file after each epoch
            torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
            print(f'End-of-epoch: average loss = {av_losses_per_epoch}')

        print('plotting : ', av_losses_per_epoch)
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.plot(av_losses_per_epoch, label='training')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(output_directory+'loss_v_epoch.png')
    
    if testing_switch:
        output_directory = workingdir+'/sampling_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        sample_batch_size = 100
        model=Gen(4, 20, 128, 3, 1, 0, marginal_prob_std=marginal_prob_std_fn)
        load_name = workingdir+'training_20230223_1501_output/ckpt_tmp_29.pth'
        model.load_state_dict(torch.load(load_name, map_location=device))
        sampled_energies = sorted(energies[:])
        sampled_energies = random.sample(sampled_energies, sample_batch_size)
        sampled_energies = torch.tensor(sampled_energies) # Converting tensor from list of ndarrays is very slow (should convert to single ndarray first)
        sampled_energies = torch.squeeze(sampled_energies)
        print('sampled_energies: ', sampled_energies)
        sampler = pc_sampler
        # Get a sample of point clouds
        samples = sampler(model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_energies, sample_batch_size, device=device)
        print('Samples: ', samples.shape)


    if plotting_switch == 1:
        output_directory = workingdir+'/training_data_plots_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Make a simple plot of the total energy deposited by a shower in a given z layer
        point_clouds_loader = DataLoader(custom_data,batch_size=load_n_clouds,shuffle=False)
        cloud_features = CloudFeatures(point_clouds_loader)
        energy_means = cloud_features.calculate_mean_energies()
        hits_means = cloud_features.calculate_mean_nhits()
        total_e_ = cloud_features.all_energies
        total_hits_ = cloud_features.all_hits
        
        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Average deposited energy [GeV]')
        plt.xlabel('Layer number')
        plt.plot(energy_means, label='Geant4')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'avE_per_layer.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Average nhits [GeV]')
        plt.xlabel('Layer number')
        plt.plot(hits_means, label='Geant4')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'avHits_per_layer.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Entries')
        plt.xlabel('Deposited energy [GeV] (per shower)')
        plt.hist(total_e_, 20,label='Geant4')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'deposited_energy_per_cloud.png')

        fig, ax = plt.subplots(ncols=1, figsize=(10,10))
        plt.title('')
        plt.ylabel('Entries')
        plt.yscale('log')
        plt.xlabel('# hits (per shower)')
        plt.hist(total_hits_, 20, label='Geant4')
        plt.legend(loc='upper right')
        fig.savefig(output_directory+'nhits_per_cloud.png')

        

if __name__=='__main__':
    start = time.time()
    main()
    fin = time.time()
    elapsed_time = fin-start
    print('Time elapsed: {:3f}'.format(elapsed_time))