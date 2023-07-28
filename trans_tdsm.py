import time, functools, torch, os, sys, random, fnmatch, psutil, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
from torch.utils.data import Dataset, DataLoader
import utils
from prettytable import PrettyTable
import util.dataset_structure, util.display, util.model
import tqdm
from pickle import load

#import torch.multiprocessing as mp
#import wandb

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps"""
    def __init__(self, embed_dim, scale=30):
        super().__init__()
        # Time information incorporated via Gaussian random feature encoding
        # Randomly sampled weights initialisation. Fixed during optimisation i.e. not trainable
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, time):
        # Multiply batch of times by network weights
        time_proj = time[:, None] * self.W[None, :] * 2 * np.pi
        # Output [sin(2pi*wt);cos(2pi*wt)]
        gauss_out = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)
        return gauss_out

class AdaptiveBatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class Dense(nn.Module):
    """Fully connected layer that reshapes output of embedded conditional variable to feature maps"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        """Dense nn layer output must have same dimensions as input data:
            [batchsize, (dummy)nhits, (dummy)features]
        """
        return self.dense(x)[..., None]

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        """Encoder block:
        Args:
        embed_dim: length of embedding / dimension of the model
        num_heads: number of parallel attention heads to use
        hidden: dimensionaliy of hidden layer
        dropout: regularising layer
        """
        super().__init__()
        # batch_first=True because normally in NLP the batch dimension would be the second dimension
        # In everything(?) else it is the first dimension so this flag is set to true to match other conventions
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True, dropout=0)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.fc1_cls = nn.Linear(embed_dim, hidden)
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.act_dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self,x,x_cls,src_key_padding_mask=None,):
        # Stash original embedded input
        residual = x.clone()
        # Multiheaded self-attention but replacing query with a single mean field approximator
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

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ffnn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        #self.norm1 = AdaptiveBatchNorm2d(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        #self.norm2 = AdaptiveBatchNorm2d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        #self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, src_key_padding_mask=None):
        '''_src = self.attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        _src = self.dropout1(_src)
        src = src + _src.clone()
        src = self.norm1( src )
        _src = self.ffnn(src)
        _src = self.dropout2(_src)
        src = src + _src.clone()
        src = self.norm2( src )'''

        # Attention section
        attn_out = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Fully-connected section
        ffnn_out = self.ffnn(x)
        x = x + self.dropout(ffnn_out)
        x = self.norm2(x)

        return x

class Gen(nn.Module):
    def __init__(self, n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std, **kwargs):
        '''Transformer encoder model
        Arguments:
        n_feat_dim = number of features
        embed_dim = dimensionality to embed input
        hidden_dim = dimensionaliy of hidden layer
        num_encoder_blocks = number of encoder blocks
        num_attn_heads = number of parallel attention heads to use
        dropout_gen = regularising layer
        marginal_prob_std = standard deviation of Gaussian perturbation captured by SDE
        '''
        super().__init__()
        # Embedding: size of input (n_feat_dim) features -> size of output (embed_dim)
        self.embed = nn.Linear(n_feat_dim, embed_dim)
        # Seperate embedding for (time/incident energy) conditional inputs (small NN with fixed weights)
        self.embed_t = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # Boils embedding down to single value
        self.dense1 = Dense(embed_dim, 1)
        # Module list of encoder blocks
        self.encoder = nn.ModuleList(
            [
                #Block(
                #    embed_dim=embed_dim,
                #    num_heads=num_attn_heads,
                #    hidden=hidden_dim,
                #    dropout=dropout_gen,
                #)
                EncoderBlock(
                    embed_dim=embed_dim,
                    n_heads=num_attn_heads,
                    dropout=dropout_gen,
                    hidden_dim=hidden_dim,
                )
                for i in range(num_encoder_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(embed_dim, n_feat_dim)
        # token simply has same dimension as input feature embedding
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim), requires_grad=True)
        self.act = nn.GELU()
        # Swish activation function
        self.act_sig = lambda x: x * torch.sigmoid(x)
        # Standard deviation of SDE
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, e, mask=None):
        # Embed 4-vector input 
        x = self.embed(x)
        # Embed 'time' condition
        embed_t_ = self.act_sig( self.embed_t(t) )
        # Match dimensions and append to input
        x += self.dense1(embed_t_).clone()
        # Embed incident particle energy
        embed_e_ = self.embed_t(e)
        embed_e_ = self.act_sig(embed_e_)
        # Match dimensions and append to input
        x += self.dense1(embed_e_).clone()
        # 'class' token (mean field)
        #x_cls = self.cls_token.expand(x.size(0), 1, -1)

        # Feed input embeddings into encoder block
        for layer in self.encoder:
            # Each encoder block takes previous blocks output as input
            #x = layer(x, x_cls, mask) # Block layers 
            x = layer(x, mask) # EncoderBlock layers
        
        # Rescale models output (helps capture the normalisation of the true scores)
        mean_ , std_ = self.marginal_prob_std(x,t)
        output = self.out(x) / std_[:, None, None]
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
    # Generate padding mask for padded entries
    # Positions with True are ignored while False values will be unchanged
    padding_mask = (x[:,:,0]==-20).type(torch.bool)
    # Inverse mask to ignore for when 0-padded hits should be ignored
    output_mask = (x[:,:,0]!=-20).type(torch.int)
    output_mask = output_mask.unsqueeze(-1)
    output_mask = output_mask.expand(output_mask.size()[0], output_mask.size()[1],4)
    
    # Tensor of randomised conditional variable 'time' steps
    random_t = torch.rand(incident_energies.shape[0], device=device) * (1. - eps) + eps
    
    # Noise input
    # Multiply by mask so we don't go perturbing zero padded values to have some non-sentinel value
    z = torch.randn_like(x)*output_mask
    z = z.to(device)
    
    # Sample from standard deviation of noise
    mean_, std_ = marginal_prob_std(x,random_t)
    
    # Add noise to input
    perturbed_x = x + z * std_[:, None, None]
    
    # Evaluate model (aim: to estimate the score function of each noise-perturbed distribution)
    scores = model(perturbed_x, random_t, incident_energies, mask=padding_mask)
    
    # Calculate loss 
    loss = (scores*std_[:,None,None] + z)**2

    # Zero losses calculated over padded inputs
    loss = loss*output_mask
    
    # Sum loss across all hits and 4-vectors (normalise by number of hits)
    sum_loss = torch.sum( loss, dim=(1,2))

    # Average across batch
    batch_loss = torch.mean( sum_loss )
    
    return batch_loss

class pc_sampler:
    def __init__(self, snr=0.16, sampler_steps=100, device='cuda', eps=1e-3, jupyternotebook=False):
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
        self.snr = snr
        self.sampler_steps = sampler_steps
        self.device = device
        self.eps = eps
        self.jupyternotebook = jupyternotebook
        
        self.deposited_energy_t1 = []
        self.deposited_energy_t25 = []
        self.deposited_energy_t50 = []
        self.deposited_energy_t75 = []
        self.deposited_energy_t99 = []
        
        self.av_x_pos_t1 = []
        self.av_x_pos_t25 = []
        self.av_x_pos_t50 = []
        self.av_x_pos_t75 = []
        self.av_x_pos_t99 = []
        
        self.av_y_pos_t1 = []
        self.av_y_pos_t25 = []
        self.av_y_pos_t50 = []
        self.av_y_pos_t75 = []
        self.av_y_pos_t99 = []
        
        self.incident_e_t1 = []
        self.incident_e_t25 = []
        self.incident_e_t50 = []
        self.incident_e_t75 = []
        self.incident_e_t99 = []
    
    def __call__(self, score_model, marginal_prob_std, diffusion_coeff, sampled_energies, init_x, batch_size=1, energy_trans_file='', x_trans_file='', y_trans_file='', ine_trans_file=''):
        
        t = torch.ones(batch_size, device=self.device)
        
        # Mean is the only thing related to input hits, std only related to conditional
        mean_,std_ = marginal_prob_std(init_x,t)
        std_.to(self.device)

        # Padding masks defined by initial # hits / zero padding
        padding_mask = (init_x[:,:,0]==-20).type(torch.bool)
        
        # Inverse mask to ignore models output for 0-padded hits in loss
        output_mask = (init_x[:,:,0]!=-20).type(torch.int)
        output_mask = output_mask.unsqueeze(-1)
        output_mask = output_mask.expand(output_mask.size()[0], output_mask.size()[1],4)

        init_x = init_x * std_[:,None,None]
        time_steps = np.linspace(1., self.eps, self.sampler_steps)
        step_size = time_steps[0]-time_steps[1]

        if self.jupyternotebook:
            time_steps = tqdm.notebook.tqdm(time_steps)
        
        # Input shower is just some noise * std from SDE
        x = init_x
        diffusion_step_ = 0
        with torch.no_grad():
             for time_step in time_steps:
                diffusion_step_+=1
                if not self.jupyternotebook:
                    print(f"Sampler step: {time_step:.4f}") 
                batch_time_step = torch.ones(batch_size, device=x.device) * time_step

                # matrix multiplication in GaussianFourier projection doesnt like float64
                sampled_energies = sampled_energies.to(x.device, torch.float32)
                alpha = torch.ones_like(torch.tensor(time_step))

                # Corrector step (Langevin MCMC)
                # First calculate Langevin step size using the predicted scores
                grad = score_model(x, batch_time_step, sampled_energies, mask=padding_mask)

                # Noise to add to input
                noise = torch.randn_like(x)
                
                # Multiply by mask so we don't add noise to padded values / use gradients for padding in loss
                noise = noise * output_mask
                grad = grad * output_mask

                # Langevin step size calculation: snr * ratio of gradients in noise / prediction used to calculate
                flattened_scores = grad.reshape(grad.shape[0], -1)
                grad_norm = torch.linalg.norm( flattened_scores, dim=-1 ).mean()
                flattened_noise = noise.reshape(noise.shape[0],-1)
                noise_norm = torch.linalg.norm( flattened_noise, dim=-1 ).mean()
                langevin_step_size =  (self.snr * noise_norm / grad_norm)**2 * 2 * alpha

                # Adjust inputs according to scores using Langevin iteration rule
                x_mean = x + langevin_step_size * grad
                x = x_mean + torch.sqrt(2 * langevin_step_size) * noise
                
                # Adjust inputs according to scores using Euler-Maruyama predictor iteration rule
                drift, diff = diffusion_coeff(x,batch_time_step)
                x_mean = x + (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies, mask=padding_mask) * step_size
                x = x_mean + torch.sqrt(diff**2 * step_size)[:, None, None] * torch.randn_like(x)
                
                # Store distributions at different stages of diffusion
                if diffusion_step_== 1:
                    # Load saved pre-processor
                    print(f'Loading file for hit e transformation inversion: {energy_trans_file}')
                    if ine_trans_file != '':
                        scalar_ine = load(open(ine_trans_file, 'rb'))
                    if energy_trans_file != '':
                        scalar_e = load(open(energy_trans_file, 'rb'))
                    if x_trans_file != '':
                        scalar_x = load(open(x_trans_file, 'rb'))
                    if y_trans_file != '':
                        scalar_y = load(open(y_trans_file, 'rb'))

                    for shower_idx in range(0,len(x_mean)):
                        masked_output = x_mean*output_mask
                        
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()[0]
                        self.incident_e_t1.append( all_ine )
                        
                        all_e = np.array( masked_output[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        total_deposited_energy = sum( all_e )
                        self.deposited_energy_t1.append(total_deposited_energy)
                        
                        all_x = np.array( masked_output[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        av_x_position = np.mean( all_x )
                        self.av_x_pos_t1.append( av_x_position )
                        
                        all_y = np.array( masked_output[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        av_y_position = np.mean( all_y )
                        self.av_y_pos_t1.append( av_y_position )
                if diffusion_step_== 25:
                    for shower_idx in range(0,len(x_mean)):
                        masked_output = x_mean*output_mask
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()[0]
                        self.incident_e_t25.append( all_ine )
                        
                        all_e = np.array( masked_output[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        total_deposited_energy = sum( all_e )
                        self.deposited_energy_t25.append(total_deposited_energy)
                        
                        all_x = np.array( masked_output[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        av_x_position = np.mean( all_x )
                        self.av_x_pos_t25.append( av_x_position )
                        
                        all_y = np.array( masked_output[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        av_y_position = np.mean( all_y )
                        self.av_y_pos_t25.append( av_y_position )
                if diffusion_step_== 50:
                    for shower_idx in range(0,len(x_mean)):
                        masked_output = x_mean*output_mask
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()[0]
                        self.incident_e_t50.append( all_ine )
                        
                        all_e = np.array( masked_output[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        total_deposited_energy = sum( all_e )
                        self.deposited_energy_t50.append(total_deposited_energy)
                        
                        all_x = np.array( masked_output[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        av_x_position = np.mean( all_x )
                        self.av_x_pos_t50.append( av_x_position )
                        
                        all_y = np.array( masked_output[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        av_y_position = np.mean( all_y )
                        self.av_y_pos_t50.append( av_y_position )
                if diffusion_step_== 75:
                    for shower_idx in range(0,len(x_mean)):
                        masked_output = x_mean*output_mask
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()[0]
                        self.incident_e_t75.append( all_ine )
                        
                        all_e = np.array( masked_output[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        total_deposited_energy = sum( all_e )
                        self.deposited_energy_t75.append(total_deposited_energy)
                        
                        all_x = np.array( masked_output[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        av_x_position = np.mean( all_x )
                        self.av_x_pos_t75.append( av_x_position )
                        
                        all_y = np.array( masked_output[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        av_y_position = np.mean( all_y )
                        self.av_y_pos_t75.append( av_y_position )
                if diffusion_step_== 99:
                    for shower_idx in range(0,len(x_mean)):
                        masked_output = x_mean*output_mask
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()[0]
                        self.incident_e_t99.append( all_ine )
                        
                        all_e = np.array( masked_output[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        total_deposited_energy = sum( all_e )
                        self.deposited_energy_t99.append(total_deposited_energy)
                        
                        all_x = np.array( masked_output[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        av_x_position = np.mean( all_x )
                        self.av_x_pos_t99.append( av_x_position )
                        
                        all_y = np.array( masked_output[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        av_y_position = np.mean( all_y )
                        self.av_y_pos_t99.append( av_y_position )

        # Do not include noise in last step
        # Need to remove padded hits?
        x_mean = x_mean
        return x_mean

def check_mem():
    # Resident set size memory (non-swap physical memory process has used)
    process = psutil.Process(os.getpid())
    # Print bytes in GB
    print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))
    return

def random_sampler(pdf,xbin):
    myCDF = np.zeros_like(xbin,dtype=float)
    myCDF[1:] = np.cumsum(pdf)
    a = np.random.uniform(0, 1)
    return xbin[np.argmax(myCDF>=a)-1]

def get_prob_dist(x,y,nbins):
    '''
    2D histogram:
    x = incident energy per shower
    y = # valid hits per shower
    '''
    hist,xbin,ybin = np.histogram2d(x,y,bins=nbins,density=False)
    # Normalise histogram
    sum_ = hist.sum(axis=-1)
    sum_ = sum_[:,None]
    hist = hist/sum_
    # Remove NaN
    hist[np.isnan(hist)] = 0.0
    return hist, xbin, ybin

def generate_hits(prob, xbin, ybin, x_vals, max_hits, n_features, device='cpu'):
    ind = np.digitize(x_vals, xbin) - 1
    ind[ind==len(xbin)-1] = len(xbin)-2
    ind[ind==-1] = 0    
    y_pred = []
    pred_nhits = []
    prob_ = prob[ind,:]
    for i in range(len(prob_)):
        nhits = int(random_sampler(prob_[i],ybin + 1))
        pred_nhits.append(nhits)
        y_pred.append(torch.randn(nhits, n_features, device=device))
    
    return pred_nhits, y_pred


def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-o','--output',dest='output_path', help='Path to output directory', default='', type=str)
    argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that run: evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
    argparser.add_argument('-i','--inputs',dest='inputs', help='Path to input directory', default='quantile_gauss_transformer', type=str)
    args = argparser.parse_args()
    workingdir = args.output_path
    indir = args.inputs
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Running on device: ', device)
    if torch.cuda.is_available():
        print('Cuda used to build pyTorch: ',torch.version.cuda)
        print('Current device: ', torch.cuda.current_device())
        print('Cuda arch list: ', torch.cuda.get_arch_list())
    
    print('Working directory: ' , workingdir)
    
    # Useful when debugging gradient issues
    torch.autograd.set_detect_anomaly(True)

    ### HYPERPARAMETERS ###
    train_ratio = 0.8
    batch_size = 64
    lr = 0.00001
    n_epochs = 5
    ### SDE PARAMETERS ###
    SDE = 'VE'
    sigma_max = 50.
    ### MODEL PARAMETERS ###
    n_feat_dim = 4
    embed_dim = 512
    hidden_dim = 128
    num_encoder_blocks = 3
    num_attn_heads = 8
    dropout_gen = 0
    # SAMPLER PARAMETERS
    sampler_steps = 100
    n_showers_2_gen = 64

    model_params = f'''
    ### PARAMS ###
    ### SDE PARAMETERS ###
    SDE = {SDE}
    sigma/beta max. = {sigma_max}
    ### MODEL PARAMETERS ###
    batch_size = {batch_size}
    lr = {lr}
    n_epochs = {n_epochs}
    n_feat_dim = {n_feat_dim}
    embed_dim = {embed_dim}
    hidden_dim = {hidden_dim}
    num_encoder_blocks = {num_encoder_blocks}
    num_attn_heads = {num_attn_heads}
    dropout_gen = {dropout_gen}
    # SAMPLE PARAMETERS
    sampler_steps = {sampler_steps}
    n_showers_2_gen = {n_showers_2_gen}
    '''
    
    # Instantiate stochastic differential equation
    if SDE == 'VP':
        sde = utils.VPSDE(beta_max=sigma_max,device=device)
    if SDE == 'VE':
        sde = utils.VESDE(sigma_max=sigma_max,device=device)
    marginal_prob_std_fn = functools.partial(sde.marginal_prob)
    diffusion_coeff_fn = functools.partial(sde.sde)

    # List of training input files
    #training_file_path = '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/datasets/'
    #training_file_path = '/eos/user/t/tihsu/database/ML_hackthon/bucketed_tensor/'
    training_file_path = os.path.join('/eos/user/j/jthomasw/tdsm_encoder/datasets/', indir)
    files_list_ = []
    print(f'Training files found in: {training_file_path}')
    for filename in os.listdir(training_file_path):
        if fnmatch.fnmatch(filename, 'dataset_1_photons_padded*.pt'):
            files_list_.append(os.path.join(training_file_path,filename))
    print(f'Files: {files_list_}')

    #dataset_store_path = "/eos/user/j/jthomasw/tdsm_encoder/datasets/quantile_gauss_transformer/"

    #### Input plots ####
    if switches_ & trigger:
        # Limit to N showers to use for plots
        nshowers_2_plot = 100
        # Transformed variables
        dists_trans = util.display.plot_distribution(files_list_, nshowers_2_plot)
        entries = dists_trans[0]
        all_incident_e_trans = dists_trans[1]
        total_deposited_e_shower_trans = dists_trans[2]
        all_e_trans = dists_trans[3]
        all_x_trans = dists_trans[4]
        all_y_trans = dists_trans[5]
        all_z_trans = dists_trans[6]
        all_hit_ine_trans = dists_trans[7]
        average_x_shower_trans = dists_trans[8]
        average_y_shower_trans = dists_trans[9]

        # Non-transformed variables
        dists = util.display.plot_distribution(files_list_, nshowers_2_plot, energy_trans_file='transform_e.pkl', x_trans_file='transform_x.pkl', y_trans_file='transform_y.pkl', ine_trans_file='rescaler_y.pkl')
        entries = dists[0]
        all_incident_e = dists[1]
        total_deposited_e_shower = dists[2]
        all_e = dists[3]
        all_x = dists[4]
        all_y = dists[5]
        all_z = dists[6]
        all_hit_ine = dists[7]
        average_x_shower = dists[8]
        average_y_shower = dists[9]

        ### 2D distributions
        
        distributions = [(('X', 'Hit energy [GeV]', 'Incident energy [GeV]') , (all_x, all_e, all_hit_ine, all_x_trans, all_e_trans, all_hit_ine_trans))]
        util.display.make_plot(distributions,training_file_path)

        distributions = [(('Incident energy [GeV]', 'Hit energy [GeV]', 'Incident energy [GeV]') , (all_hit_ine, all_e, all_hit_ine, all_hit_ine_trans, all_e_trans, all_hit_ine_trans))]
        util.display.make_plot(distributions,training_file_path)

        distributions = [(('Av. X Position', 'Av. Y Position', 'Incident energy [GeV]') , (average_x_shower, average_y_shower, all_incident_e, average_x_shower_trans, average_y_shower_trans, all_incident_e_trans))]
        util.display.make_plot(distributions,training_file_path)

        distributions = [(('Incident energy [GeV]', 'Av. Energy Deposited [GeV]', 'Incident energy [GeV]') , (all_incident_e, total_deposited_e_shower, all_incident_e, all_incident_e_trans, total_deposited_e_shower_trans, all_incident_e_trans))]
        util.display.make_plot(distributions,training_file_path)

        fig, ax = plt.subplots(3,3, figsize=(12,12))
        print('Plot # entries')
        ax[0][0].set_ylabel('# entries')
        ax[0][0].set_xlabel('Hit entries')
        ax[0][0].hist(entries, 50, color='orange', label='Geant4')
        ax[0][0].legend(loc='upper right')

        print('Plot hit energies')
        ax[0][1].set_ylabel('# entries')
        ax[0][1].set_xlabel('Hit energy [GeV]')
        ax[0][1].hist(all_e_trans, 50, color='orange', label='Geant4')
        ax[0][1].set_yscale('log')
        ax[0][1].legend(loc='upper right')

        print('Plot hit x')
        ax[0][2].set_ylabel('# entries')
        ax[0][2].set_xlabel('Hit x position')
        ax[0][2].hist(all_x_trans, 50, color='orange', label='Geant4')
        ax[0][2].set_yscale('log')
        ax[0][2].legend(loc='upper right')

        print('Plot hit y')
        ax[1][0].set_ylabel('# entries')
        ax[1][0].set_xlabel('Hit y position')
        ax[1][0].hist(all_y_trans, 50, color='orange', label='Geant4')
        ax[1][0].set_yscale('log')
        ax[1][0].legend(loc='upper right')

        print('Plot hit z')
        ax[1][1].set_ylabel('# entries')
        ax[1][1].set_xlabel('Hit z position')
        ax[1][1].hist(all_z_trans, color='orange', label='Geant4')
        ax[1][1].set_yscale('log')
        ax[1][1].legend(loc='upper right')

        print('Plot incident energies')
        ax[1][2].set_ylabel('# entries')
        ax[1][2].set_xlabel('Incident energies [GeV]')
        ax[1][2].hist(all_incident_e_trans, 50, color='orange', label='Geant4')
        ax[1][2].set_yscale('log')
        ax[1][2].legend(loc='upper right')

        print('Plot total deposited hit energy per shower')
        ax[2][0].set_ylabel('# entries')
        ax[2][0].set_xlabel('Deposited energy [GeV]')
        ax[2][0].hist(total_deposited_e_shower_trans, 50, color='orange', label='Geant4')
        ax[2][0].set_yscale('log')
        ax[2][0].legend(loc='upper right')

        print('Plot av. X position per shower')
        ax[2][1].set_ylabel('# entries')
        ax[2][1].set_xlabel('Average X position [GeV]')
        ax[2][1].hist(average_x_shower_trans, 50, color='orange', label='Geant4')
        ax[2][1].set_yscale('log')
        ax[2][1].legend(loc='upper right')

        print('Plot av. Y position per shower')
        ax[2][2].set_ylabel('# entries')
        ax[2][2].set_xlabel('Average Y position [GeV]')
        ax[2][2].hist(average_y_shower_trans, 50, color='orange', label='Geant4')
        ax[2][2].set_yscale('log')
        ax[2][2].legend(loc='upper right')

        save_name = os.path.join(training_file_path,'input_dists_transformed.png')
        fig.savefig(save_name)

    #### Training ####
    if switches_>>1 & trigger:
        output_directory = workingdir+'/training_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        print('Output directory: ', output_directory)
        if not os.path.exists(output_directory):
            print(f'Making new directory: {output_directory}')
            os.makedirs(output_directory)
        
        model=Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
        table = PrettyTable(['Module name', 'Parameters listed'])
        t_params = 0
        for name_ , para_ in model.named_parameters():
            if not para_.requires_grad: continue
            param = para_.numel()
            table.add_row([name_, param])
            t_params+=param
        print(table)
        print(f'Sum of trainable parameters: {t_params}')    
        
        print('model: ', model)
        if torch.cuda.device_count() > 1:
            print(f'Lets use {torch.cuda.device_count()} GPUs!')
            model = nn.DataParallel(model)

        # Optimiser needs to know model parameters for to optimise
        optimiser = RAdam(model.parameters(),lr=lr)
        
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
            training_batches_per_epoch = 0
            testing_batches_per_epoch = 0
            
            # For debugging purposes
            #files_list_ = files_list_[:1]

            # Load files
            for filename in files_list_:
                file_counter+=1

                # Rescaling now done in padding package
                custom_data = utils.cloud_dataset(filename, device=device)
                train_size = int(train_ratio * len(custom_data.data))
                test_size = len(custom_data.data) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])
                print(f'Size of training dataset in (file {file_counter}/{len(files_list_)}): {len(train_dataset)} showers, batch = {batch_size} showers')

                n_training_showers+=train_size
                n_testing_showers+=test_size
                # Load clouds for each epoch of data dataloaders length will be the number of batches
                shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

                # Accumuate number of batches per epoch
                training_batches_per_epoch += len(shower_loader_train)
                testing_batches_per_epoch += len(shower_loader_test)
                
                # Load shower batch for training
                for i, (shower_data,incident_energies) in enumerate(shower_loader_train,0):
                    # Move model to device and set dtype as same as data (note torch.double works on both CPU and GPU)
                    model.to(device, shower_data.dtype)
                    model.train()
                    shower_data.to(device)
                    incident_energies.to(device)
                    if len(shower_data) < 1:
                        print('Very few hits in shower: ', len(shower_data))
                        continue
                    # Zero any gradients from previous steps
                    optimiser.zero_grad()
                    # Loss average for each batch
                    loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, device=device)
                    cumulative_epoch_loss+=float(loss)
                    # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                    loss.backward()
                    # Update value of x += -lr * x.grad
                    optimiser.step()
                
                # Testing on subset of file
                for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
                    with torch.no_grad():
                        model.to(device, shower_data.dtype)
                        model.eval()
                        shower_data = shower_data.to(device)
                        incident_energies = incident_energies.to(device)
                        test_loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, device=device)
                        cumulative_test_epoch_loss+=float(test_loss)

            # Add the batch size just used to the total number of clouds
            # Calculate average loss per epoch
            av_training_losses_per_epoch.append(cumulative_epoch_loss/training_batches_per_epoch)
            av_testing_losses_per_epoch.append(cumulative_test_epoch_loss/testing_batches_per_epoch)
            print(f'End-of-epoch: average train loss = {av_training_losses_per_epoch}, average test loss = {av_testing_losses_per_epoch}')
            if epoch % 50 == 0:
                # Save checkpoint file after each epoch
                torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
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
        
        torch.save(model.state_dict(), output_directory+'ckpt_tmp_'+str(epoch)+'.pth')
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
    
    #### Sampling ####
    if switches_>>2 & trigger:    
        output_directory = workingdir+'/sampling_'+str(sampler_steps)+'samplersteps_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
        if not os.path.exists(output_directory):
            print(f'Making new directory: {output_directory}')
            os.makedirs(output_directory)
        
        # Load saved model
        model=Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
        load_name = os.path.join(workingdir,'training_20230727_1756_output/ckpt_tmp_4.pth')
        model.load_state_dict(torch.load(load_name, map_location=device))
        model.to(device)

        # Get max. N hits / N valid hits in batch
        n_valid_hits_per_shower = []
        incident_e_per_shower = []
        geant_deposited_energy = []
        geant_x_pos = []
        geant_y_pos = []
        geant_ine = []
        N_geant_showers = 0
        max_hits = -1

        # For diffusion plots in 'physical' feature space, add files here
        energy_trans_file = ''#os.path.join(files_list_[0].rsplit('/',1)[0],'transform_e.pkl')
        x_trans_file = ''#os.path.join(files_list_[0].rsplit('/',1)[0],'transform_x.pkl')
        y_trans_file = ''#os.path.join(files_list_[0].rsplit('/',1)[0],'transform_y.pkl')
        ine_trans_file = ''#os.path.join(files_list_[0].rsplit('/',1)[0],'rescaler_y.pkl')

        # Load saved pre-processor
        if ine_trans_file != '':
            print(f'energy_trans_file: {energy_trans_file}')
            scalar_ine = load(open(ine_trans_file, 'rb'))
        if energy_trans_file != '':
            scalar_e = load(open(energy_trans_file, 'rb'))
        if x_trans_file != '':
            scalar_x = load(open(x_trans_file, 'rb'))
        if y_trans_file != '':
            scalar_y = load(open(y_trans_file, 'rb'))
        
        for file in files_list_:
            # Load shower data
            custom_data = utils.cloud_dataset(file, device=device)
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=False)
            # Loop over batches
            for i, (shower_data, incident_energies) in enumerate(point_clouds_loader,0):

                # Copy data
                valid_event = []
                data_np = shower_data.cpu().numpy().copy()
                energy_np = incident_energies.cpu().numpy().copy()

                # Mask for padded values
                masking = data_np[:,:,0] != -20
                
                # Loop over each shower in batch
                for j in range(len(data_np)):
                    # Mask padded hits and count valid hits for shower j in batch
                    valid_hits = data_np[j][masking[j]]
                    n_valid_hits_per_shower.append(len(valid_hits))
                    if len(valid_hits)>max_hits:
                        max_hits = len(valid_hits)
                    incident_e_per_shower.append(energy_np[j])

                    # ONLY for plotting purposes
                    if N_geant_showers < n_showers_2_gen:
                        all_ine = np.array(energy_np[j]).reshape(-1,1)
                        # Rescale the conditional input for each shower
                        if ine_trans_file != '':
                            all_ine = scalar_ine.inverse_transform(all_ine)
                        all_ine = all_ine.flatten().tolist()
                        geant_ine.append(all_ine)
                        
                        all_e = valid_hits[:,0].reshape(-1,1)
                        if energy_trans_file != '':
                            all_e = scalar_e.inverse_transform(all_e)
                        all_e = all_e.flatten().tolist()
                        geant_deposited_energy.append( sum( all_e ) )
                        
                        all_x = valid_hits[:,1].reshape(-1,1)
                        if x_trans_file != '':
                            all_x = scalar_x.inverse_transform(all_x)
                        all_x = all_x.flatten().tolist()
                        geant_x_pos.append( np.mean(all_x) )
                        
                        all_y = valid_hits[:,2].reshape(-1,1)
                        if y_trans_file != '':
                            all_y = scalar_y.inverse_transform(all_y)
                        all_y = all_y.flatten().tolist()
                        geant_y_pos.append( np.mean(all_y) )

                    N_geant_showers+=1
            del custom_data

        # Arrays of Nvalid hits in showers, incident energies per shower
        n_valid_hits_per_shower = np.array(n_valid_hits_per_shower)
        incident_e_per_shower = np.array(incident_e_per_shower)
        max_incident_e = max(incident_e_per_shower)
        min_incident_e = min(incident_e_per_shower)

        # Generate 2D pdf of incident E vs N valid hits from the training file(s)
        n_bins_prob_dist = 50
        e_vs_nhits_prob, x_bin, y_bin = get_prob_dist(incident_e_per_shower, n_valid_hits_per_shower, n_bins_prob_dist)

        # Plot 2D histogram (sanity check)
        fig0, (ax0) = plt.subplots(ncols=1, sharey=True)
        heatmap = ax0.pcolormesh(y_bin, x_bin, e_vs_nhits_prob, cmap='rainbow')
        ax0.plot(n_valid_hits_per_shower, n_valid_hits_per_shower, 'k-')
        ax0.set_xlim(n_valid_hits_per_shower.min(), n_valid_hits_per_shower.max())
        ax0.set_ylim(incident_e_per_shower.min(), incident_e_per_shower.max())
        ax0.set_xlabel('n_valid_hits_per_shower')
        ax0.set_ylabel('incident_e_per_shower')
        cbar = plt.colorbar(heatmap)
        cbar.ax.set_ylabel('PDF', rotation=270)
        ax0.set_title('histogram2d')
        ax0.grid()
        savefigname = os.path.join(output_directory,'validhits_ine_2D.png')
        fig0.savefig(savefigname)

        # Generate tensor sampled from the appropriate range of injection energies
        idx_list_incident_e = torch.randint(0, len(incident_e_per_shower), (n_showers_2_gen,))
        in_energies = np.random.choice( incident_e_per_shower, n_showers_2_gen )
        in_energies = torch.from_numpy(in_energies)

        # Sample from 2D pdf = nhits per shower vs incident energies -> nhits and a tensor of randomly initialised hit features
        nhits, gen_hits = generate_hits(e_vs_nhits_prob, x_bin, y_bin, in_energies, max_hits, 4, device=device)

        # Save
        torch.save([gen_hits, in_energies],'tmp.pt')

        # Load the showers of noise
        gen_hits = utils.cloud_dataset('tmp.pt', device=device)
        # Pad showers with values of -20
        gen_hits.padding(-20)
        # Load len(gen_hits_loader) number of batches each with batch_size number of showers
        gen_hits_loader = DataLoader(gen_hits, batch_size=batch_size, shuffle=False)

        # Remove noise shower file
        os.system("rm tmp.pt")

        # Instantiate sampler
        sample = []
        sampler = pc_sampler(snr=0.16, sampler_steps=sampler_steps, device=device, jupyternotebook=False)

        # Loop over each batch of noise showers
        print(f'# batches: {len(gen_hits_loader)}' )
        for i, (gen_hit, sampled_energies) in enumerate(gen_hits_loader,0):
            
            print(f'Generation batch {i}: showers per batch: {gen_hit.shape[0]}, max. hits per shower: {gen_hit.shape[1]}, features per hit: {gen_hit.shape[2]}, sampled_energies: {len(sampled_energies)}')    
            sys.stdout.write('\r')
            sys.stdout.write("Progress: %d/%d \n" % ((i+1), len(gen_hits_loader)))
            sys.stdout.flush()
            
            # Run reverse diffusion sampler
            generative = sampler(model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_energies, gen_hit, batch_size=gen_hit.shape[0], energy_trans_file=energy_trans_file, x_trans_file=x_trans_file , y_trans_file = y_trans_file, ine_trans_file=ine_trans_file)
            
            # Create first sample or concatenate sample to sample list
            if i == 0:
                sample = generative
            else:
                sample = torch.cat([sample,generative])
            
            print(f'sample: {sample.shape}')
            
        sample_ = []
        sample_np = sample.cpu().numpy()

        for i in range(len(sample_np)):
            tmp_sample = sample_np[i][:nhits[i]]
            sample_.append(torch.tensor(tmp_sample))
        torch.save([sample_,in_energies], os.path.join(output_directory, 'sample.pt'))

        # Create plots of distributions evolving with diffusion steps
        distributions = [
        ( ('X', 'Y'), 
        (geant_x_pos,
        geant_y_pos,
        sampler.av_x_pos_t1,
        sampler.av_y_pos_t1, 
        sampler.av_x_pos_t25,
        sampler.av_y_pos_t25,
        sampler.av_x_pos_t50,
        sampler.av_y_pos_t50,
        sampler.av_x_pos_t75,
        sampler.av_y_pos_t75,
        sampler.av_x_pos_t99,
        sampler.av_y_pos_t99) )
        ]
        util.display.make_diffusion_plot(distributions, output_directory)

        distributions = [
        ( ('X', 'Total deposited energy [GeV]'), 
        (geant_x_pos,
        geant_deposited_energy,
        sampler.av_x_pos_t1,
        sampler.deposited_energy_t1, 
        sampler.av_x_pos_t25,
        sampler.deposited_energy_t25,
        sampler.av_x_pos_t50,
        sampler.deposited_energy_t50,
        sampler.av_x_pos_t75,
        sampler.deposited_energy_t75,
        sampler.av_x_pos_t99,
        sampler.deposited_energy_t99) )
        ]
        util.display.make_diffusion_plot(distributions, output_directory)

        distributions = [
        ( ('Total deposited energy', 'Incident particle energy [GeV]'), 
        (geant_deposited_energy,
        incident_e_per_shower[:n_showers_2_gen],
        sampler.deposited_energy_t1,
        sampler.incident_e_t1, 
        sampler.deposited_energy_t25,
        sampler.incident_e_t25,
        sampler.deposited_energy_t50,
        sampler.incident_e_t50,
        sampler.deposited_energy_t75,
        sampler.incident_e_t75,
        sampler.deposited_energy_t99,
        sampler.incident_e_t99) )
        ]
        util.display.make_diffusion_plot(distributions, output_directory)

    #### Evaluation plots ####
    if switches_>>3 & trigger:
        #output_directory = workingdir+'/evaluation_'+str(sampler_steps)+'samplersteps_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
        #print(f'Evaluation outputs stored here: {output_directory}')
        #if not os.path.exists(output_directory):
        #    os.makedirs(output_directory)

        # Distributions object for generated files
        print(f'Generated inputs')
        output_directory = 'sampling_100samplersteps_20230728_1523_output'
        print(f'Evaluation outputs stored here: {output_directory}')
        plot_file_name = os.path.join(output_directory, 'sample.pt')
        custom_data = utils.cloud_dataset(plot_file_name,device=device)
        # when providing just cloud dataset, energy_trans_file needs to include full path
        #dists_gen = util.display.plot_distribution(custom_data, energy_trans_file='/eos/user/j/jthomasw/tdsm_encoder/datasets/power_transformer/transform_e.pkl', nshowers_2_plot=n_showers_2_gen)
        dists_gen = util.display.plot_distribution(custom_data, nshowers_2_plot=n_showers_2_gen)

        entries_gen = dists_gen[0]
        all_incident_e_gen = dists_gen[1]
        total_deposited_e_shower_gen = dists_gen[2]
        all_e_gen = dists_gen[3]
        all_x_gen = dists_gen[4]
        all_y_gen = dists_gen[5]
        all_z_gen = dists_gen[6]
        all_hit_ine_gen = dists_gen[7]
        average_x_shower_gen = dists_gen[8]
        average_y_shower_gen = dists_gen[9]

        print(f'Geant4 inputs')
        # Distributions object for Geant4 files
        #dists = util.display.plot_distribution(files_list_, energy_trans_file='transform_e.pkl', nshowers_2_plot=n_showers_2_gen)
        dists = util.display.plot_distribution(files_list_, nshowers_2_plot=n_showers_2_gen)

        entries = dists[0]
        all_incident_e = dists[1]
        total_deposited_e_shower = dists[2]
        all_e = dists[3]
        all_x = dists[4]
        all_y = dists[5]
        all_z = dists[6]
        all_hit_ine_geant = dists[7]
        average_x_shower_geant = dists[8]
        average_y_shower_geant = dists[9]

        print('Plot # entries')
        bins=np.histogram(np.hstack((entries,entries_gen)), bins=50)[1]
        fig, ax = plt.subplots(3,3, figsize=(12,12))
        ax[0][0].set_ylabel('# entries')
        ax[0][0].set_xlabel('Hit entries')
        ax[0][0].hist(entries, bins, alpha=0.5, color='orange', label='Geant4')
        ax[0][0].hist(entries_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[0][0].legend(loc='upper right')

        print('Plot hit energies')
        bins=np.histogram(np.hstack((all_e,all_e_gen)), bins=50)[1]
        ax[0][1].set_ylabel('# entries')
        ax[0][1].set_xlabel('Hit energy [GeV]')
        ax[0][1].hist(all_e, bins, alpha=0.5, color='orange', label='Geant4')
        ax[0][1].hist(all_e_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[0][1].set_yscale('log')
        ax[0][1].legend(loc='upper right')

        print('Plot hit x')
        bins=np.histogram(np.hstack((all_x,all_x_gen)), bins=50)[1]
        ax[0][2].set_ylabel('# entries')
        ax[0][2].set_xlabel('Hit x position')
        ax[0][2].hist(all_x, bins, alpha=0.5, color='orange', label='Geant4')
        ax[0][2].hist(all_x_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[0][2].set_yscale('log')
        ax[0][2].legend(loc='upper right')

        print('Plot hit y')
        bins=np.histogram(np.hstack((all_y,all_y_gen)), bins=50)[1]
        ax[1][0].set_ylabel('# entries')
        ax[1][0].set_xlabel('Hit y position')
        ax[1][0].hist(all_y, bins, alpha=0.5, color='orange', label='Geant4')
        ax[1][0].hist(all_y_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[1][0].set_yscale('log')
        ax[1][0].legend(loc='upper right')

        print('Plot hit z')
        bins=np.histogram(np.hstack((all_z,all_z_gen)), bins=50)[1]
        ax[1][1].set_ylabel('# entries')
        ax[1][1].set_xlabel('Hit z position')
        ax[1][1].hist(all_z, bins, alpha=0.5, color='orange', label='Geant4')
        ax[1][1].hist(all_z_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[1][1].set_yscale('log')
        ax[1][1].legend(loc='upper right')

        print('Plot incident energies')
        bins=np.histogram(np.hstack((all_incident_e,all_incident_e_gen)), bins=50)[1]
        ax[1][2].set_ylabel('# entries')
        ax[1][2].set_xlabel('Incident energies [GeV]')
        ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
        ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[1][2].set_yscale('log')
        ax[1][2].legend(loc='upper right')

        print('Plot total deposited hit energy')
        bins=np.histogram(np.hstack((total_deposited_e_shower,total_deposited_e_shower_gen)), bins=50)[1]
        ax[2][0].set_ylabel('# entries')
        ax[2][0].set_xlabel('Deposited energy [GeV]')
        ax[2][0].hist(total_deposited_e_shower, bins, alpha=0.5, color='orange', label='Geant4')
        ax[2][0].hist(total_deposited_e_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[2][0].set_yscale('log')
        ax[2][0].legend(loc='upper right')

        print('Plot average hit X position')
        bins=np.histogram(np.hstack((average_x_shower_geant,average_x_shower_gen)), bins=50)[1]
        ax[2][1].set_ylabel('# entries')
        ax[2][1].set_xlabel('Average X pos.')
        ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
        ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[2][1].set_yscale('log')
        ax[2][1].legend(loc='upper right')

        print('Plot average hit Y position')
        bins=np.histogram(np.hstack((average_y_shower_geant,average_y_shower_gen)), bins=50)[1]
        ax[2][2].set_ylabel('# entries')
        ax[2][2].set_xlabel('Average Y pos.')
        ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
        ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
        ax[2][2].set_yscale('log')
        ax[2][2].legend(loc='upper right')

        fig_name = os.path.join(output_directory, 'Geant_Gen_comparison.png')
        print(f'Figure name: {fig_name}')
        fig.savefig(fig_name)


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