import functools, torch, tqdm
import numpy as np

class pc_sampler:
    def __init__(self, sde, padding_value, snr=0.2, sampler_steps=100, steps2plot=(), device='cuda', eps=1e-3, jupyternotebook=False):
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
        self.sde = sde
        self.diffusion_coeff_fn = functools.partial(self.sde.sde)
        self.snr = snr
        self.padding_value = padding_value
        self.sampler_steps = sampler_steps
        self.steps2plot = steps2plot
        self.device = device
        self.eps = eps
        self.jupyternotebook = jupyternotebook
        
        # Dictionary objects  hold lists of variable values at various stages of diffusion process
        # Used for visualisation and diagnostic purposes only
        self.hit_energy_stages = { x:[] for x in self.steps2plot}
        self.hit_x_stages = { x:[] for x in self.steps2plot}
        self.hit_y_stages = { x:[] for x in self.steps2plot}
        self.deposited_energy_stages = { x:[] for x in self.steps2plot}
        self.av_x_stages = { x:[] for x in self.steps2plot}
        self.av_y_stages = { x:[] for x in self.steps2plot}
        self.incident_e_stages = { x:[] for x in self.steps2plot}
    
    def __call__(self, score_model, sampled_energies, init_x, batch_size=1):
        
        # Time array
        t = torch.ones(batch_size, device=self.device)
        # Padding masks defined by initial # hits / zero padding
        padding_mask = (init_x[:,:,0]== self.padding_value).type(torch.bool)
        # Create array of time steps
        time_steps = np.linspace(1., self.eps, self.sampler_steps)
        step_size = time_steps[0]-time_steps[1]

        if self.jupyternotebook:
            time_steps = tqdm.notebook.tqdm(time_steps)
        
        # Input shower is just some noise * std from SDE
        x = init_x
        
        diffusion_step_ = 0
        with torch.no_grad():
            # Matrix multiplication in GaussianFourier projection doesnt like float64
            sampled_energies = sampled_energies.to(x.device, torch.float32)
            
            # Iterate through time steps
            for time_step in time_steps:
                
                if not self.jupyternotebook:
                    print(f"Sampler step: {time_step:.4f}") 
                
                batch_time_step = torch.ones(batch_size, device=x.device) * time_step

                alpha = torch.ones_like(torch.tensor(time_step))

                # Corrector step (Langevin MCMC)
                # Noise to add to input
                z = torch.normal(0,1,size=x.shape, device=x.device)
                
                # Conditional score prediction gives estimate of noise to remove
                grad = score_model(x, batch_time_step, sampled_energies, mask=padding_mask)
                
                nc_steps = 100
                for n_ in range(nc_steps):
                    # Langevin corrector
                    noise = torch.normal(0,1,size=x.shape, device=x.device)
                    # step size calculation: snr * ratio of gradients in noise / prediction used to calculate
                    flattened_scores = grad.reshape(grad.shape[0], -1)
                    grad_norm = torch.linalg.norm( flattened_scores, dim=-1 ).mean()
                    flattened_noise = noise.reshape(noise.shape[0],-1)
                    noise_norm = torch.linalg.norm( flattened_noise, dim=-1 ).mean()
                    langevin_step_size =  (self.snr * noise_norm / grad_norm)**2 * 2 * alpha
                    # Adjust inputs according to scores using Langevin iteration rule
                    x_mean = x + langevin_step_size * grad
                    x = x_mean + torch.sqrt(2 * langevin_step_size) * noise
                
                # Euler-Maruyama Predictor
                # Adjust inputs according to scores
                drift, diff = self.diffusion_coeff_fn(x,batch_time_step)
                drift = drift - (diff**2)[:, None, None] * score_model(x, batch_time_step, sampled_energies, mask=padding_mask)
                x_mean = x - drift*step_size
                x = x_mean + torch.sqrt(diff**2*step_size)[:, None, None] * z
                
                # Store distributions at different stages of diffusion (for visualisation purposes only)
                if diffusion_step_ in self.steps2plot:
                    step_incident_e = []
                    step_hit_e = []
                    step_hit_x = []
                    step_hit_y = []
                    step_deposited_energy = []
                    step_av_x_pos = []
                    step_av_y_pos = []
                    for shower_idx in range(0,len(x_mean)):
                        all_ine = np.array( sampled_energies[shower_idx].cpu().numpy().copy() ).reshape(-1,1)
                        all_ine = all_ine.flatten().tolist()
                        step_incident_e.extend( all_ine )
                        
                        all_e = np.array( x_mean[shower_idx,:,0].cpu().numpy().copy() ).reshape(-1,1)
                        total_deposited_energy = np.sum( all_e )
                        all_e = all_e.flatten().tolist()
                        step_hit_e.extend( all_e )
                        step_deposited_energy.extend( [total_deposited_energy] )
                        
                        all_x = np.array( x_mean[shower_idx,:,1].cpu().numpy().copy() ).reshape(-1,1)
                        av_x_position = np.mean( all_x )
                        all_x = all_x.flatten().tolist()
                        step_hit_x.extend(all_x)
                        step_av_x_pos.extend( [av_x_position] )
                        
                        all_y = np.array( x_mean[shower_idx,:,2].cpu().numpy().copy() ).reshape(-1,1)
                        av_y_position = np.mean( all_y )
                        all_y = all_y.flatten().tolist()
                        step_hit_y.extend(all_y)
                        step_av_y_pos.extend( [av_y_position] )
                    
                    self.incident_e_stages[diffusion_step_].extend(step_incident_e)
                    self.hit_energy_stages[diffusion_step_].extend(step_hit_e)
                    self.hit_x_stages[diffusion_step_].extend(step_hit_x)
                    self.hit_y_stages[diffusion_step_].extend(step_hit_y)
                    self.deposited_energy_stages[diffusion_step_].extend(step_deposited_energy)
                    self.av_x_stages[diffusion_step_].extend(step_av_x_pos)
                    self.av_y_stages[diffusion_step_].extend(step_av_y_pos)
                       
                diffusion_step_+=1
                
        # Do not include noise in last step
        x_mean = x_mean
        return x_mean
    
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

def generate_hits(prob, xbin, ybin, x_vals, n_features, device='cpu'):
    '''
    prob = 2D PDF of nhits vs incident energy
    x/ybin = histogram bins
    x_vals = sample of incident energies (sampled from GEANT4)
    n_features = # of feature dimensions e.g. (E,X,Y,Z) = 4
    Returns:
    pred_nhits = array of nhit values, one for each shower
    y_pred = array of tensors (one for each shower) of initial noise values for features of each hit, sampled from normal distribution
    '''
    # bin index each incident energy falls into
    ind = np.digitize(x_vals, xbin) - 1
    ind[ind==len(xbin)-1] = len(xbin)-2
    ind[ind==-1] = 0
    # Construct list of nhits for given incident energies
    prob_ = prob[ind,:]
    
    y_pred = []
    pred_nhits = []
    for i in range(len(prob_)):
        nhits = int(random_sampler(prob_[i],ybin + 1))
        pred_nhits.append(nhits)
        # Generate random values for features in all hits
        ytmp = torch.normal(0,1,size=(nhits, n_features), device=device)
        y_pred.append( ytmp )
    return pred_nhits, y_pred