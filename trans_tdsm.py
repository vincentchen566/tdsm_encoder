import time, functools, torch, os, sys, random, fnmatch, psutil, argparse, tqdm, yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Pytorch libs
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader

# WandB setup
import wandb
os.environ['WANDB_NOTEBOOK_NAME'] = 'NCSM_condor'
wandb.login()

# TDSM libs
# Adds util directory to directories interpreter will search for modules
#sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/util')
sys.path.insert(1, 'util') # Local path / user independent
import data_utils as utils
import score_model as score_model
import sdes as sdes
import display
import samplers as samplers
import Convertor as Convertor
from Convertor import Preprocessor


def train_log(loss, batch_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=batch_ct)

def build_dataset(filename, train_ratio, batch_size, device):
    # Build dataset
    custom_data = utils.cloud_dataset(filename, device=device)
    train_size = int(train_ratio * len(custom_data.data))
    test_size = len(custom_data.data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])
    shower_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    shower_loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return shower_loader_train, shower_loader_test

def check_mem():
    # Resident set size memory (non-swap physical memory process has used)
    process = psutil.Process(os.getpid())
    # Print bytes in GB
    print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))
    return

def train_model(files_list_, device='cpu'):

    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config
    print(f'training config: {config}')

    wd = os.getcwd()
    #wd = '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/'
    output_files = 'training_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
    output_directory = os.path.join(wd, output_files)
    print('Training directory: ', output_directory)
    if not os.path.exists(output_directory):
        print(f'Making new dir . . . . . ')
        os.makedirs(output_directory)

    # Instantiate stochastic differential equation
    if config.SDE == 'subVP':
        sde = sdes.subVPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VP':
        sde = sdes.VPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VE':
        sde = sdes.VESDE(sigma_max=config.sigma_max,device=device)
    marginal_prob_std_fn = functools.partial(sde.marginal_prob)

    # Instantiate model
    loss_fn = score_model.ScoreMatchingLoss()
    model = score_model.Gen(config.n_feat_dim, config.embed_dim, config.hidden_dim, config.num_encoder_blocks, config.num_attn_heads, config.dropout_gen, marginal_prob_std=marginal_prob_std_fn)

    table = PrettyTable(['Module name', 'Parameters listed'])
    t_params = 0
    for name_ , para_ in model.named_parameters():
        if not para_.requires_grad: continue
        param = para_.numel()
        table.add_row([name_, param])
        t_params+=param
    print(table)
    print(f'Sum of trainable parameters: {t_params}')    
    
    if torch.cuda.device_count() > 1:
        print(f'Lets use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    # Optimiser needs to know model parameters for to optimise
    optimiser = RAdam(model.parameters(),lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.99)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    
    eps_ = []
    batch_ct = 0
    for epoch in range(0, config.epochs ):
        sys.stdout.write('\r')
        sys.stdout.write('Progress: %d/%d'%((epoch+1), config.epochs)) # Local Progress Tracker
        sys.stdout.flush()
        eps_.append(epoch)

        # Create/clear per epoch variables
        cumulative_epoch_loss = 0.
        file_counter = 0
        training_batches_per_epoch = 0
        testing_batches_per_epoch = 0

        # Load files
        for filename in files_list_:
            file_counter+=1

            # Build dataset
            shower_loader_train, shower_loader_test = build_dataset(filename, config.train_ratio, config.batch_size, device)

            # Accumuate number of batches per epoch
            training_batches_per_epoch += len(shower_loader_train)
            testing_batches_per_epoch += len(shower_loader_test)
            
            # Load shower batch for training
            for i, (shower_data,incident_energies) in enumerate(shower_loader_train,0):
                batch_ct+=1
                # Move model to device and set dtype as same as data (note torch.double works on both CPU and GPU)
                model.to(device, shower_data.dtype)
                model.train()
                shower_data.to(device)
                incident_energies.to(device)
                if len(shower_data) < 1:
                    continue

                # Zero any gradients from previous steps
                optimiser.zero_grad()
                # Loss average for each batch
                loss = loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value=0.0, device=device, diffusion_on_mask=False)
                # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
                loss.backward()
                cumulative_epoch_loss+=loss.item()
                # Update value of x += -lr * x.grad
                optimiser.step()
                # Report metrics every 5th batch
                if ((batch_ct + 1) % 5) == 0:
                    train_log(loss, batch_ct, epoch)
            
            # Testing on subset of file
            for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
                with torch.no_grad():
                    model.to(device, shower_data.dtype)
                    model.eval()
                    shower_data = shower_data.to(device)
                    incident_energies = incident_energies.to(device)
                    test_loss = score_model.loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value=0.0, device=device)

        scheduler.step()
        
        # Save checkpoints
        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(output_directory, 'ckpt_tmp_'+str(epoch)+'.pth' ))
    
    save_name = os.path.join(output_directory, 'ckpt_tmp_'+str(epoch)+'.pth' )
    torch.save(model.state_dict(), save_name)
    return save_name


def generate(files_list_, load_filename, device='cpu'):

    wd = os.getcwd()
    output_file = 'sampling_'+datetime.now().strftime('%Y%m%d_%H%M')+'_output/'
    output_directory = os.path.join(wd, output_file)
    print('Sampling directory: ', output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    config = wandb.config

    # Instantiate stochastic differential equation
    if config.SDE == 'subVP':
        sde = sdes.subVPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VP':
        sde = sdes.VPSDE(beta_max=config.sigma_max, beta_min=config.sigma_min, device=device)
    if config.SDE == 'VE':
        sde = sdes.VESDE(sigma_max=config.sigma_max,device=device)
    marginal_prob_std_fn = functools.partial(sde.marginal_prob)
    diffusion_coeff_fn = functools.partial(sde.sde)

    # Load saved model
    model=score_model.Gen(config.n_feat_dim, config.embed_dim, config.hidden_dim, config.num_encoder_blocks, config.num_attn_heads, config.dropout_gen, marginal_prob_std=marginal_prob_std_fn)
    if load_filename == '':
        load_name = os.path.join(wd,'training_20240408_1350_output/ckpt_tmp_299.pth')
    else:
        load_name = os.path.join(wd,load_filename)

    model.load_state_dict(torch.load(load_name, map_location=device))
    model.to(device)


    geant_deposited_energy = []
    geant_x_pos = []
    geant_y_pos = []
    geant_ine = np.array([])
    N_geant_showers = 0

    # Step to plot
    N_steps_2_plot = 5
    plotsteps = []
    tmp_step = 0
    plotsteps.append(tmp_step)
    stepsize=round(config.sampler_steps/N_steps_2_plot, 0)
    for i in range(N_steps_2_plot):
        tmp_step = tmp_step+stepsize
        plotsteps.append(tmp_step)
    
    n_files = len(files_list_)
    print(f'n_files: {n_files}')
    nshowers_per_file = [config.n_showers_2_gen//n_files for x in range(n_files)]
    r_ = config.n_showers_2_gen % nshowers_per_file[0]
    nshowers_per_file[-1] = nshowers_per_file[-1]+r_
    shower_counter = 0

    # create list to store final samples
    sample_ = []
    # instantiate sampler 
    sampler = samplers.pc_sampler(sde=sde, padding_value=0.0, snr=0.16, sampler_steps=config.sampler_steps, steps2plot=plotsteps, device=device, jupyternotebook=False)

    # Collect Geant4 shower information
    for file_idx in range(len(files_list_)):

        # N valid hits used for 2D PDF
        n_valid_hits_per_shower = np.array([])
        # Incident particle energy for 2D PDF
        incident_e_per_shower = np.array([])

        max_hits = -1
        file = files_list_[file_idx]
        print(f'file: {file}')
        shower_counter = 0

        # Load shower data
        custom_data = utils.cloud_dataset(file, device=device)
        point_clouds_loader = DataLoader(custom_data, batch_size=config.batch_size, shuffle=True)
        # Loop over batches
        for i, (shower_data, incident_energies) in enumerate(point_clouds_loader,0):
            # Copy data
            valid_event = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()

            # Mask for padded values (padded values set to 0)
            masking = data_np[:,:,0] != 0.0

            # Loop over each shower in batch
            for j in range(len(data_np)):

                # valid hits for shower j in batch used for GEANT plot distributions
                valid_hits = data_np[j]

                # real (unpadded) hit multiplicity needed for the 2D PDF later
                n_valid_hits = data_np[j][masking[j]]

                n_valid_hits_per_shower = np.append(n_valid_hits_per_shower, len(n_valid_hits))
                if len(valid_hits)>max_hits:
                    max_hits = len(valid_hits)

                incident_e_per_shower = np.append(incident_e_per_shower, energy_np[j])

                # ONLY for plotting purposes
                if shower_counter >= nshowers_per_file[file_idx]:
                    break
                else:
                    shower_counter+=1

                    all_ine = energy_np[j].reshape(-1,1)

                    # Rescale the conditional input for each shower
                    all_ine = all_ine.flatten().tolist()
                    geant_ine = np.append(geant_ine,all_ine[0])
                    
                    all_e = valid_hits[:,0].reshape(-1,1)
                    all_e = all_e.flatten().tolist()
                    geant_deposited_energy.append( sum( all_e ) )
                    
                    all_x = valid_hits[:,1].reshape(-1,1)
                    all_x = all_x.flatten().tolist()
                    geant_x_pos.append( np.mean(all_x) )
                    
                    all_y = valid_hits[:,2].reshape(-1,1)
                    all_y = all_y.flatten().tolist()
                    geant_y_pos.append( np.mean(all_y) )

                N_geant_showers+=1
        del custom_data

        # Arrays of Nvalid hits in showers, incident energies per shower
        n_valid_hits_per_shower = np.array(n_valid_hits_per_shower)
        incident_e_per_shower = np.array(incident_e_per_shower)

        # Generate 2D pdf of incident E vs N valid hits from the training file(s)
        n_bins_prob_dist = 50
        e_vs_nhits_prob, x_bin, y_bin = sampler.get_prob_dist( incident_e_per_shower, n_valid_hits_per_shower, n_bins_prob_dist )

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
        in_energies = torch.from_numpy(np.random.choice( incident_e_per_shower, nshowers_per_file[file_idx] ))
        if file_idx == 0:
            sampled_ine = in_energies
        else:
            sampled_ine = torch.cat([sampled_ine,in_energies])

        # Sample from 2D pdf = nhits per shower vs incident energies -> nhits and a tensor of randomly initialised hit features
        nhits, gen_hits = sampler.generate_hits(e_vs_nhits_prob, x_bin, y_bin, in_energies, 4, device=device)

        # Save
        torch.save([gen_hits, in_energies],'tmp.pt')

        # Load the showers of noise
        gen_hits = utils.cloud_dataset('tmp.pt', device=device)

        #Set the max_nhits according to geant4 data
        gen_hits.max_nhits = max_hits
        
        # Pad showers with values of 0
        gen_hits.padding(0.0)
        # Load len(gen_hits_loader) number of batches each with batch_size number of showers
        gen_hits_loader = DataLoader(gen_hits, batch_size=config.batch_size, shuffle=False)

        # Remove noise shower file
        os.system("rm tmp.pt")

        # Create instance of sampler
        sample = []
        # Loop over each batch of noise showers
        print(f'# batches: {len(gen_hits_loader)}' )
        for i, (gen_hit, sampled_energies) in enumerate(gen_hits_loader,0):
            print(f'Generation batch {i}: showers per batch: {gen_hit.shape[0]}, max. hits per shower: {gen_hit.shape[1]}, features per hit: {gen_hit.shape[2]}, sampled_energies: {len(sampled_energies)}')    
            sys.stdout.write('\r')
            sys.stdout.write("Progress: %d/%d \n" % ((i+1), len(gen_hits_loader)))
            sys.stdout.flush()
            
            # Run reverse diffusion sampler
            #generative = sampler(model, marginal_prob_std_fn, diffusion_coeff_fn, sampled_energies, gen_hit, batch_size=gen_hit.shape[0], energy_trans_file=energy_trans_file, x_trans_file=x_trans_file , y_trans_file = y_trans_file, ine_trans_file=ine_trans_file)
            generative = sampler(model, sampled_energies, gen_hit, batch_size=gen_hit.shape[0])
            # Create first sample or concatenate sample to sample list
            if i == 0:
                sample = generative
            else:
                sample = torch.cat([sample,generative])
            
            print(f'sample: {sample.shape}')
            
        sample_np = sample.cpu().numpy()

        for i in range(len(sample_np)):
            tmp_sample = sample_np[i]
            sample_.append(torch.tensor(tmp_sample))
    
    print(f'sample_: {len(sample_)}, sampled_ine: {len(sampled_ine)}')
    sample_savename = os.path.join(output_directory, 'sample.pt')
    torch.save([sample_,sampled_ine], sample_savename)

    gen_data = utils.cloud_dataset(sample_savename,device=device)
    # Generated distributions
    dists_gen = display.plot_distribution(gen_data, nshowers_2_plot=config.n_showers_2_gen, padding_value=0.0)
    # Distributions object for Geant4 files
    dists = display.plot_distribution(files_list_, nshowers_2_plot=config.n_showers_2_gen, padding_value=0.0)
    comparison_fig = display.comparison_summary(dists, dists_gen, output_directory)#, erange=(-5,3), xrange=(-2.5,2.5), yrange=(-2.5,2.5), zrange=(0,1))
    # Add evaluation plots to keep on wandb
    wandb.log({"summary" : wandb.Image(comparison_fig)})
    return output_directory
def main(config=None):
   
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
    
    print('Working directory: ' , os. getcwd())

    # Useful when debugging gradient issues
    torch.autograd.set_detect_anomaly(True)

    padding_value = 0.0

    # List of training input files
    training_file_path = os.path.join(indir) # change indir to be absolute path
    files_list_ = []
    print(f'Training files found in: {training_file_path}')
    for filename in os.listdir(training_file_path):
        if fnmatch.fnmatch(filename, 'dataset_2_padded_nentry1033To1161.pt'):
            files_list_.append(os.path.join(training_file_path,filename))
    print(f'Files: {files_list_}')

    with wandb.init(config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        #### Input plots ####
        if switches_ & trigger:
            # Limited to n_showers_2_gen showers in for plots
            # Transformed variables
            dists_trans = display.plot_distribution(files_list_, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)
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

            ### 1D histograms
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


        #train_model_name = "/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/training_20230830_1430_output/ckpt_tmp_499.pth" #Default model name 
        #### Training ####
        if switches_>>1 & trigger:
            trained_model_name = train_model(files_list_, device=device)
        
        #### Sampling ####
        if switches_>>2 & trigger:
            # If a new training was run and you want to use it
            if switches_>>1 & trigger:
                output_directory = generate(files_list_, load_filename=trained_model_name, device=device)
            # To use an older training file
            # n.b. you'll need to make sure the config hyperparams are the same as the model being used
            else:
#                trained_model_name = 'training_20240408_1350_output/ckpt_tmp_299.pth'
                trained_model_name = 'training_20240429_1211_output/ckpt_tmp_2.pth'
                output_directory = generate(files_list_, load_filename=trained_model_name, device=device)
            

        #### Evaluation plots ####
        if switches_>>3 & trigger:
            # Distributions object for generated files
            print(f'Generated inputs')
            workingdir = os.getcwd()
            if not switches_>>2 & trigger:
              output_directory = os.path.join(workingdir,'sampling_100samplersteps_20230829_1606_output')
            print(f'Evaluation outputs stored here: {output_directory}')
            plot_file_name = os.path.join(output_directory, 'sample.pt')
            custom_data = utils.cloud_dataset(plot_file_name,device=device)
            # when providing just cloud dataset, energy_trans_file needs to include full path
            dists_gen = display.plot_distribution(custom_data, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)

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
            dists = display.plot_distribution(files_list_, nshowers_2_plot=config.n_showers_2_gen, padding_value=padding_value)

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
            #ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            print('Plot hit x')
            bins=np.histogram(np.hstack((all_x,all_x_gen)), bins=50)[1]
            ax[0][2].set_ylabel('# entries')
            ax[0][2].set_xlabel('Hit x position')
            ax[0][2].hist(all_x, bins, alpha=0.5, color='orange', label='Geant4')
            ax[0][2].hist(all_x_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[0][2].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            print('Plot hit y')
            bins=np.histogram(np.hstack((all_y,all_y_gen)), bins=50)[1]
            ax[1][0].set_ylabel('# entries')
            ax[1][0].set_xlabel('Hit y position')
            ax[1][0].hist(all_y, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][0].hist(all_y_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            print('Plot hit z')
            bins=np.histogram(np.hstack((all_z,all_z_gen)), bins=50)[1]
            ax[1][1].set_ylabel('# entries')
            ax[1][1].set_xlabel('Hit z position')
            ax[1][1].hist(all_z, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][1].hist(all_z_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][1].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            print('Plot incident energies')
            bins=np.histogram(np.hstack((all_incident_e,all_incident_e_gen)), bins=50)[1]
            ax[1][2].set_ylabel('# entries')
            ax[1][2].set_xlabel('Incident energies [GeV]')
            ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
            ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')

            print('Plot total deposited hit energy')
            bins=np.histogram(np.hstack((total_deposited_e_shower,total_deposited_e_shower_gen)), bins=50)[1]
            ax[2][0].set_ylabel('# entries')
            ax[2][0].set_xlabel('Deposited energy [GeV]')
            ax[2][0].hist(total_deposited_e_shower, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][0].hist(total_deposited_e_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][0].set_yscale('log')
            ax[2][0].legend(loc='upper right')

            print('Plot average hit X position')
            bins=np.histogram(np.hstack((average_x_shower_geant,average_x_shower_gen)), bins=50)[1]
            ax[2][1].set_ylabel('# entries')
            ax[2][1].set_xlabel('Average X pos.')
            ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][1].set_yscale('log')
            ax[2][1].legend(loc='upper right')

            print('Plot average hit Y position')
            bins=np.histogram(np.hstack((average_y_shower_geant,average_y_shower_gen)), bins=50)[1]
            ax[2][2].set_ylabel('# entries')
            ax[2][2].set_xlabel('Average Y pos.')
            ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
            ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
            #ax[2][2].set_yscale('log')
            ax[2][2].legend(loc='upper right')

            fig_name = os.path.join(output_directory, 'Geant_Gen_comparison.png')
            print(f'Figure name: {fig_name}')
            fig.savefig(fig_name)



            # Convert Generated file
            Converter_ = Convertor.Convertor(plot_file_name, 0.0, preprocessor=args.preprocessor)
#            Converter_ = Convertor.Convertor(files_list_[0], 0.0, preprocessor=args.preprocessor)
            Converter_.invert(-99)
            Converter_.digitize()
            Converter_.to_h5py(os.path.join(output_directory, 'Gen.h5'))
            # Convert Reference file: TODO: multifile management
            Converter_ = Convertor.Convertor(files_list_[0], 0.0, preprocessor=args.preprocessor)
            Converter_.invert(-99)
            Converter_.digitize()
            Converter_.to_h5py(os.path.join(output_directory, 'Reference.h5'))

   
            os.system('python util/evaluate_image_based.py -m avg --output_dir {outdir} --input_file {Gen_file} --reference_file {Geant4_file} --dataset 2'.format(Gen_file = os.path.join(output_directory, 'Gen.h5'), Geant4_file = os.path.join(output_directory, 'Reference.h5'), outdir = os.path.join(output_directory, 'calo_score')))
            wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'reference_average_shower_dataset_2.png'))})
            wandb.log({"summary" : wandb.Image(os.path.join(output_directory, 'calo_score', 'average_shower_dataset_2.png'))})


if __name__=='__main__':

    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-s','--switches',dest='switches', help='Binary representation of switches that run: evaluation plots, training, sampling, evaluation plots', default='0000', type=str)
    argparser.add_argument('-i','--inputs',dest='inputs', help='Path to input directory', default='', type=str)
    argparser.add_argument('-c', '--config', dest='config', help='Configuration file for parameter monitoring (relative path)', default='', type=str)
    argparser.add_argument('-p', '--preprocessor', dest='preprocessor', help='pickle files of preprocessor', default='', type=str)
    args = argparser.parse_args()
    
    # WandB configuration
    cfg_name = args.config
    print(f'Using config: {cfg_name}')

    project_name = cfg_name.split('.')[0].split('_',1)[1]
    print(f'Starting project: {project_name}')

    #if not os.path.exists(cfg_name):
    #    cfg_name = os.path.join('../configs', cfg_name)

    with open(cfg_name) as ymlfile:
        sweep_yml = yaml.safe_load(ymlfile)
    
    # Run main function using sweep agents reading from configs
    # Sweeps run by setting range of parameter values to explore, else set single parameter value
    # Running from yaml files facilitates submitting (several) jobs to condor
    n_runs = 1
    sweep_id = wandb.sweep(sweep_yml, project="NCSM-"+project_name)
    wandb.agent(sweep_id, main, count=n_runs)

