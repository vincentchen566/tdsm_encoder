import time, functools, torch, os, sys, random, fnmatch, psutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
import util.data_utils, util.display, util.score_model, util.sdes, util.transfer_learning, util.samplers, util.Evaluate
from pickle import load
from IPython import display
import optparse, argparse
import torch.optim.lr_scheduler as lr_scheduler
from termcolor import colored

def training(padding_value,
             dataset,
             preproc_dataset_name,
             keyword,
             dataset_dir,
             working_dir,
             initial_model,
             SDE,
             embed_dim,
             hidden_dim,
             num_encoder_blocks,
             num_attn_heads,
             dropout_gen,
             mask_diff,
             train_ratio,
             batch_size,
             n_epochs,
             initial_lr,
             transfer_learning_series,
             postfix_,
             sampler_steps,
             n_showers_2_gen,
             serialized_model,
             cp_chunks):
  
  ###################
  ##  Environment  ##
  ###################
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"
  os.system('nvidia-smi')
  dataset_store_path = os.path.join(dataset_dir, preproc_dataset_name) 
  print(postfix_)
  ###########
  ##  SDE  ##
  ###########
  if SDE == 'VP':
    beta_max = 1.0
    beta_min = 0.001
    sde = util.sdes.VPSDE(beta_max = beta_max, beta_min = beta_min, device=device)
  elif SDE == 'VE':
    sigma_max = 20.0
    sigma_min = 0.1
    sde = util.sdes.VESDE(sigma_max = sigma_max, sigma_min = sigma_min, device=device)
  marginal_prob_std_fn = functools.partial(sde.marginal_prob)
  diffusion_coeff_fn   = functools.partial(sde.sde)
  ########################
  ##  Model Parameters  ##
  ######################## 

  n_feat_dim = 4

  #########################
  ##  Print Information  ##
  #########################
  print('torch version: ', torch.__version__)
  print('Running on device: ', device)
  if torch.cuda.is_available():
    print('Cuda used to build pyTorch: ', torch.version.cuda)
    print('Current device: ', torch.cuda.current_device())
    print('Cuda arch list: ', torch.cuda.get_arch_list())
  print('Working directory: ', working_dir)

  #################
  ##  File List  ##
  #################
  files_list_ = []
  for filename in os.listdir(dataset_store_path):
    if fnmatch.fnmatch(filename, dataset + keyword):
      files_list_.append(os.path.join(dataset_store_path, filename))
  print('file list: ', files_list_)

  #####################
  ##  Initial_Model  ##
  #####################
  if True: #not os.path.exists(initial_model):
      if not serialized_model:
        model = util.score_model.Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
      else:
        model = util.score_model.get_seq_model(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
      torch.save(model.state_dict(), initial_model)

  ######################
  ##  Memory Control  ##
  ######################
  Tune_cp_chunks = True
  while serialized_model and Tune_cp_chunks:
    custom_data = util.data_utils.cloud_dataset(files_list_[0], device=device)
    try:
        for i, (shower_data, incident_energies) in enumerate(DataLoader(custom_data, batch_size=int(batch_size*1.2))): # Preserve 20% memory to buffer
            shower_data.to(device)
            model.to(device, shower_data.dtype)
            incident_energies = incident_energies.to(device)
            # Loss average for each batch
            loss = util.score_model.loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value, device=device, serialized_model=serialized_model, cp_chunks=cp_chunks)
            Tune_cp_chunks = False
            print(loss)
            break
    except Exception as error:
        print(colored('[Error Occur] {}'.format(error), 'yellow'))
        if 'CUDA out of memory' in str(error):
            cp_chunks += 1
            print(colored("[Solution] Tune gradient check point number from %d to %d"%(cp_chunks-1, cp_chunks), 'blue'))
        else:
            print(colored("[Break] Memory out of use and number of gradient check points already saturated. Please decrease batch size", 'red', attrs=['bold']))
            return
  torch.no_grad()
  torch.cuda.empty_cache()



  ################
  ##  Training  ##
  ################

  torch.cuda.empty_cache()
  if not serialized_model:
    model = util.score_model.Gen(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
  else:
    model = util.score_model.get_seq_model(n_feat_dim, embed_dim, hidden_dim, num_encoder_blocks, num_attn_heads, dropout_gen, marginal_prob_std=marginal_prob_std_fn)
  state_dict = torch.load(os.path.join(working_dir, initial_model))
  model.load_state_dict(state_dict)

  optimiser = RAdam(model.parameters(), lr=initial_lr)
  scheduler = lr_scheduler.ExponentialLR(optimiser, gamma=0.99)
  table = PrettyTable(['Module name', 'Parameters listed'])
  t_params = 0
  for name_, para_ in model.named_parameters():
    if not para_.requires_grad: continue
    param = para_.numel()
    table.add_row([name_, param])
    t_params += param
  print(table)
  print(f'Sum of trainable parameters: {t_params}')

  model_final = util.transfer_learning.transfer_learning(working_dir, preproc_dataset_name, files_list_, initial_lr,
                                           n_epochs, train_ratio, batch_size, model, optimiser, scheduler,
                                           marginal_prob_std_fn, padding_value, transfer_learning_series=transfer_learning_series,
                                           device=device, mask_diff=mask_diff, postfix_=postfix_, 
                                           serialized_model=serialized_model, cp_chunks=cp_chunks)
  ################
  ##  Sampling  ##
  ################

  output_directory = os.path.join(working_dir, 'record_' + postfix_, 'sampling')
  sampling_output_directory = output_directory
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  geant_hit_energies = []
  geant_hit_x = []
  geant_hit_y = []
  geant_deposited_energy = []
  geant_x_pos = []
  geant_y_pos = []
  geant_ine   = np.array([])
  N_geant_showers = 0

  n_files = len(files_list_)
  nshowers_per_file = [n_showers_2_gen//n_files for x in range(n_files)] #TODO: not true if we have multiple files
  r_ =  n_showers_2_gen % nshowers_per_file[0]
  nshowers_per_file[-1] += r_
  print(f'# showers per file: {nshowers_per_file}')
  shower_counter = 0

  sample_ = []
  sampler = util.samplers.pc_sampler(sde=sde, padding_value=padding_value, snr=0.16, sampler_steps=sampler_steps, device=device, jupyternotebook=False, serialized_model=serialized_model)
  for file_idx in range(len(files_list_)):
    n_valid_hits_per_shower = np.array([])
    incident_e_per_shower = np.array([])
  
    max_hits = -1
    file = files_list_[file_idx]
    print(f'file: {file}')
    shower_counter = 0

    custom_data = util.data_utils.cloud_dataset(file, device=device)
    point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)

    for i, (shower_data, incident_energies) in enumerate(point_clouds_loader, 0):
      valid_event = []
      data_np     = shower_data.cpu().numpy().copy()
      energy_np   = incident_energies.cpu().numpy().copy()
      masking     = data_np[:,:,0] != padding_value
      for j in range(len(data_np)):
        valid_hits = data_np[j]
        n_valid_hits = data_np[j][masking[j]]
        if len(valid_hits) > max_hits:
          max_hits = len(valid_hits)
        n_valid_hits_per_shower = np.append(n_valid_hits_per_shower, len(n_valid_hits))
        incident_e_per_shower   = np.append(incident_e_per_shower, energy_np[j])

        if shower_counter >= nshowers_per_file[file_idx]:
          break
        else:
          shower_counter += 1
          all_ine = energy_np[j].reshape(-1,1)
          all_ine = all_ine.flatten()
          all_ine = all_ine.tolist()
          geant_ine = np.append(geant_ine, all_ine[0])
     
          all_e = valid_hits[:,0].reshape(-1,1)
          all_e = all_e.flatten().tolist()
          geant_hit_energies.extend(all_e)
          geant_deposited_energy.extend( [sum( all_e)])
  
          all_x = valid_hits[:,1].reshape(-1,1)
          all_x = all_x.flatten().tolist()
          geant_hit_x.extend(all_x)
          geant_x_pos.extend( [np.sum(all_x)])
          
          all_y = valid_hits[:,2].reshape(-1,1)
          all_y = all_y.flatten().tolist()
          geant_hit_y.extend(all_y)
          geant_y_pos.extend( [np.sum(all_y)])

        N_geant_showers += 1
    del custom_data
    print(f'max_hits: {max_hits}')

    n_valid_hits_per_shower = np.array(n_valid_hits_per_shower)
    incident_e_per_shower   = np.array(incident_e_per_shower)

    n_bins_prob_dist = 20
    e_vs_nhits_prob, x_bin, y_bin = util.samplers.get_prob_dist(incident_e_per_shower, n_valid_hits_per_shower, n_bins_prob_dist)

    in_energies = torch.from_numpy(np.random.choice( incident_e_per_shower, nshowers_per_file[file_idx]))
    if file_idx ==0:
      sampled_ine = in_energies
    else:
      sampled_ine = torch.cat([sampled_ine, in_energies])

    nhits, gen_hits = util.samplers.generate_hits(e_vs_nhits_prob, x_bin, y_bin, in_energies, 4, device=device)
    torch.save([gen_hits, in_energies], os.path.join(working_dir, 'tmp.pt'))
    gen_hits = util.data_utils.cloud_dataset(os.path.join(working_dir, 'tmp.pt'), device=device)
    gen_hits.max_nhits = max_hits
    gen_hits.padding(value=padding_value)
    gen_hits_loader = DataLoader(gen_hits, batch_size=batch_size, shuffle=False)
    os.system('rm {}'.format(os.path.join(working_dir, 'tmp.pt')))

    sample = []
    for i, (gen_hit, sampled_energies) in enumerate(gen_hits_loader,0):
      print(f'Batch: {i}')
      print(f'Generation batch{i}: showers per batch: {gen_hit.shape[0]}, max. hits per shower: {gen_hit.shape[1]}, features per hit: {gen_hit.shape[2]}, sample_energies:{len(sampled_energies)}')
      sys.stdout.write('\r')
      sys.stdout.write('Progress: %d/%d'%((i+1), len(gen_hits_loader)))
      sys.stdout.flush()

      generative = sampler(model_final, sampled_energies, gen_hit, batch_size=gen_hit.shape[0], diffusion_on_mask=mask_diff)
      if i == 0:
        sample = generative
      else:
        sample = torch.cat([sample, generative])
      print(f'sample: {sample.shape}')

    sample_np = sample.cpu().numpy()
    for i in range(len(sample_np)):
      tmp_sample = sample_np[i]
      sample_.append(torch.tensor(tmp_sample))
  torch.save([sample_, sampled_ine], os.path.join(output_directory, 'sample.pt'))

  ##############################
  ##  Plot Generative result  ##
  ##############################

  plot_file_name = os.path.join(output_directory, 'sample.pt')
  custom_data    = util.data_utils.cloud_dataset(plot_file_name, device=device)
  dists_gen      = util.display.plot_distribution(custom_data, nshowers_2_plot=n_showers_2_gen, padding_value=padding_value, masking=True)

  entries_gen    = dists_gen[0]
  all_incident_e_gen = dists_gen[1]
  total_deposited_e_shower_gen = dists_gen[2]
  all_e_gen = dists_gen[3]
  all_x_gen = dists_gen[4]
  all_y_gen = dists_gen[5]
  all_z_gen = dists_gen[6]
  all_hit_ine_gen = dists_gen[7]
  average_x_shower_gen = dists_gen[8]
  average_y_shower_gen = dists_gen[9]

  dists = util.display.plot_distribution(files_list_, nshowers_2_plot=n_showers_2_gen, padding_value=padding_value, masking=True)
  
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
  print(f'entries: {len(entries)}')
  print(f'entries_gen: {len(entries_gen)}')

  bins = np.histogram(np.hstack((entries, entries_gen)), bins=25)[1]
  fig, ax = plt.subplots(3,3, figsize=(12,12))
  ax[0][0].set_ylabel('# entries')
  ax[0][0].set_xlabel('Hit entries')
  ax[0][0].hist(entries, bins, alpha=0.5, color='orange', label='Geant4')
  ax[0][0].hist(entries_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[0][0].legend(loc='upper right')

  bins = np.linspace(0,1,25)
  ax[0][1].set_ylabel('# entries')
  ax[0][1].set_xlabel('Hit energy [GeV]')
  ax[0][1].hist(all_e, bins, alpha=0.5, color='orange', label='Geant4')
  ax[0][1].hist(all_e_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[0][1].legend(loc='upper right')

  ax[0][2].set_ylabel('# entries')
  ax[0][2].set_xlabel('Hit x position')
  ax[0][2].hist(all_x, bins, alpha=0.5, color='orange', label='Geant4')
  ax[0][2].hist(all_x_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[0][2].legend(loc='upper right')

  ax[1][0].set_ylabel('# entries')
  ax[1][0].set_xlabel('Hit y position')
  ax[1][0].hist(all_y, bins, alpha=0.5, color='orange', label='Geant4')
  ax[1][0].hist(all_y_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[1][0].legend(loc='upper right')

  ax[1][1].set_ylabel('# entries')
  ax[1][1].set_xlabel('Hit z position')
  ax[1][1].hist(all_z, bins, alpha=0.5, color='orange', label='Geant4')
  ax[1][1].hist(all_z_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[1][1].legend(loc='upper right')
  
  bins = np.histogram(np.hstack((all_incident_e, all_incident_e_gen)), bins=25)[1]
  ax[1][2].set_ylabel('# entries')
  ax[1][2].set_xlabel('Incident energies [GeV]')
  ax[1][2].hist(all_incident_e, bins, alpha=0.5, color='orange', label='Geant4')
  ax[1][2].hist(all_incident_e_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[1][2].legend(loc='upper right')

  bins = np.histogram(np.hstack((all_hit_ine_geant, all_hit_ine_gen)), bins=25)[1]
  ax[2][0].set_ylabel('# entries')
  ax[2][0].set_xlabel('Total deposited energy [GeV]')
  ax[2][0].hist(all_hit_ine_geant, bins, alpha=0.5, color='orange', label='Geant4')
  ax[2][0].hist(all_hit_ine_gen,   bins, alpha=0.5, color='blue', label='Gen')
  ax[2][0].legend(loc='upper right')
 
  bins = np.histogram(np.hstack((average_x_shower_geant, average_x_shower_gen)), bins=25)[1]
  ax[2][1].set_ylabel('# entries')
  ax[2][1].set_xlabel('Shower Sum X')
  ax[2][1].hist(average_x_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
  ax[2][1].hist(average_x_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[2][1].legend(loc='upper right')

  bins = np.histogram(np.hstack((average_y_shower_geant, average_y_shower_gen)), bins=25)[1]
  ax[2][2].set_ylabel('# entries')
  ax[2][2].set_xlabel('Shower Sum Y')
  ax[2][2].hist(average_y_shower_geant, bins, alpha=0.5, color='orange', label='Geant4')
  ax[2][2].hist(average_y_shower_gen, bins, alpha=0.5, color='blue', label='Gen')
  ax[2][2].legend(loc='upper right')

  fig.show()
  fig_name = os.path.join(output_directory, 'Geant_Gen_comparison.png')
  fig.savefig(fig_name)

  ############################
  ##  Performance Checking  ##
  ############################
  performance_output_directory = os.path.join(working_dir, 'record_' + postfix_, 'performance')
  if not os.path.exists(performance_output_directory):
      os.system('mkdir -p {}'.format(performance_output_directory))
  # TODO base_dataset from file to list
  evaluator = util.Evaluate.evaluator(base_dataset_name = files_list_[0],
                                      gen_dataset_name   = os.path.join(sampling_output_directory, "sample.pt"),
                                      device = device,
                                      digitize=False)

  evaluator.draw_distribution(performance_output_directory)
  indices = [0,1,2,3] # 0: e, 1: x, 2: y, 3:z (choose parameters set used for training)
  model = util.Evaluate.Classifier(n_dim=len(indices),
                                   embed_dim=64,
                                   hidden_dim=64,
                                   n_layers=2,
                                   n_layers_cls=1,
                                   n_heads=2,
                                   dropout=0)

  evaluator.separate_ttv(0.8,0.1)
  evaluator.train(
            model=model,
            jupyternotebook = False,
            mask = True,
            n_epochs = 150,
            device = device,
            indices = indices,
            output_directory=performance_output_directory)
  performance_score = evaluator.evulate_score(model = model, indices = indices, output_directory=performance_output_directory)
  print(colored('accuracy score (maximum 1.0, meaning the classifer can fully distinguish gen & geant4 data): {}'.format(performancei_score), 'green'))

if __name__ == '__main__':

  usage = 'usage: %prog [options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--padding_value', dest='padding_value', default = 0.0, type=float)
  parser.add_argument('--dataset', dest='dataset', default = 'dataset_2_padded_nentry', type=str)
  parser.add_argument('--preproc_name', dest='preproc_name', default = 'tanh_transform', type=str)
  parser.add_argument('--keyword', dest='keyword', default = '11*.pt', type=str)
  parser.add_argument('--dataset_dir', dest='dataset_dir', default = '/eos/user/t/tihsu/SWAN_projects/ML_hackthon_transferlearning/tdsm_encoder/datasets/', type=str)
  parser.add_argument('--working_dir', dest='working_dir', default = './', type=str)
  parser.add_argument('--initial_model', dest='initial_model', default='initial_model.pt', type=str)
  parser.add_argument('--SDE', dest='SDE', default='VP', type=str)
  parser.add_argument('--n_epochs', dest='n_epochs', default=[3], nargs='+', type=int)
  parser.add_argument('--embed_dim', dest='embed_dim', default=512, type=int)
  parser.add_argument('--hidden_dim', dest='hidden_dim', default=128, type=int)
  parser.add_argument('--num_encoder_blocks', dest='num_encoder_blocks', default=8, type=int)
  parser.add_argument('--num_attn_heads', dest='num_attn_heads', default=16, type=int)
  parser.add_argument('--dropout_gen', dest='dropout_gen', default=0, type=float)
  parser.add_argument('--mask_diff', action='store_true')
  parser.add_argument('--serialized_model', action='store_true')
  parser.add_argument('--cp_chunks', default=0, type=int)
  parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
  parser.add_argument('--batch_size', dest='batch_size', default=16, type=int)
  parser.add_argument('--initial_lr', dest='initial_lr', default=1e-3, type=float)
  parser.add_argument('--transfer_learning_series', dest='transfer_learning_series', default=[-0.1], nargs='+')
  parser.add_argument('--postfix', dest='postfix', default='test', type=str)
  parser.add_argument('--sampler_steps', dest='sampler_steps', default=200, type=int)
  parser.add_argument('--n_showers_2_gen', dest='n_showers_2_gen', default=500, type=int)
  args = parser.parse_args()

  padding_value = args.padding_value
  dataset       = args.dataset
  preproc_dataset_name = args.preproc_name
  keyword       = args.keyword
  dataset_dir   = args.dataset_dir
  working_dir   = args.working_dir
  initial_model = args.initial_model
  SDE           = args.SDE
  embed_dim     = args.embed_dim
  hidden_dim    = args.hidden_dim
  num_encoder_blocks = args.num_encoder_blocks
  num_attn_heads= args.num_attn_heads
  dropout_gen   = args.dropout_gen
  mask_diff     = args.mask_diff
  train_ratio   = args.train_ratio
  batch_size    = args.batch_size
  n_epochs      = args.n_epochs
  initial_lr    = args.initial_lr
  transfer_learning_series = args.transfer_learning_series
  postfix       = args.postfix
  sampler_steps = args.sampler_steps
  n_showers_2_gen = args.n_showers_2_gen
  serialized_model = args.serialized_model
  cp_chunks     = args.cp_chunks

  print('series', transfer_learning_series)
  training(padding_value,
             dataset,
             preproc_dataset_name,
             keyword,
             dataset_dir,
             working_dir,
             initial_model,
             SDE,
             embed_dim,
             hidden_dim,
             num_encoder_blocks,
             num_attn_heads,
             dropout_gen,
             mask_diff,
             train_ratio,
             batch_size,
             n_epochs,
             initial_lr,
             transfer_learning_series,
             postfix_ = postfix,
             sampler_steps = sampler_steps,
             n_showers_2_gen = n_showers_2_gen,
             serialized_model=serialized_model,
             cp_chunks=cp_chunks) 
