import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time, functools, torch, os, sys, random, fnmatch, psutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,RAdam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
import util.data_utils, util.display, util.score_model, util.memory
import tqdm
from pickle import load
from IPython import display
from datasets.pad_events_threshold import transform_hit_e
import pandas as pd 
from torch.utils.checkpoint import checkpoint_sequential

def training(workingdir, preproc_dataset_name, files_list_, initial_lr, n_epochs, train_ratio, batch_size, model, optimiser, scheduler, marginal_prob_std_fn, padding_value, threshold_value = -0.1, device='cpu', notebook=False, cls_ffnn=False, mask_diff=False, postfix_='', serialized_model=False, cp_chunks=0):

  torch.cuda.empty_cache()
  output_directory = os.path.join(workingdir, 'record_'+postfix_, 'training', preproc_dataset_name)
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
  av_training_losses_per_epoch = []
  av_testing_losses_per_epoch = []

  fig, ax = plt.subplots(ncols=2, figsize=(8,4))
  dh = display.display(fig, display_id=True)
  ax[0].set_ylabel('Loss')
  ax[0].set_xlabel('Epoch')
  ax[0].set_yscale('log')
  ax[1].set_xlabel('lr')
  ax[1].set_xlim(initial_lr*0.99**(n_epochs), initial_lr)
  ax[1].set_xscale('log')
  ax[1].set_yscale('log')
  ax[1].tick_params('both', length=10, width=1, which='both')

  lrs_ = []

  print(files_list_)
  eps_ = []
  if notebook:
    epochs = tqdm.notebook.trange(n_epochs)
  else:
    epochs = range(n_epochs)

  ######################
  ##  Memory Control  ##
  ######################
  mem_log = []
  exp     = 'baseline'

  for epoch in epochs:
    sys.stdout.write('\r')
    sys.stdout.write('Progress: %d/%d'%((epoch+1), n_epochs))
    sys.stdout.flush()
    eps_.append(epoch)
    # Create/clear per epoch variables
    cumulative_epoch_loss = 0.
    cumulative_test_epoch_loss = 0.

    file_counter = 0
    n_training_showers = 0
    n_testing_showers = 0
    training_batches_per_epoch = 0
    testing_batches_per_epoch = 0

    # Load files
    for filename in files_list_:
        if not float(threshold_value) < 0:
          filename = filename.replace('transform','transform_threshold' + str(threshold_value) )
        custom_data = util.data_utils.cloud_dataset(filename, device=device)
        train_size = int(train_ratio * len(custom_data.data))
        test_size = len(custom_data.data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(custom_data, [train_size, test_size])
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
            shower_data = shower_data.to(device)
            incident_energies = incident_energies.to(device)

            if len(shower_data) < 1:
                print('Very few hits in shower: ', len(shower_data))
                continue
            # Zero any gradients from previous steps
            optimiser.zero_grad()
            # Loss average for each batch
            
            # Memory Hook -- Only see first iteration --> 1 iteration = 1 forward + 1 backward in 1 batch
            if len(mem_log) == 0:
              start_time_batch = time.time()
              mem_log_ = mem_log or []
              exp     = exp or f'exp_{len(mem_log_)}'
              hr      = []
              for _idx, _module in enumerate(model.modules()):
                util.memory._add_memory_hooks(_idx, _module, mem_log_, exp, hr)

            loss = util.score_model.loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value, device=device, diffusion_on_mask=mask_diff, serialized_model=serialized_model, cp_chunks=cp_chunks)
            # Accumulate batch loss per epoch
            cumulative_epoch_loss+=float(loss)
            # collect dL/dx for any parameters (x) which have requires_grad = True via: x.grad += dL/dx
            loss.backward()
            # Update value of x += -lr * x.grad
            optimiser.step()

            # Memory Hook -- Release
            if len(mem_log) == 0:
                end_time_batch = time.time()
                process_time_batch = end_time_batch - start_time_batch
                [h.remove() for h in hr]
                mem_log = mem_log_
                df = pd.DataFrame(mem_log)
                print(df)
                util.memory.plot_mem(df, exps=['baseline'], output_file='baseline_memory_plot.png', normalize_mem_all=False, normalize_call_idx=False, time=process_time_batch, batch_size=batch_size)
                return #TODO: Debugging


        # Testing on subset of file
        for i, (shower_data,incident_energies) in enumerate(shower_loader_test,0):
            with torch.no_grad():
                model.eval()
                shower_data = shower_data.to(device)
                incident_energies = incident_energies.to(device)
                test_loss = util.score_model.loss_fn(model, shower_data, incident_energies, marginal_prob_std_fn, padding_value, device=device, diffusion_on_mask=mask_diff)
                cumulative_test_epoch_loss+=float(test_loss)

    # Calculate average loss per epoch
    av_training_losses_per_epoch.append(cumulative_epoch_loss/training_batches_per_epoch)
    av_testing_losses_per_epoch.append(cumulative_test_epoch_loss/testing_batches_per_epoch)
    
    lr_ = optimiser.param_groups[0]['lr']
    if notebook:
      epochs.set_description('Average Loss: {:5f}(Train) {:5f}(Test) {:5f}(lr)'.format(cumulative_epoch_loss/training_batches_per_epoch, cumulative_test_epoch_loss/testing_batches_per_epoch, lr_))
    ax[0].plot(av_training_losses_per_epoch[1:], c='blue', label='training')
    ax[0].plot(av_testing_losses_per_epoch[1:], c='red', label='testing')
    
    # End of epoch, change the learning rate
    before_lr = optimiser.param_groups[0]['lr']
    scheduler.step()
    after_lr = optimiser.param_groups[0]['lr']
    lrs_.append(before_lr)
    ax[1].plot(lrs_[1:], av_training_losses_per_epoch[1:], c='blue')
    if epoch == 0:
        ax[0].legend(loc='upper right')
    if notebook:
      dh.update(fig)
    if n_epochs%5 == 0:
        torch.save(model.state_dict(), os.path.join(output_directory,'ckpt_tmp_'+str(epoch)+'.pth'))


  fig.savefig(os.path.join(output_directory, 'loss_v_epoch.png'))
  torch.save(model.state_dict(), os.path.join(output_directory,'ckpt_tmp_'+str(epoch)+'.pth'))

  util.display.plot_loss_vs_epoch(eps_, av_training_losses_per_epoch, av_testing_losses_per_epoch, odir=output_directory, zoom=True)
  return model, optimiser, scheduler

def transfer_learning(workingdir, preproc_dataset_name, files_list_, initial_lr, n_epochs, train_ratio, batch_size, model, optimiser, scheduler, marginal_prob_std_fn, padding_value, transfer_learning_series, device='cpu', mask_diff=False, notebook=False, postfix_='', serialized_model=False, cp_chunks=0):

  model_tmp = model
  optimiser_tmp = optimiser
  scheduler_tmp = scheduler
  for idx, threshold in enumerate(transfer_learning_series):
    model_tmp, optimiser_tmp, scheduler_tmp = training(workingdir, preproc_dataset_name + "_threshold" + str(threshold), files_list_, optimiser.param_groups[0]['lr'], n_epochs[idx], train_ratio, batch_size, model_tmp, optimiser_tmp, scheduler_tmp, marginal_prob_std_fn, padding_value, threshold, device=device, notebook = notebook, mask_diff=mask_diff, postfix_=postfix_, serialized_model=serialized_model, cp_chunks=cp_chunks)
  return model_tmp
