import torch, utils, sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from typing import Union
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, PowerTransformer
from pickle import load
sys.path.insert(1, '../')

def plot_distribution(files_:Union[ list , utils.cloud_dataset], nshowers_2_plot=5000, padding_value=-20, batch_size=1, energy_trans_file=''):
    
    '''
    files_ = can be a list of input files or a cloud dataset object
    nshowers_2_plot = # showers you want to plot. Limits memory required. Samples evenly from several files if files input is used.
    padding_value = value used for padded entries
    batch_size = # showers to load at a time
    energy_trans_file = pickle file containing fitted input transformation function. Only provide a file name if you want to plot distributions where the transformation has been inverted and applied to inputs to transform back to the original distributions.
    '''
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    shower_counter = 0
    
    shower_hit_energies = []
    all_x = []
    all_y = []
    all_z = []

    total_deposited_e_shower = []
    all_incident_e = []
    entries = []
    GeV = 1/1000
    
    if type(files_) == list:
        print(f'plot_distribution running on input type \'files\'')
        if energy_trans_file != '':
            energy_trans_file = os.path.join(files_[0].rsplit('/',1)[0],energy_trans_file)
            print(f'Loading file for transformation inversion: {energy_trans_file}')
            # Load saved pre-processor
            scalar_e = load(open(energy_trans_file, 'rb'))
        
        # Using several files so want to take even # samples from each file for plots
        n_files = len(files_)
        print(f'n_files {type(n_files)}: {n_files}')
        print(f'nshowers_2_plot {type(nshowers_2_plot)}: {nshowers_2_plot}')
        nshowers_2_plot = nshowers_2_plot/n_files
        
        for filename in files_:
            shower_counter=0
            print(f'File: {filename}')
            fdir = filename.rsplit('/',1)[0]
            custom_data = utils.cloud_dataset(filename, device=device)
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
            
            # For each batch in file
            print(f'# batches: {len(point_clouds_loader)}')
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0): 
                valid_hits = []
                data_np = shower_data.cpu().numpy().copy()
                energy_np = incident_energies.cpu().numpy().copy()
                mask = ~(data_np[:,:,3] == padding_value)
                
                # For each shower in batch
                for j in range(len(data_np)):
                    if shower_counter >= nshowers_2_plot:
                        continue
                    shower_counter+=1
                    
                    # Only use non-padded values
                    valid_hits = data_np[j][mask[j]]
                    
                    # To transform back to original energies for plots
                    all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                    if energy_trans_file != '':
                        all_e = scalar_e.inverse_transform(all_e)
                    all_e = all_e.flatten().tolist()
                    
                    # Store features of individual hits in shower
                    shower_hit_energies += all_e
                    all_x += ((valid_hits).copy()[:,1]).flatten().tolist()
                    all_y += ((valid_hits).copy()[:,2]).flatten().tolist()
                    all_z += ((valid_hits).copy()[:,3]).flatten().tolist()
                    
                    # Number of valid hits
                    entries += [len(valid_hits)]
                    # Total energy deposited by shower
                    total_deposited_e_shower.append( sum(all_e) )
                    # Incident energies
                    all_incident_e.append(energy_np[j])
                    
    elif type(files_) == utils.cloud_dataset:
        print(f'plot_distribution running on input type \'cloud_dataset\'')
        point_clouds_loader = DataLoader(files_, batch_size=batch_size, shuffle=True)
        if energy_trans_file != '':
            # Load saved pre-processor
            print(f'Loading file for transformation inversion: {energy_trans_file}')
            scalar_e = load(open(energy_trans_file, 'rb'))
        for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0): 
            valid_hits = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()
            mask = ~(data_np[:,:,3] == padding_value)
            
            # For each shower in batch
            for j in range(len(data_np)):
                if shower_counter >= nshowers_2_plot:
                    continue
                shower_counter+=1
                valid_hits = data_np[j][mask[j]]
                
                # To transform back to original energies for plots                    
                all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                if energy_trans_file != '':
                    all_e = scalar_e.inverse_transform(all_e)
                all_e = all_e.flatten().tolist()
                
                # Store features of individual hits in shower
                shower_hit_energies += all_e
                all_x += ((valid_hits).copy()[:,1]).flatten().tolist()
                all_y += ((valid_hits).copy()[:,2]).flatten().tolist()
                all_z += ((valid_hits).copy()[:,3]).flatten().tolist()
                
                # Number of valid hits
                entries += [len(valid_hits)]
                # Total energy deposited by shower
                total_deposited_e_shower.append( sum(all_e) )
                # Incident energies
                all_incident_e.append(energy_np[j])
    #print(f'entries: {entries}')
    return [entries, all_incident_e, total_deposited_e_shower, shower_hit_energies, all_x, all_y, all_z]
    
