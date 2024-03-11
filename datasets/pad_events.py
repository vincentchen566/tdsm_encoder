import torch, os, sys, fnmatch, argparse
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler, FunctionTransformer
from pickle import dump
sys.path.insert(1, '../')
import utils, psutil

# tanh with enhanced gradient for hit energy transformation
def transform_hit_e(hit_energies):
    #new_e = 2/(1+np.exp(-10*hit_energies)) - 1
    new_e = -(1/15.)*np.log(hit_energies/(1+hit_energies))
    new_e = np.nan_to_num(new_e)
    new_e = np.reshape(new_e,(-1,))
    return new_e

# Sigmoid with reduced gradient for hit positions transformation
def transform_hit_xy(hit_pos):
    new_pos = 1/(1+np.exp(-0.04*hit_pos))
    new_pos = np.nan_to_num(new_pos)
    new_pos = np.reshape(new_pos,(-1,))
    return new_pos

# Min max for z layer
def transform_hit_z(z_):
    maxz_ = np.max(z_)
    minz_ = np.min(z_)
    z_ = (z_ - minz_) / (maxz_ - minz_)
    return z_

# Min-max incident energy transformation
def transform_incident_energy(ine_):
    maxe_ = np.max(ine_)
    mine_ = np.min(ine_)
    new_ine = (ine_ - mine_) / (maxe_ - mine_)
    return new_ine





def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-i','--indir',dest='indir', help='input directory', default='', type=str)
    argparser.add_argument('-o','--odir',dest='odir', help='output directory', default='test', type=str)
    argparser.add_argument('-t','--transform',dest='transform', help='Perform transform', default=0, type=int)
    args = argparser.parse_args()
    indir = args.indir
    odir = args.odir
    transform = args.transform
    
    GeV = 1/1000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    
    # Fit the transformation functions before adding any padding to the dataset
    E_ = []
    X_ = []
    Y_ = []
    Z_ = []
    all_files_inenergy = torch.tensor([])
    min_e_subset = 10
    min_e = 10
    max_e_subset = 0
    max_e = 0
    
    # For each file
    for infile in os.listdir(indir):
        if fnmatch.fnmatch(infile, 'dataset_2_tensor_no_pedding_euclidian_nentry*.pt'):
            filename = os.path.join(indir,infile)
            ofile = infile.replace("tensor_no_pedding_euclidian", "padded" )
            opath = '/eos/user/j/jthomasw/tdsm_encoder/datasets/'
            odir = os.path.join(opath,odir)
            if not os.path.exists(odir):
                os.makedirs(odir)
            outfilename = os.path.join(odir,ofile)
            print(f'infile: {infile}')
            print(f'ofile: {ofile}')

            loaded_file = torch.load(filename)
            showers = loaded_file[0]
            incident_energies = loaded_file[1]*GeV
            print(f'File {filename} contains:')
            print(f'{type(showers)} of {len(showers)} {type(showers[0])} showers')
            print(f'{type(incident_energies)} of {len(incident_energies)} {type(incident_energies[0])} incidient energies')

            # Find the maximum number of hits for a shower in the dataset
            max_nhits = 0
            custom_data = utils.cloud_dataset(filename, device=device)
            padded_showers = []
            for shower in custom_data.data:
                if shower.shape[0] > max_nhits:
                    max_nhits = shower.shape[0]
            
            # Transform the incident energies
            incident_energies = np.asarray( incident_energies ).reshape(-1, 1)
            incident_energies = transform_incident_energy(incident_energies)
            incident_energies = torch.from_numpy( incident_energies.flatten() )
            print(f'incident_energies:{incident_energies}')
            
            # Rescale hit energy and position and do padding
            shower_count = 0
            print(f'Maximum number of hits for all showers in file: {max_nhits}')
            print(f'custom_data {type(custom_data.data)}: {len(custom_data)}')
            
            # For each shower
            for showers in custom_data.data:
                pad_hits = max_nhits-showers.shape[0]
                if showers.shape[0] == 0:
                    print(f'incident e: {incident_energies[shower_count]} with {showers.shape[0]} hits')
                    continue
                
                # Transform the inputs
                E_ = np.asarray(showers[:,0]*GeV).reshape(-1, 1)
                X_ = np.asarray(showers[:,1]).reshape(-1, 1)
                Y_ = np.asarray(showers[:,2]).reshape(-1, 1)
                Z_ = np.asarray(showers[:,3]).reshape(-1, 1)
                    
                if transform == 1:
                    E_ = transform_hit_e(E_)
                    X_ = transform_hit_xy(X_)
                    Y_ = transform_hit_xy(Y_)
                    Z_ = transform_hit_z(Z_)
                    
                E_ = torch.from_numpy( E_.flatten() )
                X_ = torch.from_numpy( X_.flatten() )
                Y_ = torch.from_numpy( Y_.flatten() )
                Z_ = torch.from_numpy( Z_.flatten() )
                shower_data_transformed = torch.stack((E_,X_,Y_,Z_), -1)
                
                # Homogenise data with padding to make all showers the same length
                padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=0.0)
                
                # normal padding
                padded_showers.append(padded_shower)
                
                shower_count+=1
             
            torch.save([padded_showers,incident_energies], outfilename)

            ############
            # Check the transformation and padding by making some plots
            ############
            
            padded_loaded_file = torch.load(outfilename)
            padded_showers = padded_loaded_file[0]
            padded_incident_e = padded_loaded_file[1]
            print(f'File {infile} contains:')
            print(f'{type(padded_showers)} of {len(padded_showers)} showers of type {type(padded_showers[0])}')
            print(f'{type(padded_incident_e)} of {len(padded_incident_e)} showers of type {type(padded_incident_e[0])}')
            
            #### Input plots ####
            # Do you want to invert transformation for plotting purposes?
            invert_transform = 0
            
            print('Padded filename: ', outfilename)
            padded_data = utils.cloud_dataset(outfilename, device=device)
            plot_file = ofile.replace(".pt", "" )
            plots_dir = 'featureplots_'+plot_file+'/'
            plots_path = os.path.join(odir,plots_dir)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            # Load input data
            padded_data_loader = DataLoader(padded_data, batch_size=1, shuffle=False)
            n_hits = []
            all_x = []
            all_y = []
            all_z = []
            all_e = []
            all_incident_e = []
            total_shower_e = []
            
            # For each batch
            for i, (shower_data,incident_energies) in enumerate(padded_data_loader,0):

                hit_energies = shower_data[0][:,0]
                hit_xs = shower_data[0][:,1]
                hit_ys = shower_data[0][:,2]
                hit_zs = shower_data[0][:,3]

                #mask = hit_energies != 0.0
                #real_hit_energies = torch.masked_select(hit_energies,mask)

                real_hit_energies = hit_energies
                real_hit_xs = hit_xs
                real_hit_ys = hit_ys
                real_hit_zs = hit_zs
                
                real_hit_energies = real_hit_energies.tolist()
                real_hit_xs = real_hit_xs.tolist()
                real_hit_ys = real_hit_ys.tolist()
                real_hit_zs = real_hit_zs.tolist()
                
                n_hits.extend( [len(real_hit_energies)] )
                all_e.extend(real_hit_energies)
                all_x.extend(real_hit_xs)
                all_y.extend(real_hit_ys)
                all_z.extend(real_hit_zs)
                hits_sum_e = sum(real_hit_energies)
                total_shower_e.extend( [hits_sum_e] )
                
                all_incident_e.extend( [incident_energies.item()] )
            
            plot_e = all_e
            plot_x = all_x
            plot_y = all_y
            plot_z = all_z
            
            bins = np.linspace(0,10,100)

            fig, ax = plt.subplots(3,3, figsize=(14,14))
            ax[0][0].set_ylabel('# entries')
            ax[0][0].set_xlabel('Real hit entries')
            ax[0][0].hist(n_hits, 30, label='Geant4')
            ax[0][0].legend(loc='upper right')

            ax[0][1].set_ylabel('# entries')
            ax[0][1].set_xlabel('Real hit energy [GeV]')
            ax[0][1].hist(plot_e, bins=np.linspace(0,1,15), label='Geant4')
            ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            ax[0][2].set_ylabel('# entries')
            ax[0][2].set_xlabel('Real hit x')
            ax[0][2].hist(plot_x, bins=np.linspace(0,1,15), label='Geant4')
            ax[2][0].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            ax[1][0].set_ylabel('# entries')
            ax[1][0].set_xlabel('Real hit y')
            ax[1][0].hist(plot_y, bins=np.linspace(0,1,15), label='Geant4')
            ax[2][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            ax[1][1].set_ylabel('# entries')
            ax[1][1].set_xlabel('Real hit z')
            ax[1][1].hist(plot_z, bins=np.linspace(0,1,15), label='Geant4')
            ax[1][1].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            ax[1][2].set_ylabel('# entries')
            ax[1][2].set_xlabel('Incident energy')
            ax[1][2].hist(all_incident_e, 15, label='Geant4')
            ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')
            
            ax[2][0].set_ylabel('# entries')
            ax[2][0].set_xlabel('Real hit deposited energy [GeV]')
            ax[2][0].hist(total_shower_e, 15, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[2][0].legend(loc='upper right')
            if invert_transform == 0:
                fig_save_name = plots_path+'hit_inputs_transformed.png'
            else:
                fig_save_name = plots_path+'hit_inputs_non_transformed.png'
            fig.savefig(fig_save_name)

if __name__=='__main__':
    main()
    
