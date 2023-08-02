import torch, os, sys, fnmatch, argparse
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from pickle import dump
sys.path.insert(1, '../')
import utils, psutil

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
    min_val_subset = 10
    min_val = 10
    for infile in os.listdir(indir):
        # It seems that using a single file to fit the transformation function,
        # creates a stable result and smooth distributions for all files
        if fnmatch.fnmatch(infile, 'dataset_1_photons_tensor_no_pedding_euclidian_nentry12*.pt'):
            filename = os.path.join(indir,infile)
            print(f'filename: {filename}')
            loaded_file = torch.load(filename)
            all_files_inenergy = torch.cat([all_files_inenergy,loaded_file[1]*GeV])
            print(f'# showers: {len(loaded_file[0])}')
            nshowers = 0
            for shower in loaded_file[0]:
                
                if sorted(set(shower[:,0]))[0] < min_val:
                    min_val = sorted(set(shower[:,0]))[0]
                if nshowers%2 == 0:
                    if sorted(set(shower[:,0]))[0] < min_val_subset:
                        min_val_subset = sorted(set(shower[:,0]))[0]
                    E_.extend( shower[:,0]*GeV )
                    X_.extend( shower[:,1] )
                    Y_.extend( shower[:,2] )
                    Z_.extend( shower[:,3] )
                    nshowers+=1
            
            process = psutil.Process(os.getpid())
            print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))
    print(f'min_val_subset: {min_val_subset}')
    print(f'min_val: {min_val}')
    print(f'# showers: {len(E_)}')
    E_ = np.array(E_).reshape(-1, 1)
    X_ = np.array(X_).reshape(-1, 1)
    Y_ = np.array(Y_).reshape(-1, 1)
    Z_ = np.array(Z_).reshape(-1, 1)
    
    # Fit the transformation functions
    if transform == 1:
        print(f'Fitting transformation function for hit energies')
        #transform_e = QuantileTransformer(output_distribution='normal').fit(E_)
        #transform_e = QuantileTransformer(output_distribution='uniform').fit(E_)
        #transform_e = RobustScaler().fit(E_)
        transform_e = PowerTransformer(method="box-cox").fit(E_)
        print(f'Fitting transformation function for hit X')
        #transform_x = QuantileTransformer(output_distribution='normal').fit(X_)
        #transform_x = QuantileTransformer(output_distribution='uniform').fit(X_)
        #transform_x = RobustScaler().fit(X_)
        transform_x = PowerTransformer(method="yeo-johnson").fit(X_)
        print(f'Fitting transformation function for hit Y')
        #transform_y = QuantileTransformer(output_distribution='normal').fit(Y_)
        #transform_y = QuantileTransformer(output_distribution='uniform').fit(Y_)
        #transform_y = RobustScaler().fit(Y_)
        transform_y = PowerTransformer(method="yeo-johnson").fit(Y_)
    
    all_files_inenergy = torch.reshape(all_files_inenergy, (-1,1))
    print(f'Fitting transformation function for incident energies')
    #rescaler_y = QuantileTransformer(output_distribution='normal').fit(all_files_inenergy)
    #rescaler_y = QuantileTransformer(output_distribution='uniform').fit(all_files_inenergy)
    #rescaler_y = RobustScaler().fit(all_files_inenergy)
    rescaler_y = PowerTransformer("box-cox").fit(all_files_inenergy)
    
    # For each file
    for infile in os.listdir(indir):
        if fnmatch.fnmatch(infile, 'dataset_1_photons_tensor_no_pedding_euclidian_nentry*.pt'):
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
            #if transform == 1:
                # Rescale the conditional input for each shower
            #    incident_energies = rescaler_y.transform(incident_energies)
            incident_energies = torch.from_numpy( incident_energies.flatten() )
            
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
                if transform == 1:
                    E_ = transform_e.transform(E_)
                    X_ = transform_x.transform(X_)
                    Y_ = transform_y.transform(Y_)
                    
                E_ = torch.from_numpy( E_.flatten() )
                X_ = torch.from_numpy( X_.flatten() )
                Y_ = torch.from_numpy( Y_.flatten() )
                Z_ = showers[:,3]
                shower_data_transformed = torch.stack((E_,X_,Y_,Z_), -1)
                
                # Homogenise data with padding to make all showers the same length
                padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=-20)
                padded_showers.append(padded_shower)
                shower_count+=1
                
            torch.save([padded_showers,incident_energies], outfilename)
            
            if transform == 1:
                dump(rescaler_y, open( os.path.join(odir,'rescaler_y.pkl') , 'wb'))
                dump(transform_e, open( os.path.join(odir,'transform_e.pkl') , 'wb'))
                dump(transform_x, open(os.path.join(odir,'transform_x.pkl') , 'wb'))
                dump(transform_y, open(os.path.join(odir,'transform_y.pkl') , 'wb'))
                
            # Check
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

                mask = hit_energies != -20.0

                real_hit_energies = torch.masked_select(hit_energies,mask)
                real_hit_xs = torch.masked_select(hit_xs,mask)
                real_hit_ys = torch.masked_select(hit_ys,mask)
                real_hit_zs = torch.masked_select(hit_zs,mask)
                
                # Invert transformation for plots in original variable space (also checks inversion works)
                if invert_transform == 1:
                    real_hit_energies = real_hit_energies.tolist()
                    real_hit_energies = np.asarray(real_hit_energies).reshape(-1, 1)
                    real_hit_energies = transform_e.inverse_transform(real_hit_energies)
                    real_hit_energies = torch.from_numpy( real_hit_energies.flatten() )
                    
                    
                    real_hit_xs = real_hit_xs.tolist()
                    real_hit_xs = np.asarray(real_hit_xs).reshape(-1, 1)
                    real_hit_xs = transform_x.inverse_transform(real_hit_xs)
                    real_hit_xs = torch.from_numpy( real_hit_xs.flatten() )
                    
                    real_hit_ys = real_hit_ys.tolist()
                    real_hit_ys = np.asarray(real_hit_ys).reshape(-1, 1)
                    real_hit_ys = transform_y.inverse_transform(real_hit_ys)
                    real_hit_ys = torch.from_numpy( real_hit_ys.flatten() )
                
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
                
                # Transform the incident energies
                incident_energies = np.asarray( incident_energies ).reshape(-1, 1)
                if invert_transform == 1:
                    # Rescale the conditional input for each shower
                    incident_energies = rescaler_y.inverse_transform(incident_energies)
                incident_energies = torch.from_numpy( incident_energies.flatten() )
                all_incident_e.extend( [incident_energies.item()] )
            
            plot_e = all_e
            plot_x = all_x
            plot_y = all_y
            plot_z = all_z

            fig, ax = plt.subplots(3,3, figsize=(14,14))
            ax[0][0].set_ylabel('# entries')
            ax[0][0].set_xlabel('Hit entries')
            ax[0][0].hist(n_hits, 30, label='Geant4')
            ax[0][0].legend(loc='upper right')

            ax[0][1].set_ylabel('# entries')
            ax[0][1].set_xlabel('Hit energy [GeV]')
            ax[0][1].hist(plot_e, 50, label='Geant4')
            ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            ax[0][2].set_ylabel('# entries')
            ax[0][2].set_xlabel('Hit x')
            ax[0][2].hist(plot_x, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            ax[1][0].set_ylabel('# entries')
            ax[1][0].set_xlabel('Hit y')
            ax[1][0].hist(plot_y, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            ax[1][1].set_ylabel('# entries')
            ax[1][1].set_xlabel('Hit z')
            ax[1][1].hist(plot_z, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            ax[1][2].set_ylabel('# entries')
            ax[1][2].set_xlabel('Incident energy')
            ax[1][2].hist(all_incident_e, 30, label='Geant4')
            ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')
            
            ax[2][0].set_ylabel('# entries')
            ax[2][0].set_xlabel('Deposited energy [GeV]')
            ax[2][0].hist(total_shower_e, 30, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[2][0].legend(loc='upper right')
            if invert_transform == 0:
                fig_save_name = plots_path+'hit_inputs_transformed.png'
            else:
                fig_save_name = plots_path+'hit_inputs_non_transformed.png'
            fig.savefig(fig_save_name)

if __name__=='__main__':
    main()
    