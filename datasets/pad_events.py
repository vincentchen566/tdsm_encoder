import torch, os, sys, fnmatch, argparse
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from pickle import dump
sys.path.insert(1, '../')
import utils, psutil

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-i','--indir',dest='indir', help='input directory', default='', type=str)
    argparser.add_argument('-o','--odir',dest='odir', help='output directory', default='test', type=str)
    args = argparser.parse_args()
    indir = args.indir
    odir = args.odir
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    
    E_ = np.array([])
    X_ = np.array([])
    Y_ = np.array([])
    Z_ = np.array([])
    all_files_inenergy = torch.tensor([])
    for infile in os.listdir(indir):
        if fnmatch.fnmatch(infile, 'dataset_1_photons_tensor_no_pedding_euclidian_nentry*.pt'):
            filename = os.path.join(indir,infile)
            print(f'filename: {filename}')
            loaded_file = torch.load(filename)
            all_files_inenergy = torch.cat([all_files_inenergy,loaded_file[1]])
            print(f'# showers: {len(loaded_file[0])}')
            nshowers = 0
            for shower in loaded_file[0]:
                if nshowers%100 == 0:
                    E_ = np.append(E_,np.asarray(shower[:,0]).reshape(-1, 1))
                    X_ = np.append(X_,np.asarray(shower[:,1]).reshape(-1, 1))
                    Y_ = np.append(Y_,np.asarray(shower[:,2]).reshape(-1, 1))
                    Z_ = np.append(Z_,np.asarray(shower[:,3]).reshape(-1, 1))
                nshowers+=1
            
            process = psutil.Process(os.getpid())
            print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))
    
    E_ = E_.reshape(-1, 1)
    X_ = X_.reshape(-1, 1)
    Y_ = Y_.reshape(-1, 1)
    Z_ = Z_.reshape(-1, 1)
    
    #transform_e = RobustScaler().fit(E_)
    transform_e = PowerTransformer().fit(E_)
    #transform_x = RobustScaler().fit(X_)
    transform_x = PowerTransformer().fit(X_)
    #transform_y = RobustScaler().fit(Y_)
    transform_y = PowerTransformer().fit(Y_)
    
    all_files_inenergy = torch.reshape(all_files_inenergy, (-1,1))
    #rescaler_y = RobustScaler().fit(all_files_inenergy)
    rescaler_y = PowerTransformer().fit(all_files_inenergy)
    rescaler = utils.rescale_energies()
    
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
            incident_energies = loaded_file[1]
            print(f'File {filename} contains:')
            print(f'{type(showers)} of {len(showers)} {type(showers[0])} showers')
            print(f'{type(incident_energies)} of {len(incident_energies)} {type(incident_energies[0])} incidient energies')

            max_nhits = 0
            custom_data = utils.cloud_dataset(filename, device=device)

            # Have to homogenise with data padding before using dataloaders
            padded_showers = []
            for shower in custom_data.data:
                if shower.shape[0] > max_nhits:
                    max_nhits = shower.shape[0]
            
            # Rescale the conditional input for each shower
            incident_energies = torch.reshape(incident_energies, (-1,1))
            incident_e_transformed = rescaler_y.transform(incident_energies)
            incident_e_transformed = torch.reshape( torch.tensor(incident_e_transformed), (-1,) )
            
            # Rescale hit energy and position and do padding
            shower_count = 0
            print(f'Maximum number of hits for all showers in file: {max_nhits}')
            print(f'custom_data {type(custom_data.data)}: {len(custom_data)}')
            for showers in custom_data.data:
                #print(f'showers: {type(showers)}')
                pad_hits = max_nhits-showers.shape[0]
                if showers.shape[0] == 0:
                    print(f'incident e: {incident_e_transformed[shower_count]} with {showers.shape[0]} hits')
                    continue
                    
                # Energy rescaled wrt to non-scaled incident energy
                #shower_data_transformed = rescaler(showers,incident_energies[shower_count])
                
                E_ = np.asarray(showers[:,0]).reshape(-1, 1)
                X_ = np.asarray(showers[:,1]).reshape(-1, 1)
                Y_ = np.asarray(showers[:,2]).reshape(-1, 1)

                e_ = transform_e.transform(E_)
                e_ = torch.from_numpy( e_.flatten() )

                x_ = transform_x.transform(X_)
                x_ = torch.from_numpy( x_.flatten() )

                y_ = transform_y.transform(Y_)
                y_ = torch.from_numpy( y_.flatten() )

                #e_ = normalize(features[:,1], dim=0)
                #x_ = normalize(features[:,1], dim=0)
                #y_ = normalize(features[:,2], dim=0)

                z_ = showers[:,3]
                
                shower_data_transformed = torch.stack((e_,x_,y_,z_), -1)
                
                # Pad with sentinel value to make the showers uniform in length
                padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=-20)
                padded_showers.append(padded_shower)
                shower_count+=1
            #print(f'incident_e_transformed: {incident_e_transformed}')
            torch.save([padded_showers,incident_e_transformed], outfilename)
            #dump(rescaler, open('rescaler.pkl', 'wb'))
            dump(rescaler_y, open('rescaler_y.pkl', 'wb'))
            dump(transform_e, open('transform_e.pkl', 'wb'))
            dump(transform_x, open('transform_x.pkl', 'wb'))
            dump(transform_y, open('transform_y.pkl', 'wb'))
            # Check
            padded_loaded_file = torch.load(outfilename)
            padded_showers = padded_loaded_file[0]
            padded_incident_e = padded_loaded_file[1]
            print(f'File {infile} contains:')
            print(f'{type(padded_showers)} of {len(padded_showers)} showers of type {type(padded_showers[0])}')
            print(f'{type(padded_incident_e)} of {len(padded_incident_e)} showers of type {type(padded_incident_e[0])}')

            #### Input plots ####
            print('Padded filename: ', outfilename)
            padded_data = utils.cloud_dataset(outfilename, device=device)
            plot_file = ofile.replace(".pt", "" )
            plots_dir = 'featureplots_'+plot_file+'/'
            plots_path = os.path.join(odir,plots_dir)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            # Load input data
            padded_data_loader = DataLoader(padded_data,batch_size=1,shuffle=False)
            n_hits = []
            all_x = []
            all_y = []
            all_z = []
            all_e = []
            all_incident_e = []

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

                real_hit_energies = real_hit_energies.tolist()
                real_hit_xs = real_hit_xs.tolist()
                real_hit_ys = real_hit_ys.tolist()
                real_hit_zs = real_hit_zs.tolist()

                n_hits.append( len(real_hit_energies) )
                all_e.append(real_hit_energies)
                all_x.append(real_hit_xs)
                all_y.append(real_hit_ys)
                all_z.append(real_hit_zs)
                all_incident_e.append(incident_energies.item())

            plot_e = np.concatenate(all_e)
            plot_x = np.concatenate(all_x)
            plot_y = np.concatenate(all_y)
            plot_z = np.concatenate(all_z)

            fig, ax = plt.subplots(2,3, figsize=(12,12))
            ax[0][0].set_ylabel('# entries')
            ax[0][0].set_xlabel('Hit entries')
            ax[0][0].hist(n_hits, 30, label='Geant4')
            ax[0][0].legend(loc='upper right')

            ax[0][1].set_ylabel('# entries')
            ax[0][1].set_xlabel('Hit energy')
            ax[0][1].hist(plot_e, 50, label='Geant4')
            ax[0][1].legend(loc='upper right')

            ax[0][2].set_ylabel('# entries')
            ax[0][2].set_xlabel('Hit x')
            ax[0][2].hist(plot_x, 50, label='Geant4')
            ax[0][2].legend(loc='upper right')

            ax[1][0].set_ylabel('# entries')
            ax[1][0].set_xlabel('Hit y')
            ax[1][0].hist(plot_y, 50, label='Geant4')
            ax[1][0].legend(loc='upper right')

            ax[1][1].set_ylabel('# entries')
            ax[1][1].set_xlabel('Hit z')
            ax[1][1].hist(plot_z, 50, label='Geant4')
            ax[1][1].legend(loc='upper right')

            ax[1][2].set_ylabel('# entries')
            ax[1][2].set_xlabel('Incident energy')
            ax[1][2].hist(all_incident_e, 30, label='Geant4')
            ax[1][2].legend(loc='upper right')

            fig.savefig(plots_path+'hit_inputs.png')

if __name__=='__main__':
    main()
    