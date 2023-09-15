import torch, os, sys, fnmatch, argparse
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler
from pickle import dump
sys.path.insert(1, '../')
import utils, psutil

# bespoke minmax function
class minmax():

    def __init__(self, indir=''):
        self.indir = indir
        self.min_e = 1000
        self.max_e = 0
        self.min_x = 1000
        self.max_x = 0
        self.min_y = 1000
        self.max_y = 0
        self.GeV = 1/1000

    def get_minmax(self):
        # We want to fit a transformation function for the distribution of showers from all files
        for infile in os.listdir(self.indir):
            # Using a minmax function we need to ensure we get the max and min value of the entire dataset
            if fnmatch.fnmatch(infile, 'dataset_1_photons_tensor_no_pedding_euclidian_nentry*.pt'):
                filename = os.path.join(self.indir,infile)
                print(f'check filename: {filename} for minmax')
                loaded_file = torch.load(filename)
                
                for shower in loaded_file[0]:
                    sorted_e = sorted(set(shower[:,0]* self.GeV))
                    sorted_x = sorted(set(shower[:,1]))
                    sorted_y = sorted(set(shower[:,2]))
                    if sorted_e[0] < self.min_e:
                        self.min_e = sorted_e[0].item()
                    if sorted_e[-1] > self.max_e:
                        self.max_e = sorted_e[-1].item()
                    if sorted_x[0] < self.min_x:
                        self.min_x = sorted_x[0].item()
                    if sorted_x[-1] > self.max_x:
                        self.max_x = sorted_x[-1].item()
                    if sorted_y[0] < self.min_y:
                        self.min_y = sorted_y[0].item()
                    if sorted_y[-1] > self.max_y:
                        self.max_y = sorted_y[-1].item()
    
        print(f'Final min e: {self.min_e}, max: {self.max_e}')
        print(f'Final min x: {self.min_x}, max: {self.max_x}')
        print(f'Final min y: {self.min_y}, max: {self.max_y}')
        return
    
    def transform_e(self, E, range_min, range_max):
        E_std = (E - self.min_e) / (self.max_e - self.min_e)
        E_scaled = E_std * (range_max - range_min) + range_min
        return E_scaled
    
    def invert_transform_e(self, E_scaled, range_min, range_max):
        E_std = (E_scaled - range_min) / (range_max - range_min)
        E = E_std * (self.max_e - self.min_e) +  self.min_e
        return E
    
    def transform_x(self, X, range_min, range_max):
        X_std = (X - self.min_x) / (self.max_x - self.min_x)
        X_scaled = X_std * (range_max - range_min) + range_min
        return X_scaled
    
    def transform_y(self, Y, range_min, range_max):
        Y_std = (Y - self.min_y) / (self.max_y - self.min_y)
        Y_scaled = Y_std * (range_max - range_min) + range_min
        return Y_scaled

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
    
    # Instantiate transform object and get minmax values
    #trans_ = minmax(indir=indir)
    #trans_.get_minmax()
    
    
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

    # We want to fit a transformation function for the distribution of showers from all files
    for infile in os.listdir(indir):
        if fnmatch.fnmatch(infile, 'dataset_1_photons_tensor_no_pedding_euclidian_nentry*.pt'):
            filename = os.path.join(indir,infile)
            print(f'Add filename: {filename} to list for fit')
            loaded_file = torch.load(filename)
            all_files_inenergy = torch.cat([all_files_inenergy,loaded_file[1]*GeV])
            print(f'# showers {type(loaded_file[0])}: {len(loaded_file[0])}')
            every_n_files = len(loaded_file[0])//1000
            print(f'sample every {every_n_files} showers')
            nshowers = 0
            for shower in loaded_file[0]:
                
                if nshowers % every_n_files == 0:
                    
                    # Sort the showers to see the max and min values (warning, there are outliers)
                    sorted_shower = sorted(set(shower[:,0]))
                    if sorted_shower[0] < min_e:
                        min_e = sorted_shower[0]
                    if sorted_shower[-1] > max_e:
                        max_e = sorted_shower[-1]

                    # Only using some of the showers to fit pre-processing transformation function
                    if sorted_shower[0] < min_e_subset:
                        min_e_subset = sorted_shower[0]
                    if sorted_shower[-1] > max_e_subset:
                        max_e_subset = sorted_shower[-1]
                    E_.extend( shower[:,0]*GeV )
                    X_.extend( shower[:,1] )
                    Y_.extend( shower[:,2] )
                    Z_.extend( shower[:,3] )

                nshowers+=1
            
            process = psutil.Process(os.getpid())
            print('Memory usage of current process 0 [GB]: ', process.memory_info().rss/(1024 * 1024 * 1024))

    print(f'min_e_subset: {min_e_subset}')
    print(f'min_e: {min_e}')
    print(f'max_e_subset: {max_e_subset}')
    print(f'max_e: {max_e}')

    E_ = np.array(E_).reshape(-1, 1)
    X_ = np.array(X_).reshape(-1, 1)
    Y_ = np.array(Y_).reshape(-1, 1)
    Z_ = np.array(Z_).reshape(-1, 1)
    all_files_inenergy = torch.reshape(all_files_inenergy, (-1,1))
    
    # Fit the transformation functions
    if transform == 1:
        print(f'Fitting transformation function for hit energies')
        #transform_e = MinMaxScaler(feature_range=(1,2)).fit(E_)
        transform_e = QuantileTransformer(output_distribution="uniform").fit(E_)
        print(f'Fitting transformation function for hit X')
        #transform_x = MinMaxScaler(feature_range=(1,2)).fit(X_)
        transform_x = QuantileTransformer(output_distribution="uniform").fit(X_)
        print(f'Fitting transformation function for hit Y')
        #transform_y = MinMaxScaler(feature_range=(1,2)).fit(Y_)
        transform_y = QuantileTransformer(output_distribution="uniform").fit(Y_)
        print(f'Fitting transformation function for incident energies')
        #rescaler_y = MinMaxScaler(feature_range=(1,2)).fit(all_files_inenergy)
        rescaler_y = QuantileTransformer(output_distribution="uniform").fit(all_files_inenergy)
    

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
            if transform == 1:
                # Rescale the conditional input for each shower
                incident_energies = rescaler_y.transform(incident_energies)
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
                Z_ = np.asarray(showers[:,3]).reshape(-1, 1)
                if transform == 1:

                    #E_ = trans_.transform_e(E_, 1, 2)
                    #X_ = trans_.transform_x(X_, 1, 2)
                    #Y_ = trans_.transform_y(Y_, 1, 2)
                    E_ = transform_e.transform(E_)+1
                    X_ = transform_x.transform(X_)+1
                    Y_ = transform_y.transform(Y_)+1
                    Z_ = Z_+1
                    
                E_ = torch.from_numpy( E_.flatten() )
                X_ = torch.from_numpy( X_.flatten() )
                Y_ = torch.from_numpy( Y_.flatten() )
                Z_ = torch.from_numpy( Z_.flatten() )
                shower_data_transformed = torch.stack((E_,X_,Y_,Z_), -1)
                
                # Homogenise data with padding to make all showers the same length
                padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=0.0)
                
                # normal padding
                padded_showers.append(padded_shower)
                
                # Smear padded data with values == 0
                # Be careful to ensure that valid data doesn't have values == 0
                # May need to cap the values to ensure padded values not smeared into valid distribution?
                #noise = torch.normal(0,0.2,size=padded_shower.shape)
                #noise = torch.rand(padded_shower.shape)
                #mask_valid = (padded_shower[:,:] == 0).type(torch.int)
                #noise = noise*mask_valid
                #tmp = noise.clone()
                #tmp[noise==0]=1
                #new_padded_shower = padded_shower.clone()
                #new_padded_shower[padded_shower==0]=1
                #new_padded_shower = new_padded_shower*tmp
                #padded_showers.append(new_padded_shower)
                
                shower_count+=1
             
            torch.save([padded_showers,incident_energies], outfilename)
            

            ############
            # Check the transformation and padding by making some plots
            ############

            # Need to save transformation functions if we want to invert the same transformation later
            if transform == 1:
                dump(rescaler_y, open( os.path.join(odir,'rescaler_y.pkl') , 'wb'))
                dump(transform_e, open( os.path.join(odir,'transform_e.pkl') , 'wb'))
                dump(transform_x, open(os.path.join(odir,'transform_x.pkl') , 'wb'))
                dump(transform_y, open(os.path.join(odir,'transform_y.pkl') , 'wb'))
            
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

                mask = hit_energies != 0.0

                #real_hit_energies = torch.masked_select(hit_energies,mask)
                real_hit_energies = hit_energies
                #real_hit_xs = torch.masked_select(hit_xs,mask)
                real_hit_xs = hit_xs
                #real_hit_ys = torch.masked_select(hit_ys,mask)
                real_hit_ys = hit_ys
                #real_hit_zs = torch.masked_select(hit_zs,mask)
                real_hit_zs = hit_zs
                
                '''
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
                    
                    # Rescale the conditional input for each shower
                    incident_energies = np.asarray( incident_energies ).reshape(-1, 1)
                    incident_energies = rescaler_y.inverse_transform(incident_energies)
                    incident_energies = torch.from_numpy( incident_energies.flatten() )
                    '''
                
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

            fig, ax = plt.subplots(3,3, figsize=(14,14))
            ax[0][0].set_ylabel('# entries')
            ax[0][0].set_xlabel('Real hit entries')
            ax[0][0].hist(n_hits, 30, label='Geant4')
            ax[0][0].legend(loc='upper right')

            ax[0][1].set_ylabel('# entries')
            ax[0][1].set_xlabel('Real hit energy [GeV]')
            ax[0][1].hist(plot_e, 50, label='Geant4')
            ax[0][1].set_yscale('log')
            ax[0][1].legend(loc='upper right')

            ax[0][2].set_ylabel('# entries')
            ax[0][2].set_xlabel('Real hit x')
            ax[0][2].hist(plot_x, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[0][2].legend(loc='upper right')

            ax[1][0].set_ylabel('# entries')
            ax[1][0].set_xlabel('Real hit y')
            ax[1][0].hist(plot_y, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[1][0].legend(loc='upper right')

            ax[1][1].set_ylabel('# entries')
            ax[1][1].set_xlabel('Real hit z')
            ax[1][1].hist(plot_z, 50, label='Geant4')
            ax[2][0].set_yscale('log')
            ax[1][1].legend(loc='upper right')

            ax[1][2].set_ylabel('# entries')
            ax[1][2].set_xlabel('Incident energy')
            ax[1][2].hist(all_incident_e, 30, label='Geant4')
            ax[1][2].set_yscale('log')
            ax[1][2].legend(loc='upper right')
            
            ax[2][0].set_ylabel('# entries')
            ax[2][0].set_xlabel('Real hit deposited energy [GeV]')
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
    