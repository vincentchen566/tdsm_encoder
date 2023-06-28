import torch, os, sys, fnmatch, argparse
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
sys.path.insert(1, '../')
import utils

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-i','--infile',dest='infilename', help='input filenme', default='', type=str)
    args = argparser.parse_args()
    infilename = args.infilename
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    filename = '/eos/user/t/tihsu/database/ML_hackthon/bucketed_tensor/'+infilename
    loaded_file = torch.load(filename)
    showers = loaded_file[0]
    incident_energies = loaded_file[1]
    print(f'File {filename} contains:')
    print(f'{type(showers)} of {len(showers)} showers of type {type(showers[0])}')
    print(f'{type(incident_energies)} of {len(incident_energies)} incidient energies of type {type(incident_energies[0])}')
    
    max_nhits = 0
    custom_data = utils.cloud_dataset(filename, transform=utils.rescale_energies(), transform_y=utils.rescale_conditional(), device=device)
    
    # Have to homogenise with data padding before using dataloaders
    padded_showers = []
    rescaled_incident_e = []
    for shower in custom_data.data:
        if shower.shape[0] > max_nhits:
            max_nhits = shower.shape[0]

    shower_count = 0
    print(f'Maximum number of hits for all showers in file: {max_nhits}')
    for showers in custom_data.data:
        pad_hits = max_nhits-showers.shape[0]
        # Firstly, do any rescaling of the inputs
        rescaler = utils.rescale_energies()

        if showers.shape[0] == 0:
            print(f'incident e: {incident_energies[shower_count]} with {showers.shape[0]} hits')
            continue
        
        shower_data_transformed = rescaler(showers,incident_energies[shower_count])
        if torch.sum(showers.gt(1e10)) > 0:
               #print('showers: ', showers)
               continue
        
        if showers.shape[0] < 10:
            print('showers.shape[0]: ', showers.shape[0])
            print('showers: ', showers)
            print('shower_data_transformed: ', shower_data_transformed)
            continue
        
        # Next, pad with sentinel value to make the showers uniform in length
        padded_shower = F.pad(input = shower_data_transformed, pad=(0,0,0,pad_hits), mode='constant', value=-20)
        padded_showers.append(padded_shower)
        # Rescale the conditional input for each shower
        rescaler_y = utils.rescale_conditional()
        incident_e_transformed = rescaler_y(incident_energies[shower_count],custom_data.min_y,custom_data.max_y)
        rescaled_incident_e.append(incident_e_transformed)
        shower_count+=1

    torch.save([padded_showers,torch.as_tensor(rescaled_incident_e)], '/afs/cern.ch/work/j/jthomasw/private/NTU/fast_sim/tdsm_encoder/datasets/'+infilename)
    # Check
    padded_loaded_file = torch.load(infilename)
    padded_showers = padded_loaded_file[0]
    padded_incident_e = padded_loaded_file[1]
    print(f'File {infilename} contains:')
    print(f'{type(padded_showers)} of {len(padded_showers)} showers of type {type(padded_showers[0])}')
    print(f'{type(padded_incident_e)} of {len(padded_incident_e)} showers of type {type(padded_incident_e[0])}')

    #### Input plots ####
    print('filename: ', infilename)
    custom_data = utils.cloud_dataset(infilename)
    output_directory = './feature_plots_padded_dataset_1_photons_'+datetime.now().strftime('%Y%m%d_%H%M')+'/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load input data
    point_clouds_loader = DataLoader(custom_data,batch_size=1,shuffle=False)
    n_hits = []
    all_x = []
    all_y = []
    all_z = []
    all_e = []
    all_incident_e = []
    for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0):
        # Limit number fo showers to plot for memories sake
        if i>500:
            break

        hit_energies = shower_data[0][:,0]
        hit_xs = shower_data[0][:,1]
        hit_ys = shower_data[0][:,2]
        hit_zs = shower_data[0][:,3]

        mask = hit_energies.gt(-20.0)

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
    
    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('# Hits')
    plt.hist(n_hits, 100, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'nhit.png')
    
    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit energy')
    plt.hist(plot_e, 100, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_energies.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit x position')
    plt.hist(plot_x, 100, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_x.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit y position')
    plt.hist(plot_y, 100, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_y.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Hit z position')
    plt.hist(plot_z, 50, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_z.png')

    fig, ax = plt.subplots(ncols=1, figsize=(10,10))
    plt.title('')
    plt.ylabel('# entries')
    plt.xlabel('Incident energy')
    plt.hist(all_incident_e, 100, label='Geant4')
    plt.legend(loc='upper right')
    fig.savefig(output_directory+'hit_incident_e.png')

if __name__=='__main__':
    main()
    