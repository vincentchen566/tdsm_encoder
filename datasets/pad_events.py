import torch, os, sys, fnmatch, argparse, psutil
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, MinMaxScaler
from pickle import dump
sys.path.insert(1, '../util')
import data_utils as utils
import display

def transform_hit_e(x_, ine_):
    # alpha and ine multiplier used to ensure hit energy > 0
    alpha = 1e-06
    t_ = 1-alpha
    y_ = t_ * x_/(8*ine_)
    rescaled_e = -(1/5)*torch.log(y_/(1-y_))
    return rescaled_e

def transform_hit_xy(x_):
    max_xy = 40
    min_xy = -40
    y_ = (x_-min_xy)/(max_xy-min_xy)
    return y_

def transform_hit_z(z_):
    z_ = (z_ / 45)+0.5
    return z_

def transform_ine(ine_):
    new_ine = np.log(ine_)
    return new_ine

def main():
    usage=''
    argparser = argparse.ArgumentParser(usage)
    argparser.add_argument('-i','--indir',dest='indir', help='input directory', default='', type=str)
    argparser.add_argument('-f','--infile',dest='infile', help='input file', default='', type=str)
    argparser.add_argument('-o','--odir',dest='odir', help='output directory', default='test', type=str)
    argparser.add_argument('-t','--transform',dest='transform', help='Perform transform', default=0, type=int)
    args = argparser.parse_args()
    indir = args.indir
    infile = args.infile
    odir = args.odir
    transform = args.transform
    
    GeV = 1/1000

    print('pytorch: ', torch.__version__)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    print(f'pad_events.py: check # cuda devices: {torch.cuda.device_count()} ({torch._C._cuda_getDeviceCount()}), current device: {torch.cuda.current_device()}')
    
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
    #for infile in os.listdir(indir):
    #if fnmatch.fnmatch(infile, 'dataset_2_tensor_no_pedding_euclidian_nentry*.pt'):
    
    filename = os.path.join(indir,infile)
    ofile = infile.replace("tensor_no_pedding_euclidian", "padded" )
    opath = '/eos/user/j/jthomasw/tdsm_encoder/datasets/'
    odir = os.path.join(opath,odir)
    if not os.path.exists(odir):
        os.makedirs(odir)
    outfilename = os.path.join(odir,ofile)
    print(f'Input file from: {filename}')
    print(f'Output file to: {outfilename}')

    loaded_file = torch.load(filename)
    showers = loaded_file[0]
    incident_energies = loaded_file[1]*GeV
    print(f'File {filename} contains:')
    print(f'{len(showers)} {type(showers[0])} showers')

    # Find the maximum number of hits for a shower in the dataset for padding
    max_nhits = 0
    custom_data = utils.cloud_dataset(filename, device=device)
    padded_showers = []
    for shower in custom_data.data:
        if shower.shape[0] > max_nhits:
            max_nhits = shower.shape[0]

    # Transform the incident energies
    original_incident_e = incident_energies.tolist()
    trans_incident_e = transform_ine(original_incident_e)

    # Rescale hit energy and position and do padding
    shower_count = 0
    print(f'Maximum number of hits for all showers in file: {max_nhits}')

    if not len(custom_data) == len(showers):
        print('Error: # showers in data object != loaded file. Exiting . . . ')
        return 1

    # For each shower
    for i, showers in enumerate(custom_data.data):
        pad_hits = max_nhits-showers.shape[0]
        if showers.shape[0] == 0:
            print(f'Shower {i}: {trans_incident_e[shower_count]} and {showers.shape[0]} hits.')
            print('Skipping . . .')
            continue

        # Transform the inputs
        E_ = showers[:,0]*GeV
        X_ = showers[:,1]
        Y_ = showers[:,2]
        Z_ = showers[:,3]
        shower_e = trans_incident_e[i]

        if transform == 1:
            E_ = transform_hit_e(E_, shower_e)
            X_ = transform_hit_xy(X_)
            Y_ = transform_hit_xy(Y_)
            Z_ = transform_hit_z(Z_)

        E_ = E_.flatten()
        X_ = X_.flatten()
        Y_ = Y_.flatten()
        Z_ = Z_.flatten()
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

    #### Input plots ####
    # Do you want to invert transformation for plotting purposes?
    invert_transform = 0
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

    name1 = infile.split('nentry',1)[1].split('.')[0]
    plots = [all_incident_e,all_e,all_x,all_y,all_z]
    titles = ['incident energy','hit energy', 'hit x', 'hit y', 'hit z']
    n_plots = len(plots)
    # 9x16 radial x angular / eta x phi bins, 45 layers
    n_bins = [30,30,9,16,45]
    plotter = display.recursive_plot(len(plots), name1, plots, n_bins, titles)
    plotter.rec_plot()

    # Save file name
    if invert_transform == 0:
        savename = 'Transformed_feats_'+name1
    else:
        savename = 'Untransformed_feats_'+name1

    plotter.save(os.path.join(plots_path,savename)+'.png')

if __name__=='__main__':
    main()
