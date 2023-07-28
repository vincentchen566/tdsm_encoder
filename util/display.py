import torch, utils, sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from typing import Union
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, PowerTransformer, minmax_scale
from pickle import load
from matplotlib import cm
sys.path.insert(1, '../')

def plot_distribution(files_:Union[ list , utils.cloud_dataset], nshowers_2_plot=100, padding_value=-20, batch_size=1, energy_trans_file='', x_trans_file='', y_trans_file='', ine_trans_file=''):
    
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
    shower_hit_x = []
    shower_hit_y = []
    all_z = []
    shower_hit_ine = []

    total_deposited_e_shower = []
    average_x_shower = []
    average_y_shower = []
    average_z_shower = []
    all_incident_e = []
    entries = []
    GeV = 1/1000
    
    if type(files_) == list:
        print(f'plot_distribution running on input type \'files\'')
        if energy_trans_file != '':
            energy_trans_file = os.path.join(files_[0].rsplit('/',1)[0],energy_trans_file)
            print(f'Loading file for hit e transformation inversion: {energy_trans_file}')
            # Load saved pre-processor
            scalar_e = load(open(energy_trans_file, 'rb'))
        if x_trans_file != '':
            x_trans_file = os.path.join(files_[0].rsplit('/',1)[0],x_trans_file)
            print(f'Loading file for hit x transformation inversion: {x_trans_file}')
            # Load saved pre-processor
            scalar_x = load(open(x_trans_file, 'rb'))
        if y_trans_file != '':
            y_trans_file = os.path.join(files_[0].rsplit('/',1)[0],y_trans_file)
            print(f'Loading file for hit y transformation inversion: {y_trans_file}')
            # Load saved pre-processor
            scalar_y = load(open(y_trans_file, 'rb'))
        if ine_trans_file != '':
            ine_trans_file = os.path.join(files_[0].rsplit('/',1)[0],ine_trans_file)
            print(f'Loading file for incident e transformation inversion: {ine_trans_file}')
            # Load saved pre-processor
            scalar_ine = load(open(ine_trans_file, 'rb'))
        
        # Using several files so want to take even # samples from each file for plots
        n_files = len(files_)
        print(f'n_files {type(n_files)}: {n_files}')
        print(f'# showers {type(nshowers_2_plot)} to plot: {nshowers_2_plot}')
        nshowers_2_plot = nshowers_2_plot/n_files
        
        for filename in files_:
            shower_counter=0
            print(f'File: {filename}')
            fdir = filename.rsplit('/',1)[0]
            custom_data = utils.cloud_dataset(filename, device=device)
            # Note: Shuffling can be turned off if you want to see exactly the same showers before and after transformation
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
            
            # For each batch in file
            print(f'# batches: {len(point_clouds_loader)}')
            for i, (shower_data,incident_energies) in enumerate(point_clouds_loader,0): 
                valid_hits = []
                data_np = shower_data.cpu().numpy().copy()
                incident_energies = incident_energies.cpu().numpy().copy()
                mask = ~(data_np[:,:,3] == padding_value)
                
                incident_energies = np.array(incident_energies).reshape(-1,1)
                if ine_trans_file != '':
                    # Rescale the conditional input for each shower
                    incident_energies = scalar_ine.inverse_transform(incident_energies)
                incident_energies = incident_energies.flatten().tolist()
                
                # For each shower in batch
                for j in range(len(data_np)):
                    if shower_counter >= nshowers_2_plot:
                        continue
                    shower_counter+=1
                    
                    # Only use non-padded values
                    valid_hits = data_np[j][mask[j]]
                    
                    # To transform back to original energies for plots
                    all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                    all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                    all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                    if energy_trans_file != '':
                        all_e = scalar_e.inverse_transform(all_e)
                    if x_trans_file != '':
                        all_x = scalar_x.inverse_transform(all_x)
                    if y_trans_file != '':
                        all_y = scalar_y.inverse_transform(all_y)
                    
                    all_e = all_e.flatten().tolist()
                    all_x = all_x.flatten().tolist()
                    all_y = all_y.flatten().tolist()
                    
                    # Store features of individual hits in shower
                    shower_hit_energies.extend( all_e )
                    shower_hit_x.extend( all_x )
                    shower_hit_y.extend( all_y )
                    all_z.extend( ((valid_hits).copy()[:,3]).flatten().tolist() )
                    hits_ine = [ incident_energies[j] for x in range(0,len(valid_hits[:,0])) ]
                    shower_hit_ine.extend( hits_ine )
                    
                    # Store full shower features
                    # Number of valid hits
                    entries.extend( [len(valid_hits)] )
                    # Total energy deposited by shower
                    total_deposited_e_shower.extend([ sum(all_e) ])
                    # Incident energies
                    all_incident_e.extend( [incident_energies[j]] )
                    # Hit position averages
                    average_x_shower.extend( [np.mean(all_x)] )
                    average_y_shower.extend( [np.mean(all_y)] )
                    average_z_shower.extend( [np.mean(all_z)] )
                    
    elif type(files_) == utils.cloud_dataset:
        print(f'plot_distribution running on input type \'cloud_dataset\'')
        # Note: Shuffling can be turned off if you want to see exactly the same showers before and after transformation
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
                all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                if energy_trans_file != '':
                    all_e = scalar_e.inverse_transform(all_e)
                    all_x = scalar_x.inverse_transform(all_x)
                    all_y = scalar_y.inverse_transform(all_y)
                all_e = all_e.flatten().tolist()
                all_x = all_x.flatten().tolist()
                all_y = all_y.flatten().tolist()
                
                # Store features of individual hits in shower
                shower_hit_energies.extend( all_e )
                shower_hit_x.extend( all_x )
                shower_hit_y.extend( all_y )
                all_z.extend( ((valid_hits).copy()[:,3]).flatten().tolist() )
                
                shower_hit_ine.extend( [incident_energies[j] for x in valid_hits[:,0]] )
                
                # Number of valid hits
                entries.extend( [len(valid_hits)] )
                # Total energy deposited by shower
                total_deposited_e_shower.extend( [sum(all_e)] )
                # Incident energies
                all_incident_e.extend( [energy_np[j]] )
                # Hit position averages
                average_x_shower.extend( [np.mean(all_x)] )
                average_y_shower.extend( [np.mean(all_y)] )
                average_z_shower.extend( [np.mean(all_z)] )

    return [entries, all_incident_e, total_deposited_e_shower, shower_hit_energies, shower_hit_x, shower_hit_y, all_z, shower_hit_ine, average_x_shower, average_y_shower, average_z_shower]

def create_axes():

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)
    
    # define the axis for the first colorbar
    left, width_c = width + left + 0.1, 0.01
    rect_colorbar = [left, bottom, width_c, height]
    ax_colorbar = plt.axes(rect_colorbar)
    
    # define the axis for the transformation plot
    left = left + width_c + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_trans = plt.axes(rect_scatter)
    ax_histx_trans = plt.axes(rect_histx)
    ax_histy_trans = plt.axes(rect_histy)

    # define the axis for the second colorbar
    left, width_c = left + width + 0.1, 0.01
    rect_colorbar_trans = [left, bottom, width_c, height]
    ax_colorbar_trans = plt.axes(rect_colorbar_trans)
    
    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_trans, ax_histy_trans, ax_histx_trans),
        (ax_colorbar, ax_colorbar_trans)
    )

def plot_xy(axes, X1, X2, y, ax_colorbar, hist_nbins=50, zlabel="", x0_label="", x1_label="", name=""):
    
    # scale the output between 0 and 1 for the colorbar
    y_full = y
    y = minmax_scale(y_full)
    
    # The scatter plot
    cmap = cm.get_cmap('winter')
    
    ax, hist_X2, hist_X1 = axes
    ax.set_title(name)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    
    colors = cmap(y)
    ax.scatter(X1, X2, alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for x-axis (along top)
    hist_X1.set_xlim(ax.get_xlim())
    hist_X1.hist(X1, bins=hist_nbins, orientation="vertical", color="red", ec="red")
    hist_X1.axis("off")
    
    # Histogram for y-axis (along RHS)
    hist_X2.set_ylim(ax.get_ylim())
    hist_X2.hist(X2, bins=hist_nbins, orientation="horizontal", color="grey", ec="grey")
    hist_X2.axis("off")
    
    norm = Normalize(min(y_full), max(y_full))
    cb1 = ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label=zlabel,
    )
    return

def make_plot(distributions, outdir=''):
    
    fig = plt.figure(figsize=(12, 8))
    
    X1, X2, y_X, T1, T2, y_T = distributions[0][1]
    xlabel, ylabel, zlabel = distributions[0][0]

    ax_X, ax_T, ax_colorbar = create_axes()
    axarr = (ax_X, ax_T)
    
    title = 'Non-transformed'
    plot_xy(
        axarr[0],
        X1,
        X2,
        y_X,
        ax_colorbar[0],
        hist_nbins=200,
        x0_label=xlabel,
        x1_label=ylabel,
        zlabel=zlabel,
        name=title
    )
    
    title='Transformed'
    plot_xy(
        axarr[1],
        T1,
        T2,
        y_T,
        ax_colorbar[1],
        hist_nbins=200,
        x0_label=xlabel,
        x1_label=ylabel,
        zlabel=zlabel,
        name=title
    )
    
    save_name = xlabel+'_'+ylabel+'.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    print(f'save_name: {save_name}')
    fig.savefig(os.path.join(outdir,save_name))
    
    return

def create_axes_diffusion(n_plots):
    
    axes_ = ()
    
    # Define the axis for the first plot
    # Histogram width
    width_h = 0.02
    left = 0.02
    width_buffer = 0.05
    # Scatter plot width
    width = (1-(n_plots*(width_h+width_buffer))-left)/n_plots
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, width_h, height]

    # Add axes with dimensions above in normalized units
    ax_scatter = plt.axes(rect_scatter)
    # Horizontal histogram along x-axis (Top)
    ax_histx = plt.axes(rect_histx)
    # Vertical histogram along y-axis (RHS)
    ax_histy = plt.axes(rect_histy)
    
    axes_ += ((ax_scatter, ax_histy, ax_histx),)
    
    # define the axis for the next plots
    for idx in range(0,n_plots-1):
        print(idx)
        #left = left + width + 0.22
        left = left + width + width_h + width_buffer
        left_h = left + width + 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, width_h, height]

        ax_scatter_diff = plt.axes(rect_scatter)
        # Horizontal histogram along x-axis (Top)
        ax_histx_diff = plt.axes(rect_histx)
        # Vertical histogram along y-axis (RHS)
        ax_histy_diff = plt.axes(rect_histy)
        
        axes_ += ((ax_scatter_diff, ax_histy_diff, ax_histx_diff),)
    
    return axes_

def plot_diffusion_xy(axes, X1, X2, GX1, GX2, hist_nbins=50, x0_label="", x1_label="", name="", xlim=(-1,1), ylim=(-1,1)):
    
    # The scatter plot
    ax, hist_X1, hist_X0 = axes
    ax.set_title(name)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    ax.scatter(X1, X2, alpha=0.5, marker="o", s=5, lw=0, c='blue',label='Gen')
    ax.scatter(GX1, GX2, alpha=0.5, marker="o", s=5, lw=0, c='orange',label='Geant4')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    ax.legend(loc='upper left')

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Horizontal histogram along x-axis (Top)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X1, bins=hist_nbins, orientation="vertical", color="grey", ec="grey")
    hist_X0.axis("off")
    
    # Vertical histogram along y-axis (RHS)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X2, bins=hist_nbins, orientation="horizontal", color="red", ec="red")
    hist_X1.axis("off")
    
    return

def make_diffusion_plot(distributions, outdir=''):
    
    fig = plt.figure(figsize=(50, 10))
    #fig.set_tight_layout(True)

    xlabel, ylabel = distributions[0][0]
    # Geant4/Gen distributions for x- and y-axes
    geant_x, geant_y, gen_x_t1, gen_y_t1, gen_x_t25, gen_y_t25, gen_x_t50, gen_y_t50, gen_x_t75, gen_y_t75, gen_x_t99, gen_y_t99  = distributions[0][1]
    
    # Number of plots depends on the number of diffusion steps to plot
    n_plots = (len(distributions[0][1])-2)/2
    
    # Labels of variables to plot
    ax_X, ax_T1, ax_T2, ax_T3, ax_T4 = create_axes_diffusion(int(n_plots))
    axarr = (ax_X, ax_T1, ax_T2, ax_T3, ax_T4)
    
    x_lim = ( min(min(gen_x_t1),min(geant_x)) , max(max(gen_x_t1),max(geant_x)) )
    y_lim = ( min(min(gen_y_t1),min(geant_y)) , max(max(gen_y_t1),max(geant_y)) )
    
    plot_diffusion_xy(
        axarr[0],
        gen_x_t1,
        gen_y_t1,
        geant_x,
        geant_y,
        hist_nbins=100,
        x0_label=xlabel,
        x1_label=ylabel,
        name='t=1',
        xlim=x_lim,
        ylim=y_lim
    )
    
    plot_diffusion_xy(
        axarr[1],
        gen_x_t25,
        gen_y_t25,
        geant_x,
        geant_y,
        hist_nbins=100,
        x0_label=xlabel,
        x1_label=ylabel,
        name='t=25',
        xlim=x_lim,
        ylim=y_lim
    )
    
    plot_diffusion_xy(
        axarr[2],
        gen_x_t50,
        gen_y_t50,
        geant_x,
        geant_y,
        hist_nbins=100,
        x0_label=xlabel,
        x1_label=ylabel,
        name='t=50',
        xlim=x_lim,
        ylim=y_lim
    )
    
    plot_diffusion_xy(
        axarr[3],
        gen_x_t75,
        gen_y_t75,
        geant_x,
        geant_y,
        hist_nbins=100,
        x0_label=xlabel,
        x1_label=ylabel,
        name='t=75',
        xlim=x_lim,
        ylim=y_lim
    )
    
    plot_diffusion_xy(
        axarr[4],
        gen_x_t99,
        gen_y_t99,
        geant_x,
        geant_y,
        hist_nbins=100,
        x0_label=xlabel,
        x1_label=ylabel,
        name='t=99',
        xlim=x_lim,
        ylim=y_lim
    )
    print(f'plt.axis(): {plt.axis()}')

    save_name = xlabel+'_'+ylabel+'.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    save_name = os.path.join(outdir,save_name)
    print(f'save_name: {save_name}')
    fig.savefig(save_name)

    return