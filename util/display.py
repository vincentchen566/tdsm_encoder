import torch, sys, os
import data_utils as utils
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

def invert_transform_e(e_):
    original_e = 0.5 * np.log( (1+np.array(e_)) / (1-np.array(e_)) )
    original_e = np.nan_to_num(original_e)
    original_e = np.reshape(original_e,(-1,))
    return original_e

# Pass a list of plots, bins and titles and function will recursively loop through and plot
class recursive_plot:
    def __init__(self, n_plots, name1, vals_list, x_titles, n_bins=0, xvals_list=[], y_titles=[], colors=[]):
        '''
        Plot list of any number plots
        Args:
            nplots: number of plots to make
            name1: save name
            vals_list: list of lists/arrays of values/datapoints to plot
            n_bins: number of bins
            x_titles: x-axis label for each histogram
        '''
        self.n_plots = n_plots
        self.fig, self.ax = plt.subplots(1,n_plots, figsize=(25,4))
        self.fig.suptitle(name1)
        self.vals_list = vals_list
        self.xvals_list = xvals_list
        self.n_bins = n_bins
        self.x_titles = x_titles
        self.y_titles = y_titles
        self.colors = colors
    
    def rec_plot(self):
        if len(self.vals_list) == 0:
            return None
        plot_idx = self.n_plots-len(self.vals_list)
        self.ax[plot_idx].hist(self.vals_list[0], bins=self.n_bins[0])
        self.ax[plot_idx].set_xlabel(self.x_titles[0])
        self.vals_list.pop(0)
        self.n_bins.pop(0)
        self.x_titles.pop(0)
        self.ax[plot_idx].set_yscale('log')
        self.rec_plot()

    def rec_scatter(self):
        if len(self.vals_list) == 0:
            return None
        if len(self.xvals_list) == 0:
            print('WARNING: no xvals provided for scatter plot')
            return None
        plot_idx = self.n_plots-len(self.vals_list)
        self.ax[plot_idx].scatter(self.xvals_list[0],self.vals_list[0])
        self.ax[plot_idx].set_xlabel(self.x_titles[0])
        self.ax[plot_idx].set_ylabel(self.y_titles[0])
        self.vals_list.pop(0)
        self.xvals_list.pop(0)
        self.y_titles.pop(0)
        self.x_titles.pop(0)
        self.ax[plot_idx].set_yscale('log')
        self.rec_scatter()

    def save(self, savename):
        self.fig.savefig(savename)
        return

def plot_loss_vs_epoch(eps_, train_losses, test_losses, odir='', zoom=False):
    
    fig_, ax_ = plt.subplots(ncols=1, figsize=(4,4))
    
    if zoom==True:
        # Only plot the last 80% of the epochs
        ax_.set_title('zoom')
        zoom_split = int(len(train_losses) * 0.8)
    else:
        ax_.set_title('Loss vs. epoch')
        zoom_split = 0
        
    ax_.set_ylabel('Loss')
    ax_.set_xlabel('Epoch')
    ax_.set_yscale('log')
    eps_zoom = eps_[zoom_split:]
    train_loss_zoom = train_losses[zoom_split:]
    test_loss_zoom = test_losses[zoom_split:]
    ax_.plot(eps_zoom,train_loss_zoom, c='blue', label='training')
    ax_.plot(eps_zoom,test_loss_zoom, c='red', label='testing')
    ax_.legend(loc='upper right')
    
    if zoom==True:
        z = np.polyfit(eps_zoom, train_loss_zoom, 1)
        trend = np.poly1d(z)
        ax_.plot(eps_zoom,trend(eps_zoom), c='black', label='trend')
        fig_.savefig(os.path.join(odir,'loss_v_epoch_zoom.png'))
    else:
        fig_.savefig(os.path.join(odir,'loss_v_epoch.png'))
    
    return

def plot_distribution(files_:Union[ list , utils.cloud_dataset], nshowers_2_plot=100, padding_value=0.0, batch_size=1, energy_trans=False, masking=False):
    
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
    sum_x_shower = []
    sum_y_shower = []
    sum_z_shower = []
    
    sum_e_shower = []
    mean_x_shower = []
    mean_y_shower = []
    mean_z_shower = []
    
    all_incident_e = []
    entries = []
    GeV = 1/1000
    print(f'# showers to plot: {nshowers_2_plot}')
    if type(files_) == list:
        print(f'plot_distribution running on input type \'files\'')
        
        # Using several files so want to take even # samples from each file for plots
        n_files = len(files_)
        print(f'# files: {n_files}')
        nshowers_per_file = [nshowers_2_plot//n_files for x in range(n_files)]
        r_ = nshowers_2_plot % nshowers_per_file[0]
        nshowers_per_file[-1] = nshowers_per_file[-1]+r_
        print(f'# showers per file: {nshowers_per_file}')
        
        for file_idx in range(len(files_)):
            filename = files_[file_idx]
            shower_counter=0
            print(f'File: {filename}')
            fdir = filename.rsplit('/',1)[0]
            custom_data = utils.cloud_dataset(filename, device=device)
            # Note: Shuffling can be turned off if you want to see exactly the same showers before and after transformation
            point_clouds_loader = DataLoader(custom_data, batch_size=batch_size, shuffle=True)
            
            # For each batch in file
            print(f'# batches: {len(point_clouds_loader)}')
            for i, (shower_data,incident_energies,_) in enumerate(point_clouds_loader,0): 
                valid_hits = []
                data_np = shower_data.cpu().numpy().copy()
                incident_energies = incident_energies.cpu().numpy().copy()
                
                # Mask for padded values
               # mask = ~(data_np[:,:,0] <= 0.01)
                mask = ~(data_np[:,:,0] == padding_value)

                incident_energies = np.array(incident_energies).reshape(-1,1)
                incident_energies = incident_energies.flatten().tolist()
                
                # For each shower in batch
                for j in range(len(data_np)):
                    if shower_counter >= nshowers_per_file[file_idx]:
                        break
                    shower_counter+=1
                    
                    # Only use non-padded values for plots
                    valid_hits = data_np[j]#[mask[j]]
                    if masking:
                        valid_hits = data_np[j][mask[j]]
                    # To transform back to original energies for plots
                    all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                    all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                    all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                    
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
                    # Hit means
                    sum_e_shower.extend( [np.sum(all_e)] )
                    mean_x_shower.extend( [np.mean(all_x)] )
                    mean_y_shower.extend( [np.mean(all_y)] )
                    mean_z_shower.extend( [np.mean(all_z)] )
                    
    elif type(files_) == utils.cloud_dataset:
        print(f'plot_distribution running on input type \'cloud_dataset\'')
        
        # Note: Shuffling can be turned off if you want to see specific showers
        point_clouds_loader = DataLoader(files_, batch_size=batch_size, shuffle=True)

        for i, (shower_data,incident_energies,_) in enumerate(point_clouds_loader,0):
            valid_hits = []
            data_np = shower_data.cpu().numpy().copy()
            energy_np = incident_energies.cpu().numpy().copy()
            
            mask = ~(data_np[:,:,0] == padding_value)
            # For each shower in batch
            for j in range(len(data_np)):
                if shower_counter >= nshowers_2_plot:
                    break
                    
                shower_counter+=1
                valid_hits = data_np[j]#[mask[j]]
                if masking:
                    valid_hits = data_np[j][mask[j]]
                # To transform back to original energies for plots                    
                all_e = np.array(valid_hits[:,0]).reshape(-1,1)
                all_x = np.array(valid_hits[:,1]).reshape(-1,1)
                all_y = np.array(valid_hits[:,2]).reshape(-1,1)
                    
                all_e = all_e.flatten().tolist()
                all_x = all_x.flatten().tolist()
                all_y = all_y.flatten().tolist()
                
                # Store features of individual hits in shower
                shower_hit_energies.extend( all_e )
                shower_hit_x.extend( all_x )
                shower_hit_y.extend( all_y )
                all_z.extend( ((valid_hits).copy()[:,3]).flatten().tolist() )
                
                shower_hit_ine.extend( [energy_np[j] for x in valid_hits[:,0]] ) #Use CPU version of incident_energies
                
                # Number of valid hits
                entries.extend( [len(valid_hits)] )
                # Total energy deposited by shower
                total_deposited_e_shower.extend( [sum(all_e)] )
                # Incident energies
                all_incident_e.extend( [energy_np[j]] )
                # Hit means
                sum_e_shower.extend( [np.sum(all_e)] )
                mean_x_shower.extend( [np.mean(all_x)] )
                mean_y_shower.extend( [np.mean(all_y)] )
                mean_z_shower.extend( [np.mean(all_z)] )

    return [entries, all_incident_e, shower_hit_ine, shower_hit_energies, shower_hit_x, shower_hit_y, all_z, sum_e_shower, mean_x_shower, mean_y_shower, mean_z_shower]

def perturbation_1D(distributions, titles, outdir=''):
    xlabel = distributions[0][0]
    p0, p1, p2, p3, p4, p5 = distributions[0][1]
    
    fig, axs_1 = plt.subplots(1,5, figsize=(24,8), sharey=True)
    bins=np.histogram(np.hstack((p0,p1)), bins=25)[1]
    axs_1[0].set_title(titles[0])
    axs_1[0].set_xlabel(xlabel)
    axs_1[0].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[0].hist(p1, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[0].set_yscale('log')
    axs_1[0].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p2)), bins=25)[1]
    axs_1[1].set_title(titles[1])
    axs_1[1].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[1].hist(p2, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[1].set_yscale('log')
    axs_1[1].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p3)), bins=25)[1]
    axs_1[2].set_title(titles[2])
    axs_1[2].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[2].hist(p3, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[2].set_yscale('log')
    axs_1[2].legend(loc='upper right')

    bins=np.histogram(np.hstack((p0,p4)), bins=25)[1]
    axs_1[3].set_title(titles[3])
    axs_1[3].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[3].hist(p4, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[3].set_yscale('log')
    axs_1[3].legend(loc='upper right')
    
    bins=np.histogram(np.hstack((p0,p5)), bins=25)[1]
    axs_1[4].set_title(titles[4])
    axs_1[4].hist(p0, bins, alpha=0.5, color='orange', label='un-perturbed')
    axs_1[4].hist(p5, bins, alpha=0.5, color='red', label='perturbed')
    axs_1[4].set_yscale('log')
    axs_1[4].legend(loc='upper right')
    
    fig.show()
    save_name = xlabel+'_perturbation_1D.png'
    save_name = save_name.replace(' ','').replace('[','').replace(']','')
    save_name = os.path.join(outdir,save_name)
    print(f'save_name: {save_name}')
    fig.savefig(save_name)
    return

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
    
    fig = plt.figure(figsize=(12, 10))
    
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
    ax.scatter(GX1, GX2, alpha=0.5, marker="o", s=8, lw=0, c='orange',label='Geant4')
    ax.scatter(X1, X2, alpha=0.5, marker="o", s=8, lw=0, c='blue',label='Gen')
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

def make_diffusion_plot(distributions, titles=[], outdir=''):
    
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
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[0]} (noisy)',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t25),min(geant_x)) , max(max(gen_x_t25),max(geant_x)) )
    y_lim = ( min(min(gen_y_t25),min(geant_y)) , max(max(gen_y_t25),max(geant_y)) )
    plot_diffusion_xy(
        axarr[1],
        gen_x_t25,
        gen_y_t25,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[1]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t50),min(geant_x)) , max(max(gen_x_t50),max(geant_x)) )
    y_lim = ( min(min(gen_y_t50),min(geant_y)) , max(max(gen_y_t50),max(geant_y)) )
    plot_diffusion_xy(
        axarr[2],
        gen_x_t50,
        gen_y_t50,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[2]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t75),min(geant_x)) , max(max(gen_x_t75),max(geant_x)) )
    y_lim = ( min(min(gen_y_t75),min(geant_y)) , max(max(gen_y_t75),max(geant_y)) )
    plot_diffusion_xy(
        axarr[3],
        gen_x_t75,
        gen_y_t75,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[3]}',
        xlim=x_lim,
        ylim=y_lim
    )
    
    x_lim = ( min(min(gen_x_t99),min(geant_x)) , max(max(gen_x_t99),max(geant_x)) )
    y_lim = ( min(min(gen_y_t99),min(geant_y)) , max(max(gen_y_t99),max(geant_y)) )
    plot_diffusion_xy(
        axarr[4],
        gen_x_t99,
        gen_y_t99,
        geant_x,
        geant_y,
        hist_nbins=50,
        x0_label=xlabel,
        x1_label=ylabel,
        name=f't={titles[4]}',
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

def make_diffusion_plot_v2(distributions, titles=[], outdir='', steps=[]):
  for key in distributions:
    fig = plt.figure(figsize=(50, 10))
    n_plots = len(steps)
    axarr = create_axes_diffusion(n_plots)
    for idx, step in enumerate(steps):
      xlabel, ylabel = distributions[key][step][0]
      geant_x, geant_y, gen_x_t, gen_y_t = distributions[key][step][1]
      x_lim = (min(min(gen_x_t), min(geant_x)), max(max(gen_x_t), max(geant_x)))
      y_lim = (min(min(gen_y_t), min(geant_y)), max(max(gen_y_t), max(geant_y)))
      plot_diffusion_xy(
         axarr[idx],
         gen_x_t,
         gen_y_t,
         geant_x,
         geant_y,
         hist_nbins=50,
         x0_label=xlabel,
         x1_label=ylabel,
         name=f't={step} (noisy)',
         xlim = x_lim,
         ylim = y_lim
      )
    fig.savefig(os.path.join(outdir, '{}_diffusion_2D.png'.format(key)))
  return
