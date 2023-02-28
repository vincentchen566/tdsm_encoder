import os, math, sys
import numpy as np
import matplotlib.pyplot as plt

class CloudFeatures:
    '''Compute high-level (averaged) features of the point clouds
    based on hits. Also reads in the detector geometry (from .xml) in order 
    to create plots/images. For the time being, we base a lot of this on the 
    HighLevelFeature.py and XMLHandler.py scripts in the CaloChallenge code as 
    we are using the same datasets.
    '''

    def __init__(self, layer_set):
        '''
        '''
        
        self.layer_set = layer_set
        self.num_layers = len(self.layer_set)
        self.z_binning = np.linspace(0, self.num_layers, self.num_layers)
        
        self.total_energy = []
        self.n_hits = []
        self.layers_mean_e = {}
        self.layers_mean_h = {}
        for layer_ in range(0,self.num_layers+1):
                self.layers_mean_e[layer_] = []
                self.layers_mean_h[layer_] = []
        
    def basic_quantities(self, loader):

        self.n_hits.clear()
        self.total_energy.clear()
        for layer_ in range(0,self.num_layers+1):
            self.layers_mean_e[layer_].clear()
            self.layers_mean_h[layer_].clear()

        self.total_energies_sum = 0
        for i, (cloud_data,injection_energy) in enumerate(loader,0):
            if i>10:
                break
            print('cloud: ', i)
            # Individual cloud features
            cloud_hit_energies = []
            cloud_hit_x = []
            cloud_hit_y = []
            cloud_hit_z = []
            cloud_layers_e = {}
            cloud_layers_h = {}
            cloud_nhits = 0
            cloud_total_energy = 0
            for layer_ in range(0,self.num_layers+1):
                cloud_layers_e[layer_] = []
                cloud_layers_h[layer_] = []
            
            for hit in cloud_data.x:
                # All hits energy deposits in cloud in GeV
                cloud_hit_energies.append(hit[0].item()/1000)
                # All hits x position in cloud
                cloud_hit_x.append(hit[1].item())
                # All hits y position in cloud
                cloud_hit_y.append(hit[2].item())
                # All hits z position in cloud
                cloud_hit_z.append(hit[3].item())
                # To which layer does hit belong
                layer_match = self.closest_value(self.layer_set,hit[3].item())
                # Append hit to relevant layer entry in dictionary
                cloud_layers_e[layer_match].append(hit[0].item()/1000)
            
            # Total energy deposited by cloud
            cloud_total_energy = sum(cloud_hit_energies)
            self.total_energy.append(cloud_total_energy)
            # Number of hits in a cloud 
            cloud_nhits = len(cloud_hit_energies)
            self.n_hits.append(cloud_nhits)
            
            # Append the energy per layer from each cloud to relativelist
            for layer_ in range(0,self.num_layers):
                self.layers_mean_e[layer_].append(cloud_layers_e[layer_])
                self.layers_mean_h[layer_].append(len(cloud_layers_e[layer_]))


    def calculate_mean_energies(self):
        '''Calculate the average energy deposited (per layer) for all clouds
        '''
        layers_av = []
        for layer_ in range(0,self.num_layers):
            layer_average = sum(self.layers_mean_e[layer_][0])/len(self.layers_mean_e[layer_])
            layers_av.append( layer_average )
        return layers_av
    
    def calculate_mean_nhits(self):
        '''Calculate the average number of hits (per layer) for all clouds
        '''
        layers_av = []
        for layer_ in range(0,self.num_layers):
            layer_average = sum(self.layers_mean_h[layer_])/len(self.layers_mean_h[layer_])
            layers_av.append(layer_average)
        
        return layers_av

    def closest_value(self, inlist, invalue):
        arr_ = list(inlist)
        #print('arr_: ', arr_)
        invalue = invalue
        diff = [abs(x-invalue) for x in arr_]
        match_index = int(np.argmin(diff))
        #print('min_ index: ', match_index)
        return arr_[match_index]