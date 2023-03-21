import os, math, sys
import numpy as np
import matplotlib.pyplot as plt

class CloudFeatures:
    '''Compute high-level (averaged) features of the point clouds
    based on hits. Also reads in the detector geometry (from .xml) in order 
    to create plots/images.
    '''

    def __init__(self, layer_set):
        self.layer_set = layer_set
        self.num_layers = len(self.layer_set)
        self.z_binning = np.linspace(0, self.num_layers, self.num_layers)
        
        self.total_energy = 0
        self.n_hits = -1
        self.layers_sum_e = {}
        self.layers_nhits = {}
        for layer_ in range(0,self.num_layers):
                self.layers_sum_e[layer_] = []
                self.layers_nhits[layer_] = []
        
    def basic_quantities(self, cloud, injection_energy):
        for layer_ in range(0,self.num_layers):
            self.layers_sum_e[layer_].clear()
            self.layers_nhits[layer_].clear()

        #self.total_energies_sum = 0
        
        # Individual cloud features
        cloud_hit_energies = []
        cloud_hit_x = []
        cloud_hit_y = []
        cloud_hit_z = []
        cloud_layers_e = {}
        cloud_layers_h = {}
        cloud_nhits = 0
        for layer_ in range(0,self.num_layers):
            cloud_layers_e[layer_] = [0]
            cloud_layers_h[layer_] = [0]
        for hit in cloud.x:
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
        self.total_energy = sum(cloud_hit_energies)
        # Number of hits in a cloud 
        cloud_nhits = len(cloud_hit_energies)
        self.n_hits = cloud_nhits
        
        # Append the total energy / nhits per layer for the cloud to relative lists
        for layer_ in range(0,self.num_layers):
            self.layers_sum_e[layer_].append(sum(cloud_layers_e[layer_]))
            self.layers_nhits[layer_].append(len(cloud_layers_e[layer_]))

    def closest_value(self, inlist, invalue):
        arr_ = list(inlist)
        #print('arr_: ', arr_)
        invalue = invalue
        diff = [abs(x-invalue) for x in arr_]
        match_index = int(np.argmin(diff))
        #print('min_ index: ', match_index)
        return arr_[match_index]