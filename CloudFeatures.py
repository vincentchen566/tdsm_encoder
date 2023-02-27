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

    def __init__(self, point_clouds_loader):
        '''
        '''
        cloud = next(iter(point_clouds_loader))        
        self.num_layers = len(set([tensor.item() for tensor in cloud[0].x[:,3]]))
        self.num_clouds = len(point_clouds_loader)
        self.layers_e = {}
        self.layers_h = {}
        for layer_ in range(0,self.num_layers):
            self.layers_e[layer_] = []
            self.layers_h[layer_] = []
        
        self.layer_1_energies = []
        self.layer_2_energies = []
        self.layer_3_energies = []
        self.layer_4_energies = []
        self.layer_5_energies = []
        self.all_energies = []
        self.layer_1_hits = []
        self.layer_2_hits = []
        self.layer_3_hits = []
        self.layer_4_hits = []
        self.layer_5_hits = []
        self.all_hits = []
        self.layers_mean_e = []
        self.layers_mean_h = []
        
        for i, (cloud_data,injection_energy) in enumerate(point_clouds_loader,0):
            if i>10:
                break
            
            # Sum of the total energy deposited by a cloud per layer
            for layer_ in range(0,self.num_layers):
                self.layers_e[layer_].append( sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == layer_]) )    
                self.layers_h[layer_].append( len([z for z in cloud_data.x[:,3] if z == layer_]) )
            
            # Sum of the total energy deposited by a cloud
            self.all_energies.append(sum([E.item()/1000 for E in cloud_data.x[:,0]]))
            # Number of hits in a cloud
            self.all_hits.append(len([z for z in cloud_data.x[:,3]]))
            
            

    def calculate_mean_energies(self):
        '''Calculate the average energy deposited (per layer) for all clouds
        '''
        for layer_ in range(0,self.num_layers):
            self.layers_mean_e.append(sum(self.layers_e[layer_])/self.num_clouds)
        
        return self.layers_mean_e
    
    def calculate_mean_nhits(self):
        '''Calculate the average number of hits (per layer) for all clouds
        '''
        for layer_ in range(0,self.num_layers):
            self.layers_mean_h.append( sum(self.layers_h[layer_])/self.num_clouds )
        
        return self.layers_mean_h