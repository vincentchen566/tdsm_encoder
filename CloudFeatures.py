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
        self.num_clouds = len(point_clouds_loader)
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
        for i, (cloud_data,injection_energy) in enumerate(point_clouds_loader,0):
            if i>1000:
                break
                
            # Sum of the total energy deposited by a cloud per layer
            self.layer_1_energies.append(sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == 0]))
            self.layer_2_energies.append(sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == 1]))
            self.layer_3_energies.append(sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == 2]))
            self.layer_4_energies.append(sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == 3]))
            self.layer_5_energies.append(sum([E.item()/1000 for E,z in zip(cloud_data.x[:,0], cloud_data.x[:,3]) if z == 4]))
            
            # Sum of the total energy deposited by a cloud
            self.all_energies.append(sum([E.item()/1000 for E in cloud_data.x[:,0]]))

            # Number of hits per layer in a cloud
            self.layer_1_hits.append(len([z for z in cloud_data.x[:,3] if z == 0]))
            self.layer_2_hits.append(len([z for z in cloud_data.x[:,3] if z == 1]))
            self.layer_3_hits.append(len([z for z in cloud_data.x[:,3] if z == 2]))
            self.layer_4_hits.append(len([z for z in cloud_data.x[:,3] if z == 3]))
            self.layer_5_hits.append(len([z for z in cloud_data.x[:,3] if z == 4]))
            
            # Number of hits in a cloud
            self.all_hits.append(len([z for z in cloud_data.x[:,3]]))

    def calculate_mean_energies(self):
        '''Calculate the average energy deposited (per layer) for all clouds
        '''
        layer_1_meanenergy = sum(self.layer_1_energies)/self.num_clouds
        layer_2_meanenergy = sum(self.layer_2_energies)/self.num_clouds
        layer_3_meanenergy = sum(self.layer_3_energies)/self.num_clouds
        layer_4_meanenergy = sum(self.layer_4_energies)/self.num_clouds
        layer_5_meanenergy = sum(self.layer_5_energies)/self.num_clouds
        layers_mean = [layer_1_meanenergy, layer_2_meanenergy, layer_3_meanenergy, layer_4_meanenergy, layer_5_meanenergy]
        
        return layers_mean
    
    def calculate_mean_nhits(self):
        '''Calculate the average number of hits (per layer) for all clouds
        '''
        layer_1_meanhits = sum(self.layer_1_hits)/self.num_clouds
        layer_2_meanhits = sum(self.layer_2_hits)/self.num_clouds
        layer_3_meanhits = sum(self.layer_3_hits)/len(self.layer_3_hits)
        layer_4_meanhits = sum(self.layer_4_hits)/len(self.layer_4_hits)
        layer_5_meanhits = sum(self.layer_5_hits)/len(self.layer_5_hits)
        layers_mean = [layer_1_meanhits,layer_2_meanhits,layer_3_meanhits,layer_4_meanhits,layer_5_meanhits]
        return layers_mean