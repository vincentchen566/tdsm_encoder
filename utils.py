import h5py
import math
import torch
from torch.utils.data import Dataset

class custom_dataset(Dataset):
    def __init__(self,file_name, train_test, train_test_split):
        file_ = h5py.File(file_name, 'r')
        n_examples = math.floor(file_['incident_energies'].shape[0]*train_test_split)
        self.showers_ = file_['showers']
        self.energies_ = file_['incident_energies']
    
    def __len__(self):
        return len(self.showers_)
    
    def __getitem__(self, idx):
        '''Return item requested by idx (this returns a single sample)
        Pytorch DataLoader class will use this method to make an iterable for train/test/val loops'''
        shower_ = self.showers_[idx]
        energy_ = self.energies_[idx]
        return shower_, energy_

class uniform_energy_sampler():
    def __init__ (self, filename, sample_batch_size):
        file_ = h5py.File(filename, 'r')
        self.energies = file_['incident_energies']
        self.min_energy = min(file_['incident_energies'][:])
        self.max_energy = max(file_['incident_energies'][:])
        # Get (1 x sample_batch_size) matrix of numbers in uniform range [0,1)
        sampled_energies_ = torch.rand(sample_batch_size)
        # Produce numbers in the range [0, max_energy-min_energy)
        energy_span = (self.max_energy-self.min_energy)
        # Produce numbers uniformaly distributed between min and max
        self.energy_samples = (energy_span[0] * sampled_energies_) + self.min_energy

    def __len__(self,idx):
        return len(self.energy_samples)

    def __getitem__(self,idx):
        energy_sample_ = self.energy_samples[ix]
        return energy_sample_

class cloud_dataset(Dataset):
    def __init__(self, data, condition, transform=None):
        self.data = data
        self.condition = torch.LongTensor(condition)
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index]
        y = self.condition[index]
        return x,y
    def __len__(self):
        return len(self.data)