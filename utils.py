import h5py, math, torch
from torch.utils.data import Dataset
#from torch_geometric.data import Dataset

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

class uniform_energy_sampler():
    def __init__ (self, filename, sample_batch_size):
        ''' 
        '''
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
