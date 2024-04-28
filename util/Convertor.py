import time, functools, torch, os, sys, random, fnmatch, psutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util.XMLHandler import XMLHandler
import pandas as pd
import util.Evaluate
from datasets.pad_events_threshold import Preprocessor
import pickle
import h5py
class Convertor:
    def __init__(self, dataset_name, label, padding_value = 0, device='cpu', preprocessor='datasets/test/dataset_2_padded_nentry1129To1269_preprocessor.pkl'):

        dataset = torch.load(dataset_name, map_location = torch.device(device))
        label_tensor = torch.ones(len(dataset[0]), device=device) * label
        self.dataset = util.Evaluate.evaluate_dataset(dataset[0], dataset[1], label_tensor, device)
        self.device  = device
        self.padding_value = padding_value
        dbfile = open(preprocessor, 'rb')
        self.preprocessor  = pickle.load(dbfile)
        dbfile.close()
        print(len(self.dataset))

    def padding(self):
        self.dataset.padding(self.padding_value)

    def digitize(self, particle='electron',xml_bin='dataset_generation_code/binning_dataset_2.xml'):
      self.dataset.digitize(particle, xml_bin, pad_value=self.padding_value)
        
    def invert(self, new_padding_value=0.0):
        invert_data = []
        invert_inE  = []
        for index, data_ in enumerate(self.dataset):
          E_ = np.asarray((data_[0][:,0])).reshape(-1,1)
          X_ = np.asarray((data_[0][:,1])).reshape(-1,1)
          Y_ = np.asarray((data_[0][:,2])).reshape(-1,1)
          Z_ = np.asarray((data_[0][:,3])).reshape(-1,1)
          inE_ = data_[1]
          new_inE_ = self.preprocessor.inverse_transform_incident_energy(inE_)
          new_inE_ = new_inE_.item()
          new_E_, new_X_, new_Y_, new_Z_ = self.preprocessor.inverse_transform_hit(E_, X_, Y_, Z_, new_inE_, self.padding_value, new_padding_value)
          new_E_ = torch.from_numpy( new_E_.flatten())
          new_X_ = torch.from_numpy( new_X_.flatten())
          new_Y_ = torch.from_numpy( new_Y_.flatten())
          new_Z_ = torch.from_numpy( new_Z_.flatten())
          invert_data.append(torch.stack((new_E_, new_X_, new_Y_, new_Z_),-1))
          invert_inE.append(new_inE_)

        self.dataset.data = invert_data
        self.dataset.inE  = torch.tensor(invert_inE, device=self.device)
        self.padding_value = new_padding_value


    def transform(self, new_padding_value=0.0): 
        data = []
        for index, data_ in enumerate(self.dataset):
          E_ = np.asarray((data_[0][:,0])).reshape(-1,1)
          X_ = np.asarray((data_[0][:,1])).reshape(-1,1)
          Y_ = np.asarray((data_[0][:,2])).reshape(-1,1)
          Z_ = np.asarray((data_[0][:,3])).reshape(-1,1)
          mask = E_ == self.padding_value
          inE_ = data[1]
          new_E_, new_X_, new_Y_, new_Z_ = self.preprocessor.transform(E_, X_, Y_, Z_, inE_)
          mask = mask.reshape(np.shape(new_E_))
          new_E_[mask] = (np.ones(np.shape(new_E_)) * new_padding_value)[mask]
          new_X_[mask] = (np.ones(np.shape(new_X_)) * new_padding_value)[mask]
          new_Y_[mask] = (np.ones(np.shape(new_Y_)) * new_padding_value)[mask]
          new_Z_[mask] = (np.ones(np.shape(new_Z_)) * new_padding_value)[mask]
          

          new_E_ = torch.from_numpy( new_E_.flatten())
          new_X_ = torch.from_numpy( new_X_.flatten())
          new_Y_ = torch.from_numpy( new_Y_.flatten())
          new_Z_ = torch.from_numpy( new_Z_.flatten())
          data.append(torch.stack((new_E_, new_X_, new_Y_, new_Z_),-1))
        self.dataset.data = data
        self.padding_value = new_padding_value

    def to_h5py(self, outfile):
        h5f = h5py.File(outfile, 'w')
        h5f.create_dataset('showers', data=self.dataset.data_np)
        h5f.create_dataset('incident_energies', data=torch.unsqueeze((self.dataset.inE),1).cpu().numpy().copy())
        h5f.close()
