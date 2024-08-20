import time, functools, torch, os, sys, random, fnmatch, psutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

import XMLHandler
import pandas as pd
import Evaluate
import pickle
import h5py

class Preprocessor:
    def __init__(self):
        self.maxe_ = 5000.
        self.mine_ = 0.2
        self.maxz_ = 12.0
        self.minz_ = 0.0

    ########################
    ##  Incident  Energy  ##
    ########################

    def fit_incident_energy(self, ine_):
        self.maxe_ = np.max(ine_)
        self.mine_ = np.min(ine_)
        return

    def transform_incident_energy(self, ine_):
        new_ine = (ine_ - self.mine_) / (self.maxe_ - self.mine_)
        return new_ine

    def inverse_transform_incident_energy(self, ine_):
        new_ine = (self.maxe_ - self.mine_)*ine_ + self.mine_
        return new_ine

    ######################
    ##  Transform XYZE  ##
    ######################

    def fit(self,Z):
      if self.maxz_ is None:
        self.maxz_ = np.max(Z)
        self.minz_ = np.min(Z)
      else:
        self.maxz_ = max(np.max(Z), self.maxz_)
        self.minz_ = min(np.min(Z), self.minz_)


    def transform_hit_xy(self, hit_pos):
      new_pos = 1/(1+np.exp(-0.07*hit_pos))
      new_pos = np.nan_to_num(new_pos)
      new_pos = np.reshape(new_pos, (-1,))
      return new_pos

    def inverse_transform_hit_xy(self, new_pos, mask, new_padding_value):
      new_pos = np.log(1./new_pos - 1)/(-0.07)
      pad  = np.ones((len(new_pos),1)) * new_padding_value
      new_pos[mask] = pad[mask]
      new_pos = np.nan_to_num(new_pos)
      new_pos = np.reshape(new_pos, (-1,))
      return new_pos

    def transform_hit_z(self, z_):
      z_ = (z_ - self.minz_) / (self.maxz_ - self.minz_)
      return z_

    def inverse_transform_hit_z(self, z_, mask, new_padding_value):
      z_ = (self.maxz_ - self.minz_)*z_ + self.minz_
      z_[mask] = (np.ones((len(z_),1)) * new_padding_value)[mask]
      z_ = np.reshape(z_, (-1,))
      return z_

    def transform_hit_e(self, e_, incident_energy):
      #with np.errstate(invalid="raise"):
      new_e = e_ / (incident_energy * 2.) 
      new_e = 1e-6 + (1.- 2e-6)*new_e
      new_e = (np.log(new_e/(1-new_e)))
      
      new_e = np.nan_to_num(new_e)
      new_e = np.reshape(new_e, (-1,))
      return new_e

    def inverse_transform_hit_e(self, e_, mask, new_padding_value, incident_energy):

      new_e = (((np.exp(e_))/(1.+(np.exp(e_)))) - 1e-6) * 2. * incident_energy / (1.-2e-6)

      new_e[mask] = (np.ones((len(new_e), 1))*new_padding_value)[mask]
      new_e = np.reshape(new_e, (-1,))
      return new_e

    def transform(self, E,X,Y,Z, incident_energy):
      new_E = self.transform_hit_e(E, incident_energy)
      new_X = self.transform_hit_xy(X)
      new_Y = self.transform_hit_xy(Y)
      new_Z = self.transform_hit_z(Z)
      return new_E, new_X, new_Y, new_Z

    def inverse_transform_hit(self, E,X,Y,Z, incident_energy, padding_value, new_padding_value):
      mask = (E == padding_value)
      new_E = self.inverse_transform_hit_e(E, mask, new_padding_value, incident_energy)
      new_X = self.inverse_transform_hit_xy(X, mask, new_padding_value)
      new_Y = self.inverse_transform_hit_xy(Y, mask, new_padding_value)
      new_Z = self.inverse_transform_hit_z(Z, mask, new_padding_value)
      return new_E, new_X, new_Y, new_Z


class Convertor:
    def __init__(self, dataset_name, label, padding_value = 0, device='cpu', preprocessor='datasets/test/dataset_2_padded_nentry1129To1269_preprocessor.pkl'):

        dataset = torch.load(dataset_name, map_location = torch.device(device))
        label_tensor = torch.ones(len(dataset[0]), device=device) * label
        self.dataset = Evaluate.evaluate_dataset(dataset[0], dataset[1], label_tensor, device)
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
