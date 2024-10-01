# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class data_set(Dataset):
    
    def __init__(self,Y_train,Z_train,X_train,N,L):
        self.N=N
        self.L=L
   
        """
        

        Parameters
        ----------
        X_train : training data. N data points of length L
        Y_train : Output data. N data points
        N : Number of training points.
        L : length of input data points.

        Returns
        -------
        None.

        """
        if not torch.is_tensor(X_train):
            self.y = torch.from_numpy(Y_train)
        else:
            self.y=Y_train
            
        if not torch.is_tensor(Z_train):
            self.Z = torch.from_numpy(Z_train)
        else:
            self.Z=Z_train
            
        if not torch.is_tensor(X_train):
            self.X = torch.from_numpy(X_train)
        else:
            self.X=X_train
        self.y=self.y.reshape([N,-1])
        self.X=self.X.reshape([N,L])
        
        self.torch=True
        
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_data=self.X[idx,:]
        y_data=self.y[idx,:]
        z_data=self.Z[idx,:]
        in_two_channels =torch.stack([y_data, z_data])

        return in_two_channels , x_data
    def from_numpy(self):
        if torch.is_tensor(self.X):
            self.X=self.X.detach().numpy()
            self.Y=self.Y.detach().numpy()
        else:
            pass
    def to_torch(self):
        if not torch.is_tensor(self.X):
            self.X=torch.from_numpy(X)
            self.Y=torch.from_numpy(Y)
        else:
            pass
        
    
def next_batch(ds,batchSize):
	# loop over the dataset
	for i in range(0, len(ds), batchSize):
		# yield a tuple of the current batched data and labels
		yield i,(ds.y[i:i + batchSize].float(), ds.X[i:i + batchSize].float())
        
        
        
