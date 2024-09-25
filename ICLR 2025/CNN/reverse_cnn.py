# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:09:42 2022

@author: Joe
"""

import numpy as np
import torch
import time
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import dataloader
from . import utilities as u




    
class reverse_CNN(nn.Module):
    def __init__(self,nodes,num_hidden_layers,channels,L=6000,dilation=1,device='cuda'):
        super(reverse_CNN, self).__init__()
        ###work out the padding needed for consistent shape
        nodes=nodes+1 if nodes%2==0 else nodes
        self.nodes=nodes
        self.pad=nodes//2
        self.dilation=dilation
        self.device=device
        ###define the parameters
        self.L=L
        self.inputs=1
        self.lr='none'
        self.channels=channels
        self.num_hidden_layers=num_hidden_layers
        #store the results in the CNN object
        self.losses=[]
        self.history=[]
        self.Ks=[]
        self.layers={}
        ##define activation function
        self.activ = nn.ReLU()
        self.M=L
        self.M_p=300  ###number of parameters to define K
        self.points=nn.Parameter(torch.Tensor([1]*(self.M_p+2)))  ##points representing K
   
        self.initialise_newtork_yx()

   
        
    def initialise_newtork_yx(self):
        #define batch normalisations for the different numbers of channels
        ##input batch normalisation
        self.batch1=nn.BatchNorm1d(num_features=self.inputs)
        ##hidden layer batch normalisation
        self.batch2=nn.BatchNorm1d(num_features=self.channels)
        #create the layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels=self.inputs, out_channels=self.channels, kernel_size=self.nodes,padding=self.pad,dilation=self.dilation))
        for i in range(1,self.num_hidden_layers):
            self.layers.append(nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=self.nodes,padding=self.pad))
        self.layers.append(nn.Conv1d(in_channels=self.channels, out_channels=1, kernel_size=1,padding=0))
        #initialise the weights
        torch.nn.init.xavier_uniform_(self.layers[-1].weight)
        self.to(self.device)
        
    def sigmoid(self,x):
        return torch.sigmoid(torch.clamp(x, min=-10, max=10))
    
    def linear_piecewise(self):
        K=torch.nn.functional.interpolate(self.points.reshape(1,1,-1),size=self.M//2,mode='linear',align_corners=True).reshape(-1,)
        K=torch.concat([K,K.flip(dims=[0])]).reshape(1,-1)
        self.K=self.sigmoid(K)
        
    def K_update(self,X):
        yn=X[:,0,:]
        z=X[:,1,:]
        Yn=torch.fft.fft(yn)
        Z=torch.fft.fft(z)
        self.linear_piecewise()
        Y=Yn*self.K+Z*(1-self.K)
        y=torch.fft.ifft(Y)
        
        return y.real    
    def EYY(self,yn,z):
        A=(self.K.reshape(-1,))/(1-self.K.reshape(-1,))
        Yn=torch.fft.fft(yn)
        Z=torch.fft.fft(z)
        YYn=abs(torch.mean(torch.multiply(Yn,Yn.conj()),axis=0))
        ZZ=abs(torch.mean(torch.multiply(Z,Z.conj()),axis=0))
        YZ=abs(torch.mean(torch.multiply(Yn,Z.conj()),axis=0)).real
        
        return (A*YYn-ZZ+2*YZ)/(A+1),YYn,ZZ
        
    def coherence(self,yn,z):
        Yf,YYn,ZZ=self.EYY(yn,z)
        
        co=(torch.divide(torch.square(abs(Yf)),torch.multiply(Yf,YYn)))

        return co.real.cpu().detach().numpy()
    
       
    def forward(self, y_n):
 
        L=y_n.shape[2]
        y_hat=self.K_update(y_n).reshape(-1,1,L) 
        
        
        z=self.batch1(y_hat)
   
        
        for i in range(self.num_hidden_layers):

            out=self.layers[i](z)

            z = self.activ(out)
            z=self.batch2(z)
            
          
       

        x_hat=self.layers[-1](z)
        
        x_hat=x_hat.reshape(-1,L)
        y_hat=y_hat.reshape(-1,L)
        
        return x_hat,y_hat
    



def predict(CNN,ds,device='cuda'):

    N=ds.N
    L=ds.L
    X=np.zeros((N,L))
    Y=np.zeros((N,L))

    k=0
    for i,batch in enumerate(ds):
        batchX,batchY=batch[0].to(device,dtype=torch.float).reshape(-1,2,L),batch[1].to(device,dtype=torch.float)
        batch_size_i=batchX.shape[0]
              
        Xi,Yi=CNN(batchX)
 
        X[k:k+batch_size_i,:]=Xi[:L].cpu().detach().numpy()
        Y[k:k+batch_size_i,:]=Yi[:L].cpu().detach().numpy()
        k+=batch_size_i

    return X,Y
        