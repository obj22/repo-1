# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

###function to evaluate normalised mean squared error performance of trained model
def evaluate(model,val_loader,rmsy,rmsx,device):

  batch_lossx=[]
  batch_lossy=[]
  k=0
  for i,batch in enumerate(val_loader):
    batchY,batchX=batch[0].to(device, dtype=torch.float),batch[1].to(device, dtype=torch.float)
    batch_size_i=batchY.shape[0]

    Xi,Yi = model(batchY)

    Xi,targetsx=normalise(Xi,batchX,rmsx)
    
    lossx=MMSELoss(Xi,targetsx)
    
    Yi,targetsy=normalise(Yi,batchY[:,0,:],rmsy)
    lossy=MMSELoss(Yi,targetsy)
    batch_lossx.append(lossx.data.cpu().numpy())
    batch_lossy.append(lossy.data.cpu().numpy())
   

  return np.round(np.mean(batch_lossx),6),np.round(np.mean(batch_lossy),6)


        
##mean squared error loss
def MMSELoss(outputs, targets):
  N=outputs.shape[0]
  MSE=torch.mean(torch.square(outputs-targets))
  return MSE

#normalise outputs and targets
def normalise(outputs, targets,rms):
  return torch.div(outputs,rms),torch.div(targets,rms)
