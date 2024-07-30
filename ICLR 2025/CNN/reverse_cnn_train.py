# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:16:46 2022

@author: Joe
"""
import numpy as np
import torch
import time
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import torch.nn.init as init

from . import dataloader
from . import utilities as u


def train_cnn(CNN,dL_train,dL_test,N,L,rmsy,rmsx,num_epochs=1000,lr=0.1,BATCH_SIZE=1,device="cpu"):
 
    optimizer = optim.Adam(CNN.parameters(), lr=lr)
    iterator = (tqdm(range(num_epochs), desc="Epoch"))
    #####vary from 0.001 to 0.999
    lambda_p=0.001

    for e in iterator:
        batch_loss=torch.zeros((1,math.ceil(N/BATCH_SIZE)))
        for i,batch in enumerate(dL_train):
       
            batchY,batchX=batch[0].to(device, dtype=torch.float).reshape(-1,2,L),batch[1].to(device, dtype=torch.float).reshape(-1,L)            
            # zero the parameter gradients
            optimizer.zero_grad()

            Xi,Yi = CNN(batchY)
            
            Xi,targetsx=u.normalise(Xi,batchX,rmsx)
            Yi,targetsy=u.normalise(Yi,batchY[:,0,:],rmsx)
            
            lossx=u.MMSELoss(Xi,targetsx)
            lossyn=u.MMSELoss(Yi,targetsy)
                
     
      
            loss=lossx*(1-lambda_p)+lambda_p*lossyn
           
            loss.backward()
            
            optimizer.step()
       
            batch_loss[0,i]=loss.data
             
        loss_i=np.round(torch.mean(batch_loss).cpu().item(),4)

        CNN.losses.append(loss_i)  
        
        resultx,resulty = u.evaluate(CNN,dL_test,rmsy,rmsx,device)
  
        CNN.history.append([resultx,resulty]) 
        CNN.Ks.append(CNN.K.cpu().detach().numpy())
        iterator.set_postfix(loss=[loss_i,resultx,resulty])

    return CNN


            

    