# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:10:59 2024

@author: jtm44
"""

import numpy as np
import torch 

def batch(X,l):
    #total length
    L=X.shape[1]
    N=X.shape[0]
    new_X=torch.zeros((L//l)*N,l)
    i=0
  
    for n in range(N):
        k=0
        while True:
            new_X[i,:]=X[n,k*l:(k+1)*l]
            k+=1
            i+=1
            if (k+1)*l>L:
                break
    return new_X
            
            