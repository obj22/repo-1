# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:26:41 2021

@author: Joe
"""
import numpy as np
import matplotlib.pyplot as plt
from . import psd

        
def coherence(x,y,dt,window_length):
              
        f,S_xy=psd.psd_calc_batch(x,y,dt,window_length)
        f,S_xx=psd.psd_calc_batch(x,x,dt,window_length)
        f,S_yy=psd.psd_calc_batch(y,y,dt,window_length)
        g=np.square(abs(np.divide(np.square(abs(S_xy)),np.multiply(S_yy,S_xx))))
        return f,g
    
        
        