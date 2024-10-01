# -*- coding: utf-8 -*-

import numpy as np

def mse(x,y):
    return np.mean(np.square(x-y))
def n_mse(x,y):
    return (np.mean(np.square(x-y))/np.mean(np.square(x)))**0.5