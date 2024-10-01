# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz,welch


def psd_calc(x,y,dt,window_length=None  ,window_overlap=0.5):
    
    x=np.reshape(x,(1,-1))
    y=np.reshape(y,(1,-1))
    N=x.shape[1]
    if window_length==None:
        window_length=N
    
        
    XY_star_T=np.zeros((1,window_length))*1j
    start=0
    i=0
    
    while window_length+start<=N:
        x_w=x[0,start:start+window_length]
        y_w=y[0,start:start+window_length]
        window=np.hanning(len(x_w))
        # window=np.ones(len(x_w))
        X=(2*dt/np.sum(window))*fft(window*x_w)
        Y=(2*dt/np.sum(window))*fft(window*y_w)
        XY_star=np.multiply(X,Y.conj())
        XY_star_T+=XY_star
        i+=1
        start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)

    XY_star_T=XY_star_T[0,0:window_length//2]/i
    return xf,abs(XY_star_T)

def psd_calc_batch(X,Y,dt,window_length=None  ,window_overlap=0.5):
    if len(X.shape)==1:
        X=X.reshape(1,-1)
        Y=Y.reshape(1,-1)
    L=X.shape[1]
    N=X.shape[0]
    
    if window_length==None:
        window_length=L
    
        
    XY_star_T=np.zeros((1,window_length))*1j

    for j in range(N):
        start=0
        i=0
        while window_length+start<=L:
            x_w=X[j,start:start+window_length]
            y_w=Y[j,start:start+window_length]
            # window=np.hanning(len(x_w))
            window=np.ones(len(x_w))
            Xf=(2*dt/np.sum(window))*fft(window*x_w)
            Yf=(2*dt/np.sum(window))*fft(window*y_w)
            XY_star=np.multiply(Xf,Yf.conj())
            XY_star_T+=XY_star
            i+=1
            start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)

    XY_star_T=XY_star_T[0,0:window_length//2]/(i*N)
    return xf,abs(XY_star_T)

def cross_calc_batch(X,Y,dt,window_length=None  ,window_overlap=0.5):
    if len(X.shape)==1:
        X=X.reshape(1,-1)
        Y=Y.reshape(1,-1)
    L=X.shape[1]
    N=X.shape[0]
    
    if window_length==None:
        window_length=L
    
        
    XY_star_T=np.zeros((1,window_length))*1j

    for j in range(N):
        start=0
        i=0
        while window_length+start<=L:
            x_w=X[j,start:start+window_length]
            y_w=Y[j,start:start+window_length]
            # window=np.hanning(len(x_w))
            window=np.ones(len(x_w))
            Xf=fft(window*x_w)
            Yf=fft(window*y_w)
            XY_star=np.multiply(Xf,Yf.conj())
            XY_star_T+=XY_star
            i+=1
            start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)

    XY_star_T=XY_star_T[0,0:window_length]#/(i*N)
    return xf,XY_star_T

def expect_calc_batch(X,dt,window_length=None  ,window_overlap=0.5):
    if len(X.shape)==1:
        X=X.reshape(1,-1)
    L=X.shape[1]
    N=X.shape[0]
    
    if window_length==None:
        window_length=L
    
        
    X_star_T=np.zeros((1,window_length))*1j

    for j in range(N):
        start=0
        i=0
        while window_length+start<=L:
            x_w=X[j,start:start+window_length]

            # window=np.hanning(len(x_w))
            window=np.ones(len(x_w))
            Xf=(2*dt/np.sum(window))*fft(window*x_w)
            X_star=Xf
            X_star_T+=X_star
            i+=1
            start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)

    X_star_T=X_star_T[0,0:window_length]/(i*N)
    return xf,X_star_T




def welch_sxy_batch(X,Y,dt,window_length=None  ,window_overlap=0.5):
    if len(X.shape)==1:
        X=X.reshape(1,-1)
        Y=Y.reshape(1,-1)
    L=X.shape[1]
    N=X.shape[0]
    
    if window_length==None:
        window_length=L
    
        
    XY_star_T=np.zeros((1,window_length))*1j

    for j in range(N):
        start=0
        i=0
        while window_length+start<=L:
            x_w=X[j,start:start+window_length]
            y_w=Y[j,start:start+window_length]
            window=np.hanning(len(x_w))
            Xf=(2*dt/np.sum(window))*fft(window*x_w)
            Yf=(2*dt/np.sum(window))*fft(window*y_w)
            XY_star=np.multiply(Xf,Yf.conj())
            XY_star_T+=XY_star
            i+=1
            start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(-1.0/(2.0*dt), 1.0/(2.0*dt), window_length)
    
    XY_star_T=XY_star_T/(i*N)  #check i here
    return xf,XY_star_T



def welch_sxy(x,y,dt,window_length=None  ,window_overlap=0.5):
    x=np.reshape(x,(1,-1))
    y=np.reshape(y,(1,-1))
    N=x.shape[1]
    if window_length==None:
        window_length=N
    
        
    XY_star_T=np.zeros((1,window_length))*1j
    start=0
    i=0
    
    while window_length+start<=N:
        x_w=x[0,start:start+window_length]
        y_w=y[0,start:start+window_length]
        window=np.hanning(len(x_w))
        X=(2*dt/np.sum(window))*fft(window*x_w)
        Y=(2*dt/np.sum(window))*fft(window*y_w)
        XY_star=np.multiply(X,Y.conj())
        XY_star_T+=XY_star
        i+=1
        start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)

    XY_star_T=XY_star_T/i
    return xf,XY_star_T


    
def kernel_calc(x,y,dt,window_length=None  ,window_overlap=0.5):
    f,s_xx=welch_sxy(x,x,dt,window_length=window_length  ,window_overlap=window_overlap)
    f,s_xy=welch_sxy(y,x,dt,window_length=window_length  ,window_overlap=window_overlap)
    G=s_xy/s_xx
    g=ifft(G)
    return np.real(g)


def wiener_Y(x,y,dt,window_length=None,window_overlap=0.5):
    x=np.reshape(x,(1,-1))
    y=np.reshape(y,(1,-1))
    N=x.shape[1]
    if window_length==None:
        window_length=N
    
    Y1_Total=np.zeros((window_length,1))*1j
    start=0
    i=0
    
    while window_length+start<=N:
        x_w=x[0,start:start+window_length]
        y_w=y[0,start:start+window_length]
        window=np.hanning(len(x_w))
        X=(2*dt/np.sum(window))*fft(window*x_w)
        Y=(2*dt/np.sum(window))*fft(window*y_w)
        X=X.reshape(-1,1)
        Y=Y.reshape(-1,1)
        XX_plus=1/np.dot(X.T,X)
        Y1=np.matmul(X,np.matmul(XX_plus,np.matmul(X.T,Y)))
        Y1_Total+=Y1
        i+=1
        start+=int((1-window_overlap)*window_length)
        
    xf = np.linspace(0.0, 1.0/(2.0*dt), window_length//2)
    

    

    return xf,abs(Y1_Total[0:window_length//2]/i)

    



