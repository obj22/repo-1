###############################################################################################################################
######################################Import Modules#############################################################################
################################################################################################################################
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import time
from data_analysis import coherence as coh
from data_analysis import psd
import data_analysis.filters as filters
import os
from data_analysis import evaluation as e
from tqdm import tqdm
from CNN import utilities as u
from torch.utils.data import Dataset, DataLoader
from CNN import dataloader,reverse_cnn,reverse_cnn_train
plt.rcParams["mathtext.default"]= "regular"
matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["font.family"] = "Times New Roman"
###set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
###############################################################################################################################
######################################Import Data#############################################################################
################################################################################################################################
#%%
########Important######
#increase \lambda until the averaged L_x increases by 0.001 above the minimum value. This is a minor error in the paper; it is 0.001 not 0.01.


#%%

# #######import data from lab
l=6000  #frame length
dts=1/500   ##sampling frequency
sigma=0.5##set the noise level 0.05,0.2,0.5 to replicate low, moderate and high noise levels in paper
X=torch.from_numpy(np.load('Y_n/X_lab.npy'))
Y=torch.from_numpy(np.load('Y_n/Y_lab.npy'))
Z=torch.Tensor(np.load('Y_z/'+"lab"+".npy")) 
f_low=0.1*int(1/dts)
f_high=0.13*int(1/dts)
L=X.shape[1]
##split into training,test and validation
N_train=10
N_val=10
N_test=160
X_train,Y_train,Z_train=X[:N_train,:],Y[:N_train,:],Z[:N_train,:]
X_val,Y_val,Z_val=X[N_train:N_train+N_val,:],Y[N_train:N_train+N_val,:],Z[N_train:N_train+N_val,:]
X_test,Y_test,Z_test=X[N_train+N_val:N_train+N_val+N_test,:],Y[N_train+N_val:N_train+N_val+N_test,:],Z[N_train+N_val:N_train+N_val+N_test,:]
#%%
####load friction
#####load data###########
dts=1/500   ##sampling frequency
sigma=1 ##set the noise level 1,5,10 to replicate low, moderate and high noise levels in paper 
X=torch.Tensor(np.load('Y_n/X_'+"friction"+".npy")) ##divide by 5000 so the input and output are both similar magnitudes around 1
Y=torch.Tensor(np.load('Y_n/Y_'+"friction"+".npy"))
Z=torch.Tensor(np.load('Y_z/'+"friction"+".npy"))
print(X.shape,Y.shape,Z.shape)
f_low=0.095*int(1/dts)
f_high=0.105*int(1/dts)
L=X.shape[1]
N_train=10
N_val=10
N_test=800
X_train,Y_train,Z_train=X[:N_train,:],Y[:N_train,:],Z[:N_train,:]
X_val,Y_val,Z_val=X[N_train:N_train+N_val,:],Y[N_train:N_train+N_val,:],Z[N_train:N_train+N_val,:]
X_test,Y_test,Z_test=X[N_train+N_val:N_train+N_val+N_test,:],Y[N_train+N_val:N_train+N_val+N_test,:],Z[N_train+N_val:N_train+N_val+N_test,:]

#%%
####load  saturated stiffness
#####load data###########
dts=1/200   ##sampling frequency
sigma=1 ##set the noise level 0.1,1,3 to replicate low, moderate and high noise levels in paper  
X=torch.Tensor(np.load('Y_n/X_'+"saturating_stiffness"+".npy"))/5000 ##divide by 5000 so the input and output are both similar magnitudes around 1
Y=torch.Tensor(np.load('Y_n/Y_'+"saturating_stiffness"+".npy"))
Z=torch.Tensor(np.load('Y_z/'+"saturating_stiffness"+".npy"))
f_low=0.066*int(1/dts)
f_high=0.12*int(1/dts)
L=X.shape[1]
N_train=10
N_val=10
N_test=800
X_train,Y_train,Z_train=X[:N_train,:],Y[:N_train,:],Z[:N_train,:]
X_val,Y_val,Z_val=X[N_train:N_train+N_val,:],Y[N_train:N_train+N_val,:],Z[N_train:N_train+N_val,:]
X_test,Y_test,Z_test=X[N_train+N_val:N_train+N_val+N_test,:],Y[N_train+N_val:N_train+N_val+N_test,:],Z[N_train+N_val:N_train+N_val+N_test,:]
#%%
####load nonlinear_stiffness data
#####load data###########
dts=1/20000   ##sampling frequency
sigma=1 ##set the noise level 0.5,1,3 to replicate low, moderate and high noise levels in paper
X=torch.Tensor(np.load('Y_n/X_'+"nonlinear_stiffness"+".npy")) ##divide by 5000 so the input and output are both similar magnitudes around 1
Y=torch.Tensor(np.load('Y_n/Y_'+"nonlinear_stiffness"+".npy")) 
Z=torch.Tensor(np.load('Y_z/'+"nonlinear_stiffness"+".npy"))
L=X.shape[1]
f_low=0.025*int(1/dts)
f_high=0.05*int(1/dts)
N_train=10
N_val=10
N_test=700
X_train,Y_train,Z_train=X[:N_train,:],Y[:N_train,:],Z[:N_train,:]
X_val,Y_val,Z_val=X[N_train:N_train+N_val,:],Y[N_train:N_train+N_val,:],Z[N_train:N_train+N_val,:]
X_test,Y_test,Z_test=X[N_train+N_val:N_train+N_val+N_test,:],Y[N_train+N_val:N_train+N_val+N_test,:],Z[N_train+N_val:N_train+N_val+N_test,:]
###############################################################################################################################
######################################Define Model#############################################################################
################################################################################################################################
#%%
##specify CNN parameters

kernel_width=7 
num_hidden_layers=5
num_channels=5
dilation=1

###create CNN~
CNN=reverse_cnn.reverse_CNN(kernel_width, num_hidden_layers, num_channels,L,dilation,device)
CNN=CNN.to(device)
#%%
###############################################################################################################################
######################################Add bandlimited noise to data#############################################################################
################################################################################################################################

rmsx=torch.mean(torch.square(X)).item()**0.5
rmsy=torch.mean(torch.square(Y)).item()**0.5



noise_train=np.random.normal(0,1,(Y_train.shape))
noise_train=filters.butter_bandpass_filter(noise_train,f_low,f_high,int(1/dts))
noise_train=torch.tensor(noise_train)*sigma


noise_val=np.random.normal(0,1,(Y_val.shape))
noise_val=filters.butter_bandpass_filter(noise_val,f_low,f_high,int(1/dts))
noise_val=torch.tensor(noise_val)*sigma

noise_test=np.random.normal(0,1,(Y_test.shape))
noise_test=filters.butter_bandpass_filter(noise_test,f_low,f_high,int(1/dts))
noise_test=torch.tensor(noise_test)*sigma


Y_train=Y_train+noise_train
Y_val=Y_val+noise_val
Y_test_clean=Y_test.clone()
Y_test=Y_test+noise_test

###############################################################################################################################
######################################Create Dataloader Objects#############################################################################
################################################################################################################################

#%%
BATCH_SIZE=1

ds_train=dataloader.data_set(Y_train,Z_train,X_train,N_train,L)
ds_test=dataloader.data_set(Y_test,Z_test,X_test,N_test,L)
ds_val=dataloader.data_set(Y_val,Z_val,X_val,N_val,L)

dL_train=DataLoader(ds_train,shuffle=True,batch_size=BATCH_SIZE,pin_memory=True)
dL_test=DataLoader(ds_test,shuffle=False,batch_size=N_test,pin_memory=True)
dL_val=DataLoader(ds_val,shuffle=True,batch_size=N_val,pin_memory=True)

#%%
########################################################################################
##################################train CNN#############################################
#######################################################################################
num_epochs=100

CNN=reverse_cnn_train.train_cnn(CNN,dL_train,dL_val,N_train,L,rmsy,rmsx,num_epochs,0.01,BATCH_SIZE,device)

#%%
####evaluate L_x and L_y on the validation dataset
 
mse=u.evaluate(CNN,dL_val,rmsy,rmsx,device)


print(np.mean(np.array(CNN.history)[-100:,:],axis=0))
# plt.plot(CNN.history)
#%%
########################################################################################
##################################plot time domain results##################################
#######################################################################################

##define time vector
X_test_pred,Y_test_pred=reverse_cnn.predict(CNN,ds_test,device) 
t=np.linspace(0,dts*L,L)
k=np.random.randint(0,N_test)   #choose a random test sample

#extract the chosen 1 dimensional vectors to plot
x=X_test[k,:]
y=Y_test[k,:]

y_pred=Y_test_pred[k,:]
x_pred=X_test_pred[k,:]


plt.plot(t,x,color='black',label='True X')  #plot the clean X


plt.plot(t,x_pred,color='blue', linestyle='dashdot', label='CNN X Prediction',linewidth=3)

plt.legend()
plt.xlabel('$Time (secs)$')
plt.ylabel('$Acceleration \ (ms^{-2})$')
plt.tight_layout()
#%%
########################################################################################
##################################plot frequency domain results##################################
#######################################################################################
X_test_pred,Y_test_pred=reverse_cnn.predict(CNN,ds_test,device) 
k=np.random.randint(0,N_test)   #choose which test sample
window=L//1


f,Xf=psd.cross_calc_batch(X.numpy(),X.numpy(),dts,window)
f,Yf=psd.cross_calc_batch(Y.numpy(),Y.numpy(),dts,window)
f,Y_CNNf=psd.cross_calc_batch(Y_test_pred,Y_test_pred,dts,window)
f,Zf=psd.cross_calc_batch(Z.numpy(),Z.numpy(),dts,window)



plt.figure(1)
plt.plot(np.log10(Xf[:L//2]),'green',linestyle=':',label='X PSD')
plt.xlabel('$\omega$')
plt.ylabel('Input PSD') #(dB/Hz)
plt.legend(loc='upper right',fontsize='15')
plt.tight_layout()

plt.figure(2)
plt.plot(Yf[:L//2],'Black',label='$Y_{n}$ PSD')
plt.plot(Zf[:L//2],'g:',label='$Y_{linear}$ PSD')

plt.xlim([500,700])
# plt.ylim([0,140000])

plt.xlabel('$\omega$')
plt.ylabel('Output PSD') #(dB/Hz)
plt.legend(loc='upper right',fontsize='15')

plt.tight_layout()

#%%##############################################################################################
#########################################Plot power spectrum densities##############################
###################################################################################################
Yff=torch.fft.fft(Y_test_clean)
eee=torch.fft.fft(noise_test)
YY=abs(torch.mean(torch.multiply(Yff,Yff.conj()),axis=0))
ee=abs(torch.mean(torch.multiply(eee,eee.conj()),axis=0))
Y_pred,Ynn,Yzz=CNN.EYY(Y_test.to(device),Z_test.to(device))
plt.plot(Y_pred[:L//2].detach().cpu(),'r--',label='$Y_{pred}$ PSD')
plt.plot(Ynn[:L//2].detach().cpu(),'blue',label='$Y_{n}$ PSD')
plt.plot(Yzz[:L//2].detach().cpu(),'g:',label='$Y_{z}$ PSD')
plt.plot(YY[:L//2].detach().cpu(),'black',label='$Y_{}$ PSD')
plt.plot(ee[:L//2].detach().cpu(),'black',label='$e_{n}$ PSD')
plt.yscale('log')
#%%
########################################################################################
##################################Calculate coherence##################################
#######################################################################################
X_test_pred,Y_test_pred=reverse_cnn.predict(CNN,ds_test,device) 
window=L//1
Y_test_clean=Y[N_train+N_val:N_train+N_val+N_test,:]
####coherence between the true input and predicted input
f,Xf_1=psd.cross_calc_batch(X_test.numpy(),X_test.numpy(),dts,window_length=window)
f,Yf_1=psd.cross_calc_batch(X_test_pred,X_test_pred,dts,window_length=window)
f,XYf_1=psd.cross_calc_batch(X_test_pred,X_test.numpy(),dts,window_length=window)

co_1=np.divide(np.square(abs(XYf_1)),np.multiply(Xf_1,Yf_1))

###True nonlinear coherence
f,Xf_2=psd.cross_calc_batch(Y_test.numpy(),Y_test.numpy(),dts,window_length=window)
f,Yf_2=psd.cross_calc_batch(Y_test_clean.numpy(),Y_test_clean.numpy(),dts,window_length=window)
f,XYf_2=psd.cross_calc_batch(Y_test.numpy(),Y_test_clean.numpy(),dts,window_length=window)

co_2=np.divide(np.square(abs(XYf_2)),np.multiply(Xf_2,Yf_2))


###linear coherence
f,Xf_3=psd.cross_calc_batch(Y_test.numpy(),Y_test.numpy(),dts,window_length=window)
f,Yf_3=psd.cross_calc_batch(X_test.numpy(),X_test.numpy(),dts,window_length=window)
f,XYf_3=psd.cross_calc_batch(Y_test.numpy(),X_test.numpy(),dts,window_length=window)

co_3=np.divide(np.square(abs(XYf_3)),np.multiply(Xf_3,Yf_3))

##Y_z with Y_n coherence
f,Xf_4=psd.cross_calc_batch(Y_test.numpy(),Y_test.numpy(),dts,window_length=window)
f,Yf_4=psd.cross_calc_batch(Z_test.numpy(),Z_test.numpy(),dts,window_length=window)
f,XYf_4=psd.cross_calc_batch(Y_test.numpy(),Z_test.numpy(),dts,window_length=window)

co_4=np.divide(np.square(abs(XYf_4)),np.multiply(Xf_4,Yf_4))
###predicted nonlinear coherence
co_5=CNN.coherence(Y_test.to(device),Z_test.to(device))




########################################################################################
##################################plot coherence##################################
#######################################################################################


plt.figure()
plt.plot(co_2[:L//2],'black',label='$Co(Y,Y_{n})$')
plt.plot((co_4)[:L//2],'b:',label='$Co(Y_{CNN},Y_{n})$',alpha=1)
plt.plot((co_3)[:L//2],'g',label='$Co(X,Y_{n})$',alpha=1)
plt.plot((co_5)[:L//2],'r--',label='Prediction',alpha=1)
plt.xlabel('$\omega$')
# plt.xlabel('$f \ (Hz)$')
plt.ylabel('$\gamma^{2}(\omega)$')
plt.xlim([0,600])
plt.legend(fontsize='15')
plt.tight_layout()
plt.ylim([-0.05,1.05])

