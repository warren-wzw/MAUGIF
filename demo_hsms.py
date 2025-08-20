import torch
from torch import nn, optim 
#from model import *
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
from network import Spatial_Spectral_Autoencoder, Spatial_Encoder, Spatial_Decoder, Spectral_Encoder, Spectral_Decoder
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2) 

Pre = scipy.io.loadmat('.\data\Pre.mat')
HSI = Pre['HSI'].transpose(2,0,1)
MSI = Pre['MSI'].transpose(2,0,1)
HRHSI = Pre['true_image']

subspace = 15
HSI3 = HSI.reshape(HSI.shape[0],-1)
U,S,V= np.linalg.svd(np.dot(HSI3,HSI3.T))
Dic  = U[:,0:subspace]
Coe =  np.tensordot(Dic.T,  HSI, axes=([1], [0]))

HSI =  torch.from_numpy(HSI).to('cuda:0').to(torch.float32)
MSI =  torch.from_numpy(MSI).to('cuda:0').to(torch.float32)
Dic = torch.from_numpy(Dic).to('cuda:0').to(torch.float32)
Coe = torch.from_numpy(Coe).to('cuda:0').to(torch.float32)


################### 
# Here are the hyperparameters. 
lr_real = 0.0003
max_iter =  8001

[L, m, n]    = HSI.shape
[l, M, N]    = MSI.shape
sf = 32

LR_MSI = MSI[:,0:M:sf,0:N:sf]

encoder_spa  = Spatial_Encoder(1)
decoder_spa  = Spatial_Decoder(1)
encoder_spec = Spectral_Encoder(l,subspace)
decoder_spec = Spectral_Decoder(l,subspace)

centre = torch.Tensor(L,M,N).cuda()
stdv = 1 / math.sqrt(centre.size(2))
centre.data.uniform_(-stdv, stdv)
model  = Spatial_Spectral_Autoencoder(encoder_spa, decoder_spa, encoder_spec, decoder_spec).cuda() 
params = []
params += [x for x in model.parameters()]
# centre.requires_grad=True
# params += [centre]
optimizer    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 

  
'''------------------------------------main -----------------------------------------'''
for iter in range(max_iter):
    
    # LR_MSI1, LR_MSI2, HR_HSI1, HR_HSI2, MSI1, HSI1,  X_MSI, X = model(HSI, MSI, centre)
    
    # loss1 = 1*torch.norm(HSI1-HSI,2)+ 0*torch.norm(MSI1-MSI,2)
    # loss2 = torch.norm(LR_MSI1-LR_MSI,2)
    # loss3 = torch.norm(LR_MSI2-LR_MSI,2)
    # loss4 = torch.norm(X_MSI-MSI,2)

    # loss = 0.1*loss1 + 0*loss2 + 10*loss3
    
    LR_MSI2, HR_Coe1,  Coe1 = model(Coe, MSI)
    loss1 = torch.norm(Coe1-Coe,2)
    loss2 = torch.norm(LR_MSI2-LR_MSI,2)
    
    loss = 0.1*loss1 + 1*loss2

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if iter % 500 == 0:
        HR_HSI1 = np.tensordot(Dic.cpu().detach().numpy(), HR_Coe1.cpu().detach().numpy(), axes=([1], [0]))
        HR_HSI1 = torch.from_numpy(HR_HSI1).to('cuda:0').to(torch.float32)
        HR_HSIt = HR_HSI1.permute(1,2,0)
        ps = peak_signal_noise_ratio(np.clip(HRHSI,0,1),HR_HSIt.cpu().detach().numpy())
        print('iteration:',iter,'PSNR',ps)
                
        # plt.figure(figsize=(15,45))
        # show = [10,20,30] 
        # plt.subplot(121)
        # plt.imshow(np.clip(np.stack((HRHSI[:,:,show[0]],
        #                      HRHSI[:,:,show[1]],
        #                      HRHSI[:,:,show[2]]),2),0,1))
        # plt.title('GT')
        
        # plt.subplot(122)
        # plt.imshow(np.clip(np.stack((HR_HSIt[:,:,show[0]].cpu().detach().numpy(),
        #                      HR_HSIt[:,:,show[1]].cpu().detach().numpy(),
        #                      HR_HSIt[:,:,show[2]].cpu().detach().numpy()),2),0,1))
        # plt.title('out')
        # plt.show()