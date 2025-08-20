import torch
from torch import nn, optim 
#from model import *
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
from network import IR_RGB_Autoencoder, IR_Encoder, IR_Decoder, RGB_Encoder, RGB_Decoder
import random
from PIL import Image
from torchvision import transforms
from FusionImageEvalution.Metric_Python.eval_one_image import evaluation_one



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2) 

# 图像路径
image_path_ir = r'.\data\ir2.png'
image_path_rgb = r'.\data\rgb2.png'

# 加载图像
ir  = Image.open(image_path_ir)

rgb = Image.open(image_path_rgb)

# 定义一个转换，将PIL图像转换为tensor
convert_tensor = transforms.ToTensor()

# 应用转换并加载图像作为tensor
ir  = convert_tensor(ir).to('cuda:0')
rgb = convert_tensor(rgb).to('cuda:0')



ir3 = torch.rand_like(rgb)
for iter in range(3):
    ir3[iter,:,:] = 1*ir+0*rgb[iter,:,:]

# plt.imshow(ir3.permute(1,2,0).cpu().detach().numpy())
# plt.title('fusion_rgb')
# plt.show()
        
################### 
# Here are the hyperparameters. 
lr_real = 0.00025
max_iter =  5001

[L, M, N]    = rgb.shape


encoder_spa  = IR_Encoder()
decoder_spa  = IR_Decoder()
encoder_spec = RGB_Encoder()
decoder_spec = RGB_Decoder()
model  = IR_RGB_Autoencoder(encoder_spa, decoder_spa, encoder_spec, decoder_spec).cuda() 
params = []
params += [x for x in model.parameters()]
optimizer    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 

  
  
'''------------------------------------main -----------------------------------------'''
for iter in range(max_iter):
    
    ir_en, ir_de, rgb_en, rgb_de, fusion_ir, fusion_rgb = model(ir, rgb)
    # loss1 = torch.norm(ir_de-ir,2)+torch.norm(rgb_de-rgb,2)
    # loss2 = torch.norm(ir_en-ir3,2)
    # loss3 = torch.norm(rgb_en-ir3,2)
    
    loss1 = 1*torch.norm(rgb_de-rgb,1) + torch.norm(rgb_de-ir3,1)
    loss2 = torch.norm(ir_en-rgb_en,1)
    #loss3 = torch.norm(fusion_rgb-rgb,2)
    loss = 1*loss1 + 1*loss2 

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if iter % 2000 == 0:
        HRHSI   = fusion_ir.permute(1,2,0)
        HR_HSIt = rgb_de.permute(1,2,0)

        plt.figure(figsize=(15,45))
        plt.subplot(131)
        plt.imshow(ir.cpu().detach().numpy().squeeze())
        plt.title('IR')
           
        plt.subplot(132)
        plt.imshow(rgb.permute(1,2,0).cpu().detach().numpy())
        plt.title('RGB')
        
        plt.subplot(133)
        plt.imshow(HR_HSIt.cpu().detach().numpy())
        plt.title('fusion')
        plt.show()
        
        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir, rgb, fusion_rgb)
        print('EN:', round(EN, 4))
        print('MI:', round(MI, 4))
        print('SF:', round(SF, 4))
        print('AG:', round(AG, 4))
        print('SD:', round(SD, 4))
        print('CC:', round(CC, 4))
        print('SCD:', round(SCD, 4))
        print('VIF:', round(VIF, 4))
        print('MSE:', round(MSE, 4))
        print('PSNR:', round(PSNR, 4))
        print('Qabf:', round(Qabf, 4))
        print('Nabf:', round(Nabf, 4))
        print('SSIM:', round(SSIM, 4))
        print('MS_SSIM:', round(MS_SSIM, 4))
        print('============================')