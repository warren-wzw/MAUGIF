import torch
from torch import nn, optim 
#from model import *
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio
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
image_path_1 = '.\data\mf1.png'
image_path_2 = '.\data\mf2.png'
image_path_gt = '.\data\mf_gt.jpg'

# 加载图像
image1 = Image.open(image_path_1)
image2 = Image.open(image_path_2)
gt     = Image.open(image_path_gt)

# 定义一个转换，将PIL图像转换为tensor
convert_tensor = transforms.ToTensor()

# 应用转换并加载图像作为tensor
ir  = convert_tensor(image1).to('cuda:0')
rgb = convert_tensor(image2).to('cuda:0')
gt  = convert_tensor(gt).to('cuda:0')

lr_real = 0.00025
max_iter =  3001

[L, M, N]    = rgb.shape


'''-----------------------------------------------------------------------'''
# IR编码器
class IR_Encoder(nn.Module):
    def __init__(self):
        super(IR_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(6,   3, kernel_size=9, padding=4)  

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        return x3

# IR解码器
class IR_Decoder(nn.Module):
    def __init__(self):
        super(IR_Decoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(6,   3, kernel_size=9, padding=4)  

    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        return x3


# RGB编码器
class RGB_Encoder(nn.Module): 
    def __init__(self):
        super(RGB_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(6,   3, kernel_size=9, padding=4)  
        
    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        return x3
    
# RGB解码器
class RGB_Decoder(nn.Module): 
    def __init__(self):
        super(RGB_Decoder, self).__init__()
        self.conv3 = torch.nn.Conv2d(3,   3, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(6,   3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = torch.sin(self.conv3(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv4(x2)
        return x3
    
class IR_RGB_Autoencoder(nn.Module):
    def __init__(self, ir_encoder, ir_decoder, rgb_encoder, rgb_decoder):
        super(IR_RGB_Autoencoder, self).__init__()
        self.encoder_ir  = ir_encoder
        self.decoder_ir  = ir_decoder
        self.encoder_rgb = rgb_encoder
        self.decoder_rgb = rgb_decoder

    def forward(self, ir, rgb):
        ir_en = self.encoder_ir(ir)
        rgb_en = self.encoder_rgb(rgb)
        
        
        ir_de  = self.decoder_ir(ir_en)
        rgb_de = self.decoder_rgb(rgb_en)
    
        fusion_ir  = self.decoder_ir(rgb)
        fusion_rgb = self.decoder_rgb(ir)
        return ir_en, ir_de, rgb_en, rgb_de, fusion_ir, fusion_rgb
        # gg   = self.encoder_ir(ir-rgb)
        # gg1  = self.encoder_ir(rgb)
        # rgb1 = self.decoder_rgb(gg1)
        # ir1  = self.decoder_ir(gg1)
        
        # fusion1 = self.decoder_rgb(ir)
        # fusion2 = self.decoder_ir(rgb)
        
        # return gg, gg1, rgb1, ir1, fusion1, fusion2
    
class IR_RGB_Autoencoder1(nn.Module):
    def __init__(self, ir_encoder, ir_decoder, rgb_encoder, rgb_decoder):
        super(IR_RGB_Autoencoder1, self).__init__()
        self.encoder_ir  = ir_encoder
        self.decoder_ir  = ir_decoder
        self.encoder_rgb = rgb_encoder
        self.decoder_rgb = rgb_decoder

    def forward(self, gg, rgb, ir):

        rgb1 = self.decoder_rgb(gg)
        ir1  = self.decoder_ir(gg)
        
        fusion1 = self.decoder_rgb(ir)
        fusion2 = self.decoder_ir(rgb)
        
        return  rgb1, ir1, fusion1, fusion2

encoder_spa  = IR_Encoder()
decoder_spa  = IR_Decoder()
encoder_spec = RGB_Encoder()
decoder_spec = RGB_Decoder()
model  = IR_RGB_Autoencoder(encoder_spa, decoder_spa, encoder_spec, decoder_spec).cuda() 
params = []
params += [x for x in model.parameters()]
optimizer    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 

# a = 0.5
# median_rgb = a*ir+(1-a)*rgb
# median_ir = (1-a)*ir+a*rgb
  
'''------------------------------------main -----------------------------------------'''
for iter in range(max_iter):
    
    ir_en, ir_de, rgb_en, rgb_de, fusion_ir, fusion_rgb = model(ir, rgb)
    fusion = fusion_ir
    loss1 = 1*torch.norm(rgb_de-rgb,2) +  torch.norm(ir_de-ir,2)
    loss2 = 1*torch.norm(ir_en-rgb_en,2) + 1*torch.norm(ir_en-ir,1)+ 1*torch.norm(rgb_en-rgb,1)
    #loss3 = 1*torch.norm(fusion-rgb,2) +  torch.norm(fusion-ir,2)
    #loss3 = torch.norm(fusion_rgb-rgb,2)
    loss = 1*loss1 + 1*loss2 
    
    # gg, gg1, rgb1, ir1, fusion1, fusion2 = model(ir, rgb)
    # fusion = fusion1
    # loss1 = 1*torch.norm(rgb1-rgb,2) +  torch.norm(ir1-ir,2)
    # loss2 = torch.norm(gg,2)
    # loss3 = 1*torch.norm(gg1-ir,2) + 1*torch.norm(gg1-rgb,2)
    # loss  = loss1+ 0.1*loss2 + loss3

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if iter % 1000 == 0:
        HR_HSI = fusion
        HR_HSIt = HR_HSI.permute(1,2,0)
        ps = peak_signal_noise_ratio(np.clip(HR_HSIt.cpu().detach().numpy(),0,1),gt.permute(1,2,0).cpu().detach().numpy())
        print('iteration:',iter,'PSNR',ps)
        
    if iter % 1000 == 0:
        HR_HSI = fusion
        HR_HSIt = HR_HSI.permute(1,2,0)
        plt.figure(figsize=(15,45))
        plt.subplot(131)
        plt.imshow(ir.permute(1,2,0).cpu().detach().numpy().squeeze())
        plt.title('IR')
           
        plt.subplot(132)
        plt.imshow(rgb.permute(1,2,0).cpu().detach().numpy())
        plt.title('RGB')
        
        plt.subplot(133)
        plt.imshow(HR_HSIt.cpu().detach().numpy())
        plt.title('fusion')
        plt.show()
        
        
'''--------------------------------------------------------------------------------'''       
# model1  = IR_RGB_Autoencoder1(encoder_spa, decoder_spa, encoder_spec, decoder_spec).cuda() 
# params1 = []
# params1 += [x for x in model1.parameters()]
# optimizer1    = optim.Adam(params1, lr=lr_real, weight_decay=1e-8) 
    
# for iter in range(max_iter):
    
#     rgb1, ir1, fusion1, fusion2 = model1( fusion, ir, rgb)
#     loss1 = 1*torch.norm(rgb1-rgb,2) 

#     optimizer1.zero_grad()
#     loss1.backward(retain_graph=True)
#     optimizer1.step()
#     if iter % 500 == 0:
#         HR_HSI = fusion1
#         HR_HSIt = HR_HSI.permute(1,2,0)
#         ps = peak_signal_noise_ratio(np.clip(HR_HSIt.cpu().detach().numpy(),0,1),gt.permute(1,2,0).cpu().detach().numpy())
#         print('iteration:',iter,'PSNR',ps)
        
#     if iter % 500 == 0:
#         HR_HSI = fusion1
#         HR_HSIt = HR_HSI.permute(1,2,0)
#         plt.figure(figsize=(15,45))
#         plt.subplot(131)
#         plt.imshow(ir.permute(1,2,0).cpu().detach().numpy().squeeze())
#         plt.title('IR')
           
#         plt.subplot(132)
#         plt.imshow(rgb.permute(1,2,0).cpu().detach().numpy())
#         plt.title('RGB')
        
#         plt.subplot(133)
#         plt.imshow(HR_HSIt.cpu().detach().numpy())
#         plt.title('fusion')
#         plt.show()
