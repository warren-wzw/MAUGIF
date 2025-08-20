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
image_path_1 = r'.\data\ir1.png'
image_path_2 = r'.\data\rgb1.png'
# 加载图像
image1 = Image.open(image_path_1)
image2 = Image.open(image_path_2)

# 定义一个转换，将PIL图像转换为tensor
convert_tensor = transforms.ToTensor()

# 应用转换并加载图像作为tensor
ir2  = convert_tensor(image1).to('cuda:0')
rgb = convert_tensor(image2).to('cuda:0')
ir = torch.rand_like(rgb)
ir[0,:,:] = ir2
ir[1,:,:] = ir2
ir[2,:,:] = ir2

# ir  = convert_tensor(image1).to('cuda:0')
# rgb2 = convert_tensor(image2).to('cuda:0')
# rgb = torch.rand_like(ir)
# rgb[0,:,:] = rgb2
# rgb[1,:,:] = rgb2
# rgb[2,:,:] = rgb2


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
        self.conv3 = torch.nn.Conv2d(6,   3, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(9,   3, kernel_size=3, padding=1)
        
    def forward(self, x, y):
        x = torch.cat([x,y],0)
        x1 = torch.sin(self.conv3(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv4(x2)
        return x3
    
class IR_RGB_encoder(nn.Module):
    def __init__(self, ir_encoder,  rgb_encoder):
        super(IR_RGB_encoder, self).__init__()
        self.encoder_ir  = ir_encoder
        self.encoder_rgb = rgb_encoder

    def forward(self, ir, rgb):

        gg   = self.encoder_ir(ir-rgb)
        gg1  = self.encoder_ir(rgb)

        return gg, gg1
    
class IR_RGB_decoder(nn.Module):
    def __init__(self, ir_decoder, rgb_decoder):
        super(IR_RGB_decoder, self).__init__()
        self.decoder_ir  = ir_decoder
        self.decoder_rgb = rgb_decoder

    def forward(self, gg1, ir, rgb):

        rgb_detail = self.decoder_rgb(ir, rgb)
        rgb1 = gg1+rgb_detail
        ir1  = self.decoder_ir(gg1)
        
        fusion1 = ir+rgb_detail
        fusion2 = self.decoder_ir(rgb)
        
        return  rgb1, ir1, fusion1, fusion2

encoder_spa  = IR_Encoder()
encoder_spec = RGB_Encoder()
model_en  = IR_RGB_encoder(encoder_spa,  encoder_spec).cuda() 
params = []
params += [x for x in model_en.parameters()]
optimizer    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 

# a = 0.5
# median_rgb = a*ir+(1-a)*rgb
# median_ir = (1-a)*ir+a*rgb
  
'''------------------------------------main -----------------------------------------'''
for iter in range(max_iter):
    
    gg, gg1 = model_en(ir, rgb)
    fusion = gg1
    #loss1 = 1*torch.norm(rgb1-rgb,2) +  torch.norm(ir1-ir,2)
    loss2 = torch.norm(gg,2)
    loss3 = 1*torch.norm(gg1-ir,2) + 1*torch.norm(gg1-rgb,2)
    loss  = 0.75*loss2 + loss3

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # if iter % 1000 == 0:
    #     HR_HSI = fusion
    #     HR_HSIt = HR_HSI.permute(1,2,0)
    #     ps = peak_signal_noise_ratio(np.clip(HR_HSIt.cpu().detach().numpy(),0,1),gt.permute(1,2,0).cpu().detach().numpy())
    #     print('iteration:',iter,'PSNR',ps)
        
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
    
decoder_ir  = IR_Decoder()
decoder_rgb = RGB_Decoder() 
model_de  = IR_RGB_decoder(decoder_ir,  decoder_rgb).cuda() 
params = []
params += [x for x in model_de.parameters()]
optimizer1    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 
for iter in range(5001):
    
    rgb1, ir1, fusion1, fusion2 = model_de(gg1, ir, rgb)
    fusion = fusion1
    loss1 = torch.norm(rgb1-rgb,2) 
    
    loss  = loss1

    optimizer1.zero_grad()
    loss.backward(retain_graph=True)
    optimizer1.step()
    if iter % 1000 == 0:
        HR_HSI = fusion
        HR_HSIt = HR_HSI.permute(1,2,0)
        # ps = peak_signal_noise_ratio(np.clip(HR_HSIt.cpu().detach().numpy(),0,1),gt.permute(1,2,0).cpu().detach().numpy())
        # print('iteration:',iter,'PSNR',ps)
        
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
        
        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir1, rgb, fusion)
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