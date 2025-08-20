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
from FusionImageEvalution.SSIM_loss import SSIM_loss


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


lr_real = 0.00025
max_iter =  3001

[L, M, N]    = rgb.shape


'''-----------------------------------------------------------------------'''
# IR编码器
class IR_Encoder(nn.Module):
    def __init__(self):
        super(IR_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,   2, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(3,   1, kernel_size=9, padding=4)  
        self.conv3 = torch.nn.Conv2d(4,   1, kernel_size=7, padding=3)  

    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        x4 = torch.cat([x3,x2],0)
        x5 = self.conv3(x4)
        return x5

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
        self.conv2 = torch.nn.Conv2d(6,   1, kernel_size=9, padding=4)  
        
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
        
        ir3 = torch.rand_like(rgb)
        ir3[0,:,:] = rgb_en
        ir3[1,:,:] = ir_en
        ir3[2,:,:] = ir
        
        ir4 = torch.rand_like(rgb)
        ir4[0,:,:] = ir
        ir4[1,:,:] = ir
        ir4[2,:,:] = ir
        
        ir_de  = self.decoder_ir(ir3)
        rgb_de = self.decoder_rgb(ir4)
    
        fusion_ir  = self.decoder_ir(rgb)
        fusion_rgb = self.decoder_rgb(ir3)
        return ir_en, ir_de, rgb_en, rgb_de, fusion_ir, fusion_rgb

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
    
    loss1 = 1*torch.norm(rgb_de-rgb,1) + torch.norm(ir_de-ir,1)
    #losss = 1-SSIM_loss(rgb_de,rgb)-SSIM_loss(ir_de,ir)
    loss2 = torch.norm(ir_en-rgb_en,1)
    loss3 = 1.3*torch.norm(fusion_rgb-rgb,1) + torch.norm(fusion_rgb-ir3,1)
    #loss3 = torch.norm(fusion_rgb-rgb,2)
    loss = 1*loss1 + 1*loss2 + 10*loss3 # + 10*losss

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if iter % 300 == 0:
        HR_HSI = fusion_rgb
        HR_HSIt = HR_HSI.permute(1,2,0)
        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir, rgb, HR_HSI)
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
    if iter % 500 == 0:
        HR_HSI = fusion_rgb
        HR_HSIt = HR_HSI.permute(1,2,0)
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
    
    
