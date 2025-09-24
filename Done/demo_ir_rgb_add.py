import torch
from torch import nn, optim 
#from model import *
import os
import sys
os.chdir(sys.path[0])
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import math
import random
from PIL import Image
from torchvision import transforms
DATA=f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/test/'
FILE="00131D.png"



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2) 

# 图像路径
image_path_1 = DATA+"/ir/"+FILE
image_path_2 = DATA+"/vi/"+FILE
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

        # gg   = self.encoder_ir(ir-rgb)
        # gg1  = self.encoder_ir(rgb)
        # return gg, gg1
        g1 = self.encoder_ir(ir)
        gg1 = self.encoder_rgb(rgb)
        return g1, gg1
    
class IR_RGB_decoder(nn.Module):
    def __init__(self, ir_decoder, rgb_decoder):
        super(IR_RGB_decoder, self).__init__()
        self.decoder_ir  = ir_decoder
        self.decoder_rgb = rgb_decoder

    def forward(self, gg1, ir, rgb):
        ir_detail = self.decoder_rgb(ir, rgb)
        ir1 = gg1 + ir_detail
        ir_detail1 = torch.where(ir_detail< 0.2, torch.tensor(0.0).to(ir_detail.device), ir_detail)
        fusion1 = rgb+ir_detail1
        return  ir1,  fusion1, ir_detail1

encoder_spa  = IR_Encoder()
encoder_spec = RGB_Encoder()
model_en  = IR_RGB_encoder(encoder_spa,  encoder_spec).cuda() 
params = []
params += [x for x in model_en.parameters()]
optimizer    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 

  
'''------------------------------------main -----------------------------------------'''
for iter in range(max_iter):
    
    gg, gg1 = model_en(ir, rgb)
    fusion = gg1
    loss2 = torch.norm(gg-gg1,2)
    loss3 = 1*torch.norm(gg1-ir,2) + 0.9*torch.norm(gg1-rgb,2)
    loss  = 0.2*loss2 + loss3

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
  
plt.figure(figsize=(6, 6))
plt.imshow(gg.permute(1,2,0).cpu().detach().numpy())
plt.axis('off')  # 关闭坐标轴
plt.savefig('.gg.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭图像，释放内存


decoder_ir  = IR_Decoder()
decoder_rgb = RGB_Decoder() 
model_de  = IR_RGB_decoder(decoder_ir,  decoder_rgb).cuda() 
params = []
params += [x for x in model_de.parameters()]
optimizer1    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 
for iter in range(5001): 
    irr,  fusion1, ir_detail = model_de(gg1, ir, rgb)
    fusion = fusion1
    loss1 = torch.norm(irr-ir,2) 
    loss  = loss1
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    
plt.figure(figsize=(6, 6))
plt.axis('off')  # 关闭坐标轴
plt.savefig(r'.fusion_irvis.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭图像，释放内存

plt.figure(figsize=(6, 6))

plt.axis('off')  # 关闭坐标轴
plt.savefig(r'.feature_irvis.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()  # 关闭图像，释放内存