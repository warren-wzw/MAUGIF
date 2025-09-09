import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import sys
import os
os.chdir(sys.path[0])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from PIL import Image
from torchvision import transforms
import cv2
from model.model import *
FILE="00131D.png"
# 图像路径
image_path_ir = f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/test/ir/{FILE}'
image_path_rgb = f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/test/vi/{FILE}'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def main():
    setup_seed(2002)
    lr_real = 0.00025
    max_iter =  5001 
    """load image"""
    image_path_1 = image_path_rgb
    image_path_2 = image_path_ir
    image1 = Image.open(image_path_1)
    image2 = Image.open(image_path_2)
    convert_tensor = transforms.ToTensor()
    ir  = convert_tensor(image1).to('cuda:0')
    rgb2 = convert_tensor(image2).to('cuda:0')
    rgb = torch.rand_like(ir)
    rgb[0,:,:] = rgb2
    rgb[1,:,:] = rgb2
    rgb[2,:,:] = rgb2

    [L, M, N]    = rgb.shape
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
        loss.backward()
        optimizer.step()
    torch.save(model_en.state_dict(), "./checkpoints/model_en.pt")
    with torch.no_grad():
        gg, gg1 = model_en(ir, rgb) 



    decoder_ir  = IR_Decoder()
    decoder_rgb = RGB_Decoder() 
    model_de  = IR_RGB_decoder(decoder_ir,  decoder_rgb).cuda() 
    params = []
    params += [x for x in model_de.parameters()]
    optimizer1    = optim.Adam(params, lr=lr_real, weight_decay=1e-8) 
    for iter in range(5001):
        irr,  fusion1, ir_detail = model_de(gg1, ir, rgb)
        fusion = fusion1
        loss = torch.norm(irr-ir,2) 
        optimizer1.zero_grad()
        loss.backward()   
        optimizer1.step()
    torch.save(model_de.state_dict(), "./checkpoints/model_de.pt")
    fusion_np_rgb = fusion.permute(1, 2, 0).cpu().detach().numpy()
    fusion_np_bgr = cv2.cvtColor(fusion_np_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./out/{FILE}',fusion_np_bgr*255)


if __name__=="__main__":
    main()