import os
import sys
import torch
import cv2
import random
import numpy as np
os.chdir(sys.path[0])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from PIL import Image
from torch import nn, optim  
from torchvision import transforms
from torch.utils.data import (DataLoader)

from model.model import *
from model.utils import *
from model.dataset import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=6
EPOCH=20
image_path_rgb = f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/train/vi/'
image_path_ir = f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/train/ir/'


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    setup_seed(2002)
    lr_real = 0.00025
    """model define"""
    encoder_spa  = IR_Encoder()
    encoder_spec = RGB_Encoder()
    model_en  = Encoder(encoder_spa,  encoder_spec).to(DEVICE)
    decoder_ir  = IR_Decoder()
    decoder_rgb = RGB_Decoder() 
    model_de  = Decoder(decoder_ir,  decoder_rgb).to(DEVICE)
    """dataset"""
    dataset = RGBT_Dataset(image_path_rgb,image_path_ir,transforms==True)
    num_work = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    dataloader_train = DataLoader(dataset=dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True,
                        pin_memory=True,
                        num_workers=num_work,
                        collate_fn=dataset.collate_fn)

    optimizer    = optim.Adam(model_en.parameters(), lr=lr_real, weight_decay=1e-8) 

    """Encoder training"""
    best_loss = float("inf")  
    best_epoch = -1
    model_en.train()
    torch.cuda.empty_cache()
    for epoch_index in range(EPOCH):
        loss_sum=0
        train_iterator = tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (vis, ir) in enumerate(train_iterator):
            vis, ir= vis.to(DEVICE),ir.to(DEVICE)
            Y_vis,Cb_vis,Cr_vis= RGB2YCrCb(vis)
            vis=Y_vis
            ir_en, vis_en = model_en(ir, vis) #b 1 H W
            loss2 = torch.norm(ir_en-vis_en,2)
            loss3 = 1*torch.norm(vis_en-ir,2) + 0.9*torch.norm(vis_en-vis,2)
            loss  = 0.2*loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum=loss_sum+loss.item()
            avg_loss = loss_sum / (step + 1)
            train_iterator.set_description(f"Epoch={epoch_index} loss={avg_loss:.6f}")
        epoch_loss = loss_sum / len(train_iterator)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch_index
            torch.save(model_en.state_dict(), "./checkpoints/model_en.pt")
            print(f"✔ Epoch {epoch_index}: new best model saved with loss {best_loss:.6f}")
    print("Enoder training done")

    """Decoder training"""
    best_loss = float("inf")  
    best_epoch = -1
    print("加载已训练的编码器权重用于解码器训练...")
    model_en.load_state_dict(torch.load("./checkpoints/model_en.pt"))
    model_en.to(DEVICE)
    model_en.eval()
    optimizer1    = optim.Adam(model_de.parameters(), lr=lr_real, weight_decay=1e-8) 
    model_de.train()
    torch.cuda.empty_cache()
    for epoch_index in range(EPOCH):
        loss_sum=0
        train_iterator = tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (vis, ir) in enumerate(train_iterator):
            vis, ir= vis.to(DEVICE),ir.to(DEVICE)
            Y_vis,Cb_vis,Cr_vis= RGB2YCrCb(vis)
            vis=Y_vis
            with torch.no_grad():
                ir_en, vis_en = model_en(ir, vis)
            irr,  fusion_Y, ir_detail = model_de(vis_en, ir, vis)
            loss = torch.norm(irr-ir,2) 
            optimizer1.zero_grad()
            loss.backward()
            loss_sum=loss_sum+loss.item()   
            optimizer1.step()
            avg_loss = loss_sum / (step + 1)
            train_iterator.set_description(f"Epoch={epoch_index} loss={avg_loss:.6f}")
        epoch_loss = loss_sum / len(train_iterator)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch_index
            torch.save(model_en.state_dict(), "./checkpoints/model_en.pt")
            print(f"✔ Epoch {epoch_index}: new best model saved with loss {best_loss:.6f}")



if __name__=="__main__":
    main()