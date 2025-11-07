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
from model.loss import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=8
EPOCH=100
image_path_rgb = f'/home/BlueDisk/Dataset/FusionDataset/Done/MultiFocus_MFI-WHU/train/vi/'
image_path_ir = f'/home/BlueDisk/Dataset/FusionDataset/Done/MultiFocus_MFI-WHU/train/ir/'


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    setup_seed(2002)
    lr_real = 0.0005
    """model define"""
    encoder_ir  = IR_Encoder().to(DEVICE)
    encoder_vis = RGB_Encoder().to(DEVICE)
    model_en  = Encoder(encoder_ir,  encoder_vis).to(DEVICE)
    decoder_ir  = IR_Decoder().to(DEVICE)
    decoder_vis = RGB_Decoder().to(DEVICE)
    model_de  = Decoder().to(DEVICE)
    """loss define"""
    Grad=Fusionloss()
    """dataset"""
    dataset = RGBT_Dataset(image_path_rgb,image_path_ir,transforms==True)
    num_work = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    dataloader_train = DataLoader(dataset=dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True,
                        pin_memory=True,
                        num_workers=num_work,
                        collate_fn=dataset.collate_fn)

    optimizer_en    = optim.Adam(model_en.parameters(), lr=lr_real, weight_decay=1e-8)
    optimizer_de    = optim.Adam(model_de.parameters(), lr=lr_real, weight_decay=1e-8)  

    """Encoder training"""
    best_loss = float("inf")  
    best_epoch = -1
    model_en.train()
    torch.cuda.empty_cache()
    for epoch_index in range(EPOCH):
        loss_sum=0
        train_iterator = tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (vis, ir) in enumerate(train_iterator):
            vis, ir= vis.to(DEVICE),ir.to(DEVICE) #b 3 H W b 1 H W  
            Y_vis,_,_= RGB2YCrCb(vis)
            vis=Y_vis
            ir_en, vis_en = model_en(ir, vis) #b 1 H W b 1 H W->
            ir_pred=decoder_ir(ir_en)
            vis_pred=decoder_vis(vis_en)
            loss2 = torch.norm(ir_en-vis_en,2)
            loss3 = 1*torch.norm(vis_en-ir,2) + 0.9*torch.norm(vis_en-Y_vis,2)
            loss4=torch.norm(vis_pred-vis,2) + torch.norm(ir_pred-ir,2)
            loss  = 0.2*loss2 + loss3+0*loss4
            optimizer_en.zero_grad()
            loss.backward()
            optimizer_en.step()
            loss_sum=loss_sum+loss.item()
            avg_loss = loss_sum / (step + 1)
            train_iterator.set_description(f"Epoch={epoch_index} loss={avg_loss:.6f}")
            # ir_detail = vis_en[0].permute(1, 2, 0).cpu().detach().numpy()
            # ir_detail = cv2.cvtColor(ir_detail, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f'./out/tmp.png',ir_detail*255)
        epoch_loss = loss_sum / len(train_iterator)
        
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model_en.state_dict(), "./checkpoints/model_en.pt")
            print(f"✔ Epoch {epoch_index}: new best model saved with loss {best_loss:.6f}")
    print("Enoder training done")

    """Decoder training"""
    best_loss = float("inf")  
    best_epoch = -1
    model_en.load_state_dict(torch.load("./checkpoints/model_en.pt"))
    model_en.to(DEVICE)
    model_en.eval()
    model_de.train()
    torch.cuda.empty_cache()
    for epoch_index in range(EPOCH):
        loss_sum=0
        train_iterator = tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (vis, ir) in enumerate(train_iterator):
            vis, ir= vis.to(DEVICE),ir.to(DEVICE)
            Y_vis,_,_= RGB2YCrCb(vis)
            vis=Y_vis
            with torch.no_grad():
                ir_en, vis_en = model_en(ir, vis)
            irr,  fusion_Y, ir_detail = model_de(vis_en, ir, vis)
            grad_loss=Grad(vis,ir,fusion_Y)
            loss = torch.norm(irr-ir,2) 
            loss =loss+ 66*grad_loss
            optimizer_de.zero_grad()
            loss.backward()
            loss_sum=loss_sum+loss.item()   
            optimizer_de.step()
            avg_loss = loss_sum / (step + 1)
            train_iterator.set_description(f"Epoch={epoch_index} loss={avg_loss:.6f} grad_loss={grad_loss:.6f}")
        epoch_loss = loss_sum / len(train_iterator)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model_de.state_dict(), "./checkpoints/model_de.pt")
            print(f"✔ Epoch {epoch_index}: new best model saved with loss {best_loss:.6f}")



if __name__=="__main__":
    main()