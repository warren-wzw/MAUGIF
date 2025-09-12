import torch
from torch import nn, optim 
    
class RGB_Encoder(nn.Module):
    def __init__(self):
        super(RGB_Encoder, self).__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=11, padding=5),
            nn.ReLU(inplace=True) # Use ReLU instead of sin
        )
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True) # Adding an activation here is also good practice
        )
        
    def forward(self, x):
        # x shape: [B, 1, H, W]
        features = self.encoder_block1(x) # Shape: [B, 3, H, W]
        concatenated = torch.cat([x, features], dim=1) # Shape: [B, 4, H, W]
        output = self.encoder_block2(concatenated) # Shape: [B, 1, H, W]
        
        return output

class IR_Encoder(nn.Module):
    def __init__(self):
        super(IR_Encoder, self).__init__()
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=11, padding=5),
            nn.ReLU(inplace=True) # Use ReLU instead of sin
        )
        self.encoder_block2 = nn.Sequential(
            # Input channels are 4 (1 from original input + 3 from block1)
            nn.Conv2d(4, 1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True) # Adding an activation here is also good practice
        )
        
    def forward(self, x):
        # x shape: [B, 1, H, W]
        features = self.encoder_block1(x) # Shape: [B, 3, H, W]
        concatenated = torch.cat([x, features], dim=1) # Shape: [B, 4, H, W]
        output = self.encoder_block2(concatenated) # Shape: [B, 1, H, W]
        
        return output

class RGB_Decoder(nn.Module): 
    def __init__(self):
        super(RGB_Decoder, self).__init__()
        self.conv3 = torch.nn.Conv2d(2,   3, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(5,   1, kernel_size=3, padding=1)
        
    def forward(self, x, y):
        x = torch.cat([x,y],dim=1)
        x1 = torch.sin(self.conv3(x))
        x2 = torch.cat([x,x1],dim=1)
        x3 = self.conv4(x2)
        return x3 #1 W H
    
class IR_Decoder(nn.Module):
    def __init__(self):
        super(IR_Decoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(4,   1, kernel_size=9, padding=4)  

    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],dim=1)
        x3 = self.conv2(x2)
        return x3
    
class Encoder(nn.Module):
    def __init__(self, ir_encoder,  rgb_encoder):
        super(Encoder, self).__init__()
        self.encoder_ir  = ir_encoder
        self.encoder_rgb = rgb_encoder

    def forward(self, ir, rgb):
        vis_en = self.encoder_rgb(rgb) #B 1 H W
        ir_en = self.encoder_ir(ir) #B 1 H W
        return ir_en, vis_en 
    
class Decoder(nn.Module):
    def __init__(self, ir_decoder, rgb_decoder):
        super(Decoder, self).__init__()
        self.decoder_ir  = ir_decoder
        self.decoder_rgb = rgb_decoder

    def forward(self,vis_en, ir, vis):
        ir_detail = self.decoder_rgb(ir, vis)
        ir1 = vis_en + ir_detail
        ir_detail = torch.where(ir_detail< 0, torch.tensor(0.0).to(ir_detail.device), ir_detail)
        fusion1 = vis+ir_detail
        return  ir1,  fusion1, ir_detail
    

