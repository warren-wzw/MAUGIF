import torch
from torch import nn, optim 

class RGB_Encoder(nn.Module): 
    def __init__(self):
        super(RGB_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(4,   1, kernel_size=9, padding=4)  
        
    def forward(self, x):
        #inpout B, 1, H, W
        x1 = torch.sin(self.conv1(x)) #B 3 H W
        x2 = torch.cat([x,x1],dim=1)
        x3 = self.conv2(x2)
        return x3

class IR_Encoder(nn.Module):
    def __init__(self):
        super(IR_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(4,   1, kernel_size=9, padding=4)  

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat([x,x1],dim=1)
        x3 = self.conv2(x2)
        return x3

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
        self.conv1 = torch.nn.Conv2d(3,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(6,   3, kernel_size=9, padding=4)  

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

    def forward(self, gg1, ir, rgb):
        ir_detail = self.decoder_rgb(ir, rgb)
        ir1 = gg1 + ir_detail
        ir_detail1 = torch.where(ir_detail< 0.0, torch.tensor(0.0).to(ir_detail.device), ir_detail)
        fusion1 = rgb+ir_detail1
        return  ir1,  fusion1, ir_detail1

# # 卷积块：Conv + BN + ReLU
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, k, p):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.block(x)

# # ---------------- Encoders ----------------
# class RGB_Encoder(nn.Module): 
#     def __init__(self):
#         super(RGB_Encoder, self).__init__()
#         self.conv1 = ConvBlock(1, 3, 11, 5)
#         self.conv2 = ConvBlock(4, 8, 9, 4)   # 4通道 -> 8通道
#         self.out   = nn.Conv2d(8, 1, kernel_size=3, padding=1)

#     def forward(self, x):  # x: (C,H,W)
#         x = x.unsqueeze(0)            # -> (1,C,H,W) 方便Conv2d
#         x1 = self.conv1(x)            # (1,3,H,W)
#         x2 = torch.cat([x, x1], 1)    # 通道拼接 (保持dim=1更直观)
#         x3 = self.conv2(x2)
#         return self.out(x3).squeeze(0)  # 输出 (C,H,W)


# class IR_Encoder(nn.Module):
#     def __init__(self):
#         super(IR_Encoder, self).__init__()
#         self.conv1 = ConvBlock(1, 3, 11, 5)
#         self.conv2 = ConvBlock(4, 8, 9, 4)
#         self.out   = nn.Conv2d(8, 1, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = x.unsqueeze(0)
#         x1 = self.conv1(x)
#         x2 = torch.cat([x, x1], 1)
#         x3 = self.conv2(x2)
#         return self.out(x3).squeeze(0)

# # ---------------- Decoders ----------------
# class RGB_Decoder(nn.Module): 
#     def __init__(self):
#         super(RGB_Decoder, self).__init__()
#         self.conv3 = ConvBlock(2, 4, 1, 0)
#         self.conv4 = ConvBlock(6, 8, 3, 1)
#         self.out   = nn.Conv2d(8, 1, kernel_size=3, padding=1)

#     def forward(self, x, y):
#         x = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], 1)  # (1,2,H,W)
#         x1 = self.conv3(x)
#         x2 = torch.cat([x, x1], 1)
#         x3 = self.conv4(x2)
#         return self.out(x3).squeeze(0)  # (C,H,W)


# class IR_Decoder(nn.Module):
#     def __init__(self):
#         super(IR_Decoder, self).__init__()
#         self.conv1 = ConvBlock(3, 6, 11, 5)
#         self.conv2 = ConvBlock(9, 8, 9, 4)
#         self.out   = nn.Conv2d(8, 3, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = x.unsqueeze(0)
#         x1 = self.conv1(x)
#         x2 = torch.cat([x, x1], 1)
#         x3 = self.conv2(x2)
#         return self.out(x3).squeeze(0)

# # ---------------- Fusion Wrapper ----------------
# class Encoder(nn.Module):
#     def __init__(self, ir_encoder, rgb_encoder):
#         super(Encoder, self).__init__()
#         self.encoder_ir  = ir_encoder
#         self.encoder_rgb = rgb_encoder

#     def forward(self, ir, rgb):
#         vis_en = self.encoder_rgb(rgb)  # (C,H,W)
#         ir_en  = self.encoder_ir(ir)    # (C,H,W)
#         return ir_en, vis_en


# class Decoder(nn.Module):
#     def __init__(self, ir_decoder, rgb_decoder):
#         super(Decoder, self).__init__()
#         self.decoder_ir  = ir_decoder
#         self.decoder_rgb = rgb_decoder

#     def forward(self, gg1, ir, rgb):
#         ir_detail = self.decoder_rgb(ir, rgb)   # (C,H,W)
#         ir1 = gg1 + ir_detail
#         ir_detail1 = torch.relu(ir_detail)      # 简化版 ReLU
#         fusion1 = rgb + ir_detail1
#         return ir1, fusion1, ir_detail1