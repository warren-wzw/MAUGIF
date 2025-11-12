import torch
from torch import nn

# 空间编码器
class Spatial_Encoder(nn.Module):
    def __init__(self, H):
        super(Spatial_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(H, H, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Conv2d(H, H, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(H, H, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Conv2d(H, H, kernel_size=4, stride=2, padding=1)# 32x32 -> 16x16
        )

    def forward(self, x):
        return self.encoder(x)

# 空间解码器
class Spatial_Decoder(nn.Module):
    def __init__(self, H):
        super(Spatial_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(H, H, kernel_size=4, stride=2, padding=1),     # 16x16 -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(H, H, kernel_size=4, stride=2, padding=1),     # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


# 光谱编码器
class Spectral_Encoder(nn.Module): 
    def __init__(self, h, H):
        super(Spectral_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(H,     h, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(H+h,   h, kernel_size=9, padding=4)  
        self.conv3 = torch.nn.Conv2d(H+2*h, h, kernel_size=7, padding=3)  
        
    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        x4 = torch.cat([x2,x3],0)
        x5 = self.conv3(x4)
        return x5
    
# 光谱解码器
class Spectral_Decoder(nn.Module): 
    def __init__(self, h, H):
        super(Spectral_Decoder, self).__init__()
        self.conv3 = torch.nn.Conv2d(h,   H, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(H+h, H, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(H+h, H, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = torch.sin(self.conv3(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv4(x2)
        return x3
   
    
# 光谱自编码器组合 Encoder + Decoder  
class Spatial_Spectral_Autoencoder(nn.Module):
    def __init__(self, encoder_spa, decoder_spa, encoder_spec, decoder_spec):
        super(Spatial_Spectral_Autoencoder, self).__init__()
        self.encoder_spa  = encoder_spa
        self.decoder_spa  = decoder_spa
        self.encoder_spec = encoder_spec
        self.decoder_spec = decoder_spec

    def forward(self, Coe, MSI):
        #LR_MSI1 = self.encoder_spa(MSI)
        LR_MSI2 = self.encoder_spec(Coe)
        Coe1 = self.decoder_spec(LR_MSI2)
        HR_Coe1 = self.decoder_spec(MSI)

        
        return  LR_MSI2, HR_Coe1,  Coe1
    

'''========================================================'''
'''========================================================'''
'''========================================================'''

# IR编码器
class IR_Encoder(nn.Module):
    def __init__(self):
        super(IR_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,   3, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(4,   1, kernel_size=9, padding=4)  

    def forward(self, x):
        x1 = torch.sin(self.conv1(x))
        x2 = torch.cat([x,x1],0)
        x3 = self.conv2(x2)
        return x3

# IR解码器
class IR_Decoder(nn.Module):
    def __init__(self):
        super(IR_Decoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(5,   1, kernel_size=11, padding=5)   
        self.conv2 = torch.nn.Conv2d(6,   1, kernel_size=9, padding=4)  

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
        self.conv3 = torch.nn.Conv2d(5,   3, kernel_size=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8,   3, kernel_size=3, padding=1)
        
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
        
        M = ir.shape[1]
        N = ir.shape[2]
        ir5 = torch.zeros(5,M,N).cuda()
        ir5[0,:,:] = ir_en
        ir5[1,:,:] = ir
        ir5[2:5,:,:] = rgb
        
        ir_de = self.decoder_ir(ir5)
        rgb_de = self.decoder_rgb(ir5)
        
        
        fusion_ir  = self.decoder_ir(ir5)
        fusion_rgb = self.decoder_rgb(ir5)
        return ir_en, ir_de, rgb_en, rgb_de, fusion_ir, fusion_rgb
    