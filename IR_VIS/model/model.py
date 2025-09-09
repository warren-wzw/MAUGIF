import torch
from torch import nn, optim 
'''---------------------------------Model--------------------------------------'''
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
        ir_detail1 = torch.where(ir_detail< 0.0, torch.tensor(0.0).to(ir_detail.device), ir_detail)
        fusion1 = rgb+ir_detail1
        return  ir1,  fusion1, ir_detail1