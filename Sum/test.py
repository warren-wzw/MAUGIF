import os
from socket import CAN_ISOTP
import sys
os.chdir(sys.path[0])
from PIL import Image
from torchvision import transforms
import cv2
from thop import profile
from model.model import *
from model.utils import *
import time

WEIGHTS_ENCODER = './checkpoints/model_en.pt'
WEIGHTS_DECODER = './checkpoints/model_de.pt'
DATA=f'/home/BlueDisk/Dataset/FusionDataset/MultiFocus/WHU/test/'
    
def main():
    encoder_ir  = IR_Encoder()
    encoder_vis = RGB_Encoder() 
    decoder_ir  = IR_Decoder()
    decoder_vis = RGB_Decoder() 
    model_en  = Encoder(encoder_ir,  encoder_vis).cuda()
    model_de  = Decoder().cuda() 
    """load checkpoints"""
    model_en.load_state_dict(torch.load(WEIGHTS_ENCODER))
    model_de.load_state_dict(torch.load(WEIGHTS_DECODER))
    files=os.listdir(DATA+"/ir/")
    convert_tensor = transforms.ToTensor()
    start=time.time()
    with torch.no_grad():
        for file in files:
            vis=DATA+'vi/'+file
            ir=DATA+'ir/'+file
            image1 = Image.open(vis).convert("RGB")
            image2 = Image.open(ir).convert("L")
            vis  = convert_tensor(image1).to('cuda:0')
            ir = convert_tensor(image2).to('cuda:0').unsqueeze(0)
            Y_vis,Cb_vis,Cr_vis= RGB2YCrCb(vis.unsqueeze(0))
            vis=Y_vis

            ir_en, vis_en = model_en(ir, vis)
            irr,  fusion_Y, ir_detail = model_de(ir_en+vis_en, ir, vis)
            """FLOPS"""
            # flops, params = profile(model_en, inputs=(ir,vis))
            # gflops = flops / 1e9  
            # params_million = params / 1e6  
            # print(f"Total FLOPs: {gflops:.2f} GFLOPs") 
            # print(f"Total Parameters: {params_million:.2f} M") 
            
            # flops, params = profile(model_de, inputs=(ir_en+vis_en,ir,vis))
            # gflops = flops / 1e9  
            # params_million = params / 1e6  # 
            # print(f"Total FLOPs: {gflops:.2f} GFLOPs") 
            # print(f"Total Parameters: {params_million:.2f} M") 
            """output"""
            fusion_out=YCbCr2RGB(fusion_Y,Cb_vis,Cr_vis)
            fusion_out = fusion_out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            fusion_out = cv2.cvtColor(fusion_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'./out/{file}',fusion_out*255)
            print(file)
    end=time.time()
    print(end-start)
            

if __name__ == "__main__":
    main()

