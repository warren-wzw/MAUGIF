import os
import sys
os.chdir(sys.path[0])
from PIL import Image
from torchvision import transforms
import cv2
from model.model import *
from model.utils import *

WEIGHTS_ENCODER = './checkpoints/model_en.pt'
WEIGHTS_DECODER = './checkpoints/model_de.pt'
DATA=f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MultiFocus/test/'
FILE="00131D.png"
    
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
    files=os.listdir(DATA+"/FAR/")
    convert_tensor = transforms.ToTensor()
    with torch.no_grad():
        vis=DATA+'FAR/'+FILE
        ir=DATA+'Near/'+FILE
        image1 = Image.open(vis)
        image2 = Image.open(ir).convert("L")
        vis  = convert_tensor(image1).to('cuda:0')
        ir = convert_tensor(image2).to('cuda:0').unsqueeze(0)
        Y_vis,Cb_vis,Cr_vis= RGB2YCrCb(vis.unsqueeze(0))
        vis=Y_vis

        ir_en, vis_en = model_en(ir, vis)
        irr,  fusion_Y, ir_detail = model_de(vis_en, ir, vis)

        fusion_out=YCbCr2RGB(fusion_Y,Cb_vis,Cr_vis)
        fusion_out = fusion_out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        fusion_out = cv2.cvtColor(fusion_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./out/{FILE}',fusion_out*255)


if __name__ == "__main__":
    main()