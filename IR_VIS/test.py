import os
import sys
os.chdir(sys.path[0])
from PIL import Image
from torchvision import transforms
import cv2
from model.model import *

WEIGHTS_ENCODER = './checkpoints/model_en.pt'
WEIGHTS_DECODER = './checkpoints/model_de.pt'
DATA=f'/home/BlueDisk/Dataset/FusionDataset/RGBT/MSRS/test/'
    
def main():
    encoder_spa  = IR_Encoder()
    encoder_spec = RGB_Encoder()
    model_en  = IR_RGB_encoder(encoder_spa,  encoder_spec).cuda() 
    model_en.load_state_dict(torch.load(WEIGHTS_ENCODER))
    decoder_ir  = IR_Decoder()
    decoder_rgb = RGB_Decoder() 
    model_de  = IR_RGB_decoder(decoder_ir,  decoder_rgb).cuda() 
    model_de.load_state_dict(torch.load(WEIGHTS_DECODER))
    files=os.listdir(DATA+"/vi/")
    convert_tensor = transforms.ToTensor()
    with torch.no_grad():
        for file in files:
            vi=DATA+'vi/'+file
            ir=DATA+'ir/'+file
            image1 = Image.open(vi)
            image2 = Image.open(ir)
            vi  = convert_tensor(image1).to('cuda:0')
            ir = convert_tensor(image2).to('cuda:0').repeat(3, 1, 1)

            gg, gg1 = model_en(ir, vi)
            irr,  fusion, ir_detail = model_de(gg1, ir, vi)
            fusion_np_rgb = fusion.permute(1, 2, 0).cpu().detach().numpy()
            fusion_np_bgr = cv2.cvtColor(fusion_np_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'./out/{file}',fusion_np_bgr*255)
            print(file)

if __name__ == "__main__":
    main()

