from PIL import Image
from FusionImageEvalution.Metric_Python.Metric import *
from time import time
import warnings

warnings.filterwarnings("ignore")


def evaluation_one(ir_name, vi_name, f_name):
    f_img = tensor_to_gray_img(f_name)
    ir_img = tensor_to_gray_img(ir_name)
    vi_img = tensor_to_gray_img(vi_name)
    # f_img = f_name.cpu().detach().numpy()
    # ir_img = ir_name.cpu().detach().numpy()
    # vi_img = vi_name.cpu().detach().numpy()
    f_img_int = np.array(f_img).astype(np.int32)

    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)

    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)
    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM

def tensor_to_gray_img(tensor):
    # Ensure tensor is 3D: (C, H, W)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0) # (H, W) -> (1, H, W)
    elif tensor.ndim == 1:
        raise ValueError(f"Tensor too small: shape {tensor.shape}")

    np_img = tensor.cpu().detach().numpy()

    if np_img.shape[0] == 1:
        # Single channel image, shape (1, H, W)
        np_img = np.squeeze(np_img, axis=0) # -> (H, W)
        np_img = (np_img * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode='L')

    elif np_img.shape[0] == 3:
        # RGB image, shape (3, H, W)
        np_img = np.transpose(np_img, (1, 2, 0)) # -> (H, W, 3)
        np_img = (np_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        return pil_img.convert('L')

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

if __name__ == '__main__':
    f_name = r'E:\Desktop\metric\Test\Results\TNO\GTF\01.png'
    ir_name = r'E:\Desktop\metric\Test\datasets\TNO\ir\01.png'
    vi_name = r'E:\Desktop\metric\Test\datasets\TNO\vi\01.png'
    EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name, f_name)
    print('EN:', round(EN, 4))
    print('MI:', round(MI, 4))
    print('SF:', round(SF, 4))
    print('AG:', round(AG, 4))
    print('SD:', round(SD, 4))
    print('CC:', round(CC, 4))
    print('SCD:', round(SCD, 4))
    print('VIF:', round(VIF, 4))
    print('MSE:', round(MSE, 4))
    print('PSNR:', round(PSNR, 4))
    print('Qabf:', round(Qabf, 4))
    print('Nabf:', round(Nabf, 4))
    print('SSIM:', round(SSIM, 4))
    print('MS_SSIM:', round(MS_SSIM, 4))
