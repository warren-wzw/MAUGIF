import os
import torch

from PIL import Image
from torch.utils.data import  Dataset
import torchvision.transforms as transforms

class JointTransform:
    def __init__(self, size=480):
        self.random_resized_crop = transforms.RandomResizedCrop(size)
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, vis, ir):
        # 先把它们堆在一起，保证同样的 transform 参数
        stacked = Image.merge("RGBA", (vis.split()[:3] + (ir,)))
        stacked = self.random_resized_crop(stacked)
        stacked = self.random_horizontal_flip(stacked)

        # 拆回 RGB + IR
        vis = stacked.convert("RGB")
        ir  = stacked.split()[-1]  # 拿最后一通道

        vis = self.to_tensor(vis)
        ir  = self.to_tensor(ir)
        return vis, ir

class RGBT_Dataset(Dataset):   #only when json is standard json form,it will speed up
    def __init__(self, vis_path: str, ir_path: str, transform=False):
        self.transform=transform
        self.img_paths = sorted([os.path.join(vis_path, f) for f in os.listdir(vis_path)])
        self.ir_paths  = sorted([os.path.join(ir_path,  f) for f in os.listdir(ir_path)])
        assert len(self.img_paths) == len(self.ir_paths), "Vis and IR image counts do not match!"
        self.total_num = len(self.img_paths)
        self.transform = JointTransform(size=480)

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        vis = Image.open(self.img_paths[idx]).convert("RGB")
        ir  = Image.open(self.ir_paths[idx]).convert("L")   # 假设红外是单通道

        if self.transform is not None:
            vis, ir = self.transform(vis, ir)

        return vis, ir

    @staticmethod
    def collate_fn(batch):
        vis, ir = tuple(zip(*batch))
        vis = torch.stack(vis, dim=0)  # 将图像堆叠成一个张量
        ir = torch.stack(ir, dim=0)  # 将图像堆叠成一个张量
        return vis, ir