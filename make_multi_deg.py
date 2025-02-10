import os
import cv2
import torch
from torchvision import transforms
import pdb
import numpy as np

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def Deg(x, scale, deg):
    if deg == 'sr':
        A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
        Ap = lambda z: MeanUpsample(z,scale)
        return Ap(A(x))
    else: # inp
        loaded = np.load(f"exp/inp_masks/{deg}.npy")
        mask = torch.from_numpy(loaded)
        A = lambda z: z*mask
        return A(x)

def downsample_and_save(input_folder, output_folder, scale, deg):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取输入文件夹下的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 读取图片
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # 转换为PyTorch张量
            img_tensor = transforms.ToTensor()(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)  # 添加批次维度

            # 进行Deg, sr or inp(mask, draw, facemask, half)
            downsampled_img = Deg(img_tensor, 4, deg)

            # 转回NumPy数组
            downsampled_img = downsampled_img.squeeze().numpy()
            downsampled_img = np.transpose(downsampled_img, (1,2,0))
            # 保存下采样后的图片
            output_path = os.path.join(output_folder, f"{deg}_{filename}")
            cv2.imwrite(output_path, downsampled_img*255)
            # cv2.imwrite(output_path, (downsampled_img * 255).astype('uint8'))  # 乘以255并转为uint8


if __name__ == "__main__":
    deg = 'draw' # sr or inp(mask, draw, facemask, half)

    input_folder = "exp/datasets/celeba_hq_sr/face/"
    output_folder = f"exp/datasets/celeba_hq_sr_{deg}/face/"
    scale_factor = 4  # 可根据需求调整下采样的比例
    

    downsample_and_save(input_folder, output_folder, scale_factor, deg)
