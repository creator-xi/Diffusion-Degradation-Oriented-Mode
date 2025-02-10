import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from functions.jpeg import jpeg_decode, jpeg_encode
import pdb

class Inpainting_SR:
    # RGB
    
    def __init__(self, channels, img_dim, missing_indices, ratio, device):
        from functions.svd_operators import SuperResolution
        from functions.svd_operators import Inpainting
        self.ipt = Inpainting(channels, img_dim, missing_indices, device) 
        self.sr = SuperResolution(channels, img_dim, ratio, device)
    
    def A(self, vec):
        temp = self.ipt.A(vec)
        out = self.sr.A(vec)
        return out
    
    # def At(self, vec):
    #     return jpeg_decode(vec, self.jpeg_qf)
    
    def A_pinv(self, vec):
        temp = self.sr.A_pinv(vec)
        out = self.ipt.A_pinv(temp)
        return out
    