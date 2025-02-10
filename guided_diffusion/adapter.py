import torch
import torch.nn as nn
import math
import pdb

import numpy as np
import os


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, ksize=3, sk=False):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None


    def forward(self, x, temb):
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)
        # pdb.set_trace()
        h = x + temb[:,:,None,None]

        h = self.block1(h)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class Adapter(nn.Module):
    def __init__(self, nums_rb=20, cin=64, ksize=3, sk=True, args=None):
        super(Adapter, self).__init__()
        self.args = args
        self.unshuffle = nn.PixelUnshuffle(4)
        self.nums_rb = nums_rb
        self.body = nn.ModuleList()

        self.conv_in = nn.Conv2d(48, 96, 3, 1, 1)

        for i in range(nums_rb):
            self.body.append(ResnetBlock(96, 96, ksize=ksize, sk=sk))

        self.conv_out = nn.Conv2d(96, 48, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(4)

        if self.args.add_temb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(96, 96),
                torch.nn.Linear(96, 96),
            ])


        

    def forward(self, x, t):
        if self.args.add_temb:
            temb = get_timestep_embedding(t, 96)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = torch.zeros([1,96]).cuda()

        ori_x = x
        x = self.unshuffle(x)
        x = self.conv_in(x)

        for i in range(self.nums_rb):
            x = self.body[i](x, temb)

        x = self.conv_out(x)
        x = self.shuffle(x)

        if self.args.res_adap:
            x = x + ori_x

        return x


    def load_pth(self, pth, device):
        stata_dict = torch.load(pth, map_location=device)
        
        weights_dict = {}
        for k, v in stata_dict['state_dict'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v


        self.load_state_dict(weights_dict)
        

