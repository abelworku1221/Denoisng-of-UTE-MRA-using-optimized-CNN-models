import numpy as np
import glob
import random
import math
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from numba.cuda.simulator.reduction import reduce
from pthflops import count_ops
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchsummary import summary
import warnings
from skimage.exposure import match_histograms


class ConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class AttentionBlock3D(nn.Module):
    def __init__(self, f_g, f_l):
        super().__init__()
        self.w_g = nn.Conv3d(f_g, 1, kernel_size=1, stride=1, padding=0)
        self.w_x = nn.Conv3d(f_l, 1, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)

        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.act(g1 + x1)
        psi = self.sig(self.psi(psi))
        # print(psi.shape)

        out = psi * x

        return out


class Denoise_AUnet(nn.Module):
    def __init__(self):
        super(Denoise_AUnet, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.UpSample3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.UpSample2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.UpSample1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

        # Encoder
        self.enc1 = ConvBlock3D(ch_in=1, ch_out=32)
        self.enc2 = ConvBlock3D(ch_in=32, ch_out=64)
        self.enc3 = ConvBlock3D(ch_in=64, ch_out=128)

        self.bk = ConvBlock3D(ch_in=128, ch_out=256)

        # Decoder

        self.dec3 = ConvBlock3D(ch_in=256, ch_out=128)
        self.dec2 = ConvBlock3D(ch_in=128, ch_out=64)
        self.dec1 = ConvBlock3D(ch_in=64, ch_out=32)

        self.att3 = AttentionBlock3D(f_g=128, f_l=128)
        self.att2 = AttentionBlock3D(f_g=64, f_l=64)
        self.att1 = AttentionBlock3D(f_g=32, f_l=32)

        self.norm3 = nn.BatchNorm3d(128)
        self.norm2 = nn.BatchNorm3d(64)
        self.norm1 = nn.BatchNorm3d(32)

        # Final layer
        self.last_conv = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()




    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)                      # 32 x 128 x 128 x 128
        x2 = self.enc2(self.max_pool(x1))      # 64 x 64 x 64 x 64
        x3 = self.enc3(self.max_pool(x2))      # 128 x 32 x 32 x 32

        bk = self.bk(self.max_pool(x3))        # 256 x 16 x 16 x 16

        # Decoder + Concat
        d3 = self.UpSample3(bk)     # 128 x 32 x 32 x 32
        x3 = self.att3(g=d3, x=x3)              # 128 x 32 x 32 x 32
        d3 = torch.cat((x3, d3), dim=1)  # 256 x 32 x 32 x 32
        d3 = self.dec3(d3)                      # 128 x 32 x 32 x 32

        d2 = self.UpSample2(d3)                 # 64 x 64 x 64 x 64
        x2 = self.att2(g=d2, x=x2)              # 64 x 64 x 64 x 64
        d2 = torch.cat((x2, d2), dim=1)  # 128 x 64 x 64 x 64
        d2 = self.dec2(d2)                      # 64 x 64 x 64 x 64

        d1 = self.UpSample1(d2)                 # 32 x 128 x 128 x 128
        x1 = self.att1(g=d1, x=x1)              # 32 x 128 x 128 x 128
        d1 = torch.cat((x1, d1), dim=1)  # 64 x 128 x 128 x 128
        d1 = self.dec1(d1)                      # 32 x 128 x 128 x 128

        d1 = self.last_conv(d1)                # 1 x 128 x 128 x 128
        out = self.sig(d1)
        return out

