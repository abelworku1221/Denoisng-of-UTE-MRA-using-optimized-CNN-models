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


class Denoise_Unet(nn.Module):
    def __init__(self):
        super(Denoise_Unet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        # Final output
        self.conv_final = nn.Conv3d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool3d(enc2, kernel_size=2))
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc3, kernel_size=2))
        # Decoder
        up3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        # Final output
        return torch.sigmoid(self.conv_final(dec1))
