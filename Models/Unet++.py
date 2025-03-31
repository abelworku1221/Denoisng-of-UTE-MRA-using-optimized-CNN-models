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
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out


class Denoise_Unetpp(nn.Module):
    def __init__(self, input_channels=1, deep_supervision=True):
        super(Denoise_Unetpp, self).__init__()

        num_filter = [32, 64, 128, 256]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.UpSample = nn.Upsample(mode="nearest", scale_factor=2.0)


        # DownSampling
        self.conv0_0 = ConvBlock3D(input_channels, num_filter[0])
        self.conv1_0 = ConvBlock3D(num_filter[0], num_filter[1])
        self.conv2_0 = ConvBlock3D(num_filter[1], num_filter[2])

        self.conv_bk = ConvBlock3D(num_filter[2], num_filter[3])

        # Upsampling & Dense skip
        # N to 1 skip

        self.conv0_1 = ConvBlock3D(num_filter[0] + num_filter[1], num_filter[0])
        self.conv1_1 = ConvBlock3D(num_filter[1] + num_filter[2], num_filter[1])
        self.conv2_1 = ConvBlock3D(num_filter[2] + num_filter[3], num_filter[2])

        # N to 2 skip
        self.conv0_2 = ConvBlock3D(num_filter[0] * 2 + num_filter[1], num_filter[0])
        self.conv1_2 = ConvBlock3D(num_filter[1] * 2 + num_filter[2], num_filter[1])

        self.conv0_3 = ConvBlock3D(num_filter[0] * 3 + num_filter[1], num_filter[0])

        self.output = nn.Conv3d(num_filter[0], 1, kernel_size=3, padding = 1)

        self.sig = nn.Sigmoid()


    def forward(self, x):   # (Batch, 1, 128, 128, 128)
        x0_0 = self.conv0_0(x)   # (Batch, 32, 128, 128, 128)
        x1_0 = self.conv1_0(self.pool(x0_0))  # (Batch, 64, 64, 64, 64)
        x2_0 = self.conv2_0(self.pool(x1_0))  # (Batch, 128, 32, 32, 32)

        x_bk = self.conv_bk(self.pool(x2_0))   # (Batch, 256, 16, 16, 16)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.UpSample(x1_0)], dim=1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.UpSample(x2_0)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.UpSample(x_bk)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.UpSample(x1_1)], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.UpSample(x2_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.UpSample(x1_2)], dim=1))

        if self.deep_supervision:
            output1 = self.sig(self.output(x0_1))
            output2 = self.sig(self.output(x0_2))
            output3 = self.sig(self.output(x0_3))


            output = (output1 + output2 + output3) / 3.0
        else:
            output = self.sig(self.output(x0_3))

        return output

