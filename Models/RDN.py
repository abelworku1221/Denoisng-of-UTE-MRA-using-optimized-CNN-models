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


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(num_filter + 0, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(num_filter + 16, 16, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv3d(num_filter + 32, num_filter, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.act(self.conv1(x))
        out2 = self.act(self.conv2(torch.cat([x, out1], 1)))
        out = self.conv(torch.cat([x, out1, out2], 1))
        return out + x



class Denoise_RDN(nn.Module):
    def __init__(self):
        super(Denoise_RDN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.RDB = ResidualDenseBlock(32)

        self.norm = nn.InstanceNorm3d(32)
        self.act = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))

        temp = x.clone()

        for i in range(2):
            x = self.RDB(x)

        x = self.act(self.conv2(x + temp))

        x = self.sig(self.conv3(x))

        return x

