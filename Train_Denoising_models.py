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
#import pytorch_lightning as pl
from pthflops import count_ops
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchsummary import summary
import pytorch_msssim
import warnings
from skimage.exposure import match_histograms


warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("gpu is detected")


path = sorted(glob.glob("Data/combined numpy data/*.npy"))


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


#  ====== network design ===========

from Models.Unet import Denoise_Unet  # importing models. please change the model depending on which architecture

model = Denoise_Unet()
model.to(device)


MSE_loss = nn.MSELoss(reduction = "mean")


from data_loader import TrainDataset, ValidDataset

batch = 8

valid_dataset = ValidDataset()
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch, shuffle=True, num_workers=0)

train_dataset = TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=0)

# loss function
def custom_loss(y_pred, y_true):
    mse_loss = MSE_loss(y_true, y_pred)
    dtrue = y_true[:, :, 1:, 1:, 1:] - y_true[:, :, :-1, :-1, :-1]
    dpred = y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, :-1, :-1]
    gd_loss = MSE_loss(dtrue, dpred)
    return mse_loss + gd_loss


def validate_one_epoch():
    model.eval()
    cumulative_loss = 0
    for i, (ns_im, nf_im) in enumerate(valid_dataloader):
        ns_im = ns_im.to(device)
        nf_im = nf_im.to(device)
        with torch.no_grad():
            pred = model(ns_im)
            loss = custom_loss(nf_im, pred)
        cumulative_loss += loss.item()
    return cumulative_loss / len(valid_dataloader)


def train_model(epochs):
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
    train_loss_history = [] # used for checking training and plotting
    valid_loss_history = [] # used for checking training and plotting
    print("training is started...!")
    for e in range(epochs):
        train_loss = 0
        model.train()
        for i, (n_im, nf_im) in enumerate(train_dataloader):
            n_im = n_im.to(device)
            nf_im = nf_im.to(device)
            optimizer.zero_grad()
            pred = model(n_im)
            loss = custom_loss(pred, nf_im)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        valid_loss = validate_one_epoch()
        valid_loss_history.append(valid_loss)
        train_loss_history.append(train_loss)
        best_point = np.min(np.array(valid_loss_history))
        if best_point == valid_loss:
            print("better model found at epoch = ", e)
            pth = "Models/Denoising_Unet.pth"
            scripted = torch.jit.script(model)
            scripted.save(pth)
        print(e, train_loss, valid_loss)
        scheduler.step()
    best = np.argmin(np.array(valid_loss_history))
    print("Training is finished, and The best model is at epoch :", best, "the validation loss = ", valid_loss_history[best])

train_model(200)


