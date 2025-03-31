import numpy as np
import glob
import random
import math
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import nibabel as nib
import torch.nn.functional as F
import torchvision
from pthflops import count_ops
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchsummary import summary


import warnings
from skimage.metrics import structural_similarity, mean_squared_error
warnings.filterwarnings("ignore")

device = torch.device("cpu")


test_paths = sorted(glob.glob("Data/Validation/*.npy"))  # testing dataset

# some helper functions
def maximum_intensity_projection_sagittal(data):
    res = data.shape[0]
    temp = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            lines = data[:, i, j]
            temp[i, j] = np.max(lines)
    return temp


def maximum_intensity_projection_coronal(data):
    res = data.shape[0]
    temp = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            lines = data[i, :, j]
            temp[i, j] = np.max(lines)

    return temp


def maximum_intensity_projection_axial(data):
    res = data.shape[0]
    temp = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            lines = data[i, j, :]
            temp[j, i] = np.max(lines)
    return temp


model = torch.jit.load("Models/Denoising_Unet.pth", map_location=device) # read the trained model

out_path = "Result/Unet/"  # please change this path based on the type of the network used


if not os.path.exists(out_path):
    os.makedirs(out_path)

model.to(device)

MSE_loss = nn.MSELoss()
import time


def denoise_the_image(image):
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0).astype(np.float32)
    image = torch.from_numpy(image)
    image = image.to(device)

    with torch.no_grad():
        pred = model(image)
        pred = np.squeeze(pred.detach().numpy()).astype(np.float32)
    return pred


def res_evaluation(ytrue, ypred):
    mse = mean_squared_error(ytrue, ypred)
    ssm = structural_similarity(ytrue, ypred, data_range=1.0)
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return mse, ssm, psnr


validation_MSE = []
validation_PSNR = []
validation_SSIM = []



for p in test_paths:
    print(p)
    name = os.path.splitext(os.path.basename(p))[0]
    path = out_path + str(name) + "/"

    if not os.path.exists(path):
        os.makedirs(path)

    data = np.load(p)

    image = data[0]
    y_true  = data[1]

    BM4D_noise = np.abs(data[2])

    plt.imsave(path + "original_axial.png", maximum_intensity_projection_axial(image), cmap = "gray", vmax = 1.0)
    plt.imsave(path + "original_coronal.png", maximum_intensity_projection_coronal(image), cmap="gray", vmax=1.0)
    plt.imsave(path + "original_sagittal.png", maximum_intensity_projection_sagittal(image), cmap="gray", vmax=1.0)

    plt.imsave(path + "BM4D_axial.png", maximum_intensity_projection_axial(y_true), cmap = "gray", vmax = 1.0)
    plt.imsave(path + "BM4D_coronal.png", maximum_intensity_projection_coronal(y_true), cmap="gray", vmax=1.0)
    plt.imsave(path + "BM4D_sagittal.png", maximum_intensity_projection_sagittal(y_true), cmap="gray", vmax=1.0)

    plt.imsave(path + "BM4D_noise_axial.png", maximum_intensity_projection_axial(BM4D_noise), cmap = "gray", vmax = 1.0)
    plt.imsave(path + "BM4D_noise_coronal.png", maximum_intensity_projection_coronal(BM4D_noise), cmap="gray", vmax=1.0)
    plt.imsave(path + "BM4D_noise_sagittal.png", maximum_intensity_projection_sagittal(BM4D_noise), cmap="gray", vmax=1.0)


    ti = time.time()
    y_pred = denoise_the_image(image)
    print((time.time() - ti)/60)

    plt.imsave(path + "Denoising_NN_axial.png", maximum_intensity_projection_axial(y_pred), cmap="gray", vmax=1.0)
    plt.imsave(path + "Denoising_NN_coronal.png", maximum_intensity_projection_coronal(y_pred), cmap="gray", vmax=1.0)
    plt.imsave(path + "Denoising_NN_sagittal.png", maximum_intensity_projection_sagittal(y_pred), cmap="gray", vmax=1.0)


    NN_noise = np.abs(image - y_pred)

    plt.imsave(path + "Denoising_NN_noise_axial.png", maximum_intensity_projection_axial(NN_noise), cmap = "gray", vmax = 1.0)
    plt.imsave(path + "Denoising_NN_noise_coronal.png", maximum_intensity_projection_coronal(NN_noise), cmap="gray", vmax=1.0)
    plt.imsave(path + "Denoising_NN_noise_sagittal.png", maximum_intensity_projection_sagittal(NN_noise), cmap="gray", vmax=1.0)

    mse, ssm, psn = res_evaluation(y_true, y_pred)

    validation_MSE.append(mse)
    validation_PSNR.append(psn)
    validation_SSIM.append(ssm)


validation_MSE = np.array(validation_MSE)
validation_PSNR = np.array(validation_PSNR)
validation_SSIM = np.array(validation_SSIM)

print(np.around(validation_MSE, 6))
print(np.around(validation_PSNR, 3))
print(np.round(validation_SSIM, 4))


print(" ========================= SSIM =================================")
print("The model achieved SSIM of ", np.mean(validation_SSIM), " with standard deviation of ", np.std(validation_SSIM))


print(" ========================= PSNR =================================")
print("The model achieved PSNR of ", np.mean(validation_PSNR), " with standard deviation of ", np.std(validation_PSNR))


print(" ========================= MSE =================================")
print("The model achieved MSE of ", np.mean(validation_MSE), " with standard deviation of ", np.std(validation_MSE))

