import numpy as np
import glob
import random
import math
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader, Dataset

Train_path = sorted(glob.glob("Data/Training/*.npy"))
valid_path = sorted(glob.glob("Data/Validation/*.npy"))

def normalize_data(data):
    data -= np.min(data)
    data /= np.max(data)
    return data.astype(np.float32)


def loading_data(paths):
    input_data_tensor = []
    output_data_tensor = []

    def patch_splitting(nd_image, nf_image):

        pti = [0, 64, 128, 192, 256, 320, 384]
        ptf = [128, 192, 256, 320, 384, 448, 512]

        for i in range(7):
            for j in range(7):
                for k in range(7):
                    nimg = nd_image[pti[i]:ptf[i], pti[j]:ptf[j], pti[k]:ptf[k]]
                    nfimg = nf_image[pti[i]:ptf[i], pti[j]:ptf[j], pti[k]:ptf[k]]

                    nimg = np.expand_dims(nimg, axis=0)
                    nfimg = np.expand_dims(nfimg, axis=0)

                    input_data_tensor.append(nimg)
                    output_data_tensor.append(nfimg)
                    

    for p in paths:
        print(p)
        data = np.load(p).astype(np.float32)
        noised_image = data[0]
        noise_free_image = data[1]
        noise = data[2]  # if you need the extracted noise

        patch_splitting(noised_image, noise_free_image)

    input_data_tensor = np.asarray(input_data_tensor).astype(np.float32)
    output_data_tensor = np.asarray(output_data_tensor).astype(np.float32)

    return input_data_tensor, output_data_tensor


class TrainDataset(Dataset):
    def __init__(self):
        data = loading_data(Train_path)
        self.free_image = torch.from_numpy(data[1])
        self.n_image = torch.from_numpy(data[0])
        self.n_samples = self.n_image.shape[0]
        print(self.n_image.shape, self.free_image.shape)

    def __getitem__(self, index):
        return self.n_image[index], self.free_image[index]

    def __len__(self):
        return self.n_samples


class ValidDataset(Dataset):
    def __init__(self):
        data = loading_data(valid_path)
        self.free_image = torch.from_numpy(data[1])
        self.n_image = torch.from_numpy(data[0])
        self.n_samples = self.n_image.shape[0]
        print(self.n_image.shape, self.free_image.shape)

    def __getitem__(self, index):
        return self.n_image[index], self.free_image[index]

    def __len__(self):
        return self.n_samples
