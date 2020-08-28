import os

import numpy as np
import torch
import imageio

from torch.utils.data import Dataset

from tqdm import tqdm

from utils import Timer


class Mirflickr25kDataset(Dataset):
    def __init__(self, data_root='data'):
        with Timer('Loading npy into Mirflickr25kDataset'):
            self.texts = np.load(f'{data_root}/texts.npy')
            self.labels = np.load(f'{data_root}/labels.npy')
            self.images = np.load(f'{data_root}/images.npy', allow_pickle=True)

        # print(self.texts.shape)
        # print(self.labels.shape)
        # print(self.images.shape)
        # print(self.images[0], self.images[0].shape)

