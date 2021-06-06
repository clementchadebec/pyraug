import typing

import torch
from models.config import ModelConfig
from models.vae_models import RHVAE
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, digits, labels, binarize=False):

        self.labels = labels

        if binarize:
            self.data = (torch.rand_like(digits) < digits).type(torch.float)

        else:
            self.data = digits.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]

        # Load data and get label
        # X = torch.load('data/' + DATA + '.pt')
        y = self.labels[index]

        return X, y
