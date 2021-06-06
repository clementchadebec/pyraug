import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils import data

from pyraug.exception.customexception import LoadError

PATH = os.path.dirname(os.path.abspath(__file__))
DEMOS = ["mnist"]


class Checker(ABC):
    def __init__(self):
        pass


class DataChecker:
    def __init__(self):
        pass

    def check_data(self, data):
        if not torch.is_tensor(data):
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    " Data must be either of type "
                    f"< 'torch.Tensor' > or < 'np.ndarray' > ({type(data)} provided). "
                    f" Check data"
                )

            else:
                try:
                    data = torch.tensor(data).type(torch.float)

                except TypeError as e:
                    raise TypeError(
                        str(e.args) + ". Potential issues:\n"
                        "- input data has not the same shape in array\n"
                        "- input data with unhandable type"
                    ) from e

        # Detect potential nan
        if (data != data).sum() > 0:
            raise ValueError("Nan detected in input data!")

        elif data.min() < 0 or data.max() > 1:
            data = self.normalize_data(data.type(torch.float))

        return data.reshape(data.shape[0], -1)

    def normalize_data(self, data):

        data_reshaped = data.reshape(data.shape[0], -1)
        clean_data = (
            data_reshaped - data_reshaped.min(dim=-1).values.unsqueeze(-1)
        ) / (
            data_reshaped.max(dim=-1).values.unsqueeze(-1)
            - data_reshaped.min(dim=-1).values.unsqueeze(-1)
        )

        return clean_data.reshape_as(data)


class DataGetter:
    def __init__(self):
        pass

    def get_data(
        self, path_to_data: str, verbose: bool = False, logger: logging.Logger = None
    ) -> torch.Tensor:

        if verbose:
            logger.info(f"Fetching data from '{path_to_data}'")

        data_checker = DataGetter()
        data = data_checker.load_data(path_to_data)

        return data

    def load_data(self, path_to_data: str) -> torch.Tensor:
        try:
            data = torch.load(path_to_data)
        except:
            raise LoadError(
                "data must be leadable using torch.load(). "
                f"Check data path ({path_to_data})"
            )

        return data


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
