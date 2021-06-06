import logging

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


class Data_Checker(object):
    def __init__(self):
        pass

    def check_type(self, database):
        """
        Checks if database is of type torch.tensor
        """

        if torch.is_tensor(database):
            pass

        elif isinstance(database, np.ndarray):

            database = torch.tensor

        else:
            raise TypeError(
                f"Database should be torch.tensor or ndarray, object of type {type(database)} provided."
            )

    def __check_0_1(self, database):
        pass
