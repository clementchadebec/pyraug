from pyraug.config import BaseConfig
from pydantic.dataclasses import dataclass

from typing import Union


@dataclass
class TrainingConfig(BaseConfig):
    """
    TrainingConfig is the class in which all the training arguments are stored.
    This instance is then provided to a `Trainer` instance to perform a model
    training.

    Parameters:

        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.

        batch_size (int): The number of training samples per batch. Default 50

        max_epochs (int): The maximal number of epochs for training. Default: 10000

        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-3

        train_early_stopping (int): The maximal number of epochs authorized without train loss
            improvement. Default: 50

        eval_early_stopping (int): The maximal number of epochs authorized without eval loss
            improvement. Default: 50

        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch 

        seed (int): The random seed for reprodicibility

        no_cuda (bool): Disable `cuda` training. Default: False

        verbose (bool): Allow verbosity

        
    .. tip::
        Scheduler will be added  
    """

    output_dir: str = None
    batch_size: int = 50
    max_epochs: int = 10000
    learning_rate: float = 1e-4
    train_early_stopping:Union[int, None] = 50
    eval_early_stopping: Union[int, None] = None
    steps_saving: int = 1000
    seed: int = 8
    no_cuda: bool = False
    verbose: bool = True