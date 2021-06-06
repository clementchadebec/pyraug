from typing import Union

import torch
from pydantic.dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 180
    max_epochs: int = 10000
    learning_rate: float = 1e-3
    early_stopping_epochs: int = 50
    no_cuda: bool = False
    verbose: bool = True
    device: str = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"


@dataclass
class ModelConfig:
    input_dim: int = None
    latent_dim: int = 10
    n_lf: int = 3
    eps_lf: float = 0.001
    beta_zero: float = 0.3
    temperature: float = 1.5
    regularization: float = 0.01
    encoder: str = "default"
    decoder: str = "default"
    metric: str = "default"
    no_cuda: bool = False
    device: str = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"


@dataclass
class GenerationConfig:
    num_samples: int = None
    batch_size: int = 50
    # use_classic_gen: bool = None
    # generation_type: str = None
    mcmc_steps_nbr: int = 100
    n_lf: int = 15
    eps_lf: int = 0.03
    verbose: bool = False
    random_start: bool = False
    no_cuda: bool = False
    device: str = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
