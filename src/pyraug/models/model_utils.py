from collections import OrderedDict
import typing
from typing import Tuple, Any
from pyraug.config import BaseConfig

from pydantic.dataclasses import dataclass

@dataclass
class ModelConfig(BaseConfig):
    """This is the base configuration instance of the models
    
    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        default_encoder (bool): Whether the encoder default. Default: True.
        default_encoder (bool): Whether the encoder default. Default: True."""

    input_dim: int = None
    latent_dim: int = 10
    uses_default_encoder: bool = True
    uses_default_decoder: bool = True


class ModelOuput(OrderedDict):
    """Base ModelOuput class fixing the output type from the models. This class is inspired from 
    the ``ModelOutput`` class from hugginface transformres library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())
