""" Here will be implemented the main Variational Autoencoders architecture you
may use to perform data augmentation
"""

from .rhvae.rhvae_model import RHVAE
from .base.base_vae import BaseVAE

__all__ = [
    "BaseVae",
    "RHVAE"
]
