from .base_pipeline import Pipeline
from pyraug.models import BaseVAE, BaseSampler


import typing
from typing import Union

from pyraug.models.model_config import SamplerConfig
from pyraug.models.rhvae import RHVAESamplerConfig
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler

class GenerationPipeline(Pipeline):
    """
    This pipelines allows to generate new samples from a pre-trained model
    
    Parameters:
        model (BaseVAE): The model
        sampler (BaseSampler): The sampler to use to sampler from the model

        .. warning::
            You must ensure that the sampler used handled the model provided
    """

    def __init__(
        self,
        model: BaseVAE,
        sampler: BaseSampler=None):
    

            self.model = model

            if sampler is None:
                sampler = RHVAESampler(model=model, sampler_config=RHVAESamplerConfig())

            self.sampler = sampler
    
    def __call__(self, samples_number):

        self.sampler.sample(samples_number)

            