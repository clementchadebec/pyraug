from .base_pipeline import Pipeline
from pyraug.models import BaseVAE, BaseSampler


import typing
from typing import Union

from pyraug.models.model_config import SamplerConfig

class GenerationPipeline(Pipeline):
    """
    This pipelines allows to generate new samples from a pre-trained model
    
    Parameters:
        model (BaseVAE): The model itself of a path to the model folder
        sampler (BaseSampler): The sampler to use to sampler from the model

        .. warning::
            You must ensure that the sampler used handled the model provided
    """

    def __init__(
        self,
        model: BaseVAE,
        sampler: BaseSampler):
    

            self.model = model
            self.sampler = sampler
    
    def __call__(self, samples_number):

        self.sampler.sample(samples_number)

            