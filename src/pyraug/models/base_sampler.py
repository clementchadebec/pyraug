import typing
import os
import logging
import torch
from pyraug.models.model_config import SamplerConfig
from pyraug.models import BaseVAE


logger = logging.getLogger(__name__)

class BaseSampler:
    """Base class for sampler used to generate from the VAEs models
    
    Args:
        model (BaseVAE): The vae model to sample from.
        sampler_config (SamplerConfig): An instance of SamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None
    """

    def __init__(
        self,
        model: BaseVAE,
        sampler_config: SamplerConfig = None
    ):

        if sampler_config.output_dir is None:
            output_dir = 'dummy_output_dir'
            sampler_config.output_dir = output_dir

        if not os.path.exists(sampler_config.output_dir):
            os.makedirs(sampler_config.output_dir)
            logger.info(f"Created {sampler_config.output_dir} folder since did not exist. "
                "Generated data will be saved here.")
        
        self.model = model
        self.sampler_config = sampler_config

        self.samples_number = sampler_config.samples_number
        self.batch_size = sampler_config.batch_size
        self.samples_per_save = self.sampler_config.samples_per_save

        self.device = "cuda" if torch.cuda.is_available() and not sampler_config.no_cuda else 'cpu'

        full_batch_nbr = int(self.sampler_config.samples_number / self.sampler_config.batch_size)
        last_batch_samples_nbr = self.sampler_config.samples_number % self.sampler_config.batch_size

        if last_batch_samples_nbr == 0:
            batch_number = full_batch_nbr

        else:
            batch_number = full_batch_nbr + 1
        
        self.batch_number = batch_number
        self.full_batch_nbr = full_batch_nbr
        self.last_batch_samples_nbr= last_batch_samples_nbr

    def sample(self):
        """Main sampling function of the samplers

        Retruns:
            (torch.Tensor): The generated data

        """
        raise not NotImplementedError()


    def save(self, dir_path):
        """Method to save the sampler config. The config is saved a as ``sampler_config.json`` 
        file in ``dir_path``"""

        self.sampler_config.save_json(dir_path, "sampler_config")
        

    def save_data_batch(self, data, dir_path, number_of_samples, batch_idx):
        """
        Method to save a batch of generated data. The data will be saved in the 
        ``dir_path`` folder. The batch of data
        is saved in a file named ``generated_data_{number_of_samples}_{batch_idx}.pt``

        Args:
            data (torch.Tensor): The data to save
            dir_path (str): The folder where the data and config file must be saved
            batch_idx (int): The batch idx

        .. note::
            You can then easily reload the generated data using

            .. code-block:

                >>> import torch
                >>> import os
                >>> data = torch.load(
                    os.path.join(
                        'dir_path', 'generated_data_{number_of_samples}_{batch_idx}.pt'))
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        torch.save(
            data, os.path.join(dir_path, f'generated_data_{number_of_samples}_{batch_idx}.pt'))