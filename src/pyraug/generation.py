import dataclasses
import json
import logging
import os

import torch

from pyraug.models import RHVAE
from pyraug.config import GenerationConfig, ModelConfig
from pyraug.config_loader import ConfigParserFromJSON
from pyraug.models.vae_models import RHVAE
from pyraug.sampler import hmc_manifold_sampling


class Generator:
    def __init__(
        self,
        model: ,
        generation_args:
        sampler
        )

class Generator:
    def __init__(self, generation_config_file: str):
        generation_config = self.get_config(generation_config_file)

        self.generation_config = generation_config

        # if torch.cuda.is_available():
        #    self.generation_config.device = "cuda"

    #
    # else:
    #    self.generation_config.device = "cpu"

    def save_data(
        self,
        dir_path: str,
        generated_data: torch.Tensor,
        verbose: bool = True,
        logger: logging.Logger = None,
    ) -> None:

        with open(os.path.join(dir_path, "generation_config.json"), "w") as fp:
            json.dump(dataclasses.asdict(self.generation_config), fp)

        torch.save(generated_data, os.path.join(dir_path, "generated_data.data"))

        if verbose:
            logger.info(f"Data saved in {dir_path}")

    def generate_data(
        self,
        model: RHVAE,
        num_samples: int,
        verbose: bool = False,
        logger: logging.Logger = None,
    ) -> torch.Tensor:
        model.to(self.generation_config.device)
        self.generation_config.num_samples = num_samples
        self.generation_config.batch_size = min(
            model.centroids_tens.shape[0], self.generation_config.batch_size
        )
        full_batch_nbr = int(
            self.generation_config.num_samples / self.generation_config.batch_size
        )
        last_batch_samples_nbr = int(
            self.generation_config.num_samples % self.generation_config.batch_size
        )

        if verbose:
            logger.info("Launching generation !\n")

        generated_samples = []
        for i in range(full_batch_nbr):
            samples = hmc_manifold_sampling(
                model,
                latent_dim=model.latent_dim,
                n_samples=self.generation_config.batch_size,
                step_nbr=self.generation_config.mcmc_steps_nbr,
                n_lf=self.generation_config.n_lf,
                eps_lf=self.generation_config.eps_lf,
                device=self.generation_config.device,
                random_start=self.generation_config.random_start,
                verbose=False,
            )

            x_gen = model.decoder(z=samples).detach()
            generated_samples.append(x_gen)

        if last_batch_samples_nbr > 0:
            samples = hmc_manifold_sampling(
                model,
                latent_dim=model.latent_dim,
                n_samples=last_batch_samples_nbr,
                step_nbr=self.generation_config.mcmc_steps_nbr,
                n_lf=self.generation_config.n_lf,
                eps_lf=self.generation_config.eps_lf,
                device=self.generation_config.device,
                random_start=self.generation_config.random_start,
                verbose=False,
            )

            x_gen = model.decoder(z=samples).detach()
            generated_samples.append(x_gen)

        return torch.cat(generated_samples)

    def get_config(self, generation_config_file: str) -> GenerationConfig:
        config_parser = ConfigParserFromJSON()
        generation_config = config_parser.parse_generation(generation_config_file)
        return generation_config
