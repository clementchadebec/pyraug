import logging
import os

import dill
import torch

from pyraug.config import ModelConfig
from pyraug.config_loader import ConfigParserFromJSON
from pyraug.demo.default_variables import (
    NEEDED_FILES_FOR_MODEL_LOADING, NEEDED_KEYS_FOR_RHVAE_WEIGHTS_LOADING)
from pyraug.models.vae_models import RHVAE



class ModelLoaderFromFolder:
    def __init__(self):
        self.needed_files = NEEDED_FILES_FOR_MODEL_LOADING
        pass

    def __check_files(self, model_folder: str):
        """
        Check all needed files are in folder for model loading
        """

        files_in_folder = os.listdir(model_folder)

        if not set(self.needed_files).issubset(files_in_folder):
            raise FileNotFoundError(
                "Cannot perform model rebuilding. "
                f"Missing files in folder {model_folder}. Check that all {set(self.needed_files)} are in "
                f"'{model_folder}'"
            )

    def get_config(self, model_config_file: str) -> ModelConfig:
        config_parser = ConfigParserFromJSON()
        model_config = config_parser.parse_model(model_config_file)
        return model_config

    def load_model(
        self, model_folder: str, verbose: bool = False, logger: logging.Logger = None
    ):

        self.__check_files(model_folder)

        model_config_file = os.path.join(model_folder, "model_config.json")
        model_config = self.get_config(model_config_file)

        if torch.cuda.is_available():
            model_config.device = "cuda"

        else:
            model_config.device = "cpu"

        model = RHVAE(model_config)

        if "custom" in model_config.encoder:
            assert "encoder.pkl" in os.listdir(model_folder), (
                "Did not find 'encoder.pkl' file.." f"Check '{model_folder}'"
            )
            print(os.path.join(model_folder, "encoder.pkl"))
            with open(os.path.join(model_folder, "encoder.pkl"), "rb") as fp:
                encoder = dill.load(fp)

        else:
            from pyraug.demo.default_architectures import Encoder_MLP

            encoder = Encoder_MLP(model_config)

        if "custom" in model_config.decoder:
            assert "decoder.pkl" in os.listdir(model_folder), (
                "Did not find 'encoder.pkl' file.." f"Check '{model_folder}'"
            )
            with open(os.path.join(model_folder, "decoder.pkl"), "rb") as fp:
                decoder = dill.load(fp)

        else:
            from pyraug.demo.default_architectures import Decoder_MLP

            decoder = Decoder_MLP(model_config)

        if "custom" in model_config.metric:
            assert "metric.pkl" in os.listdir(model_folder), (
                "Did not find 'encoder.pkl' file.." f"Check '{model_folder}'"
            )
            with open(os.path.join(model_folder, "metric.pkl"), "rb") as fp:
                metric = dill.load(fp)

        else:
            from pyraug.demo.default_architectures import Metric_MLP

            metric = Metric_MLP(model_config)

        model.set_encoder(encoder)
        model.set_decoder(decoder)
        model.set_metric(metric)

        weights = torch.load(os.path.join(model_folder, "model.pt"))

        model.M_tens = weights["M"]
        model.centroids_tens = weights["centroids"]
        model.G = self.__create_metric(model, device=model_config.device)
        model.G_inv = self.__create_inverse_metric(model, device=model_config.device)

        model.load_state_dict(weights["model_state_dict"])

        if verbose:
            logger.info("Model successfully loaded!\n")

        # Return the model in eval mode
        model.eval()

        return model

    def __create_metric(self, model, device="cpu"):
        def G(z):
            return torch.inverse(
                (
                    model.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (model.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + model.lbd * torch.eye(model.latent_dim).to(device)
            )

        return G

    def __create_inverse_metric(self, model, device="cpu"):
        def G_inv(z):
            return (
                model.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + model.lbd * torch.eye(model.latent_dim).to(device)

        return G_inv


class _LoadRHVAEWeightsFromDict:
    """
    This class loads the weights of the RHVAE model along with the 'M' matrices and 'centroids'
    to rebuild the metric.
    """

    def __init__(self):
        self.needed_keys = NEEDED_KEYS_FOR_RHVAE_WEIGHTS_LOADING

    def _check_dict(self, model_dict):
        """
        Check the model dict assert all needed keys to rebuild model are provided.
        """
        if not set(model_dict.keys()).issubset(set(self.needed_keys)):
            raise KeyError(
                f"Missing keys in model dict. {set(model_dict.keys())} should be in"
                f" in {set(self.needed_keys)}"
            )

    def _check_has_nets(self, model):
        """
        This function checks if a provided model has an encoder, decoder and metric.
        These nets are needed to reload the weights.
        """

        if model.encoder is None:
            raise AttributeError(
                "The model has no encoder."
                "Please provided one before loading the weights."
            )

        if model.decoder is None:
            raise AttributeError(
                "The model has no decoder."
                "Please provided one before loading the weights."
            )

        if model.metric is None:
            raise AttributeError(
                "The model has no metric."
                "Please provided one before loading the weights."
            )

    def load_weights(self, model_dict, model):
        self._check_has_nets(model)
        self._check_has_nets(model)
