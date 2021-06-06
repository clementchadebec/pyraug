import dataclasses
import json
import logging
import os
import typing
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import dill
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyraug.config import ModelConfig, TrainingConfig
from pyraug.config_loader import ConfigParserFromJSON
from pyraug.data_loader import DataChecker, DataGetter, Dataset
from pyraug.exception.customexception import *
from pyraug.make_training import training_loops
from pyraug.models.base_architectures import *
from pyraug.models.vae_models import RHVAE


class TrainerFromJSON:
    def __init__(self, model_config_file: str, training_config_file: str):

        model_config, training_config = self.get_config(
            model_config_file, training_config_file
        )
        self.model_config = model_config
        self.training_config = training_config

        # if torch.cuda.is_available():
        #    self.training_config.device = "cuda"
        #    self.model_config.device = "cuda"

    #
    # else:
    #    self.training_config.device = "cpu"
    #    self.model_config.device = "cpu"

    def build_model(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        metric: Optional[nn.Module] = None,
        verbose: bool = False,
        logger: logging.Logger = None,
    ) -> RHVAE:

        assert (
            self.model_config.input_dim is not None
        ), "Load data before building model to adjust input dimension by calling 'trainer.get_dataloader()'"

        if verbose:
            logger.info("Loading model")
        model = RHVAE(self.model_config)

        if encoder is not None:
            self.model_config.encoder = f"custom ({type(encoder).__name__})"

        else:
            from pyraug.demo.default_architectures import Encoder_MLP

            encoder = Encoder_MLP(self.model_config)
            self.model_config.encoder = f"default ({type(encoder).__name__})"

        model.set_encoder(encoder)
        self.model_encoder = encoder

        if decoder is not None:
            self.model_config.decoder = f"custom ({type(decoder).__name__})"

        else:
            from pyraug.demo.default_architectures import Decoder_MLP

            decoder = Decoder_MLP(self.model_config)
            self.model_config.decoder = f"default ({type(decoder).__name__})"
        model.set_decoder(decoder)
        self.model_decoder = decoder

        if metric is not None:
            self.model_config.metric = f"custom ({type(metric).__name__})"

        else:
            from pyraug.demo.default_architectures import Metric_MLP

            metric = Metric_MLP(self.model_config)
            self.model_config.metric = f"default ({type(metric).__name__})"
        model.set_metric(metric)
        self.model_metric = metric

        model.to(self.training_config.device)

        self._check_model_archi(model)

        if verbose:
            logger.info("Model loaded!\n")

        if verbose:
            logger.info(
                f"Model hyper-params:\n"
                f" - Latent dim: {self.model_config.latent_dim}\n"
                f" - input dim: {self.model_config.input_dim}\n"
                f" - n_lf: {self.model_config.n_lf}\n"
                f" - eps_lf: {self.model_config.eps_lf}\n"
                f" - T: {self.model_config.temperature}\n"
                f" - lbd: {self.model_config.regularization}\n"
            )

            logger.info(f"Model Architecture: {model}\n")
        return model

    def _check_model_archi(self, model: RHVAE):

        dummy_latent_dim = self.model_config.latent_dim
        dummy_input_dim = self.model_config.input_dim
        dummy_batch_size = self.training_config.batch_size

        dummy_input_data = torch.randn(dummy_batch_size, dummy_input_dim).to(
            self.training_config.device
        )

        if not issubclass(type(model.encoder), Base_Encoder):
            raise BadInheritance(
                (
                    "Encoder must inherit from Base_Encoder class from "
                    "pyraug.models.base_architectures.Base_Encoder. Refer to documentation."
                )
            )

        if not issubclass(type(model.decoder), Base_Decoder):
            raise BadInheritance(
                (
                    "Decoder must inherit from Base_Decoder class from "
                    "pyraug.models.base_architectures.Base_Decoder. Refer to documentation."
                )
            )

        if not issubclass(type(model.metric), Base_Metric):
            raise BadInheritance(
                (
                    "Metric must inherit from Base_Metric class from "
                    "pyraug.models.base_architectures.Base_Metric. Refer to documentation."
                )
            )

        try:
            output = model.encoder(dummy_input_data)

        except RuntimeError as e:
            raise EncoderInputError(
                "Enable to encode input data. Potential fixes\n"
                " - check the encoder architecture (should take input of shape (-1, input_dim)\n"
                "Exception catch: " + str(e)
            ) from e

        if len(output) != 2:
            if type(output) == tuple:
                output_shape = len(output)

            else:
                output_shape = 1

            raise EncoderOutputError(
                "Expects the encoder to output the mean and log_std of "
                f"posterior distribution q(z|x). Got {output_shape} output tensor(s). Expected output: (mu, log_sigma)"
            )

        elif output[0].shape[-1] != self.model_config.latent_dim:
            raise EncoderOutputError(
                "Mean output shape differs from provided latent dim. Got mean of "
                f" shape {output[0].shape[-1]} and latent dim of shape {self.model_config.latent_dim}"
            )

        elif output[1].shape[-1] != self.model_config.latent_dim:
            raise EncoderOutputError(
                "Log std output shape differs from provided latent dim. Got log_std of "
                f" shape {output[1].shape[-1]} and latent dim of shape {self.model_config.latent_dim}"
                "Check encoder architecture. Reminder: the variance is diagonal and so only the diag"
                " coefficient are expected."
            )

        try:
            output = model.decoder(output[0])

        except RuntimeError as e:
            raise DecoderInputError(
                "Enable to decode latent data. Potential fixes\n"
                " - check the decoder architecture (should take input of shape (-1, latent_dim)\n"
                "Exception catch: " + str(e)
            ) from e

        if type(output) == tuple and len(output) != 1:
            raise DecoderOutputError(
                "Expects the decoder to output only the mean of "
                f"condition distribution p(x|z). Got {len(output)} output tensor(s). Expected output: (mu)"
            )

        elif output[0].shape[-1] != self.model_config.input_dim:
            raise DecoderOutputError(
                "Mean output shape differs from data input dim. Got mean of "
                f" shape {output[0].shape[-1]} and input dim of shape {self.model_config.input_dim}."
                " Check decoder architecture."
            )

        try:
            output = model.metric(dummy_input_data)

        except RuntimeError as e:
            raise MetricInputError(
                "Enable to encode input data in metric network. Potential fixes\n"
                " - check the metric architecture (should take input of shape (-1, input_dim)\n"
                "Exception catch: " + str(e)
            ) from e

        if type(output) == tuple and len(output) != 1:
            raise MetricOutputError(
                "Expects the metric network to only output the L_{psi_i} "
                f"matrices of metric. Got {len(output)} output tensor(s). Expected output: (L)"
            )

        elif len(output.shape) != 3:
            raise MetricOutputError(
                "Expects the metric network to only output the L_{psi_i} "
                "matrices of metric. Expected output of shape: (batch_size, latent_dim, latent_dim)"
                f" got output of shape {output.shape}"
            )

        elif (
            output.shape[-1] != self.model_config.latent_dim
            or output.shape[-2] != self.model_config.latent_dim
        ):
            raise MetricOutputError(
                "Expects the metric network to only output the L_{psi_i} "
                "matrices of metric. Expected output of shape: (batch_size, latent_dim, latent_dim)"
                f" got output of shape {output.shape}"
            )

    def build_optimizer(
        self, model: RHVAE, verbose: bool = False, logger=None
    ) -> torch.optim.Optimizer:
        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )
        if verbose:
            logger.info(f"Optimizer \n --> {optimizer}\n")
        return optimizer

    def save_model( # call model.save, config.save
        self,
        dir_path: str,
        best_model_dict: dict,
        verbose: bool = False,
        logger: logging.Logger = None,
    ) -> None:

        with open(os.path.join(dir_path, "model_config.json"), "w") as fp:
            json.dump(dataclasses.asdict(self.model_config), fp)

        with open(os.path.join(dir_path, "training_config.json"), "w") as fp:
            json.dump(dataclasses.asdict(self.training_config), fp)

        # only save .pkl if custom architecture provided
        if "custom" in self.model_config.encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                dill.dump(self.model_encoder, fp)

        if "custom" in self.model_config.decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                dill.dump(self.model_decoder, fp)

        if "custom" in self.model_config.metric:
            with open(os.path.join(dir_path, "metric.pkl"), "wb") as fp:
                dill.dump(self.model_metric, fp)

        torch.save(best_model_dict, os.path.join(dir_path, "model.pt"))

        if verbose:
            logger.info(f"Model saved in {dir_path}")

    def get_dataloader(
        self, data: torch.Tensor, verbose: bool = False, logger: logging.Logger = None
    ) -> torch.utils.data.DataLoader:

        data_checker = DataChecker() # In preproces
        clean_data = data_checker.check_data(data) # In preproces

        dummy_targets = torch.ones(clean_data.shape)
        dataset = Dataset(clean_data, dummy_targets)
        loader = DataLoader(
            dataset=dataset, batch_size=self.training_config.batch_size, shuffle=True
        )

        self.model_config.input_dim = loader.dataset.data.reshape(
            loader.dataset.data.shape[0], -1
        ).shape[-1]
        if verbose:
            logger.info("Data loaded !\n")

        return loader

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        model: RHVAE,
        optimizer: torch.optim.Optimizer,
        verbose: bool = False,
        logger: logging.Logger = None,
    ) -> dict:

        # if verbose:
        #    logger.info("Setting device\n")
        # if torch.cuda.is_available():
        #    if verbose:
        #        logger.info("Using cuda !\n")
        #    self.training_config.device = "cuda"
        #    self.model_config.device = "cuda"
        #
        # else:
        #    if verbose:
        #        logger.info("Using cpu !\n")
        #    self.training_config.device = "cpu"
        #    self.model_config.device = "cpu"

        if verbose:
            logger.info(
                f"Training params:\n - max_epochs: {self.training_config.max_epochs}\n"
                f" - es: {self.training_config.early_stopping_epochs}\n"
                f" - batch_size: {self.training_config.batch_size}\n"
            )

            # logger.info(f'Model Architecture: {model}\n')
        best_model_dict = training_loops(
            self.model_config,
            self.training_config,
            train_loader,
            model,
            optimizer,
            logger=logger,
        )

        return best_model_dict

    def get_config(
        self, model_config_file: str, training_config_file: str
    ) -> Tuple[ModelConfig, TrainingConfig]:
        config_parser = ConfigParserFromJSON()
        model_config = config_parser.parse_model(model_config_file)
        training_config = config_parser.parse_training(training_config_file)
        return model_config, training_config
