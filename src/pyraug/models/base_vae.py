import os
import typing
from typing import Optional, Dict
from copy import deepcopy
from typing import Optional

import dill
import torch
import torch.nn as nn

from pyraug.models.nn import Base_Encoder, Base_Decoder
from pyraug.customexception import BadInheritanceError
from pyraug.models.model_utils import ModelOuput
from pyraug.models.model_config import ModelConfig

from pyraug.models.nn.default_architectures import Encoder_MLP, Decoder_MLP


class BaseVAE(nn.Module):
    """Base class for VAE based models
    
    Args:
        model_config (ModelConfig): An instance of ModelConfig in which any model's parameters is 
            made available.

        encoder (Base_Encoder): An instance of Base_Encoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple [Multi Layer Preception]
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (Base_Decoder): An instance of Base_decoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple [Multi Layer Preception]
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        .. note::
            If you provide your own encoder or decoder networks


    """

    def __init__(
        self,
        model_config: ModelConfig,
        encoder: Optional[Base_Encoder] = None,
        decoder: Optional[Base_Decoder] = None):

        nn.Module.__init__(self)

        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        
        self.model_config = model_config

        if encoder is None:
            encoder = Encoder_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False
        
        if decoder is None:
            decoder = Decoder_MLP(model_config)
            self.model_config.uses_default_decoder = True

        else:
            self.model_config.uses_default_decoder = False

        self.set_encoder(encoder)
        self.set_decoder(decoder)

        self.device = None

    def forward(self, inputs):
        """Main forward pass outputing the VAE outputs
        This function should output an model_output instance gathering all the model outputs
        
        Args:
            inputs (Dict[str, torch.Tensor]): The training data with labels, masks etc...
            
        Returns:
            (ModelOutput): The output of the model.
            
        .. note::
            The loss must be computed in this forward pass and accessed through 
            ``loss = model_output.loss`` """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training.

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def save(self, dir_path):
        """Method to save the model at a specific location
        
        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        model_path = dir_path

        model_dict = {
            "model_state_dict": deepcopy(self.state_dict()),
        }

        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path)

            except FileNotFoundError as e:
                raise e

        self.model_config.save_json(model_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(model_path, "encoder.pkl"), "wb") as fp:
                dill.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(model_path, "decoder.pkl"), "wb") as fp:
                dill.dump(self.decoder, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))
        

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = ModelConfig.from_json_file(path_to_model_config)

        return model_config

    @classmethod
    def _load_model_weights_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

        if "model.pt" not in file_list:
            raise FileNotFoundError(f"Missing model weights file ('model.pt') file in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_weights = os.path.join(dir_path, "model.pt")

        try:
            model_weights = torch.load(path_to_model_weights, map_location='cpu')

        except:
            RuntimeError("Enable to load model weights. Ensure they are saves in a '.pt' format.")
        
        if 'model_state_dict' not in model_weights.keys():
            raise KeyError("Model state dict is not available in 'model.pt' file. Got keys:"
                f"{model_weights.keys()}")

        model_weights = model_weights['model_state_dict']

        return model_weights

    @classmethod
    def _load_custom_encoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

        if "encoder.pkl" not in file_list:
                raise FileNotFoundError(f"Missing encoder pkl file ('encoder.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom encoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
                encoder = dill.load(fp)

        return encoder

    @classmethod
    def _load_custom_decoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)

        if "decoder.pkl" not in file_list:
                raise FileNotFoundError(f"Missing decoder pkl file ('decoder.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom decoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
                decoder = dill.load(fp)

        return decoder


    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder
        
        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:
                - a ``model_config.json`` and a ``model.pt`` if no custom architectures were
                provided

                or 
                - a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp. 
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """
        
        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)
       

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model

    def set_encoder(self, encoder: Base_Encoder) -> None:
        """Set the encoder of the model"""
        if not issubclass(type(encoder), Base_Encoder):
            raise BadInheritanceError(
                (
                    "Encoder must inherit from Base_Encoder class from "
                    "pyraug.models.base_architectures.Base_Encoder. Refer to documentation."
                )
            )
        self.encoder = encoder

    def set_decoder(self, decoder: Base_Decoder) -> None:
        """Set the decoder of the model"""
        if not issubclass(type(decoder), Base_Decoder):
            raise BadInheritanceError(
                (
                    "Decoder must inherit from Base_Decoder class from "
                    "pyraug.models.base_architectures.Base_Decoder. Refer to documentation."
                )
            )
        self.decoder = decoder


#    def build_networks(
#        self,
#        encoder: Optional[Base_Encoder] = None,
#        decoder: Optional[Base_Decoder] = None,
#    ):
#
#        assert (
#            self.model_config.input_dim is not None
#        ), "Load data before building model to adjust input dimension by calling 'trainer.get_dataloader()'"
#
#
#        model = RHVAE(self.model_config)
#
#        if encoder is not None:
#            self.model_config.encoder = f"custom ({type(encoder).__name__})"
#
#        else:
#            from pyraug.demo.default_architectures import Encoder_MLP
#
#            encoder = Encoder_MLP(self.model_config)
#            self.model_config.encoder = f"default ({type(encoder).__name__})"
#
#        model.set_encoder(encoder)
#        self.model_encoder = encoder
#
#        if decoder is not None:
#            self.model_config.decoder = f"custom ({type(decoder).__name__})"
#
#        else:
#            from pyraug.demo.default_architectures import Decoder_MLP
#
#            decoder = Decoder_MLP(self.model_config)
#            self.model_config.decoder = f"default ({type(decoder).__name__})"
#        model.set_decoder(decoder)
#        self.model_decoder = decoder
#
#        if metric is not None:
#            self.model_config.metric = f"custom ({type(metric).__name__})"
#
#        else:
#            from pyraug.demo.default_architectures import Metric_MLP
#
#            metric = Metric_MLP(self.model_config)
#            self.model_config.metric = f"default ({type(metric).__name__})"
#        model.set_metric(metric)
#        self.model_metric = metric
#
#        model.to(self.training_config.device)
#
#        self._check_model_archi(model)
#
#        if verbose:
#            logger.info("Model loaded!\n")
#
#        if verbose:
#            logger.info(
#                f"Model hyper-params:\n"
#                f" - Latent dim: {self.model_config.latent_dim}\n"
#                f" - input dim: {self.model_config.input_dim}\n"
#                f" - n_lf: {self.model_config.n_lf}\n"
#                f" - eps_lf: {self.model_config.eps_lf}\n"
#                f" - T: {self.model_config.temperature}\n"
#                f" - lbd: {self.model_config.regularization}\n"
#            )
#
#            logger.info(f"Model Architecture: {model}\n")
#        return model
#
#    def _check_model_archi(self, model: RHVAE):

#        dummy_latent_dim = self.model_config.latent_dim
#        dummy_input_dim = self.model_config.input_dim
#        dummy_batch_size = self.training_config.batch_size
#
#        dummy_input_data = torch.randn(dummy_batch_size, dummy_input_dim).to(
#            self.training_config.device
#        )
#
#        if not issubclass(type(model.encoder), Base_Encoder):# TODO (remove)
#            raise BadInheritance(
#                (
#                    "Encoder must inherit from Base_Encoder class from "
#                    "pyraug.models.base_architectures.Base_Encoder. Refer to documentation."
#                )
#            )
#
#        if not issubclass(type(model.decoder), Base_Decoder): # TODO (remove)
#            raise BadInheritance(
#                (
#                    "Decoder must inherit from Base_Decoder class from "
#                    "pyraug.models.base_architectures.Base_Decoder. Refer to documentation."
#                )
#            )
#
#        if not issubclass(type(model.metric), Base_Metric): # TODO (remove)
#            raise BadInheritance(
#                (
#                    "Metric must inherit from Base_Metric class from "
#                    "pyraug.models.base_architectures.Base_Metric. Refer to documentation."
#                )
#            )
#
#        try:
#            output = model.encoder(dummy_input_data)
#
#        except RuntimeError as e:
#            raise EncoderInputError(
#                "Enable to encode input data. Potential fixes\n"
#                " - check the encoder architecture (should take input of shape (-1, input_dim)\n"
#                "Exception catch: " + str(e)
#            ) from e
#
#        if len(output) != 2:
#            if type(output) == tuple:
#                output_shape = len(output)
#
#            else:
#                output_shape = 1
#
#            raise EncoderOutputError(
#                "Expects the encoder to output the mean and log_std of "
#                f"posterior distribution q(z|x). Got {output_shape} output tensor(s). Expected output: (mu, log_sigma)"
#            )
#
#        elif output[0].shape[-1] != self.model_config.latent_dim:
#            raise EncoderOutputError(
#                "Mean output shape differs from provided latent dim. Got mean of "
#                f" shape {output[0].shape[-1]} and latent dim of shape {self.model_config.latent_dim}"
#            )
#
#        elif output[1].shape[-1] != self.model_config.latent_dim:
#            raise EncoderOutputError(
#                "Log std output shape differs from provided latent dim. Got log_std of "
#                f" shape {output[1].shape[-1]} and latent dim of shape {self.model_config.latent_dim}"
#                "Check encoder architecture. Reminder: the variance is diagonal and so only the diag"
#                " coefficient are expected."
#            )
#
#        try:
#            output = model.decoder(output[0])
#
#        except RuntimeError as e:
#            raise DecoderInputError(
#                "Enable to decode latent data. Potential fixes\n"
#                " - check the decoder architecture (should take input of shape (-1, latent_dim)\n"
#                "Exception catch: " + str(e)
#            ) from e
#
#        if type(output) == tuple and len(output) != 1:
#            raise DecoderOutputError(
#                "Expects the decoder to output only the mean of "
#                f"condition distribution p(x|z). Got {len(output)} output tensor(s). Expected output: (mu)"
#            )
#
#        elif output[0].shape[-1] != self.model_config.input_dim:
#            raise DecoderOutputError(
#                "Mean output shape differs from data input dim. Got mean of "
#                f" shape {output[0].shape[-1]} and input dim of shape {self.model_config.input_dim}."
#                " Check decoder architecture."
#            )
#
#        try:
#            output = model.metric(dummy_input_data)
#
#        except RuntimeError as e:
#            raise MetricInputError(
#                "Enable to encode input data in metric network. Potential fixes\n"
#                " - check the metric architecture (should take input of shape (-1, input_dim)\n"
#                "Exception catch: " + str(e)
#            ) from e
#
#        if type(output) == tuple and len(output) != 1:
#            raise MetricOutputError(
#                "Expects the metric network to only output the L_{psi_i} "
#                f"matrices of metric. Got {len(output)} output tensor(s). Expected output: (L)"
#            )
#
#        elif len(output.shape) != 3:
#            raise MetricOutputError(
#                "Expects the metric network to only output the L_{psi_i} "
#                "matrices of metric. Expected output of shape: (batch_size, latent_dim, latent_dim)"
#                f" got output of shape {output.shape}"
#            )
#
#        elif (
#            output.shape[-1] != self.model_config.latent_dim
#            or output.shape[-2] != self.model_config.latent_dim
#        ):
#            raise MetricOutputError(
#                "Expects the metric network to only output the L_{psi_i} "
#                "matrices of metric. Expected output of shape: (batch_size, latent_dim, latent_dim)"
#                f" got output of shape {output.shape}"
#            )