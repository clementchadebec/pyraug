
import os
import typing
from typing import Any, Optional, Tuple, Union, Dict, Any
import logging
from copy import deepcopy
import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from pyraug.customexception import *



from torch.utils.data import Dataset
from pyraug.models import BaseVAE
from pyraug.trainers.trainer_utils import set_seed
from pyraug.trainers.training_config import TrainingConfig
from pyraug.customexception import ModelError



logger = logging.getLogger(__name__)

class Trainer:
    """Trainer is the main class to perform model training.

    Args:
        model (BaseVAE): The model to train

        train_dataset (Dataset): The training dataset if type :class:`~torch.utils.data.Dataset`

        training_args (TrainingConfig): The training arguments summarizing the main parameters used
            for training. If None, a basic training instance of :class:`TrainingConfig` is used.
            Default: None.
         
        optimizer (torch.optim.Optimizer): An instance of `torch.optim.Optimizer` used for
            training. If nothing is provided, a basic ``torch.optim.Adam`` is used.
    """

    def __init__(
        self,
        model: BaseVAE,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset]=None,
        training_config: Optional[TrainingConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
    
        if training_config is None:
            training_config = TrainingConfig()

        if training_config.output_dir is None:
            output_dir = 'dummy_output_dir'
            training_config.output_dir = output_dir

        if not os.path.exists(training_config.output_dir):
            os.makedirs(training_config.output_dir)
            logger.info(f"Created {training_config.output_dir} folder since did not exist.")

            

        
        self.training_config = training_config

        set_seed(self.training_config.seed)

        device = "cuda" if torch.cuda.is_available() and not training_args.no_cuda else 'cpu'

        # place model on device
        model = model.to(device)

        # set optimizer
        if optimizer is None:
            optimizer = self.set_default_optimizer(model)


        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.model = model
        self.optimizer = optimizer

        self.device = device

        # set early stopping flags
        self._set_earlystopping_flags(train_dataset, eval_dataset, training_config)        

        # Define the loaders
        train_loader = self.get_train_dataloader(train_dataset)

        if eval_dataset is not None:
            eval_loader = self.get_eval_dataloader(eval_dataset)

        else:
            eval_loader = None


        self.train_loader = train_loader
        self.eval_loader = eval_loader

    
    def get_train_dataloader(
        self, train_dataset: Dataset
    ) -> torch.utils.data.DataLoader:

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True
        )

    def get_eval_dataloader(
        self, eval_dataset: Dataset) -> torch.utils.data.DataLoader:
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False
        )
        
    def set_default_optimizer(
        self,
        model: BaseVAE
    ) -> torch.optim.Optimizer:

        optimizer = optim.Adam(
            model.parameters(), lr=self.training_config.learning_rate
        )

        return optimizer

    def _run_model_sanity_check(self, model, train_dataset):
        try:
            model(train_dataset[:2])

        except Exception as e:
            raise ModelError("Error when calling forward method from model. Potential issues: \n"
                " - Wrong model architecture -> check encoder, decoder and metric architecture if "
                "you provide yours \n"
                " - The data input dimension provided is wrong -> when no encoder, decoder or metric "
                "provided, a network is built automatically but requires the shape of the flatten " 
                "input data.\n"
                f"Exception raised: {type(e)} with message: " + str(e)) from e
#
    def _set_earlystopping_flags(self, train_dataset, eval_dataset, training_config):

        # Initialize early_stopping flags
        self.make_eval_early_stopping = False
        self.make_train_early_stopping = False

        if training_config.train_early_stopping is not None:
            self.make_train_early_stopping = True

        # Check if eval_dataset is provided
        if eval_dataset is not None and training_config.eval_early_stopping is not None:
            self.make_eval_early_stopping = True
            
            # By default we make the early stopping on evaluation dataset
            self.make_train_early_stopping = False


    def _set_inputs_to_device(self, inputs: Dict[str, Any]):
        
        inputs_on_device = inputs

        if self.device == 'cuda':
                cuda_inputs = dict.fromkeys(inputs)
                for key in inputs.keys():
                    if torch.is_tensor(inputs[key]):
                        cuda_inputs[key] = inputs[keys].cuda()

                    else:
                        cuda_inputs = inputs[keys]

                inputs_on_device = cuda_inputs

        return inputs_on_device
        
    def train(self, log_output_dir: str = None
    ):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        # run sanity check on the model
        self._run_model_sanity_check(self.model, self.train_dataset)

        self._training_signature = str(
            datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")

        training_dir = os.path.join(
            self.training_config.output_dir, f"training_{self._training_signature}"
        )

        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            logger.info(f"Created {training_dir}."
                "Training config, checkpoints and final model will be saved here.")

        log_verbose = False

        # set up log file
        if log_output_dir is not None:
            log_dir = log_output_dir
            log_verbose = True

            # if dir does not exist create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                logger.info(f"Created {log_dir} folder since did not exists. Training logs will be "
                    "recodered here")

            # create and set logger
            file_logger = logging.getLogger("log_dir")
            file_logger.setLevel(logging.INFO)
            f_handler = logging.FileHandler(os.path.join(log_dir, f"training_logs.log"))
            f_handler.setLevel(logging.INFO)
            file_logger.addHandler(f_handler)

            # Do not output logs in the console
            file_logger.propagate = False


        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        epoch_es_train = 0
        epoch_es_eval = 0

        for epoch in range(1, self.training_config.max_epochs):

            epoch_train_loss = self.train_step()
            

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step()


            # early stopping
            if self.make_eval_early_stopping:

                if epoch_eval_loss < best_eval_loss:
                    epoch_es_eval = 0
                    best_eval_loss = epoch_eval_loss

                else:
                    epoch_es_eval += 1

                    if epoch_es_eval > self.training_config.eval_early_stopping and log_verbose:
                        logger.info(f"Training ended at epoch {epoch}! "
                            f" Eval loss did not improve for {epoch_es_eval} epochs.")
                        file_logger.info(f"Training ended at epoch {epoch}! "
                            f" Eval loss did not improve for {epoch_es_eval} epochs.")

                        break

            elif self.make_train_early_stopping:

                if epoch_train_loss < best_train_loss:
                    epoch_es_train = 0
                    best_train_loss = epoch_train_loss

                else:
                    epoch_es_train += 1

                    if epoch_es_train > self.training_config.train_early_stopping and log_verbose:
                        logger.info(f"Training ended at epoch {epoch}! "
                            f" Train loss did not improve for {epoch_es_train} epochs.")
                        file_logger.info(f"Training ended at epoch {epoch}! "
                            f" Train loss did not improve for {epoch_es_train} epochs.")

                        break
                
            # save checkpoints
            if epoch % self.training_config.steps_saving == 0:
                self.save_checkpoint(dir_path=training_dir, epoch=epoch)

            if log_verbose and epoch % 1 == 0:
                file_logger.info(self.make_eval_early_stopping)
                if self.eval_dataset is not None:
                    if self.make_eval_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                            f"- Eval Early Stopping: {epoch_es_eval}/{self.training_config.eval_early_stopping}"
                            f" (Best: {best_eval_loss:.2f})\n"
                        )
                    elif self.make_train_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                            f"- Train Early Stopping: {epoch_es_train}/{self.training_config.train_early_stopping}"
                            f" (Best: {best_train_loss:.2f})\n"
                        )

                    else:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Current Eval loss: {epoch_eval_loss:.2f}\n"
                        )
                else:
                    if self.make_train_early_stopping:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                            f"- Train Early Stopping: {epoch_es_train}/{self.training_config.train_early_stopping}"
                            f" (Best: {best_train_loss:.2f})\n"
                        )

                    else:
                        file_logger.info(
                            f"Epoch {epoch} / {self.training_config.max_epochs}\n"
                            f"- Current Train loss: {epoch_train_loss:.2f}\n"
                        )

        final_dir = os.path.join(training_dir, 'final_model')

        self.save_model(dir_path=final_dir)
        logger.info("----------------------------------")
        logger.info("Training ended!")
        logger.info(f"Saved final model in {final_dir}")


    def eval_step(self):
        """Perform an evaluation step

        Retruns:
            (torch.Tensor): The evaluation loss
        """

        self.model.eval()

        epoch_loss = 0


        for (batch_idx, inputs) in enumerate(self.eval_loader):
            
            inputs = self._set_inputs_to_device(inputs)

            model_output = self.model(inputs)

            loss = model_output.loss

            epoch_loss += loss.item()

        epoch_loss /= len(self.eval_loader)

        return epoch_loss


    def train_step(self):
        """The trainer performs training loop over the train_loader.
        
        Retruns:
            (torch.Tensor): The step training loss 
        """
        # set model in train model
        self.model.train()

        epoch_loss = 0

        for (batch_idx, inputs) in enumerate(self.train_loader):
            
            inputs = self._set_inputs_to_device(inputs)

            self.optimizer.zero_grad()

            model_output = self.model(inputs)

            loss = model_output.loss

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()


        # Allows model updates if needed
        self.model.update()

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    def save_model(self, dir_path):
        """This method saves the final model along with the config files
        
        Args:
            dir_path (str): The folder where the model and config files should be saved
        """

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # save model
        self.model.save(dir_path)

        # save training config
        self.training_config.save_json(dir_path, "training_config")


    def save_checkpoint(self, dir_path, epoch: int):
        """Saves a checkpoint alowing to restart training from here
        
        Args:
            dir_path (str): The folder where the checkpoint should be saved
            epochs_signature (int): The epoch number"""

        checkpoint_dir = os.path.join(dir_path,
            f"checkpoint_epoch_{epoch}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # save optimizer
        torch.save(deepcopy(self.optimizer.state_dict()), os.path.join(checkpoint_dir, 'optimizer.pt'))

        # save model
        self.model.save(checkpoint_dir)


    

    #def _load_checkpoint(self, checkpoint_dir):
#
    #    if not os.path.isdir(checkpoint_dir):
    #        raise FileNotFoundError("Checkpoint directory not found or is not a directory. Check"
    #            f" path: '{checkpoint_dir}'")
#
    #    files_list = os.listdir(checkpoint_dir)
#
    #    if not 'model.pt' in files_list:
    #        raise FileNotFoundError("Cannot retrieve model weights! Check that 'model.pt' is in"
    #            f"{checkpoint_dir}" )
#
    #    if not 'model.pt' in files_list:
    #        raise FileNotFoundError("Cannot retrieve optimizer state! Check that 'optimizer.pt' is"
    #            f" in {checkpoint_dir}" )
#
    #    model_weights = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
#
#

#class TrainerFromJSON:
#    def __init__(self, model_config_file: str, training_config_file: str):
#
#        model_config, training_config = self.get_config(
#            model_config_file, training_config_file
#        )
#        self.model_config = model_config
#        self.training_config = training_config
#
#        # if torch.cuda.is_available():
#        #    self.training_config.device = "cuda"
#        #    self.model_config.device = "cuda"
#
#    #
#    # else:
#    #    self.training_config.device = "cpu"
#    #    self.model_config.device = "cpu"
#
#    def build_model(
#        self,
#        encoder: Optional[nn.Module] = None,
#        decoder: Optional[nn.Module] = None,
#        metric: Optional[nn.Module] = None,
#        verbose: bool = False,
#        logger: logging.Logger = None,
#    ) -> BaseVAE(model_config):
#
#        assert (
#            self.model_config.input_dim is not None
#        ), "Load data before building model to adjust input dimension by calling 'trainer.get_dataloader()'"
#
#        if verbose:
#            logger.info("Loading model")
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
#
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
#
#    def build_optimizer(
#        self, model: RHVAE, verbose: bool = False, logger=None
#    ) -> torch.optim.Optimizer:
#        optimizer = optim.Adam(
#            model.parameters(), lr=self.training_config.learning_rate
#        )
#        if verbose:
#            logger.info(f"Optimizer \n --> {optimizer}\n")
#        return optimizer
#
#    def save_model( # call model.save, config.save
#        self,
#        dir_path: str,
#        best_model_dict: dict,
#        verbose: bool = False,
#        logger: logging.Logger = None,
#    ) -> None:
#
#        with open(os.path.join(dir_path, "model_config.json"), "w") as fp:
#            json.dump(dataclasses.asdict(self.model_config), fp)
#
#        with open(os.path.join(dir_path, "training_config.json"), "w") as fp:
#            json.dump(dataclasses.asdict(self.training_config), fp)
#
#        # only save .pkl if custom architecture provided
#        if "custom" in self.model_config.encoder:
#            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
#                dill.dump(self.model_encoder, fp)
#
#        if "custom" in self.model_config.decoder:
#            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
#                dill.dump(self.model_decoder, fp)
#
#        if "custom" in self.model_config.metric:
#            with open(os.path.join(dir_path, "metric.pkl"), "wb") as fp:
#                dill.dump(self.model_metric, fp)
#
#        torch.save(best_model_dict, os.path.join(dir_path, "model.pt"))
#
#        if verbose:
#            logger.info(f"Model saved in {dir_path}")
#
#    def get_dataloader(
#        self, data: torch.Tensor, verbose: bool = False, logger: logging.Logger = None
#    ) -> torch.utils.data.DataLoader:
#
#        data_checker = DataChecker() # In preproces
#        clean_data = data_checker.check_data(data) # In preproces
#
#        dummy_targets = torch.ones(clean_data.shape)
#        dataset = Dataset(clean_data, dummy_targets)
#        loader = DataLoader(
#            dataset=dataset, batch_size=self.training_config.batch_size, shuffle=True
#        )
#
#        self.model_config.input_dim = loader.dataset.data.reshape(
#            loader.dataset.data.shape[0], -1
#        ).shape[-1]
#        if verbose:
#            logger.info("Data loaded !\n")
#
#        return loader
#
#    def train_model(
#        self,
#        train_loader: torch.utils.data.DataLoader,
#        model: RHVAE,
#        optimizer: torch.optim.Optimizer,
#        verbose: bool = False,
#        logger: logging.Logger = None,
#    ) -> dict:
#
#        # if verbose:
#        #    logger.info("Setting device\n")
#        # if torch.cuda.is_available():
#        #    if verbose:
#        #        logger.info("Using cuda !\n")
#        #    self.training_config.device = "cuda"
#        #    self.model_config.device = "cuda"
#        #
#        # else:
#        #    if verbose:
#        #        logger.info("Using cpu !\n")
#        #    self.training_config.device = "cpu"
#        #    self.model_config.device = "cpu"
#
#        if verbose:
#            logger.info(
#                f"Training params:\n - max_epochs: {self.training_config.max_epochs}\n"
#                f" - es: {self.training_config.early_stopping_epochs}\n"
#                f" - batch_size: {self.training_config.batch_size}\n"
#            )
#
#            # logger.info(f'Model Architecture: {model}\n')
#        best_model_dict = training_loops(
#            self.model_config,
#            self.training_config,
#            train_loader,
#            model,
#            optimizer,
#            logger=logger,
#        )
#
#        return best_model_dict
#
#    def get_config(
#        self, model_config_file: str, training_config_file: str
#    ) -> None:
#        config_parser = ConfigParserFromJSON()
#        model_config = config_parser.parse_model(model_config_file)
#        training_config = config_parser.parse_training(training_config_file)
#        return model_config, training_config
#