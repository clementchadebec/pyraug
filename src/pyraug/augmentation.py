"""This module provides function to perform data augmentation with your own data
Your can either decide to only train a generative to generate new data with yours 
using the `train_my_model` function or only generate data from a pre-trained model
using ``generate_from_pretrained``. Finally, you can directly augment your data with the 
``augment_data`` function which 1) trains a generative model on your data and 2) generate 
the requested number of samples
"""

import datetime
import logging
import os
import shutil
import typing
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from pyraug.demo.default_variables import (PATH_TO_DEFAULT_GENERATION_CONFIG,
                                           PATH_TO_DEFAULT_MODEL_CONFIG,
                                           PATH_TO_DEFAULT_TRAINING_CONFIG)
from pyraug.generation import Generator
from pyraug.model_loader import ModelLoaderFromFolder
from pyraug.trainers.trainers import TrainerFromJSON

from .models.vae_models import RHVAE

PATH = os.path.dirname(os.path.abspath(__file__))

# logging.basicConfig(
#     level=logging.INFO,
#     format= '%(message)s',
#     datefmt='%H:%M:%S'
# )
console_logs = logging.getLogger("console")

# make it print to the console.
console = logging.StreamHandler()
console_logs.addHandler(console)
console_logs.setLevel(logging.INFO)


def augment_from_pretrained(
    number_of_samples: int,
    model: RHVAE = None,
    path_to_model_folder: str = None,
    path_to_generation_config: str = None,
    path_to_save_data: str = None,
    verbose: bool = True,
) -> torch.Tensor:
    r"""Augmentation function from a pretrained generative model. Useful is you have trained a model 
    and want to reuse it to generated new data. This function can be used with:
    * a folder containing the trained model which was created by calling for instance 
    * a RHVAE instance directly 

    Args:
        number_of_samples (int): The number of synthetic samples to generate
        
        model (RHVAE, optional): A RHVAE model instance which is used to augment the data.
            It should be a trained model you want to generate from.

        path_to_model_folder (str, optional): Path to the folder containing the trained model. The 
            folder should contain the following files: ['model.pt', 'model_config.json'] and ['encoder.pkl',
            'decoder.pkl', 'metric.pkl'] in case the model was trained with custom architectures.
            If both 'model' and 'path_to_model_folder' are provided, the function uses the model 
            by default.

        path_to_generation_config (str, optional): Path to generation config.
            See (https://arxiv.org/pdf/2105.00026.pdf) for hyper-parameters discussion. If None,
            default parameters are used. Default: None

        path_to_save_data (str, optional): Path to the folder in which the generated data will be saved.
            The data are saved in the format '.data' along with a 'generation_config.json'.
            Data are then  loadable using 'torch.load(data)'. If None, generated data are not saved.


    **Example 1**: Using a folder

    .. code-block::

        >>> train_my_model(data=my_data, path_model_folder='path/to/save/my/model')
        >>> generated_data = augment_from_pretrained(number_of_samples=10, path_to_model_folder='path/to/save/my/model')


    **Example 2**: Using a model (RHVAE)

    .. code-block::

        >>> my_model = train_my_model(data=my_data, output_model=True)
        >>> generated_data = augment_from_pretrained(number_of_samples=10, model=my_model)

    
    .. note::
        You can directly save the data in a specific location unsing the ``path_to_save_data``
        argument
    """

    signature = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")

    if model is None and path_to_model_folder is None:
        raise ValueError(
            "Provided either a model or a path to a folder to augment re-augment your"
            "data with a pre-trained model"
        )

    if path_to_generation_config is None:
        path_to_generation_config = os.path.join(
            PATH, PATH_TO_DEFAULT_GENERATION_CONFIG
        )
        if verbose:
            console_logs.info("No generation_config provided. Using default.\n")

    if model is None:
        if not os.path.isdir(path_to_model_folder):
            raise FileNotFoundError(
                f"The provided folder '{path_to_model_folder}' does not exist or "
                "is not a directory"
            )

        model_loader = ModelLoaderFromFolder()
        model = model_loader.load_model(
            path_to_model_folder, verbose=verbose, logger=console_logs
        )

    # create generator and generate data
    generator = Generator(path_to_generation_config)
    generated_data = generator.generate_data(
        model, num_samples=number_of_samples, verbose=verbose, logger=console_logs
    )

    if path_to_save_data is not None:
        # If model to save provided data are saved in the same folder as the model
        data_path = path_to_save_data
        generated_data_path = os.path.join(data_path, f"generation_{signature}")
        if not os.path.exists(generated_data_path):
            os.makedirs(generated_data_path)
            if verbose:
                console_logs.info(f"Created {generated_data_path}")

        if verbose:
            console_logs.info(f"Data will be saved in {generated_data_path}\n")

        # save data
        generator.save_data(
            generated_data_path, generated_data, verbose=verbose, logger=console_logs
        )

    return generated_data


def train_my_model(
    data: Union[torch.Tensor, np.ndarray],
    path_to_model_config: str = None,
    path_to_training_config: str = None,
    encoder: str = None,
    decoder: str = None,
    metric: str = None,
    path_to_logs: str = None,
    path_to_save_model: str = None,
    output_model=True,
    verbose: bool = True,
) -> RHVAE:
    """
    Function to train a RHVAE model. This function allows to output a RHVAE or save it in a given 
    folder.

    Args:
        data (torch.Tensor, numpy.ndarray): Shape (n_samples, x_shape, y_shape, ...) or (n_samples, -1)

        path_to_config_model (str): Path to model config. It must be a .json file containing the
            main parameters. See (https://arxiv.org/pdf/2010.11518.pdf) for hyper-parameters
            discussion. If None, default parameters are used. Default: None

        path_to_training_config (str): Path to training config. 
            It must be a .json file containing the main parameters. If None, default parameters 
            are used. Default: None

        encoder (torch.nn.Module): The custom encoder network you wish to use. If None, default architecture
            (MLP) is used.

        decoder (torch.nn.Module): The custom decoder network you wish to use. If None, default architecture
            (MLP) is used

        metric (torch.nn.Module): The custom metric network you wish to use. If None, default architecture
            (MLP) is used

        path_to_logs (str): Path to save the logs of training and generation. This is usefull to 
            follow the model training. If None, logs are not saved.  Default: None

        path_to_save_model (str): Path to the folder in which the trained model will be saved. 
            The model is saved in a folder named 'training_YYYY-MM-DD_hh-mm-ss' in the format '.pt'
            along with a 'model_config.json' and  'training_config.json' files for future use. If 
            you plan to reuse the model, specify a path. If you provide your own encoder, decoder or
            metric, they also be save in a '.pkl' format for 
            future use. If None, the model is not saved.

        output_model (bool): If True, the trained model will be output by the function. If False, this 
            function output nothing
    
        verbose (bool): verbosity in console. If True this will diplay the major steps of the
            training.

        Returns:
            model (RHVAE): Optional. The trained model if 'output_model' is True.

    .. warning::

        If you want to reuse the model do not forget to set the argument ``path_to_save_model``
        to the path where you want to store the model or set ``output_model`` to *True*
    """

    # This is used if path to logs is provided
    log_verbose = False

    # Initialize logger to None
    logger = None

    signature = str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")

    # if path to log provided create logger and save in file
    if path_to_logs is not None:
        log_dir = path_to_logs
        log_verbose = True

        # if dir does nit exists create it
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # create and set logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        f_handler = logging.FileHandler(os.path.join(log_dir, f"logs_{signature}.out"))
        f_handler.setLevel(logging.INFO)
        logger.addHandler(f_handler)

        # Do not output logs in the console
        logger.propagate = False

        if verbose:
            console_logs.info(
                f"Launching augmentation. Logs will be save in '{log_dir}'\n"
            )

    # if no model config provided. Use default.
    if path_to_model_config is None:
        path_to_model_config = os.path.join(PATH, PATH_TO_DEFAULT_MODEL_CONFIG)
        if verbose:
            console_logs.info("No training_config provided. Using default.\n")

    # if no training config provided. Use default.
    if path_to_training_config is None:
        path_to_training_config = os.path.join(PATH, PATH_TO_DEFAULT_TRAINING_CONFIG)
        if verbose:
            console_logs.info("No model_config provided. Using default.\n")

    # if no encoder provided. Use default.
    if verbose and encoder is None:
        console_logs.info("No encoder architecture provided. Using default.\n")

    # if decoder encoder provided. Use default.
    if verbose and decoder is None:
        console_logs.info("No decoder architecture provided. Using default.\n")

    # if no metric provided. Use default.
    if verbose and metric is None:
        console_logs.info("No metric architecture provided. Using default.\n")

    # Build folder where the model wil be saved
    if path_to_save_model is not None:
        recording_path = path_to_save_model
        erase_repo = False

    # creates a dummy repo to rebuild completely the model
    else:
        recording_path = "dummy_recording"
        erase_repo = True

    model_path = os.path.join(recording_path, f"training_{signature}")

    if not os.path.exists(model_path):
        try:
            os.makedirs(model_path)
            if verbose:
                console_logs.info(f"Created {model_path}")

        except FileNotFoundError as e:
            raise e
    if verbose:
        if not erase_repo:
            console_logs.info(f"Model will be saved in {model_path}\n")
        else:
            console_logs.info(
                f"Creating {model_path}. Repo will be deleted"
                " at the end of training\n"
            )

    # set trainer
    trainer = TrainerFromJSON(path_to_model_config, path_to_training_config)

    # get data loader
    train_loader = trainer.get_dataloader(data=data, verbose=log_verbose, logger=logger)

    # build model
    model = trainer.build_model(
        encoder=encoder,
        decoder=decoder,
        metric=metric,
        verbose=log_verbose,
        logger=logger,
    )

    console_logs.info(model)

    # build optimizer
    optimizer = trainer.build_optimizer(model=model, verbose=log_verbose, logger=logger)

    if verbose:
        console_logs.info("training successfully launched !")

    # train the model
    best_model_dict = trainer.train_model(
        train_loader, model, optimizer, verbose=log_verbose, logger=logger
    )

    # save model
    trainer.save_model(model_path, best_model_dict, verbose=log_verbose, logger=logger)

    if output_model:
        # reload the model completely to output trained model
        model_loader = ModelLoaderFromFolder()
        model = model_loader.load_model(model_path, verbose=False)

        if erase_repo:
            shutil.rmtree(recording_path)

        return model


def augment_data(
    data: Union[torch.Tensor, np.ndarray],
    number_of_samples: int,
    path_to_model_config: str = None,
    path_to_training_config: str = None,
    path_to_generation_config: str = None,
    encoder: str = None,
    decoder: str = None,
    metric: str = None,
    path_to_logs: str = None,
    path_to_save_model: str = None,
    path_to_save_data: str = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Advanced augmentation function. Allows to provided different model_config, training_config,
    autoencoding architectures and hyper-parameters for augmentation. 


    Args:
        data (torch.Tensor, numpy.array): Shape (n_samples, x_shape, y_shape, ...)

        number_of_samples (int): The number of synthetic samples to generate

        path_to_config_model (str, optional):Path to model config. It must be a .json file 
            containing the main parameters. See (https://arxiv.org/pdf/2010.11518.pdf) for 
            hyper-parameters discussion. If None, default parameters are used. Default: None

        path_to_training_config (str, optional): Path to training config. 
            It must be a .json file containing the main parameters. If None, default parameters
            are used. Default: None

        path_to_generation_config (str, optional): Path to model config. It must be a .json file 
            containing the main parameters. See (https://arxiv.org/pdf/2105.00026.pdf) for 
            hyper-parameters discussion. If None, default parameters are used.

        encoder (torch.nn.Module, optional): The custom encoder network you wish to use. If None, 
            default architecture (MLP) is used. Default: None

        decoder (torch.nn.Module, optional): The custom decoder network you wish to use. If None, 
            default architecture (MLP) is used. Default: None

        metric (torch.nn.Module, optional): The custom metric network you wish to use. If None,
            default architecture (MLP) is used. Default: None
        path_to_logs (str, optional): Path to save the logs of training and generation. This is usefull to 
            follow the model training. If None, logs are not saved. Default: None

        path_to_save_model (str, optional): Path to the folder in which the trained model will be saved. 
            The model is saved in a folder named 'training_YYYY-MM-DD_hh-mm-ss' in the format '.pt'
            along with a 'model_config.json' and  'training_config.json' files for future use. If 
            you plan to reuse the model, specify a path. If you provide your own encoder, decoder or
            metric, they also be save in a '.pkl' format for 
            future use. If None, the model is not saved.

        path_to_save_data (str, optional): Path to the folder in which the generated data will be saved. 
            If path_to_save model provided, the data are stored in the same folder as the model.
            The data are saved in the format '.data' along with a 'generation_config.json'.
            Data are then  loadable using 'torch.load(data)'. If None, generated data are not saved. 
            Defaults: None

        verbose (bool): verbosity., Default: False

    Returns:
        (torch.Tensor): The generated data

    .. note::
        You can directly save the data in a specific location unsing the ``path_to_save_data``
        argument

    .. warning::
        If you want to reuse the model do not forget to set the argument ``path_to_save_model``
        to the path where you want to store the model.

    """

    # check if genration_config is available.
    # this is done in train my model for model_config
    # and training_config files
    if path_to_generation_config is not None:
        _ = Generator(path_to_generation_config)

    model = train_my_model(
        data=data,
        path_to_model_config=path_to_model_config,
        path_to_training_config=path_to_training_config,
        encoder=encoder,
        decoder=decoder,
        metric=metric,
        path_to_logs=path_to_logs,
        path_to_save_model=path_to_save_model,
        output_model=True,
        verbose=verbose,
    )

    generated_data = augment_from_pretrained(
        number_of_samples=number_of_samples,
        model=model,
        path_to_model_folder=None,
        path_to_generation_config=path_to_generation_config,
        path_to_save_data=path_to_save_data,
        verbose=verbose,
    )

    # generator = Generator(path_to_generation_config)
    # generated_data = generator.generate_data(
    #    model, num_samples=number_of_samples, verbose=log_verbose, logger=logger
    # )
    #
    # if path_to_save_data is not None:
    #    # If model to save provided data are saved in the same folder as the model
    #    data_path = path_to_save_data
    #    generated_data_path = os.path.join(data_path, f"generation_{signature}")
    #    if not os.path.exists(generated_data_path):
    #        os.makedirs(generated_data_path)
    #        if verbose:
    #            console_logs.info(f"Created {generated_data_path}")
    #
    #    if verbose:
    #        console_logs.info(f"Data will be saved in {generated_data_path}\n")
    #
    #    # save data
    #    generator.save_data(
    #        generated_data_path, generated_data, verbose=log_verbose, logger=logger
    #    )
    #
    return generated_data
