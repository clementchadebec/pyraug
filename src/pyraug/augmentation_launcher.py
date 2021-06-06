import argparse
import datetime
import importlib.util
import inspect
import logging
import os

import torch

from pyraug.demo.default_variables import PATH_TO_DEFAULT_MODEL_CONFIG
from pyraug.generation import Generator
from pyraug.model_loader import ModelLoaderFromConfigDict
from pyraug.trainers.trainers import TrainerFromJSON

# Get the top-level logger object
console_logs = logging.getLogger()

# make it print to the console.
console = logging.StreamHandler()
console_logs.addHandler(console)
console_logs.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))
RUNNING_PATH = os.getcwd()

ap = argparse.ArgumentParser()


# Training setting
ap.add_argument(
    "--path_to_data",
    type=str,
    default=None,
    help="path to the data set to augment",
    required=True,
)
ap.add_argument(
    "--number_of_samples", type=int, help="number of samples to generate", required=True
)
ap.add_argument(
    "--path_to_model_config",
    help="path to model config file (expected json file)",
    default=os.path.join(PATH, "demo/configs_files/model_config.json"),
)
ap.add_argument(
    "--path_to_custom_archi",
    help="path to the file containing the custom encoder, decoder or metric networks architectures."
    "Must be provided as 'path/to/custom_architectures'. The modules must be python class the name "
    "of which starts with 'Encoder' for encoder, 'Decoder' for decoder or 'Metric' for metric "
    "(ex: class Encoder_Custom(nn.Module)). If not None, default architectures wil be used.",
    default=None,
)
ap.add_argument(
    "--path_to_training_config",
    help="path_to_model_config_file (expected json file)",
    default=os.path.join(PATH, "demo/configs_files/training_config.json"),
)
ap.add_argument(
    "--path_to_logs",
    help="specific folder save to log files",
    default=os.path.join("outputs/"),
)
ap.add_argument(
    "--path_to_save_models",
    help="folder where the model will saved. If None, the model is not saved",
    default=None,
)
ap.add_argument(
    "--path_to_save_data",
    help="folder where the generated will be saved. If None, the data are stored in './generated_data'",
    default=os.path.join("generated_data/"),
)
ap.add_argument(
    "--no_verbose",
    action="store_true",
    default=False,
    help="no verbosity (default: False",
)


args = ap.parse_args()

args.verbose = not args.no_verbose


def main(args):
    log_dir = args.path_to_logs

    args.signature = (
        str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.verbose:
        console_logs.info(f"Launching training. Logs will be save in '{log_dir}'\n")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(os.path.join(log_dir, f"logs_{args.signature}.out"))
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    if not args.verbose:
        logger.propagate = False

    if args.path_to_save_models is not None:
        # DIRECTORY FOR SAVING
        recording_path = args.path_to_save_models
        model_path = os.path.join(recording_path, f"training_{args.signature}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            if args.verbose:
                logger.info(f"Created {model_path}")
        if args.verbose:
            logger.info(f"Model will be saved in {model_path}\n")

    else:
        model_path = None

    if args.path_to_save_data is not None:
        # DIRECTORY FOR SAVING
        generated_data_path = args.path_to_save_data
        generated_dir = os.path.join(
            generated_data_path, f"generated_data_{args.signature}"
        )
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)
            if args.verbose:
                logger.info(f"Created {generated_dir}")
        if args.verbose:
            logger.info(f"Generated data will be saved in {generated_dir}\n")

    if args.path_to_custom_archi is not None:
        if args.verbose:
            logger.info("Loading custom architectures")

        if not os.path.isfile(args.path_to_custom_archi):
            raise FileNotFoundError(
                "File to custom architecture not found. Please check path "
                f"'{args.path_to_custom_archi}'. Must be 'path/to/custom_architectures.py."
            )

        if args.path_to_custom_archi.split(".")[-1] != "py":
            raise ModuleNotFoundError(
                f"Expected '.py' file got '.{args.path_to_custom_archi.split('.')[-1]}'. "
                f"Check path '{args.path_to_custom_archi}'"
            )

        module_name = args.path_to_custom_archi.split("/")[-1].split(".")[0]
        spec = importlib.util.spec_from_file_location(
            module_name, args.path_to_custom_archi
        )
        archis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archis)

    trainer = TrainerFromJSON(args.path_to_model_config, args.path_to_training_config)
    data = trainer.get_data(args.path_to_data, verbose=True, logger=logger)

    encoder = None
    decoder = None
    metric = None

    for name, obj in inspect.getmembers(archis):
        if "Encoder" in name:
            # encoder = archis.obj
            encoder = obj(trainer.model_config)
            if args.verbose:
                logger.info("Built custom encoder")

        elif "Decoder" in name:
            decoder = obj(trainer.model_config)
            if args.verbose:
                logger.info("Built custom decoder")

        elif "Metric" in name:
            metric = obj(trainer.model_config)
            if args.verbose:
                logger.info("Built custom metric")

    train_loader = trainer.get_dataloader(data=data, verbose=True, logger=logger)
    model = trainer.build_model(
        encoder=encoder, decoder=decoder, metric=metric, verbose=True, logger=logger
    )
    optimizer = trainer.build_optimizer(model=model, verbose=True, logger=logger)
    best_model_dict = trainer.train_model(
        train_loader, model, optimizer, verbose=True, logger=logger
    )

    if model_path is not None:
        trainer.save_model(model_path, best_model_dict, verbose=True, logger=logger)

    generation_config = trainer.generation_config
    model_config = trainer.model_config

    model = ModelLoaderFromConfigDict(model_config, best_model_dict).load_model(
        logger=logger
    )
    # print(model)
    generator = Generator(generation_config)
    generated_data = generator.generate_data(model, verbose=True, logger=logger)
    generator.save_data(generated_dir, generated_data, verbose=True, logger=logger)


if __name__ == "__main__":
    main(args)
