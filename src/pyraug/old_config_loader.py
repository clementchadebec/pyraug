import json
import logging
import logging.config
from abc import ABC, abstractmethod
from typing import Tuple, Union

from pyraug.config import GenerationConfig, ModelConfig, TrainingConfig
from pyraug.demo.default_variables import (GENERATION_KEYS, MODEL_KEYS,
                                           TRAINING_KEYS)
from pyraug.exception.customexception import SizeMismatchError


class ConfigParserFromJSON:
    def __init__(self):
        pass

    def __get_config(self, json_path: str) -> dict:
        # logger.info('Reading json file')
        try:
            with open(json_path) as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except TypeError as e:
                    raise e(f"File {json_path} not loadable. Maybe not json ?")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )
        # logger.info('Loaded config !')

    def __check_dict(
        self,
        dict_to_check: dict,
        true_dict: dict,
        location: str = "",
        filename: str = "",
    ):
        for key in dict_to_check.keys():
            if key not in true_dict.keys():
                raise KeyError(
                    f"'{key}' key must be in {set(true_dict.keys())} in {location} ."
                    f"Check json file ({filename})."
                )
            if type(dict_to_check[key]) not in true_dict[key]:
                raise TypeError(
                    f"'Wrong type provided for '{key}' in {location}. "
                    f"Got {type(dict_to_check[key])}, expected {set(true_dict[key])}. "
                    f"Check json file ({filename})."
                )

    def __check_model_config(self, model_dict_config: dict, filename: str) -> None:
        self.__check_dict(model_dict_config, MODEL_KEYS, filename=filename)

    def __check_generation_config(
        self, generation_dict_config: dict, filename: str
    ) -> None:

        self.__check_dict(generation_dict_config, GENERATION_KEYS, filename=filename)

    def __check_training_config(
        self, training_dict_config: dict, filename: str
    ) -> None:
        self.__check_dict(training_dict_config, TRAINING_KEYS, filename=filename)

    def __populate_model(self, model_dict: dict) -> ModelConfig:
        model_config = ModelConfig(
            latent_dim=model_dict["latent_dim"],
            n_lf=model_dict["n_lf"],
            eps_lf=model_dict["eps_lf"],
            temperature=model_dict["temperature"],
            regularization=model_dict["regularization"],
            beta_zero=model_dict["beta_zero"],
            # encoder=model_dict["networks"]["encoder"],
            # decoder=model_dict["networks"]["decoder"],
            # metric=model_dict["networks"]["metric"],
        )

        return model_config

    def __populate_pretrained_model(self, model_dict: dict) -> ModelConfig:
        model_config = ModelConfig(
            input_dim=model_dict["input_dim"],
            latent_dim=model_dict["latent_dim"],
            n_lf=model_dict["n_lf"],
            eps_lf=model_dict["eps_lf"],
            temperature=model_dict["temperature"],
            regularization=model_dict["regularization"],
            beta_zero=model_dict["beta_zero"],
            encoder=model_dict["encoder"],
            decoder=model_dict["decoder"],
            metric=model_dict["metric"],
            device=model_dict["device"],
        )

        return model_config

    def __populate_generation(self, generation_dict: dict) -> GenerationConfig:
        generation_config = GenerationConfig(
            batch_size=generation_dict["batch_size"],
            mcmc_steps_nbr=generation_dict["mcmc_steps_nbr"],
            n_lf=generation_dict["n_lf"],
            eps_lf=generation_dict["eps_lf"],
        )
        return generation_config

    def __populate_training(self, training_dict: dict) -> TrainingConfig:
        training_config = TrainingConfig(
            batch_size=training_dict["batch_size"],
            max_epochs=training_dict["max_epochs"],
            learning_rate=training_dict["learning_rate"],
            early_stopping_epochs=training_dict["early_stopping"],
        )
        return training_config

    def parse_model(self, json_path: str, pre_trained=False) -> ModelConfig:
        model_dict = self.__get_config(json_path)
        self.__check_model_config(model_dict, filename=json_path)
        if pre_trained:
            model_config = self.__populate_pretrained_model(model_dict)

        else:
            model_config = self.__populate_model(model_dict)

        return model_config

    def parse_generation(self, json_path: str) -> GenerationConfig:
        generation_dict = self.__get_config(json_path)
        self.__check_generation_config(generation_dict, filename=json_path)
        generation_config = self.__populate_generation(generation_dict)
        return generation_config

    def parse_training(self, json_path: str) -> TrainingConfig:
        training_dict = self.__get_config(json_path)
        self.__check_training_config(training_dict, filename=json_path)
        training_config = self.__populate_training(training_dict)
        return training_config
