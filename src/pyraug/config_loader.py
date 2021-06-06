import json
import logging
import logging.config
from typing import Tuple, Union

from pydantic import ValidationError

from pyraug.config import GenerationConfig, ModelConfig, TrainingConfig
from pyraug.exception.customexception import SizeMismatchError


class ConfigParserFromJSON:
    def __init__(self):
        pass

    def _get_config(self, json_path: str) -> dict:
        # logger.info('Reading json file')
        try:
            with open(json_path) as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except (TypeError, json.JSONDecodeError) as e:
                    raise TypeError(f"File {json_path} not loadable. Maybe not json ?")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )

    def _populate_model(self, model_config) -> ModelConfig:
        try:
            model_config = ModelConfig(**model_config)
        except (ValidationError, TypeError) as e:
            raise e
        return model_config

    def _populate_generation(self, generation_dict: dict) -> GenerationConfig:
        try:
            generation_config = GenerationConfig(**generation_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return generation_config

    def _populate_training(self, training_dict: dict) -> TrainingConfig:
        try:
            training_config = TrainingConfig(**training_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return training_config

    def parse_model(self, json_path: str) -> ModelConfig:
        model_dict = self._get_config(json_path)
        model_config = self._populate_model(model_dict)
        return model_config

    def parse_generation(self, json_path: str) -> GenerationConfig:
        generation_dict = self._get_config(json_path)
        generation_config = self._populate_generation(generation_dict)
        return generation_config

    def parse_training(self, json_path: str) -> TrainingConfig:
        training_dict = self._get_config(json_path)
        training_config = self._populate_training(training_dict)
        return training_config
