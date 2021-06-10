from typing import Union, Dict, Any

import torch
import json
import os
from dataclasses import asdict
from pydantic.dataclasses import dataclass
from pydantic import ValidationError
from dataclasses import field



@dataclass
class BaseConfig:
    """This is the BaseConfig class which defines all the useful loading and saving methods
    of the configs"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Creates a :class:`~pyraug.config.BaseConfig` instance from a dictionnary
        
        Args:
            config_dict (dict): The Python dictionnary containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        try:
            config = cls(**config_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return config 

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
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

    @classmethod
    def from_json_file(cls, json_path: Union[str, os.PathLike]) -> "BaseConfig":
        """Creates a :class:`~pyraug.config.BaseConfig` instance from a JSON config file
        
        Args:
            json_path (str, os.PathLike): The path to the json file containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance   
            """
        
        config_dict = cls._dict_from_json(json_path)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Transforms object into a Python dictionnary
        
        Returns:
            (dict): The dictionnary containing all the parameters"""
        return asdict(self)

    def to_json_string(self):
        """Transforms object into a JSON str
        
        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict())

    def save_json(self, dir_path, filename):
        with open(os.path.join(dir_path, f"{filename}.json"), "w", encoding="utf-8") as fp:
            fp.write(self.to_json_string())
