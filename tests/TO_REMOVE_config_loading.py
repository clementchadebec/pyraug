# TODO (remove these test -> test_config.py)




import json

import pytest
from pydantic import ValidationError

from pyraug.config import BaseConfig, TrainingConfig, RHVAEConfig, RHVAEGenerationConfig
from pyraug.config_loader import ConfigParserFromJSON


# RHVAE loading tests
class Test_Load_RHVAE_Config_From_JSON: #TODO (remove)
    @pytest.fixture(
        params=[
            "tests/data/rhvae/configs/model_config00.json",
            "tests/data/rhvae/configs/training_config00.json",
            "tests/data/rhvae/configs/generation_config00.json",
        ]
    )
    def custom_config_path(self, request):
        return request.param

    @pytest.fixture
    def corrupted_config_path(self):
        return "corrupted_path"

    @pytest.fixture
    def not_json_config_path(self):
        return "tests/data/rhvae/configs/not_json_file.md"

    @pytest.fixture(
        params=[
            [
                "tests/data/rhvae/configs/model_config00.json",
                RHVAEConfig(
                    latent_dim=11,
                    n_lf=2,
                    eps_lf=0.00001,
                    temperature=0.5,
                    regularization=0.1,
                    beta_zero=0.8,
                ),
            ],
            [
                "tests/data/rhvae/configs/training_config00.json",
                TrainingConfig(
                    batch_size=3,
                    max_epochs=2,
                    learning_rate=1e-5,
                    early_stopping_epochs=10,
                ),
            ],
            [
                "tests/data/rhvae/configs/generation_config00.json",
                RHVAEConfig(
                    batch_size=3,
                    mcmc_steps_nbr=3,
                    n_lf=2,
                    eps_lf=0.003,
                    random_start=False,
                ),
            ],
        ]
    )
    def custom_config_path_with_true_config(self, request):
        return request.param

    def test_load_custom_config(self, custom_config_path_with_true_config):
        parser = ConfigParserFromJSON() # TODO (remove)

        config_path = custom_config_path_with_true_config[0]
        true_config = custom_config_path_with_true_config[1]

        if config_path == "tests/data/rhvae/configs/model_config00.json":
            parsed_config = parser.parse_model(config_path)

        elif config_path == "tests/data/rhvae/configs/training_config00.json":
            parsed_config = parser.parse_training(config_path)

        else:
            parsed_config = parser.parse_generation(config_path)

        assert parsed_config == true_config

    def test_load_dict_default_config(self, custom_config_path): # TODO (remove)
        parser = ConfigParserFromJSON()
        model_dict = parser._get_config(custom_config_path)
        assert type(model_dict) == dict

    def test_raise_load_file_not_found(self, corrupted_config_path):
        parser = ConfigParserFromJSON()
        with pytest.raises(FileNotFoundError):
            model_dict = parser._get_config(corrupted_config_path)

    def test_raise_not_json_file(self, not_json_config_path):
        parser = ConfigParserFromJSON()
        with pytest.raises(TypeError):
            model_dict = parser._get_config(not_json_config_path)


class Test_Load_Config_From_Dict:
    @pytest.fixture(params=[{"latant_dim": 10}, {"batsh_size": 1}, {"mcmc_steps": 12}])
    def corrupted_keys_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_keys_dict_config):
        parser = ConfigParserFromJSON()
        if set(corrupted_keys_dict_config.keys()).issubset(["latant_dim"]):
            with pytest.raises(TypeError):
                parser._populate_model(**corrupted_keys_dict_config)

        elif set(corrupted_keys_dict_config.keys()).issubset(["batsh_size"]):
            with pytest.raises(TypeError):
                parser._populate_training(**corrupted_keys_dict_config)

        else:
            with pytest.raises(TypeError):
                parser._populate_generation(**corrupted_keys_dict_config)

    @pytest.fixture(
        params=[
            {"latent_dim": "bad_type"},
            {"batch_size": "bad_type"},
            {"mcmc_steps_nbr": "bad_type"},
        ]
    )
    def corrupted_type_dict_config(self, request):
        return request.param

    def test_raise_type_error_corrupted_keys(self, corrupted_type_dict_config):
        parser = ConfigParserFromJSON()
        if set(corrupted_type_dict_config.keys()).issubset(["latent_dim"]):
            with pytest.raises(ValidationError):
                parser._populate_model(corrupted_type_dict_config)

        elif set(corrupted_type_dict_config.keys()).issubset(["batch_size"]):
            with pytest.raises(ValidationError):
                parser._populate_training(corrupted_type_dict_config)

        else:
            with pytest.raises(ValidationError):
                parser._populate_generation(corrupted_type_dict_config)
