import os
from copy import deepcopy

import dill
import pytest
import torch

from pyraug.augmentation import train_my_model
from pyraug.config import ModelConfig
from pyraug.config_loader import ConfigParserFromJSON
from tests.data.rhvae.custom_architectures import Decoder_Conv, Encoder_Conv


@pytest.fixture()
def custom_config_paths(request):
    return (
        "tests/data/rhvae/configs/model_config00.json",
        "tests/data/rhvae/configs/training_config00.json",
    )


@pytest.fixture
def demo_data():
    return torch.load(
        "tests/data/demo_mnist_data"
    )  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture


@pytest.fixture()
def dummy_model_config_with_input_dim():
    return ModelConfig(
        input_dim=784,  # Simulates data loading (where the input shape is computed). This needed to
        # create dummy custom encoders and decoders
        latent_dim=11,
        n_lf=12,
        eps_lf=0.00001,
        temperature=15,
        regularization=10,
        beta_zero=0.8,
    )


@pytest.fixture
def custom_encoder(dummy_model_config_with_input_dim):
    return Encoder_Conv(dummy_model_config_with_input_dim)


@pytest.fixture
def custom_decoder(dummy_model_config_with_input_dim):
    return Decoder_Conv(dummy_model_config_with_input_dim)


class Test_Model_Saving:
    def test_save_default_model(self, tmpdir, demo_data, custom_config_paths):
        model = train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            output_model=True,
        )

        dir_path = os.path.join(tmpdir, "dummy_saving")

        # save model
        model.save(path_to_save_model=dir_path)

        model_dict = {
            "M": deepcopy(model.M_tens),
            "centroids": deepcopy(model.centroids_tens),
            "model_state_dict": deepcopy(model.state_dict()),
        }

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt"]
        ), f"{os.listdir(dir_path)}"

        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))

        ## check model_state_dict
        assert torch.equal(model_dict["M"], rec_model_dict["M"])
        assert torch.equal(model_dict["centroids"], rec_model_dict["centroids"])

        assert (
            sum(
                [
                    not torch.equal(
                        rec_model_dict["model_state_dict"][key],
                        model_dict["model_state_dict"][key],
                    )
                    for key in model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )

        ## check model and training configs
        parser = ConfigParserFromJSON()
        rec_model_config = parser.parse_model(
            os.path.join(dir_path, "model_config.json")
        )

        assert rec_model_config.__dict__ == model.model_config.__dict__

    def test_save_default_model(
        self, tmpdir, demo_data, custom_encoder, custom_decoder, custom_config_paths
    ):
        model = train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            output_model=True,
            encoder=custom_encoder,
            decoder=custom_decoder,
        )

        dir_path = os.path.join(tmpdir, "dummy_saving")

        # save model
        model.save(path_to_save_model=dir_path)

        model_dict = {
            "M": deepcopy(model.M_tens),
            "centroids": deepcopy(model.centroids_tens),
            "model_state_dict": deepcopy(model.state_dict()),
        }

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "decoder.pkl", "encoder.pkl"]
        ), f"{os.listdir(dir_path)}"

        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))

        ## check model_state_dict
        assert torch.equal(model_dict["M"], rec_model_dict["M"])
        assert torch.equal(model_dict["centroids"], rec_model_dict["centroids"])

        assert (
            sum(
                [
                    not torch.equal(
                        rec_model_dict["model_state_dict"][key],
                        model_dict["model_state_dict"][key],
                    )
                    for key in model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )

        ## check model and training configs
        parser = ConfigParserFromJSON()
        rec_model_config = parser.parse_model(
            os.path.join(dir_path, "model_config.json")
        )

        assert rec_model_config.__dict__ == model.model_config.__dict__

        ## check custom encoder and decoder
        with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
            rec_encoder = dill.load(fp)
        with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
            rec_decoder = dill.load(fp)

        assert type(rec_encoder) == type(model.encoder)
        assert type(rec_decoder) == type(model.decoder)
