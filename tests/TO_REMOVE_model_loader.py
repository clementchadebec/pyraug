import os

import pytest
import torch

from pyraug.config import ModelConfig
from pyraug.demo.default_architectures import Metric_MLP
from pyraug.model_loader import (ModelLoaderFromFolder,
                                 _LoadRHVAEWeightsFromDict)
from pyraug.models.vae_models import RHVAE
from pyraug.trainers.trainers import TrainerFromJSON
from tests.data.rhvae.custom_architectures import Decoder_Conv, Encoder_Conv


@pytest.fixture
def demo_training_data():
    return torch.load(
        "tests/data/demo_mnist_data"
    )  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture


@pytest.fixture()
def custom_config_paths():
    return (
        "tests/data/rhvae/configs/model_config00.json",
        "tests/data/rhvae/configs/training_config00.json",
    )


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


@pytest.fixture
def trainer_from_config(custom_config_paths):
    trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
    return trainer


class Test_RHVAE_Weigths_Loading_From_Dict:
    @pytest.fixture
    def corrupted_model_dict(self):
        return {"missing keys": 0.0}

    @pytest.fixture(
        params=[
            {"encoder": "void_encoder", "decoder": "void_decoder"},
            {"metric": "void_metric", "decoder": "void_decoder"},
            {"metric": "void_metric", "encoder": "void_encoder"},
        ]
    )
    def dummy_autoencoding_config(self, request):
        return request.param

    def test_raises_missing_keys(self, corrupted_model_dict):
        weights_loader = _LoadRHVAEWeightsFromDict()
        with pytest.raises(KeyError):
            weights_loader._check_dict(corrupted_model_dict)

    def test_raises_missing_nets(
        self, dummy_model_config_with_input_dim, dummy_autoencoding_config
    ):
        model = RHVAE(dummy_model_config_with_input_dim)

        if "encoder" in dummy_autoencoding_config.keys():
            model.set_encoder(dummy_autoencoding_config["encoder"])
        if "decoder" in dummy_autoencoding_config.keys():
            model.set_decoder(dummy_autoencoding_config["decoder"])
        if "metric" in dummy_autoencoding_config.keys():
            model.set_metric(dummy_autoencoding_config["metric"])

        weights_loader = _LoadRHVAEWeightsFromDict()
        with pytest.raises(AttributeError):
            weights_loader._check_has_nets(model)


class Test_Model_Loading_From_Folder:
    @pytest.fixture
    def corrupted_folder(self, tmpdir):
        tmpdir.mkdir("training00")
        dir_path = os.path.join(tmpdir, "training00")
        return dir_path

    def test_reload_default_model_after_training(
        self, tmpdir, demo_training_data, trainer_from_config
    ):

        tmpdir.mkdir("training00")
        dir_path = os.path.join(tmpdir, "training00")

        # make training
        train_loader = trainer_from_config.get_dataloader(demo_training_data)
        model = trainer_from_config.build_model()
        optimizer = trainer_from_config.build_optimizer(model)
        best_model_dict = trainer_from_config.train_model(
            train_loader=train_loader, model=model, optimizer=optimizer
        )

        trainer_from_config.save_model(
            dir_path=dir_path, best_model_dict=best_model_dict
        )

        model_loader = ModelLoaderFromFolder()
        rec_model = model_loader.load_model(dir_path)

        # check state dict is loaded
        assert (
            sum(
                [
                    not torch.equal(
                        rec_model.state_dict()[key],
                        best_model_dict["model_state_dict"][key],
                    )
                    for key in best_model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )
        assert torch.equal(rec_model.M_tens, best_model_dict["M"])
        assert torch.equal(rec_model.centroids_tens, best_model_dict["centroids"])

        # check loaded default nets architectures
        assert type(rec_model.encoder).__name__ == "Encoder_MLP"
        assert type(rec_model.decoder).__name__ == "Decoder_MLP"
        assert type(rec_model.metric).__name__ == "Metric_MLP"

        # Assert set model in evl mode by default
        assert not rec_model.training

    def test_reload_custom_model_after_training(
        self,
        tmpdir,
        demo_training_data,
        trainer_from_config,
        custom_encoder,
        custom_decoder,
    ):

        tmpdir.mkdir("training00")
        dir_path = os.path.join(tmpdir, "training00")

        # make training
        train_loader = trainer_from_config.get_dataloader(demo_training_data)
        model = trainer_from_config.build_model(
            encoder=custom_encoder, decoder=custom_decoder
        )
        optimizer = trainer_from_config.build_optimizer(model)
        best_model_dict = trainer_from_config.train_model(
            train_loader=train_loader, model=model, optimizer=optimizer
        )

        trainer_from_config.save_model(
            dir_path=dir_path, best_model_dict=best_model_dict
        )

        model_loader = ModelLoaderFromFolder()
        rec_model = model_loader.load_model(dir_path)

        # check state dict is loaded
        assert (
            sum(
                [
                    not torch.equal(
                        rec_model.state_dict()[key],
                        best_model_dict["model_state_dict"][key],
                    )
                    for key in best_model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )

        assert torch.equal(rec_model.M_tens, best_model_dict["M"])
        assert torch.equal(rec_model.centroids_tens, best_model_dict["centroids"])

        # check loaded custom nets architectures
        assert type(rec_model.encoder).__name__ == type(custom_encoder).__name__
        assert type(rec_model.decoder).__name__ == type(custom_decoder).__name__
        assert type(rec_model.metric).__name__ == "Metric_MLP"

    def test_raise_missing_files(self, corrupted_folder):

        model_loader = ModelLoaderFromFolder()
        with pytest.raises(FileNotFoundError):
            model_loader.load_model(corrupted_folder)
