import os
from copy import deepcopy

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn

from pyraug.config import ModelConfig, TrainingConfig
from pyraug.config_loader import ConfigParserFromJSON
from pyraug.demo.default_architectures import Metric_MLP
from pyraug.exception.customexception import *
from pyraug.trainers.trainers import TrainerFromJSON
from tests.data.rhvae.custom_architectures import *

#################################### Test Loading ##################################################


@pytest.fixture()
def custom_config_paths(request):
    return (
        "tests/data/rhvae/configs/model_config00.json",
        "tests/data/rhvae/configs/training_config00.json",
    )


@pytest.fixture()
def dummy_model_config():
    return ModelConfig(
        latent_dim=11,
        n_lf=2,
        eps_lf=0.00001,
        temperature=0.5,
        regularization=0.1,
        beta_zero=0.8,
    )


@pytest.fixture()
def dummy_training_config():
    return TrainingConfig(
        batch_size=3, max_epochs=2, learning_rate=1e-5, early_stopping_epochs=10
    )


@pytest.fixture(
    params=[
        np.array([[10.0, 1], [2, 0.0], [0.0, 1.0]]),
        torch.tensor([[10.0, 1, 10.0], [0.0, 2.0, 1.0]]),
        torch.load(
            "tests/data/demo_mnist_data"
        ),  # This is an extract of 3 data from MNIST (unnormalized)
    ]
)
def dummy_training_data(request):
    return request.param


class Test_Trainer_Loading:
    def test_load_config(
        self, custom_config_paths, dummy_model_config, dummy_training_config
    ):
        trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])

        assert trainer.model_config == dummy_model_config
        assert trainer.training_config == dummy_training_config

    def test_raise_missing_data_to_compute_input_dim(self, custom_config_paths):
        trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
        with pytest.raises(AssertionError):
            model = trainer.build_model()

    def test_load_data_loader(self, dummy_training_data, custom_config_paths):
        trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
        data_loader = trainer.get_dataloader(dummy_training_data)

        # check data was normalized
        assert all(data_loader.dataset.data.min(dim=1)[0] <= 0) and all(
            data_loader.dataset.data.max(dim=1)[0] <= 1
        )

        # check updated input check with data
        assert trainer.model_config.input_dim == dummy_training_data[0].shape[0]


#################################### Test Model Building ###########################################


@pytest.fixture
def trainer_from_config_with_input_dim(custom_config_paths):
    trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
    trainer.model_config.input_dim = (
        784
    )  # Simulates data loading (where the input shape is computed)
    return trainer


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


class Test_Trainer_Build_RHVAE:
    @pytest.fixture
    def default_metric(self, dummy_model_config_with_input_dim):
        return Metric_MLP(dummy_model_config_with_input_dim)

    @pytest.fixture
    def encoder_wrong_input_dim(self, dummy_model_config_with_input_dim):
        return EncoderWrongInputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def decoder_wrong_input_dim(self, dummy_model_config_with_input_dim):
        return DecoderWrongInputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def metric_wrong_input_dim(self, dummy_model_config_with_input_dim):
        return MetricWrongInputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def encoder_wrong_output(self, dummy_model_config_with_input_dim):
        return EncoderWrongOutput(dummy_model_config_with_input_dim)

    @pytest.fixture
    def decoder_wrong_output(self, dummy_model_config_with_input_dim):
        return DecoderWrongOutput(dummy_model_config_with_input_dim)

    @pytest.fixture
    def metric_wrong_output(self, dummy_model_config_with_input_dim):
        return MetricWrongOutput(dummy_model_config_with_input_dim)

    @pytest.fixture
    def encoder_wrong_output_dim(self, dummy_model_config_with_input_dim):
        return EncoderWrongOutputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def decoder_wrong_output_dim(self, dummy_model_config_with_input_dim):
        return DecoderWrongOutputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def metric_wrong_output_dim(self, dummy_model_config_with_input_dim):
        return MetricWrongOutputDim(dummy_model_config_with_input_dim)

    @pytest.fixture
    def metric_wrong_output_dim_bis(self, dummy_model_config_with_input_dim):
        return MetricWrongOutputDimBis(dummy_model_config_with_input_dim)

    @pytest.fixture
    def net_bad_inheritance(self, dummy_model_config_with_input_dim):
        return NetBadInheritance(dummy_model_config_with_input_dim)

    def test_build_architecture(
        self,
        trainer_from_config_with_input_dim,
        custom_encoder,
        custom_decoder,
        default_metric,
    ):
        model = trainer_from_config_with_input_dim.build_model(
            encoder=custom_encoder, decoder=custom_decoder
        )

        # check we savedauto encoding architectures in config
        assert (
            type(custom_encoder).__name__
            in trainer_from_config_with_input_dim.model_config.encoder
        )
        assert (
            type(custom_decoder).__name__
            in trainer_from_config_with_input_dim.model_config.decoder
        )
        assert (
            type(default_metric).__name__
            in trainer_from_config_with_input_dim.model_config.metric
        )

        # check we saved default metric in config
        assert "custom" in trainer_from_config_with_input_dim.model_config.encoder
        assert "custom" in trainer_from_config_with_input_dim.model_config.decoder
        assert "default" in trainer_from_config_with_input_dim.model_config.metric

        # check we built the right architecture
        assert type(custom_encoder).__name__ == type(model.encoder).__name__
        assert type(custom_decoder).__name__ == type(model.decoder).__name__
        assert type(default_metric).__name__ == type(model.metric).__name__

    def test_raises_cannot_encode(
        self, trainer_from_config_with_input_dim, encoder_wrong_input_dim
    ):
        with pytest.raises(EncoderInputError):
            model = trainer_from_config_with_input_dim.build_model(
                encoder=encoder_wrong_input_dim
            )

    def test_raises_cannot_decode(
        self, trainer_from_config_with_input_dim, decoder_wrong_input_dim
    ):
        with pytest.raises(DecoderInputError):
            model = trainer_from_config_with_input_dim.build_model(
                decoder=decoder_wrong_input_dim
            )

    def test_raises_cannot_decode(
        self, trainer_from_config_with_input_dim, metric_wrong_input_dim
    ):
        with pytest.raises(MetricInputError):
            model = trainer_from_config_with_input_dim.build_model(
                metric=metric_wrong_input_dim
            )

    def test_raises_wrong_encoder_output(
        self, trainer_from_config_with_input_dim, encoder_wrong_output
    ):
        with pytest.raises(EncoderOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                encoder=encoder_wrong_output
            )

    def test_raises_wrong_decoder_output(
        self, trainer_from_config_with_input_dim, decoder_wrong_output
    ):
        with pytest.raises(DecoderOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                decoder=decoder_wrong_output
            )

    def test_raises_wrong_metric_output(
        self, trainer_from_config_with_input_dim, metric_wrong_output
    ):
        with pytest.raises(MetricOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                metric=metric_wrong_output
            )

    def test_raises_wrong_encoder_output_dim(
        self, trainer_from_config_with_input_dim, encoder_wrong_output_dim
    ):
        with pytest.raises(EncoderOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                encoder=encoder_wrong_output_dim
            )

    def test_raises_wrong_decoder_output_dim(
        self, trainer_from_config_with_input_dim, decoder_wrong_output_dim
    ):
        with pytest.raises(DecoderOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                decoder=decoder_wrong_output_dim
            )

    def test_raises_wrong_metric_output_dim(
        self, trainer_from_config_with_input_dim, metric_wrong_output_dim
    ):
        with pytest.raises(MetricOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                metric=metric_wrong_output_dim
            )

    def test_raises_wrong_metric_output_dim_bis(
        self, trainer_from_config_with_input_dim, metric_wrong_output_dim_bis
    ):
        with pytest.raises(MetricOutputError):
            model = trainer_from_config_with_input_dim.build_model(
                metric=metric_wrong_output_dim_bis
            )

    def test_raises_encoder_bad_inheritance(
        self, trainer_from_config_with_input_dim, net_bad_inheritance
    ):
        with pytest.raises(BadInheritance):
            model = trainer_from_config_with_input_dim.build_model(
                encoder=net_bad_inheritance
            )

    def test_raises_encoder_bad_inheritance(
        self, trainer_from_config_with_input_dim, net_bad_inheritance
    ):
        with pytest.raises(BadInheritance):
            model = trainer_from_config_with_input_dim.build_model(
                decoder=net_bad_inheritance
            )

    def test_raises_encoder_bad_inheritance(
        self, trainer_from_config_with_input_dim, net_bad_inheritance
    ):
        with pytest.raises(BadInheritance):
            model = trainer_from_config_with_input_dim.build_model(
                metric=net_bad_inheritance
            )


#################################### Test Optimizer Building #######################################


@pytest.fixture
def trainer_from_config(custom_config_paths):
    trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
    return trainer


@pytest.fixture
def model_from_config(trainer_from_config_with_input_dim):
    model = trainer_from_config_with_input_dim.build_model()
    return model


class Test_Trainer_Build_Optim:
    def test_build_optim(self, trainer_from_config, model_from_config):
        optimizer = trainer_from_config.build_optimizer(model_from_config)
        assert (
            list(model_from_config.parameters()) == optimizer.param_groups[0]["params"]
        )


#################################### Test Training #################################################


class Test_Trainer_Training:
    def test_training(self, dummy_training_data, trainer_from_config):
        train_loader = trainer_from_config.get_dataloader(dummy_training_data)
        model = trainer_from_config.build_model()
        start_model_state_dict = deepcopy(model.state_dict())
        optimizer = trainer_from_config.build_optimizer(model)
        best_model_dict = trainer_from_config.train_model(
            train_loader=train_loader, model=model, optimizer=optimizer
        )

        # check output keys
        assert set(best_model_dict.keys()).issubset(
            set(["M", "centroids", "model_state_dict"])
        )
        assert set(start_model_state_dict.keys()).issubset(
            best_model_dict["model_state_dict"]
        )

        # check that weights were updated
        assert not torch.equal(
            best_model_dict["model_state_dict"]["metric.lower.weight"],
            start_model_state_dict["metric.lower.weight"],
        )

        # check all M and centroids were recorded
        assert best_model_dict["M"].shape[0] == dummy_training_data.shape[0]
        assert best_model_dict["centroids"].shape[0] == dummy_training_data.shape[0]

        # check M and centroids have right shape (ie latent space dim)
        assert (
            best_model_dict["M"].shape[1] == trainer_from_config.model_config.latent_dim
        )
        assert (
            best_model_dict["centroids"].shape[1]
            == trainer_from_config.model_config.latent_dim
        )


#################################### Test Saving ###################################################


@pytest.fixture
def demo_training_data():
    return torch.load(
        "tests/data/demo_mnist_data"
    )  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture


class Test_Trainer_Saving:
    @pytest.fixture
    def dummy_best_default_model_dict(self, dummy_training_data, trainer_from_config):
        train_loader = trainer_from_config.get_dataloader(dummy_training_data)
        model = trainer_from_config.build_model()
        start_model_state_dict = deepcopy(model.state_dict())
        optimizer = trainer_from_config.build_optimizer(model)
        best_model_dict = trainer_from_config.train_model(
            train_loader=train_loader, model=model, optimizer=optimizer
        )
        return best_model_dict

    @pytest.fixture
    def dummy_best_custom_model_dict(
        self, demo_training_data, trainer_from_config, custom_encoder, custom_decoder
    ):
        train_loader = trainer_from_config.get_dataloader(demo_training_data)

        model = trainer_from_config.build_model(
            encoder=custom_encoder, decoder=custom_decoder
        )

        optimizer = trainer_from_config.build_optimizer(model)
        best_model_dict = trainer_from_config.train_model(
            train_loader=train_loader, model=model, optimizer=optimizer
        )
        return best_model_dict

    def test_default_model_saving(
        self, tmpdir, trainer_from_config, dummy_best_default_model_dict
    ):
        tmpdir.mkdir("training00")
        dir_path = os.path.join(tmpdir, "training00")
        trainer_from_config.save_model(
            dir_path=dir_path, best_model_dict=dummy_best_default_model_dict
        )

        assert os.listdir(dir_path) == [
            "training_config.json",
            "model_config.json",
            "model.pt",
        ]

        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))

        # check everything has be saved

        ## check model_state_dict
        assert torch.equal(dummy_best_default_model_dict["M"], rec_model_dict["M"])
        assert torch.equal(
            dummy_best_default_model_dict["centroids"], rec_model_dict["centroids"]
        )

        assert (
            sum(
                [
                    not torch.equal(
                        rec_model_dict["model_state_dict"][key],
                        dummy_best_default_model_dict["model_state_dict"][key],
                    )
                    for key in dummy_best_default_model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )

        ## check model and training configs
        parser = ConfigParserFromJSON()
        rec_model_config = parser.parse_model(
            os.path.join(dir_path, "model_config.json")
        )
        rec_training_config = parser.parse_training(
            os.path.join(dir_path, "training_config.json")
        )

        assert rec_model_config.__dict__ == trainer_from_config.model_config.__dict__
        assert (
            rec_training_config.__dict__ == trainer_from_config.training_config.__dict__
        )

    def test_custom_model_saving(
        self, tmpdir, trainer_from_config, dummy_best_custom_model_dict
    ):
        tmpdir.mkdir("training00")
        dir_path = os.path.join(tmpdir, "training00")
        trainer_from_config.save_model(
            dir_path=dir_path, best_model_dict=dummy_best_custom_model_dict
        )

        # check if the custom encoder and decoder were saved (for future use)
        assert set(os.listdir(dir_path)) == set(
            [
                "training_config.json",
                "model_config.json",
                "model.pt",
                "decoder.pkl",
                "encoder.pkl",
            ]
        ), f"{os.listdir(dir_path)}"

        rec_model_dict = torch.load(os.path.join(dir_path, "model.pt"))

        # check everything has be saved

        ## check model_state_dict
        assert torch.equal(dummy_best_custom_model_dict["M"], rec_model_dict["M"])
        assert torch.equal(
            dummy_best_custom_model_dict["centroids"], rec_model_dict["centroids"]
        )

        assert (
            sum(
                [
                    not torch.equal(
                        rec_model_dict["model_state_dict"][key],
                        dummy_best_custom_model_dict["model_state_dict"][key],
                    )
                    for key in dummy_best_custom_model_dict["model_state_dict"].keys()
                ]
            )
            == 0
        )

        ## check model and training configs
        parser = ConfigParserFromJSON()
        rec_model_config = parser.parse_model(
            os.path.join(dir_path, "model_config.json")
        )
        rec_training_config = parser.parse_training(
            os.path.join(dir_path, "training_config.json")
        )
        assert rec_model_config.__dict__ == trainer_from_config.model_config.__dict__
        assert (
            rec_training_config.__dict__ == trainer_from_config.training_config.__dict__
        )

        ## check custom encoder and decoder
        with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
            rec_encoder = dill.load(fp)
        with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
            rec_decoder = dill.load(fp)

        assert type(rec_encoder) == type(trainer_from_config.model_encoder)
        assert type(rec_decoder) == type(trainer_from_config.model_decoder)
