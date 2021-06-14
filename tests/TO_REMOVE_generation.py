import os

import pytest
import torch

from pyraug.config import GenerationConfig
from pyraug.generation import Generator
from pyraug.model_loader import ModelLoaderFromFolder
from pyraug.trainers.trainers import TrainerFromJSON

#################################### Test Loading ##################################################


@pytest.fixture
def custom_generation_config_path():
    return "tests/data/rhvae/configs/generation_config00.json"


@pytest.fixture
def dummy_generation_config():
    return GenerationConfig(
        batch_size=3, mcmc_steps_nbr=3, n_lf=2, eps_lf=0.003, random_start=False
    )


class Test_Generator_Loading:
    def test_load_config(self, custom_generation_config_path, dummy_generation_config):
        generator = Generator(custom_generation_config_path)
        assert generator.generation_config == dummy_generation_config


#################################### Test Generation ###############################################


@pytest.fixture
def custom_config_paths():
    return (
        "tests/data/rhvae/configs/model_config00.json",
        "tests/data/rhvae/configs/training_config00.json",
    )


@pytest.fixture
def demo_training_data():
    return torch.load(
        "tests/data/demo_mnist_data"
    )  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture


@pytest.fixture
def trainer_from_config(custom_config_paths):
    trainer = TrainerFromJSON(custom_config_paths[0], custom_config_paths[1])
    return trainer


@pytest.fixture
def generator_from_config(custom_generation_config_path):
    generator = Generator(custom_generation_config_path)
    return generator


def make_training(tmpdir, demo_training_data, trainer_from_config):

    tmpdir.mkdir("training00")
    dir_path = os.path.join(tmpdir, "training00")

    # make training
    train_loader = trainer_from_config.get_dataloader(demo_training_data)
    model = trainer_from_config.build_model()
    optimizer = trainer_from_config.build_optimizer(model)
    best_model_dict = trainer_from_config.train_model(
        train_loader=train_loader, model=model, optimizer=optimizer
    )

    # save model
    trainer_from_config.save_model(dir_path=dir_path, best_model_dict=best_model_dict)

    model_loader = ModelLoaderFromFolder()
    rec_model = model_loader.load_model(dir_path)

    return rec_model


class Test_RHVAE_Generator:
    @pytest.fixture(params=[1.0, 5.0, 10])
    def dummy_number_of_samples(sef, request):
        return request.param

    @pytest.mark.filterwarnings("ignore")
    def test_data_generation(
        self,
        tmpdir,
        demo_training_data,
        trainer_from_config,
        generator_from_config,
        dummy_number_of_samples,
    ):
        model = make_training(tmpdir, demo_training_data, trainer_from_config)
        gen_data = generator_from_config.generate_data(model, dummy_number_of_samples)

        # assert number of generated data equals number of samples requested
        assert gen_data.shape[0] == dummy_number_of_samples
