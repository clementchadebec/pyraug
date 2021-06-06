import os
import shutil

import numpy as np
import pytest
import torch

from pyraug.augmentation import (augment_data, augment_from_pretrained,
                                 train_my_model)
from pyraug.config import ModelConfig
from pyraug.config_loader import ConfigParserFromJSON
from pyraug.models.vae_models import RHVAE
from tests.data.rhvae.custom_architectures import Decoder_Conv, Encoder_Conv


@pytest.fixture()
def custom_config_paths(request):
    return (
        "tests/data/rhvae/configs/model_config00.json",
        "tests/data/rhvae/configs/training_config00.json",
        "tests/data/rhvae/configs/generation_config00.json",
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


@pytest.fixture
def custom_dummy_folder(tmpdir):
    return os.path.join(tmpdir, "dummy_folder")


@pytest.fixture
def corrupted_config_path():
    return "corrupted_file"


@pytest.fixture
def not_json_file():
    return "tests/data/rhvae/configs/not_json_file.md"


#################################### Test Loading ##################################################


class Test_Main_Augmentation:
    @pytest.mark.filterwarnings("ignore")
    def test_creates_custom_logs_files(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        pass
        augment_data(
            data=demo_data,
            number_of_samples=4,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_generation_config=custom_config_paths[2],
            path_to_logs=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check log file is created
        assert len(os.listdir(custom_dummy_folder)) == 1

    @pytest.mark.filterwarnings("ignore")
    def test_creates_model_files(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        pass
        augment_data(
            data=demo_data,
            number_of_samples=4,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_generation_config=custom_config_paths[2],
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check training_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        training_folder = os.listdir(custom_dummy_folder)[0]
        training_folder_path = os.path.join(custom_dummy_folder, training_folder)
        training_folder_files = os.listdir(training_folder_path)

        # check saved the training_config_file
        assert "training_config.json" in training_folder_files
        assert "model_config.json" in training_folder_files
        assert "model.pt" in training_folder_files

    @pytest.mark.filterwarnings("ignore")
    def test_creates_generation_files(
        self,
        demo_data,
        custom_config_paths,
        custom_dummy_folder,
        custom_encoder,
        custom_decoder,
    ):

        augment_data(
            demo_data,
            number_of_samples=4,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_generation_config=custom_config_paths[2],
            encoder=custom_encoder,
            decoder=custom_decoder,
            path_to_save_data=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check generation_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        generation_folder = os.listdir(custom_dummy_folder)[0]
        generation_folder_path = os.path.join(custom_dummy_folder, generation_folder)
        generation_folder_files = os.listdir(generation_folder_path)

        # check saved the files
        assert "generated_data.data" in generation_folder_files
        assert "generation_config.json" in generation_folder_files

    @pytest.mark.filterwarnings("ignore")
    def test_creates_model_files_with_custom_archi(
        self,
        demo_data,
        custom_config_paths,
        custom_dummy_folder,
        custom_encoder,
        custom_decoder,
    ):

        augment_data(
            demo_data,
            number_of_samples=4,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_generation_config=custom_config_paths[2],
            encoder=custom_encoder,
            decoder=custom_decoder,
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check training_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        training_folder = os.listdir(custom_dummy_folder)[0]
        training_folder_path = os.path.join(custom_dummy_folder, training_folder)
        training_folder_files = os.listdir(training_folder_path)

        # check saved the files
        assert "training_config.json" in training_folder_files
        assert "model_config.json" in training_folder_files
        assert "model.pt" in training_folder_files
        assert "encoder.pkl" in training_folder_files
        assert "decoder.pkl" in training_folder_files
        # assert 0

    def test_raises_missing_config_main_augmentation(
        self, demo_data, corrupted_config_path
    ):

        # assert raises file not found
        with pytest.raises(FileNotFoundError):
            augment_data(
                demo_data,
                number_of_samples=4,
                path_to_model_config=corrupted_config_path,
            )

        with pytest.raises(FileNotFoundError):
            augment_data(
                demo_data,
                number_of_samples=4,
                path_to_training_config=corrupted_config_path,
            )

        with pytest.raises(FileNotFoundError):
            augment_data(
                demo_data,
                number_of_samples=4,
                path_to_generation_config=corrupted_config_path,
            )

    def test_raises_not_json_config_main_augmentation(self, demo_data, not_json_file):

        return 0
        # assert raises file not found
        with pytest.raises(TypeError):
            augment_data(
                demo_data, number_of_samples=4, path_to_model_config=not_json_file
            )

        with pytest.raises(TypeError):
            augment_data(
                demo_data, number_of_samples=4, path_to_training_config=not_json_file
            )

        with pytest.raises(TypeError):
            augment_data(
                demo_data, number_of_samples=4, path_to_generation_config=not_json_file
            )


class Test_Train_MyModel:
    def test_creates_custom_logs_files(
        self, demo_data, custom_dummy_folder, custom_config_paths
    ):
        return 0
        train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_logs=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check log file is created
        assert len(os.listdir(custom_dummy_folder)) == 1

    def test_creates_model_files(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        return 0
        train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check training_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        training_folder = os.listdir(custom_dummy_folder)[0]
        training_folder_path = os.path.join(custom_dummy_folder, training_folder)
        training_folder_files = os.listdir(training_folder_path)

        # check saved the training_config_file
        assert "training_config.json" in training_folder_files
        assert "model_config.json" in training_folder_files
        assert "model.pt" in training_folder_files

    def test_output_model_in_eval(self, demo_data, custom_config_paths):
        return 0
        model = train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            output_model=True,
            verbose=False,
        )

        assert isinstance(model, RHVAE)
        assert not model.training

    def test_output_model_by_default(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        return 0
        model = train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            output_model=True,
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        assert isinstance(model, RHVAE)
        assert not model.training

    def test_creates_model_files_with_custom_archi(
        self,
        demo_data,
        custom_config_paths,
        custom_dummy_folder,
        custom_encoder,
        custom_decoder,
    ):
        return 0
        train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            encoder=custom_encoder,
            decoder=custom_decoder,
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check training_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        training_folder = os.listdir(custom_dummy_folder)[0]
        training_folder_path = os.path.join(custom_dummy_folder, training_folder)
        training_folder_files = os.listdir(training_folder_path)

        # check saved the files
        assert "training_config.json" in training_folder_files
        assert "model_config.json" in training_folder_files
        assert "model.pt" in training_folder_files
        assert "encoder.pkl" in training_folder_files
        assert "decoder.pkl" in training_folder_files

    def test_raises_missing_config_train_my_model(
        self, demo_data, corrupted_config_path
    ):
        return 0
        # assert raises file not found
        with pytest.raises(FileNotFoundError):
            train_my_model(demo_data, path_to_model_config=corrupted_config_path)

        with pytest.raises(FileNotFoundError):
            train_my_model(demo_data, path_to_training_config=corrupted_config_path)

    def test_raises_not_json_config_train_my_model(self, demo_data, not_json_file):

        return 0
        # assert raises file not found
        with pytest.raises(TypeError):
            train_my_model(demo_data, path_to_model_config=not_json_file)

        with pytest.raises(TypeError):
            train_my_model(demo_data, path_to_training_config=not_json_file)


class Test_Augment_From_PreTrained:
    @pytest.mark.filterwarnings("ignore")
    def test_creates_generation_files(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        # created a trained model
        model = train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_save_model=None,
            verbose=False,
        )

        augment_from_pretrained(
            number_of_samples=4,
            model=model,
            path_to_model_folder=None,
            path_to_generation_config=custom_config_paths[2],
            path_to_save_data=custom_dummy_folder,
        )

        # check folder is created
        assert os.path.isdir(custom_dummy_folder)

        # check generation_with_signature_folder is created
        assert len(os.listdir(custom_dummy_folder)) == 1

        generation_folder = os.listdir(custom_dummy_folder)[0]
        generation_folder_path = os.path.join(custom_dummy_folder, generation_folder)
        generation_folder_files = os.listdir(generation_folder_path)

        # check saved the files
        assert "generated_data.data" in generation_folder_files
        assert "generation_config.json" in generation_folder_files

    def test_raises_missing_config_augment_from_pretrained(
        self, dummy_model_config_with_input_dim, corrupted_config_path
    ):
        model = RHVAE(dummy_model_config_with_input_dim)
        # assert raises file not found
        with pytest.raises(FileNotFoundError):
            augment_from_pretrained(
                number_of_samples=4,
                model=model,
                path_to_generation_config=corrupted_config_path,
            )

    def test_raises_not_json_augment_from_pretrained(
        self, dummy_model_config_with_input_dim, not_json_file
    ):
        model = RHVAE(dummy_model_config_with_input_dim)
        # assert raises file not found
        with pytest.raises(TypeError):
            augment_from_pretrained(
                number_of_samples=4,
                model=model,
                path_to_generation_config=not_json_file,
            )

    def test_raises_missing_model(self):
        # check that it raised error if no model provided in augment from pre-trained
        with pytest.raises(ValueError):
            augment_from_pretrained(number_of_samples=4)

    def test_raises_corrupted_path_to_model(self, corrupted_config_path):
        with pytest.raises(FileNotFoundError):
            augment_from_pretrained(
                number_of_samples=4, path_to_model_folder=corrupted_config_path
            )


class Test_Custom_Config_Using:
    @pytest.mark.filterwarnings("ignore")
    def test_use_custom_config(
        self, demo_data, custom_config_paths, custom_dummy_folder
    ):
        train_my_model(
            demo_data,
            path_to_model_config=custom_config_paths[0],
            path_to_training_config=custom_config_paths[1],
            path_to_save_model=custom_dummy_folder,
            verbose=False,
        )

        parser_config_provided = ConfigParserFromJSON()

        # parse provided configs
        provided_model_config = parser_config_provided.parse_model(
            custom_config_paths[0]
        )

        provided_training_config = parser_config_provided.parse_training(
            custom_config_paths[1]
        )

        provided_generation_config = parser_config_provided.parse_generation(
            custom_config_paths[2]
        )

        training_folder = os.listdir(custom_dummy_folder)[0]
        training_folder_path = os.path.join(custom_dummy_folder, training_folder)

        # get path to saved configs
        model_used_path_to_json = os.path.join(
            training_folder_path, "model_config.json"
        )
        training_used_path_to_json = os.path.join(
            training_folder_path, "training_config.json"
        )

        # parse saved configs
        parser_config_saved = ConfigParserFromJSON()

        saved_model_config = parser_config_saved.parse_model(model_used_path_to_json)
        saved_training_config = parser_config_saved.parse_training(
            training_used_path_to_json
        )

        assert provided_model_config.latent_dim == saved_model_config.latent_dim
        assert (
            provided_training_config.learning_rate
            == saved_training_config.learning_rate
        )

        augment_from_pretrained(
            number_of_samples=4,
            path_to_model_folder=training_folder_path,
            path_to_generation_config=custom_config_paths[2],
            path_to_save_data=custom_dummy_folder,
        )

        # get generation folder (sorted since there will be training_sign and generation_sign
        # folders in cusstom_dummy_folder)
        generation_folder = sorted(os.listdir(custom_dummy_folder))[0]
        generation_folder_path = os.path.join(custom_dummy_folder, generation_folder)

        generation_used_path_to_json = os.path.join(
            generation_folder_path, "generation_config.json"
        )

        # parse saved configs
        parser_config_saved = ConfigParserFromJSON()

        saved_generation_config = parser_config_saved.parse_generation(
            generation_used_path_to_json
        )

        assert (
            provided_generation_config.mcmc_steps_nbr
            == saved_generation_config.mcmc_steps_nbr
        )
