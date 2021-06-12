import pytest
import torch
import numpy as np
import os

from torch.optim import RMSprop, SGD, Adadelta, Adagrad, Adam
from copy import deepcopy

from pyraug.models import RHVAE
from pyraug.models.rhvae import RHVAEConfig
from pyraug.trainers.training_config import TrainingConfig
from pyraug.pipelines.training import TrainingPipeline



@pytest.fixture(params=[
            [
                [
                100 * torch.rand(3, 20, 15, 30), # train data
                torch.rand(3, 10, 25, 10),
                torch.rand(3, 10, 10, 30)
                ],
                (3, 3, 10, 10, 10), # target shape
                [
                100 * torch.rand(3, 20, 15, 30), # eval data (should be compatible with target shape)
                torch.rand(3, 10, 25, 10),
                torch.rand(3, 10, 10, 30)
                ],
            ],
            [
                [
                100 * torch.rand(1, 20, 15, 30),
                torch.rand(1, 10, 25, 10),
                torch.rand(1, 10, 10, 30),
                10000*torch.rand(1, 10, 10, 30),
                100*torch.rand(1, 100, 30, 30)
                ],
                (5, 1, 10, 10, 10),
                np.random.randn(2, 1, 10, 10, 10),
            ]
            ,
            [
                torch.randn(4, 12, 10),
                (4, 12, 10),
                None
            ],
            [
                np.random.randn(10, 2, 17, 28),
                (10, 2, 17, 28),
                [
                    torch.rand(2, 17, 28),
                    torch.rand(2, 17, 28)
                ]
            ]
        ])
def messy_data(request):
    return request.param



class Test_Pipeline:

    @pytest.fixture(params=[
        TrainingConfig(
            max_epochs=3
        ),
        TrainingConfig(
            max_epochs=3,
            learning_rate=1e-8
        ),
        TrainingConfig(
            max_epochs=3,
            batch_size=12,
            train_early_stopping=False
        ),
        TrainingConfig(
            max_epochs=3,
            batch_size=12,
            train_early_stopping=False,
            eval_early_stopping=2
        ),
    ])
    def training_config(self, tmpdir, request):
        tmpdir.mkdir('dummy_folder')
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir =  dir_path
        return request.param


    @pytest.fixture
    def rhvae_config(self, messy_data):
        return RHVAEConfig(input_dim=np.prod(messy_data[1][1:]))

    @pytest.fixture
    def rhvae_sample(self, rhvae_config):
        return RHVAE(rhvae_config)

    @pytest.fixture(params=[
        Adagrad,
        Adam,
        Adadelta,
        SGD,
        RMSprop
    ])
    def optimizer(self, request, rhvae_sample, training_config):
        
        optimizer = request.param(rhvae_sample.parameters(), lr=training_config.learning_rate)
        return optimizer

    def test_pipeline(self, messy_data, rhvae_sample, optimizer, training_config):
        pipe = TrainingPipeline(
            model=rhvae_sample,
            optimizer=optimizer,
            training_config=training_config
        )
    
        start_model = deepcopy(rhvae_sample)

        pipe(
            train_data=messy_data[0],
            eval_data=messy_data[2])

        assert not all([
            torch.equal(
                pipe.model.state_dict()[key],
                start_model.state_dict()[key]) for key in start_model.state_dict().keys()
        ])

        

    