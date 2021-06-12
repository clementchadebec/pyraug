from .base_pipeline import Pipeline
from pyraug.models import BaseVAE
from pyraug.trainers.training_config import TrainingConfig
import numpy as np
import torch

from typing import Union, Optional
from pyraug.trainers import Trainer
from pyraug.customexception import LoadError


from pyraug.data.loader import DataGetter
from pyraug.data.preprocessors import DataProcessor
from torch.optim import Optimizer

class TrainingPipeline(Pipeline):
    
    def __init__(self,
        data_loader: Optional[DataGetter] = None,
        data_processor: Optional[DataProcessor] = None,
        model: Optional[BaseVAE] = None,
        optimizer: Optional[Optimizer] = None,
        training_config:Optional[TrainingConfig]=None):

        #model_name = model_name.upper()
        
        self.data_loader = data_loader

        if data_processor is None:
            data_processor = DataProcessor
    
        self.data_processor = data_processor
        self.model = model
        self.optimizer = optimizer
        self.training_config = training_config

    def __call__(
        self,
        train_data: Union[str, np.ndarray, torch.Tensor],
        eval_data: Union[str, np.ndarray, torch.Tensor]=None):
        """
        Launch a model training

        Args:
            training_data (Union[str, np.ndarray, torch.Tensor]): The training data coming from 
                a folder in which each file is a data or a np.ndarray or torch.Tensor of shape 
                (-1, data_shape)
            eval_data (Union[str, np.ndarray, torch.Tensor]): The evaluation data coming from 
                a folder in which each file is a data or a np.ndarray or torch.Tensor of shape 
                (-1, data_shape) 
        """

        if self.data_loader is None:
            if isinstance(train_data, str):
                raise TypeError("No loader provided. Train data is expected to come as np.ndarray or "
                    "torch.Tensor")

        else:
            try:
                train_data = self.data_loader.load(train_data)

            except Exception as e:
                raise LoadError(
                    f"Enable to load training data. Exception catch: {type(e)} with message: "
                    + str(e))

        train_data = self.data_processor.process_data(train_data)
        train_dataset = self.data_processor.to_dataset(train_data)
       

        if eval_data is not None:
            if self.data_loader is None:
                if isinstance(eval_data, str):
                    raise TypeError("No loader provided. Eval data is expected to come as np.ndarray or"
                        " torch.Tensor")

            else:
                try:
                    eval_data = self.data_loader.load(eval_data)

                except Exception as e:
                    raise LoadError(
                        f"Enable to load eval data. Exception catch: {type(e)} with message: "
                        + str(e))
            eval_data = self.data_processor.process_data(eval_data)
            eval_dataset = self.data_processor.to_dataset(eval_data)

        else: 
            eval_dataset = None


        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_config=self.training_config,
            optimizer=self.optimizer)

        trainer.train()
