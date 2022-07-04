import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from config_setup import config
from cbctgen import CBCTGenerator
import os
#from src.pyraug.trainers.training_config import TrainingConfig
from src.pyraug.pipelines.training import TrainingPipeline

## For reproducible results    
def seed_all(s):
    np.random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s) 
    torch.manual_seed(s)
    print(f'Seeds set to {s}!')


if __name__ == "__main__":
    seed_all(42) #For reproducible results

    gen = CBCTGenerator(config)

    # Load the data
    train_set = gen.train_loader

    #Define training configuration for TrainingPipeline
    pyraug_config = TrainingConfig(
        output_dir='my_model',
        train_early_stopping=50,
        learning_rate=1e-4,
        batch_size=8, # Set to 200 for demo purposes to speed up (default: 50)
        max_epochs=50000 # Set to 500 for demo purposes. Augment this in your case to access to better generative model (default: 20000)
    )

    # This creates the Pipeline
    pipeline = TrainingPipeline(training_config=pyraug_config)

    # This will launch the Pipeline on the data
    pipeline(train_data=train_set, log_output_dir='output_logs')


