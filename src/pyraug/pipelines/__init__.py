"""The Pipelines module is created to facilitate the use of the library. It provides ways to 
perform end-to-end operation such as model training or generation. A typical Pipeline is composed by 
several Pyraug's instances which are articulated together.

A __call__ function is defined and use to launch the Pipeline. """

from pyraug.pipelines.generation import GenerationPipeline
from pyraug.pipelines.training import TrainingPipeline

__all__ = [
    "TrainingPipeline",
    "GenerationPipeline"
]