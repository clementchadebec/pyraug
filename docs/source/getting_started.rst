**********************************
Getting stated
**********************************

Description
###########

This library provides a way to perform Data Augmentation using the Variational Autoencoders in a 
reliable way in the challenging High Dimensional Low Sample size setting.


Installation
############


To install the librairy run:

.. code-block:: bash

    $ pip install pyraug 


or alternatively clone the github repo to access to test and tutorials.

.. code-block:: bash

    $ git clone 

Augmenting your Data
####################

There exists two way to augment your data pretty straightforwardly using Pyraug's built-in functions. 
A typical augmentation process is divided into 2 distinct parts:

    - Train a model using the Pyraug's :class:`~pyraug.pipelines.TrainingPipeline`
    - Generate new data from a trained model using Pyraug's :class:`~pyraug.pipelines.GenerationPipeline`



Using the provided commandline
-------------------------------

Pyraug provides two scripts allowing you to augment your data directly with commandlines.

.. note::
    To access to the predefined scripts you should first clone the Pyraug's repositority.
    The following scripts are located in ``pyraug/scripts`` folder. For the time being, only :class:`~pyraug.models.RHVAE` model training and generation is handled by the provided script. Models will be added as they are implemented in :ref:`pyraug.models` 

Launching a model training:
''''''''''''''''''''''''''''

To launch a model training, run 

.. code-block:: bash

    $ python scripts/training.py --path_to_train_data 'path/to/your/data/folder' 

The model will be saved in a folder ``outputs/my_model_from_script/training_YYYY-MM-DD_hh-mm-ss/final_model``


.. tip::
   In the simplest configuration, default ``training_config.json`` and ``model_config.json`` are used. You can easily overide these parameters by defining your own ``.json`` file and passing them the to the parser arguments.

    .. code-block:: bash

        $ python scripts/training.py 
            --path_to_train_data 'path/to/your/data/folder'
            --path_to_model_config 'path/to/your/model/config.json'
            --path_to_training_config 'path/to/your/training/config.json'

    See :ref:`setting-own-config` for a more in depth example.



Launching data generation:
''''''''''''''''''''''''''''

To launch the data generation process from a trained model, run 

.. code-block:: bash

    $ python scripts/training.py --num_samples 10 --path_model_folder 'path/to_your/trained/model/folder' 

The generated data is stored in several ``.pt`` files in ``outputs/my_generated_data_from_script/generation_YYYY-MM-DD_hh_mm_ss``

.. tip::
    In the simplest configuration, default ``sampler_config.json`` is used. You can easily override these parameters by defining your own ``.json`` file and passing them the to the parser arguments.  See :ref:`setting-own-config` and tutorials.

    .. code-block:: bash

        $ python scripts/training.py 
            --path_to_train_data 'path/to/your/data/folder'
            --path_to_sampler_config 'path/to/your/training/config.json'
        
    See tutorials for a more in depth example.



Retrieve generated data
''''''''''''''''''''''''''''

Generated data can then be loaded pretty easily by running

.. code-block:: python

    >>> import torch
    >>> data = torch.load('path/to/generated_data.pt')




Using Pyraug's Pipelines
-------------------------------

Pyraug provides you with two pipelines that you may use to either train a model on your own data or generate new data with a pretrained model.


.. tip::

    If you want to access to more advanced feature such as defining your own autoencoding architecture, you can use the predefined pipelines which are independent of the choice of the model and sampler.  

Launching a model training
''''''''''''''''''''''''''''

To launch a model training, you only need instantiate your own model.
For instance, if you want to instantiate a basic :class:`~pyraug.models.RHVAE` run:


.. code-block:: python
    
    >>> from pyraug.models import RHVAE
    >>> from pyraug.models.rhvae import RHVAEConfig
    >>> model_config = RHVAEConfig(
    ...    input_dim=int(intput_dim)
    ... ) # input_dim is the shape of a flatten input data
    ...   # needed if you do not provided your own architectures
    >>> model = RHVAE(model_config)

Then the :class:`~pyraug.pipelines.TrainingPipeline` can be launched by running:

.. code-block:: python

    >>> from pyraug.pipelines import TrainingPipeline
    >>> pipe = TrainingPipeline(model=model)
    >>> pipe(train_data='path/to/your/data/folder')

The trained model is saved ``dummy_output_dir/training_YYYY-MM-DD_hh-mm-ss/final_model``.

.. tip::
    A full example is provided in the tutorials (demo1.ipynb)

.. tip::
    In the simplest configuration, defaults training and model parameters are used. You can easily override these parameters by instantiating your own :class:`~pyraug.trainers.training_config.TrainginConfig` and :class:`~pyraug.models.base.base_config.BaseModelConfig` file and passing them the to the :class:`~pyraug.pipelines.TrainingPipeline` see :ref:`setting-own-config`

    Example for a :class:`~pyraug.models.RHVAE` run:

    .. code-block:: python

        >>> from pyraug.models import RHVAE
        >>> from pyraug.model.rhvae import RHVAEConfig
        >>> from pyraug.trainers.training_config import TrainingConfig
        >>> from pyraug.pipelines import TrainingPipeline
        >>> custom_model_config = RHVAEConfig(
        ...    input_dim=input_dim,
        ...    *my_args,
        ...    **my_kwargs
        ... )
        >>> model = RHAVE(custom_model_config)
        >>> custom_training_config = TrainingConfig(
        ...    *my_args,
        ...    **my_kwargs
        ... )
        >>> pipe = TrainingPipeline(
        ...    model = model,
        ...    training_config=custom_training_config
        ... )
        
    See tutorials for a more in depth example.


Launching data generation:
''''''''''''''''''''''''''''

To launch the data generation process from a trained model, run 

.. code-block:: python

    >>> from pyraug.pipelines import GenerationPipeline
    >>> model = RHVAE.load_from_folder('path/to/your/trained/model')
    >>> pipe = GenerationPipeline(
    ...    model=model
    ... )
    >>> pipe(samples_number=10) # This will generate 10 data points

The generated data is in ``.pt`` files in ``dummy_output_dir/generation_YYYY-MM-DD_hh-mm-ss``.


Retrieve generated data
''''''''''''''''''''''''''''

Generated data can then be loaded pretty easily by running

.. code-block:: python

    >>> import torch
    >>> data = torch.load('path/to/generated_data.pt')
