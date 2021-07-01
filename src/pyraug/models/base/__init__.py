"""This is the base VAE architecture module from which all future models
should inherit.
It contains:

- | a :class:`~pyraug.models.base.base_config.BaseModelConfig` instance containing the main model's
   parameters (*e.g.* latent dimension ...)
- | a :class:`~pyraug.models.BaseVAE` instance which creates a BaseVAE model having a basic
   autoencoding architecture
- | a :class:`~pyraug.models.base.base_config.BaseSamplerConfig` instance containing the main
   sampler's parameters used to sample from the latent space of the BaseVAE
- a :class:`~pyraug.models.base.base_sampler.BaseSampler` instance which creates a BaseSampler.

.. note::
   If you want ot build upon this work try to make any new model inherit from these 4 classes.
"""
