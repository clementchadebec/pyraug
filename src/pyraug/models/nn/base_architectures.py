import torch.nn as nn


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns (mu, log_var) in that order.
        If you decide to provide your own encoder network, you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_Encoder(BaseEncoder):

                def __init__(self):
                    BaseEncoder.__init__(self)
                    # your code

                def forward(self, x):
                    # your code
                    return mu, log_var

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            (Tuple[torch.Tensor, torch.Tensor): The mean :math:`\mu_{\phi}`  and the **log** of the variance
            ():math:`\log \Sigma`) of the approximate distribution. As is common :`Sigma` is
            diagonal and so only the diagonal coefficients should be output

        .. warning::
            The output order in here important. Do not forget to set :math:`\mu` as first argument and
            the log variance then.
        """
        NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the latent data and returns (mu). If you decide to provide
        your own decoder network  you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_decoder(BaseDecoder):

                def __init__(self):
                    BaseDecoder.__init__(self)
                    # your code

                def forward(self, z):
                    # your code
                    return mu

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            (torch.Tensor): The mean of the conditional distribution :math:`p_{\theta}(x|z)`

        .. note::

            By convention, the output tensors :math:`\mu` should be in [0, 1]

        """
        raise NotImplementedError()


class BaseMetric(nn.Module):
    """This is a base class for Metrics neural networks
    (only applicable for Riemannian based VAE)
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns (L_psi).
        If you decide to provide your own metric network, you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_Metric(BaseMetric):

                def __init__(self):
                    BaseMetric.__init__(self)
                    # your code

                def forward(self, x):
                    # your code
                    return L

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            (torch.Tensor): The :math:`L_{\psi}` matrices of the metric
        """
        raise NotImplementedError()
