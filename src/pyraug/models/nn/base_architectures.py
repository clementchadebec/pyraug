import typing

import torch
import torch.nn as nn


class Base_Encoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns (mu, log_var) in that order. 
        If you decide to provide your own encoder network, you must make your
        model inherit from this class by setting and the define your forward function as 
        such:

        .. code-block::

            class My_Encoder(Base_Encoder):

                def __init__(self):
                    Base_Encoder.__init__(self)
                    # your code

                def forward(self, x):
                    # your code
                    return mu, log_var
        
        Parameters:
            x (torch.Tensor): The input data that must be encoded
            
        Returns:
            (Tuple[torch.Tensor, torch.Tensor): The mean :math:`\mu_{\phi}`  and the **log** of the variance
            ():math:`\log \Sigma`) of the approximate distribution. As is common :`Sigma` is diagonal and so only the diagonal 
            coefficients should be output
                
        .. warning::
            The output order in here important. Do not forget to set :math:`\mu` as first argument and
            the log variance then.
        """
        NotImplementedError()


class Base_Decoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the latent data and returns (mu). If you decide to provide
        your own decoder network  you must make your
        model inherit from this class by setting and the define your forward function as 
        such:

        .. code-block::

            class My_decoder(Base_Decoder):

                def __init__(self):
                    Base_Decoder.__init__(self)
                    # your code

                def forward(self, z):
                    # your code
                    return mu
        
        Parameters:
            z (torch.Tensor): The latent data that must be decoded
            
        Returns:
            (torch.Tensor): The mean of the conditional distribution :math:`p_{\theta}(x|z)`
        """
        raise NotImplementedError()


class Base_Metric(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns (L_psi). 
        If you decide to provide your own metric network, you must make your
        model inherit from this class by setting and the define your forward function as 
        such:

        .. code-block::

            class My_Metric(Base_Metric):

                def __init__(self):
                    Base_Metric.__init__(self)
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
