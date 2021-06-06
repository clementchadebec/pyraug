import dataclasses
import json
import os
import typing
from abc import ABC, abstractmethod
from copy import deepcopy

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class BaseVAE(nn.Module):
    """Base class for VAE based models"""

    def __init__(self, model_config):

        nn.Module.__init__(self)

        # self.data_type = data_type
        self.input_dim = model_config.input_dim
        self.latent_dim = model_config.latent_dim
        self.device = model_config.device

        self.encoder = None
        self.decoder = None
        self.metric = None

    def forward(self, x):
        """Main forward pass outputing the VAE outputs
        This function should output an model_output instance gathering all the model outputs"""
        raise NotImplementedError()

    def loss_function(self):
        """Return the loss function associated to the model"""
        raise NotImplementedError()

    def save(self, path_to_save_model):
        """Method to save the model at a specific location"""

        model_path = path_to_save_model

        model_dict = {
            "M": deepcopy(self.M_tens),
            "centroids": deepcopy(self.centroids_tens),
            "model_state_dict": deepcopy(self.state_dict()),
        }

        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path)

            except FileNotFoundError as e:
                raise e

        with open(os.path.join(model_path, "model_config.json"), "w") as fp:
            json.dump(dataclasses.asdict(self.model_config), fp)

        # only save .pkl if custom architecture provided
        if "custom" in self.model_config.encoder:
            with open(os.path.join(model_path, "encoder.pkl"), "wb") as fp:
                dill.dump(self.encoder, fp)

        if "custom" in self.model_config.decoder:
            with open(os.path.join(model_path, "decoder.pkl"), "wb") as fp:
                dill.dump(self.decoder, fp)

        if "custom" in self.model_config.metric:
            with open(os.path.join(model_path, "metric.pkl"), "wb") as fp:
                dill.dump(self.metric, fp)

        torch.save(model_dict, os.path.join(model_path, "model.pt"))

    def set_encoder(self, encoder: nn.Module) -> None:
        """Set the encoder of the model"""
        self.encoder = encoder

    def set_decoder(self, decoder: nn.Module) -> None:
        """Set the decoder of the model"""
        self.decoder = decoder


class RHVAE(BaseVAE):
    r"""
    This class is  an implementation of the VAE model proposed in 
    (https://arxiv.org/pdf/2010.11518.pdf). This model provides a way to 
    learn the Riemannian latent structure of a given set of data set through a parametrized 
    Riemannian metric having the following shape:
    :math:`\mathbf{G}^{-1}(z) = \sum \limits _{i=1}^N L_{\psi_i} L_{\psi_i}^{\top} \exp \Big(-\frac{\lVert z - c_i \rVert_2^2}{T^2} \Big) + \lambda I_d`

    and to generate new data. It is particularly well suited for High 
    Dimensional data combined with low sample number and proved relevant for Data Augmentation as 
    proved in (https://arxiv.org/pdf/2105.00026.pdf).
    """

    def __init__(self, model_config):
        BaseVAE.__init__(self, model_config)

        self.model_config = model_config

        self.T = nn.Parameter(
            torch.Tensor([model_config.temperature]), requires_grad=False
        )
        self.lbd = nn.Parameter(
            torch.Tensor([model_config.regularization]), requires_grad=False
        )
        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([model_config.beta_zero]), requires_grad=False
        )
        self.n_lf = model_config.n_lf
        self.eps_lf = model_config.eps_lf

        # this is used to store the matrices and centroids throughout trainning for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        # define a starting metric (gamma_i = 0 & L = I_d)
        def G(z):
            return (
                torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G

        # define a N(0, I) distribution
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.latent_dim).to(self.device),
            covariance_matrix=torch.eye(self.latent_dim).to(self.device),
        )

    def set_metric(self, metric: nn.Module) -> None:
        r"""This method is called to set the metric network outputing the
        :math:`L_{\psi_i}` of the metric matrices 

        Args:
            metric (torch.nn.Module): The metric module that need to be set to the model.
                
        """
        self.metric = metric

    def forward(self, x):
        r"""
        The input data is first encoded. The reparametrization is used to produce a sample 
        :math:`z_0` from the approximate posterior :math:`q_{\phi}(z|x)`. Then Riemannian
        Hamiltonian equations are solved using the generalized leapfrog integrator. In the meantime,
        the input data :math:`x` is fed to the metric network outputing the matrices 
        :math:`L_{\psi}`. The metric is computed and used with the integrator.

        Args:
            x (torch.tensor): The input data of shape (batch_size, -1)
        """
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        z0, eps0 = self._sample_gauss(mu, std)

        z = z0

        if self.training:
            # update the metric using batch data points
            L = self.metric(x)

            M = L @ torch.transpose(L, 1, 2)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decoder(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self._leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self._leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decoder(z)

            if self.training:

                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self._leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det

    def _leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self._hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def _leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self._hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def _leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        """
        H = self._hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def _hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self._log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def update_metric(self):
        r"""
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(self.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self._log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            (
                -0.5
                * (
                    torch.transpose(rhoK.unsqueeze(-1), 1, 2)
                    @ G_inv
                    @ rhoK.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det
            )
            - torch.log(torch.tensor([2 * np.pi]).to(self.device)) * self.latent_dim / 2
        )  # log p(\rho_K)

        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).sum()

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    def _log_p_x_given_z(self, recon_x, x, reduction="none"):
        r"""Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction=reduction
        ).sum(dim=1)

    def _log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def _log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self._log_p_x_given_z(recon_x, x)
        logpz = self._log_z(z)
        return logpxz + logpz
