import functools
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad


class Geodesic_linear(object):
    def __init__(
        self,
        model,
        starting_pos=None,
        ending_pos=None,
        granularity=20.0,
        step=1e-3,
        eps=1e-3,
    ):
        self.model = model
        self.granularity = granularity
        self.dt = 1 / granularity
        self.eps = eps
        self.step = step
        self.z0 = starting_pos
        self.zT = ending_pos

    def grad_metric_inv(self, z):
        return (
            -torch.transpose(
                (
                    -2
                    / (self.model.T ** 2)
                    * (
                        self.model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)
                    ).unsqueeze(2)
                    @ (
                        self.model.M_tens.unsqueeze(0)
                        * torch.exp(
                            -torch.norm(
                                self.model.centroids_tens.unsqueeze(0) - z.unsqueeze(1),
                                dim=-1,
                            )
                            ** 2
                            / (self.model.T ** 2)
                        )
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                ).sum(dim=1),
                1,
                2,
            )
        ).squeeze(0)

    def compute_grad(self, z, z_fol):

        G = self.model.G(z).squeeze(0)

        return (
            -(G @ self.grad_metric_inv(z)).T @ G @ ((z.T @ z) + 2 * z_fol.T @ z)
        )  # + 5 *2 * (z - z_fol)

    def fit(self):
        max_it = 100
        grad_E_norm = 1e10
        i = 0
        t = torch.linspace(0, 1, int(self.granularity)).unsqueeze(-1)
        z = self.z0 * (1 - t) + self.zT * t
        z = z.unsqueeze(1)
        grad_prev = 0
        while (
            i < max_it
            and grad_E_norm > self.eps
            and abs(grad_E_norm - grad_prev) > self.eps
        ):
            grad_prev = grad_E_norm
            grad_E_norm = 0.0
            i += 1

            for t in range(1, int(self.granularity) - 1):
                # print(z[t].shape)
                grad = self.compute_grad(z[t], z[t - 1])
                # print(grad.shape)
                z[t] = z[t] - self.step * grad

                grad_E_norm += torch.norm(grad)
            print(grad_E_norm)

        return z


class Exponential_map(object):
    def __init__(self, latent_dim=2):
        self.starting_pos = None
        self.starting_velo = None
        self.metric_inv = None
        self.latent_dim = latent_dim

    def _velocity_to_momentum(self, v, p=None):
        if p is None:
            return torch.inverse(self.metric_inv(self.starting_pos)) @ v.unsqueeze(-1)

        else:
            return torch.inverse(self.metric_inv(p)) @ v.unsqueeze(-1)

    def _momentum_to_velocity(self, q, p=None):
        if p is None:
            return self.metric_inv(self.starting_pos) @ q.unsqueeze(-1)

        else:
            return self.metric_inv(p) @ q.unsqueeze(-1)

    def _rk_step(self, p, q, dp, dt):
        # print(p.shape, q.shape)
        # print(q)
        mean_p = p.unsqueeze(-1) + 0.5 * dt * self.metric_inv(p) @ q.unsqueeze(-1)
        # print(self.metric_inv(p))
        # print(dp(p, q).shape, p.shape)
        mean_q = q.unsqueeze(-1) - 0.5 * dt * dp(p, q)

        # print(mean_q.shape, p.shape, self.metric_inv(mean_p.squeeze(-1)).shape)
        return (
            p.unsqueeze(-1) + dt * self.metric_inv(mean_p.squeeze(-1)) @ mean_q,
            q.unsqueeze(-1) - dt * dp(mean_p.squeeze(-1), mean_q.squeeze(-1)),
        )

    def hamiltonian(self, p, q):
        return (
            0.5
            * torch.transpose(q.unsqueeze(-1), 1, 2)
            @ self.metric_inv(p)
            @ q.unsqueeze(-1)
        )

    @staticmethod
    def dH_dp(p, q, model):
        # print(q.shape)
        a = (
            torch.transpose(q.unsqueeze(-1).unsqueeze(1), 2, 3)
            @ model.M_tens.unsqueeze(0)
            @ q.unsqueeze(-1).unsqueeze(1)
        )
        b = model.centroids_tens.unsqueeze(0) - p.unsqueeze(1)
        # print(a)
        return (
            -1
            / (model.T ** 2)
            * b.unsqueeze(-1)
            @ a
            * (
                torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - p.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.T ** 2)
                )
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1)

    def shoot(self, p=None, v=None, q=None, model=None, n_steps=10):
        """
        Geodesic shooting using Hamiltonian dynamics

        Inputs:
        -------

        p (tensor): Starting position
        v (tensor): Starting velocity
        q (tensor): Starting momentum
        inverse_metric (function): The inverse Riemannian metric should output the matrix form
        """
        assert (
            p is not None
        ), "Provide a starting position (i.e. where the exponential is computed"
        assert (
            v is not None or p is not None
        ), "Provide at least a starting velocity or momentum"

        self.metric_inv = model.G_inv

        if q is None:
            q = self._velocity_to_momentum(v, p=p).reshape(-1, self.latent_dim)

        dp = functools.partial(Exponential_map.dH_dp, model=model)

        dt = 1 / float(n_steps)

        pos_path = torch.zeros(p.shape[0], n_steps + 1, self.latent_dim).requires_grad_(
            False
        )
        mom_path = torch.zeros(p.shape[0], n_steps + 1, self.latent_dim).requires_grad_(
            False
        )

        # print(p.shape)
        pos_path[:, 0, :] = p.reshape(-1, self.latent_dim)
        mom_path[:, 0, :] = q.reshape(-1, self.latent_dim)

        for i in range(n_steps):
            p_t, q_t = self._rk_step(p, q, dp, dt)

            p, q = p_t.reshape(-1, self.latent_dim), q_t.reshape(-1, self.latent_dim)

            pos_path[:, i + 1, :] = p.detach()
            mom_path[:, i + 1, :] = q.detach()

            # print((q @ self.metric_inv(p) @ q.T).sqrt())

        return pos_path, mom_path


class Geodesic_autodiff(nn.Module):
    def __init__(
        self,
        metric=None,
        starting_pos=None,
        ending_pos=None,
        starting_velo=None,
        latent_dim=2,
        reg=0.0,
        granularity=100,
        early_stopping=100,
        device="cpu",
        seed=8,
    ):
        """
        Geodesic NN model

        Inputs:
        -------

        metric (function): The metric used to compute the geodesic path
        starting_pos (tensor): The starting point of the
        ending_pos (tensor): The ending point of the path
        starting_velo (tensor) [optional]: The initial velocity (for further use)
        latent_dim (int): Latent space dimension
        reg (float): L-2 regularization factor
        granularity (int): The discretization granularity
        """
        torch.manual_seed(seed)
        nn.Module.__init__(self)

        self.compute_with_ending_point = False
        self.compute_with_velo = False
        self.device = device
        self.early_stopping = early_stopping

        if starting_pos is None:
            starting_pos = torch.zeros(1, latent_dim).to(self.device)

        if ending_pos is None:
            ending_pos = torch.ones_like(starting_pos).to(self.device)

        else:
            self.compute_with_ending_point = True

        if starting_velo is None:
            starting_velo = torch.zeros(1, latent_dim).to(self.device)

        else:
            self.compute_with_velo = True

        self.starting_pos = starting_pos
        self.ending_pos = ending_pos
        self.starting_velo = starting_velo

        self.metric = metric
        self.reg = reg
        self.gran = granularity
        self.latent_dim = latent_dim
        self.length = None

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, latent_dim)

    def forward(self, t):
        """
        The geodesic model
        """
        h1 = torch.tanh(self.fc1(t))
        h2 = torch.tanh(self.fc2(h1))
        out = self.fc3(h2)

        return out

    def loss_function(self, curve_t, gt_):
        # gt_ = torch.zeros(self.latent_dim, 1)
        #
        # for i in range(self.latent_dim):
        #    gt_[i] = grad(curve_t[0][i], t, create_graph=True)[0]
        #    #print(gt_)

        # print(torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_))

        return (
            torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_)
            + self.reg * self.metric(curve_t).norm()
        )

    def fit(self, n_epochs=10, lr=1e-2):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        best_curve_model = deepcopy(self)
        best_loss = 1e20

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = 0
            loss += torch.sqrt(
                self.starting_velo
                @ self.metric(self.starting_pos)
                @ self.starting_velo.T
            )

            curve_t0 = self(torch.tensor([0.0]).to(self.device))
            curve_t1 = self(torch.tensor([1.0]).to(self.device))

            if self.compute_with_velo:
                gt_0 = torch.zeros(1, self.latent_dim)

                for t in range(1):
                    t = (
                        torch.tensor([t])
                        .type(torch.float)
                        .requires_grad_(True)
                        .to(self.device)
                    )
                    curve_t = self(t).reshape(1, self.latent_dim)

                    for i in range(self.latent_dim):
                        gt_0[0][i] = grad(curve_t[0][i], t, retain_graph=True)[0]
                # print(self.starting_velo, gt_0)
                a = self.starting_velo / gt_0
                b = a * curve_t0

            elif self.compute_with_ending_point:
                a = (self.starting_pos - self.ending_pos) / (curve_t0 - curve_t1)
                b = (self.starting_pos * curve_t1 - self.ending_pos * curve_t0) / (
                    curve_t0 - curve_t1
                )

            for t in range(0, self.gran + 1):
                t = (
                    torch.tensor([t / self.gran])
                    .type(torch.float)
                    .requires_grad_(True)
                    .to(self.device)
                )
                curve_t = self(t).reshape(1, self.latent_dim)

                curve_t = a * curve_t - b

                gt_ = torch.zeros(self.latent_dim, 1).to(self.device)

                for i in range(self.latent_dim):
                    gt_[i] = grad(curve_t[0][i], t, retain_graph=True)[0]
                # print(torch.norm(gt_, dim=0))
                # gt_ / torch.norm(gt_, dim=0)

                loss += self.loss_function(curve_t, gt_)

            loss /= self.gran

            if loss < best_loss:
                es_epoch = 0
                print("better", loss)
                best_curve_model = deepcopy(self)
                best_loss = loss
                best_a, best_b = a, b
                length = best_loss

            elif self.early_stopping > 0:
                es_epoch += 1

                if es_epoch >= self.early_stopping:
                    print(
                        f"Early Stopping at epoch {epoch} ! Loss did not improve in {self.early_stopping} epochs"
                    )
                    break

            # print(loss)
            if epoch % 50 == 0:
                print("-----")
                print(loss)

            loss.backward()
            optimizer.step()

        return best_curve_model, best_a, best_b, length


class Geodesic_hand(nn.Module):
    def __init__(
        self,
        metric=None,
        starting_pos=None,
        ending_pos=None,
        starting_velo=None,
        latent_dim=2,
        reg=0.0,
        granularity=100,
        device="cpu",
        seed=8,
    ):
        """
        Geodesic NN model

        Inputs:
        -------

        metric (function): The metric used to compute the geodesic path
        starting_pos (tensor): The starting point of the
        ending_pos (tensor): The ending point of the path
        starting_velo (tensor) [optional]: The initial velocity (for further use)
        latent_dim (int): Latent space dimension
        reg (float): L-2 regularization factor
        granularity (int): The discretization granularity
        """
        torch.manual_seed(seed)
        nn.Module.__init__(self)

        self.compute_with_ending_point = False
        self.compute_with_velo = False
        self.device = device

        if starting_pos is None:
            starting_pos = torch.zeros(1, latent_dim).to(self.device)

        if ending_pos is None:
            ending_pos = torch.ones_like(starting_pos).to(self.device)

        else:
            self.compute_with_ending_point = True

        if starting_velo is None:
            starting_velo = torch.zeros(1, latent_dim).to(self.device)

        else:
            self.compute_with_velo = True

        self.starting_pos = starting_pos
        self.ending_pos = ending_pos
        self.starting_velo = starting_velo

        self.metric = metric
        self.reg = reg
        self.gran = granularity
        self.latent_dim = latent_dim
        self.length = None

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, latent_dim)

    def forward(self, t):
        """
        The geodesic model
        """
        h1 = torch.tanh(self.fc1(t))
        h2 = torch.tanh(self.fc2(h1))
        out = self.fc3(h2)

        return out

    def loss_function(self, curve_t, gt_):
        # gt_ = torch.zeros(self.latent_dim, 1)
        #
        # for i in range(self.latent_dim):
        #    gt_[i] = grad(curve_t[0][i], t, create_graph=True)[0]
        #    #print(gt_)

        # print(torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_))

        return (
            torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_)
            + self.reg * self.metric(curve_t).norm()
        )

    def _gt(self, t):
        """
        Comutes the gradient of the curve and evalutate it at t
        """

        h1 = self.fc1(t)
        h2 = self.fc2(h1)
        # out = self.fc3(h2)

        dh1_dt = (1 - h1 ** 2).unsqueeze(-1) * self.fc1.weight
        # print(dh1_dt.shape)
        dh2_dh1 = (1 - h2 ** 2).unsqueeze(-1) * self.fc2.weight
        # print(dh2_dh1.shape)
        dout_dh2 = self.fc3.weight
        # print(dout_dh2.shape)

        gt = dout_dh2 @ dh2_dh1 @ dh1_dt

        # print(torch.norm(gt, dim=1).unsqsqueeze(1).shape)

        return gt  # / torch.norm(gt, dim=1).unsqueeze(1)

    def fit(self, n_epochs=10, lr=1e-2):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        best_curve_model = deepcopy(self)
        best_loss = 1e20

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = 0
            loss += torch.sqrt(
                self.starting_velo
                @ self.metric(self.starting_pos)
                @ self.starting_velo.T
            )

            curve_t0 = self(torch.tensor([0.0]).to(self.device))
            curve_t1 = self(torch.tensor([1.0]).to(self.device))

            if self.compute_with_velo:
                gt_0 = torch.zeros(1, self.latent_dim)

                for t in range(1):
                    t = (
                        torch.tensor([t])
                        .type(torch.float)
                        .requires_grad_(True)
                        .to(self.device)
                    )
                    curve_t = self(t).reshape(1, self.latent_dim)

                    for i in range(self.latent_dim):
                        gt_0[0][i] = grad(curve_t[0][i], t, retain_graph=True)[0]
                # print(self.starting_velo, gt_0)
                a = self.starting_velo / gt_0
                b = a * curve_t0

            elif self.compute_with_ending_point:
                a = (self.starting_pos - self.ending_pos) / (curve_t0 - curve_t1)
                b = (self.starting_pos * curve_t1 - self.ending_pos * curve_t0) / (
                    curve_t0 - curve_t1
                )

            for t in range(0, self.gran + 1):
                t = (
                    torch.tensor([t / self.gran])
                    .type(torch.float)
                    .requires_grad_(True)
                    .to(self.device)
                )
                curve_t = self(t).reshape(1, self.latent_dim)

                curve_t = a * curve_t - b

                gt_ = self._gt(t)

                # for i in range(self.latent_dim):
                #    gt_[i] = grad(curve_t[0][i], t, create_graph=True)[0]
                # print(torch.norm(gt_, dim=0))
                # gt_ / torch.norm(gt_, dim=0)

                loss += self.loss_function(curve_t, gt_)

            loss /= self.gran

            if loss < best_loss:
                print("better", loss)
                best_curve_model = deepcopy(self)
                best_loss = loss
                best_a, best_b = a, b
                length = best_loss

            # print(loss)
            if _ % 50 == 0:
                print("-----")
                print(loss)

            loss.backward()
            optimizer.step()

        return best_curve_model, best_a, best_b, length


# class Geodesic(nn.Module):
#
#    def __init__(self, metric=None, starting_pos=None, ending_pos=None, starting_velo=None, latent_dim=2, reg=0., granularity=100, early_stopping=100, device='cpu', seed=8):
#        """
#        Geodesic NN model
#
#        Inputs:
#        -------
#
#        metric (function): The metric used to compute the geodesic path
#        starting_pos (tensor): The starting point of the
#        ending_pos (tensor): The ending point of the path
#        starting_velo (tensor) [optional]: The initial velocity (for further use)
#        latent_dim (int): Latent space dimension
#        reg (float): L-2 regularization factor
#        granularity (int): The discretization granularity
#        """
#        torch.manual_seed(seed)
#        nn.Module.__init__(self)
#
#        self.compute_with_ending_point = False
#        self.compute_with_velo = False
#        self.device = device
#
#        if starting_pos is None:
#            starting_pos = torch.zeros(1, latent_dim).to(self.device)
#
#        if ending_pos is None:
#            ending_pos = torch.ones_like(starting_pos).to(self.device)
#
#        else:
#            self.compute_with_ending_point = True
#
#        if starting_velo is None:
#            starting_velo = torch.zeros(1, latent_dim).to(self.device)
#
#        else:
#            self.compute_with_velo = True
#
#        self.starting_pos = starting_pos
#        self.ending_pos = ending_pos
#        self.starting_velo = starting_velo
#
#        self.metric = metric
#        self.reg = reg
#        self.gran = granularity
#        self.latent_dim = latent_dim
#        self.early_stopping = early_stopping
#
#        self.fc1 = nn.Linear(1, 100)
#        self.fc2 = nn.Linear(100, 100)
#        self.fc3 = nn.Linear(100, latent_dim)
#
#    def forward(self, t):
#        """
#        The geodesic model
#        """
#        h1 = torch.tanh(self.fc1(t))
#        h2 = torch.tanh(self.fc2(h1))
#        out = self.fc3(h2)
#
#        return out
#
#    def _gt(self, t):
#        """
#        Comutes the gradient of the curve and evalutate it at t
#        """
#
#
#        h1 = self.fc1(t)
#        h2 = self.fc2(h1)
#        #out = self.fc3(h2)
#
#        dh1_dt = (1 - h1 ** 2).unsqueeze(-1) * self.fc1.weight
#        #print(dh1_dt.shape)
#        dh2_dh1 = (1 - h2 ** 2).unsqueeze(-1) * self.fc2.weight
#        #print(dh2_dh1.shape)
#        dout_dh2 = self.fc3.weight
#        #print(dout_dh2.shape)
#
#        return dout_dh2 @ dh2_dh1 @ dh1_dt
#
#
#    def loss_function(self, curve_t, gt_):
#        #gt_ = torch.zeros(self.latent_dim, 1)
#        #
#        #for i in range(self.latent_dim):
#        #    gt_[i] = grad(curve_t[0][i], t, create_graph=True)[0]
#        #    #print(gt_)
#
#        #print(torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_))
#        return torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_) + self.reg * self.metric(curve_t).norm()
#
#
#    def fit(self, n_epochs=10, lr=1e-2):
#        optimizer = optim.Adam(self.parameters(), lr=lr)
#        self.train()
#
#        best_curve_model = deepcopy(self)
#        best_loss = 1e20
#
#        for epoch in range(n_epochs):
#            optimizer.zero_grad()
#            loss = 0
#            loss += torch.sqrt(self.starting_velo @ self.metric(self.starting_pos) @ self.starting_velo.T)
#
#            curve_t0 = self(torch.tensor([0.]).to(self.device))
#            curve_t1 = self(torch.tensor([1.]).to(self.device))
#
#            if self.compute_with_velo:
#                gt_0 = torch.zeros(1, self.latent_dim)
#
#                for t in range(1):
#                    t = torch.tensor([t]).type(torch.float).requires_grad_(True).to(self.device)
#                    curve_t = self(t).reshape(1, self.latent_dim)
#
#                    for i in range(self.latent_dim):
#                        gt_0[0][i] = grad(curve_t[0][i], t, retain_graph=True)[0]
#                #print(self.starting_velo, gt_0)
#                a = (self.starting_velo / gt_0)
#                b = a * curve_t0
#
#            elif self.compute_with_ending_point:
#                a = (self.starting_pos - self.ending_pos) / (curve_t0 - curve_t1)
#                b = (self.starting_pos * curve_t1 - self.ending_pos * curve_t0) / (curve_t0 - curve_t1)
#
#            for t in range(0, self.gran+1):
#                t = torch.tensor([t / self.gran]).type(torch.float).requires_grad_(True).to(self.device)
#                curve_t = self(t).reshape(1, self.latent_dim)
#
#                curve_t = a * curve_t - b
#
#                # gt_ = torch.zeros(self.latent_dim, 1).to(self.device)
#
#                gt_ = self._gt(t)
#                #print(gt_.shape)
#                #for i in range(self.latent_dim):
#                #    gt_[i] = grad(curve_t[0][i], t, retain_graph=True)[0]
#
#                loss += self.loss_function(curve_t, gt_)
#
#            loss /= self.gran
#
#            if loss < best_loss:
#                es_epoch = 0
#                print('better', loss)
#                best_curve_model = deepcopy(self)
#                best_loss = loss
#                best_a, best_b = a, b
#                length = best_loss
#
#            elif self.early_stopping > 0:
#                es_epoch += 1
#
#                if es_epoch >= self.early_stopping:
#                        print(f'Early Stopping at epoch {epoch} ! Loss did not improve in {self.early_stopping} epochs')
#                        break
#
#
#            #print(loss)
#            if epoch % 50 == 0:
#                print('-----')
#                print(loss)
#
#            loss.backward()
#            optimizer.step()
#
#
#        return best_curve_model, best_a, best_b, length
#


class Geodesic(nn.Module):
    def __init__(
        self,
        metric=None,
        starting_pos=None,
        ending_pos=None,
        starting_velo=None,
        compute_curve=True,
        latent_dim=2,
        reg=0.0,
        granularity=100,
        early_stopping=100,
        device="cpu",
        verbose=True,
        seed=8,
    ):
        """
        Geodesic NN model

        Inputs:
        -------

        metric (function): The metric used to compute the geodesic path
        starting_pos (tensor): The starting point of the
        ending_pos (tensor): The ending point of the path
        starting_velo (tensor) [optional]: The initial velocity (for further use)
        latent_dim (int): Latent space dimension
        reg (float): L-2 regularization factor
        granularity (int): The discretization granularity
        """
        torch.manual_seed(seed)
        nn.Module.__init__(self)

        self.compute_with_ending_point = False
        self.compute_with_velo = False
        self.device = device
        self.verbose = verbose

        if starting_pos is None:
            starting_pos = torch.zeros(1, latent_dim).to(self.device)

        if ending_pos is None:
            ending_pos = torch.ones_like(starting_pos).to(self.device)

        else:
            self.compute_with_ending_point = True

        if starting_velo is None:
            starting_velo = torch.zeros(1, latent_dim).to(self.device)

        else:
            self.compute_with_velo = True

        self.starting_pos = starting_pos
        self.ending_pos = ending_pos
        self.starting_velo = starting_velo

        self.metric = metric
        self.reg = reg
        self.gran = granularity
        self.compute_curve = compute_curve
        self.latent_dim = latent_dim
        self.early_stopping = early_stopping

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, latent_dim)

    def forward(self, t):
        """
        The geodesic model
        """
        h1 = torch.tanh(self.fc1(t))
        h2 = torch.tanh(self.fc2(h1))
        out = self.fc3(h2)

        return out

    def _gt(self, t):
        """
        Comutes the gradient of the curve and evalutate it at t
        """

        h1 = self.fc1(t)
        h2 = self.fc2(h1)
        # out = self.fc3(h2)

        dh1_dt = (1 - h1 ** 2).unsqueeze(-1) * self.fc1.weight
        # print(dh1_dt.shape)
        dh2_dh1 = (1 - h2 ** 2).unsqueeze(-1) * self.fc2.weight
        # print(dh2_dh1.shape)
        dout_dh2 = self.fc3.weight
        # print(dout_dh2.shape)

        gt = dout_dh2 @ dh2_dh1 @ dh1_dt

        # print(torch.norm(gt, dim=1).unsqsqueeze(1).shape)

        return gt / torch.norm(gt, dim=1).unsqueeze(1)

    def loss_function(self, curve_t, gt_):
        # gt_ = torch.zeros(self.latent_dim, 1)
        #
        # for i in range(self.latent_dim):
        #    gt_[i] = grad(curve_t[0][i], t, create_graph=True)[0]
        #    #print(gt_)

        # print(torch.sqrt(gt_.T @ self.metric(curve_t) @ gt_))
        # print(gt_.shape, self.metric(curve_t).shape, curve_t.shape)
        return (
            torch.sqrt(
                torch.transpose(gt_, 1, 2) @ self.metric(curve_t.squeeze(1)) @ gt_
            )
            + self.reg * self.metric(curve_t).norm()
        )

    def fit(self, n_epochs=10, lr=1e-2):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

        if self.compute_curve:
            best_curve_model = deepcopy(self)
        best_loss = 1e20

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = 0
            loss += torch.sqrt(
                self.starting_velo
                @ self.metric(self.starting_pos)
                @ self.starting_velo.T
            )

            curve_t0 = self(torch.tensor([0.0]).to(self.device))
            curve_t1 = self(torch.tensor([1.0]).to(self.device))

            if self.compute_with_velo:
                gt_0 = torch.zeros(1, self.latent_dim)

                for t in range(1):
                    t = (
                        torch.tensor([t])
                        .type(torch.float)
                        .requires_grad_(True)
                        .to(self.device)
                    )
                    curve_t = self(t).reshape(1, self.latent_dim)

                    for i in range(self.latent_dim):
                        gt_0[0][i] = grad(curve_t[0][i], t, retain_graph=True)[0]
                # print(self.starting_velo, gt_0)
                a = self.starting_velo / gt_0
                b = a * curve_t0

            elif self.compute_with_ending_point:
                a = (self.starting_pos - self.ending_pos) / (curve_t0 - curve_t1)
                b = (self.starting_pos * curve_t1 - self.ending_pos * curve_t0) / (
                    curve_t0 - curve_t1
                )

            t = (
                torch.linspace(0, 1, self.gran)
                .requires_grad_(True)
                .reshape(self.gran, 1)
                .to(self.device)
            )

            curve_t = self(t).reshape(self.gran, 1, self.latent_dim)
            curve_t = a * curve_t - b
            gt_ = self._gt(t)

            loss = self.loss_function(curve_t, gt_).sum()

            loss /= self.gran

            if loss < best_loss:
                es_epoch = 0
                if self.verbose:
                    print("better", loss)
                if self.compute_curve:
                    best_curve_model = deepcopy(self)
                best_loss = loss
                best_a, best_b = a, b
                length = best_loss

            elif self.early_stopping > 0:
                es_epoch += 1

                if es_epoch >= self.early_stopping:
                    print(
                        f"Early Stopping at epoch {epoch} ! Loss did not improve in {self.early_stopping} epochs"
                    )
                    break

            # print(loss)
            if epoch % 50 == 0 and self.verbose:
                print("-----")
                print(loss)

            loss.backward()
            optimizer.step()

        if self.compute_curve:
            return best_curve_model, best_a, best_b, length

        return length


def exponential(v, p, metric, n_epochs=100):
    geo = Geodesic(metric=metric, starting_pos=p, starting_velo=v)
    curve, _, _ = geo.fit(n_epochs=n_epochs)
    return curve(torch.tensor([1.0]))


def create_dG_dzi(model, i):
    def dG_dzi(z):

        dG_inv_dzi = -(
            model.M_tens.unsqueeze(0)
            * 2
            * (z[i].unsqueeze(1) - model.centroid_tens[i].unsqueeze(0))
            * torch.exp(
                -torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1)
                ** 2
                / (model.T ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1)

        return -model.G(z) @ dG_inv_dzi @ model.G(z)

    return dG_dzi
