import torch


def hmc_manifold_sampling(
    model,
    log_pi=None,
    grad_func=None,
    latent_dim=2,
    step_nbr=10,
    z0=None,
    n_lf=15,
    eps_lf=0.05,
    n_samples=1,
    beta_zero_sqrt=1.0,
    device="cpu",
    record_path=False,
    return_acc=False,
    random_start=True,
    verbose=False,
):
    beta_zero_sqrt = torch.tensor([beta_zero_sqrt]).to(device)
    eps_lf = torch.tensor([eps_lf]).to(device)
    n_lf = torch.tensor([n_lf]).to(device)

    if log_pi is None:
        log_pi = log_sqrt_det_G_inv

    if grad_func is None:
        grad_func = grad_log_prop

    acc_nbr = torch.zeros((n_samples, 1)).to(device)
    with torch.no_grad():
        if z0 is None:
            if random_start:
                z0 = torch.distributions.MultivariateNormal(
                    loc=torch.zeros(latent_dim).to(device),
                    covariance_matrix=torch.eye(latent_dim).to(device),
                ).sample(sample_shape=(n_samples,))

            else:
                assert (
                    n_samples <= model.centroids_tens.shape[0]
                ), "Provide n_samples < to centroids numbers"
                idx = torch.randperm(model.centroids_tens.shape[0])
                z0 = model.centroids_tens[idx][:n_samples]

        n_samples = z0.shape[0]

        Z_i = torch.zeros((step_nbr + 1, z0.shape[0], z0.shape[1]))
        Z_i[0] = z0

        beta_sqrt_old = beta_zero_sqrt
        z = z0
        for i in range(step_nbr):
            # print(i)
            if verbose and i % 50 == 0:
                print(f"Generating samples {i} / {step_nbr}")
            gamma = torch.randn_like(z, device=device)
            rho = gamma / beta_zero_sqrt
            H0 = -log_pi(z, model) + 0.5 * torch.norm(rho, dim=1) ** 2
            # print(model.G_inv(z).det())

            for k in range(n_lf):

                g = -grad_func(z, model).reshape(n_samples, latent_dim)
                # step 1
                rho_ = rho - (eps_lf / 2) * g

                # step 2
                z = z + eps_lf * rho_
                g = -grad_func(z, model).reshape(n_samples, latent_dim)
                # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                # step 3
                rho__ = rho_ - (eps_lf / 2) * g

                # tempering
                beta_sqrt = tempering(k + 1, n_lf, beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho__
                beta_sqrt_old = beta_sqrt

            H = -log_pi(z, model) + 0.5 * torch.norm(rho, dim=1) ** 2
            alpha = torch.exp(-H) / (torch.exp(-H0))
            acc = torch.rand(n_samples).to(device)
            moves = torch.tensor(acc < alpha).type(torch.int).reshape(n_samples, 1)
            z = z * moves + (1 - moves) * z0
            acc_nbr += moves
            if record_path:
                Z_i[i] = z

            z0 = z
    if return_acc:
        return z, acc_nbr
    else:
        return z


def tempering(k, K, beta_zero_sqrt):
    beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt

    return 1 / beta_k


def log_sqrt_det_G_inv(z, model):
    return torch.log(torch.sqrt(torch.det(model.G_inv(z))) + 1e-10)


def grad_log_sqrt_det_G_inv(z, model):
    return (
        -0.5
        * torch.transpose(model.G(z), 1, 2)
        @ torch.transpose(
            (
                -2
                / (model.T ** 2)
                * (model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(2)
                @ (
                    model.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (model.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            ).sum(dim=1),
            1,
            2,
        )
    )


def grad_prop_manifold(z, grad_log_density, model):
    # print(grad_log_sqrt_det_G_inv(z, model))
    return grad_log_sqrt_det_G_inv(z, model) + grad_log_density(z).reshape(-1, 2, 1)


def grad_log_prop(z, model):
    def grad_func(z, model):
        return grad_log_sqrt_det_G_inv(z, model)

    return grad_func(z, model)
