import typing

import torch
from models.config import ModelConfig
from models.vae_models import RHVAE
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, digits, labels, binarize=False):

        self.labels = labels

        if binarize:
            self.data = (torch.rand_like(digits) < digits).type(torch.float)

        else:
            self.data = digits.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]

        # Load data and get label
        # X = torch.load('data/' + DATA + '.pt')
        y = self.labels[index]

        return X, y


def load_from_config(model_config: ModelConfig) -> RHVAE:

    return RHVAE(model_config)


def load_dict(model: RHVAE) -> RHVAE:

    if not set(["M_tens", "centroids_tens", "model_state_dict"]).issubset(
        model_dict.keys()
    ):
        raise KeyError(
            "Missing key in model dict. Ensure ['M_tens', 'centroids_tens', "
            "'model_state_dict'] are saved"
        )
    model.M_tens = model_dict["M_tens"]
    model.centroids_tens = self.model_dict["centroids_tens"]
    model.G_inv = create_inverse_metric(model)
    model.G = create_metric(model)

    return model


# To rebuild the metric of a trained model (only the centroids and LL where saved)
def create_metric(model, device="cpu"):
    def G(z):
        return torch.inverse(
            (
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
            ).sum(dim=1)
            + model.lbd * torch.eye(model.latent_dim).to(device)
        )

    return G


def create_inverse_metric(model, device="cpu"):
    def G_inv(z):
        return (
            model.M_tens.unsqueeze(0)
            * torch.exp(
                -torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1)
                ** 2
                / (model.T ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1) + model.lbd * torch.eye(model.latent_dim).to(device)

    return G_inv
