import logging
import math
import os
import typing
from copy import deepcopy
from typing import Tuple

import torch

from pyraug.trainers.training_config import TrainingConfig
from pyraug.models.model_utils import ModelConfig
from pyraug.models.base_vae import BaseVAE


def train_vae(
    epoch: int,
    training_config: TrainingConfig,
    model: BaseVAE,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Tuple[BaseVAE, float]:
    train_loss = 0

    # Set model on training mode
    model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        if training_config.device == "cuda":
            data, _ = data.cuda(), _

        optimizer.zero_grad()

        # forward pass
        (recon_batch, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det) = model(
            data
        )
        # loss computation
        loss = model.loss_function(
            recon_batch, data, z0, z, rho, eps0, gamma, mu, log_var, G_inv, G_log_det
        )

        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()

    model.update_metric()
    # calculate final loss
    train_loss /= len(train_loader)
    return model, train_loss


def training_loops(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    train_loader: torch.utils.data.DataLoader,
    model: BaseVAE,
    optimizer: torch.optim.Adam,
    dir_path: str = None,
    verbose: bool = False,
    logger: logging.Logger = None,
) -> dict:
    best_loss = 1e10
    e = 0
    train_loss_recording = []
    val_loss_recording = []

    for epoch in range(1, training_config.max_epochs):

        model, train_loss = train_vae(
            epoch, training_config, model, train_loader, optimizer
        )

        train_loss_recording.append(train_loss)

        if training_config.verbose and epoch % 100 == 0:
            if verbose:
                logger.info(
                    f"Epoch {epoch} / {training_config.max_epochs}\n"
                    f"- Train loss: {train_loss:.2f}\n"
                    f"- Early Stopping: {e}/{training_config.early_stopping_epochs} (Best: {best_loss:.2f})\n"
                )

        # Early stopping
        if train_loss < best_loss:
            e = 0
            best_loss = train_loss

            # print('Best model saved !')
            # best_model = deepcopy(model)

            best_model_dict = { ## To go in model.save_checkpoint()
                "M": deepcopy(model.M_tens),
                "centroids": deepcopy(model.centroids_tens),
                "model_state_dict": deepcopy(model.state_dict()),
            }

        else:
            e += 1
            if e >= training_config.early_stopping_epochs:
                if verbose:
                    logger.info(
                        f"Training ended at epoch {epoch}! "
                        "(Loss did not improved in {e} epochs)\n"
                    )

                break

        if math.isnan(train_loss):
            logger.error("NaN detected !")
            break

    return best_model_dict
