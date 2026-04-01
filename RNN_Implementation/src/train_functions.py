# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """
    # TODO
    # Define metric lists
    # Loss
    losses: list[float] = []
    # mae
    maes: list[float] = []

    # Start train
    model.train()

    for sequences, targets in train_data:
        sequences = sequences.to(device)
        targets = targets.to(device)

        # Fordward
        outputs = model(sequences)
        loss_value = loss(outputs, targets)

        # Unnormalize outputs and target
        outputs = outputs * std + mean
        targets = targets * std + mean

        # METRICS
        # loss
        losses.append(loss_value.item())
        # maes
        diff_abs = torch.abs(outputs - targets)
        mae = float(torch.mean(diff_abs))
        maes.append(mae)

        # Backward
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/mae", np.mean(maes), epoch)

    return float(np.mean(losses)), float(np.mean(maes))


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # TODO
    # Define metric lists
    # Loss
    losses: list[float] = []
    # maes
    maes: list[float] = []

    # Start eval
    model.eval()

    for sequences, targets in val_data:
        with torch.no_grad():
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Forward
            outputs = model(sequences)
            loss_value = loss(outputs, targets)

        # Unnormalize outputs and target
        outputs = outputs * std + mean
        targets = targets * std + mean

        # METRICS
        # loss
        losses.append(loss_value.item())

        # mae
        diff_abs = torch.abs(outputs - targets)
        mae = float(torch.mean(diff_abs))
        maes.append(mae)

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/mae", np.mean(maes), epoch)

    return float(np.mean(losses)), float(np.mean(maes))


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    # TODO
    # Define metric lists
    # Accuracies
    maes: list[float] = []

    # Start eval mode
    model.eval()

    for sequences, targets in test_data:
        with torch.no_grad():
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Fordward
            outputs = model(sequences)

        # Unnormalize outputs and target
        outputs = outputs * std + mean
        targets = targets * std + mean

        # METRICS
        # mae
        diff_abs = torch.abs(outputs - targets)
        mae = float(torch.mean(diff_abs))
        maes.append(mae)

    return float(np.mean(maes))
