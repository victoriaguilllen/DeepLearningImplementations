# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm.auto import tqdm
import signal

# own modules
from src.models import CNNModel
from src.data import load_data
from src.utils import Accuracy, save_model, set_seed

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 10


def main() -> None:
    """
    This function is the main program for the training.
    """

    # TODO
    # ----HYPERPARAMETERS----
    epochs: int = 100
    lr: float = 0.001
    batch_size: int = 64

    # Scheduler
    weight_decay = 1e-1
    # milestones: list[int] = [10, 20, 60]
    # gamma = 0.1
    max_iter = 20
    lr_min = 1e-5
    # -----------------------

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    print("Loading data...")
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(DATA_PATH, batch_size)
    print("DONE")

    # name and writer
    name = f"cnn_{lr}_{batch_size}_{epochs}_AdamW_wd_{weight_decay}_CosSch_data_aug"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = CNNModel(inputs.shape[1], 10).to(device)

    # loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, lr_min)

    # train loop
    for epoch in tqdm(range(epochs)):
        # train step
        train_mean_accuracy = train_step(
            model, train_data, loss, optimizer, writer, epoch, device
        )

        # val step
        val_mean_accuracy = val_step(model, val_data, loss, writer, epoch, device)

        print(
            f"Train and Val. accuracies in epoch {epoch}, lr {scheduler.get_lr()}:",
            (round(train_mean_accuracy, 4), round(val_mean_accuracy, 4)),
        )

        scheduler.step()

    save_model(model, name)

    return None


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # Define metric lists
    # Loss
    losses: list[float] = []
    # Accuracy
    accuracy_object = Accuracy()
    accuracies: list[float] = []

    # Start train
    model.train()

    # TODO
    for images, labels in train_data:
        images = images.to(device)
        labels = labels.to(device)

        # Fordward pass
        outputs = model(images)
        loss_value = loss(outputs, labels)

        # METRICS
        # loss
        losses.append(loss_value.item())
        # accuracy
        accuracy_object.reset()
        accuracy_object.update(outputs, labels)
        accuracy = accuracy_object.compute()
        accuracies.append(accuracy)

        # Backward
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(accuracies))


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """
    # TODO
    # Define metric lists
    # Loss
    losses: list[float] = []
    # Accuracy
    accuracy_object = Accuracy()
    accuracies: list[float] = []

    # Start eval
    model.eval()
    # TODO
    for images, labels in val_data:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss_value = loss(outputs, labels)

        # METRICS
        # loss
        losses.append(loss_value.item())

        # accuracy
        accuracy_object.reset()
        accuracy_object.update(outputs, labels)
        accuracy = accuracy_object.compute()
        accuracies.append(accuracy)

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(accuracies))


if __name__ == "__main__":
    main()
