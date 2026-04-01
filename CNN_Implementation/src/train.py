# 3pps
import torch
import numpy as np
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# own modules
from src.models import CNNModel
from src.utils import (
    load_imagenette_data,
    Accuracy,
    save_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(device)

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
    # HIPERPARÁMETROS
    epochs = 90
    lr = 0.00008
    batch = 64
    hidden_blocks = 300, 512

    # DATOS
    train_data, val_data, _ = load_imagenette_data(path=DATA_PATH, batch_size=batch)

    # NOMBRE DEL MODELO
    model_name = f"model_2_{lr}_{hidden_blocks}_{batch}_{epochs}_adamW"

    # WRITER
    writer = SummaryWriter(f"runs/{model_name}")

    # DEFINIMOS EL MODELO
    inputs = next(iter(train_data))[0]
    input_channels = inputs.shape[1]
    model = CNNModel(hidden_sizes=hidden_blocks, input_channels=input_channels).to(
        device=device
    )

    # DEFINIMOS LA PÉRDIDA
    loss = torch.nn.CrossEntropyLoss()

    # DEFINIMOS EL OPTIMIZADOR
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # TRAIN LOOP
    for epoch in tqdm(range(epochs)):
        train_accuracy = train_step(
            model, train_data, loss, optimizer, writer, epoch, device
        )
        val_accuracy = val_step(model, val_data, loss, writer, epoch, device)

        print(f"Train accuracy: {train_accuracy}")
        print(f"Validation accuracy: {val_accuracy}")

    # GUARDAMOS EL MODELO
    save_model(model=model, name=model_name)

    return None


def train_step(
    model: torch.nn.Module,
    train_data: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float: 
    # LODD
    losses = []

    # Acuracy
    accuracy_object = Accuracy()
    accuracies = []

    # COMENZAMOS A ENTRENAR
    model.train()

    for images, labels in train_data:
        images = images.to(device)
        labels = labels.to(device)

        # FOWARD
        outputs = model(images)
        loss_value = loss(outputs, labels)

        # METRICS
        losses.append(loss_value.item())

        accuracy_object.reset()
        accuracy_object.update(outputs, labels)
        accuracy = accuracy_object.compute()
        accuracies.append(accuracy)

        # BACKWARD
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return np.mean(accuracies) # type: ignore


def val_step(
    model: torch.nn.Module,
    val_data: torch.utils.data.DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> float: 
    # LOSS
    losses = []

    # ACCURACY
    accuracy_object = Accuracy()
    accuracies = []

    # COMENZAMOS A EVALUAR
    model.eval()

    for images, labels in val_data:
        with torch.no_grad():  # me aseguro que no se están iniciando con el gradiente activado, ya que los datos no los uso para entrenar
            images = images.to(device)
            labels = labels.to(device)

            # FOWARD
            outputs = model(images)
            loss_value = loss(outputs, labels)

        # METRICS
        losses.append(loss_value.item())

        accuracy_object.reset()
        accuracy_object.update(outputs, labels)
        accuracy = accuracy_object.compute()
        accuracies.append(accuracy)

    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)

    return np.mean(accuracies) # type: ignore


if __name__ == "__main__":
    main()
