# 3pps
import torch
from torch.jit import RecursiveScriptModule
import torch.utils
import torch.utils.data
import numpy as np

# own modules
from src.utils import (
    load_imagenette_data,
    Accuracy,
    load_model,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    test_data: torch.utils.data.DataLoader
    _, _, test_data = load_imagenette_data(DATA_PATH, batch_size=64)

    model = load_model(f"{name}").to(device)

    accuracy = test_step(model, test_data, device)

    return accuracy


def test_step(
    model: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    # ACCURACY
    accuracy_object = Accuracy()
    accuracies = []

    # COMENZAMOS A EVALUAR
    model.eval()

    for images, labels in test_data:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            # FOWARD
            outputs = model(images)

        ## METRICS
        accuracy_object.reset()
        accuracy_object.update(outputs, labels)
        accuracy = accuracy_object.compute()
        accuracies.append(accuracy)

    return float(np.mean(accuracies))


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
