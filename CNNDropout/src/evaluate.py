import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader

import numpy as np

# own modules
from src.data import load_data
from src.utils import (
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

    # TODO
    # load data
    test_data: DataLoader
    _, _, test_data = load_data(DATA_PATH, batch_size=64)

    # define model
    model: RecursiveScriptModule = load_model(f"{name}").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)

    return accuracy


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """

    # TODO
    # Define metric lists
    # Accuracy
    accuracy_object = Accuracy()  # For evaluating accuracy
    accuracies: list[float] = []

    # Start eval mode
    model.eval()

    for images, labels in test_data:
        with torch.no_grad():
            # Pass data and labels to the correct device
            images = images.to(device)
            labels = labels.to(device)

            # Fordward pass
            outputs = model(images)

        # METRICS
        # Save accuracy metric
        accuracy_object.reset()  # reset
        accuracy_object.update(outputs, labels)  # update counts of correct and total
        accuracy = accuracy_object.compute()  # compute
        accuracies.append(accuracy)

    return float(np.mean(accuracies))


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
