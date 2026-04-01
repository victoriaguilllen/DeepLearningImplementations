# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model, parameters_to_double

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # TODO
    # ----HYPERPARAMETERS----
    epochs: int = 400
    lr: float = 0.001
    batch_size: int = 256
    past_days: int = 7
    hidden_size: int = 100
    num_layers: int = 1
    dropout: float = 0.4
    bidirectional = False

    # Scheduler
    weight_decay = 1e-1
    max_iter = 100
    lr_min = 1e-6

    # Load data
    print("Loading data...")
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _, mean, std = load_data(
        DATA_PATH, batch_size=batch_size, past_days=past_days
    )
    print("Fata loaded")

    open("nohup.out", "w").close()
    # name = f"\
    #     GRU_layers_{num_layers}_bidirectional_{bidirectional}_{hidden_size}\
    #     _{lr}_{batch_size}_{epochs}_{past_days},{dropout}_\
    #     L1Loss_AdamW_wd_{weight_decay}_CosSch_2"
    name = "best_model"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = MyModel(
        inputs.shape[2], 24, hidden_size, num_layers, dropout, bidirectional
    ).to(device)
    parameters_to_double(model)

    # loss
    loss: torch.nn.Module = torch.nn.L1Loss()

    # optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # shceduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, lr_min)

    for epoch in tqdm(range(epochs)):
        train_mean_loss, train_mean_mae = train_step(
            model, train_data, mean, std, loss, optimizer, writer, epoch, device
        )

        val_mean_loss, val_mean_mae = val_step(
            model, val_data, mean, std, loss, scheduler, writer, epoch, device
        )

        print(
            f"Train and Val. accuracy in epoch {epoch}, lr {scheduler.get_lr()}:",
            (round(train_mean_mae, 4), round(val_mean_mae, 4)),
        )

        scheduler.step()

    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
