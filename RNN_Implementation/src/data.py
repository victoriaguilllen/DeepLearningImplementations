# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# other libraries
import os


class ElectricDataset(Dataset):
    """
    This class is the dataset loading the data.

    Attr:
        dataset: tensor with all the prices data. Dimensions:
            [number of days, 24].
        past_days: length used for predicting the next value.
    """

    dataset: torch.Tensor
    past_days: int

    def __init__(self, dataset: pd.DataFrame, past_days: int) -> None:
        """
        Constructor of ElectricDataset.

        Args:
            dataset: dataset in dataframe format. It has three columns
                (price, feature 1, feature 2) and the index is
                Timedelta format.
            past_days: number of past days to use for the
                prediction.
        """

        # TODO
        self.dataset = torch.tensor(dataset["Price"].values, dtype=torch.float64).view(
            -1, 24
        )
        self.past_days = past_days

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            number of days in the dataset.
        """

        # TODO
        return self.dataset.shape[0] - self.past_days

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method returns an element from the dataset based on the
        index. It only has to return the prices.

        Args:
            index: index of the element.

        Returns:
            past values, starting to collecting those in the zero
                index. Dimensions: [sequence length, 24].
            current values. Start to collect those in the index
                self.sequence. Dimensions: [24].
        """

        # TODO
        return (
            self.dataset[index : index + self.past_days],
            self.dataset[index + self.past_days],
        )


def load_data(
    save_path: str,
    past_days: int = 7,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    This method returns Dataloaders of the chosen dataset. Use  the
    last 42 weeks of the training dataframe for the validation set.

    Args:
        save_path: path to save the data.
        past_days: number of past days to use for the prediction.
        batch_size: size of batches that wil be created.
        shuffle: indicator of shuffle the data. Defaults to true.
        drop_last: indicator to drop the last batch since it is not
            full. Defaults to False.
        num_workers: num workers for loading the data. Defaults to 0.

    Returns:
        train dataloader.
        val dataloader.
        test dataloader.
        means of price.
        stds of price.
    """

    # TODO
    tr_df, ts_df = download_data(save_path)

    mean_price = tr_df["Price"].mean()
    std_price = tr_df["Price"].std()

    # split train and val
    validation_hours = 43 * 7 * 24
    training_hours = 42 * 7 * 24
    val_df = tr_df.iloc[-validation_hours:]
    tr_df = tr_df.iloc[:-training_hours]

    # concatenate the last week of validation data with the test data
    ts_df = pd.concat([val_df.iloc[-(7 * 24) :], ts_df])

    # normalization
    tr_df["Price"] = (tr_df["Price"] - mean_price) / std_price
    val_df["Price"] = (val_df["Price"] - mean_price) / std_price
    ts_df["Price"] = (ts_df["Price"] - mean_price) / std_price

    # datasets
    tr_dataset: ElectricDataset = ElectricDataset(tr_df, past_days)
    val_dataset: ElectricDataset = ElectricDataset(val_df, past_days)
    test_dataset: ElectricDataset = ElectricDataset(ts_df, past_days)

    # dataloaders
    tr_dataloader: DataLoader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return tr_dataloader, val_dataloader, test_dataloader, mean_price, std_price


def download_data(
    path, years_test=2, begin_test_date=None, end_test_date=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to download data from day-ahead electricity markets.

    Args:
        path: path to save the data
        years_test: year for the test data. Defaults to 2.
        begin_test_date: beginning test date. Defaults to None.
        end_test_date: end test date. Defaults to None.

    Raises:
        IOError: Error when reading dataset with pandas.
        Exception: Starting date for test dataset should be midnight.
        Exception: End date for test dataset should be at 0h or 23h.

    Returns:
        training dataset.
        testing dataset.

    Example:
        >>> from epftoolbox.data import read_data
        >>> df_train, df_test = read_data(path='.', dataset='PJM',
                                            begin_test_date='01-01-2016',
        ...                                 end_test_date='01-02-2016')
        Test datasets: 2016-01-01 00:00:00 - 2016-02-01 23:00:00
        >>> df_train.tail()
                                Price  Exogenous 1  Exogenous 2
        Date
        2015-12-31 19:00:00  29.513832     100700.0      13015.0
        2015-12-31 20:00:00  28.440134      99832.0      12858.0
        2015-12-31 21:00:00  26.701700      97033.0      12626.0
        2015-12-31 22:00:00  23.262253      92022.0      12176.0
        2015-12-31 23:00:00  22.262431      86295.0      11434.0
        >>> df_test.head()
                                Price  Exogenous 1  Exogenous 2
        Date
        2016-01-01 00:00:00  20.341321      76840.0      10406.0
        2016-01-01 01:00:00  19.462741      74819.0      10075.0
        2016-01-01 02:00:00  17.172706      73182.0       9795.0
        2016-01-01 03:00:00  16.963876      72300.0       9632.0
        2016-01-01 04:00:00  17.403722      72535.0       9566.0
        >>> df_test.tail()
                                Price  Exogenous 1  Exogenous 2
        Date
        2016-02-01 19:00:00  28.056729      99400.0      12680.0
        2016-02-01 20:00:00  26.916456      97553.0      12495.0
        2016-02-01 21:00:00  24.041505      93983.0      12267.0
        2016-02-01 22:00:00  22.044896      88535.0      11747.0
        2016-02-01 23:00:00  20.593339      82900.0      10974.0
    """

    dataset: str = "NP"

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset in ["PJM", "NP", "FR", "BE", "DE"]:
        file_path = os.path.join(path, dataset + ".csv")

        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0)
        else:
            url_dir = "https://zenodo.org/records/4624805/files/"
            data = pd.read_csv(url_dir + dataset + ".csv", index_col=0)
            data.to_csv(file_path)
    else:
        try:
            file_path = os.path.join(path, dataset + ".csv")
            data = pd.read_csv(file_path, index_col=0)
        except IOError as e:
            raise IOError("%s: %s" % (path, e.strerror))

    data.index = pd.to_datetime(data.index)

    columns = ["Price"]
    n_exogeneous_inputs = len(data.columns) - 1

    for n_ex in range(1, n_exogeneous_inputs + 1):
        columns.append("Exogenous " + str(n_ex))

    data.columns = columns

    if begin_test_date is None and end_test_date is None:
        number_datapoints = len(data.index)
        number_training_datapoints = number_datapoints - 24 * 364 * years_test

        df_train = data.loc[
            : data.index[0] + pd.Timedelta(hours=number_training_datapoints - 1), :
        ]
        df_test = data.loc[
            data.index[0] + pd.Timedelta(hours=number_training_datapoints) :, :
        ]

    else:
        try:
            begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
            end_test_date = pd.to_datetime(end_test_date, dayfirst=True)
        except ValueError:
            print("Provided values for dates are not valid")

        if begin_test_date.hour != 0:
            raise Exception("Starting date for test dataset should be midnight")
        if end_test_date.hour != 23:
            if end_test_date.hour == 0:
                end_test_date = end_test_date + pd.Timedelta(hours=23)
            else:
                raise Exception("End date for test dataset should be at 0h or 23h")

        print("Test datasets: {} - {}".format(begin_test_date, end_test_date))
        df_train = data.loc[: begin_test_date - pd.Timedelta(hours=1), :]
        df_test = data.loc[begin_test_date:end_test_date, :]

    return df_train, df_test
