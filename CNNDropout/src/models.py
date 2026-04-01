# deep learning libraries
import torch

# own modules
from src.utils import get_dropout_random_indexes


class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        # TODO
        self.p = p
        self.inplace = inplace
        # super().__init__()
        super(Dropout, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """

        # TODO

        if not self.training or self.p == 0:
            return inputs

        dropout_mask = 1 - get_dropout_random_indexes(inputs.shape, self.p)

        if self.inplace:
            inputs.mul_(dropout_mask).div_(1 - self.p)
            return inputs
        else:
            return inputs * (dropout_mask) / (1 - self.p)


class CNNModel(torch.nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 10,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            input_channels: input channels of the model.
        """

        # TODO

        super().__init__()

        layers: list[torch.nn.Module] = []

        layers.append(
            torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), padding=3, stride=2)
        )
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        m = [
            torch.nn.Conv2d(64, 516, 3, padding=1),
            torch.nn.BatchNorm2d(516),
            torch.nn.ReLU(),
            torch.nn.Conv2d(516, 1024, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
        ]
        layers += m

        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        # Flatten the result of the Adaptive Pooling so we get: [batch, channels]
        layers.append(torch.nn.Flatten(-3))
        layers.append(torch.nn.Dropout(0.6))
        layers.append(torch.nn.Linear(1024, output_channels))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, output_channels].
        """

        # TODO
        return self.net(inputs)
