# standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F


class ReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        ctx.save_for_backward(inputs)
        return torch.maximum(
            torch.tensor(0, dtype=inputs.dtype, device=inputs.device), inputs
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        inputs = ctx.saved_tensors[0]
        # input_gradients = torch.empty_like(inputs)
        input_gradients = torch.where(
            inputs <= 0, 0, 1
        )  # meter como arg 'other=input_gradients (el creado arriba)'

        return input_gradients * grad_output


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.fn = ReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs)


class LinearFunction(torch.autograd.Function):
    """
    This class implements the forward and backward of the Linear layer.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is the forward pass of the Linear layer.

        Args:
            ctx: Contex for saving elements for the backward.
            inputs: Inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: Bias tensor. Dimensions: [output dimension].

        Returns:
            Outputs tensor. Dimensions: [batch, output dimension].
        """

        # TODO
        ctx.save_for_backward(inputs, weight)
        return torch.matmul(inputs, weight.T) + bias

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            Inputs gradients. Dimensions: [batch, input dimension].
            Weights gradients. Dimensions: [output dimension,
                input dimension].
            Bias gradients. Dimension: [output dimension].
        """

        # TODO
        inputs, weights = ctx.saved_tensors
        input_gradients: torch.Tensor = torch.matmul(grad_output, weights)
        weight_gradients = torch.matmul(grad_output.T, inputs)
        bias_gradients = torch.sum(grad_output, dim=0)
        return input_gradients, weight_gradients, bias_gradients


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight torch parameter. Dimensions: [output dimension,
            input dimension].
        bias: Bias torch parameter. Dimensions: [output dimension].
        fn: Autograd function.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_dim))

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = LinearFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.bias)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: Weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: Bias of the layer. Dimensions: [output channels].
            padding: padding parameter.
            stride: stride parameter.

        Returns:
            Output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """
        ctx.save_for_backward(inputs, weight, bias)
        ctx.padding = padding
        ctx.stride = stride

        kernel_size = weight.shape[2]
        out_height = int((inputs.shape[2] + 2 * padding - kernel_size) / stride + 1)
        out_width = int((inputs.shape[3] + 2 * padding - kernel_size) / stride + 1)

        unfold = torch.nn.Unfold(
            kernel_size=weight.shape[-1], padding=padding, stride=stride
        )
        unfolded_inputs: torch.Tensor = unfold(inputs)

        # quiero multiplicar matricualmente (matmul) el kernel con el unfolded input.

        # las dimensiones del kernel y del unfolded input son:
        # -> unfolded_input: [batch, kernel_size**2 * input_chanels, out_height * out_width]
        # -> kernel: [output_channels, input channels, kernel_size, kernel_size]

        # reordeno de forma que  se puedan multiplicar
        # como son solo quiero intercambiar dos dimensiones podria unar transpose,
        # y como solo tengo dos dimensiones también podría hacer .T o .t()
        unfolded_outputs = torch.matmul(
            unfolded_inputs.permute(0, 2, 1),
            weight.view(weight.size(0), -1).permute(1, 0),
        ).permute(0, 2, 1)

        # añado el bias aunque también lo podría hacer despues del fold
        unfolded_outputs += bias.view(1, bias.shape[0], 1)

        # para hacer el fold necesito un tamaño de kernel, sin embargo debido a las multiplicaciones la parte de k^2 ha desaparecido.
        # por lo tanto ahora el kernel que necesitaremos es de tamaño (1, 1)
        fold = torch.nn.Fold(
            output_size=(out_height, out_width), kernel_size=(1, 1)
        )  # devuelve [N, C, output_size[0], ...]
        outputs: torch.Tensor = fold(
            unfolded_outputs
        )  # outputs: [batch, output_channels, out_height, out_width]

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: Context for loading elements needed in the backward.
            grad_output: Outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            Inputs gradients. Dimensions: [batch, input channels,
                height, width].
            Weight gradients. Dimensions: [output channels,
                input channels, kernel size, kernel size].
            Bias gradients. Dimensions: [output channels].
            None.
            None.
        """

        # TODO
        inputs, weights, bias = ctx.saved_tensors
        padding, stride = ctx.padding, ctx.stride

        # WEIGHTS GRADIENT
        # Operando nos damos cuenta de que el resultado es una convolucion de los inputs y y el output kernel
        # Si metemos dentro de esta ecuación (width + 2*padding - kernel size) / stride + 1] el output_kernel_size (=(width + 2*padding - kernel size) / stride + 1])
        # y igualamos el resultado a Kernel_size, nos da que el padding y el stride de esta convolución son 0 y 1 respectivamente (así se cumplen las dimensiones).
        # -> inputs : [batch, input channels, height, width]
        # -> grad_output : [batch, output channels,new kernel height size, new kernel width size]
        unfold = torch.nn.Unfold(kernel_size=grad_output.shape[2])
        unfolded_inputs = unfold(
            inputs.permute(1, 0, 2, 3)
        )  # unfold([input_channels, barch, h, w]) = [input_channels, H*W*batch, k^2]

        unfolded_outputs = torch.matmul(
            grad_output.view(
                grad_output.shape[1], -1
            ),  # [output_channels, W x H x batch]
            unfolded_inputs,
        ).permute(
            1, 0, 2
        )  # [output_channel, input_channel, k^2]

        ## OTRA FORMA
        # unfolded_outputs = torch.matmul(
        #     unfolded_inputs.permute(0,2,1),
        #     grad_output.view(-1, grad_output.shape[1])
        # ).permute(2,0,1)

        fold = torch.nn.Fold(
            (weights.shape[2], weights.shape[3]), (1, 1)
        )  # input: (N, C*kernel_size, L) , en nuestro caso podemos ver que kernel_size es (1, 1)
        weights_gradients = fold(unfolded_outputs)

        # INPUT GRADIENT
        unfold = torch.nn.Unfold(kernel_size=(1, 1))
        unfolded_inputs_gradients = torch.matmul(
            weights.view(weights.size(0), -1).t(), unfold(grad_output)
        )

        fold = torch.nn.Fold(
            output_size=inputs.shape[2:],
            kernel_size=weights.shape[2:],
            padding=padding,
            stride=stride,
        )
        inputs_gradients = fold(unfolded_inputs_gradients)

        # BIAS GRADIENT
        # (dLoss/dBias = dLoss/dOutput * dOutput/dBias) -> dOutput/dBias = I -> dLoss/dBias = dLoss/dOutput
        # -> grad_output : [batch, output channels, new kernel height size, new kernel width size]
        # -> bias_gradient : [output channels]
        bias_gradient = grad_output.sum(dim=(0, 2, 3))

        return inputs_gradients, weights_gradients, bias_gradient, None, None


class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight pytorch parameter. Dimensions: [output channels,
            input channels, kernel size, kernel size].
        bias: Bias torch parameter. Dimensions: [output channels].
        padding: Padding parameter.
        stride: Stride parameter.
        fn: Autograd function.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: Input dimension.
            output_channels: Output dimension.
            kernel_size: Kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).

    Attributes:
        net: Sequential containing all the layers.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of the Block class. It is composed of
        3x(conv(kernel=3) + ReLU). Only the second conv
        will have stride. Use a Sequential for encapsulating all the
        layers. Clue: convs may have padding to fit into the correct
        dimensions.

        Args:
            input_channels: Input channels for Block.
            output_channels: Output channels for Block.
            stride: Stride only for the second convolution of the
                Block.
        """

        # TODO
        super().__init__()

        # Usaré los módulos de torch
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass.

        Args:
            inputs: Inputs batch of tensors.
                Dimensions: [batch, input_channels, height, width].

        Returns:
            Outputs batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        # TODO
        return self.block(inputs)


class CNNModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],  # canales de output de mis bloques
        input_channels: int = 3,
        output_channels: int = 10,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: Output channel dimensions of the Blocks.
            input_channels: Input channels of the model.
        """

        # TODO
        super().__init__()

        # Primeras capas
        layers: list[torch.nn.Module] = []
        layers.append(
            torch.nn.Conv2d(input_channels, 32, kernel_size=(7, 7), padding=3, stride=2)
        )
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Capas intermedias usando Block
        last_output_size = 32
        for hidden_size in hidden_sizes:
            layers.append(Block(last_output_size, hidden_size, 2))
            last_output_size = hidden_size

        # Adaptive-Pooling (average sin tener que poner las dim de entrada, solo le doy lo que quiero que salga)
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))

        # Flatten [batch, channels, 1, 1]
        layers.append(torch.nn.Flatten(-3))  # [batch, channels]

        # Capa final
        layers.append(torch.nn.Linear(last_output_size, output_channels))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the
        neural network.

        Args:
            inputs: Inputs batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            Outputs batch of logits. Dimensions: [batch,
                output_channels].
        """

        # TODO
        return self.net(inputs)
