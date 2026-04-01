# deep learning libraries
import torch

# other libraries
import math
from typing import Any


def relu_backward(relu_inputs: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    """
    For simulating the backward of the ReLU

    Args:
        relu_inputs: The inputs that came into the relu for computing its outputs. [*]
        grad_output: The incoming gradients of the computed otputs. [*]

    Return:
        inputs gradients. Dimensions, same as grad_output and relu_inputs
    """

    relu_grad = torch.clone(grad_output)
    relu_grad[relu_inputs <= 0] = 0

    return relu_grad


class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # TODO

        # dimensiones
        batch = inputs.shape[0]
        sequence = inputs.shape[1]
        input_size = inputs.shape[2]
        hidden_size = h0.shape[2]

        # creo los tensorez de salida con todo ceros
        outputs = torch.zeros((batch, sequence, hidden_size), dtype=inputs.dtype)
        pht = torch.zeros((sequence, batch, hidden_size)).type_as(h0)
        ht = torch.zeros((sequence, batch, hidden_size)).type_as(h0)

        # permut los inputs para que sean de la dimension [sequence, batch, input_size] y así poder hacer solo una iteración en el bucle
        inputs_permuted = inputs.permute(1, 0, 2)

        # comenzamos la iteracion
        h = h0
        for i in range(inputs_permuted.shape[0]):

            x = inputs_permuted[i, :, :]

            h = (
                torch.matmul(x, weight_ih.permute(1, 0))
                + bias_ih
                + torch.matmul(h, weight_hh.permute(1, 0))
                + bias_hh
            )

            pht[i, :, :] = h  # gusrdamos para el backward
            h[h <= 0] = 0  # relu
            ht[i, :, :] = h  # gusrdamos para el backward
            outputs[:, i : i + 1, :] = h.permute(1, 0, 2)  # obtenemos outputs

        # Save context
        ctx.save_for_backward(
            inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, pht, ht
        )

        return (outputs, h)

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # TODO
        (
            inputs,
            h0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            pre_activations,
            hidden_states,
        ) = ctx.saved_tensors
        sequence_length = inputs.shape[1]

        grad_output_sequence = grad_output

        grad_hidden = torch.zeros_like(hidden_states).type_as(inputs)
        grad_next_hidden = torch.zeros_like(h0)
        grad_inputs = torch.zeros_like(inputs)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_bias_ih = torch.zeros_like(bias_ih)
        ones_bias_ih = torch.ones_like(bias_ih)
        grad_bias_hh = torch.zeros_like(bias_hh)
        ones_bias_hh = torch.ones_like(bias_hh)

        for t in reversed(range(sequence_length)):
            grad_output_t = grad_output_sequence[:, t, :]
            grad_hidden[t, :, :] = grad_next_hidden + grad_output_t

            relu_grad = relu_backward(pre_activations[t, :, :], grad_hidden[t, :, :])

            grad_next_hidden = torch.matmul(relu_grad, weight_hh)

            # Input gradient:
            grad_inputs[:, t, :] = torch.matmul(relu_grad, weight_ih)

            # Weight_hh gradient:
            prev_hidden = h0.squeeze() if t == 0 else hidden_states[t - 1, :, :]
            grad_weight_hh += torch.matmul(relu_grad.t(), prev_hidden)

            # Weight_ih gradient:
            grad_weight_ih += torch.matmul(relu_grad.t(), inputs[:, t, :])

            # Bias_ih gradients:
            grad_bias_ih += torch.matmul(relu_grad, ones_bias_ih).sum(dim=0)

            # Bias_hh gradients:
            grad_bias_hh += torch.matmul(relu_grad, ones_bias_hh).sum(dim=0)

        # Hidden gradient: dL/dh0
        grad_initial_hidden = grad_next_hidden.unsqueeze(0)

        return (
            grad_inputs,
            grad_initial_hidden,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )


class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None


class MyModel(torch.nn.Module):
    def __init__(
        self,
        inputs_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """

        super().__init__()

        self.gru: torch.nn.Module = torch.nn.GRU(
            inputs_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = (
            torch.nn.Linear(2 * hidden_size, output_size)
            if bidirectional
            else torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        # TODO

        gru, _ = self.gru(inputs)
        gru_last = gru[:, -1, :]
        dropout = self.dropout(gru_last)
        outputs = self.linear(dropout)

        return outputs
