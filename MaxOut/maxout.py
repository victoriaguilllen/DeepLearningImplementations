"""
This module contains the code for MaxOut.
"""

# 3pps
import torch


class MaxOut(torch.nn.Module):
    """
    This class implements the MaxOut layer without loops.

    Attr:
        num_units: Number of linear layers the MaxOut s going to use.
        weight: Tensor object with all the weights of the different
            layers. Dimensions: [num_units, output dim, input dim].
            The dtype is a double.
    """

    # Define attributes
    num_units: int
    weight: torch.Tensor

    def __init__(self, num_units: int, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the class.
        
        Returns:
            None.
        """

        # Call super class
        super().__init__()

        # TODO
        self.num_units = num_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(torch.rand(size=(self.num_units, self.output_dim, self.input_dim))).to(torch.float64)

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method reshapes the inputs in a way later can be used to perform
        matrix multiplication.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Inputs reshaped. Dimensions: [number of units * batch size,
                1, input dim].
        """

        # TODO
        batch, in_dim = inputs.shape
        expanded_inputs = inputs.expand((self.num_units, batch, in_dim)) # [num_units, batch, in_dim]
        flatten_expanded_inputs = expanded_inputs.flatten(start_dim=0, end_dim=1) # [num_units*batch, in_dim]
        reshaped_inputs = flatten_expanded_inputs.view(-1, 1, in_dim) # [num_units*batch, 1, in_dim]
        return reshaped_inputs


    def reshape_weight(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function transform the dimensions of the weight so it can
        be multiplied with bmm.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Weights reshaped.
        """

        # TODO
        # para que los weights se puedan multiplicar con los inputs, tienen que tener dimensiones: [num_units*batch, in_dim, out_dim]
        # los weights ahora tienen dimensiones: [num_units, output dim, input dim]
        # donde num_features son las fimensiones que los pesos

        batch, in_dim = inputs.shape
        weights = self.weight.view(self.num_units, 1, self.output_dim, self.input_dim) # [num_units, 1, out_dim, in_dim]
        expanded_weights = weights.expand((self.num_units, batch, self.output_dim, self.input_dim)) # [num_units, batch, out_dim, in_dim]
        flatten_expanded_weights = expanded_weights.flatten(start_dim=0, end_dim=1) # [num_units*batch, out_dim, in_dim]
        reshaped_weights = flatten_expanded_weights.permute(0, 2, 1) # [num_units*batch, in_dim, out_dim]
        return reshaped_weights


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Output tensor. Dimensions: [batch size, output dim].
        """

        # TODO
        reshaped_inputs = self.reshape_inputs(inputs) #[number of units * batch size,1, input dim].
        reshaped_weights = self.reshape_weight(inputs) # [num_units*batch, in_dim, out_dim]
        outputs_bmm = torch.bmm(reshaped_inputs, reshaped_weights) # [num_units*batch, 1, out_dim]
        pre_outputs = outputs_bmm.view(self.num_units, inputs.shape[0], self.output_dim) # [num_units, batch, out_dim]

        # ahora quiero para cada num unit coger el elemento que sea mayor
        max_out, idx = torch.max(pre_outputs, dim=0, keepdim=True) # [1, batch, out_dim]
        outputs = max_out.squeeze(0) # [batch, out_dim]
        return outputs


