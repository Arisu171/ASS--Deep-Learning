"""Neural network model definitions and parameter introspection utilities.

CLI execution:
  python -m srcs.libraries.models --hidden_dims 32 32 --activation relu

Inputs:
  None

Outputs:
  Console display of model architecture layers and total parameter count.
"""

import argparse
from typing import List, Union
import torch
from torch import nn

DEFAULT_INPUT_DIM = 2
DEFAULT_OUTPUT_DIM = 2
DEFAULT_HIDDEN_DIMS = (32, 32)
DEFAULT_ACTIVATION = "relu"
DEFAULT_DROPOUT_RATE = 0.0


def resolve_activation(activation_name: str) -> nn.Module:
    """Instantiate a PyTorch activation layer based on a string identifier.

    Input:  activation_name -- lowercase string name ('relu', 'tanh', 'sigmoid').
    Output: corresponding initialized nn.Module activation instance.
    Flow:   normalize identifier string to lowercase, match against supported
            activation types, and return the corresponding PyTorch activation module.
    """
    normalized = activation_name.lower().strip()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation function: {activation_name}. Choose relu, tanh, or sigmoid.")


class ConfigurableMLP(nn.Module):
    """Multi-Layer Perceptron with configurable depth, width, activation, and dropout.

    Input:  input_dim    -- integer number of input features (default DEFAULT_INPUT_DIM);
            hidden_dims  -- list or tuple of integers specifying hidden layer sizes (default DEFAULT_HIDDEN_DIMS);
            output_dim   -- integer number of target classes or outputs (default DEFAULT_OUTPUT_DIM);
            activation   -- string activation name (default DEFAULT_ACTIVATION);
            dropout_rate -- probability of dropout regularization (default DEFAULT_DROPOUT_RATE).
    Output: instantiated ConfigurableMLP nn.Module.
    Flow:   construct linear input layer, sequentially append hidden layers with activations
            and optional dropout, append output linear layer, and store layer chain in nn.Sequential.
    """

    def __init__(self,
                 input_dim: int = DEFAULT_INPUT_DIM,
                 hidden_dims: Union[List[int], tuple] = DEFAULT_HIDDEN_DIMS,
                 output_dim: int = DEFAULT_OUTPUT_DIM,
                 activation: str = DEFAULT_ACTIVATION,
                 dropout_rate: float = DEFAULT_DROPOUT_RATE):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(resolve_activation(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-layer perceptron.

        Input:  x -- input tensor of shape (batch_size, input_dim).
        Output: raw logit tensor of shape (batch_size, output_dim).
        Flow:   pass input tensor through sequential layer chain and return resulting logits.
        """
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    """Calculate the total number of trainable parameters in a neural network.

    Input:  model -- PyTorch nn.Module instance.
    Output: integer total count of trainable parameters.
    Flow:   iterate through model parameters, filter by requires_grad,
            sum up numel counts for all trainable weight and bias tensors.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_moons_mlp(hidden_dim: int,
                    input_dim: int = DEFAULT_INPUT_DIM,
                    output_dim: int = DEFAULT_OUTPUT_DIM,
                    activation: str = DEFAULT_ACTIVATION) -> nn.Module:
    """Construct standard two-hidden-layer MLP architecture for make_moons benchmark.

    Input:  hidden_dim -- number of neurons per hidden layer;
            input_dim  -- number of input features (default DEFAULT_INPUT_DIM);
            output_dim -- number of output classes (default DEFAULT_OUTPUT_DIM);
            activation -- string activation name (default DEFAULT_ACTIVATION).
    Output: nn.Sequential model matching onclass and homework assignment topology.
    Flow:   build sequential network with two hidden linear layers, intermediate
            activation modules, and a final classification projection layer.
    """
    act_fn = resolve_activation(activation)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        act_fn,
        nn.Linear(hidden_dim, hidden_dim),
        resolve_activation(activation),
        nn.Linear(hidden_dim, output_dim)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model construction inspection CLI.")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[32, 32], help="Hidden layer dimensions.")
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, help="Activation function.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT_RATE, help="Dropout probability.")
    args = parser.parse_args()

    test_model = ConfigurableMLP(
        input_dim=DEFAULT_INPUT_DIM,
        hidden_dims=args.hidden_dims,
        output_dim=DEFAULT_OUTPUT_DIM,
        activation=args.activation,
        dropout_rate=args.dropout
    )
    print(test_model)
    print(f"Total trainable parameters: {count_parameters(test_model)}")

