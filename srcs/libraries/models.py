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


class CNN2Blocks(nn.Module):
    """Two-block Convolutional Neural Network with configurable channels, kernel, norm, and dropout.

    Input:  in_channels -- number of input image color channels (default 1);
            side        -- spatial width and height of square input images (default 28);
            channels    -- initial convolutional filter channels (default 16);
            kernel_size -- square kernel dimension for 2D convolutions (default 3);
            batchnorm   -- boolean flag enabling 2D batch normalization (default False);
            dropout     -- probability of dropout applied before final projection (default 0.0);
            num_classes -- number of target output classification categories (default 10).
    Output: instantiated CNN2Blocks nn.Module.
    Flow:   initialize two convolutional blocks with optional batch normalization, ReLU, and pooling,
            calculate downsampled flattened spatial dimensions, attach linear projection layers,
            and propagate inputs through the convolutional and classification pipeline.
    """

    def __init__(self,
                 in_channels: int = 1,
                 side: int = 28,
                 channels: int = 16,
                 kernel_size: int = 3,
                 batchnorm: bool = False,
                 dropout: float = 0.0,
                 num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.side = side
        self.channels = channels
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.num_classes = num_classes

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, 2 * channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(2 * channels) if batchnorm else nn.Identity()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        flattened_dim = 2 * channels * (side // 4) * (side // 4)
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional blocks, pooling, and dense layers.

        Input:  x -- input image tensor of shape (batch_size, in_channels, side, side).
        Output: raw logit tensor of shape (batch_size, num_classes).
        Flow:   apply conv1, bn1, relu, pool1, then conv2, bn2, relu, pool2, flatten,
                project through fc1, relu, drop, and final fc2 logits layer.
        """
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)



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

