"""High-level utility workflows aggregating foundation libraries for experiments.

Single execution:
  python -c "import srcs.utilities; print('Utilities imported successfully')"

Inputs:
  None

Outputs:
  None
"""

from srcs.libraries.seeding import seed_everything
from srcs.libraries.models import build_moons_mlp, ConfigurableMLP, count_parameters
from srcs.libraries.datasets import create_moons_data, load_fashion_mnist_data
from srcs.libraries.training import train_classifier, train_one_epoch, evaluate
from srcs.libraries.metrics import calculate_metrics
from srcs.libraries.serialization import save_checkpoint, load_checkpoint, save_experiment_results

__all__ = [
    "seed_everything",
    "build_moons_mlp",
    "ConfigurableMLP",
    "count_parameters",
    "create_moons_data",
    "load_fashion_mnist_data",
    "train_classifier",
    "train_one_epoch",
    "evaluate",
    "calculate_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "save_experiment_results",
]

