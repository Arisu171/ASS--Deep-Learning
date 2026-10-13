"""Convenience wrapper re-exporting common seeding and training routines.

CLI execution:
  python -m srcs.libraries.common

Inputs:
  None

Outputs:
  Console confirmation of utility availability.
"""

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import train_classifier, train_one_epoch, evaluate
from srcs.libraries.models import build_moons_mlp, ConfigurableMLP, count_parameters
from srcs.libraries.datasets import create_moons_data, load_fashion_mnist_data
from srcs.libraries.metrics import calculate_metrics

if __name__ == "__main__":
    print("Common library bindings loaded successfully.")

