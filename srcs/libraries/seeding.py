"""Random seed initialization utilities for deterministic reproducibility.

CLI execution:
  python -m srcs.libraries.seeding --seed 42

Inputs:
  None

Outputs:
  Console confirmation of initialized seed states.
"""

import os
import random
import argparse
import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across standard library, numpy, and pytorch backends.

    Input:  seed -- integer random seed value (default DEFAULT_SEED).
    Output: None.
    Flow:   set python hash seed, set random seed, set numpy seed,
            set torch manual seed, set cuda manual seed if available,
            and configure cudnn deterministic flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic seeding utility.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed integer.")
    args = parser.parse_args()
    seed_everything(seed=args.seed)
    print(f"Random seed successfully set to: {args.seed}")

