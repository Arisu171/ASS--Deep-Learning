"""Dataset creation, downloading, and data loader preparation pipelines.

CLI execution:
  python -m srcs.libraries.datasets --dataset moons
  python -m srcs.libraries.datasets --dataset fashion_mnist --data_dir data

Inputs:
  data/ (destination folder for downloaded dataset archives)

Outputs:
  Console summary of sample counts, batch dimensions, and tensor shapes.
"""

import os
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets, transforms
from sklearn.datasets import make_moons

from srcs.libraries.seeding import seed_everything

DEFAULT_MOONS_SAMPLES = 1200
DEFAULT_MOONS_NOISE = 0.28
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_BATCH_SIZE = 32
DEFAULT_VAL_BATCH_SIZE = 256
DEFAULT_RANDOM_STATE = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_FASHION_SUBSET = 10000


def create_moons_data(n_samples: int = DEFAULT_MOONS_SAMPLES,
                      noise: float = DEFAULT_MOONS_NOISE,
                      train_ratio: float = DEFAULT_TRAIN_RATIO,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                      random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Any]:
    """Generate synthetic two-moons classification dataset and construct PyTorch DataLoaders.

    Input:  n_samples      -- total number of data points to generate (default DEFAULT_MOONS_SAMPLES);
            noise          -- standard deviation of Gaussian noise added to data (default DEFAULT_MOONS_NOISE);
            train_ratio    -- fraction of data allocated to training split (default DEFAULT_TRAIN_RATIO);
            batch_size     -- batch size for training DataLoader (default DEFAULT_BATCH_SIZE);
            val_batch_size -- batch size for validation DataLoader (default DEFAULT_VAL_BATCH_SIZE);
            random_state   -- integer seed controlling dataset generation (default DEFAULT_RANDOM_STATE).
    Output: dict containing train_loader, val_loader, raw X, raw y, X_train, y_train, X_val, y_val.
    Flow:   generate 2D coordinates using make_moons, convert arrays to float32 and int64 tensors,
            partition into training and validation splits using torch random_split,
            wrap splits into DataLoaders, and package raw coordinates for decision boundary plotting.
    """
    seed_everything(random_state)
    features, labels = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    tensor_features = torch.tensor(features, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(tensor_features, tensor_labels)

    train_size = int(train_ratio * n_samples)
    val_size = n_samples - train_size
    generator = torch.Generator().manual_seed(random_state)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)

    train_indices = train_subset.indices
    val_indices = val_subset.indices

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X": features,
        "y": labels,
        "X_train": features[train_indices],
        "y_train": labels[train_indices],
        "X_val": features[val_indices],
        "y_val": labels[val_indices],
        "input_dim": 2,
        "num_classes": 2,
    }


def load_fashion_mnist_data(data_dir: str = DEFAULT_DATA_DIR,
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                            subset_size: int = DEFAULT_FASHION_SUBSET,
                            random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Any]:
    """Download Fashion-MNIST dataset into data directory and construct DataLoaders.

    Input:  data_dir       -- root folder where dataset files are cached (default DEFAULT_DATA_DIR);
            batch_size     -- batch size for training DataLoader (default DEFAULT_BATCH_SIZE);
            val_batch_size -- batch size for evaluation DataLoader (default DEFAULT_VAL_BATCH_SIZE);
            subset_size    -- optional limit on training sample count for fast benchmarking (default DEFAULT_FASHION_SUBSET);
            random_state   -- integer seed controlling dataset subsampling (default DEFAULT_RANDOM_STATE).
    Output: dict containing train_loader, val_loader, test_loader, input_dim, and class_names.
    Flow:   ensure destination directory exists, download torchvision FashionMNIST train and test sets,
            flatten 28x28 images and normalize pixel values, split training set into train and validation,
            optionally apply subset filtering for responsiveness, and create DataLoaders.
    """
    seed_everything(random_state)
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda tensor: torch.flatten(tensor))
    ])

    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    total_train = len(full_train_dataset)
    val_count = 10000
    train_count = total_train - val_count
    generator = torch.Generator().manual_seed(random_state)
    train_split, val_split = random_split(full_train_dataset, [train_count, val_count], generator=generator)

    if subset_size is not None and subset_size < train_count:
        rng = np.random.RandomState(random_state)
        sampled_indices = rng.choice(train_count, size=subset_size, replace=False)
        train_split = Subset(train_split, sampled_indices)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_dim": 28 * 28,
        "num_classes": 10,
        "class_names": full_train_dataset.classes
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loading CLI.")
    parser.add_argument("--dataset", type=str, default="moons", choices=["moons", "fashion_mnist"], help="Dataset to load.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Destination directory.")
    args = parser.parse_args()

    if args.dataset == "moons":
        data_bundle = create_moons_data()
        print(f"Moons dataset ready: {len(data_bundle['X_train'])} train samples, {len(data_bundle['X_val'])} val samples.")
    else:
        data_bundle = load_fashion_mnist_data(data_dir=args.data_dir, subset_size=2000)
        print(f"Fashion-MNIST ready: {data_bundle['input_dim']} input features, {data_bundle['num_classes']} classes.")

