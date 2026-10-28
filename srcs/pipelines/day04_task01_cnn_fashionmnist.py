"""Day 4 Task 1: Comparative evaluation of MLP and 2-Block CNN on FashionMNIST.

CLI execution:
  python -m srcs.pipelines.day04_task01_cnn_fashionmnist --epochs 5

Inputs:
  data/FashionMNIST (raw torchvision archives)

Outputs:
  docs/day04/task01/image_learning_curve.png
  bin/day04/task01/results/learning_curve.png
  docs/day04/task01/image_confusion_matrix.png
  bin/day04/task01/results/confusion_matrix.png
  Console logs detailing layer shapes, training progression, and performance tables.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.libraries.models import count_parameters, CNN2Blocks
from srcs.libraries.metrics import compute_confusion_matrix

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_BATCH_SIZE = 256
DEFAULT_TRAIN_SIZE = 6000
DEFAULT_VAL_SIZE = 1200
DEFAULT_LR = 1e-3
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day04", "task01")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day04", "task01", "results")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


class FashionMLP(nn.Module):
    """Two-layer multi-layer perceptron baseline for FashionMNIST benchmark.

    Input:  input_dim  -- dimension of flattened input images (default 784);
            hidden_dim -- hidden layer dimension (default 256);
            num_classes -- number of output categories (default 10).
    Output: instantiated FashionMLP nn.Module.
    Flow:   construct sequential pipeline containing flatten, hidden linear, relu, and output linear.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Input:  x -- input image tensor of shape (batch, 1, 28, 28) or (batch, 784).
        Output: logit tensor of shape (batch, num_classes).
        Flow:   propagate input tensor through sequential subnetwork and return output logits.
        """
        return self.net(x)


def prepare_fashion_splits(data_dir: str = DEFAULT_DATA_DIR,
                           train_size: int = DEFAULT_TRAIN_SIZE,
                           val_size: int = DEFAULT_VAL_SIZE,
                           seed: int = DEFAULT_SEED) -> Tuple[Subset, Subset]:
    """Load FashionMNIST dataset and partition deterministic train and validation subsets.

    Input:  data_dir   -- directory path caching raw dataset files;
            train_size -- number of samples allocated for training;
            val_size   -- number of samples allocated for validation;
            seed       -- random seed for deterministic permutation.
    Output: tuple of (train_subset, val_subset).
    Flow:   download training split, compute pseudo-random index permutation with seed,
            and slice into train and validation subset partitions.
    """
    all_train = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_train), generator=generator).tolist()
    train_subset = Subset(all_train, indices[:train_size])
    val_subset = Subset(all_train, indices[train_size:train_size + val_size])
    return train_subset, val_subset


def create_data_loaders(train_ds: Subset,
                        val_ds: Subset,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                        seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader]:
    """Package train and validation subsets into configured PyTorch DataLoaders.

    Input:  train_ds       -- training subset dataset;
            val_ds         -- validation subset dataset;
            batch_size     -- minibatch size for training;
            val_batch_size -- minibatch size for validation;
            seed           -- random seed for DataLoader shuffling.
    Output: tuple of (train_loader, val_loader).
    Flow:   instantiate seeded training DataLoader with shuffle and un-shuffled validation DataLoader.
    """
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def inspect_cnn_shapes(model: nn.Module) -> List[Tuple[str, Tuple[int, ...]]]:
    """Inspect and display tensor shapes through sequential stages of 2-block CNN.

    Input:  model -- instantiated CNN2Blocks model.
    Output: list of tuples mapping layer stage names to output tensor shape tuples.
    Flow:   pass dummy zero tensor through successive model layers under torch.no_grad,
            record tensor shape at each milestone, and return shape history list.
    """
    model.eval()
    shapes = []
    with torch.no_grad():
        x = torch.zeros(1, 1, 28, 28)
        shapes.append(("input", tuple(x.shape)))
        x = model.conv1(x)
        shapes.append(("conv1", tuple(x.shape)))
        x = model.pool(model.relu(x))
        shapes.append(("pool1", tuple(x.shape)))
        x = model.conv2(x)
        shapes.append(("conv2", tuple(x.shape)))
        x = model.pool(model.relu(x))
        shapes.append(("pool2", tuple(x.shape)))
        x = model.flatten(x)
        shapes.append(("flatten", tuple(x.shape)))
        x = model.fc2(model.relu(model.fc1(x)))
        shapes.append(("logits", tuple(x.shape)))
    return shapes


def train_single_epoch(model: nn.Module,
                       loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       criterion: nn.Module,
                       comp_device: torch.device) -> float:
    """Execute one complete training epoch over DataLoader minibatches.

    Input:  model       -- neural network to optimize;
            loader      -- DataLoader yielding training minibatches;
            optimizer   -- optimization algorithm;
            criterion   -- loss criterion;
            comp_device -- computation device.
    Output: sample-averaged scalar training loss float.
    Flow:   set model to train mode, iterate through minibatches, zero gradients,
            perform forward propagation, calculate loss, execute backward pass,
            step optimizer, accumulate sample-weighted loss, and compute epoch mean.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(comp_device)
        yb = yb.to(comp_device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   comp_device: torch.device) -> Tuple[float, float]:
    """Evaluate loss and classification accuracy over validation DataLoader.

    Input:  model       -- neural network to evaluate;
            loader      -- DataLoader yielding evaluation minibatches;
            criterion   -- loss criterion;
            comp_device -- computation hardware device.
    Output: tuple of (validation_loss, validation_accuracy).
    Flow:   set model to evaluation mode, loop through evaluation minibatches,
            accumulate cross entropy loss and correct prediction matches,
            and return mean loss and categorical accuracy fractions.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for xb, yb in loader:
        xb = xb.to(comp_device)
        yb = yb.to(comp_device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total_samples += xb.size(0)
    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return mean_loss, accuracy


@torch.no_grad()
def collect_all_predictions(model: nn.Module,
                            loader: DataLoader,
                            comp_device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ground truth labels and categorical predictions over entire DataLoader.

    Input:  model       -- neural network model;
            loader      -- DataLoader for evaluation;
            comp_device -- computation hardware device.
    Output: tuple of (ground_truth_array, prediction_array).
    Flow:   set model to evaluation mode, pass minibatches without gradient tracking,
            extract argmax predictions, and concatenate batch results into 1D numpy arrays.
    """
    model.eval()
    all_targets = []
    all_preds = []
    for xb, yb in loader:
        xb = xb.to(comp_device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return y_true, y_pred


def run_model_benchmark(model_cls: type,
                        model_name: str,
                        train_ds: Subset,
                        val_ds: Subset,
                        epochs: int = DEFAULT_EPOCHS,
                        lr: float = DEFAULT_LR,
                        seed: int = DEFAULT_SEED,
                        comp_device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Train and evaluate single model configuration, recording epoch histories and timings.

    Input:  model_cls   -- class constructor for neural network;
            model_name  -- identifier label for benchmark configuration;
            train_ds    -- training subset dataset;
            val_ds      -- validation subset dataset;
            epochs      -- training epoch limit;
            lr          -- learning rate for Adam optimizer;
            seed        -- random generator seed;
            comp_device -- computation hardware device.
    Output: dictionary packaging trained model, parameter count, history, and timing metrics.
    Flow:   seed environment, instantiate model and optimizer, initialize history buffers,
            iterate through epochs executing training and validation passes,
            record durations and loss trajectories, and package benchmark results.
    """
    if comp_device is None:
        comp_device = torch.device(device())

    seed_everything(seed)
    model = model_cls().to(comp_device)
    params = count_parameters(model)
    train_loader, val_loader = create_data_loaders(train_ds, val_ds, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    print(f"\n[*] Training {model_name} (Trainable Parameters: {params:,})")
    for epoch in range(1, epochs + 1):
        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        train_loss = train_single_epoch(model, train_loader, optimizer, criterion, comp_device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, comp_device)

        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        print(f"  [{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc * 100:.2f}% | "
              f"Time: {elapsed:.2f}s")

    return {
        "name": model_name,
        "model": model,
        "params": params,
        "history": history,
        "val_loader": val_loader,
        "final_val_acc": history["val_acc"][-1],
        "sec_per_epoch": float(np.mean(history["epoch_times"]))
    }


def render_learning_curves(histories: Dict[str, Dict[str, Any]],
                           epochs: int,
                           output_paths: List[str]) -> None:
    """Render two-panel figure displaying loss and validation accuracy curves.

    Input:  histories    -- dictionary mapping model names to trial history dictionaries;
            epochs       -- integer total epochs;
            output_paths -- list of file destination paths to save figure.
    Output: None.
    Flow:   initialize figure with two subplots, plot train and validation trajectories,
            configure legends, axis titles, and styling, and serialize plot to all destination paths.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ep_range = range(1, epochs + 1)
    colors = {"MLP": "#e67e22", "CNN": "#2980b9"}

    for name, hist in histories.items():
        c = colors.get(name, "#333333")
        axes[0].plot(ep_range, hist["train_loss"], "-o", color=c, linewidth=1.8, label=f"{name} Train")
        axes[0].plot(ep_range, hist["val_loss"], "--s", color=c, linewidth=1.8, label=f"{name} Val")
        axes[1].plot(ep_range, [acc * 100 for acc in hist["val_acc"]], "-o", color=c, linewidth=1.8, label=name)

    axes[0].set_title("Loss across Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].set_title("Validation Accuracy across Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Day 4 Task 1: Learning Curves Comparison (MLP vs 2-Block CNN)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def render_confusion_matrices(models_dict: Dict[str, nn.Module],
                              val_loader: DataLoader,
                              comp_device: torch.device,
                              output_paths: List[str]) -> None:
    """Render side-by-side confusion matrix heatmaps comparing MLP and CNN classifications.

    Input:  models_dict  -- dictionary mapping model name to PyTorch model;
            val_loader   -- validation DataLoader;
            comp_device  -- computation hardware device;
            output_paths -- list of image file destination paths.
    Output: None.
    Flow:   extract model predictions, calculate confusion matrices, create dual heatmap subplots,
            overlay text annotations per cell, format tick labels, and save figures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    for ax, (name, model) in zip(axes, models_dict.items()):
        y_true, y_pred = collect_all_predictions(model, val_loader, comp_device)
        cm = compute_confusion_matrix(y_true, y_pred)
        acc = float((y_true == y_pred).mean())

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"Confusion Matrix: {name} (Val Acc: {acc * 100:.2f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.suptitle("Day 4 Task 1: Confusion Matrices on Validation Set (MLP vs CNN)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def run_pipeline(epochs: int = DEFAULT_EPOCHS,
                 data_dir: str = DEFAULT_DATA_DIR,
                 seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute complete Day 4 Task 1 pipeline and generate all reporting artifacts.

    Input:  epochs   -- training epoch count per trial;
            data_dir -- local directory storing dataset archives (default DEFAULT_DATA_DIR);
            seed     -- random seed for reproducibility.
    Output: dictionary mapping model names to benchmark payloads.
    Flow:   print architecture shapes, partition data from data_dir, benchmark MLP and CNN models,
            render learning curves and confusion matrices, print comparison summary,
            and return comprehensive execution payload.
    """
    comp_device = torch.device(device())
    print("=" * 75)
    print("DAY 4 TASK 1: COMPARATIVE STUDY OF MLP AND 2-BLOCK CNN ON FASHIONMNIST")
    print(f"Device: {comp_device} | Epochs: {epochs} | Data: {data_dir} | Random Seed: {seed}")
    print("=" * 75)

    print("\n--- TODO 1: TENSOR SHAPES THROUGH CNN LAYERS ---")
    demo_cnn = CNN2Blocks()
    shape_milestones = inspect_cnn_shapes(demo_cnn)
    for stage, shape in shape_milestones:
        print(f"  {stage:<15}: {shape}")

    train_ds, val_ds = prepare_fashion_splits(data_dir=data_dir, seed=seed)
    print(f"\nDataset Partitions: Train = {len(train_ds):,} samples | Val = {len(val_ds):,} samples")

    benchmarks = {}
    benchmarks["MLP"] = run_model_benchmark(FashionMLP, "MLP", train_ds, val_ds, epochs=epochs, seed=seed, comp_device=comp_device)
    benchmarks["CNN"] = run_model_benchmark(CNN2Blocks, "CNN", train_ds, val_ds, epochs=epochs, seed=seed, comp_device=comp_device)

    print("\n--- TODO 4: PERFORMANCE COMPARISON TABLE ---")
    print(f"{'Model':<10}{'Parameters':>15}{'Val Accuracy':>16}{'Sec/Epoch (avg)':>18}")
    print("-" * 60)
    for name, res in benchmarks.items():
        print(f"{name:<10}{res['params']:>15,}{res['final_val_acc'] * 100:>15.2f}%{res['sec_per_epoch']:>18.2f}s")

    curve_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_learning_curve.png"),
        os.path.join(DEFAULT_BIN_DIR, "learning_curve.png")
    ]
    render_learning_curves(
        histories={name: res["history"] for name, res in benchmarks.items()},
        epochs=epochs,
        output_paths=curve_paths
    )
    print(f"\n[V] Saved learning curves to: {curve_paths}")

    cm_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_confusion_matrix.png"),
        os.path.join(DEFAULT_BIN_DIR, "confusion_matrix.png")
    ]
    render_confusion_matrices(
        models_dict={name: res["model"] for name, res in benchmarks.items()},
        val_loader=benchmarks["CNN"]["val_loader"],
        comp_device=comp_device,
        output_paths=cm_paths
    )
    print(f"[V] Saved confusion matrices to: {cm_paths}")

    return benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 4 Task 1 standalone CNN pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs per model.")
    args = parser.parse_args()

    run_pipeline(epochs=args.epochs)

