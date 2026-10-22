"""Day 3 Task 1: Pipeline for standard PyTorch 5-step training loop and learning rate exploration.

CLI execution:
  python -m srcs.pipelines.day03_task01_training_loop --epochs 5

Inputs:
  data/FashionMNIST (dataset archives)

Outputs:
  docs/day03/task01/image_training_loop_curves.png
  bin/day03/task01/results/training_loop_curves.png
  docs/day03/task01/image_confusion_matrix.png
  bin/day03/task01/results/confusion_matrix.png
  Console logs detailing parameters and epoch-by-epoch loss trajectories.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any, Tuple, Optional
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
from srcs.libraries.training import train_one_epoch, evaluate_classifier, device, collect_predictions
from srcs.libraries.models import count_parameters
from srcs.libraries.metrics import compute_confusion_matrix
from srcs.utilities.visualization import plot_confusion_matrix

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_BATCH_SIZE = 256
DEFAULT_TRAIN_SAMPLES = 6000
DEFAULT_VAL_SAMPLES = 1000
DEFAULT_LEARNING_RATES = (0.01, 0.001, 0.0001)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DOCS_PLOT_PATH = os.path.join(PROJECT_ROOT, "docs", "day03", "task01", "image_training_loop_curves.png")
DEFAULT_BIN_PLOT_PATH = os.path.join(PROJECT_ROOT, "bin", "day03", "task01", "results", "training_loop_curves.png")
DEFAULT_DOCS_CM_PATH = os.path.join(PROJECT_ROOT, "docs", "day03", "task01", "image_confusion_matrix.png")
DEFAULT_BIN_CM_PATH = os.path.join(PROJECT_ROOT, "bin", "day03", "task01", "results", "confusion_matrix.png")
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def build_fashion_mlp(input_dim: int = 784,
                      hidden_dim: int = 256,
                      output_dim: int = 10) -> nn.Module:
    """Instantiate a sequential multi-layer perceptron for FashionMNIST classification.

    Input:  input_dim  -- dimension of flattened input images (default 784);
            hidden_dim -- number of hidden units in linear layer (default 256);
            output_dim -- number of output class logits (default 10).
    Output: nn.Sequential model containing flatten, linear, relu, and projection layers.
    Flow:   assemble sequential pipeline comprising flatten, hidden projection, activation, and logits.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def prepare_fashion_loaders(data_dir: str = DEFAULT_DATA_DIR,
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                            num_train: int = DEFAULT_TRAIN_SAMPLES,
                            num_val: int = DEFAULT_VAL_SAMPLES,
                            seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader]:
    """Download FashionMNIST dataset and partition into training and validation DataLoaders.

    Input:  data_dir       -- local directory caching raw FashionMNIST archives;
            batch_size     -- batch size for training DataLoader;
            val_batch_size -- batch size for validation DataLoader;
            num_train      -- sample count allocated to training partition;
            num_val        -- sample count allocated to validation partition;
            seed           -- random generator seed value.
    Output: tuple of (train_loader, val_loader).
    Flow:   instantiate tensor transforms, fetch train and test datasets, create deterministic
            subsets based on slice indexing, and package partitions into DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_full = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    val_full = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_subset = Subset(train_full, list(range(min(num_train, len(train_full)))))
    val_subset = Subset(val_full, list(range(min(num_val, len(val_full)))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def run_training_session(learning_rate: float,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         epochs: int = DEFAULT_EPOCHS,
                         seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute standard 5-step training loop across specified epochs for a learning rate trial.

    Input:  learning_rate -- floating point learning rate step size;
            train_loader  -- DataLoader yielding training batches;
            val_loader    -- DataLoader yielding evaluation batches;
            epochs        -- total number of training epochs to execute;
            seed          -- integer random seed for reproducibility.
    Output: dictionary packaging trained model, loss histories, accuracy logs, and timing metrics.
    Flow:   seed execution environment, build MLP, instantiate CrossEntropyLoss and Adam optimizer,
            iterate through epochs running 5-step training routine, evaluate validation loss and accuracy,
            record trajectories, measure elapsed duration, and return comprehensive trial results.
    """
    seed_everything(seed)
    comp_device = device()
    model = build_fashion_mlp().to(comp_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "lr": learning_rate,
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=comp_device
        )

        val_metrics = evaluate_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=comp_device
        )

        epoch_duration = time.perf_counter() - epoch_start
        history["epoch_times"].append(epoch_duration)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  [LR={learning_rate}] Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
              f"Time: {epoch_duration:.2f}s")

    total_time = time.perf_counter() - start_time
    history["total_time"] = total_time
    history["model"] = model
    return history


def plot_learning_rate_comparison(histories: List[Dict[str, Any]],
                                  output_paths: List[str]) -> None:
    """Plot multi-panel training loss, validation loss, and accuracy across learning rates.

    Input:  histories    -- list of trial dictionaries containing metrics across epochs;
            output_paths -- list of file paths to persist the rendered figure.
    Output: None.
    Flow:   construct 3-panel figure, plot train loss curves, plot validation loss curves,
            plot accuracy trajectories, configure styles and legends, and serialize to all paths.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    colors = {0.01: "#d9534f", 0.001: "#337ab7", 0.0001: "#5cb85c"}

    for hist in histories:
        lr = hist["lr"]
        epochs_range = range(1, len(hist["train_loss"]) + 1)
        color = colors.get(lr, "#333333")

        axes[0].plot(epochs_range, hist["train_loss"], marker="o", color=color, linewidth=1.8, label=f"LR = {lr}")
        axes[1].plot(epochs_range, hist["val_loss"], marker="s", color=color, linewidth=1.8, label=f"LR = {lr}")
        axes[2].plot(epochs_range, [acc * 100 for acc in hist["val_acc"]],
                     marker="^", color=color, linewidth=1.8, label=f"LR = {lr}")

    axes[0].set_title("Training Loss across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Validation Loss across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    axes[2].set_title("Validation Accuracy across Epochs", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.suptitle("Day 3 Task 1: Learning Rate Comparison on FashionMNIST", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def compare_learning_rates(learning_rates: Tuple[float, ...] = DEFAULT_LEARNING_RATES,
                           epochs: int = DEFAULT_EPOCHS,
                           seed: int = DEFAULT_SEED,
                           output_paths: Optional[List[str]] = None,
                           cm_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Coordinate multi-trial learning rate benchmark and produce comparative artifact visualizations.

    Input:  learning_rates -- sequence of learning rate floats to explore (default DEFAULT_LEARNING_RATES);
            epochs         -- number of training epochs per trial (default DEFAULT_EPOCHS);
            seed           -- random seed value for deterministic initialization;
            output_paths   -- destination paths for comparison loss curve plot;
            cm_paths       -- destination paths for confusion matrix plot.
    Output: list of trial result dictionaries.
    Flow:   display hardware target and parameter counts, prepare data loaders,
            execute training sessions across learning rates, generate loss curve visualization,
            extract validation predictions from best model, compute and plot confusion matrix,
            persist image artifacts, and return history list.
    """
    if output_paths is None:
        output_paths = [DEFAULT_DOCS_PLOT_PATH, DEFAULT_BIN_PLOT_PATH]
    if cm_paths is None:
        cm_paths = [DEFAULT_DOCS_CM_PATH, DEFAULT_BIN_CM_PATH]

    comp_device = device()
    sample_model = build_fashion_mlp()
    total_params = count_parameters(sample_model)

    print("=" * 75)
    print("DAY 3 TASK 1: TRAINING LOOP AND LEARNING RATE EXPLORATION")
    print(f"Device: {comp_device.upper()} | Model Parameters: {total_params:,}")
    print("=" * 75)

    train_loader, val_loader = prepare_fashion_loaders(seed=seed)
    histories = []

    for lr in learning_rates:
        print(f"\n[*] Starting Training Session with Learning Rate: {lr}")
        hist = run_training_session(
            learning_rate=lr,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            seed=seed
        )
        histories.append(hist)

    plot_learning_rate_comparison(histories, output_paths)

    best_hist = max(histories, key=lambda h: h["val_acc"][-1])
    print(f"\n[*] Generating Confusion Matrix for Best Model (LR = {best_hist['lr']})...")
    y_true, y_pred = collect_predictions(best_hist["model"], val_loader, comp_device)
    cm = compute_confusion_matrix(y_true, y_pred)
    for path in cm_paths:
        plot_confusion_matrix(
            cm=cm,
            class_names=FASHION_CLASSES,
            title=f"Fashion-MNIST Confusion Matrix (Best LR = {best_hist['lr']})",
            normalize=False,
            save_path=path
        )
    print(f"  Saved confusion matrix to: {cm_paths}")

    return histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3 Task 1 training loop pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epoch count per LR trial.")
    args = parser.parse_args()

    results = compare_learning_rates(epochs=args.epochs)
    print("\n[V] Day 3 Task 1 pipeline execution completed successfully.")

