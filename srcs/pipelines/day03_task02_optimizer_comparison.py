"""Day 3 Task 2: Comparative evaluation of optimization algorithms and Early Stopping on FashionMNIST.

CLI execution:
  python -m srcs.pipelines.day03_task02_optimizer_comparison --epochs 12

Inputs:
  data/FashionMNIST (dataset archives)

Outputs:
  docs/day03/task02/image_optimizer_learning_curves.png
  bin/day03/task02/results/optimizer_learning_curves.png
  docs/day03/task02/image_confusion_matrix.png
  bin/day03/task02/results/confusion_matrix.png
  Console logs showing convergence milestones, best epochs, and Early Stopping activations.
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
from srcs.libraries.training import device, evaluate_classifier, EarlyStopping, collect_predictions
from srcs.libraries.metrics import compute_confusion_matrix
from srcs.utilities.visualization import plot_comparative_confusion_matrices

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 128
DEFAULT_VAL_BATCH_SIZE = 256
DEFAULT_TRAIN_SAMPLES = 8000
DEFAULT_VAL_SAMPLES = 1500
DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DOCS_PLOT_PATH = os.path.join(PROJECT_ROOT, "docs", "day03", "task02", "image_optimizer_learning_curves.png")
DEFAULT_BIN_PLOT_PATH = os.path.join(PROJECT_ROOT, "bin", "day03", "task02", "results", "optimizer_learning_curves.png")
DEFAULT_DOCS_CM_PATH = os.path.join(PROJECT_ROOT, "docs", "day03", "task02", "image_confusion_matrix.png")
DEFAULT_BIN_CM_PATH = os.path.join(PROJECT_ROOT, "bin", "day03", "task02", "results", "confusion_matrix.png")
FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def build_optimizer_mlp(input_dim: int = 784,
                        hidden_dim: int = 128,
                        output_dim: int = 10) -> nn.Module:
    """Build two-layer multi-layer perceptron architecture for optimizer benchmarks.

    Input:  input_dim  -- number of input pixel features (default 784);
            hidden_dim -- hidden layer neuron width (default 128);
            output_dim -- number of output classes (default 10).
    Output: instantiated nn.Sequential PyTorch model.
    Flow:   assemble sequential model with flatten, linear projection, relu, and linear head.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def prepare_optimizer_loaders(data_dir: str = DEFAULT_DATA_DIR,
                              batch_size: int = DEFAULT_BATCH_SIZE,
                              val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                              num_train: int = DEFAULT_TRAIN_SAMPLES,
                              num_val: int = DEFAULT_VAL_SAMPLES,
                              seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader]:
    """Prepare balanced DataLoaders for optimizer comparison benchmarks.

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


def create_optimizer(optimizer_name: str,
                     model: nn.Module) -> Tuple[torch.optim.Optimizer, str, float]:
    """Instantiate optimizer algorithm and configure default hyperparameter specifications.

    Input:  optimizer_name -- string identifier ('sgd', 'momentum', 'adam');
            model          -- PyTorch model whose parameters will be optimized.
    Output: tuple of (optimizer instance, descriptive title, learning rate).
    Flow:   normalize identifier, match against supported algorithms, initialize optimizer,
            and return configured instance along with descriptive metadata.
    """
    norm_name = optimizer_name.lower().strip()
    if norm_name == "sgd":
        lr = 0.05
        return torch.optim.SGD(model.parameters(), lr=lr), "SGD (lr=0.05)", lr
    if norm_name == "momentum":
        lr = 0.03
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9), "SGD + Momentum (lr=0.03, beta=0.9)", lr
    if norm_name == "adam":
        lr = 0.001
        return torch.optim.Adam(model.parameters(), lr=lr), "Adam (lr=0.001)", lr
    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose 'sgd', 'momentum', or 'adam'.")


def run_optimizer_trial(optimizer_name: str,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        epochs: int = DEFAULT_EPOCHS,
                        patience: int = DEFAULT_PATIENCE,
                        min_delta: float = DEFAULT_MIN_DELTA,
                        seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Train MLP with specified optimizer and monitor validation loss using Early Stopping.

    Input:  optimizer_name -- string optimizer identifier;
            train_loader   -- DataLoader yielding training batches;
            val_loader     -- DataLoader yielding evaluation batches;
            epochs         -- total epochs allocated for training;
            patience       -- consecutive epochs allowed without loss improvement;
            min_delta      -- minimum loss change considered an improvement;
            seed           -- random seed for deterministic initialization.
    Output: dictionary packaging training history, timing metrics, and Early Stopping records.
    Flow:   seed environment, instantiate model and optimizer, initialize EarlyStopping monitor,
            iterate through epochs updating weights and tracking validation loss, trigger early stop
            if plateau occurs, restore best weights, and return structured performance history.
    """
    seed_everything(seed)
    comp_device = device()
    model = build_optimizer_mlp().to(comp_device)
    criterion = nn.CrossEntropyLoss()
    optimizer, opt_label, lr = create_optimizer(optimizer_name, model)

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=True
    )

    history = {
        "name": optimizer_name,
        "label": opt_label,
        "lr": lr,
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_times": [],
        "stopped_epoch": epochs,
        "early_stopped": False,
        "best_epoch": 0,
        "best_val_loss": float("inf")
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(comp_device)
            targets = targets.to(comp_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        val_metrics = evaluate_classifier(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=comp_device
        )

        epoch_duration = time.perf_counter() - epoch_start
        history["epoch_times"].append(epoch_duration)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  [{opt_label}] Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy'] * 100:.2f}% | "
              f"Time: {epoch_duration:.2f}s")

        should_stop = early_stopping.step(val_metrics["loss"], model, epoch=epoch)
        if should_stop:
            print(f"  --> Early stopping triggered at Epoch {epoch}. Best epoch was {early_stopping.best_epoch}.")
            history["stopped_epoch"] = epoch
            history["early_stopped"] = True
            break

    total_time = time.perf_counter() - start_time
    history["total_time"] = total_time
    history["best_epoch"] = early_stopping.best_epoch
    history["best_val_loss"] = early_stopping.best_loss
    history["model"] = model

    final_metrics = evaluate_classifier(model, val_loader, criterion, comp_device)
    history["final_val_loss"] = final_metrics["loss"]
    history["final_val_acc"] = final_metrics["accuracy"]

    return history


def plot_optimizer_comparison(histories: List[Dict[str, Any]],
                              output_paths: List[str]) -> None:
    """Plot multi-panel training loss, validation loss, and accuracy across optimizer algorithms.

    Input:  histories    -- list of trial dictionaries containing metrics across epochs;
            output_paths -- list of file paths to persist the rendered figure.
    Output: None.
    Flow:   construct 3-panel figure, plot train loss curves, plot validation loss curves,
            plot accuracy trajectories, mark early stopping events, and serialize to all paths.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    color_map = {
        "sgd": "#d9534f",
        "momentum": "#337ab7",
        "adam": "#2ca02c"
    }

    for hist in histories:
        name = hist["name"]
        color = color_map.get(name, "#333333")
        label = hist["label"]
        epochs_range = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs_range, hist["train_loss"], marker="o", color=color, linewidth=1.8, label=label)
        axes[1].plot(epochs_range, hist["val_loss"], marker="s", color=color, linewidth=1.8, label=label)
        axes[2].plot(epochs_range, [acc * 100 for acc in hist["val_acc"]],
                     marker="^", color=color, linewidth=1.8, label=label)

        if hist["early_stopped"]:
            best_ep = hist["best_epoch"]
            best_loss = hist["best_val_loss"]
            axes[1].scatter([best_ep], [best_loss], color="red", s=100, zorder=5,
                            label=f"{name.upper()} Early Stop (Ep {best_ep})")

    axes[0].set_title("Training Loss across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Validation Loss across Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    axes[2].set_title("Validation Accuracy across Epochs", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=9)

    plt.suptitle("Day 3 Task 2: Comparative Study of Optimizers (SGD, Momentum, Adam) and Early Stopping",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def compare_optimizers(optimizers: Tuple[str, ...] = ("sgd", "momentum", "adam"),
                       epochs: int = DEFAULT_EPOCHS,
                       patience: int = DEFAULT_PATIENCE,
                       seed: int = DEFAULT_SEED,
                       output_paths: Optional[List[str]] = None,
                       cm_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Coordinate benchmarking of optimization algorithms combined with Early Stopping.

    Input:  optimizers   -- sequence of optimizer names to evaluate;
            epochs       -- maximum epoch limit for training;
            patience     -- Early Stopping patience threshold;
            seed         -- random generator seed;
            output_paths -- destination paths for comparative learning curves;
            cm_paths     -- destination paths for comparative confusion matrices.
    Output: list of trial result dictionaries.
    Flow:   prepare data loaders, iterate through optimizers running training sessions,
            produce visual comparison figure, gather validation predictions across optimizers,
            compute normalized confusion matrices, render comparative heatmaps,
            persist image artifacts, and return history records.
    """
    if output_paths is None:
        output_paths = [DEFAULT_DOCS_PLOT_PATH, DEFAULT_BIN_PLOT_PATH]
    if cm_paths is None:
        cm_paths = [DEFAULT_DOCS_CM_PATH, DEFAULT_BIN_CM_PATH]

    comp_device = device()
    print("=" * 75)
    print("DAY 3 TASK 2: OPTIMIZER COMPARISON AND EARLY STOPPING BENCHMARK")
    print(f"Device: {comp_device.upper()} | Dataset: FashionMNIST | Max Epochs: {epochs} | Patience: {patience}")
    print("=" * 75)

    train_loader, val_loader = prepare_optimizer_loaders(seed=seed)
    histories = []

    for opt_name in optimizers:
        print(f"\n[*] Evaluating Optimizer: {opt_name.upper()}")
        hist = run_optimizer_trial(
            optimizer_name=opt_name,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience,
            seed=seed
        )
        histories.append(hist)

    plot_optimizer_comparison(histories, output_paths)

    matrices = {}
    for hist in histories:
        y_true, y_pred = collect_predictions(hist["model"], val_loader, comp_device)
        cm = compute_confusion_matrix(y_true, y_pred)
        matrices[hist["label"]] = cm

    print("\n[*] Generating Comparative Confusion Matrices for Optimizers...")
    for path in cm_paths:
        plot_comparative_confusion_matrices(
            matrices_dict=matrices,
            class_names=FASHION_CLASSES,
            title="Day 3 Task 2: Comparative Confusion Matrices (Normalized by True Class)",
            normalize=True,
            save_path=path
        )
    print(f"  Saved comparative confusion matrices to: {cm_paths}")

    return histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3 Task 2 optimizer comparison pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epoch limit.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    args = parser.parse_args()

    results = compare_optimizers(epochs=args.epochs, patience=args.patience)
    print("\n[V] Day 3 Task 2 pipeline execution completed successfully.")

