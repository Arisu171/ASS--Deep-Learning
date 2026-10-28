"""Day 4 Task 2: Comparative ablation study of CNN architectural variants on FashionMNIST.

CLI execution:
  python -m srcs.pipelines.day04_task02_cnn_variants --epochs 5

Inputs:
  data/FashionMNIST (raw dataset archives)

Outputs:
  docs/day04/task02/image_learning_curves.png
  bin/day04/task02/results/learning_curves.png
  docs/day04/task02/image_confusion_matrix_test.png
  bin/day04/task02/results/confusion_matrix_test.png
  docs/day04/task02/image_five_correct.png
  bin/day04/task02/results/five_correct.png
  docs/day04/task02/image_five_wrong.png
  bin/day04/task02/results/five_wrong.png
  Console logs detailing benchmark tables and official test set verification.
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
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day04", "task02")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day04", "task02", "results")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_ablation_configurations() -> List[Dict[str, Any]]:
    """Define ablation configurations varying one single architectural hyperparameter.

    Input:  None.
    Output: list of configuration dictionaries containing hyperparameters.
    Flow:   assemble parameter dictionaries for baseline, kernel5, ch8, ch32, batchnorm, and dropout0.3.
    """
    return [
        {"name": "baseline",   "channels": 16, "kernel_size": 3, "batchnorm": False, "dropout": 0.0},
        {"name": "kernel5",    "channels": 16, "kernel_size": 5, "batchnorm": False, "dropout": 0.0},
        {"name": "ch8",        "channels": 8,  "kernel_size": 3, "batchnorm": False, "dropout": 0.0},
        {"name": "ch32",       "channels": 32, "kernel_size": 3, "batchnorm": False, "dropout": 0.0},
        {"name": "batchnorm",  "channels": 16, "kernel_size": 3, "batchnorm": True,  "dropout": 0.0},
        {"name": "dropout0.3", "channels": 16, "kernel_size": 3, "batchnorm": False, "dropout": 0.3},
    ]


def prepare_datasets(data_dir: str = DEFAULT_DATA_DIR,
                     train_size: int = DEFAULT_TRAIN_SIZE,
                     val_size: int = DEFAULT_VAL_SIZE,
                     seed: int = DEFAULT_SEED) -> Tuple[Subset, Subset, datasets.FashionMNIST]:
    """Load FashionMNIST and create train, validation, and official test datasets.

    Input:  data_dir   -- local cache folder path;
            train_size -- train subset size;
            val_size   -- validation subset size;
            seed       -- random seed for partition.
    Output: tuple of (train_ds, val_ds, test_ds).
    Flow:   instantiate FashionMNIST train and test collections, compute deterministic permutation,
            and extract train and validation subsets while preserving test dataset.
    """
    transform = transforms.ToTensor()
    full_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_ds = Subset(full_train, indices[:train_size])
    val_ds = Subset(full_train, indices[train_size:train_size + val_size])

    return train_ds, val_ds, test_ds


def create_loaders(train_ds: Subset,
                   val_ds: Subset,
                   test_ds: datasets.FashionMNIST,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
                   seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap subsets into train, validation, and test DataLoaders.

    Input:  train_ds       -- training subset;
            val_ds         -- validation subset;
            test_ds        -- full test dataset;
            batch_size     -- batch size for training;
            val_batch_size -- batch size for evaluation;
            seed           -- random seed for training shuffle.
    Output: tuple of (train_loader, val_loader, test_loader).
    Flow:   instantiate seeded training DataLoader and un-shuffled validation and test DataLoaders.
    """
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def train_single_epoch(model: nn.Module,
                       loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       criterion: nn.Module,
                       comp_device: torch.device) -> float:
    """Execute training step across all minibatches of DataLoader.

    Input:  model       -- neural network;
            loader      -- DataLoader yielding training minibatches;
            optimizer   -- gradient optimizer;
            criterion   -- loss function;
            comp_device -- computation device.
    Output: mean training loss float.
    Flow:   switch model to train mode, iterate minibatches, zero gradients, compute forward pass,
            evaluate loss, backward pass, step optimizer, and compute mean loss.
    """
    model.train()
    total_loss = 0.0
    total_count = 0
    for xb, yb in loader:
        xb = xb.to(comp_device)
        yb = yb.to(comp_device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
    return total_loss / total_count if total_count > 0 else 0.0


@torch.no_grad()
def evaluate_dataset(model: nn.Module,
                     loader: DataLoader,
                     criterion: nn.Module,
                     comp_device: torch.device) -> Tuple[float, float]:
    """Compute mean loss and categorical accuracy without gradient updates.

    Input:  model       -- neural network model;
            loader      -- DataLoader yielding evaluation batches;
            criterion   -- loss criterion;
            comp_device -- computation hardware device.
    Output: tuple of (mean_loss, accuracy).
    Flow:   set model to evaluation mode, pass minibatches without gradient tracking,
            accumulate cross entropy loss and correct predictions, and return summary metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_count = 0
    for xb, yb in loader:
        xb = xb.to(comp_device)
        yb = yb.to(comp_device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total_count += xb.size(0)
    mean_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = correct / total_count if total_count > 0 else 0.0
    return mean_loss, accuracy


@torch.no_grad()
def collect_dataset_predictions(model: nn.Module,
                                loader: DataLoader,
                                comp_device: torch.device) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Gather input tensors, ground truth labels, and predictions over entire DataLoader.

    Input:  model       -- neural network model;
            loader      -- DataLoader yielding image batches;
            comp_device -- computation hardware device.
    Output: tuple of (images_tensor, y_true_array, y_pred_array).
    Flow:   iterate through loader minibatches under evaluation mode, extract argmax predictions,
            gather image tensors and label arrays, and return combined tensors and arrays.
    """
    model.eval()
    all_images = []
    all_targets = []
    all_preds = []
    for xb, yb in loader:
        logits = model(xb.to(comp_device))
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds.numpy())
        all_targets.append(yb.numpy())
        all_images.append(xb.cpu())
    x_all = torch.cat(all_images, dim=0)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return x_all, y_true, y_pred


def run_ablation_study(epochs: int = DEFAULT_EPOCHS,
                       lr: float = DEFAULT_LR,
                       data_dir: str = DEFAULT_DATA_DIR,
                       seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Train all 6 CNN variant architectures and coordinate comparative analysis.

    Input:  epochs   -- training epoch limit per trial;
            lr       -- learning rate step size;
            data_dir -- local cache directory for dataset archives (default DEFAULT_DATA_DIR);
            seed     -- random generator seed.
    Output: dictionary packaging benchmark results, best model payload, and loader references.
    Flow:   prepare data partitions from data_dir, loop over ablation configurations initializing models,
            run training sessions, record metrics, identify optimal configuration, and return results.
    """
    comp_device = torch.device(device())
    print("=" * 75)
    print("DAY 4 TASK 2: CNN ARCHITECTURAL VARIANTS ABLATION STUDY")
    print(f"Device: {comp_device} | Epochs: {epochs} | Data: {data_dir} | Random Seed: {seed}")
    print("=" * 75)

    train_ds, val_ds, test_ds = prepare_datasets(data_dir=data_dir, seed=seed)
    train_loader, val_loader, test_loader = create_loaders(train_ds, val_ds, test_ds, seed=seed)

    configs = get_ablation_configurations()
    results = []

    for cfg in configs:
        seed_everything(seed)
        model = CNN2Blocks(
            in_channels=1,
            side=28,
            channels=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            batchnorm=cfg["batchnorm"],
            dropout=cfg["dropout"],
            num_classes=10
        ).to(comp_device)

        params = count_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print(f"\n[*] Training Configuration: {cfg['name']} (Parameters: {params:,})")

        history = []
        for epoch in range(1, epochs + 1):
            if comp_device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            train_loss = train_single_epoch(model, train_loader, optimizer, criterion, comp_device)
            val_loss, val_acc = evaluate_dataset(model, val_loader, criterion, comp_device)

            if comp_device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            history.append((epoch, train_loss, val_loss, val_acc, elapsed))
            print(f"  [{cfg['name']}] Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc * 100:.2f}% | "
                  f"Time: {elapsed:.2f}s")

        results.append({
            "config": cfg,
            "model": model,
            "params": params,
            "history": history,
            "final_train_loss": history[-1][1],
            "final_val_loss": history[-1][2],
            "final_val_acc": history[-1][3],
            "sec_per_epoch": float(np.mean([item[4] for item in history]))
        })

    best_result = max(results, key=lambda r: (round(r["final_val_acc"], 6), -r["final_val_loss"]))
    print(f"\n[+] Best Model Selected by Validation: {best_result['config']['name']} "
          f"(Val Acc: {best_result['final_val_acc'] * 100:.2f}%, Val Loss: {best_result['final_val_loss']:.4f})")

    return {
        "results": results,
        "best_result": best_result,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "comp_device": comp_device
    }


def render_comparative_learning_curves(results: List[Dict[str, Any]],
                                       epochs: int,
                                       output_paths: List[str]) -> None:
    """Render 3-panel figure showing train loss, validation loss, and validation accuracy.

    Input:  results      -- list of trial payload dictionaries;
            epochs       -- total epoch count;
            output_paths -- list of image file destination paths.
    Output: None.
    Flow:   create 3 subplots for train loss, validation loss, and accuracy,
            plot each configuration trajectory, attach titles and legends, and serialize images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=150)
    ep_range = list(range(1, epochs + 1))

    for res in results:
        name = res["config"]["name"]
        hist = res["history"]
        axes[0].plot(ep_range, [h[1] for h in hist], "-o", label=name)
        axes[1].plot(ep_range, [h[2] for h in hist], "-o", label=name)
        axes[2].plot(ep_range, [h[3] * 100 for h in hist], "-o", label=name)

    titles = ["Train Loss across Epochs", "Validation Loss across Epochs", "Validation Accuracy across Epochs"]
    ylabels = ["Loss", "Loss", "Accuracy (%)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ep_range)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Day 4 Task 2: Comparative Learning Curves across CNN Variants", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def render_test_confusion_matrix(model: nn.Module,
                                 test_loader: DataLoader,
                                 comp_device: torch.device,
                                 model_name: str,
                                 output_paths: List[str]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
    """Evaluate optimal model on official test dataset and render annotated confusion matrix.

    Input:  model        -- best trained model instance;
            test_loader  -- official test DataLoader;
            comp_device  -- computation hardware device;
            model_name   -- name label of evaluated model;
            output_paths -- list of destination paths for confusion matrix plot.
    Output: tuple of (x_test_tensor, y_true_array, y_pred_array, test_accuracy).
    Flow:   collect predictions on test split, calculate confusion matrix, render heatmap with annotations,
            save to disk, and return prediction arrays and accuracy.
    """
    x_test, y_true, y_pred = collect_dataset_predictions(model, test_loader, comp_device)
    cm = compute_confusion_matrix(y_true, y_pred)
    test_acc = float((y_true == y_pred).mean())

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Confusion Matrix (Test): {model_name} (Accuracy: {test_acc * 100:.2f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()

    return x_test, y_true, y_pred, test_acc


def render_prediction_samples(x_test: torch.Tensor,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              indices: np.ndarray,
                              title: str,
                              output_paths: List[str]) -> None:
    """Render a horizontal grid of 5 sample images with true and predicted labels.

    Input:  x_test       -- test dataset image tensor;
            y_true       -- ground truth label array;
            y_pred       -- predicted label array;
            indices      -- sample index array of length 5;
            title        -- figure super title;
            output_paths -- list of file destination paths.
    Output: None.
    Flow:   initialize 5-panel subplot row, display grayscale image, set titles with true and predicted labels,
            turn off axis coordinates, adjust layout, and save images.
    """
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), dpi=150)
    for ax, idx in zip(axes, indices):
        img = x_test[idx]
        ax.imshow(img.squeeze(0), cmap="gray")
        t_label = CLASS_NAMES[y_true[idx]]
        p_label = CLASS_NAMES[y_pred[idx]]
        ax.set_title(f"True: {t_label}\nPred: {p_label}", fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def run_pipeline(epochs: int = DEFAULT_EPOCHS,
                 data_dir: str = DEFAULT_DATA_DIR,
                 seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute complete Day 4 Task 2 pipeline and generate all reporting artifacts.

    Input:  epochs   -- training epoch limit per configuration;
            data_dir -- local directory caching dataset archives (default DEFAULT_DATA_DIR);
            seed     -- random generator seed.
    Output: dictionary containing ablation study payload and official test results.
    Flow:   execute ablation trials with data from data_dir, print comparison table, render learning curves,
            evaluate best model on official test set, render test confusion matrix,
            display 5 correct and 5 misclassified image grids, and save artifacts.
    """
    study_data = run_ablation_study(epochs=epochs, data_dir=data_dir, seed=seed)
    results = study_data["results"]
    best_res = study_data["best_result"]
    test_loader = study_data["test_loader"]
    comp_device = study_data["comp_device"]

    print("\n--- TODO 2: COMPARISON TABLE ACROSS CNN CONFIGURATIONS ---")
    print(f"{'Configuration':<14}{'Parameters':>12}{'Train Loss':>12}{'Val Loss':>11}{'Val Acc':>11}{'Sec/Epoch':>12}")
    print("-" * 72)
    for res in results:
        cfg_name = res["config"]["name"]
        print(f"{cfg_name:<14}{res['params']:>12,}{res['final_train_loss']:>12.4f}"
              f"{res['final_val_loss']:>11.4f}{res['final_val_acc'] * 100:>10.2f}%"
              f"{res['sec_per_epoch']:>12.2f}s")

    curves_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_learning_curves.png"),
        os.path.join(DEFAULT_BIN_DIR, "learning_curves.png")
    ]
    render_comparative_learning_curves(results, epochs, curves_paths)
    print(f"\n[V] Saved learning curves to: {curves_paths}")

    test_cm_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_confusion_matrix_test.png"),
        os.path.join(DEFAULT_BIN_DIR, "confusion_matrix_test.png")
    ]
    x_test, y_true, y_pred, test_acc = render_test_confusion_matrix(
        model=best_res["model"],
        test_loader=test_loader,
        comp_device=comp_device,
        model_name=best_res["config"]["name"],
        output_paths=test_cm_paths
    )
    print(f"[V] Saved test confusion matrix to: {test_cm_paths}")
    print(f"  Official Test Accuracy ({best_res['config']['name']}): {test_acc * 100:.2f}%")

    correct_indices = np.where(y_true == y_pred)[0][:5]
    wrong_indices = np.where(y_true != y_pred)[0][:5]

    correct_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_five_correct.png"),
        os.path.join(DEFAULT_BIN_DIR, "five_correct.png")
    ]
    render_prediction_samples(x_test, y_true, y_pred, correct_indices, "5 Correctly Classified Test Samples", correct_paths)
    print(f"[V] Saved 5 correct sample images to: {correct_paths}")

    wrong_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_five_wrong.png"),
        os.path.join(DEFAULT_BIN_DIR, "five_wrong.png")
    ]
    render_prediction_samples(x_test, y_true, y_pred, wrong_indices, "5 Misclassified Test Samples", wrong_paths)
    print(f"[V] Saved 5 misclassified sample images to: {wrong_paths}")

    return study_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 4 Task 2 CNN ablation pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs per variant.")
    args = parser.parse_args()

    run_pipeline(epochs=args.epochs)

