"""Comparative evaluation of ResNet18 Transfer Learning modes on CIFAR-10.

CLI execution:
  python -m srcs.pipelines.day05_task01_transfer_learning --epochs 4 --batch_size 32

Inputs:
  data/ (CIFAR-10 raw dataset archives)

Outputs:
  docs/day05/task01/image_learning_curve.png
  docs/day05/task01/image_confusion_matrix.png
  bin/day05/task01/results/learning_curve.png
  bin/day05/task01/results/confusion_matrix.png
  bin/day05/task01/results/classwise_recall.csv
  bin/day05/task01/models/resnet18_frozen.pt
  bin/day05/task01/models/resnet18_finetune.pt
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets
from sklearn.metrics import confusion_matrix

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.libraries.models import count_parameters

DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 2000
DEFAULT_VAL_SIZE = 500
DEFAULT_EPOCHS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_LAYER4 = 1e-4
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day05", "task01")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day05", "task01", "results")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day05", "task01", "models")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def build_resnet18(finetune_layer4: bool = False,
                   num_classes: int = 10) -> nn.Module:
    """Construct pretrained ResNet18 adapted for CIFAR-10 classification.

    Input:  finetune_layer4 -- whether to unfreeze layer4 weights (default False);
            num_classes     -- integer count of output categories (default 10).
    Output: initialized ResNet18 nn.Module with modified classification head.
    Flow:   instantiate torchvision resnet18 with default pretrained weights,
            freeze all backbone parameters by setting requires_grad to False,
            replace fc layer with linear layer sized for num_classes,
            optionally unfreeze layer4 parameters when finetune_layer4 is True,
            and set all BatchNorm2d modules to eval mode to preserve ImageNet statistics.
    """
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if finetune_layer4:
        for param in net.layer4.parameters():
            param.requires_grad = True
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return net


def prepare_cifar10_loaders(data_dir: str = DEFAULT_DATA_DIR,
                            train_size: int = DEFAULT_TRAIN_SIZE,
                            val_size: int = DEFAULT_VAL_SIZE,
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Partition CIFAR-10 training set deterministically and construct PyTorch DataLoaders.

    Input:  data_dir   -- local cache folder path for CIFAR-10 dataset (default DEFAULT_DATA_DIR);
            train_size -- total count of training samples (default DEFAULT_TRAIN_SIZE);
            val_size   -- total count of validation samples (default DEFAULT_VAL_SIZE);
            batch_size -- batch size for DataLoaders (default DEFAULT_BATCH_SIZE);
            seed       -- random seed for reproducible permutation (default DEFAULT_SEED).
    Output: tuple of train DataLoader, validation DataLoader, and dictionary of subset indices.
    Flow:   retrieve default ImageNet preprocessing transforms from ResNet18 weights,
            instantiate torchvision CIFAR10 dataset from data_dir,
            generate reproducible index permutation via numpy RandomState,
            extract train and validation indices,
            wrap into Subset instances, and package into DataLoaders.
    """
    transform = models.ResNet18_Weights.DEFAULT.transforms()
    full_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    permuted_indices = rng.permutation(len(full_dataset))
    train_indices = permuted_indices[:train_size]
    val_indices = permuted_indices[train_size:train_size + val_size]

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    indices_dict = {
        "train_idx": train_indices,
        "val_idx": val_indices
    }
    return train_loader, val_loader, indices_dict


def set_train_mode(model: nn.Module) -> None:
    """Set model to training mode while freezing BatchNorm statistics.

    Input:  model -- neural network module to configure.
    Output: None.
    Flow:   invoke model.train to activate dropout,
            then iterate through submodules setting all BatchNorm2d layers to eval mode.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def run_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              comp_device: torch.device,
              optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[float, float]:
    """Execute forward pass and optional backward optimization pass across DataLoader batches.

    Input:  model       -- neural network module under evaluation;
            loader      -- DataLoader yielding image and label batches;
            criterion   -- loss calculation module;
            comp_device -- computation device for tensor placement;
            optimizer   -- optional optimizer for weight updates (default None).
    Output: tuple of float mean loss and float classification accuracy.
    Flow:   determine train mode based on presence of optimizer,
            configure training state via set_train_mode or model.eval,
            iterate across minibatches computing forward pass and metric accumulators,
            perform backward step and parameter update when training,
            and return mean loss and categorical accuracy.
    """
    is_train = optimizer is not None
    if is_train:
        set_train_mode(model)
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs = inputs.to(comp_device)
            targets = targets.to(comp_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_resnet_model(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       comp_device: torch.device,
                       epochs: int = DEFAULT_EPOCHS,
                       lr_head: float = DEFAULT_LR_HEAD,
                       lr_layer4: Optional[float] = None,
                       mode: str = "frozen") -> List[Dict[str, Any]]:
    """Train ResNet18 model across multiple epochs and capture progression history.

    Input:  model        -- ResNet18 instance to train;
            train_loader -- DataLoader for training subset;
            val_loader   -- DataLoader for validation subset;
            comp_device  -- target PyTorch compute device;
            epochs       -- total epoch iterations (default DEFAULT_EPOCHS);
            lr_head      -- learning rate for classification head (default DEFAULT_LR_HEAD);
            lr_layer4    -- optional learning rate for layer4 weights (default None);
            mode         -- string label indicating transfer learning setup (default 'frozen').
    Output: list of dictionaries recording metrics per epoch.
    Flow:   move model to compute device, initialize Adam optimizer with custom parameter groups,
            iterate over epochs tracking elapsed time and running train and validation passes,
            log progression to console, and return metrics history list.
    """
    model = model.to(comp_device)
    param_groups = [{"params": model.fc.parameters(), "lr": lr_head}]
    if lr_layer4 is not None:
        layer4_trainable = [p for p in model.layer4.parameters() if p.requires_grad]
        if layer4_trainable:
            param_groups.append({"params": layer4_trainable, "lr": lr_layer4})

    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()
    n_params = count_parameters(model)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, comp_device, optimizer=optimizer)

        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        sec_per_epoch = time.time() - start_time

        val_loss, val_acc = run_epoch(model, val_loader, criterion, comp_device, optimizer=None)

        record = {
            "mode": mode,
            "epoch": epoch,
            "trainable_params": n_params,
            "lr_head": lr_head,
            "lr_layer4": lr_layer4,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "sec_per_epoch": sec_per_epoch
        }
        history.append(record)
        print(f"{mode:<15} | Ep {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% | {sec_per_epoch:.2f}s")

    return history


def collect_predictions(model: nn.Module,
                        loader: DataLoader,
                        comp_device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Gather ground-truth labels and categorical predictions over DataLoader batches.

    Input:  model       -- trained neural network model;
            loader      -- DataLoader yielding evaluation batches;
            comp_device -- computation device.
    Output: tuple of numpy arrays (ground_truth_labels, predicted_labels).
    Flow:   set model to evaluation mode, loop through batches storing targets and argmax predictions,
            and concatenate accumulated tensors into numpy arrays.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(comp_device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return np.array(all_targets), np.array(all_preds)


def compute_confusion_matrices(models_dict: Dict[str, nn.Module],
                               val_loader: DataLoader,
                               comp_device: torch.device) -> Dict[str, np.ndarray]:
    """Compute confusion matrices for each trained model on validation data.

    Input:  models_dict -- dictionary mapping mode names to model instances;
            val_loader  -- DataLoader for validation set;
            comp_device -- computation device.
    Output: dictionary mapping mode names to 10x10 confusion matrix numpy arrays.
    Flow:   iterate over models, collect predictions using collect_predictions,
            and invoke sklearn confusion_matrix with labels 0 to 9.
    """
    cms = {}
    for name, model in models_dict.items():
        y_true, y_pred = collect_predictions(model, val_loader, comp_device)
        cms[name] = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return cms


def compute_classwise_recall(cms: Dict[str, np.ndarray],
                             classes: List[str] = CLASSES) -> pd.DataFrame:
    """Calculate per-class recall rates and performance delta across modes.

    Input:  cms     -- dictionary mapping mode names to confusion matrix arrays;
            classes -- list of string class names (default CLASSES).
    Output: pandas DataFrame indexed by class names with recall rates and difference.
    Flow:   extract diagonal and row sums for each class in confusion matrix,
            calculate recall percentages, and compute delta between finetune and frozen.
    """
    data = {}
    for mode, matrix in cms.items():
        row_sums = matrix.sum(axis=1)
        diag = matrix.diagonal()
        recall = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)
        data[mode] = recall

    df = pd.DataFrame(data, index=classes)
    if "finetune_layer4" in df.columns and "frozen" in df.columns:
        df["finetune - frozen"] = df["finetune_layer4"] - df["frozen"]
    return df


def render_learning_curves(results_dict: Dict[str, List[Dict[str, Any]]],
                           output_paths: List[str]) -> None:
    """Plot and persist comparative loss and accuracy curves for ResNet18 modes.

    Input:  results_dict -- dictionary mapping mode name to history records;
            output_paths -- list of target image file paths.
    Output: None.
    Flow:   create 1x2 subplot layout for loss and accuracy,
            plot train and val trajectories across epochs for frozen and fine-tuned modes,
            configure labels, legends, grids, and write output files.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colors = {
        "frozen": {"train": "#1f77b4", "val": "#aec7e8"},
        "finetune_layer4": {"train": "#d62728", "val": "#ff9896"}
    }

    for mode, records in results_dict.items():
        epochs = [r["epoch"] for r in records]
        train_loss = [r["train_loss"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        train_acc = [r["train_acc"] * 100 for r in records]
        val_acc = [r["val_acc"] * 100 for r in records]

        t_color = colors.get(mode, {}).get("train", "blue")
        v_color = colors.get(mode, {}).get("val", "orange")

        axes[0].plot(epochs, train_loss, label=f"{mode} train", color=t_color, linestyle="--", marker="o")
        axes[0].plot(epochs, val_loss, label=f"{mode} val", color=v_color, linestyle="-", marker="s")

        axes[1].plot(epochs, train_acc, label=f"{mode} train", color=t_color, linestyle="--", marker="o")
        axes[1].plot(epochs, val_acc, label=f"{mode} val", color=v_color, linestyle="-", marker="s")

    axes[0].set_title("Cross-Entropy Loss qua các Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Độ chính xác (Accuracy %) qua các Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def render_confusion_matrices(cms: Dict[str, np.ndarray],
                              classes: List[str] = CLASSES,
                              output_paths: Optional[List[str]] = None) -> None:
    """Plot side-by-side heatmaps of confusion matrices for ResNet18 modes.

    Input:  cms          -- dictionary mapping mode names to confusion matrix arrays;
            classes      -- list of class labels (default CLASSES);
            output_paths -- optional list of file destinations for plot image.
    Output: None.
    Flow:   initialize 1x2 subplots with color map Blues,
            render pixel matrix values as text annotations,
            configure axis ticks and labels, and save figure.
    """
    mode_names = list(cms.keys())
    fig, axes = plt.subplots(1, len(mode_names), figsize=(7 * len(mode_names), 6))
    if len(mode_names) == 1:
        axes = [axes]

    for idx, (mode, matrix) in enumerate(cms.items()):
        ax = axes[idx]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({mode})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

        for row in range(len(classes)):
            for col in range(len(classes)):
                val = matrix[row, col]
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(col, row, str(val), ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_paths:
        for path in output_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_pipeline(data_dir: str = DEFAULT_DATA_DIR,
                 epochs: int = DEFAULT_EPOCHS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute complete Day 5 Task 1 pipeline and generate reporting artifacts.

    Input:  data_dir   -- local directory caching CIFAR-10 data (default DEFAULT_DATA_DIR);
            epochs     -- epoch count per training trial (default DEFAULT_EPOCHS);
            batch_size -- batch size for minibatches (default DEFAULT_BATCH_SIZE);
            seed       -- random seed for reproducibility (default DEFAULT_SEED).
    Output: dictionary packaging benchmark histories, model instances, and recall DataFrame.
    Flow:   detect computing device, partition CIFAR-10 data loaders from data_dir,
            instantiate and train frozen backbone ResNet18,
            instantiate and train fine-tuned layer4 ResNet18,
            persist model checkpoints to models folder,
            render learning curve and confusion matrix comparison plots,
            compute class-wise recall DataFrame and save CSV,
            and return comprehensive execution dictionary.
    """
    comp_device = torch.device(device())
    print("=" * 75)
    print("DAY 5 TASK 1: RESNET18 TRANSFER LEARNING MODES ON CIFAR-10")
    print(f"Device: {comp_device} | Epochs: {epochs} | Data: {data_dir} | Random Seed: {seed}")
    print("=" * 75)

    train_loader, val_loader, _ = prepare_cifar10_loaders(
        data_dir=data_dir,
        train_size=DEFAULT_TRAIN_SIZE,
        val_size=DEFAULT_VAL_SIZE,
        batch_size=batch_size,
        seed=seed
    )

    models_trained = {}
    histories = {}

    print("\n--- Huấn luyện Chế độ 1: Frozen Backbone (chỉ train head fc) ---")
    seed_everything(seed)
    model_frozen = build_resnet18(finetune_layer4=False, num_classes=len(CLASSES))
    histories["frozen"] = train_resnet_model(
        model=model_frozen,
        train_loader=train_loader,
        val_loader=val_loader,
        comp_device=comp_device,
        epochs=epochs,
        lr_head=DEFAULT_LR_HEAD,
        lr_layer4=None,
        mode="frozen"
    )
    models_trained["frozen"] = model_frozen

    print("\n--- Huấn luyện Chế độ 2: Fine-tune Layer 4 (train fc và layer4) ---")
    seed_everything(seed)
    model_finetune = build_resnet18(finetune_layer4=True, num_classes=len(CLASSES))
    histories["finetune_layer4"] = train_resnet_model(
        model=model_finetune,
        train_loader=train_loader,
        val_loader=val_loader,
        comp_device=comp_device,
        epochs=epochs,
        lr_head=DEFAULT_LR_HEAD,
        lr_layer4=DEFAULT_LR_LAYER4,
        mode="finetune_layer4"
    )
    models_trained["finetune_layer4"] = model_finetune

    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    torch.save(model_frozen.state_dict(), os.path.join(DEFAULT_MODELS_DIR, "resnet18_frozen.pt"))
    torch.save(model_finetune.state_dict(), os.path.join(DEFAULT_MODELS_DIR, "resnet18_finetune.pt"))

    curve_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_learning_curve.png"),
        os.path.join(DEFAULT_BIN_DIR, "learning_curve.png")
    ]
    render_learning_curves(histories, curve_paths)

    cms = compute_confusion_matrices(models_trained, val_loader, comp_device)
    cm_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_confusion_matrix.png"),
        os.path.join(DEFAULT_BIN_DIR, "confusion_matrix.png")
    ]
    render_confusion_matrices(cms, classes=CLASSES, output_paths=cm_paths)

    df_recall = compute_classwise_recall(cms, classes=CLASSES)
    os.makedirs(DEFAULT_BIN_DIR, exist_ok=True)
    df_recall.to_csv(os.path.join(DEFAULT_BIN_DIR, "classwise_recall.csv"))

    all_records = []
    for mode_name, records in histories.items():
        all_records.extend(records)
    pd.DataFrame(all_records).to_csv(os.path.join(DEFAULT_BIN_DIR, "training_history.csv"), index=False)

    print("\n" + "=" * 75)
    print("HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ DAY 5 TASK 1")
    print("=" * 75)

    return {
        "models": models_trained,
        "histories": histories,
        "cms": cms,
        "recall": df_recall,
        "val_loader": val_loader,
        "comp_device": comp_device
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 5 Task 1: ResNet18 Transfer Learning on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Minibatch size.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Local dataset directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random reproducibility seed.")
    cli_args = parser.parse_args()

    run_pipeline(
        data_dir=cli_args.data_dir,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        seed=cli_args.seed
    )

