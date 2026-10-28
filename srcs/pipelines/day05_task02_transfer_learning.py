"""Comparative evaluation of Data Augmentation and Transfer Learning on CIFAR-10.

CLI execution:
  python -m srcs.pipelines.day05_task02_transfer_learning --epochs 4 --batch_size 32

Inputs:
  data/ (CIFAR-10 raw dataset archives)

Outputs:
  docs/day05/task02/image_learning_curves.png
  docs/day05/task02/image_confusion_matrix.png
  docs/day05/task02/image_examples_cnn_baseline.png
  docs/day05/task02/image_examples_cnn_augmentation.png
  docs/day05/task02/image_examples_mobilenet.png
  bin/day05/task02/results/learning_curves.png
  bin/day05/task02/results/confusion_matrix.png
  bin/day05/task02/results/summary_metrics.csv
  bin/day05/task02/models/cnn_baseline.pt
  bin/day05/task02/models/cnn_augmentation.pt
  bin/day05/task02/models/mobilenetv3_small_pretrained.pt
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
from torchvision import models, datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
DEFAULT_LR = 1e-3
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day05", "task02")
DEFAULT_BIN_DIR = os.path.join(PROJECT_ROOT, "bin", "day05", "task02", "results")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day05", "task02", "models")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


class SmallCNN(nn.Module):
    """Two-block Convolutional Neural Network baseline for CIFAR-10.

    Input:  num_classes -- integer count of target classes (default 10).
    Output: instantiated SmallCNN nn.Module.
    Flow:   construct two convolutional feature extraction blocks with MaxPool2d,
            append dense classifier with dropout and linear layers, and forward input tensor.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass through feature extractor and classifier.

        Input:  x -- input image tensor of shape (batch_size, 3, 32, 32).
        Output: logits tensor of shape (batch_size, num_classes).
        Flow:   pass through convolutional features then dense classifier.
        """
        return self.classifier(self.features(x))


def build_mobilenet_v3_small(num_classes: int = 10) -> nn.Module:
    """Construct MobileNetV3-Small with frozen backbone and new 10-class head.

    Input:  num_classes -- integer count of target categories (default 10).
    Output: configured MobileNetV3-Small nn.Module instance.
    Flow:   load torchvision mobilenet_v3_small with default pretrained weights,
            freeze all parameters across the network,
            replace final classifier linear layer with new linear module,
            and set all BatchNorm2d modules to eval mode.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    net = models.mobilenet_v3_small(weights=weights)
    for param in net.parameters():
        param.requires_grad = False
    in_features = net.classifier[3].in_features
    net.classifier[3] = nn.Linear(in_features, num_classes)
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return net


def get_transforms(augment: bool,
                   pretrained: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build torchvision transform pipelines for training and validation splits.

    Input:  augment    -- boolean flag enabling random crop and flip augmentation;
            pretrained -- boolean flag indicating pretrained ImageNet dimensions.
    Output: tuple of (train_transform, val_transform).
    Flow:   select normalization constants based on pretrained flag,
            construct deterministic validation pipeline,
            optionally prepend random data augmentation transforms for training pipeline,
            and return configured transform pair.
    """
    if not pretrained:
        norm = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            norm
        ])
        if augment:
            train_tf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm
            ])
        else:
            train_tf = val_tf
    else:
        norm = transforms.Normalize(IMNET_MEAN, IMNET_STD)
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm
        ])
        if augment:
            train_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm
            ])
        else:
            train_tf = val_tf

    return train_tf, val_tf


def prepare_partition_indices(data_dir: str = DEFAULT_DATA_DIR,
                              train_size: int = DEFAULT_TRAIN_SIZE,
                              val_size: int = DEFAULT_VAL_SIZE,
                              seed: int = DEFAULT_SEED) -> Tuple[datasets.CIFAR10, np.ndarray, np.ndarray]:
    """Generate fixed train and validation index partitions on CIFAR-10 dataset.

    Input:  data_dir   -- local cache folder path for CIFAR-10 (default DEFAULT_DATA_DIR);
            train_size -- sample count allocated for training (default DEFAULT_TRAIN_SIZE);
            val_size   -- sample count allocated for validation (default DEFAULT_VAL_SIZE);
            seed       -- random seed for partition reproducibility (default DEFAULT_SEED).
    Output: tuple of (raw_cifar_dataset, train_indices_array, val_indices_array).
    Flow:   instantiate raw CIFAR10 dataset without transforms for image viewing,
            generate permutation array with RandomState,
            and split into train and validation integer index slices.
    """
    raw_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(len(raw_dataset))
    train_idx = permuted[:train_size]
    val_idx = permuted[train_size:train_size + val_size]
    return raw_dataset, train_idx, val_idx


def create_dataloaders(data_dir: str,
                       train_idx: np.ndarray,
                       val_idx: np.ndarray,
                       augment: bool,
                       pretrained: bool,
                       batch_size: int = DEFAULT_BATCH_SIZE,
                       seed: int = DEFAULT_SEED) -> Tuple[DataLoader, DataLoader]:
    """Package train and validation subsets into configured PyTorch DataLoaders.

    Input:  data_dir   -- local cache path;
            train_idx  -- array of training sample indices;
            val_idx    -- array of validation sample indices;
            augment    -- boolean flag enabling data augmentation;
            pretrained -- boolean flag specifying ImageNet resolution;
            batch_size -- batch size for minibatches (default DEFAULT_BATCH_SIZE);
            seed       -- random seed for loader shuffling (default DEFAULT_SEED).
    Output: tuple of (train_loader, val_loader).
    Flow:   retrieve transforms via get_transforms,
            instantiate torchvision CIFAR10 instances with appropriate transforms,
            wrap into Subset instances using precomputed indices,
            and package into DataLoaders with seeded generator.
    """
    train_tf, val_tf = get_transforms(augment, pretrained)
    ds_train = datasets.CIFAR10(root=data_dir, train=True, transform=train_tf)
    ds_val = datasets.CIFAR10(root=data_dir, train=True, transform=val_tf)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(ds_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0
    )
    val_loader = DataLoader(
        Subset(ds_val, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader


def set_train_mode(model: nn.Module,
                   frozen_backbone: bool = False) -> None:
    """Set model to training state while preserving BatchNorm statistics for frozen models.

    Input:  model           -- PyTorch model to configure;
            frozen_backbone -- boolean flag indicating if backbone parameters are frozen.
    Output: None.
    Flow:   call model.train to activate dropout,
            and if frozen_backbone is True, set all BatchNorm2d layers to eval mode.
    """
    model.train()
    if frozen_backbone:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   comp_device: torch.device) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate loss, accuracy, and Macro-F1 score over validation DataLoader.

    Input:  model       -- model under evaluation;
            loader      -- DataLoader yielding evaluation batches;
            criterion   -- loss calculation module;
            comp_device -- compute device.
    Output: tuple of (mean_loss, accuracy, macro_f1, all_targets, all_preds).
    Flow:   set model to eval mode, iterate over loader batches without gradients,
            accumulate predictions and targets, compute loss, accuracy, and macro-F1.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(comp_device)
            targets = targets.to(comp_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    mean_loss = running_loss / len(y_true)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return mean_loss, acc, f1, y_true, y_pred


def train_single_configuration(config_name: str,
                               model: nn.Module,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               comp_device: torch.device,
                               frozen_backbone: bool = False,
                               epochs: int = DEFAULT_EPOCHS,
                               lr: float = DEFAULT_LR) -> Dict[str, Any]:
    """Train single experimental configuration and track metrics per epoch.

    Input:  config_name     -- string name identifier for configuration;
            model           -- neural network model instance;
            train_loader    -- training DataLoader;
            val_loader      -- validation DataLoader;
            comp_device     -- target device;
            frozen_backbone -- boolean flag indicating whether backbone is frozen;
            epochs          -- training epochs count (default DEFAULT_EPOCHS);
            lr              -- learning rate (default DEFAULT_LR).
    Output: dictionary packaging model, history records, parameters, and final predictions.
    Flow:   move model to comp_device, filter trainable parameters, initialize Adam optimizer,
            loop through epochs executing train batches, evaluate on validation set,
            record loss, accuracy, and Macro-F1 progression, and return consolidated dictionary.
    """
    model = model.to(comp_device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    param_count = count_parameters(model)

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "sec_per_epoch": []
    }

    final_y_true = np.array([])
    final_y_pred = np.array([])

    for epoch in range(1, epochs + 1):
        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        set_train_mode(model, frozen_backbone=frozen_backbone)
        running_train_loss = 0.0
        train_targets = []
        train_preds = []

        for inputs, targets in train_loader:
            inputs = inputs.to(comp_device)
            targets = targets.to(comp_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_train_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        if comp_device.type == "cuda":
            torch.cuda.synchronize()
        sec_elapsed = time.time() - start_time

        tr_loss = running_train_loss / len(train_targets)
        tr_acc = accuracy_score(train_targets, train_preds)
        tr_f1 = f1_score(train_targets, train_preds, average="macro")

        va_loss, va_acc, va_f1, final_y_true, final_y_pred = evaluate_model(
            model, val_loader, criterion, comp_device
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["sec_per_epoch"].append(sec_elapsed)

        print(f"{config_name:<28} | Ep {epoch}/{epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc * 100:.2f}% F1: {tr_f1:.4f} | Val Loss: {va_loss:.4f} Acc: {va_acc * 100:.2f}% F1: {va_f1:.4f} | {sec_elapsed:.2f}s")

    return {
        "name": config_name,
        "model": model,
        "history": history,
        "param_count": param_count,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_f1": history["val_f1"][-1],
        "avg_sec": float(np.mean(history["sec_per_epoch"])),
        "y_true": final_y_true,
        "y_pred": final_y_pred
    }


def render_comparative_curves(results: List[Dict[str, Any]],
                              output_paths: List[str],
                              epochs: int = DEFAULT_EPOCHS) -> None:
    """Plot 3-subplot comparison across configurations for Loss, Accuracy, and Macro-F1.

    Input:  results      -- list of trial result dictionaries;
            output_paths -- list of file path destinations;
            epochs       -- total epoch count (default DEFAULT_EPOCHS).
    Output: None.
    Flow:   initialize 1x3 subplots for loss, accuracy, and macro-F1,
            plot train dashed lines and val solid lines for each configuration,
            configure labels, legends, grids, and save output images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    epoch_axis = list(range(1, epochs + 1))

    colors = {
        "cnn_baseline": "#1f77b4",
        "cnn_augmentation": "#ff7f0e",
        "mobilenetv3_small_pretrained": "#2ca02c"
    }

    for res in results:
        name = res["name"]
        hist = res["history"]
        color = colors.get(name, "black")

        axes[0].plot(epoch_axis, hist["train_loss"], linestyle="--", color=color, alpha=0.7)
        axes[0].plot(epoch_axis, hist["val_loss"], linestyle="-", marker="o", color=color, label=name)

        axes[1].plot(epoch_axis, [x * 100 for x in hist["train_acc"]], linestyle="--", color=color, alpha=0.7)
        axes[1].plot(epoch_axis, [x * 100 for x in hist["val_acc"]], linestyle="-", marker="s", color=color, label=name)

        axes[2].plot(epoch_axis, hist["train_f1"], linestyle="--", color=color, alpha=0.7)
        axes[2].plot(epoch_axis, hist["val_f1"], linestyle="-", marker="^", color=color, label=name)

    axes[0].set_title("Cross-Entropy Loss qua các Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy (%) qua các Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Macro-F1 Score qua các Epochs", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def render_confusion_matrices(cms: Dict[str, np.ndarray],
                              classes: List[str] = CLASSES,
                              output_paths: Optional[List[str]] = None) -> None:
    """Plot side-by-side 1x3 subplots of confusion matrices for 3 configurations.

    Input:  cms          -- dictionary mapping configuration name to confusion matrix array;
            classes      -- list of category names (default CLASSES);
            output_paths -- optional list of image file destinations.
    Output: None.
    Flow:   initialize 1x3 subplots with Blues colormap,
            render cell text values, configure tick markers, and write file.
    """
    config_names = list(cms.keys())
    fig, axes = plt.subplots(1, len(config_names), figsize=(7 * len(config_names), 6))

    for idx, name in enumerate(config_names):
        ax = axes[idx]
        matrix = cms[name]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight="bold")
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


def render_example_predictions(raw_dataset: datasets.CIFAR10,
                               val_idx: np.ndarray,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               config_name: str,
                               output_paths: List[str],
                               classes: List[str] = CLASSES) -> None:
    """Render 2x5 grid showing 5 correct predictions and 5 misclassifications.

    Input:  raw_dataset  -- torchvision CIFAR10 dataset yielding unnormalized PIL images;
            val_idx      -- array of validation sample indices;
            y_true       -- ground-truth category indices;
            y_pred       -- predicted category indices;
            config_name  -- name of the model configuration;
            output_paths -- destination image file paths;
            classes      -- list of class label strings (default CLASSES).
    Output: None.
    Flow:   find first 5 matching and first 5 mismatching indices,
            create 2x5 subplot figure, draw images with green or red title text,
            and persist figure to output paths.
    """
    correct_indices = np.where(y_true == y_pred)[0][:5]
    wrong_indices = np.where(y_true != y_pred)[0][:5]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    for col_idx, item_idx in enumerate(correct_indices):
        ax = axes[0, col_idx]
        orig_img, _ = raw_dataset[val_idx[item_idx]]
        ax.imshow(orig_img)
        t_label = classes[y_true[item_idx]]
        p_label = classes[y_pred[item_idx]]
        ax.set_title(f"ĐÚNG: {t_label}\nPred: {p_label}", fontsize=9, color="green", fontweight="bold")
        ax.axis("off")

    for col_idx, item_idx in enumerate(wrong_indices):
        ax = axes[1, col_idx]
        orig_img, _ = raw_dataset[val_idx[item_idx]]
        ax.imshow(orig_img)
        t_label = classes[y_true[item_idx]]
        p_label = classes[y_pred[item_idx]]
        ax.set_title(f"SAI: {t_label}\nPred: {p_label}", fontsize=9, color="red", fontweight="bold")
        ax.axis("off")

    axes[0, 0].set_ylabel("Dự đoán ĐÚNG", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Dự đoán SAI", fontsize=11, fontweight="bold")
    fig.suptitle(f"Mẫu ảnh dự đoán Đúng và Sai ({config_name})", fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_pipeline(data_dir: str = DEFAULT_DATA_DIR,
                 epochs: int = DEFAULT_EPOCHS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Execute complete Day 5 Task 2 pipeline and generate all artifacts.

    Input:  data_dir   -- local cache folder path for CIFAR-10 (default DEFAULT_DATA_DIR);
            epochs     -- epoch count per training trial (default DEFAULT_EPOCHS);
            batch_size -- batch size for minibatches (default DEFAULT_BATCH_SIZE);
            seed       -- random seed for reproducibility (default DEFAULT_SEED).
    Output: dictionary packaging trial results, models, and DataFrames.
    Flow:   initialize computing device, prepare deterministic index partition,
            iterate over 3 experimental setups (cnn_baseline, cnn_augmentation, mobilenetv3_small_pretrained),
            train models and record progression metrics,
            save model weights into models directory,
            render 3-subplot comparative learning curves,
            compute confusion matrices and class-wise recall,
            render prediction sample grids for each configuration,
            and export summary tables to CSV.
    """
    comp_device = torch.device(device())
    print("=" * 75)
    print("DAY 5 TASK 2: DATA AUGMENTATION AND TRANSFER LEARNING ON CIFAR-10")
    print(f"Device: {comp_device} | Epochs: {epochs} | Data: {data_dir} | Random Seed: {seed}")
    print("=" * 75)

    raw_dataset, train_idx, val_idx = prepare_partition_indices(
        data_dir=data_dir,
        train_size=DEFAULT_TRAIN_SIZE,
        val_size=DEFAULT_VAL_SIZE,
        seed=seed
    )

    configs = [
        {"name": "cnn_baseline", "augment": False, "pretrained": False, "model_fn": lambda: SmallCNN(len(CLASSES)), "frozen": False},
        {"name": "cnn_augmentation", "augment": True, "pretrained": False, "model_fn": lambda: SmallCNN(len(CLASSES)), "frozen": False},
        {"name": "mobilenetv3_small_pretrained", "augment": True, "pretrained": True, "model_fn": lambda: build_mobilenet_v3_small(len(CLASSES)), "frozen": True}
    ]

    results = []
    cms = {}
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n--- Bắt đầu huấn luyện cấu hình: {cfg_name} ---")
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            augment=cfg["augment"],
            pretrained=cfg["pretrained"],
            batch_size=batch_size,
            seed=seed
        )

        seed_everything(seed)
        model = cfg["model_fn"]()

        trial_data = train_single_configuration(
            config_name=cfg_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            comp_device=comp_device,
            frozen_backbone=cfg["frozen"],
            epochs=epochs,
            lr=DEFAULT_LR
        )
        results.append(trial_data)

        torch.save(model.state_dict(), os.path.join(DEFAULT_MODELS_DIR, f"{cfg_name}.pt"))
        cm = confusion_matrix(trial_data["y_true"], trial_data["y_pred"], labels=list(range(len(CLASSES))))
        cms[cfg_name] = cm

        example_paths = [
            os.path.join(DEFAULT_DOCS_DIR, f"image_examples_{cfg_name}.png"),
            os.path.join(DEFAULT_BIN_DIR, f"examples_{cfg_name}.png")
        ]
        render_example_predictions(
            raw_dataset=raw_dataset,
            val_idx=val_idx,
            y_true=trial_data["y_true"],
            y_pred=trial_data["y_pred"],
            config_name=cfg_name,
            output_paths=example_paths,
            classes=CLASSES
        )

    curves_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_learning_curves.png"),
        os.path.join(DEFAULT_BIN_DIR, "learning_curves.png")
    ]
    render_comparative_curves(results, curves_paths, epochs=epochs)

    cm_paths = [
        os.path.join(DEFAULT_DOCS_DIR, "image_confusion_matrix.png"),
        os.path.join(DEFAULT_BIN_DIR, "confusion_matrix.png")
    ]
    render_confusion_matrices(cms, classes=CLASSES, output_paths=cm_paths)

    df_summary = pd.DataFrame([
        {
            "Cấu hình": res["name"],
            "Số tham số train": f"{res['param_count']:,}",
            "Train Loss": f"{res['history']['train_loss'][-1]:.4f}",
            "Val Loss": f"{res['final_val_loss']:.4f}",
            "Val Accuracy": f"{res['final_val_acc'] * 100:.2f}%",
            "Macro-F1": f"{res['final_val_f1']:.4f}",
            "Giây/epoch": f"{res['avg_sec']:.2f}s"
        }
        for res in results
    ])
    os.makedirs(DEFAULT_BIN_DIR, exist_ok=True)
    df_summary.to_csv(os.path.join(DEFAULT_BIN_DIR, "summary_metrics.csv"), index=False)

    print("\n--- BẢNG SO SÁNH TỔNG HỢP 3 CẤU HÌNH ---")
    print(df_summary.to_string(index=False))

    return {
        "results": results,
        "cms": cms,
        "summary": df_summary,
        "raw_dataset": raw_dataset,
        "train_idx": train_idx,
        "val_idx": val_idx
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 5 Task 2: Data Augmentation and Transfer Learning on CIFAR-10.")
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

