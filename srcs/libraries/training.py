"""Training and evaluation routines for neural network classifiers.

CLI execution:
  python -m srcs.libraries.training

Inputs:
  None

Outputs:
  Console log of sample training progress and final validation performance.
"""

import time
import argparse
from typing import Dict, Any, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader

from srcs.libraries.metrics import calculate_metrics

DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: str = DEFAULT_DEVICE) -> Dict[str, float]:
    """Execute a single training epoch across all batches in the DataLoader.

    Input:  model     -- PyTorch neural network to train;
            loader    -- DataLoader yielding training minibatches (features, targets);
            criterion -- loss function module;
            optimizer -- gradient optimization algorithm;
            device    -- computation device string (default DEFAULT_DEVICE).
    Output: dictionary containing mean training loss, accuracy, and Macro-F1.
    Flow:   set model to train mode, iterate through minibatches, transfer tensors to device,
            zero gradients, perform forward pass, compute loss, backward pass, step optimizer,
            accumulate predictions and compute epoch summary metrics.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    total_samples = len(loader.dataset)
    epoch_loss = running_loss / total_samples
    cat_preds = torch.cat(all_preds)
    cat_targets = torch.cat(all_targets)
    metrics = calculate_metrics(cat_targets, cat_preds)

    return {
        "loss": float(epoch_loss),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"])
    }


def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: str = DEFAULT_DEVICE) -> Dict[str, float]:
    """Evaluate neural network performance on an evaluation dataset without gradient tracking.

    Input:  model     -- PyTorch neural network to evaluate;
            loader    -- DataLoader yielding evaluation minibatches;
            criterion -- loss function module;
            device    -- computation device string (default DEFAULT_DEVICE).
    Output: dictionary containing evaluation loss, accuracy, and Macro-F1.
    Flow:   set model to eval mode, enter torch.no_grad context, iterate through minibatches,
            accumulate loss, gather predicted class labels, and compute final metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

    total_samples = len(loader.dataset)
    eval_loss = running_loss / total_samples
    cat_preds = torch.cat(all_preds)
    cat_targets = torch.cat(all_targets)
    metrics = calculate_metrics(cat_targets, cat_preds)

    return {
        "loss": float(eval_loss),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"])
    }


def train_classifier(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     epochs: int = DEFAULT_EPOCHS,
                     lr: float = DEFAULT_LR,
                     weight_decay: float = DEFAULT_WEIGHT_DECAY,
                     device: str = DEFAULT_DEVICE,
                     verbose: bool = True) -> Dict[str, Any]:
    """Train a classifier model across multiple epochs and record epoch history.

    Input:  model        -- PyTorch neural network;
            train_loader -- DataLoader for training split;
            val_loader   -- DataLoader for validation split;
            epochs       -- total training epochs (default DEFAULT_EPOCHS);
            lr           -- learning rate for Adam optimizer (default DEFAULT_LR);
            weight_decay -- L2 regularization coefficient (default DEFAULT_WEIGHT_DECAY);
            device       -- computation hardware target (default DEFAULT_DEVICE);
            verbose      -- whether to print per-epoch progress logs (default True).
    Output: history dictionary containing epoch trajectories, final metrics, and total elapsed runtime.
    Flow:   move model to device, initialize CrossEntropyLoss and Adam optimizer,
            run training and validation loop for specified epochs, record performance history,
            measure elapsed execution time, and return structured history dictionary.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "epoch_times": []
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_duration = time.perf_counter() - epoch_start
        history["epoch_times"].append(epoch_duration)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["macro_f1"])
        history["val_f1"].append(val_metrics["macro_f1"])

        if verbose:
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['macro_f1']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['macro_f1']:.4f} | "
                  f"Time: {epoch_duration:.3f}s")

    total_time = time.perf_counter() - start_time
    history["total_time"] = total_time
    history["final_train_loss"] = history["train_loss"][-1]
    history["final_val_loss"] = history["val_loss"][-1]
    history["final_train_acc"] = history["train_acc"][-1]
    history["final_val_acc"] = history["val_acc"][-1]
    history["final_train_f1"] = history["train_f1"][-1]
    history["final_val_f1"] = history["val_f1"][-1]

    return history


if __name__ == "__main__":
    from srcs.libraries.datasets import create_moons_data
    from srcs.libraries.models import build_moons_mlp

    data = create_moons_data(n_samples=500)
    mlp = build_moons_mlp(hidden_dim=16)
    hist = train_classifier(mlp, data["train_loader"], data["val_loader"], epochs=3, verbose=True)
    print(f"Sample run finished in {hist['total_time']:.2f}s with Val Acc = {hist['final_val_acc']:.4f}")

