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
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from srcs.libraries.metrics import calculate_metrics

DEFAULT_EPOCHS = 15
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_PATIENCE = 3
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def device() -> str:
    """Detect and return available PyTorch computation device.

    Input:  None.
    Output: string 'cuda' if GPU is available else 'cpu'.
    Flow:   query torch.cuda.is_available and return corresponding device identifier string.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


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


class EarlyStopping:
    """Early stopping monitor tracking validation loss to halt training on stagnation.

    Input:  patience             -- consecutive non-improving epochs before stopping (default DEFAULT_PATIENCE);
            min_delta            -- minimal decrease required to register improvement (default DEFAULT_MIN_DELTA);
            mode                 -- 'min' for minimizing loss or 'max' for maximizing metric (default 'min');
            restore_best_weights -- whether to restore model weights from best epoch (default True).
    Output: instantiated EarlyStopping monitor.
    Flow:   record baseline metric criteria, track epoch progress,
            detect improvements beyond min_delta threshold, cache best model state dictionary,
            increment violation counter upon stagnation, and signal termination when patience expires.
    """

    def __init__(self,
                 patience: int = DEFAULT_PATIENCE,
                 min_delta: float = DEFAULT_MIN_DELTA,
                 mode: str = "min",
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def step(self, current_value: float, model: nn.Module, epoch: int) -> bool:
        """Evaluate current validation metric and update early stopping status.

        Input:  current_value -- scalar validation metric observed in current epoch;
                model         -- PyTorch model whose state dictionary may be cached;
                epoch         -- current integer epoch index.
        Output: boolean flag indicating whether training should be terminated.
        Flow:   check if current metric improves baseline by at least min_delta,
                record new best score and snapshot model parameters when improved,
                increment stagnation counter otherwise, and trigger termination upon exceeding patience.
        """
        if self.best_score is None:
            self.best_score = current_value
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self.mode == "min":
            improved = current_value < (self.best_score - self.min_delta)
        else:
            improved = current_value > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_value
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Restore cached optimal model parameter weights into target model.

        Input:  model -- PyTorch neural network to receive optimal weights.
        Output: None.
        Flow:   verify cached weights exist and load state dictionary into model.
        """
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    @property
    def best_loss(self) -> Optional[float]:
        """Expose best recorded metric value for backward compatibility.

        Input:  None.
        Output: float or None representing best recorded metric value.
        Flow:   return self.best_score attribute directly.
        """
        return self.best_score


class MetricDict(dict):
    """Dictionary supporting both key-based indexing and two-element tuple unpacking."""

    def __iter__(self):
        yield self["loss"]
        yield self["accuracy"]


def evaluate_classifier(model: nn.Module,
                        loader: DataLoader,
                        criterion: nn.Module,
                        device: str = DEFAULT_DEVICE) -> MetricDict:
    """Evaluate classifier model and return mean loss and accuracy payload.

    Input:  model     -- PyTorch neural network to evaluate;
            loader    -- DataLoader yielding evaluation batches;
            criterion -- loss function module;
            device    -- computation device string (default DEFAULT_DEVICE).
    Output: MetricDict instance supporting both dict indexing and tuple unpacking.
    Flow:   set model to eval mode, disable gradient computation, iterate minibatches,
            accumulate cross entropy loss and correct prediction counts,
            and return mean loss and classification accuracy inside MetricDict.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

    mean_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return MetricDict({"loss": float(mean_loss), "accuracy": float(accuracy)})


def collect_predictions(model: nn.Module,
                        loader: DataLoader,
                        device: str = DEFAULT_DEVICE) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ground truth labels and model predictions over an entire DataLoader.

    Input:  model  -- PyTorch neural network to evaluate;
            loader -- DataLoader yielding minibatches of features and targets;
            device -- computation device string (default DEFAULT_DEVICE).
    Output: tuple of (y_true, y_pred) as 1D numpy arrays.
    Flow:   set model to eval mode, iterate minibatches under torch.no_grad,
            compute argmax of model logits, collect target and prediction batches,
            concatenate into 1D numpy arrays, and return tuple.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    return y_true, y_pred


def train_classifier(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     epochs: int = DEFAULT_EPOCHS,
                     lr: float = DEFAULT_LR,
                     weight_decay: float = DEFAULT_WEIGHT_DECAY,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     criterion: Optional[nn.Module] = None,
                     early_stopping: Optional[EarlyStopping] = None,
                     device: str = DEFAULT_DEVICE,
                     verbose: bool = True) -> Dict[str, Any]:
    """Train a classifier model across multiple epochs and record epoch history.

    Input:  model          -- PyTorch neural network;
            train_loader   -- DataLoader for training split;
            val_loader     -- DataLoader for validation split;
            epochs         -- total training epochs (default DEFAULT_EPOCHS);
            lr             -- learning rate for default Adam optimizer (default DEFAULT_LR);
            weight_decay   -- L2 regularization coefficient (default DEFAULT_WEIGHT_DECAY);
            optimizer      -- optional explicit PyTorch optimizer instance;
            criterion      -- optional explicit loss criterion module;
            early_stopping -- optional EarlyStopping monitor instance;
            device         -- computation hardware target (default DEFAULT_DEVICE);
            verbose        -- whether to print per-epoch progress logs (default True).
    Output: history dictionary containing epoch trajectories, final metrics, and total elapsed runtime.
    Flow:   move model to device, initialize loss function and optimizer if omitted,
            run training and validation loop for specified epochs, record performance history,
            monitor early stopping criteria and halt prematurely if stagnation triggers,
            restore best weights if early stopping was active, and return structured history dictionary.
    """
    model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
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

        if early_stopping is not None:
            should_terminate = early_stopping.step(val_metrics["loss"], model, epoch)
            if should_terminate:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch:02d}. "
                          f"Restoring best weights from epoch {early_stopping.best_epoch:02d} "
                          f"with Val Loss: {early_stopping.best_score:.4f}.")
                early_stopping.restore(model)
                break

    total_time = time.perf_counter() - start_time
    history["total_time"] = total_time
    history["final_train_loss"] = history["train_loss"][-1]
    history["final_val_loss"] = history["val_loss"][-1]
    history["final_train_acc"] = history["train_acc"][-1]
    history["final_val_acc"] = history["val_acc"][-1]
    history["final_train_f1"] = history["train_f1"][-1]
    history["final_val_f1"] = history["val_f1"][-1]
    history["early_stopped"] = early_stopping.should_stop if early_stopping is not None else False
    history["best_epoch"] = early_stopping.best_epoch if early_stopping is not None else len(history["train_loss"])
    history["best_val_loss"] = early_stopping.best_score if early_stopping is not None else history["final_val_loss"]

    return history


if __name__ == "__main__":
    from srcs.libraries.datasets import create_moons_data
    from srcs.libraries.models import build_moons_mlp

    data = create_moons_data(n_samples=500)
    mlp = build_moons_mlp(hidden_dim=16)
    hist = train_classifier(mlp, data["train_loader"], data["val_loader"], epochs=3, verbose=True)
    print(f"Sample run finished in {hist['total_time']:.2f}s with Val Acc = {hist['final_val_acc']:.4f}")

