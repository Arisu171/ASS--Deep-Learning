"""Evaluation metrics computation including Accuracy and Macro-F1 score.

CLI execution:
  python -m srcs.libraries.metrics

Inputs:
  None

Outputs:
  Console output of sample metric calculations.
"""

from typing import Dict, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_AVERAGE_MODE = "macro"


def calculate_metrics(y_true: Union[torch.Tensor, np.ndarray],
                      y_pred: Union[torch.Tensor, np.ndarray],
                      average: str = DEFAULT_AVERAGE_MODE) -> Dict[str, float]:
    """Compute accuracy and Macro-F1 score for multi-class classification predictions.

    Input:  y_true  -- ground truth labels as 1D tensor or numpy array;
            y_pred  -- predicted class labels as 1D tensor or numpy array;
            average -- averaging method for F1 score calculation (default DEFAULT_AVERAGE_MODE).
    Output: dictionary mapping metric names to computed floating point values.
    Flow:   convert tensors to numpy arrays if necessary, flatten shapes to 1D vectors,
            compute categorical accuracy, compute Macro-F1 score, and assemble metrics dictionary.
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = np.asarray(y_pred)

    y_true_flat = y_true_np.reshape(-1)
    y_pred_flat = y_pred_np.reshape(-1)

    acc = float(accuracy_score(y_true_flat, y_pred_flat))
    f1 = float(f1_score(y_true_flat, y_pred_flat, average=average, zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": f1
    }


if __name__ == "__main__":
    sample_true = np.array([0, 1, 0, 1, 1, 0])
    sample_pred = np.array([0, 1, 0, 0, 1, 0])
    metrics_result = calculate_metrics(sample_true, sample_pred)
    print(f"Metrics test: Accuracy = {metrics_result['accuracy']:.4f}, Macro-F1 = {metrics_result['macro_f1']:.4f}")

