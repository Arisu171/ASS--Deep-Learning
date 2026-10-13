"""Plotting and visualization routines for decision boundaries, learning curves, and metrics.

CLI execution:
  python -m srcs.utilities.visualization

Inputs:
  None

Outputs:
  bin/results/ (saved plot figures)
"""

import os
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

DEFAULT_FIGURE_DPI = 150
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "results")
DEFAULT_MESH_RESOLUTION = 250


def ensure_directory(path: str) -> None:
    """Ensure destination directory exists before saving figures.

    Input:  path -- file or directory path string.
    Output: None.
    Flow:   extract parent folder and invoke os.makedirs if directory is absent.
    """
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def plot_decision_boundaries(models_dict: Dict[str, nn.Module],
                             X: np.ndarray,
                             y: np.ndarray,
                             title: str = "Decision Boundaries Across Hidden Neuron Counts",
                             save_path: Optional[str] = None) -> plt.Figure:
    """Render 2D decision boundary heatmaps for multiple classification models side by side.

    Input:  models_dict -- dictionary mapping model label strings to trained PyTorch models;
            X           -- 2D feature matrix of shape (N, 2);
            y           -- 1D ground truth label array of shape (N,);
            title       -- figure super title (default 'Decision Boundaries Across Hidden Neuron Counts');
            save_path   -- optional file path to save rendered image figure.
    Output: matplotlib Figure instance containing grid of subplots.
    Flow:   generate 2D coordinate meshgrid covering feature domain, evaluate each model
            on grid points to obtain class prediction probabilities, render filled contour plot,
            scatter true data points with distinct markers, adjust formatting, and optionally save figure.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, DEFAULT_MESH_RESOLUTION),
                         np.linspace(y_min, y_max, DEFAULT_MESH_RESOLUTION))

    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models_dict.items()):
        model.eval()
        with torch.no_grad():
            logits = model(grid_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy().reshape(xx.shape)

        contour = ax.contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.75)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")

        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=30, alpha=0.85)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_directory(save_path)
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_learning_curves(histories_dict: Dict[str, Dict[str, Any]],
                         title: str = "Training and Validation Learning Curves",
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot cross-epoch training and validation loss and accuracy trajectories for comparison.

    Input:  histories_dict -- dictionary mapping model names to history dictionaries;
            title          -- super title for overall figure;
            save_path      -- optional file path to save generated chart.
    Output: matplotlib Figure instance with dual-axis or side-by-side subplots.
    Flow:   initialize figure with two subplots for loss and accuracy, iterate over histories,
            plot train curves as solid lines and validation curves as dashed lines,
            configure legends and titles, apply clean grid styling, and optionally export to disk.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")

    for idx, (name, hist) in enumerate(histories_dict.items()):
        color = cmap(idx % 10)
        epochs_range = range(1, len(hist["train_loss"]) + 1)

        ax_loss.plot(epochs_range, hist["train_loss"], label=f"{name} (Train)", color=color, linestyle="-", linewidth=1.8)
        ax_loss.plot(epochs_range, hist["val_loss"], label=f"{name} (Val)", color=color, linestyle="--", linewidth=1.8)

        ax_acc.plot(epochs_range, hist["train_acc"], label=f"{name} (Train)", color=color, linestyle="-", linewidth=1.8)
        ax_acc.plot(epochs_range, hist["val_acc"], label=f"{name} (Val)", color=color, linestyle="--", linewidth=1.8)

    ax_loss.set_title("Cross-Entropy Loss Trajectory", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=9, loc="upper right")
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    ax_acc.set_title("Classification Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(fontsize=9, loc="lower right")
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_directory(save_path)
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_metrics_comparison(summary_df: pd.DataFrame,
                            label_col: str = "Model",
                            metrics_cols: Optional[List[str]] = None,
                            title: str = "Performance Metrics Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
    """Generate multi-panel bar charts comparing performance metrics and resource consumption.

    Input:  summary_df   -- pandas DataFrame containing summary statistics across configurations;
            label_col    -- column name containing configuration labels (default 'Model');
            metrics_cols -- list of numeric metric columns to display;
            title        -- super title of the chart;
            save_path    -- optional output file path.
    Output: matplotlib Figure instance.
    Flow:   determine display columns, create subplot grid corresponding to metric count,
            plot bar plots for each metric with value annotations on top of bars,
            format layout, and optionally save figure.
    """
    if metrics_cols is None:
        metrics_cols = ["Val Accuracy", "Val Macro-F1", "Parameters", "Time (s)"]

    num_metrics = len(metrics_cols)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4.2 * num_metrics, 4.2))
    if num_metrics == 1:
        axes = [axes]

    labels = summary_df[label_col].tolist()
    positions = np.arange(len(labels))

    colors = ["#2b5c8f", "#3c887e", "#d87341", "#c74440"]

    for idx, (ax, col) in enumerate(zip(axes, metrics_cols)):
        values = summary_df[col].tolist()
        bar_color = colors[idx % len(colors)]
        bars = ax.bar(positions, values, color=bar_color, alpha=0.85, edgecolor="black", width=0.55)

        for bar in bars:
            height = bar.get_height()
            format_str = f"{height:,.0f}" if height >= 100 else f"{height:.4f}"
            ax.annotate(format_str,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        ensure_directory(save_path)
        fig.savefig(save_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization testing CLI.")
    args = parser.parse_args()
    print("Visualization module ready.")

