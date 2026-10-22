"""Day 1 Task 1: Standalone pipeline for MLP construction and hidden neuron width study.

CLI execution:
  python -m srcs.pipelines.day01_task01_mlp_construction --epochs 20

Inputs:
  None (synthesizes make_moons dataset in memory)

Outputs:
  bin/day01/task01/models/mlp_neurons_*.pt (trained model checkpoints)
  bin/day01/task01/results/summary_table.md (performance comparison markdown)
  bin/day01/task01/results/decision_boundaries.png (decision boundary visualization)
  bin/day01/task01/results/learning_curves.png (loss and accuracy trajectory plot)
  bin/day01/task01/results/metrics_comparison.png (resource and accuracy comparison bars)
"""

import os
import sys
import argparse
from typing import Dict, Any, List
import pandas as pd
import torch

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.datasets import create_moons_data
from srcs.libraries.models import build_moons_mlp, count_parameters
from srcs.libraries.training import train_classifier, device
from srcs.libraries.serialization import save_checkpoint, save_experiment_results
from srcs.utilities.visualization import (
    plot_decision_boundaries,
    plot_learning_curves,
    plot_metrics_comparison
)
from srcs.utilities.tables import format_summary_table

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_SAMPLES = 1200
DEFAULT_NOISE = 0.28
DEFAULT_NEURON_LIST = (8, 32, 64, 128)
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task01", "models")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task01", "results")
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "day01", "task01")


def run_pipeline(neuron_list: List[int] = DEFAULT_NEURON_LIST,
                 epochs: int = DEFAULT_EPOCHS,
                 lr: float = DEFAULT_LR,
                 seed: int = DEFAULT_SEED,
                 models_dir: str = DEFAULT_MODELS_DIR,
                 results_dir: str = DEFAULT_RESULTS_DIR,
                 docs_dir: str = DEFAULT_DOCS_DIR) -> Dict[str, Any]:
    """Execute Day 1 Task 1 training, benchmarking, plotting, and artifact persistence.

    Input:  neuron_list -- sequence of hidden neuron counts (default DEFAULT_NEURON_LIST);
            epochs      -- training epoch limit per architecture (default DEFAULT_EPOCHS);
            lr          -- learning rate for optimization (default DEFAULT_LR);
            seed        -- random generator seed (default DEFAULT_SEED);
            models_dir  -- destination folder for saved weights (default DEFAULT_MODELS_DIR);
            results_dir -- destination folder for metric artifacts (default DEFAULT_RESULTS_DIR);
            docs_dir    -- documentation directory for reports (default DEFAULT_DOCS_DIR).
    Output: dictionary packaging trained models, training histories, and summary DataFrame.
    Flow:   initialize deterministic seeds, construct make_moons data loaders,
            train each MLP topology, persist checkpoints, calculate metrics,
            render decision boundaries and learning curves, save summary tables,
            and return structured pipeline results.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    seed_everything(seed)
    data_bundle = create_moons_data(n_samples=DEFAULT_SAMPLES, noise=DEFAULT_NOISE, random_state=seed)
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    comp_device = device()

    models_dict = {}
    histories_dict = {}
    summary_rows = []

    for hidden in neuron_list:
        model_name = f"MLP_hidden_{hidden}"
        seed_everything(seed)
        model = build_moons_mlp(hidden_dim=hidden, input_dim=2, output_dim=2, activation="relu")
        params = count_parameters(model)

        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=comp_device,
            verbose=False
        )

        models_dict[model_name] = model
        histories_dict[model_name] = history

        ckpt_path = os.path.join(models_dir, f"mlp_neurons_{hidden}.pt")
        save_checkpoint(model, filepath=ckpt_path, metadata={
            "hidden": hidden,
            "params": params,
            "val_acc": history["final_val_acc"],
            "val_f1": history["final_val_f1"]
        })

        summary_rows.append({
            "Model": model_name,
            "Hidden Neurons": hidden,
            "Parameters": params,
            "Train Loss": round(history["final_train_loss"], 4),
            "Val Loss": round(history["final_val_loss"], 4),
            "Train Accuracy": round(history["final_train_acc"], 4),
            "Val Accuracy": round(history["final_val_acc"], 4),
            "Val Macro-F1": round(history["final_val_f1"], 4),
            "Time (s)": round(history["total_time"], 3)
        })

    summary_df = pd.DataFrame(summary_rows)
    table_path = os.path.join(results_dir, "summary_table.md")
    format_summary_table(summary_df, export_md_path=table_path)
    save_experiment_results(summary_df.to_dict(orient="records"),
                            filepath=os.path.join(results_dir, "neuron_comparison.json"))

    plot_decision_boundaries(
        models_dict=models_dict,
        X=data_bundle["X_train"],
        y=data_bundle["y_train"],
        title="Day 1 Task 1: Decision Boundaries Across Hidden Neuron Counts",
        save_path=os.path.join(results_dir, "decision_boundaries.png")
    )

    plot_learning_curves(
        histories_dict=histories_dict,
        title="Day 1 Task 1: Training and Validation Learning Curves",
        save_path=os.path.join(results_dir, "learning_curves.png")
    )

    plot_metrics_comparison(
        summary_df=summary_df,
        label_col="Model",
        metrics_cols=["Val Accuracy", "Val Macro-F1", "Parameters", "Time (s)"],
        title="Day 1 Task 1: Metrics and Resource Trade-offs",
        save_path=os.path.join(results_dir, "metrics_comparison.png")
    )

    return {
        "models": models_dict,
        "histories": histories_dict,
        "summary_df": summary_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 1 Task 1 MLP construction pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Optimization learning rate.")
    args = parser.parse_args()

    results = run_pipeline(epochs=args.epochs, lr=args.lr)
    print("\nDay 1 Task 1 Pipeline Execution Completed Successfully:")
    print(results["summary_df"].to_string(index=False))

