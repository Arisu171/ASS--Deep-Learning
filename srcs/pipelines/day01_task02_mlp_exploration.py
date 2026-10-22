"""Day 1 Task 2: Standalone pipeline for comprehensive MLP architectural and hyperparameter exploration.

CLI execution:
  python -m srcs.pipelines.day01_task02_mlp_exploration --epochs 12

Inputs:
  data/FashionMNIST (dataset archives)

Outputs:
  bin/day01/task02/models/*.pt (trained model checkpoints)
  bin/day01/task02/results/overall_summary.csv (consolidated benchmark results)
  bin/day01/task02/results/overall_summary.md (markdown comparison table)
  bin/day01/task02/results/*_curves.png (depth, width, activation, regularization plots)
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
from srcs.libraries.datasets import load_fashion_mnist_data
from srcs.libraries.training import device
from srcs.utilities.experiments import run_bai02_ablation_study
from srcs.utilities.visualization import plot_learning_curves
from srcs.utilities.tables import format_summary_table

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 12
DEFAULT_REG_EPOCHS = 15
DEFAULT_LR = 1e-3
DEFAULT_SUBSET_SIZE = 6000
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task02", "models")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task02", "results")


def run_pipeline(epochs: int = DEFAULT_EPOCHS,
                 reg_epochs: int = DEFAULT_REG_EPOCHS,
                 lr: float = DEFAULT_LR,
                 seed: int = DEFAULT_SEED,
                 data_dir: str = DEFAULT_DATA_DIR,
                 models_dir: str = DEFAULT_MODELS_DIR,
                 results_dir: str = DEFAULT_RESULTS_DIR) -> Dict[str, Any]:
    """Execute Day 1 Task 2 ablation experiments for depth, width, activations, and regularization.

    Input:  epochs      -- epoch count for depth, width, and activation benchmarks (default DEFAULT_EPOCHS);
            reg_epochs  -- epoch count for regularization study (default DEFAULT_REG_EPOCHS);
            lr          -- learning rate for optimization (default DEFAULT_LR);
            seed        -- random generator seed (default DEFAULT_SEED);
            data_dir    -- dataset directory path (default DEFAULT_DATA_DIR);
            models_dir  -- destination folder for saved weights (default DEFAULT_MODELS_DIR);
            results_dir -- destination folder for metric artifacts (default DEFAULT_RESULTS_DIR).
    Output: dictionary mapping study identifiers to summary DataFrames and plot handles.
    Flow:   initialize seeds, prepare FashionMNIST dataset loaders, execute depth ablation,
            execute width ablation, execute activation ablation, execute regularization ablation,
            save checkpoints and trajectory plots, and synthesize consolidated markdown and CSV tables.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    seed_everything(seed)
    data_bundle = load_fashion_mnist_data(
        data_dir=data_dir,
        batch_size=64,
        val_batch_size=256,
        subset_size=DEFAULT_SUBSET_SIZE,
        random_state=seed
    )
    comp_device = device()

    depth_configs = [
        {"name": "depth_1_layer", "hidden_dims": [128], "activation": "relu"},
        {"name": "depth_2_layers", "hidden_dims": [128, 128], "activation": "relu"},
        {"name": "depth_3_layers", "hidden_dims": [128, 128, 128], "activation": "relu"},
        {"name": "depth_4_layers", "hidden_dims": [128, 128, 128, 128], "activation": "relu"}
    ]
    study_depth = run_bai02_ablation_study(
        data_bundle=data_bundle,
        configurations=depth_configs,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=comp_device,
        study_name="depth",
        verbose=False
    )
    plot_learning_curves(study_depth["histories"],
                         title="Day 1 Task 2: Learning Curves Across Network Depth",
                         save_path=os.path.join(results_dir, "depth_curves.png"))

    width_configs = [
        {"name": "width_32_neurons", "hidden_dims": [32, 32], "activation": "relu"},
        {"name": "width_64_neurons", "hidden_dims": [64, 64], "activation": "relu"},
        {"name": "width_128_neurons", "hidden_dims": [128, 128], "activation": "relu"},
        {"name": "width_256_neurons", "hidden_dims": [256, 256], "activation": "relu"}
    ]
    study_width = run_bai02_ablation_study(
        data_bundle=data_bundle,
        configurations=width_configs,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=comp_device,
        study_name="width",
        verbose=False
    )
    plot_learning_curves(study_width["histories"],
                         title="Day 1 Task 2: Learning Curves Across Network Width",
                         save_path=os.path.join(results_dir, "width_curves.png"))

    act_configs = [
        {"name": "activation_relu", "hidden_dims": [128, 64], "activation": "relu"},
        {"name": "activation_tanh", "hidden_dims": [128, 64], "activation": "tanh"},
        {"name": "activation_sigmoid", "hidden_dims": [128, 64], "activation": "sigmoid"}
    ]
    study_act = run_bai02_ablation_study(
        data_bundle=data_bundle,
        configurations=act_configs,
        epochs=epochs,
        lr=lr,
        seed=seed,
        device=comp_device,
        study_name="activation",
        verbose=False
    )
    plot_learning_curves(study_act["histories"],
                         title="Day 1 Task 2: Learning Curves Across Activation Functions",
                         save_path=os.path.join(results_dir, "activation_curves.png"))

    reg_configs = [
        {"name": "regularization_baseline_noreg", "hidden_dims": [256, 128, 64], "activation": "relu",
         "dropout_rate": 0.0, "weight_decay": 0.0},
        {"name": "regularization_dropout_p03", "hidden_dims": [256, 128, 64], "activation": "relu",
         "dropout_rate": 0.3, "weight_decay": 0.0},
        {"name": "regularization_weightdecay_1e4", "hidden_dims": [256, 128, 64], "activation": "relu",
         "dropout_rate": 0.0, "weight_decay": 1e-4},
        {"name": "regularization_combined_drop_wd", "hidden_dims": [256, 128, 64], "activation": "relu",
         "dropout_rate": 0.3, "weight_decay": 1e-4}
    ]
    study_reg = run_bai02_ablation_study(
        data_bundle=data_bundle,
        configurations=reg_configs,
        epochs=reg_epochs,
        lr=lr,
        seed=seed,
        device=comp_device,
        study_name="regularization",
        verbose=False
    )
    plot_learning_curves(study_reg["histories"],
                         title="Day 1 Task 2: Learning Curves Across Regularization Strategies",
                         save_path=os.path.join(results_dir, "regularization_curves.png"))

    overall_df = pd.concat([
        study_depth["summary_df"],
        study_width["summary_df"],
        study_act["summary_df"],
        study_reg["summary_df"]
    ], ignore_index=True)

    format_summary_table(
        overall_df,
        export_csv_path=os.path.join(results_dir, "overall_summary.csv"),
        export_md_path=os.path.join(results_dir, "overall_summary.md")
    )

    return {
        "overall_df": overall_df,
        "depth": study_depth,
        "width": study_width,
        "activation": study_act,
        "regularization": study_reg
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 1 Task 2 MLP exploration pipeline CLI.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Ablation epoch count.")
    parser.add_argument("--reg_epochs", type=int, default=DEFAULT_REG_EPOCHS, help="Regularization epoch count.")
    args = parser.parse_args()

    results = run_pipeline(epochs=args.epochs, reg_epochs=args.reg_epochs)
    print("\nDay 1 Task 2 Pipeline Completed Successfully:")
    print(results["overall_df"][["Experiment", "Hidden Layers", "Activation", "Val Accuracy", "Val Macro-F1"]].to_string())

