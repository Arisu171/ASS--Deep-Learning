"""High-level experiment orchestrators for MLP architecture and hyperparameter studies.

CLI execution:
  python -m srcs.utilities.experiments --task bai01
  python -m srcs.utilities.experiments --task bai02 --epochs 10

Inputs:
  data/ (cached or downloaded datasets)

Outputs:
  bin/models/ (saved model checkpoint binaries)
  bin/results/ (serialized metric comparisons and records)
"""

import os
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import torch

from srcs.libraries.seeding import seed_everything
from srcs.libraries.datasets import create_moons_data, load_fashion_mnist_data
from srcs.libraries.models import build_moons_mlp, ConfigurableMLP, count_parameters
from srcs.libraries.training import train_classifier
from srcs.libraries.serialization import save_checkpoint, save_experiment_results

DEFAULT_TASK = "bai01"
DEFAULT_NEURONS = [8, 32, 64, 128]
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_SEED = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "bin", "models")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "results")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_bai01_neuron_study(neuron_list: Optional[List[int]] = None,
                           lr: float = DEFAULT_LR,
                           epochs: int = DEFAULT_EPOCHS,
                           n_samples: int = 1200,
                           noise: float = 0.28,
                           seed: int = DEFAULT_SEED,
                           device: str = DEFAULT_DEVICE,
                           verbose: bool = False) -> Dict[str, Any]:
    """Execute comparative study of MLP architectures on make_moons across specified neuron widths.

    Input:  neuron_list -- list of integer neuron counts (default DEFAULT_NEURONS: [8, 32, 64, 128]);
            lr          -- learning rate for optimization (default DEFAULT_LR);
            epochs      -- training epoch count per configuration (default DEFAULT_EPOCHS);
            n_samples   -- number of synthetic moon coordinates (default 1200);
            noise       -- coordinate Gaussian noise magnitude (default 0.28);
            seed        -- random generator seed (default DEFAULT_SEED);
            device      -- target hardware accelerator or cpu (default DEFAULT_DEVICE);
            verbose     -- whether to output per-epoch logs (default False).
    Output: dictionary packaging models, training histories, tabular summary DataFrame, and raw dataset.
    Flow:   seed random generators, create synthetic two-moons data loaders,
            iterate across specified neuron configurations, instantiate two-layer MLP for each,
            record trainable parameter count, train network and record metrics,
            save checkpoint binaries to bin/models/, assemble comparative metrics DataFrame,
            serialize summary results to bin/results/, and return comprehensive experiment bundle.
    """
    if neuron_list is None:
        neuron_list = DEFAULT_NEURONS

    seed_everything(seed)
    data_bundle = create_moons_data(n_samples=n_samples, noise=noise, random_state=seed)
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]

    models_dict = {}
    histories_dict = {}
    summary_rows = []

    for hidden in neuron_list:
        model_name = f"MLP_hidden_{hidden}"
        seed_everything(seed)
        model = build_moons_mlp(hidden_dim=hidden, input_dim=2, output_dim=2, activation="relu")
        param_count = count_parameters(model)

        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            verbose=verbose
        )

        models_dict[model_name] = model
        histories_dict[model_name] = history

        checkpoint_path = os.path.join(DEFAULT_MODEL_DIR, f"bai01_mlp_neurons_{hidden}.pt")
        save_checkpoint(model, filepath=checkpoint_path, metadata={
            "hidden": hidden,
            "params": param_count,
            "final_val_acc": history["final_val_acc"],
            "final_val_f1": history["final_val_f1"],
            "total_time": history["total_time"]
        })

        summary_rows.append({
            "Model": model_name,
            "Hidden Neurons": hidden,
            "Parameters": param_count,
            "Train Loss": round(history["final_train_loss"], 4),
            "Val Loss": round(history["final_val_loss"], 4),
            "Train Accuracy": round(history["final_train_acc"], 4),
            "Val Accuracy": round(history["final_val_acc"], 4),
            "Val Macro-F1": round(history["final_val_f1"], 4),
            "Time (s)": round(history["total_time"], 3)
        })

    summary_df = pd.DataFrame(summary_rows)
    results_path = os.path.join(DEFAULT_RESULTS_DIR, "bai01_neuron_comparison.json")
    save_experiment_results(summary_df.to_dict(orient="records"), filepath=results_path)

    return {
        "models": models_dict,
        "histories": histories_dict,
        "summary_df": summary_df,
        "data_bundle": data_bundle
    }


def run_bai02_ablation_study(data_bundle: Dict[str, Any],
                             configurations: List[Dict[str, Any]],
                             epochs: int = DEFAULT_EPOCHS,
                             lr: float = DEFAULT_LR,
                             seed: int = DEFAULT_SEED,
                             device: str = DEFAULT_DEVICE,
                             study_name: str = "bai02_ablation",
                             verbose: bool = False) -> Dict[str, Any]:
    """Execute ablation experiments varying depth, width, activations, and regularization.

    Input:  data_bundle    -- dictionary containing train_loader, val_loader, input_dim, num_classes;
            configurations -- list of dictionaries specifying name, hidden_dims, activation, dropout_rate, weight_decay;
            epochs         -- training epoch limit per configuration (default DEFAULT_EPOCHS);
            lr             -- optimization learning rate (default DEFAULT_LR);
            seed           -- random generator seed (default DEFAULT_SEED);
            device         -- target computation device (default DEFAULT_DEVICE);
            study_name     -- identifier used for artifact naming (default 'bai02_ablation');
            verbose        -- whether to output progress logs per epoch (default False).
    Output: dictionary containing model instances, histories, and summary DataFrame.
    Flow:   iterate across configuration specifications, construct ConfigurableMLP with requested
            dimensions, activation, and dropout, initialize optimizer with designated weight decay,
            train model and capture training trajectories, serialize checkpoints to bin/models/,
            synthesize comparative evaluation table, and serialize findings to bin/results/.
    """
    input_dim = data_bundle["input_dim"]
    num_classes = data_bundle["num_classes"]
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]

    models_dict = {}
    histories_dict = {}
    summary_rows = []

    for cfg in configurations:
        name = cfg["name"]
        hidden_dims = cfg.get("hidden_dims", [64, 64])
        activation = cfg.get("activation", "relu")
        dropout_rate = cfg.get("dropout_rate", 0.0)
        weight_decay = cfg.get("weight_decay", 0.0)
        cfg_lr = cfg.get("lr", lr)
        cfg_epochs = cfg.get("epochs", epochs)

        seed_everything(seed)
        model = ConfigurableMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            activation=activation,
            dropout_rate=dropout_rate
        )
        param_count = count_parameters(model)

        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg_epochs,
            lr=cfg_lr,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose
        )

        models_dict[name] = model
        histories_dict[name] = history

        checkpoint_path = os.path.join(DEFAULT_MODEL_DIR, f"{study_name}_{name}.pt")
        save_checkpoint(model, filepath=checkpoint_path, metadata={
            "name": name,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "params": param_count,
            "val_acc": history["final_val_acc"],
            "val_f1": history["final_val_f1"]
        })

        summary_rows.append({
            "Experiment": name,
            "Hidden Layers": str(hidden_dims),
            "Num Layers": len(hidden_dims),
            "Activation": activation,
            "Dropout": dropout_rate,
            "Weight Decay": weight_decay,
            "Parameters": param_count,
            "Train Loss": round(history["final_train_loss"], 4),
            "Val Loss": round(history["final_val_loss"], 4),
            "Train Accuracy": round(history["final_train_acc"], 4),
            "Val Accuracy": round(history["final_val_acc"], 4),
            "Val Macro-F1": round(history["final_val_f1"], 4),
            "Time (s)": round(history["total_time"], 3)
        })

    summary_df = pd.DataFrame(summary_rows)
    results_path = os.path.join(DEFAULT_RESULTS_DIR, f"{study_name}_summary.json")
    save_experiment_results(summary_df.to_dict(orient="records"), filepath=results_path)

    return {
        "models": models_dict,
        "histories": histories_dict,
        "summary_df": summary_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runner CLI.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, choices=["bai01", "bai02"], help="Task to execute.")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count.")
    args = parser.parse_args()

    if args.task == "bai01":
        study = run_bai01_neuron_study(epochs=args.epochs, verbose=True)
        print("\nBai 01 Results:")
        print(study["summary_df"].to_string())
    else:
        fashion_data = load_fashion_mnist_data(data_dir=DEFAULT_DATA_DIR, subset_size=2000)
        demo_cfgs = [
            {"name": "Depth_1", "hidden_dims": [64], "activation": "relu"},
            {"name": "Depth_2", "hidden_dims": [64, 64], "activation": "relu"}
        ]
        study = run_bai02_ablation_study(fashion_data, demo_cfgs, epochs=args.epochs, verbose=True)
        print("\nBai 02 Demo Results:")
        print(study["summary_df"].to_string())

