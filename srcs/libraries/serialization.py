"""Model checkpoint persistence and experiment artifact serialization utilities.

CLI execution:
  python -m srcs.libraries.serialization

Inputs:
  bin/models/ (checkpoint storage directory)
  bin/results/ (serialized results directory)

Outputs:
  Saved PyTorch model weights (.pt) and JSON evaluation records (.json).
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
import torch
from torch import nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "bin", "models")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "results")
DEFAULT_CHECKPOINT_NAME = "mlp_checkpoint.pt"
DEFAULT_RESULTS_NAME = "experiment_results.json"


def ensure_directory(path: str) -> None:
    """Create directory structure if it does not already exist.

    Input:  path -- target directory path string.
    Output: None.
    Flow:   inspect directory existence and create parent folders recursively if absent.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_checkpoint(model: nn.Module,
                    filepath: str = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_CHECKPOINT_NAME),
                    metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save model state dictionary and associated metadata to disk.

    Input:  model    -- PyTorch neural network instance;
            filepath -- destination file path (default bin/models/mlp_checkpoint.pt);
            metadata -- optional dictionary containing hyperparameter or metric information.
    Output: absolute string path to the saved checkpoint file.
    Flow:   extract target directory, ensure directory exists, package state dictionary
            with optional metadata payload, serialize checkpoint using torch.save,
            and return the destination path string.
    """
    directory = os.path.dirname(filepath)
    ensure_directory(directory)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {}
    }
    torch.save(payload, filepath)
    return os.path.abspath(filepath)


def load_checkpoint(model: nn.Module,
                    filepath: str = os.path.join(DEFAULT_MODEL_DIR, DEFAULT_CHECKPOINT_NAME)) -> Dict[str, Any]:
    """Load model state dictionary and metadata from disk into a PyTorch model.

    Input:  model    -- PyTorch neural network instance to populate;
            filepath -- source file path to existing checkpoint.
    Output: dictionary containing saved metadata.
    Flow:   verify checkpoint file presence, load serialized payload via torch.load,
            load state dictionary into model instance, and return stored metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    payload = torch.load(filepath, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})


def save_experiment_results(results: Any,
                            filepath: str = os.path.join(DEFAULT_RESULTS_DIR, DEFAULT_RESULTS_NAME)) -> str:
    """Serialize dictionary or list of experiment metrics to a JSON file.

    Input:  results  -- dictionary or list containing tabular metrics or epoch histories;
            filepath -- target json file path (default bin/results/experiment_results.json).
    Output: absolute string path to the generated JSON artifact.
    Flow:   ensure destination directory exists, normalize input payload to serializable structure,
            write JSON string with four-space indentation, and return absolute path.
    """
    directory = os.path.dirname(filepath)
    ensure_directory(directory)

    payload = results
    if isinstance(results, dict):
        payload = {}
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                payload[key] = value
            else:
                payload[key] = str(value)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    return os.path.abspath(filepath)


if __name__ == "__main__":
    from srcs.libraries.models import build_moons_mlp

    sample_net = build_moons_mlp(hidden_dim=8)
    saved_path = save_checkpoint(sample_net, metadata={"test": True, "neurons": 8})
    print(f"Checkpoint verification saved at: {saved_path}")

