"""Execute standardized notebooks and persist rendered cell outputs.

CLI execution:
  python tools/notebooks_execute.py [--notebooks path1 path2 ...]

Inputs:
  srcs/notebooks/*.ipynb

Outputs:
  Populated notebooks with rendered charts, tables, and execution logs.
"""

import os
import argparse
from typing import List, Optional
import nbformat
from nbclient import NotebookClient

DEFAULT_NOTEBOOKS = [
    "srcs/notebooks/day01_task01_mlp_construction.ipynb",
    "srcs/notebooks/day01_task02_mlp_exploration.ipynb",
    "srcs/notebooks/day03_task01_training_loop.ipynb",
    "srcs/notebooks/day03_task02_optimizer_comparison.ipynb",
    "srcs/notebooks/day04_task01_cnn_fashionmnist.ipynb",
    "srcs/notebooks/day04_task02_cnn_variants.ipynb",
    "srcs/notebooks/day05_task01_transfer_learning.ipynb",
    "srcs/notebooks/day05_task02_transfer_learning.ipynb",
    "srcs/notebooks/day06_task01_lstm_text_classification.ipynb",
    "srcs/notebooks/day06_task02_lstm_variants.ipynb"
]
DEFAULT_TIMEOUT = 1200


def execute_notebook(notebook_path: str, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Execute a single Jupyter Notebook and write evaluated outputs back to disk.

    Input:  notebook_path -- relative or absolute string path to notebook file;
            timeout       -- execution timeout in seconds per cell (default DEFAULT_TIMEOUT).
    Output: None.
    Flow:   read notebook content via nbformat, initialize NotebookClient pointing to the notebook folder,
            execute all cells sequentially, and serialize populated notebook back to original path.
    """
    print(f"\n==========================================")
    print(f"Executing: {notebook_path}")
    print(f"==========================================")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(notebook_path))}}
    )
    client.execute()

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully executed and saved: {notebook_path}")


def run_all_notebooks(notebook_paths: Optional[List[str]] = None) -> None:
    """Execute configured notebooks sequentially.

    Input:  notebook_paths -- list of notebook file paths to execute (default DEFAULT_NOTEBOOKS).
    Output: None.
    Flow:   resolve targets, iterate through notebook list, and invoke execute_notebook for each.
    """
    targets = notebook_paths or DEFAULT_NOTEBOOKS
    for nb_path in targets:
        execute_notebook(nb_path)
    print("\nAll notebooks executed and verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notebook execution runner CLI.")
    parser.add_argument("--notebooks", nargs="+", default=None, help="List of notebook paths to execute.")
    args = parser.parse_args()

    run_all_notebooks(notebook_paths=args.notebooks)
