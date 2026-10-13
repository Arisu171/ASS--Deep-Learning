"""Tabular reporting, formatting, and markdown generation utilities for experiments.

CLI execution:
  python -m srcs.utilities.tables

Inputs:
  None

Outputs:
  Formatted console tables and markdown summary exports.
"""

import os
import argparse
from typing import Optional
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "results")


def ensure_directory(path: str) -> None:
    """Ensure directory structure exists prior to saving files.

    Input:  path -- destination directory path string.
    Output: None.
    Flow:   check existence of target directory and create folders if absent.
    """
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def format_summary_table(df: pd.DataFrame,
                         export_csv_path: Optional[str] = None,
                         export_md_path: Optional[str] = None) -> str:
    """Format DataFrame as a clean Markdown table and optionally export to disk.

    Input:  df              -- pandas DataFrame containing experiment metrics;
            export_csv_path -- optional file path to save CSV artifact;
            export_md_path  -- optional file path to save Markdown artifact.
    Output: string representation of the formatted markdown table.
    Flow:   convert DataFrame to formatted markdown string with tabulate or fallback to string,
            optionally serialize to CSV and Markdown files, and return the formatted table string.
    """
    try:
        md_table = df.to_markdown(index=False)
    except (ImportError, Exception):
        md_table = df.to_string(index=False)

    if export_csv_path:
        ensure_directory(export_csv_path)
        df.to_csv(export_csv_path, index=False)

    if export_md_path:
        ensure_directory(export_md_path)
        with open(export_md_path, "w", encoding="utf-8") as f:
            f.write(md_table)

    return md_table


if __name__ == "__main__":
    test_df = pd.DataFrame([
        {"Model": "MLP_8", "Params": 100, "Val Accuracy": 0.88, "Time": 1.2},
        {"Model": "MLP_32", "Params": 800, "Val Accuracy": 0.95, "Time": 1.5}
    ])
    print(format_summary_table(test_df))

