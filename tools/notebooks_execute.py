"""Execute notebooks and write back cell outputs.

CLI execution:
  python temp/run_and_save_notebooks.py

Inputs:
  srcs/notebooks/*.ipynb

Outputs:
  Populated notebooks with rendered charts, tables, and stdout logs.
"""

import os
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "srcs/notebooks/bai01_xay_dung_mlp.ipynb",
    "srcs/notebooks/bai02_khao_sat_mlp.ipynb"
]

for nb_path in NOTEBOOKS:
    print(f"\n==========================================")
    print(f"Executing: {nb_path}")
    print(f"==========================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(nb_path))}}
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully executed and saved: {nb_path}")

print("\nAll notebooks executed and verified successfully!")

