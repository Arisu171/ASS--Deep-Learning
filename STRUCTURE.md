# Repository Structure and Naming Standards

This document establishes the mandatory architectural layout, naming conventions, and artifact persistence standards for this repository. All contributors and automated agents must adhere strictly to these rules.

---

## 1. Directory Layout

The repository is partitioned into distinct tiers to maintain clean separation between reusable components, single-task execution pipelines, documentation, and generated artifacts:

```text
ASS--Deep-Learning/
├── .agents/                          # Agent harness, skills, and rule specifications
│   ├── rules/                        # Agent rule definitions (.mdc)
│   └── skills/                       # Modular agent capability bundles
├── bin/                              # Generated binaries and experiment artifacts
│   └── dayXX/                        # Day folder (e.g., day01, day02, day03)
│       └── taskYY/                   # Task folder (e.g., task01, task02)
│           ├── models/               # Saved model weights (.pt checkpoints)
│           └── results/              # Evaluation metrics, tables, and charts (.png, .json, .csv)
├── data/                             # Raw and downloaded datasets cached at project root
├── docs/                             # Human-facing and technical documentation
│   └── dayXX/                        # Day folder (e.g., day01, day02, day03)
│       └── taskYY/                   # Task folder (e.g., task01, task02)
│           ├── <task_name>.md        # Task analytical report and findings
│           └── <images>.png          # Embedded plots and diagrams
├── srcs/                             # Source code implementations
│   ├── libraries/                    # Core reusable foundation building blocks
│   │   ├── seeding.py                # Reproducibility controller
│   │   ├── datasets.py               # Dataset loading and partitioning routines
│   │   ├── models.py                 # Neural network architectures and introspection
│   │   ├── metrics.py                # Evaluation metrics (Accuracy, Macro-F1)
│   │   ├── training.py               # 5-step PyTorch training loops and validation
│   │   ├── serialization.py          # Checkpoint persistence and disk utilities
│   │   └── common.py                 # Convenience library aggregations
│   ├── utilities/                    # Cross-task utilities, orchestrators, and exporters
│   │   ├── experiments.py            # Automated multi-configuration study orchestrators
│   │   ├── visualization.py          # Matplotlib charting and decision boundary rendering
│   │   └── tables.py                 # Tabular markdown and CSV formatting utilities
│   ├── pipelines/                    # Single-task standalone executable pipelines
│   │   ├── day01_task01_mlp_construction.py
│   │   ├── day01_task02_mlp_exploration.py
│   │   ├── day02_task_evidence_generation.py
│   │   ├── day03_task01_training_loop.py
│   │   └── day03_task02_optimizer_comparison.py
│   └── notebooks/                    # Interactive walkthrough Jupyter Notebooks
│       ├── day01_task01_mlp_construction.ipynb
│       ├── day01_task02_mlp_exploration.ipynb
│       ├── day03_task01_training_loop.ipynb
│       └── day03_task02_optimizer_comparison.ipynb
├── temp/                             # Scratch and reference input files
├── tools/                            # Code generation, automation, and validation scripts
│   ├── notebooks_generate.py         # Notebook generator script
│   └── notebooks_execute.py          # Notebook execution and output persistence runner
├── .gitignore                        # Git ignore patterns
├── AGENTS.md                         # Top-level agent workspace rules
├── COMMENT.md                        # Commenting and docstring coding standards
├── README.md                         # High-level project summary and results
├── requirements.txt                  # Virtual environment dependencies
└── STRUCTURE.md                      # This specification document
```

---

## 2. Naming Conventions

### 2.1 Language Standard
- **English Only**: Every file name, directory name, variable identifier, module name, and commit message must be in English.
- No Vietnamese accents, diacritics, or transliterated Vietnamese terms in paths or file names (e.g., use `day01_task01_mlp_construction.py`, never `bai01_xay_dung_mlp.py`).

### 2.2 Day and Task Indexing
- Use two-digit zero-padded numbers for day and task designations:
  - Days: `day01`, `day02`, `day03`, etc.
  - Tasks: `task01`, `task02`, `task03`, etc.
- In `docs/` and `bin/`, each day must have its own directory (`docs/dayXX/`, `bin/dayXX/`), and within each day directory, each task must have its own subdirectory (`taskYY/`).

### 2.3 Standalone Pipelines (`srcs/pipelines/`)
- Any Python script dedicated to a single task must reside in `srcs/pipelines/`.
- File naming must start with the day and task prefix:
  `dayXX_taskYY_<description>.py`
- Example:
  - `srcs/pipelines/day01_task01_mlp_construction.py`
  - `srcs/pipelines/day03_task02_optimizer_comparison.py`

### 2.4 Notebooks (`srcs/notebooks/`)
- Notebooks must include both day index and task index in lowercase snake_case:
  `dayXX_taskYY_<description>.ipynb`
- Example:
  - `srcs/notebooks/day01_task01_mlp_construction.ipynb`
  - `srcs/notebooks/day03_task01_training_loop.ipynb`

---

## 3. Library vs. Pipeline Architecture

1. **`srcs/libraries/`**:
   - Reusable across multiple days and tasks.
   - Contains pure functions, classes, models, and low-level routines.
   - Must not contain task-specific hardcoded configuration logic.
   - Never executes standalone business experiments on import.

2. **`srcs/utilities/`**:
   - Aggregates libraries to provide high-level workflows (multi-trial experiment sweeps, figure plotting, table formatting).
   - Serves as the middle tier between low-level libraries and high-level pipelines/notebooks.

3. **`srcs/pipelines/`**:
   - Self-contained CLI execution targets for a single task.
   - Imports from `srcs.libraries` and `srcs.utilities`.
   - Parses CLI arguments (`--epochs`, etc.) and persists model checkpoints to `bin/dayXX/taskYY/models/` and evaluation results to `bin/dayXX/taskYY/results/`.

4. **`srcs/notebooks/`**:
   - Interactive educational and analytical demonstration of each task.
   - Calls utilities and pipelines rather than reimplementing core mathematical logic.
   - Automatically derives `PROJECT_ROOT` to prevent generating nested directory artifacts.

---

## 4. Typography and Code Documentation Rules

All code and prose must respect the repository-wide constraints:
1. **No Em-Dashes**: Never use em-dashes (`—`) anywhere in prose, markdown files, docstrings, or code comments. Use colons, commas, periods, or parentheses instead.
2. **Zero Inline Comments**: No inline comments inside function bodies or on the same line as executable code (per `COMMENT.md`). Code must be self-documenting.
3. **Four-Section Docstrings**: Every public function and class must include `Summary`, `Input`, `Output`, and `Flow`.
4. **Hyperparameter Constants**: All thresholds, default paths, and hyperparameters must be declared as `DEFAULT_*` constants directly after module imports.

