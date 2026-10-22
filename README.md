# Deep Learning Practice: Multi-Layer Perceptrons and Optimization

This repository contains modular implementations, experimental studies, and rigorous empirical benchmarks exploring the mechanics of Multi-Layer Perceptrons (MLP), hyperparameter sensitivity, optimization algorithms, and regularization techniques across synthetic and real-world datasets.

---

## 1. Project Organization and Architecture

In accordance with [STRUCTURE.md](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/STRUCTURE.md), the codebase is organized into distinct functional tiers:

```text
ASS--Deep-Learning/
├── .agents/                          # Agent harness, skills, and rule specifications
│   └── rules/                        # Coding and repository standards (.mdc)
├── bin/                              # Generated binaries and experiment artifacts
│   ├── day01/                        # Day 1 artifacts (task01, task02)
│   ├── day02/                        # Day 2 synthetic evidence plots
│   └── day03/                        # Day 3 artifacts (task01, task02)
├── data/                             # Raw and downloaded datasets cached at project root
├── docs/                             # Analytical reports and documentation
│   ├── day01/                        # Day 1 reports (task01, task02)
│   ├── day02/                        # Day 2 D2L deep learning chapters (task01 to task05)
│   └── day03/                        # Day 3 reports (task01, task02)
├── srcs/                             # Core source code implementations
│   ├── libraries/                    # Reusable low-level foundation building blocks
│   │   ├── seeding.py                # Deterministic random seed controller
│   │   ├── datasets.py               # Data loaders for make_moons and Fashion-MNIST
│   │   ├── models.py                 # Configurable MLP and parameter introspection
│   │   ├── metrics.py                # Evaluation metrics (Accuracy, Macro-F1)
│   │   ├── training.py               # 5-step PyTorch training loops and validation
│   │   ├── serialization.py          # Model checkpoint and metric persistence
│   │   └── common.py                 # Convenience library wrappers
│   ├── utilities/                    # High-level aggregated workflows and orchestrators
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
├── temp/                             # Reference input files
├── tools/                            # Code generation, automation, and validation scripts
│   ├── notebooks_generate.py         # Notebook generator script
│   └── notebooks_execute.py          # Notebook execution and output persistence runner
├── .gitignore                        # Git ignore patterns
├── AGENTS.md                         # Top-level agent workspace rules
├── COMMENT.md                        # Strict commenting and docstring coding standards
├── README.md                         # Project documentation and summary
├── requirements.txt                  # Virtual environment dependencies
└── STRUCTURE.md                      # Repository layout and naming specification
```

---

## 2. Standards Compliance

All Python modules and pipelines in this repository strictly enforce the rules documented in `COMMENT.md` and `STRUCTURE.md`:
- **English-Only Naming**: All files, directories, symbols, variables, functions, and documentation are strictly in English.
- **Zero Inline Comments**: No inline comments inside function bodies or on executable code lines. Code is self-documenting.
- **Four-Section Docstrings**: Every public function and class documents `Summary`, `Input`, `Output`, and `Flow`.
- **Configurable Constants**: All hyperparameters, paths, and thresholds are declared as `DEFAULT_*` constants directly after imports.
- **No Em-Dashes**: Prose exclusively uses colons, commas, periods, or parentheses.

---

## 3. Experimental Curriculum and Results

### 3.1 Day 1 Task 1: MLP Construction on `make_moons`
- **Objective**: Binary classification of two interleaving half circles (`make_moons`, 1,200 samples, noise = 0.28). Compare models with hidden widths of **8, 32, 64, and 128** neurons across Loss, Accuracy, Macro-F1, Parameter count, and Training Latency.
- **Key Findings**:
  - **8 Neurons**: Low capacity (114 parameters). The decision boundary is coarse, resulting in mild underfitting with lower validation accuracy (86.25%).
  - **32 and 64 Neurons (Sweet Spot)**: Smooth, continuous non-linear boundaries conforming cleanly to the data distribution. Reaches peak Validation Accuracy (90.00%) and Macro-F1 (0.8999).
  - **128 Neurons**: High capacity (17,154 parameters). Train loss decreases further, but validation accuracy stabilizes around 89.58% without generalization gains, showing early local curvature around individual noisy points.
- **Pipeline & Notebook**:
  - Pipeline: [srcs/pipelines/day01_task01_mlp_construction.py](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/pipelines/day01_task01_mlp_construction.py)
  - Notebook: [srcs/notebooks/day01_task01_mlp_construction.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/day01_task01_mlp_construction.ipynb)
  - Documentation: [docs/day01/task01/task01_mlp_construction.md](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/docs/day01/task01/task01_mlp_construction.md)
  - Artifacts: [bin/day01/task01/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/bin/day01/task01/)

### 3.2 Day 1 Task 2: Comprehensive MLP Exploration (Fashion-MNIST)
- **Objective**: 10-class image classification on Fashion-MNIST (784 input dimensions). Systematic ablation across:
  1. **Network Depth**: 1, 2, 3, and 4 hidden layers (128 neurons each).
  2. **Network Width**: 32, 64, 128, and 256 neurons per hidden layer (2 hidden layers).
  3. **Activation Functions**: ReLU vs Tanh vs Sigmoid.
  4. **Regularization**: Baseline (No Reg) vs Dropout ($p=0.3$) vs Weight Decay ($\lambda = 10^{-4}$) vs Combined.
- **Key Findings**:
  - **Sigmoid Saturation**: Suffers from gradient vanishing in deeper networks, converging significantly slower than ReLU.
  - **Baseline Overfitting**: Deep unregularized networks show diverging training and validation loss curves.
  - **Regularization Benefit**: Dropout ($p=0.3$) and Weight Decay successfully suppress overfitting, closing the generalization gap.
- **Pipeline & Notebook**:
  - Pipeline: [srcs/pipelines/day01_task02_mlp_exploration.py](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/pipelines/day01_task02_mlp_exploration.py)
  - Notebook: [srcs/notebooks/day01_task02_mlp_exploration.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/day01_task02_mlp_exploration.ipynb)
  - Documentation: [docs/day01/task02/task02_mlp_exploration.md](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/docs/day01/task02/task02_mlp_exploration.md)
  - Artifacts: [bin/day01/task02/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/bin/day01/task02/)

### 3.3 Day 2: D2L Theoretical Foundations and Empirical Evidence
- **Coverage**: Chapters 1 through 5 of Dive into Deep Learning (D2L):
  - Task 1: Introduction and Deep Learning Overview
  - Task 2: Preliminaries (Data Manipulation, Linear Algebra, Calculus, Autograd, Probability)
  - Task 3: Linear Neural Networks (Linear Regression, Concise Implementation, Weight Decay)
  - Task 4: Softmax Regression and Distribution Shifts
  - Task 5: Multilayer Perceptrons, Overfitting/Underfitting, Kaggle House Prices
- **Pipeline & Documentation**:
  - Pipeline: [srcs/pipelines/day02_task_evidence_generation.py](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/pipelines/day02_task_evidence_generation.py)
  - Documentation: [docs/day02/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/docs/day02/)
  - Artifacts: [bin/day02/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/bin/day02/)

### 3.4 Day 3 Task 1: Standard PyTorch Training Loop and Learning Rate Exploration
- **Objective**: Implement the standard 5-step PyTorch training cycle (`zero_grad` -> `forward` -> `loss` -> `backward` -> `optimizer.step`) and evaluate sensitivity across learning rates ($0.01$, $0.001$, $0.0001$).
- **Key Findings**:
  - $LR = 0.01$: Fast initial descent but oscillates around the minimum with high loss variance.
  - $LR = 0.001$: Optimal balance of stability and convergence speed, achieving peak validation accuracy (~85.10%).
  - $LR = 0.0001$: Steps are too small, lagging significantly behind within 5 epochs.
- **Pipeline & Notebook**:
  - Pipeline: [srcs/pipelines/day03_task01_training_loop.py](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/pipelines/day03_task01_training_loop.py)
  - Notebook: [srcs/notebooks/day03_task01_training_loop.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/day03_task01_training_loop.ipynb)
  - Documentation: [docs/day03/task01/task01_training_loop.md](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/docs/day03/task01/task01_training_loop.md)
  - Artifacts: [bin/day03/task01/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/bin/day03/task01/)

### 3.5 Day 3 Task 2: Optimizer Comparison and Early Stopping
- **Objective**: Comparative empirical evaluation of SGD, SGD with Momentum ($\beta = 0.9$), and Adam, coupled with an Early Stopping mechanism (`patience = 3`, `min_delta = 1e-4`).
- **Key Findings**:
  - **Adam**: Fastest convergence, adapting per-parameter learning rates and reaching the lowest loss in the fewest epochs.
  - **SGD with Momentum**: Overcomes ravine oscillations, matching or exceeding Adam in final validation generalization (~85.47%).
  - **Early Stopping**: Effectively intercepts overfitting and restores optimal model weights when validation loss starts degrading.
- **Pipeline & Notebook**:
  - Pipeline: [srcs/pipelines/day03_task02_optimizer_comparison.py](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/pipelines/day03_task02_optimizer_comparison.py)
  - Notebook: [srcs/notebooks/day03_task02_optimizer_comparison.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/day03_task02_optimizer_comparison.ipynb)
  - Documentation: [docs/day03/task02/task02_optimizer_comparison.md](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/docs/day03/task02/task02_optimizer_comparison.md)
  - Artifacts: [bin/day03/task02/](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/bin/day03/task02/)

---

## 4. Execution Guide

### 4.1 Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Running Standalone Pipelines
Each pipeline can be executed independently from the repository root:

```powershell
# Day 1 Task 1: MLP Construction on make_moons
python -m srcs.pipelines.day01_task01_mlp_construction --epochs 20

# Day 1 Task 2: Comprehensive MLP Ablation on Fashion-MNIST
python -m srcs.pipelines.day01_task02_mlp_exploration --epochs 12

# Day 2: D2L Evidence Generation
python -m srcs.pipelines.day02_task_evidence_generation

# Day 3 Task 1: Training Loop and Learning Rate Benchmarks
python -m srcs.pipelines.day03_task01_training_loop --epochs 5

# Day 3 Task 2: Optimizer Comparison and Early Stopping
python -m srcs.pipelines.day03_task02_optimizer_comparison --epochs 12
```

### 4.3 Notebook Generation and Automated Verification
```powershell
# Regenerate all standardized notebooks
.venv\Scripts\python.exe tools/notebooks_generate.py

# Execute all notebooks end-to-end and persist outputs
.venv\Scripts\python.exe tools/notebooks_execute.py
```
