# Deep Learning Practice: MLP Architecture and Hyperparameter Analysis

This repository contains modular implementations and experimental studies analyzing the mechanics of Multi-Layer Perceptrons (MLP), evaluating the empirical effects of layer depth, layer width, activation functions, and regularization techniques (Dropout and Weight Decay).

---

## 1. Project Organization and Architecture

In accordance with clean engineering design, the codebase is partitioned into distinct tiers:

```text
ASS--Deep-Learning/
├── bin/                              # Build outputs and generated model artifacts
│   ├── models/                       # Serialized PyTorch model checkpoints (.pt)
│   └── results/                      # Tables (.md, .csv), metrics (.json), plots (.png)
├── data/                             # Raw and processed datasets (e.g. FashionMNIST)
├── srcs/
│   ├── libraries/                    # Core foundation building blocks (zero inline comments)
│   │   ├── seeding.py                # Deterministic random seed controller
│   │   ├── datasets.py               # Data loaders for make_moons and FashionMNIST
│   │   ├── models.py                 # Configurable MLP and parameter introspection
│   │   ├── metrics.py                # Accuracy and Macro-F1 score calculation
│   │   ├── training.py               # Train and evaluation loops with timing
│   │   ├── serialization.py          # Model and metric persistence to disk
│   │   └── common.py                 # Convenience library wrappers
│   ├── utilities/                    # High-level aggregated workflows and orchestrators
│   │   ├── experiments.py            # Automated experiment orchestrators for Bai 01 and Bai 02
│   │   ├── visualization.py          # Decision boundary, learning curves, and bar plots
│   │   └── tables.py                 # Tabular markdown and CSV formatting utilities
│   └── notebooks/                    # Interactive step-by-step experiment notebooks
│       ├── bai01_xay_dung_mlp.ipynb  # Bai 1: make_moons MLP neuron width analysis
│       └── bai02_khao_sat_mlp.ipynb  # Bai 2: Comprehensive MLP ablation and regularization study
├── requirements.txt                  # Virtual environment dependencies
├── COMMENT.md                        # Strict commenting and docstring coding standards
└── README.md                         # Project documentation and summary
```

---

## 2. Standards Compliance (COMMENT.md)

All Python modules in `srcs/` strictly enforce the standards specified in `COMMENT.md`:
- **Zero Inline Comments**: No inline comments inside function bodies or on executable code lines. Code is self-documenting.
- **Module-Level Docstrings**: Structured on line 1 with purpose, CLI usage, inputs, and outputs.
- **Three-Section Docstrings**: Every function and class documents `Summary`, `Input`, `Output`, and `Flow`.
- **Configurable Constants**: All hyperparameters declared as `DEFAULT_*` constants at top of modules.
- **No Em-Dashes**: Prose exclusively uses colons, commas, periods, or parentheses.

---

## 3. Bai 1: Xay Dung MLP Tren `make_moons`

### 3.1 Objective
Binary classification of two interleaving half circles (`make_moons`, 1,200 samples, noise = 0.28). Compare models with hidden widths of **8, 32, 64, and 128** neurons across Loss, Accuracy, Macro-F1, Parameter count, and Training Latency.

### 3.2 Quantitative Results

| Model | Hidden Neurons | Parameters | Train Loss | Val Loss | Train Accuracy | Val Accuracy | Val Macro-F1 | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP_hidden_8** | 8 | 114 | 0.3158 | 0.3734 | 0.8740 | 0.8625 | 0.8624 | 0.931 |
| **MLP_hidden_32** | 32 | 1,218 | 0.2105 | 0.2748 | 0.9062 | 0.8958 | 0.8957 | 0.909 |
| **MLP_hidden_64** | 64 | 4,482 | 0.1892 | 0.2752 | 0.9208 | 0.9000 | 0.8999 | 0.876 |
| **MLP_hidden_128** | 128 | 17,154 | 0.1835 | 0.2784 | 0.9177 | 0.8958 | 0.8957 | 0.971 |

### 3.3 Visualizations and Findings
- **8 Neurons**: Low capacity (114 parameters). The decision boundary is coarse, resulting in mild underfitting with lower validation accuracy (86.25%).
- **32 and 64 Neurons (Optimal Sweet Spot)**: Generates a smooth, continuous non-linear boundary conforming cleanly to the data distribution. Reaches the peak Validation Accuracy (90.00%) and Macro-F1 (0.8999) with balanced computational resource footprint.
- **128 Neurons**: High capacity (17,154 parameters, roughly 4x larger than 64 neurons). Train loss decreases further, but validation accuracy stabilizes around 89.58% without generalization gains, showing early local curvature around individual noisy points.
- Artifacts generated:
  - Checkpoint: `bin/models/bai01_mlp_neurons_64.pt`
  - Visualizations: `bin/results/bai01_decision_boundaries.png`, `bin/results/bai01_learning_curves.png`, `bin/results/bai01_metrics_comparison.png`.

---

## 4. Bai 2: Khao Sat Toan Dien MLP (Fashion-MNIST)

### 4.1 Objective
Multi-class classification on the Fashion-MNIST dataset (10 categories, 784 flattened input dimensions). Systematically assess:
1. **Network Depth**: 1, 2, 3, and 4 hidden layers (128 neurons each).
2. **Network Width**: 32, 64, 128, and 256 neurons per hidden layer (2 hidden layers).
3. **Activation Functions**: ReLU vs Tanh vs Sigmoid.
4. **Regularization (Advanced)**: Baseline (No Reg) vs Dropout ($p=0.3$) vs Weight Decay ($\lambda = 10^{-4}$) vs Combined.

### 4.2 Quantitative Results

| Experiment | Hidden Layers | Num Layers | Activation | Dropout | Weight Decay | Parameters | Train Loss | Val Loss | Train Acc | Val Acc | Val Macro-F1 | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Depth_1_Layer** | [128] | 1 | relu | 0.0 | 0.0 | 101,770 | 0.2107 | 0.4431 | 0.9247 | 0.8529 | 0.8534 | 33.94 |
| **Depth_2_Layers** | [128, 128] | 2 | relu | 0.0 | 0.0 | 118,282 | 0.2162 | 0.4966 | 0.9192 | 0.8436 | 0.8433 | 35.69 |
| **Depth_3_Layers** | [128, 128, 128] | 3 | relu | 0.0 | 0.0 | 134,794 | 0.2269 | 0.4680 | 0.9160 | 0.8489 | 0.8508 | 35.72 |
| **Depth_4_Layers** | [128, 128, 128, 128] | 4 | relu | 0.0 | 0.0 | 151,306 | 0.2595 | 0.4883 | 0.8980 | 0.8380 | 0.8418 | 35.50 |
| **Width_32_Neurons** | [32, 32] | 2 | relu | 0.0 | 0.0 | 26,506 | 0.3056 | 0.4632 | 0.8878 | 0.8389 | 0.8373 | 33.93 |
| **Width_64_Neurons** | [64, 64] | 2 | relu | 0.0 | 0.0 | 55,050 | 0.2498 | 0.4483 | 0.9088 | 0.8467 | 0.8479 | 34.68 |
| **Width_128_Neurons** | [128, 128] | 2 | relu | 0.0 | 0.0 | 118,282 | 0.2162 | 0.4966 | 0.9192 | 0.8436 | 0.8433 | 34.43 |
| **Width_256_Neurons** | [256, 256] | 2 | relu | 0.0 | 0.0 | 269,322 | 0.1783 | 0.4871 | 0.9312 | 0.8530 | 0.8536 | 35.39 |
| **Activation_ReLU** | [128, 64] | 2 | relu | 0.0 | 0.0 | 109,386 | 0.2259 | 0.4544 | 0.9165 | 0.8472 | 0.8474 | 34.84 |
| **Activation_Tanh** | [128, 64] | 2 | tanh | 0.0 | 0.0 | 109,386 | 0.2208 | 0.4480 | 0.9180 | 0.8488 | 0.8487 | 35.12 |
| **Activation_Sigmoid** | [128, 64] | 2 | sigmoid | 0.0 | 0.0 | 109,386 | 0.3288 | 0.4371 | 0.8882 | 0.8499 | 0.8490 | 34.79 |
| **Baseline_NoReg** | [256, 128, 64] | 3 | relu | 0.0 | 0.0 | 242,762 | 0.1602 | 0.5141 | 0.9388 | 0.8516 | 0.8525 | 44.63 |
| **Dropout_p03** | [256, 128, 64] | 3 | relu | 0.3 | 0.0 | 242,762 | 0.3305 | 0.4530 | 0.8790 | 0.8498 | 0.8482 | 45.92 |
| **WeightDecay_1e4** | [256, 128, 64] | 3 | relu | 0.0 | 0.0001 | 242,762 | 0.1695 | 0.5181 | 0.9382 | 0.8504 | 0.8509 | 51.37 |
| **Combined_Drop_WD** | [256, 128, 64] | 3 | relu | 0.3 | 0.0001 | 242,762 | 0.3333 | 0.4532 | 0.8753 | 0.8493 | 0.8463 | 58.02 |

### 4.3 Analysis of Underfitting and Overfitting
1. **Underfitting Dynamics**:
   - **Sigmoid Saturation**: The Sigmoid activation suffers from gradient vanishing in deeper networks as activations saturate outside $[-2, 2]$. Consequently, Train Loss converges much more slowly (0.3288 vs 0.2259 for ReLU).
   - **Small Width**: An MLP with only 32 neurons per layer lacks representation capacity, showing higher training loss (0.3056) and lower overall accuracy.
2. **Overfitting Dynamics**:
   - **Baseline Unregularized Network**: For a deep network (`[256, 128, 64]`), Train Loss drops aggressively to 0.1602 (Train Accuracy 93.88%), but Validation Loss balloons to 0.5141. The prominent divergence between training and validation loss curves is the definitive hallmark of overfitting.
3. **Mitigation Through Regularization**:
   - **Dropout ($p=0.3$)**: Successfully suppresses overfitting. Validation loss drops from 0.5141 down to 0.4530. The generalization gap narrows significantly because randomly masking activations prevents co-adaptation of features.
   - **Combined Regularization**: Pairing Dropout with Weight Decay ($L_2$ regularization) yields a stable model with well-behaved validation trajectories.

---

## 5. Execution Guide

### 5.1 Setting up Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 5.2 Running via CLI
```powershell
# Run Bai 01 (make_moons neuron analysis)
python -m srcs.utilities.experiments --task bai01 --epochs 20

# Run Bai 02 (Fashion-MNIST ablation study)
python -m srcs.utilities.experiments --task bai02 --epochs 12
```

### 5.3 Interactive Exploration via Jupyter
Launch Jupyter or open directly in VS Code:
- [srcs/notebooks/bai01_xay_dung_mlp.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/bai01_xay_dung_mlp.ipynb)
- [srcs/notebooks/bai02_khao_sat_mlp.ipynb](file:///e:/Data/Devdata/Freelancers/Practice/ASS--Deep-Learning/srcs/notebooks/bai02_khao_sat_mlp.ipynb)

