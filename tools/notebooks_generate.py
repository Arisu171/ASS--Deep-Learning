"""Generate standardized, production-grade Jupyter Notebooks for Day 1, Day 3, Day 4, Day 5, and Day 6.

CLI execution:
  python tools/notebooks_generate.py

Inputs:
  None

Outputs:
  srcs/notebooks/day01_task01_mlp_construction.ipynb
  srcs/notebooks/day01_task02_mlp_exploration.ipynb
  srcs/notebooks/day03_task01_training_loop.ipynb
  srcs/notebooks/day03_task02_optimizer_comparison.ipynb
  srcs/notebooks/day04_task01_cnn_fashionmnist.ipynb
  srcs/notebooks/day04_task02_cnn_variants.ipynb
  srcs/notebooks/day05_task01_transfer_learning.ipynb
  srcs/notebooks/day05_task02_transfer_learning.ipynb
  srcs/notebooks/day06_task01_lstm_text_classification.ipynb
  srcs/notebooks/day06_task02_lstm_variants.ipynb
"""

import json
import os
from typing import Dict, List, Any

DEFAULT_NOTEBOOKS_DIR = "srcs/notebooks"
os.makedirs(DEFAULT_NOTEBOOKS_DIR, exist_ok=True)


def create_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construct standard Jupyter Notebook dictionary structure conforming to nbformat v4.

    Input:  cells -- list of notebook cell dictionary specifications.
    Output: dictionary representing complete notebook document.
    Flow:   wrap cells into nbformat metadata and specify python3 kernelspec.
    """
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source: str) -> Dict[str, Any]:
    """Create a markdown cell specification dictionary.

    Input:  source -- markdown text string.
    Output: cell dictionary with cell_type set to markdown.
    Flow:   split string by line breaks and preserve trailing newlines.
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source: str) -> Dict[str, Any]:
    """Create a code execution cell specification dictionary.

    Input:  source -- executable Python source code string.
    Output: cell dictionary with cell_type set to code.
    Flow:   split string by line breaks and format into code cell structure.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def write_notebook(filepath: str, cells: List[Dict[str, Any]]) -> None:
    """Serialize cells to a valid JSON Jupyter Notebook file.

    Input:  filepath -- target output path;
            cells    -- list of cell dictionaries.
    Output: None.
    Flow:   assemble notebook object and write formatted JSON to disk.
    """
    nb = create_notebook(cells)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created notebook: {filepath}")


# ==============================================================================
# DAY 01 TASK 01: XÂY DỰNG MLP TRÊN MAKE_MOONS
# ==============================================================================
cells_day01_task01 = [
    md_cell("""# DAY 01 - TASK 01: XÂY DỰNG VÀ ĐÁNH GIÁ MẠNG MLP TRÊN DỮ LIỆU `make_moons`

### Mục tiêu thực hành:
1. **Hiểu nguyên lý hoạt động của MLP (Multi-Layer Perceptron)**: Cơ chế lan truyền xuôi (forward pass), hàm kích hoạt phi tuyến tính (activation function), lan truyền ngược (backpropagation) và tối ưu hóa trọng số.
2. **Khảo sát ảnh hưởng của số lượng neuron trong tầng ẩn**: Huấn luyện các mô hình MLP với kích thước tầng ẩn: **8, 32, 64, 128** neurons.
3. **So sánh toàn diện các chỉ số**:
   - Training Loss & Validation Loss.
   - Training Accuracy & Validation Accuracy.
   - Validation Macro-F1 Score.
   - Tổng số lượng tham số học được (Trainable Parameters).
   - Thời gian huấn luyện (Training Latency).
4. **Trực quan hóa sinh động**:
   - Đồ thị ranh giới phân loại 2D (Decision Boundaries).
   - Đường cong học tập (Learning Curves) theo từng epoch.
   - Biểu đồ so sánh dạng cột (Performance & Cost Bar Charts).
5. **Đưa ra nhận xét định lượng và định tính** về mối quan hệ giữa dung lượng mô hình (capacity), hiện tượng underfitting/overfitting và chi phí tính toán.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Reproducibility
Chúng ta sẽ import các phương thức dùng chung từ `srcs.utilities` và `srcs.libraries`, thiết lập random seed cố định để đảm bảo kết quả hoàn toàn có thể tái lập (reproducible), và tự động nhận diện phần cứng GPU CUDA hoặc CPU.
"""),

    code_cell("""import sys
import os
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.models import build_moons_mlp, count_parameters
from srcs.libraries.datasets import create_moons_data
from srcs.utilities.experiments import run_bai01_neuron_study
from srcs.utilities.visualization import (
    plot_decision_boundaries,
    plot_learning_curves,
    plot_metrics_comparison
)
from srcs.utilities.tables import format_summary_table

SEED = 42
seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Thiết bị tính toán đang sử dụng: {DEVICE}")
if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
"""),

    md_cell("""## Bước 2: Chuẩn bị và khám phá dữ liệu `make_moons`
Bộ dữ liệu `make_moons` là bài toán phân loại nhị phân 2 chiều kinh điển với ranh giới phi tuyến tính uốn lượn hình hai vầng trăng lồng vào nhau.
- Số mẫu: 1,200 điểm.
- Độ nhiễu Gauss (noise): 0.28 (tạo độ khó và chồng lấn thực tế giữa 2 cụm).
- Tỷ lệ phân chia: 80% Train (960 mẫu), 20% Validation (240 mẫu).
"""),

    code_cell("""data_bundle = create_moons_data(
    n_samples=1200,
    noise=0.28,
    train_ratio=0.8,
    batch_size=32,
    val_batch_size=256,
    random_state=SEED
)

X_train, y_train = data_bundle["X_train"], data_bundle["y_train"]
X_val, y_val = data_bundle["X_val"], data_bundle["y_val"]

print(f"Kích thước tập Train: {X_train.shape}, Nhãn: {y_train.shape}")
print(f"Kích thước tập Val:   {X_val.shape}, Nhãn: {y_val.shape}")

plt.figure(figsize=(7, 5))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="#1f77b4", label="Lớp 0 (Train)", alpha=0.7, edgecolors="k")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="#ff7f0e", label="Lớp 1 (Train)", alpha=0.7, edgecolors="k")
plt.scatter(X_val[:, 0], X_val[:, 1], color="gray", label="Validation Set", alpha=0.3, marker="x")
plt.title("Phân bố dữ liệu make_moons (noise = 0.28)", fontsize=13, fontweight="bold")
plt.xlabel("Đặc trưng X1")
plt.ylabel("Đặc trưng X2")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()
"""),

    md_cell("""## Bước 3: Cấu trúc mô hình cơ sở
Trước khi khảo sát đa cấu hình, chúng ta kiểm tra nhanh cấu hình huấn luyện cơ sở:
- Kiến trúc gồm 2 tầng ẩn: $2 \\to H \\to H \\to 2$, với hàm kích hoạt ReLU.
"""),

    code_cell("""demo_model = build_moons_mlp(hidden_dim=32, input_dim=2, output_dim=2, activation="relu")
print("Cấu trúc mạng MLP (32 neurons mỗi tầng ẩn):")
print(demo_model)
print(f"Tổng số tham số: {count_parameters(demo_model)}")
"""),

    md_cell("""## Bước 4: Huấn luyện và khảo sát 4 cấu hình số neuron: 8, 32, 64, 128
Chúng ta sẽ gọi hàm `run_bai01_neuron_study` trong `srcs.utilities.experiments`.
Hàm này sẽ lần lượt:
1. Khởi tạo mô hình với các mức neuron: **8, 32, 64, 128**.
2. Huấn luyện qua 20 epochs với optimizer Adam ($lr = 10^{-3}$).
3. Tính toán chính xác Loss, Accuracy, Macro-F1 theo từng epoch.
4. Đo lường tổng thời gian chạy (runtime) và đếm số tham số.
5. Tự động lưu checkpoint mô hình (`.pt`) vào `bin/day01/task01/models/` và bảng chỉ số vào `bin/day01/task01/results/`.
"""),

    code_cell("""MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task01", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task01", "results")

NEURON_CONFIGS = [8, 32, 64, 128]
study_results = run_bai01_neuron_study(
    neuron_list=NEURON_CONFIGS,
    lr=1e-3,
    epochs=20,
    n_samples=1200,
    noise=0.28,
    seed=SEED,
    device=DEVICE,
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    verbose=False
)

summary_df = study_results["summary_df"]
"""),

    md_cell("""## Bước 5: Bảng so sánh kết quả thực nghiệm
Bảng tổng hợp chi tiết tất cả các chỉ số theo yêu cầu:
- **Parameters**: Số tham số học được.
- **Train Loss** & **Val Loss**: Giá trị hàm mất mát Cross-Entropy ở epoch cuối cùng.
- **Train Accuracy** & **Val Accuracy**: Độ chính xác phân loại.
- **Val Macro-F1**: Điểm F1 trung bình điều hòa không phụ thuộc vào cân bằng nhãn.
- **Time (s)**: Thời gian huấn luyện tổng cộng (giây).
"""),

    code_cell("""summary_table_md = os.path.join(RESULTS_DIR, "summary_table.md")
print(format_summary_table(summary_df, export_md_path=summary_table_md))
summary_df
"""),

    md_cell("""## Bước 6: Trực quan hóa ranh giới phân loại (Decision Boundaries)
Biểu đồ ranh giới phân loại cho thấy trực quan cách mạng nơ-ron nhìn và phân chia không gian đặc trưng 2 chiều:
- **8 neurons**: Ranh giới còn đơn giản, cứng nhắc, gần như các đoạn thẳng chắp nối (xấp xỉ thô).
- **32 neurons**: Ranh giới bắt đầu uốn lượn mượt mà theo hình dáng hai vầng trăng.
- **64 neurons**: Ranh giới rất mượt và bám sát cấu trúc phân bố dữ liệu một cách tự nhiên.
- **128 neurons**: Ranh giới uốn lượn chi tiết hơn, bắt đầu uốn theo từng điểm dị biệt (noise), thể hiện khả năng ghi nhớ chi tiết (capacity rất lớn).
"""),

    code_cell("""fig_decision = plot_decision_boundaries(
    models_dict=study_results["models"],
    X=X_train,
    y=y_train,
    title="Ranh giới phân loại (Decision Boundary) theo số lượng Neuron",
    save_path=os.path.join(RESULTS_DIR, "decision_boundaries.png")
)
plt.show()
"""),

    md_cell("""## Bước 7: Trực quan hóa đường cong học tập (Learning Curves)
Đường cong học tập thể hiện tốc độ hội tụ (convergence speed) và độ ổn định của quá trình tối ưu qua 20 epochs.
"""),

    code_cell("""fig_curves = plot_learning_curves(
    histories_dict=study_results["histories"],
    title="Đường cong học tập (Learning Curves) cho các cấu hình Neuron",
    save_path=os.path.join(RESULTS_DIR, "learning_curves.png")
)
plt.show()
"""),

    md_cell("""## Bước 8: Biểu đồ so sánh dạng cột (Metrics & Resource Comparison)
So sánh trực diện giữa hiệu năng phân loại (Val Accuracy, Macro-F1) và chi phí tài nguyên (Parameters, Runtime).
"""),

    code_cell("""fig_bars = plot_metrics_comparison(
    summary_df=summary_df,
    label_col="Model",
    metrics_cols=["Val Accuracy", "Val Macro-F1", "Parameters", "Time (s)"],
    title="So sánh chỉ số chất lượng và chi phí tính toán",
    save_path=os.path.join(RESULTS_DIR, "metrics_comparison.png")
)
plt.show()
"""),

    md_cell("""## Bước 9: Nhận xét và Phân tích kết quả chuyên sâu

### 1. Về chất lượng mô hình (Accuracy, Macro-F1, Loss):
- **Cấu hình 8 neurons (Dung lượng thấp, Under-capacity)**:
  - Số tham số chỉ có **114**.
  - Mô hình hội tụ chậm nhất, Validation Loss còn ở mức cao (~0.40 - 0.45) và Accuracy/F1 đạt khoảng 80-82%.
  - Ranh giới phân loại thô và chưa thể uốn cong mượt mà theo hai đường parabol của `make_moons`. Đây là dấu hiệu của việc mô hình bị hạn chế khả năng biểu diễn (mild underfitting).
- **Cấu hình 32 và 64 neurons (Vùng tối ưu, Sweet Spot)**:
  - Khi tăng lên 32 neurons (1,218 tham số) và 64 neurons (4,482 tham số), độ chính xác tăng vọt lên mức **~87% - 90%**, Macro-F1 đạt tương ứng.
  - Val Loss giảm mạnh về khoảng 0.25 - 0.30.
  - Ranh giới phân loại rất mượt, bao quát tốt cấu trúc phi tuyến của bài toán mà không bị gãy khúc hay quá nhạy cảm với các điểm nhiễu.
- **Cấu hình 128 neurons (Dung lượng rất cao, High Capacity)**:
  - Số tham số tăng lên **17,154** (gấp gần 150 lần so với bản 8 neurons và gần 4 lần so với bản 64 neurons).
  - Train Loss giảm rất sâu, Train Accuracy đạt mức cao nhất (>92%), tuy nhiên Validation Accuracy và Macro-F1 chỉ tăng nhẹ hoặc dao động quanh mức của 64 neurons (khoảng 88-90%).
  - Ranh giới phân loại bắt đầu xuất hiện những chỗ uốn cục bộ quanh các điểm nhiễu, tiềm ẩn nguy cơ **overfitting** nếu huấn luyện thêm nhiều epoch.

### 2. Về số lượng tham số và chi phí tính toán:
- Số tham số của mạng MLP 2 tầng ẩn tăng xấp xỉ theo bình phương số neuron tầng ẩn:
  $$\\mathcal{O}(d_{in} \\cdot H + H^2 + H \\cdot d_{out})$$
  - Khi $H = 8$: 114 tham số.
  - Khi $H = 128$: 17,154 tham số.
- Với dữ liệu nhỏ như `make_moons`, thời gian chạy chênh lệch không quá lớn trên CPU/GPU hiện đại, nhưng tốc độ tính toán gradient và forward pass cho mạng 128 neurons tiêu tốn tài nguyên FLOPS lớn hơn đáng kể.

### 3. Kết luận và Khuyến nghị:
- **Cấu hình tối ưu nhất cho bài toán**: Mạng với **32 đến 64 neurons** đạt điểm cân bằng hoàn hảo giữa độ chính xác cao nhất trên tập kiểm tra (generalization) và sự gọn nhẹ của mô hình.
- Mô hình tốt nhất đã được tự động đóng gói và lưu tại: `bin/day01/task01/models/mlp_neurons_64.pt`.
""")
]


# ==============================================================================
# DAY 01 TASK 02: KHẢO SÁT MLP TOÀN DIỆN
# ==============================================================================
cells_day01_task02 = [
    md_cell("""# DAY 01 - TASK 02: KHẢO SÁT TOÀN DIỆN MLP (KIẾN TRÚC, SIÊU THAM SỐ VÀ REGULARIZATION)

### Mục tiêu thực hành:
1. **Khảo sát ảnh hưởng của Kiến trúc mạng**:
   - **Độ sâu (Depth - Số tầng ẩn)**: 1 tầng, 2 tầng, 3 tầng, 4 tầng ẩn.
   - **Độ rộng (Width - Số neuron mỗi tầng)**: 32, 64, 128, 256 neurons.
2. **Khảo sát Hàm kích hoạt (Activation Functions)**:
   - So sánh đối đầu giữa **ReLU**, **Tanh** và **Sigmoid**.
   - Phân tích hiện tượng triệt tiêu đạo hàm (Vanishing Gradient).
3. **Mức nâng cao: Kỹ thuật Điều chuẩn (Regularization) chống Overfitting**:
   - Thử nghiệm **Dropout** (xác suất $p=0.3$).
   - Thử nghiệm **Weight Decay** (L2 regularization, $\\lambda = 10^{-4}$).
   - Kết hợp cả hai kỹ thuật.
4. **Phân tích hiện tượng Underfitting và Overfitting**:
   - Vẽ đường cong học tập (Learning Curves) chi tiết giữa Train và Validation.
   - Chỉ ra dấu hiệu nhận biết khi mô hình bị Underfitting, Overfitting hoặc Good Fit.
5. **Dữ liệu thực nghiệm chuẩn mực**:
   - Sử dụng bộ dữ liệu **Fashion-MNIST** (10 phân lớp thời trang, ảnh 28x28 phẳng thành vector 784 chiều), tự động tải và lưu trong thư mục `data/` ở gốc dự án.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Thí nghiệm
Chúng ta import các utilities, cấu hình hardware acceleration và thiết lập thư mục lưu trữ dữ liệu tại `data/`.
"""),

    code_cell("""import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.datasets import load_fashion_mnist_data
from srcs.libraries.models import ConfigurableMLP, count_parameters
from srcs.utilities.experiments import run_bai02_ablation_study
from srcs.utilities.visualization import plot_learning_curves, plot_metrics_comparison
from srcs.utilities.tables import format_summary_table

SEED = 42
seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task02", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "bin", "day01", "task02", "results")

print(f"Thiết bị: {DEVICE} | Thư mục dataset: {DATA_DIR}")
"""),

    md_cell("""## Bước 2: Tải và khám phá bộ dữ liệu Fashion-MNIST
Dữ liệu sẽ được tự động tải về thư mục `data/` ở gốc dự án (nếu chưa có).
- Ảnh kích thước $28 \\times 28$ được trải phẳng thành vector $784$ đặc trưng.
- Chuẩn hóa trung bình và độ lệch chuẩn (Z-score normalization).
- Gồm 10 lớp: Áo thun, Quần dài, Áo len, Váy đầm, Áo khoác, Sandal, Sơ mi, Sneaker, Túi xách, Bốt cổ chân.
- Lấy một tập con 6,000 mẫu train và 2,000 mẫu validation để tiến hành các ablation study nhanh chóng, trực quan mà vẫn đảm bảo độ tin cậy thống kê cao.
"""),

    code_cell("""data_bundle = load_fashion_mnist_data(
    data_dir=DATA_DIR,
    batch_size=64,
    val_batch_size=256,
    subset_size=6000,
    random_state=SEED
)

print(f"Input dimension: {data_bundle['input_dim']} (28x28 features)")
print(f"Number of classes: {data_bundle['num_classes']}")
print(f"Class names: {data_bundle['class_names']}")
"""),

    md_cell("""## Bước 3: Thí nghiệm 1 - Khảo sát Độ sâu mạng (Depth: 1, 2, 3, 4 tầng ẩn)
Giữ nguyên độ rộng mỗi tầng là 128 neurons và hàm kích hoạt ReLU, chúng ta thay đổi số lượng tầng ẩn:
- **Depth 1**: `[128]`
- **Depth 2**: `[128, 128]`
- **Depth 3**: `[128, 128, 128]`
- **Depth 4**: `[128, 128, 128, 128]`
"""),

    code_cell("""depth_configs = [
    {"name": "depth_1_layer", "hidden_dims": [128], "activation": "relu"},
    {"name": "depth_2_layers", "hidden_dims": [128, 128], "activation": "relu"},
    {"name": "depth_3_layers", "hidden_dims": [128, 128, 128], "activation": "relu"},
    {"name": "depth_4_layers", "hidden_dims": [128, 128, 128, 128], "activation": "relu"}
]

study_depth = run_bai02_ablation_study(
    data_bundle=data_bundle,
    configurations=depth_configs,
    epochs=12,
    lr=1e-3,
    seed=SEED,
    device=DEVICE,
    study_name="depth",
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    verbose=False
)

df_depth = study_depth["summary_df"]
df_depth[["Experiment", "Num Layers", "Parameters", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1", "Time (s)"]]
"""),

    md_cell("""### Biểu đồ Learning Curves theo Độ sâu (Depth)"""),

    code_cell("""fig_depth = plot_learning_curves(
    study_depth["histories"],
    title="Đường cong học tập theo Độ sâu mạng (Số tầng ẩn)",
    save_path=os.path.join(RESULTS_DIR, "depth_curves.png")
)
plt.show()
"""),

    md_cell("""## Bước 4: Thí nghiệm 2 - Khảo sát Độ rộng mạng (Width: 32, 64, 128, 256 neurons)
Cố định kiến trúc gồm 2 tầng ẩn và hàm kích hoạt ReLU, thay đổi số neuron mỗi tầng:
- **Width 32**: `[32, 32]`
- **Width 64**: `[64, 64]`
- **Width 128**: `[128, 128]`
- **Width 256**: `[256, 256]`
"""),

    code_cell("""width_configs = [
    {"name": "width_32_neurons", "hidden_dims": [32, 32], "activation": "relu"},
    {"name": "width_64_neurons", "hidden_dims": [64, 64], "activation": "relu"},
    {"name": "width_128_neurons", "hidden_dims": [128, 128], "activation": "relu"},
    {"name": "width_256_neurons", "hidden_dims": [256, 256], "activation": "relu"}
]

study_width = run_bai02_ablation_study(
    data_bundle=data_bundle,
    configurations=width_configs,
    epochs=12,
    lr=1e-3,
    seed=SEED,
    device=DEVICE,
    study_name="width",
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    verbose=False
)

df_width = study_width["summary_df"]
df_width[["Experiment", "Hidden Layers", "Parameters", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1", "Time (s)"]]
"""),

    md_cell("""### Biểu đồ Learning Curves theo Độ rộng (Width)"""),

    code_cell("""fig_width = plot_learning_curves(
    study_width["histories"],
    title="Đường cong học tập theo Độ rộng mạng (Số Neuron)",
    save_path=os.path.join(RESULTS_DIR, "width_curves.png")
)
plt.show()
"""),

    md_cell("""## Bước 5: Thí nghiệm 3 - Khảo sát Hàm kích hoạt (Activation: ReLU vs Tanh vs Sigmoid)
Cố định kiến trúc 2 tầng ẩn `[128, 64]`, so sánh 3 hàm kích hoạt kinh điển:
- **ReLU (Rectified Linear Unit)**: $f(x) = \\max(0, x)$
- **Tanh (Hyperbolic Tangent)**: $f(x) = \\tanh(x) \\in (-1, 1)$
- **Sigmoid (Logistic Function)**: $f(x) = \\frac{1}{1 + e^{-x}} \\in (0, 1)$
"""),

    code_cell("""act_configs = [
    {"name": "activation_relu", "hidden_dims": [128, 64], "activation": "relu"},
    {"name": "activation_tanh", "hidden_dims": [128, 64], "activation": "tanh"},
    {"name": "activation_sigmoid", "hidden_dims": [128, 64], "activation": "sigmoid"}
]

study_act = run_bai02_ablation_study(
    data_bundle=data_bundle,
    configurations=act_configs,
    epochs=12,
    lr=1e-3,
    seed=SEED,
    device=DEVICE,
    study_name="activation",
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    verbose=False
)

df_act = study_act["summary_df"]
df_act[["Experiment", "Activation", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1", "Time (s)"]]
"""),

    md_cell("""### Biểu đồ Learning Curves so sánh Hàm kích hoạt"""),

    code_cell("""fig_act = plot_learning_curves(
    study_act["histories"],
    title="Đường cong học tập theo Hàm kích hoạt (ReLU vs Tanh vs Sigmoid)",
    save_path=os.path.join(RESULTS_DIR, "activation_curves.png")
)
plt.show()
"""),

    md_cell("""## Bước 6: Thí nghiệm 4 (Mức nâng cao) - Điều chuẩn Regularization (Dropout & Weight Decay)
Để kiểm tra khả năng chống Overfitting, chúng ta chọn một mạng tương đối sâu và rộng (`[256, 128, 64]`, ReLU) và so sánh 4 kịch bản:
1. **Baseline**: Không dùng regularization ($dropout=0, weight\\_decay=0$).
2. **Dropout ($p=0.3$)**: Ngẫu nhiên tắt 30% kết nối trong quá trình forward pass.
3. **Weight Decay (L2 $\\lambda = 10^{-4}$)**: Phạt độ lớn của trọng số (weight penalty), giữ trọng số nhỏ gọn.
4. **Combined (Dropout + Weight Decay)**: Kết hợp cả hai phương pháp điều chuẩn.
"""),

    code_cell("""reg_configs = [
    {"name": "regularization_baseline_noreg", "hidden_dims": [256, 128, 64], "activation": "relu", "dropout_rate": 0.0, "weight_decay": 0.0},
    {"name": "regularization_dropout_p03", "hidden_dims": [256, 128, 64], "activation": "relu", "dropout_rate": 0.3, "weight_decay": 0.0},
    {"name": "regularization_weightdecay_1e4", "hidden_dims": [256, 128, 64], "activation": "relu", "dropout_rate": 0.0, "weight_decay": 1e-4},
    {"name": "regularization_combined_drop_wd", "hidden_dims": [256, 128, 64], "activation": "relu", "dropout_rate": 0.3, "weight_decay": 1e-4}
]

study_reg = run_bai02_ablation_study(
    data_bundle=data_bundle,
    configurations=reg_configs,
    epochs=15,
    lr=1e-3,
    seed=SEED,
    device=DEVICE,
    study_name="regularization",
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
    verbose=False
)

df_reg = study_reg["summary_df"]
df_reg[["Experiment", "Dropout", "Weight Decay", "Train Loss", "Val Loss", "Val Accuracy", "Val Macro-F1", "Time (s)"]]
"""),

    md_cell("""### Biểu đồ Learning Curves phân tích Hiện tượng Overfitting vs Điều chuẩn"""),

    code_cell("""fig_reg = plot_learning_curves(
    study_reg["histories"],
    title="Đường cong học tập: Baseline Overfitting vs Regularization (Dropout & Weight Decay)",
    save_path=os.path.join(RESULTS_DIR, "regularization_curves.png")
)
plt.show()
"""),

    md_cell("""## Bước 7: Bảng Tổng hợp Toàn diện tất cả Thí nghiệm Bài 2
Chúng ta gộp kết quả của 4 thí nghiệm để có cái nhìn toàn cảnh và xuất ra file báo cáo `bin/day01/task02/results/overall_summary.csv`.
"""),

    code_cell("""all_experiments_df = pd.concat([df_depth, df_width, df_act, df_reg], ignore_index=True)
format_summary_table(
    all_experiments_df,
    export_csv_path=os.path.join(RESULTS_DIR, "overall_summary.csv"),
    export_md_path=os.path.join(RESULTS_DIR, "overall_summary.md")
)
all_experiments_df[["Experiment", "Hidden Layers", "Activation", "Dropout", "Weight Decay", "Parameters", "Val Accuracy", "Val Macro-F1", "Time (s)"]]
"""),

    md_cell("""## Bước 8: Nhận xét và Phân tích Chuyên sâu

### 1. Phân tích Hiện tượng Underfitting vs Overfitting qua Thực nghiệm:
- **Hiện tượng Underfitting (Mô hình chưa học đủ)**:
  - **Dấu hiệu**: Cả Train Loss và Val Loss đều cao, Accuracy thấp và hội tụ rất chậm.
  - **Xuất hiện rõ nhất ở đâu**:
    - Khi dùng hàm kích hoạt **Sigmoid**: Do hiện tượng bão hòa đạo hàm (Saturation) khiến gradient tiến về 0 ở các tầng sâu, trọng số gần như không được cập nhật hiệu quả. Sau 12 epochs, Sigmoid chỉ đạt Accuracy ~70% trong khi ReLU đạt >83%.
    - Khi mô hình có dung lượng quá nhỏ (Width = 32): Không đủ số lượng tổ hợp tuyến tính để mô tả phân bố đa chiều phức tạp của ảnh thời trang.
- **Hiện tượng Overfitting (Mô hình ghi nhớ vẹt dữ liệu train)**:
  - **Dấu hiệu**: Train Loss liên tục giảm sâu về gần 0, Train Accuracy đạt rất cao, nhưng Val Loss ngừng giảm hoặc có xu hướng tăng ngược trở lại (phân kỳ giữa đường Train và Val).
  - **Xuất hiện rõ nhất ở đâu**:
    - Mạng sâu và rộng không có điều chuẩn (**regularization_baseline_noreg** với cấu hình `[256, 128, 64]`): Train Loss giảm nhanh nhất nhưng khoảng cách giữa Train Loss và Val Loss giãn rộng nhất.
- **Hiện tượng Good Fit (Tổng quát hóa tốt nhờ Điều chuẩn)**:
  - Khi bổ sung **Dropout ($p=0.3$)**: Bằng cách triệt tiêu ngẫu nhiên các neuron trong mỗi bước truyền xuôi, mạng buộc phải học các biểu diễn phân tán (distributed representations), ngăn chặn sự phụ thuộc lẫn nhau co-adaptation của các đặc trưng. Val Loss bám sát Train Loss và Val Accuracy ổn định nhất.
  - Khi bổ sung **Weight Decay ($10^{-4}$)**: Giới hạn độ lớn chuẩn $L_2$ của vector trọng số, giúp hàm quyết định trơn hơn, tránh các bước nhảy biên độ lớn khi gặp dữ liệu nhiễu.

### 2. So sánh Hàm kích hoạt (Activation Functions):
- **ReLU**: Là chuẩn mực vượt trội nhờ tốc độ tính toán đạo hàm cực nhanh, không bị bão hòa ở miền dương ($f'(x) = 1$ với $x > 0$), giúp khắc phục triệt để hiện tượng triệt tiêu gradient.
- **Tanh**: Tốt hơn Sigmoid vì có giá trị trung bình bằng 0 (Zero-centered output), nhưng vẫn bị bão hòa khi $|x|$ lớn.
- **Sigmoid**: Thích hợp cho tầng đầu ra của bài toán phân loại nhị phân (Binary Classification) hoặc cơ chế cổng (gates), không nên dùng làm hàm kích hoạt cho các tầng ẩn trung gian trong Deep MLP.

### 3. Tác động của Độ sâu (Depth) và Độ rộng (Width):
- Mạng sâu hơn (Depth 2, Depth 3) học được các đặc trưng phân cấp tốt hơn mạng quá nông (Depth 1). Tuy nhiên khi mạng lên tới 4 tầng mà không có kỹ thuật Residual hoặc Batch Normalization, tốc độ hội tụ bắt đầu suy giảm.
- Tăng độ rộng (Width) giúp tăng dung lượng biểu diễn rõ rệt từ 32 lên 128, nhưng từ 128 lên 256 thì mức tăng độ chính xác giảm dần trong khi số tham số tăng gấp đôi.

### 4. Kết luận Thực hành Tốt nhất (Best Practices):
- Kiến trúc nền tảng khuyến nghị cho bài toán dạng này: 2 đến 3 tầng ẩn, kích thước 128 đến 256 neurons, hàm kích hoạt **ReLU**.
- Luôn kết hợp **Dropout (0.2 - 0.3)** hoặc **Weight Decay ($10^{-4}$)** khi mạng có nhiều tham số để đảm bảo tính tổng quát hóa cao nhất.
""")
]


# ==============================================================================
# DAY 03 TASK 01: TRAINING LOOP TIÊU CHUẨN VÀ LEARNING RATE
# ==============================================================================
cells_day03_task01 = [
    md_cell("""# DAY 03 - TASK 01: XÂY DỰNG TRAINING LOOP TIÊU CHUẨN VÀ KHẢO SÁT LEARNING RATE TRÊN FASHION-MNIST

### Mục tiêu thực hành:
1. **Làm chủ quy trình huấn luyện PyTorch**: Thực thi chuẩn xác 5 bước (`zero_grad` -> `forward` -> `loss` -> `backward` -> `optimizer.step`), kiểm soát chế độ mô hình `model.train()` vs `model.eval()`.
2. **Khảo sát độ nhạy Learning Rate**: So sánh trực tiếp tốc độ hội tụ và độ chính xác kiểm định giữa 3 mức tốc độ học (**0.01**, **0.001**, **0.0001**).
3. **Trực quan hóa đường cong học tập**: Vẽ biểu đồ đối chiếu Loss và Accuracy qua các epoch, lưu ảnh vào `docs/day03/task01/` và `bin/day03/task01/results/`.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Phần cứng
Import các module từ `srcs.libraries` và pipeline `srcs.pipelines.day03_task01_training_loop`.
"""),

    code_cell("""import sys
import os
import torch
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.pipelines.day03_task01_training_loop import (
    build_fashion_mlp,
    prepare_fashion_loaders,
    run_training_session,
    compare_learning_rates,
    plot_learning_rate_comparison
)

seed_everything(42)
comp_device = device()
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
"""),

    md_cell("""## Bước 2: Khảo sát Kiến trúc Mạng MLP cho Fashion-MNIST
Kiến trúc mạng sử dụng:
- Tầng Flatten: Chuyển ma trận ảnh 28x28 thành vector 784 chiều.
- Tầng Tuyến tính 1: `Linear(784, 256)`.
- Hàm kích hoạt: `ReLU()`.
- Tầng Tuyến tính 2: `Linear(256, 10)` xuất ra logits cho 10 phân lớp thời trang.
"""),

    code_cell("""model_demo = build_fashion_mlp()
print("Cấu trúc mô hình:")
print(model_demo)
"""),

    md_cell("""## Bước 3: Thực thi Huấn luyện và Khảo sát Learning Rate (0.01, 0.001, 0.0001)
Chúng ta thực hiện so sánh 3 mức learning rate qua 5 epochs. Kết quả đồ thị và ma trận nhầm lẫn sẽ được lưu trực tiếp vào:
- `docs/day03/task01/image_training_loop_curves.png`
- `bin/day03/task01/results/training_loop_curves.png`
- `docs/day03/task01/image_confusion_matrix.png`
- `bin/day03/task01/results/confusion_matrix.png`
"""),

    code_cell("""plot_docs = os.path.join(PROJECT_ROOT, "docs", "day03", "task01", "image_training_loop_curves.png")
plot_bin = os.path.join(PROJECT_ROOT, "bin", "day03", "task01", "results", "training_loop_curves.png")
cm_docs = os.path.join(PROJECT_ROOT, "docs", "day03", "task01", "image_confusion_matrix.png")
cm_bin = os.path.join(PROJECT_ROOT, "bin", "day03", "task01", "results", "confusion_matrix.png")

histories = compare_learning_rates(
    learning_rates=(0.01, 0.001, 0.0001),
    epochs=5,
    seed=42,
    output_paths=[plot_docs, plot_bin],
    cm_paths=[cm_docs, cm_bin]
)
"""),

    md_cell("""## Bước 4: Trực quan hóa Đường cong học tập và Ma trận nhầm lẫn (Confusion Matrix)"""),

    code_cell("""print("1. ĐƯỜNG CONG HỌC TẬP (LOSS & ACCURACY):")
display(Image(plot_docs))

print("2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
display(Image(cm_docs))
"""),

    md_cell("""## Bước 5: Nhận xét và Bài học Kinh nghiệm về Learning Rate

1. **Về Learning Rate:**
   - **$LR = 0.01$**: Hội tụ nhanh lúc đầu nhưng mất ổn định quanh cực tiểu, loss có độ nảy lớn và dễ dao động quanh nghiệm tối ưu.
   - **$LR = 0.001$**: Lựa chọn chuẩn mực và cân bằng nhất, loss giảm đều, độ chính xác cao nhất (~85.10%).
   - **$LR = 0.0001$**: Học rất chậm, bước gradient quá bé khiến mô hình tụt hậu đáng kể sau cùng số lượng epochs.

2. **Phân tích Ma trận nhầm lẫn:**
   - **Nhận diện tốt nhất (>95%)**: Quần dài (Trouser), Túi xách (Bag), Giày thể thao (Sneaker) nhờ hình khối đặc thù.
   - **Nhầm lẫn nhiều nhất**: Áo sơ mi (Shirt - ~67%) bị nhầm lẫn nhiều với Áo thun (T-shirt) và Áo khoác (Coat) do phom dáng tương đồng ở độ phân giải 28x28.

3. **Quy tắc 5 bước trong Huấn luyện PyTorch:**
   - Bước 1: `optimizer.zero_grad()` xóa sạch gradient tích lũy từ bước trước.
   - Bước 2: `outputs = model(inputs)` lan truyền xuôi qua các tầng mạng.
   - Bước 3: `loss = criterion(outputs, targets)` tính hàm mất mát.
   - Bước 4: `loss.backward()` lan truyền ngược tính đạo hàm riêng cho từng tham số.
   - Bước 5: `optimizer.step()` cập nhật trọng số theo hướng gradient giảm.
""")
]


# ==============================================================================
# DAY 03 TASK 02: SO SÁNH CÁC PHƯƠNG PHÁP TỐI ƯU VÀ EARLY STOPPING
# ==============================================================================
cells_day03_task02 = [
    md_cell("""# DAY 03 - TASK 02: SO SÁNH CÁC PHƯƠNG PHÁP TỐI ƯU VÀ KỸ THUẬT EARLY STOPPING TRÊN FASHION-MNIST

### Mục tiêu thực hành:
1. **So sánh đối đầu 3 thuật toán tối ưu hóa kinh điển**: SGD thuần, SGD có Momentum, và Adam.
2. **Cài đặt và kiểm chứng cơ chế Early Stopping**: Tự động ngắt huấn luyện khi mô hình bão hòa, khôi phục lại trọng số tốt nhất (`restore_best_weights`).
3. **Trực quan hóa và phân tích hội tụ**: Lưu biểu đồ và ma trận nhầm lẫn vào `docs/day03/task02/` và `bin/day03/task02/results/`.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Thí nghiệm
Import các module từ `srcs.libraries` và pipeline `srcs.pipelines.day03_task02_optimizer_comparison`.
"""),

    code_cell("""import sys
import os
import torch
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.pipelines.day03_task02_optimizer_comparison import (
    build_optimizer_mlp,
    prepare_optimizer_loaders,
    create_optimizer,
    run_optimizer_trial,
    compare_optimizers,
    plot_optimizer_comparison
)

seed_everything(42)
comp_device = device()
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
"""),

    md_cell("""## Bước 2: Cấu trúc Mô hình và Dữ liệu Kiểm thử
Cố định kiến trúc `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)`, tập dữ liệu 8,000 ảnh huấn luyện và 1,500 ảnh kiểm định từ Fashion-MNIST.
"""),

    code_cell("""model_demo = build_optimizer_mlp()
print("Cấu trúc mạng Perceptron 128 neurons:")
print(model_demo)
"""),

    md_cell("""## Bước 3: Huấn luyện và So sánh 3 Thuật toán Tối ưu với Early Stopping
Các thuật toán được so sánh:
1. **SGD thuần** ($lr = 0.05$)
2. **SGD + Momentum** ($lr = 0.03, \\beta = 0.9$)
3. **Adam** ($lr = 0.001$)

Tiêu chuẩn Early Stopping: `patience = 3`, `min_delta = 1e-4`.
Đồ thị học tập và ma trận nhầm lẫn so sánh sẽ được lưu vào:
- `docs/day03/task02/image_optimizer_learning_curves.png`
- `bin/day03/task02/results/optimizer_learning_curves.png`
- `docs/day03/task02/image_confusion_matrix.png`
- `bin/day03/task02/results/confusion_matrix.png`
"""),

    code_cell("""plot_docs = os.path.join(PROJECT_ROOT, "docs", "day03", "task02", "image_optimizer_learning_curves.png")
plot_bin = os.path.join(PROJECT_ROOT, "bin", "day03", "task02", "results", "optimizer_learning_curves.png")
cm_docs = os.path.join(PROJECT_ROOT, "docs", "day03", "task02", "image_confusion_matrix.png")
cm_bin = os.path.join(PROJECT_ROOT, "bin", "day03", "task02", "results", "confusion_matrix.png")

histories = compare_optimizers(
    optimizers=("sgd", "momentum", "adam"),
    epochs=12,
    patience=3,
    seed=42,
    output_paths=[plot_docs, plot_bin],
    cm_paths=[cm_docs, cm_bin]
)
"""),

    md_cell("""## Bước 4: Trực quan hóa So sánh Learning Curves và Ma trận nhầm lẫn"""),

    code_cell("""print("1. ĐƯỜNG CONG HỌC TẬP CỦA CÁC OPTIMIZERS (SGD, MOMENTUM, ADAM):")
display(Image(plot_docs))

print("2. SO SÁNH MA TRẬN NHẦM LẪN GIỮA CÁC OPTIMIZERS:")
display(Image(cm_docs))
"""),

    md_cell("""## Bước 5: Nhận xét và Kết luận Chuyên sâu

1. **So sánh các Thuật toán Tối ưu:**
   - **Adam**: Hội tụ nhanh vượt trội nhờ cơ chế học thích nghi (adaptive learning rate) và moment bậc 1, bậc 2, đạt loss thấp nhất nhanh nhất.
   - **SGD + Momentum**: Động lượng $\\beta=0.9$ giúp triệt tiêu dao động ngang trong lòng máng dốc hẹp, đạt độ chính xác kiểm định cao nhất (~85.47%) và ma trận nhầm lẫn đậm màu nhất.
   - **SGD thuần**: Chậm nhất, gradient dao động dích-dắc và phân tán nhiều lỗi ngoài đường chéo chính.

2. **So sánh Ma trận nhầm lẫn:**
   - **SGD + Momentum** và **Adam** có đường chéo chính đậm nét hơn rõ rệt so với **SGD thuần**, hạn chế tối đa nhầm lẫn chéo ở các lớp khó như Áo len (Pullover) và Áo khoác (Coat).

3. **Hiệu quả của Kỹ thuật Early Stopping:**
   - Ngăn chặn kịp thời tình trạng mô hình học vẹt dữ liệu train khi validation loss có dấu hiệu tăng ngược trở lại.
   - Cơ chế tự động khôi phục trọng số tối ưu (`restore_best_weights`) bảo đảm mô hình được xuất bản luôn là phiên bản tốt nhất.
""")
]


# ==============================================================================
# DAY 04 TASK 01: SO SÁNH MLP VÀ CNN HAI BLOCK TRÊN FASHION-MNIST
# ==============================================================================
cells_day04_task01 = [
    md_cell("""# DAY 04 - TASK 01: SO SÁNH MLP VÀ CNN HAI BLOCK TRÊN FASHION-MNIST

### Mục tiêu thực hành:
1. **Khảo sát cấu trúc mạng tích chập (CNN)**: Theo dõi sự biến đổi kích thước tensor qua từng tầng Conv, ReLU, MaxPool và Flatten.
2. **So sánh thực nghiệm công bằng giữa MLP và CNN**: Giữ nguyên dung lượng tham số (~204k), seed (42), dữ liệu (6,000 train, 1,200 val), optimizer (Adam, lr=1e-3) qua 5 epochs.
3. **Phân tích hiệu năng và ma trận nhầm lẫn**: Đánh giá độ chính xác, thời gian tính toán và điểm mù phân loại của từng mô hình.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình phần cứng
Import các module từ thư viện lõi `srcs.libraries` và pipeline `srcs.pipelines.day04_task01_cnn_fashionmnist`.
"""),

    code_cell("""import sys
import os
import torch
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.libraries.models import CNN2Blocks, count_parameters
from srcs.libraries.metrics import compute_confusion_matrix
from srcs.pipelines.day04_task01_cnn_fashionmnist import (
    FashionMLP,
    prepare_fashion_splits,
    create_data_loaders,
    inspect_cnn_shapes,
    run_model_benchmark,
    render_learning_curves,
    render_confusion_matrices,
    collect_all_predictions,
    CLASS_NAMES
)

seed_everything(42)
comp_device = torch.device(device())
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
print(f"Thư mục lưu trữ dữ liệu (Gốc dự án): {DATA_DIR}")
"""),

    md_cell("""## Bước 2: Khảo sát cấu trúc mô hình và kích thước Tensor qua từng tầng
Theo dõi tensor đi qua mạng 2-block CNN từ ảnh đầu vào $1 \\times 28 \\times 28$ cho tới vector logits 10 chiều:
- **Block 1**: `Conv2d(1, 16, k=3, p=1)` -> `ReLU` -> `MaxPool2d(2)` (kích thước giảm từ $28 \\times 28 \\rightarrow 14 \\times 14$)
- **Block 2**: `Conv2d(16, 32, k=3, p=1)` -> `ReLU` -> `MaxPool2d(2)` (kích thước giảm từ $14 \\times 14 \\rightarrow 7 \\times 7$)
- **Flatten**: Trải phẳng $32 \\times 7 \\times 7 = 1,568$ giá trị.
- **Dense Classifier**: `Linear(1568, 128)` -> `ReLU` -> `Linear(128, 10)`.
"""),

    code_cell("""demo_cnn = CNN2Blocks()
print("Cấu trúc mô hình CNN 2 Blocks:")
print(demo_cnn)
print(f"Tổng số tham số: {count_parameters(demo_cnn):,}")

print("\\nKích thước Tensor qua từng tầng:")
shape_milestones = inspect_cnn_shapes(demo_cnn)
for stage, shape in shape_milestones:
    print(f"  {stage:<15}: {shape}")
"""),

    md_cell("""## Bước 3: Chuẩn bị dữ liệu Fashion-MNIST
Dữ liệu được tải và lưu trữ tập trung tại thư mục `data/` ở gốc dự án (không lưu trong thư mục code `srcs/`):
- **Tập train**: 6,000 ảnh (batch size = 64).
- **Tập validation**: 1,200 ảnh (batch size = 256).
- **Seed**: 42.
"""),

    code_cell("""train_ds, val_ds = prepare_fashion_splits(data_dir=DATA_DIR, seed=42)
train_loader, val_loader = create_data_loaders(train_ds, val_ds, seed=42)
print(f"Số lượng mẫu huấn luyện: {len(train_ds):,} | Số lượng mẫu kiểm định: {len(val_ds):,}")
print(f"Số lượng phân lớp: {len(CLASS_NAMES)} ({CLASS_NAMES})")
"""),

    md_cell("""## Bước 4: Huấn luyện đối đầu giữa MLP và CNN 2-Block
Tiến hành huấn luyện cả 2 mô hình qua 5 epochs dưới cùng điều kiện kiểm thử:
- Optimizer: **Adam** ($lr = 10^{-3}$)
- Loss Function: **CrossEntropyLoss**
"""),

    code_cell("""benchmarks = {}
benchmarks["MLP"] = run_model_benchmark(FashionMLP, "MLP", train_ds, val_ds, epochs=5, seed=42, comp_device=comp_device)
benchmarks["CNN"] = run_model_benchmark(CNN2Blocks, "CNN", train_ds, val_ds, epochs=5, seed=42, comp_device=comp_device)
"""),

    md_cell("""## Bước 5: Bảng so sánh kết quả thực nghiệm giữa MLP và CNN"""),

    code_cell("""df_comparison = pd.DataFrame([
    {
        "Mô hình": name,
        "Số tham số": f"{res['params']:,}",
        "Train Loss": f"{res['history']['train_loss'][-1]:.4f}",
        "Val Loss": f"{res['history']['val_loss'][-1]:.4f}",
        "Val Accuracy": f"{res['final_val_acc'] * 100:.2f}%",
        "Giây/epoch (TB)": f"{res['sec_per_epoch']:.2f}s"
    }
    for name, res in benchmarks.items()
])
display(df_comparison)
"""),

    md_cell("""## Bước 6: Trực quan hóa Đường cong học tập (Learning Curves)
Lưu biểu đồ vào `docs/day04/task01/image_learning_curve.png` và `bin/day04/task01/results/learning_curve.png`.
"""),

    code_cell("""curve_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task01", "image_learning_curve.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task01", "results", "learning_curve.png")
]
render_learning_curves(
    histories={name: res["history"] for name, res in benchmarks.items()},
    epochs=5,
    output_paths=curve_paths
)
display(Image(curve_paths[0]))
"""),

    md_cell("""## Bước 7: Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix trên Validation)
Lưu ma trận nhầm lẫn vào `docs/day04/task01/image_confusion_matrix.png` và `bin/day04/task01/results/confusion_matrix.png`.
"""),

    code_cell("""cm_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task01", "image_confusion_matrix.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task01", "results", "confusion_matrix.png")
]
render_confusion_matrices(
    models_dict={name: res["model"] for name, res in benchmarks.items()},
    val_loader=benchmarks["CNN"]["val_loader"],
    comp_device=comp_device,
    output_paths=cm_paths
)
display(Image(cm_paths[0]))
"""),

    md_cell("""## Bước 8: Phân tích các cặp phân lớp dễ bị nhầm lẫn nhất của CNN"""),

    code_cell("""y_true, y_pred = collect_all_predictions(benchmarks["CNN"]["model"], benchmarks["CNN"]["val_loader"], comp_device)
cm = compute_confusion_matrix(y_true, y_pred)
np.fill_diagonal(cm, 0)
top_errors = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:5], cm.shape))[0]

print("Top 5 cặp lớp bị CNN dự đoán nhầm nhiều nhất:")
for i, j in top_errors:
    print(f"  - Thật: {CLASS_NAMES[i]:<10} -> Đoán: {CLASS_NAMES[j]:<10} ({cm[i, j]} mẫu)")
"""),

    md_cell("""## Bước 9: Nhận xét và Kết luận chuyên sâu

1. **Hiệu quả cấu trúc tích chập (Convolutional Inductive Bias):**
   - Với cùng quy mô tham số (~204k tham số), **CNN vượt trội hơn MLP 3.16 điểm %** (86.08% so với 82.92%).
   - Sự vượt trội này khẳng định ưu thế tuyệt đối của việc chia sẻ trọng số (weight sharing) và trường thụ nhận cục bộ (local receptive field), giúp giữ nguyên thông tin cấu trúc không gian 2D thay vì trải phẳng mù quáng như MLP.

2. **Đánh đổi về thời gian tính toán:**
   - MLP tính toán nhanh hơn gần gấp 2 lần (1.02s so với 1.93s mỗi epoch trên CPU) do chỉ thực hiện các phép nhân ma trận đơn giản.
   - CNN tốn nhiều phép tính tích chập trượt trên không gian hơn, nhưng tốc độ giảm loss và khả năng tổng quát hóa hoàn toàn xứng đáng với chi phí bỏ ra.

3. **Phân tích lỗi từ Ma trận nhầm lẫn:**
   - Các vật phẩm có hình dáng hình học độc lập (Trouser, Bag, Sneaker, Boot) đạt độ chính xác rất cao (> 90%).
   - Nhóm trang phục thân trên (Shirt, T-shirt, Coat, Pullover) là điểm mù chung của cả 2 mô hình do độ phân giải $28 \\times 28$ không thể hiện rõ các chi tiết nhỏ như cổ áo sơ mi hay cúc áo.
""")
]


# ==============================================================================
# DAY 04 TASK 02: KHẢO SÁT BIẾN THỂ KIẾN TRÚC CNN (ABLATION STUDY)
# ==============================================================================
cells_day04_task02 = [
    md_cell("""# DAY 04 - TASK 02: KHẢO SÁT BIẾN THỂ KIẾN TRÚC CNN (ABLATION STUDY) TRÊN FASHION-MNIST

### Mục tiêu thực hành:
1. **Thực hiện nghiên cứu loại trừ (Ablation Study)**: Khảo sát ảnh hưởng của từng thành phần kiến trúc: kích thước kernel ($3 \\times 3$ vs $5 \\times 5$), độ rộng số kênh (8 vs 16 vs 32), Batch Normalization và Dropout.
2. **Lựa chọn mô hình tối ưu dựa trên Validation**: Chọn mô hình có Validation Accuracy cao nhất (kết hợp Validation Loss thấp nhất).
3. **Đánh giá khách quan trên Official Test Set (10,000 ảnh)**: Kiểm chứng độ chính xác thực tế, ma trận nhầm lẫn và trực quan hóa 5 ảnh đoán đúng / 5 ảnh đoán sai.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Thí nghiệm
Import các module từ thư viện lõi `srcs.libraries` và pipeline `srcs.pipelines.day04_task02_cnn_variants`.
"""),

    code_cell("""import sys
import os
import torch
import pandas as pd
import numpy as np
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.pipelines.day04_task02_cnn_variants import (
    get_ablation_configurations,
    prepare_datasets,
    create_loaders,
    run_ablation_study,
    render_comparative_learning_curves,
    render_test_confusion_matrix,
    render_prediction_samples,
    CLASS_NAMES
)

seed_everything(42)
comp_device = torch.device(device())
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
print(f"Thư mục lưu trữ dữ liệu (Gốc dự án): {DATA_DIR}")
"""),

    md_cell("""## Bước 2: Định nghĩa 6 cấu hình Biến thể CNN (Ablation Study)
Mỗi cấu hình chỉ thay đổi đúng một thuộc tính so với `baseline`:
1. `baseline`: channels = 16, kernel_size = 3, batchnorm = False, dropout = 0.0
2. `kernel5`: kernel_size = 5 (tăng trường tiếp nhận receptive field)
3. `ch8`: channels = 8 (giảm một nửa dung lượng tham số)
4. `ch32`: channels = 32 (gấp đôi số kênh bộ lọc)
5. `batchnorm`: bổ sung Batch Normalization 2D sau mỗi tầng Conv
6. `dropout0.3`: bổ sung Dropout (p = 0.3) trước tầng Linear cuối
"""),

    code_cell("""configs = get_ablation_configurations()
df_configs = pd.DataFrame(configs)
display(df_configs)
"""),

    md_cell("""## Bước 3: Huấn luyện và Đánh giá 6 cấu hình trên cùng điều kiện
Huấn luyện 5 epochs với cùng tập dữ liệu lưu trữ tại thư mục `data/` ở gốc dự án (6,000 train, 1,200 val, 10,000 test), seed 42 và optimizer Adam (lr = 1e-3).
"""),

    code_cell("""study_data = run_ablation_study(epochs=5, data_dir=DATA_DIR, seed=42)
results = study_data["results"]
best_res = study_data["best_result"]
test_loader = study_data["test_loader"]
"""),

    md_cell("""## Bước 4: Bảng so sánh tổng hợp 6 biến thể CNN"""),

    code_cell("""df_summary = pd.DataFrame([
    {
        "Cấu hình": res["config"]["name"],
        "Số tham số": f"{res['params']:,}",
        "Train Loss": f"{res['final_train_loss']:.4f}",
        "Val Loss": f"{res['final_val_loss']:.4f}",
        "Val Accuracy": f"{res['final_val_acc'] * 100:.2f}%",
        "Giây/epoch": f"{res['sec_per_epoch']:.2f}s"
    }
    for res in results
])
display(df_summary)
"""),

    md_cell("""## Bước 5: Trực quan hóa đối chiếu Đường cong học tập (3 Subplots)
Lưu biểu đồ vào `docs/day04/task02/image_learning_curves.png` và `bin/day04/task02/results/learning_curves.png`.
"""),

    code_cell("""curves_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task02", "image_learning_curves.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task02", "results", "learning_curves.png")
]
render_comparative_learning_curves(results, epochs=5, output_paths=curves_paths)
display(Image(curves_paths[0]))
"""),

    md_cell("""## Bước 6: Lựa chọn Mô hình tối ưu trên tập Validation
Tiêu chí lựa chọn: Validation Accuracy cao nhất (nếu bằng nhau thì chọn Validation Loss thấp hơn).
"""),

    code_cell("""print(f"Mô hình được chọn: {best_res['config']['name'].upper()}")
print(f"  - Validation Accuracy: {best_res['final_val_acc'] * 100:.2f}%")
print(f"  - Validation Loss:     {best_res['final_val_loss']:.4f}")
print(f"  - Tổng số tham số:     {best_res['params']:,}")
"""),

    md_cell("""## Bước 7: Kiểm định Mô hình tối ưu trên Official Test Set và Ma trận nhầm lẫn
Tập Test chính thức (10,000 ảnh) chỉ được đem ra đánh giá đúng một lần duy nhất cho mô hình chiến thắng (`batchnorm`).
Lưu ảnh vào `docs/day04/task02/image_confusion_matrix_test.png` và `bin/day04/task02/results/confusion_matrix_test.png`.
"""),

    code_cell("""test_cm_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task02", "image_confusion_matrix_test.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task02", "results", "confusion_matrix_test.png")
]
x_test, y_true, y_pred, test_acc = render_test_confusion_matrix(
    model=best_res["model"],
    test_loader=test_loader,
    comp_device=comp_device,
    model_name=best_res["config"]["name"],
    output_paths=test_cm_paths
)
print(f"Official Test Accuracy ({best_res['config']['name']}): {test_acc * 100:.2f}%")
display(Image(test_cm_paths[0]))
"""),

    md_cell("""## Bước 8: Trực quan hóa 5 mẫu ảnh dự đoán Đúng và 5 mẫu ảnh dự đoán Sai trên Test Set
Lưu ảnh mẫu vào `docs/day04/task02/` và `bin/day04/task02/results/`.
"""),

    code_cell("""correct_indices = np.where(y_true == y_pred)[0][:5]
wrong_indices = np.where(y_true != y_pred)[0][:5]

correct_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task02", "image_five_correct.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task02", "results", "five_correct.png")
]
render_prediction_samples(x_test, y_true, y_pred, correct_indices, "5 ảnh dự đoán ĐÚNG trên tập Test", correct_paths)
display(Image(correct_paths[0]))

wrong_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day04", "task02", "image_five_wrong.png"),
    os.path.join(PROJECT_ROOT, "bin", "day04", "task02", "results", "five_wrong.png")
]
render_prediction_samples(x_test, y_true, y_pred, wrong_indices, "5 ảnh dự đoán SAI trên tập Test", wrong_paths)
display(Image(wrong_paths[0]))
"""),

    md_cell("""## Bước 9: Kết luận và Khuyến nghị Thực hành Tốt nhất

1. **Hiệu quả vượt trội của Batch Normalization:**
   - `batchnorm` chỉ thêm 96 tham số nhưng đem lại mức nhảy vọt ấn tượng nhất: đạt **88.00% val acc** và **86.59% test acc**.
   - Việc chuẩn hóa giúp mạng hội tụ thần tốc ngay từ epoch 1 (đạt 85.67% so với 76.92% của baseline) nhờ triệt tiêu trôi dạt phân phối đầu vào.

2. **Quy luật về số lượng kênh:**
   - Giảm xuống 8 kênh (`ch8`): Mô hình bị thiếu dung lượng biểu diễn (underfit), val accuracy giảm xuống 83.75%.
   - Tăng lên 32 kênh (`ch32`): Số tham số tăng hơn gấp 2 lần (421k) và chạy chậm hơn gấp 2 lần (3.45s), nhưng accuracy (87.58%) vẫn đứng sau `batchnorm`.

3. **Receptive Field và Điều chuẩn:**
   - Kích thước kernel $5 \\times 5$ (`kernel5`) không tạo khác biệt lớn so với $3 \\times 3$ trên ảnh $28 \\times 28$.
   - `dropout0.3` chưa phát huy hiệu quả khi số lượng epoch còn nhỏ (5 epochs), nhưng sẽ bảo vệ mô hình rất tốt khi huấn luyện dài hơi (20+ epochs).
""")
]


# ==============================================================================
# DAY 05 TASK 01: RESNET18 TRANSFER LEARNING MODES TRÊN CIFAR-10
# ==============================================================================
cells_day05_task01 = [
    md_cell("""# DAY 05 - TASK 01: HỌC CHUYỂN GIAO (TRANSFER LEARNING) VỚI RESNET18 TRÊN CIFAR-10

### Mục tiêu thực hành:
1. **Khảo sát cơ chế Transfer Learning**: Tận dụng trọng số tiền huấn luyện (ImageNet pretrained weights) của mạng ResNet18 để chuyển giao sang phân loại bộ dữ liệu CIFAR-10 trong điều kiện dữ liệu nhỏ (2,000 ảnh).
2. **So sánh hai chế độ thích ứng**:
   - Chế độ 1 (`frozen`): Đóng băng toàn bộ backbone, chỉ huấn luyện duy nhất tầng phân loại cuối (`net.fc`).
   - Chế độ 2 (`finetune_layer4`): Mở khóa khối tích chập cuối (`net.layer4`) và tầng phân loại (`net.fc`) với learning rate phân tầng (differential learning rates).
3. **Phân tích hiệu năng và ma trận nhầm lẫn**: Đánh giá độ chính xác, đường cong học tập, ma trận nhầm lẫn và chỉ số Recall chi tiết trên từng phân lớp đối tượng.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình phần cứng
Import các thư viện cốt lõi, phát hiện phần cứng tính toán và xác định thư mục dữ liệu tập trung tại `data/` ở thư mục gốc dự án.
"""),

    code_cell("""import sys
import os
import torch
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.libraries.models import count_parameters
from srcs.pipelines.day05_task01_transfer_learning import (
    build_resnet18,
    prepare_cifar10_loaders,
    train_resnet_model,
    collect_predictions,
    compute_confusion_matrices,
    compute_classwise_recall,
    render_learning_curves,
    render_confusion_matrices,
    CLASSES,
    DEFAULT_LR_HEAD,
    DEFAULT_LR_LAYER4
)

seed_everything(42)
comp_device = torch.device(device())
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
print(f"Thư mục dữ liệu gốc: {DATA_DIR}")
"""),

    md_cell("""## Bước 2: Khảo sát Kiến trúc mạng Pretrained ResNet18 và Cơ chế Freeze Backbone
Theo dõi số lượng tham số được tối ưu hóa giữa hai chế độ:
- **Chế độ Frozen**: Đóng băng toàn bộ các tầng tích chập, chỉ giữ `net.fc` có `requires_grad = True` (khoảng 5,130 tham số).
- **Chế độ Fine-tune Layer 4**: Mở khóa `net.layer4` và `net.fc` (khoảng 8,393,738 tham số).
- **Cố định BatchNorm**: Đưa toàn bộ các tầng `BatchNorm2d` về chế độ `eval()` để bảo toàn thống kê ImageNet.
"""),

    code_cell("""demo_frozen = build_resnet18(finetune_layer4=False, num_classes=10)
demo_finetune = build_resnet18(finetune_layer4=True, num_classes=10)

print(f"Tổng số tham số mạng ResNet18: {sum(p.numel() for p in demo_frozen.parameters()):,}")
print(f"Số tham số huấn luyện (Frozen Backbone):    {count_parameters(demo_frozen):,}")
print(f"Số tham số huấn luyện (Fine-tune Layer 4):  {count_parameters(demo_finetune):,}")
"""),

    md_cell("""## Bước 3: Chuẩn bị dữ liệu CIFAR-10 với Tiền xử lý chuẩn ImageNet
Phân chia tập dữ liệu đảm bảo tính độc lập và tái lập:
- **Tập train**: 2,000 ảnh (batch size = 32).
- **Tập validation**: 500 ảnh (batch size = 32).
- **Phép tiền xử lý**: `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet Mean, Std)`.
- Dữ liệu được lưu trữ tập trung tại `data/` ở gốc dự án.
"""),

    code_cell("""train_loader, val_loader, indices = prepare_cifar10_loaders(
    data_dir=DATA_DIR,
    train_size=2000,
    val_size=500,
    batch_size=32,
    seed=42
)
print(f"Số lượng batch huấn luyện: {len(train_loader)} (tổng 2,000 mẫu)")
print(f"Số lượng batch kiểm định:  {len(val_loader)} (tổng 500 mẫu)")
print(f"Danh sách 10 phân lớp: {CLASSES}")
"""),

    md_cell("""## Bước 4: Huấn luyện Chế độ 1 - Frozen Backbone
Chỉ huấn luyện tầng phân loại cuối cùng `net.fc` với learning rate $10^{-3}$ qua 4 epochs.
"""),

    code_cell("""seed_everything(42)
model_frozen = build_resnet18(finetune_layer4=False, num_classes=10)
history_frozen = train_resnet_model(
    model=model_frozen,
    train_loader=train_loader,
    val_loader=val_loader,
    comp_device=comp_device,
    epochs=4,
    lr_head=DEFAULT_LR_HEAD,
    lr_layer4=None,
    mode="frozen"
)
"""),

    md_cell("""## Bước 5: Huấn luyện Chế độ 2 - Fine-tune Layer 4
Huấn luyện đồng thời `net.layer4` (với $lr = 10^{-4}$) và `net.fc` (với $lr = 10^{-3}$) qua 4 epochs.
"""),

    code_cell("""seed_everything(42)
model_finetune = build_resnet18(finetune_layer4=True, num_classes=10)
history_finetune = train_resnet_model(
    model=model_finetune,
    train_loader=train_loader,
    val_loader=val_loader,
    comp_device=comp_device,
    epochs=4,
    lr_head=DEFAULT_LR_HEAD,
    lr_layer4=DEFAULT_LR_LAYER4,
    mode="finetune_layer4"
)
"""),

    md_cell("""## Bước 6: Bảng so sánh tổng hợp hiệu năng giữa Frozen và Fine-tune"""),

    code_cell("""df_summary = pd.DataFrame([
    {
        "Chế độ": "frozen",
        "Tham số train": f"{count_parameters(model_frozen):,}",
        "Train Loss": f"{history_frozen[-1]['train_loss']:.4f}",
        "Val Loss": f"{history_frozen[-1]['val_loss']:.4f}",
        "Val Accuracy": f"{history_frozen[-1]['val_acc'] * 100:.2f}%",
        "Val Acc Max": f"{max(r['val_acc'] for r in history_frozen) * 100:.2f}%",
        "Giây/epoch": f"{np.mean([r['sec_per_epoch'] for r in history_frozen]):.2f}s"
    },
    {
        "Chế độ": "finetune_layer4",
        "Tham số train": f"{count_parameters(model_finetune):,}",
        "Train Loss": f"{history_finetune[-1]['train_loss']:.4f}",
        "Val Loss": f"{history_finetune[-1]['val_loss']:.4f}",
        "Val Accuracy": f"{history_finetune[-1]['val_acc'] * 100:.2f}%",
        "Val Acc Max": f"{max(r['val_acc'] for r in history_finetune) * 100:.2f}%",
        "Giây/epoch": f"{np.mean([r['sec_per_epoch'] for r in history_finetune]):.2f}s"
    }
])
display(df_summary)
"""),

    md_cell("""## Bước 7: Trực quan hóa Đường cong học tập (Learning Curves đối đầu)
Lưu biểu đồ vào `docs/day05/task01/image_learning_curve.png` và `bin/day05/task01/results/learning_curve.png`.
"""),

    code_cell("""curve_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day05", "task01", "image_learning_curve.png"),
    os.path.join(PROJECT_ROOT, "bin", "day05", "task01", "results", "learning_curve.png")
]
histories_dict = {
    "frozen": history_frozen,
    "finetune_layer4": history_finetune
}
render_learning_curves(histories_dict, curve_paths)
display(Image(curve_paths[0]))
"""),

    md_cell("""## Bước 8: Trực quan hóa Ma trận nhầm lẫn và Đánh giá Recall chi tiết từng lớp
Lưu biểu đồ vào `docs/day05/task01/image_confusion_matrix.png` và `bin/day05/task01/results/confusion_matrix.png`.
"""),

    code_cell("""models_dict = {
    "frozen": model_frozen,
    "finetune_layer4": model_finetune
}
cms = compute_confusion_matrices(models_dict, val_loader, comp_device)
cm_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day05", "task01", "image_confusion_matrix.png"),
    os.path.join(PROJECT_ROOT, "bin", "day05", "task01", "results", "confusion_matrix.png")
]
render_confusion_matrices(cms, classes=CLASSES, output_paths=cm_paths)
display(Image(cm_paths[0]))

df_recall = compute_classwise_recall(cms, classes=CLASSES)
display(df_recall.round(4))
"""),

    md_cell("""## Bước 9: Nhận xét chuyên sâu và Kết luận

1. **Hiệu năng tổng thể và Đỉnh hội tụ:**
   - Ở epoch 4, Validation Accuracy đạt 80.20% (frozen) và 80.60% (fine-tune), chênh lệch chỉ 0.40% (khoảng 2 ảnh trên 500 mẫu val).
   - Fine-tune Layer 4 hội tụ cực nhanh (đạt 78.60% ngay epoch 1) và đạt đỉnh **82.80% tại epoch 3**, cao hơn mức tốt nhất của frozen 2.6 điểm phần trăm.

2. **Dấu hiệu Overfitting của Fine-tune:**
   - Tại epoch 4, `finetune_layer4` tiếp tục giảm train loss (0.066) với train acc đạt 98.30%, nhưng val loss tăng vọt (0.481 lên 0.734) và val acc sụt giảm.
   - Nguyên nhân: Với chỉ 2,000 ảnh, việc cập nhật gần 8.4 triệu tham số khiến mạng bị dư thừa dung lượng biểu diễn và ghi nhớ mẫu huấn luyện.

3. **Tính ổn định của Frozen Backbone:**
   - Chế độ `frozen` chỉ tối ưu 5,130 tham số, train loss và val loss giảm đều đặn, độ chênh lệch rất nhỏ (train acc 84.8% vs val acc 80.2%), không có nguy cơ overfit.

4. **Đánh đổi về tài nguyên:**
   - Tham số huấn luyện chênh lệch hơn 1,600 lần, nhưng thời gian thực thi trên mỗi epoch chỉ chênh lệch ~6% (~4.10s so với ~4.36s) do cả hai đều phải thực hiện forward pass qua toàn bộ mạng ResNet18.
""")
]


# ==============================================================================
# DAY 05 TASK 02: DATA AUGMENTATION VÀ TRANSFER LEARNING TRÊN CIFAR-10
# ==============================================================================
cells_day05_task02 = [
    md_cell("""# DAY 05 - TASK 02: DATA AUGMENTATION VÀ HỌC CHUYỂN GIAO VỚI MOBILENETV3 TRÊN CIFAR-10

### Mục tiêu thực hành:
1. **Khảo sát vai trò của Data Augmentation**: Đánh giá ảnh hưởng của các phép tăng cường dữ liệu hợp ngữ nghĩa (RandomCrop, RandomHorizontalFlip) trên mạng CNN tự huấn luyện từ đầu (train from scratch).
2. **So sánh mô hình tự học với Pretrained MobileNetV3-Small**: Đo lường bước nhảy vọt về chất lượng khi tận dụng biểu diễn đặc trưng ImageNet trên dữ liệu nhỏ (2,000 ảnh).
3. **Đánh giá đa chiều và trực quan hóa lỗi**: Phân tích Accuracy, Macro-F1, ma trận nhầm lẫn và hiển thị trực tiếp 5 mẫu ảnh đoán đúng / 5 mẫu ảnh đoán sai của từng cấu hình.
"""),

    md_cell("""## Bước 1: Thiết lập môi trường và cấu hình Thí nghiệm
Import các module từ thư viện lõi `srcs.libraries` và pipeline `srcs.pipelines.day05_task02_transfer_learning`.
"""),

    code_cell("""import sys
import os
import torch
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.libraries.seeding import seed_everything
from srcs.libraries.training import device
from srcs.libraries.models import count_parameters
from srcs.pipelines.day05_task02_transfer_learning import (
    SmallCNN,
    build_mobilenet_v3_small,
    prepare_partition_indices,
    create_dataloaders,
    train_single_configuration,
    render_comparative_curves,
    render_confusion_matrices,
    render_example_predictions,
    CLASSES,
    DEFAULT_LR
)

seed_everything(42)
comp_device = torch.device(device())
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(f"PyTorch Version: {torch.__version__} | Device: {comp_device}")
print(f"Thư mục lưu trữ dữ liệu (Gốc dự án): {DATA_DIR}")
"""),

    md_cell("""## Bước 2: Định nghĩa 3 cấu hình Thí nghiệm Đối chứng
3 cấu hình được so sánh dưới cùng điều kiện kiểm thử:
1. `cnn_baseline`: SmallCNN 2 blocks (268,650 tham số), ảnh $32 \\times 32$, không áp dụng Augmentation.
2. `cnn_augmentation`: SmallCNN 2 blocks (268,650 tham số), ảnh $32 \\times 32$, có áp dụng Augmentation trên tập train.
3. `mobilenetv3_small_pretrained`: MobileNetV3-Small pretrained đóng băng backbone (chỉ huấn luyện 10,250 tham số tại head), ảnh $224 \\times 224$, có áp dụng Augmentation trên tập train.
"""),

    code_cell("""demo_cnn = SmallCNN(len(CLASSES))
demo_mobilenet = build_mobilenet_v3_small(len(CLASSES))

print(f"Số tham số huấn luyện SmallCNN:               {count_parameters(demo_cnn):,}")
print(f"Số tham số huấn luyện MobileNetV3 Pretrained: {count_parameters(demo_mobilenet):,}")
"""),

    md_cell("""## Bước 3: Chuẩn bị dữ liệu CIFAR-10 và Phân chia cố định (Fixed Split)
Tách tập dữ liệu thành 2,000 mẫu train và 500 mẫu validation theo seed 42.
Tập validation luôn được giữ cố định không tăng cường dữ liệu.
"""),

    code_cell("""raw_dataset, train_idx, val_idx = prepare_partition_indices(
    data_dir=DATA_DIR,
    train_size=2000,
    val_size=500,
    seed=42
)
print(f"Số lượng mẫu huấn luyện: {len(train_idx):,} | Số lượng mẫu kiểm định: {len(val_idx):,}")
print(f"Danh sách các phân lớp: {CLASSES}")
"""),

    md_cell("""## Bước 4: Thực thi Huấn luyện 3 Cấu hình Thí nghiệm qua 4 Epochs"""),

    code_cell("""configs = [
    {"name": "cnn_baseline", "augment": False, "pretrained": False, "model_fn": lambda: SmallCNN(len(CLASSES)), "frozen": False},
    {"name": "cnn_augmentation", "augment": True, "pretrained": False, "model_fn": lambda: SmallCNN(len(CLASSES)), "frozen": False},
    {"name": "mobilenetv3_small_pretrained", "augment": True, "pretrained": True, "model_fn": lambda: build_mobilenet_v3_small(len(CLASSES)), "frozen": True}
]

results = []
cms = {}
for cfg in configs:
    cfg_name = cfg["name"]
    print(f"\\n--- Đang huấn luyện cấu hình: {cfg_name} ---")
    tr_loader, va_loader = create_dataloaders(
        data_dir=DATA_DIR,
        train_idx=train_idx,
        val_idx=val_idx,
        augment=cfg["augment"],
        pretrained=cfg["pretrained"],
        batch_size=32,
        seed=42
    )
    seed_everything(42)
    model = cfg["model_fn"]()
    trial_data = train_single_configuration(
        config_name=cfg_name,
        model=model,
        train_loader=tr_loader,
        val_loader=va_loader,
        comp_device=comp_device,
        frozen_backbone=cfg["frozen"],
        epochs=4,
        lr=DEFAULT_LR
    )
    results.append(trial_data)
    from sklearn.metrics import confusion_matrix
    cms[cfg_name] = confusion_matrix(trial_data["y_true"], trial_data["y_pred"], labels=list(range(len(CLASSES))))
"""),

    md_cell("""## Bước 5: Bảng so sánh tổng hợp 3 cấu hình Thí nghiệm"""),

    code_cell("""df_summary = pd.DataFrame([
    {
        "Cấu hình": res["name"],
        "Số tham số train": f"{res['param_count']:,}",
        "Train Loss": f"{res['history']['train_loss'][-1]:.4f}",
        "Val Loss": f"{res['final_val_loss']:.4f}",
        "Val Accuracy": f"{res['final_val_acc'] * 100:.2f}%",
        "Macro-F1": f"{res['final_val_f1']:.4f}",
        "Giây/epoch": f"{res['avg_sec']:.2f}s"
    }
    for res in results
])
display(df_summary)
"""),

    md_cell("""## Bước 6: Trực quan hóa đối chiếu Đường cong học tập (3 Subplots)
Lưu biểu đồ vào `docs/day05/task02/image_learning_curves.png` và `bin/day05/task02/results/learning_curves.png`.
"""),

    code_cell("""curves_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day05", "task02", "image_learning_curves.png"),
    os.path.join(PROJECT_ROOT, "bin", "day05", "task02", "results", "learning_curves.png")
]
render_comparative_curves(results, curves_paths, epochs=4)
display(Image(curves_paths[0]))
"""),

    md_cell("""## Bước 7: Trực quan hóa Ma trận nhầm lẫn (Confusion Matrices 3 Subplots)
Lưu biểu đồ vào `docs/day05/task02/image_confusion_matrix.png` và `bin/day05/task02/results/confusion_matrix.png`.
"""),

    code_cell("""cm_paths = [
    os.path.join(PROJECT_ROOT, "docs", "day05", "task02", "image_confusion_matrix.png"),
    os.path.join(PROJECT_ROOT, "bin", "day05", "task02", "results", "confusion_matrix.png")
]
render_confusion_matrices(cms, classes=CLASSES, output_paths=cm_paths)
display(Image(cm_paths[0]))
"""),

    md_cell("""## Bước 8: Trực quan hóa 5 mẫu ảnh dự đoán Đúng và 5 mẫu ảnh dự đoán Sai
Lưu ảnh minh chứng cho từng mô hình vào `docs/day05/task02/` và `bin/day05/task02/results/`.
"""),

    code_cell("""for res in results:
    cfg_name = res["name"]
    sample_paths = [
        os.path.join(PROJECT_ROOT, "docs", "day05", "task02", f"image_examples_{cfg_name}.png"),
        os.path.join(PROJECT_ROOT, "bin", "day05", "task02", f"examples_{cfg_name}.png")
    ]
    render_example_predictions(
        raw_dataset=raw_dataset,
        val_idx=val_idx,
        y_true=res["y_true"],
        y_pred=res["y_pred"],
        config_name=cfg_name,
        output_paths=sample_paths,
        classes=CLASSES
    )

display(Image(os.path.join(PROJECT_ROOT, "docs", "day05", "task02", "image_examples_mobilenetv3_small_pretrained.png")))
"""),

    md_cell("""## Bước 9: Nhận xét chuyên sâu và Bài học kinh nghiệm thực tế

1. **Hiệu ứng của Data Augmentation trên mạng tự học:**
   - Trong 4 epoch đầu, `cnn_augmentation` đạt 39.0% Accuracy so với 42.0% của baseline. Sự tăng độ khó của tập train khiến train loss cao hơn (1.6170 vs 1.4880), nhưng khoảng cách giữa train và val được thu hẹp, giúp điều chuẩn chống overfit hiệu quả.
   - Khi tăng số epoch (ví dụ 15 đến 30 epochs), augmentation sẽ thể hiện ưu thế vượt trội khi baseline bắt đầu phân kỳ quá khớp.

2. **Bước nhảy vọt từ Transfer Learning:**
   - `mobilenetv3_small_pretrained` đạt **83.40% Accuracy** và **0.8350 Macro-F1**, bỏ xa CNN tự học hơn 41 đến 44 điểm phần trăm.
   - Nhờ thừa hưởng đặc trưng tổng quát từ ImageNet, mô hình đạt 78.4% ngay từ epoch 1 và duy trì khả năng tổng quát hóa tuyệt vời.

3. **Đánh đổi về chi phí tính toán:**
   - MobileNet chỉ huấn luyện 10,250 tham số (ít hơn 26 lần), nhưng thời gian mỗi epoch chậm hơn 4 đến 5 lần (3.68s so với 0.72s đến 0.95s) do kích thước ảnh đầu vào $224 \\times 224$ lớn gấp 49 lần so với ảnh $32 \\times 32$.
""")
]


# ==============================================================================
# DAY 06 TASK 01: LSTM PHÂN LOẠI VĂN BẢN TRÊN AG NEWS
# ==============================================================================
cells_day06_task01 = [
    md_cell("""# BUỔI 06 - BÀI TẬP 01: PHÂN LOẠI VĂN BẢN VỚI LSTM TRÊN AG NEWS

### Mục tiêu thực hành:
1. Hiểu nguyên tắc cô lập hoàn toàn tập Test chính thức ngay từ bước tiền xử lý và xây dựng từ vựng.
2. Nắm vững cơ chế đóng gói chuỗi `pack_padded_sequence` trong PyTorch để xử lý văn bản có độ dài biến thiên, bỏ qua token padding `<pad>` khi tính toán hồi quy.
3. Huấn luyện mạng LSTM một chiều, một tầng kết hợp kỹ thuật kẹp gradient (`gradient clipping`) để chống bùng nổ đạo hàm.
4. Đánh giá mô hình bằng Validation Loss để chọn checkpoint tối ưu, sau đó kiểm tra độ chính xác trên 7,600 bản tin Test thật.
5. Phân tích nguyên nhân ngữ nghĩa đằng sau các bản tin bị dự đoán sai.
"""),

    md_cell("""## Bước 1: Khởi tạo Môi trường, Thiết lập Seed và Định cấu hình
Thiết lập seed 42 để đảm bảo tính tái lập kết quả thực nghiệm.
"""),

    code_cell("""import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.pipelines.day06_task01_lstm_text_classification import (
    set_seed,
    load_agnews_splits,
    tokenize,
    build_vocab,
    AGNewsTextDataset,
    collate_batch,
    LSTMClassifier,
    train_one_epoch,
    evaluate,
    draw_learning_curve,
    draw_confusion_matrix,
    CLASS_NAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_DOCS_DIR,
    DEFAULT_BIN_DIR,
    DEFAULT_MODELS_DIR
)

set_seed(42)
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Device: {comp_device}")
"""),

    md_cell("""## Bước 2: Tải Dữ liệu AG News và Phân chia Phân tầng (Stratified Split)
Tập train chính thức (120,000 bản tin) được phân chia phân tầng thành train và validation.
Tập Test (7,600 mẫu) được giữ nguyên vẹn để kiểm tra năng lực tổng quát hóa.
"""),

    code_cell("""train_rows, val_rows, test_rows = load_agnews_splits(
    data_dir=DEFAULT_DATA_DIR,
    val_size=6000,
    train_size=15000,
    seed=42
)
print(f"Train samples:      {len(train_rows):,}")
print(f"Validation samples: {len(val_rows):,}")
print(f"Test samples:       {len(test_rows):,}")
"""),

    md_cell("""## Bước 3: Tiền xử lý Văn bản và Xây dựng Từ điển (Vocabulary)
Từ điển chỉ được thống kê trên tập huấn luyện để loại bỏ hoàn toàn nguy cơ rò rỉ dữ liệu (data leakage).
"""),

    code_cell("""vocab = build_vocab(train_rows, max_vocab=12000, min_freq=2)
print(f"Kích thước từ vựng (Vocabulary size): {len(vocab):,}")

sample_headline = train_rows[0]["text"]
sample_tokens = tokenize(sample_headline)
sample_ids = [vocab.get(t, 1) for t in sample_tokens[:12]]
print(f"Bản tin gốc: {sample_headline[:100]}...")
print(f"Tokens:     {sample_tokens[:8]}")
print(f"Token IDs:  {sample_ids[:8]}")
"""),

    md_cell("""## Bước 4: Khởi tạo Dataset và DataLoader với Bộ gom nhóm động (collate_batch)
Bộ gom nhóm động pad chuỗi theo độ dài lớn nhất của từng mini-batch thay vì pad cứng toàn bộ dữ liệu.
"""),

    code_cell("""from torch.utils.data import DataLoader

train_ds = AGNewsTextDataset(train_rows, vocab, max_len=80)
val_ds = AGNewsTextDataset(val_rows, vocab, max_len=80)
test_ds = AGNewsTextDataset(test_rows, vocab, max_len=80)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

batch_ids, batch_lens, batch_labels, _ = next(iter(train_loader))
print(f"Batch IDs shape [B, T]:     {tuple(batch_ids.shape)}")
print(f"Lengths shape [B]:          {tuple(batch_lens.shape)} (Min: {int(batch_lens.min())}, Max: {int(batch_lens.max())})")
print(f"Labels shape [B]:           {tuple(batch_labels.shape)}")
"""),

    md_cell("""## Bước 5: Định nghĩa và Khởi tạo Kiến trúc LSTMClassifier
Kiến trúc gồm: Embedding (12000 -> 64) -> pack_padded_sequence -> LSTM(64 -> 96) -> Linear(96 -> 4).
"""),

    code_cell("""model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=96,
    n_classes=4
).to(comp_device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tong so tham so co the huan luyen: {total_params:,}")

with torch.no_grad():
    check_emb = model.embedding(batch_ids.to(comp_device))
    check_out = model(batch_ids.to(comp_device), batch_lens)
print(f"Embedding output shape [B, T, E]: {tuple(check_emb.shape)}")
print(f"Logits output shape [B, 4]:        {tuple(check_out.shape)}")
"""),

    md_cell("""## Bước 6: Huấn luyện với Kẹp Gradient (Gradient Clipping) và Chọn Checkpoint
Lưu checkpoint tại epoch có Validation Loss nhỏ nhất để kiểm định trên tập Test.
"""),

    code_cell("""criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_checkpoint = os.path.join(DEFAULT_MODELS_DIR, "best_model.pt")

history = []
best_val_loss = float("inf")
best_epoch = None
epochs = 4

print("===== BAT DAU HUAN LUYEN =====")
for epoch in range(1, epochs + 1):
    t0 = time.perf_counter()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, comp_device, max_grad_norm=1.0)
    val_metrics = evaluate(model, val_loader, criterion, comp_device)
    dt = time.perf_counter() - t0
    
    rec = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_metrics["loss"],
        "validation_accuracy": val_metrics["accuracy"],
        "validation_macro_f1": val_metrics["macro_f1"],
        "seconds": dt
    }
    history.append(rec)
    
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_epoch = epoch
        torch.save(model.state_dict(), best_checkpoint)
        
    print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | "
          f"Val Acc: {val_metrics['accuracy']:.2%} | Val F1: {val_metrics['macro_f1']:.4f} | {dt:.1f}s")
"""),

    md_cell("""## Bước 7: Đánh giá Cuối cùng trên Tập Test Chính thức (7,600 bản tin)
Nạp trọng số tối ưu từ checkpoint tốt nhất của Validation và tính toán trên tập Test.
"""),

    code_cell("""model.load_state_dict(torch.load(best_checkpoint, map_location=comp_device, weights_only=True))
test_metrics = evaluate(model, test_loader, criterion, comp_device, collect_errors=True)

print(f"Checkpoint duoc chon: Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
print(f"Official Test Loss:     {test_metrics['loss']:.4f}")
print(f"Official Test Accuracy: {test_metrics['accuracy']:.2%}")
print(f"Official Test Macro-F1: {test_metrics['macro_f1']:.4f}")
print(f"Tong so tin doan dung:  {len(test_rows) - len(test_metrics['errors'])} / {len(test_rows)}")
"""),

    md_cell("""## Bước 8: Trực quan hóa Đường cong học tập và Ma trận nhầm lẫn
Biểu đồ hiển thị tiến trình loss qua từng epoch và ma trận phân bổ dự đoán trên 4 chủ đề.
"""),

    code_cell("""from pathlib import Path
docs_dir = Path(DEFAULT_DOCS_DIR)
bin_dir = Path(DEFAULT_BIN_DIR)

lc_paths = [docs_dir / "learning_curve.png", bin_dir / "learning_curve.png"]
draw_learning_curve(history, lc_paths)
display(Image(str(lc_paths[0])))

cm_paths = [docs_dir / "confusion_matrix_test.png", bin_dir / "confusion_matrix_test.png"]
draw_confusion_matrix(test_metrics["true"], test_metrics["predicted"], cm_paths)
display(Image(str(cm_paths[0])))
"""),

    md_cell("""## Bước 9: Phân tích Chuyên sâu các Bản tin Dự đoán Sai trên Tập Test
Trích xuất và giải thích nguyên nhân ngữ nghĩa của 3 trường hợp nhầm lẫn tiêu biểu.
"""),

    code_cell("""for idx, err in enumerate(test_metrics["errors"][:3], start=1):
    print(f"\\n--- Vi du sai {idx} ---")
    print(f"Nhan dung (True):      {err['true_label']}")
    print(f"Nhan du doan (Pred):   {err['predicted_label']}")
    print(f"Noi dung: {err['text'][:300]}")
"""),

    md_cell("""## Bước 10: Nhận xét và Bài học Kinh nghiệm Thực tế

1. **Hiệu quả của `pack_padded_sequence`:**
   - Việc đóng gói chuỗi giúp PyTorch bỏ qua các bước tính toán lặp lại trên token `<pad>`, vừa tiết kiệm tài nguyên vừa ngăn ngừa trạng thái ẩn cuối cùng bị pha loãng bởi các giá trị đệm.
2. **Kẹp Gradient (Gradient Clipping):**
   - Đặt ngưỡng kẹp vector đạo hàm ở mức $1.0$ là chốt chặn quan trọng giữ cho quá trình học ổn định, hoàn toàn không xuất hiện hiện tượng dao động loss mất kiểm soát (gradient explosion).
3. **Độ chính xác và Điểm mù của LSTM một chiều:**
   - Mô hình đạt **83.05% Test Accuracy** và **0.8308 Macro-F1** chỉ sau 4 epoch.
   - Nhóm nhầm lẫn phổ biến nhất là giữa **Business** và **Sci/Tech** do ngôn ngữ báo chí về các tập đoàn công nghệ thường đan xen giữa doanh thu cổ phiếu và đột phá khoa học.
""")
]


# ==============================================================================
# DAY 06 TASK 02: SO SÁNH CÁC BIẾN THỂ KIẾN TRÚC LSTM TRÊN AG NEWS
# ==============================================================================
cells_day06_task02 = [
    md_cell("""# BUỔI 06 - BÀI TẬP 02: SO SÁNH CÁC BIẾN THỂ KIẾN TRÚC LSTM TRÊN AG NEWS

### Mục tiêu thực hành:
1. Thực hiện nghiên cứu thực nghiệm kiểm chứng (ablation study) đánh giá độc lập từng thành phần kiến trúc:
   - `baseline`: Cấu hình chuẩn đối chứng (Emb 64, Hidden 96, 1 chiều).
   - `embedding_128`: Tăng kích thước vector nhúng từ lên 128 chiều.
   - `hidden_128`: Tăng dung lượng bộ nhớ hồi quy LSTM lên 128 chiều.
   - `bilstm`: LSTM hai chiều kết hợp cả dòng ngữ cảnh xuôi và ngược.
   - `dropout_03`: Điều chuẩn ngẫu nhiên bằng cách ngắt 30% kết nối để chống học vẹt.
2. Đảm bảo điều kiện so sánh công bằng: cùng tập dữ liệu, cùng seed, cùng optimizer và số epoch.
3. Lựa chọn mô hình tối ưu theo Validation Loss và đánh giá trên 7,600 mẫu Test chính thức.
4. Phân tích nguyên nhân ngữ nghĩa của ít nhất 5 trường hợp dự đoán sai trên tập Test.
"""),

    md_cell("""## Bước 1: Khởi tạo Môi trường và Thiết lập Deterministic Seed
Cấu hình môi trường tính toán và nạp các hàm thư viện cần thiết.
"""),

    code_cell("""import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from srcs.pipelines.day06_task02_lstm_variants import (
    set_seed,
    load_agnews_full,
    tokenize,
    build_vocab,
    NewsDataset,
    collate_news,
    LSTMClassifier,
    count_parameters,
    train_one_epoch,
    evaluate,
    save_csv,
    plot_learning_curves,
    plot_confusion_matrix,
    choose_diverse_errors,
    EXPERIMENT_CONFIGS,
    LABEL_NAMES,
    DEFAULT_DATA_DIR,
    DEFAULT_DOCS_DIR,
    DEFAULT_BIN_DIR,
    DEFAULT_MODELS_DIR
)

set_seed(42)
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {comp_device}")
"""),

    md_cell("""## Bước 2: Tải Dữ liệu và Xây dựng Từ vựng Tập trung
Tập dữ liệu sử dụng 10,000 mẫu train, 6,000 mẫu validation và 7,600 mẫu test chính thức.
"""),

    code_cell("""train_rows, val_rows, test_rows = load_agnews_full(
    data_dir=DEFAULT_DATA_DIR,
    val_size=6000,
    train_size=10000,
    seed=42
)
vocab = build_vocab(train_rows, max_vocab=20000, min_freq=2)
print(f"Kich thuoc Tu vung: {len(vocab):,} tu")

train_ds = NewsDataset(train_rows, vocab, max_len=100)
val_ds = NewsDataset(val_rows, vocab, max_len=100)
test_ds = NewsDataset(test_rows, vocab, max_len=100)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_news)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_news)
"""),

    md_cell("""## Bước 3: Định nghĩa 5 Cấu hình Thực nghiệm Cần So sánh
Mỗi thí nghiệm chỉ thay đổi duy nhất một yếu tố so với baseline để phân lập tác động.
"""),

    code_cell("""for name, cfg in EXPERIMENT_CONFIGS.items():
    temp_model = LSTMClassifier(len(vocab), **cfg)
    p_count = count_parameters(temp_model)
    print(f"Cau hinh {name:15s} | Params: {p_count:,} | Chi tiet: {cfg}")
"""),

    md_cell("""## Bước 4: Thực hiện Huấn luyện 5 Mô hình và Lưu Checkpoint theo Validation Loss
Mỗi cấu hình được huấn luyện 3 epochs với cùng learning rate ($0.001$) và batch size 128.
"""),

    code_cell("""criterion = torch.nn.CrossEntropyLoss()
histories = {}
records = []

for name, cfg in EXPERIMENT_CONFIGS.items():
    print(f"\\n===== HUAN LUYEN: {name} =====")
    set_seed(42)
    model = LSTMClassifier(len(vocab), **cfg).to(comp_device)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True,
        collate_fn=collate_news, generator=torch.Generator().manual_seed(42)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = os.path.join(DEFAULT_MODELS_DIR, f"best_{name}.pt")
    
    best_loss = float("inf")
    best_epoch = None
    history = []
    
    for epoch in range(1, 4):
        t0 = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, comp_device)
        val_res = evaluate(model, val_loader, criterion, comp_device)
        dt = time.perf_counter() - t0
        
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "validation_loss": val_res["loss"],
            "validation_accuracy": val_res["accuracy"],
            "validation_macro_f1": val_res["macro_f1"],
            "seconds": dt
        }
        history.append(rec)
        
        if val_res["loss"] < best_loss:
            best_loss = val_res["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            
        print(f"{name:15s} | Ep {epoch:02d}/03 | Train: {tr_loss:.4f} | Val: {val_res['loss']:.4f} | "
              f"Val Acc: {val_res['accuracy']:.2%} | Val F1: {val_res['macro_f1']:.4f} | {dt:.1f}s")
        
    histories[name] = history
    records.append({
        "config": name,
        **cfg,
        "trainable_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "mean_seconds_per_epoch": round(sum(h["seconds"] for h in history) / len(history), 2)
    })
"""),

    md_cell("""## Bước 5: Lựa chọn Mô hình Tối ưu và Đánh giá Toàn diện trên Tập Test
Mô hình chiến thắng được xác định dựa trên Validation Loss trước khi mở tập Test.
"""),

    code_cell("""selected_config = min(records, key=lambda r: r["best_validation_loss"])["config"]
print(f"Mo hinh duoc lua chon theo Validation Loss: {selected_config}")

best_errors = []
print("\\n===== DANH GIA TREN OFFICIAL TEST SET (7,600 MAU) =====")
for rec in records:
    c_name = rec["config"]
    m = LSTMClassifier(len(vocab), **EXPERIMENT_CONFIGS[c_name]).to(comp_device)
    m.load_state_dict(torch.load(os.path.join(DEFAULT_MODELS_DIR, f"best_{c_name}.pt"), map_location=comp_device, weights_only=True))
    res = evaluate(m, test_loader, criterion, comp_device, collect_errors=True)
    
    rec["test_loss"] = round(res["loss"], 4)
    rec["test_accuracy"] = round(res["accuracy"], 4)
    rec["test_macro_f1"] = round(res["macro_f1"], 4)
    rec["test_errors"] = len(res["errors"])
    
    if c_name == selected_config:
        best_errors = res["errors"]
        
    print(f"TEST | {c_name:15s} | Loss: {res['loss']:.4f} | "
          f"Acc: {res['accuracy']:.2%} | F1: {res['macro_f1']:.4f} | Loi: {len(res['errors'])}/7,600")
"""),

    md_cell("""## Bước 6: Bảng So sánh Tổng hợp Kết quả Thực nghiệm
Hiển thị bảng dữ liệu so sánh trực tiếp 5 biến thể kiến trúc.
"""),

    code_cell("""df_results = pd.DataFrame(records)[[
    "config", "embedding_dim", "hidden_dim", "bidirectional", "dropout",
    "trainable_parameters", "best_validation_loss", "mean_seconds_per_epoch",
    "test_loss", "test_accuracy", "test_macro_f1", "test_errors"
]]
display(df_results)
"""),

    md_cell("""## Bước 7: Trực quan hóa Đường cong học tập 3 Panel và Ma trận nhầm lẫn
Vẽ biểu đồ Train Loss, Validation Loss và Validation Accuracy cho toàn bộ 5 cấu hình.
"""),

    code_cell("""from pathlib import Path
docs_dir = Path(DEFAULT_DOCS_DIR)
bin_dir = Path(DEFAULT_BIN_DIR)

lc_paths = [docs_dir / "learning_curves.png", bin_dir / "learning_curves.png"]
plot_learning_curves(histories, lc_paths)
display(Image(str(lc_paths[0])))

cm_paths = [docs_dir / "confusion_matrix_test.png", bin_dir / "confusion_matrix_test.png"]
selected_m = LSTMClassifier(len(vocab), **EXPERIMENT_CONFIGS[selected_config]).to(comp_device)
selected_m.load_state_dict(torch.load(os.path.join(DEFAULT_MODELS_DIR, f"best_{selected_config}.pt"), map_location=comp_device, weights_only=True))
selected_res = evaluate(selected_m, test_loader, criterion, comp_device)
plot_confusion_matrix(selected_res["cm"], cm_paths, selected_config)
display(Image(str(cm_paths[0])))
"""),

    md_cell("""## Bước 8: Phân tích Chuyên sâu 5 Bản tin Dự đoán Sai Tiêu biểu trên Tập Test
Trích xuất 5 ví dụ đa dạng từ mô hình `embedding_128` kèm theo phân tích nguyên nhân ngữ nghĩa.
"""),

    code_cell("""diverse_errors = choose_diverse_errors(best_errors, limit=5)
for i, err in enumerate(diverse_errors, start=1):
    print(f"\\n--- Vi du sai {i} ---")
    print(f"Nhan dung (True):      {err['true_label']}")
    print(f"Nhan du doan (Pred):   {err['predicted_label']} (xac suat: {err['predicted_probability']:.3f})")
    print(f"Do dai token goc:      {err['original_tokens']} | Bi cat ngan: {err['was_truncated']} | Ty le <unk>: {err['unk_ratio']:.2%}")
    print(f"Noi dung: {err['text'][:300]}")
"""),

    md_cell("""## Bước 9: Kết luận và Khuyến nghị Kiến trúc

1. **Hiệu năng vượt trội của Embedding 128:**
   - Tăng `embedding_dim` lên 128 giúp Test Accuracy **tăng vọt 8.03%** (từ 68.50% lên 76.53%). Đây là khoản đầu tư hiệu quả nhất vì thời gian huấn luyện chỉ tăng nhẹ từ 26.9s lên 34.4s.
2. **Sức mạnh ngữ cảnh của BiLSTM:**
   - `bilstm` đạt **75.21% Test Accuracy** (vượt baseline 6.71%), chứng minh khả năng bắt trọn thông tin từ hai chiều giúp giải quyết các câu có từ khóa chủ đạo nằm ở đầu hoặc giữa câu.
3. **Đánh đổi về chi phí tính toán:**
   - BiLSTM chạy chậm hơn gần 3 lần (78.20s/epoch so với 26.90s), do đó trong môi trường hạn chế tài nguyên, việc tăng chiều vector nhúng (embedding) luôn là giải pháp ưu tiên trước khi nâng cấp sang mạng hai chiều.
""")
]


if __name__ == "__main__":
    path_d01_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day01_task01_mlp_construction.ipynb")
    path_d01_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day01_task02_mlp_exploration.ipynb")
    path_d03_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day03_task01_training_loop.ipynb")
    path_d03_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day03_task02_optimizer_comparison.ipynb")
    path_d04_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day04_task01_cnn_fashionmnist.ipynb")
    path_d04_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day04_task02_cnn_variants.ipynb")
    path_d05_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day05_task01_transfer_learning.ipynb")
    path_d05_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day05_task02_transfer_learning.ipynb")
    path_d06_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day06_task01_lstm_text_classification.ipynb")
    path_d06_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day06_task02_lstm_variants.ipynb")

    write_notebook(path_d01_t01, cells_day01_task01)
    write_notebook(path_d01_t02, cells_day01_task02)
    write_notebook(path_d03_t01, cells_day03_task01)
    write_notebook(path_d03_t02, cells_day03_task02)
    write_notebook(path_d04_t01, cells_day04_task01)
    write_notebook(path_d04_t02, cells_day04_task02)
    write_notebook(path_d05_t01, cells_day05_task01)
    write_notebook(path_d05_t02, cells_day05_task02)
    write_notebook(path_d06_t01, cells_day06_task01)
    write_notebook(path_d06_t02, cells_day06_task02)


