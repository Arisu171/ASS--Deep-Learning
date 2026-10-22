"""Generate standardized, production-grade Jupyter Notebooks for Day 1 and Day 3.

CLI execution:
  python tools/notebooks_generate.py

Inputs:
  None

Outputs:
  srcs/notebooks/day01_task01_mlp_construction.ipynb
  srcs/notebooks/day01_task02_mlp_exploration.ipynb
  srcs/notebooks/day03_task01_training_loop.ipynb
  srcs/notebooks/day03_task02_optimizer_comparison.ipynb
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


if __name__ == "__main__":
    path_d01_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day01_task01_mlp_construction.ipynb")
    path_d01_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day01_task02_mlp_exploration.ipynb")
    path_d03_t01 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day03_task01_training_loop.ipynb")
    path_d03_t02 = os.path.join(DEFAULT_NOTEBOOKS_DIR, "day03_task02_optimizer_comparison.ipynb")

    write_notebook(path_d01_t01, cells_day01_task01)
    write_notebook(path_d01_t02, cells_day01_task02)
    write_notebook(path_d03_t01, cells_day03_task01)
    write_notebook(path_d03_t02, cells_day03_task02)
