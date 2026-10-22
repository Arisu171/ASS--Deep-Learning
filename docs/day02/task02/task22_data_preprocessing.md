# 2. Preliminaries: Dive into Deep Learning 1.0.3

## 2.2. Data Preprocessing: Tiền xử lý dữ liệu

Nguồn tham khảo: [2.2.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_preliminaries/pandas.html#exercises).

---

### Bài 1: Inspecting the UCI Abalone dataset

**Đề bài:** Tải tập dữ liệu UCI Abalone, thực hiện phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA), kiểm tra tình trạng dữ liệu khuyết thiếu (missing values) và phân loại các biến thành biến định lượng (numerical), biến định danh (categorical) hoặc văn bản tự do (free text).

**Phân tích và Lời giải:**

Tập dữ liệu Abalone từ UCI Machine Learning Repository phục vụ bài toán hồi quy dự báo tuổi của bào ngư thông qua số lượng vòng vỏ (`Rings`).

```python
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = [
    "Sex", "Length", "Diameter", "Height", "Whole_weight",
    "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"
]
df = pd.read_csv(url, names=columns)

# Kiểm tra chiều không gian và tính toàn vẹn dữ liệu
print("Kích thước ma trận dữ liệu (N, D):", df.shape)
print("Tổng số giá trị khuyết thiếu (NaN):", df.isna().sum().sum())
print("Kiểu dữ liệu tương ứng của các thuộc tính:\n", df.dtypes)
```

**Kết quả phân tích thống kê:**
- Tập dữ liệu gồm $N = 4.177$ mẫu quan sát và $D = 9$ biến thuộc tính.
- Mức độ toàn vẹn: Tỷ lệ khuyết thiếu đạt $0\%$ trên toàn bộ tập dữ liệu (không chứa giá trị `NaN` hoặc `null`).

| Loại biến thuộc tính | Số lượng | Tỷ lệ (%) | Danh sách thuộc tính cụ thể |
| :--- | :---: | :---: | :--- |
| **Biến định lượng liên tục (Continuous Numerical)** | 7 | 77,78% | `Length`, `Diameter`, `Height`, `Whole_weight`, `Shucked_weight`, `Viscera_weight`, `Shell_weight` (đơn vị: mm và gram, kiểu `float64`) |
| **Biến định lượng rời rạc (Discrete Numerical / Target)** | 1 | 11,11% | `Rings` (đếm số vòng vỏ, kiểu `int64`, tương ứng số tuổi = `Rings` + 1.5) |
| **Biến định danh (Nominal Categorical)** | 1 | 11,11% | `Sex` (nhãn danh mục: `M`, `F`, `I` [Infant], kiểu `object`) |
| **Văn bản tự do (Free Text)** | 0 | 0,00% | Không có |

---

### Bài 2: Selecting columns by name

**Đề bài:** Thực hiện trích xuất tập con đặc trưng (feature subsetting) theo tên biến thay vì truy xuất dựa trên chỉ số vị trí số học (positional index).

**Phân tích và Lời giải:**

Trong kỹ nghệ dữ liệu, việc truy xuất thuộc tính bằng tên danh định (explicit label-based indexing) thông qua `.loc[]` hoặc truy xuất khóa dictionary đảm bảo tính bền vững (robustness) của pipeline xử lý. Cách tiếp cận này ngăn chặn lỗi rò rỉ dữ liệu hoặc sai lệch đặc trưng khi thứ tự các cột bị biến đổi trong quá trình tích hợp dữ liệu.

```python
# Trích xuất tập con đặc trưng phục vụ phân tích
features_subset = df.loc[:, ["Sex", "Length", "Rings"]]
target_series = df["Rings"]
```

---

### Bài 3: Estimating the practical dataset size

**Đề bài:** Ước tính quy mô tập dữ liệu tối đa có thể xử lý trực tiếp trong bộ nhớ (in-memory processing) bằng pandas. Phân tích các nút thắt cổ chai tài nguyên trên máy trạm cá nhân và máy chủ tính toán.

**Phân tích và Lời giải:**

1. **Hạn chế bộ nhớ trong của pandas (Memory Overhead):**
   - Một `DataFrame` trong pandas tiêu tốn lượng RAM thực tế gấp từ 3 đến 5 lần so với kích thước tệp dữ liệu thô (raw CSV) trên ổ đĩa do cấu trúc đối tượng Python (overhead của metadata con trỏ kiểu, block manager, và tính chất bất biến trong một số thao tác sao chép ngầm).
   - *Ví dụ thực tế:* Tập dữ liệu Abalone có dung lượng đĩa xấp xỉ $191 \text{ KB}$, nhưng khi tải vào pandas với `memory_usage(deep=True)` chiếm khoảng $0,454 \text{ MiB}$.
   - Đối với hệ thống có dung lượng bộ nhớ khả dụng $M_{\text{avail}}$, quy mô tệp thô an toàn tối đa có thể tải trực tiếp chỉ nên thỏa mãn $S_{\text{file}} \le \frac{1}{3} M_{\text{avail}}$ đến $\frac{1}{5} M_{\text{avail}}$.

2. **Các giải pháp kiến trúc cho dữ liệu quy mô lớn (Out-of-Core Processing):**
   - **Xử lý theo khối (Chunking):** Sử dụng tham số `chunksize` trong `pd.read_csv()` để tạo các trình lặp generator cho phép xử lý từng đợt dữ liệu.
   - **Tối ưu hóa biểu diễn kiểu dữ liệu (Downcasting):** Chuyển đổi `float64` sang `float32`, `int64` sang `int16`/`int8`, và các biến phân loại chuỗi sang kiểu `category`.
   - **Áp dụng công nghệ lưu trữ theo cột và tính toán phân tán:** Thay thế pandas bằng các công cụ hiện đại như **Apache Arrow / Polars** (tối ưu hóa bộ nhớ không cần sao chép - zero-copy, xử lý đa luồng), **DuckDB** (truy vấn SQL trên tệp Parquet ngoài bộ nhớ), hoặc **Dask / PySpark** cho tính toán cụm phân tán.

---

### Bài 4: Handling high-cardinality categories

**Đề bài:** Phân tích chiến lược xử lý các biến phân loại có lực lượng lớn (high-cardinality categorical features). Đánh giá tính khả thi khi đưa các thuộc tính định danh duy nhất (unique identifiers) vào mô hình học sâu.

**Phân tích và Lời giải:**

1. **Thách thức của phương pháp One-Hot Encoding:**
   - Khi một biến danh mục có $K$ giá trị khác biệt với $K \gg 10^3$ (ví dụ: mã bưu chính, ID người dùng), biểu diễn One-Hot sẽ tạo ra một ma trận có số chiều cực lớn và phân tán cao (sparse matrix). Điều này dẫn đến sự bùng nổ tham số ở lớp tuyến tính đầu tiên, làm suy giảm nghiêm trọng hiệu quả tính toán và gây hiện tượng quá khớp (curse of dimensionality).

2. **Chiến lược xử lý tối ưu:**
   - **Nhúng thực thể (Entity Embeddings):** Ánh xạ từng danh mục vào một không gian vector liên tục chiều thấp $\mathbb{R}^d$ ($d \ll K$), trong đó các vector nhúng được tối ưu hóa đồng thời cùng tham số mạng thông qua lan truyền ngược (backpropagation).
   - **Gộp danh mục tần suất thấp (Frequency Thresholding):** Gom các danh mục xuất hiện hiếm gặp vào một lớp chung `Others`.
   - **Target Encoding có điều chuẩn (Regularized Target Encoding):** Thay thế danh mục bằng kỳ vọng của biến mục tiêu kèm kỹ thuật làm mịn (smoothing) hoặc kiểm định chéo để triệt tiêu hiện tượng rò rỉ mục tiêu (target leakage).
   - **Biến băm (Hashing Trick):** Ánh xạ không gian danh mục vô hạn về một không gian có số chiều cố định bằng hàm băm.

3. **Thuộc tính định danh duy nhất (Primary Keys / IDs):**
   - Thuộc tính có giá trị duy nhất trên từng bản ghi (như mã giao dịch, số CMND/CCCD) có dung lượng thông tin tương hỗ bằng không đối với quy luật tổng quát của bài toán.
   - Nếu đưa vào mô hình, mạng nơ-ron có thể "học vẹt" (memorization) thay vì khái quát hóa (generalization). Về nguyên tắc học máy, các thuộc tính định danh duy nhất **bắt buộc phải bị loại bỏ** khỏi ma trận đặc trưng đầu vào $\mathbf{X}$.

---

### Bài 5: Alternatives to pandas

**Đề bài:** Tổng hợp và đánh giá các công cụ tiền xử lý dữ liệu thay thế pandas trong các hệ thống học sâu hiện đại đối với dữ liệu mảng, hình ảnh và bảng quy mô lớn.

**Phân tích và Lời giải:**

| Miền dữ liệu | Công cụ chuyên dụng | Đặc tính kỹ thuật nổi bật |
| :--- | :--- | :--- |
| **Dữ liệu mảng số học đa chiều (Tensors / Arrays)** | **NumPy, Zarr, HDF5** | Cung cấp cơ chế bộ nhớ ánh xạ (`np.memmap`), lưu trữ nhị phân nén dạng khối (`.npy`, `.zarr`) cho phép đọc ngẫu nhiên nhanh và tiết kiệm RAM khi huấn luyện mini-batch. |
| **Dữ liệu thị giác (Images / Vision)** | **Pillow (PIL), OpenCV, Torchvision, Albumentations** | Tối ưu hóa trên bộ nhớ C/C++, hỗ trợ xử lý tăng cường dữ liệu đa luồng (multi-threaded data augmentations) trực tiếp hoặc gia tốc trên GPU thông qua CUDA/Kornia. |
| **Dữ liệu dạng bảng quy mô lớn (Large-scale Tabular Data)** | **Polars, Apache Arrow, DuckDB, Dask, Ray Data** | Kiến trúc định hướng cột (columnar format), tính toán lười (lazy evaluation), tối ưu hóa đồ thị truy vấn, tận dụng chỉ thị SIMD và đa nhân CPU. |
| **Đường ống nạp dữ liệu học sâu (Deep Learning Data Pipelines)** | **PyTorch `DataLoader` / `IterableDataset`, WebDataset, TFRecord** | Hỗ trợ nạp dữ liệu không đồng bộ (asynchronous prefetching), đa tiến trình (`num_workers > 0`), nạp dữ liệu phân tán (distributed data parallel) và ghim bộ nhớ RAM (`pin_memory=True`) giúp tối đa hóa thông lượng đưa dữ liệu vào GPU. |
