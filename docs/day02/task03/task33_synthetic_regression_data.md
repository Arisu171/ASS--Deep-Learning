# 3.3. Synthetic Regression Data: Tạo Dữ liệu Hồi quy Tổng hợp và Cơ chế Lấy mẫu

Nguồn tham khảo: [3.3.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/synthetic-regression-data.html#exercises).

---

### Bài 1: Incomplete final minibatch

**Đề bài:** Phân tích ảnh hưởng của lô dữ liệu cuối cùng khi số lượng mẫu $N$ không chia hết cho kích thước lô $B$ ($N \not\equiv 0 \pmod B$). Đánh giá tùy chọn `drop_last` trong các bộ nạp dữ liệu chuẩn.

**Phân tích và Lời giải:**

Khi phân chia $N$ mẫu thành các mini-batch kích thước $B$:
- Số lượng batch hoàn chỉnh: $\lfloor N / B \rfloor$.
- Kích thước của batch dư cuối cùng: $R = N \pmod B$ ($0 < R < B$).

1. **Tác động lên quá trình tối ưu hóa:**
   - Nếu hàm mất mát tính tổng ($\mathcal{L}_{\text{sum}}$), batch dư có kích thước $R < B$ sẽ tạo ra chuẩn gradient nhỏ hơn đáng kể, làm sai lệch thang đo bước cập nhật SGD.
   - Nếu hàm mất mát tính trung bình ($\mathcal{L}_{\text{mean}}$), kỳ vọng toán của gradient không đổi, nhưng phương sai của gradient ước lượng sẽ tăng tỷ lệ với $\frac{1}{R} > \frac{1}{B}$, tạo nhiễu ngẫu nhiên lớn hơn vào bước nhảy cuối cùng của epoch.
   - Đặc biệt trong các kiến trúc có lớp Chuẩn hóa theo lô (Batch Normalization), một batch quá nhỏ ($R = 1$) sẽ gây lỗi chia cho 0 khi tính phương sai mẫu hoặc ước lượng thống kê không ổn định.
2. **Tham số `drop_last` trong PyTorch `DataLoader`:**
   - `drop_last=False` (mặc định): Giữ lại batch dư cuối, đảm bảo không bỏ sót mẫu dữ liệu nào trong epoch huấn luyện.
   - `drop_last=True`: Loại bỏ $R$ mẫu cuối cùng, giữ cho mọi batch đều có kích thước $B$ đồng nhất. Đây là cấu hình bắt buộc khi huấn luyện phân tán đa GPU (Distributed Data Parallel) để tránh hiện tượng mất đồng bộ giao tiếp (deadlock/hang) giữa các tiến trình.

---

### Bài 2: Huge datasets stored on disk

**Đề bài:** Thiết kế kiến trúc nạp và xáo trộn dữ liệu ngoài bộ nhớ (Out-of-Core Data Shuffling) cho tập dữ liệu có quy mô vượt quá dung lượng RAM khả dụng, giảm thiểu chi phí truy xuất đĩa I/O ngẫu nhiên.

**Kiến trúc Đề xuất:**

1. **Phân mảnh và nạp theo khối (Sharding & Buffered Streaming):**
   - Dữ liệu trên đĩa được lưu thành $K$ tệp phân mảnh tuần tự (shards), ví dụ theo định dạng Apache Arrow, WebDataset (tarball), hoặc TFRecord.
   - Mỗi epoch, thực hiện hoán vị ngẫu nhiên danh sách các shard: $\pi_{\text{shards}} \in \mathcal{S}_K$.
   - Bộ nạp duy trì một bộ đệm vòng (ring buffer) kích thước hữu hạn $M$ mẫu trong RAM. Khi dữ liệu được stream tuần tự từ shard vào bộ đệm, giải thuật Reservoir Sampling hoặc xáo trộn ngẫu nhiên cục bộ trong buffer được áp dụng để sinh ra các mini-batch.
2. **Hoán vị chỉ số giả ngẫu nhiên toàn cục (Pseudo-Random Index Permutation):**
   - Thay vì xáo trộn vật lý dữ liệu trên đĩa, ta tạo một hàm song ánh giả ngẫu nhiên ánh xạ chỉ số $f: \{0, \dots, N-1\} \to \{0, \dots, N-1\}$ sử dụng mạng mật mã Feistel kết hợp kỹ thuật Cycle-Walking.
   - Để tránh việc đọc đĩa ngẫu nhiên làm giảm băng thông IOPS, các chỉ số sau hoán vị được gom cụm (chunked) theo khối địa chỉ vật lý lân cận trước khi tải.

---

### Bài 3: Generate fresh data on every iteration

**Đề bài:** Thiết lập một bộ sinh dữ liệu động (Dynamic Data Generator) tạo ra một ma trận dữ liệu tổng hợp hoàn toàn mới tại mỗi epoch, mô phỏng quá trình huấn luyện trực tuyến trên dòng dữ liệu vô hạn.

**Hiện thực hóa chuẩn OOP PyTorch:**

```python
import torch

class DynamicRegressionData:
    def __init__(self, w: torch.Tensor, b: float, num_examples: int, batch_size: int, base_seed: int = 42):
        self.w = w.reshape(-1, 1)
        self.b = b
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.epoch = 0

    def __iter__(self):
        # Đảm bảo mỗi epoch sinh dữ liệu mới nhưng có thể tái lập thông qua seed tăng dần
        gen = torch.Generator()
        gen.manual_seed(self.base_seed + self.epoch)
        self.epoch += 1

        # Sinh ma trận đặc trưng chuẩn hóa và nhãn hồi quy tuyến tính có nhiễu Gauss
        X = torch.randn(self.num_examples, self.w.numel(), generator=gen)
        noise = 0.01 * torch.randn(self.num_examples, 1, generator=gen)
        y = torch.matmul(X, self.w) + self.b + noise

        # Xáo trộn chỉ số
        indices = torch.randperm(self.num_examples, generator=gen)
        for i in range(0, self.num_examples, self.batch_size):
            batch_idx = indices[i: i + self.batch_size]
            yield X[batch_idx], y[batch_idx]
```

---

### Bài 4: Reproducible data on repeated calls

**Đề bài:** Thiết kế cơ chế đảm bảo tính tái lập kết quả thực nghiệm (Determinism and Reproducibility) tuyệt đối cho các bộ sinh dữ liệu ngẫu nhiên.

**Phân tích và Hiện thực hóa:**

Để đảm bảo tính tái lập mà không gây ảnh hưởng phụ (side effects) đến trạng thái ngẫu nhiên toàn cục của chương trình (`torch.manual_seed`), cần tạo và truyền một đối tượng `torch.Generator` cục bộ cho mọi phép toán sinh số ngẫu nhiên:

```python
def generate_deterministic_synthetic_data(
    w: torch.Tensor, b: float, num_examples: int, seed: int = 1337
) -> tuple[torch.Tensor, torch.Tensor]:
    # Khởi tạo generator với không gian trạng thái độc lập
    local_gen = torch.Generator().manual_seed(seed)
    
    num_features = w.numel()
    X = torch.randn(num_examples, num_features, generator=local_gen)
    epsilon = 0.01 * torch.randn(num_examples, 1, generator=local_gen)
    y = torch.matmul(X, w.reshape(-1, 1)) + b + epsilon
    
    return X, y
```

*Nguyên tắc kỹ thuật:* Việc đóng gói trạng thái ngẫu nhiên vào đối tượng `torch.Generator` độc lập là chuẩn mực thiết kế trong các thử nghiệm nghiên cứu học máy, đảm bảo rằng việc thay đổi số lượng luồng nạp dữ liệu hay lời gọi hàm ở nơi khác trong mã nguồn không làm biến đổi phân phối dữ liệu đầu vào.
