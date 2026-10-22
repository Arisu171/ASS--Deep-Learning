# 3.2. Object-Oriented Design for Implementation: Thiết kế Hướng đối tượng trong Framework Học sâu

Nguồn tham khảo: [3.2.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/oo-design.html#exercises).

---

### Bài 1: Locate the complete D2L class implementations

**Đề bài:** Phương pháp định vị và phân tích mã nguồn hoàn chỉnh của các lớp cơ sở (`DataModule`, `Module`, `Trainer`) trong thư viện D2L thông qua kỹ thuật phản chiếu (Reflection/Introspection) trong Python.

**Phân tích và Lời giải:**

Trong thiết kế hệ thống học sâu hiện đại (tương tự như PyTorch Lightning hay Hugging Face Accelerate), việc tách biệt ba thực thể chính:
1. `DataModule`: Đóng gói quy trình tiền xử lý, chia tập dữ liệu và nạp dữ liệu mini-batch.
2. `Module` (kế thừa `torch.nn.Module`): Đóng gói kiến trúc mô hình, hàm tính forward pass và hàm tính loss.
3. `Trainer`: Điều phối vòng lặp huấn luyện, tối ưu hóa gradient, tính toán số liệu đánh giá và quản lý phần cứng (CPU/GPU).

Để kiểm tra trực tiếp mã nguồn của lớp cơ sở trong môi trường thực thi, ta sử dụng mô-đun chuẩn `inspect`:

```python
import inspect
from d2l import torch as d2l

# Xác định đường dẫn tệp mã nguồn vật lý
print("Đường dẫn mã nguồn DataModule:", inspect.getfile(d2l.DataModule))

# Trích xuất toàn bộ mã nguồn của lớp
print(inspect.getsource(d2l.DataModule))
```

*Phân tích kỹ thuật:* Việc đào sâu mã nguồn lớp cha giúp làm sáng tỏ cơ chế giải phóng bộ đệm (state caching), cách kích hoạt các callback/hooks tại từng epoch, và luồng dữ liệu giữa `validation_step` và `training_step`.

---

### Bài 2: Remove `save_hyperparameters` from class B

**Đề bài:** Phân tích ảnh hưởng khi loại bỏ lời gọi `save_hyperparameters()` trong hàm khởi tạo `__init__` của một lớp con. Giải thích cơ chế nội tại của phương thức này trong Python.

**Phân tích và Lời giải:**

1. **Cơ chế hoạt động của `save_hyperparameters()`:**
   Phương thức tiện ích này kiểm tra khung ngăn xếp thực thi (execution stack frame) của hàm `__init__` thông qua `inspect.currentframe().f_back`. Nó tự động trích xuất toàn bộ các đối số truyền vào (`*args`, `**kwargs`) và gán chúng thành các thuộc tính của đối tượng `self` (hoặc lưu trong thuộc tính từ điển `self.hparams`).

2. **Hệ quả khi loại bỏ:**
   Nếu loại bỏ `self.save_hyperparameters()` trong lớp con `B`, các đối số như `a` và `b` chỉ là các biến cục bộ (local scope) tồn tại trong thời gian thực thi hàm khởi tạo. Sau khi hàm kết thúc, các biến này bị bộ dọn rác (garbage collector) thu hồi. Lời gọi truy xuất `self.a` hoặc `self.b` từ bên ngoài sẽ kích hoạt ngoại lệ `AttributeError: 'B' object has no attribute 'a'`.

3. **Cài đặt tường minh tương đương chuẩn PEP 8:**

```python
class B:
    def __init__(self, a, b):
        # Gán thuộc tính tường minh không phụ thuộc phản chiếu runtime
        self.a = a
        self.b = b
```

*Đánh giá kiến trúc:* Mặc dù `save_hyperparameters()` giảm thiểu đáng kể mã lặp (boilerplate code) khi thử nghiệm hàng loạt siêu tham số kiến trúc, việc gán tường minh trong các thư viện production giúp các công cụ kiểm tra kiểu tĩnh (như `mypy`) và IDE nhận diện thuộc tính chính xác hơn.
