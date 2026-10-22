# 2. Preliminaries: Dive into Deep Learning 1.0.3

## 2.1. Data Manipulation: Thao tác với Tensor

Nguồn tham khảo: [2.1.8. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_preliminaries/ndarray.html#exercises).

---

### Bài 1: Changing an elementwise comparison

**Đề bài:** Thay thế phép so sánh bằng (`==`) giữa hai tensor bằng các toán tử quan hệ nhỏ hơn (`<`) hoặc lớn hơn (`>`), phân tích đặc điểm của tensor kết quả.

**Phân tích và Lời giải:**

Trong các thư viện tính toán tensor (như PyTorch hay JAX), các toán tử so sánh quan hệ được thực thi theo cơ chế từng phần tử (element-wise operations). Nếu hai tensor có cùng kích thước (shape), phép so sánh sẽ đánh giá điều kiện độc lập tại từng vị trí chỉ số và trả về một tensor kiểu logic (Boolean tensor, `torch.bool`) có cùng kích thước.

```python
import torch

X = torch.tensor([[1, 2], [3, 4]])
Y = torch.tensor([[1, 1], [4, 4]])

print("X == Y:\n", X == Y)
print("X < Y:\n", X < Y)
print("X > Y:\n", X > Y)
```

**Kết quả thực nghiệm:**

```text
X == Y:
tensor([[ True, False],
        [False,  True]])
X < Y:
tensor([[False, False],
        [ True, False]])
X > Y:
tensor([[False,  True],
        [False, False]])
```

*Nhận xét:* Tensor logic thu được thường đóng vai trò mặt nạ nhị phân (binary mask) để phục vụ các kỹ thuật trích xuất phần tử có điều kiện (boolean indexing), tính độ chính xác phân loại (classification accuracy), hoặc cài đặt các hàm kích hoạt ngắt (như Heaviside step function).

---

### Bài 2: Testing broadcasting with other tensor shapes

**Đề bài:** Khảo sát cơ chế lan truyền (broadcasting mechanism) với các tensor đa chiều có số chiều và kích thước khác nhau. Phân tích điều kiện tương thích và nguyên lý cấp phát bộ nhớ.

**Phân tích và Lời giải:**

Cơ chế lan truyền (Broadcasting) cho phép thực hiện các phép toán số học giữa hai tensor có kích thước khác nhau mà không cần sao chép dữ liệu thực tế trong bộ nhớ.

1. **Quy tắc tương thích chiều (Broadcasting Semantics):**
   - Hai chiều tại cùng vị trí (tính từ trục cuối cùng căn sang trái, right-to-left) được xem là tương thích nếu:
     1. Kích thước của chúng bằng nhau, hoặc
     2. Một trong hai chiều có kích thước bằng 1.
   - Nếu một tensor có số chiều ít hơn, các chiều kích thước 1 sẽ được bổ sung ngầm định vào phía trước (bên trái).

2. **Cơ chế quản lý bộ nhớ (Memory Striding):**
   - Về mặt vật lý, backend tensor không nhân bản phần tử trong RAM mà điều chỉnh bước nhảy bộ nhớ (strides) của trục có chiều dài 1 về giá trị 0. Khi chỉ số thay đổi dọc theo trục đó, con trỏ bộ nhớ vẫn trỏ về cùng một vị trí ô nhớ ban đầu.

**Thực nghiệm với tensor 3 chiều:**

```python
import torch

# Khởi tạo tensor với kích thước khác nhau
a = torch.arange(6).reshape(2, 1, 3)  # Kích thước: (2, 1, 3)
b = torch.arange(4).reshape(1, 4, 1)  # Kích thước: (1, 4, 1)

# Áp dụng phép cộng lan truyền
c = a + b
print("Kích thước tensor kết quả c:", c.shape)
print("Lát cắt c[0]:\n", c[0])
```

**Kết quả:**
- Kích thước của $c$ sau broadcasting là $(2, 4, 3)$, do $\max(2, 1) = 2$, $\max(1, 4) = 4$, và $\max(3, 1) = 3$.
- Lát cắt $c[0] \in \mathbb{R}^{4 \times 3}$ thể hiện sự kết hợp cộng tuyến tính giữa vector dòng và vector cột dọc theo các trục được mở rộng logic.
