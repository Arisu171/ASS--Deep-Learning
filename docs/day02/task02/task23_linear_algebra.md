# 2. Preliminaries: Dive into Deep Learning 1.0.3

## 2.3. Linear Algebra: Đại số tuyến tính trong Học sâu

Nguồn tham khảo: [2.3.13. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_preliminaries/linear-algebra.html#exercises).

---

### Bài 1: Transpose twice

**Đề bài:** Chứng minh rằng phép chuyển vị hai lần liên tiếp của một ma trận trả về chính ma trận ban đầu: $(\mathbf{A}^\top)^\top = \mathbf{A}$.

**Chứng minh:**

Xét ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$. Theo định nghĩa phép chuyển vị ma trận, phần tử tại vị trí $(i, j)$ của ma trận $\mathbf{A}^\top \in \mathbb{R}^{n \times m}$ được xác định bởi:

$$[\mathbf{A}^\top]_{i, j} = [\mathbf{A}]_{j, i}, \quad \forall i \in \{1, \dots, n\}, \, j \in \{1, \dots, m\}$$

Áp dụng tiếp phép chuyển vị trên ma trận $\mathbf{B} = \mathbf{A}^\top$, ta có phần tử tại vị trí $(k, l)$ của $(\mathbf{A}^\top)^\top$ thỏa mãn:

$$[(\mathbf{A}^\top)^\top]_{k, l} = [\mathbf{A}^\top]_{l, k} = [\mathbf{A}]_{k, l}, \quad \forall k \in \{1, \dots, m\}, \, l \in \{1, \dots, n\}$$

Do mọi phần tử tương ứng đều đồng nhất và hai ma trận có cùng kích thước $m \times n$, ta có điều phải chứng minh: $(\mathbf{A}^\top)^\top = \mathbf{A}$.

---

### Bài 2: Transpose and addition

**Đề bài:** Chứng minh rằng phép chuyển vị có tính chất phân phối đối với phép cộng ma trận: $(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$.

**Chứng minh:**

Xét hai ma trận cùng kích thước $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$. Xét phần tử tổng quát tại chỉ số $(i, j)$ của ma trận chuyển vị của tổng $(\mathbf{A} + \mathbf{B})^\top$:

$$[(\mathbf{A} + \mathbf{B})^\top]_{i, j} = [\mathbf{A} + \mathbf{B}]_{j, i} = [\mathbf{A}]_{j, i} + [\mathbf{B}]_{j, i}$$

Mặt khác, xét phần tử tương ứng của tổng hai ma trận chuyển vị $\mathbf{A}^\top + \mathbf{B}^\top$:

$$[\mathbf{A}^\top + \mathbf{B}^\top]_{i, j} = [\mathbf{A}^\top]_{i, j} + [\mathbf{B}^\top]_{i, j} = [\mathbf{A}]_{j, i} + [\mathbf{B}]_{j, i}$$

Từ tính chất giao hoán của phép cộng trên trường số thực, ta suy ra:

$$[(\mathbf{A} + \mathbf{B})^\top]_{i, j} = [\mathbf{A}^\top + \mathbf{B}^\top]_{i, j}, \quad \forall i \in \{1, \dots, n\}, \, j \in \{1, \dots, m\}$$

Vậy phép cộng ma trận giao hoán với phép chuyển vị: $(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$.

---

### Bài 3: Symmetry of A plus its transpose

**Đề bài:** Cho $\mathbf{A} \in \mathbb{R}^{n \times n}$ là một ma trận vuông bất kỳ. Chứng minh rằng ma trận $\mathbf{S} = \mathbf{A} + \mathbf{A}^\top$ luôn là một ma trận đối xứng.

**Chứng minh:**

Một ma trận vuông $\mathbf{S}$ được định nghĩa là đối xứng nếu và chỉ nếu $\mathbf{S}^\top = \mathbf{S}$.

Áp dụng tính chất phân phối của phép chuyển vị đối với phép cộng (Bài 2) và tính chất chuyển vị của chuyển vị (Bài 1):

$$\mathbf{S}^\top = (\mathbf{A} + \mathbf{A}^\top)^\top = \mathbf{A}^\top + (\mathbf{A}^\top)^\top = \mathbf{A}^\top + \mathbf{A}$$

Do phép cộng ma trận có tính chất giao hoán trên từng phần tử:

$$\mathbf{A}^\top + \mathbf{A} = \mathbf{A} + \mathbf{A}^\top = \mathbf{S}$$

Suy ra $\mathbf{S}^\top = \mathbf{S}$. Do đó, $\mathbf{A} + \mathbf{A}^\top$ luôn là ma trận đối xứng $\forall \mathbf{A} \in \mathbb{R}^{n \times n}$.

---

### Bài 4: Length of a three-dimensional tensor

**Đề bài:** Với một tensor ba chiều $\mathbf{X}$ có kích thước $(2, 3, 4)$, hàm `len(X)` trong Python trả về giá trị bao nhiêu?

**Phân tích và Lời giải:**

Trong Python nói chung và thư viện PyTorch nói riêng, hàm dựng sẵn `len()` khi áp dụng lên một tensor đa chiều chỉ trả về kích thước của chiều (trục) đầu tiên (tức `X.shape[0]`):

$$\text{len}(\mathbf{X}) = \text{kích thước của trục 0} = 2$$

Tổng số phần tử vô hướng của tensor được xác định bởi tích các chiều: $2 \times 3 \times 4 = 24$ (truy xuất qua phương thức `X.numel()`).

---

### Bài 5: Meaning of `len` for arbitrary shapes

**Đề bài:** Phân tích ý nghĩa cấu trúc của hàm `len(X)` đối với một tensor có kích thước tùy ý $(d_0, d_1, \dots, d_{k-1})$.

**Phân tích và Lời giải:**

Đối với tensor $k$ chiều ($k \ge 1$), `len(X)` luôn tương ứng với lực lượng của chiều ngoài cùng ($d_0$). Về mặt ngữ nghĩa lập trình, tensor được xem là một chuỗi (sequence) lồng nhau gồm $d_0$ lát cắt tensor con có số chiều là $k-1$. Đối với tensor vô hướng (scalar, $k = 0$), hàm `len()` không xác định và sẽ kích hoạt ngoại lệ `TypeError: len() of a 0-d tensor`.

---

### Bài 6: Dividing by a reduction along axis 1

**Đề bài:** Phân tích kết quả và giải thích lỗi khi thực thi lệnh `A / A.sum(axis=1)` với ma trận $\mathbf{A} \in \mathbb{R}^{2 \times 3}$. Đề xuất giải pháp chuẩn hóa đúng theo hàng.

**Phân tích và Lời giải:**

Khi tính tổng rút gọn dọc theo trục 1 mà không bảo toàn số chiều (`keepdims=False`), vector kết quả $\mathbf{s} = \mathbf{A}.\text{sum}(\text{axis}=1)$ sẽ có kích thước là $(2,)$.

Khi áp dụng phép chia lan truyền:
- Tensor $\mathbf{A}$ có kích thước $(2, 3)$.
- Tensor $\mathbf{s}$ có kích thước $(2,)$.
- Theo quy tắc broadcasting căn chỉnh từ phải sang trái, trục cuối có kích thước $3$ và $2$ không tương thích ($3 \ne 2$ và không có chiều nào bằng 1). Hệ thống sẽ trả về lỗi: `RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1`.

**Giải pháp chuẩn hóa hàng chuẩn tắc:**
Giữ nguyên số chiều rút gọn bằng tham số `keepdims=True` (hoặc `keepdim=True` trong PyTorch):

```python
import torch

A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
row_normalized_A = A / A.sum(dim=1, keepdim=True)
# A.sum(dim=1, keepdim=True) có kích thước (2, 1), cho phép broadcast hợp lệ sang (2, 3)
```

Khi đó, tổng của mỗi hàng sau chuẩn hóa thỏa mãn $\sum_{j} [\text{row\_normalized\_A}]_{i, j} = 1.0$, tạo thành các phân phối xác suất hợp lệ trên từng hàng (nguyên lý của lớp Softmax).

---

### Bài 7: Distance on a Manhattan street grid

**Đề bài:** Xác định khoảng cách di chuyển giữa hai điểm trên lưới quy hoạch đường phố Manhattan. Phân tích mối liên hệ với chuẩn $L_1$.

**Phân tích và Lời giải:**

Xét hai điểm $\mathbf{x}_1 = (a_1, s_1)^\top$ và $\mathbf{x}_2 = (a_2, s_2)^\top$ trên mặt phẳng tọa độ, trong đó các trục tương ứng với các đại lộ (avenues) và đường phố (streets) trực giao nhau.

Do quy tắc giao thông chỉ cho phép di chuyển song song với các trục tọa độ mà không thể đi theo đường chéo trực tiếp, quỹ đạo ngắn nhất giữa hai điểm chính là **khoảng cách Manhattan** (Manhattan distance), tương đương chuẩn $L_1$ của vector sai lệch:

$$d_1(\mathbf{x}_1, \mathbf{x}_2) = \|\mathbf{x}_1 - \mathbf{x}_2\|_1 = |a_1 - a_2| + |s_1 - s_2|$$

Trong học sâu, chuẩn $L_1$ được sử dụng làm hàm mất mát (Mean Absolute Error - MAE) giúp tăng tính bền vững (robustness) trước các điểm ngoại lai (outliers), hoặc làm số hạng điều chuẩn (Lasso Regularization) để tạo tính thưa (sparsity) cho trọng số mô hình.

---

### Bài 8: Shapes after summing tensor axes

**Đề bài:** Xác định kích thước của tensor kết quả khi thực hiện phép tính tổng rút gọn tensor $\mathbf{X} \in \mathbb{R}^{2 \times 3 \times 4}$ lần lượt dọc theo các trục 0, 1 và 2.

**Phân tích và Lời giải:**

Khi thực hiện phép tính tổng rút gọn dọc theo trục $i$ với tham số `keepdims=False`, trục thứ $i$ sẽ bị loại bỏ khỏi cấu trúc kích thước:

| Trục rút gọn | Biểu thức toán học | Kích thước (`keepdims=False`) | Kích thước (`keepdims=True`) |
| :---: | :---: | :---: | :---: |
| **Trục 0** | $[\mathbf{S}_0]_{j, k} = \sum_{i=1}^2 X_{i, j, k}$ | $(3, 4)$ | $(1, 3, 4)$ |
| **Trục 1** | $[\mathbf{S}_1]_{i, k} = \sum_{j=1}^3 X_{i, j, k}$ | $(2, 4)$ | $(2, 1, 4)$ |
| **Trục 2** | $[\mathbf{S}_2]_{i, j} = \sum_{k=1}^4 X_{i, j, k}$ | $(2, 3)$ | $(2, 3, 1)$ |

---

### Bài 9: Norm of a tensor with three or more axes

**Đề bài:** Trình bày định nghĩa và công thức tính chuẩn Euclid (Frobenius norm) mở rộng cho một tensor có số chiều lớn hơn hoặc bằng 3.

**Phân tích và Lời giải:**

Đối với một tensor đa chiều $\mathbf{X} \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_k}$, chuẩn Frobenius mở rộng (thường gọi là chuẩn $L_2$ của tensor) được định nghĩa là căn bậc hai của tổng bình phương toàn bộ các phần tử vô hướng:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i_1=1}^{d_1} \sum_{i_2=1}^{d_2} \dots \sum_{i_k=1}^{d_k} X_{i_1, i_2, \dots, i_k}^2}$$

Về mặt hình học, chuẩn này tương đương với chuẩn $L_2$ của vector phẳng (flattened vector) biểu diễn toàn bộ tensor trong không gian Euclid đa chiều $\mathbb{R}^{\prod_{j=1}^k d_j}$. Phép tính được hiện thực hóa thông qua hàm `torch.linalg.norm(X)`.

---

### Bài 10: Choosing the multiplication order for three matrices

**Đề bài:** So sánh độ phức tạp tính toán (FLOPs) và dung lượng bộ nhớ trung gian giữa hai phương án kết hợp ma trận $(\mathbf{A}\mathbf{B})\mathbf{C}$ và $\mathbf{A}(\mathbf{B}\mathbf{C})$ với kích thước: $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^5}$, $\mathbf{C} \in \mathbb{R}^{2^5 \times 2^{14}}$.

**Phân tích định lượng:**

Phép nhân hai ma trận kích thước $(m \times k)$ và $(k \times n)$ đòi hỏi $2mkn$ phép toán dấu phẩy động (FLOPs, gồm $mkn$ phép nhân và $mkn$ phép cộng) và tạo ra ma trận có kích thước $(m \times n)$.

1. **Phương án 1: Tính $(\mathbf{A}\mathbf{B})\mathbf{C}$**
   - **Bước 1: Tính $\mathbf{T}_1 = \mathbf{A}\mathbf{B}$**
     - Kích thước trung gian: $\mathbf{T}_1 \in \mathbb{R}^{2^{10} \times 2^5}$ ($1.024 \times 32 = 32.768$ phần tử $\approx 131 \text{ KB}$ ở định dạng `float32`).
     - Số phép tính FLOPs: $2 \times 2^{10} \times 2^{16} \times 2^5 = 2^{32} \text{ FLOPs}$.
   - **Bước 2: Tính $\mathbf{T}_1 \mathbf{C}$**
     - Kích thước kết quả: $\mathbb{R}^{2^{10} \times 2^{14}}$ ($2^{24}$ phần tử).
     - Số phép tính FLOPs: $2 \times 2^{10} \times 2^5 \times 2^{14} = 2^{30} \text{ FLOPs}$.
   - **Tổng chi phí:** $2^{32} + 2^{30} = 5 \times 2^{30} \approx 5,37 \times 10^9 \text{ FLOPs}$.

2. **Phương án 2: Tính $\mathbf{A}(\mathbf{B}\mathbf{C})$**
   - **Bước 1: Tính $\mathbf{T}_2 = \mathbf{B}\mathbf{C}$**
     - Kích thước trung gian: $\mathbf{T}_2 \in \mathbb{R}^{2^{16} \times 2^{14}}$ ($2^{30}$ phần tử $\approx 1,07 \times 10^9$ phần tử $\approx \mathbf{4 \text{ GiB}}$ ở định dạng `float32`).
     - Số phép tính FLOPs: $2 \times 2^{16} \times 2^5 \times 2^{14} = 2^{36} \text{ FLOPs}$.
   - **Bước 2: Tính $\mathbf{A} \mathbf{T}_2$**
     - Số phép tính FLOPs: $2 \times 2^{10} \times 2^{16} \times 2^{14} = 2^{41} \text{ FLOPs}$.
   - **Tổng chi phí:** $2^{36} + 2^{41} \approx 2,27 \times 10^{12} \text{ FLOPs}$.

**Kết luận:** Phương án $(\mathbf{A}\mathbf{B})\mathbf{C}$ tiết kiệm hơn xấp xỉ **422 lần về số lượng phép tính** và giải phóng **4 GiB bộ nhớ đệm**, hoàn toàn loại trừ nguy cơ tràn bộ nhớ VRAM. Điều này minh chứng tầm quan trọng của việc tối ưu hóa thứ tự nhân ma trận (Matrix Chain Multiplication) trong các framework biên dịch đồ thị học sâu (như PyTorch TorchDynamo / XLA).

---

### Bài 11: Multiplying by a transposed matrix view

**Đề bài:** So sánh tính tương đương về đại số và cơ chế bộ nhớ khi thực hiện $\mathbf{A}\mathbf{B}$ so với $\mathbf{A}\mathbf{C}^\top$ trong trường hợp $\mathbf{C} = \mathbf{B}^\top$ được tạo dưới dạng một khung nhìn (view).

**Phân tích và Lời giải:**

1. **Về mặt toán học:** Do $(\mathbf{B}^\top)^\top = \mathbf{B}$, ta có $\mathbf{A}\mathbf{C}^\top = \mathbf{A}(\mathbf{B}^\top)^\top = \mathbf{A}\mathbf{B}$. Hai phép toán cho kết quả số học hoàn toàn đồng nhất.
2. **Về cơ chế quản lý bộ nhớ và tính toán:**
   - Trong PyTorch, thao tác $\mathbf{C} = \mathbf{B}.\text{t}()$ không cấp phát thêm bộ nhớ mới mà tạo một đối tượng tensor view chia sẻ chung khối bộ nhớ vật lý (storage data pointer) với $\mathbf{B}$, chỉ thay đổi siêu dữ liệu về bước nhảy (`stride`) và kích thước (`shape`).
   - Ma trận $\mathbf{C}$ sau chuyển vị sẽ trở thành tensor không liền kề trong bộ nhớ (non-contiguous memory layout). Các kernel tính toán tối ưu của BLAS (như cuBLAS GEMM) có các cờ xử lý riêng biệt cho ma trận chuyển vị (`CUBLAS_OP_T`), cho phép thực thi trực tiếp phép tính mà không cần gọi thao tác sao chép tốn kém `B.clone()`.

---

### Bài 12: Stacking matrices and selecting a slice

**Đề bài:** Thực hiện xếp chồng ba ma trận $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$ dọc theo trục thứ 3 và trích xuất lại chính xác ma trận $\mathbf{B}$.

**Phân tích và Hiện thực hóa:**

Thao tác xếp chồng (stacking) mở rộng một chiều mới tại vị trí chỉ định. Với `dim=2`, ba ma trận hai chiều $(100, 200)$ sẽ tạo thành một tensor 3 chiều có kích thước $(100, 200, 3)$.

```python
import torch

# Khởi tạo ma trận ngẫu nhiên
A = torch.randn(100, 200)
B = torch.randn(100, 200)
C = torch.randn(100, 200)

# Xếp chồng dọc theo trục thứ ba (chỉ số trục dim=2)
stacked = torch.stack([A, B, C], dim=2)
assert stacked.shape == (100, 200, 3)

# Khôi phục ma trận B thông qua lát cắt trục 2 tại chỉ mục 1
recovered_B = stacked[:, :, 1]
assert torch.equal(recovered_B, B)
```
