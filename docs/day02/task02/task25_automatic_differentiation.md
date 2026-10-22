# 2. Preliminaries: Dive into Deep Learning 1.0.3

## 2.5. Automatic Differentiation: Tự động vi phân trong Học sâu

Nguồn tham khảo: [2.5.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_preliminaries/autograd.html#exercises).

---

### Bài 1: Cost of second derivatives

**Đề bài:** Phân tích bản chất toán học và độ phức tạp tính toán/bộ nhớ của đạo hàm cấp hai (Hessian Matrix) so với đạo hàm cấp một (Gradient Vector) trong các mạng nơ-ron sâu.

**Phân tích và Lời giải:**

Xét hàm mục tiêu mất mát $\ell(\boldsymbol{\theta}): \mathbb{R}^d \to \mathbb{R}$ với vector tham số $\boldsymbol{\theta} \in \mathbb{R}^d$.
- **Đạo hàm cấp một (Gradient):** $\nabla_{\boldsymbol{\theta}} \ell \in \mathbb{R}^d$ gồm $d$ đạo hàm riêng bậc nhất. Độ phức tạp tính toán và bộ nhớ lưu trữ tăng tuyến tính $\mathcal{O}(d)$.
- **Đạo hàm cấp hai (Hessian):** $\mathbf{H} = \nabla_{\boldsymbol{\theta}}^2 \ell \in \mathbb{R}^{d \times d}$ gồm $d^2$ đạo hàm riêng cấp hai liên hiệp:
  $$H_{i, j} = \frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j}$$
  - **Độ phức tạp không gian (Bộ nhớ):** Cần $\mathcal{O}(d^2)$ ô nhớ. Khi mô hình có $d \sim 10^7 - 10^{11}$ tham số, ma trận Hessian đòi hỏi dung lượng Petabytes đến Exabytes, vượt quá giới hạn phần cứng thực tế.
  - **Độ phức tạp thời gian:** Việc tính ma trận Hessian đầy đủ đòi hỏi $d$ lượt lan truyền ngược hoặc xây dựng đồ thị tính toán bậc hai phức tạp.
  - *Giải pháp trong Học sâu:* Sử dụng các phương pháp xấp xỉ Hessian cấp thấp (Quasi-Newton như BFGS/L-BFGS) hoặc kỹ thuật tích Hessian-Vector (Hessian-Vector Products - HVP: $\mathbf{H}\mathbf{v} = \nabla(\nabla \ell^\top \mathbf{v})$) chỉ tốn $\mathcal{O}(d)$ thông qua hai lần vi phân tự động mà không cần hiện thực hóa tường minh ma trận $\mathbf{H}$.

---

### Bài 2: Repeating backpropagation

**Đề bài:** Phân tích hành vi nội tại của PyTorch Autograd khi gọi liên tiếp nhiều lần phương thức `.backward()` trên cùng một đồ thị tính toán. Trình bày cơ chế tích lũy gradient (`gradient accumulation`).

**Phân tích và Lời giải:**

1. **Cơ chế thu dọn đồ thị tính toán (Graph Lifecycle):**
   - Theo thiết kế mặc định, sau khi thực thi `loss.backward()`, PyTorch sẽ giải phóng toàn bộ các bộ đệm kích hoạt trung gian (intermediate activation buffers) cấu thành đồ thị tính toán vô hướng có hướng (DAG) để tối ưu hóa bộ nhớ RAM/VRAM.
   - Nếu gọi `.backward()` lần thứ hai mà không tái tạo forward pass, hệ thống sẽ kích hoạt ngoại lệ: `RuntimeError: Trying to backward through the graph a second time...`.
2. **Tham số `retain_graph=True`:**
   - Để giữ lại đồ thị phục vụ tính toán gradient bậc cao hoặc nhiều hàm mất mát nhánh, cần thiết lập `retain_graph=True`.
3. **Cơ chế tích lũy gradient:**
   - Gradient tính toán ở mỗi lần gọi `.backward()` sẽ **được cộng dồn** vào thuộc tính `.grad` của các tensor lá (`leaf tensors`): $\mathbf{g}_{\text{acc}} \leftarrow \mathbf{g}_{\text{acc}} + \nabla_{\boldsymbol{\theta}} \ell$.
   - Do đó, trong vòng lặp tối ưu hóa tiêu chuẩn, lập trình viên bắt buộc phải gọi `optimizer.zero_grad()` để làm sạch bộ tích lũy trước mỗi bước cập nhật trọng số. Kỹ thuật này cũng được khai thác chủ động để mô phỏng kích thước batch lớn (Gradient Accumulation over micro-batches) khi VRAM hạn chế.

---

### Bài 3: Vector or matrix input in the control-flow example

**Đề bài:** Khi đầu ra của mô hình là một vector hoặc ma trận (non-scalar output), cơ chế tự động vi phân yêu cầu thành phần bổ trợ gì để thực thi lan truyền ngược?

**Phân tích và Lời giải:**

Đối với hàm ánh xạ vector-sang-vector $\mathbf{y} = f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}^m$, đạo hàm của $f$ là một ma trận Jacobian $\mathbf{J} \in \mathbb{R}^{m \times n}$, trong đó $J_{i, j} = \frac{\partial y_i}{\partial x_j}$.

Engine tự động vi phân ngược (Reverse-mode autodiff) không thiết kế để tính trực tiếp ma trận Jacobian $\mathbf{J}$ vì chi phí bộ nhớ quá lớn. Thay vào đó, engine tính **tích Vector-Jacobian** (Vector-Jacobian Product - VJP):

$$\mathbf{v}^\top \mathbf{J} = \sum_{i=1}^m v_i \nabla_{\mathbf{x}} y_i \in \mathbb{R}^{1 \times n}$$

Trong đó vector $\mathbf{v} \in \mathbb{R}^m$ được gọi là vector gradient hạt giống (seed gradient / cotangent vector), được truyền vào đối số `gradient` của hàm `.backward()`: `y.backward(gradient=v)`. Nếu muốn tính gradient của một đại lượng vô hướng tổng thể, vector hạt giống thường là vector đơn vị hoặc vector toàn một $\mathbf{1}$.

---

### Bài 4: Plotting sin and its autodiff derivative

**Đề bài:** Trực quan hóa hàm số $y = \sin(x)$ cùng gradient của nó tính bằng tự động vi phân (PyTorch Autograd) trên đoạn $[0, 2\pi]$ mà không sử dụng công thức giải tích đạo hàm $\cos(x)$.

**Hiện thực hóa:**

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(0, 2 * torch.pi, 200, requires_grad=True)
y = torch.sin(x)

# Do y là vector, áp dụng VJP với hạt giống toàn 1 (tương đương backward của y.sum())
y.backward(torch.ones_like(y))

# x.grad chứa gradient số học bằng autograd
print("Sai số cực đại giữa autograd và cos giải tích:", torch.max(torch.abs(x.grad - torch.cos(x))).item())
```

Kết quả sai lệch cực đại chỉ ở cấp độ chính xác máy số thực ($\approx 10^{-7}$). Đồ thị tương ứng được biểu diễn ở bảng trực quan hóa bên dưới.

---

### Bài 5: Dependency graph for a composite function

**Đề bài:** Xây dựng đồ thị tính toán có hướng (Directed Acyclic Graph - DAG) cho hàm số hợp: $f(x) = \ln(x^2) \cdot \sin(x) + x^{-1}$ trên miền xác định $x \ne 0$.

**Cấu trúc Đồ thị Tính toán (Computational DAG):**

```text
       ┌───> [ (·)² ] ───> u₁ ───> [ ln(·) ] ───> u ───┐
       │                                               │
(x) ───┼───> [ sin(·) ] ────────────────────────> v ───┴───> [ (·) × (·) ] ───> p ───┐
       │                                                                              │
       └───> [ (·)⁻¹ ] ─────────────────────────> w ──────────────────────────────────┴───> [ (+) ] ───> f(x)
```

**Các biến trạng thái trung gian:**
- $u_1 = x^2$
- $u = \ln(u_1) = \ln(x^2)$
- $v = \sin(x)$
- $p = u \cdot v$
- $w = x^{-1}$
- $f = p + w$

---

### Bài 6: Differentiating the composite function

**Đề bài:** Vận dụng quy tắc đạo hàm chuỗi để xác định đạo hàm giải tích $f'(x)$ và biểu diễn dòng lan truyền gradient trên đồ thị.

**Lời giải:**

Dựa vào cấu trúc phân rã ở Bài 5, tính vi phân từng nút:
- $\frac{du}{dx} = \frac{1}{x^2} \cdot 2x = \frac{2}{x}$
- $\frac{dv}{dx} = \cos(x)$
- $\frac{dw}{dx} = -x^{-2} = -\frac{1}{x^2}$

Áp dụng quy tắc đạo hàm chuỗi cho hàm mục tiêu $f = u \cdot v + w$:

$$f'(x) = \frac{\partial f}{\partial u}\frac{du}{dx} + \frac{\partial f}{\partial v}\frac{dv}{dx} + \frac{\partial f}{\partial w}\frac{dw}{dx} = v \cdot \frac{2}{x} + u \cdot \cos(x) + 1 \cdot \left(-\frac{1}{x^2}\right)$$

Thay lại các biến gốc:

$$f'(x) = \frac{2\sin(x)}{x} + \ln(x^2)\cos(x) - \frac{1}{x^2}$$

*Kiểm chứng tại $x = 2$:*
- $f(2) = \ln(4)\sin(2) + \frac{1}{2} \approx 1,386294 \times 0,909297 + 0,5 \approx 1,760554$
- $f'(2) = \frac{2\sin(2)}{2} + \ln(4)\cos(2) - \frac{1}{4} = \sin(2) - \ln(4)\cos(2) - 0,25 \approx 0,082395$

---

### Bài 7: Forward and backward differentiation

**Đề bài:** So sánh cơ chế vận hành của hai chế độ vi phân: Vi phân xuôi (Forward-mode Autodiff) và Vi phân ngược (Reverse-mode Autodiff).

**Phân tích và Lời giải:**

1. **Vi phân chế độ xuôi (Forward Mode - Tangent Linearization):**
   - Lan truyền đồng thời giá trị nguyên thủy (primal value) và tiếp tuyến đạo hàm (tangent/dual number $\dot{v} = \frac{\partial v}{\partial x}$) từ nút đầu vào hướng về nút đầu ra.
   - Thích hợp khi tính đạo hàm của hàm $f: \mathbb{R} \to \mathbb{R}^m$ (số đầu vào ít hơn số đầu ra, $n \ll m$).
   - Ưu điểm: Bộ nhớ $\mathcal{O}(1)$ vì không cần lưu trữ vết tính toán (tape).
2. **Vi phân chế độ ngược (Reverse Mode - Adjoint Sensitivity / Backpropagation):**
   - **Giai đoạn xuôi (Forward pass):** Tính và lưu trữ toàn bộ các giá trị kích hoạt trung gian vào bộ nhớ tape.
   - **Giai đoạn ngược (Backward pass):** Lan truyền vector adjoint (gradient phụ tá $\bar{v} = \frac{\partial \ell}{\partial v}$) từ ngọn đồ thị ngược về các nút gốc.
   - Thích hợp tối ưu cho các bài toán học sâu nơi $f: \mathbb{R}^n \to \mathbb{R}$ (hàm mất mát vô hướng phụ thuộc vào hàng triệu tham số $n \gg 1$).

---

### Bài 8: Choosing forward or backward differentiation

**Đề bài:** Thiết lập tiêu chuẩn lựa chọn thuật toán vi phân tự động dựa trên cấu trúc không gian ánh xạ và ràng buộc phần cứng.

**Tiêu chuẩn lựa chọn kiến trúc:**

| Tiêu chí kỹ thuật | Vi phân chế độ xuôi (Forward Mode) | Vi phân chế độ ngược (Reverse Mode / Backprop) |
| :--- | :--- | :--- |
| **Kích thước không gian** | $f: \mathbb{R}^n \to \mathbb{R}^m$ với $n \ll m$ | $f: \mathbb{R}^n \to \mathbb{R}^m$ với $n \gg m$ (đặc biệt $m = 1$) |
| **Độ phức tạp bộ nhớ** | $\mathcal{O}(1)$ (bộ nhớ hằng số, không lưu activation) | $\mathcal{O}(N)$ (lưu vết toàn bộ đồ thị DAG kích thước $N$) |
| **Đặc trưng tác vụ** | Tính tích Jacobian-Vector ($Jv$), điều khiển tối ưu thời gian thực | Huấn luyện mạng nơ-ron sâu, tính gradient của hàm mất mát |
| **Tối ưu hóa bộ nhớ** | Tự nhiên, không có nguy cơ tràn bộ nhớ | Cần các kỹ thuật hỗ trợ: Gradient Checkpointing, Recomputation |

---

### Đồ thị trực quan hóa kết quả

Biểu đồ thực nghiệm minh họa:
- Đồ thị hàm số và tiếp tuyến giải tích tại $x = 1$.
- So sánh đạo hàm hàm $\sin(x)$ tính bằng PyTorch Autograd và đường cong giải tích chuẩn.

![Biểu đồ bài tập giải tích và tự động vi phân](image2_graph.png)
