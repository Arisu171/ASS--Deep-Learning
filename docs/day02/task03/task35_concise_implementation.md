# 3.5. Concise Implementation of Linear Regression: Cài đặt Tinh gọn Hồi quy Tuyến tính

Nguồn tham khảo: [3.5.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/linear-regression-concise.html#exercises).

---

### Bài 1: Sum loss versus mean loss

**Đề bài:** Phân tích mối tương quan giữa hàm mất mát tính tổng ($\mathcal{L}_{\text{sum}}$) và hàm mất mát tính trung bình ($\mathcal{L}_{\text{mean}}$) đối với một mini-batch kích thước $B$. Thiết lập quy tắc điều chỉnh tốc độ học (Learning Rate $\eta$) tương ứng.

**Phân tích và Lời giải:**

Xét mini-batch $\mathcal{B}$ gồm $B$ mẫu dữ liệu. Hai quy ước tính mất mát được xác định:

$$\mathcal{L}_{\text{sum}}(\boldsymbol{\theta}) = \sum_{i \in \mathcal{B}} \ell_i(\boldsymbol{\theta}), \quad \mathcal{L}_{\text{mean}}(\boldsymbol{\theta}) = \frac{1}{B} \sum_{i \in \mathcal{B}} \ell_i(\boldsymbol{\theta})$$

Lấy gradient theo tham số $\boldsymbol{\theta}$:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{mean}}(\boldsymbol{\theta}) = \frac{1}{B} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{sum}}(\boldsymbol{\theta})$$

Bước cập nhật tham số SGD tương ứng:
- Với loss trung bình: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_{\text{mean}} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{mean}}$
- Với loss tổng: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_{\text{sum}} \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{sum}} = \boldsymbol{\theta}_t - \eta_{\text{sum}} \cdot B \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{mean}}$

Để hai quy trình tối ưu hóa tạo ra cùng một quỹ đạo cập nhật trong không gian tham số, tốc độ học phải tuân theo quy tắc tỷ lệ:

$$\eta_{\text{sum}} = \frac{\eta_{\text{mean}}}{B} \iff \eta_{\text{mean}} = B \cdot \eta_{\text{sum}}$$

*Khuyến nghị kỹ thuật:* Các thư viện học sâu chuẩn mực (PyTorch, TensorFlow) đều thiết lập `reduction='mean'` làm mặc định. Khi chuyển đổi mã nguồn hoặc tái lập bài báo khoa học, cần kiểm tra kỹ quy ước reduction để tránh sai lệch bậc độ lớn của learning rate.

---

### Bài 2: Replace squared loss with Huber loss

**Đề bài:** Cài đặt hàm mất mát Huber loss tùy chỉnh với tham số ngưỡng $\delta > 0$ và đánh giá tính chất toán học của gradient.

**Hiện thực hóa PyTorch:**

```python
import torch

def custom_huber_loss(y_hat: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    r"""
    Cài đặt hàm mất mát Huber có điều chuẩn độ cong:
    L_\delta(e) = 0.5 * e^2 / \delta nếu |e| <= \delta, ngược lại |e| - 0.5 * \delta
    """
    error = torch.abs(y_hat - y)
    quadratic = 0.5 * (error ** 2) / delta
    linear = error - 0.5 * delta
    return torch.where(error <= delta, quadratic, linear)
```

*Phân tích gradient:* Đạo hàm riêng của hàm mất mát này theo $\hat{y}$ bị chặn tuyệt đối trong đoạn $[-1, 1]$ với mọi sai số:
$$\left|\frac{\partial \mathcal{L}_\delta}{\partial \hat{y}}\right| = \min\left(1, \frac{|\hat{y} - y|}{\delta}\right) \le 1$$
Tính chất này mang lại khả năng chống bùng nổ gradient cực kỳ hiệu quả khi tập dữ liệu huấn luyện chứa các giá trị ngoại lai dị thường.

---

### Bài 3: Accessing the weight gradient

**Đề bài:** Trình bày phương thức truy xuất gradient của ma trận trọng số và vector bias trong mô-đun cấp cao `torch.nn.Linear` và cơ chế tích lũy gradient.

**Phân tích và Lời giải:**

Sau khi thực thi `loss.backward()`, đồ thị tính toán tự động lan truyền ngược và lưu trữ các đạo hàm riêng tương ứng vào thuộc tính `.grad` của từng tham số `torch.nn.Parameter`:

```python
# Giả sử mô hình tuyến tính: model = nn.Sequential(nn.Linear(d, 1))
weight_gradient = model[0].weight.grad  # Kích thước: (1, d)
bias_gradient = model[0].bias.grad      # Kích thước: (1,)
```

- **Nguyên lý tích lũy:** Phương thức `.backward()` không ghi đè mà thực hiện phép cộng dồn vào `.grad`. Do đó, lệnh `optimizer.zero_grad()` hoặc `model.zero_grad()` bắt buộc phải được gọi trước mỗi lượt backward mới để tránh hiện tượng cộng dồn gradient từ các mini-batch trước.

---

### Bài 4: Learning rate and number of epochs

**Đề bài:** Đánh giá sự tương tác giữa tốc độ học $\eta$, số lượng epoch và tính chất bề mặt tối ưu hóa (Loss Landscape) trong việc xác định điểm hội tụ.

**Phân tích và Lời giải:**

1. **Bề mặt hàm mất mát lồi (Convex Landscape):** Đối với bài toán hồi quy tuyến tính bình phương tối thiểu, hàm mục tiêu là hàm lồi mạnh với cực tiểu toàn cục duy nhất. Nếu $\eta < \frac{2}{\lambda_{\max}}$, thuật toán chắc chắn hội tụ về điểm cực tiểu này bất kể điểm khởi tạo.
2. **Hiện tượng bão hòa Epoch:** Khi vector tham số đã tiến vào lân cận bán kính sai số $\mathcal{O}(\eta)$ quanh điểm tối ưu toàn cục, việc tiếp tục tăng số lượng epoch chỉ làm dao động quanh nghiệm mà không cải thiện độ chính xác. Để đạt nghiệm chính xác cao hơn, cần kết hợp chiến lược giảm tốc độ học (Learning Rate Decay / Cosine Annealing) hoặc sử dụng tiêu chuẩn dừng sớm (Early Stopping) dựa trên tập kiểm định validation.

---

### Bài 5: Estimation error versus dataset size

**Đề bài:** Thực hiện khảo sát thực nghiệm về tốc độ suy giảm sai số ước lượng tham số $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ và $|\hat{b} - b^*|$ theo quy mô tập dữ liệu $n$, giải thích lý do quy mô mẫu nên tăng theo cấp số nhân (logarithmic scale).

**Phân tích lý thuyết và Thực nghiệm:**

1. **Tốc độ hội tụ tiệm cận trong Thống kê học:**
   Theo định lý tiệm cận chuẩn của ước lượng OLS, khi các giả thiết Gauss-Markov thỏa mãn, ma trận hiệp phương sai của ước lượng tham số thỏa mãn:
   $$\operatorname{Cov}(\hat{\mathbf{w}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1} \approx \frac{\sigma^2}{n} \boldsymbol{\Sigma}_{\mathbf{x}}^{-1}$$
   Do đó, chuẩn sai số kỳ vọng suy giảm theo quy luật căn bậc hai nghịch đảo:
   $$\mathbb{E}\left[\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2\right] = \mathcal{O}(n^{-1/2})$$
2. **Lý do sử dụng thang đo cấp số nhân ($n \in \{5, 10, 20, 50, 100, 500, 1000, 10000\}$):**
   Để quan sát được sự suy giảm sai số theo từng bậc độ lớn (orders of magnitude) trên đồ thị log-log ($\log(\text{Error}) \sim -\frac{1}{2}\log(n)$), khoảng cách giữa các điểm thử nghiệm cần tăng theo cấp số nhân. Việc tăng tuyến tính (ví dụ: $10, 20, 30, \dots$) sẽ làm dày đặc các điểm dữ liệu ở vùng kích thước nhỏ mà không bao quát được hành vi tiệm cận ở vùng dữ liệu lớn.

**Kết quả thực nghiệm mô phỏng (Trung bình qua 12 hạt giống ngẫu nhiên):**
- Tại $n = 5$: Sai số ước lượng trọng số $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2 \approx 1,66 \times 10^{-2}$.
- Tại $n = 10.000$: Sai số suy giảm chính xác về $1,96 \times 10^{-4}$ (thu hẹp xấp xỉ gần hai bậc độ lớn tương ứng với $\sqrt{10.000 / 5} \approx 44,7$ lần).
- Đồ thị bên trái biểu diễn đường cong suy giảm sai số ước lượng thực tế theo quy mô mẫu; đồ thị bên phải minh họa đường cong điều chuẩn Ridge theo tham số $\lambda$.

![Sai số ước lượng theo kích thước dữ liệu và MSE ridge theo lambda](image35_37_experiment.png)
