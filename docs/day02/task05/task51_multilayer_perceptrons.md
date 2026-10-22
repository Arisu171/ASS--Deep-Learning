# 5.1. Multilayer Perceptrons: Perceptron Đa lớp và Các Hàm Kích hoạt Phi tuyến

Nguồn tham khảo: [5.1.4. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html#exercises).

---

### Bài 1: Expressive power of deep linear networks

**Đề bài:** Chứng minh rằng việc xếp chồng nhiều tầng tuyến tính không làm tăng khả năng biểu diễn của mạng nơ-ron. Phân tích trường hợp việc xuất hiện một tầng ẩn có chiều hẹp (bottleneck) còn làm suy giảm dung lượng biểu diễn so với một tầng tuyến tính đơn lẻ.

**Chứng minh:**

Xét mạng nơ-ron gồm $L$ tầng tuyến tính liên tiếp không có hàm kích hoạt:

$$\mathbf{h}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1, \quad \mathbf{h}_2 = \mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2, \quad \dots, \quad \hat{\mathbf{y}} = \mathbf{h}_{L-1} \mathbf{W}_L + \mathbf{b}_L$$

Áp dụng phép thế tuần tự:

$$\hat{\mathbf{y}} = \mathbf{x} (\mathbf{W}_1 \mathbf{W}_2 \dots \mathbf{W}_L) + \sum_{l=1}^{L-1} \mathbf{b}_l \left(\prod_{j=l+1}^L \mathbf{W}_j\right) + \mathbf{b}_L = \mathbf{x} \mathbf{W}_{\text{eff}} + \mathbf{b}_{\text{eff}}$$

Trong đó $\mathbf{W}_{\text{eff}} = \prod_{l=1}^L \mathbf{W}_l$ và $\mathbf{b}_{\text{eff}}$ là các biến đổi affine tương đương.
- *Khả năng biểu diễn:* Mạng $L$ tầng thuần tuyến tính hoàn toàn tương đương với một tầng affine duy nhất, do đó không thể biểu diễn các quy luật phi tuyến cơ bản như hàm logic XOR hay mặt cong bậc hai.
- *Suy giảm dung lượng do Nút thắt (Rank Bottleneck):*
  Xét ánh xạ từ $\mathbb{R}^d \to \mathbb{R}^d$. Một tầng tuyến tính đơn lẻ có ma trận $\mathbf{W} \in \mathbb{R}^{d \times d}$ với hạng tối đa bằng $d$.
  Nếu thiết kế hai tầng $\mathbf{W}_1 \in \mathbb{R}^{d \times k}$ và $\mathbf{W}_2 \in \mathbb{R}^{k \times d}$ với tầng ẩn có kích thước $k < d$:
  $$\operatorname{rank}(\mathbf{W}_{\text{eff}}) = \operatorname{rank}(\mathbf{W}_1 \mathbf{W}_2) \le \min(\operatorname{rank}(\mathbf{W}_1), \operatorname{rank}(\mathbf{W}_2)) \le k < d$$
  Ma trận hiệu dụng bị giới hạn hạng nghiêm trọng, không thể biểu diễn được các biến đổi đẳng cấu đầy đủ (chẳng hạn như ma trận đơn vị $\mathbf{I}_d$). Khi đó, kiến trúc sâu lại có dung lượng biểu diễn kém hơn một tầng đơn đầy đủ.

---

### Bài 2: Derivative of pReLU

**Đề bài:** Thiết lập biểu thức đạo hàm và dưới vi phân (subgradient) của hàm kích hoạt PReLU (Parametric Rectified Linear Unit) với tham số góc nghiêng học được $\alpha \in \mathbb{R}$.

**Phân tích và Lời giải:**

Hàm PReLU được định nghĩa:

$$f(x) = \max(0, x) + \alpha \min(0, x) = \begin{cases} 
x & \text{khi } x > 0 \\ 
\alpha x & \text{khi } x \le 0 
\end{cases}$$

1. **Đạo hàm theo biến đầu vào $x$:**
   $$\frac{\partial f}{\partial x} = \begin{cases} 
   1 & \text{khi } x > 0 \\ 
   \alpha & \text{khi } x < 0 
   \end{cases}$$
   Tại điểm kỳ dị $x = 0$, hàm số không khả vi cổ điển nếu $\alpha \ne 1$. Tập dưới vi phân Clarke tại điểm 0 là đoạn lồi: $\partial f(0) = [\min(1, \alpha), \max(1, \alpha)]$. Trong cài đặt framework, giá trị subgradient tại 0 thường được quy ước bằng $\alpha$ hoặc 1.
2. **Đạo hàm theo tham số học $\alpha$:**
   $$\frac{\partial f}{\partial \alpha} = \min(0, x) = \begin{cases} 
   0 & \text{khi } x \ge 0 \\ 
   x & \text{khi } x < 0 
   \end{cases}$$
   Gradient này cho phép cập nhật tham số $\alpha$ đồng thời với các trọng số mạng thông qua thuật toán lan truyền ngược.

---

### Bài 3: Derivative of Swish

**Đề bài:** Tính đạo hàm của hàm kích hoạt Swish (SiLU) $f(x) = x \cdot \sigma(\beta x)$ theo biến đầu vào $x$ và theo tham số độ dốc $\beta$.

**Phân tích và Lời giải:**

Đặt $s = \sigma(\beta x) = \frac{1}{1 + e^{-\beta x}}$. Ta có đạo hàm cơ bản của sigmoid: $\frac{ds}{dx} = \beta s (1 - s)$.

1. **Đạo hàm theo biến đầu vào $x$:**
   Áp dụng quy tắc đạo hàm của tích:
   $$\frac{df}{dx} = \frac{d}{dx}[x \cdot s] = 1 \cdot s + x \cdot \frac{ds}{dx} = \sigma(\beta x) + \beta x \sigma(\beta x) (1 - \sigma(\beta x))$$
   Biểu diễn rút gọn qua chính hàm $f(x)$:
   $$\frac{df}{dx} = \beta f(x) + \sigma(\beta x)(1 - \beta f(x))$$
2. **Đạo hàm theo tham số $\beta$:**
   $$\frac{\partial f}{\partial \beta} = x \cdot \frac{\partial}{\partial \beta}[\sigma(\beta x)] = x \cdot [x \sigma(\beta x)(1 - \sigma(\beta x))] = x^2 \sigma(\beta x)(1 - \sigma(\beta x))$$
   Đạo hàm luôn không âm ($\frac{\partial f}{\partial \beta} \ge 0$), thể hiện tính đơn điệu tăng của đầu ra theo độ nhạy $\beta$.

---

### Bài 4: ReLU networks are continuous piecewise linear

**Đề bài:** Chứng minh rằng một mạng nơ-ron Perceptron đa lớp sử dụng hàm kích hoạt ReLU (hoặc PReLU) xác định một hàm số liên tục từng khúc tuyến tính (Continuous Piecewise Linear - CPWL).

**Chứng minh:**

1. **Tính chất của từng nơ-ron:**
   Mỗi nơ-ron thứ $i$ ở tầng 1 nhận đầu vào affine $z_i(\mathbf{x}) = \mathbf{w}_i^\top \mathbf{x} + b_i$. Hàm kích hoạt ReLU $\sigma(z_i) = \max(0, z_i)$ phân chia không gian đầu vào $\mathbb{R}^d$ thành hai nửa không gian (half-spaces) ngăn cách bởi siêu phẳng $H_i = \{\mathbf{x} \mid \mathbf{w}_i^\top \mathbf{x} + b_i = 0\}$:
   - Khi $\mathbf{w}_i^\top \mathbf{x} + b_i > 0$: $a_i(\mathbf{x}) = \mathbf{w}_i^\top \mathbf{x} + b_i$ (hàm tuyến tính affine).
   - Khi $\mathbf{w}_i^\top \mathbf{x} + b_i \le 0$: $a_i(\mathbf{x}) = 0$ (hàm hằng, là một dạng affine suy biến).
2. **Sự kết hợp đa tầng:**
   Sự giao nhau của các siêu phẳng từ tất cả các tầng ẩn phân chia không gian $\mathbb{R}^d$ thành một tập hợp các đa diện lồi liên thông (convex polyhedra).
   - Bên trong mỗi đa diện lồi, trạng thái kích hoạt của mọi nơ-ron (bật hoặc tắt) là cố định. Toàn bộ mạng tương đương với một chuỗi các phép biến đổi ma trận affine, do đó là một hàm affine cục bộ.
   - Do hàm ReLU là hàm liên tục khắp nơi và phép hợp của các hàm liên tục là liên tục, toàn bộ hàm mạng $F(\mathbf{x})$ liên tục trên toàn miền xác định.

**Kết luận:** Mạng MLP-ReLU là một hàm liên tục từng khúc tuyến tính. Số lượng vùng tuyến tính (linear regions) có thể tăng theo cấp số nhân theo độ sâu của mạng ($\mathcal{O}\left(\left(\frac{N}{d}\right)^{Ld}\right)$), minh chứng cho ưu thế biểu diễn của kiến trúc sâu so với kiến trúc nông.

---

### Bài 5: Equivalence of sigmoid and tanh networks

**Đề bài:** Chứng minh hệ thức đồng nhất $\tanh(x) = 2\sigma(2x) - 1$. Chứng minh rằng hai họ mạng nơ-ron MLP sử dụng hàm kích hoạt Sigmoid và Tanh có cùng năng lực biểu diễn khi các tầng tuyến tính có tham số độ chệch (bias).

**Chứng minh:**

1. **Hệ thức giải tích:**
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2e^x - (e^x + e^{-x})}{e^x + e^{-x}} = \frac{2e^{2x}}{e^{2x} + 1} - 1 = \frac{2}{1 + e^{-2x}} - 1 = 2\sigma(2x) - 1$$
   $$\iff \sigma(x) = \frac{1}{2}\tanh\left(\frac{x}{2}\right) + \frac{1}{2}$$
2. **Tính tương đương về không gian giả thuyết:**
   Xét một tầng ẩn trong mạng Sigmoid:
   $$\mathbf{h}_{\sigma} = \sigma(\mathbf{x} \mathbf{W} + \mathbf{b})$$
   Biểu diễn qua hàm Tanh:
   $$\mathbf{h}_{\sigma} = \frac{1}{2}\tanh\left(\frac{\mathbf{x} \mathbf{W} + \mathbf{b}}{2}\right) + \frac{1}{2} = \frac{1}{2}\tanh\left(\mathbf{x} \mathbf{W}' + \mathbf{b}'\right) + \frac{1}{2} \mathbf{1}^\top$$
   Trong đó $\mathbf{W}' = \frac{1}{2}\mathbf{W}$ và $\mathbf{b}' = \frac{1}{2}\mathbf{b}$.
   Khi đưa $\mathbf{h}_{\sigma}$ vào tầng kế tiếp $\mathbf{z}_{\text{next}} = \mathbf{h}_{\sigma} \mathbf{W}_{\text{next}} + \mathbf{b}_{\text{next}}$:
   $$\mathbf{z}_{\text{next}} = \left[\frac{1}{2}\tanh\left(\mathbf{x} \mathbf{W}' + \mathbf{b}'\right) + \frac{1}{2} \mathbf{1}^\top\right] \mathbf{W}_{\text{next}} + \mathbf{b}_{\text{next}} = \tanh\left(\mathbf{x} \mathbf{W}' + \mathbf{b}'\right) \left(\frac{1}{2}\mathbf{W}_{\text{next}}\right) + \left(\mathbf{b}_{\text{next}} + \frac{1}{2} \mathbf{1}^\top \mathbf{W}_{\text{next}}\right)$$
   Số hạng độ lệch $\frac{1}{2} \mathbf{1}^\top \mathbf{W}_{\text{next}}$ được hấp thụ hoàn toàn vào vector bias mới $\mathbf{b}'_{\text{next}}$. Do đó, mọi hàm số biểu diễn được bằng mạng Sigmoid đều biểu diễn được chính xác bằng mạng Tanh có cùng số nơ-ron và ngược lại.

---

### Bài 6: Minibatch-dependent nonlinearities

**Đề bài:** Phân tích các hệ lụy và hạn chế khi đưa các toán tử phi tuyến phụ thuộc vào toàn bộ mini-batch (như Batch Normalization) vào kiến trúc mạng sâu.

**Phân tích và Lời giải:**

Toán tử Batch Normalization chuẩn hóa kích hoạt dựa trên trung bình và phương sai của lô dữ liệu hiện tại: $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$.
1. **Phá vỡ tính độc lập giữa các mẫu:** Đầu ra của mẫu $\mathbf{x}_i$ bị ràng buộc phi tuyến với toàn bộ các mẫu khác trong cùng mini-batch $\mathcal{B}$. Điều này làm cho dự đoán của một mẫu bị thay đổi nếu thành phần các mẫu đi kèm trong batch bị thay đổi.
2. **Sự không nhất quán giữa Huấn luyện và Suy luận (Train-Inference Discrepancy):** Trong pha huấn luyện, mô hình dùng thống kê của batch ($\mu_{\mathcal{B}}, \sigma_{\mathcal{B}}$); trong pha suy luận (Inference/Serving), mô hình phải dùng thống kê tích lũy động (running moving averages). Nếu phân phối dữ liệu thực tế biến động hoặc kích thước batch suy luận bằng 1, độ lệch này gây suy giảm độ chính xác.
3. **Thất bại ở quy mô Batch nhỏ:** Khi kích thước batch $B \le 4$, ước lượng phương sai mẫu có phương sai cực lớn, gây mất ổn định huấn luyện.
- *Giải pháp hiện đại:* Chuyển đổi sang các phương pháp chuẩn hóa độc lập với batch như **Layer Normalization** (chuẩn hóa trong Transformers), **Group Normalization**, hoặc **RMSNorm**.

---

### Bài 7: Vanishing gradients with sigmoid

**Đề bài:** Chứng minh hiện tượng biến mất gradient (Vanishing Gradient Problem) khi xếp chồng nhiều tầng ẩn sử dụng hàm kích hoạt Sigmoid thông qua cận giải tích của đạo hàm.

**Chứng minh:**

Đạo hàm của hàm sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ thỏa mãn:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Do $\sigma(z) \in (0, 1)$, khảo sát cực trị của hàm $g(u) = u(1 - u)$ trên $(0, 1)$:

$$g'(u) = 1 - 2u = 0 \iff u = \frac{1}{2} \implies g_{\max} = \frac{1}{2}\left(1 - \frac{1}{2}\right) = \frac{1}{4}$$

$$\implies 0 < \sigma'(z) \le \frac{1}{4} = 0,25, \quad \forall z \in \mathbb{R}$$

Xét mạng sâu $L$ tầng. Theo quy tắc đạo hàm chuỗi, gradient của hàm mất mát đối với kích hoạt ở tầng đầu tiên $\mathbf{h}_1$ chứa tích của $L - 1$ ma trận Jacobian của hàm kích hoạt:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{l=1}^{L-1} \left( \mathbf{W}_{l+1}^\top \operatorname{diag}(\sigma'(\mathbf{z}_l)) \right)$$

Chuẩn của số hạng co bóp qua mỗi tầng bị chặn trên bởi:

$$\|\operatorname{diag}(\sigma'(\mathbf{z}))\|_2 \le \max_i |\sigma'(z_i)| \le \frac{1}{4}$$

Nếu ma trận trọng số $\|\mathbf{W}\|_2 \approx 1$, gradient lan truyền ngược về tầng 1 sẽ bị suy giảm theo cấp số nhân:

$$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1}\right\| \le \left(\frac{1}{4}\right)^{L-1} \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}\right\|$$

Với mạng chỉ 10 tầng ($L = 10$), hệ số suy giảm đạt $(\frac{1}{4})^9 \approx 3,81 \times 10^{-6}$. Các tầng đầu tiên gần như không nhận được tín hiệu gradient để cập nhật trọng số, khiến mạng không thể học được biểu diễn đặc trưng. Đây là lý do hàm kích hoạt **ReLU** ($\sigma'(z) = 1$ với $z > 0$) trở thành giải pháp đột phá thay thế Sigmoid trong học sâu hiện đại.
