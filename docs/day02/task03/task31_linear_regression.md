# 3.1. Linear Regression: Hồi quy Tuyến tính

Nguồn tham khảo: [3.1.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/linear-regression.html#exercises).

---

### Bài 1: Least squares and absolute error for a constant

**Đề bài:** Xác định hằng số $b \in \mathbb{R}$ tối thiểu hóa tổng bình phương sai số $\sum_{i=1}^n (x_i - b)^2$ và giải thích mối liên hệ với phân phối chuẩn thông qua nguyên lý hợp lý cực đại. Xác định nghiệm tối ưu khi hàm mục tiêu là tổng sai số tuyệt đối $\sum_{i=1}^n |x_i - b|$.

**Phân tích và Chứng minh:**

1. **Tổng bình phương sai số ($L_2$ Loss):**
   Hàm mục tiêu: $\mathcal{L}_2(b) = \sum_{i=1}^n (x_i - b)^2$.
   Lấy đạo hàm bậc nhất theo $b$ và giải phương trình điểm dừng:
   $$\frac{d\mathcal{L}_2}{db} = -2 \sum_{i=1}^n (x_i - b) = 0 \iff n b = \sum_{i=1}^n x_i \iff \hat{b}_{\text{LS}} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}$$
   Do $\frac{d^2\mathcal{L}_2}{db^2} = 2n > 0$, nghiệm duy nhất là **trung bình mẫu** (sample mean).
   - *Liên hệ với ước lượng hợp lý cực đại (MLE):* Giả sử mô hình sinh dữ liệu $x_i = b + \epsilon_i$ với nhiễu Gauss độc lập $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Hàm log-hợp lý (Log-Likelihood):
     $$\ln p(\mathbf{x} \mid b, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - b)^2$$
     Tối đa hóa log-hợp lý tương đương trực tiếp với tối thiểu hóa tổng bình phương sai số. Do đó, trung bình mẫu chính là ước lượng hợp lý cực đại của tham số vị trí phân phối chuẩn.

2. **Tổng sai số tuyệt đối ($L_1$ Loss):**
   Hàm mục tiêu: $\mathcal{L}_1(b) = \sum_{i=1}^n |x_i - b|$.
   Hàm số khả vi từng khúc, dưới vi phân (subgradient) theo $b$:
   $$\partial \mathcal{L}_1(b) = - \sum_{i=1}^n \operatorname{sign}(x_i - b) = 0 \iff \sum_{i: x_i > b} 1 = \sum_{i: x_i < b} 1$$
   Điểm dừng đạt được khi số lượng điểm dữ liệu nằm bên phải $b$ cân bằng với số lượng điểm nằm bên trái $b$. Nghiệm tối ưu chính là **trung vị mẫu** (sample median). Nếu cỡ mẫu $n$ chẵn, mọi giá trị $b$ trong khoảng đóng giữa hai phần tử chính giữa $[x_{(n/2)}, x_{(n/2+1)}]$ đều đạt cực tiểu toàn cục.
   - *Bản chất xác suất:* $L_1$ loss tương ứng với ước lượng MLE khi mô hình chịu nhiễu Laplace (nhiễu hai phía phân rã hàm mũ).

---

### Bài 2: Affine functions as linear functions with an extra coordinate

**Đề bài:** Chứng minh rằng mọi hàm affine $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{w} + b$ trên không gian $\mathbb{R}^d$ có thể được biểu diễn đồng nhất dưới dạng một hàm tuyến tính thuần túy trên không gian mở rộng $\mathbb{R}^{d+1}$.

**Chứng minh:**

Định nghĩa vector đặc trưng mở rộng (augmented feature vector) $\tilde{\mathbf{x}} \in \mathbb{R}^{d+1}$ và vector tham số mở rộng $\tilde{\mathbf{w}} \in \mathbb{R}^{d+1}$:

$$\tilde{\mathbf{x}} = \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix}, \quad \tilde{\mathbf{w}} = \begin{bmatrix} \mathbf{w} \\ b \end{bmatrix}$$

Tích vô hướng giữa hai vector mở rộng:

$$\tilde{\mathbf{x}}^\top \tilde{\mathbf{w}} = \begin{bmatrix} \mathbf{x}^\top & 1 \end{bmatrix} \begin{bmatrix} \mathbf{w} \\ b \end{bmatrix} = \mathbf{x}^\top \mathbf{w} + 1 \cdot b = \mathbf{x}^\top \mathbf{w} + b = f(\mathbf{x})$$

*Ý nghĩa kỹ thuật:* Biểu diễn này cho phép hấp thụ hoàn toàn hệ số độ chệch (bias term) vào ma trận trọng số, đơn giản hóa các phép toán đại số tuyến tính trong cài đặt GPU và suy dẫn lý thuyết.

---

### Bài 3: Representing quadratic functions with a network

**Đề bài:** Trình bày phương pháp biểu diễn hàm toàn phương $f(\mathbf{x}) = c + \sum_{i=1}^d w_i x_i + \sum_{i \le j} w_{ij} x_i x_j$ bằng kiến trúc mạng nơ-ron.

**Phân tích và Lời giải:**

1. **Biểu diễn chính xác thông qua ánh xạ mở rộng đặc trưng (Feature Mapping):**
   Định nghĩa không gian đặc trưng đa thức bậc hai:
   $$\boldsymbol{\phi}(\mathbf{x}) = \left[ 1, \, x_1, \dots, x_d, \, x_1^2, \, x_1 x_2, \dots, x_d^2 \right]^\top \in \mathbb{R}^{1 + d + \frac{d(d+1)}{2}}$$
   Khi đó hàm toàn phương trở thành một hàm tuyến tính trong không gian đặc trưng mới: $f(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\theta}$.
2. **Biểu diễn qua mạng nơ-ron sâu:**
   - Nếu mạng sử dụng hàm kích hoạt ReLU ($\sigma(z) = \max(0, z)$), do bản chất ReLU là hàm tuyến tính từng khúc (piecewise linear), mạng nơ-ron sâu chiều rộng hữu hạn **không thể biểu diễn chính xác** một hàm bậc hai liên tục trên toàn không gian không bị chặn $\mathbb{R}^d$.
   - Mạng chỉ có thể **xấp xỉ tùy ý** (theo Định lý xấp xỉ phổ quát) hàm bậc hai trên một tập compact hữu hạn bằng cách ghép nối vô số đoạn thẳng nhỏ.
   - Để biểu diễn chính xác trong kiến trúc mạng, cần sử dụng các tầng tương tác tích (Bilinear Layers / Multiplicative Blocks) hoặc hàm kích hoạt đa thức bậc hai $\sigma(z) = z^2$.

---

### Bài 4: Rank-deficient design matrix

**Đề bài:** Phân tích hiện tượng ma trận thiết kế $\mathbf{X} \in \mathbb{R}^{n \times d}$ không đạt hạng đầy đủ (rank-deficient, $\operatorname{rank}(\mathbf{X}) < d$) và các hệ quả trong bài toán hồi quy bình phương tối thiểu thông thường (OLS).

**Phân tích và Lời giải:**

Khi các cột của $\mathbf{X}$ phụ thuộc tuyến tính (hiện tượng đa cộng tuyến hoàn hảo - multicollinearity):
1. Ma trận Gram $\mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{d \times d}$ là ma trận suy biến (singular, $\det(\mathbf{X}^\top \mathbf{X}) = 0$).
2. Phương trình chuẩn tắc (Normal Equations) $\mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}$ có vô số nghiệm thỏa mãn cùng giá trị loss huấn luyện tối thiểu. Không gian nghiệm là một affine subspace $\mathbf{w}^* + \operatorname{Null}(\mathbf{X})$.
3. **Các giải pháp xử lý chuẩn mực:**
   - **Nghịch đảo giả Moore–Penrose ($\mathbf{X}^+$):** Chọn nghiệm có chuẩn $L_2$ nhỏ nhất: $\mathbf{w}_{\text{min}} = \mathbf{X}^+ \mathbf{y}$.
   - **Điều chuẩn Ridge / Tikhonov ($L_2$ Regularization):** Bổ sung số hạng $\lambda \mathbf{I}$ với $\lambda > 0$:
     $$\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$
     Do $\mathbf{X}^\top \mathbf{X}$ bán xác định dương, $\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}$ luôn xác định dương và khả nghịch tuyệt đối, đảm bảo tính ổn định số học.

---

### Bài 5: Likelihood and loss for Laplace noise

**Đề bài:** Giả định sai số mô hình hồi quy tuân theo phân phối Laplace với hàm mật độ $p(\epsilon) = \frac{1}{2b} \exp\left(-\frac{|\epsilon|}{b}\right)$. Thiết lập hàm mất mát Negative Log-Likelihood (NLL).

**Chứng minh:**

Xét mẫu dữ liệu quan sát độc lập $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ với $y_i = \mathbf{x}_i^\top \mathbf{w} + \epsilon_i$, $\epsilon_i \sim \text{Laplace}(0, b)$.
Hàm hợp lý toàn cục:

$$\mathcal{L}(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \prod_{i=1}^n \frac{1}{2b} \exp\left(-\frac{|y_i - \mathbf{x}_i^\top \mathbf{w}|}{b}\right)$$

Lấy logarit tự nhiên và nhân với âm:

$$\text{NLL}(\mathbf{w}) = - \ln \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n \left[ \ln(2b) + \frac{|y_i - \mathbf{x}_i^\top \mathbf{w}|}{b} \right] = n \ln(2b) + \frac{1}{b} \sum_{i=1}^n |y_i - \mathbf{x}_i^\top \mathbf{w}|$$

Loại bỏ hằng số và thừa số tỷ lệ không phụ thuộc vào $\mathbf{w}$, bài toán tối ưu hóa tương đương tối thiểu hóa hàm mất mát khoảng cách tuyệt đối $L_1$ (Mean Absolute Error):

$$\min_{\mathbf{w}} \sum_{i=1}^n |y_i - \mathbf{x}_i^\top \mathbf{w}|$$

*Kết luận:* Mất mát $L_1$ chính là ước lượng hợp lý cực đại khi dữ liệu chịu nhiễu Laplace, mang lại khả năng kháng điểm ngoại lai (outliers) vượt trội so với phân phối Gauss.

---

### Bài 6: Composition of two linear layers

**Đề bài:** Chứng minh rằng sự xếp chồng liên tiếp của hai tầng tuyến tính không kèm hàm kích hoạt phi tuyến không mở rộng dung lượng biểu diễn của mô hình.

**Chứng minh:**

Xét mạng hai tầng: tầng 1 có $\mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$; tầng 2 có $\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$.
Thay thế trực tiếp $\mathbf{h}$ vào phương trình tầng 2:

$$\mathbf{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2)$$

Đặt $\mathbf{W}_{\text{eff}} = \mathbf{W}_2 \mathbf{W}_1$ và $\mathbf{b}_{\text{eff}} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$, ta thu được:

$$\mathbf{y} = \mathbf{W}_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}$$

Ánh xạ thu được hoàn toàn là một hàm affine đơn tầng. Nếu không có các hàm kích hoạt phi tuyến tính (non-linear activation functions) xen giữa các tầng, mạng nơ-ron dù sâu bao nhiêu tầng cũng chỉ biểu diễn được cùng một lớp hàm tuyến tính, triệt tiêu hoàn toàn ưu thế của học sâu.

---

### Bài 7: Why Gaussian additive noise is unsuitable for prices

**Đề bài:** Phân tích những sai lệch lý thuyết và thực tiễn khi giả định giá tài sản (bất động sản, chứng khoán) chịu nhiễu cộng Gauss. Đề xuất mô hình Log-Normal và các lưu ý giao dịch đối với cổ phiếu giá thấp (penny stocks).

**Phân tích và Lời giải:**

1. **Hạn chế của giả định nhiễu cộng Gauss $P = \hat{P} + \epsilon$ ($\epsilon \sim \mathcal{N}(0, \sigma^2)$):**
   - Phân phối Gauss có giá trị hỗ trợ trên toàn trục số thực $(-\infty, +\infty)$, dẫn đến xác suất khác không gán cho các giá trị giá âm ($\mathbb{P}(P < 0) > 0$), vi phạm nguyên lý kinh tế về trách nhiệm hữu hạn.
   - Biến động giá tài sản trong thực tế thường mang tính tương đối (tỷ lệ phần trăm) và phân phối lợi suất có hiện tượng đuôi dày (fat tails / kurtosis cao) cùng độ lệch dương (skewness).
2. **Mô hình Hồi quy Logarit (Log-Normal Regression):**
   - Mô hình hóa $\ln P = \mathbf{x}^\top \mathbf{w} + \epsilon \implies P = \exp(\mathbf{x}^\top \mathbf{w}) \exp(\epsilon) > 0$, đảm bảo giá dự đoán luôn dương.
   - *Hiệu chỉnh kỳ vọng (Jensen's Inequality):* Lưu ý rằng $\mathbb{E}[P] = \mathbb{E}[\exp(\ln P)] = \exp\left(\mathbf{x}^\top \mathbf{w} + \frac{\sigma^2}{2}\right) > \exp(\mathbf{x}^\top \mathbf{w})$. Giá trị $\exp(\mathbf{x}^\top \mathbf{w})$ là trung vị (median) của phân phối giá thực tế chứ không phải kỳ vọng toán (mean).
3. **Đặc thù của cổ phiếu giá thấp (Penny Stocks):**
   - Bị chi phối bởi bước giá tối thiểu (minimum tick size) và chênh lệch giá mua-bán (bid-ask spread) lớn. Một bước nhảy giá nhỏ tạo ra biến động phần trăm khổng lồ, khiến mô hình liên tục bị phá vỡ và cần các mô hình vi cấu trúc thị trường rời rạc.

---

### Bài 8: Poisson model for sales counts

**Đề bài:** Chứng minh kỳ vọng của phân phối Poisson bằng tham số $\lambda$. Thiết lập hàm mất mát Negative Log-Likelihood cho mô hình hồi quy Poisson với hàm liên kết log.

**Chứng minh:**

Biến ngẫu nhiên đếm $Y \in \{0, 1, 2, \dots\}$ tuân theo phân phối Poisson:

$$\mathbb{P}(Y = y \mid \lambda) = \frac{\lambda^y e^{-\lambda}}{y!}, \quad \lambda > 0$$

1. **Kỳ vọng toán:**
   $$\mathbb{E}[Y] = \sum_{y=0}^\infty y \frac{\lambda^y e^{-\lambda}}{y!} = \sum_{y=1}^\infty \frac{\lambda \cdot \lambda^{y-1} e^{-\lambda}}{(y-1)!} = \lambda e^{-\lambda} \sum_{k=0}^\infty \frac{\lambda^k}{k!} = \lambda e^{-\lambda} e^\lambda = \lambda$$

2. **Hàm mất mát Negative Log-Likelihood (Poisson Loss):**
   Với quan sát $(y_i, \mathbf{x}_i)$, đặt hàm liên kết chuẩn (canonical log-link): $\eta_i = \ln \lambda_i = \mathbf{x}_i^\top \mathbf{w} \implies \lambda_i = e^{\eta_i} = \exp(\mathbf{x}_i^\top \mathbf{w}) > 0$.
   Log-likelihood của một mẫu:
   $$\ln \mathbb{P}(y_i \mid \eta_i) = y_i \ln \lambda_i - \lambda_i - \ln(y_i!) = y_i \eta_i - e^{\eta_i} - \ln(y_i!)$$
   Negative Log-Likelihood (bỏ hằng số $\ln(y_i!)$ không phụ thuộc tham số):
   $$\ell(\mathbf{w}) = \sum_{i=1}^n \left[ \exp(\mathbf{x}_i^\top \mathbf{w}) - y_i (\mathbf{x}_i^\top \mathbf{w}) \right]$$
   Hàm mất mát này là hàm lồi đối với $\mathbf{w}$, đảm bảo tối ưu hóa hội tụ về cực tiểu toàn cục duy nhất.
