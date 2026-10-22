# 3.4. Linear Regression Implementation from Scratch: Cài đặt Hồi quy Tuyến tính từ Đầu

Nguồn tham khảo: [3.4.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html#exercises).

---

### Bài 1: Effect of zero and very large weight initialization

**Đề bài:** Khảo sát ảnh hưởng động học của việc khởi tạo trọng số bằng 0 ($\mathbf{w} = \mathbf{0}$) và khởi tạo với phương sai cực lớn ($\sigma^2 = 1000$) đối với mô hình hồi quy tuyến tính đơn tầng.

**Phân tích và Lời giải:**

1. **Khởi tạo trọng số bằng 0 ($\mathbf{w}_0 = \mathbf{0}$):**
   - Khác với mạng nơ-ron sâu nhiều tầng có các nơ-ron ẩn đối xứng (symmetry breaking issue), mô hình hồi quy tuyến tính là một hàm lồi đơn tầng.
   - Gradient ban đầu theo $\mathbf{w}$ là $\nabla_{\mathbf{w}} \mathcal{L} = -\frac{1}{n} \mathbf{X}^\top (\mathbf{y} - b_0 \mathbf{1}) = -\frac{1}{n} \mathbf{X}^\top \mathbf{y}$ (nếu $b_0 = 0$).
   - Nếu $\mathbf{X}^\top \mathbf{y} \ne \mathbf{0}$, gradient khác 0 và thuật toán SGD vẫn có thể kéo trọng số rời khỏi gốc tọa độ về điểm hội tụ. Tuy nhiên, nếu đầu vào và nhãn trực giao hoặc dữ liệu đã được chuẩn hóa mean=0, bước cập nhật ban đầu có thể bị trễ hoặc rơi vào điểm dừng suy biến.
2. **Khởi tạo trọng số với phương sai cực lớn ($\sigma^2 = 1000 \implies \sigma \approx 31,62$):**
   - Trọng số ban đầu có độ lớn rất lớn so với thang đo chuẩn của dữ liệu.
   - Dự đoán $\hat{y} = \mathbf{x}^\top \mathbf{w}$ có độ lệch cực lớn, dẫn đến hàm mất mát MSE ban đầu bùng nổ: $\mathcal{L} \sim \mathcal{O}(\sigma^2 \|\mathbf{x}\|^2)$.
   - Vector gradient $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{B}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$ nhận giá trị khổng lồ, khiến thuật toán SGD bước các bước nhảy vọt (gradient explosion), dẫn đến hiện tượng dao động dữ dội hoặc tràn số học (`NaN` / `Inf`).

---

### Bài 2: Learning Ohm’s law parameters with autodiff

**Đề bài:** Thiết lập bài toán tối ưu hóa tham số điện trở $R$ trong Định luật Ohm ($V = R \cdot I$) sử dụng cơ chế tự động vi phân và ước lượng bình phương tối thiểu.

**Phân tích và Lời giải:**

Xét tập dữ liệu thực nghiệm gồm các cặp đo đạc cường độ dòng điện và điện áp $\{(I_j, V_j)\}_{j=1}^m$.
Hàm mất mát bình phương sai số:

$$\mathcal{L}(R) = \frac{1}{m} \sum_{j=1}^m (V_j - R I_j)^2$$

1. **Gradient tính bằng Autodiff:**
   $$\frac{d\mathcal{L}}{dR} = -\frac{2}{m} \sum_{j=1}^m I_j (V_j - R I_j)$$
2. **Nghiệm giải tích bình phương tối thiểu (OLS không hệ số chặn):**
   $$\frac{d\mathcal{L}}{dR} = 0 \iff \hat{R} = \frac{\sum_{j=1}^m I_j V_j}{\sum_{j=1}^m I_j^2}$$
   Để đảm bảo điều kiện vật lý $R > 0$, trong thực hành học sâu có thể tham số hóa $R = \operatorname{softplus}(\rho) = \ln(1 + e^\rho)$ và tối ưu hóa theo tham số thực vô điều kiện $\rho \in \mathbb{R}$.

---

### Bài 3: Estimating temperature from Planck’s law

**Đề bài:** Áp dụng phương pháp tối ưu hóa dựa trên gradient và hàm mất mát log-scale để ước lượng nhiệt độ vật đen $T$ từ định luật bức xạ phổ Planck.

**Phân tích và Hiện thực hóa:**

Định luật bức xạ Planck mô tả mật độ năng lượng phổ tại bước sóng $\lambda$ ở nhiệt độ tuyệt đối $T$:

$$B(\lambda, T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp\left(\frac{h c}{\lambda k_B T}\right) - 1}$$

Trong đó $h \approx 6,626 \times 10^{-34} \text{ J}\cdot\text{s}$, $c \approx 2,998 \times 10^8 \text{ m/s}$, $k_B \approx 1,381 \times 10^{-23} \text{ J/K}$.
Do $B(\lambda, T)$ biến thiên qua nhiều bậc độ lớn từ dải tử ngoại đến hồng ngoại, hàm mất mát đo lường trong không gian logarit tự nhiên đảm bảo độ ổn định số học và cân bằng đóng góp của mọi bước sóng:

```python
import torch
import torch.nn.functional as F

h = 6.62607015e-34
c = 299792458.0
kB = 1.380649e-23

def estimate_planck_temperature(wavelengths_m: torch.Tensor, observed_spectral_radiance: torch.Tensor) -> float:
    # Tham số hóa T qua softplus để ràng buộc T > 0 tuyệt đối
    raw_T = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw_T], lr=0.05)

    for step in range(2500):
        optimizer.zero_grad()
        T = F.softplus(raw_T) + 1.0  # T > 1 Kelvin
        
        # Sử dụng torch.expm1 để tính ổn định: expm1(u) = exp(u) - 1
        exponent = (h * c) / (wavelengths_m * kB * T)
        predicted = (2 * h * c**2 / wavelengths_m**5) / torch.expm1(exponent)
        
        # Log-space Mean Squared Error
        loss = torch.mean((torch.log(predicted) - torch.log(observed_spectral_radiance))**2)
        loss.backward()
        optimizer.step()
        
    return (F.softplus(raw_T) + 1.0).item()
```

---

### Bài 4: Problems with second derivatives of the loss

**Đề bài:** Phân tích các rào cản tính toán và giải tích khi áp dụng đạo hàm cấp hai (Phương pháp Newton / Hessian) trong tối ưu hóa hàm mất mát mạng nơ-ron.

**Phân tích và Lời giải:**

1. **Chi phí bộ nhớ và tính toán:**
   Ma trận Hessian $\mathbf{H} \in \mathbb{R}^{d \times d}$ đòi hỏi lưu trữ $\mathcal{O}(d^2)$ phần tử và giải hệ phương trình tuyến tính $\mathbf{H} \Delta \boldsymbol{\theta} = -\mathbf{g}$ tốn $\mathcal{O}(d^3)$ phép tính.
2. **Tính không khả vi cấp hai (Non-smoothness):**
   - Với hàm mất mát $L_1$ tuyệt đối hoặc Huber, đạo hàm cấp hai tại các điểm phân ranh bị triệt tiêu hoặc không xác định.
   - Đối với mạng sâu sử dụng hàm kích hoạt ReLU, đạo hàm cấp hai bằng 0 hầu như khắp nơi ($\sigma''(z) = 0, \forall z \ne 0$) và không xác định tại $z = 0$. Khi đó ma trận Hessian suy biến hoàn toàn thành ma trận 0, khiến các phương pháp bậc hai thuần túy không thể hoạt động.

---

### Bài 5: Why reshape labels in the loss

**Đề bài:** Phân tích cạm bẫy lan truyền kích thước (Broadcasting Trap) trong PyTorch khi vector dự đoán $\hat{\mathbf{y}}$ có kích thước $(B, 1)$ và vector nhãn $\mathbf{y}$ có kích thước $(B,)$.

**Phân tích và Lời giải:**

Nếu thực hiện phép trừ trực tiếp `loss = (y_hat - y)**2` mà không đồng nhất số chiều:
- `y_hat` có kích thước $(B, 1)$.
- `y` có kích thước $(B,)$.
- Theo quy tắc broadcasting của PyTorch, `y` được xem như $(1, B)$, và phép trừ sẽ tự động mở rộng cả hai tensor thành ma trận kích thước $(B, B)$!
- Thay vì tính sai số giữa từng cặp mẫu tương ứng ($B$ giá trị sai số đường chéo), ma trận $(B, B)$ chứa toàn bộ các sai số chéo giữa mẫu thứ $i$ và nhãn của mẫu thứ $j$ ($B^2$ phần tử).
- Giá trị mất mát trung bình và gradient tính toán sẽ bị sai lệch hoàn toàn về mặt toán học. Do đó, việc gọi `y.reshape(y_hat.shape)` là bắt buộc.

---

### Bài 6: Learning rate, epochs, and convergence

**Đề bài:** Đánh giá động học hội tụ của SGD theo tốc độ học (Learning Rate $\eta$) và số epoch. Phân tích giới hạn của việc tăng số epoch huấn luyện.

**Phân tích và Lời giải:**

1. **Động học bước học $\eta$:**
   - Trong bài toán tối ưu hóa lồi mạnh với độ trơn $L$ (hồi quy bình phương tối thiểu), điều kiện cần và đủ để thuật toán Gradient Descent hội tụ là $0 < \eta < \frac{2}{L}$, trong đó $L = \lambda_{\max}(\frac{1}{n}\mathbf{X}^\top \mathbf{X})$.
   - Nếu $\eta > \frac{2}{L}$, quỹ đạo tham số phân kỳ theo cấp số nhân.
   - Nếu $\eta$ quá nhỏ, thời gian hội tụ bị kéo dài không cần thiết ($\mathcal{O}(1/\epsilon)$).
2. **Giới hạn của việc tăng Epoch:**
   - Khi mô hình đã đạt đến lân cận nghiệm tối ưu Bayes hoặc cực tiểu trên tập huấn luyện, việc tăng thêm epoch không mang lại cải thiện sai số xấp xỉ.
   - Ngược lại, đối với mô hình có dung lượng tham số lớn hơn kích thước dữ liệu, việc huấn luyện quá nhiều epoch sẽ chuyển dịch mô hình từ trạng thái học quy luật sang trạng thái học thuộc nhiễu quan sát (overfitting), làm gia tăng khoảng cách tổng quát hóa (generalization gap).

---

### Bài 7: Incomplete final minibatch

**Đề bài:** Đánh giá định lượng sai số gradient khi duy trì batch cuối có kích thước nhỏ hơn batch tiêu chuẩn và so sánh giữa hai quy ước tính loss: `reduction='mean'` và `reduction='sum'`.

**Phân tích và Lời giải:**

Với mini-batch kích thước $b < B$:
- Khi sử dụng `reduction='sum'`: Gradient tích lũy có độ lớn $\mathbf{g}_{\text{sum}} = \sum_{i=1}^b \nabla \ell_i$. Vì $b < B$, độ lớn của gradient này nhỏ hơn độ lớn trung bình của các batch trước, làm suy giảm quán tính của optimizer.
- Khi sử dụng `reduction='mean'`: Gradient trung bình $\mathbf{g}_{\text{mean}} = \frac{1}{b}\sum_{i=1}^b \nabla \ell_i$ duy trì cùng kỳ vọng toán $\mathbb{E}[\mathbf{g}_{\text{mean}}] = \nabla \mathcal{L}_{\text{pop}}$, nhưng phương sai của ước lượng gradient tỷ lệ nghịch với $b$ ($\operatorname{Var} \propto \frac{1}{b}$). Do đó batch nhỏ tạo ra một cú huých ngẫu nhiên lớn hơn vào không gian tham số.

---

### Bài 8: Absolute and Huber losses under outliers

**Đề bài:** So sánh tính bền vững (robustness) của hàm mất mát bình phương ($L_2$), hàm mất mát tuyệt đối ($L_1$), và hàm mất mát Huber khi tập dữ liệu bị ô nhiễm bởi các điểm ngoại lai có biên độ cực lớn ($y_i \to \infty$).

**Phân tích toán học:**

Hàm mất mát Huber kết hợp tính chất trơn khả vi của hàm bình phương tại lân cận 0 và tốc độ tăng tuyến tính của hàm trị tuyệt đối ở vùng ngoài biên:

$$\mathcal{L}_\delta(e) = \begin{cases} 
\frac{1}{2\delta} e^2 & \text{khi } |e| \le \delta \\ 
|e| - \frac{\delta}{2} & \text{khi } |e| > \delta 
\end{cases}, \quad e = \hat{y} - y$$

Đạo hàm theo sai số $e$:

$$\frac{d\mathcal{L}_\delta}{de} = \begin{cases} 
\frac{e}{\delta} & \text{khi } |e| \le \delta \\ 
\operatorname{sign}(e) & \text{khi } |e| > \delta 
\end{cases}$$

- **Mất mát bình phương ($L_2$):** Gradient $\propto e$. Khi xuất hiện điểm ngoại lai với sai số $e = 10.000$, gradient bùng nổ lên $10.000$, chi phối toàn bộ hướng cập nhật của SGD, kéo mô hình lệch khỏi phân phối của đa số điểm dữ liệu chuẩn.
- **Mất mát Huber:** Gradient bị chặn cứng trong đoạn $[-1, 1]$ với mọi sai số dù lớn đến đâu ($|e| \to \infty \implies |\nabla| = 1$). Nhờ đó, Huber loss triệt tiêu sự thống trị của các điểm nhiễu ngoại lai trong khi vẫn duy trì tính khả vi liên tục bậc một tại $e = 0$, khắc phục nhược điểm góc nhọn không khả vi của $L_1$ loss.

---

### Bài 9: Why shuffle training examples

**Đề bài:** Giải thích cơ sở thống kê học của việc xáo trộn ngẫu nhiên (Random Shuffling) tập dữ liệu trước mỗi epoch huấn luyện trong giải thuật Stochastic Gradient Descent (SGD).

**Phân tích và Lời giải:**

1. **Đảm bảo tính không chệch của gradient ngẫu nhiên:**
   Nền tảng lý thuyết của SGD dựa trên giả định rằng gradient của mỗi mini-batch là một ước lượng không chệch của gradient kỳ vọng trên toàn bộ phân phối:
   $$\mathbb{E}_{\mathcal{B}}[\nabla \mathcal{L}_{\mathcal{B}}(\boldsymbol{\theta})] = \nabla \mathcal{L}(\boldsymbol{\theta})$$
   Nếu dữ liệu được sắp xếp theo thứ tự có tính quy luật (ví dụ: các nhãn giá trị thấp xếp trước, nhãn giá trị cao xếp sau), gradient của các batch đầu epoch sẽ liên tục kéo tham số về cực trị của nhóm đầu, và các batch cuối epoch lại kéo tham số về nhóm sau. Điều này gây ra hiện tượng dao động tuần hoàn giới hạn (limit cycle oscillation) và ngăn cản sự hội tụ.
2. **Phá vỡ mối tự tương quan nội tại (Intra-batch Autocorrelation):**
   Xáo trộn mẫu làm triệt tiêu sự phụ thuộc tương quan giữa các mẫu liên tiếp, tối thiểu hóa phương sai của vector gradient mini-batch và hỗ trợ SGD vượt qua các cực tiểu cục bộ nông hoặc điểm yên ngựa (saddle points).
