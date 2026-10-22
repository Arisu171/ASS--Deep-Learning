# 5.4. Numerical Stability and Initialization: Độ Ổn định Số học và Khởi tạo Tham số trong Mạng Sâu

Nguồn tham khảo: [5.4.4. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#exercises).

---

### Bài 1: Other symmetries that require breaking

**Đề bài:** Phân tích các dạng đối xứng cấu trúc (Structural Symmetries) trong mạng nơ-ron sâu bên cạnh tính đối xứng hoán vị nơ-ron (Permutation Symmetry) đòi hỏi phải phá vỡ khi khởi tạo tham số.

**Phân tích và Lời giải:**

1. **Đối xứng Hoán vị nơ-ron ẩn (Permutation Symmetry):**
   Nếu khởi tạo mọi nơ-ron trong cùng một tầng ẩn với các trọng số và độ chệch giống hệt nhau, các nơ-ron này sẽ nhận cùng một kích hoạt forward và cùng một tín hiệu gradient backward. Khi đó các nơ-ron cập nhật giống nhau và suy biến thành các bản sao vô nghĩa. Việc khởi tạo ngẫu nhiên phá vỡ đối xứng hoán vị này.
2. **Đối xứng Co giãn (Scale / Rescaling Invariance trong mạng ReLU):**
   Do tính chất thuần nhất dương của hàm ReLU ($\sigma(c \cdot z) = c \cdot \sigma(z)$ với $c > 0$), xét hai tầng liên tiếp:
   $$\mathbf{h}_2 = \sigma\left(\sigma(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2\right)$$
   Nếu nhân $\mathbf{W}_1$ với $c$ và nhân $\mathbf{W}_2$ với $\frac{1}{c}$, hàm số đầu ra hoàn toàn không đổi:
   $$\sigma(\mathbf{x}(c\mathbf{W}_1))\left(\frac{1}{c}\mathbf{W}_2\right) = c \sigma(\mathbf{x}\mathbf{W}_1)\left(\frac{1}{c}\mathbf{W}_2\right) = \sigma(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$$
   Điều này tạo ra một thung lũng nghiệm vô hạn tương đương trong không gian tham số, gây khó khăn cho việc định hướng gradient nếu không có kỹ thuật chuẩn hóa (như Batch/Layer Normalization).
3. **Đối xứng Xoay trong Phân rã Ma trận Tuyến tính (Unitary / Orthogonal Equivalence):**
   Trong phép phân rã ma trận $\mathbf{W}_{\text{eff}} = \mathbf{W}_1 \mathbf{W}_2$, với bất kỳ ma trận trực giao $\mathbf{Q}$ ($\mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$), ta có $(\mathbf{W}_1 \mathbf{Q})(\mathbf{Q}^\top \mathbf{W}_2) = \mathbf{W}_1 \mathbf{W}_2$.

---

### Bài 2: Equal initialization for linear and softmax regression

**Đề bài:** Phân tích lý do việc khởi tạo toàn bộ trọng số bằng cùng một hằng số (ví dụ bằng 0) vẫn hoạt động được trong hồi quy tuyến tính và hồi quy Softmax nhưng lại thất bại hoàn toàn trong mạng Perceptron đa lớp.

**Phân tích toán học:**

1. **Hồi quy Tuyến tính và Softmax (Mô hình Đơn tầng):**
   - Không chứa tầng ẩn nào: các nơ-ron đầu ra kết nối trực tiếp với các kênh đặc trưng đầu vào khác nhau của dữ liệu $\mathbf{x} = [x_1, \dots, x_d]$.
   - Dù khởi tạo $\mathbf{W} = \mathbf{0}$, vector gradient tại bước đầu tiên là $\nabla_{\mathbf{W}} \ell = \mathbf{X}^\top (\mathbf{p} - \mathbf{y})$. Do các cột đặc trưng của dữ liệu $\mathbf{X}$ có giá trị khác nhau và vector nhãn $\mathbf{y}$ mang thông tin phân tách lớp, các thành phần của gradient sẽ nhận các giá trị khác nhau. Điều này lập tức đẩy các trọng số tách biệt nhau ngay từ bước cập nhật đầu tiên.
2. **Mạng Nơ-ron Đa lớp (MLP):**
   - Nếu khởi tạo toàn bộ trọng số tầng ẩn bằng cùng một giá trị $c$: $[\mathbf{W}_1]_{i, j} = c$.
   - Mọi nơ-ron ẩn $j$ đều tính ra cùng một giá trị pre-activation: $z_j = c \sum_{i=1}^d x_i$.
   - Mọi nơ-ron ẩn nhận cùng một đạo hàm ngược và được cập nhật cùng một bước nhảy $\Delta W$. Mạng sâu dù có hàng triệu nơ-ron cũng chỉ hoạt động tương đương với một nơ-ron duy nhất (Rank-1 Collapse).

---

### Bài 3: Matrix products, singular values, and gradient conditioning

**Đề bài:** Vận dụng các bất đẳng thức về giá trị kỳ dị (Singular Values) và bán kính phổ của tích ma trận để phân tích điều kiện ổn định của dòng lan truyền gradient qua $L$ tầng sâu.

**Chứng minh Giải tích:**

Xét dòng lan truyền gradient qua chuỗi $L$ tầng tuyến tính: $\mathbf{g}_0 = \mathbf{g}_L \prod_{l=L}^1 \mathbf{W}_l$.
Đặt ma trận tích chuyển tiếp $\mathbf{M} = \prod_{l=1}^L \mathbf{W}_l$.

Theo bất đẳng thức giá trị kỳ dị cực đại (chuẩn phổ ma trận $\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A})$):

$$\sigma_{\max}(\mathbf{A}\mathbf{B}) \le \sigma_{\max}(\mathbf{A}) \sigma_{\max}(\mathbf{B})$$

$$\implies \|\mathbf{M}\|_2 = \sigma_{\max}\left(\prod_{l=1}^L \mathbf{W}_l\right) \le \prod_{l=1}^L \sigma_{\max}(\mathbf{W}_l)$$

Mặt khác, đối với giá trị kỳ dị cực tiểu của các ma trận khả nghịch:

$$\sigma_{\min}\left(\prod_{l=1}^L \mathbf{W}_l\right) \ge \prod_{l=1}^L \sigma_{\min}(\mathbf{W}_l)$$

1. **Hiện tượng Bùng nổ Gradient (Gradient Explosion):**
   Nếu giá trị kỳ dị cực đại của các ma trận trọng số thỏa mãn $\sigma_{\max}(\mathbf{W}_l) > 1 + \epsilon$ với $\epsilon > 0$, chuẩn của gradient sẽ tăng theo hàm mũ:
   $$\|\mathbf{g}_0\|_2 \sim \mathcal{O}\left((1 + \epsilon)^L\right) \xrightarrow{L \to \infty} +\infty$$
2. **Hiện tượng Biến mất Gradient (Gradient Vanishing):**
   Nếu $\sigma_{\max}(\mathbf{W}_l) < 1 - \epsilon$, chuẩn gradient suy giảm theo hàm mũ về 0:
   $$\|\mathbf{g}_0\|_2 \sim \mathcal{O}\left((1 - \epsilon)^L\right) \xrightarrow{L \to \infty} 0$$
3. **Số điều kiện của tích ma trận (Condition Number):**
   $$\kappa(\mathbf{M}) = \frac{\sigma_{\max}(\mathbf{M})}{\sigma_{\min}(\mathbf{M})} \le \prod_{l=1}^L \kappa(\mathbf{W}_l)$$
   Số điều kiện tăng cấp số nhân làm biến dạng cảnh quan tối ưu hóa, gây ra các khe hẹp bất đối xứng.
- *Giải pháp:* Khởi tạo trực giao (Orthogonal Initialization) đảm bảo $\sigma_i(\mathbf{W}) = 1, \forall i$, giúp bảo toàn đẳng cự chuẩn gradient qua các tầng sâu.

---

### Bài 4: Correcting divergent terms after they occur

**Đề bài:** Phân tích tính bất khả thi của việc xử lý các giá trị phân kỳ (`NaN` / `Inf`) sau khi chúng đã xuất hiện trong tensor. Trình bày các kỹ thuật phòng ngừa chủ động: Cắt tỉa gradient (Gradient Clipping) và Co giãn tốc độ học thích ứng theo tầng (LARS).

**Phân tích và Lời giải:**

1. **Tính bất thuận nghịch của tràn số học (`NaN` Propagation):**
   Một khi giá trị số học dấu phẩy động chạm ngưỡng vô hạn `Inf` hoặc không xác định `NaN` (do chia cho 0 hoặc $\ln(0)$), theo chuẩn IEEE-754, mọi phép toán số học tiếp theo chứa `NaN` đều trả về `NaN`:
   $$\text{NaN} + x = \text{NaN}, \quad \text{NaN} \times x = \text{NaN}$$
   Toàn bộ thông tin định lượng của gradient và tham số bị hủy hoại vĩnh viễn; việc cắt tỉa tensor sau khi đã xuất hiện `NaN` là hoàn toàn vô hiệu.
2. **Các Kỹ thuật Phòng ngừa Chủ động:**
   - **Cắt tỉa Gradient theo Chuẩn (Gradient Norm Clipping - Pascanu et al.):**
     Kiểm tra chuẩn Euclidean của toàn bộ vector gradient $\mathbf{g}$ trước bước cập nhật của optimizer:
     $$\tilde{\mathbf{g}} = \mathbf{g} \cdot \min\left(1, \, \frac{\theta}{\|\mathbf{g}\|_2}\right)$$
     Nếu chuẩn gradient vượt ngưỡng trần $\theta$, vector gradient được co giãn tỷ lệ về đúng độ dài $\theta$ mà vẫn bảo toàn hoàn hảo hướng di chuyển trong không gian tham số.
   - **Co giãn Tốc độ học Thích ứng theo Tầng (Layer-wise Adaptive Rate Scaling - LARS):**
     Được thiết kế để huấn luyện mạng sâu với kích thước batch cực lớn ($B \ge 8192$), LARS điều chỉnh tốc độ học cục bộ của từng tầng $l$ dựa trên tỷ số giữa chuẩn trọng số và chuẩn gradient:
     $$\eta_l = \eta \cdot \frac{\|\mathbf{w}_l\|_2}{\|\nabla_{\mathbf{w}_l} \mathcal{L}\|_2 + \beta \|\mathbf{w}_l\|_2}$$
     Ngăn chặn hiện tượng gradient quá lớn ở các tầng ngoài làm phá hủy các biểu diễn đặc trưng ổn định đã học được ở các tầng sâu bên trong.
