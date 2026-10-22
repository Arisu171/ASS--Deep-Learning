# 4.1. Softmax Regression: Hồi quy Softmax và Phân loại Đa lớp

Nguồn tham khảo: [4.1.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/softmax-regression.html#exercises).

---

### Bài 1: Cross-entropy Hessian and softmax variance

**Đề bài:** Tính toán ma trận đạo hàm cấp hai (Hessian Matrix) của hàm mất mát Cross-Entropy theo các giá trị logits $\mathbf{o} \in \mathbb{R}^K$. Chứng minh rằng ma trận Hessian này đồng nhất với ma trận hiệp phương sai của phân phối xác suất danh mục (Categorical Distribution).

**Chứng minh:**

Xét vector logits $\mathbf{o} = [o_1, \dots, o_K]^\top \in \mathbb{R}^K$. Phân phối xác suất Softmax được xác định bởi:

$$p_i = \operatorname{softmax}(\mathbf{o})_i = \frac{\exp(o_i)}{\sum_{j=1}^K \exp(o_j)}, \quad \forall i \in \{1, \dots, K\}$$

Hàm mất mát Cross-Entropy đối với nhãn one-hot $\mathbf{y} \in \{0, 1\}^K$ ($\sum_{i=1}^K y_i = 1$):

$$\ell(\mathbf{o}, \mathbf{y}) = - \sum_{i=1}^K y_i \ln p_i = - \sum_{i=1}^K y_i o_i + \ln \sum_{j=1}^K \exp(o_j)$$

1. **Gradient bậc nhất theo logits:**
   $$\frac{\partial \ell}{\partial o_k} = - y_k + \frac{\exp(o_k)}{\sum_{j=1}^K \exp(o_j)} = p_k - y_k \implies \nabla_{\mathbf{o}} \ell = \mathbf{p} - \mathbf{y}$$
2. **Ma trận đạo hàm cấp hai (Hessian):**
   Đạo hàm riêng của $p_k$ theo $o_m$:
   $$\frac{\partial p_k}{\partial o_m} = \begin{cases} 
   p_k (1 - p_k) & \text{khi } k = m \\ 
   - p_k p_m & \text{khi } k \ne m 
   \end{cases}$$
   Do đó, phần tử tổng quát của ma trận Hessian $\mathbf{H} = \nabla_{\mathbf{o}}^2 \ell \in \mathbb{R}^{K \times K}$:
   $$H_{k, m} = \frac{\partial^2 \ell}{\partial o_k \partial o_m} = \frac{\partial p_k}{\partial o_m} = \operatorname{diag}(\mathbf{p})_{k, m} - p_k p_m$$
   Dưới dạng ma trận:
   $$\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^\top$$
3. **Mối liên hệ với ma trận Hiệp phương sai:**
   Xét biến ngẫu nhiên vector one-hot $\mathbf{Z} \in \{0, 1\}^K$ tuân theo phân phối đa thức với $\mathbb{P}(\mathbf{Z} = \mathbf{e}_i) = p_i$.
   - Vector kỳ vọng: $\mathbb{E}[\mathbf{Z}] = \mathbf{p}$.
   - Ma trận mô-men cấp hai: $\mathbb{E}[\mathbf{Z} \mathbf{Z}^\top] = \sum_{i=1}^K p_i (\mathbf{e}_i \mathbf{e}_i^\top) = \operatorname{diag}(\mathbf{p})$.
   - Ma trận hiệp phương sai của $\mathbf{Z}$:
     $$\operatorname{Cov}(\mathbf{Z}) = \mathbb{E}[\mathbf{Z} \mathbf{Z}^\top] - \mathbb{E}[\mathbf{Z}] \mathbb{E}[\mathbf{Z}]^\top = \operatorname{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^\top = \mathbf{H}$$
   Do ma trận hiệp phương sai luôn bán xác định dương ($\mathbf{v}^\top \mathbf{H} \mathbf{v} = \operatorname{Var}(\mathbf{v}^\top \mathbf{Z}) \ge 0, \forall \mathbf{v}$), hàm mất mát Cross-Entropy theo logits là **hàm lồi** (convex function), đảm bảo tính duy nhất của điểm cực tiểu toàn cục.

---

### Bài 2: Binary codes for three equally likely classes

**Đề bài:** Phân tích giới hạn lý thuyết thông tin khi mã hóa ba lớp có cùng xác suất xuất hiện ($p_i = \frac{1}{3}$) bằng các chuỗi bit nhị phân. Đánh giá tính tối ưu khi mã hóa gom cụm nhiều quan sát liên tiếp (Block Coding).

**Phân tích và Lời giải:**

1. **Entropy Shannon của nguồn tin:**
   $$\mathcal{H}(X) = - \sum_{i=1}^3 \frac{1}{3} \log_2\left(\frac{1}{3}\right) = \log_2 3 \approx 1,58496 \text{ bits/biểu tượng}$$
2. **Hạn chế khi mã hóa từng mẫu đơn lẻ:**
   - Một bit chỉ có thể biểu diễn $2^1 = 2$ trạng thái. Do đó cần tối thiểu 2 bit cố định ($2^2 = 4$ trạng thái, dư thừa 1 trạng thái), tạo ra độ dài trung bình $L = 2 \text{ bits} > 1,585 \text{ bits}$.
   - Nếu áp dụng mã tiền tố Huffman (Prefix Code), tập mã có thể là $\{0, 10, 11\}$ với độ dài tương ứng là $\{1, 2, 2\}$. Độ dài trung bình:
     $$L_{\text{Huffman}} = \frac{1}{3}(1) + \frac{1}{3}(2) + \frac{1}{3}(2) = \frac{5}{3} \approx 1,6667 \text{ bits/biểu tượng}$$
3. **Mã hóa theo khối (Block Coding / Shannon's Source Coding Theorem):**
   - Khi gom $n$ quan sát độc lập liên tiếp, không gian trạng thái mở rộng thành $3^n$ khả năng đồng xác suất. Số bit cần thiết để mã hóa toàn bộ khối là $\lceil \log_2(3^n) \rceil = \lceil n \log_2 3 \rceil$.
   - Chi phí mã hóa trung bình cho mỗi quan sát:
     $$\lim_{n \to \infty} \frac{\lceil n \log_2 3 \rceil}{n} = \log_2 3 \approx 1,585 \text{ bits}$$
   - *Ví dụ:* Với khối $n = 2$, có $3^2 = 9$ trạng thái, cần $\lceil \log_2 9 \rceil = 4$ bit, đạt hiệu suất $\frac{4}{2} = 2,0$ bits/mẫu. Với khối $n = 5$, có $3^5 = 243$ trạng thái, cần $\lceil \log_2 243 \rceil = 8$ bit, đạt hiệu suất $\frac{8}{5} = 1,60$ bits/mẫu (rất gần giới hạn entropy lý thuyết).

---

### Bài 3: Encoding integers with ternary signals

**Đề bài:** Xác định số lượng ký hiệu điều chế biên độ xung tam phân (PAM-3) tối thiểu để biểu diễn các số nguyên trong tập $\{0, 1, \dots, 7\}$. Phân tích ưu thế và thách thức kỹ thuật của tín hiệu tam phân so với nhị phân.

**Phân tích và Lời giải:**

1. **Dung lượng mã hóa tam phân:**
   Mỗi ký hiệu PAM-3 sở hữu 3 mức điện áp khả dĩ $\{-1, 0, +1\}$.
   - Với $k$ ký hiệu, số trạng thái tối đa có thể mã hóa là $3^k$.
   - Để mã hóa 8 trạng thái ($\{0, \dots, 7\}$), ta cần $3^k \ge 8 \implies k \ge \lceil \log_3 8 \rceil = \lceil 1,8928 \rceil = 2$ ký hiệu.
   - Hai ký hiệu tam phân cung cấp $3^2 = 9$ tổ hợp, đủ để biểu diễn từ $00_3 = 0$ đến $21_3 = 7$ (dư một tổ hợp $22_3 = 8$).
2. **Đánh giá kỹ thuật phần cứng:**
   - **Ưu thế:** Tăng mật độ thông tin truyền dẫn trên một chu kỳ xung nhịp ($\log_2 3 \approx 1,585 \text{ bits/baud}$ so với $1 \text{ bit/baud}$ của nhị phân), giảm tần số chuyển mạch (switching frequency) và tiết kiệm băng thông vật lý.
   - **Thách thức:** Với cùng một điện áp nguồn $V_{\text{DD}}$, khoảng cách điện áp phân tách giữa các mức logic bị thu hẹp còn $\frac{V_{\text{DD}}}{2}$ thay vì $V_{\text{DD}}$. Điều này làm suy giảm tỷ số tín hiệu trên nhiễu (SNR margin), đòi hỏi bộ giải mã ADC có độ nhạy cao hơn và mạch khử nhiễu phức tạp hơn.

---

### Bài 4: Softmax and Bradley–Terry preferences

**Đề bài:** Thiết lập tính tương đương toán học giữa hàm phân phối Softmax và mô hình lựa chọn sở thích Bradley–Terry. Trình bày phương pháp mở rộng mô hình cho lựa chọn loại trừ (lựa chọn "không chọn phương án nào").

**Phân tích và Chứng minh:**

1. **Mô hình Bradley–Terry:**
   Xét hai đối tượng $i$ và $j$ với điểm tiềm ẩn (latent utilities) $u_i, u_j$. Xác suất đối tượng $i$ được ưu tiên hơn đối tượng $j$:
   $$\mathbb{P}(i \succ j) = \frac{\pi_i}{\pi_i + \pi_j}, \quad \pi_i = \exp(o_i)$$
   Biểu diễn qua logits $o_i, o_j$:
   $$\mathbb{P}(i \succ j) = \frac{\exp(o_i)}{\exp(o_i) + \exp(o_j)} = \frac{1}{1 + \exp(-(o_i - o_j))} = \sigma(o_i - o_j)$$
   Khi mở rộng ra tập hợp $K$ lựa chọn, xác suất lựa chọn phương án $i$ thỏa mãn:
   $$\mathbb{P}(i) = \frac{\exp(o_i)}{\sum_{j=1}^K \exp(o_j)} = \operatorname{softmax}(\mathbf{o})_i$$
   Tính đơn điệu: $\frac{\partial \mathbb{P}(i)}{\partial o_i} = \mathbb{P}(i)(1 - \mathbb{P}(i)) > 0$, chứng minh rằng việc tăng logit của một lựa chọn luôn làm tăng xác suất lựa chọn phương án đó.
2. **Mở rộng cho tùy chọn loại trừ ("Neither / Outside Option"):**
   Gán cho lựa chọn không hành động một giá trị logit cơ sở $o_0$ (thường chuẩn hóa $o_0 = 0$). Phân phối xác suất mở rộng trên $K+1$ trạng thái:
   $$\mathbb{P}(\text{neither}) = \frac{\exp(o_0)}{\exp(o_0) + \sum_{j=1}^K \exp(o_j)} = \frac{1}{1 + \sum_{j=1}^K \exp(o_j - o_0)}$$

---

### Bài 5: Properties of the two-input log-sum-exp

**Đề bài:** Chứng minh rằng hàm $\operatorname{LSE}(a, b) = \ln(e^a + e^b) > \max(a, b)$, xác định cận infimum của độ lệch và chứng minh tính hội tụ tiệm cận về hàm $\max$ khi tham số nhiệt độ $T \to 0$.

**Chứng minh:**

Không mất tính tổng quát, giả sử $a \ge b \implies \max(a, b) = a$.

1. **Tính chặn trên ngặt:**
   $$\operatorname{LSE}(a, b) = \ln\left(e^a (1 + e^{b - a})\right) = a + \ln(1 + e^{b - a})$$
   Do $e^{b - a} > 0, \forall a, b \in \mathbb{R}$, ta có $1 + e^{b - a} > 1 \implies \ln(1 + e^{b - a}) > 0$.
   $$\implies \operatorname{LSE}(a, b) > a = \max(a, b)$$
2. **Cực trị của độ lệch $\Delta(a, b) = \operatorname{LSE}(a, b) - \max(a, b) = \ln(1 + e^{-|a - b|})$:**
   - Độ lệch cực đại đạt được khi $a = b$: $\Delta_{\max} = \ln(1 + e^0) = \ln 2 \approx 0,69315$.
   - Độ lệch cực tiểu tiệm cận (Infimum): Khi $|a - b| \to \infty$, $e^{-|a - b|} \to 0 \implies \inf \Delta(a, b) = \lim_{|a - b| \to \infty} \ln(1 + e^{-|a - b|}) = 0$.
3. **Giới hạn tham số nhiệt độ (Temperature Parameter $\tau > 0$):**
   Định nghĩa hàm LSE có trọng số nhiệt độ: $f_\tau(a, b) = \tau \ln\left(e^{a/\tau} + e^{b/\tau}\right)$.
   $$f_\tau(a, b) = \max(a, b) + \tau \ln\left(1 + e^{-\frac{|a - b|}{\tau}}\right)$$
   Lấy giới hạn khi nhiệt độ đóng băng $\tau \to 0^+$:
   $$\lim_{\tau \to 0^+} f_\tau(a, b) = \max(a, b) + \lim_{\tau \to 0^+} \tau \ln\left(1 + e^{-\frac{|a - b|}{\tau}}\right) = \max(a, b) + 0 = \max(a, b)$$
   Do đó, Log-Sum-Exp chính là một xấp xỉ trơn khả vi vô hạn ($C^\infty$) của toán tử không trơn $\max(a, b)$.

---

### Bài 6: Convexity, translation, and stable log-partition

**Đề bài:** Chứng minh tính lồi của hàm phân hoạch log $g(\mathbf{x}) = \ln \sum_{i=1}^n \exp(x_i)$, phân tích tính bất biến tịnh tiến và trình bày giải thuật ổn định số học.

**Chứng minh:**

1. **Tính lồi (Convexity):**
   Gradient bậc nhất: $\nabla g(\mathbf{x}) = \operatorname{softmax}(\mathbf{x}) = \mathbf{p}$.
   Ma trận Hessian: $\nabla^2 g(\mathbf{x}) = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.
   Với mọi vector bất kỳ $\mathbf{v} \in \mathbb{R}^n$:
   $$\mathbf{v}^\top \nabla^2 g(\mathbf{x}) \mathbf{v} = \sum_{i=1}^n p_i v_i^2 - \left(\sum_{i=1}^n p_i v_i\right)^2 = \mathbb{E}[V^2] - (\mathbb{E}[V])^2 = \operatorname{Var}(V) \ge 0$$
   Do ma trận Hessian luôn nửa xác định dương trên toàn miền $\mathbb{R}^n$, hàm $g(\mathbf{x})$ là hàm lồi.
2. **Tính tịnh tiến:**
   $$g(\mathbf{x} + c \mathbf{1}) = \ln \sum_{i=1}^n \exp(x_i + c) = \ln\left(e^c \sum_{i=1}^n \exp(x_i)\right) = c + g(\mathbf{x})$$
3. **Kỹ thuật tính toán ổn định số học (Log-Sum-Exp Trick):**
   Để triệt tiêu nguy cơ tràn số học (overflow) khi $x_i \gg 0$, ta chọn $m = \max_{j} x_j$:
   $$g(\mathbf{x}) = m + \ln \sum_{i=1}^n \exp(x_i - m)$$
   Khi đó số mũ lớn nhất thỏa mãn $x_i - m = 0 \implies \exp(0) = 1$, hoàn toàn loại bỏ khả năng tràn số thực trong biểu diễn FP32/FP16.

---

### Bài 7: Temperature scaling via a power transform

**Đề bài:** Xét phép biến đổi lũy thừa $Q(i) \propto P(i)^\alpha$ với $\alpha > 0$. Xác định mối liên hệ với tham số nhiệt độ $T$ trong Softmax và khảo sát hành vi tại các điểm giới hạn $T \to 0$ và $T \to \infty$.

**Phân tích và Lời giải:**

Nếu phân phối gốc tuân theo Softmax với logits $o_i$: $P(i) \propto \exp(o_i)$.
Phép biến đổi lũy thừa:

$$Q(i) \propto P(i)^\alpha \propto \left(\exp(o_i)\right)^\alpha = \exp(\alpha o_i) = \exp\left(\frac{o_i}{T}\right) \implies \alpha = \frac{1}{T}$$

1. **Hiệu chỉnh nhiệt độ:**
   - Tăng gấp đôi nhiệt độ ($T' = 2T$) tương ứng hệ số lũy thừa giảm một nửa: $\alpha' = \frac{\alpha}{2}$.
   - Giảm một nửa nhiệt độ ($T' = \frac{T}{2}$) tương ứng hệ số lũy thừa tăng gấp đôi: $\alpha' = 2\alpha$.
2. **Hành vi tiệm cận:**
   - **Khi $T \to 0^+$ ($\alpha \to \infty$ - Nhiệt độ đóng băng):** Phân phối suy biến thành phân phối tập trung tuyệt đối vào lớp có logit cực đại (Argmax / One-hot collapse). Nếu có nhiều lớp cùng đạt giá trị cực đại, xác suất được chia đều giữa các lớp đồng hạng.
   - **Khi $T \to \infty$ ($\alpha \to 0^+$ - Nhiệt độ vô hạn):** Phân phối hội tụ về phân phối đều (Uniform Distribution) trên không gian các lớp: $Q(i) \to \frac{1}{K}$, entropy đạt cực đại.
