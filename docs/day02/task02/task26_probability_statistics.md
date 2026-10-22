# 2. Preliminaries: Dive into Deep Learning 1.0.3

## 2.6. Probability and Statistics: Xác suất và Thống kê trong Học sâu

Nguồn tham khảo: [2.6.8. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_preliminaries/probability.html#exercises).

---

### Bài 1: Uncertainty reduced by more observations

**Đề bài:** Phân tích bản chất toán học của quá trình giảm thiểu độ bất định nhận thức (Epistemic Uncertainty) khi tăng quy mô mẫu quan sát.

**Phân tích và Lời giải:**

Xét mô hình quan sát các biến ngẫu nhiên độc lập cùng phân phối (i.i.d.) $X_1, X_2, \dots, X_n \sim \text{Bernoulli}(p)$ với tham số xác suất thành công $p \in (0, 1)$ chưa biết.

Ước lượng hợp lý cực đại (MLE) của $p$ là tần suất mẫu: $\hat{p}_n = \frac{1}{n} \sum_{i=1}^n X_i$.
- Theo **Luật số lớn yếu (Weak Law of Large Numbers - WLLN)**: $\hat{p}_n \xrightarrow{P} p$ khi $n \to \infty$.
- Sai số chuẩn của ước lượng: $\text{SE}(\hat{p}_n) = \sqrt{\frac{\operatorname{Var}(X)}{n}} = \sqrt{\frac{p(1-p)}{n}} = \mathcal{O}(n^{-1/2})$.

Khi cỡ mẫu quan sát $n \to \infty$, phương sai của ước lượng tiệm cận về 0, nghĩa là độ bất định về tham số $p$ có thể được giảm thiểu tùy ý. Đây chính là bản chất của **độ bất định nhận thức** (Epistemic / Model Uncertainty) – dạng bất định phát sinh do thiếu dữ liệu và có thể triệt tiêu khi bổ sung thêm thông tin.

---

### Bài 2: Irreducible uncertainty

**Đề bài:** Phân tích độ bất định ngẫu nhiên cố hữu (Aleatoric Uncertainty) và chứng minh rằng việc tăng thêm dữ liệu không thể làm biến mất hoàn toàn sự bất định này.

**Phân tích và Lời giải:**

Xét bài toán dự báo kết quả của một lần tung đồng xu công bằng ở bước tiếp theo $X_{n+1}$, ngay cả khi tham số $p = 0.5$ đã được biết chính xác tuyệt đối (độ bất định nhận thức bằng 0).
- Biến cố $X_{n+1}$ vẫn tuân theo phân phối nhị thức với entropy cực đại:
  $$\mathcal{H}(X_{n+1}) = - \sum_{x \in \{0, 1\}} P(x) \log_2 P(x) = 1 \text{ bit}$$
- Sai số dự báo Bayes tối ưu (Bayes Error Rate) bị chặn dưới bởi $\min(p, 1-p) = 0.5$.

Đây được gọi là **độ bất định ngẫu nhiên / cố hữu** (Aleatoric / Data Uncertainty), phát sinh từ sự ngẫu nhiên vốn có của hệ thống vật lý hoặc nhiễu không thể quan sát trong quá trình đo lường. Dù tập dữ liệu huấn luyện mở rộng đến vô hạn ($n \to \infty$), sai số Bayes cố hữu này không thể bị triệt tiêu.

---

### Bài 3: Variance and Chebyshev bound for a coin estimate

**Đề bài:** Xác định phương sai của ước lượng tham số $\hat{p}_n$; thiết lập bất đẳng thức Chebyshev và so sánh với xấp xỉ tiệm cận của Định lý Giới hạn Trung tâm (CLT).

**Phân tích và Lời giải:**

Xét $n$ biến cố độc lập $X_i \sim \text{Bernoulli}(p)$, ước lượng $\hat{p}_n = \frac{1}{n} \sum_{i=1}^n X_i$.
- Kỳ vọng toán: $\mathbb{E}[\hat{p}_n] = p$ (ước lượng không chệch).
- Phương sai: $\operatorname{Var}(\hat{p}_n) = \frac{\operatorname{Var}(X_i)}{n} = \frac{p(1-p)}{n}$.

1. **Bất đẳng thức Chebyshev:**
   Với mọi ngưỡng sai số $\epsilon > 0$:
   $$\mathbb{P}\left(|\hat{p}_n - p| \ge \epsilon\right) \le \frac{\operatorname{Var}(\hat{p}_n)}{\epsilon^2} = \frac{p(1-p)}{n \epsilon^2}$$
   Do $p(1-p) \le \frac{1}{4}$ với mọi $p \in [0, 1]$, ta có chặn trên phân phối độc lập:
   $$\mathbb{P}\left(|\hat{p}_n - p| \ge \epsilon\right) \le \frac{1}{4n \epsilon^2}$$

2. **So sánh với Định lý Giới hạn Trung tâm (CLT):**
   - Theo CLT, khi $n \to \infty$, thống kê chuẩn hóa hội tụ về phân phối chuẩn tắc:
     $$\frac{\hat{p}_n - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} \mathcal{N}(0, 1)$$
   - Đuôi xác suất xấp xỉ theo hàm phân phối Gauss $\Phi$: $\mathbb{P}(|\hat{p}_n - p| \ge \epsilon) \approx 2\left(1 - \Phi\left(\frac{\epsilon\sqrt{n}}{\sqrt{p(1-p)}}\right)\right) \sim \mathcal{O}\left(e^{-2n\epsilon^2}\right)$ (tiệm cận lũy thừa giảm theo bất đẳng thức Hoeffding).
   - *Nhận xét:* Bất đẳng thức Chebyshev đúng với mọi phân phối không cần giả định quy mô mẫu nhưng cung cấp chặn trên lỏng ($\mathcal{O}(n^{-1})$) so với đuôi hàm mũ thực tế ($\mathcal{O}(e^{-n})$).

---

### Bài 4: Sample means and dependence

**Đề bài:** Cho các biến ngẫu nhiên độc lập $X_i$ có $\mathbb{E}[X_i] = 0$ và $\operatorname{Var}(X_i) = 1$. Xét dãy giá trị trung bình tích lũy $z_m = \frac{1}{m} \sum_{i=1}^m X_i$. Đánh giá tính độc lập giữa các biến $z_m$ và phân tích việc áp dụng bất đẳng thức Chebyshev.

**Phân tích và Lời giải:**

1. **Phân phối biên của từng $z_m$:**
   $$\mathbb{E}[z_m] = 0, \quad \operatorname{Var}(z_m) = \frac{1}{m^2} \sum_{i=1}^m \operatorname{Var}(X_i) = \frac{1}{m}$$
   Áp dụng bất đẳng thức Chebyshev cho từng phần tử độc lập:
   $$\mathbb{P}(|z_m| \ge \epsilon) \le \frac{1}{m \epsilon^2}$$

2. **Tính phụ thuộc giữa các $z_m$ khác nhau:**
   Xét $m < k$:
   $$\operatorname{Cov}(z_m, z_k) = \operatorname{Cov}\left(\frac{1}{m}\sum_{i=1}^m X_i, \, \frac{1}{k}\sum_{j=1}^k X_j\right) = \frac{1}{mk} \sum_{i=1}^m \operatorname{Var}(X_i) = \frac{m}{mk} = \frac{1}{k} > 0$$
   Do $\operatorname{Cov}(z_m, z_k) \ne 0$, các biến trung bình mẫu $z_m$ và $z_k$ **hoàn toàn phụ thuộc thống kê lẫn nhau** do chia sẻ chung $m$ mẫu quan sát đầu tiên.
   - *Hệ quả:* Không thể tính xác suất đồng thời $\mathbb{P}(|z_m| < \epsilon \land |z_k| < \epsilon)$ bằng tích các xác suất biên. Để thiết lập chặn trên đồng thời cho toàn bộ quỹ đạo của dãy số $\{z_m\}$, cần vận dụng các bất đẳng thức Martingale (như bất đẳng thức Doob hoặc Kolmogorov).

---

### Bài 5: Bounds for union and intersection

**Đề bài:** Thiết lập các cận trên và cận dưới giải tích cho xác suất của biến cố hợp $\mathbb{P}(A \cup B)$ và biến cố giao $\mathbb{P}(A \cap B)$ theo $p_A = \mathbb{P}(A)$ và $p_B = \mathbb{P}(B)$.

**Phân tích và Lời giải:**

Theo nguyên lý cộng xác suất: $\mathbb{P}(A \cup B) = \mathbb{P}(A) + \mathbb{P}(B) - \mathbb{P}(A \cap B)$.

1. **Cận của biến cố hợp $\mathbb{P}(A \cup B)$:**
   - Cận dưới: Do $A \subseteq A \cup B$ và $B \subseteq A \cup B$, nên $\mathbb{P}(A \cup B) \ge \max(p_A, p_B)$.
   - Cận trên: Theo bất đẳng thức Boole (Union Bound) và tiên đề xác suất $\le 1$:
   $$\max(p_A, p_B) \le \mathbb{P}(A \cup B) \le \min(1, p_A + p_B)$$

2. **Cận của biến cố giao $\mathbb{P}(A \cap B)$ (Bất đẳng thức Fréchet):**
   - Cận trên: Do $A \cap B \subseteq A$ và $A \cap B \subseteq B$, nên $\mathbb{P}(A \cap B) \le \min(p_A, p_B)$.
   - Cận dưới: Từ $\mathbb{P}(A \cup B) \le 1 \implies \mathbb{P}(A \cap B) = p_A + p_B - \mathbb{P}(A \cup B) \ge p_A + p_B - 1$, kết hợp điều kiện không âm:
   $$\max(0, p_A + p_B - 1) \le \mathbb{P}(A \cap B) \le \min(p_A, p_B)$$

---

### Bài 6: Joint probability in a Markov chain

**Đề bài:** Khai triển phân phối xác suất đồng thời $\mathbb{P}(A, B, C)$ cho chuỗi các biến ngẫu nhiên tuân theo tính chất Markov: $A \to B \to C$.

**Phân tích và Lời giải:**

Theo quy tắc nhân xác suất tổng quát (Chain Rule of Probability):

$$\mathbb{P}(A, B, C) = \mathbb{P}(A) \mathbb{P}(B \mid A) \mathbb{P}(C \mid A, B)$$

Theo định nghĩa của tính chất Markov bậc một, biến cố tương lai $C$ độc lập có điều kiện với biến cố quá khứ $A$ khi đã biết trạng thái hiện tại $B$:

$$\mathbb{P}(C \mid A, B) = \mathbb{P}(C \mid B)$$

Thay vào biểu thức trên, ta thu được dạng nhân tử hóa chuẩn tắc của đồ thị xác suất định hướng:

$$\mathbb{P}(A, B, C) = \mathbb{P}(A) \mathbb{P}(B \mid A) \mathbb{P}(C \mid B)$$

---

### Bài 7: Two conditionally dependent diagnostic tests

**Đề bài:** Cho tỷ lệ nhiễm bệnh tiên nghiệm $\mathbb{P}(H = 1) = 0,0015$. Mỗi xét nghiệm độc lập có độ nhạy (True Positive Rate) là $99\%$ ($\text{FNR} = 1\%$) và tỷ lệ dương tính giả (False Positive Rate - FPR) là $10\%$. Giả sử hai xét nghiệm phụ thuộc nhau trên nhóm người khỏe với xác suất đồng thời dương tính là $\mathbb{P}(D_1 = 1, D_2 = 1 \mid H = 0) = 0,02$. Tính xác suất hậu nghiệm mắc bệnh sau một lần và sau hai lần xét nghiệm dương tính liên tiếp.

**Phân tích và Lời giải:**

1. **Bảng phân phối xác suất có điều kiện:**
   - Đối với nhóm người khỏe ($H = 0$): $\mathbb{P}(D_1=1 \mid H=0) = 0,1$, $\mathbb{P}(D_2=1 \mid H=0) = 0,1$, $\mathbb{P}(D_1=1, D_2=1 \mid H=0) = 0,02$.
   - Đối với nhóm người mắc bệnh ($H = 1$): Hai xét nghiệm độc lập có điều kiện: $\mathbb{P}(D_1=1, D_2=1 \mid H=1) = 0,99 \times 0,99 = 0,9801$.

| Trạng thái sức khỏe | $D_1=1, D_2=1$ | $D_1=1, D_2=0$ | $D_1=0, D_2=1$ | $D_1=0, D_2=0$ |
| :--- | :---: | :---: | :---: | :---: |
| **Khỏe mạnh ($H = 0$)** | 0,0200 | 0,0800 | 0,0800 | 0,8200 |
| **Nhiễm bệnh ($H = 1$)** | 0,9801 | 0,0099 | 0,0099 | 0,0001 |

2. **Xác suất hậu nghiệm sau một lần xét nghiệm dương tính:**
   Áp dụng định lý Bayes:
   $$\mathbb{P}(H = 1 \mid D_1 = 1) = \frac{\mathbb{P}(D_1 = 1 \mid H = 1)\mathbb{P}(H = 1)}{\mathbb{P}(D_1 = 1 \mid H = 1)\mathbb{P}(H = 1) + \mathbb{P}(D_1 = 1 \mid H = 0)\mathbb{P}(H = 0)}$$
   $$\mathbb{P}(H = 1 \mid D_1 = 1) = \frac{0,99 \times 0,0015}{0,99 \times 0,0015 + 0,10 \times 0,9985} = \frac{0,001485}{0,001485 + 0,09985} \approx \mathbf{1,465\%}$$

3. **Xác suất hậu nghiệm sau hai lần xét nghiệm dương tính đồng thời:**
   $$\mathbb{P}(H = 1 \mid D_1 = 1, D_2 = 1) = \frac{\mathbb{P}(D_1 = 1, D_2 = 1 \mid H = 1)\mathbb{P}(H = 1)}{\mathbb{P}(D_1 = 1, D_2 = 1 \mid H = 1)\mathbb{P}(H = 1) + \mathbb{P}(D_1 = 1, D_2 = 1 \mid H = 0)\mathbb{P}(H = 0)}$$
   $$\mathbb{P}(H = 1 \mid D_1 = 1, D_2 = 1) = \frac{0,9801 \times 0,0015}{0,9801 \times 0,0015 + 0,0200 \times 0,9985} = \frac{0,00147015}{0,00147015 + 0,01997} \approx \mathbf{6,857\%}$$

*Bình luận học thuật:* Đây là minh họa kinh điển cho hiện tượng **Nghịch lý tỷ lệ nền (Base Rate Fallacy)**. Dù cả hai xét nghiệm đều dương tính, xác suất thực tế mắc bệnh chỉ đạt xấp xỉ $6,86\%$ do căn bệnh có tỷ lệ mắc tiên nghiệm cực thấp ($0,15\%$) kết hợp với sự tương quan dương tính giả ($2\%$) giữa hai xét nghiệm.

---

### Bài 8: Expected return, variance, and portfolio optimization

**Đề bài:** Xây dựng công thức lợi suất kỳ vọng, phương sai rủi ro của danh mục đầu tư và thiết lập bài toán tối ưu hóa danh mục Markowitz dưới dạng bài toán quy hoạch toàn phương (Quadratic Programming).

**Phân tích và Lời giải:**

Xét danh mục gồm $d$ tài sản với vector tỷ trọng phân bổ vốn $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_d)^\top \in \mathbb{R}^d$, vector lợi suất kỳ vọng $\boldsymbol{\mu} \in \mathbb{R}^d$ và ma trận hiệp phương sai bán xác định dương $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$.

1. **Lợi suất kỳ vọng của danh mục:**
   $$\mathbb{E}[R_p] = \sum_{i=1}^d \alpha_i \mu_i = \boldsymbol{\alpha}^\top \boldsymbol{\mu}$$

2. **Phương sai rủi ro của danh mục:**
   $$\operatorname{Var}(R_p) = \operatorname{Var}\left(\boldsymbol{\alpha}^\top \mathbf{R}\right) = \boldsymbol{\alpha}^\top \operatorname{Cov}(\mathbf{R}) \boldsymbol{\alpha} = \boldsymbol{\alpha}^\top \boldsymbol{\Sigma} \boldsymbol{\alpha}$$

3. **Bài toán Quy hoạch Tối ưu hóa Markowitz (Quadratic Programming Formulation):**
   Mục tiêu tối đa hóa lợi suất với mức trần rủi ro $\gamma > 0$ và ràng buộc bảo toàn vốn (không bán khống $\alpha_i \ge 0$):

   $$\begin{aligned}
   \max_{\boldsymbol{\alpha}} \quad & \boldsymbol{\alpha}^\top \boldsymbol{\mu} \\
   \text{thỏa mãn} \quad & \boldsymbol{\alpha}^\top \boldsymbol{\Sigma} \boldsymbol{\alpha} \le \gamma \\
   & \mathbf{1}^\top \boldsymbol{\alpha} = 1 \\
   & \boldsymbol{\alpha} \succeq \mathbf{0}
   \end{aligned}$$

   *Liên hệ với Học sâu:* Cấu trúc toán học này hoàn toàn tương đồng với bài toán tối ưu hóa có điều chuẩn trong học máy, trong đó số hạng toàn phương $\boldsymbol{\alpha}^\top \boldsymbol{\Sigma} \boldsymbol{\alpha}$ đóng vai trò điều chuẩn rủi ro (tương tự như số hạng Weight Decay trong hồi quy Ridge hay tối ưu hóa hàm mục tiêu có ràng buộc Karush-Kuhn-Tucker - KKT).
