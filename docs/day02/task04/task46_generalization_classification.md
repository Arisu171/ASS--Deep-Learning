# 4.6. Generalization in Classification: Khả năng Khái quát hóa trong Phân loại và Lý thuyết Độ phức tạp

Nguồn tham khảo: [4.6.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/generalization-classification.html#exercises).

---

### Bài 1: Samples needed for a precise fixed-model error estimate

**Đề bài:** Xác định số lượng mẫu kiểm thử độc lập $n$ tối thiểu theo Bất đẳng thức Hoeffding để ước lượng tỷ lệ sai số của một mô hình cố định với sai số biên không vượt quá $\epsilon = 10^{-4}$ tại mức độ tin cậy $1 - \delta \ge 99,9\%$ ($\delta = 0,001$).

**Phân tích và Chứng minh:**

Xét một mô hình phân loại cố định $f$. Biến cố sai số trên từng mẫu quan sát là biến ngẫu nhiên nhị phân $Z_i = \mathbb{I}(f(\mathbf{x}_i) \ne y_i) \in \{0, 1\}$.
Tỷ lệ sai số thực nghiệm: $\hat{\epsilon}_n = \frac{1}{n} \sum_{i=1}^n Z_i$. Sai số thực tế kỳ vọng: $\epsilon = \mathbb{E}[Z_i]$.

Theo **Bất đẳng thức Hoeffding** cho tổng các biến ngẫu nhiên độc lập bị chặn trong $[0, 1]$:

$$\mathbb{P}\left(|\hat{\epsilon}_n - \epsilon| \ge \delta_{\text{tol}}\right) \le 2 \exp\left(-2 n \delta_{\text{tol}}^2\right)$$

Để đảm bảo xác suất vi phạm không vượt quá $\delta = 0,001$ với dung sai $\delta_{\text{tol}} = 10^{-4}$:

$$2 \exp\left(-2 n \delta_{\text{tol}}^2\right) \le \delta \iff -2 n \delta_{\text{tol}}^2 \le \ln\left(\frac{\delta}{2}\right) \iff n \ge \frac{\ln(2/\delta)}{2 \delta_{\text{tol}}^2}$$

Thay các giá trị số học:
- $\ln\left(\frac{2}{0,001}\right) = \ln(2000) \approx 7,600902$
- $\delta_{\text{tol}}^2 = (10^{-4})^2 = 10^{-8}$

$$n \ge \frac{7,600902}{2 \times 10^{-8}} = 3,800451 \times 10^8 \approx \mathbf{380.045.000 \text{ mẫu}}$$

*Bình luận học thuật:* Chặn Hoeffding là chặn phi phân phối (distribution-free bound), đảm bảo đúng cho mọi phân phối dữ liệu xấu nhất nhưng có tính chất bảo thủ. Trong thực tế, nếu sai số thực sự $\epsilon$ rất nhỏ ($\epsilon \ll 1$), việc áp dụng các bất đẳng thức chặt hơn như Bernstein hoặc Chernoff sẽ thu hẹp đáng kể yêu cầu về cỡ mẫu.

---

### Bài 2: Test-set leakage through unrestricted model queries

**Đề bài:** Phân tích giao thức tấn công giải mã nhãn tập kiểm thử (Test-Set Label Extraction Attack) thông qua việc truy vấn sai số mất mát tổng thể trên tập kiểm thử không nhãn gồm $N$ mẫu.

**Phân tích và Lời giải:**

Giả sử tập kiểm thử gồm $N$ mẫu nhị phân phân biệt $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ với $y_i \in \{0, 1\}$. Người dùng chỉ nhận được giá trị sai số 0–1 tổng thể $\text{Err} = \sum_{i=1}^N \mathbb{I}(f(\mathbf{x}_i) \ne y_i)$.

**Giao thức trích xuất nhãn với $N + 1$ lượt truy vấn:**
1. **Truy vấn cơ sở (Baseline Query - Lần 1):**
   Gửi mô hình hằng số $f_0(\mathbf{x}) = 0, \forall \mathbf{x}$. Sai số nhận được là số lượng mẫu có nhãn thực tế bằng 1:
   $$E_0 = \sum_{i=1}^N y_i$$
2. **Truy vấn từng điểm (Per-sample Inversion Queries - $N$ lần tiếp theo):**
   Với mỗi mẫu thứ $k \in \{1, \dots, N\}$, xây dựng mô hình $f_k$:
   $$f_k(\mathbf{x}_i) = \begin{cases} 
   1 & \text{khi } i = k \\ 
   0 & \text{khi } i \ne k 
   \end{cases}$$
   Ghi nhận sai số $E_k$. Khi đó độ chênh lệch sai số thỏa mãn:
   $$E_k - E_0 = \begin{cases} 
   -1 & \text{nếu nhãn thực tế } y_k = 1 \ (\text{dự đoán chuyển từ sai thành đúng}) \\ 
   +1 & \text{nếu nhãn thực tế } y_k = 0 \ (\text{dự đoán chuyển từ đúng thành sai}) 
   \end{cases}$$
   Suy ra nhãn chính xác tuyệt đối của từng mẫu:
   $$y_k = \frac{1 - (E_k - E_0)}{2}$$

*Ý nghĩa an toàn học máy:* Chỉ cần $N + 1$ truy vấn điểm số vô hướng, toàn bộ nhãn ẩn của tập kiểm thử sẽ bị giải mã hoàn toàn. Sau đó, kẻ tấn công có thể huấn luyện một mô hình ghi nhớ nhãn tra bảng để đạt độ chính xác $100\%$ trên bảng xếp hạng (Leaderboard overfitting). Đây là lý do các nền tảng thi đấu như Kaggle bắt buộc phải giới hạn số lượt nộp bài mỗi ngày và chia tập kiểm thử thành Public Test và Private Test độc lập.

---

### Bài 3: VC dimension of fifth-order polynomials

**Đề bài:** Xác định số chiều Vapnik–Chervonenkis (VC Dimension) của lớp các hàm phân loại dựa trên đa thức một biến bậc không quá 5: $\mathcal{H} = \{x \mapsto \operatorname{sign}(P(x)) \mid \deg(P) \le 5\}$.

**Chứng minh:**

1. **Khả năng phân tách (Shattering) tập 6 điểm:**
   Xét đa thức một biến $P(x) = \sum_{j=0}^5 w_j x^j$. Đa thức này sở hữu $6$ tham số tự do (hệ số $[w_0, \dots, w_5]$). Với $6$ điểm phân biệt bất kỳ $x_1 < x_2 < \dots < x_6$, ma trận Vandermonde cỡ $6 \times 6$ luôn khả nghịch, cho phép tìm được đa thức $P(x)$ nhận bất kỳ giá trị dấu nào tại 6 điểm này. Do đó, $\text{VC}(\mathcal{H}) \ge 6$.
2. **Không thể phân tách tập 7 điểm:**
   Xét 7 điểm phân biệt trên trục số thực: $x_1 < x_2 < \dots < x_7$. Gán các nhãn xen kẽ liên tục: $+1, -1, +1, -1, +1, -1, +1$.
   Để một đa thức liên tục đổi dấu giữa các cặp điểm xen kẽ này, theo Định lý giá trị trung gian (Intermediate Value Theorem), đa thức bắt buộc phải có ít nhất một nghiệm thực nằm giữa mỗi cặp khoảng mở $(x_i, x_{i+1})$. Do có 6 khoảng chuyển tiếp nhãn, đa thức phải có ít nhất 6 nghiệm thực phân biệt.
   Tuy nhiên, theo Định lý cơ bản của Đại số, một đa thức bậc không quá 5 khác không có tối đa 5 nghiệm thực. Điều này dẫn đến mâu thuẫn toán học. Do đó không thể phân tách 7 điểm.

**Kết luận:** Số chiều VC của lớp đa thức một biến bậc 5 bằng chính xác:

$$\operatorname{VC}(\mathcal{H}) = 6$$

---

### Bài 4: VC dimension of axis-aligned rectangles in 2D

**Đề bài:** Xác định số chiều VC của lớp các bộ phân loại hình chữ nhật có cạnh song song với các trục tọa độ trong không gian 2 chiều $\mathbb{R}^2$.

**Chứng minh:**

Xét lớp giả thuyết $\mathcal{H}_{\text{rect}} = \{h_R(\mathbf{x}) = \mathbb{I}(\mathbf{x} \in R) \mid R = [x_1, x_2] \times [y_1, y_2]\}$.

1. **Chứng minh $\text{VC}(\mathcal{H}_{\text{rect}}) \ge 4$:**
   Chọn 4 điểm tạo thành hình thoi: $(0, 1), (1, 0), (0, -1), (-1, 0)$. Với bất kỳ tập con nào được gán nhãn dương ($+1$), ta luôn có thể dựng một hình chữ nhật song song trục nhỏ nhất (Bounding Box) vừa khít bao trọn các điểm dương đó mà không chứa các điểm âm còn lại. Khi không có điểm dương, chọn hình chữ nhật rỗng. Do đó 4 điểm được phân tách hoàn toàn.
2. **Chứng minh không thể phân tách 5 điểm bất kỳ ($\text{VC}(\mathcal{H}_{\text{rect}}) < 5$):**
   Xét một tập hợp 5 điểm bất kỳ trong $\mathbb{R}^2$. Tìm các điểm cực trị biên theo 4 hướng:
   - Điểm có hoành độ nhỏ nhất (Cực tả - Leftmost).
   - Điểm có hoành độ lớn nhất (Cực hữu - Rightmost).
   - Điểm có tung độ lớn nhất (Cực thượng - Topmost).
   - Điểm có tung độ nhỏ nhất (Cực hạ - Bottommost).
   Bốn hướng biên này được định hình bởi tối đa 4 điểm dữ liệu phân biệt. Do tập hợp có 5 điểm, chắc chắn tồn tại ít nhất một điểm thứ năm $\mathbf{x}^*$ nằm ở vùng trong hoặc trên biên của hộp bao hình chữ nhật tạo bởi 4 điểm cực trị nói trên.
   Nếu ta gán 4 điểm cực trị nhãn dương ($+1$) và gán điểm $\mathbf{x}^*$ nhãn âm ($0$), thì mọi hình chữ nhật song song trục bao trọn 4 điểm cực trị **bắt buộc phải bao trùm luôn cả điểm $\mathbf{x}^*$**. Do đó không thể thực hiện phép phân loại này.

**Kết luận:** Số chiều VC của lớp hình chữ nhật song song trục trong $\mathbb{R}^2$ bằng chính xác:

$$\operatorname{VC}(\mathcal{H}_{\text{rect}}) = 4$$
