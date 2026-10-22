# 4.3. The Base Classification Model: Mô hình Phân loại Cơ sở và Lý thuyết Quyết định Bayes

Nguồn tham khảo: [4.3.4. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/classification.html#exercises).

---

### Bài 1: Exact validation loss from a quick minibatch estimate

**Đề bài:** Thiết lập biểu thức giải tích biểu diễn sai số kiểm định chính xác $L_v$ theo ước lượng nhanh không trọng số $L_v^q$, mất mát của mini-batch cuối cùng $\ell_v^b$, quy mô tập dữ liệu $N$, kích thước lô $B$ và số lượng mini-batch $m$.

**Phân tích và Chứng minh:**

Giả sử tập kiểm định gồm $N$ mẫu được chia thành $m$ mini-batch, trong đó:
- $m - 1$ batch đầu tiên có kích thước đầy đủ $B$.
- Batch thứ $m$ cuối cùng có kích thước $r = N - (m - 1)B$ mẫu ($1 \le r \le B$).

Gọi $\ell_j$ là sai số mất mát trung bình của batch thứ $j$ ($j \in \{1, \dots, m\}$) và $\ell_v^b = \ell_m$.
Ước lượng trung bình không trọng số nhanh (Quick Estimate):

$$L_v^q = \frac{1}{m} \sum_{j=1}^m \ell_j \iff \sum_{j=1}^{m-1} \ell_j = m L_v^q - \ell_v^b$$

Sai số kiểm định chính xác trên từng mẫu quan sát:

$$L_v = \frac{1}{N} \left[ \sum_{j=1}^{m-1} B \ell_j + r \ell_v^b \right] = \frac{B \sum_{j=1}^{m-1} \ell_j + r \ell_v^b}{N}$$

Thay thế tổng tích lũy của $m - 1$ batch đầu vào phương trình:

$$L_v = \frac{B (m L_v^q - \ell_v^b) + r \ell_v^b}{N} = \frac{m B L_v^q + (r - B)\ell_v^b}{N}$$

*Nhận xét:* Khi $r = B$ (tập dữ liệu chia hết hoàn hảo cho kích thước batch, $N = mB$), số hạng $(r - B)$ triệt tiêu và ta thu được $L_v = L_v^q$. Khi $r < B$, ước lượng nhanh $L_v^q$ bị lệch do gán trọng số tương đương cho batch thiếu mẫu.

---

### Bài 2: Unbiasedness of the quick estimate

**Đề bài:** Chứng minh tính không chệch của ước lượng nhanh $L_v^q$ dưới mô hình lấy mẫu ngẫu nhiên độc lập và giải thích lý do thực tế vẫn cần tính toán sai số kiểm định chính xác $L_v$.

**Phân tích và Lời giải:**

1. **Chứng minh tính không chệch (Unbiasedness):**
   Nếu các mini-batch được tạo thành thông qua quá trình lấy mẫu ngẫu nhiên đồng nhất từ phân phối tổng thể $P(X, Y)$, thì kỳ vọng toán của hàm mất mát trên bất kỳ batch nào (bất kể kích thước $B$ hay $r$) đều bằng rủi ro kỳ vọng lý thuyết:
   $$\mathbb{E}[\ell_j] = \mathcal{R}_{\text{true}}, \quad \forall j \in \{1, \dots, m\}$$
   Lấy kỳ vọng toán của ước lượng nhanh:
   $$\mathbb{E}[L_v^q] = \frac{1}{m} \sum_{j=1}^m \mathbb{E}[\ell_j] = \frac{1}{m} \cdot m \mathcal{R}_{\text{true}} = \mathcal{R}_{\text{true}}$$
   Do đó, $L_v^q$ là một ước lượng không chệch của rủi ro tổng thể.
2. **Sự cần thiết của Sai số kiểm định chính xác $L_v$:**
   Trong thực hành học sâu, tập kiểm định (Validation Set) là một tập dữ liệu hữu hạn cố định $\mathcal{D}_{\text{val}}$. Mục tiêu của ta là đo lường rủi ro thực nghiệm chính xác trên tập này:
   $$\mathcal{R}_{\text{emp}}(\mathcal{D}_{\text{val}}) = \frac{1}{N} \sum_{i=1}^N \ell(f(\mathbf{x}_i), y_i)$$
   Ước lượng $L_v^q$ gán trọng số cho mỗi mẫu trong batch cuối là $\frac{1}{m \cdot r}$, lớn hơn trọng số của các mẫu trong batch chuẩn ($\frac{1}{m \cdot B}$). Sự thiên lệch mẫu này có thể làm sai lệch việc lựa chọn checkpoint tối ưu hoặc kích hoạt sai điều kiện dừng sớm (Early Stopping) khi batch cuối tình cờ chứa các điểm dữ liệu dị thường.

---

### Bài 3: Bayes-optimal prediction for a general classification loss

**Đề bài:** Thiết lập quy tắc quyết định Bayes tối ưu (Bayes-Optimal Decision Rule) cho bài toán phân loại đa lớp dưới ma trận chi phí tổn thất tổng quát $C(y, \hat{y})$ và phân phối hậu nghiệm $P(y \mid \mathbf{x})$.

**Phân tích toán học:**

Trong lý thuyết quyết định thống kê (Statistical Decision Theory), xét không gian nhãn thực tế $y \in \{1, \dots, K\}$ và không gian hành động quyết định $a \in \{1, \dots, K\}$.
Gọi $C(y, a) \ge 0$ là chi phí phải trả khi nhãn thực tế là $y$ nhưng mô hình đưa ra dự đoán hành động $a$.

Với một vector đặc trưng quan sát $\mathbf{x}$, rủi ro có điều kiện (Conditional Risk / Expected Loss) khi chọn hành động $a$:

$$\mathcal{R}(a \mid \mathbf{x}) = \mathbb{E}_{Y \mid \mathbf{x}}[C(Y, a)] = \sum_{y=1}^K C(y, a) P(Y = y \mid \mathbf{x})$$

Quy tắc quyết định Bayes tối ưu là hành động cực tiểu hóa rủi ro có điều kiện:

$$a^*(\mathbf{x}) = \arg\min_{a \in \{1, \dots, K\}} \sum_{y=1}^K C(y, a) P(Y = y \mid \mathbf{x})$$

1. **Trường hợp tổn thất 0–1 (Zero-One Loss):**
   Nếu $C(y, a) = \mathbb{I}(y \ne a) = 1 - \delta_{y, a}$:
   $$\mathcal{R}(a \mid \mathbf{x}) = \sum_{y \ne a} P(Y = y \mid \mathbf{x}) = 1 - P(Y = a \mid \mathbf{x})$$
   $$\implies a^*(\mathbf{x}) = \arg\min_a [1 - P(Y = a \mid \mathbf{x})] = \arg\max_a P(Y = a \mid \mathbf{x})$$
   Quy tắc suy biến về phép chọn nhãn có xác suất hậu nghiệm cực đại (Maximum A Posteriori - MAP / Argmax).
2. **Trường hợp tổn thất phi đối xứng (Asymmetric Costs trong Chẩn đoán Y tế):**
   Nếu chi phí bỏ sót bệnh nhân mắc bệnh hiểm nghèo (False Negative) đắt gấp 100 lần chi phí xét nghiệm lại người khỏe mạnh (False Positive), tức $C(\text{bệnh}, \text{khỏe}) \gg C(\text{khỏe}, \text{bệnh})$, ngưỡng quyết định tối ưu không còn ở mức xác suất $50\%$ mà sẽ dịch chuyển về phía xác suất thấp (ví dụ: chỉ cần $P(\text{bệnh} \mid \mathbf{x}) > 1\%$ đã phải đưa ra quyết định cảnh báo bệnh).
