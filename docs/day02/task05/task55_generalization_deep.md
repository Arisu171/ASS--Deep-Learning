# 5.5. Generalization in Deep Learning: Khả năng Khái quát hóa trong Mạng Nơ-ron Siêu tham số

Nguồn tham khảo: [5.5.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html#exercises).

---

### Bài 1: Limits of traditional complexity measures

**Đề bài:** Phân tích lý do các thước đo độ phức tạp thống kê cổ điển (Số chiều VC, Rademacher Complexity, Đếm số lượng tham số) thất bại trong việc giải thích khả năng khái quát hóa vượt trội của mạng nơ-ron sâu trong chế độ siêu tham số (Overparameterized Regime).

**Phân tích và Luận giải Khoa học:**

1. **Nghịch lý Siêu tham số (Zhang et al. - "Understanding Deep Learning Requires Rethinking Generalization"):**
   Theo lý thuyết học máy cổ điển, một mô hình có số lượng tham số $P$ vượt xa số mẫu quan sát $N$ ($P \gg N$) sẽ rơi vào hiện tượng quá khớp trầm trọng. Thực nghiệm chứng minh các mạng nơ-ron sâu hiện đại có khả năng khớp hoàn hảo (đạt $100\%$ training accuracy) ngay cả trên tập dữ liệu bị xáo trộn nhãn hoàn toàn ngẫu nhiên.
   - Các cận tổng quát hóa cổ điển dựa trên số chiều VC $\mathcal{O}\left(\sqrt{\frac{\text{VC}}{N}}\right)$ trở nên vô nghĩa vì $\text{VC} \ge P \gg N$, dẫn đến cận trên của sai số lớn hơn 1 (vacuous bounds).
2. **Hiện tượng Suy giảm Kép (Double Descent Phenomenon - Belkin et al.):**
   Khi dung lượng mô hình vượt qua ngưỡng nội suy (Interpolation Boundary, nơi $P = N$ và rủi ro thực nghiệm đạt 0), đường cong rủi ro kiểm định không tiếp tục tăng mà bắt đầu suy giảm trở lại, đạt hiệu năng khái quát hóa cao hơn cả vùng mô hình đơn giản.
3. **Thiên kiến Quy nạp Ngầm của Thuật toán Tối ưu (Implicit Regularization of SGD):**
   Trong vô số cấu hình tham số có khả năng nội suy toàn bộ dữ liệu huấn luyện, thuật toán Stochastic Gradient Descent (SGD) với bước học nhỏ và khởi tạo ngẫu nhiên gần gốc tọa độ luôn có xu hướng ngầm hội tụ về các nghiệm có chuẩn nhỏ nhất (Minimum-Norm Solutions) và rơi vào các thung lũng cực tiểu phẳng (Flat Minima). Các cực tiểu phẳng sở hữu vết ma trận Hessian nhỏ, có độ nhạy thấp trước các biến động phân phối của dữ liệu kiểm thử.

---

### Bài 2: Early stopping as regularization

**Đề bài:** Thiết lập tính tương đương toán học giữa kỹ thuật Dừng sớm (Early Stopping) và điều chuẩn suy giảm trọng số $L_2$ (Weight Decay / Ridge Penalty) trong không gian tham số.

**Chứng minh Giải tích:**

Xét bài toán hồi quy bình phương tối thiểu lồi với khởi tạo tại gốc tọa độ $\mathbf{w}_0 = \mathbf{0}$.
Hàm mất mát: $\mathcal{L}(\mathbf{w}) = \frac{1}{2n} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$.
Ma trận hiệp phương sai mẫu: $\mathbf{A} = \frac{1}{n} \mathbf{X}^\top \mathbf{X}$.
Phân rã phổ của ma trận đối xứng: $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$, trong đó $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_d)$.

1. **Quỹ đạo Gradient Descent liên tục sau thời gian $t$:**
   $$\mathbf{w}(t) = \mathbf{Q} \left[ \mathbf{I} - \exp(-\eta t \boldsymbol{\Lambda}) \right] \boldsymbol{\Lambda}^{-1} \mathbf{Q}^\top \left(\frac{1}{n}\mathbf{X}^\top \mathbf{y}\right)$$
   Đối với mỗi thành phần vector riêng thứ $i$:
   $$w_i(t) = \frac{1 - \exp(-\eta t \lambda_i)}{\lambda_i} [\mathbf{Q}^\top \mathbf{w}_{\text{OLS}}]_i$$
2. **Nghiệm Hồi quy Ridge với hệ số điều chuẩn $\lambda_{\text{ridge}}$:**
   $$w_i^{\text{ridge}} = \frac{1}{\lambda_i + \lambda_{\text{ridge}}} [\mathbf{Q}^\top \mathbf{w}_{\text{OLS}}]_i = \frac{\lambda_i}{\lambda_i + \lambda_{\text{ridge}}} \frac{1}{\lambda_i} [\mathbf{Q}^\top \mathbf{w}_{\text{OLS}}]_i$$
3. **Đồng nhất hệ số co:**
   So sánh hệ số co trọng số:
   $$1 - \exp(-\eta t \lambda_i) \approx \frac{\lambda_i}{\lambda_i + \lambda_{\text{ridge}}} \iff \eta t \approx \frac{1}{\lambda_{\text{ridge}}}$$
   - *Kết luận:* Số lượng bước cập nhật $t$ trong Gradient Descent đóng vai trò nghịch đảo chính xác với hệ số điều chuẩn $\lambda_{\text{ridge}}$:
     $$\text{Thời gian huấn luyện } t \sim \frac{1}{\lambda_{\text{ridge}}}$$
   Dừng sớm tại thời điểm $t$ tương đương với việc áp dụng một mức phạt điều chuẩn $L_2$ tỷ lệ nghịch với $t$. Nó cho phép các thành phần đặc trưng có giá trị riêng lớn ($\lambda_i \gg 0$) hội tụ trước, trong khi các hướng nhiễu có giá trị riêng nhỏ bị triệt tiêu không kịp cập nhật.

---

### Bài 3: Choosing an early-stopping criterion

**Đề bài:** Thiết lập tiêu chuẩn dừng sớm chuẩn mực trong nghiên cứu học sâu: tham số kiên nhẫn (Patience), ngưỡng cải thiện tối thiểu ($\Delta_{\min}$) và kỹ thuật làm mịn số liệu kiểm định.

**Quy trình Vận hành Tiêu chuẩn:**

1. **Giám sát Hàm mất mát trên Tập kiểm định độc lập:**
   Theo dõi giá trị Cross-Entropy Loss hoặc độ đo mục tiêu chính $\mathcal{M}_{\text{val}}^{(t)}$ tại cuối mỗi epoch $t$.
2. **Thiết lập Cửa sổ Kiên nhẫn (Patience Window $p$):**
   Duy trì kỷ lục tốt nhất $\mathcal{M}^* = \min_{\tau \le t} \mathcal{M}_{\text{val}}^{(\tau)}$ và chỉ số epoch tương ứng $t^*$.
   - Nếu $\mathcal{M}_{\text{val}}^{(t)} < \mathcal{M}^* - \Delta_{\min}$: Cập nhật kỷ lục $\mathcal{M}^* \leftarrow \mathcal{M}_{\text{val}}^{(t)}$, lưu checkpoint trọng số $\boldsymbol{\theta}^* \leftarrow \boldsymbol{\theta}_t$, và đặt lại bộ đếm kiên nhẫn về 0.
   - Nếu sau $p$ epoch liên tiếp không có sự cải thiện vượt ngưỡng $\Delta_{\min}$: Ngắt vòng lặp huấn luyện ngay lập tức và khôi phục trạng thái mô hình từ checkpoint $\boldsymbol{\theta}^*$.
3. **Làm mịn Số liệu (Metric Smoothing):**
   Để ngăn chặn việc dừng sớm ngẫu nhiên do biến động nhiễu của mini-batch kiểm định, áp dụng trung bình trượt hàm mũ (Exponential Moving Average - EMA) trên đường cong loss kiểm định trước khi đánh giá tiêu chuẩn dừng.

---

### Bài 4: When early stopping helps most

**Đề bài:** Phân tích các điều kiện tiền đề của bài toán khiến kỹ thuật Dừng sớm phát huy tối đa hiệu quả kiểm soát khoảng cách tổng quát hóa (Generalization Gap).

**Các Điều kiện Tối ưu:**

1. **Dữ liệu có Tỷ lệ Nhiễu Nhãn cao (Noisy Labels):**
   Mạng sâu có thứ tự học ưu tiên (Spectral Bias): học các quy luật đơn giản, khái quát trước trong các epoch đầu và chỉ bắt đầu học thuộc các mẫu nhiễu ở các epoch muộn. Dừng sớm ngắt quá trình huấn luyện đúng thời điểm trước khi mạng bắt đầu ghi nhớ nhiễu.
2. **Chế độ Dữ liệu Hạn chế (Low-data Regime):**
   Khi quy mô mẫu $N$ nhỏ so với số tham số $P$, nguy cơ quá khớp tăng vọt; dừng sớm là chốt chặn hữu hiệu nhất ngăn ngừa việc bão hòa rủi ro thực nghiệm.
3. **Sự Không tương thích của Lịch trình Tốc độ học (Suboptimal Learning Rate Schedule):**
   Khi chưa tìm được lịch suy giảm tốc độ học tối ưu, dừng sớm ngăn cản mô hình đi sâu vào các thung lũng quá khớp hẹp.

---

### Bài 5: Other benefits of early stopping

**Đề bài:** Đánh giá các lợi ích phụ trợ của kỹ thuật Dừng sớm về mặt kinh tế học tính toán (Green AI), tăng tốc tìm kiếm siêu tham số và ổn định số học.

**Lợi ích Kỹ thuật Bổ trợ:**

1. **Tiết kiệm Chi phí Tính toán và Năng lượng (FLOPs & Carbon Footprint):**
   Cắt giảm từ $30\%$ đến $70\%$ tổng số epoch huấn luyện dư thừa, tiết kiệm chi phí thuê cụm GPU đám mây.
2. **Tăng tốc Tối ưu hóa Siêu tham số (Asynchronous Hyperband / ASHA):**
   Cho phép các bộ điều phối tự động ngắt sớm hàng trăm nhánh thử nghiệm có triển vọng kém để dồn tài nguyên GPU cho các cấu hình kiến trúc tối ưu.
3. **Ổn định Số học:**
   Tránh được các pha bùng nổ gradient muộn khi mô hình bị đẩy vào các vùng cực đoan của hàm mục tiêu.
