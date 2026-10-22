# 3.6. Generalization: Khả năng Khái quát hóa trong Học máy

Nguồn tham khảo: [3.6.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-regression/generalization.html#exercises).

---

### Bài 1: Exact polynomial regression

**Đề bài:** Phân tích các điều kiện toán học cần và đủ để bài toán hồi quy đa thức giải được nghiệm giải tích chính xác hoàn hảo.

**Phân tích và Lời giải:**

Xét bài toán khớp đa thức bậc $d$: $y = \sum_{j=0}^d w_j x^j$. Để tìm được nghiệm giải tích hoàn hảo khôi phục chính xác các hệ số thực $w_j^*$:
1. **Dữ liệu thuần nhất (Noise-free regime):** Tập dữ liệu quan sát không chứa nhiễu đo lường ngẫu nhiên ($\epsilon_i = 0, \forall i$).
2. **Khớp đúng lớp giả thuyết (Well-specified hypothesis class):** Quy luật vật lý sinh ra dữ liệu thực sự là một đa thức bậc không vượt quá $d$.
3. **Điều kiện ma trận Vandermonde không suy biến:** Cần ít nhất $n \ge d + 1$ điểm dữ liệu đầu vào phân biệt đôi một ($x_i \ne x_j, \forall i \ne j$). Khi đó ma trận thiết kế Vandermonde đạt hạng đầy đủ ($\operatorname{rank}(\mathbf{X}) = d+1$), đảm bảo tồn tại duy nhất một đa thức nội suy đi qua toàn bộ các điểm.
- *Hiện tượng Runge trong thực tế:* Khi bậc đa thức $d$ tăng cao, ma trận Vandermonde có số điều kiện (condition number) bùng nổ cấp số nhân, dẫn đến sự mất ổn định số học và hiện tượng dao động dữ dội ở biên (Runge's phenomenon).

---

### Bài 2: Dependent observations that violate IID

**Đề bài:** Phân tích các tình huống thực tiễn mà giả định các biến quan sát độc lập cùng phân phối (Independent and Identically Distributed - IID) bị vi phạm nghiêm trọng.

**Phân tích và Lời giải:**

Trong học máy lý thuyết, giả định IID là tiền đề cho các cận khái quát hóa (PAC bounds). Trong thực tế, giả định này thường xuyên bị phá vỡ:
1. **Tự tương quan thời gian trong chuỗi thời gian (Temporal Autocorrelation):** Giá cổ phiếu, tín hiệu điện tâm đồ (ECG), số đo cảm biến IoT tại thời điểm $t$ phụ thuộc trực tiếp vào trạng thái tại thời điểm $t-1$.
2. **Tương quan không gian (Spatial Autocorrelation - Định luật Tobler):** Dữ liệu địa lý, hình ảnh viễn thám, dự báo thời tiết có các điểm lân cận về không gian mang đặc trưng tương đồng cao.
3. **Phân cụm thực thể (Group / Cluster Dependencies):** Nhiều bản ghi y tế đến từ cùng một bệnh nhân; nhiều hành vi mua sắm đến từ cùng một người dùng mạng xã hội.
4. **Cấu trúc ngữ cảnh tuần tự (Markovian / Syntactic Dependencies):** Các từ trong một câu văn bản hoặc các khung hình (frames) liên tiếp trong một đoạn video.
5. **Dữ liệu đồ thị mạng lưới (Network / Graph Dependencies):** Các nút trong mạng xã hội hoặc mạng trích dẫn bài báo khoa học.

*Hệ quả kỹ thuật:* Nếu chia ngẫu nhiên (random train-test split) trên dữ liệu phụ thuộc, hiện tượng rò rỉ thông tin (data leakage) sẽ xảy ra, làm cho kết quả đánh giá trên tập kiểm thử lạc quan một cách giả tạo nhưng thất bại hoàn toàn khi triển khai thực tế. Bắt buộc phải áp dụng các chiến lược chia dữ liệu theo thời gian (Time-series split) hoặc theo cụm nhóm (GroupKFold).

---

### Bài 3: Zero training and generalization error

**Đề bài:** Phân tích khả năng đạt sai số huấn luyện bằng 0 ($\mathcal{R}_{\text{emp}} = 0$) và các điều kiện để sai số khái quát hóa thực tế bằng 0 ($\mathcal{R}_{\text{true}} = 0$).

**Phân tích và Lời giải:**

1. **Sai số huấn luyện bằng 0 ($\mathcal{R}_{\text{emp}} = 0$):**
   Hoàn toàn khả thi khi dung lượng mô hình vượt ngưỡng nội suy (overparameterized regime, số tham số lớn hơn số mẫu), dữ liệu không chứa các điểm mâu thuẫn (cùng $\mathbf{x}$ nhưng khác $y$). Tuy nhiên, điều này không đồng nghĩa với việc mô hình học được quy luật tổng quát; mô hình có thể chỉ ghi nhớ máy móc (memorization) toàn bộ nhiễu của tập huấn luyện.
2. **Sai số khái quát hóa bằng 0 ($\mathcal{R}_{\text{true}} = 0$):**
   Chỉ có thể đạt được trên lý thuyết khi và chỉ khi:
   - Bài toán là bài toán tất định (deterministic), không chứa nhiễu ngẫu nhiên không thể khử (Bayes error rate bằng 0: $\mathcal{R}^* = 0$).
   - Hàm mục tiêu thực tế nằm trọn vẹn trong lớp giả thuyết của mô hình ($f^* \in \mathcal{H}$).
   - Không có sự dịch chuyển phân phối giữa môi trường huấn luyện và môi trường kiểm thử ($P_{\text{train}}(\mathbf{x}, y) = P_{\text{test}}(\mathbf{x}, y)$).

---

### Bài 4: Cost of K-fold cross-validation

**Đề bài:** Đánh giá độ phức tạp tính toán và chi phí phần cứng của phương pháp kiểm định chéo $K$-gấp ($K$-Fold Cross-Validation) trong các bài toán học sâu.

**Phân tích định lượng:**

Phương pháp kiểm định chéo $K$-fold chia tập dữ liệu thành $K$ phần bằng nhau, luân phiên huấn luyện trên $K-1$ phần và kiểm định trên 1 phần còn lại.
- **Độ phức tạp tính toán:** Chi phí huấn luyện tăng gấp chính xác $K$ lần so với phương pháp chia tập đơn (Hold-out validation):
  $$\text{FLOPs}_{\text{CV}} \approx K \times \text{FLOPs}_{\text{train}}$$
- **Thách thức trong Học sâu:** Đối với các mô hình học sâu quy mô lớn (như Large Language Models hay thị giác máy tính quy mô lớn) nơi một chu trình huấn luyện tốn hàng trăm ngàn giờ GPU và hàng triệu USD, việc nhân $K$ lần chi phí là bất khả thi về mặt kinh tế. Do đó, $K$-fold CV chủ yếu được áp dụng cho dữ liệu dạng bảng quy mô vừa/nhỏ hoặc các bài toán y tế có cỡ mẫu hạn chế.

---

### Bài 5: Bias of the K-fold cross-validation estimate

**Đề bài:** Phân tích bản chất toán học về độ chệch (statistical bias) của ước lượng sai số thông qua $K$-Fold Cross-Validation.

**Phân tích và Lời giải:**

1. **Độ chệch bi quan (Pessimistic Bias về kích thước tập huấn luyện):**
   Ở mỗi fold, mô hình chỉ được huấn luyện trên $\frac{K-1}{K} n$ mẫu dữ liệu, ít hơn quy mô toàn bộ tập huấn luyện $n$ mà mô hình cuối cùng sẽ sử dụng. Do đường cong học tập (learning curve) thường có sai số giảm đơn điệu theo cỡ mẫu, sai số trung bình của $K$-fold thường cao hơn (bi quan hơn) so với sai số kỳ vọng của mô hình được huấn luyện trên toàn bộ $n$ mẫu. Khi $K$ càng lớn (ví dụ $K = 10$ hoặc Leave-One-Out $K = n$), độ chệch bi quan này càng được thu hẹp.
2. **Độ chệch lạc quan (Optimistic Selection Bias do dò quét siêu tham số):**
   Nếu quy trình tìm kiếm siêu tham số (Hyperparameter Tuning) sử dụng trực tiếp điểm số trung bình của $K$-fold để chọn mô hình tốt nhất mà không có tập kiểm thử độc lập ngoài (Nested CV), điểm số tốt nhất được báo cáo sẽ bị chệch lạc quan do hiện tượng quá khớp tập kiểm định (overfitting the validation splits).

---

### Bài 6: Limitations of VC dimension

**Đề bài:** Phân tích những hạn chế của số chiều Vapnik–Chervonenkis (VC Dimension) trong việc đo lường độ phức tạp và khả năng khái quát hóa của mạng nơ-ron sâu hiện đại.

**Phân tích và Lời giải:**

1. **Tính chất phân phối độc lập (Distribution-free nature):**
   Số chiều VC đo lường khả năng phân tách nhãn tối đa trên trường hợp xấu nhất (worst-case scenario) đối với một cấu hình điểm bất kỳ. Trong thực tế, dữ liệu tự nhiên (ảnh, âm thanh, văn bản) nằm trên các đa tạp con số chiều thấp (low-dimensional manifolds) với cấu trúc phân phối rất đặc thù chứ không phân bố ngẫu nhiên trong toàn bộ không gian.
2. **Sự không nhạy với độ lớn trọng số và lề phân tách (Scale-invariance):**
   VC dimension của lớp các bộ phân loại tuyến tính trong $\mathbb{R}^d$ là $d + 1$, hoàn toàn không phụ thuộc vào độ lớn của vector trọng số $\|\mathbf{w}\|$ hay độ rộng của lề phân tách (margin $\gamma$). Trong học sâu, hai mạng có cùng kiến trúc tham số có thể có khả năng khái quát hóa hoàn toàn khác nhau phụ thuộc vào chuẩn chuẩn hóa trọng số ($L_2$ norm) và phương pháp điều chuẩn.
3. **Các lý thuyết thay thế hiện đại:**
   Lý thuyết học máy hiện đại ưu tiên sử dụng **Độ phức tạp Rademacher** (Rademacher Complexity), **Cận dựa trên chuẩn trọng số** (Norm-based Generalization Bounds), và **Lý thuyết PAC-Bayes** để giải thích khả năng khái quát hóa vượt trội của mạng nơ-ron cực lớn (overparameterized deep networks).

---

### Bài 7: Justifying the need for more data

**Đề bài:** Đề xuất phương pháp khoa học sử dụng đường cong học tập (Learning Curves) thông qua việc giảm kích thước tập dữ liệu để chứng minh sự cần thiết của việc đầu tư thu thập thêm dữ liệu huấn luyện.

**Quy trình Thực nghiệm:**

1. **Thiết lập chuỗi thực nghiệm trên các tập con giảm dần:**
   Cố định kiến trúc mô hình và pipeline siêu tham số. Huấn luyện mô hình trên các tập con dữ liệu có quy mô tăng dần: $S \in \{10\%, 20\%, 40\%, 60\%, 80\%, 100\%\} \times N$.
2. **Đo lường sai số trên tập kiểm định cố định:**
   Ghi nhận sai số huấn luyện $\mathcal{R}_{\text{train}}(S)$ và sai số kiểm định $\mathcal{R}_{\text{val}}(S)$ độc lập qua nhiều hạt giống ngẫu nhiên để vẽ đường cong học tập cùng khoảng tin cậy.
3. **Biện luận định lượng:**
   - Nếu đường cong $\mathcal{R}_{\text{val}}(S)$ tiếp tục suy giảm với độ dốc đáng kể trên thang đo log-log khi tiến từ $80\%$ lên $100\%$ và khoảng cách tổng quát hóa $(\mathcal{R}_{\text{val}} - \mathcal{R}_{\text{train}})$ vẫn còn lớn, điều đó chứng minh mô hình đang ở chế độ bị giới hạn bởi dữ liệu (data-limited regime). Việc ngoại suy đường cong theo luật lũy thừa (Power Law Scaling) sẽ cung cấp luận cứ kinh tế chính xác về việc mở rộng quy mô dữ liệu sẽ làm giảm sai số bao nhiêu phần trăm.
