# 5.3. Forward Propagation, Backward Propagation, and Computational Graphs: Lan truyền Xuôi, Lan truyền Ngược và Đồ thị Tính toán

Nguồn tham khảo: [5.3.6. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html#exercises).

---

### Bài 1: Shape of a scalar function's gradient

**Đề bài:** Xác định kích thước của vector gradient $\nabla_{\mathbf{X}} f$ khi biến đầu vào của hàm vô hướng $f(\mathbf{X}): \mathbb{R}^{n \times m} \to \mathbb{R}$ là một ma trận hai chiều.

**Phân tích và Lời giải:**

Theo quy ước giải tích ma trận (Matrix Calculus / Layout Convention):
Gradient của một hàm mục tiêu vô hướng $f$ đối với một ma trận $\mathbf{X} \in \mathbb{R}^{n \times m}$ là một ma trận có cùng kích thước $n \times m$, trong đó phần tử tại vị trí $(i, j)$ là đạo hàm riêng của $f$ theo biến vô hướng tương ứng:

$$[\nabla_{\mathbf{X}} f]_{i, j} = \frac{\partial f}{\partial X_{i, j}}, \quad \forall i \in \{1, \dots, n\}, \, j \in \{1, \dots, m\}$$

Quy ước này bảo toàn cấu trúc hình học của không gian tham số, cho phép thực hiện trực tiếp phép cập nhật gradient descent dạng ma trận: $\mathbf{X}^{(t+1)} = \mathbf{X}^{(t)} - \eta \nabla_{\mathbf{X}} f$.

---

### Bài 2: Add hidden bias and derive forward/backward equations

**Đề bài:** Thiết lập hệ phương trình lan truyền xuôi và lan truyền ngược chi tiết cho mạng Perceptron hai tầng có chứa vector độ chệch tầng ẩn $\mathbf{b}_1 \in \mathbb{R}^h$, hàm kích hoạt $\sigma$, hàm mất mát bình phương Frobenius và số hạng điều chuẩn trọng số (không điều chuẩn bias).

**Chứng minh Giải tích:**

Xét mini-batch đầu vào $\mathbf{X} \in \mathbb{R}^{m \times d}$, ma trận nhãn mục tiêu $\mathbf{T} \in \mathbb{R}^{m \times q}$.
Các tham số: $\mathbf{W}_1 \in \mathbb{R}^{d \times h}$, $\mathbf{b}_1 \in \mathbb{R}^h$, $\mathbf{W}_2 \in \mathbb{R}^{h \times q}$.

1. **Lan truyền xuôi (Forward Propagation):**
   $$\mathbf{Z}_1 = \mathbf{X} \mathbf{W}_1 + \mathbf{1}_m \mathbf{b}_1^\top \in \mathbb{R}^{m \times h}$$
   $$\mathbf{H} = \sigma(\mathbf{Z}_1) \in \mathbb{R}^{m \times h}$$
   $$\hat{\mathbf{Y}} = \mathbf{H} \mathbf{W}_2 \in \mathbb{R}^{m \times q}$$
   Hàm mất mát toàn cục có điều chuẩn $L_2$ (Weight Decay với hệ số $\lambda > 0$):
   $$\mathcal{L} = \frac{1}{2m} \|\hat{\mathbf{Y}} - \mathbf{T}\|_F^2 + \frac{\lambda}{2} \left( \|\mathbf{W}_1\|_F^2 + \|\mathbf{W}_2\|_F^2 \right)$$

2. **Lan truyền ngược (Backward Propagation):**
   - **Gradient đối với đầu ra dự đoán:**
     $$\mathbf{G}_2 = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{Y}}} = \frac{1}{m} (\hat{\mathbf{Y}} - \mathbf{T}) \in \mathbb{R}^{m \times q}$$
   - **Gradient đối với ma trận trọng số $\mathbf{W}_2$:**
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \mathbf{H}^\top \mathbf{G}_2 + \lambda \mathbf{W}_2 \in \mathbb{R}^{h \times q}$$
   - **Lan truyền gradient ngược về tầng ẩn:**
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{H}} = \mathbf{G}_2 \mathbf{W}_2^\top \in \mathbb{R}^{m \times h}$$
   - **Gradient đối với pre-activation $\mathbf{Z}_1$:**
     Áp dụng quy tắc nhân Hadamard (từng phần tử) với đạo hàm kích hoạt:
     $$\mathbf{G}_1 = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_1} = \left(\mathbf{G}_2 \mathbf{W}_2^\top\right) \odot \sigma'(\mathbf{Z}_1) \in \mathbb{R}^{m \times h}$$
   - **Gradient đối với ma trận trọng số $\mathbf{W}_1$:**
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \mathbf{X}^\top \mathbf{G}_1 + \lambda \mathbf{W}_1 \in \mathbb{R}^{d \times h}$$
   - **Gradient đối với vector độ chệch $\mathbf{b}_1$:**
     Do $\mathbf{b}_1$ được broadcast trên toàn bộ $m$ mẫu của batch:
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \mathbf{G}_1^\top \mathbf{1}_m = \sum_{i=1}^m [\mathbf{G}_1]_{i, :} \in \mathbb{R}^h$$

---

### Bài 3: Memory footprint for training and prediction

**Đề bài:** Phân tích định lượng dung lượng bộ nhớ RAM/VRAM tiêu tốn trong pha huấn luyện (Training) so với pha suy luận (Inference/Prediction) của mạng Perceptron đa lớp.

**Phân tích So sánh Bộ nhớ:**

| Thành phần Bộ nhớ | Pha Huấn luyện (Training) | Pha Suy luận (Inference) |
| :--- | :--- | :--- |
| **Trọng số mô hình (Model Parameters)** | Lưu trữ tham số $\boldsymbol{\theta}$ ($P$ phần tử). | Lưu trữ tham số $\boldsymbol{\theta}$ ($P$ phần tử). |
| **Gradient của tham số** | Lưu trữ $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ ($P$ phần tử). | **Không cần** (Giải phóng hoàn toàn). |
| **Trạng thái Bộ tối ưu (Optimizer States)** | Lưu trữ các biến phụ tá: Adam cần 2 moment $m_t, v_t$ ($2P$ phần tử). | **Không cần**. |
| **Bộ đệm kích hoạt (Activation Tape)** | **Bắt buộc lưu vết toàn bộ** các tensor kích hoạt trung gian $\{\mathbf{Z}_l, \mathbf{H}_l\}_{l=1}^L$ của toàn bộ mini-batch kích thước $m$ để tính đạo hàm ngược ($\mathcal{O}(m \sum h_l)$). | Chỉ lưu kích hoạt tạm thời của 2 tầng liên tiếp; bộ đệm được tái sử dụng ngay lập tức ($\mathcal{O}(m \max h_l)$). |

*Kết luận:* Pha huấn luyện tiêu tốn bộ nhớ gấp từ **4 đến 10 lần** so với pha suy luận. Để giảm tải bộ nhớ kích hoạt trong huấn luyện mạng cực sâu, kỹ thuật **Kiểm tra chốt gradient** (Gradient Checkpointing / Activation Recomputation) được áp dụng: chỉ lưu một số chốt kích hoạt và tái tính toán các kích hoạt còn lại trong pha backward.

---

### Bài 4: Second derivatives and computational graph

**Đề bài:** Phân tích sự biến đổi của đồ thị tính toán khi tính đạo hàm cấp hai (Hessian-Vector Products) và đánh giá chi phí tính toán.

**Phân tích và Lời giải:**

Để tính toán đạo hàm cấp hai thông qua cơ chế tự động vi phân:
1. Trong pha backward đầu tiên, framework phải thiết lập cờ `create_graph=True`. Điều này chỉ thị cho hệ thống không chỉ tính gradient số học mà còn xây dựng một đồ thị tính toán phụ (Derivative Computational Graph) cho chính các phép toán tạo ra gradient đó.
2. Đồ thị mở rộng này đòi hỏi lượng bộ nhớ tăng gấp đôi để lưu giữ các nút đạo hàm trung gian.
3. **Kỹ thuật Tích Hessian-Vector (Pearlmutter Trick / $R$-operator):**
   Thay vì tính ma trận Hessian đầy đủ $\mathbf{H} \in \mathbb{R}^{P \times P}$ tốn $\mathcal{O}(P^2)$ bộ nhớ và $\mathcal{O}(P^3)$ thời gian, tích $\mathbf{H}\mathbf{v} = \nabla_{\boldsymbol{\theta}}(\nabla_{\boldsymbol{\theta}}\mathcal{L}^\top \mathbf{v})$ có thể được tính toán với độ phức tạp thời gian và bộ nhớ chỉ tương đương $\mathcal{O}(P)$ thông qua hai lượt backward liên tiếp.

---

### Bài 5: Partitioning a large graph across GPUs

**Đề bài:** Phân tích các chiến lược phân vùng đồ thị tính toán quy mô lớn trên hệ thống phân tán đa GPU (Song song mô hình vs Song song đường ống) và so sánh với phương án giảm kích thước mini-batch.

**Các Chiến lược Phân vùng:**

1. **Song song theo Tầng / Đường ống (Pipeline Parallelism - GPipe / Megatron-LM):**
   Chia các tầng của mạng sâu cho các GPU nối tiếp: GPU 0 giữ tầng 1–10, GPU 1 giữ tầng 11–20.
   - *Ưu điểm:* Cho phép huấn luyện mô hình có kích thước trọng số vượt quá VRAM của một GPU đơn lẻ.
   - *Thách thức:* Xuất hiện hiện tượng "Bong bóng đường ống" (Pipeline Bubble) khi các GPU sau phải chờ đợi dữ liệu kích hoạt từ các GPU trước. Cần chia mini-batch thành các micro-batches để cải thiện hiệu suất chồng lấn tính toán.
2. **Song song Tensor (Tensor Parallelism):**
   Chia nhỏ ma trận trọng số của từng tầng tuyến tính theo hàng (Row Parallel) hoặc theo cột (Column Parallel) và đồng bộ thông qua toán tử `All-Reduce`.
3. **So sánh với giải pháp giảm Mini-batch:**
   - Giảm batch size chỉ giải quyết được vấn đề quá tải bộ nhớ kích hoạt (Activation Memory) nhưng hoàn toàn bất lực nếu bản thân ma trận trọng số của mô hình đã lớn hơn dung lượng VRAM của thiết bị. Khi đó, phân vùng mô hình đa GPU là giải pháp bắt buộc duy nhất.
