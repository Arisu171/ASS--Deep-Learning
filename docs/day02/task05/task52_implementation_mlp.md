# 5.2. Implementation of Multilayer Perceptrons: Cài đặt Perceptron Đa lớp và Động học Tối ưu hóa

Nguồn tham khảo: [5.2.4. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/mlp-implementation.html#exercises).

---

### Bài 1: Number of hidden units

**Đề bài:** Khảo sát ảnh hưởng của số lượng nơ-ron tầng ẩn ($h \in \{8, 32, 64, \dots\}$) đến khả năng khái quát hóa và độ chính xác kiểm định. Thiết lập phương pháp thực nghiệm khoa học để lựa chọn độ rộng tối ưu.

**Phân tích và Thực nghiệm:**

Độ rộng của tầng ẩn quy định trực tiếp chiều không gian đặc trưng ẩn và dung lượng tham số của mô hình:
- **Số nơ-ron quá nhỏ ($h < h^*$):** Không gian biểu diễn bị thắt hẹp, mô hình không đủ sức mạnh biểu diễn các biên quyết định phi tuyến phức tạp (hiện tượng dưới khớp - Underfitting).
- **Số nơ-ron quá lớn ($h \gg h^*$):** Dung lượng mô hình vượt xa thông tin mẫu, dễ ghi nhớ chi tiết nhiễu và đòi hỏi chi phí tính toán lớn (hiện tượng quá khớp - Overfitting).

**Kết quả Thực nghiệm Mô phỏng:**
Thực nghiệm đánh giá mạng MLP-ReLU trên tập dữ liệu tổng hợp 2D phi tuyến với 10 epoch, bộ tối ưu SGD ($\eta = 0.1$):
- $h = 8$ nơ-ron ẩn: Validation Accuracy đạt $61,3\%$ (bị giới hạn do dưới khớp).
- $h = 32$ nơ-ron ẩn: Validation Accuracy tăng vọt lên $94,4\%$.
- $h = 64$ nơ-ron ẩn: Validation Accuracy đạt $98,1\%$.

Biểu đồ bên trái ở hình dưới minh họa sự gia tăng độ chính xác kiểm định khi mở rộng độ rộng tầng ẩn; biểu đồ bên phải so sánh hiệu năng các cấu hình phân bổ Dropout.

![Validation accuracy theo số nút ẩn và vị trí dropout](image5_experiment.png)

---

### Bài 2: Adding another hidden layer

**Đề bài:** Phân tích sự đánh đổi khi tăng độ sâu của mạng (thêm một tầng ẩn thứ hai) về mặt lý thuyết biểu diễn phân cấp (Hierarchical Representations) và tối ưu hóa gradient.

**Phân tích và Lời giải:**

1. **Ưu thế lý thuyết của Độ sâu (Depth Efficiency):**
   Theo các định lý về độ phức tạp mạch nơ-ron, một mạng sâu có thể biểu diễn một số lớp hàm tuần hoàn hoặc phân cấp (như bài toán chẵn lẻ Parity hoặc nhân ma trận) với số lượng nơ-ron tăng tuyến tính $\mathcal{O}(L)$, trong khi một mạng nông đơn tầng đòi hỏi số nơ-ron bùng nổ theo cấp số nhân $\mathcal{O}(2^d)$. Tầng ẩn thứ hai cho phép học các tổ hợp đặc trưng trừu tượng cấp cao hơn từ các đặc trưng nguyên thủy của tầng ẩn thứ nhất.
2. **Thách thức tối ưu hóa:**
   Mỗi tầng ẩn bổ sung làm phức tạp hóa cảnh quan mất mát (Loss Landscape), tạo ra nhiều điểm yên ngựa và làm tăng nguy cơ suy giảm gradient. Cần kết hợp kỹ thuật khởi tạo trọng số chuẩn mực (He/Xavier) và chuẩn hóa kích hoạt.

---

### Bài 3: A single-neuron bottleneck

**Đề bài:** Phân tích lý do việc thiết kế một tầng ẩn chỉ có duy nhất một nơ-ron ($h = 1$) ở giữa mạng sâu là một khiếm khuyết kiến trúc nghiêm trọng.

**Phân tích và Lời giải:**

Xét kiến trúc mạng: $\mathbf{x} \in \mathbb{R}^d \xrightarrow{\mathbf{W}_1} z \in \mathbb{R}^1 \xrightarrow{\sigma} a \in \mathbb{R}^1 \xrightarrow{\mathbf{W}_2} \hat{\mathbf{y}} \in \mathbb{R}^K$.
1. **Nút thắt thông tin nghiêm trọng (Information Bottleneck):**
   Toàn bộ vector đặc trưng đa chiều của mẫu đầu vào bị nén ép cưỡng bức thành một đại lượng vô hướng đơn lẻ $a = \sigma(\mathbf{x}\mathbf{w}_1 + b_1) \in \mathbb{R}$.
2. **Suy biến hạng ma trận:**
   Biến đổi của tầng kế tiếp $\hat{\mathbf{y}} = a \mathbf{w}_2^\top + \mathbf{b}_2$ chỉ có thể tạo ra các vector dự đoán nằm trên một đường thẳng (không gian affine 1 chiều) trong $\mathbb{R}^K$. Toàn bộ khả năng phân biệt độc lập giữa nhiều lớp đối tượng bị triệt tiêu hoàn toàn.

---

### Bài 4: Learning rate and number of epochs

**Đề bài:** Khảo sát mối quan hệ tương hỗ giữa tốc độ học $\eta$ và số lượng epoch huấn luyện trong việc định hình quỹ đạo tối ưu hóa của MLP.

**Phân tích và Lời giải:**

- **Tốc độ học nhỏ kèm số epoch lớn:** Cho phép thuật toán SGD thăm dò cảnh quan mất mát một cách mịn màng, giảm phương sai dao động nhưng tốn kém tài nguyên tính toán.
- **Tốc độ học lớn kèm số epoch nhỏ:** Tăng tốc độ hội tụ ban đầu nhưng có thể dao động quanh khe tối ưu mà không đạt được điểm cực tiểu chính xác.
- *Nguyên lý Scaling:* Để tối ưu hóa hiệu quả, việc tăng kích thước lô dữ liệu (Mini-batch size) thường đòi hỏi tăng tuyến tính hoặc căn bậc hai tốc độ học ($\eta \propto \sqrt{B}$ hoặc $\eta \propto B$) kết hợp với kỹ thuật khởi động mềm (Learning Rate Warmup).

---

### Bài 5: Joint hyperparameter optimization

**Đề bài:** Trình bày các chiến lược tối ưu hóa đồng thời không gian siêu tham số đa chiều (Joint Hyperparameter Search) cho mạng MLP: tốc độ học, số tầng, số nơ-ron và hệ số điều chuẩn.

**Phương pháp Luận Khoa học:**

1. **Tìm kiếm Ngẫu nhiên (Random Search - Bergstra & Bengio):**
   Vượt trội hoàn toàn so với Tìm kiếm theo Lưới (Grid Search) trong không gian nhiều chiều, do các siêu tham số có độ nhạy (importance) khác nhau và Random Search thăm dò nhiều giá trị phân biệt hơn trên các chiều quan trọng.
2. **Tối ưu hóa Bayes (Bayesian Optimization via Gaussian Processes / TPE):**
   Xây dựng mô hình đại diện xác suất (Surrogate Model) để dự đoán hiệu năng và sử dụng hàm thu nhận (Acquisition Function như Expected Improvement - EI) để cân bằng giữa khai thác (Exploitation) và khám phá (Exploration).
3. **Giải thuật Cắt nửa Liên tiếp (Successive Halving / Hyperband):**
   Phân bổ ngân sách tính toán (epochs) linh hoạt: khởi tạo hàng trăm cấu hình với số epoch nhỏ, tự động loại bỏ các cấu hình kém hiệu quả nhất và chỉ cấp ngân sách huấn luyện đầy đủ cho nhóm cấu hình tiềm năng.

---

### Bài 6: Scratch versus framework speed

**Đề bài:** So sánh hiệu năng thực thi giữa việc cài đặt mạng nơ-ron từ đầu bằng vòng lặp Python thuần và việc sử dụng framework tối ưu hóa (PyTorch / C++ backend).

**Phân tích Hiệu năng:**

- **Cài đặt từ đầu (Pure Python Loops):** Bị giới hạn nghiêm trọng bởi cơ chế thông dịch từng dòng của Python, overhead của con trỏ kiểu động và khóa thông dịch toàn cục (GIL). Hiệu năng tính toán cực kỳ thấp ($\mathcal{O}(10^2 - 10^3)$ lần chậm hơn).
- **Framework chuyên dụng (PyTorch):** Sử dụng các thư viện đại số tuyến tính tối ưu viết bằng C++/Fortran (Intel MKL / OpenBLAS cho CPU; cuBLAS / cuDNN cho GPU NVIDIA). Các thư viện này tận dụng tối đa tập lệnh vector hóa SIMD (AVX-512) và kiến trúc nhân Tensor Cores, giúp thông lượng tính toán đạt hàng TeraFLOPs.

---

### Bài 7: Aligned and misaligned matrix multiplication

**Đề bài:** Đo lường độ trễ của phép nhân ma trận vuông $\mathbf{A} \mathbf{B}$ với kích thước $n \in \{1024, 1025, 1026, 1028, 1032\}$. Phân tích ảnh hưởng của sự căn chỉnh bộ nhớ (Memory Alignment) và kiến trúc dòng bộ nhớ đệm (Cache Line).

**Phân tích Cơ chế Phần cứng:**

1. **Căn chỉnh địa chỉ bộ nhớ (Memory Alignment & Tiling):**
   - Các dòng cache CPU hiện đại có kích thước tiêu chuẩn là 64 bytes (tương đương 16 phần tử `float32`).
   - Các kernel nhân ma trận hiệu năng cao (GEMM) chia ma trận thành các khối con (tiles / blocks) có kích thước là lũy thừa của 2 (ví dụ $16 \times 16$, $32 \times 32$, $64 \times 64$).
   - Khi kích thước ma trận là bội số của kích thước khối (như $n = 1024$), dữ liệu được căn chỉnh hoàn hảo trên các ranh giới cache, giúp tối ưu hóa truy cập trực tiếp và loại bỏ xung đột địa chỉ (Cache Conflict Misses).
2. **Kích thước không căn chỉnh ($n = 1025$):**
   Đòi hỏi kernel phải bổ sung các khối đệm (Padding) hoặc xử lý biên lặt vặt (Boundary conditions), làm phát sinh hiện tượng phân mảnh truy xuất và giảm hiệu quả nạp SIMD.

---

### Bài 8: Activation functions

**Đề bài:** Phân loại và so sánh ưu nhược điểm của các hàm kích hoạt chuẩn tắc trong học sâu: ReLU, Leaky ReLU, Tanh, GELU và SiLU/Swish.

| Hàm Kích hoạt | Công thức Toán học | Ưu điểm Nổi bật | Hạn chế Cần lưu ý |
| :--- | :---: | :--- | :--- |
| **ReLU** | $\max(0, x)$ | Tính toán cực nhanh, gradient không suy giảm ở miền dương. | Hiện tượng nơ-ron chết (Dying ReLU) khi $x \le 0$. |
| **Leaky ReLU / PReLU** | $\max(\alpha x, x)$ | Khắc phục hoàn toàn nơ-ron chết nhờ duy trì gradient nhỏ ở miền âm. | Bổ sung siêu tham số cần tinh chỉnh. |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Đối xứng qua gốc tọa độ 0 (zero-centered), giúp ổn định cập nhật gradient. | Bão hòa ở hai cực, gây biến mất gradient. |
| **GELU** | $x \Phi(x) = x P(X \le x)$ | Làm mượt xác suất, chuẩn mực trong các mô hình Transformers (BERT, GPT). | Tốn kém chi phí tính toán hơn ReLU. |
| **SiLU / Swish** | $x \cdot \sigma(x)$ | Tự điều tiết độ dốc, không đơn điệu, tối ưu trong các mạng thị giác hiện đại (EfficientNet, YOLO). | Chi phí tính toán hàm sigmoid. |

---

### Bài 9: Weight initialization

**Đề bài:** Phân tích nguyên lý bảo toàn phương sai của phương pháp khởi tạo Xavier/Glorot và He/Kaiming.

**Phân tích toán học:**

Để ngăn chặn hiện tượng bùng nổ hoặc tiêu biến tín hiệu kích hoạt khi lan truyền qua $L$ tầng sâu, mục tiêu là bảo toàn phương sai kích hoạt giữa các tầng: $\operatorname{Var}(h_l) = \operatorname{Var}(h_{l-1})$.

1. **Khởi tạo Xavier / Glorot (Dành cho Tanh / Sigmoid):**
   $$\mathbf{W}_{i, j} \sim \mathcal{N}\left(0, \, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{hoặc} \quad \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$
2. **Khởi tạo He / Kaiming (Dành riêng cho ReLU):**
   Do ReLU triệt tiêu một nửa số kích hoạt âm về 0, phương sai của tín hiệu bị giảm đi một nửa ($\mathbb{E}[\sigma(z)^2] = \frac{1}{2}\operatorname{Var}(z)$). Để bù đắp độ suy giảm này, phương sai khởi tạo của ma trận trọng số phải được nhân đôi:
   $$\mathbf{W}_{i, j} \sim \mathcal{N}\left(0, \, \frac{2}{n_{\text{in}}}\right)$$
   Việc khởi tạo sai phương sai (như dùng chuẩn tắc $\mathcal{N}(0, 1)$ cho mạng 50 tầng) sẽ khiến gradient bùng nổ hoặc biến mất hoàn toàn ngay từ epoch đầu tiên.
