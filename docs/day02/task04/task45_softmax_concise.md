# 4.5. Concise Implementation of Softmax Regression: Cài đặt Hồi quy Softmax Tinh gọn và Định dạng Số học

Nguồn tham khảo: [4.5.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/softmax-regression-concise.html#exercises).

---

### Bài 1: Exponential input ranges for floating-point formats

**Đề bài:** Xác định miền đầu vào an toàn của hàm số mũ $\exp(x)$ trước khi kích hoạt hiện tượng tràn trên (Overflow) hoặc tràn dưới (Underflow) trên các định dạng số thực dấu phẩy động chuẩn IEEE-754: FP64, FP32, BFLOAT16, FP16 và TF32.

**Phân tích và Khảo sát Lý thuyết:**

Một định dạng số thực dấu phẩy động được đặc trưng bởi số bit dấu ($1$), số bit mũ ($E$), và số bit định trị ($M$). Ngưỡng tràn trên xảy ra khi $\exp(x) > \text{Max\_Finite\_Float}$, và ngưỡng tràn dưới xảy ra khi $\exp(x) < \text{Min\_Normal\_Float}$ (hoặc $\text{Min\_Subnormal\_Float}$).

$$\text{Miền an toàn của } x \approx [\ln(\text{Min\_Positive}), \, \ln(\text{Max\_Positive})]$$

| Định dạng | Cấu trúc (Dấu, Mũ, Định trị) | $x_{\min}$ (Chạm ngưỡng Subnormal) | $x_{\min}$ (Ngưỡng Normal) | $x_{\max}$ (Ngưỡng Overflow) | Nhận xét chuyên môn |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FP64 (Double)** | $1 + 11 + 52$ | $\approx -744,44$ | $\approx -708,40$ | $\approx +709,78$ | Độ chính xác khoa học cao, chi phí bộ nhớ lớn. |
| **FP32 (Single)** | $1 + 8 + 23$ | $\approx -103,28$ | $\approx -87,34$ | $\approx +88,72$ | Chuẩn mực huấn luyện truyền thống. |
| **BF16 (Bfloat16)**| $1 + 8 + 7$ | $\approx -92,19$ | $\approx -87,34$ | $\approx +88,72$ | Cùng dải số mũ với FP32; ngăn tràn số hiệu quả trong huấn luyện mô hình ngôn ngữ lớn (LLMs). |
| **FP16 (Half)** | $1 + 5 + 10$ | $\approx -16,64$ | $\approx -9,70$ | $\approx +11,09$ | Dải số mũ rất hẹp; dễ tràn số nếu không có kỹ thuật Loss Scaling. |
| **TF32 (TensorFloat)**| $1 + 8 + 10$ | $\approx -92,19$ | $\approx -87,34$ | $\approx +88,72$ | Định dạng tính toán nội bộ trên nhân Tensor Core của kiến trúc NVIDIA Ampere/Hopper. |

---

### Bài 2: Extending INT8 dynamic range

**Đề bài:** Phương pháp mở rộng dải động (Dynamic Range) của kiểu số nguyên 8-bit (INT8) trong kỹ thuật lượng tử hóa mạng nơ-ron (Model Quantization). Phân tích sự biến đổi của các phép toán số học cộng và nhân.

**Phân tích và Lời giải:**

1. **Lượng tử hóa Ánh xạ Tuyến tính Tuyệt đối (Affine / Asymmetric Quantization):**
   Biểu diễn một số thực liên tục $x \in \mathbb{R}$ thông qua số nguyên 8-bit $q \in [-128, 127]$ (hoặc $[0, 255]$) bằng một hệ số tỷ lệ $s \in \mathbb{R}^+$ (scale factor) và điểm không $z \in \mathbb{Z}$ (zero-point):
   $$x \approx s (q - z) \iff q = \operatorname{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, \, q_{\min}, \, q_{\max}\right)$$
   - Để mở rộng dải động mà không tăng số lượng bit, ta tăng hệ số bước lượng tử $s$. Điều này mở rộng khoảng giá trị bao phủ nhưng làm giảm độ phân giải số học.
   - Để hạn chế suy giảm độ chính xác cục bộ, kỹ thuật lượng tử hóa theo khối (Block-wise / Per-channel Quantization) được áp dụng, gán hệ số $s$ riêng biệt cho từng hàng hoặc từng kênh của ma trận trọng số.
2. **Biến đổi Phép toán Số học:**
   - **Phép nhân:** Cho $x_1 = s_1(q_1 - z_1)$ và $x_2 = s_2(q_2 - z_2)$:
     $$x_1 x_2 = s_1 s_2 \left( q_1 q_2 - q_1 z_2 - q_2 z_1 + z_1 z_2 \right)$$
     Tích hai số nguyên 8-bit đòi hỏi bộ tích lũy 32-bit (INT32 Accumulator) để chống tràn số học trước khi thực hiện bước tái lượng tử hóa (Re-quantization) về INT8.
   - **Phép cộng:** Nếu hai tensor có hệ số scale khác nhau ($s_1 \ne s_2$), không thể cộng trực tiếp các số nguyên mà bắt buộc phải thực hiện bước cân chỉnh hệ số tỷ lệ (Rescaling alignment).

---

### Bài 3: Validation accuracy after more epochs

**Đề bài:** Phân tích cơ chế suy giảm độ chính xác kiểm định (Validation Accuracy) khi kéo dài số epoch huấn luyện và đề xuất các giải pháp kỹ thuật khắc phục.

**Phân tích và Lời giải:**

1. **Hiện tượng Quá khớp (Overfitting):**
   Khi tiếp tục tối ưu hóa sau điểm bão hòa, hàm mất mát huấn luyện vẫn tiếp tục giảm nhưng mô hình bắt đầu học thuộc các đặc trưng nhiễu cụ thể của tập train. Biểu hiện cụ thể là sự phân kỳ giữa đường cong Cross-Entropy Loss của tập huấn luyện và tập kiểm định.
2. **Giải pháp kiểm soát chất lượng:**
   - **Cơ chế Dừng sớm (Early Stopping):** Lưu trữ checkpoint có validation accuracy cực đại (hoặc validation loss cực tiểu) và tự động ngắt huấn luyện khi không có sự cải thiện sau một khoảng thời gian kiên nhẫn (patience).
   - **Điều chuẩn mô hình:** Bổ sung hệ số suy giảm trọng số ($L_2$ weight decay), kỹ thuật làm mịn nhãn (Label Smoothing), hoặc tăng cường dữ liệu (Data Augmentation).

---

### Bài 4: Comparing learning rates

**Đề bài:** So sánh động thái của các đường cong mất mát (Loss Curves) dưới các mức tốc độ học khác nhau và thiết lập tiêu chí lựa chọn tốc độ học tối ưu.

**Phân tích Động học:**

- **Tốc độ học quá nhỏ ($\eta \ll \eta^*$):** Đường cong mất mát suy giảm tuyến tính cực kỳ chậm, tiêu tốn tài nguyên tính toán và có nguy cơ mắc kẹt tại các vùng yên ngựa phẳng.
- **Tốc độ học tối ưu ($\eta \approx \eta^*$):** Mất mát suy giảm nhanh theo cấp số nhân trong các epoch đầu và tiệm cận mượt mà về cực tiểu toàn cục.
- **Tốc độ học quá lớn ($\eta \gg \eta^*$):** Xảy ra hiện tượng dao động dữ dội xung quanh khe tối ưu (oscillations), hoặc bùng nổ phân kỳ mất mát ($\mathcal{L} \to \infty$ hoặc tràn số `NaN`).
