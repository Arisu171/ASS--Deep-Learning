# 4.7. Environment and Distribution Shift: Dịch chuyển Phân phối và Thích ứng Miền trong Học máy

Nguồn tham khảo: [4.7.7. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html#exercises).

---

### Bài 1: Consequences of changing a search engine

**Đề bài:** Phân tích các vòng lặp phản hồi động (Feedback Loops) và sự dịch chuyển phân phối dữ liệu khi thay đổi thuật toán xếp hạng của một công cụ tìm kiếm đối với hành vi người dùng và chiến lược của nhà quảng cáo.

**Phân tích và Lời giải:**

Trong các hệ thống học máy tương tác thực tế (Interactive ML Systems), mô hình không chỉ quan sát thụ động mà hành động của mô hình làm thay đổi trực tiếp môi trường sinh dữ liệu (Performative Prediction):
1. **Dịch chuyển Hiệp biến (Covariate Shift trên hành vi truy vấn):**
   Khi thuật toán thay đổi thứ hạng hiển thị, người dùng thay đổi cách thức tương tác (tỷ lệ nhấp chuột CTR tại các vị trí đầu, thời gian dừng trên trang - dwell time). Nếu kết quả không đáp ứng kỳ vọng ban đầu, người dùng sẽ tự điều chỉnh cú pháp truy vấn, dẫn đến phân phối từ khóa đầu vào $P_{\text{test}}(\mathbf{x})$ dịch chuyển lệch khỏi phân phối huấn luyện $P_{\text{train}}(\mathbf{x})$.
2. **Dịch chuyển Khái niệm (Concept Shift trên cơ chế đấu giá quảng cáo):**
   Các nhà quảng cáo sẽ phản ứng chiến lược (Strategic Manipulation) bằng cách thay đổi giá thầu (bid price), tối ưu hóa từ khóa SEO và định vị lại nội dung landing page để khai thác thuật toán mới. Mối quan hệ xác suất có điều kiện giữa nội dung quảng cáo và ý định mua sắm của người dùng $P(y \mid \mathbf{x})$ bị biến đổi hoàn toàn.
3. **Chiến lược kiểm soát rủi ro:** Bắt buộc phải triển khai thử nghiệm A/B Testing có đối chứng, áp dụng học tăng cường chống thiên lệch (Bandits with Off-policy Evaluation), và liên tục giám sát các chỉ số thống kê (Drift Monitoring) theo thời gian.

---

### Bài 2: Implementing a covariate-shift detector

**Đề bài:** Xây dựng bộ phân loại đối kháng (Domain Discriminator / Classifier) để phát hiện sự dịch chuyển phân phối hiệp biến giữa miền nguồn (Source Domain) và miền đích (Target Domain) sử dụng độ đo ROC-AUC.

**Nguyên lý và Hiện thực hóa scikit-learn:**

Gộp dữ liệu từ hai miền, gán nhãn nhị phân miền $D = 0$ cho miền nguồn và $D = 1$ cho miền đích. Huấn luyện một mô hình phân loại để phân biệt hai miền dựa trên vector đặc trưng $\mathbf{x}$.
- Nếu $P_{\text{source}}(\mathbf{x}) \equiv P_{\text{target}}(\mathbf{x})$: Hai miền hoàn toàn đồng nhất về mặt phân phối, bộ phân loại không thể phân biệt tốt hơn đoán ngẫu nhiên $\implies \text{ROC-AUC} \approx 0,5$.
- Nếu có sự dịch chuyển phân phối đáng kể: Bộ phân loại dễ dàng tách biệt hai miền $\implies \text{ROC-AUC} \gg 0,5$ (tiệm cận $1,0$).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def detect_covariate_shift(X_source: np.ndarray, X_target: np.ndarray) -> float:
    # Gộp mẫu và tạo nhãn miền
    X = np.concatenate([X_source, X_target], axis=0)
    y_domain = np.concatenate([
        np.zeros(len(X_source), dtype=int),
        np.ones(len(X_target), dtype=int)
    ])
    
    # Phân chia tập huấn luyện và tập kiểm định phân tầng (Stratified Split)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_domain, test_size=0.3, stratify=y_domain, random_state=42
    )
    
    detector = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    detector.fit(X_train, y_train)
    
    # Tính điểm ROC-AUC trên tập kiểm định độc lập
    val_probs = detector.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_probs)
    return float(auc_score)
```

---

### Bài 3: Correcting covariate shift

**Đề bài:** Thiết lập phương pháp Hiệu chỉnh Dịch chuyển Hiệp biến bằng Trọng số Tầm quan trọng (Importance Weighting / Density Ratio Estimation) và phân tích kỹ thuật cắt tỉa trọng số (Weight Clipping) để kiểm soát phương sai.

**Phân tích toán học:**

Giả thiết Covariate Shift: $P_{\text{source}}(Y \mid \mathbf{X}) = P_{\text{target}}(Y \mid \mathbf{X})$, nhưng $P_{\text{source}}(\mathbf{X}) \ne P_{\text{target}}(\mathbf{X})$.
Mục tiêu tối thiểu hóa rủi ro trên miền đích:

$$\mathcal{R}_{\text{target}}(f) = \mathbb{E}_{(\mathbf{X}, Y) \sim P_{\text{target}}}[\ell(f(\mathbf{X}), Y)] = \int \ell(f(\mathbf{x}), y) P(y \mid \mathbf{x}) P_{\text{target}}(\mathbf{x}) d\mathbf{x} dy$$

Nhân và chia cho mật độ nguồn $P_{\text{source}}(\mathbf{x})$:

$$\mathcal{R}_{\text{target}}(f) = \int \ell(f(\mathbf{x}), y) P(y \mid \mathbf{x}) \frac{P_{\text{target}}(\mathbf{x})}{P_{\text{source}}(\mathbf{x})} P_{\text{source}}(\mathbf{x}) d\mathbf{x} dy = \mathbb{E}_{(\mathbf{X}, Y) \sim P_{\text{source}}}[w(\mathbf{X}) \ell(f(\mathbf{X}), Y)]$$

Trong đó tỷ số mật độ (Density Ratio / Importance Weight) $w(\mathbf{x}) = \frac{p_{\text{target}}(\mathbf{x})}{p_{\text{source}}(\mathbf{x})}$.

1. **Ước lượng $w(\mathbf{x})$ thông qua Bộ phân loại Miền:**
   Theo định lý Bayes với nhãn miền $D \in \{0, 1\}$:
   $$P(\mathbf{x} \mid D = 1) = \frac{P(D = 1 \mid \mathbf{x}) P(\mathbf{x})}{P(D = 1)}, \quad P(\mathbf{x} \mid D = 0) = \frac{P(D = 0 \mid \mathbf{x}) P(\mathbf{x})}{P(D = 0)}$$
   Lập tỷ số:
   $$w(\mathbf{x}) = \frac{P(\mathbf{x} \mid D = 1)}{P(\mathbf{x} \mid D = 0)} = \frac{P(D = 1 \mid \mathbf{x})}{P(D = 0 \mid \mathbf{x})} \cdot \frac{P(D = 0)}{P(D = 1)} = \frac{P(D = 1 \mid \mathbf{x})}{1 - P(D = 1 \mid \mathbf{x})} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}$$
2. **Kỹ thuật Cắt tỉa Trọng số (Weight Clipping) và Kích thước Mẫu Hiệu dụng:**
   - Khi $P(D = 1 \mid \mathbf{x}) \to 1$, tỷ số $w(\mathbf{x})$ bùng nổ tiến về $+\infty$, khiến phương sai của ước lượng rủi ro bùng nổ.
   - Để kiểm soát phương sai, ta cắt tỉa trọng số: $\tilde{w}(\mathbf{x}) = \min(w(\mathbf{x}), M)$ với ngưỡng trần $M > 0$.
   - Giám sát **Kích thước Mẫu Hiệu dụng** (Effective Sample Size - ESS):
     $$\text{ESS} = \frac{\left(\sum_{i=1}^n w_i\right)^2}{\sum_{i=1}^n w_i^2}$$
     Nếu $\text{ESS} \ll n$, trọng số bị chi phối bởi một số ít mẫu ngoại lai, đòi hỏi các giải pháp thích ứng miền sâu (Domain Adaptation / Optimal Transport).

---

### Bài 4: Other factors affecting empirical risk

**Đề bài:** Tổng hợp các yếu tố cấu trúc bên cạnh dịch chuyển phân phối khiến cho rủi ro thực nghiệm (Empirical Risk) sai lệch so với rủi ro thực tế (True Population Risk).

**Hệ thống các Nguyên nhân Gây sai lệch Rủi ro:**

1. **Quy mô mẫu hữu hạn và Phương sai Lấy mẫu (Finite-sample Variance):** Bị chi phối bởi cận tốc độ hội tụ $\mathcal{O}(n^{-1/2})$ theo định lý giới hạn trung tâm.
2. **Vi phạm giả định IID (Tự tương quan và Gom cụm):** Mẫu dữ liệu có quan hệ chuỗi thời gian hoặc quan hệ gia đình/cụm làm suy giảm số bậc tự do thực tế của dữ liệu.
3. **Ô nhiễm nhãn và Sai lệch Đo lường (Label Noise & Measurement Errors):** Nhãn bị gán sai ngẫu nhiên hoặc gán sai có hệ thống bởi người gắn nhãn (annotator bias).
4. **Mất cân bằng lớp nghiêm trọng (Severe Class Imbalance):** Mẫu thuộc lớp thiểu số quá hiếm khiến rủi ro thực nghiệm tối thiểu hóa bằng cách dự đoán toàn bộ lớp đa số.
5. **Rò rỉ thông tin dữ liệu (Data Leakage):** Thông tin từ tương lai hoặc thông tin nhãn bị rò rỉ vào ma trận đặc trưng đầu vào trong bước tiền xử lý chuẩn hóa.
6. **Thiên lệch Lựa chọn Mô hình (Optimization & Model Selection Bias):** Việc dò quét quá nhiều cấu hình kiến trúc trên cùng một tập kiểm định dẫn đến hiện tượng quá khớp vô hình.
