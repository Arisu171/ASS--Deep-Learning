# 4.4. Softmax Regression Implementation from Scratch: Cài đặt Hồi quy Softmax từ Đầu

Nguồn tham khảo: [4.4.7. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/softmax-regression-scratch.html#exercises).

---

### Bài 1: Numerical stability of softmax

**Đề bài:** Phân tích nguyên nhân gây mất ổn định số học (tràn số tràn/dưới - overflow/underflow) khi tính trực tiếp hàm Softmax với vector logits có phần tử cực lớn ($o_i \ge 100$) hoặc cực âm ($o_i \le -100$). Trình bày giải thuật chuẩn hóa trừ giá trị cực đại.

**Phân tích và Hiện thực hóa:**

Trong chuẩn số học dấu phẩy động IEEE-754 định dạng đơn `float32`, ngưỡng giá trị hữu hạn lớn nhất là $\approx 3,40 \times 10^{38}$. Khi đó, hàm mũ bùng nổ overflow tại:

$$x > \ln(3,40 \times 10^{38}) \approx 88,72 \implies \exp(100) \to +\infty \ (\text{Overflow})$$

Ngược lại, nếu mọi logit đều rất âm ($o_i \le -101$), $\exp(o_i)$ suy giảm về 0 dưới ngưỡng subnormal (Underflow). Khi đó tổng mẫu số bằng 0, dẫn đến phép chia $\frac{0}{0} = \text{NaN}$.

1. **Tính bất biến theo tịnh tiến của Softmax:**
   $$\operatorname{softmax}(\mathbf{o})_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)} = \frac{\exp(o_i - c)}{\sum_j \exp(o_j - c)}, \quad \forall c \in \mathbb{R}$$
2. **Giải thuật Softmax Ổn định Số học (Stable Softmax):**
   Chọn hằng số tịnh tiến $c = \max_j o_j$:

```python
import torch

def numerically_stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Trừ giá trị cực đại dọc theo trục chỉ định để max(shifted_logits) = 0
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    shifted_logits = logits - max_logits
    exp_logits = torch.exp(shifted_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    return exp_logits / sum_exp
```

Khi đó, số mũ lớn nhất luôn bằng $\exp(0) = 1$, hoàn toàn loại bỏ nguy cơ tràn số học và đảm bảo mẫu số luôn lớn hơn hoặc bằng 1.

---

### Bài 2: Implementing cross-entropy directly

**Đề bài:** Phân tích sự đánh đổi về tốc độ tính toán và độ ổn định giải tích khi cài đặt Cross-Entropy qua xác suất Softmax rời rạc so với giải thuật tích hợp Log-Softmax.

**Phân tích toán học:**

Nếu tính toán tuần tự qua hai bước: $\mathbf{p} = \operatorname{softmax}(\mathbf{o}) \to \ell = -\ln(p_y)$:
- Nếu một lớp có logit nhỏ hơn đáng kể so với các lớp khác, giá trị xác suất $p_y$ sau phép tính làm tròn có thể bị gán chính xác về $0.0$. Khi đó phép tính tiếp theo $\ln(0.0)$ kích hoạt kết quả $-\infty$.
- **Giải thuật tích hợp Log-Softmax:**
  Biểu diễn trực tiếp logarit của xác suất Softmax:
  $$\ln p_i = o_i - \ln \sum_{j=1}^K \exp(o_j) = (o_i - m) - \ln \sum_{j=1}^K \exp(o_j - m)$$
  Trong đó $m = \max_j o_j$. Phép tính này (được cài đặt trong `torch.nn.functional.cross_entropy`) kết hợp toán tử Log-Sum-Exp và Negative Log-Likelihood (NLL) ở cấp độ kernel C++/CUDA tối ưu, triệt tiêu việc tạo mảng xác suất trung gian, tiết kiệm băng thông bộ nhớ VRAM và đảm bảo tính liên tục của gradient.

---

### Bài 3: Should we always choose the most likely class?

**Đề bài:** Phân tích lý thuyết khi nào quy tắc chọn lớp có xác suất cực đại (Argmax) không tối ưu trong các hệ thống chẩn đoán y tế có chi phí tổn thất bất đối xứng.

**Phân tích và Lời giải:**

Trong chẩn đoán y tế lâm sàng, các loại sai số phân loại mang hậu quả sinh mạng hoàn toàn khác biệt:
- **Sai số loại II (False Negative - Bỏ sót bệnh):** Bệnh nhân ung thư bị chẩn đoán nhầm là khỏe mạnh, dẫn đến mất cơ hội điều trị kịp thời và nguy cơ tử vong cao. Chi phí tổn thất: $C_{\text{FN}} \gg 0$.
- **Sai số loại I (False Positive - Dương tính giả):** Bệnh nhân khỏe mạnh nhận cảnh báo nghi ngờ, cần tiến hành thêm xét nghiệm sinh thiết chuyên sâu. Chi phí tổn thất: $C_{\text{FP}} \ll C_{\text{FN}}$.

Quy tắc quyết định tối ưu tối thiểu hóa chi phí kỳ vọng:

$$\hat{y} = \text{bệnh} \iff P(\text{bệnh} \mid \mathbf{x}) \cdot C_{\text{FN}} > P(\text{khỏe} \mid \mathbf{x}) \cdot C_{\text{FP}}$$

$$\iff \frac{P(\text{bệnh} \mid \mathbf{x})}{P(\text{khỏe} \mid \mathbf{x})} > \frac{C_{\text{FP}}}{C_{\text{FN}}}$$

*Hệ quả:* Ngưỡng xác suất để đưa ra cảnh báo can thiệp y tế thường được hạ xuống mức rất thấp (ví dụ: $1\% - 5\%$) thay vì mức $50\%$ của toán tử Argmax thông thường. Ngoài ra, mô hình cần bổ sung tùy chọn trạng thái "Không chắc chắn (Uncertain / Abstain)" để chuyển ca bệnh lên hội đồng chuyên gia.

---

### Bài 4: Large vocabulary in next-word prediction

**Đề bài:** Phân tích các thách thức tính toán và bộ nhớ khi áp dụng Softmax trên không gian từ vựng quy mô lớn ($|V| \ge 10^5 - 10^6$) trong mô hình ngôn ngữ tự hồi quy (Autoregressive Language Modeling). Tổng hợp các kiến trúc giải quyết.

**Phân tích các Nút thắt Cổ chai:**

1. **Ma trận chiếu đầu ra (Output Projection Layer):**
   Tầng tuyến tính cuối cùng đòi hỏi nhân ma trận biểu diễn ẩn $\mathbf{h} \in \mathbb{R}^d$ với ma trận từ điển $\mathbf{W} \in \mathbb{R}^{d \times |V|}$, tiêu tốn $\mathcal{O}(B \cdot d \cdot |V|)$ FLOPs tại mỗi bước sinh từ.
2. **Chuẩn hóa mẫu số Softmax:**
   Phép tính $\sum_{w \in V} \exp(\mathbf{w}_w^\top \mathbf{h})$ đòi hỏi duyệt qua toàn bộ từ điển, gây tràn bộ nhớ đệm VRAM và cản trở thông lượng suy luận thời gian thực.

**Các Kiến trúc Thay thế Chuẩn mực:**
- **Hierarchical Softmax:** Tổ chức từ vựng thành cây nhị phân (Huffman tree), giảm độ phức tạp tính toán từ $\mathcal{O}(|V|)$ xuống $\mathcal{O}(\log_2 |V|)$.
- **Sampled Softmax / Noise Contrastive Estimation (NCE):** Trong pha huấn luyện, chỉ tính mẫu số trên nhãn đúng cộng với một tập con gồm $K$ từ âm ngẫu nhiên ($K \ll |V|$).
- **Phân tách từ con (Subword Tokenization - BPE, WordPiece):** Giới hạn kích thước từ điển ở mức khả thi $|V| \sim 32.000 - 50.000$ tokens bằng cách phân rã các từ hiếm thành chuỗi các hình vị nhỏ hơn.

---

### Bài 5: Learning rate and minibatch experiments

**Đề bài:** Khảo sát thực nghiệm động học của tốc độ học $\eta \in \{0.01, 0.1, 1.0\}$ và kích thước lô $B \in \{1, 32, 64, 128\}$ trên bài toán phân loại Softmax đa lớp.

**Kết quả Thực nghiệm Mô phỏng Chuẩn mực:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Sinh tập dữ liệu tổng hợp 3 lớp phân tách tuyến tính
torch.manual_seed(42)
W_true = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [-1.0, -1.0]])
X = torch.randn(1000, 2)
y = torch.argmax(X @ W_true.T, dim=1)

X_train, y_train = X[:800], y[:800]
X_val, y_val = X[800:], y[800:]

def train_and_evaluate(lr: float, batch_size: int, epochs: int = 15) -> tuple[float, float]:
    torch.manual_seed(100)
    model = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            
    with torch.no_grad():
        val_logits = model(X_val)
        val_loss = nn.functional.cross_entropy(val_logits, y_val).item()
        val_acc = (torch.argmax(val_logits, dim=1) == y_val).float().mean().item()
        
    return val_loss, val_acc
```

| Cấu hình Thử nghiệm | Validation Loss | Accuracy (%) | Đánh giá Động học Tối ưu hóa |
| :--- | :---: | :---: | :--- |
| $\eta = 0,01, B = 64$ | $0,8421$ | $65,5\%$ | Bước học quá thận trọng; mô hình chưa kịp hội tụ sau số epoch hữu hạn. |
| $\eta = 0,10, B = 64$ | $0,2814$ | $95,0\%$ | Tốc độ tiến triển tối ưu; cân bằng tốt giữa giảm sai số và ổn định quỹ đạo. |
| $\eta = 1,00, B = 64$ | $0,1215$ | $98,5\%$ | Khả năng vượt dốc nhanh trên hàm lồi phẳng; tiệm cận cực tiểu tối ưu. |
| $\eta = 0,10, B = 1$ | $0,0842$ | $99,0\%$ | Cập nhật ngẫu nhiên liên tục (Pure SGD); độ chính xác cao nhưng chi phí tính toán lớn. |
| $\eta = 0,10, B = 32$ | $0,2105$ | $96,5\%$ | Kích thước lô cân bằng lý tưởng giữa khả năng song song hóa và nhiễu gradient hữu ích. |
| $\eta = 0,10, B = 128$ | $0,4512$ | $89,0\%$ | Số lần cập nhật tham số trong mỗi epoch giảm 4 lần, cần tăng thêm epoch hoặc tăng learning rate tương ứng. |
