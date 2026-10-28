# Bài 1: Transfer Learning với ResNet18 trên CIFAR-10

## Vấn đề là gì?

Khảo sát phương pháp học chuyển giao (Transfer Learning) bằng mô hình mạng nơ-ron tích chập ResNet18 đã được huấn luyện trước (Pretrained) trên tập dữ liệu ImageNet (hơn 1.2 triệu ảnh, 1,000 phân lớp) để chuyển giao sang phân loại bộ dữ liệu CIFAR-10 trong điều kiện dữ liệu nhỏ (chỉ 2,000 ảnh huấn luyện).

Thí nghiệm so sánh trực tiếp hai cơ chế học chuyển giao chính:
1. **Đóng băng Backbone (Frozen)**: Đóng băng toàn bộ trọng số của mạng ResNet18, chỉ huấn luyện duy nhất tầng phân loại cuối cùng (`net.fc`).
2. **Tinh chỉnh Tầng cuối (Fine-tuning Layer 4)**: Giữ đông phần lớn backbone, nhưng mở khóa (unfreeze) khối tích chập cuối cùng (`net.layer4`) và tầng phân loại (`net.fc`) để thích ứng với đặc trưng của CIFAR-10.

3 câu hỏi cần trả lời:
1. Khi chỉ có 2,000 ảnh huấn luyện, việc mở khóa Layer 4 có giúp tăng độ chính xác vượt trội so với việc chỉ đóng băng backbone không?
2. Chi phí về tham số huấn luyện và thời gian tính toán giữa hai chế độ chênh lệch như thế nào?
3. Khi phân tích ma trận nhầm lẫn và recall theo từng lớp, hai phương pháp bộc lộ ưu nhược điểm gì ở các phân lớp cụ thể?

---

## Dữ liệu trông như thế nào?

**CIFAR-10**: Tập dữ liệu chuẩn gồm 60,000 ảnh màu RGB kích thước $32 \times 32$ chia đều cho 10 phân lớp:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

Quy trình chuẩn bị dữ liệu:
* **Tập train**: 2,000 ảnh trích xuất ngẫu nhiên từ tập train gốc (seed = 42, batch size = 32).
* **Tập validation**: 500 ảnh trích xuất ngẫu nhiên độc lập từ tập train gốc (seed = 42, batch size = 32).
* **Không đụng đến tập test chính thức** để đảm bảo tính khách quan trong việc đánh giá và chọn mô hình.
* **Tiền xử lý theo chuẩn ImageNet**:
  Ảnh được phóng to và cắt ngẫu nhiên theo chuẩn `ResNet18_Weights.DEFAULT.transforms()`:
  `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

---

## Làm thế nào để giải quyết?

### 1. Kiến trúc mô hình ResNet18 Pretrained
* **Khung xương (Backbone)**: Gồm tầng tích chập khởi đầu $7 \times 7$, khối MaxPool, theo sau bởi 4 khối phần dư (Layer 1, 2, 3, 4) và tầng AdaptiveAvgPool.
* **Tầng phân loại mới (Head)**: Thay thế tầng `net.fc` nguyên bản (1,000 lớp ImageNet) bằng `Linear(512, 10)` phục vụ 10 lớp của CIFAR-10.
* **Cơ chế BatchNorm**: Toàn bộ các tầng `BatchNorm2d` trong backbone đều được cố định ở chế độ kiểm định (`eval()`) trong suốt quá trình huấn luyện nhằm bảo toàn giá trị kỳ vọng (running mean) và phương sai (running variance) học được từ ImageNet, tránh hiện tượng lệch phân phối khi batch size nhỏ.

### 2. Thiết lập 2 chế độ thí nghiệm
Giữ cố định toàn bộ điều kiện: cùng tập dữ liệu (2,000 train, 500 val), seed 42, optimizer Adam, 4 epochs:

1. **Chế độ 1 (`frozen`)**:
   - Khóa toàn bộ backbone (`requires_grad = False`).
   - Chỉ huấn luyện tầng `fc`: **5,130** tham số được cập nhật.
   - Learning rate: $10^{-3}$.

2. **Chế độ 2 (`finetune_layer4`)**:
   - Mở khóa toàn bộ tham số của `layer4` và `fc`: **8,393,738** tham số được cập nhật (gấp ~1,636 lần chế độ frozen).
   - Tối ưu hóa với 2 nhóm tham số (differential learning rate):
     - Head `fc`: $lr = 10^{-3}$
     - Backbone `layer4`: $lr = 10^{-4}$ (nhỏ hơn 10 lần để tránh phá vỡ các đặc trưng tổng quát đã học).

---

## Kết quả thực nghiệm

| Chế độ | Tham số train | Train Loss (Ep 4) | Val Loss (Ep 4) | Val Accuracy (Ep 4) | Val Accuracy Tối đa | Giây/Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`frozen`** | **5,130** | 0.5580 | **0.6380** | 80.20% | 80.20% (Ep 4) | **4.10s** |
| **`finetune_layer4`** | 8,393,738 | **0.0660** | 0.7340 | **80.60%** | **82.80% (Ep 3)** | 4.36s |

### 1. Đường cong học tập qua 4 Epochs

![Đường cong học tập so sánh Frozen và Fine-tune Layer 4](image_learning_curve.png)

### 2. Ma trận nhầm lẫn (Confusion Matrix) trên tập Validation

![Ma trận nhầm lẫn so sánh Frozen và Fine-tune](image_confusion_matrix.png)

### 3. Bảng phân tích Recall theo từng lớp thời trang/vật thể

| Lớp đối tượng | Recall (`frozen`) | Recall (`finetune_layer4`) | Chênh lệch (Finetune - Frozen) |
| :--- | :---: | :---: | :---: |
| **airplane** | 86.0% | 86.0% | 0.0% |
| **automobile** | 87.0% | 89.1% | +2.1% |
| **bird** | 56.6% | **69.8%** | **+13.2%** |
| **cat** | 60.4% | **77.4%** | **+17.0%** |
| **deer** | 82.7% | 86.5% | +3.8% |
| **dog** | **71.1%** | 31.1% | **-40.0%** |
| **frog** | 93.8% | 93.8% | 0.0% |
| **horse** | 89.6% | 75.0% | -14.6% |
| **ship** | 85.1% | **97.9%** | **+12.8%** |
| **truck** | 88.2% | 90.2% | +2.0% |

---

## Nhận xét chuyên sâu

1. **Hiệu năng tổng thể cuối cùng tương đương, nhưng quá trình hội tụ khác biệt:**
   - Ở epoch 4, Validation Accuracy đạt 80.20% (frozen) và 80.60% (fine-tune), khoảng cách chỉ 0.40% (tương đương 2 ảnh trên tập 500 mẫu).
   - Fine-tune Layer 4 học nhanh vượt bậc ngay từ epoch 1 (78.60% so với 72.20%) và đạt đỉnh cao nhất là **82.80%** tại epoch 3.

2. **Dấu hiệu Overfitting rõ rệt của Fine-tuning ở Epoch 4:**
   - Sang epoch 4, `finetune_layer4` tiếp tục giảm Train Loss từ 0.1560 xuống 0.0660 (Train Acc lên tới 98.30%), nhưng Validation Loss bắt đầu tăng vọt từ 0.4810 lên 0.7340, kéo Validation Accuracy tụt xuống 80.60%.
   - Điều này khẳng định quy luật: Khi dữ liệu mục tiêu nhỏ (2,000 ảnh), việc mở khóa 8.4 triệu tham số khiến mô hình có dung lượng quá lớn so với dữ liệu, dẫn đến ghi nhớ mẫu huấn luyện (memorization).

3. **Tính ổn định của Frozen Backbone:**
   - Chế độ `frozen` cho thấy đường loss train và val giảm đều, độ chênh giữa train acc (84.80%) và val acc (80.20%) rất hẹp, hoàn toàn không có dấu hiệu overfitting.

4. **Đánh đổi về thời gian tính toán và bộ nhớ:**
   - Dù tham số huấn luyện chênh lệch hơn 1,600 lần, thời gian/epoch giữa hai chế độ chỉ chênh lệch khoảng 6% (~4.10s so với ~4.36s) vì ở cả hai chế độ, toàn bộ ảnh đều phải thực hiện bước lan truyền xuôi (forward pass) qua tất cả các tầng của ResNet18.
   - Tuy nhiên, `finetune_layer4` đòi hỏi bộ nhớ VRAM và trạng thái tối ưu hóa (Adam momentum và variance states) lớn hơn đáng kể.

5. **Hiện tượng ẩn sau Accuracy tổng thể:**
   - Phân tích chi tiết theo từng lớp cho thấy dù Accuracy xấp xỉ nhau, sự dịch chuyển phân loại là rất lớn. Fine-tune cải thiện rất mạnh ở các lớp khó như `cat` (+17.0%), `bird` (+13.2%), `ship` (+12.8%), nhưng lại làm suy giảm mạnh lớp `dog` (-40.0%) và `horse` (-14.6%). Lỗi nhầm giữa `dog` sang `cat` tăng đột biến, chứng tỏ mô hình bị thiên lệch khi cố gắng tối ưu hóa trên tập train nhỏ.

---

## Kết luận và Khuyến nghị

* **Khi nào nên chọn Frozen Backbone:** Khi tập dữ liệu nhỏ (< 5,000 mẫu), tài nguyên tính toán hạn chế hoặc cần mô hình dự đoán ổn định, ít rủi ro suy giảm hiệu năng.
* **Khi nào nên Fine-tune:** Khi cần đạt độ chính xác đỉnh cao nhất (như đạt 82.80% ở epoch 3), nhưng **bắt buộc** phải kết hợp kỹ thuật Dừng sớm (Early Stopping), kiểm soát learning rate nhỏ và áp dụng Data Augmentation để ngăn ngừa overfitting.

