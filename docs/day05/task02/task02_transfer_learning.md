# Bài 2: Tăng cường Dữ liệu (Data Augmentation) và Học chuyển giao với MobileNetV3 trên CIFAR-10

## Vấn đề là gì?

Thực hiện khảo sát đối chiếu có hệ thống giữa hai chiến lược nâng cao hiệu năng trong thị giác máy tính:
1. **Tăng cường Dữ liệu (Data Augmentation)** trên mạng CNN tự xây dựng từ đầu (train from scratch) nhằm chống lại hiện tượng ghi nhớ mẫu.
2. **Học chuyển giao (Transfer Learning)** với kiến trúc mạng tối ưu hóa cho thiết bị di động MobileNetV3-Small đã được huấn luyện sẵn trên ImageNet.

3 cấu hình được so sánh trực tiếp dưới cùng điều kiện:
* `cnn_baseline`: Mạng CNN 2 blocks tự huấn luyện từ đầu, không áp dụng Data Augmentation.
* `cnn_augmentation`: Mạng CNN 2 blocks tự huấn luyện từ đầu, có áp dụng Data Augmentation trên tập huấn luyện.
* `mobilenetv3_small_pretrained`: Mô hình MobileNetV3-Small pretrained đóng băng backbone, thay thế tầng phân loại 10 lớp và áp dụng Data Augmentation.

Các câu hỏi cần trả lời:
1. Kỹ thuật Data Augmentation có lập tức mang lại lợi ích rõ rệt trên mạng CNN tự huấn luyện trong số lượng epoch ngắn (4 epochs) không?
2. Mô hình Pretrained MobileNetV3-Small vượt trội hơn mô hình tự học bao nhiêu về Accuracy và chỉ số vĩ mô Macro-F1?
3. Đánh đổi giữa số lượng tham số huấn luyện và thời gian chạy thực tế trên mỗi epoch diễn ra như thế nào?
4. Những cặp lớp nào hay bị nhầm lẫn nhất và các ảnh bị dự đoán sai có đặc điểm chung gì?

---

## Dữ liệu trông như thế nào?

**CIFAR-10**: Tập dữ liệu 10 lớp đối tượng tự nhiên kích thước $32 \times 32$:
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

Thiết lập phân chia tập dữ liệu:
* **Tập train**: 2,000 ảnh trích xuất ngẫu nhiên (seed = 42, batch size = 32).
* **Tập validation**: 500 ảnh trích xuất ngẫu nhiên độc lập (seed = 42, batch size = 32).
* **Nguyên tắc vàng**: Tập Validation luôn được cố định không áp dụng Augmentation. Mọi kỹ thuật biến đổi tăng cường chỉ áp dụng duy nhất cho tập huấn luyện (train split). Không sử dụng tập test chính thức trong quá trình so sánh và lựa chọn mô hình.

Quy tắc tiền xử lý:
* **Đối với CNN tự học (32x32)**:
  - Validation: `ToTensor() -> Normalize(CIFAR_MEAN, CIFAR_STD)`.
  - Augmentation: `RandomCrop(32, padding=4) -> RandomHorizontalFlip() -> ToTensor() -> Normalize(CIFAR_MEAN, CIFAR_STD)`.
* **Đối với MobileNetV3 (224x224)**:
  - Validation: `Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(IMNET_MEAN, IMNET_STD)`.
  - Augmentation: `Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip() -> ToTensor() -> Normalize(IMNET_MEAN, IMNET_STD)`.

---

## Làm thế nào để giải quyết?

### 1. Kiến trúc hai mô hình thử nghiệm
* **SmallCNN (2 blocks)**:
  - Block 1: `Conv2d(3, 16, k=3, p=1) -> ReLU -> MaxPool2d(2)` (kích thước giảm $32 \rightarrow 16$).
  - Block 2: `Conv2d(16, 32, k=3, p=1) -> ReLU -> MaxPool2d(2)` (kích thước giảm $16 \rightarrow 8$).
  - Classifier: `Flatten(2048) -> Linear(2048, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)`.
  - Tổng số tham số huấn luyện: **268,650**.

* **MobileNetV3-Small Pretrained**:
  - Khung xương MobileNetV3 gồm các khối Inverted Residuals và Squeeze-and-Excitation tối ưu.
  - Đóng băng toàn bộ backbone (`requires_grad = False`).
  - Thay thế đầu phân loại `classifier[3]`: `Linear(1024, 10)`.
  - Toàn bộ `BatchNorm2d` được giữ ở chế độ `eval()`.
  - Tổng số tham số huấn luyện: **10,250** (ít hơn 26 lần so với SmallCNN).

### 2. Thiết lập siêu tham số chung
Cố định seed 42, optimizer Adam, learning rate $10^{-3}$, batch size 32 qua 4 epochs.

---

## Kết quả thực nghiệm

| Cấu hình | Mô hình | Augmentation | Tham số train | Train Loss (Ep 4) | Val Loss (Ep 4) | Val Accuracy (Ep 4) | Macro-F1 (Ep 4) | Giây/Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cnn_baseline` | SmallCNN | Không | 268,650 | 1.4880 | 1.6150 | 42.00% | 0.4120 | **0.72s** |
| `cnn_augmentation` | SmallCNN | Có | 268,650 | 1.6170 | 1.6920 | 39.00% | 0.3880 | 0.95s |
| **`mobilenetv3_small_pretrained`** | MobileNetV3 | Có | **10,250** | **0.4280** | **0.5260** | **83.40%** | **0.8350** | 3.68s |

### 1. Đường cong học tập qua 4 Epochs

![Đường cong học tập so sánh 3 cấu hình](image_learning_curves.png)

### 2. Ma trận nhầm lẫn (Confusion Matrix) đối chiếu giữa 3 cấu hình

![Ma trận nhầm lẫn đối chiếu 3 cấu hình](image_confusion_matrix.png)

### 3. Minh chứng mẫu ảnh dự đoán Đúng và Sai tiêu biểu

* **Mẫu dự đoán của MobileNetV3-Small Pretrained:**

![Mẫu dự đoán MobileNet](image_examples_mobilenetv3_small_pretrained.png)

---

## Nhận xét chuyên sâu

### 1. Hiệu ứng của Data Augmentation trên mạng CNN tự học
* **Hiện tượng trong 4 epoch đầu:** `cnn_augmentation` có độ chính xác thấp hơn baseline một chút (39.0% so với 42.0%, Macro-F1 0.388 so với 0.412).
* **Bản chất cơ chế:** Data Augmentation tạo ra dữ liệu biến đổi liên tục (cắt xén, lật ngang ngẫu nhiên), khiến bài toán tối ưu trên tập train trở nên khó hơn. Do đó, Train Loss của mô hình augmentation cao hơn baseline (1.6170 so với 1.4880). Tuy nhiên, độ chênh lệch giữa train loss và val loss thu hẹp lại, chứng minh kỹ thuật này làm đúng nhiệm vụ hạn chế việc ghi nhớ máy móc.
* **Tại sao chưa thấy lợi ích vượt trội:** Cả hai mô hình SmallCNN mới chỉ học 4 epochs, loss vẫn đang trên đà giảm dốc, tức là **mô hình chưa hề bị overfit**. Data Augmentation chỉ thực sự phát huy sức mạnh khi số lượng epoch đủ lớn (ví dụ 15 đến 30 epochs), thời điểm mà mô hình không có augmentation bắt đầu phân kỳ overfit.

### 2. Sức mạnh áp đảo của Transfer Learning
* **Khoảng cách chất lượng:** `mobilenetv3_small_pretrained` đạt **83.40% Validation Accuracy** và **0.8350 Macro-F1**, cao hơn mô hình CNN tự học hơn **41 đến 44 điểm phần trăm**.
* **Tốc độ học:** Ngay từ epoch đầu tiên, MobileNet đã chạm mốc 78.40%, trong khi CNN tự học chỉ đạt 26 đến 27%.
* **Tổng quát hóa tuyệt đối:** Khoảng cách giữa Train Acc (86.70%) và Val Acc (83.40%) chỉ là 3.3%, Val Loss liên tục giảm ổn định xuống 0.5260 mà không hề có bất kỳ dấu hiệu quá khớp nào.
* **Nguyên nhân:** Khung xương MobileNetV3 đã tích lũy hàng triệu tri thức về cấu trúc biên cạnh, chất liệu và hình khối từ tập ImageNet khổng lồ. Với tập dữ liệu nhỏ (2,000 ảnh), việc chỉ phải huấn luyện 10,250 tham số tại tầng phân loại cuối cùng giúp mô hình nắm bắt phân phối nhãn một cách hoàn hảo.

### 3. Đánh đổi tính toán thực tế
* MobileNet chỉ cần tối ưu 10,250 tham số (ít hơn 26 lần), nhưng thời gian thực tế mỗi epoch lại chậm hơn gần 4 đến 5 lần (3.68s so với 0.72s đến 0.95s).
* Lý do là kích thước ảnh đầu vào của MobileNet là $224 \times 224$ (lớn hơn gấp 49 lần so với ảnh gốc $32 \times 32$), và toàn bộ phép tính trượt của mạng tích chập sâu vẫn phải diễn ra đầy đủ ở bước lan truyền xuôi. Như vậy: **Số tham số huấn luyện ít không đồng nghĩa với khối lượng tính toán ít**.

### 4. Phân tích lỗi và ma trận nhầm lẫn
* **CNN tự học:** Các lỗi nhầm lẫn tập trung dày đặc ở các cặp vật thể có hình khối tương đồng: xe tải sang ô tô (`truck` sang `automobile`), mèo sang chó/ngựa/ếch (`cat` sang `dog`/`horse`/`frog`), chim sang hươu (`bird` sang `deer`). Đặc trưng trích xuất cấp thấp tự học còn rất mỏng manh.
* **MobileNetV3:** Lỗi phân bố rất thưa thớt (cao nhất chỉ 4 đến 5 ảnh ở mỗi cặp). Các lớp xe cộ, động vật lớn đạt độ chính xác cực cao (`truck` 92.2%, `deer` 90.4%, `frog` 89.6%).
* **Đặc điểm ảnh dự đoán sai:** Hầu hết các ảnh sai đều là các mẫu có độ phân giải $32 \times 32$ bị nhiễu nền phức tạp, góc chụp khuất hoặc vật thể chiếm diện tích quá nhỏ trong khung hình.

---

## Kết luận và Bài học rút ra

1. **Với bài toán dữ liệu nhỏ (vài nghìn ảnh):** Sử dụng mô hình Pretrained (Transfer Learning) mang lại bước nhảy vọt về chất lượng (+40% accuracy) mà không bất kỳ kỹ thuật điều chỉnh siêu tham số hay data augmentation nào trên mô hình tự học có thể sánh được.
2. **Data Augmentation:** Là phương pháp điều hòa chống overfit tuyệt vời cho quá trình huấn luyện dài hạn, nhưng không thể thay thế được sự thiếu hụt tri thức biểu diễn sâu khi dữ liệu quá ít.
3. **Thực tế triển khai:** Cần cân nhắc giữa yêu cầu về độ chính xác và ràng buộc về độ trễ suy luận (FPS/thời gian tính toán).

