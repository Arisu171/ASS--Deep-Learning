# Bài 2: Khảo sát MLP trên Fashion-MNIST

## Vấn đề là gì?

Phân loại 10 loại sản phẩm thời trang từ ảnh grayscale 28x28. Bài này thử nghiệm nhiều cách cấu hình mạng để xem yếu tố nào ảnh hưởng nhiều đến kết quả.

4 câu hỏi cần trả lời:

1. Mạng **sâu hơn** (nhiều tầng hơn) có tốt hơn không?
2. Mạng **rộng hơn** (nhiều neuron hơn) có tốt hơn không?
3. **ReLU, Tanh, Sigmoid** cái nào tốt hơn?
4. **Dropout và Weight Decay** có giúp mạng tổng quát hóa tốt hơn không?

---

## Dữ liệu trông như thế nào?

**Fashion-MNIST**: 70,000 ảnh quần áo, giày dép, túi xách. Thí nghiệm này dùng 6,000 ảnh để train và 2,000 ảnh để kiểm tra.

10 lớp phân loại:

| Nhãn | Sản phẩm | Nhãn | Sản phẩm      |
| :---: | :--------- | :---: | :-------------- |
|   0   | Áo thun   |   5   | Dép quai       |
|   1   | Quần dài |   6   | Áo sơ mi      |
|   2   | Áo len    |   7   | Giày thể thao |
|   3   | Váy đầm |   8   | Túi xách      |
|   4   | Áo khoác |   9   | Bốt cổ chân  |

Mỗi ảnh được trải phẳng thành vector 784 số trước khi đưa vào mạng. Bài toán này khó hơn Bài 1 vì: nhiều lớp hơn, dữ liệu phức tạp hơn, một số lớp trông rất giống nhau (Áo thun vs Áo sơ mi).

---

## Kết quả 4 thí nghiệm

### 1. Ảnh hưởng của Độ sâu (số tầng ẩn)

Giữ cố định 128 neurons mỗi tầng, chỉ thay đổi số tầng:

| Cấu hình | Số tầng |   Val Accuracy   |
| :--------- | :-------: | :--------------: |
| Depth_1    |     1     | **85.29%** |
| Depth_2    |     2     |      84.36%      |
| Depth_3    |     3     |      84.89%      |
| Depth_4    |     4     |      83.80%      |

**Kết luận:** Trong thí nghiệm này, mạng 1 tầng lại tốt nhất. Mạng sâu hơn cần nhiều thời gian học hơn, với chỉ 12 epochs thì chưa đủ để khai thác được ưu điểm của độ sâu.

---

### 2. Ảnh hưởng của Độ rộng (số neuron)

Giữ cố định 2 tầng ẩn, chỉ thay đổi số neurons:

| Cấu hình | Neurons/tầng | Tham số |   Val Accuracy   |
| :--------- | :-----------: | :------: | :--------------: |
| Width_32   |      32      |  26,506  |      83.89%      |
| Width_64   |      64      |  55,050  |      84.67%      |
| Width_128  |      128      | 118,282 |      84.36%      |
| Width_256  |      256      | 269,322 | **85.30%** |

**Kết luận:** Nhiều neurons hơn có xu hướng tốt hơn, nhưng tham số tăng 10 lần (26K lên 269K) chỉ đổi lại 1.4% accuracy. Không hiệu quả về chi phí.

---

### 3. So sánh hàm kích hoạt

Cùng kiến trúc [128, 64], chỉ thay hàm kích hoạt:

| Hàm kích hoạt | Train Accuracy | Val Accuracy | Nhận xét              |
| :--------------- | :------------: | :----------: | :---------------------- |
| **ReLU**   |     91.65%     |    84.72%    | Học nhanh, mạnh nhất |
| Tanh             |     91.80%     |    84.88%    | Tương đương ReLU   |
| Sigmoid          |     88.82%     | **84.99%** | Học chậm, train yếu  |

**Kết luận:** ReLU và Tanh cho kết quả gần bằng nhau. Sigmoid học chậm hơn nhiều (Train Accuracy thấp nhất 88.82%) vì gradient bị triệt tiêu ở các tầng sâu. Dùng **ReLU** làm mặc định là an toàn nhất.

---

### 4. Hiệu quả của Điều chuẩn (chống overfitting)

Dùng mạng lớn [256, 128, 64], đây là điều kiện dễ bị overfitting nhất:

| Cấu hình      |    Train Acc    | Val Acc |     Val Loss     | Khoảng cách Train/Val |
| :-------------- | :--------------: | :-----: | :--------------: | :---------------------: |
| Không có gì  | **93.88%** | 85.16% |      0.5141      |  Rộng (đang overfit)  |
| Dropout (p=0.3) |      87.90%      | 84.98% | **0.4530** |  Hẹp lại (tốt hơn)  |
| Weight Decay    |      93.82%      | 85.04% |      0.5181      | Gần như không đổi |
| Cả hai         |      87.53%      | 84.93% |      0.4532      | Hẹp, nhưng chậm hơn |

**Kết luận:**

- **Không dùng gì**: Train accuracy 93.88% nhưng Val accuracy chỉ 85.16%. Mô hình đang "học vẹt" dữ liệu train, không áp dụng được tốt ra ngoài. Đây là overfitting.
- **Dropout**: Kéo Train accuracy xuống 87.90% nhưng Val Loss giảm rõ rệt. Mạng học tổng quát hơn thay vì ghi nhớ.
- **Weight Decay**: Gần như không có tác dụng với cấu hình này.
- **Tốt nhất về Val Loss**: Dropout đơn thuần.

---

## Tổng kết

| Câu hỏi                         | Câu trả lời từ thực nghiệm                            |
| :-------------------------------- | :---------------------------------------------------------- |
| Sâu hơn có tốt hơn không?   | Không tự động, cần nhiều epoch hơn để hội tụ     |
| Rộng hơn có tốt hơn không?  | Có, nhưng hiệu quả giảm dần, tốn tài nguyên nhiều |
| Hàm kích hoạt nào tốt?       | ReLU là mặc định an toàn nhất                         |
| Nên dùng regularization không? | Có, Dropout hiệu quả nhất trong thí nghiệm này       |

> **Giới hạn của MLP:** Kết quả tốt nhất chỉ đạt khoảng 85-86% trên Fashion-MNIST. Để vượt qua ngưỡng này cần dùng CNN (mạng tích chập) vì CNN khai thác được cấu trúc không gian của ảnh, còn MLP xem mỗi pixel độc lập.
