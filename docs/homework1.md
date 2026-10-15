# Bài 1: Xây dựng MLP trên `make_moons`

## Vấn đề là gì?

Cho 1,200 điểm dữ liệu 2 chiều thuộc 2 nhóm, hãy xây dựng mạng neuron phân loại chúng. Đồng thời so sánh xem mạng to hơn (nhiều neuron hơn) có cho kết quả tốt hơn không.

---

## Dữ liệu trông như thế nào?

`make_moons` tạo ra 2 nhóm điểm hình hai vầng trăng lồng vào nhau. Mỗi điểm có 2 tọa độ $(x_1, x_2)$ và một nhãn là 0 hoặc 1.

Đặc điểm quan trọng: **không thể vẽ một đường thẳng để tách 2 nhóm** vì chúng uốn lượn vào nhau. Thêm vào đó, noise = 0.28 làm cho một số điểm bị "lạc" sang vùng của nhóm kia, giống thực tế.

Chia tập: 960 mẫu để train, 240 mẫu để kiểm tra (validation).

---

## Làm thế nào để giải quyết?

Dùng mạng MLP 2 tầng ẩn, thay đổi số neuron H:

```
Input(2) → Linear → ReLU → Linear → ReLU → Output(2 lớp)
```

Chạy 4 cấu hình: **H = 8, 32, 64, 128 neurons**, tất cả các thứ khác giữ nguyên (20 epoch, lr = 0.001).

---

## Kết quả

| Mô hình        |   Neurons   |  Số tham số  |   Val Accuracy   |     Val F1     |   Thời gian   |
| :--------------- | :----------: | :-------------: | :--------------: | :-------------: | :-------------: |
| MLP_8            |      8      |       114       |      86.25%      |      0.862      |      0.93s      |
| MLP_32           |      32      |      1,218      |      89.58%      |      0.896      |      0.91s      |
| **MLP_64** | **64** | **4,482** | **90.00%** | **0.900** | **0.88s** |
| MLP_128          |     128     |     17,154     |      89.58%      |      0.896      |      0.97s      |

---

## Kết quả nói lên điều gì?

**Mạng 8 neurons:** Quá nhỏ. Ranh giới phân loại vẽ ra còn thô, không uốn cong theo hình trăng được. Accuracy thấp nhất: 86.25%.

**Mạng 32 và 64 neurons:** Vùng tốt nhất. Ranh giới mượt mà, bám theo hình dạng dữ liệu tốt. Mạng 64 neurons đạt đỉnh: **90% accuracy**.

**Mạng 128 neurons:** Có tới 17,154 tham số (gấp ~4 lần mạng 64) nhưng accuracy lại giảm về bằng mạng 32. Mạng quá to bắt đầu "nhớ nhiễu" thay vì học quy luật thực sự.

> **Bài học chính:** Nhiều neuron hơn không phải lúc nào cũng tốt hơn. Cấu hình **64 neurons** là điểm cân bằng tốt nhất giữa hiệu năng và chi phí tính toán.
