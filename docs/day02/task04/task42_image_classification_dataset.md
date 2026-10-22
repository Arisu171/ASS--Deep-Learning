# 4.2. The Image Classification Dataset: Bộ Dữ liệu Phân loại Ảnh và Hiệu năng Tải Dữ liệu

Nguồn tham khảo: [4.2.5. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_linear-classification/image-classification-dataset.html#exercises).

---

### Bài 1: Effect of reducing batch size

**Đề bài:** Phân tích ảnh hưởng của việc giảm kích thước mini-batch (đặc biệt khi $B = 1$) đến thông lượng (throughput) và độ trễ (latency) của đường ống nạp dữ liệu thị giác máy tính.

**Phân tích và Lời giải:**

1. **Thông lượng xử lý tổng thể (Samples per Second):**
   - Giảm $B$ xuống $1$ làm tăng số lượng lời gọi lặp iterator lên $N$ lần đối với một epoch gồm $N$ ảnh. Mỗi lời gọi kích hoạt chi phí điều phối (Python runtime overhead, Global Interpreter Lock - GIL contention, và CUDA kernel launch overhead).
   - Phép toán xử lý trên GPU yêu cầu kích thước lô đủ lớn để làm bão hòa hàng ngàn nhân CUDA (Streaming Multiprocessors - SMs). Với $B = 1$, GPU rơi vào trạng thái thiếu tải (under-utilization), băng thông bộ nhớ và thông lượng tính toán suy giảm nghiêm trọng.
2. **Độ trễ suy luận (Inference Latency):**
   - Mặc dù thông lượng giảm, độ trễ xử lý của từng mẫu đơn lẻ (tính từ thời điểm nhận ảnh đến khi trả kết quả) lại đạt giá trị thấp nhất vì không phải chờ đợi quá trình gom cụm lô (batch aggregation delay). Điều này phù hợp với các ứng dụng thời gian thực tương tác (như camera giám sát hay xe tự hành).

---

### Bài 2: Profiling and improving the data iterator

**Đề bài:** Thiết lập phương pháp đo lường điểm nghẽn hiệu năng (Profiling) trên đường ống nạp dữ liệu và đề xuất các giải pháp kỹ thuật tối ưu hóa thông lượng đưa dữ liệu vào GPU.

**Quy trình Đo lường và Tối ưu hóa:**

1. **Phân rã các giai đoạn của Pipeline:**
   Sử dụng công cụ `torch.profiler` để đo thời gian tiêu tốn độc lập tại các chặng:
   - I/O đọc tệp từ ổ cứng (Disk Read).
   - Giải mã định dạng ảnh JPEG/PNG (Image Decoding).
   - Biến đổi tăng cường dữ liệu hình học và màu sắc (Data Augmentations).
   - Ghép nối lô và chuyển đổi kiểu dữ liệu (Batch Collation).
   - Truyền dữ liệu từ bộ nhớ CPU chủ sang bộ nhớ GPU thiết bị (Host-to-Device Memory Copy via PCIe).
2. **Các giải pháp kỹ thuật tăng tốc:**
   - **Đa tiến trình xử lý ngầm (`num_workers > 0`):** Tận dụng kiến trúc đa nhân CPU để giải mã và tiền xử lý ảnh song song trước khi GPU yêu cầu.
   - **Ghim bộ nhớ trang (`pin_memory=True`):** Cấp phát tensor trên bộ nhớ không bị hoán trang (page-locked / pinned memory), cho phép kích hoạt cơ chế truy cập bộ nhớ trực tiếp (Direct Memory Access - DMA) qua bus PCIe mà không cần sự can thiệp của CPU.
   - **Sao chép dữ liệu không đồng bộ (`non_blocking=True`):** Thực thi lệnh `.to(device, non_blocking=True)` đồng thời với quá trình tính toán của kernel trên các dòng lệnh CUDA (CUDA Streams) riêng biệt.
   - **Duy trì worker liên tục (`persistent_workers=True`):** Ngăn chặn việc tạo mới và hủy tiến trình worker tốn kém ở mỗi epoch.

---

### Bài 3: Other datasets in the framework

**Đề bài:** Khảo sát hệ sinh thái các bộ dữ liệu thị giác máy tính chuẩn tắc trong thư viện `torchvision.datasets` và phân loại theo đặc thù tác vụ học sâu.

**Tổng hợp các Bộ dữ liệu Tiêu biểu:**

| Bộ dữ liệu | Quy mô (Mẫu / Lớp) | Độ phân giải | Đặc thù và Ứng dụng Học máy |
| :--- | :---: | :---: | :--- |
| **Fashion-MNIST** | $70.000$ ảnh / $10$ lớp | $28 \times 28$ (Grayscale) | Thay thế trực tiếp MNIST; phù hợp kiểm thử nhanh giải thuật phân loại và mô hình tuyến tính. |
| **CIFAR-10 / CIFAR-100** | $60.000$ ảnh / $10$ hoặc $100$ lớp | $32 \times 32$ (RGB) | Ảnh màu tự nhiên độ phân giải thấp; thử nghiệm mạng tích chập sâu (ResNet, VGG) và kỹ thuật điều chuẩn. |
| **ImageNet-1K (ILSVRC)** | $1,28 \times 10^6$ ảnh / $1.000$ lớp | Biến thiên (Chuẩn hóa $224 \times 224$) | Chuẩn mực đánh giá thị giác quy mô lớn; huấn luyện tiền mô hình nền tảng (Foundation Models / Vision Transformers). |
| **MS COCO / PASCAL VOC** | Hàng trăm ngàn ảnh | Độ phân giải cao | Tác vụ phức tạp: Phát hiện vật thể (Object Detection), Phân đoạn đối tượng (Instance/Semantic Segmentation). |
