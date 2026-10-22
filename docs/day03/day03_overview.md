
Hai bài tập cuối trong tài liệu thuộc phần thực hành huấn luyện mô hình MLP trên tập dữ liệu FashionMNIST, bao gồm **Thực hành 1 (Xây dựng Training Loop)** và **Thực hành 2 (So sánh các phương pháp tối ưu)**.

Dưới đây là giải thích chi tiết nội dung và yêu cầu của từng bài tập:

---

### **1. Thực hành 1: Xây dựng Training Loop**

* **Mục tiêu:** Hoàn thiện quy trình huấn luyện và đánh giá mô hình MLP trên tập con dữ liệu FashionMNIST.
* **Nhiệm vụ cần thực hiện:**
  * **Tệp mã nguồn:** Sử dụng file khung gợi ý `bai02_class_training_loop.py`.
  * **Vòng lặp huấn luyện (Training loop):** Viết đầy đủ 5 bước tiêu chuẩn trong PyTorch bao gồm: `zero_grad` \\(\rightarrow\\) `forward` \\(\rightarrow\\) `loss` \\(\rightarrow\\) `backward` \\(\rightarrow\\) `optimizer.step`.
  * **Chế độ mô hình:** Thiết lập đúng `model.train()` khi thực hiện huấn luyện và `model.eval()` khi đánh giá mô hình.
  * **Theo dõi và in kết quả:** Hiển thị thiết bị tính toán đang sử dụng (CPU/GPU), tổng số tham số của mô hình, cùng giá trị `train loss` và `validation accuracy` sau mỗi epoch.
  * **Thử nghiệm Learning Rate:** Huấn luyện mô hình với **3 mức learning rate** (\\(0.01\\), \\(0.001\\), \\(0.0001\\)), giữ nguyên các tham số cấu hình khác để so sánh tốc độ hội tụ và độ chính xác.
* **Sản phẩm cần nộp:** Mã nguồn Python hoàn chỉnh chạy được; log kết quả huấn luyện qua từng epoch; bảng so sánh kết quả của 3 mức learning rate kèm nhận xét ngắn về tốc độ hội tụ.

---

### **2. Thực hành 2: So sánh các phương pháp tối ưu**

* **Mục tiêu:** Đánh giá ảnh hưởng của các thuật toán tối ưu hóa (optimizer) khác nhau đến quá trình huấn luyện mô hình MLP trên FashionMNIST.
* **Các mức độ yêu cầu:**
  * **Mức Cơ bản:** Huấn luyện cùng một kiến trúc MLP bằng 2 thuật toán tối ưu là **SGD** và **Adam** (giữ cố định dữ liệu, số epoch và ghi rõ learning rate).
  * **Mức Khá:** Vẽ biểu đồ đường (learning curve) biểu diễn `train loss` và `validation loss` theo từng epoch để so sánh hai optimizer.
  * **Mức Nâng cao:** Cài đặt cơ chế **Early Stopping** (tự động dừng huấn luyện khi `validation loss` không cải thiện sau một số epoch liên tiếp).
* **Sản phẩm cần nộp:** Mã nguồn chạy được; bảng so sánh cấu hình và kết quả; biểu đồ learning curve; bài nhận xét phân tích ưu/nhược điểm của SGD vs Adam, phân tích nguyên nhân gây lỗi và đề xuất hướng cải thiện.

---

end
