# 1. Introduction: Dive into Deep Learning 1.0.3

## 1.9. Exercises: Bài tập lý thuyết và tổng quan

Nguồn tham khảo: [1. Introduction, mục 1.9: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_introduction/index.html#exercises).

---

### Bài 1: What parts of code could be learned?

**Đề bài:** Xác định các thành phần trong một hệ thống phần mềm có khả năng được tối ưu hóa hoặc thay thế bằng mô hình học từ dữ liệu. Nêu các phương pháp phỏng đoán (heuristics) truyền thống hiện dùng và yêu cầu về tập dữ liệu tương ứng.

**Phân tích và Lời giải:**

Trong kỹ nghệ phần mềm truyền thống (Software 1.0), logic chương trình được đặc tả tường minh thông qua các quy tắc logic cố định (`if-else`, ngưỡng tĩnh). Ngược lại, các bài toán có không gian đầu vào phức tạp, phi tuyến tính và phân bố biến thiên liên tục (Software 2.0) là đối tượng tối ưu để áp dụng mô hình học từ dữ liệu (Machine Learning / Deep Learning).

1. **Các thành phần phần mềm tiềm năng:**
   - **Hệ thống lọc thư rác (Spam Filtering):** Thay thế hệ thống lọc dựa trên từ khóa heuristic bằng mô hình phân loại văn bản (Text Classification). Mô hình trích xuất đặc trưng ngữ nghĩa và học phân phối xác suất có điều kiện $P(y = \text{spam} \mid \mathbf{x})$.
   - **Dự báo thời gian ước tính (ETA Prediction):** Thay vì các công thức vật lý tĩnh cơ bản, mô hình học máy ánh xạ phi tuyến từ vector đặc trưng đa chiều (tuyến đường, tọa độ GPS, thời tiết, mật độ giao thông lịch sử) sang giá trị thời gian thực tế $y \in \mathbb{R}^+$.
   - **Hệ thống gợi ý (Recommender Systems):** Chuyển dịch từ các bộ lọc quy tắc thủ công sang các kiến trúc phân rã ma trận (Matrix Factorization) hoặc mạng nơ-ron sâu học biểu diễn ẩn (latent embeddings) của người dùng và vật phẩm từ lịch sử tương tác.

2. **Hạn chế của các phương pháp Heuristics:**
   - Heuristics thường áp đặt các ngưỡng giá trị tĩnh hoặc cây quyết định định sẵn do con người thiết kế.
   - Các quy tắc này có độ nhạy kém trước sự thay đổi phân phối dữ liệu (concept drift), dễ dẫn đến sai số loại I (false positive) và sai số loại II (false negative), đồng thời chi phí bảo trì tăng theo cấp số nhân khi độ phức tạp bài toán mở rộng.

3. **Yêu cầu đối với dữ liệu huấn luyện:**
   - **Tính đại diện và độc lập cùng phân phối (i.i.d.):** Dữ liệu thu thập cần phản ánh đúng phân phối thực tế của môi trường triển khai.
   - **Tín hiệu phản hồi / Nhãn mục tiêu ($y$):** Cần đảm bảo độ chính xác của nhãn (nhãn nhị phân cho lọc spam, nhãn liên tục ground-truth cho bài toán hồi quy thời gian, tín hiệu tương tác ẩn/hiện cho hệ thống gợi ý).
   - **Kiểm soát thiên lệch và đạo đức dữ liệu:** Giảm thiểu mất cân bằng lớp (class imbalance), loại bỏ thiên lệch thu thập (sampling bias) và tuân thủ các quy chuẩn bảo mật thông tin (privacy-preserving learning).

---

### Bài 2: Which problems have examples but no explicit automation procedure?

**Đề bài:** Phân tích các lớp bài toán sở hữu nguồn dữ liệu quan sát phong phú nhưng khó hoặc không thể giải quyết bằng các giải thuật lập trình tường minh (imperative algorithms).

**Phân tích và Lời giải:**

Những bài toán này thường thuộc lớp bài toán nhận thức (perception problems) và ra quyết định phức tạp, nơi con người thực hiện trực giác dễ dàng nhưng không thể giải thích hay mô hình hóa tường minh thành chuỗi chỉ thị logic hình thức.

1. **Các ví dụ tiêu biểu:**
   - **Nhận diện và phân loại ảnh (Visual Object Recognition):** Con người có thể nhận biết vật thể tức thì, nhưng biểu diễn của vật thể ở cấp độ điểm ảnh (pixel space) bị ảnh hưởng phức tạp bởi góc chiếu, điều kiện chiếu sáng, biến dạng hình học và hiện tượng che khuất (occlusion). Việc thiết lập các quy tắc nhận diện cạnh/góc thủ công không đảm bảo tính khái quát hóa.
   - **Xử lý ngôn ngữ tự nhiên và Hiểu ý định (Natural Language Understanding - NLU):** Một ý định ngữ nghĩa có thể được diễn đạt qua vô hạn cấu trúc cú pháp, kèm ngữ cảnh, thành ngữ và từ đồng nghĩa. Các hệ thống luật ngữ pháp không thể bao quát tính đa nghĩa và phong phú của ngôn ngữ tự nhiên.
   - **Phát hiện bất thường trong tín hiệu chuỗi thời gian (Anomaly Detection in Time Series):** Dữ liệu từ mạng lưới cảm biến công nghiệp thể hiện mối tương quan không gian - thời gian phức tạp. Ranh giới giữa trạng thái bình thường và trạng thái bất thường thường mang tính phi tuyến và thay đổi linh hoạt theo ngữ cảnh vận hành.

2. **Tiêu chuẩn áp dụng Học sâu (Deep Learning):**
   - Sự sẵn có của hàm mục tiêu toán học xác định (objective/loss function) và tập dữ liệu huấn luyện quy mô đủ lớn.
   - Tồn tại tính chất bất biến cục bộ (local invariance) hoặc cấu trúc phân cấp (hierarchical structure) trong dữ liệu, phù hợp với định lý xấp xỉ phổ quát (Universal Approximation Theorem) của mạng nơ-ron.
   - Đối với các bài toán yêu cầu tính tất định tuyệt đối và có thể xác minh hình thức (formal verification) trong các hệ thống an toàn sinh mạng, các giải thuật logic tường minh vẫn là lựa chọn ưu tiên hơn mô hình xác suất học sâu.

---

### Bài 3: How are algorithms, data, and computation related?

**Đề bài:** Trình bày mối tương quan cấu trúc giữa ba trụ cột: Thuật toán (Algorithms), Dữ liệu (Data), và Năng lực tính toán (Computation). Phân tích cách đặc điểm dữ liệu và tài nguyên tính toán chi phối việc lựa chọn kiến trúc mô hình.

**Phân tích và Lời giải:**

Học sâu hiện đại phát triển dựa trên sự giao thoa mật thiết của ba yếu tố mang tính liên kết động:

$$\text{Khả năng khái quát hóa} \propto f(\text{Dung lượng mô hình (Thuật toán)}, \text{Kích thước \& Đa dạng (Dữ liệu)}, \text{Ngân sách huấn luyện (Tính toán)})$$

1. **Mối quan hệ tương hỗ:**
   - **Dữ liệu làm nền tảng:** Cung cấp thông tin thống kê để ước lượng hàm mục tiêu. Dữ liệu quyết định cận trên của hiệu năng mô hình (data determines the upper bound).
   - **Thuật toán là công cụ khai phá:** Bao gồm kiến trúc mạng (CNN, Transformer, MLP) và giải thuật tối ưu hóa (SGD, AdamW). Thuật toán quyết định mức độ tiệm cận cận trên nói trên thông qua thiên kiến quy nạp (inductive bias).
   - **Tính toán là phương tiện hiện thực hóa:** Năng lực tính toán song song (GPU, TPU) quyết định kích thước mô hình có thể huấn luyện và thông lượng xử lý dữ liệu trong khung thời gian khả thi.

2. **Chi phối trong quyết định kỹ thuật:**
   - **Chế độ dữ liệu hạn chế (Low-data regime):** Mô hình có dung lượng tham số lớn (high capacity) dễ rơi vào tình trạng quá khớp (overfitting). Cần ưu tiên các mô hình có thiên kiến quy nạp mạnh (strong inductive bias), áp dụng kỹ thuật điều chuẩn (regularization), học chuyển giao (transfer learning) hoặc mô hình tuyến tính đơn giản.
   - **Chế độ dữ liệu quy mô lớn (High-data regime):** Theo quy luật mở rộng (Scaling Laws), mô hình có dung lượng lớn và thiên kiến quy nạp mềm (weak inductive bias, ví dụ: Transformer) sẽ thể hiện ưu thế vượt trội khi được hỗ trợ bởi tài nguyên tính toán mạnh.
   - **Ràng buộc tài nguyên tính toán:** Khi ngân sách tính toán (FLOPs) hoặc bộ nhớ (VRAM) bị giới hạn, việc tối ưu hóa phải chuyển hướng sang các giải pháp lượng tử hóa (quantization), cắt tỉa mô hình (pruning), chưng cất tri thức (knowledge distillation), hoặc sử dụng các biến thể tính toán nhẹ (như MobileNet, Depthwise Separable Convolutions).

---

### Bài 4: Where might end-to-end training be useful?

**Đề bài:** Phân tích các tình huống mà phương pháp huấn luyện đầu-cuối (End-to-End Deep Learning) mang lại ưu thế vượt trội so với các quy trình thiết kế theo tầng truyền thống (Pipelined Systems).

**Phân tích và Lời giải:**

Huấn luyện đầu-cuối là phương pháp tối ưu hóa trực tiếp một hàm ánh xạ vi phân được $f_\theta: \mathcal{X} \to \mathcal{Y}$ nối liền không gian đầu vào thô $\mathcal{X}$ với không gian nhãn đầu ra mục tiêu $\mathcal{Y}$, loại bỏ các mô-đun tiền xử lý hoặc trích xuất đặc trưng thủ công trung gian.

1. **Các kịch bản ứng dụng tối ưu:**
   - **Điều khiển Robot và Xe tự hành (Autonomous Driving):** Ánh xạ trực tiếp từ dữ liệu cảm biến đa phương thức (Camera, LiDAR) sang tín hiệu điều khiển góc lái và gia tốc ($[v, \omega]$), khắc phục sự tích lũy sai số (error compounding) giữa các phân hệ tách biệt (nhận diện vật thể $\to$ định vị $\to$ lập kế hoạch quỹ đạo).
   - **Nhận dạng giọng nói tự động (Automatic Speech Recognition - ASR):** Thay thế chuỗi hệ thống phức tạp (phân tích âm học $\to$ mô hình phát âm $\to$ mô hình ngôn ngữ HMM) bằng mạng đầu-cuối (như CTC hoặc Conformer-Transducer) ánh xạ trực tiếp từ spectrogram sang chuỗi văn bản.
   - **Dịch máy nơ-ron (Neural Machine Translation - NMT):** Thay thế các hệ thống dịch dựa trên luật cú pháp và cụm từ thống kê (Statistical Machine Translation) bằng các kiến trúc Sequence-to-Sequence / Transformer được tối ưu hóa toàn cục dựa trên độ đo cross-entropy.

2. **Đánh đổi kỹ thuật (Trade-offs):**
   - **Ưu điểm:** Tối ưu hóa toàn cục theo một hàm mục tiêu duy nhất; trích xuất được các đặc trưng biểu diễn tối ưu cho bài toán cuối; loại bỏ sự phụ thuộc vào các giả định heuristic của con người ở các bước trung gian.
   - **Hạn chế và Rủi ro:** Đòi hỏi kích thước tập dữ liệu huấn luyện rất lớn để hội tụ; giảm khả năng diễn giải (interpretability) và truy vết nguyên nhân lỗi; khó khăn trong việc cài đặt các ràng buộc an toàn tiền nghiệm (prior safety constraints) trong các hệ thống đòi hỏi độ tin cậy tuyệt đối.
