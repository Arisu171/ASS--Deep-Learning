# 5.7. Predicting House Prices on Kaggle: Dự đoán Giá nhà trên Kaggle và Kỹ nghệ Dữ liệu Dạng bảng

Nguồn tham khảo: [5.7.9. Exercises: Dive into Deep Learning 1.0.3](https://d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html#exercises).

---

### Bài 1: Submitting predictions to Kaggle

**Đề bài:** Trình bày quy trình kỹ thuật chuẩn hóa để xây dựng, xuất tệp và nộp kết quả dự báo giá nhà (`SalePrice`) lên nền tảng thi đấu Kaggle. Phân tích nguyên tắc diễn giải điểm số giữa Public Leaderboard và Private Leaderboard.

**Quy trình Kỹ thuật Chuẩn hóa:**

1. **Huấn luyện và Suy luận:**
   - Huấn luyện mô hình tối ưu trên tập `train.csv`.
   - Sinh vector dự đoán giá nhà $\hat{\mathbf{y}}_{\text{test}}$ cho các bản ghi trong tập `test.csv`.
2. **Kiểm tra Tính Toàn vẹn Dữ liệu (Sanity Checks):**
   - Định dạng tệp xuất ra theo đúng quy chuẩn của `sample_submission.csv` gồm hai cột: `Id` và `SalePrice`.
   - Bắt buộc kiểm tra điều kiện không tồn tại giá trị khuyết thiếu (`NaN`), giá trị vô hạn (`Inf`), hoặc giá trị âm ($\hat{y} \le 0$).
   - Sử dụng hàm mũ `np.expm1()` để khôi phục thang đo giá thực tế nếu mô hình được huấn luyện trên không gian logarit $\ln(y + 1)$.
3. **Phân biệt Public và Private Leaderboard:**
   - **Public Leaderboard:** Chỉ tính điểm trên một tập con nhỏ (thường khoảng $30\% - 50\%$) của tập kiểm thử. Điểm số này chỉ mang tính tham khảo tạm thời trong quá trình phát triển.
   - **Hiện tượng Rung lắc Bảng xếp hạng (Leaderboard Shakeup):** Nếu người tham gia liên tục tinh chỉnh siêu tham số để tối ưu hóa điểm số Public Leaderboard, mô hình sẽ bị quá khớp với tập con này và tụt hạng thảm hại khi kết quả chung cuộc được chấm trên Private Leaderboard ($50\% - 70\%$ còn lại). Tiêu chuẩn tối thượng để lựa chọn mô hình cuối cùng phải dựa trên điểm kiểm định chéo $K$-Fold nội bộ.

---

### Bài 2: When mean imputation is inappropriate

**Đề bài:** Phân tích những sai lệch nghiêm trọng khi áp dụng phương pháp điền giá trị trung bình (Mean Imputation) cho dữ liệu khuyết thiếu không ngẫu nhiên (Missing Not at Random - MNAR). Đề xuất giải pháp kỹ thuật phù hợp.

**Phân tích và Lời giải:**

1. **Phân loại Cơ chế Dữ liệu Khuyết thiếu:**
   - **Khuyết thiếu Hoàn toàn Ngẫu nhiên (MCAR):** Xác suất khuyết thiếu độc lập với mọi biến quan sát và chưa quan sát.
   - **Khuyết thiếu Không Ngẫu nhiên (MNAR):** Sự vắng mặt của dữ liệu mang một ý nghĩa thông tin ngữ nghĩa cụ thể.
2. **Cạm bẫy của Mean Imputation trong Bài toán Giá nhà:**
   - Thuộc tính `GarageArea` (Diện tích nhà để xe) hoặc `PoolArea` (Diện tích bể bơi) bị khuyết thiếu (`NaN`) đa phần phản ánh sự thật rằng ngôi nhà đó **hoàn toàn không có nhà để xe hoặc không có bể bơi**.
   - Nếu áp dụng phương pháp điền giá trị trung bình, hệ thống sẽ gán diện tích giả định ví dụ $470 \text{ sq ft}$ cho một ngôi nhà không có nhà để xe. Điều này làm sai lệch bản chất vật lý của bất động sản và đánh lừa mạng nơ-ron.
3. **Giải pháp Kỹ thuật Đúng chuẩn:**
   - Đối với các biến số lượng mang ý nghĩa không tồn tại: Điền giá trị hằng số $0$.
   - Đối với các biến định danh (như `GarageType`): Gán nhãn danh mục riêng biệt `None` hoặc `Missing`.
   - **Bổ sung Cột Mặt nạ Chỉ báo (Missingness Indicator):** Tạo thêm biến nhị phân $M_{\text{feat}} = \mathbb{I}(\text{feat is NaN}) \in \{0, 1\}$ để cho phép mô hình học được mối tương quan giữa sự vắng mặt của dữ liệu và mức giá mục tiêu.
   - *Ngăn chặn Rò rỉ Dữ liệu:* Mọi thông số thống kê dùng để điền khuyết (mean, median) bắt buộc chỉ được tính toán trên các nếp huấn luyện (training folds) và áp dụng sang nếp kiểm định (validation fold), tuyệt đối không tính trên toàn bộ tập dữ liệu trước khi chia fold.

---

### Bài 3: Improve score with K-fold cross-validation

**Đề bài:** Thiết lập quy trình tối ưu hóa siêu tham số bằng Kiểm định chéo $K$-Gấp ($K$-Fold Cross-Validation) chuẩn tắc để tối thiểu hóa độ đo đánh giá chính thức của cuộc thi (Root Mean Squared Logarithmic Error - RMSLE).

**Phân tích Toán học và Quy trình:**

Độ đo đánh giá chính thức của cuộc thi Kaggle House Prices:

$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \ln(y_i + 1) - \ln(\hat{y}_i + 1) \right)^2}$$

1. **Chuyển đổi Không gian Mục tiêu (Target Transformation):**
   Thay vì cố gắng điều chỉnh hàm mất mát phức tạp, ta biến đổi trực tiếp biến mục tiêu trước khi huấn luyện: $z_i = \ln(y_i + 1) = \operatorname{log1p}(y_i)$. Khi đó, hàm mất mát bình phương trung bình (MSE) tiêu chuẩn trên $z$:
   $$\text{MSE}(z, \hat{z}) = \frac{1}{N} \sum_{i=1}^N (\hat{z}_i - z_i)^2$$
   sẽ đồng nhất hoàn toàn với đại lượng $\text{RMSLE}^2$. Dự đoán cuối cùng được khôi phục qua $\hat{y} = \exp(\hat{z}) - 1 = \operatorname{expm1}(\hat{z})$.
2. **Quy trình Kiểm định Chéo $K$-Fold ($K = 5$ hoặc $K = 10$):**
   - Phân chia tập huấn luyện thành $K$ nếp phân vùng.
   - Tại mỗi nếp, thực hiện toàn bộ quy trình: Xử lý ngoại lai $\to$ Điền giá trị thiếu $\to$ Chuẩn hóa đặc trưng $\to$ Huấn luyện mô hình $\to$ Đo lường điểm RMSLE trên nếp kiểm định.
   - Điểm đánh giá mô hình là trung bình cộng $\overline{\text{RMSLE}} = \frac{1}{K}\sum_{k=1}^K \text{RMSLE}_k$ kèm độ lệch chuẩn để đo độ ổn định.

---

### Bài 4: Improve the model architecture and regularization

**Đề bài:** Đề xuất các cải tiến kiến trúc mạng Perceptron đa lớp, kỹ thuật điều chuẩn và kỹ nghệ đặc trưng (Feature Engineering) để nâng cao thứ hạng dự báo giá nhà.

**Chiến lược Nâng cao Hiệu năng:**

1. **Kỹ nghệ Đặc trưng Chuyên sâu cho Dữ liệu Dạng bảng (Tabular Feature Engineering):**
   - Tạo các đặc trưng tổng hợp miền nghiệp vụ: $\text{TotalSF} = \text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$ (Tổng diện tích sử dụng).
   - Đặc trưng tuổi thọ bất động sản: $\text{HouseAge} = \text{YrSold} - \text{YearBuilt}$.
   - Biến đổi logarit tự nhiên cho các thuộc tính số có độ lệch phân phối lớn (Skewed Features) như diện tích đất `LotArea`.
2. **Kiến trúc Mạng MLP Tối ưu:**
   - Cấu hình mạng hình phễu thu hẹp dần: $\text{Input} \to \text{Linear}(256) \to \text{BatchNorm/LayerNorm} \to \text{ReLU} \to \text{Dropout}(0,3) \to \text{Linear}(64) \to \text{ReLU} \to \text{Linear}(1)$.
   - Áp dụng kỹ thuật kết nối tắt dư (Residual Skip Connections) giữa các tầng nếu mở rộng độ sâu.
3. **Mô hình Hỗn hợp Đa dạng (Ensembling / Stacking):**
   Trong miền dữ liệu dạng bảng, các mô hình cây quyết định tăng cường gradient (Gradient Boosted Decision Trees - **LightGBM, XGBoost, CatBoost**) thường thể hiện ưu thế vượt trội nhờ khả năng xử lý tự nhiên các biên quyết định dạng bậc thang và dữ liệu khuyết thiếu. Việc kết hợp trung bình có trọng số giữa dự đoán của mạng nơ-ron MLP và mô hình GBDT sẽ mang lại điểm số cao nhất nhờ tính bù trừ sai số đa dạng.

---

### Bài 5: Omitting feature standardization

**Đề bài:** Phân tích các tác động tiêu cực về mặt giải tích hình học và động học tối ưu hóa gradient descent khi bỏ qua bước chuẩn hóa các đặc trưng số liên tục (Feature Standardization).

**Phân tích toán học:**

Xét tập dữ liệu giá nhà với các thuộc tính có thang đo vật lý cực kỳ chênh lệch:
- Diện tích đất `LotArea`: Thang đo $\sim 10^3 - 10^5 \text{ sq ft}$.
- Số lượng phòng ngủ `BedroomAbvGr`: Thang đo $\sim 1 - 5 \text{ phòng}$.

1. **Biến dạng Cảnh quan Mất mát (Ill-conditioned Loss Landscape):**
   Các phần tử của ma trận Hessian của hàm mất mát tỷ lệ với tích của các đặc trưng:
   $$H_{j, k} = \frac{\partial^2 \mathcal{L}}{\partial w_j \partial w_k} = \frac{1}{N} \sum_{i=1}^N x_{i, j} x_{i, k}$$
   - Số điều kiện của ma trận Hessian $\kappa(\mathbf{H}) = \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}(\mathbf{H})}$ sẽ bùng nổ lên tới $10^8$.
   - Các đường mức của hàm mục tiêu bị kéo dài thành các thung lũng hình elip cực kỳ dẹp và hẹp.
2. **Động học Gradient Descent bị Triệt tiêu:**
   - Gradient theo trọng số của đặc trưng lớn ($\nabla_{w_{\text{area}}} \mathcal{L}$) cực lớn, trong khi gradient theo trọng số của đặc trưng nhỏ ($\nabla_{w_{\text{bed}}} \mathcal{L}$) cực nhỏ.
   - Thuật toán Gradient Descent sẽ liên tục dao động dữ dội qua lại giữa hai vách thung lũng dốc và tiến triển cực kỳ chậm dọc theo đáy thung lũng, đòi hỏi phải hạ tốc độ học $\eta$ xuống mức cực nhỏ để không phân kỳ.
3. **Lệch lạc trong Điều chuẩn Suy giảm Trọng số ($L_2$ Weight Decay):**
   Số hạng phạt điều chuẩn $\frac{\lambda}{2} \sum_j w_j^2$ phạt đồng đều độ lớn của mọi trọng số. Do đặc trưng có thang đo lớn đòi hỏi trọng số $w$ rất nhỏ để cho ra cùng một đóng góp dự đoán, số hạng phạt $L_2$ sẽ vô tình đè nén bất công các trọng số của các đặc trưng có thang đo nhỏ.
- *Giải pháp Bắt buộc:* Áp dụng chuẩn hóa $z$-score: $\tilde{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$, đưa mọi đặc trưng về phân phối có kỳ vọng bằng 0 và phương sai bằng 1, giúp cảnh quan mất mát có dạng hình cầu đẳng hướng hoàn hảo.
