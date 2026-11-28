# Notebooks — Hệ thống Phân loại Ý định Hội thoại NPC (Vietnamese RPG Intent Classification)

Thư mục `Notebooks/` tập trung toàn bộ các Jupyter Notebooks phục vụ **khám phá dữ liệu**, **phát triển & đối sánh mô hình**, và **suy luận triển khai thực tế** cho bài toán nhận diện ý định người chơi khi tương tác với NPC trong game nhập vai (RPG) tiếng Việt.

## Cấu trúc thư mục

```text
Notebooks/
├── Exploratory/
│   └── 01_exploratory_data_analysis.ipynb        # Khám phá dữ liệu, độ đồng thuận gán nhãn & trực quan hóa TF-IDF
├── Modeling/
│   ├── 01_classical_ml_models.ipynb              # Huấn luyện, tinh chỉnh & đánh giá Naive Bayes, Linear SVM, Logistic Regression
│   └── 02_pytorch_mlp_and_benchmark.ipynb        # Huấn luyện mạng nơ-ron PyTorch MLP (ReLU vs SiLU) & Đối sánh 5 mô hình
├── Inference/
│   └── 01_npc_intent_inference_deployment.ipynb  # Triển khai suy luận đa mô hình, kiểm thử nhiễu game thủ & giả lập hội thoại NPC
└── README.md
```

### a. `Exploratory/` — Khám phá và Trực quan hóa Dữ liệu
- **[01_exploratory_data_analysis.ipynb](./Exploratory/01_exploratory_data_analysis.ipynb)**:
  - Phân tích tập dữ liệu gán nhãn thủ công ban đầu (`dataset_ver1.csv`) và đo độ đồng thuận giữa các người gán nhãn (*Inter-Annotator Agreement — Cohen's Kappa*).
  - Kiểm tra phân phối lớp trên tập huấn luyện (`9.215 mẫu`) và tập kiểm thử độc lập *Zero-Leakage* (`2.000 mẫu`, chứa 70% cấu trúc câu chưa từng xuất hiện khi train).
  - Thống kê đặc trưng ngôn ngữ chat của game thủ Việt: độ dài câu/từ, tỷ lệ câu không dấu, mật độ *teencode* / từ viết tắt và các mẫu gây nhiễu (*Hard Negatives*).
  - Trực quan hóa không gian đặc trưng `TF-IDF` (2.000 chiều, `ngram_range=(1, 2)`): độ thưa (*sparsity*), Top N-grams đặc trưng nhất cho từng nhãn và chiếu giảm chiều 2D bằng `TruncatedSVD`.

### b. `Modeling/` — Phát triển, Huấn luyện và Đánh giá Mô hình
- **[01_classical_ml_models.ipynb](./Modeling/01_classical_ml_models.ipynb)**:
  - Xây dựng đường ống trích xuất đặc trưng chống rò rỉ dữ liệu (*Zero-Leakage TF-IDF Pipeline*).
  - Phát triển 3 mô hình Học máy kinh điển: `Multinomial Naive Bayes`, `Linear SVM (LinearSVC)` và `Logistic Regression`.
  - Khảo sát siêu tham số (`alpha` cho Naive Bayes, `C` cho SVM/Logistic Regression) với kiểm định chéo phân tầng (*Stratified 5-Fold Cross-Validation*) và phân tích trọng số từ vựng theo từng lớp.
- **[02_pytorch_mlp_and_benchmark.ipynb](./Modeling/02_pytorch_mlp_and_benchmark.ipynb)**:
  - Thiết kế kiến trúc mạng nơ-ron đa tầng `PyTorch MLP (2000 -> 512 -> 128 -> 5)` tích hợp `Dropout(0.5)`.
  - Thực nghiệm đối sánh hai hàm kích hoạt phi tuyến `ReLU` và `SiLU (Swish)`, trực quan hóa đường cong hội tụ (*Loss*, *Accuracy*, *Macro F1* theo từng Epoch).
  - Bảng xếp hạng tổng thể cả **5 mô hình** trên 5 tiêu chí: *Accuracy*, *Macro F1*, *Weighted F1*, *Độ trễ suy luận (ms/query)*, *Dung lượng lưu trữ (KB/MB)* kèm ma trận nhầm lẫn (*Confusion Matrix*) và phân tích lỗi sai (*Error Analysis*).

### c. `Inference/` — Triển khai và Sử dụng Mô hình Dự đoán
- **[01_npc_intent_inference_deployment.ipynb](./Inference/01_npc_intent_inference_deployment.ipynb)**:
  - Đóng gói bộ suy luận hợp nhất `NPCDialogueEngine` nạp đồng thời từ điển `TF-IDF` và cả 5 mô hình (`Scikit-Learn` & `PyTorch`), xuất xác suất dự đoán và độ trễ thời gian thực.
  - Kiểm thử độ bền vững (*Robustness Stress Test*) trước câu gõ thiếu dấu, *teencode*, *hard negatives* và câu hỏi ngoài luồng (*Out-of-Domain*).
  - Dự đoán theo lô (*Batch Inference*), đối sánh mức độ đồng thuận giữa 5 mô hình và xuất kết quả ra CSV/JSON.
  - Giả lập kịch bản kích hoạt tính năng trong game (*In-Game Action Dispatcher*: Mở cửa hàng, Nhận nhiệm vụ, Cảnh báo chiến đấu...) và mẫu gọi REST API (`FastAPI`).

---

## Cơ chế Tự động Cấp phát Dữ liệu & Mô hình (Standalone-Ready 3 Tầng)

Mỗi file `.ipynb` đều được thiết kế **độc lập hoàn toàn (Self-Contained)** để người dùng có thể tải lẻ bất kỳ 1 file `.ipynb` nào về chạy trên **Google Colab**, **Kaggle**, hoặc máy cá nhân mà không bắt buộc phải tải toàn bộ kho mã nguồn:

1. **Tầng 1 — Tự động nhận diện kho mã nguồn cục bộ (`Local Repo Discovery`)**: Nếu chạy bên trong dự án `intent-classification`, notebook tự động liên kết tới `data/` và `network/models/` sẵn có.
2. **Tầng 2 — Tự động tải từ GitHub (`Remote Raw Fetch`)**: Nếu chỉ tải lẻ file `.ipynb` và có kết nối Internet, ô khởi tạo sẽ tự động tải đúng các tệp dữ liệu (`.json`, `.csv`, `.npz`) hoặc trọng số mô hình (`.pkl`, `.pth`) cần thiết từ nhánh `main` của kho lưu trữ GitHub về thư mục tạm `./standalone_workspace/`.
3. **Tầng 3 — Dự phòng Ngoại tuyến 100% (`Offline Synthetic & Auto-Train Fallback`)**: Trong trường hợp chạy lẻ hoàn toàn **không có Internet**, notebook tự động kích hoạt bộ sinh dữ liệu hội thoại RPG tiếng Việt thu nhỏ (giữ nguyên cấu trúc 5 nhãn, *teencode*, bỏ dấu) và tự huấn luyện nhanh bộ `TF-IDF` + mô hình ngay trong RAM để mọi ô mã lệnh và biểu đồ phía sau vẫn vận hành trơn tru.
