
# Quant AutoML Pipeline Design

- Nạp 2 file riêng biệt features và labels(.parquet hoặc .csv) và đồng bộ hóa chỉ mục thời gian (timestamp).
- Tự động suy luận loại bài toán (**Binary Classification**, **Multiclass**) từ dữ liệu cột nhãn.
- Chia tập dữ liệu theo chuỗi thời gian chống rò rỉ (Train / Calibration/ Validation / Out-of-Sample Test).
- (Lọc 1): Loại bỏ các feature có tỷ lệ khuyết thiếu (NaN) hoặc giá trị vô hạn (Inf) vượt ngưỡng.
- (Lọc 2): Loại bỏ các feature có phương sai gần bằng 0 hoặc giá trị gần như bất biến (Quasi-constant).
- (Lọc 3): Loại bỏ các feature bị trôi dạt phân phối theo thời gian qua kiểm định tính dừng (KS-Test / Drift Gate).
- (Lọc 4): Loại bỏ các feature bị trùng lặp đa cộng tuyến tuyến tính
- (Lọc 5): Gom cụm các họ chỉ báo tương đồng và chọn đại diện trung tâm (Hierarchical Feature Clustering).
- (Lọc 6): Lọc bỏ các feature có mức độ liên quan phi tuyến thấp với nhãn (Mutual Information Gate).
- (Lọc 7): Lọc bỏ các feature có hiệu suất dự báo không ổn định xuyên suốt các chu kỳ thời gian (Temporal ICIR Gate).
- (Lọc 8): Tuyển chọn tập feature "vàng" trực giao và tối ưu tương hỗ cuối cùng bằng thuật toán MRMR.
- Đấu loại sơ bộ toàn bộ Model Candidates (LightGBM, CatBoost, XGBoost, RandomForest...) với cấu hình mặc định.
- Tự động quyết định lựa chọn Duy nhất 1 Model Tốt nhất hay chọn Top-K Model để phối hợp.
- Tối ưu hóa siêu tham số bằng Optuna (dùng TPESampler + MedianPruner, tối ưu theo Log-Loss hoặc Rank IC).
- Huấn luyện (các) mô hình tối ưu trên tập Train và hiệu chuẩn xác suất (Probability Calibration).
- Tối ưu hóa vector trọng số tĩnh kết hợp các mô hình bằng thuật toán tối ưu lồi (SLSQP).
- Đóng gói toàn bộ mô hình và phép cộng trọng số thành 1 file .onnx tĩnh duy nhất.
- Xuất file metadata features_schema.json lưu đúng danh sách và thứ tự các feature được giữ lại.
- Nạp file .onnx và dữ liệu nến OHLCV vào backtesting.py để chạy bài kiểm tra nghiệm thu độc lập cuối cùng.
