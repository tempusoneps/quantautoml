# Project / Feature Scope: Quant AutoML Pipeline

**Feature/Topic:** Quant AutoML Pipeline (Giao dịch Định lượng Chuyên sâu)  
**Spec Reference:** `quant-automl-srd.md` (`v1.0.0-PROD-SPEC`) + Quant Master Review  
**Date:** 2026-11-01  
**Status:** Approved  

---

## 1. Objectives & Goals

- **Mục tiêu cốt lõi:** Xây dựng Pipeline AutoML chuyên biệt cho giao dịch định lượng (Quantitative Trading), tự động hóa toàn diện từ nạp dữ liệu, thanh lọc feature chống rò rỉ (leakage-free), đấu loại mô hình, tối ưu siêu tham số, hiệu chuẩn xác suất đến đóng gói chuẩn ONNX Model Bundle và nghiệm thu thực tế sau chi phí ma sát.
- **Tiêu chí thành công (Success Criteria):**
  - **Zero Look-ahead Leakage:** Tuyệt đối không rò rỉ dữ liệu tương lai qua cơ chế phân chia thời gian nghiêm ngặt có Purging & Embargo (hỗ trợ Adaptive Partitioning: 5-way static khi $N \ge 10,000$ hoặc OOF-assisted khi $N < 5,000$).
  - **Metric Consistency:** Nhất quán tối ưu theo Rank IC (Spearman correlation) và IC Sharpe Ratio xuyên suốt tất cả các giai đoạn.
  - **Earn Its Place:** Ensemble chỉ được sử dụng khi vượt trội hơn mô hình đơn lẻ và Equal Weights ít nhất 5% IC Sharpe.
  - **Chuẩn hóa Inference & Sai số số học:** Sai số giữa Python gốc và ONNX Runtime $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ (chuẩn `float32`).
  - **OOS Acceptance Gate:** Chiến lược kiểm thử mù (Blind Test) trên tập OOS đạt Out-of-Sample Rank IC $> 0.035$, ICIR $> 0.40$, Net Annualized Sharpe $> 1.20$ **(sau khi trừ phí giao dịch & trượt giá với cơ chế kiểm soát vòng quay danh mục Turnover)**, MDD $< 18\%$, độ trễ suy luận $< 5\text{ ms/row}$.

---

## 2. In-Scope (Must Have)

### Module 1: Ingestion, Task Inference & Adaptive Temporal Splitter
- [ ] Nạp dữ liệu độc lập từ `features.{parquet,csv}` và `labels.{parquet,csv}`.
- [ ] Chuẩn hóa timestamp về UTC ISO-8601 (`datetime64[ns]`), inner join đồng bộ, sắp xếp tăng dần đơn điệu (`shuffle=False`).
- [ ] Tự động suy luận bài toán: Binary Classification ($Y \in \{0, 1\}$), Multiclass ($3 \le \text{nunique}(Y) \le 10$), hoặc Clustering/Unsupervised.
- [ ] **Adaptive Temporal Split:**
  - Nếu $N \ge 10,000$: Phân chia 5 vùng tĩnh `Train (60%) | Cal-A (10%) | Cal-B (10%) | Val (10%) | OOS (10%)`.
  - Nếu $N < 5,000$: Phân chia `Train (80%) | Val (10%) | OOS (10%)`, trong đó Cal-A và Cal-B được trích xuất từ Out-of-Fold (OOF) predictions của 5-fold Walk-Forward trên Train để tránh nhiễu mẫu nhỏ.
- [ ] Áp dụng Purging & Embargo $h$ nến giữa các ranh giới theo horizon của nhãn.

### Module 2: Leakage-Free Feature Funnel (Bộ lọc 7 tầng cải tiến)
- [ ] **F1 (NaN/Inf Triage):** Phân loại MCAR/MAR/MNAR, sinh indicator `is_nan_X`, tính điểm kết hợp $\text{Score} = \text{MI}(X, Y) \times (1 - \text{NaN\_Ratio})$.
- [ ] **F2 (Multi-Metric Variance):** Lọc rác đồng thời qua Hệ số biến thiên ($CV$), Peak Rolling Variance (phân vị 90, cửa sổ 63 nến), và Unique Ratio.
- [ ] **F3+7 (IC Stability & Multi-Horizon Gate có FDR):** Đánh giá tính ổn định $P(Y \mid X)$ trên horizon mục tiêu, áp dụng kiểm soát đa giả thuyết **Benjamini-Hochberg (FDR $\le 0.05$)**, và kiểm tra tính đơn điệu suy giảm của IC qua các kỳ hạn $\{1, 5, 10, 21\}$.
- [ ] **F4 (Fast dCor Collinearity):** Đo phụ thuộc phi tuyến bằng Fast Distance Correlation ($\mathcal{O}(N \log N)$ hoặc Subsampling khi $N > 5,000$); triệt tiêu cộng tuyến phi tuyến ($> 0.85$).
- [ ] **F5 (IC-Behavior Clustering):** Gom cụm phân cấp theo hành vi dự báo $1 - |\text{corr}(\text{IC}_{X_i}, \text{IC}_{X_j})|$, đại diện cụm chọn theo Max-MI/Rank-IC.
- [ ] **F6 (Regime-Aware MI & Synergy Gate):** Tính MI theo chế độ thị trường (Bull/Bear/Sideways) và phát hiện cặp tính năng cộng hưởng (Synergy).
- [ ] **F8 (JMI - Joint Mutual Information):** Tuyển chọn tập Golden Feature Set tối ưu hóa thông tin tương hỗ liên kết.

### Module 3: Model Tournament & Hyperparameter Optimization
- [ ] **Modern GBDT Trio Pool:** LightGBM, CatBoost, XGBoost (loại bỏ Random Forest để bảo toàn kích thước file ONNX và độ trễ $< 5\text{ ms}$).
- [ ] Tournament sơ bộ với ngân sách 30 trials/model bằng `RandomSampler` trên 5-fold Expanding Window Walk-Forward.
- [ ] Decision Engine: Kiểm tra điều kiện Diversity ($< 0.88$), Marginal Gain ($+1.5\%$ Rank IC), và WF Consistency ($\ge 60\%$) để quyết định Ensemble hay Single Best Model (tối đa $K \le 3$).
- [ ] Full Optuna Tuning: `TPESampler(multivariate=True)` kết hợp `HyperbandPruner` với hàm mục tiêu $\text{IC Sharpe}_{WF} - 0.05 \times \text{Complexity\_Score}$.

### Module 4: Ensemble Lồi & Hiệu chuẩn Xác suất
- [ ] Tối ưu hóa trọng số SLSQP lồi trên tập **Cal-B**: $0.05 \le w_i \le 0.60$, $\sum w_i = 1$.
- [ ] **Hàm mục tiêu SLSQP khả vi $C^2$:** Tối ưu hóa Pearson Correlation trên ma trận dự đoán đã được Rank-Transform ($\tilde{Y} \in [0, 1]$), đảm bảo hội tụ ổn định và chính xác.
- [ ] Bootstrap SLSQP ($N=200$, lấy mẫu $80\%$) kết hợp Shrinkage về Equal Weights ($1/K$) và 5% Guard Rail.
- [ ] **Probability Calibration bằng Temperature Scaling / Platt Scaling:** Bỏ Isotonic Regression để tránh hiện tượng bậc thang (ties), bảo toàn $100\%$ thứ tự phân hạng Rank IC gốc và đạt $\text{ECE} < 0.05$.

### Module 5: ONNX Packaging & Blind OOS Backtesting Thực Chiến
- [ ] Đóng gói dạng Model Bundle: `preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, `features_schema.json`, và `predictor.py`.
- [ ] Kiểm tra sai số số học bắt buộc: $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ trên chuẩn `float32`.
- [ ] Cảnh báo phân phối dịch chuyển (`DistributionShiftWarning` khi $z\text{-score} > 3.0$).
- [ ] Runner kiểm thử độc lập qua `backtesting.py` trên tập OOS phong tỏa:
  - Tích hợp **Ma trận chi phí ma sát (Taker fee + Slippage bps)**.
  - Cơ chế **Hysteresis Buffer Bands** để kiểm soát vòng quay (Turnover Cap).
  - Kiểm tra bổ sung chỉ số **Downside / Stress Period IC** trong các chu kỳ sụt giảm mạnh.

---

## 3. Out-of-Scope (Non-Goals)

- **Live Trading & Order Execution Engine:** Không kết nối API sàn thực tế hoặc quản lý lệnh thực tế (thực hiện độc lập qua execution bot).
- **Deep Learning / LLM Models:** Không triển khai mô hình Deep Learning (LSTM, Transformer) trong candidate pool (tập trung GBDT tối ưu cho tabular quant).
- **High-Frequency Tick Data:** Không xử lý order book tick-by-tick micro-second; hệ thống thiết kế cho nến (bar-level data: 1m, 5m, 15m, 1h, 1d).
- **Web UI / Graphical Dashboard:** Cung cấp API và CLI hoàn chỉnh, không làm web dashboard cồng kềnh.
- **Online Continuous Real-time Retraining:** Không tự động học lại streaming real-time (retraining theo batch định kỳ).

---

## 4. Constraints & Assumptions

- **Ràng buộc ngôn ngữ & môi trường:** Python $\ge 3.12$, tương thích thư viện máy học lõi: `lightgbm`, `xgboost`, `catboost`, `optuna`, `scipy`, `onnx`, `onnxruntime`, `polars`/`pandas`, `numpy`.
- **Ràng buộc kiểu dữ liệu số học:** Bắt buộc chuẩn hóa `np.float32` từ đầu vào để đảm bảo tính tương thích và hiệu năng của ONNX runtime.
- **Ràng buộc chuỗi thời gian:** Chỉ mục thời gian bắt buộc tăng dần đơn điệu, không trùng lặp và không xáo trộn ngẫu nhiên.
- **Tham chiếu Trade-offs:** Chi tiết cơ sở toán học và đánh đổi kỹ thuật xem tại [`docs/plan/TRADE-OFF.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/TRADE-OFF.md).
