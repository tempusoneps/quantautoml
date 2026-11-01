# Project / Feature Scope: Quant AutoML Pipeline

**Feature/Topic:** Quant AutoML Pipeline (Giao dịch Định lượng Chuyên sâu)  
**Spec Reference:** `quant-automl-srd.md` (`v1.0.0-PROD-SPEC`) + Quant Master Review  
**Date:** 2026-11-01  
**Status:** Approved  

> File này chứa phạm vi **tổng thể**. Phạm vi chi tiết (In-Scope checklist) của từng step nằm trong `SCOPE.md` của thư mục step tương ứng (xem bảng ở §2).

---

## 1. Objectives & Goals

- **Mục tiêu cốt lõi:** Xây dựng Pipeline AutoML chuyên biệt cho giao dịch định lượng (Quantitative Trading), tự động hóa toàn diện từ nạp dữ liệu, thanh lọc feature chống rò rỉ (leakage-free), đấu loại mô hình, tối ưu siêu tham số, hiệu chuẩn xác suất đến đóng gói chuẩn ONNX Model Bundle và nghiệm thu thực tế sau chi phí ma sát.
- **Tiêu chí thành công (Success Criteria):**
  - **Zero Look-ahead Leakage:** Tuyệt đối không rò rỉ dữ liệu tương lai qua cơ chế phân chia thời gian nghiêm ngặt có Purging & Embargo (hỗ trợ Adaptive Partitioning: 5-way static khi $N \ge 10,000$ hoặc OOF-assisted khi $N < 10,000$).
  - **Metric Consistency:** Nhất quán tối ưu theo Rank IC (Spearman correlation) và IC Sharpe Ratio xuyên suốt tất cả các giai đoạn.
  - **Earn Its Place:** Ensemble chỉ được sử dụng khi vượt trội hơn mô hình đơn lẻ và Equal Weights ít nhất 5% IC Sharpe.
  - **Chuẩn hóa Inference & Sai số số học:** Sai số giữa Python gốc và ONNX Runtime $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ (chuẩn `float32`).
  - **OOS Acceptance Gate:** Chiến lược kiểm thử mù (Blind Test) trên tập OOS đạt Out-of-Sample Rank IC $> 0.035$, ICIR $> 0.40$, Net Annualized Sharpe $> 1.20$ **(sau khi trừ phí giao dịch & trượt giá với cơ chế kiểm soát vòng quay danh mục Turnover)**, MDD $< 18\%$, độ trễ suy luận $< 5\text{ ms/row}$.

---

## 2. In-Scope theo Step

| Step | Thư mục | Scope chi tiết |
| :--- | :--- | :--- |
| 01 | [`01-ingestion-split/`](01-ingestion-split/SCOPE.md) | Module 1: Ingestion, Task Inference & Adaptive Temporal Splitter |
| 02 | [`02-feature-selection/`](02-feature-selection/SCOPE.md) | Module 2: Leakage-Free Feature Funnel (7 tầng) |
| 03 | [`03-model-selection/`](03-model-selection/SCOPE.md) | Module 3 (phần 1): GBDT Trio Pool, Tournament, Decision Engine |
| 04 | [`04-tuning/`](04-tuning/SCOPE.md) | Module 3 (phần 2): Full Optuna Tuning |
| 05 | [`05-ensemble-calibration/`](05-ensemble-calibration/SCOPE.md) | Module 4: SLSQP Ensemble & Probability Calibration |
| 06 | [`06-packaging/`](06-packaging/SCOPE.md) | Module 5 (phần 1): ONNX Bundle, Predictor, Precision Validator |
| 07 | [`07-backtest/`](07-backtest/SCOPE.md) | Module 5 (phần 2): Friction-Aware Blind OOS Backtest |

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
- **Tham chiếu Trade-offs:** Xem index tại [`TRADE-OFF.md`](TRADE-OFF.md); chi tiết nằm trong `TRADE-OFF.md` của từng step.
