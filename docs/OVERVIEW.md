# Quant AutoML Pipeline

> **End-to-End Autonomous Quantitative AutoML Pipeline** — Hệ thống tự động hóa xây dựng, tinh tuyển đặc trưng, huấn luyện tổ hợp mô hình GBDT, đóng gói ONNX siêu nhẹ (`< 5 ms/row`) và kiểm thử mù có tính chi phí ma sát (Friction-Aware Blind OOS Backtesting) cho dữ liệu chuỗi thời gian tài chính.

![Quant AutoML Pipeline Architecture](docs/diagrams/architecture.svg)

---

## Điểm Nổi Bật & Quy Tắc Bất Biến (Core Hard Gates)

| Mã Quy Tắc | Tên Quy Tắc | Bản Chất Kỹ Thuật & Ngưỡng Ràng Buộc |
| :--- | :--- | :--- |
| **RULE-01** | **Zero Look-Ahead Leakage** | `fit()` chỉ được gọi trên tập **Train**. Không xáo trộn dữ liệu (`shuffle=False`). Áp dụng **Purging & Embargo** $h$ nến giữa mọi phân vùng thời gian (`Train`, `Cal-A`, `Cal-B`, `Val`, `OOS`). |
| **RULE-02** | **Rank Metric Consistency** | Mọi đánh giá và tối ưu xuyên suốt các giai đoạn (Feature Funnel, Tournament, Optuna, SLSQP) đều căn cứ vào **Spearman Rank IC, ICIR & IC Sharpe**. |
| **RULE-03** | **Rank Invariance Calibration** | Hiệu chuẩn xác suất bằng **Temperature Scaling** ($T > 0$) trên tập `Cal-A`, đảm bảo $\text{ECE} < 0.05$ và **bảo toàn 100% thứ hạng Rank IC** (tuyệt đối không dùng Isotonic Regression gây hiện tượng bậc thang / ties). |
| **RULE-04** | **Tính Khả Vi $C^2$ trong SLSQP** | Tối ưu trọng số tổ hợp bằng `SLSQP` trên tập `Cal-B` với hàm mục tiêu **Pearson Correlation trên dữ liệu đã Rank-Transform** ($\tilde{Y} \in [0, 1]$), kết hợp Bootstrap $N=200$ và Shrinkage về $1/K$. |
| **RULE-05** | **Earn Its Place (Hurdle Rule)** | Chỉ kích hoạt Ensemble khi Diversity thỏa mãn ($\text{corr} < 0.88$) và vượt qua Single Best Model / Equal Weights ít nhất **$+5\%$ IC Sharpe**; ngược lại tự động fallback về Single Best Model. |
| **RULE-06** | **Numerical Precision Contract** | Toàn bộ ma trận số học dùng chuẩn `np.float32`. Sai số giữa Python gốc và ONNX Runtime bắt buộc thỏa mãn $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ trên tập `Val`. |
| **RULE-07** | **Decoupled Inference Engine** | Script suy luận `bundle/predictor.py` hoàn toàn độc lập (chỉ import `numpy`, `onnxruntime` và thư viện chuẩn), độ trễ $< 5.0\text{ ms/row}$, tích hợp cảnh báo `DistributionShiftWarning` khi $|z\text{-score}| > 3.0$. |
| **RULE-08** | **Kiểm Soát Đa Giả Thuyết (FDR)** | Phễu lọc F3+7 kiểm soát **Benjamini-Hochberg FDR $\le 0.05$** và kiểm tra tính đơn điệu suy giảm của ICIR qua cấu trúc kỳ hạn $\{1, 5, 10, 21\}$. |
| **RULE-09** | **Dual-Tier Friction Backtest** | Nghiệm thu độc lập trên tập `OOS` với phí giao dịch (`fee_bps`) + trượt giá (`slippage_bps`), dải đệm trễ **Hysteresis Buffer Bands** (Tier 1 NumPy) và khớp lệnh thực chiến tại `Open[t+1]` qua `backtesting.py` (Tier 2). |
