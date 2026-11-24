# Technical Trade-Offs & Design Decisions — `v1.1-optimize3`

Tài liệu này ghi nhận các phương án thay thế và lý do lựa chọn thiết kế cho từng thành phần của `v1.1-optimize3`.

---

## 1. Cửa sổ Rolling IC trong `bootstrap.py` & `tournament.py` (Phase 1)

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
| :--- | :--- | :--- | :--- |
| **A. Dùng 1 scalar `Rank IC` toàn đoạn (không rolling)** | Không bao giờ bị Simpson's Paradox; tính toán cực nhanh. | Mất thông tin về độ ổn định theo thời gian (variance của IC), vi phạm tinh thần `IC Sharpe` (`RULE-02`). | Loại bỏ |
| **B. Cố định `window = 500` cứng cho mọi dataset** | Khớp `backtest.ic_window = 500` trên tập 76k dòng. | Thất bại / degenerate trên dataset nhỏ (`N < 1,000`) và làm hỏng unit tests tổng hợp. | Loại bỏ |
| **C. Thích ứng `min(500, N // div)` + cho phép ghi đè qua `cfg.ic_window` (clamp theo `N // 4`)** | Tự khớp `500` trên dữ liệu intraday lớn (`N >= 5,000`), tự co lại hợp lý trên dữ liệu nhỏ (`N = 100..500`), tránh suy biến còn 2 cửa sổ (`ddof=1`) khi `ic_window` lớn trên `OOF_ASSISTED`, và cho phép tinh chỉnh theo khung thời gian (1m/5m/1h/1d). | Cần chọn `div` hợp lý (`div = 10` ở `Cal-A`, `div = 4` ở mỗi fold WF) để luôn có $\ge 9\text{–}15$ cửa sổ tính `std`. | **Chọn (C)** |

**Ghi chú chấp nhận có chủ đích (Tail Remainder):** `calc_rolling_ic` dùng cửa sổ không chồng lấn (`n_windows = len(x) // window`) nên bỏ qua phần dư `n % window` ở cuối đoạn (tối đa `499` nến ≈ `6.6%` của `7,585` nến khi `w = 500`). Chấp nhận đánh đổi này để tránh trùng lặp quan sát giữa các cửa sổ gây thổi phồng `ICIR`.

---

## 2. Xử lý Turnover & Phí Ma Sát trong Backtest (Phase 2)

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
| :--- | :--- | :--- | :--- |
| **A. Đưa EMA smoothing vào trong `predictor.py` (ONNX bundle)** | Tín hiệu ra từ predictor đã mượt sẵn. | Biến `predictor.py` thành stateful (phụ thuộc lịch sử các lần gọi `predict` trước đó), phá vỡ hợp đồng suy luận từng dòng độc lập (`RULE-07`) và làm phức tạp việc kiểm tra precision (`RULE-06`). | Loại bỏ |
| **B. Đặt Causal EMA (`smooth_scores`) + `min_hold_bars` trong `HysteresisBuffer` (`backtest/buffer.py`)** | Giữ `predictor.py` 100% stateless & decoupled (`RULE-07`); tầng thực thi (execution/position sizing) chịu trách nhiệm kiểm soát turnover; Tier 2 (`RealWorldBacktestRunner`) tự động hưởng lợi; mặc định `span=1, min_hold=0` bảo toàn tương thích ngược 100%. | Người dùng live trading cần cấu hình cùng `signal_ema_span` & `min_hold_bars` ở execution layer. | **Chọn (B)** |
| **C. Chỉ nới `top_pct` / `exit_pct` mà không thêm smoothing** | Không cần viết thêm code. | Không giải quyết được hiện tượng score giật 1 nến qua ngưỡng rồi quay đầu ngay ở nến sau; nới quá rộng sẽ giảm mạnh số lượng giao dịch có edge. | Bổ trợ, không thay thế (B) |

**Nguyên tắc chống OOS Overfitting (`RULE-01`):** `signal_ema_span` và `min_hold_bars` phải được chọn dựa trên **prior lý thuyết của horizon nhãn** (ví dụ `tb5_label` horizon 5 nến $\to$ `span` / `min_hold` khoảng `3..5`) hoặc kiểm chứng trên `Val`, tuyệt đối không grid-search trên `OOS`.

---

## 3. Chẩn đoán Đa Regime & Khớp Lệnh `next_open` (Phase 3)

| Phương án | Ưu điểm | Nhược điểm | Quyết định |
| :--- | :--- | :--- | :--- |
| **A. Dùng `Val` + `Cal-B` để tuning lại ngưỡng Hysteresis** | Có thể tối ưu Net Sharpe trên các đoạn cũ. | Nguy cơ overfit tham số backtest và vi phạm nguyên tắc blind evaluation nếu đụng vào OOS (`RULE-01`). | Loại bỏ |
| **B. Module chẩn đoán Read-Only (`backtest/diagnostics.py`) + CLI `quantautoml diagnose`** | Hoàn toàn không `fit()`, không rò rỉ dữ liệu, không làm chậm `quantautoml run` mặc định; phơi bày minh bạch điểm yếu regime (như `Cal-B` 2024) và chênh lệch `close` vs `next_open` trước khi quyết định deploy. | Không tự động sửa model khi gặp regime yếu (nhưng cung cấp bằng chứng định lượng chính xác để quyết định nâng cấp feature/regime filter ở version sau). | **Chọn (B)** |
