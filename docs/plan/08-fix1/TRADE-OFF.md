# Step 08-fix1 — Trade-off: Dual-Tier Backtesting Architecture

**Feature/Topic:** Nâng cấp Backtest Layer thành Realistic Execution Simulation (Step 08-fix1)
**Date:** 2026-11-19
**Status:** Approved

---

## 1. Kiến Trúc: Tier 1 (NumPy Fast) vs Tier 2 (backtesting.py Realistic) vs Thay thế hoàn toàn

### Các hướng tiếp cận:

- **Phương án A — Giữ nguyên Tier 1 (hiện tại):** Chỉ dùng NumPy vectorized return.
  - *Pros:* Siêu nhanh, không phụ thuộc thêm thư viện, test đầy đủ.
  - *Cons:* Giả định khớp lệnh tại giá Close (thiên kiến), không có trade log, không phát hiện được các chiến lược bị hủy diệt bởi intrabar Stop-loss, không có biểu đồ trực quan.

- **Phương án B — Thay thế hoàn toàn bằng `backtesting.py`:** Dùng `backtesting.py` làm engine duy nhất.
  - *Pros:* Mô phỏng thực chiến đầy đủ nhất.
  - *Cons:* `backtesting.py` chậm hơn NumPy vectorized ~10-50x do vòng lặp Python theo từng lệnh; phụ thuộc nặng vào `pandas`/`bokeh`; không tính Rank IC/ICIR trực tiếp; phá vỡ toàn bộ test suite hiện tại.

- **Phương án C — Dual-Tier (Được chọn):** Tier 1 (NumPy, hiện tại) là gate thống kê chính thức → Tier 2 (`backtesting.py`) chạy sau khi Tier 1 PASS để bổ sung báo cáo thực chiến.
  - *Pros:*
    1. **Không phá vỡ kiến trúc hiện tại:** Tier 1 vẫn là gate `accepted` chính thức theo RULE-02 và RULE-09; test suite cũ không cần sửa.
    2. **Bổ sung Execution Realism có chọn lọc:** Tier 2 chỉ chạy khi caller cung cấp OHLCV DataFrame — tức là optional, không ảnh hưởng pipeline AutoML tự động.
    3. **Tín hiệu dùng chung:** `HysteresisBuffer.positions()` đã tính ở Tier 1 được tái sử dụng trực tiếp cho Tier 2 — không tính lại alpha, không thêm look-ahead.
    4. **Trade log và HTML report:** Phục vụ kiểm tra thủ công của Quant researcher trước khi triển khai live.
  - *Cons:* Thêm dependency `backtesting` + `bokeh` vào dev group; cần map `positions` array → `backtesting.py` Strategy callback (một lớp adapter mỏng).

> **Quyết định:** Áp dụng **Phương án C (Dual-Tier)**. Tier 1 bảo toàn tốc độ và tính nghiêm ngặt thống kê; Tier 2 bổ sung tính thực chiến và khả năng quan sát (observability) cho researcher.

---

## 2. Thời Điểm Khớp Lệnh: `trade_on_close=True` vs `trade_on_close=False`

- **`trade_on_close=True`:** Lệnh phát tại thanh nến $t$ khớp ngay tại giá `Close[t]`.
  - *Cons:* **Look-ahead bias tại bar boundary** — tín hiệu và giá khớp cùng một thời điểm, không có độ trễ xử lý. Vi phạm giả định nhân quả.

- **`trade_on_close=False` (Được chọn — Next-bar Open):** Lệnh phát tại thanh $t$ khớp tại `Open[t+1]`.
  - *Pros:* Phản ánh đúng thực tế: tín hiệu được tính sau khi nến $t$ đóng, lệnh được gửi đến sàn trước khi nến $t+1$ mở. Phù hợp với môi trường VN30F (futures).
  - *Cons:* Slippage tại `Open` có thể lớn hơn khi có tin tức qua đêm.

> **Quyết định:** Mặc định `trade_on_close=False`. Cho phép cấu hình qua `RealWorldConfig.trade_on_close`.

---

## 3. Vị Thế: `[-1, 0, +1]` Discrete vs Fractional Sizing

- **Discrete `{-1, 0, +1}` (Được chọn):** Mỗi thanh nến chỉ có 3 trạng thái: Short, Flat, Long. Phù hợp với giao dịch hợp đồng tương lai nguyên lot (1 hợp đồng VN30F).
  - *Pros:* Nhất quán 100% với Tier 1 (HysteresisBuffer trả về `{-1, 0, +1}`); dễ kiểm thử, không ambiguity trong số lượng hợp đồng.
  - *Cons:* Không mô phỏng Kelly Criterion hay Position Sizing theo biến động.

- **Fractional Sizing theo Volatility:** `size = Kelly_fraction * account_equity / ATR`.
  - *Cons:* Phức tạp hóa không cần thiết ở bước nghiệm thu OOS; phù hợp hơn với module riêng trong v1.1.

> **Quyết định:** Dùng Discrete `{-1, 0, +1}` với `size=1` (1 hợp đồng) nhất quán với Tier 1. Position Sizing nâng cao thuộc v1.1.

---

## 4. Tín Hiệu Vào Lệnh: Tính lại trong Strategy vs Pre-computed Array Injection

- **Tính lại trong Strategy Callback:** Gọi lại `HysteresisBuffer` và `predictor.predict()` bên trong `Strategy.next()`.
  - *Cons:* Chậm hơn nhiều (mỗi nến gọi ONNX inference); logic tính vị thế bị nhân đôi, dễ drift so với Tier 1 (phiên bản khác nhau gây inconsistency).

- **Pre-computed Array Injection (Được chọn):** `RealWorldBacktestRunner` nhận vào `positions` array đã tính ở Tier 1 (hoặc tính lại trước khi gọi backtesting), inject vào `Strategy` như một cột phụ trên OHLCV DataFrame.
  - *Pros:* **100% consistency với Tier 1** — cùng một mảng vị thế, chỉ khác giá khớp (`Open[t+1]` thay vì `Close[t]/Close[t+1]`); nhanh, không gọi ONNX trong vòng lặp.

> **Quyết định:** Pre-computed injection. `BacktestingPyStrategy` đọc cột `_position` (đã được ghép vào DataFrame trước khi khởi tạo `Backtest`) và chỉ phát `self.buy()` / `self.sell()` / `self.position.close()` theo giá trị của cột đó.

---

## 5. Phụ Thuộc: `backtesting` vào Core vs Optional Dev Group

- **Core dependency:** Mọi người dùng `quantautoml` đều phải cài `backtesting`.
  - *Cons:* Kéo theo `pandas`, `bokeh` — nặng, không cần thiết cho inference engine. Vi phạm tinh thần RULE-07 (Decoupled Inference Engine).

- **Optional dev dependency (Được chọn):**
  ```toml
  [dependency-groups]
  dev = ["pytest>=8.0.0", "backtesting>=0.3.3", "bokeh>=3.0.0"]
  ```
  - *Pros:* Core library vẫn gọn nhẹ. `RealWorldBacktestRunner` lazy-import `backtesting` và ném `ImportError` có hướng dẫn rõ ràng nếu thiếu.

> **Quyết định:** Thêm vào `[dependency-groups] dev`, không thêm vào `[project] dependencies`.
