# Step 08-fix1 — Scope: Realistic OOS Simulation với backtesting.py

**Feature/Topic:** Nâng cấp step backtest thành mô phỏng thực chiến sát thực tế (Execution Realism Layer)
**Date:** 2026-11-19
**Status:** In Progress

---

## 1. Objectives & Goals

- **Vấn đề cần giải quyết:** Step 07 hiện tại dùng Vectorized Bar-to-Bar Return (NumPy) — cực nhanh để đánh giá Alpha thống kê (Rank IC, ICIR), nhưng **bỏ qua hoàn toàn các yếu tố thực chiến** của khớp lệnh thực tế (giá khớp Next-bar Open, quản lý vốn/ký quỹ, lot sizing, intrabar risk, trade log chi tiết).
- **Mục tiêu chính (Measurable Outcome):**
  1. Tích hợp `backtesting.py >= 0.3.3` như một **Tier 2 Execution Realism Validator** chạy sau khi Tier 1 (NumPy fast evaluator, hiện tại) đã PASS.
  2. Xuất **Trade Log** (`trades.csv`) và **Báo cáo HTML tương tác** (`backtest_report.html`) với biểu đồ nến + điểm vào/ra lệnh.
  3. Giữ nguyên kiến trúc Tier 1 (không phá vỡ các test hiện tại), Tier 2 chỉ được gọi nếu Tier 1 đã `accepted=True`.

---

## 2. In-Scope (Must Have)

- [ ] **Dependency:** Thêm `backtesting>=0.3.3` vào `pyproject.toml` (dev optional group, không bắt buộc runtime inference).
- [ ] **`BacktestingPyStrategy`** (`src/quantautoml/backtest/strategy.py`): Lớp `Strategy` kế thừa từ `backtesting.lib.Strategy`, đọc mảng tín hiệu (vị thế) đã tính sẵn từ `HysteresisBuffer` và phát lệnh Market tại giá `Open` nến kế tiếp.
- [ ] **`RealWorldBacktestRunner`** (`src/quantautoml/backtest/realworld_runner.py`): Điều phối toàn bộ luồng Tier 2: nạp OHLCV DataFrame → gắn tín hiệu → chạy `backtesting.Backtest` → trích xuất metrics → xuất HTML + CSV.
- [ ] **`RealWorldMetrics`** (Pydantic): Các chỉ số bổ sung từ `backtesting.py`: `win_rate`, `profit_factor`, `max_trade_loss_pct`, `avg_holding_bars`, `n_trades`, `equity_final`.
- [ ] **Config mở rộng:** Thêm `RealWorldConfig` vào `common/config.py`: `ohlcv_price_col: str = "Close"`, `cash: float = 1_000_000.0`, `commission: float = 0.0005`, `margin: float = 1.0`, `trade_on_close: bool = False` (mặc định: khớp lệnh tại Open nến kế tiếp).
- [ ] **Integration:** `OOSBacktestRunner.run()` trả về `(OOSMetrics, RealWorldMetrics | None)` — Tier 2 chỉ chạy khi caller truyền tham số `ohlcv_df: pd.DataFrame | None`.
- [ ] **Tests:**
  - `tests/unit/test_realworld_runner.py`: Unit test `BacktestingPyStrategy` với OHLCV giả lập nhỏ.
  - `tests/integration/test_realworld_backtest.py`: Integration test đầy đủ với bundle thật + tập OOS ~4000 nến.

---

## 3. Out-of-Scope (Non-Goals)

- **Không thay thế Tier 1 (NumPy vectorized).** Tier 1 vẫn là gate kiểm tra chính thức (`accepted` flag). Tier 2 chỉ bổ sung báo cáo thực chiến.
- **Không hỗ trợ danh mục nhiều tài sản (Multi-asset Portfolio):** Step này chỉ xử lý 1 tài sản (đúng quyết định D2 từ PLAN.md Phase 5). Multi-asset là phạm vi v1.1.
- **Không tích hợp Live/Paper Trading:** Chỉ phục vụ nghiệm thu offline OOS.
- **Không thay đổi logic `HysteresisBuffer` hay `FrictionCostModel`:** Đây là thành phần đã ổn định của Tier 1.
- **Không tối ưu hóa thêm siêu tham số từ kết quả Tier 2:** Tier 2 là kiểm tra nghiệm thu, không phải vòng lặp tối ưu.
- **Không xây dựng Walk-Forward backtesting:** Nằm ngoài phạm vi v1.0.

---

## 4. Constraints & Assumptions

**Constraints:**
- `backtesting.py` phụ thuộc `pandas` và `bokeh` (cho báo cáo HTML) — phải đưa vào **optional dev dependency**, không nhúng vào core inference bundle.
- `predictor.py` (RULE-07) **tuyệt đối không import** `backtesting` — Tier 2 chỉ được gọi từ `backtest/realworld_runner.py`.
- Khớp lệnh Tier 2 phải cấu hình `trade_on_close=False` mặc định để tránh thiên kiến giá đóng cửa (Look-Ahead bias tại bar boundary).
- Python ≥ 3.12, type hints 100%, exception kế thừa `QuantAutoMLError`.

**Assumptions:**
- Caller của `RealWorldBacktestRunner` cần truyền vào `pd.DataFrame` OHLCV với index là `DatetimeIndex` và các cột `Open`, `High`, `Low`, `Close`, `Volume` (tên tiêu chuẩn của `backtesting.py`).
- Tín hiệu vị thế (`positions` array từ `HysteresisBuffer`) đã được tính ở Tier 1 và truyền trực tiếp — Tier 2 **không tính lại tín hiệu**, chỉ replay theo tín hiệu đó.
- OOS đã được xác nhận không rò rỉ bởi Tier 1 (`DataLeakageError` đã được kiểm tra trước khi gọi Tier 2).
