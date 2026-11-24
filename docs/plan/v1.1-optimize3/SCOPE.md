# Project / Feature Scope — `v1.1-optimize3` (IC Window Alignment, Turnover Control & Regime Diagnostics)

**Feature/Topic:** Horizon-Aware Rolling IC Window, Turnover-Controlled Hysteresis & Cross-Regime Execution Verification (`v1.1-optimize3`)  
**Branch:** `develop/v1.1`  
**Date:** 2026-11-24  
**Status:** Ready for Implementation (Reviewed & Aligned with Codebase)

---

## 1. Objectives & Goals

Dựa trên phân tích thực nghiệm đối chiếu hai run liên tiếp [`runs/tb5_label_nov24v2`](../../../runs/tb5_label_nov24v2/) (FAIL 2/6 gates: `icir`, `net_sharpe`) và [`runs/tb5_label_nov24v3`](../../../runs/tb5_label_nov24v3/) (PASS 6/6 gates: `Net Sharpe = 1.447`, `OOS Rank IC = 0.0846`, `ICIR = 0.497`), mục tiêu của `v1.1-optimize3` là giải quyết dứt điểm 3 điểm nghẽn định lượng trước khi bước sang `v1.1-fix2` Phase 2 (`Train+Cal Refit` & `Multi-Window Walk-Forward OOS`):

1. **Triệt tiêu lỗi "Simpson's Paradox" khi tính IC Sharpe với cửa sổ quá ngắn (`w = 50–63` nến):**
   - Thực nghiệm trên dữ liệu 5 phút với `tb5_label` cho thấy `ICIR(w=50)` tạo giá trị **âm giả**:
     - Trên `v3`: trên `cal_a` (nơi `BootstrapShrinkageEngine` fit theo thứ tự nhân quả `v1.1-fix1`), `model_0.onnx` đạt `overall_IC = +0.1120` và `ICIR(500) = +0.4752` (sau `T=1.2694`: `+0.1078` / `+0.4451`), nhưng `ICIR(50) = -0.1564` (ghi trong `01_ensemble_report.json`); ngay trên `oos`, `overall_IC = +0.0846` (`+0.0830` trước `T`) và `ICIR(500) = +0.4970`, nhưng `ICIR(50) = -0.0447` (`-0.0418` trước `T`).
     - Trên `v2`: trên `cal_a`, cả 3 model có `ICIR(500)` dương mạnh (`+0.464` đến `+0.627`) nhưng đều âm giả ở `w=50` (`-0.1267`, `-0.0694`, `-0.0207`), khiến `BootstrapShrinkageEngine` từ chối ensemble và fallback về CatBoost (`weights = [0, 0, 1]`, sau đó yếu nhất trên `OOS` của `v2`: `+0.0585` vs LightGBM `+0.0789`, XGBoost `+0.0853`). Còn `ModelTournament` chấm CatBoost `ic_sharpe = -0.0178` dù cả 5 fold đều dương (`mean_ic = +0.0963`).
   - **Tiêu chí thành công:** Cửa sổ rolling IC trong `tournament.py`, `tuner.py` và `bootstrap.py` co giãn hợp lý theo cỡ mẫu / cấu hình `ic_window` (tối đa `500` nến), chấm đúng dấu `ICIR` trên dữ liệu intraday mà không phá vỡ unit test mẫu nhỏ hoặc chế độ `OOF_ASSISTED`.
   - **Giới hạn kỳ vọng:** Phase 1 khắc phục sai lệch **phép đo & lựa chọn model/ensemble**, không tự triệt tiêu hiện tượng lệch phân phối (regime shift) thực sự trên các giai đoạn yếu như `cal_b` của `v3` (`overall_IC = +0.0142`, `ICIR(500) = +0.0223`).
2. **Giảm hao hụt Sharpe do chi phí ma sát (`Gross 2.324 → Net 1.447`, mất 38% Sharpe):**
   - `mean_turnover = 0.0367/bar` (~443x/năm) trên khung 5m do phản ứng từng nến với raw score cho nhãn horizon 5 nến.
   - **Tiêu chí thành công:** Hỗ trợ làm mượt tín hiệu nhân quả (`signal_ema_span`) và thời gian nắm giữ tối thiểu (`min_hold_bars`) trong `HysteresisBuffer` (opt-in, mặc định giữ nguyên hành vi cũ), kéo giảm turnover và cải thiện tỷ lệ `Net Sharpe / Gross Sharpe` mà không grid-search trên `OOS` (`RULE-01`).
3. **Kiểm chứng thực chiến đa regime và khớp lệnh giá `next_open` / Tier 2 (`realworld`):**
   - `v3` mạnh ở `Cal-A` (`IC = +0.1078` sau `T`) và `OOS` (`IC = +0.0846`), trung bình ở `Val` (`IC = +0.0441`), yếu ở `Cal-B` (`IC = +0.0142`), và mới chỉ kiểm chứng ở `execution_price = "close"`.
   - **Tiêu chí thành công:** Tiện ích chẩn đoán đa segment (read-only, zero-leakage) cho thấy rõ hiệu năng trên từng regime và độ bền của `Net Sharpe` khi chuyển sang `execution_price = "next_open"` + Tier 2 (`RealWorldBacktestRunner`).

---

## 2. In-Scope (Must Have)

### Phase 1 — Horizon-Aware Rolling IC Window & Ensemble Diagnostics *(Done)*
- [x] Thêm `ic_window: int | None = None` (`ge=20`) vào `ModelsConfig` và `EnsembleConfig` (`src/quantautoml/common/config.py`). Khi `None`, tự động chọn cửa sổ thích ứng theo độ dài chuỗi (`max(10, min(500, n // 10))` cho `bootstrap.py`, `max(20, min(500, n // 4))` cho `fold_subwindow_ics` trong `tournament.py`); khi cấu hình tường minh, clamp theo `n // 4` (khi `n` đủ lớn) để tránh suy biến về 2 cửa sổ (`ddof=1`).
- [x] Truyền `ic_window` xuyên suốt `ModelTournament`, `DeepOptunaTuner`, và `BootstrapShrinkageEngine`.
- [x] Bổ sung các trường chẩn đoán vào `EnsembleWeightsResult` (`ic_window_used: int`, `single_model_scores: list[float]`, `overall_rank_ic: float`) ở **cả nhánh `n_models == 1` lẫn `n_models > 1`**, và sửa điều kiện fallback `score == 0.0` trong `_calc_series_ic_sharpe` để chỉ fallback sang `Rank IC` khi không đủ cửa sổ hợp lệ (`valid < 2` hoặc `std <= 1e-12`).
- [x] Đồng bộ `scripts/generate_config.py` và `docs/CONFIGURATION.md`.
- [x] Unit tests theo TDD trong `test_models_config.py`, `test_ensemble_config.py`, `test_tournament.py`, `test_tuner.py`, `test_bootstrap_shrinkage.py` bao phủ cả mẫu nhỏ (`n=100..250`), chuỗi intraday dài (`n=4500..7500`), truyền `cfg.ic_window` qua `DeepOptunaTuner`, và kiểm tra biên `ic_window < 3`.

### Phase 2 — Causal Signal Smoothing & Minimum Holding Period *(In Progress)*
- [ ] Thêm `signal_ema_span: int = 1` (`ge=1`) và `min_hold_bars: int = 0` (`ge=0`) vào `BacktestConfig`.
- [ ] Hiện thực hóa `smooth_scores()` (EMA nhân quả an toàn với `NaN`/`±inf`, identity khi `span == 1`), nâng cấp `rolling_percentile()` thành NaN-safe (không biến `NaN` thành percentile âm giả kích hoạt lệnh Short), và thêm logic `min_hold_bars` trong `HysteresisBuffer` (`src/quantautoml/backtest/buffer.py`); nối từ `OOSBacktestRunner` (vẫn giữ **raw** `scores` khi chấm `rank_ic`/`icir`/`downside_ic` theo `RULE-02`).
- [ ] Unit tests trong `test_backtest_config.py`, `test_buffer.py`, và `test_runner.py` bảo đảm **Zero Look-Ahead Leakage** (`positions[:t+1]` bất biến khi thay đổi `scores[t+1:]`), xử lý biên `NaN` ở `positions()`, biên `min_hold_bars`, truyền tham số từ `OOSBacktestRunner`, và giảm turnover trên chuỗi nhiễu.

### Phase 3 — Cross-Regime Diagnostic & Realistic Execution Verification *(Planned)*
- [ ] Thêm tiện ích chẩn đoán đa segment (`src/quantautoml/backtest/diagnostics.py`, read-only, zero-leakage, dùng chung helper tính return/metric với `runner.py`) + lệnh CLI `quantautoml diagnose --run-dir <dir>` đo Rank IC, ICIR, Gross/Net Sharpe (`close` vs `next_open`) trên `cal_a`, `cal_b`, `val`, `oos`, xuất `06_backtest/03_regime_diagnostics.json`. Dùng `SegmentDiagnostic` tách biệt khỏi `OOSMetrics` để không báo trượt gate `"latency"` giả khi `latency_ms=None`; tự skip mượt khi segment thiếu/ngắn/giá `NaN`/`<= 0`/thiếu `y`, và fallback về `BacktestConfig()` mặc định nếu `config.toml` không có `[backtest]`.
- [ ] Unit/CLI tests trong `test_backtest_diagnostics.py` và `test_cli.py`; cập nhật `docs/OUTPUTS.md` (đồng bộ cây `01_data/`..`06_backtest/`) và `docs/STRUCTURE.md`.
- [ ] Chạy nghiệm thu so sánh `execution_price = "next_open"` và Tier 2 (`RealWorldBacktestRunner`) trên `tb5_label` (ghi rõ bộ tham số `config.toml` dùng cho benchmark và lưu ý Tier 2 chỉ tự kích hoạt trong `OOSBacktestRunner` khi Tier 1 `accepted == true`).

---

## 3. Out-of-Scope (Non-Goals)

- Không thay đổi phễu lọc feature `02_features` (`F1 → F8`) hoặc biên regularization đã hoàn thiện ở `v1.1-fix2` Phase 1.
- Không sửa đổi cấu trúc ONNX Model Bundle hay `predictor.py` (`RULE-06`, `RULE-07`).
- Không thay đổi cơ chế Temperature Scaling trên `Cal-B` (`RULE-03`) hay hàm mục tiêu $C^2$ của SLSQP trên `Cal-A` (`RULE-04`).
- Không gộp phần `Train+Cal Refit` hay `Multi-Window Walk-Forward OOS` (thuộc `v1.1-fix2` Phase 2, thực hiện sau `v1.1-optimize3`).

---

## 4. Constraints & Assumptions

- **Hard Gates:** Tuân thủ nghiêm ngặt `RULE-01` đến `RULE-10` trong `AGENTS.md`.
- **Tương thích ngược 100%:** Mọi trường config mới đều có giá trị mặc định an toàn (`ic_window = None`, `signal_ema_span = 1`, `min_hold_bars = 0`); các file `config.toml` cũ chạy không lỗi.
- **TDD & Commit Policy:** Viết failing test trước trong từng task (RED → GREEN → REFACTOR); 1 commit sạch cho mỗi Phase sau khi `uv run pytest tests/` PASS 100% và `graphify update .` hoàn tất.
