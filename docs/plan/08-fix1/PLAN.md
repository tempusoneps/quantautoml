# Step 08-fix1 — PLAN.md: Realistic OOS Simulation với backtesting.py

> **For agentic workers:** TDD bắt buộc (RED → GREEN → REFACTOR); cập nhật `- [x]` sau mỗi task theo RULE-10.

**Active Phase:** Step 08-fix1 — Realistic Execution Layer (Fix 1 on Phase 5)
**Branch:** `develop/v1.0` (nhánh hiện tại — không tạo nhánh mới)
**Goal:** Tích hợp `backtesting.py` như một **Tier 2 Execution Realism Validator** chạy sau Tier 1 (NumPy fast evaluator). Tier 1 vẫn là gate `accepted` chính thức. Tier 2 bổ sung: khớp lệnh tại `Open[t+1]`, trade log (`trades.csv`), báo cáo HTML tương tác, và `RealWorldMetrics` (Win Rate, Profit Factor, Max Trade Loss...).
**Tech Stack:** `backtesting>=0.3.3` (dev group), `pandas`, `bokeh`, `numpy`, `pydantic` v2, `pytest`.
**Spec:** [`SCOPE.md`](SCOPE.md), [`DESIGN.md`](DESIGN.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Quyết định đã xác nhận

- **D1:** `backtesting` là **optional dev dependency** — không thêm vào `[project] dependencies`. Lazy import trong `realworld_runner.py`.
- **D2:** Tín hiệu vị thế `{-1, 0, +1}` được tính **một lần** ở Tier 1 (HysteresisBuffer) rồi **inject trực tiếp** vào `backtesting.py` qua cột `"_position"` trên OHLCV DataFrame. Không tính lại tín hiệu trong `Strategy.next()`.
- **D3:** Mặc định `trade_on_close=False` — khớp lệnh tại `Open` nến kế tiếp để tránh thiên kiến giá đóng cửa.
- **D4:** Tier 2 chỉ chạy khi `OOSBacktestRunner.run()` nhận `ohlcv_df` không phải `None` **và** Tier 1 trả về `accepted=True`.
- **D5:** `OOSBacktestRunner.run()` signature mở rộng: thêm tham số `ohlcv_df: pd.DataFrame | None = None`, trả về `tuple[OOSMetrics, RealWorldMetrics | None]`. *Backward compatible:* caller cũ bỏ qua `ohlcv_df` vẫn nhận `(OOSMetrics, None)`.

---

## Global Constraints & Hard Gates

- **RULE-01:** Không gọi `fit()` tại bất kỳ đâu trong step này.
- **RULE-07:** `predictor.py` trong bundle tuyệt đối không import `backtesting`. Chỉ `realworld_runner.py` lazy-import.
- **RULE-09:** Tier 2 bổ sung báo cáo thực chiến; Tier 1 vẫn là gate Net Sharpe chính thức sau phí.
- **No Regression:** Toàn bộ test suite hiện tại (`uv run pytest tests/`) phải PASS sau mọi commit.
- Python ≥ 3.12, type hints 100%, exception kế thừa `QuantAutoMLError`.

---

## Execution Order Overview

- [x] **Task 1: `uv add backtesting` (dev) + `RealWorldConfig`**
- [x] **Task 2: `BacktestingPyStrategy` (`backtest/strategy.py`)**
- [x] **Task 3: `RealWorldMetrics` + `RealWorldBacktestRunner` (`backtest/realworld_runner.py`)**
- [ ] **Task 4: Mở rộng `OOSBacktestRunner.run()` — Tier 2 integration**
- [ ] **Task 5: Integration test + docs + commit tổng**

---

## Detailed Task Breakdown

---

### [x] Task 1: Dependency + `RealWorldConfig`

**Files:**
- Modify: `pyproject.toml` — thêm `backtesting>=0.3.3` vào `[dependency-groups] dev`
- Modify: `src/quantautoml/common/config.py` — thêm `RealWorldConfig`
- Test: `tests/unit/test_realworld_config.py`

**Interfaces — Produces:**
```python
class RealWorldConfig(BaseModel):  # frozen, extra="forbid"
    cash: float = Field(default=1_000_000.0, gt=0)
    commission: float = Field(default=0.0005, ge=0)
    margin: float = Field(default=1.0, gt=0)
    trade_on_close: bool = False
    out_html: str | None = None
    out_trades_csv: str | None = None
    ohlcv_open_col: str = "Open"
    ohlcv_high_col: str = "High"
    ohlcv_low_col: str = "Low"
    ohlcv_close_col: str = "Close"
    ohlcv_volume_col: str = "Volume"
```

**Action Steps:**
- [x] Step 1: Viết test `test_defaults_match_spec` (cash=1M, commission=0.0005, trade_on_close=False), `test_cash_zero_rejected` (`pytest.raises(ValidationError)`), `test_negative_commission_rejected`, `test_extra_field_forbidden`.
- [x] Step 2: Chạy → **FAIL** (class chưa tồn tại).
- [x] Step 3: `uv add --dev backtesting`; thêm `RealWorldConfig` vào `config.py`.
- [x] Step 4: `uv run pytest tests/unit/test_realworld_config.py -v` → **PASS**. Chạy full suite `uv run pytest tests/` → không regression.
- [x] Step 5: Commit `feat(setup): add backtesting dev dep and RealWorldConfig`.

---

### [x] Task 2: `BacktestingPyStrategy`

**Files:**
- Create: `src/quantautoml/backtest/strategy.py`
- Test: `tests/unit/test_backtest_strategy.py`

**Interfaces — Produces:**
```python
class BacktestingPyStrategy:  # kế thừa backtesting.Strategy
    """Reads pre-computed _position column; issues Market orders on signal change."""
    def init(self) -> None: ...
    def next(self) -> None: ...
        # Nếu target == current: pass
        # Nếu có vị thế hiện tại: self.position.close()
        # target == 1: self.buy(); target == -1: self.sell()
```

**Action Steps:**
- [x] Step 1: Viết test với `backtesting.Backtest` chạy trên OHLCV giả lập nhỏ (100 nến):
  - `test_strategy_long_signal_opens_position`: positions = `[0]*20 + [1]*10 + [0]*70` → phải có đúng 1 trade Long.
  - `test_strategy_short_signal_opens_short`: positions short → phải có đúng 1 trade Short.
  - `test_strategy_no_signal_no_trades`: positions = `[0]*100` → `n_trades == 0`.
  - `test_strategy_flip_long_to_short_no_flat_in_between`: `[1]*50 + [-1]*50` → 2 trades, không có bar flat giữa chừng.
- [x] Step 2: Chạy → **FAIL**.
- [x] Step 3: Hiện thực `BacktestingPyStrategy` với lazy import guard:
  ```python
  try:
      from backtesting import Strategy
  except ImportError as e:
      raise ImportError("backtesting package required: uv add --dev backtesting") from e
  ```
- [x] Step 4: `uv run pytest tests/unit/test_backtest_strategy.py -v` → **PASS**. Full suite → no regression.
- [x] Step 5: Commit `feat(backtest): add BacktestingPyStrategy adapter for pre-computed signals`.

---

### [x] Task 3: `RealWorldMetrics` + `RealWorldBacktestRunner`

**Files:**
- Create: `src/quantautoml/backtest/realworld_runner.py`
- Modify: `src/quantautoml/backtest/__init__.py` — export `RealWorldBacktestRunner`, `RealWorldMetrics`
- Test: `tests/unit/test_realworld_runner.py`

**Interfaces — Produces:**
```python
class RealWorldMetrics(BaseModel):  # frozen
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_holding_bars: float
    max_trade_loss_pct: float
    equity_final: float
    equity_peak: float
    return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio_bt: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float

class RealWorldBacktestRunner:
    def __init__(self, cfg: RealWorldConfig) -> None: ...

    def run(
        self,
        *,
        ohlcv_df: pd.DataFrame,       # DatetimeIndex, cols: Open High Low Close Volume
        positions: NDArray[np.float64], # {-1, 0, +1}, len == len(ohlcv_df)
    ) -> RealWorldMetrics:
        """Inject positions → Backtest → extract metrics → optional export."""
        ...
```

**Validation trong `run()`:**
- `ohlcv_df` phải có `DatetimeIndex`.
- `len(positions) == len(ohlcv_df)` → `ValueError`.
- Các cột `Open/High/Low/Close/Volume` (theo `cfg.ohlcv_*_col`) phải tồn tại → `BacktestError`.
- `ohlcv_df` không được chứa NaN trong 4 cột giá → `BacktestError`.
- Nếu `n_trades == 0`: trả về `RealWorldMetrics` với các chỉ số bằng `0.0` (không ném exception — chiến lược flat hoàn toàn vẫn hợp lệ).

**Action Steps:**
- [x] Step 1: Viết test với OHLCV giả lập 500 nến + positions giả lập:
  - `test_metrics_has_expected_fields`: output phải là `RealWorldMetrics` với tất cả fields.
  - `test_zero_trades_returns_zero_metrics`: positions = `[0]*500` → `n_trades == 0`, không crash.
  - `test_mismatched_length_raises`: len(positions) != len(df) → `ValueError`.
  - `test_missing_ohlcv_column_raises`: df thiếu cột `High` → `BacktestError`.
  - `test_nan_in_close_raises`: `Close` có NaN → `BacktestError`.
  - `test_win_rate_range`: `0.0 <= metrics.win_rate <= 1.0`.
  - `test_equity_final_positive`: `equity_final > 0`.
  - `test_html_export_creates_file` (tmp_path): khi `cfg.out_html` được set, file HTML phải tồn tại sau `run()`.
  - `test_csv_export_creates_file` (tmp_path): khi `cfg.out_trades_csv` được set, file CSV phải tồn tại.
- [x] Step 2: Chạy → **FAIL**.
- [x] Step 3: Hiện thực `RealWorldBacktestRunner.run()`:
  1. Rename các cột OHLCV về `Open/High/Low/Close/Volume` chuẩn của `backtesting.py` nếu cần.
  2. Inject `positions` vào `ohlcv_df["_position"]`.
  3. Khởi tạo và chạy `Backtest(data=ohlcv_df, strategy=BacktestingPyStrategy, cash=cfg.cash, commission=cfg.commission, margin=cfg.margin, trade_on_close=cfg.trade_on_close)`.
  4. Trích xuất `stats` dict → build `RealWorldMetrics`.
  5. Nếu `out_html`: `bt.plot(filename=cfg.out_html, open_browser=False)`.
  6. Nếu `out_trades_csv`: `stats._trades.to_csv(cfg.out_trades_csv, index=False)`.
- [x] Step 4: `uv run pytest tests/unit/test_realworld_runner.py -v` → **PASS**. Full suite → no regression.
- [x] Step 5: Commit `feat(backtest): add RealWorldMetrics and RealWorldBacktestRunner`.

---

### [ ] Task 4: Mở Rộng `OOSBacktestRunner.run()` — Tier 2 Integration

**Files:**
- Modify: `src/quantautoml/backtest/runner.py` — mở rộng signature `run()`, tích hợp Tier 2
- Test: `tests/unit/test_runner_tier2_integration.py`

**Mở rộng signature (backward compatible):**
```python
def run(
    self,
    *,
    bundle_dir: Path,
    oos: Segment,
    feature_names: Sequence[str],
    other_segments: Sequence[Segment],
    out_path: Path | None = None,
    latency_repeats: int = 200,
    latency_budget_ms: float = 5.0,
    # --- MỚI (optional, backward compatible) ---
    ohlcv_df: pd.DataFrame | None = None,
    realworld_cfg: RealWorldConfig | None = None,
) -> tuple[OOSMetrics, RealWorldMetrics | None]:
```

**Logic Tier 2 trigger:**
```python
rw_metrics: RealWorldMetrics | None = None
if ohlcv_df is not None and metrics.accepted:
    rw_cfg = realworld_cfg or RealWorldConfig()
    rw_metrics = RealWorldBacktestRunner(rw_cfg).run(
        ohlcv_df=ohlcv_df,
        positions=positions,  # array đã tính ở Tier 1, trước khi cắt positions[:-1]
    )
return metrics, rw_metrics
```

> **Lưu ý:** `positions` ở đây là **toàn bộ** mảng vị thế (len = `n`), **không** phải `pos = positions[:-1]`. `RealWorldBacktestRunner` cần aligned đúng với `ohlcv_df` đầy đủ.

**Action Steps:**
- [ ] Step 1: Viết test:
  - `test_run_without_ohlcv_returns_none_rw`: Caller không truyền `ohlcv_df` → `rw_metrics is None`.
  - `test_run_with_ohlcv_accepted_returns_rw_metrics`: Bundle hợp lệ + `ohlcv_df` đủ chuẩn → `rw_metrics is not None` và `isinstance(rw_metrics, RealWorldMetrics)`.
  - `test_run_with_ohlcv_rejected_skips_tier2`: Tier 1 `accepted=False` (ví dụ: Sharpe quá thấp do config ngưỡng cao) → `rw_metrics is None` dù `ohlcv_df` được truyền.
  - `test_backward_compat_old_caller`: Caller cũ chỉ nhận `OOSMetrics` từ tuple → `oos_metrics, _ = runner.run(...)` vẫn hoạt động.
- [ ] Step 2: Chạy → **FAIL**.
- [ ] Step 3: Hiện thực mở rộng `runner.py`.
- [ ] Step 4: `uv run pytest tests/unit/test_runner_tier2_integration.py -v` → **PASS**. Full suite → no regression.
- [ ] Step 5: Commit `feat(backtest): extend OOSBacktestRunner with Tier 2 realworld simulation`.

---

### [ ] Task 5: Integration Test + Docs + Phase Handoff

**Files:**
- Create: `tests/integration/test_realworld_backtest.py`
- Modify: `docs/plan/PLAN.md` — đánh dấu `[x]` tất cả task
- Modify: `docs/plan/08-fix1/SCOPE.md` — cập nhật Status → `Completed`

**Integration test** (`tests/integration/test_realworld_backtest.py`):
```
test_tier2_produces_html_and_csv_on_real_bundle:
  - Dùng bundle fixture từ tests/integration/bundle_helper.py
  - Sinh OHLCV DataFrame với Close liên tục và cột Open/High/Low/Volume hợp lệ
  - Chạy OOSBacktestRunner.run(..., ohlcv_df=df, realworld_cfg=RealWorldConfig(out_html=str(tmp_html), out_trades_csv=str(tmp_csv)))
  - Assert: oos_metrics.accepted == True (hoặc kiểm tra rw_metrics không phải None)
  - Assert: tmp_html tồn tại và có nội dung > 0 bytes
  - Assert: tmp_csv tồn tại, có thể đọc bằng pandas, cột "PnL" tồn tại

test_tier2_skipped_when_tier1_rejected:
  - Cấu hình BacktestConfig với min_net_sharpe=999.0 (đảm bảo Tier 1 fail)
  - rw_metrics phải là None
```

**Action Steps:**
- [ ] Step 1: Viết integration tests.
- [ ] Step 2: Chạy → **FAIL** (Tier 2 chưa kết nối).
- [ ] Step 3: Chạy toàn bộ implementation từ Task 1-4 → integration test **PASS**.
- [ ] Step 4: `uv run pytest tests/ -v` → **100% PASS, 0 regression**.
- [ ] Step 5: Cập nhật tất cả checkbox `[x]` trong file này. Cập nhật `SCOPE.md` → `Status: Completed`. Commit `feat(backtest): full Tier 2 realistic backtest integration with backtesting.py + tests`.

---

## Final Verification Checklist

- [ ] `uv run pytest tests/` — 100% PASS, 0 regression.
- [ ] `backtesting` không xuất hiện trong `src/quantautoml/packaging/templates/predictor_template.py` (RULE-07).
- [ ] `OOSBacktestRunner.run()` không có `ohlcv_df` vẫn trả về `(OOSMetrics, None)` (backward compat).
- [ ] HTML report được tạo thành công với plot nến + điểm vào/ra lệnh.
- [ ] `trades.csv` có đủ cột: `EntryBar`, `ExitBar`, `EntryPrice`, `ExitPrice`, `PnL`, `ReturnPct`, `Duration`.
- [ ] Type hints 100% — không có `Any` không rõ nguồn gốc.
- [ ] Exceptions kế thừa `QuantAutoMLError`.
