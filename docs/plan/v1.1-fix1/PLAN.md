# Implementation Plan — `v1.1-fix1` (Phase 3: Time-Series Bootstrap & Production Backtest Realism)

**Active Phase:** Phase 3 — Time-Series Bootstrap & Production Backtest Realism (`Audit #7, #8`)
**Branch:** `develop/v1.1`
**Commit Policy:** 1 commit duy nhất sau khi hoàn thành toàn bộ Phase 3 và `uv run pytest tests/` PASS 100%.
**Prerequisites:** Phase 1 & Phase 2 Done ([`PHASES.md`](PHASES.md)), [`SCOPE.md`](SCOPE.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Execution Order Overview

- [x] **Task 1 (Audit #7):** `BootstrapShrinkageEngine` — Moving Block Bootstrap (`bootstrap_block_size`) & Out-of-Bag (OOB) Hurdle Gate (`RULE-05`)
- [x] **Task 2 (Audit #8a):** `evaluate_oos_performance` — Daily-Aggregated Net Sharpe ($\times\sqrt{252}$) & Deflated Sharpe Ratio (DSR)
- [x] **Task 3 (Audit #8b):** `BacktestConfig` & `OOSBacktestRunner` — tùy chọn khớp lệnh `execution_price = "next_open"` (`Open[t+1] → Open[t+2]`) đồng bộ với Tier 2
- [x] **Task 4:** Full verification (`uv run pytest tests/`), cập nhật `docs/CONFIGURATION.md`, `graphify update .`, đánh dấu Phase 3 Done & 1 commit gọn cho toàn bộ Phase 3

---

## Detailed Task Breakdown

### [ ] Task 1: Moving Block Bootstrap & OOB Hurdle Gate (Audit #7)
- **Files to Touch:**
  - Modify: `src/quantautoml/common/config.py` (`EnsembleConfig.bootstrap_block_size: int = 20`)
  - Modify: `src/quantautoml/ensemble/bootstrap.py` (`_sample_block_indices`, OOB predictions cho Hurdle `+5%`)
  - Modify: `tests/unit/test_bootstrap_shrinkage.py`
- **Action Steps:**
  1. **(RED)** Viết unit test:
     - `test_block_bootstrap_preserves_contiguous_blocks`: xác nhận chỉ số mẫu sinh ra bởi `_sample_block_indices` gồm các khối liên tiếp có độ dài `block_size`.
     - `test_oob_hurdle_rejects_spurious_noise_overfit`: trên dữ liệu nhiễu thuần với nhiều model, OOB hurdle không bị lạc quan in-sample.
  2. **(GREEN)** Hiện thực Moving Block Bootstrap và OOB ensemble evaluation trong `bootstrap.py`.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_bootstrap_shrinkage.py -v
  ```

---

### [ ] Task 2: Daily-Aggregated Net Sharpe & Deflated Sharpe Ratio (Audit #8a)
- **Files to Touch:**
  - Modify: `src/quantautoml/backtest/evaluator.py` (`OOSMetrics.daily_net_sharpe`, `OOSMetrics.deflated_sharpe_ratio`)
  - Modify: `src/quantautoml/backtest/runner.py` (truyền `timestamps=oos.timestamps[:-1]`)
  - Modify: `tests/unit/test_evaluator.py`
- **Action Steps:**
  1. **(RED)** Viết unit test trong `tests/unit/test_evaluator.py`:
     - `test_daily_net_sharpe_compounds_intraday_returns_and_scales_sqrt252`: kiểm tra công thức ghép lãi theo ngày và nhân $\sqrt{252}$.
     - `test_deflated_sharpe_ratio_in_unit_interval_and_monotonic`: DSR $\in [0, 1]$, cao gần 1 với alpha mạnh và thấp với nhiễu thuần.
  2. **(GREEN)** Hiện thực `_daily_net_sharpe` và `_deflated_sharpe_ratio` trong `evaluator.py`, nối vào `runner.py`.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_evaluator.py tests/unit/test_runner.py -v
  ```

---

### [ ] Task 3: Tier 1 `next_open` Execution Alignment (Audit #8b)
- **Files to Touch:**
  - Modify: `src/quantautoml/common/config.py` (`BacktestConfig.execution_price`, `BacktestConfig.open_col`)
  - Modify: `src/quantautoml/backtest/runner.py`
  - Modify: `tests/unit/test_runner.py`
- **Action Steps:**
  1. **(RED)** Viết unit test trong `tests/unit/test_runner.py`:
     - `test_runner_next_open_execution_uses_open_t_plus_1`: xác nhận `execution_price="next_open"` dùng cột `Open` (báo `BacktestError` nếu thiếu cột `Open`), và `n_bars == len(oos) - 2`.
  2. **(GREEN)** Hiện thực `execution_price` trong `BacktestConfig` và `OOSBacktestRunner`.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_runner.py tests/unit/test_backtest_config.py -v
  ```

---

### [ ] Task 4: Full Verification, Docs, Graphify & Single Phase 3 Commit
- **Verification Command:**
  ```bash
  uv run pytest tests/ && graphify update .
  ```
- **Commit Message:** `feat(backtest): add block bootstrap OOB hurdle, daily Net Sharpe, DSR, and next_open execution (v1.1-fix1 Phase 3)`

---

## Final Verification Checklist (Phase 3)
- [x] `uv run pytest tests/` PASS 100%.
- [x] Moving Block Bootstrap & OOB Hurdle hoạt động chính xác trong `BootstrapShrinkageEngine`.
- [x] `OOSMetrics` báo cáo đầy đủ `daily_net_sharpe` và `deflated_sharpe_ratio`.
- [x] `BacktestConfig.execution_price = "next_open"` đồng bộ giá khớp lệnh Tier 1 với Tier 2.
- [x] `docs/CONFIGURATION.md`, `PHASES.md`, `SCOPE.md`, `PLAN.md` và `graphify-out/` được cập nhật đầy đủ; 1 commit duy nhất cho toàn bộ Phase 3.
