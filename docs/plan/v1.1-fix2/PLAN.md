# Implementation Plan — `v1.1-fix2` (Phase 2: Regime Freshness & Multi-Window Walk-Forward OOS)

**Active Phase:** Phase 2 — Regime Freshness & Multi-Window Walk-Forward OOS  
**Branch:** `develop/v1.1`  
**Commit Policy:** 1 commit duy nhất cho toàn bộ Phase 2 sau khi toàn bộ test suite PASS 100%.  
**Prerequisites:** Phase 1 Done, `v1.1-optimize3` Phases 1–3 Done.

---

## Global Constraints & Hard Gates

- **RULE-01:** `refit_on_train_cal=True` chỉ được phép fit lại bản sao model đã khóa `best_params` trên `Train + Cal-A + Cal-B` (sau khi đã khóa `FeatureFunnel`, `weights` và `temperature`). Tuyệt đối không chạm `Val` hoặc `OOS`. Mọi cửa sổ trong `WalkForwardOOSRunner` bắt buộc có khoảng purging/embargo chính xác bằng `horizon_h` nến (`DataLeakageError` nếu vi phạm).
- **RULE-06:** Bundle xuất ra sau `refit_on_train_cal=True` vẫn phải vượt qua kiểm định sai số số học `max|y_py - y_onnx| < 1e-4` trên `Val`.
- **Tương thích ngược 100%:** `models.refit_on_train_cal` mặc định `False`; không thay đổi hành vi mặc định của pipeline.
- **TDD bắt buộc:** Viết failing tests (RED) trước khi sửa mã nguồn (GREEN).

---

## Execution Order Overview

- [x] **Task 1 (RED → GREEN):** Thêm `models.refit_on_train_cal: bool = False` vào `ModelsConfig` và tích hợp refit an toàn (`Train + Cal-A + Cal-B`, giữ nguyên `Val`/`OOS` bất biến) trong `src/quantautoml/pipeline.py` + tests (`test_models_config.py`, `test_zero_leakage_ensemble.py`)
- [x] **Task 2 (RED → GREEN):** Xây dựng `src/quantautoml/backtest/walkforward.py` (`WalkForwardWindowSpec`, `generate_walkforward_windows`, `WalkForwardWindowResult`, `WalkForwardOOSReport`, `WalkForwardOOSRunner`) và export qua `backtest/__init__.py` + tests (`test_walkforward_oos.py`)
- [x] **Task 3:** Đồng bộ `scripts/generate_config.py`, `docs/CONFIGURATION.md`, `docs/STRUCTURE.md`, chạy toàn bộ `uv run pytest tests/`, `graphify update .`, đánh dấu Phase 2 Done & commit

---

## Final Verification Checklist (Phase 2)

- [x] `uv run pytest tests/` PASS 100%.
- [x] `ModelsConfig.refit_on_train_cal` mặc định `False`; khi `True` trên `STATIC_5WAY`, model đóng gói được refit trên `Train + Cal-A + Cal-B`, bất biến hoàn toàn trước mọi đột biến trên `Val`/`OOS`, và đạt ONNX parity `< 1e-4`.
- [x] `generate_walkforward_windows` bảo đảm khoảng cách `oos_start - train_end == horizon_h` trên mọi cửa sổ và `WalkForwardOOSRunner` tổng hợp đầy đủ `WalkForwardOOSReport` + ghi JSON.
- [x] `scripts/generate_config.py`, `docs/CONFIGURATION.md`, `docs/STRUCTURE.md`, `docs/plan/v1.1-fix2/` và `graphify-out/` đã đồng bộ; 1 commit sạch cho Phase 2.
