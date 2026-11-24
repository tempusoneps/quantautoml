# Implementation Plan — `v1.1-fix2` (Phase 1: GBDT Regularization & GPU Acceleration)

**Active Phase:** Phase 1 — GBDT Regularization, Search-Space Hardening & GPU Acceleration
**Branch:** `develop/v1.1`
**Commit Policy:** 1 commit duy nhất cho toàn bộ Phase 1 sau khi toàn bộ test suite PASS 100%.
**Prerequisites:** [`SCOPE.md`](SCOPE.md), [`PHASES.md`](PHASES.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Execution Order Overview

- [x] **Task 1:** Thêm `models.device: Literal["auto", "cpu", "cuda"] = "auto"` vào `ModelsConfig` và hỗ trợ GPU cho `XGBoostCandidate` & `CatBoostCandidate` trong `models/candidates.py`
- [x] **Task 2:** Thêm `adaptive_leaf_bounds` / `adaptive_child_weight_bounds` và cập nhật biên "điểm ngọt" (`depth 3..7`, `num_leaves 7..47`, `lr 0.01..0.08`, `n_estimators 50..300`, `l2_leaf_reg 2..20`) trong `models/tournament.py` & `models/tuner.py` + truyền `device` (kể cả `_fit_oof_candidates` trong `pipeline.py`)
- [x] **Task 3:** Viết unit tests trong `tests/unit/test_candidates.py`, `tests/unit/test_tournament.py`, `tests/unit/test_tuner.py` (kể cả ONNX parity trên GPU nếu máy có CUDA)
- [x] **Task 4:** Đồng bộ `scripts/generate_config.py`, `docs/CONFIGURATION.md`, chạy toàn bộ `uv run pytest tests/`, `graphify update .`, đánh dấu Phase 1 Done & tạo 1 commit duy nhất cho Phase 1

---

## Detailed Task Breakdown

### [ ] Task 1: GPU Support (`models.device`) trong `config.py` & `models/candidates.py`
- **Files to Touch:**
  - Modify: `src/quantautoml/common/config.py` (`ModelsConfig.device: Literal["auto", "cpu", "cuda"] = "auto"`)
  - Modify: `src/quantautoml/models/candidates.py`
- **Action Steps:**
  1. Thêm `resolve_device(device: str) -> Literal["cpu", "cuda"]` (cache kết quả kiểm tra CUDA bằng `shutil.which("nvidia-smi")` / `subprocess` hoặc probe nhẹ 1 lần duy nhất; nếu `device == "cpu"` trả `"cpu"`, nếu `"cuda"` trả `"cuda"`, nếu `"auto"` trả `"cuda"` khi có GPU khả dụng ngược lại `"cpu"`).
  2. Cập nhật `BaseGBDCandidate.__init__(..., device: str = "cpu")` và `create_candidate(..., device: str = "cpu")`.
  3. Trong `XGBoostCandidate.fit()`: khi `resolved == "cuda"`, đặt `tree_method="hist"`, `device="cuda"` (loại `n_jobs` không cần thiết trên GPU).
  4. Trong `CatBoostCandidate.fit()`: khi `resolved == "cuda"`, đặt `task_type="GPU"`, `bootstrap_type="Bernoulli"` (nếu có `subsample`), loại bỏ `thread_count` (không áp dụng cho GPU).
  5. `LightGBMCandidate`: giữ nguyên CPU (`n_jobs=-1`).

---

### [ ] Task 2: Biên "Điểm Ngọt" & Adaptive Leaf Bounds trong `tournament.py`, `tuner.py`, `pipeline.py`
- **Files to Touch:**
  - Modify: `src/quantautoml/models/tournament.py`
  - Modify: `src/quantautoml/models/tuner.py`
  - Modify: `src/quantautoml/pipeline.py`
- **Action Steps:**
  1. Thêm `adaptive_leaf_bounds(n_samples)` (`lo = max(10, n_samples // 1000)`, `hi = max(30, min(300, n_samples // 200))`) và `adaptive_child_weight_bounds(n_samples)` (`lo = max(2, n_samples // 5000)`, `hi = max(15, min(60, n_samples // 1000))`).
  2. Cập nhật `_sample_params` (`tournament.py`) và `_suggest_tuning_params` (`tuner.py`):
     - `lightgbm`: `n_estimators` (`[50, 100, 150, 250]` / `50..300`), `max_depth` `3..7`, `num_leaves` `7..47`, `learning_rate` `0.01..0.08`, `min_child_samples` theo `adaptive_leaf_bounds(n_samples)`.
     - `xgboost`: `n_estimators` (`[50, 100, 150, 250]` / `50..300`), `max_depth` `3..7`, `learning_rate` `0.01..0.08`, `min_child_weight` theo `adaptive_child_weight_bounds(n_samples)`.
     - `catboost`: `n_estimators` (`[50, 100, 150, 250]` / `50..300`), `depth` `3..7`, `learning_rate` `0.01..0.08`, `l2_leaf_reg` `2.0..20.0`.
  3. Truyền `device=self.cfg.device` (và `cfg.models.device` trong `_fit_oof_candidates` của `pipeline.py`) vào mọi lời gọi `create_candidate(...)`.

---

### [ ] Task 3: Unit Tests (TDD)
- **Files to Touch:**
  - Modify: `tests/unit/test_candidates.py`
  - Modify: `tests/unit/test_tournament.py`
  - Modify: `tests/unit/test_tuner.py`
- **Action Steps:**
  1. `test_resolve_device_cpu_and_auto` + `test_xgb_and_catboost_fit_with_auto_device` (xác nhận train + predict float32 thành công trên cả `cpu` và `auto`/GPU nếu có CUDA).
  2. `test_adaptive_leaf_bounds_scales_with_sample_size` (`n=250` $\to$ `(10, 30)`; `n=50_000` $\to$ `(50, 250)`).
  3. `test_tournament_search_space_respects_regularization_caps` & `test_tuner_search_space_respects_regularization_caps` (`max_depth <= 7`, `num_leaves <= 47`, `depth <= 7`, `learning_rate <= 0.08`, `l2_leaf_reg >= 2.0`).
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_candidates.py tests/unit/test_tournament.py tests/unit/test_tuner.py tests/integration/test_zero_leakage_models.py -v
  ```

---

### [ ] Task 4: Sync Config Generator, Docs, Full Suite, Graphify & Single Phase 1 Commit
- **Files to Touch:**
  - Modify: `scripts/generate_config.py`, `docs/CONFIGURATION.md`, `docs/plan/v1.1-fix2/{PLAN,PHASES,SCOPE}.md`
- **Verification Command:**
  ```bash
  uv run pytest tests/ && graphify update .
  ```
- **Commit Message:** `perf(models): add GPU acceleration (XGBoost/CatBoost) and balanced GBDT regularization bounds (v1.1-fix2 Phase 1)`

---

## Final Verification Checklist (Phase 1)
- [x] `uv run pytest tests/` PASS 100% (không hồi quy).
- [x] `models.device` hỗ trợ `"auto" | "cpu" | "cuda"` (XGBoost + CatBoost chạy GPU trên máy có CUDA; LightGBM chạy CPU; fallback an toàn về CPU).
- [x] Không còn `max_depth = -1` hay `depth > 7` hay `num_leaves > 47` trong Tournament & Tuner.
- [x] `min_child_samples` và `min_child_weight` co giãn động theo cỡ mẫu Train.
- [x] ONNX float32 parity `< 1e-4` vẫn PASS cho cả 3 model.
- [x] `docs/CONFIGURATION.md`, `scripts/generate_config.py`, `docs/plan/v1.1-fix2/` và `graphify-out/` được cập nhật; 1 commit duy nhất cho Phase 1.
