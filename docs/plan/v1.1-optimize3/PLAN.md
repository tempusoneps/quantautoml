# Implementation Plan — `v1.1-optimize3` (Phase 1: Horizon-Aware Rolling IC Window & Ensemble Diagnostics)

**Active Phase:** Phase 1 — Horizon-Aware Rolling IC Window & Ensemble Diagnostics  
**Branch:** `develop/v1.1`  
**Commit Policy:** 1 commit duy nhất cho toàn bộ Phase 1 sau khi toàn bộ test suite PASS 100%.  
**Prerequisites:** [`SCOPE.md`](SCOPE.md), [`PHASES.md`](PHASES.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Global Constraints & Hard Gates

- **RULE-01:** Tuyệt đối không dùng dữ liệu ngoài `Train` (Tournament/Tuner) hoặc `Cal-A` / OOF fold (Bootstrap Ensemble) khi ước lượng cửa sổ hoặc IC Sharpe.
- **RULE-02:** Giữ nguyên Spearman Rank IC & IC Sharpe làm thước đo thống nhất.
- **RULE-04 & RULE-05:** Không thay đổi hàm mục tiêu $C^2$ của SLSQP trên `Cal-A` và ngưỡng hurdle $+5\%$ IC Sharpe.
- **Tương thích ngược:** `ic_window = None` mặc định hoạt động ổn định trên cả dataset unit test nhỏ (`N = 50..300`), chế độ `OOF_ASSISTED`, và dữ liệu intraday thực tế (`N >= 5,000`).
- **TDD bắt buộc:** Trong từng Task 1–3, viết unit test xác nhận FAIL (**RED**) trước khi sửa mã nguồn (**GREEN**), sau đó chạy lại toàn bộ suite để bảo đảm không hồi quy.

---

## Execution Order Overview

- [x] **Task 1 (RED → GREEN):** Thêm `ic_window: int | None = None` (`ge=20`) vào `ModelsConfig` & `EnsembleConfig` (`src/quantautoml/common/config.py`) + unit tests (`test_models_config.py`, `test_ensemble_config.py`)
- [x] **Task 2 (RED → GREEN):** Nâng cấp `fold_subwindow_ics` trong `models/tournament.py`, nối `cfg.ic_window` từ `ModelTournament` & `DeepOptunaTuner` + unit tests (`test_tournament.py`, `test_tuner.py`)
- [x] **Task 3 (RED → GREEN):** Thêm `_resolve_ic_window`, nâng cấp `_calc_series_ic_sharpe` (sửa fallback `score == 0.0`) và bổ sung trường chẩn đoán (`ic_window_used`, `single_model_scores`, `overall_rank_ic`) cho cả `K == 1` và `K > 1` trong `ensemble/bootstrap.py` + unit tests (`test_bootstrap_shrinkage.py`)
- [x] **Task 4:** Đồng bộ `scripts/generate_config.py`, `docs/CONFIGURATION.md`, chạy toàn bộ `uv run pytest tests/`, `graphify update .`, cập nhật trạng thái Phase 1 & commit

---

## Detailed Task Breakdown

### [ ] Task 1: Thêm `ic_window` vào `ModelsConfig` & `EnsembleConfig` (TDD)
- **Files to Touch:**
  - Modify: `tests/unit/test_models_config.py`, `tests/unit/test_ensemble_config.py`
  - Modify: `src/quantautoml/common/config.py`
- **Action Steps:**
  1. **(RED)** Viết unit tests trong `test_models_config.py` và `test_ensemble_config.py`:
     - Mặc định `ModelsConfig().ic_window is None` và `EnsembleConfig().ic_window is None`.
     - Chấp nhận `ic_window = 500` (cả qua constructor lẫn TOML).
     - Từ chối `ic_window < 20` (vd `19`, `0`, `-5`) bằng `pydantic.ValidationError`.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Thêm `ic_window: int | None = Field(default=None, ge=20)` vào `ModelsConfig` và `EnsembleConfig` trong `src/quantautoml/common/config.py`.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_models_config.py tests/unit/test_ensemble_config.py tests/unit/test_config.py -v
  ```

---

### [ ] Task 2: Nâng cấp `fold_subwindow_ics` trong `tournament.py` & nối qua `tuner.py` (TDD)
- **Files to Touch:**
  - Modify: `tests/unit/test_tournament.py`, `tests/unit/test_tuner.py`
  - Modify: `src/quantautoml/models/tournament.py`
  - Modify: `src/quantautoml/models/tuner.py`
- **Action Steps:**
  1. **(RED)** Viết unit tests trong `test_tournament.py` và `test_tuner.py`:
     - `test_fold_subwindow_ics_scales_up_on_long_series`: trên `n=4500`, `len(fold_subwindow_ics(pred, y)) == 9` (`4500 // 500`, không còn `4500 // 63 == 71`).
     - `test_fold_subwindow_ics_respects_explicit_ic_window`: truyền `ic_window=250` trên `n=2000` trả về `8` cửa sổ.
     - `test_fold_subwindow_ics_clamps_large_ic_window_on_small_fold`: truyền `ic_window=500` trên `n=200` tự clamp về `sub_w = 50` (`4` cửa sổ, không suy biến về `2` cửa sổ).
     - `test_fold_subwindow_ics_rejects_tiny_explicit_window`: truyền trực tiếp `ic_window=2` (`< 3`) ném `InvalidDataError`.
     - `test_tuner_forwards_explicit_ic_window` (trong `test_tuner.py`): spy `fold_subwindow_ics` khi chạy `DeepOptunaTuner(ModelsConfig(ic_window=120, ...)).tune(...)` để xác nhận `ic_window=120` được truyền xuống.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Cập nhật `src/quantautoml/models/tournament.py` và `src/quantautoml/models/tuner.py`:
     - Đặt `_MAX_SUBWINDOW = 500` (thay cho `63`).
     - Mở rộng chữ ký `fold_subwindow_ics(pred, y_fold, *, ic_window: int | None = None) -> list[float]`:
       - Nếu `ic_window is not None and ic_window < 3`: ném `InvalidDataError`.
       - Khi `n >= _MIN_FOLD_FOR_SUBWINDOWS`:
         - Nếu `ic_window is not None`: `sub_w = min(ic_window, max(_MIN_SUBWINDOW, n // 4))`.
         - Nếu `ic_window is None`: `sub_w = max(_MIN_SUBWINDOW, min(_MAX_SUBWINDOW, n // 4))`.
         - Nếu `n // sub_w < 2`: fallback về `[float(calc_rank_ic(pred, y_fold)[0])]`.
     - Trong `ModelTournament.run()` (`tournament.py`) và `DeepOptunaTuner.tune()` (`tuner.py`), truyền `ic_window=self.cfg.ic_window` vào `fold_subwindow_ics(...)`.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_tournament.py tests/unit/test_tuner.py tests/integration/test_zero_leakage_models.py -v
  ```

---

### [ ] Task 3: Nâng cấp `_calc_series_ic_sharpe` & Chẩn đoán trong `ensemble/bootstrap.py` (TDD)
- **Files to Touch:**
  - Modify: `tests/unit/test_bootstrap_shrinkage.py`
  - Modify: `src/quantautoml/ensemble/bootstrap.py`
- **Action Steps:**
  1. **(RED)** Viết unit tests trong `tests/unit/test_bootstrap_shrinkage.py`:
     - `test_bootstrap_adaptive_window_scales_on_long_intraday_series`: trên `n=7500`, `res.ic_window_used == 500`, `len(res.single_model_scores) == K`, `res.overall_rank_ic` hữu hạn và đúng dấu.
     - `test_bootstrap_explicit_ic_window_and_single_model_branch`: kiểm thử `EnsembleConfig(ic_window=300)` cho cả `K=1` và `K=3`; xác nhận nhánh `K=1` điền đầy đủ `ic_window_used`, `single_model_scores` (`len == 1`), và `overall_rank_ic`.
     - `test_resolve_ic_window_clamps_to_at_least_four_windows_when_possible`: trên `n=380` với `ic_window=500`, `_resolve_ic_window(380, 500) == 95` (`4` cửa sổ thay vì `2`), và `_resolve_ic_window(380, 2)` ném `ValueError`.
     - `test_calc_series_ic_sharpe_does_not_mix_rank_ic_when_mean_is_zero`: khi chuỗi `rolling_ic` có $\ge 2$ cửa sổ hợp lệ, `std > 0` nhưng `mean == 0.0`, kết quả trả về `0.0` (không nhảy sang `overall_rank_ic`).
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Cập nhật `src/quantautoml/ensemble/bootstrap.py`:
     - Thêm `_resolve_ic_window(n: int, ic_window: int | None) -> int`:
       - Nếu `ic_window is not None`: kiểm tra `ic_window >= 3` (ngược lại ném `ValueError`), đặt `w = min(ic_window, max(10, n // 4))`.
       - Nếu `ic_window is None`: `w = max(10, min(500, n // 10))`.
       - Nếu `n // w < 2`: `w = max(3, n // 2)`.
       - Trả về `int(w)`.
     - Cập nhật `_calc_series_ic_sharpe(pred, target, *, ic_window: int | None = None) -> float`:
       - Dùng `window = _resolve_ic_window(len(pred), ic_window)`.
       - Tính `ic_series = calc_rolling_ic(pred, target, window=window)`, lọc `valid = ic_series[~np.isnan(ic_series)]`.
       - Nếu `valid.size < 2`, `np.ptp(valid) == 0.0`, hoặc `float(np.std(valid, ddof=1)) <= 1e-12`: fallback `float(calc_rank_ic(pred, target)[0])`; ngược lại trả `float(calc_ic_sharpe(ic_series))`.
     - Mở rộng `EnsembleWeightsResult` với các trường mặc định tương thích ngược:
       - `ic_window_used: int = 0`
       - `single_model_scores: list[float] = Field(default_factory=list)`
       - `overall_rank_ic: float = 0.0` (Rank IC toàn đoạn `Cal-A` / OOF của `preds @ final_weights`).
     - Truyền `ic_window=self.cfg.ic_window` vào mọi lời gọi `_calc_series_ic_sharpe(...)` trong `BootstrapShrinkageEngine.fit()`, đồng nhất kiểm tra `single_model_scores is not None and len(single_model_scores) == 1` ở nhánh `n_models == 1`, và điền đủ 3 trường chẩn đoán ở **cả nhánh `n_models == 1` lẫn `n_models > 1`**.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_bootstrap_shrinkage.py tests/integration/test_zero_leakage_ensemble.py -v
  ```

---

### [ ] Task 4: Đồng bộ Config Generator, Docs, Full Test Suite, Graphify & Commit
- **Files to Touch:**
  - Modify: `scripts/generate_config.py` (thêm chú thích tùy chọn `# ic_window = 500` trong `[models]` & `[ensemble]`)
  - Modify: `docs/CONFIGURATION.md` (ghi rõ cơ chế `ic_window` tự động `<= 500` và cách ghi đè)
  - Modify: `docs/plan/v1.1-optimize3/{PLAN,PHASES,SCOPE}.md` (đánh dấu `[x]` và chuyển trạng thái Phase 1 → Done)
- **Verification Command:**
  ```bash
  uv run pytest tests/ && graphify update .
  ```
- **Commit Message:** `fix(metrics): use horizon-aware adaptive rolling IC window in tournament, tuner, and bootstrap ensemble (v1.1-optimize3 Phase 1)`

---

## Final Verification Checklist (Phase 1)

- [x] `uv run pytest tests/` PASS 100% (không hồi quy).
- [x] `ModelsConfig.ic_window` và `EnsembleConfig.ic_window` hỗ trợ `None` (tự động tối đa `500`) và giá trị nguyên `>= 20`.
- [x] `fold_subwindow_ics` và `_calc_series_ic_sharpe` không còn bị kẹt ở ngưỡng `50` / `63` nến trên chuỗi dài (`N >= 5,000`), và clamp an toàn theo `n // 4` trên mẫu nhỏ.
- [x] `01_ensemble_report.json` ghi rõ `ic_window_used`, `single_model_scores`, `overall_rank_ic` cho cả nhánh `K == 1` và `K > 1`.
- [x] `scripts/generate_config.py`, `docs/CONFIGURATION.md`, `docs/plan/v1.1-optimize3/` và `graphify-out/` đã đồng bộ; 1 commit sạch cho Phase 1.
