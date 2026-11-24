# Implementation Plan — `v1.1-fix1` (Phase 2: Statistical Rigor & Objective Alignment)

**Active Phase:** Phase 2 — Statistical Rigor & Objective Alignment (`Audit #3, #4, #5`)
**Branch:** `develop/v1.1`
**Target Deliverable:** Kiểm định BH-FDR không bị méo bởi autocorrelation ở F3+7, cổng monotonicity suy giảm từ `target_horizon` trở đi, `IC Sharpe` ổn định dựa trên rolling sub-windows trong Walk-Forward CV + đồng bộ thang đo `HyperbandPruner`, và thống nhất Multiclass score (`probs @ classes`) xuyên suốt pipeline.
**Prerequisites:** Phase 1 Done ([`PHASES.md`](PHASES.md)), [`SCOPE.md`](SCOPE.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Execution Order Overview

- [x] **Task 1 (Audit #3):** F3+7 `ICStabilityFilter` — 1-sample t-test trên `rolling_ic` không chồng lấn cho BH-FDR & monotonicity gate kiểm tra suy giảm từ `target_horizon` trở đi
- [ ] **Task 2 (Audit #4):** `ModelTournament` & `DeepOptunaTuner` — tính `IC Sharpe` trên sub-window rolling IC nội bộ trong mỗi fold + đồng bộ thang đo `trial.report()` với `objective()` của `HyperbandPruner`
- [ ] **Task 3 (Audit #5):** Thống nhất Multiclass expected-class score (`cand.fitted_model.predict(X)` dùng đúng `classes_`) trong `pipeline.py` và kiểm chứng khớp tuyệt đối với `predictor.py`
- [ ] **Task 4:** Chạy toàn bộ test suite (`uv run pytest tests/`), cập nhật `docs/FEATURES.md`, `graphify update .`, và chuyển Phase 2 → Done

---

## Detailed Task Breakdown

### [x] Task 1: F3+7 Autocorrelation-Safe BH-FDR & Post-Target Monotonicity (Audit #3)
- **Files to Touch:**
  - Modify: `src/quantautoml/features/ic_stability.py`
  - Modify: `tests/unit/test_ic_stability.py`
- **Action Steps:**
  1. **(RED)** Cập nhật/bổ sung unit test trong `tests/unit/test_ic_stability.py`:
     - `test_hump_at_target_horizon_kept_when_decaying_after_target`: với `target_horizon = 5`, feature có `|ICIR(1)| < |ICIR(5)| > |ICIR(10)| > |ICIR(21)|` (nhiễu microstructure ở $h=1$, đỉnh ở $h=5$ và suy giảm sau đó) **phải được giữ lại**.
     - `test_rising_after_target_horizon_dropped`: với `target_horizon = 5`, feature có `|ICIR(5)| < |ICIR(10)|` **vẫn bị drop** vì `monotonic`.
     - `test_fdr_uses_rolling_ic_ttest_controlling_autocorrelation`: trên chuỗi pure noise với nhãn overlapping `fwd_ret_10` (có serial correlation mạnh), p-value dùng 1-sample t-test trên `rolling_ic` không chồng lấn giữ tỷ lệ false positive BH-FDR $\le 5\%$.
     - Cập nhật `test_aux_nan_tail_masked_pairwise` để khớp công thức p-value từ `rolling_ic` t-test.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Sửa `src/quantautoml/features/ic_stability.py`:
     - Tính p-value tại `target_horizon` bằng `scipy.stats.ttest_1samp(valid, 0.0).pvalue` trên chuỗi `roll_target[:, j]` không chồng lấn (trả `1.0` nếu `< 2` cửa sổ hợp lệ hoặc `std == 0` / NaN).
     - Đổi `mono_ok = np.all(abs_icir[t:-1] >= abs_icir[t + 1 :] - cfg.monotonic_tolerance, axis=0)`.
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_ic_stability.py tests/unit/test_funnel.py -v
  ```
- **Commit Message:** `fix(features): use rolling-IC t-test for BH-FDR and check monotonicity from target_horizon onward`

---

### [ ] Task 2: Rolling Sub-Window `IC Sharpe` in CV & Aligned `HyperbandPruner` (Audit #4)
- **Files to Touch:**
  - Modify: `src/quantautoml/models/tournament.py`
  - Modify: `src/quantautoml/models/tuner.py`
  - Modify: `tests/unit/test_tournament.py`
  - Modify: `tests/unit/test_tuner.py`
- **Action Steps:**
  1. **(RED)** Bổ sung unit test trong `tests/unit/test_tournament.py` và `tests/unit/test_tuner.py`:
     - `test_cv_ic_sharpe_uses_subwindow_rolling_ics`: chứng minh trên tập dữ liệu lớn đủ dài mỗi fold, `ic_sharpe` được tính từ sub-window rolling ICs (không phải chỉ `std` trên 2–5 điểm fold).
     - `test_tuner_reports_aligned_objective_to_pruner`: spy `optuna.Trial.report` xác nhận giá trị báo cáo ở bước cuối khớp với `ic_sharpe - complexity_penalty * complexity` (cùng thang đo với giá trị trả về của `objective()`).
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)**
     - Thêm helper dùng chung (trong `models/validation.py` hoặc `models/tournament.py`): `_fold_subwindow_ics(pred, y_fold)` chia mỗi fold test thành các sub-window không chồng lấn (`window = max(20, min(63, len(y_fold) // 4))` nếu `len(y_fold) >= 40`, ngược lại fallback về `[fold_ic]`).
     - Dùng `calc_ic_sharpe(np.array(all_sub_ics))` trong cả `ModelTournament` và `DeepOptunaTuner`.
     - Trong `DeepOptunaTuner.objective`, tính `complexity` trước vòng lặp fold và gọi `trial.report(calc_ic_sharpe(sub_ics_so_far) - penalty * complexity, step=step)`.
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_tournament.py tests/unit/test_tuner.py tests/integration/test_zero_leakage_models.py -v
  ```
- **Commit Message:** `fix(models): compute CV IC Sharpe over fold sub-windows and align HyperbandPruner objective`

---

### [ ] Task 3: Unified Multiclass Score (`probs @ classes_`) in Pipeline (Audit #5)
- **Files to Touch:**
  - Modify: `src/quantautoml/pipeline.py`
  - Modify: `tests/unit/test_pipeline_packaging.py`
- **Action Steps:**
  1. **(RED)** Viết unit test kiểm chứng `_get_candidate_scores` với bài toán Multiclass có nhãn không đều (vd `classes_ = [-2, 0, 5]`): kết quả phải khớp `probs @ [-2, 0, 5]` (trùng khớp `BaseGBDCandidate.predict` và `Predictor.predict`), không phải `probs @ [0, 1, 2]`.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Sửa `_get_candidate_scores` trong `src/quantautoml/pipeline.py` gọi trực tiếp `cand.fitted_model.predict(X)` (vốn đã xử lý thống nhất `probs[:, 1]` cho binary và `probs @ self.classes_` cho multiclass).
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_pipeline_packaging.py tests/unit/test_pipeline.py -v
  ```
- **Commit Message:** `fix(pipeline): unify multiclass candidate scoring via fitted_model.predict using actual classes`

---

### [ ] Task 4: Full Verification, Docs & Phase 2 Handoff
- **Files to Touch:**
  - Modify: `docs/FEATURES.md` (cập nhật đặc tả F3+7)
  - Modify: `docs/plan/v1.1-fix1/PHASES.md`, `docs/plan/v1.1-fix1/SCOPE.md`, `docs/plan/v1.1-fix1/PLAN.md`
- **Action Steps:**
  1. Chạy toàn bộ test suite `uv run pytest tests/` xác nhận 100% PASS.
  2. Cập nhật `docs/FEATURES.md` và chạy `graphify update .`.
  3. Đánh dấu Phase 2 → `Done`, Phase 3 → `In Progress` trong `PHASES.md` và `SCOPE.md`.
- **Verification Command:**
  ```bash
  uv run pytest tests/ && graphify update .
  ```
- **Commit Message:** `docs(plan): update F3+7 docs and mark v1.1-fix1 Phase 2 done`

---

## Final Verification Checklist (Phase 2)
- [ ] `uv run pytest tests/` PASS 100%.
- [ ] F3+7 dùng 1-sample t-test trên `rolling_ic` cho BH-FDR và kiểm tra monotonicity từ `target_horizon` trở đi.
- [ ] `ModelTournament` và `DeepOptunaTuner` tính `IC Sharpe` trên sub-window rolling ICs; `HyperbandPruner` nhận đúng thang đo của `objective()`.
- [ ] Multiclass score thống nhất `probs @ classes_` xuyên suốt từ Tournament → Tuner → Cal-A → Cal-B → ONNX Predictor.
