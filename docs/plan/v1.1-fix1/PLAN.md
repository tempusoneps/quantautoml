# Implementation Plan — `v1.1-fix1` (Phase 1: Causality, Leakage & Feature Filter Rigor)

**Active Phase:** Phase 1 — Ingestion Date Slicing, Causality, Leakage & Feature Filter Rigor (`Ingestion + Audit #1, #2, #6`)
**Branch:** `develop/v1.1`
**Target Deliverable:** Cắt dataset theo `[date_from, date_to]` ngay đầu bước Ingestion, triệt tiêu rò rỉ in-sample ở `OOF_ASSISTED`, khôi phục thứ tự nhân quả `Cal-A → Cal-B`, và thay thế cổng `unique_ratio` phụ thuộc cỡ mẫu ở F2 bằng `max_top_freq_ratio`.
**Prerequisites:** [`SCOPE.md`](SCOPE.md), [`PHASES.md`](PHASES.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md).

---

## Execution Order Overview

- [x] **Task 1 (Data Slicing):** Thêm `date_from` / `date_to` vào `DataConfig` và cắt dataset ngay đầu bước `DataIngestion` (trước mọi xử lý)
- [x] **Task 2 (Audit #6):** Cập nhật `FeaturesConfig` và `MultiMetricVarianceFilter` (F2) sang `max_top_freq_ratio` bất biến theo $N$
- [x] **Task 3 (Audit #2):** Đảo thứ tự nhân quả `Cal-A` (SLSQP Ensemble) → `Cal-B` (Temperature Calibration) trong `pipeline.py`
- [ ] **Task 4 (Audit #1):** Sinh true OOF predictions cho Phase 4 trong chế độ `OOF_ASSISTED` (loại bỏ rò rỉ in-sample)
- [ ] **Task 5:** Chạy toàn bộ test suite, cập nhật `graphify` & tài liệu `docs/CONFIGURATION.md` / `docs/FEATURES.md`

---

## Detailed Task Breakdown

### [x] Task 1: Ingestion Date-Range Slicing (`date_from` / `date_to`)
- **Files to Touch:**
  - Modify: `src/quantautoml/common/config.py`
  - Modify: `src/quantautoml/common/types.py`
  - Modify: `src/quantautoml/data/ingestion.py`
  - Modify: `tests/unit/test_config.py`
  - Modify: `tests/unit/test_ingestion.py`
- **Action Steps:**
  1. **(RED)** Viết unit test trong `tests/unit/test_config.py` và `tests/unit/test_ingestion.py`:
     - `test_data_config_date_range_validation`: chấp nhận `None`, `YYYY-MM-DD`, ISO-8601 datetime (`YYYY-MM-DDTHH:MM:SS`, có/không `Z`); từ chối chuỗi sai định dạng và từ chối `date_from > date_to` (`ValidationError`).
     - `test_ingestion_slices_date_from_and_date_to_inclusive`: chỉ giữ các dòng thỏa `date_from <= ts <= date_to` (khi `date_to` là dạng ngày `YYYY-MM-DD` thì inclusive tới hết ngày `23:59:59.999999`).
     - `test_ingestion_supports_one_sided_bounds`: chỉ đặt `date_from` hoặc chỉ đặt `date_to` đều hoạt động.
     - `test_ingestion_date_slice_happens_before_encoding_join_and_fwd_ret`: thêm giá trị string/NaN/giá cực trị nằm ngoài `[date_from, date_to]` → không xuất hiện trong `string_columns_encoded`, không làm fail `min_match_ratio`, và không rò rỉ vào `fwd_ret_h` (cuối cửa sổ vẫn có trailing NaN thay vì nhìn sang sau `date_to`).
     - `test_ingestion_empty_after_date_slice_raises`: khoảng ngày không có dòng nào → ném `InvalidDataError`.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)**
     - Trong `DataConfig` (`src/quantautoml/common/config.py`): thêm `date_from: str | None = None`, `date_to: str | None = None` kèm validator kiểm tra parse ISO-8601 và `start <= end`.
     - Trong `IngestionReport` (`src/quantautoml/common/types.py`): thêm `date_from: str | None = None`, `date_to: str | None = None`.
     - Trong `DataIngestion.load()` (`src/quantautoml/data/ingestion.py`): ngay sau `_read_frame()` cho `features` và `labels`, áp `_slice_date_range(df, ts_col, cfg.data.date_from, cfg.data.date_to)` trước khi đo `n_feature_rows`/`n_label_rows`, trước `_encode_features`, trước `join`, và trước `_derive_forward_returns`.
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_config.py tests/unit/test_ingestion.py -v
  ```
- **Commit Message:** `feat(data): add date_from/date_to config slicing at start of ingestion`

---

### [x] Task 2: F2 `MultiMetricVarianceFilter` — Sample-Size-Invariant Near-Constant Filter (Audit #6)
- **Files to Touch:**
  - Modify: `src/quantautoml/common/config.py`
  - Modify: `src/quantautoml/features/variance.py`
  - Modify: `tests/unit/test_config.py`
  - Modify: `tests/unit/test_f2_variance.py`
- **Action Steps:**
  1. **(RED)** Viết unit test trong `tests/unit/test_f2_variance.py` và `tests/unit/test_config.py`:
     - `test_discrete_periodic_feature_survives_large_n`: cột `session_progress` (50 giá trị lặp lại trên $N = 60{,}000$) **phải được giữ lại** với config mặc định.
     - `test_near_constant_feature_dropped_by_top_freq_ratio`: cột có 99.7% giá trị `0.0` và 0.3% giá trị khác **phải bị drop** với lý do `top_freq_ratio > max_top_freq_ratio`.
     - `test_explicit_min_unique_ratio_still_works`: khi user chủ động set `min_unique_ratio = 0.01`, cổng cũ vẫn hoạt động bình thường.
     - `test_peak_var_ratio_disabled_when_none`: khi `max_peak_var_ratio = None`, không drop theo `peak_var_ratio`.
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Cập nhật `FeaturesConfig` trong `src/quantautoml/common/config.py`:
     - Thêm `max_top_freq_ratio: float = Field(default=0.995, gt=0.0, le=1.0)`.
     - Đổi default `min_unique_ratio: float = Field(default=0.0, ge=0.0, le=1.0)`.
     - Đổi `max_peak_var_ratio: float | None = Field(default=None, gt=0.0)`.
  4. Cập nhật `MultiMetricVarianceFilter._fit` trong `src/quantautoml/features/variance.py`:
     - Tính `counts = np.unique(valid, return_counts=True)[1]`, `top_freq_ratio = float(counts.max()) / valid.size`.
     - Kiểm tra theo thứ tự: `n_unique < 2` → `top_freq_ratio > cfg.max_top_freq_ratio` → (`cfg.min_unique_ratio > 0.0 and n_unique > 2 and unique_ratio < cfg.min_unique_ratio`) → `|cv| < cfg.min_cv` → (`cfg.max_peak_var_ratio is not None and peak[j] > cfg.max_peak_var_ratio`).
  5. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/unit/test_config.py tests/unit/test_f2_variance.py -v
  ```
- **Commit Message:** `fix(features): replace sample-size-dependent unique_ratio default with top_freq_ratio in F2`

---

### [x] Task 3: Khôi phục thứ tự nhân quả `Cal-A` (Ensemble) → `Cal-B` (Calibration) (Audit #2)
- **Files to Touch:**
  - Modify: `src/quantautoml/pipeline.py`
  - Modify: `src/quantautoml/ensemble/calibration.py` (docstring)
  - Modify: `src/quantautoml/ensemble/slsqp.py` (docstring)
  - Modify: `tests/integration/test_zero_leakage_ensemble.py`
- **Action Steps:**
  1. **(RED)** Bổ sung integration test trong `tests/integration/test_zero_leakage_ensemble.py`:
     - `test_static_5way_ensemble_uses_cal_a_and_calibration_uses_cal_b`: phá toàn bộ dữ liệu trên `Cal-B` (70–80%) → `ensemble_result.weights` (học trên `Cal-A`) phải **bit-identical**; ngược lại `calibration_result.temperature` (học trên `Cal-B`) phải **thay đổi**.
  2. Chạy pytest xác nhận FAIL (vì code hiện tại đang fit Ensemble trên `Cal-B` và Calibration trên `Cal-A`).
  3. **(GREEN)** Sửa `src/quantautoml/pipeline.py` trong Phase 04:
     - Fit `BootstrapShrinkageEngine` trên `cal_a_seg`.
     - Áp trọng số `ensemble_result.weights` lên `cal_b_seg` và fit `TemperatureCalibrator` trên `cal_b_seg`.
     - Cập nhật docstring liên quan ở `slsqp.py` và `calibration.py`.
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/integration/test_zero_leakage_ensemble.py tests/unit/test_pipeline_packaging.py tests/integration/test_zero_leakage_packaging.py -v
  ```
- **Commit Message:** `fix(ensemble): enforce temporal causality by fitting ensemble on Cal-A and calibration on Cal-B`

---

### [ ] Task 4: True Out-of-Fold Predictions cho Phase 4 ở chế độ `OOF_ASSISTED` (Audit #1)
- **Files to Touch:**
  - Modify: `src/quantautoml/pipeline.py`
  - Modify: `tests/integration/test_zero_leakage_ensemble.py`
- **Action Steps:**
  1. **(RED)** Viết integration test trong `tests/integration/test_zero_leakage_ensemble.py`:
     - `test_oof_assisted_phase4_never_uses_refitted_model_on_train_subset`: chạy `run_pipeline` với $N = 3{,}000$ (`OOF_ASSISTED`, có `[models]` + `[ensemble]`), spy/assert rằng:
       - `ensemble_result` và `calibration_result` được sinh ra từ các model fit riêng trên `oof_folds[-2][0]` và `oof_folds[-1][0]`, **không gọi** `refitted_model.predict` / `predict_proba` trên bất kỳ dòng nào của `split.train`.
       - Phá dữ liệu của `Val` và `OOS` → kết quả Phase 4 vẫn **bit-identical** (RULE-01).
  2. Chạy pytest xác nhận FAIL.
  3. **(GREEN)** Trong `src/quantautoml/pipeline.py`:
     - Viết helper `_oof_fold_candidates(tuning_result, train_seg, tr_idx, task_type, random_state)` tạo clone từng `TunedCandidate` bằng `create_candidate(name=cand.model_name, task_type=task_type, params=cand.best_params, random_state=random_state)` và `fit(train_seg.X[tr_idx], train_seg.y[tr_idx])`.
     - Trong nhánh `OOF_ASSISTED`:
       - Dùng `oof_folds[-2]` (`tr_a, va_a`) để tạo `oof_cands_a`, predict lên `va_a` → fit `BootstrapShrinkageEngine`.
       - Dùng `oof_folds[-1]` (`tr_b, va_b`) để tạo `oof_cands_b`, predict lên `va_b`, trộn bằng `ensemble_result.weights` → fit `TemperatureCalibrator`.
       - `tuning_result.tuned_candidates` (refitted trên full Train) chỉ dùng cho Phase 05 Packaging (`Val` check & `OOS` inference).
  4. Chạy pytest xác nhận PASS.
- **Verification Command:**
  ```bash
  uv run pytest tests/integration/test_zero_leakage_ensemble.py tests/unit/test_pipeline.py -v
  ```
- **Commit Message:** `fix(pipeline): use true out-of-fold predictions for Phase 4 in OOF_ASSISTED mode`

---

### [ ] Task 5: Full Verification, Docs & Graphify Update
- **Files to Touch:**
  - Modify: `docs/CONFIGURATION.md` (cập nhật tham số `max_top_freq_ratio`, default `min_unique_ratio`, `max_peak_var_ratio`)
  - Modify: `docs/FEATURES.md` (cập nhật đặc tả F2)
  - Modify: `docs/plan/v1.1-fix1/PHASES.md` & `docs/plan/v1.1-fix1/PLAN.md` (đánh dấu Phase 1 Done khi hoàn tất)
- **Action Steps:**
  1. Chạy toàn bộ test suite `uv run pytest tests/` xác nhận 100% PASS (0 failures).
  2. Cập nhật `docs/CONFIGURATION.md` và `docs/FEATURES.md` cho khớp với hành vi mới.
  3. Chạy `graphify update .` để đồng bộ knowledge graph.
  4. Chuyển trạng thái Phase 1 → `Done` và Phase 2 → `In Progress` trong `docs/plan/v1.1-fix1/PHASES.md`.
- **Verification Command:**
  ```bash
  uv run pytest tests/ && graphify update .
  ```
- **Commit Message:** `docs(plan): update config/feature docs and mark v1.1-fix1 Phase 1 done`

---

## Final Verification Checklist (Phase 1)
- [ ] `uv run pytest tests/` PASS 100% (không hồi quy).
- [ ] `date_from` / `date_to` cắt dữ liệu ngay đầu bước `DataIngestion` (trước encode/join/`fwd_ret_h`, zero look-ahead vượt `date_to`).
- [ ] `OOF_ASSISTED` không còn rò rỉ in-sample vào `BootstrapShrinkageEngine` và `TemperatureCalibrator`.
- [ ] `STATIC_5WAY` thuần nhân quả: `Cal-A` (Ensemble) → `Cal-B` (Calibration).
- [ ] F2 giữ lại đúng các feature rời rạc/chu kỳ (`session_progress`, `hour`, `body`, `cluster_kmeans_*`) trên $N = 60{,}000$ và loại đúng cột near-constant ($\ge 99.5\%$).
- [ ] Sai số ONNX float32 vẫn $< 10^{-4}$ (`RULE-06`) và `predictor.py` hoàn toàn độc lập (`RULE-07`).
