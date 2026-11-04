# Implementation Plan: Quant AutoML Pipeline

**Active Phase:** Phase 1 — Data Ingestion, Task Inference & Adaptive Leakage-Free Temporal Splitter  
**Target Deliverable:** Sub-system nạp `features` + `labels` (nhiều cột nhãn, chọn qua config TOML), đồng bộ UTC, suy luận loại bài toán, chia thích ứng (5-way khi $N \ge 10{,}000$, ngược lại OOF-assisted) kèm Purging `horizon_h` dòng và sinh report.  
**Prerequisites:** [`SCOPE.md`](SCOPE.md), [`PHASES.md`](PHASES.md), [`STRUCTURE.md`](STRUCTURE.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`TRADE-OFF.md`](TRADE-OFF.md), [`01-ingestion-split/DESIGN.md`](01-ingestion-split/DESIGN.md) (đặc tả input/xử lý/output/report), [`../CONFIGURATION.md`](../CONFIGURATION.md).

---

## Execution Order Overview

- [x] **Task 1:** Setup dự án, exceptions, types, report dataclasses, `PipelineConfig` + loader TOML.
- [x] **Task 2:** `DataIngestion` (UTC, join, float32, xử lý nhãn, `IngestionReport`).
- [x] **Task 3:** `infer_task_type`.
- [x] **Task 4:** `AdaptiveTimeSeriesSplitter` (5-way / OOF, Purging, `oof_folds`, `SplitReport`, `DataLeakageError`).
- [x] **Task 5:** Integration test Zero Look-Ahead Leakage.

---

## Detailed Task Breakdown

### [x] Task 1: Setup, Common Types, Exceptions & Config
- **Mục tiêu:** Môi trường chạy được `pytest`, import được package, có đủ data contract và config cho Phase 1.
- **Files to Touch:**
  - Modify: `pyproject.toml`
  - Create: `src/quantautoml/__init__.py`, `src/quantautoml/common/__init__.py`
  - Create: `src/quantautoml/common/exceptions.py`, `types.py`, `config.py`
  - Create: `tests/conftest.py`, `tests/unit/test_types.py`, `tests/unit/test_config.py`
- **Action Steps:**
  1. `pyproject.toml`: thêm `[build-system]` (hatchling), src-layout, dependencies `polars`, `numpy`, `pyarrow`, `pydantic>=2`; dev: `pytest`; cấu hình `[tool.pytest.ini_options]`. **Không** thêm `scipy`/`pandas` (YAGNI). Cài bằng `uv sync` / `uv pip install -e .`.
  2. Viết test rồi implement:
     - `exceptions.py`: `QuantAutoMLError` (gốc), `InvalidDataError`, `TimestampAlignmentError`, `DataLeakageError`.
     - `types.py`: `TaskType`, `SplitMode`, `Segment`, `Dataset`, `SplitResult` (frozen dataclass), `IngestionReport`, `SplitReport` (có `to_dict()`).
     - `config.py`: `PipelineConfig` (Pydantic v2) + `load_config(path)` bằng `tomllib`; validate `horizon_h >= 1`, `min_match_ratio` trong (0, 1], thiếu `labels.selected` khi có `labels_path` → lỗi.
- **Verification Command:**
  ```bash
  pytest tests/unit/test_types.py tests/unit/test_config.py -v
  ```
- **Commit Message:** `chore(setup): configure project, common types, exceptions and TOML config`

---

### [x] Task 2: Data Ingestion
- **Mục tiêu:** Theo DESIGN.md §2.1.
- **Files to Touch:** Create `src/quantautoml/data/__init__.py`, `ingestion.py`, `tests/unit/test_ingestion.py`
- **Action Steps (test trước):**
  1. Test: đọc parquet và csv; tz khác nhau → UTC; naive → UTC + cờ trong report; timestamp trùng / không đơn điệu → `TimestampAlignmentError`; inner join bỏ dòng lệch; khớp < `min_match_ratio` → lỗi; feature không phải số → `InvalidDataError`; tràn float32 → lỗi; NaN/Inf feature giữ nguyên; `selected` không tồn tại → lỗi (liệt kê cột); nhãn chọn NaN cuối bị drop và có trong report; NaN giữa → lỗi; nhãn phụ giữ NaN; NaN cuối lớn nhất > `horizon_h` → lỗi; `labels_path=None` → `y=None`; report đủ trường.
  2. Implement `DataIngestion(config).load() -> Dataset` (gọi `infer_task_type` nên Task 3 được làm trước khi nối vào ingestion).
- **Verification Command:** `pytest tests/unit/test_ingestion.py -v`
- **Commit Message:** `feat(data): implement DataIngestion with UTC sync, label handling and report`

---

### [x] Task 3: Task Inference
- **Mục tiêu:** Theo bảng DESIGN.md §2.2.
- **Files to Touch:** Create `src/quantautoml/data/task_inference.py`, `tests/unit/test_task_inference.py`
- **Action Steps:**
  1. Test: `None` → UNSUPERVISED; 2 giá trị `{0,1}` và `{-1,1}` → BINARY (remap, `classes_`); 3–10 nguyên → MULTICLASS; 1 giá trị, >10 giá trị, nhãn không nguyên → `InvalidDataError`.
  2. Implement `infer_task_type(y) -> TaskInference(task_type, classes, y)` và tích hợp remap vào luồng ingestion.
- **Verification Command:** `pytest tests/unit/test_task_inference.py -v`
- **Commit Message:** `feat(data): implement task inference engine`

---

### [x] Task 4: Adaptive Temporal Splitter
- **Mục tiêu:** Theo DESIGN.md §2.3.
- **Files to Touch:** Create `src/quantautoml/data/splitter.py`, `tests/unit/test_splitter.py`
- **Action Steps:**
  1. Test: chọn mode tại N = 499 (lỗi) / 500 / 9,999 / 10,000; gap giữa **mọi** cặp segment liền kề (kể cả Cal-A, Cal-B) đúng `horizon_h` dòng; thứ tự thời gian tăng; `index` không chồng lấn; OOF: 5 fold, mỗi fold `min(val) - max(train) - 1 >= h`, chỉ nằm trong Train; `cal_a/cal_b = None` ở OOF và `oof_folds = None` ở 5-way; `h` quá lớn → `InvalidDataError`; phá ranh giới có chủ đích → `DataLeakageError`; API không có tham số shuffle; `labels_aux` cắt đúng theo segment; `SplitReport` đủ trường (gồm cảnh báo mất lớp).
  2. Implement `AdaptiveTimeSeriesSplitter.split(dataset, horizon_h) -> SplitResult`.
- **Verification Command:** `pytest tests/unit/test_splitter.py -v`
- **Commit Message:** `feat(data): implement AdaptiveTimeSeriesSplitter with purging and OOF folds`

---

### [x] Task 5: Zero Look-Ahead Leakage Integration Verification
- **Mục tiêu:** End-to-end từ file TOML + file dữ liệu đến `SplitResult`, chứng minh không rò rỉ.
- **Files to Touch:** Create `tests/integration/test_zero_leakage.py`
- **Action Steps:**
  1. Sinh dữ liệu giả lập: chuỗi giá, nhãn `fwd_ret_{1,5,21}` bằng `shift(-k)` (có NaN cuối), `horizon_h = 21`; kịch bản 15,000 dòng (5-way), 3,000 dòng (OOF), và N sát ngưỡng.
  2. Chạy `load_config → DataIngestion → infer_task_type → AdaptiveTimeSeriesSplitter`.
  3. Kiểm tra bằng **dữ liệu**: với mỗi segment, nhãn `fwd_ret_21` của dòng cuối không phụ thuộc giá của dòng thuộc segment sau (tính lại từ chuỗi giá gốc theo `index`); gap đúng 21; timestamp không chồng lấn; dtype `float32`.
  4. Negative test: `horizon_h = 5` với nhãn `fwd_ret_21` → `InvalidDataError`.
- **Verification Command:** `pytest tests/integration/test_zero_leakage.py -v`
- **Commit Message:** `test(integration): verify zero look-ahead leakage across adaptive temporal splits`

---

## Final Verification Checklist
- [x] `pytest tests/` đạt 100% PASS.
- [x] Module đặt đúng theo [`STRUCTURE.md`](STRUCTURE.md); `common` không import tầng trên.
- [x] Report của ingestion và split chứa đủ các trường trong DESIGN.md.
- [x] Tuân thủ Zero Look-ahead Leakage ([`SCOPE.md`](SCOPE.md), [`TRADE-OFF.md`](TRADE-OFF.md)).
- [x] Sẵn sàng tích hợp Phase 2.
