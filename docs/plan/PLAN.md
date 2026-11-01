# Implementation Plan: Quant AutoML Pipeline

**Active Phase:** Phase 1 — Data Ingestion, Task Inference & Adaptive Leakage-Free Temporal Splitter  
**Target Deliverable:** Sub-system hoàn chỉnh tiếp nhận dữ liệu bảng tài chính (`features`, `labels`), đồng bộ hóa thời gian UTC ISO-8601, tự động suy luận loại bài toán, và thực hiện phân chia thích ứng (Adaptive Partitioning: 5-way static khi $N \ge 10,000$ hoặc OOF-assisted khi $N < 5,000$) kèm cơ chế Purging & Embargo $h$ nến loại bỏ 100% rò rỉ dữ liệu tương lai.  
**Prerequisites:** Tham chiếu [`docs/plan/SCOPE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/SCOPE.md), [`docs/plan/PHASES.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/PHASES.md), [`docs/plan/STRUCTURE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/STRUCTURE.md), [`docs/plan/ARCHITECTURE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/ARCHITECTURE.md), và [`docs/plan/TRADE-OFF.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/TRADE-OFF.md).

---

## Execution Order Overview

- [ ] **Task 1:** Thiết lập môi trường, dependencies và module types/exceptions dùng chung.
- [ ] **Task 2:** Xây dựng module Ingestion & Đồng bộ hóa thời gian UTC ISO-8601 (`ingestion.py`).
- [ ] **Task 3:** Xây dựng Engine tự động suy luận bài toán (`task_inference.py`).
- [ ] **Task 4:** Xây dựng Adaptive Temporal Splitter (5-way vs. OOF-assisted) kèm Purging & Embargo $h$ nến (`splitter.py`).
- [ ] **Task 5:** Viết bộ kiểm thử tích hợp chuyên sâu chống rò rỉ dữ liệu tương lai (`test_zero_leakage.py`).

---

## Detailed Task Breakdown

### [ ] Task 1: Setup Dependencies, Common Types & Exceptions
- **Mục tiêu:** Cập nhật `pyproject.toml` với các thư viện cần thiết cho Phase 1, định nghĩa enum `TaskType`, dataclass `SplitResult`, `DatasetContainer`, cấu hình `SplitMode`, và các ngoại lệ tùy chỉnh.
- **Files to Touch:**
  - Modify: `pyproject.toml`
  - Create: `src/quantautoml/common/__init__.py`
  - Create: `src/quantautoml/common/types.py`
  - Create: `src/quantautoml/common/exceptions.py`
- **Action Steps:**
  1. Thêm dependencies cơ sở vào `pyproject.toml`: `polars>=1.0.0`, `pandas>=2.2.0`, `numpy>=1.26.0`, `scipy>=1.13.0`, `pydantic>=2.0.0`, dev-dependencies: `pytest>=8.0.0`.
  2. Tạo `types.py` định nghĩa:
     - `TaskType(str, Enum)`: `BINARY_CLASSIFICATION`, `MULTICLASS_CLASSIFICATION`, `UNSUPERVISED`.
     - `SplitMode(str, Enum)`: `STATIC_5WAY`, `OOF_ASSISTED`.
     - `SplitResult(BaseModel)` hoặc `@dataclass`: chứa `train`, `cal_a`, `cal_b`, `val`, `oos` (mỗi phần gồm `X`, `y`, `timestamps`).
  3. Tạo `exceptions.py` định nghĩa `LeakageError`, `InvalidDataError`, `TimestampAlignmentError`.
- **Verification Command:**
  ```bash
  python -c "import quantautoml.common.types as t; print(t.TaskType.BINARY_CLASSIFICATION, t.SplitMode.STATIC_5WAY)"
  ```
- **Commit Message:** `chore(setup): configure dependencies and define common types and exceptions`

---

### [ ] Task 2: Data Ingestion & Strict UTC Timestamp Synchronization
- **Mục tiêu:** Đọc file parquet/csv của features và labels, chuyển timestamp về UTC ISO-8601 (`datetime64[ns]`), inner join theo timestamp, ép kiểu dữ liệu `np.float32`, và kiểm tra thứ tự đơn điệu tăng dần.
- **Files to Touch:**
  - Create: `src/quantautoml/data/__init__.py`
  - Create: `src/quantautoml/data/ingestion.py`
  - Create: `tests/conftest.py`
  - Create: `tests/unit/test_ingestion.py`
- **Action Steps:**
  1. Viết test thất bại trong `tests/unit/test_ingestion.py`:
     - Test đọc file parquet/csv.
     - Test chuyển đổi múi giờ khác nhau về UTC ISO-8601.
     - Test loại bỏ dòng lệch timestamp qua phép inner join.
     - Test kiểm tra lỗi nếu dữ liệu bị xáo trộn hoặc không đơn điệu tăng.
     - Test ép kiểu số học sang `np.float32`.
  2. Triển khai lớp `DataIngestion` trong `src/quantautoml/data/ingestion.py`.
  3. Chạy test để đảm bảo PASS 100%.
- **Verification Command:**
  ```bash
  pytest tests/unit/test_ingestion.py -v
  ```
- **Commit Message:** `feat(data): implement DataIngestion with UTC ISO-8601 synchronization`

---

### [ ] Task 3: Task Inference Engine
- **Mục tiêu:** Phân tích vector nhãn $Y$ để tự động xác định bản chất bài toán máy học: Binary ($Y \in \{0, 1\}$), Multiclass ($3 \le nunique(Y) \le 10$), hoặc Unsupervised.
- **Files to Touch:**
  - Create: `src/quantautoml/data/task_inference.py`
  - Create: `tests/unit/test_task_inference.py`
- **Action Steps:**
  1. Viết test trong `tests/unit/test_task_inference.py`:
     - Test với nhãn nhị phân float/int $\to$ trả về `TaskType.BINARY_CLASSIFICATION`.
     - Test với nhãn đa lớp 3-10 unique $\to$ trả về `TaskType.MULTICLASS_CLASSIFICATION`.
     - Test khi không có nhãn hoặc chỉ định unsupervised $\to$ `TaskType.UNSUPERVISED`.
     - Test báo lỗi nếu số lượng unique nhãn không hợp lệ.
  2. Hiện thực hàm `infer_task_type(y: np.ndarray) -> TaskType` trong `src/quantautoml/data/task_inference.py`.
  3. Chạy test xác nhận PASS.
- **Verification Command:**
  ```bash
  pytest tests/unit/test_task_inference.py -v
  ```
- **Commit Message:** `feat(data): implement TaskInference engine for automatic problem classification`

---

### [ ] Task 4: Adaptive Temporal Splitter (with Purging & Embargo)
- **Mục tiêu:** Xây dựng bộ chia chuỗi thời gian thích ứng:
  - Nếu $N \ge 10,000$: Phân chia tĩnh 5 vùng: Train (60%), Cal-A (10%), Cal-B (10%), Val (10%), OOS (10%).
  - Nếu $N < 5,000$: Phân chia Train (80%), Val (10%), OOS (10%), hỗ trợ trích xuất OOF indices.
  - Tự động áp dụng khoảng cách an toàn (Purging & Embargo) $h$ nến giữa các ranh giới.
- **Files to Touch:**
  - Create: `src/quantautoml/data/splitter.py`
  - Create: `tests/unit/test_splitter.py`
- **Action Steps:**
  1. Viết unit tests trong `tests/unit/test_splitter.py`:
     - Kiểm tra tự động chuyển đổi giữa `STATIC_5WAY` và `OOF_ASSISTED` theo số lượng dòng $N$.
     - Kiểm tra khoảng cách giữa điểm cuối tập trước và điểm đầu tập sau đúng bằng $h$ nến (Embargo gap).
     - Kiểm tra tính liên tục theo thời gian: $\max(t_{train}) < \min(t_{val}) < \max(t_{val}) < \min(t_{oos})$.
     - Kiểm tra cấm xáo trộn dữ liệu (`shuffle=False`).
  2. Hiện thực lớp `AdaptiveTimeSeriesSplitter` trong `src/quantautoml/data/splitter.py`.
  3. Chạy test xác nhận PASS.
- **Verification Command:**
  ```bash
  pytest tests/unit/test_splitter.py -v
  ```
- **Commit Message:** `feat(data): implement AdaptiveTimeSeriesSplitter with purging and embargo`

---

### [ ] Task 5: Zero Look-Ahead Leakage Integration Verification
- **Mục tiêu:** Tạo bộ kiểm thử tích hợp mô phỏng toàn bộ luồng Phase 1 với dữ liệu chuỗi thời gian sinh ngẫu nhiên có chèn cố tình các bẫy rò rỉ dữ liệu để đảm bảo hệ thống chặn đứng mọi nguy cơ.
- **Files to Touch:**
  - Create: `tests/integration/test_zero_leakage.py`
- **Action Steps:**
  1. Tạo kịch bản tích hợp sinh 15,000 dòng nến mô phỏng với horizon $h = 5$ nến.
  2. Nạp qua `DataIngestion`, suy luận qua `TaskInference`, và phân tách qua `AdaptiveTimeSeriesSplitter`.
  3. Kiểm tra toán học:
     - Không có giá trị timestamp nào của tập tương lai xuất hiện trong tập quá khứ.
     - Độ chênh lệch giữa các mốc biên đạt chuẩn $h$ nến embargo.
     - Dữ liệu hoàn toàn ép kiểu `float32`.
- **Verification Command:**
  ```bash
  pytest tests/integration/test_zero_leakage.py -v
  ```
- **Commit Message:** `test(integration): verify zero look-ahead leakage across adaptive temporal splits`

---

## Final Verification Checklist
- [ ] Cài đặt thành công dependencies và chạy `pytest tests/` đạt 100% PASS.
- [ ] Mọi module tuân thủ cấu trúc định nghĩa tại [`docs/plan/STRUCTURE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/STRUCTURE.md).
- [ ] Dữ liệu nạp và phân vùng tuân thủ nguyên tắc Zero Look-ahead Leakage của [`docs/plan/SCOPE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/SCOPE.md) và các trade-offs trong [`docs/plan/TRADE-OFF.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/TRADE-OFF.md).
- [ ] Sẵn sàng tích hợp sang Phase 2 (Leakage-Free Feature Funnel).
