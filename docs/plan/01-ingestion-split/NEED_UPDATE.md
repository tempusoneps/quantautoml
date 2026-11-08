# NEED_UPDATE — Step 01: Ingestion & Adaptive Temporal Split

> Tài liệu này ghi lại toàn bộ các **quyết định điều chỉnh thiết kế & sửa đổi sai lệch** so với `DESIGN.md` và `PLAN.md` gốc,
> được xác nhận bởi owner sau khi rà soát toàn diện và đối chiếu trực tiếp với dữ liệu thực tế (`datasets/feature.parquet`, `datasets/label.csv`).
> Ngày rà soát & cập nhật: 2026-11-08.

---

## Danh Sách Các Điều Chỉnh & Quyết Định Thiết Kế

### UD-01 — Chuẩn hóa độ chính xác Timestamp (Timestamp Precision Normalization)
- **Vấn đề phát hiện:**
  - `feature.parquet`: Cột timestamp có dtype `Datetime[ns]` (metadata từ Parquet).
  - `label.csv`: Cột timestamp có dtype `Datetime[μs]` (Polars CSV parser mặc định).
  - Polars từ chối join trực tiếp: `SchemaError: datatypes of join keys don't match`.
  - Thực tế dữ liệu: 100% timestamp là giây chẵn (sub-second = 0), không mất mát dữ liệu khi chuyển đổi.
- **Quyết định:**
  - `DataIngestion` **tự động chuẩn hóa** toàn bộ cột timestamp về `Datetime[μs]` (microseconds) trước khi thực hiện `inner join`.
  - Ghi nhận `timestamp_precision_normalized: bool` vào `IngestionReport`.

---

### UD-02 — Xử lý các cột Feature kiểu Boolean (Auto-cast → Float32)
- **Vấn đề phát hiện:**
  - `feature.parquet` có **25 cột kiểu `Boolean`** (ví dụ: `is_morning`, `heikin_ashi_bull`, `bb_rejection`, `adx_trend_filter`...).
- **Quyết định:**
  - `DataIngestion` **tự động ép kiểu** `Boolean → Int8 (0/1) → Float32`.
  - Ghi danh sách các cột boolean đã cast vào `IngestionReport.bool_columns_cast`.

---

### UD-03 — Xử lý các cột Feature kiểu String (Ordinal Encoding → Float32)
- **Vấn đề phát hiện:**
  - `feature.parquet` có **16 cột kiểu `String`** (ví dụ: `volume_group`, `long_trend`, `close_price_group`, `high_position`...).
- **Quyết định:**
  - `DataIngestion` **tự động áp dụng Ordinal Encoding** cho các cột String:
    - Sắp xếp danh sách unique value theo thứ tự bảng chữ cái (alphabetical) $\to$ ánh xạ sang $\{0, 1, 2, \dots\}$.
    - Cast kết quả sang `Float32`.
    - Phương pháp này thuần túy deterministic, không cần lưu trữ trạng thái `fit` phức tạp.
  - Lưu bảng ánh xạ vào `IngestionReport.string_columns_encoded` (`dict[str, dict[str, int]]`).

---

### UD-04 — Xử lý Naive Timestamp (Không có Timezone)
- **Vấn đề phát hiện:**
  - Cả `feature.parquet` và `label.csv` đều có `time_zone = None` (naive datetime).
- **Quyết định:**
  - Giữ nguyên thiết kế: Coi naive datetime là chuẩn **UTC ISO-8601**.
  - Đánh dấu cờ `IngestionReport.naive_utc_assumed = True`.

---

### UD-05 — Xử lý NaN ở giữa chuỗi dữ liệu của nhãn chọn (`mid_nulls`)
- **Vấn đề phát hiện:**
  - Trong `datasets/label.csv`, hầu hết các nhãn chiến lược intraday (`entry_filter`, `direction_filter`, `allow_entry`, `remain_session_net_move`...) đều có **5,581 dòng NaN ở giữa**, tương ứng 100% khung giờ **14:25 – 14:45** (cuối mỗi phiên giao dịch).
  - Quy tắc cũ trong `DESIGN.md`: *"NaN giữa $\to$ `InvalidDataError`"* sẽ làm sập pipeline ngay lập tức nếu chọn các nhãn này.
- **Quyết định:**
  - **Chỉ drop các dòng NaN ở cuối file** (tail NaNs phát sinh do `shift(-k)` forward-looking) để loại bỏ nhãn chưa hình thành.
  - **Không drop các dòng NaN ở giữa**: Giữ nguyên dòng để **bảo toàn tính liên tục của chuỗi thời gian** (không làm đứt gãy nến thời gian, không làm lệch phép tính lagging/rolling features sau này).
  - **Gán giá trị đặc biệt `-1`** (quy ước chuẩn `UNLABELED` / `NO_TRADE` trong Quant ML) cho các giá trị NaN ở giữa của nhãn chọn:
    - `y[mid_nans] = -1`
    - Mảng phân loại hợp lệ `classes_` vẫn chỉ chứa các lớp thực sự (ví dụ `{0, 1}`).
    - Ghi nhận `IngestionReport.mid_nan_count` để các tầng huấn luyện tiếp theo (Phase 3) biết và bỏ qua (hoặc mask) khi tính toán hàm mất mát (loss function).

---

### UD-06 — Hỗ trợ nhãn Multiclass dạng String
- **Vấn đề phát hiện:**
  - Các cột nhãn nhiều lớp trong `label.csv` đều là String (ví dụ `direction_filter` gồm `'None', 'Short', 'Long'`; `allow_entry` gồm 4 trạng thái chuỗi).
  - Thiết kế cũ bắt buộc Multiclass phải là số nguyên, nếu là String sẽ ném `InvalidDataError`.
- **Quyết định:**
  - Cho phép nhãn Multiclass ($3 \le \text{nunique} \le 10$) dạng String hoạt động đồng nhất như Binary:
    - Sắp xếp các class theo alphabet và ánh xạ sang số nguyên $\{0, 1, \dots, C-1\}$.
    - Lưu danh sách nhãn gốc vào `classes_`.

---

### UD-07 — Hệ quy chiếu Index trong `oof_folds` của `SplitResult`
- **Vấn đề phát hiện:**
  - Chưa làm rõ mảng `(train_idx, val_idx)` trong từng fold của `oof_folds` (chế độ `OOF_ASSISTED`) dùng tọa độ tương đối hay tuyệt đối.
- **Quyết định:**
  - `train_idx` và `val_idx` sử dụng **chỉ số tương đối theo tập Train**: Chạy từ `0` đến `len(train.X) - 1`.
  - Mục đích: Tối ưu cho việc indexing trực tiếp khi cross-validation ở Phase 3 (`train.X[train_idx]`). Vị trí tuyệt đối trên toàn bộ dataset đã được lưu trữ độc lập tại `Segment.index`.

---

### UD-08 — Đồng bộ hóa các sai lệch văn bản giữa `PLAN.md` và `DESIGN.md`
- **Sửa lỗi ngưỡng $N$:**
  - `PLAN.md` (dòng 4 và dòng 91) ghi nhầm $N < 5,000$. Sửa lại thành:
    - $N \ge 10{,}000$: `STATIC_5WAY`
    - $500 \le N < 10{,}000$: `OOF_ASSISTED`
    - $N < 500$: `InvalidDataError`
- **Sửa tên Exception chuẩn theo RULE-01:**
  - `PLAN.md` Task 1 ghi `LeakageError` $\to$ sửa thành `DataLeakageError`.
- **Bổ sung Module Config vào Task Breakdown:**
  - Task 1 trong `PLAN.md` bổ sung tạo `src/quantautoml/common/config.py` và `tests/unit/test_config.py` (`PipelineConfig` validate TOML qua Pydantic v2).

---

## Cấu Trúc Bổ Sung cho `IngestionReport`

```python
@dataclass(frozen=True)
class IngestionReport:
    # --- Các trường nguyên bản ---
    n_features_rows: int
    n_labels_rows: int
    n_joined_rows: int
    n_dropped_tail_nan: int
    time_min: str
    time_max: str
    feature_nan_count: int
    feature_inf_count: int

    # --- Bổ sung từ các quyết định mới ---
    timestamp_precision_normalized: bool                # UD-01: True nếu đã cast ns -> μs
    timestamp_original_time_unit: str                   # UD-01: 'ns', 'us', etc.
    bool_columns_cast: list[str]                        # UD-02: Danh sách feature bool -> float32
    string_columns_encoded: dict[str, dict[str, int]]   # UD-03: Mapping Ordinal Encoding {col: {val: idx}}
    naive_utc_assumed: bool                             # UD-04: True nếu naive datetime -> UTC
    mid_nan_count: int                                  # UD-05: Số dòng NaN ở giữa được gán nhãn -1
```

---

## Checklist Cập Nhật Tài Liệu Cần Thực Hiện

- [ ] Cập nhật [`docs/plan/01-ingestion-split/DESIGN.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/01-ingestion-split/DESIGN.md) theo các mục UD-01 đến UD-07.
- [ ] Cập nhật [`docs/plan/01-ingestion-split/SCOPE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/01-ingestion-split/SCOPE.md) làm rõ xử lý Boolean/String và nhãn mid-NaN `-1`.
- [ ] Cập nhật [`docs/plan/PLAN.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/plan/PLAN.md) theo mục UD-08 (`DataLeakageError`, ngưỡng 10,000, bổ sung `config.py`).
