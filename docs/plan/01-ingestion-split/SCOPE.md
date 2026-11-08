# Step 01 — Scope: Ingestion, Task Inference & Adaptive Temporal Splitter

> Thuộc [Phase 1](../PHASES.md). Phạm vi tổng thể xem [`../SCOPE.md`](../SCOPE.md). Cấu hình xem [`../../CONFIGURATION.md`](../../CONFIGURATION.md).

## Giả định nền

- Dữ liệu là **single-asset**: một chuỗi nến, mỗi timestamp xuất hiện đúng một lần.
- Các cột nhãn là **forward-looking** (`shift(-k)`): nhãn tại dòng t dùng dữ liệu tới t+k, nên cuối file có k dòng NaN. Do đó bắt buộc Purging bằng `horizon_h` (xem DESIGN.md).

## In-Scope (Must Have)

- [ ] Đọc cấu hình từ file TOML, validate bằng Pydantic v2 (`PipelineConfig`).
- [ ] Nạp `features.{parquet,csv}` và `labels.{parquet,csv}` (labels có thể chứa nhiều cột nhãn; `labels.selected` chọn nhãn mục tiêu; labels có thể vắng mặt → Unsupervised).
- [ ] Chuẩn hóa timestamp về UTC `datetime64[ns]`; kiểm tra tăng nghiêm ngặt và không trùng (không tự sort); inner join; kiểm tra tỷ lệ khớp `min_match_ratio`.
- [ ] Ép feature và nhãn phụ về `np.float32`; giữ nguyên NaN/Inf của feature (Phase 2 xử lý).
- [ ] Xử lý nhãn chọn: NaN ở cuối bị drop và báo cáo; NaN ở giữa → `InvalidDataError`.
- [ ] Kiểm tra an toàn horizon: số NaN cuối lớn nhất trong các cột nhãn > `horizon_h` → `InvalidDataError`.
- [ ] Tự động suy luận bài toán: Binary, Multiclass ($3 \le \text{nunique} \le 10$, nhãn nguyên), Unsupervised; nhãn liên tục/regression bị từ chối.
- [ ] **Adaptive Temporal Split:**
  - $N \ge 10{,}000$: `Train 60% | Cal-A 10% | Cal-B 10% | Val 10% | OOS 10%`.
  - $N < 10{,}000$: `Train 80% | Val 10% | OOS 10%` + `oof_folds` (5-fold expanding walk-forward đã purge trên Train). Việc tách Cal-A/Cal-B từ OOF thuộc Phase 4.
  - $N < 500$: `InvalidDataError`.
- [ ] Purging & Embargo: bỏ `horizon_h` dòng đầu của mỗi segment sau (kể cả validation block trong từng OOF fold).
- [ ] Splitter tự kiểm tra hậu điều kiện, vi phạm → `DataLeakageError`.
- [ ] Sinh `IngestionReport` và `SplitReport` (dataclass + `to_dict()` + logging).

## Out-of-Scope (Phase 1)

- Dữ liệu nhiều mã (panel), regression, impute/drop NaN của feature, sinh OOF predictions, ghi report ra file/CLI.

## Ràng buộc liên quan

- Chỉ mục thời gian bắt buộc tăng đơn điệu, không trùng lặp, không xáo trộn.
- Feature chuẩn hóa `np.float32`; nhãn classification là `int32`.
