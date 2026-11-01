# Step 01 — Phase 1: Data Ingestion, Task Inference & Adaptive Temporal Splitter

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md). Checklist task đang thực thi: [`../PLAN.md`](../PLAN.md). Chi tiết thiết kế: [`DESIGN.md`](DESIGN.md).

- **Mục tiêu:** Nạp dữ liệu features/labels theo file cấu hình TOML, đồng bộ chuỗi thời gian, suy luận loại bài toán và phân chia dữ liệu thích ứng kèm Purging (`horizon_h`), đồng thời sinh report.

## Deliverables

1. `PipelineConfig` + `load_config()`: cấu hình TOML validate bằng Pydantic.
2. `DataIngestion`: trả `Dataset` (X `float32`, `y` nhãn chọn, `labels_aux`, timestamps UTC) + `IngestionReport`.
3. `infer_task_type()`: Binary / Multiclass / Unsupervised; từ chối nhãn liên tục.
4. `AdaptiveTimeSeriesSplitter`: trả `SplitResult` (5-way hoặc 3-way + `oof_folds`) + `SplitReport`.

## Tiêu chí kiểm thử độc lập

- Timestamp lỗi (trùng, không đơn điệu, khớp thấp) bị chặn bằng `TimestampAlignmentError`.
- NaN/Inf của feature được giữ nguyên; tràn `float32` bị chặn.
- `horizon_h` nhỏ hơn số NaN cuối của nhãn bị chặn.
- Gap giữa các segment liền kề đúng bằng `horizon_h` dòng; thứ tự thời gian đúng cho cả Cal-A/Cal-B.
- Ngưỡng chuyển mode đúng tại N = 499 / 500 / 9,999 / 10,000.
- Phá ranh giới có chủ đích → `DataLeakageError`.
- Report chứa đủ các trường mô tả trong DESIGN.md.
