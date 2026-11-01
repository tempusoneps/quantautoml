# Configuration

Pipeline được cấu hình bằng file TOML, parse và validate bằng Pydantic v2 (`quantautoml.common.config.PipelineConfig`, `load_config(path)`).

## Phase 1 — Ingestion & Split

```toml
[data]
features_path   = "datasets/features.parquet"
labels_path     = "datasets/labels.parquet"   # bỏ trống → Unsupervised
timestamp_col   = "timestamp"
min_match_ratio = 0.95

[labels]
selected = "fwd_ret_5"    # cột nhãn mục tiêu; các cột nhãn còn lại → labels_aux

[split]
horizon_h = 21            # horizon (số nến) LỚN NHẤT của mọi cột nhãn trong file; >= 1
```

| Khóa | Bắt buộc | Mô tả |
| :--- | :--- | :--- |
| `data.features_path` | Có | Đường dẫn features (`.parquet` / `.csv`) |
| `data.labels_path` | Không | Đường dẫn labels; vắng → `UNSUPERVISED` |
| `data.timestamp_col` | Không (`timestamp`) | Tên cột timestamp ở cả hai file |
| `data.min_match_ratio` | Không (`0.95`) | Tỷ lệ timestamp khớp tối thiểu khi inner join |
| `labels.selected` | Có nếu có labels | Cột nhãn dùng để suy luận task và huấn luyện |
| `split.horizon_h` | Có | Số dòng bị bỏ ở mỗi ranh giới giữa các segment |

### Vì sao cần `horizon_h`

Nhãn forward-looking (`shift(-k)`) tại dòng t dùng dữ liệu tới t+k. Dòng cuối của Train sẽ "chạm" vào đầu Val nếu không chừa khoảng trống. Đặt `horizon_h` bằng horizon lớn nhất trong các cột nhãn bạn dùng (Phase 2 sẽ dùng nhiều horizon). Giá trị quá nhỏ gây rò rỉ; quá lớn chỉ tốn vài dòng dữ liệu.

Pipeline kiểm tra an toàn: nếu một cột nhãn có số NaN ở cuối lớn hơn `horizon_h` thì báo `InvalidDataError`.

### Hằng số (không cấu hình)

`STATIC_MIN_ROWS = 10_000`, `MIN_ROWS = 500`, `OOF_N_FOLDS = 5`, tỷ lệ chia 60/10/10/10/10 (5-way) và 80/10/10 (OOF).
