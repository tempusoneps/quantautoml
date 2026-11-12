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


## Phase 2 — Feature Funnel

Mục `[features]` là tùy chọn; thiếu mục này thì `PipelineConfig.features is None` (tương thích Phase 1).
`ic_horizons` ánh xạ kỳ hạn -> cột forward-return trong `labels_aux`; `target_horizon` phải là một khóa của `ic_horizons`.

```toml
[features]
target_horizon = 5
icir_min = 0.30             # ICIR(h_target) tối thiểu
fdr_alpha = 0.05            # Benjamini-Hochberg, miền hợp lệ (0, 0.05] (RULE-08)
rolling_window = 63
dcor_threshold = 0.85
cluster_distance = 0.30     # 1 - |corr(IC_i, IC_j)|
max_nan_ratio = 0.6
min_unique_ratio = 0.01
min_cv = 1e-6
max_peak_var_ratio = 50.0
max_features = 30
monotonic_tolerance = 0.05

[features.ic_horizons]
1 = "fwd_ret_1"
5 = "fwd_ret_5"
10 = "fwd_ret_10"
21 = "fwd_ret_21"
```

Khi có `[features]`, `run_pipeline` fit `FeatureFunnel` (F1 → F2 → F3+7 → F4 → F5 → F6 → F8) chỉ trên Train và ghi thêm vào `out_dir`: `funnel_report.json` (danh sách 7 báo cáo theo thứ tự tầng, mỗi phần tử gồm `stage`, `n_in`, `n_out`, `dropped` {tên: lý do}, `notes`; tầng bị bỏ qua vì hết feature có `n_in = n_out = 0` và note `"skipped: no features left"`) và `selected_features.json` (`{"selected_features": [...]}`). Các file `*.npz` của segment vẫn chứa X chưa lọc; thiếu `[features]` thì output giống hệt Phase 1. Task `unsupervised` cùng `[features]` → `InvalidDataError`.
