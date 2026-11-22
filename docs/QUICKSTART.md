# Quick Start & Hướng Dẫn Sử Dụng (`QUICKSTART`)

Tài liệu này hướng dẫn cách cài đặt môi trường, sinh file cấu hình `config.toml` bằng `scripts/generate_config.py` (bao gồm cách đổi nhãn mục tiêu `selected`), chạy toàn bộ pipeline và sử dụng mô hình ONNX xuất xưởng.

---

## 1. Yêu Cầu Hệ Thống & Cài Đặt

- **Hệ điều hành:** Linux / macOS / Windows (WSL2)
- **Python:** $\ge 3.12$
- **Trình quản lý gói:** [`uv`](https://docs.astral.sh/uv/)

```bash
# Cài đặt toàn bộ dependencies (bao gồm cả dev group và backtesting.py)
uv sync
```

---

## 2. Sinh File Cấu Hình `config.toml` Tự Động (`scripts/generate_config.py`)

Dự án cung cấp sẵn script [`scripts/generate_config.py`](scripts/generate_config.py) để tự động quét header của file dữ liệu (`datasets/feature.parquet`, `datasets/label.csv`) và sinh ra file `config.toml` hoàn chỉnh cho cả 5 Phase (có thể chạy được ngay).

### 2.1. Sinh cấu hình mặc định (nhãn nhị phân `is_peak_10`)

```bash
uv run python scripts/generate_config.py --force
```

### 2.2. Đổi nhãn mục tiêu sang nhãn giao dịch có hướng (`allow_entry`)

Khi bạn chọn nhãn dạng chữ có hướng như `allow_entry` (`"Yes - Buy"`, `"Yes - Sell"`, `"No - None"`, `"No - Sideway"`) hoặc `direction_filter` (`"Long"`, `"Short"`, `"None"`), script sẽ **tự động**:
1. Bật `ignore_other_columns = true` trong `[labels]` để bỏ qua các cột nhãn phụ không phải kiểu số trong `datasets/label.csv`.
2. Sinh bảng ánh xạ `[labels.mapping]` chuẩn (`+1` cho Buy/Long, `-1` cho Sell/Short, `0` cho None/Sideway).

```bash
# Đổi sang nhãn allow_entry và ghi đè vào config.toml
uv run python scripts/generate_config.py --selected-label allow_entry --force

# Hoặc xuất ra một file config riêng cho từng thí nghiệm
uv run python scripts/generate_config.py --selected-label allow_entry --output config_allow_entry.toml --force
uv run python scripts/generate_config.py --selected-label direction_filter --output config_direction.toml --force
uv run python scripts/generate_config.py --selected-label entry_filter --output config_entry_filter.toml --force
```

### 2.3. Tùy chỉnh kỳ hạn (`horizon-h`, `target-horizon`) và số lượng Trials

```bash
uv run python scripts/generate_config.py \
  --selected-label allow_entry \
  --horizon-h 21 \
  --target-horizon 5 \
  --tournament-trials 30 \
  --tuning-trials 50 \
  --output config.toml \
  --force
```

### 2.4. Sinh cấu hình cho bài toán Không giám sát (Unsupervised Mode)

```bash
uv run python scripts/generate_config.py --labels-path "" --output config_unsupervised.toml --force
```

### 2.5. Bảng tham số CLI của `scripts/generate_config.py`

| Tham số CLI | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `-o`, `--output` | `config.toml` | Đường dẫn file cấu hình TOML đầu ra. |
| `--features-path` | `datasets/feature.parquet` | Đường dẫn tới file đặc trưng (`.parquet` hoặc `.csv`). |
| `--labels-path` | `datasets/label.csv` | Đường dẫn tới file nhãn (truyền `""` nếu chạy Unsupervised). |
| `--timestamp-col` | Auto-detect (`Date`) | Tên cột thời gian chung giữa file features và labels. |
| `--price-col` | Auto-detect (`Close`) | Tên cột giá dùng để tự suy `fwd_ret_*` và tính P&L trong backtest. |
| `--min-match-ratio` | `0.95` | Tỷ lệ khớp timestamp tối thiểu khi inner-join features và labels. |
| `--selected-label` | Auto (`is_peak_10`) | Tên cột nhãn mục tiêu cần huấn luyện (`is_peak_10`, `allow_entry`, `direction_filter`, ...). |
| `--keep-all-labels` | `False` | Nếu bật cờ này, sẽ đặt `ignore_other_columns = false` (yêu cầu mọi cột trong file nhãn đều là số). |
| `--horizon-h` | `21` | Số nến Purging & Embargo giữa các tập dữ liệu (phải $\ge$ kỳ hạn lớn nhất trong `ic_horizons`). |
| `--target-horizon` | `5` | Kỳ hạn mục tiêu chính (nến) để tính ICIR và lọc đặc trưng ở Phase 2. |
| `--tournament-trials` | `30` | Số lượt thử ngẫu nhiên cho mỗi mô hình GBDT ở vòng đấu loại. |
| `--tuning-trials` | `50` | Số lượt tối ưu sâu bằng Optuna Bayesian TPE. |
| `--force` | `False` | Ghi đè file cấu hình đã tồn tại mà không hỏi xác nhận. |

---

## 3. Chạy Toàn Bộ Pipeline

```bash
# Chạy với config.toml mặc định và xuất toàn bộ báo cáo + model bundle vào runs/default
uv run quantautoml run --config config.toml --out runs/default

# Hoặc chạy với cấu hình nhãn allow_entry
uv run quantautoml run --config config_allow_entry.toml --out runs/allow_entry
```

---

## 4. Sử Dụng Gói Suy Luận Độc Lập (`bundle/predictor.py`)

Sau khi chạy xong pipeline, thư mục `runs/default/bundle/` chứa trọn bộ mô hình ONNX và file `predictor.py` **không phụ thuộc vào thư viện `quantautoml` hay bất kỳ framework huấn luyện nào** (chỉ cần `numpy` và `onnxruntime`):

```python
import numpy as np
from runs.default.bundle.predictor import Predictor

# 1. Khởi tạo Predictor từ thư mục bundle
predictor = Predictor("runs/default/bundle")

# 2. Chuẩn bị ma trận đặc trưng thô X (shape: [N, n_raw_features], dtype: np.float32)
X_live = np.zeros((1, 824), dtype=np.float32)

# 3. Dự báo điểm tín hiệu đã hiệu chuẩn Temperature Scaling
# - Bài toán nhị phân: trả về xác suất lớp 1 P(y=1)
# - Bài toán đa lớp có hướng (-1, 0, +1): trả về kỳ vọng lớp có hướng
scores = predictor.predict(X_live)

# 4. Hoặc lấy toàn bộ ma trận xác suất của từng lớp [N, n_classes]
probs = predictor.predict_proba(X_live)
```
