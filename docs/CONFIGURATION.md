# Configuration (`config.toml`)

Pipeline được cấu hình bằng file TOML, được kiểm tra hợp lệ chặt chẽ bằng Pydantic v2 (`quantautoml.common.config.PipelineConfig`, `load_config(path)`). Mọi khóa lạ (extra fields) hoặc sai kiểu dữ liệu đều bị từ chối ngay khi nạp.

---

## Tổng Hợp Toàn Bộ Tham Số Thời Gian & Cửa Sổ Nến (Time & Horizon Parameters)

Tất cả các tham số thời gian trong `config.toml` đều được tính theo đơn vị **số lượng cây nến (bars/candles)**. Bảng dưới đây tổng hợp ý nghĩa của từng tham số và quy đổi tương đương trên **khung nến 5 phút** (giả định ~51 nến/ngày giao dịch như VN30F1M hoặc 288 nến/ngày với thị trường 24/7):

| Tham số trong `config.toml` | Mặc định | Đơn vị | Quy đổi trên nến 5 phút (VN30 ~51 nến/ngày) | Ý nghĩa & Vai trò trong Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| `split.horizon_h` | `21` | Nến (bars) | ~105 phút (~2 giờ giao dịch) | **Khoảng đệm Purging & Embargo:** Số cây nến bị cắt bỏ tại ranh giới giữa các tập (`Train`, `Cal-A`, `Cal-B`, `Val`, `OOS`) và giữa các fold Walk-Forward để triệt tiêu rò rỉ nhãn tương lai. **Bắt buộc $\ge$ kỳ hạn lớn nhất trong `ic_horizons` và $\ge$ số NaN ở cuối của mọi cột nhãn.** |
| `features.ic_horizons` | `{1, 5, 10, 21}` | Nến $\to$ Cột | 5 phút, 25 phút, 50 phút, 105 phút | **Cấu trúc kỳ hạn đa chiều (Term Structure):** Ánh xạ từ độ trễ $h$ nến sang cột lợi suất tương lai `fwd_ret_h = Close[t+h]/Close[t] - 1`. Dùng ở tầng F3+7 để kiểm tra tính đơn điệu suy giảm của tín hiệu theo thời gian. |
| `features.target_horizon` | `5` | Nến (bars) | 25 phút (5 nến 5 phút) | **Kỳ hạn mục tiêu chính ($h_{\text{target}}$):** Kỳ hạn chuẩn (phải thuộc khóa của `ic_horizons`) dùng để tính `ICIR`, kiểm định `Benjamini-Hochberg FDR` (F3+7), so sánh loại đa cộng tuyến `dCor` (F4) và gom cụm hành vi IC (F5). |
| `features.rolling_window` | `63` | Nến (bars) | ~315 phút (~1.2 phiên giao dịch) | **Cửa sổ trượt tính đặc trưng (Feature Rolling Window):** Độ dài cửa sổ cuộn để tính: (1) Peak Rolling Variance ở tầng F2; (2) Chuỗi Rolling Spearman IC & ICIR ở tầng F3+7 và F5; (3) Phân vùng 3 pha thị trường Bull/Bear/Sideways ($\pm 0.5\sigma$) ở tầng F6. |
| `backtest.rank_window` | `2000` | Nến (bars) | ~39 phiên (~2 tháng giao dịch) | **Cửa sổ xếp hạng phân vị nhân quả (Causal Percentile Window):** Tại mỗi nến OOS $t$, điểm dự báo `scores[t]` được xếp hạng phần trăm so với `2000` cây nến quá khứ gần nhất $[t - 1999, t]$ để quyết định vào/thoát lệnh qua `HysteresisBuffer`. (`2000 - 1` nến đầu tiên của OOS sẽ ở trạng thái đứng ngoài `pos = 0` để tích lũy đủ cửa sổ). |
| `backtest.ic_window` | `500` | Nến (bars) | ~10 phiên (~2 tuần giao dịch) | **Cửa sổ đánh giá rủi ro OOS (OOS Rolling IC & Stress Window):** Dùng trong `OOSBacktestRunner` cho 2 mục đích: (1) Tính chuỗi Rolling Rank IC trên tập OOS để suy ra chỉ số `ICIR`; (2) Tính lợi suất thị trường trượt `500` nến để xác định các cây nến sụt giảm mạnh nhất (`downside_quantile` 20% thấp nhất) khi đo `Downside IC`. |

---

## Phase 1 — Ingestion, Labels & Adaptive Temporal Split (`[data]`, `[labels]`, `[split]`)

```toml
[data]
features_path   = "datasets/feature.parquet"
labels_path     = "datasets/label.csv"   # bỏ trống / lược bỏ -> Unsupervised
timestamp_col   = "Date"
min_match_ratio = 0.95

[labels]
selected             = "is_peak_10"      # cột nhãn mục tiêu; hoặc "allow_entry", "direction_filter"
ignore_other_columns = true              # true: bỏ qua các cột nhãn phụ khác trong labels_path

[split]
horizon_h = 21                           # số nến Purging & Embargo ở mọi ranh giới tập dữ liệu; >= 1
```

| Khóa | Mặc định | Bắt buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `data.features_path` | — | Có | Đường dẫn tới file ma trận đặc trưng (`.parquet` hoặc `.csv`). |
| `data.labels_path` | `None` | Không | Đường dẫn tới file nhãn (`.parquet` hoặc `.csv`); nếu lược bỏ $\to$ chạy chế độ `UNSUPERVISED`. |
| `data.timestamp_col` | `"timestamp"` | Không | Tên cột thời gian ở cả hai file. Được chuẩn hóa tự động về UTC `datetime64[us]` và kiểm tra tăng dần nghiêm ngặt. |
| `data.min_match_ratio` | `0.95` | Không | Tỷ lệ khớp timestamp tối thiểu `(0.0, 1.0]` khi inner-join `features` và `labels`. Nếu thấp hơn sẽ ném `TimestampAlignmentError`. |
| `labels.selected` | `None` | Có (nếu có `labels_path`) | Tên cột nhãn dùng để suy luận loại bài toán (`binary_classification` hoặc `multiclass_classification`) và huấn luyện mô hình. |
| `labels.ignore_other_columns` | `false` | Không | Khi đặt `true`, bộ nạp dữ liệu chỉ giữ lại đúng cột `labels.selected` và bỏ qua mọi cột khác trong file nhãn (rất hữu ích khi file `label.csv` chứa nhiều cột chữ/boolean khác không dùng tới). Các cột `fwd_ret_*` cho Phase 2 sẽ được tự động tính từ cột giá `Close`. |
| `labels.mapping` | `None` | Không | Bảng `[labels.mapping]` ánh xạ nhãn dạng **chuỗi văn bản** sang **số có hướng** trước khi suy luận bài toán (xem mục `[labels.mapping]` bên dưới). |
| `split.horizon_h` | — | Có | Số cây nến Purging & Embargo bị loại bỏ tại mỗi ranh giới giữa các tập `Train`, `Cal-A`, `Cal-B`, `Val`, `OOS` ($\ge 1$). |

### Nhãn có hướng (`[labels.mapping]`)

Khi huấn luyện trên các nhãn văn bản có hướng như `allow_entry` hoặc `direction_filter`, khai báo thêm bảng `[labels.mapping]`:

```toml
[labels]
selected = "allow_entry"
ignore_other_columns = true

[labels.mapping]
"Yes - Buy"    = 1
"Yes - Sell"   = -1
"No - None"    = 0
"No - Sideway" = 0
"NaN"          = 0        # Tùy chọn: gán các dòng trống giữa chuỗi (mid-series NaN/null) về lớp 0
```

- Nhiều nhãn chữ có thể ánh xạ về cùng một con số (ví dụ `"No - None"` và `"No - Sideway"` cùng gộp về lớp trung tính `0`).
- **Ánh xạ giá trị trống (`"NaN" = 0` hoặc `NaN = 0`):** Nếu trong `[labels.mapping]` có khai báo khóa `"NaN"` (hoặc `"nan"`, `""`, `"null"`, `"None"`), toàn bộ các dòng bị trống nhãn ở giữa chuỗi (`mid-series NaN`, ví dụ 2 cây nến cuối mỗi phiên giao dịch trong `allow_entry`) sẽ được gán thẳng về lớp `0` thay vì tạo ra mã `UNLABELED = -1`, trong khi các dòng `NaN` ở cuối toàn bộ chuỗi dữ liệu (`trailing NaN` do forward-looking horizon) vẫn được bảo toàn để cắt bỏ an toàn.
- Mô hình GBDT sẽ huấn luyện trên mã lớp `0..K-1` theo thứ tự tăng dần của số đã map (`-1` $\to$ lớp `0` Sell, `0` $\to$ lớp `1` Flat, `+1` $\to$ lớp `2` Buy).
- Khi suy luận trong `predictor.py` và chạy Backtest, điểm dự báo (`score`) của bài toán đa lớp được tính bằng kỳ vọng lớp $\sum_{c=0}^{K-1} c \cdot P(y=c) = 1 + P(\text{Buy}) - P(\text{Sell})$, đồng biến 100% với hiệu xác suất có hướng $P(\text{Buy}) - P(\text{Sell})$ mà `HysteresisBuffer` sử dụng để vào lệnh Long/Short.

### Quy tắc chia phân vùng tự động (`AdaptiveTimeSeriesSplitter`)
- **Khi $N \ge 10,000$ dòng (`STATIC_5WAY`):** Chia tĩnh 5 tập liên tiếp theo thời gian: **Train (60%) $\to$ Cal-A (10%) $\to$ Cal-B (10%) $\to$ Val (10%) $\to$ OOS (10%)**, mỗi ranh giới cách nhau đúng `horizon_h` nến.
- **Khi $500 \le N < 10,000$ dòng (`OOF_ASSISTED`):** Chia **Train (80%) $\to$ Val (10%) $\to$ OOS (10%)** và tự động tạo 5-fold Expanding Walk-Forward OOF trên Train để cấp dữ liệu cho `Cal-A` và `Cal-B`.

---

## Phase 2 — 7-Stage Leakage-Free Feature Funnel (`[features]`)

```toml
[features]
ic_horizons = { 1 = "fwd_ret_1", 5 = "fwd_ret_5", 10 = "fwd_ret_10", 21 = "fwd_ret_21" }
target_horizon = 5
price_col = "Close"
icir_min = 0.30
fdr_alpha = 0.05
rolling_window = 63
dcor_threshold = 0.85
cluster_distance = 0.30
max_nan_ratio = 0.60
min_unique_ratio = 0.01
min_cv = 1e-6
max_peak_var_ratio = 50.0
max_features = 30
monotonic_tolerance = 0.05
mi_floor = 0.0
synergy_delta = 0.01
```

| Khóa | Mặc định | Ràng buộc | Tầng lọc | Ý nghĩa chi tiết |
| :--- | :--- | :--- | :--- | :--- |
| `features.ic_horizons` | — | Bắt buộc | F3+7 | Từ điển ánh xạ kỳ hạn (số nến $h$) sang tên cột lợi suất tương lai. Nếu cột chưa có trong `labels_aux`, `DataIngestion` tự động tính `fwd_ret_h = Close[t+h]/Close[t] - 1` từ cột `price_col`. |
| `features.target_horizon` | — | Thuộc `ic_horizons` | F3–F5 | Kỳ hạn mục tiêu chính ($h_{\text{target}}$) để chấm điểm IC và ICIR. |
| `features.price_col` | `"Close"` | Tên cột trong `X` | Ingestion | Tên cột giá đóng cửa trong file features dùng để tự suy `fwd_ret_*`. |
| `features.max_nan_ratio` | `0.60` | `(0.0, 1.0]` | **F1 (Triage)** | Loại bỏ ngay các cột có tỷ lệ giá trị thiếu (`NaN`/`Inf`) vượt quá 60% trên tập Train. |
| `features.min_unique_ratio` | `0.01` | `[0.0, 1.0]` | **F2 (Variance)** | Tỷ lệ giá trị duy nhất tối thiểu $n_{\text{unique}} / N_{\text{train}}$ (miễn trừ cho biến nhị phân 2 giá trị). |
| `features.min_cv` | `1e-6` | `>= 0.0` | **F2 (Variance)** | Hệ số biến thiên tối thiểu $|\sigma / \mu|$; loại bỏ các cột gần như hằng số. |
| `features.max_peak_var_ratio` | `50.0` | `> 0.0` | **F2 (Variance)** | Tỷ số tối đa giữa phân vị 90 của phương sai trượt (`rolling_window`) và phương sai toàn tập Train. |
| `features.rolling_window` | `63` | `>= 2` | **F2, F3+7, F5, F6** | Kích thước cửa sổ cuộn (nến) để tính Rolling Variance, Rolling Rank IC và phân loại Regime. |
| `features.icir_min` | `0.30` | float | **F3+7 (IC Stability)** | Ngưỡng $|\text{ICIR}(h_{\text{target}})|$ tối thiểu để đặc trưng được giữ lại. |
| `features.fdr_alpha` | `0.05` | `(0.0, 0.05]` | **F3+7 (FDR Gate)** | Mức ý nghĩa kiểm soát tỷ lệ phát hiện sai Benjamini-Hochberg (RULE-08, tối đa 5%). |
| `features.monotonic_tolerance` | `0.05` | `>= 0.0` | **F3+7 (Term Struct)** | Dung sai cho phép khi kiểm tra tính đơn điệu suy giảm $|\text{ICIR}(h_i)| \ge |\text{ICIR}(h_{i+1})| - \epsilon$ qua các kỳ hạn $\{1, 5, 10, 21\}$. |
| `features.dcor_threshold` | `0.85` | `(0.0, 1.0]` | **F4 (Collinearity)** | Ngưỡng Fast Distance Correlation tối đa giữa 2 đặc trưng; nếu $\text{dCor} > 0.85$, loại đặc trưng có $|\text{Rank IC}|$ thấp hơn. |
| `features.cluster_distance` | `0.30` | `(0.0, 1.0)` | **F5 (Clustering)** | Ngưỡng khoảng cách gom cụm phân cấp $d = 1 - |\text{corr}(\text{Rolling IC}_i, \text{Rolling IC}_j)|$; mỗi cụm chỉ giữ 1 đại diện có Mutual Information (MI) cao nhất. |
| `features.mi_floor` | `0.0` | `>= 0.0` | **F6 (Regime MI)** | Ngưỡng MI hiệu chỉnh chi-square tối thiểu trong ít nhất một trạng thái thị trường (Bull / Bear / Sideways). |
| `features.synergy_delta` | `0.01` | `>= 0.0` | **F6 (Synergy)** | Mức tăng tương tác thông tin tối thiểu $I(X_i, X_j; Y) - I(X_i; Y) - I(X_j; Y)$ (sau hiệu chỉnh bias) để bảo vệ cặp đặc trưng hiệp đồng. |
| `features.max_features` | `30` | `>= 1` | **F8 (JMI Selector)** | Số lượng đặc trưng tối đa được chọn vào tập **Golden Feature Set** cuối cùng bằng thuật toán Joint Mutual Information. |

---

## Phase 3 — Modern GBDT Tournament & Deep Optuna Tuning (`[models]`)

```toml
[models]
tournament_trials        = 30
tuning_trials            = 50
n_wf_folds               = 5
diversity_threshold      = 0.88
marginal_gain_threshold  = 0.015
wf_consistency_threshold = 0.60
complexity_penalty       = 0.05
random_state             = 42
enabled_models           = ["lightgbm", "xgboost", "catboost"]
```

| Khóa | Mặc định | Ràng buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `models.tournament_trials` | `30` | `>= 1` | Ngân sách số lượt thử `RandomSampler` đồng nhất cấp cho **mỗi** họ mô hình ở vòng đấu loại. |
| `models.tuning_trials` | `50` | `>= 1` | Số lượt tối ưu hóa siêu tham số sâu bằng `TPESampler(multivariate=True)` + `HyperbandPruner` cho các mô hình được chọn. |
| `models.n_wf_folds` | `5` | `>= 2` | Số fold Expanding Window Walk-Forward trên tập Train (mỗi fold đều áp dụng embargo `horizon_h` nến). |
| `models.diversity_threshold` | `0.88` | `(0.0, 1.0]` | Ngưỡng tương quan dự đoán tối đa cho phép giữa các mô hình trong Ensemble. Nếu $\max \text{corr} \ge 0.88$ $\to$ tự động fallback về Single Best Model. |
| `models.marginal_gain_threshold` | `0.015` | `>= 0.0` | Mức cải thiện Rank IC tối thiểu ($+1.5\%$) của tổ hợp so với mô hình đơn tốt nhất để chấp nhận Ensemble. |
| `models.wf_consistency_threshold` | `0.60` | `(0.0, 1.0]` | Tỷ lệ số fold Walk-Forward tối thiểu ($\ge 60\%$, tức $\ge 3/5$ folds) mà tổ hợp phải thắng mô hình đơn tốt nhất. |
| `models.complexity_penalty` | `0.05` | `>= 0.0` | Hệ số phạt độ phức tạp cây trong hàm mục tiêu Optuna: $\text{Score} = \text{IC Sharpe}_{\text{WF}} - 0.05 \times \text{Complexity}$. |
| `models.random_state` | `42` | `int` | Hạt giống ngẫu nhiên đảm bảo tính tái lập 100%. |
| `models.enabled_models` | `["lightgbm", "xgboost", "catboost"]` | Tập con hợp lệ | Danh sách các họ mô hình GBDT tham gia đấu loại. |

---

## Phase 4 — Convex SLSQP Ensemble & Temperature Calibration (`[ensemble]`)

```toml
[ensemble]
min_weight             = 0.05
max_weight             = 0.60
n_bootstrap            = 200
bootstrap_sample_ratio = 0.80
shrinkage_factor       = 0.10
hurdle_ratio           = 0.05
max_ece_threshold      = 0.05
random_state           = 42
```

| Khóa | Mặc định | Ràng buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `ensemble.min_weight` | `0.05` | `[0.0, 1.0]` | Cận dưới trọng số $w_i$ của mỗi mô hình trong bài toán tối ưu lồi SLSQP trên tập `Cal-B` (`min_weight < max_weight`). |
| `ensemble.max_weight` | `0.60` | `(0.0, 1.0]` | Cận trên trọng số $w_i$ của mỗi mô hình trong SLSQP, chống tập trung rủi ro vào một mô hình duy nhất. |
| `ensemble.n_bootstrap` | `200` | `>= 10` | Số vòng lấy mẫu Bootstrap trên tập `Cal-B` để giải SLSQP và lấy vector trọng số trung vị (median weights). |
| `ensemble.bootstrap_sample_ratio` | `0.80` | `(0.0, 1.0]` | Tỷ lệ kích thước mẫu trong mỗi lần lấy mẫu Bootstrap trên `Cal-B` (80% số dòng). |
| `ensemble.shrinkage_factor` | `0.10` | `[0.0, 1.0]` | Hệ số co cụm $\lambda$ kéo trọng số trung vị về trọng số đều $1/K$: $w_{\text{shrunk}} = (1-\lambda)w_{\text{med}} + \lambda (1/K)$. |
| `ensemble.hurdle_ratio` | `0.05` | `>= 0.0` | **RULE-05 Hurdle Gate:** Trọng số SLSQP phải vượt qua cả Single Best Model và Equal Weights ($1/K$) ít nhất $+5\%$ IC Sharpe trên `Cal-B`; nếu không đạt sẽ tự động fallback về trọng số đều $1/K$. |
| `ensemble.max_ece_threshold` | `0.05` | `(0.0, 1.0]` | Ngưỡng sai số hiệu chuẩn kỳ vọng tối đa (Expected Calibration Error $< 0.05$) sau khi tối ưu Temperature Scaling ($T > 0$) trên tập `Cal-A`. |
| `ensemble.random_state` | `42` | `int` | Hạt giống ngẫu nhiên cho quá trình lấy mẫu Bootstrap. |

---

## Phase 5 — ONNX Bundle Packaging & Dual-Tier Friction Backtest (`[packaging]`, `[backtest]`)

```toml
[packaging]
precision_tol     = 1e-4
drift_z_threshold = 3.0
latency_budget_ms = 5.0
std_floor         = 1e-8
latency_repeats   = 200

[backtest]
price_col         = "Close"
fee_bps           = 4.0
slippage_bps      = 1.0
top_pct           = 0.20
exit_pct          = 0.30
rank_window       = 2000
ic_window         = 500
min_rank_ic       = 0.035
min_icir          = 0.40
min_net_sharpe    = 1.20
max_mdd           = 0.18
downside_quantile = 0.20
min_downside_ic   = 0.0
```

### 1. Tham số Đóng gói ONNX (`[packaging]`)

| Khóa | Mặc định | Ràng buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `packaging.precision_tol` | `1e-4` | `> 0.0` | **RULE-06 Gate:** Độ lệch tuyệt đối tối đa cho phép $\max |\hat{y}_{\text{py}} - \hat{y}_{\text{onnx}}|$ trên toàn bộ tập `Val`. Nếu vượt ngưỡng, pipeline ném `PrecisionValidationError` và hủy xuất bundle. |
| `packaging.drift_z_threshold` | `3.0` | `> 0.0` | Ngưỡng $|z\text{-score}|$ (so với `train_mean` và `train_std` lưu trong `features_schema.json`) để `predictor.py` phát cảnh báo `DistributionShiftWarning` khi suy luận. |
| `packaging.latency_budget_ms` | `5.0` | `> 0.0` | Ngân sách độ trễ suy luận tối đa cho phép (ms/dòng) đối với batch 1 dòng. |
| `packaging.std_floor` | `1e-8` | `> 0.0` | Giá trị sàn cho độ lệch chuẩn `train_std` nhằm tránh lỗi chia cho 0 đối với các đặc trưng có phương sai cực nhỏ. |
| `packaging.latency_repeats` | `200` | `>= 1` | Số lần lặp đo thời gian suy luận batch 1 dòng (lấy trung vị median) để nghiệm thu gate độ trễ. |

### 2. Tham số Kiểm thử mù OOS & Chi phí ma sát (`[backtest]`)

| Khóa | Mặc định | Ràng buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `backtest.price_col` | `"Close"` | Tên cột trong `X` | Cột giá đóng cửa trong ma trận đặc trưng thô của tập `OOS` dùng để tính lợi suất thị trường $r_t = \text{Close}_{t+1}/\text{Close}_t - 1$. |
| `backtest.fee_bps` | `4.0` | `>= 0.0` | Phí giao dịch (Taker fee) tính theo **Basis Points** ($1\text{ bps} = 0.01\% = 10^{-4}$). |
| `backtest.slippage_bps` | `1.0` | `>= 0.0` | Độ trượt giá khớp lệnh (Slippage) tính theo **Basis Points**. Tổng chi phí mỗi đơn vị turnover là `(fee_bps + slippage_bps) / 10000`. |
| `backtest.top_pct` | `0.20` | `(0.0, 0.5)` | **Ngưỡng mở vị thế (Entry Percentile):** Mở Long (`+1`) khi phân vị tín hiệu $\ge 1 - \text{top\_pct}$ (`0.80`, tức Top 20%); mở Short (`-1`) khi phân vị $\le \text{top\_pct}$ (`0.20`, tức Bottom 20%). |
| `backtest.exit_pct` | `0.30` | `[top_pct, 0.5)` | **Ngưỡng đóng vị thế (Hysteresis Exit Percentile):** Đang Long chỉ đóng về Flat (`0`) khi phân vị tụt xuống $< 1 - \text{exit\_pct}$ (`0.70`); đang Short chỉ đóng khi phân vị vượt lên $> \text{exit\_pct}$ (`0.30`). Khoảng đệm `[0.20, 0.30]` giúp triệt tiêu nhiễu đảo lệnh liên tục quanh ngưỡng. |
| `backtest.rank_window` | `2000` | `>= 20` | Số cây nến quá khứ trong cửa sổ trượt nhân quả dùng để tính phân vị tín hiệu cho `HysteresisBuffer`. |
| `backtest.ic_window` | `500` | `>= 20` | Cửa sổ trượt (số nến) trên tập OOS để tính chuỗi Rolling Rank IC (cho `ICIR`) và lợi suất thị trường trượt (cho `Downside IC`). |
| `backtest.min_rank_ic` | `0.035` | float | **Hard Gate 1:** Rank IC tối thiểu trên toàn tập OOS (`rank_ic > 0.035`). |
| `backtest.min_icir` | `0.40` | float | **Hard Gate 2:** ICIR tối thiểu trên tập OOS (`icir > 0.40`). |
| `backtest.min_net_sharpe` | `1.20` | float | **Hard Gate 3 (RULE-09):** Net Sharpe thường niên hóa tối thiểu sau khi đã trừ phí giao dịch và trượt giá (`net_sharpe > 1.20`). |
| `backtest.max_mdd` | `0.18` | `> 0.0` | **Hard Gate 4:** Mức sụt giảm vốn tối đa (Max Drawdown) cho phép trên đường cong vốn ròng (`max_drawdown < 0.18` tức $< 18\%$). |
| `backtest.downside_quantile` | `0.20` | `(0.0, 0.5)` | Phân vị xác định các cây nến thị trường sụt giảm mạnh (20% các kỳ có lợi suất trượt `ic_window` thấp nhất). |
| `backtest.min_downside_ic` | `0.0` | float | **Hard Gate 5:** Rank IC tối thiểu trong các kỳ thị trường sụt giảm (`downside_ic >= 0.0`), đảm bảo mô hình không gãy khi thị trường stress. |

### 3. Tham số Mô phỏng Khớp lệnh Thực chiến Tier 2 (`[realworld]` — `backtesting.py`)

```toml
[realworld]
enabled          = false         # true: tự động chạy Tier 2 backtesting.py sau khi Tier 1 accepted
cash             = 1000000.0
commission       = 0.00025
margin           = 1.0
size             = "auto"        # "auto", "start_cash" ("_start_cash"), số nguyên >= 1 (vd: 1), hoặc tỷ lệ vốn (0, 1)
trade_on_close   = false
ohlcv_open_col   = "Open"
ohlcv_high_col   = "High"
ohlcv_low_col    = "Low"
ohlcv_close_col  = "Close"
ohlcv_volume_col = "Volume"
```

| Khóa | Mặc định | Ràng buộc | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `realworld.enabled` | `false` | `bool` | Cờ bật/tắt tầng backtest thực chiến Tier 2 (`backtesting.py`). Đặt `false` để chỉ chạy Tier 1 (NumPy vectorized); đặt `true` để tự động xuất thêm `realworld_report.json`, `trades.csv` và `backtest_report.html` khi Tier 1 `accepted = true`. |
| `realworld.cash` | `1_000_000.0` | `> 0.0` | Số vốn khởi tạo ban đầu (`initial_cash`) của tài khoản mô phỏng. |
| `realworld.commission` | `0.0005` | `>= 0.0` | Tỷ lệ phí giao dịch trên mỗi chiều khớp lệnh (`0.0005` = `0.05%` = `5 bps`). |
| `realworld.margin` | `1.0` | `> 0.0` | Tỷ lệ ký quỹ (`1.0` = không dùng đòn bẩy; `0.2` = đòn bẩy 5x). |
| `realworld.size` | `"auto"` | Xem các chế độ bên dưới | **Chế độ đi khối lượng lệnh (`Size`) cho mỗi giao dịch:**<br>• `"auto"`: Dùng toàn bộ vốn khả dụng hiện tại (`equity` lãi kép / compounding, hành vi mặc định của `backtesting.py`).<br>• `"start_cash"` hoặc `"_start_cash"`: Cố định ngân sách mỗi lệnh theo vốn ban đầu `cash` (không nhân lãi kép khi thắng; tự động giới hạn theo `equity` hiện có nếu tài khoản đang sụt giảm).<br>• Số nguyên $\ge 1$ (ví dụ `size = 1` hoặc `size = 5`): **Cố định số lượng hợp đồng/cổ phiếu** cho mỗi lệnh (ví dụ `size = 1` = đánh cố định 1 hợp đồng VN30F1M để cột `PnL` trong `trades.csv` phản ánh trực tiếp số điểm chỉ số).<br>• Số thực trong khoảng `(0.0, 1.0)` (ví dụ `size = 0.5`): Dùng một tỷ lệ phần trăm cố định (50%) của vốn `equity` hiện tại cho mỗi lệnh. |
| `realworld.trade_on_close` | `false` | `bool` | `false`: Tín hiệu tại nến $t$ sẽ khớp lệnh Market tại giá `Open` của nến $t+1$ (chống thiên kiến giá đóng cửa). |
| `realworld.out_html` | `None` | `str \| None` | Đường dẫn xuất biểu đồ nến HTML tương tác (`backtest_report.html`). |
| `realworld.out_trades_csv` | `None` | `str \| None` | Đường dẫn xuất nhật ký lệnh chi tiết (`trades.csv`). |

