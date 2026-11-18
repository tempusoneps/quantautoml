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
selected = "y_dir"        # cột nhãn mục tiêu; các cột nhãn còn lại → labels_aux

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
| `labels.ignore_other_columns` | Không (mặc định `false`) | `true`: ingestion chỉ giữ cột `labels.selected`, bỏ qua mọi cột nhãn khác (không kiểm tra kiểu, không đưa vào `labels_aux`, không tính vào trailing-NaN/`horizon_h`). Nhãn chọn dạng chuỗi tự được mã hóa số (UD-06). |
| `split.horizon_h` | Có | Số dòng bị bỏ ở mỗi ranh giới giữa các segment |

### Vì sao cần `horizon_h`

Nhãn forward-looking (`shift(-k)`) tại dòng t dùng dữ liệu tới t+k. Dòng cuối của Train sẽ "chạm" vào đầu Val nếu không chừa khoảng trống. Đặt `horizon_h` bằng horizon lớn nhất trong các cột nhãn bạn dùng (Phase 2 sẽ dùng nhiều horizon). Giá trị quá nhỏ gây rò rỉ; quá lớn chỉ tốn vài dòng dữ liệu.

Pipeline kiểm tra an toàn: nếu một cột nhãn có số NaN ở cuối lớn hơn `horizon_h` thì báo `InvalidDataError`.

### Hằng số (không cấu hình)

`STATIC_MIN_ROWS = 10_000`, `MIN_ROWS = 500`, `OOF_N_FOLDS = 5`, tỷ lệ chia 60/10/10/10/10 (5-way) và 80/10/10 (OOF).


## Phase 2 — Feature Funnel

Mục `[features]` là tùy chọn; thiếu mục này thì `PipelineConfig.features is None` (tương thích Phase 1).
`ic_horizons` ánh xạ kỳ hạn -> cột forward-return trong `labels_aux`; `target_horizon` phải là một khóa của `ic_horizons`.

> **Tự suy `fwd_ret_*`:** nếu một cột trong `ic_horizons` không có trong file labels (hoặc bị bỏ qua bởi `labels.ignore_other_columns`), ingestion tính `fwd_ret_h = price[t+h]/price[t] − 1` từ cột `features.price_col` (mặc định `Close`) và đưa vào `labels_aux` *trước khi split*; dòng cuối thiếu `Close[t+h]` là NaN. Cột đã có sẵn trong file labels không bị ghi đè. Thiếu cột giá → `InvalidDataError`. `split.horizon_h` vẫn phải ≥ kỳ hạn lớn nhất (purging), nếu không ingestion báo lỗi.

> **Lưu ý:** ingestion loại cột `labels.selected` khỏi `labels_aux`, nên mọi cột trong `ic_horizons` phải là cột forward-return phụ **khác** `labels.selected` (ví dụ `selected = "y_dir"` như ví dụ Phase 1, còn `fwd_ret_1..21` nằm trong file labels). Nếu đặt `selected = "fwd_ret_5"` đồng thời ánh xạ `5 = "fwd_ret_5"`, F3+7 báo `labels_aux is missing column 'fwd_ret_5'`.

```toml
[features]
target_horizon = 5
icir_min = 0.30             # ICIR(h_target) tối thiểu
fdr_alpha = 0.05            # Benjamini-Hochberg, miền hợp lệ (0, 0.05] (RULE-08)
rolling_window = 63
dcor_threshold = 0.85
cluster_distance = 0.30     # 1 - |corr(IC_i, IC_j)|, miền (0, 1): 1.0 sẽ gộp mọi feature
max_nan_ratio = 0.6
min_unique_ratio = 0.01
min_cv = 1e-6
max_peak_var_ratio = 50.0
max_features = 30
monotonic_tolerance = 0.05
mi_floor = 0.0              # F6: ngưỡng hiệu ứng (nats) cộng thêm trên kiểm định chi2-0.99
synergy_delta = 0.01        # F6: interaction gain tối thiểu (nats) sau hiệu chỉnh bias

[features.ic_horizons]
1 = "fwd_ret_1"
5 = "fwd_ret_5"
10 = "fwd_ret_10"
21 = "fwd_ret_21"
```

| Khóa (F6) | Mặc định | Ràng buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `features.mi_floor` | `0.0` | `>= 0` | Giữ feature nếu MI tốt nhất theo regime **đã hiệu chỉnh** `max(0, I - chi2.ppf(0.99, df)/(2 n_r))` lớn hơn **nghiêm ngặt** (`>`) ngưỡng này. `0.0` = "có ý nghĩa thống kê ở mức 1% trong ít nhất một regime"; giá trị dương là ngưỡng effect size bổ sung. |
| `features.synergy_delta` | `0.01` | `>= 0` | Cặp (Xi, Xj) là synergistic nếu `I(Xi,Xj;Y) - I(Xi;Y) - I(Xj;Y) - chi2.ppf(0.99, (Bi-1)(Bj-1)(C-1))/(2N) > synergy_delta`; hai feature của cặp được giữ dù không qua gate MI. |

Khi có `[features]`, `run_pipeline` fit `FeatureFunnel` (F1 → F2 → F3+7 → F4 → F5 → F6 → F8) chỉ trên Train và ghi thêm vào `out_dir`: `funnel_report.json` (danh sách 7 báo cáo theo thứ tự tầng, mỗi phần tử gồm `stage`, `n_in`, `n_out`, `dropped` {tên: lý do}, `notes`; tầng bị bỏ qua vì hết feature có `n_in = n_out = 0` và note `"skipped: no features left"`) và `selected_features.json` (`{"selected_features": [...]}`). Các file `*.npz` của segment vẫn chứa X chưa lọc; thiếu `[features]` thì output giống hệt Phase 1. Task `unsupervised` cùng `[features]` → `InvalidDataError`.

### Giới hạn đã biết / hiệu chuẩn (Known limitations)

1. **RULE-08 / BH:** p-value của F3+7 là Spearman gộp trên toàn bộ dòng Train (giả định iid, theo plan). Với feature có tính bền (persistent) và target forward-return chồng lấn (`target_horizon > 1`), BH tự nó **không** kiểm soát FDR: mô phỏng nhiễu AR(1) φ = 0.98 với `target_horizon = 5` có 52–123/300 feature được BH chấp nhận. Cả tầng vẫn giữ ≤ 5% trong mô phỏng chỉ nhờ gate ICIR (tệ nhất 15/300). Tỷ lệ nhiễu đã kiểm chứng trong test là cho feature iid với `target_horizon = 1`.
2. **F2 `max_peak_var_ratio`:** phân vị 90 của rolling variance bị chặn khoảng 10× phương sai toàn Train (bất đẳng thức Markov; đo được tối đa 7.1), nên mặc định `50` (và mọi ngưỡng > ~10) không bao giờ kích hoạt. Cần hiệu chuẩn trên dữ liệu thật hoặc đổi metric.
3. **F3+7 sign-consistency / monotonic:** luật nghiêm ngặt có thể cắt nhầm tín hiệu thật khi IC ở kỳ hạn dài chỉ là nhiễu (2/10 seed bị loại ở IC 0.10).
4. **Công suất F6:** gate ý nghĩa chi2-0.99 với 3 regime chỉ giữ ~13/40 tín hiệu ở IC 0.05 và ~34/40 ở IC 0.08 (N = 9,000); nhiễu lọt qua ~5%.
5. **F6/F8 bỏ qua `ctx.discrete_features`:** mã categorical bị quantile-bin như biến thứ bậc (ordinal).


## Phase 3 — GBDT Tournament & Deep Tuning

Mục `[models]` là tùy chọn; thiếu mục này thì `PipelineConfig.models is None` (tương thích ngược Phase 1 & 2).

```toml
[models]
tournament_trials = 30           # Ngân sách RandomSampler cho mỗi candidate
tuning_trials = 50               # Ngân sách Optuna Bayesian TPE tuning
n_wf_folds = 5                   # Số folds Expanding Walk-Forward
diversity_threshold = 0.88       # Ngưỡng tương quan dự đoán tối đa giữa các model (< 0.88)
marginal_gain_threshold = 0.015  # Tăng trưởng Rank IC tối thiểu của Ensemble (+1.5%)
wf_consistency_threshold = 0.60  # Tỷ lệ fold vượt trội của Ensemble (>= 60%)
complexity_penalty = 0.05        # Hệ số phạt độ phức tạp mô hình
random_state = 42
enabled_models = ["lightgbm", "xgboost", "catboost"]
```

| Khóa | Mặc định | Ràng buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `models.tournament_trials` | `30` | `>= 1` | Số lượt thử ngẫu nhiên cho mỗi model ở vòng đấu loại. |
| `models.tuning_trials` | `50` | `>= 1` | Số trials tối ưu hóa siêu tham số sâu bằng Optuna TPE. |
| `models.n_wf_folds` | `5` | `>= 2` | Số fold Expanding Window Walk-Forward trên tập Train. |
| `models.diversity_threshold` | `0.88` | `(0, 1]` | Ngưỡng tương quan dự đoán tối đa (Diversity $> 0.12$). |
| `models.marginal_gain_threshold` | `0.015` | `>= 0` | Biên độ tăng Rank IC tối thiểu để Ensemble được chấp thuận. |
| `models.wf_consistency_threshold` | `0.60` | `(0, 1]` | Tỷ lệ fold Walk-Forward mà Ensemble vượt hơn Single Best Model. |
| `models.complexity_penalty` | `0.05` | `>= 0` | Trọng số phạt độ phức tạp trong hàm mục tiêu Optuna. |
| `models.random_state` | `42` | int | Random seed đảm bảo tính tái lập kết quả. |
| `models.enabled_models` | trio | subset | Danh sách model tham gia (LightGBM, XGBoost, CatBoost). |


## Phase 4 — Convex Ensemble & Probability Calibration

Mục `[ensemble]` là tùy chọn; thiếu mục này thì `PipelineConfig.ensemble is None`.

```toml
[ensemble]
min_weight = 0.05               # Trọng số tối thiểu cho mỗi model trong ensemble
max_weight = 0.60               # Trọng số tối đa cho mỗi model trong ensemble
n_bootstrap = 200               # Số lần lấy mẫu bootstrap Cal-B
bootstrap_sample_ratio = 0.80   # Tỷ lệ mẫu Cal-B cho mỗi lần bootstrap
shrinkage_factor = 0.10         # Hệ số co cụm (shrinkage) về Equal Weights (1/K)
hurdle_ratio = 0.05             # Ngưỡng vượt trội tối thiểu (+5% IC Sharpe) so với baseline
max_ece_threshold = 0.05        # Ngưỡng sai số hiệu chuẩn tối đa (ECE < 0.05)
random_state = 42               # Random seed tái lập
```

| Khóa | Mặc định | Ràng buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `ensemble.min_weight` | `0.05` | `[0.0, 1.0]` | Cận dưới trọng số của mô hình trong SLSQP. |
| `ensemble.max_weight` | `0.60` | `(0.0, 1.0]` | Cận trên trọng số của mô hình trong SLSQP. |
| `ensemble.n_bootstrap` | `200` | `>= 10` | Số vòng lấy mẫu con Cal-B để ước lượng phân phối trọng số ổn định. |
| `ensemble.bootstrap_sample_ratio` | `0.80` | `(0.0, 1.0]` | Tỷ lệ quan sát được chọn mỗi lần bootstrap. |
| `ensemble.shrinkage_factor` | `0.10` | `[0.0, 1.0]` | Mức độ kéo trọng số trung vị về vector đẳng trọng $1/K$. |
| `ensemble.hurdle_ratio` | `0.05` | `>= 0` | Mức tăng trưởng tối thiểu $+5\%$ IC Sharpe (RULE-05) để nhận ensemble. |
| `ensemble.max_ece_threshold` | `0.05` | `(0.0, 1.0]` | Ngưỡng chấp nhận Expected Calibration Error sau Temperature Scaling. |
| `ensemble.random_state` | `42` | int | Random seed tái lập. |


## Phase 5 — ONNX Packaging & Friction-Aware Blind OOS Backtest

Mục `[packaging]` và `[backtest]` đều tùy chọn. `[packaging]` chỉ chạy khi đã có ensemble + calibration (cần `[models]` và `[ensemble]`); `[backtest]` chỉ chạy khi bundle đã được tạo.

```toml
[packaging]
precision_tol = 1e-4        # RULE-06: max|y_py - y_onnx| phải < ngưỡng này trên Val
drift_z_threshold = 3.0     # DistributionShiftWarning khi |z| > 3.0 (bỏ qua cột 0/1)
latency_budget_ms = 5.0
std_floor = 1e-8            # Sàn std khi tính z-score (cột hằng)
latency_repeats = 200

[backtest]
price_col = "Close"         # Cột giá trong X thô của OOS (không qua funnel)
fee_bps = 4.0               # Taker fee
slippage_bps = 1.0
top_pct = 0.20              # Vào lệnh khi phân vị tín hiệu ở top/bottom 20%
exit_pct = 0.30             # Chỉ thoát khi phân vị vượt 30% (Hysteresis Buffer), >= top_pct
rank_window = 2000          # Cửa sổ nhân quả để tính phân vị tín hiệu
ic_window = 500             # Cửa sổ Rank IC cuốn chiếu (ICIR) và kỳ sụt giảm
min_rank_ic = 0.035
min_icir = 0.40
min_net_sharpe = 1.20       # Sau phí + slippage (RULE-09)
max_mdd = 0.18
downside_quantile = 0.20    # Nến "stress": lợi suất thị trường trượt ic_window nến thuộc 20% thấp nhất
min_downside_ic = 0.0
```

- Đầu ra: `bundle/` (`preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, `features_schema.json`, `predictor.py`, `bundle_manifest.json`) và `backtest_report.json`.
- `PrecisionValidationError` chặn xuất bundle. Trượt ngưỡng nghiệm thu OOS **không** ném lỗi: `backtest_report.json` ghi `accepted=false` và `failed_gates`.
- Chi phí: $R_{net}=R_{gross}-\text{Turnover}\times(\text{fee}+\text{slip})$, $\text{Turnover}=\tfrac12\sum|\Delta w|$ (đúng `ARCHITECTURE.md` §3.4); với 1 tài sản, đảo vị thế $-1\to+1$ tính turnover $=1$.
- Annualization suy ra từ timestamp OOS (`số nến / số năm`).

