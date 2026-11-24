# Danh Mục Báo Cáo & Artifacts Đầu Ra (`OUTPUTS`)

Khi chạy lệnh `uv run quantautoml run --config config.toml --out runs/default`, pipeline sẽ tạo ra các báo cáo JSON, tập dữ liệu phân đoạn `.npz` và gói mô hình `bundle/` trong thư mục đích. Dưới đây là ý nghĩa và cách đọc từng file:

---

## 1. Cấu Trúc Thư Mục Kết Quả (`runs/default/`)

```
runs/default/
├── 00_run_manifest.json                     # Danh mục trạng thái & artifacts của toàn bộ 6 giai đoạn
├── config.toml                              # Bản sao cấu hình chính xác đã dùng cho lần chạy
├── 01_data/
│   ├── 01_ingestion_report.json             # Báo cáo nạp dữ liệu, đồng bộ UTC, ép kiểu float32 & suy diễn fwd_ret_*
│   ├── 02_task.json                         # Loại bài toán (binary/multiclass/unsupervised), danh sách lớp & tên cột thô
│   ├── 03_split_report.json                 # Báo cáo chia tập dữ liệu (Static 5-way hoặc OOF) & số dòng Purging/Embargo
│   └── segments/
│       ├── train.npz                        # Dữ liệu phân đoạn Train (X, y, timestamps, index)
│       ├── cal_a.npz                        # Dữ liệu phân đoạn Cal-A (SLSQP Ensemble Optimization)
│       ├── cal_b.npz                        # Dữ liệu phân đoạn Cal-B (Temperature Calibration)
│       ├── val.npz                          # Dữ liệu phân đoạn Validation (kiểm tra sai số số học ONNX < 1e-4)
│       └── oos.npz                          # Dữ liệu phân đoạn Out-of-Sample (chỉ đọc bởi OOSBacktestRunner)
├── 02_features/
│   ├── 01_funnel_report.json                # Nhật ký chi tiết 8 tầng lọc đặc trưng (F1 -> F8): số cột vào/ra & lý do loại
│   └── 02_selected_features.json            # Danh sách các đặc trưng tinh tuyển cuối cùng (Golden Feature Set)
├── 03_models/
│   ├── 01_tournament_report.json            # Kết quả đấu loại 30 trials x 3 mô hình (LightGBM, CatBoost, XGBoost)
│   ├── 02_decision_report.json              # Quyết định chọn Single Best Model hay Ensemble (Diversity, Marginal Gain, WF)
│   └── 03_tuning_report.json                # Kết quả tối ưu siêu tham số chuyên sâu bằng Optuna TPE + Hyperband
├── 04_ensemble/
│   ├── 01_ensemble_report.json              # Trọng số tối ưu SLSQP (Block Bootstrap + Shrinkage 1/K + OOB Hurdle 5%, ic_window_used)
│   └── 02_calibration_report.json           # Tham số nhiệt độ T > 0, ECE trước/sau hiệu chuẩn & xác nhận Rank Invariance
├── 05_packaging/
│   └── bundle/                              # Gói mô hình ONNX độc lập sẵn sàng triển khai Production
│       ├── bundle_manifest.json             # Danh mục file trong bundle & sai số max_abs_diff thực tế trên tập Val
│       ├── ensemble_meta.json               # Trọng số mô hình, tham số nhiệt độ T, ngưỡng drift_z_threshold
│       ├── features_schema.json             # Hợp đồng dữ liệu (input_features, model_features, train_mean, train_std)
│       ├── preprocessor.onnx                # Đồ thị ONNX tiền xử lý (Inf/NaN, bool fill, sinh cột is_nan_*, trích cột)
│       ├── model_0.onnx                     # Các mô hình GBDT đã biên dịch sang ONNX
│       └── predictor.py                     # Engine suy luận độc lập (~78 dòng thuần numpy + onnxruntime)
└── 06_backtest/
    ├── 01_backtest_report.json              # Báo cáo nghiệm thu mù OOS (Rank IC, ICIR, Gross/Net/Daily Sharpe, DSR, MDD, Gates)
    ├── 02_realworld_report.json             # (Tùy chọn Tier 2) Báo cáo mô phỏng khớp lệnh Open[t+1] qua backtesting.py
    ├── 03_regime_diagnostics.json           # (Sinh bởi `quantautoml diagnose`) Chẩn đoán read-only đa segment (close vs next_open)
    ├── backtest_report.html                 # (Tùy chọn Tier 2) Biểu đồ nến HTML tương tác
    └── trades.csv                           # (Tùy chọn Tier 2) Nhật ký giao dịch chi tiết
```

---

## 2. File Nào Lưu Các Đặc Trưng Cuối Cùng Được Chọn?

Có **3 file** giúp bạn tra cứu và kiểm toán các đặc trưng (features) được chọn:

1. **`selected_features.json` (Tra cứu nhanh danh sách cuối cùng):**
   Lưu mảng `"selected_features"` chứa tên các đặc trưng sống sót sau tầng lọc cuối cùng (F8 JMI).
   ```json
   {
     "selected_features": [
       "rvi",
       "close_vs_session_range",
       "vortex_diff",
       "kdjdk"
     ]
   }
   ```
2. **`bundle/features_schema.json` (Hợp đồng đầu vào & thống kê chuẩn hóa):**
   Lưu cả `input_features` (toàn bộ 824 cột thô đầu vào), `model_features` (các cột sau `preprocessor.onnx`), cùng vector `train_mean` và `train_std` được tính thuần túy trên tập `Train` để kiểm tra cảnh báo trôi phân phối (`DistributionShiftWarning` khi $|z| > 3.0$).
3. **`funnel_report.json` (Giải thích vì sao từng đặc trưng được giữ hoặc bị loại):**
   Ghi lại toàn bộ 8 bước lọc (`f1_triage` $\to$ `f2a_variance` $\to$ `f2b_stationarity` $\to$ `f3_ic_stability` $\to$ `f4_collinearity` $\to$ `f5_clustering` $\to$ `f6_regime_synergy` $\to$ `f8_jmi`). Với mỗi cột bị loại trong trường `"dropped"`, file ghi rõ lý do định lượng (ví dụ: `non-stationary: lag1_autocorr=0.99994 > 0.995 and temporal_drift=2.270 > 1.5`, `icir |0.154| <= icir_min 0.3`, `dcor=0.9850 > 0.85 with kept 'vortex_diff'`, hoặc `cluster 11: represented by 'close_vs_session_range'`).

---

## 3. Cách Đọc Báo Cáo Nghiệm Thu `backtest_report.json`

File `backtest_report.json` tổng hợp toàn bộ chỉ số kiểm định mù trên tập `OOS` và kết quả kiểm tra **6 Hard Gates**:

| Trường trong `backtest_report.json` | Ý nghĩa định lượng | Điều kiện PASS Gate |
| :--- | :--- | :--- |
| `rank_ic` | Hệ số tương quan hạng Spearman giữa điểm dự báo và nhãn trên toàn tập OOS. | `> min_rank_ic` (`0.035`) |
| `icir` | Tỷ số $\text{Mean}(\text{Rolling IC}) / \text{Std}(\text{Rolling IC})$ với cửa sổ `ic_window` nến. | `> min_icir` (`0.40`) |
| `gross_sharpe` | Sharpe thường niên hóa trước khi trừ phí giao dịch và trượt giá. | *(Tham khảo đối chiếu)* |
| `net_sharpe` | Sharpe thường niên hóa **sau khi trừ phí sàn (`fee_bps`) + trượt giá (`slippage_bps`)**. | `> min_net_sharpe` (`1.20`) |
| `max_drawdown` | Mức sụt giảm vốn tối đa từ đỉnh (MDD) trên đường cong vốn ròng (`cumprod(1 + R_net)`). | `< max_mdd` (`0.18` tức 18%) |
| `downside_ic` | Rank IC tính riêng trên các cây nến thị trường sụt giảm mạnh nhất (`downside_quantile` 20% thấp nhất). | `>= min_downside_ic` (`0.0`) |
| `mean_turnover` | Vòng quay danh mục trung bình trên mỗi cây nến: $\frac{1}{2}|\text{pos}_t - \text{pos}_{t-1}|$. | *(Đo tần suất đảo lệnh)* |
| `latency_ms` | Độ trễ suy luận trung vị (median) cho 1 dòng dữ liệu qua `predictor.py`. | `< latency_budget_ms` (`5.0` ms) |
| `periods_per_year` | Số cây nến trung bình mỗi năm (suy tự động từ timestamps thực tế của tập OOS). | *(Dùng để thường niên hóa Sharpe)* |
| `n_overnight_gaps` | Số khoảng nghỉ qua đêm / cuối tuần (bước nhảy thời gian $> 3\times$ bước nến trung vị). | *(Kiểm tra tính liên tục chuỗi)* |
| `accepted` | `true` nếu và chỉ nếu **tất cả 6 gates** trong `"gates"` đều đạt `true`. | `failed_gates == []` |
