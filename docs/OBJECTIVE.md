# Project Objective — Quant AutoML Pipeline (`quantautoml`)

> **Tài liệu định hướng mục tiêu cấp dự án (Project Charter & Quantitative Objectives)**  
> Phác thảo bài toán cốt lõi, 5 trụ cột thiết kế, tiêu chí nghiệm thu định lượng (Hard Gates), lộ trình tiến hóa (`v1.0` $\to$ `v1.1`) và ranh giới phạm vi của **Quant AutoML Pipeline**.

---

## 1. Tuyên bố Mục tiêu Cốt lõi (North Star Objective)

Thị trường tài chính chuỗi thời gian có tỷ lệ tín hiệu trên nhiễu cực thấp (*low signal-to-noise ratio*), phân phối không dừng (*non-stationary*), tự tương quan cao theo phiên/kỳ hạn và chi phí giao dịch thực tế ăn mòn nhanh các chiến lược quay vòng nhiều.

**Mục tiêu tối hậu của `quantautoml`** là xây dựng một **hệ thống AutoML định lượng tự động khép kín (End-to-End Quantitative AutoML Pipeline)** cho dữ liệu chuỗi thời gian dạng nến (*bar-level single-asset*: 1m, 5m, 15m, 1h, 1d), có khả năng biến đổi dữ liệu thô (`features` + `labels`) thành **mô hình ONNX độc lập sẵn sàng vận hành Production**, sao cho:

1. **Chống rò rỉ dữ liệu tuyệt đối (*Zero Look-Ahead Leakage*):** Không bất kỳ thông tin tương lai nào lọt vào chọn đặc trưng, huấn luyện, tinh chỉnh siêu tham số, trộn trọng số hay hiệu chuẩn xác suất.
2. **Tối ưu đúng bản chất xếp hạng alpha (*Rank-Consistent Alpha*):** Tìm ra tín hiệu bảo toàn thứ hạng dự báo (`Spearman Rank IC` & `IC Sharpe`) ổn định qua thời gian, thay vì tối ưu sai lệch bình phương (`MSE`) hay độ chính xác thô (`Accuracy`).
3. **Sống sót sau chi phí ma sát (*Net Alpha After Friction*):** Mọi chiến lược xuất xưởng bắt buộc vượt qua kiểm định mù trên tập `OOS` **sau khi trừ phí giao dịch, trượt giá và kiểm soát vòng quay danh mục**.
4. **Thực thi độc lập, độ trễ thấp, tái lập 100%:** Đóng gói thành ONNX Model Bundle thuần `numpy` + `onnxruntime` (`< 5.0 ms/row`), kiểm định sai số số học `float32` (`< 10^-4`) và có báo cáo kiểm toán đầy đủ ở từng giai đoạn.

---

## 2. Năm Trụ cột Mục tiêu Kỹ thuật & Định lượng

| Trụ cột | Bản chất mục tiêu | Cơ chế hiện thực hóa trong hệ thống | Quy tắc bất biến liên quan |
| :--- | :--- | :--- | :--- |
| **1. Cô lập Thời gian & Chống Rò rỉ** | Đảm bảo mọi đánh giá phản ánh đúng năng lực tổng quát hóa ngoài mẫu theo chiều thời gian nhân quả. | Không xáo trộn (`shuffle=False`), Purging & Embargo `horizon_h` nến giữa mọi ranh giới; chia thích ứng (`STATIC_5WAY` 60/10/10/10/10 khi $N \ge 10{,}000$ hoặc `OOF_ASSISTED` 80/10/10 dùng true OOF folds khi $500 \le N < 10{,}000$); thứ tự nhân quả `Train → Cal-A (SLSQP) → Cal-B (Temperature) → Val → OOS`. | [`RULE-01`](.ai/RULES.md) |
| **2. Thanh lọc Đặc trưng Đa tầng Nghiêm ngặt** | Loại bỏ đặc trưng rác, không dừng, trùng lặp phi tuyến và nhiễu ngẫu nhiên (*data snooping bias*) từ hàng trăm chỉ báo kỹ thuật. | Phễu 8 bước chỉ `fit()` trên `Train`: `F1` (NaN/Inf & discrete-aware MI) $\to$ `F2a` (Variance / `top_freq_ratio`) $\to$ `F2b` (Stationarity: `lag1_autocorr` & `temporal_drift`) $\to$ `F3+7` (Rolling ICIR, BH-FDR $\le 0.05$, Term Structure) $\to$ `F4` (Fast dCor) $\to$ `F5` (IC-Behavior Clustering) $\to$ `F6` (Regime MI & Synergy) $\to$ `F8` (JMI Golden Set). | [`RULE-01`](.ai/RULES.md), [`RULE-08`](.ai/RULES.md) |
| **3. Tối ưu Hợp lý & Chống Phình Mô hình (*Earn Its Place*)** | Chỉ chấp nhận độ phức tạp mô hình/ensemble khi chứng minh được lợi ích thống kê vượt ngưỡng trên dữ liệu ngoài mẫu huấn luyện. | Đấu loại công bằng Modern GBDT Trio (`LightGBM`, `XGBoost`, `CatBoost`) trên 5-fold Expanding Walk-Forward (`RandomSampler` 30 trials/model); `ModelDecisionEngine` (Diversity `< 0.88`, Marginal Gain `>= 1.5%`, WF Consistency `>= 60%`); `DeepOptunaTuner` (TPE Multivariate + Hyperband + phạt complexity + biên regularization co giãn theo $N_{\text{train}}$); `SmoothSLSQPOptimizer` ($C^2$ Pearson trên dữ liệu rank-transform) + Moving Block Bootstrap ($N=200$) + Shrinkage $1/K$ + OOB Hurdle Gate ($+5\%$ IC Sharpe); `TemperatureCalibrator` ($\text{ECE} < 0.05$). | [`RULE-02`](.ai/RULES.md), [`RULE-03`](.ai/RULES.md), [`RULE-04`](.ai/RULES.md), [`RULE-05`](.ai/RULES.md) |
| **4. Đóng gói Chuẩn Production Tách rời** | Tách biệt hoàn toàn hạ tầng huấn luyện nặng khỏi engine suy luận vận hành thực tế. | Xuất `bundle/` gồm `preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, `features_schema.json`, `bundle_manifest.json` và `predictor.py` (~78 dòng chỉ dùng `numpy`, `onnxruntime`, stdlib); chặn xuất xưởng nếu $\max|\hat{y}_{\text{py}} - \hat{y}_{\text{onnx}}| \ge 10^{-4}$ trên `Val`; cảnh báo `DistributionShiftWarning` khi $|z| > 3.0$. | [`RULE-06`](.ai/RULES.md), [`RULE-07`](.ai/RULES.md) |
| **5. Nghiệm thu Hai Tầng Sát Thực tế** | Loại bỏ chiến lược "lãi ảo trên giấy" nhưng lỗ thực tế do tần suất đảo lệnh và trượt giá khớp lệnh. | **Tier 1** (NumPy vectorized): `FrictionCostModel` (`fee_bps` + `slippage_bps`), `HysteresisBuffer` (dải đệm phân vị nhân quả), hỗ trợ `close` hoặc `next_open` (`Open[t+1] → Open[t+2]`), báo cáo `daily_net_sharpe` ($\times\sqrt{252}$) & `deflated_sharpe_ratio`. **Tier 2** (`backtesting.py`): mô phỏng khớp lệnh `Open[t+1]` với quản lý vốn/ký quỹ và xuất nhật ký `trades.csv` + biểu đồ HTML. | [`RULE-09`](.ai/RULES.md) |

---

## 3. Tiêu chí Thành công & Cổng Nghiệm thu Định lượng (Acceptance Gates)

Một lần chạy pipeline chỉ được đánh dấu **`accepted = true`** (sẵn sàng cân nhắc triển khai) khi thỏa mãn **đồng thời tất cả 6 cổng OOS** và các hợp đồng chất lượng kỹ thuật sau:

| Hạng mục | Chỉ số đo lường | Ngưỡng mặc định (`config.py`) | Ý nghĩa định lượng |
| :--- | :--- | :--- | :--- |
| **OOS Gate 1** | `rank_ic` | `> 0.035` | Sức mạnh xếp hạng tổng thể trên toàn tập `OOS` |
| **OOS Gate 2** | `icir` | `> 0.40` | Độ ổn định của Spearman Rank IC qua các cửa sổ trượt `backtest.ic_window` (`500` nến) |
| **OOS Gate 3** | `net_sharpe` | `> 1.20` | Sharpe ròng thường niên hóa sau phí giao dịch & trượt giá (kèm đối chiếu `daily_net_sharpe` và `deflated_sharpe_ratio`) |
| **OOS Gate 4** | `max_drawdown` | `< 0.18` ($18\%$) | Mức sụt giảm vốn tối đa trên đường cong vốn ròng |
| **OOS Gate 5** | `downside_ic` | `>= 0.0` (trên $\ge 30$ stress bars) | Khả năng giữ nguyên hướng tín hiệu trong 20% giai đoạn thị trường sụt giảm mạnh nhất |
| **OOS Gate 6** | `latency_ms` | `< 5.0 ms` (median / 200 calls) | Độ trễ suy luận 1 dòng của `bundle/predictor.py` trên CPU |
| **Packaging Gate** | `max_abs_diff` | `< 1e-4` (`np.float32`) | Sai số số học tối đa giữa Python pipeline và `predictor.py` ONNX trên tập `Val` (`PrecisionValidationError` nếu vi phạm) |
| **Feature & Cal Gates** | `FDR` / `ECE` | `FDR <= 0.05` / `ECE < 0.05` | Kiểm soát phát hiện sai Benjamini-Hochberg (F3+7) và sai số hiệu chuẩn xác suất (Cal-B) |

---

## 4. Lộ trình Tiến hóa Mục tiêu (`v1.0` $\to$ `v1.1`)

### 4.1. Nền tảng đã hoàn thành (`v1.0` — Done)
- Hoàn thành toàn bộ **5 Phase** trong [`docs/plan/PHASES.md`](plan/PHASES.md) từ `DataIngestion`, `AdaptiveTimeSeriesSplitter`, phễu lọc đặc trưng không rò rỉ, `ModelTournament` + `DeepOptunaTuner`, `BootstrapShrinkageEngine` + `TemperatureCalibrator`, `ONNXBundleExporter` tới `OOSBacktestRunner` (Tier 1 & Tier 2).

### 4.2. Chuỗi Nâng cấp Chất lượng Định lượng (`develop/v1.1`)
Dựa trên kiểm toán định lượng (*Quant Research Audit*) và phân tích thực nghiệm trên dữ liệu intraday 5 phút (`tb5_label`):

1. **`v1.1-fix1` *(Done)* — Khắc phục Kiểm toán Định lượng:**
   - Cắt khoảng thời gian `date_from`/`date_to` ngay đầu Ingestion; loại bỏ rò rỉ in-sample ở chế độ `OOF_ASSISTED`; đảo đúng thứ tự nhân quả `Cal-A (SLSQP) → Cal-B (Temperature)`.
   - Sửa `F2` dùng `max_top_freq_ratio` bất biến theo cỡ mẫu; sửa `F3+7` dùng 1-sample t-test trên `rolling_ic` không chồng lấn cho BH-FDR và kiểm tra suy giảm từ `target_horizon` trở đi; đồng bộ vector `classes` thật cho bài toán đa lớp; dùng Moving Block Bootstrap + OOB Hurdle; bổ sung `daily_net_sharpe`, `deflated_sharpe_ratio` và tùy chọn khớp lệnh `execution_price = "next_open"`.
2. **`f2b_stationarity` *(Done)* — Bộ lọc Tính dừng:**
   - Thêm tầng `f2b_stationarity` (`lag1_autocorr` & `temporal_drift`) loại bỏ các đặc trưng giá/tích lũy không dừng gây sụp đổ khi thị trường dịch chuyển mặt bằng giá.
3. **`v1.1-fix2` *(Phase 1 Done, Phase 2 Planned)* — Chống Overfit GBDT, Tăng tốc GPU & Kiểm định Đa cửa sổ:**
   - **Phase 1 *(Done)*:** Co giãn động `min_child_samples` / `min_child_weight` theo $N_{\text{train}}$, khóa biên độ sâu/leaves/LR cân bằng và bật tăng tốc GPU CUDA cho XGBoost & CatBoost (`models.device`).
   - **Phase 2 *(Planned)*:** Tùy chọn refit trên `Train + Cal` và `Multi-Window Walk-Forward OOS` (`backtest/walkforward.py`) để đánh giá độ bền qua nhiều giai đoạn OOS nối tiếp.
4. **`v1.1-optimize3` *(In Progress — Phase 1 Active)* — Căn chỉnh Cửa sổ IC, Kiểm soát Vòng quay & Chẩn đoán Đa Regime:**
   - **Phase 1:** Cửa sổ rolling IC thích ứng (lên tới `500` nến thay vì bị kẹt ở trần `50`/`63`) để triệt tiêu nghịch lý Simpson (*Simpson's Paradox*) trong `ModelTournament`, `DeepOptunaTuner` và `BootstrapShrinkageEngine`.
   - **Phase 2:** Làm mượt tín hiệu nhân quả (`signal_ema_span`) và thời gian nắm giữ tối thiểu (`min_hold_bars`) trong `HysteresisBuffer` để giảm hao hụt Sharpe do chi phí giao dịch và vòng quay cao.
   - **Phase 3:** Tiện ích chẩn đoán đọc-chỉ (*read-only*) đa segment/regime (`Cal-A`, `Cal-B`, `Val`, `OOS`) và kiểm chứng thực nghiệm với `execution_price = "next_open"` + Tier 2 (`[realworld]`).

---

## 5. Ranh giới Phạm vi (Non-Goals)

Các hạng mục sau **không** thuộc mục tiêu của `quantautoml` nhằm giữ kiến trúc gọn nhẹ, chuyên sâu và dễ kiểm toán:
- **Live Order Management / Broker Execution:** Không kết nối trực tiếp API sàn giao dịch hay quản lý trạng thái lệnh live (giao tiếp qua `bundle/predictor.py` ở hệ thống thực thi riêng).
- **Tick-level / Order-book HFT:** Thiết kế cho dữ liệu nến (bar-level), không xử lý dữ liệu tick microsecond.
- **Multi-Asset Cross-Sectional Portfolio:** Kiến trúc hiện tại xử lý chuỗi thời gian đơn tài sản (*single-asset time series*).
- **Deep Learning / LLM:** Không đưa mạng nơ-ron sâu vào model pool mặc định để bảo đảm dung lượng bundle nhỏ, huấn luyện ổn định trên tabular quant và latency suy luận `< 5 ms`.
- **Web UI & Streaming Online Retraining:** Giao tiếp qua CLI (`quantautoml run`), file cấu hình TOML, báo cáo JSON/HTML tĩnh và huấn luyện lại theo chu kỳ batch.

---

## 6. Tài liệu Tham chiếu Liên quan

- Tổng quan & Quy tắc cốt lõi: [`OVERVIEW.md`](OVERVIEW.md), [`RULES.md`](.ai/RULES.md)
- Kiến trúc & Cấu trúc mã nguồn: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`STRUCTURE.md`](STRUCTURE.md)
- Đặc tả Phễu lọc Đặc trưng & Cấu hình: [`FEATURES.md`](FEATURES.md), [`CONFIGURATION.md`](CONFIGURATION.md), [`OUTPUTS.md`](OUTPUTS.md)
- Kế hoạch triển khai: [`plan/SCOPE.md`](plan/SCOPE.md), [`plan/v1.1-fix1/SCOPE.md`](plan/v1.1-fix1/SCOPE.md), [`plan/v1.1-fix2/SCOPE.md`](plan/v1.1-fix2/SCOPE.md), [`plan/v1.1-optimize3/SCOPE.md`](plan/v1.1-optimize3/SCOPE.md)
