# Architectural & Quantitative Trade-offs — `v1.1-fix1`

**Feature/Topic:** Quant Audit Remediation (`v1.1-fix1`)
**Status:** Draft

---

## 1. True OOF Predictions trong `OOF_ASSISTED` Mode (Audit #1)

### Các phương án cân nhắc:
- **Phương án A:** Bỏ hẳn Phase 4 (Ensemble & Calibration) khi $N < 10{,}000$.
  - *Ưu điểm:* Không có rủi ro leakage, code đơn giản.
  - *Nhược điểm:* Mất luôn khả năng hiệu chuẩn xác suất và kết hợp mô hình trên các dataset nhỏ (vd khung 1H/4H/1D).
- **Phương án B (Chọn):** Fit lại 2 bản sao nhẹ của `best_params` trên `oof_folds[-2][0]` (predict lên `oof_folds[-2][1]` cho `Cal-A`) và `oof_folds[-1][0]` (predict lên `oof_folds[-1][1]` cho `Cal-B`), sau đó mới refit model chính trên toàn bộ `split.train`.
  - *Ưu điểm:* 100% out-of-sample (có purging `horizon_h` sẵn từ `AdaptiveTimeSeriesSplitter`), chi phí tính toán cực thấp (chỉ thêm 2 lần `fit` GBDT trên $N < 10{,}000$, tốn $< 1\text{s}$), giữ nguyên kiến trúc ONNX bundle.
  - *Nhược điểm:* Thêm một helper nhỏ tính OOF predictions trong `pipeline.py` / `tuner.py`.

### Kết luận:
Chọn **Phương án B** vì loại bỏ triệt để in-sample leakage với chi phí gần như bằng 0.

---

## 2. Thứ tự `Cal-A` và `Cal-B` (Audit #2)

### Các phương án cân nhắc:
- **Phương án A (Hiện tại):** Fit Ensemble trên `Cal-B` (70–80%), fit Calibration trên `Cal-A` (60–70%).
  - *Nhược điểm:* Vi phạm tính nhân quả thời gian (dùng trọng số học từ tương lai áp lên quá khứ); Temperature $T$ bị lạc hậu hơn so với `Val`/`OOS`.
- **Phương án B (Chọn):** Fit Ensemble trên `Cal-A` (60–70%), fit Calibration trên `Cal-B` (70–80%).
  - *Ưu điểm:* Thuần nhân quả thời gian (`Train → Cal-A → Cal-B → Val → OOS`); `Cal-B` gần `Val`/`OOS` nhất nên xác suất hiệu chuẩn bám sát phân phối gần nhất; không cần đổi tỷ lệ split trong `splitter.py`.
  - *Nhược điểm:* Cần cập nhật comment/doc và 1 số test đang gắn nhãn ngược tên `cal_a`/`cal_b`.

### Kết luận:
Chọn **Phương án B**.

---

## 3. Thay thế `unique_ratio` trong F2 `MultiMetricVarianceFilter` (Audit #6)

### Các phương án cân nhắc:
- **Phương án A:** Giữ `unique_ratio` nhưng giảm ngưỡng mặc định từ `0.01` xuống `0.0001`.
  - *Nhược điểm:* Vẫn phụ thuộc tuyến tính nghịch vào $N$ — khi $N = 500{,}000$ lại tiếp tục loại nhầm `day_of_week`, `hour`, `session_progress`.
- **Phương án B (Chọn):** Dùng **`max_top_freq_ratio`** (tỷ lệ xuất hiện của giá trị phổ biến nhất, mặc định `0.995`) làm cổng loại cột gần hằng số, đồng thời đặt `min_unique_ratio = 0.0` (tắt mặc định nhưng vẫn giữ tham số để tương thích ngược) và `max_peak_var_ratio: float | None = None` (không bật dead gate `50.0`).
  - *Ưu điểm:* Hoàn toàn bất biến theo cỡ mẫu $N$; giữ lại 100% biến rời rạc/chu kỳ/lượng tử hóa hợp lệ (`session_progress`, `body`, `aroon_up`, `cluster_kmeans_*`); vẫn lọc sạch các cột gần như toàn một giá trị (vd 99.8% bằng 0).
  - *Nhược điểm:* Các cột binary cực lệch (vd sự kiện hiếm xuất hiện 0.2% thời gian) có thể bị loại nếu không miễn trừ `n_unique == 2` — do đó **chỉ áp `max_top_freq_ratio` hoặc cho phép cấu hình ngưỡng linh hoạt** (và vẫn loại cứng `n_unique < 2`).

### Kết luận:
Chọn **Phương án B** (kiểm tra `top_freq_ratio > max_top_freq_ratio` trên mọi cột có `n_unique >= 2`, với mặc định `0.995` — hoặc `0.999` nếu muốn bảo vệ sự kiện hiếm; chọn `0.995` cho cân bằng tốt giữa độ ổn định mẫu và lọc rác).

---

## 4. P-value cho BH-FDR trong F3+7 (Audit #3 — Phase 2)

### Các phương án cân nhắc:
- **Phương án A:** `spearmanr` trên toàn bộ mẫu Train (hiện tại).
  - *Nhược điểm:* Giả định i.i.d. sai hoàn toàn với overlapping forward returns, gây false positive hàng loạt.
- **Phương án B (Chọn):** 1-sample t-test trên chuỗi `rolling_ic` không chồng lấn (`window = rolling_window`).
  - *Ưu điểm:* Không thêm dependency mới, tốc độ tính toán gần như tức thì (tận dụng luôn ma trận `roll` đã tính sẵn ở F3+7), giảm mạnh ảnh hưởng autocorrelation ngắn hạn.
  - *Nhược điểm:* Khi Train ngắn ($M < 10$ cửa sổ), lực kiểm định (statistical power) thấp hơn — chấp nhận được vì bản chất mẫu nhỏ khó khẳng định alpha có ý nghĩa thống kê.

---

## 5. Bar-level Sharpe vs Daily-Aggregated Net Sharpe (Audit #8 — Phase 3)

### Các phương án cân nhắc:
- **Phương án A:** Chỉ dùng Bar-level Sharpe $\times \sqrt{\text{periods\_per\_year}}$ (hiện tại).
  - *Nhược điểm:* Méo mạnh trên intraday 5m có Hysteresis (serial correlation cao) và overnight gap.
- **Phương án B (Chọn):** Báo cáo **cả hai** (`net_sharpe` bar-level để tương thích ngược và `daily_net_sharpe` chuẩn $\times\sqrt{252}$ + `deflated_sharpe_ratio`), cho phép cấu hình dùng `daily_net_sharpe` làm cổng nghiệm thu chính khi OOS kéo dài $\ge 20$ ngày giao dịch.
  - *Ưu điểm:* Chuẩn ngành quỹ định lượng, không phá vỡ test unit tổng hợp ngắn ngày.
