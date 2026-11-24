# Project / Feature Scope — `v1.1-fix1` (Quant Audit Remediation)

**Feature/Topic:** Quant Research Audit Remediation (`v1.1-fix1`)
**Branch:** `develop/v1.1`
**Date:** 2026-11-24
**Status:** Draft (Pending User Approval)

---

## 1. Objectives & Goals

Khắc phục toàn bộ **8 điểm yếu và lỗi kỹ thuật định lượng** được phát hiện từ đợt Quant Research Audit trên nền tảng `v1.0`:
1. **Triệt tiêu rò rỉ ngầm và vi phạm tính nhân quả thời gian** ở chế độ `OOF_ASSISTED` và thứ tự hiệu chuẩn `Cal-A`/`Cal-B`.
2. **Nâng cao độ chặt chẽ thống kê** của phễu lọc tính năng (F2, F3+7), hàm mục tiêu Walk-Forward CV (Tournament/Tuner), và tính nhất quán điểm số (Score) giữa huấn luyện và suy luận ONNX.
3. **Tăng tính thực tế của nghiệm thu OOS** bằng Block Bootstrap, Daily-aggregated Net Sharpe ($\times\sqrt{252}$), Deflated Sharpe Ratio (DSR), và đồng bộ giá khớp lệnh `Open[t+1]`.

---

## 2. In-Scope (Must Have)

### Phase 1 — Ingestion Date Slicing, Causality, Leakage & Feature Filter Rigor *(Done)*
- [x] **Date Range Slicing (`[data]` `date_from` / `date_to`):** Cắt dataset theo khoảng thời gian `[date_from, date_to]` ngay đầu bước `DataIngestion` (trước khi encode feature, join, suy luận nhãn hay tính `fwd_ret_h`), báo cáo rõ trong `01_ingestion_report.json`.
- [x] **Audit #1 (`OOF_ASSISTED` Leakage):** Sinh true out-of-fold predictions cho Phase 4 (Ensemble & Calibration) khi $N < 10{,}000$, tuyệt đối không dùng `refitted_model.predict()` trên tập con của `split.train`.
- [x] **Audit #2 (Temporal Causality `Cal-A` ↔ `Cal-B`):** Đổi thứ tự nhân quả chuẩn — fit trọng số SLSQP Ensemble trên `Cal-A` (60–70%), sau đó áp trọng số đó lên `Cal-B` (70–80%) để fit Temperature Scaling gần `Val`/`OOS` hơn.
- [x] **Audit #6 (F2 `MultiMetricVarianceFilter`):** Thay cổng `unique_ratio = n_unique / N` (phụ thuộc cỡ mẫu $N$, loại nhầm biến rời rạc/chu kỳ) bằng ngưỡng bất biến theo $N$ (`max_top_freq_ratio`, mặc định `0.995`); giữ tương thích ngược cho `min_unique_ratio` (mặc định `0.0`); hiệu chuẩn lại hoặc vô hiệu hóa dead gate `max_peak_var_ratio = 50`.

### Phase 2 — Statistical Rigor & Objective Alignment *(Done)*
- [x] **Audit #3 (F3+7 FDR & Monotonicity):** Tính p-value BH-FDR trên chuỗi `rolling_ic` không chồng lấn (tránh t-stat ảo do overlapping returns); đổi cổng monotonicity thành kiểm tra suy giảm từ `target_horizon` trở đi thay vì ép đỉnh tại $h=1$.
- [x] **Audit #4 (CV `IC Sharpe` & HyperbandPruner):** Tính `IC Sharpe` từ chuỗi rolling IC nội bộ trong mỗi fold (tăng số điểm quan sát từ 5 lên ~30–100); đồng bộ thang đo giữa `trial.report()` và giá trị trả về của `objective()`.
- [x] **Audit #5 (Multiclass Score & Ensemble Scaling):** Dùng đúng vector `classes` thật xuyên suốt Tournament → Tuner → Cal-A/Cal-B → `predictor.py`; chuẩn hóa thang đo (logit/calibrated probs) trước khi trộn trọng số để khớp giữa bước tối ưu SLSQP và bước suy luận ONNX.

### Phase 3 — Time-Series Bootstrap & Production Backtest Realism *(Planned)*
- [ ] **Audit #7 (Block Bootstrap & OOF Hurdle):** Chuyển `BootstrapShrinkageEngine` sang Moving Block Bootstrap (block length $\ge 2h$); đánh giá cổng Hurdle `+5%` (`RULE-05`) trên dự đoán OOF nội bộ thay vì in-sample.
- [ ] **Audit #8 (Daily Sharpe, DSR, Execution Alignment & Multi-Window OOS):** Thêm Daily-aggregated Net Sharpe ($\times\sqrt{252}$) và Deflated Sharpe Ratio (DSR); hỗ trợ tùy chọn khớp lệnh `Open[t+1]` cho Tier 1 để đồng bộ Tier 2; bổ sung chế độ Multi-window Walk-Forward OOS.

---

## 3. Out-of-Scope (Non-Goals)

- Không thay đổi mô hình GBDT Trio (LightGBM, XGBoost, CatBoost) hay thêm Deep Learning / Transformer.
- Không phá vỡ hợp đồng `predictor.py` độc lập (`RULE-07`: chỉ `numpy` + `onnxruntime` + stdlib) hoặc nới lỏng ngưỡng `max|y_py - y_onnx| < 1e-4` (`RULE-06`).
- Không xây dựng hệ thống thực thi lệnh live (broker API / OMS / EMS).
- Không thay đổi cấu trúc thư mục output theo phase (`01_data/` .. `06_backtest/`, `00_run_manifest.json`) vừa triển khai ở `develop/v1.1`.

---

## 4. Constraints & Assumptions

- **Hard Gates bắt buộc tuân thủ:** Toàn bộ `RULE-01` đến `RULE-10` trong `AGENTS.md` / `docs/.ai/RULES.md`.
- **Precision & Performance:** Mọi tensor số học dùng `np.float32`; độ trễ suy luận 1 dòng $< 5.0\text{ ms}$; sai số ONNX $< 10^{-4}$.
- **Testing & Compatibility:** TDD bắt buộc (RED → GREEN → REFACTOR); toàn bộ 619+ unit & integration tests hiện có phải PASS 100% sau mỗi task; các tham số config mới phải có giá trị mặc định an toàn để không làm gãy file TOML cũ (ngoại trừ các trường đã ghi rõ nâng cấp mặc định).
