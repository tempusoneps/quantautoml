# Directory & File Structure — `v1.1-fix1`

Tài liệu này ánh xạ chính xác các file sẽ được sửa đổi hoặc bổ sung trong `v1.1-fix1`, tuân thủ nguyên tắc đơn trách nhiệm (SRP) và ranh giới phụ thuộc một chiều trong `docs/STRUCTURE.md`.

---

## Directory Tree (Chỉ liệt kê các vùng bị tác động)

```text
quantautoml/
├── src/quantautoml/
│   ├── common/
│   │   ├── config.py                        # [P1, P2, P3] Thêm date_from/date_to (DataConfig), max_top_freq_ratio, tùy chọn FDR/monotonic, block_size, daily_sharpe/execution_price
│   │   └── types.py                         # [P1] Thêm date_from/date_to vào IngestionReport
│   ├── data/
│   │   └── ingestion.py                     # [P1] Cắt [date_from, date_to] ngay sau khi chuẩn hóa UTC timestamp, trước encode/join/fwd_ret_h
│   ├── features/
│   │   ├── variance.py                      # [P1] Audit #6: max_top_freq_ratio (bất biến theo N), sửa dead gate peak_var_ratio
│   │   └── ic_stability.py                  # [P2] Audit #3: rolling-IC t-test cho BH-FDR, monotonicity sau target_horizon
│   ├── models/
│   │   ├── tournament.py                    # [P2] Audit #4: rolling-IC Sharpe nội bộ fold + unified score helper
│   │   └── tuner.py                         # [P1, P2] Audit #1: sinh true OOF predictions; Audit #4: đồng bộ HyperbandPruner & rolling-IC Sharpe
│   ├── ensemble/
│   │   ├── slsqp.py                         # [P2] Audit #5: tùy chọn tối ưu trên thang đo khớp predictor
│   │   ├── bootstrap.py                     # [P3] Audit #7: Moving Block Bootstrap + OOF Hurdle gate
│   │   └── calibration.py                   # [P1] Audit #2: ghi rõ docstring/contract fit trên Cal-B (sau Ensemble trên Cal-A)
│   ├── backtest/
│   │   ├── evaluator.py                     # [P3] Audit #8: Daily Net Sharpe (sqrt(252)) & Deflated Sharpe Ratio (DSR)
│   │   ├── runner.py                        # [P3] Audit #8: tùy chọn khớp Open[t+1] cho Tier 1
│   │   └── walkforward.py                   # [P3] Audit #8 (New): Multi-window Walk-Forward OOS runner
│   └── pipeline.py                          # [P1, P2, P3] Audit #1 (true OOF Phase 4), Audit #2 (Cal-A -> Cal-B), Audit #5 (classes thật)
└── tests/
    ├── unit/
    │   ├── test_config.py                   # [P1] Test config defaults mới cho DataConfig & FeaturesConfig
    │   ├── test_ingestion.py                # [P1] Test cắt khoảng date_from/date_to, edge cases & zero look-ahead ngoài cửa sổ
    │   ├── test_f2_variance.py              # [P1] Test max_top_freq_ratio & biến rời rạc trên N lớn
    │   ├── test_tuner.py                    # [P1, P2] Test true OOF predictions & aligned pruner
    │   ├── test_f3_ic_stability.py          # [P2] Test non-overlapping FDR & post-target monotonicity
    │   ├── test_bootstrap.py                # [P3] Test Block Bootstrap & OOF hurdle
    │   └── test_evaluator.py                # [P3] Test Daily Net Sharpe & DSR
    └── integration/
        ├── test_zero_leakage_ensemble.py    # [P1] Audit #1 & #2: chứng minh OOF_ASSISTED không leak in-sample & Cal-A độc lập Cal-B
        └── test_zero_leakage_packaging.py   # [P1, P2] Xác nhận ONNX precision < 1e-4 sau thay đổi Cal-A/Cal-B & classes
```

---

## Component & File Responsibilities (Phase 1 Active)

| File / Module | Trách Nhiệm Trong Phase 1 | Ký Hiệu / Thay Đổi Chính | Phụ Thuộc Hợp Lệ |
| :--- | :--- | :--- | :--- |
| `src/quantautoml/data/ingestion.py` | Áp bộ lọc `[date_from, date_to]` lên `features` và `labels` ngay sau `_read_frame()` (trước encode/join/`fwd_ret_h`) và ghi lại vào `IngestionReport`. | `DataIngestion.load()`, `_slice_date_range()` | `polars`, `numpy`, `common` |
| `src/quantautoml/common/types.py` | Bổ sung trường tùy chọn `date_from`, `date_to` vào `IngestionReport`. | `IngestionReport` | stdlib, `numpy` |
| `src/quantautoml/models/tuner.py` | Sinh true OOF predictions cho mỗi `TunedCandidate` khi có `oof_folds` (fit trên `tr_idx`, predict trên `va_idx` với `best_params`) trước khi refit toàn bộ Train. | `TunedCandidate.oof_cal_a_preds`, `TunedCandidate.oof_cal_b_preds` (hoặc hàm `compute_oof_predictions()`) | `numpy`, `models.candidates`, `common` |
| `src/quantautoml/pipeline.py` | 1) Dùng true OOF predictions thay vì `refitted_model.predict(train_subset)` ở `OOF_ASSISTED`. 2) Đảo thứ tự nhân quả: SLSQP Ensemble trên `Cal-A` (60–70%), Temperature Calibration trên `Cal-B` (70–80%). | `run_pipeline()` | `data`, `features`, `models`, `ensemble`, `packaging`, `backtest` |
| `src/quantautoml/common/config.py` | Thêm `max_top_freq_ratio: float = 0.995` vào `FeaturesConfig`; đổi mặc định `min_unique_ratio = 0.0` và `max_peak_var_ratio: float | None = None`. | `FeaturesConfig` | `pydantic` |
| `src/quantautoml/features/variance.py` | Đánh giá near-constant theo `max_top_freq_ratio` (bất biến theo $N$); chỉ áp `min_unique_ratio` khi `> 0.0`; chỉ áp `max_peak_var_ratio` khi không `None`. | `MultiMetricVarianceFilter` | `numpy`, `common`, `features.base` |

---

## Design Rules & Boundaries
1. **Không đảo ngược phụ thuộc:** `common` không import tầng trên; `features` không import `models`/`ensemble`; `packaging` không import `backtest`.
2. **Bảo toàn RULE-01:** `refitted_model` trong `TunedCandidate` chỉ dùng để predict trên dữ liệu **chưa từng thấy** (`cal_a`, `cal_b`, `val`, `oos` ở `STATIC_5WAY`, hoặc `val`, `oos` ở `OOF_ASSISTED`).
3. **Bảo toàn RULE-06 & RULE-07:** Không thêm bất kỳ dependency nào vào `predictor_template.py`; sai số float32 ONNX phải luôn `< 1e-4`.
