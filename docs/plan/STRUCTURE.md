# Directory & File Structure: Quant AutoML Pipeline

Tài liệu này xác định kiến trúc tổ chức thư mục nguồn, phân chia module và ranh giới trách nhiệm (Single Responsibility Principle) cho dự án `quantautoml`.

---

## Directory Tree

```
quantautoml/
├── docs/
│   ├── plan/
│   │   ├── SCOPE.md                  # Phạm vi, tiêu chí thành công, out-of-scope
│   │   ├── PHASES.md                 # Lộ trình 5 phase / sub-systems độc lập
│   │   ├── STRUCTURE.md              # Kiến trúc thư mục và phân chia trách nhiệm
│   │   ├── ARCHITECTURE.md           # Thiết kế kiến trúc, data flow & hạ tầng kỹ thuật
│   │   ├── TRADE-OFF.md              # Phân tích đánh đổi kỹ thuật & cơ sở toán học
│   │   └── PLAN.md                   # Kế hoạch thực thi chi tiết cho phase hiện tại
│   └── ...
├── src/
│   └── quantautoml/
│       ├── __init__.py               # Public API package exports
│       ├── common/
│       │   ├── __init__.py
│       │   ├── exceptions.py         # Custom exceptions (LeakageError, PrecisionError, etc.)
│       │   └── types.py              # Enums, Pydantic schemas, TaskType, SplitConfig
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py          # Đọc parquet/csv, đồng bộ timestamp UTC ISO-8601, float32
│       │   ├── task_inference.py     # Suy luận bài toán Binary/Multiclass/Unsupervised
│       │   └── splitter.py           # Adaptive Temporal Splitter (5-way vs. OOF-assisted) kèm Purging & Embargo
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── ic.py                 # Rank IC (Spearman), ICIR, IC Sharpe, Rolling IC, Downside IC
│       │   ├── fdr.py                # Benjamini-Hochberg False Discovery Rate correction
│       │   └── calibration.py        # Expected Calibration Error (ECE)
│       ├── features/
│       │   ├── __init__.py
│       │   ├── base.py               # BaseFeatureFilter interface (fit trên Train, transform trên all)
│       │   ├── triage.py             # F1: NaN/Inf Triage, MCAR/MAR/MNAR indicator, MI score
│       │   ├── variance.py           # F2: Multi-Metric Variance Check (CV, PeakVar, UniqueRatio)
│       │   ├── ic_stability.py       # F3+7: Rolling IC Stability, Multi-Horizon Term Structure & FDR Gate
│       │   ├── collinearity.py       # F4: Fast Distance Correlation (dCor O(N log N) / Subsampling)
│       │   ├── clustering.py         # F5: IC-Behavior Hierarchical Clustering với Max-MI
│       │   ├── regime_synergy.py     # F6: Regime-Aware MI & Synergistic Pairs Detection
│       │   ├── jmi.py                # F8: Joint Mutual Information feature selection
│       │   ├── _binning.py           # Private: quantile bin codes + contingency-table MI (F6, F8)
│       │   └── funnel.py             # FeatureFunnel pipeline điều phối 7 bộ lọc
│       ├── models/
│       │   ├── __init__.py
│       │   ├── candidates.py         # Modern GBDT Trio wrappers (LightGBM, CatBoost, XGBoost)
│       │   ├── tournament.py         # 30-trials budget tournament trên Walk-Forward
│       │   ├── decision_engine.py    # Quyết định Single vs. Ensemble (Diversity, Marginal Gain, WF)
│       │   └── tuner.py              # Full Optuna TPE Multivariate + HyperbandPruner
│       ├── ensemble/
│       │   ├── __init__.py
│       │   ├── slsqp.py              # Tối ưu lồi SLSQP trên Cal-B (Rank-Transformed Pearson C^2 objective)
│       │   ├── bootstrap.py          # Bootstrap SLSQP (N=200), Median weight & Shrinkage 1/K
│       │   └── calibration.py        # Probability Calibration (Temperature Scaling / Platt) trên Cal-A
│       ├── packaging/
│       │   ├── __init__.py
│       │   ├── bundle.py             # Đóng gói Model Bundle (.onnx, metadata, schema)
│       │   ├── schema.py             # Tạo và kiểm tra features_schema.json (Data Contract)
│       │   ├── validator.py          # Kiểm tra sai số số học float32 (|y_py - y_onnx| < 10^-4)
│       │   └── templates/
│       │       └── predictor_template.py # Mã nguồn predictor.py độc lập (~50 dòng numpy/onnxruntime)
│       └── backtest/
│           ├── __init__.py
│           ├── friction.py           # Mô hình hóa ma trận chi phí (Taker Fee + Slippage) & Turnover
│           ├── buffer.py             # Cơ chế Hysteresis Buffer Bands kiểm soát vòng quay danh mục
│           ├── evaluator.py          # Đánh giá tiêu chí nghiệm thu (OOS IC, Net Sharpe sau phí, MDD)
│           └── runner.py             # Thực thi kiểm thử mù trên tập OOS độc lập
├── tests/
│   ├── unit/
│   │   ├── test_ingestion.py
│   │   ├── test_task_inference.py
│   │   ├── test_splitter.py
│   │   ├── test_fast_dcor.py
│   │   ├── test_fdr.py
│   │   ├── test_feature_filters.py
│   │   ├── test_tournament.py
│   │   ├── test_slsqp_smooth.py
│   │   ├── test_temperature_calibration.py
│   │   └── test_packaging.py
│   ├── integration/
│   │   ├── test_zero_leakage.py      # Bắt buộc: Kiểm tra rò rỉ dữ liệu tương lai
│   │   ├── test_end_to_end.py        # Luồng hoàn chỉnh từ data thô đến ONNX bundle
│   │   └── test_friction_backtest.py # Kiểm tra backtest có tính phí giao dịch và turnover
│   └── conftest.py                   # Pytest fixtures tạo dữ liệu tài chính giả lập
├── pyproject.toml
└── README.md
```

---

## Component & File Responsibilities

| Module / File | Trách nhiệm duy nhất (Single Responsibility) | Interface / Exports chính | Dependencies |
| :--- | :--- | :--- | :--- |
| `data/ingestion.py` | Đọc dữ liệu, kiểm tra tính hợp lệ timestamp UTC ISO-8601, sắp xếp tăng dần, ép `float32`. | `load_dataset()`, `DataIngestion` | `polars`, `pandas`, `numpy` |
| `data/splitter.py` | Phân chia chuỗi thời gian thích ứng ($N \ge 10k$ vs $N < 5k$) kèm Purging & Embargo $h$ nến. | `AdaptiveTimeSeriesSplitter`, `SplitResult` | `numpy`, `pandas` |
| `metrics/fdr.py` | Kiểm soát tỷ lệ phát hiện sai (False Discovery Rate) theo Benjamini-Hochberg. | `apply_fdr_correction()` | `numpy`, `scipy` |
| `features/collinearity.py` | Lọc cộng tuyến phi tuyến bằng Fast Distance Correlation ($\mathcal{O}(N \log N)$). | `FastNonlinearCollinearityFilter` | `numpy`, `scipy` |
| `models/candidates.py` | Gói gọn 3 thuật toán GBDT hiện đại (LightGBM, CatBoost, XGBoost) với giao diện chuẩn. | `GBDTTrioPool` | `lightgbm`, `catboost`, `xgboost` |
| `ensemble/slsqp.py` | Tối ưu hóa trọng số lồi SLSQP với hàm mục tiêu Pearson trên dữ liệu đã Rank-Transform ($C^2$). | `SmoothSLSQPOptimizer` | `scipy.optimize` |
| `ensemble/calibration.py`| Hiệu chuẩn xác suất bảo toàn thứ hạng (Temperature Scaling, Platt) trên Cal-A. | `TemperatureCalibrator` | `scipy.optimize` |
| `backtest/friction.py` | Tính toán lợi nhuận ròng sau khi trừ chi phí giao dịch (Fee + Slippage) và Turnover. | `FrictionCostModel` | `numpy` |
| `backtest/buffer.py` | Kiểm soát vòng quay bằng dải đệm trễ Hysteresis Buffer Bands. | `HysteresisBuffer` | `numpy` |

---

## Design Rules & Boundaries

1. **Ranh giới Fit / Transform:** Phương thức `fit()` chỉ nhận tập Train; `transform()` áp dụng cho mọi tập mà không thay đổi internal state.
2. **Rank Invariance Contract:** Bất kỳ phương pháp Calibration nào được sử dụng bắt buộc phải thỏa mãn tính đơn điệu: $\text{Rank}(y_{calibrated}) \equiv \text{Rank}(y_{raw})$.
3. **Decoupled Production Inference:** File `predictor.py` độc lập hoàn toàn, không phụ thuộc vào codebase huấn luyện, chỉ dùng `numpy` và `onnxruntime`.
