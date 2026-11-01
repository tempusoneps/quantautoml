# Architectural & Quantitative Trade-offs: Quant AutoML Pipeline

**System:** Quant AutoML Pipeline  
**Version:** `v1.0.0-PROD-SPEC` (Updated with Quant Master Review)  
**Date:** 2026-11-01  
**Status:** Approved  

Tài liệu này là **index**. Phân tích chi tiết các hướng tiếp cận, sự đánh đổi (trade-offs), ưu và nhược điểm đằng sau các quyết định kiến trúc và toán học cốt lõi nằm trong `TRADE-OFF.md` của từng step.

| # | Quyết định | Kết luận | Chi tiết |
| :--- | :--- | :--- | :--- |
| 1 | Probability Calibration: Temperature vs. Platt vs. Isotonic | Temperature Scaling; **cấm Isotonic Regression** | [`05-ensemble-calibration/TRADE-OFF.md`](05-ensemble-calibration/TRADE-OFF.md) §1 |
| 2 | SLSQP Objective: Direct Rank IC vs. Pearson trên Rank-Transform | Pearson trên dữ liệu Rank-Transform ($C^2$) | [`05-ensemble-calibration/TRADE-OFF.md`](05-ensemble-calibration/TRADE-OFF.md) §2 |
| 3 | Data Partitioning: Static 5-Way vs. Adaptive OOF-Assisted | Adaptive Partitioning theo $N$ | [`01-ingestion-split/TRADE-OFF.md`](01-ingestion-split/TRADE-OFF.md) |
| 4 | Multi-Horizon Gate F3+7: "Any Horizon" vs. FDR + Term Structure | FDR (BH) + Term Structure Monotonicity | [`02-feature-selection/TRADE-OFF.md`](02-feature-selection/TRADE-OFF.md) |
| 5 | Candidate Pool: thêm Random Forest vs. GBDT Trio | GBDT Trio (LightGBM, CatBoost, XGBoost) | [`03-model-selection/TRADE-OFF.md`](03-model-selection/TRADE-OFF.md) |
| 6 | Backtest: Unconstrained Ranking vs. Turnover-Constrained + Friction | Net Sharpe sau ma sát + Hysteresis Buffer | [`07-backtest/TRADE-OFF.md`](07-backtest/TRADE-OFF.md) |

> Không tự ý quay lại các hướng tiếp cận đã bị loại bỏ (Isotonic Regression, Direct Rank SLSQP, Random Forest trong tournament mặc định).
