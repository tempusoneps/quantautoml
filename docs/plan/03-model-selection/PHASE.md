# Step 03 — Phase 3 (phần 1/2): Modern GBDT Tournament & Decision Engine

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md). Phần 2/2 (Optuna tuning): [`../04-tuning/PHASE.md`](../04-tuning/PHASE.md).

- **Mục tiêu:** Thiết lập cơ chế đấu loại công bằng giữa 3 mô hình GBDT hiện đại (LightGBM, CatBoost, XGBoost) và tự động ra quyết định Single Model vs. Ensemble.

## Sub-systems & Deliverables

1. `ModelTournament`: Cấp ngân sách đồng nhất 30 trials `RandomSampler` cho **LightGBM, CatBoost, XGBoost** trên 5-fold Expanding Window Walk-Forward (Train). Xếp hạng bằng IC Sharpe.
2. `ModelDecisionEngine`: Đánh giá 3 điều kiện (Diversity correlation $< 0.88$, Marginal Gain $+1.5\%$ Rank IC, Walk-Forward Consistency $\ge 60\%$) để quyết định chọn Ensemble hay Single Best Model (trần $K \le 3$).

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Tournament hoàn thành đúng ngân sách và chọn ra đúng Top Models.
- Decision Engine fallback về Single Model khi độ tương quan giữa các mô hình quá cao ($> 0.88$).
