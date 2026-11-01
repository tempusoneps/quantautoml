# Step 04 — Scope: Hyperparameter Optimization

> Thuộc [Phase 3](../PHASES.md). Phần đấu loại mô hình nằm ở [`../03-model-selection/`](../03-model-selection/SCOPE.md).

## In-Scope (Must Have)

- [ ] Full Optuna Tuning: `TPESampler(multivariate=True)` kết hợp `HyperbandPruner` với hàm mục tiêu $\text{IC Sharpe}_{WF} - 0.05 \times \text{Complexity\_Score}$.
