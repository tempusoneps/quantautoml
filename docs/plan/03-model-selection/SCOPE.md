# Step 03 — Scope: Model Tournament & Decision Engine

> Thuộc [Phase 3](../PHASES.md). Phần tối ưu siêu tham số nằm ở [`../04-tuning/`](../04-tuning/SCOPE.md).

## In-Scope (Must Have)

- [ ] **Modern GBDT Trio Pool:** LightGBM, CatBoost, XGBoost (loại bỏ Random Forest để bảo toàn kích thước file ONNX và độ trễ $< 5\text{ ms}$).
- [ ] Tournament sơ bộ với ngân sách 30 trials/model bằng `RandomSampler` trên 5-fold Expanding Window Walk-Forward.
- [ ] Decision Engine: Kiểm tra điều kiện Diversity ($< 0.88$), Marginal Gain ($+1.5\%$ Rank IC), và WF Consistency ($\ge 60\%$) để quyết định Ensemble hay Single Best Model (tối đa $K \le 3$).
