# Step 04 — Phase 3 (phần 2/2): Deep Hyperparameter Optimization

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md). Phần 1/2: [`../03-model-selection/PHASE.md`](../03-model-selection/PHASE.md).

- **Mục tiêu:** Tinh chỉnh siêu tham số chuyên sâu cho (các) mô hình được chọn từ Step 03.

## Sub-systems & Deliverables

1. `DeepOptunaTuner`: Tối ưu hóa chuyên sâu với `TPESampler(multivariate=True)` kết hợp `HyperbandPruner`, hàm mục tiêu: $\text{IC Sharpe}_{WF} - 0.05 \times \text{Complexity\_Score}$.

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Optuna tự động tỉa sớm (prune) các nhánh kém hiệu quả.
