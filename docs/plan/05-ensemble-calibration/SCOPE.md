# Step 05 — Scope: Convex Ensemble & Probability Calibration

> Thuộc [Phase 4](../PHASES.md). Phạm vi tổng thể xem [`../SCOPE.md`](../SCOPE.md).

## In-Scope (Must Have)

- [ ] Tối ưu hóa trọng số SLSQP lồi trên tập **Cal-B**: $0.05 \le w_i \le 0.60$, $\sum w_i = 1$.
- [ ] **Hàm mục tiêu SLSQP khả vi $C^2$:** Tối ưu hóa Pearson Correlation trên ma trận dự đoán đã được Rank-Transform ($\tilde{Y} \in [0, 1]$), đảm bảo hội tụ ổn định và chính xác.
- [ ] Bootstrap SLSQP ($N=200$, lấy mẫu $80\%$) kết hợp Shrinkage về Equal Weights ($1/K$) và 5% Guard Rail.
- [ ] **Probability Calibration bằng Temperature Scaling / Platt Scaling:** Bỏ Isotonic Regression để tránh hiện tượng bậc thang (ties), bảo toàn $100\%$ thứ tự phân hạng Rank IC gốc và đạt $\text{ECE} < 0.05$.

## Ràng buộc liên quan

- RULE-03 (cấm Isotonic), RULE-04 (SLSQP chỉ nhận Pearson trên dữ liệu Rank-Transform), RULE-05 (Ensemble phải vượt baseline ≥ 5% IC Sharpe).
