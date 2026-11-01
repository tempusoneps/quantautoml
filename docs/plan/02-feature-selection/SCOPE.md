# Step 02 — Scope: Leakage-Free Feature Funnel (bộ lọc 7 tầng)

> Thuộc [Phase 2](../PHASES.md). Phạm vi tổng thể xem [`../SCOPE.md`](../SCOPE.md).

## In-Scope (Must Have)

- [ ] **F1 (NaN/Inf Triage):** Phân loại MCAR/MAR/MNAR, sinh indicator `is_nan_X`, tính điểm kết hợp $\text{Score} = \text{MI}(X, Y) \times (1 - \text{NaN\_Ratio})$.
- [ ] **F2 (Multi-Metric Variance):** Lọc rác đồng thời qua Hệ số biến thiên ($CV$), Peak Rolling Variance (phân vị 90, cửa sổ 63 nến), và Unique Ratio.
- [ ] **F3+7 (IC Stability & Multi-Horizon Gate có FDR):** Đánh giá tính ổn định $P(Y \mid X)$ trên horizon mục tiêu, áp dụng kiểm soát đa giả thuyết **Benjamini-Hochberg (FDR $\le 0.05$)**, và kiểm tra tính đơn điệu suy giảm của IC qua các kỳ hạn $\{1, 5, 10, 21\}$.
- [ ] **F4 (Fast dCor Collinearity):** Đo phụ thuộc phi tuyến bằng Fast Distance Correlation ($\mathcal{O}(N \log N)$ hoặc Subsampling khi $N > 5,000$); triệt tiêu cộng tuyến phi tuyến ($> 0.85$).
- [ ] **F5 (IC-Behavior Clustering):** Gom cụm phân cấp theo hành vi dự báo $1 - |\text{corr}(\text{IC}_{X_i}, \text{IC}_{X_j})|$, đại diện cụm chọn theo Max-MI/Rank-IC.
- [ ] **F6 (Regime-Aware MI & Synergy Gate):** Tính MI theo chế độ thị trường (Bull/Bear/Sideways) và phát hiện cặp tính năng cộng hưởng (Synergy).
- [ ] **F8 (JMI - Joint Mutual Information):** Tuyển chọn tập Golden Feature Set tối ưu hóa thông tin tương hỗ liên kết.

## Ràng buộc liên quan

- Mọi `fit()` chỉ gọi trên tập Train (RULE-01); `Cal-A`, `Cal-B`, `Val`, `OOS` chỉ `transform()`.
- Kiểm soát FDR ≤ 0.05 (RULE-08).
