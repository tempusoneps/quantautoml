# Step 02 — Phase 2: Leakage-Free Feature Funnel (Fast dCor & FDR Control)

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md).

- **Mục tiêu:** Xây dựng phễu thanh lọc tính năng thông minh, kết hợp Hard Gate cho rác rõ ràng và Soft Evidence Scoring, kiểm soát đa giả thuyết (FDR) và tối ưu độ phức tạp tính toán.

## Sub-systems & Deliverables

1. `NaNInfTriage (F1)`: Phân loại MCAR/MAR/MNAR, sinh cột indicator, tính điểm kết hợp $\text{Score} = \text{MI} \times (1 - \text{NaN\_Ratio})$.
2. `MultiMetricVarianceFilter (F2)`: Đánh giá đồng thời Hệ số biến thiên ($CV$), Peak Rolling Variance ($w=63$), và Unique Ratio.
3. `ICStabilityFilter (F3+7)`: Đo tính ổn định $P(Y \mid X)$ trên $h_{target}$, kiểm soát tỷ lệ phát hiện sai **Benjamini-Hochberg (FDR $\le 0.05$)**, và kiểm tra tính đơn điệu suy giảm kỳ hạn trên $\{1, 5, 10, 21\}$.
4. `FastNonlinearCollinearityFilter (F4)`: Đo phụ thuộc phi tuyến bằng Fast Distance Correlation ($\mathcal{O}(N \log N)$ hoặc Subsampling khi $N > 5,000$, dCor $> 0.85$).
5. `ICBehaviorClusterFilter (F5)`: Gom cụm phân cấp theo khoảng cách tương quan hành vi IC $1 - |\text{corr}(\text{IC}_i, \text{IC}_j)|$, giữ lại đại diện Max-MI/Rank-IC.
6. `RegimeSynergyFilter (F6)`: Đánh giá MI theo Regime (Bull/Bear/Sideways) và bộ lọc bảo vệ cặp Synergistic Features.
7. `JMIFeatureSelector (F8)`: Tối ưu hóa tập Golden Feature Set theo thuật toán Joint Mutual Information.

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Kiểm thử xác nhận không có ngoại lệ hoặc data leakage (`fit` chỉ trên Train).
- Thuật toán dCor chạy nhanh với $N > 10,000$ mà không bị tràn bộ nhớ RAM.
