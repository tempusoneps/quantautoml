# Step 05 — Phase 4: Convex Ensemble Optimization ($C^2$ Objective) & Temperature Calibration

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md).

- **Mục tiêu:** Tối ưu hóa trọng số kết hợp mô hình theo phương pháp quy hoạch phi tuyến có hàm mục tiêu khả vi bậc hai liên tục trên Cal-B và hiệu chuẩn xác suất bằng Temperature Scaling trên Cal-A.

## Sub-systems & Deliverables

1. `SLSQPEnsembleOptimizer`: Giải bài toán tối ưu lồi SLSQP trên tập **Cal-B** với ràng buộc $0.05 \le w_i \le 0.60$ và $\sum w_i = 1$. **Hàm mục tiêu:** Tối đa hóa Pearson Correlation trên ma trận dự đoán đã Rank-Transform ($\tilde{Y} \in [0, 1]$), đảm bảo khả vi $C^2$ và hội tụ cực nhanh.
2. `BootstrapShrinkageEngine`: Chạy Bootstrap $N=200$ lần (lấy mẫu $80\%$), tính vector trung vị, áp dụng Shrinkage về $1/K$ và 5% Guard Rail.
3. `ProbabilityCalibrator`: Tự động áp dụng **Temperature Scaling** (hoặc Platt Scaling nếu nhị phân) trên tập **Cal-A**, cam kết $\text{ECE} < 0.05$ và **bảo toàn 100% Rank IC ban đầu** (loại bỏ Isotonic Regression).

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Trọng số $w$ luôn nằm trong khoảng $[0.05, 0.60]$ và có tổng bằng 1.
- Quá trình hiệu chuẩn không làm thay đổi thứ tự phân hạng của các mẫu (Rank Invariance test).
