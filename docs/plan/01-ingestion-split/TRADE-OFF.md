# Step 01 — Trade-off: Static 5-Way Split vs. Adaptive Split (OOF-Assisted)

## Vấn đề

Việc chia tĩnh cứng nhắc `Train (60%) | Cal-A (10%) | Cal-B (10%) | Val (10%) | OOS (10%)` rất hoàn hảo cho tập dữ liệu lớn ($N \ge 10,000$), nhưng nếu người dùng chạy trên dữ liệu nến ngày (Daily, $N \approx 1,250$ nến):
- Tập Cal-B chỉ có $125$ nến. Tối ưu hóa ma trận trọng số SLSQP trên $125$ nến với Bootstrap sẽ bị nhiễu ước lượng cực lớn (*Estimation Risk / Markowitz Curse*).

## So sánh các hướng tiếp cận

| Phương án | Cơ chế | Ưu điểm (Pros) | Nhược điểm (Cons) | Áp dụng |
| :--- | :--- | :--- | :--- | :--- |
| **Static 5-Way** | Cố định 5 phân vùng thời gian độc lập. | Triệt tiêu 100% rủi ro rò rỉ; cấu trúc pipeline đơn giản, dễ giải thích. | Tốn kém dữ liệu; Cal-A và Cal-B bị thiếu mẫu nếu $N$ nhỏ. | Khi $N \ge 10,000$ (nến 1m, 5m, 15m, 1h). |
| **Adaptive OOF-Assisted** | Dùng Out-of-Fold (OOF) predictions từ 5-fold Walk-Forward của tập Train để làm dữ liệu cho Cal-A & Cal-B; chỉ giữ lại 3 vùng thời gian: `Train (80%) | Val (10%) | OOS (10%)`. | Tối đa hóa kích thước dữ liệu huấn luyện ($80\%$ thay vì $60\%$); tập Caliberation có kích thước lớn ngang Train. | Độ phức tạp khi điều phối pipeline tăng; phải quản lý cẩn thận OOF metadata. | Khi $N < 10,000$ (nến Daily hoặc thời gian ngắn). |

> **Quyết định:** Hệ thống áp dụng **Adaptive Partitioning**:
> - Nếu $N \ge 10,000$: Kích hoạt chuẩn Static 5-Way Split (`Train 60% / Cal-A 10% / Cal-B 10% / Val 10% / OOS 10%`).
> - Nếu $N < 10,000$ (và $N \ge 500$): Tự động gom Cal-A và Cal-B vào luồng OOF của 5-fold Walk-Forward trên Train (Train 80% / Val 10% / OOS 10%) để triệt tiêu lỗi thiếu mẫu thống kê.
