# Step 07 — Trade-off: Unconstrained Ranking vs. Turnover-Constrained Portfolio with Friction

## So sánh

- **Unconstrained Ranking (Long Top 20% / Short Bottom 20% mỗi nến):**
  - *Pros:* Đo lường sức mạnh thuần túy (Theoretical Alpha) của mô hình.
  - *Cons:* Nếu mô hình đổi vị thế liên tục ở mỗi nến (Turnover $\approx 100\%$/nến), sau khi trừ phí giao dịch (ví dụ: Taker $0.05\%$) và trượt giá ($0.02\%$), Sharpe Ratio từ $+2.0$ trên lý thuyết sẽ rơi xuống $-3.0$ trong thực tế.
- **Turnover-Constrained & Friction Matrix (Được chọn):**
  - *Pros:*
    1. Đưa vào ma trận chi phí ma sát rõ ràng: $\text{Net Return} = \text{Gross Return} - \text{Turnover} \times (\text{Fee} + \text{Slippage})$.
    2. Cơ chế **Hysteresis Buffer Bands**: Chỉ thay thế một tài sản trong Top 20% nếu thứ hạng của nó tụt xuống dưới mức 30% (ngăn chặn việc ra vào lệnh chỉ vì dao động thứ hạng nhỏ ở ranh giới).
  - *Cons:* Đòi hỏi logic backtesting phức tạp hơn.

> **Quyết định:** Chuẩn hóa việc đánh giá nghiệm thu OOS phải phản ánh **Net Sharpe Ratio (sau phí ma sát)** và có cơ chế buffer band kiểm soát vòng quay danh mục.
