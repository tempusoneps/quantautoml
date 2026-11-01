# Step 02 — Trade-off: Multi-Horizon Gate (F3+7) — "Any Horizon" vs. "FDR-Controlled + Term Structure Monotonicity"

## Vấn đề

Tiêu chí cũ: *"Chỉ cần $\text{ICIR} > 0.3$ ở **bất kỳ** horizon nào trong $\{1, 5, 10, 21\}$"*.
- **Hậu quả:** Gây ra lỗi toán học nghiêm trọng về **Multiple Testing (Data Snooping Bias)**. Với 1,000 features rác hoàn toàn ngẫu nhiên, xác suất có ít nhất 1 horizon đạt $\text{ICIR} > 0.3$ ngẫu nhiên là rất cao.

## Giải pháp & Đánh đổi

- **Phương án mới (Được chọn):**
  1. Horizon mục tiêu chính ($h_{target}$) bắt buộc phải thỏa mãn: $\text{ICIR}(h_{target}) > 0.30$.
  2. Áp dụng điều chỉnh **Benjamini-Hochberg (FDR $\le 0.05$)** trên tập các features được thử nghiệm.
  3. Kiểm tra tính **Đơn điệu suy giảm của cấu trúc kỳ hạn (Term Structure Monotonicity):** Dự báo tài chính thường có quy luật phân rã thông tin mượt mà theo thời gian. Nếu $h=1$ có IC âm, $h=5$ có IC dương vọt lên, rồi $h=10$ lại sụp đổ, feature đó $99\%$ là overfit/nhiễu.

> **Quyết định:** Bắt buộc áp dụng kiểm soát FDR và kiểm tra tính đơn điệu của IC decay.
