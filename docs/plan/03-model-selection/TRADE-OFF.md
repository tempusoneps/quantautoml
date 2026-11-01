# Step 03 — Trade-off: Candidate Model Pool — thêm Random Forest vs. tinh giản GBDT Trio

## So sánh

- **Phương án 1 (Bao gồm Random Forest từ scikit-learn):**
  - *Pros:* Tăng tính đa dạng kiến trúc (Bagging song song vs. Boosting tuần tự).
  - *Cons:* Cây của Random Forest không sâu có giới hạn và không có leaf-wise histogram tối ưu như GBDT. Khi xuất sang file `.onnx`, mô hình RF thường nặng hàng trăm MB, thời gian suy luận tăng vọt lên $> 15\text{ ms/row}$, vi phạm nghiêm ngặt yêu cầu latency $< 5\text{ ms}$.
- **Phương án 2 (Bộ 3 GBDT: LightGBM + CatBoost + XGBoost - Được chọn):**
  - *Pros:* Đạt độ đa dạng thuật toán tối ưu cho dữ liệu dạng bảng (LightGBM dùng Leaf-wise histogram; CatBoost dùng Oblivious/Symmetric Trees & Target Encoding chống rò rỉ; XGBoost dùng Exact/Depth-wise greediness). Trọng lượng ONNX siêu nhẹ ($< 5\text{ MB}$), latency suy luận cực thấp ($< 1.5\text{ ms/row}$).
  - *Cons:* Đều thuộc họ Boosting (tuy nhiên 3 cơ chế xây cây và chia nhánh hoàn toàn khác biệt).

> **Quyết định:** Loại bỏ `RandomForestClassifier/Regressor` khỏi tournament mặc định, tập trung toàn bộ ngân sách 30 trials vào **LightGBM**, **CatBoost**, và **XGBoost**.
