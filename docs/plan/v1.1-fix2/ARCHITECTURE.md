# System Architecture & Search-Space Design — `v1.1-fix2`

## 1. Kiến Trúc Không Gian Tìm Kiếm Siêu Tham Số Chống Overfit (Phase 1)

Trên dữ liệu tài chính intraday (tín hiệu/nhiễu thấp, $N_{\text{train}} \approx 50{,}000$ nến 5 phút), cây GBDT sâu (`depth = 7..8`, `num_leaves = 49..63`, `min_child_samples = 8`) dễ dàng ghi nhớ từng cụm nhiễu ngắn hạn (40 phút dữ liệu/lá) dẫn tới sụp đổ IC trên `Cal-A` và `OOS`.

### Bảng Không Gian Tìm Kiếm Mới (Tuner & Tournament)

| Model | Tham số | Trước (`v1.1-fix1`) | Sau (`v1.1-fix2`) | Lý do định lượng |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | `n_estimators` | `30..200` | `50..300` (`[50,100,150,250]` Tourn) | Đủ số cây để hội tụ mượt khi `lr` nhỏ (tránh underfit) |
| **LightGBM** | `max_depth` | `-1, 4, 6, 8` (Tourn) / `3..8` (Tuner) | `3..7` | Chặn cây sâu vô hạn (`-1`), vẫn đủ dư địa cho tương tác phi tuyến |
| **LightGBM** | `num_leaves` | `15..63` | `7..47` | Cân bằng giữa biểu diễn phi tuyến và chống phân mảnh lá |
| **LightGBM** | `min_child_samples` | cố định `5..30` | động `[lo, hi]` theo $N_{\text{train}}$ | `lo = max(10, n // 1000)`, `hi = max(30, min(300, n // 200))` $\Rightarrow$ `50..250` trên 50k dòng |
| **LightGBM** | `learning_rate` | `0.01..0.1` | `0.01..0.08` | Đủ linh hoạt không underfit, tránh bước nhảy quá gắt |
| **XGBoost** | `n_estimators` | `30..200` | `50..300` | Đồng bộ LightGBM |
| **XGBoost** | `max_depth` | `3..8` | `3..7` | Đồng bộ trần độ sâu |
| **XGBoost** | `min_child_weight` | `1..15` | động `[lo, hi]` theo $N_{\text{train}}$ | `lo = max(2, n // 5000)`, `hi = max(15, min(60, n // 1000))` $\Rightarrow$ `10..50` trên 50k dòng |
| **XGBoost** | `learning_rate` | `0.01..0.1` | `0.01..0.08` | Cân bằng hội tụ & regularization |
| **CatBoost** | `n_estimators` | `30..200` | `50..300` | Đồng bộ LightGBM & XGBoost |
| **CatBoost** | `depth` | `4..8` (Tourn) / `3..8` (Tuner) | `3..7` | Oblivious tree `depth <= 7` đủ mạnh và chạy rất nhanh trên GPU |
| **CatBoost** | `l2_leaf_reg` | `1.0..10.0` | `2.0..20.0` | Tăng cường co cụm trọng số lá trên dữ liệu nhiễu |
| **CatBoost** | `learning_rate` | `0.01..0.1` | `0.01..0.08` | Mượt hóa hội tụ |

---

## 2. Tăng Tốc GPU (`models.device`)

- **`models.device = "auto"` (mặc định):** Kiểm tra GPU NVIDIA khả dụng (cache 1 lần duy nhất mỗi process).
  - **XGBoost:** Dùng `tree_method="hist"`, `device="cuda"` (nhanh hơn CPU 3–8x trên RTX 5070 Ti).
  - **CatBoost:** Dùng `task_type="GPU"`, `bootstrap_type="Bernoulli"` (bắt buộc để hỗ trợ `subsample` trên GPU; nhanh hơn CPU 5–15x).
  - **LightGBM:** Luôn chạy trên CPU đa nhân (`n_jobs=-1`) vì wheel PyPI không kèm CUDA và CPU histogram đã rất nhanh ở cỡ 50k × 30.
- **`models.device = "cpu"`:** Ép chạy 100% CPU (hữu ích cho CI/CD hoặc khi cần tái lập bit-exact tuyệt đối không phụ thuộc phần cứng GPU).
- **Bảo toàn RULE-06 (ONNX Parity):** Cây huấn luyện trên GPU vẫn export sang ONNX float32 bình thường và đi qua cổng kiểm định `< 1e-4` trên `Val`.
