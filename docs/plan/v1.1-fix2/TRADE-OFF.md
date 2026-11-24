# Architectural & Quantitative Trade-offs — `v1.1-fix2`

**Feature/Topic:** GBDT Regularization & Search-Space Hardening (`v1.1-fix2`)
**Status:** Draft

---

## 1. Co Giãn Động `min_child_samples` Theo $N_{\text{train}}$ vs. Cố Định Trong Config

### Các phương án cân nhắc:
- **Phương án A:** Đặt cứng `min_child_samples` trong `[50, 250]`.
  - *Nhược điểm:* Làm hỏng các tập dữ liệu nhỏ (khung 1H/4H/1D có $N = 1{,}000$) và các bài unit test tổng hợp (`n = 250`, nơi fold train chỉ có ~125 dòng $\to$ không split nổi một cây nào!).
- **Phương án B (Chọn):** Co giãn động theo cỡ mẫu Train `n_samples`:
  - `lo = max(10, n_samples // 1000)`
  - `hi = max(30, min(300, n_samples // 200))`
  - *Ưu điểm:* Tự động thích nghi cho mọi khung thời gian (5m có 50k dòng $\to$ `50..250`; 1H có 5k dòng $\to$ `10..30`; unit test 250 dòng $\to$ `10..30`); không cần người dùng phải tự tính tay trong `config.toml`.
  - *Nhược điểm:* Không cho phép user override trực tiếp từng biên nhỏ (chấp nhận được vì YAGNI, tránh phình config).

### Kết luận:
Chọn **Phương án B**.

---

## 2. Biên "Điểm Ngọt" (`depth <= 7`, `lr <= 0.08`, `n_estimators <= 300`) vs. Siết Quá Chặt (`depth <= 6`, `lr <= 0.06`)

### Các phương án cân nhắc:
- **Phương án A:** Siết cứng `depth <= 6`, `num_leaves <= 31`, `lr <= 0.06`, `n_estimators <= 200`.
  - *Nhược điểm:* Khi `lr = 0.015` và `n_estimators = 100–200`, tổng bước học $1.5\text{–}3.0$ có thể gây **underfit** sau khi `f2b_stationarity` đã loại sạch feature không dừng.
- **Phương án B (Chọn):** Mở biên cân bằng `depth <= 7`, `num_leaves <= 47`, `lr` $\in [0.01, 0.08]$, `n_estimators` $\in [50, 300]$, nhưng dùng **`min_child_samples` co giãn động (`50..250`)** + **`l2_leaf_reg` (`2..20`)** làm chốt chặn chống overfit chính.
  - *Ưu điểm:* Không lo underfit; vẫn chặn hoàn toàn hiện tượng chia lá 8 mẫu gây ghi nhớ nhiễu; kết hợp GPU cho XGBoost & CatBoost thì `depth=7` vẫn chạy cực nhanh.

---

## 3. Chiến Lược GPU (`XGBoost + CatBoost` GPU, `LightGBM` CPU)

### Các phương án cân nhắc:
- **Phương án A:** Build LightGBM từ source với CUDA để chạy cả 3 trên GPU.
  - *Nhược điểm:* Phá vỡ khả năng cài đặt 1 lệnh `uv sync`; phức tạp bảo trì; tốc độ LightGBM CPU trên 50k × 30 vốn đã rất nhanh.
- **Phương án B (Chọn):** Bật GPU cho **XGBoost** và **CatBoost** (có sẵn trong wheel PyPI chuẩn), giữ **LightGBM** trên CPU, điều khiển qua `models.device = "auto" | "cpu" | "cuda"`.
  - *Ưu điểm:* Zero extra build step; giảm tổng thời gian Phase 3 từ ~41 phút xuống còn ~10–15 phút trên RTX 5070 Ti; tự động fallback về CPU trên máy không có GPU.
