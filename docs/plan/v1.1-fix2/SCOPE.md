# Project / Feature Scope — `v1.1-fix2` (GBDT Regularization & OOS Robustness)

**Feature/Topic:** GBDT Search-Space Regularization & OOS Robustness (`v1.1-fix2`)
**Branch:** `develop/v1.1`
**Date:** 2026-11-24
**Status:** Draft (Pending User Approval)

---

## 1. Objectives & Goals

Tiếp nối `v1.1-fix1` và `f2b_stationarity`, mục tiêu của `v1.1-fix2` là:
1. **Triệt tiêu hiện tượng GBDT overfit trên chuỗi thời gian nhiễu cao** (như `lightgbm` đạt `num_leaves=49, max_depth=7, min_child_samples=8, lr=0.088` ở `tb5_label_nov24v1` dẫn tới IC âm trên `Cal-A` và `OOS`).
2. **Tăng tốc bước Optuna Tuning ~1.5–2x** nhờ giới hạn độ sâu cây hợp lý cho dữ liệu định lượng (`depth <= 6`, `num_leaves <= 31`).
3. **Nâng cấp khả năng tận dụng dữ liệu gần OOS và kiểm định đa cửa sổ** (Phase 2 — tùy chọn `Train+Cal` refit & Multi-Window Walk-Forward OOS).

---

## 2. In-Scope (Must Have)

### Phase 1 — GBDT Regularization, Search-Space Hardening & GPU Acceleration *(Done)*
- [x] **Adaptive Leaf Regularization:** Co giãn động `min_child_samples` (LightGBM: `[max(10, n // 1000), max(30, min(300, n // 200))]` $\approx 50\text{..}250$ trên 50k dòng) và `min_child_weight` (XGBoost: `[max(2, n // 5000), max(15, min(60, n // 1000))]` $\approx 10\text{..}50$).
- [x] **Sweet-Spot Tree & LR Bounds (Tournament + Tuner):** `max_depth` / `depth` $\in [3, 7]$ (bỏ `max_depth = -1` và `8`), `num_leaves` $\in [7, 47]$, `learning_rate` $\in [0.01, 0.08]$, `n_estimators` $\in [50, 300]$, `l2_leaf_reg` (CatBoost) $\in [2.0, 20.0]$.
- [x] **GPU Acceleration (`models.device`):** Thêm `models.device: Literal["auto", "cpu", "cuda"] = "auto"` vào `ModelsConfig`; bật CUDA tự động cho **XGBoost** (`tree_method="hist"`, `device="cuda"`) và **CatBoost** (`task_type="GPU"`, `bootstrap_type="Bernoulli"`), giữ **LightGBM** trên CPU đa nhân; tự động fallback về CPU khi không có GPU hoặc `device="cpu"`.
- [x] **Unit & Integration Tests:** Kiểm chứng giới hạn tham số mới cho cả dataset nhỏ (`n=250`) và lớn (`n=50_000`), kiểm chứng `device="cpu"` / `"auto"` / `"cuda"` (kèm ONNX float32 parity `< 1e-4`), xác nhận 0 hồi quy toàn suite.

### Phase 2 — Regime Freshness & Multi-Window Walk-Forward OOS *(Planned)*
- [ ] **Optional `Train + Cal` Refit:** Tùy chọn cấu hình refit model cuối trên `Train + Cal-A + Cal-B` sau khi đã khóa siêu tham số, trọng số và Temperature (giúp model thấy 20% dữ liệu mới nhất trước OOS).
- [ ] **Multi-Window Walk-Forward OOS (`backtest/walkforward.py`):** Đánh giá chiến lược trên nhiều cửa sổ OOS nối tiếp thay vì chỉ 1 đoạn 10% cuối cùng.

---

## 3. Out-of-Scope (Non-Goals)

- Không thay đổi bộ lọc `f2b_stationarity` vừa hoàn thiện.
- Không thêm mô hình ngoài GBDT Trio (LightGBM, XGBoost, CatBoost).
- Không thay đổi cấu trúc ONNX Model Bundle (`RULE-06`, `RULE-07`) hoặc format `01_data/`..`06_backtest/`.

---

## 4. Constraints & Assumptions

- Tuân thủ toàn bộ `RULE-01` đến `RULE-10` trong `AGENTS.md`.
- **Commit Policy:** 1 commit duy nhất cho mỗi Phase sau khi toàn bộ test suite PASS 100% và đã chạy `graphify update .`.
- Giữ tương thích ngược 100% với tất cả file `config.toml` hiện có (không bắt buộc thêm trường mới).
