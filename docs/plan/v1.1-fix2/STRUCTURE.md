# Directory & File Structure — `v1.1-fix2`

Bản đồ các module bị tác động trong `v1.1-fix2`, tuân thủ nguyên tắc SRP và ranh giới phụ thuộc trong `docs/STRUCTURE.md`.

---

## Directory Tree (Vùng bị tác động)

```text
quantautoml/
├── src/quantautoml/
│   ├── common/
│   │   └── config.py                        # [P1] Thêm models.device ("auto"|"cpu"|"cuda"); [P2] Thêm models.refit_on_train_cal
│   ├── models/
│   │   ├── candidates.py                    # [P1] Hỗ trợ device ("auto"|"cpu"|"cuda") cho XGBoost (cuda) & CatBoost (GPU)
│   │   ├── tournament.py                    # [P1] Biên chống overfit (depth 3..7, num_leaves 7..47, lr 0.01..0.08, adaptive min_child_samples) + truyền device
│   │   └── tuner.py                         # [P1] Đồng bộ biên tìm kiếm với Tournament (n_estimators 50..300) + truyền device
│   ├── backtest/
│   │   └── walkforward.py                   # [P2] (New) Multi-Window Walk-Forward OOS runner
│   └── pipeline.py                          # [P1] Truyền cfg.models.device cho _fit_oof_candidates; [P2] Hỗ trợ refit_on_train_cal
├── scripts/
│   └── generate_config.py                   # [P1] Đồng bộ models.device = "auto"
└── tests/
    └── unit/
        ├── test_candidates.py               # [P1] Test device resolution & GPU/CPU fallback cho XGBoost/CatBoost/LightGBM
        ├── test_tournament.py               # [P1] Test không gian tham số chống overfit & adaptive min_child_samples
        └── test_tuner.py                    # [P1] Test không gian tham số chống overfit trên n nhỏ & n lớn
```

---

## Component & File Responsibilities (Phase 1 Active)

| File / Module | Trách Nhiệm Trong Phase 1 | Ký Hiệu / Thay Đổi Chính | Phụ Thuộc Hợp Lệ |
| :--- | :--- | :--- | :--- |
| `src/quantautoml/common/config.py` | Thêm tùy chọn `device: Literal["auto", "cpu", "cuda"] = "auto"` vào `ModelsConfig`. | `ModelsConfig.device` | `pydantic` |
| `src/quantautoml/models/candidates.py` | Phát hiện GPU khả dụng (`resolve_device`), cấu hình CUDA cho `XGBoostCandidate` & `CatBoostCandidate`, giữ CPU cho `LightGBMCandidate`. | `resolve_device()`, `BaseGBDCandidate(..., device=...)`, `create_candidate(..., device=...)` | `lightgbm`, `xgboost`, `catboost`, `common` |
| `src/quantautoml/models/tournament.py` | Sinh siêu tham số RandomSampler với biên điểm ngọt & adaptive leaf bounds; truyền `cfg.device` vào `create_candidate`. | `adaptive_leaf_bounds()`, `adaptive_child_weight_bounds()`, `_sample_params()` | `optuna`, `numpy`, `common` |
| `src/quantautoml/models/tuner.py` | Sinh siêu tham số TPE đồng bộ biên tìm kiếm; truyền `cfg.device` vào `create_candidate`. | `_suggest_tuning_params()` | `optuna`, `numpy`, `common` |

---

## Design Rules & Boundaries
1. Không thay đổi public signature của `ModelTournament.run()` hay `DeepOptunaTuner.tune()`.
2. Giữ nguyên tính tái lập 100% (deterministic với `random_state`).
3. Không phá vỡ bất kỳ test unit hay integration nào hiện có (kể cả trên dataset tổng hợp nhỏ `n=250`).
