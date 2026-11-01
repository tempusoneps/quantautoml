# Step 03 — Design: Tournament & Decision Engine

## Data flow

```mermaid
flowchart TD
    Funnel["Từ Step 02: Golden Feature Set"] -->|"Train (5-fold WF)"| Tourn["30-Trial Budget Tournament\n(Modern GBDT Trio: LightGBM, CatBoost, XGBoost)"]
    Tourn --> DecEng{"Decision Engine:\nDiversity < 0.88\nGain >= 1.5%\nWF Cons >= 60%?"}
    DecEng -->|"Yes"| EnsemblePool["Candidate Ensemble Pool\n(K <= 3 models)"]
    DecEng -->|"No"| SingleBest["Single Best Model"]
    EnsemblePool --> Next["Sang Step 04: Deep Optuna Tuning"]
    SingleBest --> Next
```

## Fallback

- Nếu Diversity giữa các mô hình $< 0.12$ (tương quan $> 0.88$), fallback về Single Best Model.

## Source layout

```
src/quantautoml/models/
├── candidates.py         # Modern GBDT Trio wrappers (LightGBM, CatBoost, XGBoost)
├── tournament.py         # 30-trials budget tournament trên Walk-Forward
└── decision_engine.py    # Quyết định Single vs. Ensemble (Diversity, Marginal Gain, WF)
tests/unit/  test_tournament.py
```

## Responsibilities

| Module / File | Trách nhiệm duy nhất | Interface / Exports chính | Dependencies |
| :--- | :--- | :--- | :--- |
| `models/candidates.py` | Gói gọn 3 thuật toán GBDT hiện đại với giao diện chuẩn. | `GBDTTrioPool` | `lightgbm`, `catboost`, `xgboost` |

## Công nghệ

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| Machine Learning Pool | **LightGBM, CatBoost, XGBoost** | Tối ưu cho tabular quant, kích thước file nhỏ, suy luận $< 1.5\text{ ms/row}$. |
