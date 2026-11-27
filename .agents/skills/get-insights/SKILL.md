---
name: get-insights
description: Use when analyzing, summarizing, or comparing QuantAutoML pipeline run output directories (e.g. runs/<run_name>) into a standardized markdown insight report in insights/
---

# Get Insights (QuantAutoML Run Analyzer)

## Overview

Generate a **uniform, scannable, side-by-side comparable** Markdown insight report **in Vietnamese (Tiếng Việt)** from any QuantAutoML run directory (`runs/<run_name>`) and save it to `insights/<run_name>.md` (or `insights/compare_<a_vs_b>.md` for multi-run comparisons).

**Core principle:** Use the deterministic extraction script first so every run has identical tables, units, and precision — never read huge JSONs like `01_tournament_report.json` raw into context — then write Section 8 (quantitative diagnosis & next actions) **in Vietnamese** from the extracted facts. Keep standard quant terminology and config/code identifiers (e.g., `Rank IC`, `ICIR`, `Sharpe`, `Sortino`, `Calmar`, `Max Drawdown`, `Profit Factor`, `ONNX`, `signal_ema_span`) intact where natural.

## When to Use

- User asks to analyze, review, or extract insights from a run folder (e.g., `runs/tb5_label_nov24v5`).
- User asks to compare 2+ runs side by side.
- After a `quantautoml run` completes and you want a permanent experiment log in `insights/`.

## Workflow

### Step 1: Run the Deterministic Extractor

From the repository root:

```bash
# Single run -> insights/<run_name>.md
python3 .agents/skills/get-insights/scripts/extract_insights.py runs/<run_name>

# Multi-run comparison -> insights/compare_<run1>_vs_<run2>.md
python3 .agents/skills/get-insights/scripts/extract_insights.py runs/<run1> runs/<run2> --compare

# Custom output path (optional)
python3 .agents/skills/get-insights/scripts/extract_insights.py runs/<run_name> --out insights/<custom>.md
```

The script automatically:
- Parses Phases `01_data` through `06_backtest` (gracefully skipping missing optional artifacts/fields like `config.toml`, `02_realworld_report.json`, `trades.csv`, `03_regime_diagnostics.json`, `returns_curve.png`, or `classification_metrics` in older `01_backtest_report.json`).
- Renders OOS label-prediction metrics (Accuracy, Balanced Accuracy, Macro/Weighted F1, Log Loss, ROC-AUC, OOS ECE, per-class Precision/Recall/F1, and Confusion Matrix) when `classification_metrics` is present in `01_backtest_report.json`.
- Aggregates `01_tournament_report.json` (which can exceed 100k lines) into a compact best-per-family table.
- Computes Long vs Short trade breakdowns from `trades.csv` and friction drag (`gross_sharpe - net_sharpe`).
- Embeds `returns_curve.png` via relative path when present.

### Step 2: Inspect the Generated Report & Fill Section 8 (in Vietnamese)

1. Read the generated file `insights/<run_name>.md` (or `insights/compare_...md`).
2. Replace the `<!-- AGENT_INSIGHTS_SECTION -->` placeholder in the final section with **concise, evidence-backed quantitative analysis written in Vietnamese** (`Điểm mạnh`, `Rủi ro & Điểm nghẽn`, `Thử nghiệm tiếp theo`; or for comparisons: `Mô hình vượt trội & Lý do`, `Yếu tố tạo khác biệt chính`, `Khuyến nghị lần chạy tiếp theo`) covering:
   - **Điểm mạnh (Strengths):** Highlight what passed with strong margin (Rank IC, ICIR, DSR, Tier 2 Profit Factor / Sortino, ONNX precision).
   - **Rủi ro & Điểm nghẽn (Risks & Bottlenecks):** Explicitly check for:
     - **Friction drag:** `gross_sharpe - net_sharpe` > 0.8 or high `mean_turnover`.
     - **Close vs Next-Open gap:** Tier 1 `execution_price="close"` vs Tier 2 `trade_on_close=False` (or `03_regime_diagnostics.json` close vs next_open drop).
     - **Model diversity:** `single_model` fallback due to pairwise correlation >= 0.88 or low marginal gain.
     - **Side asymmetry:** Long vs Short PnL / win-rate imbalance in `trades.csv`.
     - **Regime / Fold variance:** Wide `Fold IC Min` vs `Fold IC Max`, or segment drop in `03_regime_diagnostics.json`.
     - **Generalization:** Cal-A / Train IC Sharpe vs OOS Rank IC / ICIR.
   - **Thử nghiệm tiếp theo (2–3 concrete actions):** Specific config parameter changes (e.g., `signal_ema_span`, `min_hold_bars`, `execution_price = "next_open"`, `complexity_penalty`, `refit_on_train_cal`) with expected effect.

### Step 3: Reply to the User

Link the saved report (`[tb5_label_nov24v5.md](file:///...)`) and give a 3–5 bullet executive takeaway in Vietnamese (verdict, headline Tier 1 & Tier 2 metrics, top bottleneck, top next action). Do not dump the entire report table again in chat.

## Output Contract (Fixed Schema — Vietnamese)

Every single-run report in `insights/<run_name>.md` MUST be written in **Vietnamese** and preserve these exact 8 sections in order so diffs and visual comparisons across runs always align:

1. `## 1. Bảng Điểm Tổng Quan (Executive Scorecard)`
2. `## 2. Cấu Hình Chính (Configuration Snapshot)`
3. `## 3. Dữ Liệu & Phân Chia Thời Gian (Phase 01)`
4. `## 4. Phễu Lọc Tính Năng (Phase 02)`
5. `## 5. Đấu Loại Mô Hình, Quyết Định & Tối Ưu Siêu Tham Số (Phase 03)`
6. `## 6. Ensemble, Hiệu Chuẩn & Đóng Gói ONNX (Phase 04–05)`
7. `## 7. Đánh Giá Backtest (Phase 06)`
8. `## 8. Phân Tích Định Lượng & Hành Động Tiếp Theo`

## Common Mistakes

| Mistake | Fix |
| :--- | :--- |
| Reading `01_tournament_report.json` directly with `read_file` | Always use `extract_insights.py` (raw file is 100k+ lines). |
| Leaving `_TBD_` placeholders or writing Section 8 in English | Always edit the generated `.md` to fill Section 8 with real quant diagnosis **in Vietnamese**. |
| Changing table columns or section order between runs | Keep the script's fixed 8-section schema so runs stay visually comparable. |
| Ignoring Tier 1 vs Tier 2 execution mismatch | Check `execution_price` (`close` vs `next_open`) against `trade_on_close` (`false` = next open). |
