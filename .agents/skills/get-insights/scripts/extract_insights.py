#!/usr/bin/env python3
"""Extract standardized Markdown insights from QuantAutoML run directories.

Usage:
    python3 extract_insights.py runs/tb5_label_nov24v5
    python3 extract_insights.py runs/tb5_label_nov24v5 --out insights/tb5_label_nov24v5.md
    python3 extract_insights.py runs/tb5_label_nov24v4 runs/tb5_label_nov24v5 --compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _fmt(val: Any, spec: str = ".4f", suffix: str = "") -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return "—"
        return f"{val:{spec}}{suffix}"
    return str(val)


def _pct(val: Any, digits: int = 2, already_pct: bool = False) -> str:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return "—"
    num = float(val) if already_pct else float(val) * 100.0
    return f"{num:.{digits}f}%"


def _badge(passed: bool | None) -> str:
    if passed is None:
        return "—"
    return "✅ PASS" if passed else "❌ FAIL"


def _short_date(ts: str | None) -> str:
    if not ts:
        return "—"
    return str(ts).replace("T", " ")[:16]


def _parse_trades_csv(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    if not rows:
        return {"n": 0}

    def _side_stats(subset: list[dict[str, str]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0, "wins": 0, "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0}
        pnls = [float(r["PnL"]) for r in subset if r.get("PnL")]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "n": len(pnls),
            "wins": wins,
            "win_rate": wins / len(pnls) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "avg_pnl": statistics.mean(pnls) if pnls else 0.0,
        }

    longs = [r for r in rows if float(r.get("Size", "0") or 0) > 0]
    shorts = [r for r in rows if float(r.get("Size", "0") or 0) < 0]
    all_pnls = [float(r["PnL"]) for r in rows if r.get("PnL")]
    commissions = [float(r["Commission"]) for r in rows if r.get("Commission")]
    ret_pcts = [float(r["ReturnPct"]) * 100.0 for r in rows if r.get("ReturnPct")]
    bars = [
        int(float(r["ExitBar"])) - int(float(r["EntryBar"]))
        for r in rows
        if r.get("EntryBar") and r.get("ExitBar")
    ]
    return {
        "n": len(all_pnls),
        "total_pnl": sum(all_pnls),
        "total_commission": sum(commissions) if commissions else None,
        "best_pnl": max(all_pnls) if all_pnls else None,
        "worst_pnl": min(all_pnls) if all_pnls else None,
        "best_ret_pct": max(ret_pcts) if ret_pcts else None,
        "worst_ret_pct": min(ret_pcts) if ret_pcts else None,
        "median_bars": statistics.median(bars) if bars else None,
        "max_bars": max(bars) if bars else None,
        "long": _side_stats(longs),
        "short": _side_stats(shorts),
    }


def _summarize_tournament(tour: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not tour or not isinstance(tour.get("rankings"), list):
        return []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in tour["rankings"]:
        name = str(entry.get("model_name", "unknown"))
        by_model.setdefault(name, []).append(entry)

    rows: list[dict[str, Any]] = []
    for name, entries in by_model.items():
        best = max(entries, key=lambda e: float(e.get("ic_sharpe", float("-inf"))))
        fold_ics = [float(x) for x in best.get("fold_ics", [])]
        rows.append(
            {
                "model_name": name,
                "n_trials": len(entries),
                "best_trial_id": best.get("trial_id"),
                "best_ic_sharpe": best.get("ic_sharpe"),
                "mean_ic": best.get("mean_ic"),
                "fold_min": min(fold_ics) if fold_ics else None,
                "fold_max": max(fold_ics) if fold_ics else None,
                "fold_std": statistics.stdev(fold_ics) if len(fold_ics) >= 2 else 0.0,
            }
        )
    rows.sort(key=lambda r: float(r["best_ic_sharpe"] or float("-inf")), reverse=True)
    return rows


def load_run_data(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    return {
        "name": run_dir.name,
        "path": run_dir,
        "manifest": _load_json(run_dir / "00_run_manifest.json"),
        "config": _load_toml(run_dir / "config.toml"),
        "ingestion": _load_json(run_dir / "01_data" / "01_ingestion_report.json"),
        "task": _load_json(run_dir / "01_data" / "02_task.json"),
        "split": _load_json(run_dir / "01_data" / "03_split_report.json"),
        "funnel": _load_json(run_dir / "02_features" / "01_funnel_report.json"),
        "selected_features": _load_json(run_dir / "02_features" / "02_selected_features.json"),
        "tournament": _summarize_tournament(
            _load_json(run_dir / "03_models" / "01_tournament_report.json")
        ),
        "decision": _load_json(run_dir / "03_models" / "02_decision_report.json"),
        "tuning": _load_json(run_dir / "03_models" / "03_tuning_report.json"),
        "ensemble": _load_json(run_dir / "04_ensemble" / "01_ensemble_report.json"),
        "calibration": _load_json(run_dir / "04_ensemble" / "02_calibration_report.json"),
        "bundle": _load_json(run_dir / "05_packaging" / "bundle" / "bundle_manifest.json"),
        "backtest": _load_json(run_dir / "06_backtest" / "01_backtest_report.json"),
        "realworld": _load_json(run_dir / "06_backtest" / "02_realworld_report.json"),
        "diagnostics": _load_json(run_dir / "06_backtest" / "03_regime_diagnostics.json"),
        "trades": _parse_trades_csv(run_dir / "06_backtest" / "trades.csv"),
        "returns_curve": (run_dir / "06_backtest" / "returns_curve.png")
        if (run_dir / "06_backtest" / "returns_curve.png").is_file()
        else None,
    }


def _extract_kpis(d: dict[str, Any]) -> dict[str, Any]:
    bt = d["backtest"] or {}
    rw = d["realworld"] or {}
    dec = d["decision"] or {}
    ing = d["ingestion"] or {}
    sel = d["selected_features"] or {}
    ens = d["ensemble"] or {}
    cal = d["calibration"] or {}
    bnd = d["bundle"] or {}
    cfg = d["config"] or {}
    bt_cfg = bt.get("config") or cfg.get("backtest") or {}
    rw_cfg = cfg.get("realworld") or {}

    gross = bt.get("gross_sharpe")
    net = bt.get("net_sharpe")
    friction_drag = (gross - net) if (gross is not None and net is not None) else None

    label_col = (cfg.get("labels") or {}).get("selected")
    if not label_col and ing.get("label_nan_counts"):
        keys = [k for k in ing["label_nan_counts"] if not str(k).startswith("fwd_ret_")]
        label_col = keys[0] if keys else None

    return {
        "name": d["name"],
        "accepted": bt.get("accepted"),
        "failed_gates": bt.get("failed_gates", []),
        "task_type": (d["task"] or {}).get("task_type") or bnd.get("task_type"),
        "label_col": label_col,
        "n_rows": ing.get("n_joined_rows"),
        "n_raw_features": ing.get("n_features"),
        "n_selected_features": len(sel.get("selected_features", [])) if sel else None,
        "decision": dec.get("decision"),
        "selected_models": ", ".join(dec.get("selected_models", [])) if dec else None,
        "ensemble_ic_sharpe": ens.get("ensemble_ic_sharpe"),
        "temperature": cal.get("temperature"),
        "calibrated_ece": cal.get("calibrated_ece"),
        "onnx_max_diff": bnd.get("max_abs_diff"),
        "rank_ic": bt.get("rank_ic"),
        "icir": bt.get("icir"),
        "gross_sharpe": gross,
        "net_sharpe": net,
        "friction_drag": friction_drag,
        "daily_net_sharpe": bt.get("daily_net_sharpe"),
        "dsr": bt.get("deflated_sharpe_ratio"),
        "mdd": bt.get("max_drawdown"),
        "downside_ic": bt.get("downside_ic"),
        "mean_turnover": bt.get("mean_turnover"),
        "latency_ms": bt.get("latency_ms"),
        "execution_price": bt_cfg.get("execution_price"),
        "fee_slip_bps": (
            (bt_cfg.get("fee_bps", 0) + bt_cfg.get("slippage_bps", 0))
            if "fee_bps" in bt_cfg
            else None
        ),
        "rw_trades": rw.get("n_trades"),
        "rw_win_rate": rw.get("win_rate"),
        "rw_profit_factor": rw.get("profit_factor"),
        "rw_return_pct": rw.get("return_pct"),
        "rw_buy_hold_pct": rw.get("buy_hold_return_pct"),
        "rw_sharpe": rw.get("sharpe_ratio_bt"),
        "rw_sortino": rw.get("sortino_ratio"),
        "rw_calmar": rw.get("calmar_ratio"),
        "rw_mdd_pct": rw.get("max_drawdown_pct"),
        "rw_avg_bars": rw.get("avg_holding_bars"),
        "rw_size": rw_cfg.get("size"),
        "rw_trade_on_close": rw_cfg.get("trade_on_close"),
    }


def render_single_report(d: dict[str, Any], out_file: Path) -> str:
    k = _extract_kpis(d)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = _badge(k["accepted"])
    failed = ", ".join(k["failed_gates"]) if k["failed_gates"] else "None"

    lines: list[str] = [
        f"# Run Insights: `{d['name']}`",
        "",
        f"> **Verdict:** {verdict} | **Task:** `{k['task_type'] or '—'}` | "
        f"**Label:** `{k['label_col'] or '—'}` | **Models:** `{k['selected_models'] or '—'}` | "
        f"**Generated:** {now}",
        f"> **Run Dir:** `{d['path']}` | **Failed Gates:** `{failed}`",
        "",
        "---",
        "",
        "## 1. Executive Scorecard",
        "",
        "| Metric | Value | Notes |",
        "| :--- | ---: | :--- |",
        f"| **OOS Rank IC** | {_fmt(k['rank_ic'])} | Tier 1 blind OOS |",
        f"| **OOS ICIR** | {_fmt(k['icir'])} | Rolling IC stability |",
        f"| **Gross → Net Sharpe** | {_fmt(k['gross_sharpe'], '.2f')} → **{_fmt(k['net_sharpe'], '.2f')}** | Friction drag: {_fmt(k['friction_drag'], '.2f')} |",
        f"| **Daily Net Sharpe** | {_fmt(k['daily_net_sharpe'], '.2f')} | Annualized ×√252 |",
        f"| **Deflated Sharpe (DSR)** | {_fmt(k['dsr'])} | Multiple-testing adjusted |",
        f"| **Tier 1 Max Drawdown** | {_pct(k['mdd'])} | Vectorized OOS |",
        f"| **Downside IC** | {_fmt(k['downside_ic'])} | Stress bars quantile |",
        f"| **Mean Turnover** | {_pct(k['mean_turnover'])} | Per bar |",
        f"| **ONNX Latency / Diff** | {_fmt(k['latency_ms'], '.3f', ' ms')} / {_fmt(k['onnx_max_diff'], '.2e')} | Budget < 5 ms, < 1e-4 |",
        f"| **Tier 2 Return vs B&H** | {_pct(k['rw_return_pct'], 2, True)} vs {_pct(k['rw_buy_hold_pct'], 2, True)} | tempusonebt realistic fill |",
        f"| **Tier 2 Sharpe / Sortino / Calmar** | {_fmt(k['rw_sharpe'], '.2f')} / {_fmt(k['rw_sortino'], '.2f')} / {_fmt(k['rw_calmar'], '.2f')} | Tier 2 risk-adjusted |",
        f"| **Tier 2 Trades / Win Rate / PF** | {_fmt(k['rw_trades'], 'd')} / {_pct(k['rw_win_rate'], 1)} / {_fmt(k['rw_profit_factor'], '.2f')} | MDD: {_pct(k['rw_mdd_pct'], 2, True)} |",
        "",
    ]

    # 2. Configuration Snapshot
    cfg = d["config"] or {}
    bt_cfg = (d["backtest"] or {}).get("config") or cfg.get("backtest") or {}
    rw_cfg = cfg.get("realworld") or {}
    feat_cfg = cfg.get("features") or {}
    mod_cfg = cfg.get("models") or {}
    split_cfg = cfg.get("split") or {}
    lines += [
        "## 2. Configuration Snapshot",
        "",
        "| Group | Key Parameters |",
        "| :--- | :--- |",
        f"| **Data & Split** | `horizon_h={split_cfg.get('horizon_h', (d['split'] or {}).get('horizon_h', '—'))}`, "
        f"`date_from={(cfg.get('data') or {}).get('date_from', (d['ingestion'] or {}).get('date_from', '—'))}`, "
        f"`date_to={(cfg.get('data') or {}).get('date_to', (d['ingestion'] or {}).get('date_to', '—'))}` |",
        f"| **Features** | `icir_min={feat_cfg.get('icir_min', '—')}`, `max_features={feat_cfg.get('max_features', '—')}`, "
        f"`dcor_threshold={feat_cfg.get('dcor_threshold', '—')}`, `max_temporal_drift={feat_cfg.get('max_temporal_drift', '—')}` |",
        f"| **Models** | `enabled={mod_cfg.get('enabled_models', '—')}`, `complexity_penalty={mod_cfg.get('complexity_penalty', '—')}`, "
        f"`refit_on_train_cal={mod_cfg.get('refit_on_train_cal', '—')}` |",
        f"| **Backtest (Tier 1)** | `execution_price={bt_cfg.get('execution_price', '—')}`, "
        f"`fee+slip={bt_cfg.get('fee_bps', '—')}+{bt_cfg.get('slippage_bps', '—')} bps`, "
        f"`top/exit={bt_cfg.get('top_pct', '—')}/{bt_cfg.get('exit_pct', '—')}`, "
        f"`ema={bt_cfg.get('signal_ema_span', '—')}`, `min_hold={bt_cfg.get('min_hold_bars', '—')}` |",
        f"| **RealWorld (Tier 2)** | `enabled={rw_cfg.get('enabled', d['realworld'] is not None)}`, "
        f"`cash={rw_cfg.get('cash', '—')}`, `commission={rw_cfg.get('commission', '—')}`, "
        f"`margin={rw_cfg.get('margin', '—')}`, `size={rw_cfg.get('size', '—')}`, "
        f"`trade_on_close={rw_cfg.get('trade_on_close', '—')}` |",
        "",
    ]

    # 3. Data & Temporal Split
    ing = d["ingestion"] or {}
    spl = d["split"] or {}
    lines += [
        "## 3. Data & Temporal Split (Phase 01)",
        "",
        f"- **Rows:** `{ing.get('n_joined_rows', '—')}` joined | **Raw Features:** `{ing.get('n_features', '—')}` | "
        f"**NaN / Inf:** `{ing.get('n_nan', '—')}` / `{ing.get('n_inf', '—')}` | "
        f"**Range:** `{_short_date(ing.get('t_min'))}` → `{_short_date(ing.get('t_max'))}`",
        f"- **Split Mode:** `{spl.get('mode', '—')}` | **Purge/Embargo `horizon_h`:** `{spl.get('horizon_h', '—')}`",
        "",
    ]
    if spl.get("segments"):
        lines += [
            "| Segment | Rows | Start | End | Embargo Dropped | Class Counts |",
            "| :--- | ---: | :--- | :--- | ---: | :--- |",
        ]
        for seg in spl["segments"]:
            cc = seg.get("class_counts") or {}
            cc_str = ", ".join(f"{ck}:{cv}" for ck, cv in cc.items()) if cc else "—"
            lines.append(
                f"| `{seg.get('name')}` | {seg.get('n_rows', '—')} | {_short_date(seg.get('t_start'))} | "
                f"{_short_date(seg.get('t_end'))} | {seg.get('rows_dropped_before', 0)} | `{cc_str}` |"
            )
        lines.append("")

    # 4. Feature Funnel
    lines += ["## 4. Feature Funnel (Phase 02)", ""]
    if d["funnel"]:
        lines += [
            "| Stage | In | Out | Dropped | Drop % |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for st in d["funnel"]:
            n_in = int(st.get("n_in", 0))
            n_out = int(st.get("n_out", 0))
            dropped = len(st.get("dropped") or {})
            drop_pct = (dropped / n_in * 100.0) if n_in > 0 else 0.0
            lines.append(f"| `{st.get('stage')}` | {n_in} | {n_out} | {dropped} | {drop_pct:.1f}% |")
        lines.append("")
    sel_feats = (d["selected_features"] or {}).get("selected_features")
    if sel_feats:
        lines.append(f"**Selected Features ({len(sel_feats)}):** " + ", ".join(f"`{f}`" for f in sel_feats))
        lines.append("")

    # 5. Model Tournament, Decision & Tuning
    lines += ["## 5. Model Tournament, Decision & Tuning (Phase 03)", ""]
    if d["tournament"]:
        lines += [
            "### Tournament Best Trial per Family",
            "",
            "| Model | Trials | Best IC Sharpe | Mean IC | Fold IC Min | Fold IC Max | Fold IC Std |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in d["tournament"]:
            lines.append(
                f"| `{row['model_name']}` | {row['n_trials']} | {_fmt(row['best_ic_sharpe'])} | "
                f"{_fmt(row['mean_ic'])} | {_fmt(row['fold_min'])} | {_fmt(row['fold_max'])} | "
                f"{_fmt(row['fold_std'])} |"
            )
        lines.append("")

    dec = d["decision"] or {}
    if dec:
        corrs = dec.get("pairwise_correlations") or {}
        corr_str = ", ".join(f"`{ck}={cv:.3f}`" for ck, cv in corrs.items()) if corrs else "—"
        lines += [
            "### Decision Engine",
            "",
            f"- **Decision:** `{dec.get('decision')}` → `{dec.get('selected_models')}`",
            f"- **Diversity Score:** `{_fmt(dec.get('diversity_score'))}` | **Marginal Gain:** `{_fmt(dec.get('marginal_gain'))}` | "
            f"**WF Consistency:** `{_pct(dec.get('wf_consistency'), 1)}`",
            f"- **Pairwise Correlations:** {corr_str}",
            f"- **Rationale:** {dec.get('rationale', '—')}",
            "",
        ]

    tun = d["tuning"] or {}
    if tun.get("tuned_candidates"):
        lines += [
            "### Optuna Tuning",
            "",
            "| Model | Best Score | Best IC Sharpe | Complexity | Trials (Pruned) | Key Params |",
            "| :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
        summaries = tun.get("study_summaries") or {}
        for cand in tun["tuned_candidates"]:
            mname = cand.get("model_name", "—")
            s = summaries.get(mname) or {}
            trials_str = f"{s.get('n_trials', '—')} ({s.get('n_pruned', '—')})"
            bp = cand.get("best_params") or {}
            bp_str = ", ".join(
                f"{pk}={pv:.4g}" if isinstance(pv, float) else f"{pk}={pv}" for pk, pv in bp.items()
            )
            lines.append(
                f"| `{mname}` | {_fmt(cand.get('best_score'))} | {_fmt(cand.get('best_ic_sharpe'))} | "
                f"{_fmt(cand.get('complexity_score'))} | {trials_str} | `{bp_str}` |"
            )
        lines.append("")

    # 6. Ensemble, Calibration & Packaging
    ens = d["ensemble"] or {}
    cal = d["calibration"] or {}
    bnd = d["bundle"] or {}
    lines += [
        "## 6. Ensemble, Calibration & Packaging (Phase 04–05)",
        "",
        "| Stage | Key Metrics |",
        "| :--- | :--- |",
        f"| **Ensemble (Cal-A)** | `weights={ens.get('weights', '—')}`, `ic_sharpe={_fmt(ens.get('ensemble_ic_sharpe'))}`, "
        f"`overall_rank_ic={_fmt(ens.get('overall_rank_ic'))}`, `hurdle={_pct(ens.get('hurdle_ratio_achieved'), 2)}`, "
        f"`fallback={ens.get('fallback_reason')}` |",
        f"| **Calibration (Cal-B)** | `method={cal.get('method', '—')}`, `T={_fmt(cal.get('temperature'))}`, "
        f"`ECE: {_fmt(cal.get('uncalibrated_ece'))} → {_fmt(cal.get('calibrated_ece'))}`, "
        f"`rank_ic_preserved={cal.get('rank_ic_preserved', '—')}` |",
        f"| **ONNX Bundle (Val)** | `n_models={bnd.get('n_models', '—')}`, `max_abs_diff={_fmt(bnd.get('max_abs_diff'), '.2e')}` "
        f"(< 1e-4: {_badge(bnd.get('max_abs_diff') < 1e-4 if bnd.get('max_abs_diff') is not None else None)}) |",
        "",
    ]

    # 7. Backtest Evaluation (Phase 06)
    bt = d["backtest"] or {}
    rw = d["realworld"] or {}
    lines += ["## 7. Backtest Evaluation (Phase 06)", ""]
    if bt:
        gates = bt.get("gates") or {}
        bcfg = bt.get("config") or {}
        lines += [
            "### Tier 1 — Vectorized OOS Hard Gates",
            "",
            "| Gate | Actual | Threshold | Status |",
            "| :--- | ---: | ---: | :---: |",
            f"| Rank IC | {_fmt(bt.get('rank_ic'))} | > {_fmt(bcfg.get('min_rank_ic'))} | {_badge(gates.get('rank_ic'))} |",
            f"| ICIR | {_fmt(bt.get('icir'))} | > {_fmt(bcfg.get('min_icir'))} | {_badge(gates.get('icir'))} |",
            f"| Net Sharpe | {_fmt(bt.get('net_sharpe'), '.2f')} | > {_fmt(bcfg.get('min_net_sharpe'), '.2f')} | {_badge(gates.get('net_sharpe'))} |",
            f"| Max Drawdown | {_pct(bt.get('max_drawdown'))} | < {_pct(bcfg.get('max_mdd'))} | {_badge(gates.get('mdd'))} |",
            f"| Downside IC | {_fmt(bt.get('downside_ic'))} | > {_fmt(bcfg.get('min_downside_ic'), '.2f')} | {_badge(gates.get('downside_ic'))} |",
            f"| Latency | {_fmt(bt.get('latency_ms'), '.3f', ' ms')} | < 5.000 ms | {_badge(gates.get('latency'))} |",
            "",
            f"- **OOS Bars:** `{bt.get('n_bars', '—')}` | **Stress Bars:** `{bt.get('n_stress_bars', '—')}` | "
            f"**Overnight Gaps:** `{bt.get('n_overnight_gaps', '—')}` | **Periods/Year:** `{_fmt(bt.get('periods_per_year'), '.1f')}`",
            "",
        ]

    if rw:
        lines += [
            "### Tier 2 — Realistic Execution (`tempusonebt`)",
            "",
            "| Metric | Value | Metric | Value |",
            "| :--- | ---: | :--- | ---: |",
            f"| Total Trades | {_fmt(rw.get('n_trades'), 'd')} | Equity Final / Peak | {_fmt(rw.get('equity_final'), ',.2f')} / {_fmt(rw.get('equity_peak'), ',.2f')} |",
            f"| Win Rate | {_pct(rw.get('win_rate'), 2)} | Return % vs Buy & Hold | {_pct(rw.get('return_pct'), 2, True)} vs {_pct(rw.get('buy_hold_return_pct'), 2, True)} |",
            f"| Profit Factor | {_fmt(rw.get('profit_factor'), '.3f')} | Sharpe / Sortino / Calmar | {_fmt(rw.get('sharpe_ratio_bt'), '.2f')} / {_fmt(rw.get('sortino_ratio'), '.2f')} / {_fmt(rw.get('calmar_ratio'), '.2f')} |",
            f"| Avg Holding Bars | {_fmt(rw.get('avg_holding_bars'), '.2f')} | Max Drawdown % | {_pct(rw.get('max_drawdown_pct'), 2, True)} |",
            f"| Worst Single Trade | -{_pct(rw.get('max_trade_loss_pct'), 2, True)} | — | — |",
            "",
        ]

    tr = d["trades"]
    if tr and tr.get("n", 0) > 0:
        lg, sh = tr["long"], tr["short"]
        lines += [
            "### Trade Log Breakdown (`trades.csv`)",
            "",
            "| Side | Trades | Wins | Win Rate | Total PnL | Avg PnL |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
            f"| Long | {lg['n']} | {lg['wins']} | {_pct(lg['win_rate'], 1)} | {_fmt(lg['total_pnl'], ',.2f')} | {_fmt(lg['avg_pnl'], ',.2f')} |",
            f"| Short | {sh['n']} | {sh['wins']} | {_pct(sh['win_rate'], 1)} | {_fmt(sh['total_pnl'], ',.2f')} | {_fmt(sh['avg_pnl'], ',.2f')} |",
            f"| **All** | **{tr['n']}** | **{lg['wins'] + sh['wins']}** | **{_pct((lg['wins'] + sh['wins']) / tr['n'], 1)}** | **{_fmt(tr['total_pnl'], ',.2f')}** | **{_fmt(tr['total_pnl'] / tr['n'], ',.2f')}** |",
            "",
            f"- **Total Commission:** `{_fmt(tr.get('total_commission'), ',.2f')}` | "
            f"**Best / Worst Trade Return:** `{_pct(tr.get('best_ret_pct'), 2, True)}` / `{_pct(tr.get('worst_ret_pct'), 2, True)}` | "
            f"**Median / Max Hold (bars):** `{_fmt(tr.get('median_bars'), '.1f')}` / `{tr.get('max_bars', '—')}`",
            "",
        ]

    if d["returns_curve"] is not None:
        rel_img = os.path.relpath(d["returns_curve"], out_file.resolve().parent).replace("\\", "/")
        lines += [
            "### Cumulative PnL Curve",
            "",
            f"![Cumulative PnL]({rel_img})",
            "",
        ]

    diag = d["diagnostics"]
    if diag and isinstance(diag.get("segments"), dict):
        lines += [
            "### Cross-Regime Diagnostics (`03_regime_diagnostics.json`)",
            "",
            "| Segment | Mode | Status | Rank IC | ICIR | Gross Sharpe | Net Sharpe | Daily Net Sharpe | MDD | Accepted |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
        for seg_name, seg_val in diag["segments"].items():
            for mode in ("close", "next_open"):
                m = (seg_val or {}).get(mode) or {}
                if not m:
                    continue
                lines.append(
                    f"| `{seg_name}` | `{mode}` | {m.get('status', '—')} | {_fmt(m.get('rank_ic'))} | "
                    f"{_fmt(m.get('icir'))} | {_fmt(m.get('gross_sharpe'), '.2f')} | "
                    f"{_fmt(m.get('net_sharpe'), '.2f')} | {_fmt(m.get('daily_net_sharpe'), '.2f')} | "
                    f"{_pct(m.get('max_drawdown'))} | {_badge(m.get('quant_accepted'))} |"
                )
        lines.append("")

    # 8. Placeholder for Agent Qualitative Synthesis
    lines += [
        "## 8. Quant Insights & Next Actions",
        "",
        "<!-- AGENT_INSIGHTS_SECTION: Replace this placeholder with concise quantitative diagnosis -->",
        "- **Strengths:** _TBD_",
        "- **Risks & Bottlenecks:** _TBD_",
        "- **Next Experiments:** _TBD_",
        "",
    ]
    return "\n".join(lines)


def render_comparison_report(runs: list[dict[str, Any]]) -> str:
    kpis = [_extract_kpis(r) for r in runs]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    names = [k["name"] for k in kpis]
    header = "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    sep = "| :--- | " + " | ".join("---:" for _ in names) + " |"

    rows_spec: list[tuple[str, Any]] = [
        ("Verdict", lambda k: _badge(k["accepted"])),
        ("Label", lambda k: f"`{k['label_col'] or '—'}`"),
        ("Task Type", lambda k: f"`{k['task_type'] or '—'}`"),
        ("Raw → Selected Features", lambda k: f"{k['n_raw_features'] or '—'} → {k['n_selected_features'] or '—'}"),
        ("Selected Models", lambda k: f"`{k['selected_models'] or '—'}`"),
        ("Decision", lambda k: f"`{k['decision'] or '—'}`"),
        ("Cal-A Ensemble IC Sharpe", lambda k: _fmt(k["ensemble_ic_sharpe"])),
        ("Calibration T / ECE", lambda k: f"{_fmt(k['temperature'], '.3f')} / {_fmt(k['calibrated_ece'])}"),
        ("ONNX Max Diff", lambda k: _fmt(k["onnx_max_diff"], ".2e")),
        ("**OOS Rank IC**", lambda k: f"**{_fmt(k['rank_ic'])}**"),
        ("**OOS ICIR**", lambda k: f"**{_fmt(k['icir'])}**"),
        ("Gross Sharpe", lambda k: _fmt(k["gross_sharpe"], ".2f")),
        ("**Net Sharpe (Tier 1)**", lambda k: f"**{_fmt(k['net_sharpe'], '.2f')}**"),
        ("Friction Drag (Gross-Net)", lambda k: _fmt(k["friction_drag"], ".2f")),
        ("**Daily Net Sharpe**", lambda k: f"**{_fmt(k['daily_net_sharpe'], '.2f')}**"),
        ("Deflated Sharpe (DSR)", lambda k: _fmt(k["dsr"])),
        ("Tier 1 MDD", lambda k: _pct(k["mdd"])),
        ("Downside IC", lambda k: _fmt(k["downside_ic"])),
        ("Mean Turnover", lambda k: _pct(k["mean_turnover"])),
        ("Latency (ms)", lambda k: _fmt(k["latency_ms"], ".3f")),
        ("Execution Price / Fee+Slip", lambda k: f"`{k['execution_price'] or '—'}` / {_fmt(k['fee_slip_bps'], '.1f', ' bps')}"),
        ("**Tier 2 Return %**", lambda k: f"**{_pct(k['rw_return_pct'], 2, True)}**"),
        ("Tier 2 Buy & Hold %", lambda k: _pct(k["rw_buy_hold_pct"], 2, True)),
        ("**Tier 2 Sharpe (BT)**", lambda k: f"**{_fmt(k['rw_sharpe'], '.2f')}**"),
        ("Tier 2 Sortino / Calmar", lambda k: f"{_fmt(k['rw_sortino'], '.2f')} / {_fmt(k['rw_calmar'], '.2f')}"),
        ("Tier 2 MDD %", lambda k: _pct(k["rw_mdd_pct"], 2, True)),
        ("Tier 2 Trades", lambda k: _fmt(k["rw_trades"], "d")),
        ("Tier 2 Win Rate", lambda k: _pct(k["rw_win_rate"], 1)),
        ("Tier 2 Profit Factor", lambda k: _fmt(k["rw_profit_factor"], ".2f")),
        ("Tier 2 Avg Hold (bars)", lambda k: _fmt(k["rw_avg_bars"], ".1f")),
    ]

    lines = [
        f"# Run Comparison: {' vs '.join(f'`{n}`' for n in names)}",
        "",
        f"> **Generated:** {now} | **Runs Compared:** {len(runs)}",
        "",
        "## 1. Side-by-Side KPI Scorecard",
        "",
        header,
        sep,
    ]
    for label, fn in rows_spec:
        vals = " | ".join(str(fn(k)) for k in kpis)
        lines.append(f"| {label} | {vals} |")

    lines += [
        "",
        "## 2. Comparative Insights & Recommendation",
        "",
        "<!-- AGENT_INSIGHTS_SECTION: Replace this placeholder with comparative analysis -->",
        "- **Winner & Why:** _TBD_",
        "- **Key Drivers of Difference:** _TBD_",
        "- **Recommended Next Run:** _TBD_",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract standardized QuantAutoML run insights.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="One or more run output directories")
    parser.add_argument("--out", type=Path, default=None, help="Output markdown path (default: insights/<name>.md)")
    parser.add_argument("--compare", action="store_true", help="Force multi-run comparison table mode")
    args = parser.parse_args()

    for rd in args.run_dirs:
        if not rd.is_dir():
            raise SystemExit(f"Run directory not found: {rd}")

    runs = [load_run_data(rd) for rd in args.run_dirs]
    is_compare = args.compare or len(runs) > 1

    if args.out is not None:
        out_path = args.out
    elif is_compare:
        slug = "_vs_".join(r["name"] for r in runs)
        out_path = Path("insights") / f"compare_{slug}.md"
    else:
        out_path = Path("insights") / f"{runs[0]['name']}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_comparison_report(runs) if is_compare else render_single_report(runs[0], out_path)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote insights report: {out_path}")


if __name__ == "__main__":
    main()
