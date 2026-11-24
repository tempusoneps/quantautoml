"""Multi-Window Walk-Forward OOS evaluator with strict Purging & Embargo (RULE-01)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quantautoml.common.exceptions import DataLeakageError, InvalidDataError
from quantautoml.common.types import Dataset, Segment


@dataclass(frozen=True)
class WalkForwardWindowSpec:
    """Chronological boundaries for one expanding walk-forward OOS window.

    ``train_end`` is the exclusive end of the training prefix ``[0, train_end)``;
    ``[oos_start, oos_end)`` is the blind OOS slice, separated by exactly ``horizon_h``
    purged rows (``oos_start - train_end == horizon_h``).
    """

    window_id: int
    train_end: int
    oos_start: int
    oos_end: int


class WalkForwardWindowResult(BaseModel):
    """Per-window OOS evaluation summary."""

    model_config = ConfigDict(frozen=True)

    window_id: int
    train_n_rows: int
    oos_n_rows: int
    t_start: str
    t_end: str
    status: Literal["ok", "skipped"]
    reason: str | None = None
    rank_ic: float | None = None
    icir: float | None = None
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    daily_net_sharpe: float | None = None
    max_drawdown: float | None = None
    mean_turnover: float | None = None
    downside_ic: float | None = None
    accepted: bool | None = None


class WalkForwardOOSReport(BaseModel):
    """Aggregated multi-window walk-forward OOS report."""

    model_config = ConfigDict(frozen=True)

    n_windows: int
    horizon_h: int
    n_completed: int
    n_accepted: int
    acceptance_rate: float
    mean_rank_ic: float
    std_rank_ic: float
    mean_icir: float
    mean_net_sharpe: float
    min_net_sharpe: float
    worst_mdd: float
    positive_sharpe_ratio: float
    windows: list[WalkForwardWindowResult] = Field(default_factory=list)


WindowEvaluator = Callable[[WalkForwardWindowSpec, Dataset, Segment], WalkForwardWindowResult]


def generate_walkforward_windows(
    n_rows: int,
    *,
    n_windows: int,
    horizon_h: int,
    min_train_ratio: float = 0.5,
    min_oos_bars: int = 1,
) -> list[WalkForwardWindowSpec]:
    """Generate sequential expanding walk-forward OOS windows with exact ``horizon_h`` purging."""
    if n_windows < 2:
        raise InvalidDataError(f"n_windows must be >= 2, got {n_windows}")
    if horizon_h < 1:
        raise InvalidDataError(f"horizon_h must be >= 1, got {horizon_h}")
    if not 0.0 < min_train_ratio < 1.0:
        raise InvalidDataError(f"min_train_ratio must be in (0, 1), got {min_train_ratio}")
    if min_oos_bars < 1:
        raise InvalidDataError(f"min_oos_bars must be >= 1, got {min_oos_bars}")

    init_end = int(n_rows * min_train_ratio)
    if init_end <= horizon_h:
        raise InvalidDataError(
            f"n_rows={n_rows} with min_train_ratio={min_train_ratio} leaves no train prefix after horizon_h={horizon_h}"
        )
    rem = n_rows - init_end
    if rem < n_windows * min_oos_bars:
        raise InvalidDataError(
            f"remaining {rem} rows cannot form {n_windows} OOS windows of >= {min_oos_bars} bars"
        )

    windows: list[WalkForwardWindowSpec] = []
    for k in range(n_windows):
        oos_start = init_end + (k * rem) // n_windows
        oos_end = init_end + ((k + 1) * rem) // n_windows if k < n_windows - 1 else n_rows
        train_end = oos_start - horizon_h
        if train_end <= 0:
            raise InvalidDataError(f"window {k}: non-positive train_end={train_end}")
        if oos_end - oos_start < min_oos_bars:
            raise InvalidDataError(
                f"window {k}: OOS length {oos_end - oos_start} < min_oos_bars={min_oos_bars}"
            )
        if oos_start - train_end != horizon_h:
            raise DataLeakageError(
                f"window {k}: gap {oos_start - train_end} != horizon_h={horizon_h} (RULE-01)"
            )
        windows.append(
            WalkForwardWindowSpec(
                window_id=k,
                train_end=train_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
    return windows


class WalkForwardOOSRunner:
    """Evaluates a strategy across multiple sequential OOS windows with purging (RULE-01)."""

    def __init__(
        self,
        *,
        n_windows: int = 3,
        horizon_h: int = 21,
        min_train_ratio: float = 0.5,
        min_oos_bars: int = 200,
    ) -> None:
        self.n_windows = n_windows
        self.horizon_h = horizon_h
        self.min_train_ratio = min_train_ratio
        self.min_oos_bars = min_oos_bars

    def run(
        self,
        dataset: Dataset,
        *,
        window_evaluator: WindowEvaluator,
        out_path: str | Path | None = None,
    ) -> WalkForwardOOSReport:
        """Slice ``dataset`` chronologically, enforce zero-leakage invariants, and aggregate metrics."""
        ts = dataset.timestamps
        if ts.size < 2 or np.any(np.diff(ts) <= np.timedelta64(0, "ns")):
            raise DataLeakageError("Dataset timestamps must be strictly increasing (RULE-01)")

        specs = generate_walkforward_windows(
            n_rows=len(ts),
            n_windows=self.n_windows,
            horizon_h=self.horizon_h,
            min_train_ratio=self.min_train_ratio,
            min_oos_bars=self.min_oos_bars,
        )

        results: list[WalkForwardWindowResult] = []
        for spec in specs:
            if not ts[spec.train_end - 1] < ts[spec.oos_start]:
                raise DataLeakageError(
                    f"window {spec.window_id}: training prefix timestamp overlaps OOS (RULE-01)"
                )
            prefix_ds = _slice_dataset(dataset, spec.train_end)
            oos_idx = np.arange(spec.oos_start, spec.oos_end, dtype=np.int64)
            oos_seg = Segment(
                X=dataset.X[oos_idx],
                y=None if dataset.y is None else dataset.y[oos_idx],
                labels_aux={k: v[oos_idx] for k, v in dataset.labels_aux.items()},
                timestamps=dataset.timestamps[oos_idx],
                index=oos_idx,
            )
            results.append(window_evaluator(spec, prefix_ds, oos_seg))

        report = _aggregate_report(results, n_windows=len(specs), horizon_h=self.horizon_h)
        if out_path is not None:
            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
        return report


def _slice_dataset(ds: Dataset, stop: int) -> Dataset:
    return Dataset(
        X=ds.X[:stop],
        y=None if ds.y is None else ds.y[:stop],
        labels_aux={k: v[:stop] for k, v in ds.labels_aux.items()},
        timestamps=ds.timestamps[:stop],
        feature_names=ds.feature_names,
        selected_label=ds.selected_label,
        report=ds.report,
        task_type=ds.task_type,
        classes_=ds.classes_,
    )


def _aggregate_report(
    windows: list[WalkForwardWindowResult], *, n_windows: int, horizon_h: int
) -> WalkForwardOOSReport:
    ok = [w for w in windows if w.status == "ok"]
    n_ok = len(ok)
    if n_ok == 0:
        return WalkForwardOOSReport(
            n_windows=n_windows,
            horizon_h=horizon_h,
            n_completed=0,
            n_accepted=0,
            acceptance_rate=0.0,
            mean_rank_ic=0.0,
            std_rank_ic=0.0,
            mean_icir=0.0,
            mean_net_sharpe=0.0,
            min_net_sharpe=0.0,
            worst_mdd=0.0,
            positive_sharpe_ratio=0.0,
            windows=windows,
        )
    ics = [w.rank_ic for w in ok if w.rank_ic is not None]
    icirs = [w.icir for w in ok if w.icir is not None]
    sharpes = [w.net_sharpe for w in ok if w.net_sharpe is not None]
    mdds = [w.max_drawdown for w in ok if w.max_drawdown is not None]
    n_acc = sum(1 for w in ok if w.accepted is True)
    pos_sr = sum(1 for s in sharpes if s > 0.0)
    return WalkForwardOOSReport(
        n_windows=n_windows,
        horizon_h=horizon_h,
        n_completed=n_ok,
        n_accepted=n_acc,
        acceptance_rate=float(n_acc / n_ok),
        mean_rank_ic=float(np.mean(ics)) if ics else 0.0,
        std_rank_ic=float(np.std(ics, ddof=1)) if len(ics) >= 2 else 0.0,
        mean_icir=float(np.mean(icirs)) if icirs else 0.0,
        mean_net_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
        min_net_sharpe=float(np.min(sharpes)) if sharpes else 0.0,
        worst_mdd=float(np.max(mdds)) if mdds else 0.0,
        positive_sharpe_ratio=float(pos_sr / len(sharpes)) if sharpes else 0.0,
        windows=windows,
    )
