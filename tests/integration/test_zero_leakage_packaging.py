"""RULE-01 for Phase 5: packaging/backtest never fit, and OOS cannot influence the bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from bundle_helper import make_bundle

from quantautoml.backtest import OOSBacktestRunner
from quantautoml.common.config import BacktestConfig
from quantautoml.ensemble import TemperatureCalibrator
from quantautoml.features.base import BaseFeatureFilter
from quantautoml.models.candidates import CatBoostCandidate, LightGBMCandidate, XGBoostCandidate
from quantautoml.pipeline import run_pipeline
from test_friction_backtest import NAMES, _wobbly_oos

N = 12000
TOML = """
[data]
features_path = "{feat}"
labels_path = "{lab}"
[labels]
selected = "target"
[split]
horizon_h = 5
[models]
tournament_trials = 2
tuning_trials = 2
n_wf_folds = 2
enabled_models = ["lightgbm"]
[ensemble]
n_bootstrap = 10
[packaging]
[backtest]
price_col = "px"
rank_window = 200
ic_window = 300
"""


def _write(tmp: Path, mutate: np.ndarray | None = None) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(N).astype("timedelta64[m]") * 5
    f0 = rng.standard_normal(N).astype(np.float32)
    ret = 0.001 * np.tanh(f0) + 0.0005 * rng.standard_normal(N)
    px = (100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))).astype(np.float32)
    feats = {"timestamp": ts, "feat_0": f0, "px": px}
    for i in range(1, 4):
        feats[f"feat_{i}"] = rng.standard_normal(N).astype(np.float32)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.6 * f0))).astype(np.int32)
    if mutate is not None:  # wreck every feature value on the OOS rows, labels untouched
        for k in ("feat_0", "feat_1", "feat_2", "feat_3"):
            feats[k] = feats[k].copy()
            feats[k][mutate] = 1e4
        feats["px"] = feats["px"].copy()
        feats["px"][mutate] = 7.0
    feat, lab = tmp / "f.parquet", tmp / "l.parquet"
    pl.DataFrame(feats).write_parquet(feat)
    pl.DataFrame({"timestamp": ts, "target": y}).write_parquet(lab)
    return feat, lab


def _cfg(tmp: Path, feat: Path, lab: Path) -> Path:
    p = tmp / "cfg.toml"
    p.write_text(TOML.format(feat=feat, lab=lab))
    return p


@pytest.fixture(scope="module")
def run_a(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a")
    feat, lab = _write(tmp)
    return run_pipeline(_cfg(tmp, feat, lab), tmp / "out"), tmp / "out"


def test_end_to_end_raw_data_to_bundle_and_report(run_a) -> None:
    result, out = run_a
    assert (out / "bundle" / "predictor.py").exists()
    rep = json.loads((out / "backtest_report.json").read_text())
    assert result.backtest_metrics is not None
    assert rep["accepted"] == result.backtest_metrics.accepted
    assert {"rank_ic", "net_sharpe", "max_drawdown", "downside_ic", "gates"} <= set(rep)


def test_oos_cannot_influence_bundle(run_a, tmp_path: Path) -> None:
    result_a, out_a = run_a
    feat, lab = _write(tmp_path, mutate=result_a.split.oos.index)
    run_pipeline(_cfg(tmp_path, feat, lab), tmp_path / "out")
    for name in ("ensemble_meta.json", "features_schema.json"):
        assert (out_a / "bundle" / name).read_text() == (tmp_path / "out" / "bundle" / name).read_text()


def test_packaging_and_backtest_never_call_fit(tmp_path: Path, monkeypatch) -> None:
    fx = make_bundle(tmp_path)

    def boom(*a, **k):
        raise AssertionError("fit() called after training (RULE-01)")

    for cls in (LightGBMCandidate, XGBoostCandidate, CatBoostCandidate, TemperatureCalibrator,
                BaseFeatureFilter):
        monkeypatch.setattr(cls, "fit", boom)
    OOSBacktestRunner(BacktestConfig(rank_window=100, ic_window=300)).run(
        bundle_dir=fx.dir, oos=_wobbly_oos(), feature_names=NAMES, other_segments=[]
    )


def test_pipeline_passes_packaging_latency_knobs_to_runner(tmp_path: Path, monkeypatch) -> None:
    feat, lab = _write(tmp_path)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        TOML.format(feat=feat, lab=lab).replace(
            "[packaging]", "[packaging]\nlatency_budget_ms = 2.5\nlatency_repeats = 7"
        )
    )
    seen: dict = {}
    orig = OOSBacktestRunner.run

    def spy(self, **kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        return orig(self, **kw)

    monkeypatch.setattr(OOSBacktestRunner, "run", spy)
    run_pipeline(cfg, tmp_path / "out")
    assert seen["latency_budget_ms"] == 2.5 and seen["latency_repeats"] == 7
