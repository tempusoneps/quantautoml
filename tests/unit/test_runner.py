"""OOSBacktestRunner: blind OOS run through the standalone predictor with friction + buffer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from bundle_helper import make_bundle

from quantautoml.backtest.runner import OOSBacktestRunner
from quantautoml.common.config import BacktestConfig
from quantautoml.common.exceptions import BacktestError, DataLeakageError, SchemaMismatchError
from quantautoml.common.types import Segment

NAMES = ["x0", "x1", "x2", "x3", "x4", "Close"]
CFG = BacktestConfig(rank_window=100, ic_window=300)


def _oos(n: int = 3000, *, signal: bool, seed: int = 5, start: int = 10_000) -> Segment:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    s = X[:, 0] + 0.5 * X[:, 1]
    y = ((s + 0.5 * rng.normal(size=n)) > np.median(s)).astype(np.int32)
    drive = np.tanh(X[:, 0]) if signal else rng.normal(size=n)
    ret = 0.001 * drive + 0.0005 * rng.normal(size=n)  # return from bar t to t+1
    close = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + ret[:-1])))
    X[:, 5] = close.astype(np.float32)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(n) * np.timedelta64(5, "m")
    return Segment(X=X, y=y, labels_aux={}, timestamps=ts, index=np.arange(start, start + n))


def _other(n: int = 500) -> Segment:
    z = np.zeros((n, 6), dtype=np.float32)
    return Segment(X=z, y=np.zeros(n, np.int32), labels_aux={}, timestamps=np.array([], "M8[us]"),
                   index=np.arange(n))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("bt"))


def _run(bundle, oos: Segment, cfg: BacktestConfig = CFG, out: Path | None = None, **kw):
    metrics, _ = OOSBacktestRunner(cfg).run(
        bundle_dir=bundle.dir, oos=oos, feature_names=kw.pop("names", NAMES),
        other_segments=kw.pop("others", [_other()]), out_path=out, **kw,
    )
    return metrics


def test_runner_accepts_strong_signal(bundle, tmp_path: Path) -> None:
    m = _run(bundle, _oos(signal=True), out=tmp_path / "r.json")
    assert m.accepted, (m.failed_gates, m.model_dump())
    assert m.net_sharpe < m.gross_sharpe  # friction was charged


def test_runner_rejects_pure_noise_signal(bundle) -> None:
    m = _run(bundle, _oos(signal=False))
    assert not m.accepted and m.failed_gates


def test_report_written_with_gates_and_extras(bundle, tmp_path: Path) -> None:
    out = tmp_path / "backtest_report.json"
    m = _run(bundle, _oos(signal=True), out=out)
    rep = json.loads(out.read_text())
    assert rep["accepted"] == m.accepted and set(rep["gates"]) >= {"net_sharpe", "latency"}
    assert rep["periods_per_year"] > 0 and rep["n_overnight_gaps"] == 0
    assert rep["config"]["fee_bps"] == CFG.fee_bps


def test_gap_count_reported(bundle, tmp_path: Path) -> None:
    oos = _oos(signal=True)
    ts = oos.timestamps.copy()
    for i in (1000, 2000, 2500):
        ts[i:] = ts[i:] + np.timedelta64(16, "h")
    out = tmp_path / "r.json"
    _run(bundle, Segment(oos.X, oos.y, {}, ts, oos.index), out=out)
    assert json.loads(out.read_text())["n_overnight_gaps"] == 3


def test_oos_overlap_raises_dataleakage(bundle) -> None:
    with pytest.raises(DataLeakageError):
        _run(bundle, _oos(signal=True, start=100))  # index 100.. overlaps other segment 0..499


def test_missing_price_col_raises(bundle) -> None:
    with pytest.raises(BacktestError, match="Close"):
        _run(bundle, _oos(signal=True), names=["a", "b", "c", "d", "e", "f"])


def test_nonpositive_close_raises(bundle) -> None:
    oos = _oos(signal=True)
    X = oos.X.copy()
    X[10, 5] = 0.0
    with pytest.raises(BacktestError, match="Close"):
        _run(bundle, Segment(X, oos.y, {}, oos.timestamps, oos.index))


def test_nan_close_raises(bundle) -> None:
    oos = _oos(signal=True)
    X = oos.X.copy()
    X[10, 5] = np.nan
    with pytest.raises(BacktestError, match="Close"):
        _run(bundle, Segment(X, oos.y, {}, oos.timestamps, oos.index))


def test_oos_shorter_than_window_raises(bundle) -> None:
    with pytest.raises(BacktestError, match="rank_window"):
        _run(bundle, _oos(n=50, signal=True))


def test_runner_does_not_modify_bundle_files(bundle) -> None:
    def snapshot() -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.dir.iterdir()}

    before = snapshot()
    _run(bundle, _oos(signal=True))
    assert snapshot() == before


def test_feature_name_order_mismatch_raises(bundle) -> None:
    with pytest.raises(SchemaMismatchError):
        _run(bundle, _oos(signal=True), names=["x1", "x0", "x2", "x3", "x4", "Close"])


def test_latency_budget_is_configurable(bundle) -> None:
    ok = _run(bundle, _oos(signal=True), latency_budget_ms=5.0)
    tight = _run(bundle, _oos(signal=True), latency_budget_ms=1e-9)
    assert ok.gates["latency"] and not tight.gates["latency"]


def test_latency_repeats_is_configurable(bundle, monkeypatch) -> None:
    calls = {"n": 0}
    import quantautoml.backtest.runner as r

    orig = r._latency_ms

    def spy(predictor, row, repeats):  # type: ignore[no-untyped-def]
        calls["n"] = repeats
        return orig(predictor, row, repeats)

    monkeypatch.setattr(r, "_latency_ms", spy)
    _run(bundle, _oos(signal=True), latency_repeats=7)
    assert calls["n"] == 7
