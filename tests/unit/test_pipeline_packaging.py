"""run_pipeline writes an ONNX bundle when [packaging] is configured."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantautoml.packaging import ONNXBundleExporter
from quantautoml.pipeline import run_pipeline

N, P = 12000, 5
PIPELINE_TOML = """
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
"""


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    ts = np.datetime64("2024-01-01T00:00:00", "us") + np.arange(N).astype("timedelta64[s]")
    feats: dict[str, np.ndarray] = {"timestamp": ts}
    for i in range(P):
        feats[f"feat_{i}"] = rng.standard_normal(N).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-0.6 * feats["feat_0"]))
    y = rng.binomial(1, prob).astype(np.int32)
    feat_path, lab_path = tmp_path / "f.parquet", tmp_path / "l.parquet"
    pl.DataFrame(feats).write_parquet(feat_path)
    pl.DataFrame({"timestamp": ts, "target": y}).write_parquet(lab_path)
    return feat_path, lab_path


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pkg")
    feat, lab = _write_inputs(tmp)
    toml = PIPELINE_TOML.format(feat=feat, lab=lab) + "\n[packaging]\n"
    cfg = tmp / "cfg.toml"
    cfg.write_text(toml)
    out = tmp / "out"
    return run_pipeline(cfg, out), out, tmp, feat, lab


def test_pipeline_writes_bundle_when_packaging_configured(run) -> None:
    result, out, *_ = run
    names = {p.name for p in (out / "05_packaging" / "bundle").iterdir()}
    assert {"preprocessor.onnx", "model_0.onnx", "ensemble_meta.json", "features_schema.json",
            "predictor.py", "bundle_manifest.json"} <= names
    assert result.bundle_manifest is not None
    assert result.backtest_metrics is None  # no [backtest] section -> no backtest stage
    assert not (out / "06_backtest").exists()


def test_pipeline_bundle_precision_ok(run) -> None:
    _, out, *_ = run
    manifest = json.loads((out / "05_packaging" / "bundle" / "bundle_manifest.json").read_text())
    assert manifest["max_abs_diff"] < 1e-4


def test_pipeline_skips_bundle_without_packaging_section(tmp_path: Path) -> None:
    feat, lab = _write_inputs(tmp_path)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(PIPELINE_TOML.format(feat=feat, lab=lab))
    result = run_pipeline(cfg, tmp_path / "out")
    assert result.bundle_manifest is None and not (tmp_path / "out" / "05_packaging").exists()


def test_pipeline_packaging_gets_train_and_val_only(tmp_path: Path, monkeypatch) -> None:
    feat, lab = _write_inputs(tmp_path)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(PIPELINE_TOML.format(feat=feat, lab=lab) + "\n[packaging]\n")
    seen: dict[str, np.ndarray] = {}
    orig = ONNXBundleExporter.export

    def spy(self, **kw):  # type: ignore[no-untyped-def]
        seen.update({k: v for k, v in kw.items() if isinstance(v, np.ndarray)})
        return orig(self, **kw)

    monkeypatch.setattr(ONNXBundleExporter, "export", spy)
    result = run_pipeline(cfg, tmp_path / "out")
    assert seen["X_train_raw"] is result.split.train.X
    assert seen["X_val_raw"] is result.split.val.X
    assert not any(np.shares_memory(a, result.split.oos.X) for a in seen.values())
