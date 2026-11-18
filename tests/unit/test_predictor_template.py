"""Standalone predictor: numpy/onnxruntime only, matches Python reference, drift + latency."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from bundle_helper import TEMPLATE, make_bundle

ALLOWED_IMPORTS = {"numpy", "onnxruntime", "json", "warnings", "pathlib", "__future__"}


def _load(bundle_dir: Path):
    spec = importlib.util.spec_from_file_location("bundle_predictor", bundle_dir / "predictor.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_imports_only_numpy_onnxruntime_stdlib() -> None:
    tree = ast.parse(TEMPLATE.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= ALLOWED_IMPORTS, roots - ALLOWED_IMPORTS


def test_runs_in_isolated_subprocess_without_training_libs(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    np.save(tmp_path / "x.npy", fx.X_raw[:5])
    code = (
        "import sys, numpy as np, importlib.util as u\n"
        f"s = u.spec_from_file_location('p', r'{fx.dir / 'predictor.py'}')\n"
        "m = u.module_from_spec(s); s.loader.exec_module(m)\n"
        f"p = m.Predictor(r'{fx.dir}')\n"
        f"print(p.predict(np.load(r'{tmp_path / 'x.npy'}')).shape)\n"
        "bad = [k for k in ('quantautoml','sklearn','lightgbm','xgboost','catboost','optuna','torch')"
        " if k in sys.modules]\n"
        "print('BAD', bad)\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "BAD []" in r.stdout


@pytest.mark.parametrize("n_classes", [2, 3])
def test_matches_python_reference_pipeline(tmp_path: Path, n_classes: int) -> None:
    fx = make_bundle(tmp_path, n_classes=n_classes)
    p = _load(fx.dir).Predictor(fx.dir)
    ref = fx.reference(fx.X_raw)
    out = p.predict_proba(fx.X_raw)
    assert out.dtype == np.float32
    if n_classes == 2:
        ref = np.column_stack([1 - ref, ref])
    assert np.max(np.abs(out - ref)) < 1e-4
    classes = np.asarray(fx.classes, dtype=np.float32)
    assert np.max(np.abs(p.predict(fx.X_raw) - out @ classes)) < 1e-6 if n_classes > 2 else True


def test_binary_predict_is_positive_class_probability(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    p = _load(fx.dir).Predictor(fx.dir)
    np.testing.assert_allclose(p.predict(fx.X_raw), p.predict_proba(fx.X_raw)[:, 1], atol=1e-7)


def test_nan_inf_input_never_returns_nan(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    p = _load(fx.dir).Predictor(fx.dir)
    X = np.full((4, 6), np.nan, dtype=np.float32)
    X[1] = np.inf
    X[2] = -np.inf
    out = p.predict(X)
    assert np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all()


def test_drift_warning_when_z_gt_3(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    mod = _load(fx.dir)
    p = mod.Predictor(fx.dir)
    X = np.zeros((3, 6), dtype=np.float32)
    X[1, 0] = 1e3
    with pytest.warns(mod.DistributionShiftWarning):
        p.predict(X)


def test_no_warning_in_distribution(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    mod = _load(fx.dir)
    p = mod.Predictor(fx.dir)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.predict(np.zeros((3, 6), dtype=np.float32))


def test_wrong_column_count_raises_valueerror(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    p = _load(fx.dir).Predictor(fx.dir)
    with pytest.raises(ValueError, match="columns"):
        p.predict(np.zeros((3, 5), dtype=np.float32))


def test_float64_input_is_rejected(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    p = _load(fx.dir).Predictor(fx.dir)
    with pytest.raises(ValueError, match="float32"):
        p.predict(np.zeros((3, 6), dtype=np.float64))


def test_single_row_latency_under_5ms(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    p = _load(fx.dir).Predictor(fx.dir)
    row = np.zeros((1, 6), dtype=np.float32)
    p.predict(row)  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        p.predict(row)
        times.append((time.perf_counter() - t0) * 1e3)
    assert float(np.median(times)) < 5.0


def test_rare_binary_indicator_does_not_trigger_drift_warning(tmp_path: Path) -> None:
    fx = make_bundle(tmp_path)
    mod = _load(fx.dir)
    p = mod.Predictor(fx.dir)
    X = np.zeros((3, 6), dtype=np.float32)
    X[1, 2] = np.nan  # flips the 0/1 is_nan_f2 indicator (rare in Train => |z| > 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.predict(X)
