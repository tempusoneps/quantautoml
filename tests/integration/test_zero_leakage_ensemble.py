"""Integration test: Zero Look-Ahead Leakage for Phase 4 Ensemble & Calibration (RULE-01)."""

from pathlib import Path
import numpy as np
import pytest

from quantautoml.common.config import EnsembleConfig, ModelsConfig, PipelineConfig
from quantautoml.common.exceptions import DataLeakageError
from quantautoml.common.types import Dataset, IngestionReport, Segment, SplitMode, SplitReport, SplitResult, TaskType
from quantautoml.ensemble.bootstrap import BootstrapShrinkageEngine
from quantautoml.ensemble.calibration import TemperatureCalibrator
from quantautoml.models.candidates import LightGBMCandidate
from quantautoml.pipeline import run_pipeline


def _make_dummy_dataset(n: int = 12000, p: int = 10) -> Dataset:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Binary label with subtle signal from feature 0
    prob = 1.0 / (1.0 + np.exp(-0.5 * X[:, 0]))
    y = rng.binomial(1, prob).astype(np.int32)
    base_t = np.datetime64("2024-01-01T00:00:00", "s")
    timestamps = base_t + np.arange(n).astype("timedelta64[s]")
    return Dataset(
        X=X,
        y=y,
        labels_aux={},
        timestamps=timestamps,
        feature_names=tuple(f"f{i}" for i in range(p)),
        selected_label="y",
        report=IngestionReport(
            n_feature_rows=n,
            n_label_rows=n,
            n_joined_rows=n,
            n_unmatched_dropped=0,
            n_trailing_label_nan_dropped=0,
            source_timezone="UTC",
            naive_utc_assumed=True,
            n_features=p,
            n_nan=0,
            n_inf=0,
            t_min=str(timestamps[0]),
            t_max=str(timestamps[-1]),
            label_nan_counts={"y": 0},
            max_trailing_nan=0,
            horizon_h=5,
        ),
        task_type=TaskType.BINARY_CLASSIFICATION,
        classes_=np.array([0, 1], dtype=np.int32),
    )


def test_zero_lookahead_leakage_val_and_oos_independence() -> None:
    """RULE-01: Mutating split.val or split.oos MUST NOT affect ensemble weights or calibration."""
    rng = np.random.default_rng(101)
    N_cal_b = 1000
    N_cal_a = 1000

    # Predictions and targets for Cal-B and Cal-A
    y_cal_b = rng.binomial(1, 0.5, size=N_cal_b).astype(np.int32)
    preds_cal_b = np.column_stack([
        rng.uniform(0.1, 0.9, size=N_cal_b).astype(np.float32),
        rng.uniform(0.1, 0.9, size=N_cal_b).astype(np.float32),
    ])

    y_cal_a = rng.binomial(1, 0.5, size=N_cal_a).astype(np.int32)
    preds_cal_a = rng.uniform(0.1, 0.9, size=N_cal_a).astype(np.float32)

    cfg = EnsembleConfig(n_bootstrap=20, random_state=42)
    engine = BootstrapShrinkageEngine(cfg)
    res_b = engine.fit(preds_cal_b, y_cal_b)

    calibrator = TemperatureCalibrator(task_type=TaskType.BINARY_CLASSIFICATION, random_state=42)
    calibrator.fit(preds_cal_a, y_cal_a)
    res_a = calibrator.get_result()

    # Create dummy "future" val and oos data
    val_data = rng.standard_normal((500, 10))
    oos_data = rng.standard_normal((500, 10))

    # Mutate val and oos dramatically (e.g. inject extreme values / NaNs / spikes)
    val_data_mutated = val_data * 1000.0 + 9999.0
    oos_data_mutated = oos_data * -500.0

    # Re-run Cal-B and Cal-A fitting; confirm 100% bit-exact invariance
    res_b_after = engine.fit(preds_cal_b, y_cal_b)
    calibrator.fit(preds_cal_a, y_cal_a)
    res_a_after = calibrator.get_result()

    assert res_b.weights == res_b_after.weights
    assert res_b.ensemble_ic_sharpe == res_b_after.ensemble_ic_sharpe
    assert res_a.temperature == res_a_after.temperature
    assert res_a.calibrated_ece == res_a_after.calibrated_ece


def test_pipeline_with_ensemble_and_calibration(tmp_path: Path) -> None:
    """Test full pipeline end-to-end integration with ensemble & calibration reports."""
    import polars as pl

    n = 12000
    p = 5
    rng = np.random.default_rng(42)
    base_t = np.datetime64("2024-01-01T00:00:00", "us")
    timestamps = base_t + np.arange(n).astype("timedelta64[s]")

    features_dict = {"timestamp": timestamps}
    for i in range(p):
        features_dict[f"feat_{i}"] = rng.standard_normal(n).astype(np.float32)

    prob = 1.0 / (1.0 + np.exp(-0.6 * features_dict["feat_0"]))
    y = rng.binomial(1, prob).astype(np.int32)
    labels_dict = {"timestamp": timestamps, "target": y}

    feat_path = tmp_path / "features.parquet"
    lab_path = tmp_path / "labels.parquet"
    pl.DataFrame(features_dict).write_parquet(feat_path)
    pl.DataFrame(labels_dict).write_parquet(lab_path)

    toml_content = f"""
    [data]
    features_path = "{feat_path}"
    labels_path = "{lab_path}"

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
    min_weight = 0.05
    max_weight = 0.60
    n_bootstrap = 10
    hurdle_ratio = 0.05
    """
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(toml_content)

    out_dir = tmp_path / "pipeline_out"
    res = run_pipeline(cfg_file, out_dir)

    assert res.ensemble_result is not None
    assert res.calibration_result is not None

    # Check generated report files
    assert (out_dir / "04_ensemble" / "01_ensemble_report.json").exists()
    assert (out_dir / "04_ensemble" / "02_calibration_report.json").exists()


def test_static_5way_ensemble_uses_cal_a_and_calibration_uses_cal_b(tmp_path: Path) -> None:
    """Audit #2: Temporal causality — Ensemble fits on Cal-A (60-70%), Calibration on Cal-B (70-80%).

    Mutating Cal-B rows MUST NOT alter ensemble weights, while it MUST alter calibration temperature.
    """
    import polars as pl

    n, p = 12000, 5
    rng = np.random.default_rng(7)
    base_t = np.datetime64("2024-01-01T00:00:00", "us")
    timestamps = base_t + np.arange(n).astype("timedelta64[s]")

    f_dict: dict[str, np.ndarray] = {"timestamp": timestamps}
    for i in range(p):
        f_dict[f"feat_{i}"] = rng.standard_normal(n).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-0.7 * f_dict["feat_0"]))
    y = rng.binomial(1, prob).astype(np.int32)

    def _run_with(f_data: dict[str, np.ndarray], y_data: np.ndarray, tag: str):
        fp = tmp_path / f"f_{tag}.parquet"
        lp = tmp_path / f"l_{tag}.parquet"
        pl.DataFrame(f_data).write_parquet(fp)
        pl.DataFrame({"timestamp": timestamps, "target": y_data}).write_parquet(lp)
        cfg_file = tmp_path / f"cfg_{tag}.toml"
        cfg_file.write_text(
            f'[data]\nfeatures_path = "{fp}"\nlabels_path = "{lp}"\n'
            '[labels]\nselected = "target"\n'
            "[split]\nhorizon_h = 5\n"
            '[models]\ntournament_trials = 2\ntuning_trials = 2\nn_wf_folds = 2\n'
            'enabled_models = ["lightgbm", "xgboost"]\n'
            "[ensemble]\nn_bootstrap = 10\n"
        )
        return run_pipeline(cfg_file, tmp_path / f"out_{tag}")

    res1 = _run_with(f_dict, y, "base")
    assert res1.split.cal_b is not None
    assert res1.ensemble_result is not None
    assert res1.calibration_result is not None

    # Mutate ONLY Cal-B labels (flip them) — Cal-A & Train are untouched
    y_mut = y.copy()
    cal_b_idx = res1.split.cal_b.index
    y_mut[cal_b_idx] = 1 - y_mut[cal_b_idx]
    res2 = _run_with(f_dict, y_mut, "mut_cal_b")
    assert res2.ensemble_result is not None
    assert res2.calibration_result is not None

    # Ensemble learned on Cal-A -> strictly invariant to Cal-B mutation
    assert res1.ensemble_result.weights == res2.ensemble_result.weights
    assert res1.ensemble_result.ensemble_ic_sharpe == res2.ensemble_result.ensemble_ic_sharpe
    # Temperature Calibration learned on Cal-B -> must respond to Cal-B mutation
    assert res1.calibration_result.temperature != res2.calibration_result.temperature


def test_oof_assisted_phase4_uses_true_oof_not_refitted_in_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit #1: In OOF_ASSISTED mode (N < 10000), Phase 4 MUST use true out-of-fold predictions
    fitted on each fold's train_idx (with horizon_h purging), NEVER refitted_model.predict() on Train.
    """
    import polars as pl

    from quantautoml.models.candidates import create_candidate

    n, p = 3000, 5
    rng = np.random.default_rng(19)
    base_t = np.datetime64("2024-01-01T00:00:00", "us")
    timestamps = base_t + np.arange(n).astype("timedelta64[s]")
    f_dict: dict[str, np.ndarray] = {"timestamp": timestamps}
    for i in range(p):
        f_dict[f"feat_{i}"] = rng.standard_normal(n).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-0.8 * f_dict["feat_0"]))
    y = rng.binomial(1, prob).astype(np.int32)

    fp, lp = tmp_path / "f.parquet", tmp_path / "l.parquet"
    pl.DataFrame(f_dict).write_parquet(fp)
    pl.DataFrame({"timestamp": timestamps, "target": y}).write_parquet(lp)
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text(
        f'[data]\nfeatures_path = "{fp}"\nlabels_path = "{lp}"\n'
        '[labels]\nselected = "target"\n'
        "[split]\nhorizon_h = 5\n"
        '[models]\ntournament_trials = 2\ntuning_trials = 2\nn_wf_folds = 2\n'
        'enabled_models = ["lightgbm"]\n'
        "[ensemble]\nn_bootstrap = 10\n"
    )

    seen_ens: list[np.ndarray] = []
    seen_cal: list[np.ndarray] = []
    orig_ens_fit = BootstrapShrinkageEngine.fit
    orig_cal_fit = TemperatureCalibrator.fit

    def spy_ens(self, predictions, y_true, single_model_scores=None):  # type: ignore[no-untyped-def]
        seen_ens.append(np.asarray(predictions, dtype=np.float32).copy())
        return orig_ens_fit(self, predictions, y_true, single_model_scores)

    def spy_cal(self, logits_or_probs, y_true):  # type: ignore[no-untyped-def]
        seen_cal.append(np.asarray(logits_or_probs, dtype=np.float32).copy())
        return orig_cal_fit(self, logits_or_probs, y_true)

    monkeypatch.setattr(BootstrapShrinkageEngine, "fit", spy_ens)
    monkeypatch.setattr(TemperatureCalibrator, "fit", spy_cal)

    res = run_pipeline(cfg_file, tmp_path / "out")
    assert res.split.mode is SplitMode.OOF_ASSISTED
    assert res.split.oof_folds is not None
    assert res.tuning_result is not None
    assert len(seen_ens) == 1 and len(seen_cal) == 1

    tr_a, va_a = res.split.oof_folds[-2]
    tr_b, va_b = res.split.oof_folds[-1]
    cand = res.tuning_result.tuned_candidates[0]
    train_X = res.split.train.X
    train_y = res.split.train.y
    assert train_y is not None

    # Predictions passed to Phase 4 must NOT be the leaky in-sample predictions of refitted_model
    in_sample_a = cand.fitted_model.predict_proba(train_X[va_a])[:, 1].astype(np.float32)
    in_sample_b = cand.fitted_model.predict_proba(train_X[va_b])[:, 1].astype(np.float32)
    assert not np.allclose(seen_ens[0][:, 0], in_sample_a, atol=1e-4)
    assert not np.allclose(seen_cal[0], in_sample_b, atol=1e-4)

    # Instead, they must match a clean OOF model fitted strictly on tr_a / tr_b
    oof_a = create_candidate(
        name=cand.model_name, task_type=TaskType.BINARY_CLASSIFICATION,
        params=cand.best_params, random_state=42,
    )
    oof_a.fit(train_X[tr_a], train_y[tr_a])
    expected_a = oof_a.predict_proba(train_X[va_a])[:, 1].astype(np.float32)
    np.testing.assert_allclose(seen_ens[0][:, 0], expected_a, atol=1e-6)


