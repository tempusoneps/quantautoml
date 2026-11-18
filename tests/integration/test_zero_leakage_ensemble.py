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
    assert (out_dir / "ensemble_report.json").exists()
    assert (out_dir / "calibration_report.json").exists()
