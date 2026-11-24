"""Pipeline: config -> ingestion -> task inference -> split -> artifacts on disk."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from quantautoml.common.config import FeaturesConfig
from quantautoml.pipeline import run_pipeline


def test_run_pipeline_writes_reports_and_segments(tmp_path: Path, make_frames, build_config) -> None:
    feats, labels = make_frames(n=3000)
    cfg = build_config(feats, labels, horizon_h=21)
    out = tmp_path / "out"

    result = run_pipeline(cfg, out)

    assert result.split.train.X.dtype == np.float32
    data_dir = out / "01_data"
    for name in ("01_ingestion_report.json", "02_task.json", "03_split_report.json"):
        assert (data_dir / name).is_file()
    assert json.loads((data_dir / "02_task.json").read_text())["task_type"] == "binary_classification"
    for seg in ("train", "val", "oos"):
        with np.load(data_dir / "segments" / f"{seg}.npz") as z:
            assert z["X"].dtype == np.float32
            assert z["X"].shape[0] == getattr(result.split, seg).X.shape[0]
    manifest = json.loads((out / "00_run_manifest.json").read_text())
    phases = {p["phase"]: p for p in manifest["phases"]}
    assert phases["01_data"]["status"] == "completed"
    assert phases["02_features"]["status"] == "skipped"


def test_run_pipeline_without_features_has_no_funnel(
    tmp_path: Path, make_frames, build_config
) -> None:
    feats, labels = make_frames(n=3000)
    out = tmp_path / "out"
    result = run_pipeline(build_config(feats, labels, horizon_h=21), out)
    assert result.funnel is None
    assert not (out / "02_features").exists()


def test_run_pipeline_with_features_writes_funnel_reports(
    tmp_path: Path, make_frames, build_config
) -> None:
    feats, labels = make_frames(n=3000, n_features=4)
    # f_sig[t] = fwd_ret_1[t] + noise: a (synthetic, leaky-by-construction) strong signal.
    fwd1 = labels["fwd_ret_1"].to_numpy()
    rng = np.random.default_rng(1)
    sig = np.nan_to_num(fwd1) + 0.01 * rng.normal(size=fwd1.size)
    feats = feats.with_columns(pl.Series("f_sig", sig))
    base = build_config(feats, labels, horizon_h=21)
    cfg = base.model_copy(
        update={
            "features": FeaturesConfig(
                ic_horizons={1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21"},
                target_horizon=1,
            )
        }
    )
    out = tmp_path / "out"

    result = run_pipeline(cfg, out)

    assert result.funnel is not None
    feat_dir = out / "02_features"
    report = json.loads((feat_dir / "01_funnel_report.json").read_text())
    assert [r["stage"] for r in report] == [r.stage for r in result.funnel.reports_]
    assert len(report) == 7
    for entry in report:
        assert set(entry) == {"stage", "n_in", "n_out", "dropped", "notes"}
    for a, b in zip(report, report[1:]):
        assert a["n_out"] == b["n_in"]
    selected = json.loads((feat_dir / "02_selected_features.json").read_text())
    assert selected == {"selected_features": list(result.funnel.selected_features_)}
    assert "f_sig" in selected["selected_features"]
    assert result.funnel.transform(result.split.val).X.dtype == np.float32


def test_run_pipeline_with_models_writes_reports(
    tmp_path: Path, make_frames, build_config
) -> None:
    from quantautoml.common.config import ModelsConfig

    feats, labels = make_frames(n=1000, n_features=4)
    base = build_config(feats, labels, horizon_h=21)
    cfg = base.model_copy(
        update={
            "models": ModelsConfig(
                tournament_trials=2,
                tuning_trials=2,
                n_wf_folds=2,
                enabled_models=["lightgbm"],
                random_state=42,
            )
        }
    )
    out = tmp_path / "out"
    result = run_pipeline(cfg, out)

    assert result.tournament_result is not None
    assert result.decision_report is not None
    assert result.tuning_result is not None

    models_dir = out / "03_models"
    assert (models_dir / "01_tournament_report.json").is_file()
    assert (models_dir / "02_decision_report.json").is_file()
    assert (models_dir / "03_tuning_report.json").is_file()

    t_rep = json.loads((models_dir / "01_tournament_report.json").read_text())
    assert len(t_rep["rankings"]) == 2

    d_rep = json.loads((models_dir / "02_decision_report.json").read_text())
    assert d_rep["decision"] in ("single_model", "ensemble")

    tune_rep = json.loads((models_dir / "03_tuning_report.json").read_text())
    assert len(tune_rep["tuned_candidates"]) == 1
