"""Integration test for Phase 3 models: verifying Zero Look-Ahead Leakage (RULE-01).

The *source files* are rewritten so that every row after the last Train timestamp
(purge gap, Cal-A, Cal-B, Val, OOS) carries different features AND labels, then the
whole pipeline is re-run end-to-end. Tournament, decision and tuning outputs must be
bit-identical, proving they depend strictly on split.train.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantautoml.common.config import ModelsConfig, PipelineConfig
from quantautoml.pipeline import PipelineResult, run_pipeline


@pytest.fixture
def cfg(make_frames, build_config) -> PipelineConfig:
    feats, labels = make_frames(n=1200, n_features=6)
    base_cfg = build_config(feats, labels, horizon_h=21)
    return base_cfg.model_copy(
        update={
            "models": ModelsConfig(
                tournament_trials=2,
                tuning_trials=2,
                n_wf_folds=2,
                enabled_models=["lightgbm", "xgboost"],
                random_state=42,
            )
        }
    )


def _rewrite_sources(
    cfg: PipelineConfig, tmp_path: Path, *, cutoff: np.datetime64, perturb_train: bool
) -> PipelineConfig:
    """Write copies of features/labels with perturbed rows; return a config pointing at them."""
    rng = np.random.default_rng(999)
    assert cfg.data.labels_path is not None

    def _perturb(df: pl.DataFrame, binary: set[str]) -> pl.DataFrame:
        ts = df["timestamp"].dt.replace_time_zone(None).to_numpy()
        mask = (ts <= cutoff) if perturb_train else (ts > cutoff)
        cols = []
        for name in df.columns:
            if name == "timestamp":
                continue
            vals = df[name].to_numpy().astype(np.float64).copy()
            valid = mask & ~np.isnan(vals)
            if name in binary:
                vals[valid] = rng.integers(0, 2, size=int(valid.sum())).astype(np.float64)
            else:
                vals[valid] = rng.normal(size=int(valid.sum()))
            cols.append(pl.Series(name, vals))
        return pl.DataFrame([df["timestamp"], *cols])

    feats = _perturb(pl.read_parquet(cfg.data.features_path), set())
    labels = _perturb(pl.read_parquet(cfg.data.labels_path), {"y_dir"})
    f_path, l_path = tmp_path / "features_perturbed.parquet", tmp_path / "labels_perturbed.parquet"
    feats.write_parquet(f_path)
    labels.write_parquet(l_path)

    raw = cfg.model_dump(mode="json")
    raw["data"]["features_path"] = str(f_path)
    raw["data"]["labels_path"] = str(l_path)
    return PipelineConfig.model_validate(raw)


def _signature(res: PipelineResult) -> dict[str, object]:
    assert res.tournament_result is not None
    assert res.decision_report is not None
    assert res.tuning_result is not None
    train_X = res.split.train.X
    return {
        "rankings": [(r.model_name, r.params, r.fold_ics, r.ic_sharpe) for r in res.tournament_result.rankings],
        "oof": res.tournament_result.oof_predictions,
        "decision": res.decision_report.model_dump(),
        "tuned": [(c.model_name, c.best_params, c.best_score) for c in res.tuning_result.tuned_candidates],
        "preds": [c.fitted_model.predict(train_X).tobytes() for c in res.tuning_result.tuned_candidates],
    }


def test_models_depend_only_on_train(cfg: PipelineConfig, tmp_path: Path) -> None:
    res1 = run_pipeline(cfg, tmp_path / "run1")
    report_names = (
        "01_tournament_report.json",
        "02_decision_report.json",
        "03_tuning_report.json",
    )
    for name in report_names:
        assert (tmp_path / "run1" / "03_models" / name).is_file()

    cutoff = res1.split.train.timestamps[-1]
    cfg2 = _rewrite_sources(cfg, tmp_path, cutoff=cutoff, perturb_train=False)
    res2 = run_pipeline(cfg2, tmp_path / "run2")

    # Train segment itself must be untouched, otherwise the comparison is meaningless.
    np.testing.assert_array_equal(res1.split.train.X, res2.split.train.X)
    np.testing.assert_array_equal(res1.split.train.y, res2.split.train.y)
    # ...while the future segments really did change.
    assert not np.array_equal(res1.split.oos.X, res2.split.oos.X)
    assert not np.array_equal(res1.split.val.y, res2.split.val.y)

    assert _signature(res1) == _signature(res2)
    for name in report_names:
        assert (
            (tmp_path / "run1" / "03_models" / name).read_bytes()
            == (tmp_path / "run2" / "03_models" / name).read_bytes()
        )


def test_leakage_check_has_teeth(cfg: PipelineConfig, tmp_path: Path) -> None:
    """Perturbing Train rows MUST change the outputs, so the invariance test above is meaningful."""
    res1 = run_pipeline(cfg, tmp_path / "run1")
    cutoff = res1.split.train.timestamps[-1]
    cfg2 = _rewrite_sources(cfg, tmp_path, cutoff=cutoff, perturb_train=True)
    res2 = run_pipeline(cfg2, tmp_path / "run2")
    assert _signature(res1) != _signature(res2)


def test_models_output_float32(cfg: PipelineConfig, tmp_path: Path) -> None:
    res = run_pipeline(cfg, tmp_path / "run")
    assert res.tuning_result is not None
    for c in res.tuning_result.tuned_candidates:
        assert c.fitted_model.predict(res.split.val.X).dtype == np.float32
        assert c.fitted_model.predict_proba(res.split.val.X).dtype == np.float32
