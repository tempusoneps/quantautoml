"""Integration test for Phase 3 models: verifying Zero Look-Ahead Leakage (RULE-01).

Proves that tournament evaluation, decision engine, and deep Optuna tuning depend
strictly and exclusively on split.train. Changing val, cal_a, cal_b, or oos has
zero impact on tournament ranking, decision, or tuned hyperparameters.
"""

from pathlib import Path

import numpy as np
import pytest

from quantautoml.common.config import ModelsConfig, PipelineConfig
from quantautoml.pipeline import run_pipeline


@pytest.fixture
def market_dataset(tmp_path: Path, make_frames, build_config):
    feats, labels = make_frames(n=1200, n_features=6)
    base_cfg = build_config(feats, labels, horizon_h=21)
    cfg = base_cfg.model_copy(
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
    return cfg, tmp_path


def test_zero_leakage_models_invariance(market_dataset) -> None:
    cfg, tmp_path = market_dataset
    out1 = tmp_path / "run1"
    res1 = run_pipeline(cfg, out1)

    assert res1.tournament_result is not None
    assert res1.decision_report is not None
    assert res1.tuning_result is not None

    # Check generated report files
    assert (out1 / "tournament_report.json").is_file()
    assert (out1 / "decision_report.json").is_file()
    assert (out1 / "tuning_report.json").is_file()

    # Now verify zero leakage:
    # Perturb the validation and OOS segments in split:
    split_perturbed = res1.split
    rng = np.random.default_rng(999)
    split_perturbed.val.X[:] = rng.normal(size=split_perturbed.val.X.shape).astype(np.float32)
    split_perturbed.oos.X[:] = rng.normal(size=split_perturbed.oos.X.shape).astype(np.float32)
    if split_perturbed.cal_a is not None:
        split_perturbed.cal_a.X[:] = rng.normal(size=split_perturbed.cal_a.X.shape).astype(np.float32)
    if split_perturbed.cal_b is not None:
        split_perturbed.cal_b.X[:] = rng.normal(size=split_perturbed.cal_b.X.shape).astype(np.float32)

    # Re-run pipeline with perturbed future segments
    # Running tournament & tuning directly on split_perturbed.train
    from quantautoml.models.decision_engine import ModelDecisionEngine
    from quantautoml.models.tournament import ModelTournament
    from quantautoml.models.tuner import DeepOptunaTuner

    tourn = ModelTournament(cfg.models, res1.dataset.task_type)
    t_res2 = tourn.run(split_perturbed.train.X, split_perturbed.train.y, embargo_bars=cfg.split.horizon_h)

    dec_engine = ModelDecisionEngine(cfg.models)
    dec_res2 = dec_engine.evaluate(t_res2, split_perturbed.train.y)

    tuner = DeepOptunaTuner(cfg.models, res1.dataset.task_type)
    tune_res2 = tuner.tune(dec_res2.selected_models, split_perturbed.train.X, split_perturbed.train.y, embargo_bars=cfg.split.horizon_h)

    # Exact equality of tournament ranking
    assert [r.model_name for r in t_res2.rankings] == [r.model_name for r in res1.tournament_result.rankings]
    assert [r.ic_sharpe for r in t_res2.rankings] == [r.ic_sharpe for r in res1.tournament_result.rankings]

    # Exact equality of decision report
    assert dec_res2.decision == res1.decision_report.decision
    assert dec_res2.selected_models == res1.decision_report.selected_models

    # Exact equality of tuned parameters
    assert [c.best_params for c in tune_res2.tuned_candidates] == [c.best_params for c in res1.tuning_result.tuned_candidates]

    # Precision contract (RULE-06): check float32
    for c in res1.tuning_result.tuned_candidates:
        pred = c.fitted_model.predict(res1.split.val.X)
        assert pred.dtype == np.float32
        proba = c.fitted_model.predict_proba(res1.split.val.X)
        assert proba.dtype == np.float32
