import numpy as np
import pytest

from quantautoml.common.config import ModelsConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.models.candidates import LightGBMCandidate
from quantautoml.models.tuner import DeepOptunaTuner, TunedCandidate, TuningResult


@pytest.fixture
def tuning_data():
    np.random.seed(42)
    n = 250
    X = np.random.randn(n, 5).astype(np.float32)
    y = (X[:, 0] * 0.7 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3 > 0.0).astype(np.int32)
    return X, y


def test_tuner_single_model(tuning_data) -> None:
    X, y = tuning_data
    cfg = ModelsConfig(
        tuning_trials=3,
        n_wf_folds=2,
        complexity_penalty=0.05,
        random_state=42,
    )
    tuner = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION)
    result = tuner.tune(models_to_tune=["lightgbm"], X=X, y=y, embargo_bars=2)

    assert isinstance(result, TuningResult)
    assert len(result.tuned_candidates) == 1

    candidate = result.tuned_candidates[0]
    assert isinstance(candidate, TunedCandidate)
    assert candidate.model_name == "lightgbm"
    assert isinstance(candidate.best_params, dict)
    assert "n_estimators" in candidate.best_params
    assert isinstance(candidate.fitted_model, LightGBMCandidate)

    # Fitted model can predict
    preds = candidate.fitted_model.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype == np.float32

    # Score calculation check: best_score = best_ic_sharpe - penalty * complexity
    expected_score = candidate.best_ic_sharpe - cfg.complexity_penalty * candidate.complexity_score
    assert candidate.best_score == pytest.approx(expected_score, abs=1e-5)


def test_tuner_multiple_models(tuning_data) -> None:
    X, y = tuning_data
    cfg = ModelsConfig(
        tuning_trials=2,
        n_wf_folds=2,
        random_state=42,
    )
    tuner = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION)
    result = tuner.tune(models_to_tune=["lightgbm", "xgboost"], X=X, y=y)

    assert len(result.tuned_candidates) == 2
    names = [c.model_name for c in result.tuned_candidates]
    assert "lightgbm" in names
    assert "xgboost" in names
    assert "lightgbm" in result.study_summaries
    assert "xgboost" in result.study_summaries


def test_tuner_unsupervised_rejected() -> None:
    cfg = ModelsConfig()
    with pytest.raises(InvalidDataError):
        DeepOptunaTuner(cfg=cfg, task_type=TaskType.UNSUPERVISED)


def test_tuner_empty_models_rejected(tuning_data) -> None:
    X, y = tuning_data
    cfg = ModelsConfig()
    tuner = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION)
    with pytest.raises(InvalidDataError):
        tuner.tune(models_to_tune=[], X=X, y=y)


def test_tuner_hyperband_prunes_weak_trials() -> None:
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (rng.normal(size=n) > 0).astype(np.int32)  # pure noise -> unstable fold ICs
    cfg = ModelsConfig(tuning_trials=25, n_wf_folds=5, random_state=0)
    result = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).tune(
        models_to_tune=["lightgbm"], X=X, y=y
    )
    summary = result.study_summaries["lightgbm"]
    assert summary["n_trials"] == 25
    assert summary["n_pruned"] >= 1
