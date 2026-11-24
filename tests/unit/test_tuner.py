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


def test_tuner_reports_aligned_objective_to_pruner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit #4: Intermediate values reported to HyperbandPruner must use the same
    objective scale (ic_sharpe - complexity_penalty * complexity) as the final return value.
    """
    import optuna

    reported: list[tuple[int, float]] = []
    orig_report = optuna.Trial.report

    def spy_report(self, value: float, step: int) -> None:
        reported.append((step, float(value)))
        return orig_report(self, value, step)

    monkeypatch.setattr(optuna.Trial, "report", spy_report)

    rng = np.random.default_rng(11)
    n = 1000
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = (X[:, 0] * 0.9 + rng.standard_normal(n) * 0.4 > 0.0).astype(np.int32)
    cfg = ModelsConfig(tuning_trials=1, n_wf_folds=2, complexity_penalty=0.05, random_state=42)
    res = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).tune(
        models_to_tune=["lightgbm"], X=X, y=y
    )
    cand = res.tuned_candidates[0]
    # The final step's reported value must equal the final objective score
    final_step_val = reported[-1][1]
    assert final_step_val == pytest.approx(cand.best_score, abs=1e-5)


def test_tuner_search_space_respects_regularization_caps(tuning_data) -> None:
    X, y = tuning_data
    cfg = ModelsConfig(
        tuning_trials=2,
        n_wf_folds=2,
        device="cpu",
        random_state=42,
    )
    res = DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).tune(
        models_to_tune=["lightgbm", "xgboost", "catboost"], X=X, y=y
    )
    for cand in res.tuned_candidates:
        p = cand.best_params
        assert 50 <= p["n_estimators"] <= 300
        assert 0.01 <= p["learning_rate"] <= 0.08
        if cand.model_name == "lightgbm":
            assert 3 <= p["max_depth"] <= 7
            assert 7 <= p["num_leaves"] <= 47
            assert p["min_child_samples"] >= 10
        elif cand.model_name == "xgboost":
            assert 3 <= p["max_depth"] <= 7
            assert p["min_child_weight"] >= 2
        elif cand.model_name == "catboost":
            assert 3 <= p["depth"] <= 7
            assert p["l2_leaf_reg"] >= 2.0


def test_tuner_forwards_explicit_ic_window(
    tuning_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantautoml.models.tuner as tuner_mod

    X, y = tuning_data
    seen_windows: list[int | None] = []
    orig = tuner_mod.fold_subwindow_ics

    def spy(pred, y_fold, *, ic_window=None):  # type: ignore[no-untyped-def]
        seen_windows.append(ic_window)
        return orig(pred, y_fold, ic_window=ic_window)

    monkeypatch.setattr(tuner_mod, "fold_subwindow_ics", spy)
    cfg = ModelsConfig(
        tuning_trials=1,
        n_wf_folds=2,
        ic_window=120,
        device="cpu",
        random_state=42,
    )
    DeepOptunaTuner(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).tune(
        models_to_tune=["lightgbm"], X=X, y=y
    )
    assert seen_windows and all(w == 120 for w in seen_windows)
