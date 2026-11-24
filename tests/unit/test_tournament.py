import numpy as np
import pytest

from quantautoml.common.config import ModelsConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.models.tournament import ModelTournament, TournamentResult, TrialScore


@pytest.fixture
def tournament_data():
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 6).astype(np.float32)
    # Signal with some noise
    y = (X[:, 0] * 0.8 + X[:, 1] * 0.4 + np.random.randn(n) * 0.5 > 0.0).astype(np.int32)
    return X, y


def test_tournament_small_run(tournament_data) -> None:
    X, y = tournament_data
    cfg = ModelsConfig(
        tournament_trials=2,
        n_wf_folds=2,
        enabled_models=["lightgbm", "xgboost", "catboost"],
        random_state=42,
    )
    tournament = ModelTournament(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION)
    result = tournament.run(X, y, embargo_bars=2)

    assert isinstance(result, TournamentResult)
    # 3 models * 2 trials = 6 rankings
    assert len(result.rankings) == 6
    for score in result.rankings:
        assert isinstance(score, TrialScore)
        assert len(score.fold_ics) == 2
        assert isinstance(score.ic_sharpe, float)
        assert isinstance(score.mean_ic, float)

    # Rankings must be sorted descending by ic_sharpe
    for i in range(len(result.rankings) - 1):
        assert result.rankings[i].ic_sharpe >= result.rankings[i + 1].ic_sharpe

    # Best per model has 3 entries
    assert set(result.best_per_model.keys()) == {"lightgbm", "xgboost", "catboost"}

    # oof_predictions matches test_indices length
    n_test_samples = len(result.test_indices)
    assert n_test_samples > 0
    for name in ["lightgbm", "xgboost", "catboost"]:
        assert name in result.oof_predictions
        assert len(result.oof_predictions[name]) == n_test_samples


def test_tournament_unsupervised_rejected(tournament_data) -> None:
    X, y = tournament_data
    cfg = ModelsConfig()
    with pytest.raises(InvalidDataError):
        ModelTournament(cfg=cfg, task_type=TaskType.UNSUPERVISED)


def test_tournament_insufficient_samples() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.zeros(10, dtype=np.int32)
    cfg = ModelsConfig(n_wf_folds=5)
    tournament = ModelTournament(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION)
    with pytest.raises(InvalidDataError):
        tournament.run(X, y)


def test_tournament_ic_sharpe_uses_subwindow_rolling_ics() -> None:
    """Audit #4: When fold test blocks are large enough, IC Sharpe must be computed over
    internal non-overlapping sub-windows rather than std() of just n_wf_folds (2-5) scalars.
    """
    from quantautoml.metrics.ic import calc_ic_sharpe

    rng = np.random.default_rng(7)
    n = 1200
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = (X[:, 0] * 0.8 + rng.standard_normal(n) * 0.5 > 0.0).astype(np.int32)
    cfg = ModelsConfig(
        tournament_trials=1,
        n_wf_folds=2,
        enabled_models=["lightgbm"],
        random_state=42,
    )
    result = ModelTournament(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).run(X, y)
    score = result.rankings[0]
    naive_2fold_sharpe = float(calc_ic_sharpe(np.array(score.fold_ics, dtype=np.float64)))
    # With 2 folds of ~300 rows each, sub-window rolling IC Sharpe differs markedly from 2-point std
    assert not np.isclose(score.ic_sharpe, naive_2fold_sharpe, rtol=1e-3)
    assert np.isfinite(score.ic_sharpe) and score.ic_sharpe > 0.0


def test_adaptive_leaf_bounds_scales_with_sample_size() -> None:
    from quantautoml.models.tournament import adaptive_child_weight_bounds, adaptive_leaf_bounds

    assert adaptive_leaf_bounds(250) == (10, 30)
    assert adaptive_leaf_bounds(50_000) == (50, 250)
    assert adaptive_child_weight_bounds(250) == (2, 15)
    assert adaptive_child_weight_bounds(50_000) == (10, 50)


def test_tournament_search_space_respects_regularization_caps(tournament_data) -> None:
    X, y = tournament_data
    cfg = ModelsConfig(
        tournament_trials=3,
        n_wf_folds=2,
        enabled_models=["lightgbm", "xgboost", "catboost"],
        device="cpu",
        random_state=42,
    )
    res = ModelTournament(cfg=cfg, task_type=TaskType.BINARY_CLASSIFICATION).run(X, y)
    for score in res.rankings:
        p = score.params
        assert 0.01 <= p["learning_rate"] <= 0.08
        if score.model_name == "lightgbm":
            assert 3 <= p["max_depth"] <= 7
            assert 7 <= p["num_leaves"] <= 47
            assert "min_child_samples" in p and p["min_child_samples"] >= 10
        elif score.model_name == "xgboost":
            assert 3 <= p["max_depth"] <= 7
            assert "min_child_weight" in p and p["min_child_weight"] >= 2
        elif score.model_name == "catboost":
            assert 3 <= p["depth"] <= 7
            assert p["l2_leaf_reg"] >= 2.0


def test_fold_subwindow_ics_scales_up_on_long_series() -> None:
    from quantautoml.models.tournament import fold_subwindow_ics

    rng = np.random.default_rng(12)
    n = 4500
    y = rng.standard_normal(n).astype(np.float32)
    pred = (y + 0.5 * rng.standard_normal(n)).astype(np.float32)
    ics = fold_subwindow_ics(pred, y)
    assert len(ics) == 9  # 4500 // 500


def test_fold_subwindow_ics_respects_explicit_ic_window() -> None:
    from quantautoml.models.tournament import fold_subwindow_ics

    rng = np.random.default_rng(13)
    n = 2000
    y = rng.standard_normal(n).astype(np.float32)
    pred = (y + 0.5 * rng.standard_normal(n)).astype(np.float32)
    ics = fold_subwindow_ics(pred, y, ic_window=250)
    assert len(ics) == 8  # 2000 // 250


def test_fold_subwindow_ics_clamps_large_ic_window_on_small_fold() -> None:
    from quantautoml.models.tournament import fold_subwindow_ics

    rng = np.random.default_rng(14)
    n = 200
    y = rng.standard_normal(n).astype(np.float32)
    pred = (y + 0.5 * rng.standard_normal(n)).astype(np.float32)
    ics = fold_subwindow_ics(pred, y, ic_window=500)
    assert len(ics) == 4  # clamped to n // 4 = 50 -> 4 windows


def test_fold_subwindow_ics_rejects_tiny_explicit_window() -> None:
    from quantautoml.models.tournament import fold_subwindow_ics

    y = np.arange(50, dtype=np.float32)
    with pytest.raises(InvalidDataError, match="ic_window"):
        fold_subwindow_ics(y, y, ic_window=2)
