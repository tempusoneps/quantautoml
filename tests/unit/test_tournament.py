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
