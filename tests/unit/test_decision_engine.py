import numpy as np
import pytest

from quantautoml.common.config import ModelsConfig
from quantautoml.models.decision_engine import DecisionReport, DecisionType, ModelDecisionEngine
from quantautoml.models.tournament import TournamentResult, TrialScore


def create_mock_tournament_result(
    oof_preds: dict[str, list[float]],
    test_indices: list[int],
    fold_test_lens: list[int],
    best_sharpes: dict[str, float],
) -> TournamentResult:
    best_per_model = {}
    rankings = []
    for i, (name, preds) in enumerate(oof_preds.items()):
        score = TrialScore(
            trial_id=i,
            model_name=name,
            params={"n_estimators": 50},
            fold_ics=[0.05] * len(fold_test_lens),
            mean_ic=0.05,
            ic_sharpe=best_sharpes.get(name, 1.0),
        )
        best_per_model[name] = score
        rankings.append(score)

    rankings.sort(key=lambda s: s.ic_sharpe, reverse=True)

    return TournamentResult(
        rankings=rankings,
        best_per_model=best_per_model,
        oof_predictions=oof_preds,
        test_indices=test_indices,
        fold_test_lens=fold_test_lens,
    )


def test_decision_engine_single_candidate_fallback() -> None:
    cfg = ModelsConfig()
    engine = ModelDecisionEngine(cfg=cfg)
    n = 100
    y_true = np.ones(n, dtype=np.int32)
    t_res = create_mock_tournament_result(
        oof_preds={"lightgbm": [0.5] * n},
        test_indices=list(range(n)),
        fold_test_lens=[50, 50],
        best_sharpes={"lightgbm": 1.5},
    )
    report = engine.evaluate(t_res, y_true)
    assert report.decision == DecisionType.SINGLE_MODEL
    assert report.selected_models == ["lightgbm"]


def test_decision_engine_fallback_when_high_correlation() -> None:
    cfg = ModelsConfig(diversity_threshold=0.88)
    engine = ModelDecisionEngine(cfg=cfg)
    n = 200
    y_true = np.random.randn(n).astype(np.float32)
    # Model 1 and Model 2 are nearly identical (rho ~ 0.99)
    base = np.random.randn(n)
    p1 = (base + np.random.randn(n) * 0.01).tolist()
    p2 = (base + np.random.randn(n) * 0.01).tolist()

    t_res = create_mock_tournament_result(
        oof_preds={"lightgbm": p1, "xgboost": p2},
        test_indices=list(range(n)),
        fold_test_lens=[100, 100],
        best_sharpes={"lightgbm": 1.5, "xgboost": 1.4},
    )
    report = engine.evaluate(t_res, y_true)
    assert report.decision == DecisionType.SINGLE_MODEL
    assert report.selected_models == ["lightgbm"]
    assert "Diversity too low" in report.rationale


def test_decision_engine_fallback_when_low_marginal_gain() -> None:
    cfg = ModelsConfig(diversity_threshold=0.88, marginal_gain_threshold=0.05)
    engine = ModelDecisionEngine(cfg=cfg)
    n = 300
    np.random.seed(42)
    y_true = np.random.randn(n).astype(np.float32)
    # Two diverse models but ensemble doesn't beat single best by 0.05
    p1 = (y_true * 0.5 + np.random.randn(n) * 0.5).tolist()
    p2 = (np.random.randn(n)).tolist()  # pure noise

    t_res = create_mock_tournament_result(
        oof_preds={"lightgbm": p1, "xgboost": p2},
        test_indices=list(range(n)),
        fold_test_lens=[150, 150],
        best_sharpes={"lightgbm": 1.5, "xgboost": 0.2},
    )
    report = engine.evaluate(t_res, y_true)
    assert report.decision == DecisionType.SINGLE_MODEL
    assert report.selected_models == ["lightgbm"]
    assert "Marginal gain" in report.rationale


def test_decision_engine_selects_ensemble_when_all_pass() -> None:
    cfg = ModelsConfig(
        diversity_threshold=0.88,
        marginal_gain_threshold=0.01,
        wf_consistency_threshold=0.60,
    )
    engine = ModelDecisionEngine(cfg=cfg)
    n = 500
    np.random.seed(42)
    # Synthetic target
    signal_a = np.random.randn(n)
    signal_b = np.random.randn(n)
    y_true = (signal_a + signal_b + np.random.randn(n) * 0.5).astype(np.float32)

    # Model 1 captures signal A, Model 2 captures signal B (diverse, low correlation)
    p1 = (signal_a + np.random.randn(n) * 0.5).tolist()
    p2 = (signal_b + np.random.randn(n) * 0.5).tolist()

    t_res = create_mock_tournament_result(
        oof_preds={"lightgbm": p1, "catboost": p2},
        test_indices=list(range(n)),
        fold_test_lens=[100, 100, 100, 100, 100],
        best_sharpes={"lightgbm": 1.2, "catboost": 1.1},
    )
    report = engine.evaluate(t_res, y_true)
    assert report.decision == DecisionType.ENSEMBLE
    assert set(report.selected_models) == {"lightgbm", "catboost"}
    assert report.diversity_score > 0.12
    assert report.marginal_gain >= 0.01
    assert report.wf_consistency >= 0.60


def test_wf_consistency_requires_strict_outperformance() -> None:
    """Ties between ensemble and single best must NOT count as outperformance."""
    cfg = ModelsConfig()
    engine = ModelDecisionEngine(cfg=cfg)
    n = 200
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=n).astype(np.float32)
    base = (y_true + rng.normal(size=n)).tolist()
    t_res = create_mock_tournament_result(
        oof_preds={"lightgbm": base, "xgboost": list(base)},  # identical -> ensemble == single
        test_indices=list(range(n)),
        fold_test_lens=[100, 100],
        best_sharpes={"lightgbm": 1.5, "xgboost": 1.4},
    )
    report = engine.evaluate(t_res, y_true)
    assert report.wf_consistency == 0.0
    assert report.decision == DecisionType.SINGLE_MODEL


def test_ensemble_ranks_are_normalised_per_fold() -> None:
    """A per-fold offset in one model's scale must not change the report."""
    cfg = ModelsConfig()
    engine = ModelDecisionEngine(cfg=cfg)
    n = 300
    rng = np.random.default_rng(7)
    y_true = rng.normal(size=n).astype(np.float32)
    p1 = y_true * 0.6 + rng.normal(size=n)
    p2 = y_true * 0.6 + rng.normal(size=n)
    lens = [100, 100, 100]
    offset = np.repeat([0.0, 50.0, -80.0], 100)

    def run(second: np.ndarray):
        t_res = create_mock_tournament_result(
            oof_preds={"lightgbm": p1.tolist(), "xgboost": second.tolist()},
            test_indices=list(range(n)),
            fold_test_lens=lens,
            best_sharpes={"lightgbm": 1.5, "xgboost": 1.4},
        )
        return engine.evaluate(t_res, y_true)

    a = run(p2)
    b = run(p2 + offset)
    assert a.wf_consistency == b.wf_consistency
    assert a.marginal_gain == pytest.approx(b.marginal_gain, abs=1e-12)
