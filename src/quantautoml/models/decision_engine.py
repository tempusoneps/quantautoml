"""Decision engine evaluating Single Model vs. Ensemble Pool under 3 hard conditions."""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from quantautoml.common.config import ModelsConfig
from quantautoml.metrics.ic import calc_rank_ic
from quantautoml.models.tournament import TournamentResult


class DecisionType(StrEnum):
    SINGLE_MODEL = "single_model"
    ENSEMBLE = "ensemble"


class DecisionReport(BaseModel):
    """Evaluation report containing the decision and metrics for all 3 gates."""

    model_config = ConfigDict(frozen=True)

    decision: DecisionType
    selected_models: list[str]
    pairwise_correlations: dict[str, float]
    diversity_score: float
    marginal_gain: float
    wf_consistency: float
    rationale: str


class ModelDecisionEngine:
    """Evaluates tournament candidates against 3 hurdle ensemble conditions (RULE-05)."""

    def __init__(self, cfg: ModelsConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        tournament_result: TournamentResult,
        y_true: NDArray[np.float32 | np.int32],
    ) -> DecisionReport:
        """Evaluate whether to form an Ensemble Pool (K <= 3) or fallback to Single Best Model.

        Conditions:
        1. Diversity: Pairwise correlation of predictions < diversity_threshold (diversity > 0.12).
        2. Marginal Gain: Equal-weights ensemble has >= marginal_gain_threshold over single best.
        3. WF Consistency: In >= wf_consistency_threshold of walk-forward folds, ensemble beats single best.
        """
        best_models = sorted(
            tournament_result.best_per_model.values(),
            key=lambda s: (s.ic_sharpe, s.mean_ic),
            reverse=True,
        )

        if not best_models:
            return DecisionReport(
                decision=DecisionType.SINGLE_MODEL,
                selected_models=[],
                pairwise_correlations={},
                diversity_score=0.0,
                marginal_gain=0.0,
                wf_consistency=0.0,
                rationale="No candidate models available.",
            )

        best_single_name = best_models[0].model_name

        if len(best_models) < 2:
            return DecisionReport(
                decision=DecisionType.SINGLE_MODEL,
                selected_models=[best_single_name],
                pairwise_correlations={},
                diversity_score=1.0,
                marginal_gain=0.0,
                wf_consistency=1.0,
                rationale="Fewer than 2 distinct candidate models available for ensemble.",
            )

        # Select top candidates up to K <= 3
        candidate_models = [s.model_name for s in best_models[:3]]
        y_test = np.asarray(y_true, dtype=np.float64)[tournament_result.test_indices]
        n_test = len(y_test)

        preds: dict[str, NDArray[np.float64]] = {
            m: np.asarray(tournament_result.oof_predictions[m], dtype=np.float64)
            for m in candidate_models
        }

        # 1. Diversity Condition: Pairwise Spearman correlation
        pairwise_corrs: dict[str, float] = {}
        for m1, m2 in combinations(candidate_models, 2):
            rho = calc_rank_ic(preds[m1], preds[m2])[0]
            pairwise_corrs[f"{m1}_vs_{m2}"] = float(rho)

        max_corr = max(pairwise_corrs.values()) if pairwise_corrs else 0.0
        diversity_score = float(1.0 - max_corr)
        cond_diversity = max_corr < self.cfg.diversity_threshold

        # Equal-weights rank-normalized ensemble
        rank_norms = np.array(
            [rankdata(preds[m]) / n_test for m in candidate_models],
            dtype=np.float64,
        )
        ens_pred = np.mean(rank_norms, axis=0)
        single_best_pred = preds[best_single_name]

        # 2. Marginal Gain Condition: Ensemble Rank IC vs Single Best Rank IC
        ic_single = calc_rank_ic(single_best_pred, y_test)[0]
        ic_ens = calc_rank_ic(ens_pred, y_test)[0]
        marginal_gain = float(ic_ens - ic_single)
        cond_gain = marginal_gain >= self.cfg.marginal_gain_threshold

        # 3. Walk-Forward Consistency Condition: Across folds
        fold_lens = tournament_result.fold_test_lens
        if fold_lens:
            n_consistent = 0
            offset = 0
            for flen in fold_lens:
                sl = slice(offset, offset + flen)
                fold_ic_ens = calc_rank_ic(ens_pred[sl], y_test[sl])[0]
                fold_ic_single = calc_rank_ic(single_best_pred[sl], y_test[sl])[0]
                if fold_ic_ens >= fold_ic_single:
                    n_consistent += 1
                offset += flen
            wf_consistency = float(n_consistent / len(fold_lens))
        else:
            wf_consistency = 1.0 if cond_gain else 0.0

        cond_consistency = wf_consistency >= self.cfg.wf_consistency_threshold

        reasons: list[str] = []
        if not cond_diversity:
            reasons.append(
                f"Diversity too low (max correlation {max_corr:.3f} >= {self.cfg.diversity_threshold})"
            )
        if not cond_gain:
            reasons.append(
                f"Marginal gain insufficient ({marginal_gain:+.4f} < {self.cfg.marginal_gain_threshold:+.4f})"
            )
        if not cond_consistency:
            reasons.append(
                f"Walk-Forward consistency too low ({wf_consistency:.1%} < {self.cfg.wf_consistency_threshold:.1%})"
            )

        if not reasons:
            decision = DecisionType.ENSEMBLE
            selected_models = candidate_models
            rationale = "All 3 hurdle conditions passed: Diversity, Marginal Gain, and WF Consistency verified."
        else:
            decision = DecisionType.SINGLE_MODEL
            selected_models = [best_single_name]
            rationale = "Fallback to single best model: " + "; ".join(reasons)

        return DecisionReport(
            decision=decision,
            selected_models=selected_models,
            pairwise_correlations=pairwise_corrs,
            diversity_score=diversity_score,
            marginal_gain=marginal_gain,
            wf_consistency=wf_consistency,
            rationale=rationale,
        )
