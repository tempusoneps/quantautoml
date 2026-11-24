"""Bootstrap Shrinkage Engine and Hurdle Fallback Gate (RULE-05)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import EnsembleConfig
from quantautoml.ensemble.slsqp import SmoothSLSQPOptimizer
from quantautoml.metrics.ic import calc_ic_sharpe, calc_rank_ic, calc_rolling_ic


class EnsembleWeightsResult(BaseModel):
    """Result contract of the ensemble weight optimization and hurdle evaluation."""

    model_config = ConfigDict(frozen=True)

    weights: list[float]
    raw_weights: list[float]
    shrunk_weights: list[float]
    bootstrap_std: list[float]
    is_ensemble_accepted: bool
    ensemble_ic_sharpe: float
    best_single_ic_sharpe: float
    equal_weights_ic_sharpe: float
    hurdle_ratio_achieved: float
    fallback_reason: str | None = None


def _sample_block_indices(
    rng: np.random.Generator,
    n_samples: int,
    target_size: int,
    block_size: int,
) -> NDArray[np.intp]:
    """Moving Block Bootstrap: sample contiguous time-series blocks of length ``block_size``."""
    b = max(1, min(block_size, target_size, n_samples))
    max_start = n_samples - b
    n_blocks = (target_size + b - 1) // b
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    offsets = np.arange(b, dtype=np.intp)
    idx = (starts[:, None] + offsets[None, :]).ravel()[:target_size]
    return idx.astype(np.intp, copy=False)


def _calc_series_ic_sharpe(
    pred: NDArray[np.float32],
    target: NDArray[np.float32 | np.int32],
) -> float:
    """Helper to evaluate IC Sharpe or fallback to Rank IC on small sample slices."""
    n = len(pred)
    window = max(10, min(50, n // 10))
    if n // window < 2:
        window = max(3, n // 2)

    ic_series = calc_rolling_ic(pred, target, window=window)
    score = calc_ic_sharpe(ic_series)
    if np.isnan(score) or score == 0.0:
        score = float(calc_rank_ic(pred, target)[0])
    return float(score)


class BootstrapShrinkageEngine:
    """Bootstrap weight distribution estimator with shrinkage and hurdle guard rail.

    Parameters
    ----------
    cfg : EnsembleConfig
        Ensemble configuration specifying bootstrap parameters, shrinkage factor,
        and hurdle ratio (+5% IC Sharpe requirement).
    """

    def __init__(self, cfg: EnsembleConfig) -> None:
        self.cfg = cfg

    def fit(
        self,
        predictions: NDArray[np.float32],
        y_true: NDArray[np.float32 | np.int32],
        single_model_scores: list[float] | None = None,
    ) -> EnsembleWeightsResult:
        """Fit ensemble weights on Cal-B using Bootstrap and Shrinkage.

        Parameters
        ----------
        predictions : NDArray[np.float32]
            2D array of shape (N, K) with predictions from K models.
        y_true : NDArray[np.float32 | np.int32]
            1D array of shape (N,) with ground truth targets.
        single_model_scores : list[float] | None, default=None
            Optional precomputed IC Sharpe scores for each of the K models.

        Returns
        -------
        EnsembleWeightsResult
            Container with final weights, bootstrap statistics, and acceptance flags.
        """
        preds = np.asarray(predictions, dtype=np.float32)
        target = np.asarray(y_true, dtype=np.float32)

        if preds.ndim != 2:
            raise ValueError(f"predictions must be 2D, got ndim={preds.ndim}.")
        if preds.shape[0] == 0 or preds.shape[1] == 0 or target.shape[0] == 0:
            raise ValueError("Input predictions or target cannot be empty.")
        if preds.shape[0] != target.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions has {preds.shape[0]} rows but target has {target.shape[0]} rows."
            )

        n_samples, n_models = preds.shape

        if n_models == 1:
            single_score = (
                float(single_model_scores[0])
                if single_model_scores
                else _calc_series_ic_sharpe(preds[:, 0], target)
            )
            return EnsembleWeightsResult(
                weights=[1.0],
                raw_weights=[1.0],
                shrunk_weights=[1.0],
                bootstrap_std=[0.0],
                is_ensemble_accepted=True,
                ensemble_ic_sharpe=single_score,
                best_single_ic_sharpe=single_score,
                equal_weights_ic_sharpe=single_score,
                hurdle_ratio_achieved=0.0,
                fallback_reason=None,
            )

        # 1. Moving Block Bootstrap sampling (preserves serial correlation; collects OOB preds)
        rng = np.random.default_rng(self.cfg.random_state)
        subsample_size = max(int(n_samples * self.cfg.bootstrap_sample_ratio), 10)
        subsample_size = min(subsample_size, n_samples)
        block_size = self.cfg.bootstrap_block_size

        optimizer = SmoothSLSQPOptimizer(
            min_weight=self.cfg.min_weight,
            max_weight=self.cfg.max_weight,
            random_state=self.cfg.random_state,
        )

        weights_matrix = np.zeros((self.cfg.n_bootstrap, n_models), dtype=np.float32)
        oob_sum = np.zeros(n_samples, dtype=np.float64)
        oob_cnt = np.zeros(n_samples, dtype=np.int64)
        for b in range(self.cfg.n_bootstrap):
            idx = _sample_block_indices(rng, n_samples, subsample_size, block_size)
            w_b = optimizer.optimize(preds[idx], target[idx])
            weights_matrix[b] = w_b
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[idx] = True
            oob = ~in_bag
            if oob.any():
                oob_sum[oob] += preds[oob] @ w_b.astype(np.float64)
                oob_cnt[oob] += 1

        # 2. Median and normalization
        raw_med = np.median(weights_matrix, axis=0)
        sum_raw = np.sum(raw_med)
        if sum_raw > 0.0:
            raw_med = raw_med / sum_raw
        else:
            raw_med = np.full(n_models, 1.0 / n_models, dtype=np.float32)

        boot_std = np.std(weights_matrix, axis=0)

        # 3. Shrinkage towards 1/K
        equal_w = np.full(n_models, 1.0 / n_models, dtype=np.float32)
        lam = float(self.cfg.shrinkage_factor)
        shrunk_w = (1.0 - lam) * raw_med + lam * equal_w
        shrunk_w = shrunk_w / np.sum(shrunk_w)

        # 4. Out-of-Bag Hurdle Evaluation (RULE-05, avoids in-sample optimism bias)
        if single_model_scores is not None and len(single_model_scores) == n_models:
            single_scores = [float(s) for s in single_model_scores]
        else:
            single_scores = [_calc_series_ic_sharpe(preds[:, k], target) for k in range(n_models)]

        best_single_score = max(single_scores)
        best_single_idx = int(np.argmax(single_scores))

        equal_pred = (preds @ equal_w).astype(np.float32)
        equal_score = _calc_series_ic_sharpe(equal_pred, target)

        shrunk_pred = (preds @ shrunk_w).astype(np.float32)
        if np.any(oob_cnt > 0):
            ens_pred = np.where(
                oob_cnt > 0, oob_sum / np.maximum(oob_cnt, 1), shrunk_pred
            ).astype(np.float32)
        else:
            ens_pred = shrunk_pred
        ens_score = _calc_series_ic_sharpe(ens_pred, target)

        baseline_score = max(best_single_score, equal_score)

        if baseline_score > 1e-6:
            hurdle_achieved = (ens_score - baseline_score) / abs(baseline_score)
            is_accepted = hurdle_achieved >= self.cfg.hurdle_ratio
        else:
            hurdle_achieved = ens_score - baseline_score
            is_accepted = hurdle_achieved >= self.cfg.hurdle_ratio

        if is_accepted:
            final_weights = [float(x) for x in shrunk_w]
            fallback_reason = None
        else:
            is_accepted = False
            if best_single_score >= equal_score:
                fb_w = np.zeros(n_models, dtype=np.float32)
                fb_w[best_single_idx] = 1.0
                final_weights = [float(x) for x in fb_w]
                fallback_reason = "Ensemble did not achieve +5% IC Sharpe over Single Best Model."
            else:
                final_weights = [float(x) for x in equal_w]
                fallback_reason = "Ensemble did not achieve +5% IC Sharpe over Equal Weights baseline."

        return EnsembleWeightsResult(
            weights=final_weights,
            raw_weights=[float(x) for x in raw_med],
            shrunk_weights=[float(x) for x in shrunk_w],
            bootstrap_std=[float(x) for x in boot_std],
            is_ensemble_accepted=is_accepted,
            ensemble_ic_sharpe=ens_score,
            best_single_ic_sharpe=best_single_score,
            equal_weights_ic_sharpe=equal_score,
            hurdle_ratio_achieved=float(hurdle_achieved),
            fallback_reason=fallback_reason,
        )
