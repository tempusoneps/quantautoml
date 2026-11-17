"""Walk-forward GBDT tournament with uniform budget and random sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import ModelsConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.metrics.ic import calc_ic_sharpe, calc_rank_ic
from quantautoml.models.candidates import create_candidate
from quantautoml.models.validation import ExpandingWalkForwardSplitter

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TrialScore(BaseModel):
    """Result of a single tournament trial evaluated over walk-forward folds."""

    model_config = ConfigDict(frozen=True)

    trial_id: int
    model_name: str
    params: dict[str, Any]
    fold_ics: list[float]
    mean_ic: float
    ic_sharpe: float


class TournamentResult(BaseModel):
    """Aggregated tournament results across all candidate models and trials."""

    model_config = ConfigDict(frozen=True)

    rankings: list[TrialScore]
    best_per_model: dict[str, TrialScore]
    oof_predictions: dict[str, list[float]]
    test_indices: list[int]


def _sample_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Sample hyperparameters using Optuna trial suggestions."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [30, 50, 100, 150]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8]),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [30, 50, 100, 150]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "catboost":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [30, 50, 100, 150]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }
    raise InvalidDataError(f"Unsupported model for sampling: '{model_name}'")


class ModelTournament:
    """Executes a fair budget tournament between GBDT models on Walk-Forward folds."""

    def __init__(self, cfg: ModelsConfig, task_type: TaskType) -> None:
        if task_type == TaskType.UNSUPERVISED:
            raise InvalidDataError("ModelTournament cannot be run on UNSUPERVISED tasks")
        self.cfg = cfg
        self.task_type = task_type

    def run(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32 | np.int32],
        embargo_bars: int = 0,
    ) -> TournamentResult:
        """Run tournament across enabled models on X and y.

        X and y must belong strictly to the training segment (RULE-01).
        """
        n_samples = len(X)
        if n_samples < 20:
            raise InvalidDataError(f"Insufficient samples for tournament (got {n_samples})")

        splitter = ExpandingWalkForwardSplitter(
            n_folds=self.cfg.n_wf_folds,
            min_train_ratio=0.5,
            embargo_bars=embargo_bars,
        )
        folds = list(splitter.split(n_samples))

        # Collect concatenated test indices
        test_indices = [int(idx) for _, t_idx in folds for idx in t_idx]

        rankings: list[TrialScore] = []
        best_per_model: dict[str, TrialScore] = {}
        oof_predictions: dict[str, list[float]] = {}

        global_trial_id = 0

        for model_idx, model_name in enumerate(self.cfg.enabled_models):
            sampler = optuna.samplers.RandomSampler(
                seed=self.cfg.random_state + model_idx * 1000
            )
            study = optuna.create_study(direction="maximize", sampler=sampler)

            model_trial_scores: list[TrialScore] = []
            trial_oof_map: dict[int, list[float]] = {}

            def objective(trial: optuna.Trial) -> float:
                nonlocal global_trial_id
                params = _sample_params(trial, model_name)

                fold_ics: list[float] = []
                oof_preds_list: list[float] = []

                for train_idx, test_idx in folds:
                    candidate = create_candidate(
                        name=model_name,
                        task_type=self.task_type,
                        params=params,
                        random_state=self.cfg.random_state,
                    )
                    candidate.fit(X[train_idx], y[train_idx])
                    pred = candidate.predict(X[test_idx])
                    oof_preds_list.extend(pred.tolist())

                    ic = calc_rank_ic(pred, y[test_idx])[0]
                    fold_ics.append(float(ic))

                mean_ic = float(np.mean(fold_ics))
                sharpe = float(calc_ic_sharpe(np.array(fold_ics, dtype=np.float64)))

                score = TrialScore(
                    trial_id=global_trial_id,
                    model_name=model_name,
                    params=params,
                    fold_ics=fold_ics,
                    mean_ic=mean_ic,
                    ic_sharpe=sharpe,
                )
                model_trial_scores.append(score)
                trial_oof_map[global_trial_id] = oof_preds_list
                global_trial_id += 1
                return sharpe

            study.optimize(objective, n_trials=self.cfg.tournament_trials)

            # Sort this model's trials
            model_trial_scores.sort(key=lambda s: (s.ic_sharpe, s.mean_ic), reverse=True)
            best_trial = model_trial_scores[0]
            best_per_model[model_name] = best_trial
            oof_predictions[model_name] = trial_oof_map[best_trial.trial_id]

            rankings.extend(model_trial_scores)

        # Global rankings sorted descending
        rankings.sort(key=lambda s: (s.ic_sharpe, s.mean_ic), reverse=True)

        return TournamentResult(
            rankings=rankings,
            best_per_model=best_per_model,
            oof_predictions=oof_predictions,
            test_indices=test_indices,
        )
