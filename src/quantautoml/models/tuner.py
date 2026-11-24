"""Deep hyperparameter optimization using Optuna Multivariate TPE and Hyperband pruning."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from quantautoml.common.config import ModelsConfig
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import TaskType
from quantautoml.metrics.ic import calc_ic_sharpe
from quantautoml.models.candidates import BaseGBDCandidate, create_candidate
from quantautoml.models.tournament import (
    adaptive_child_weight_bounds,
    adaptive_leaf_bounds,
    fold_subwindow_ics,
)
from quantautoml.models.validation import ExpandingWalkForwardSplitter

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TunedCandidate(BaseModel):
    """A candidate model that has been fine-tuned and refitted on the full training set."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_name: str
    best_params: dict[str, Any]
    best_score: float
    best_ic_sharpe: float
    complexity_score: float
    fitted_model: BaseGBDCandidate


class TuningResult(BaseModel):
    """Result of deep hyperparameter tuning across all requested candidate models."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tuned_candidates: list[TunedCandidate]
    study_summaries: dict[str, Any]


def _suggest_tuning_params(
    trial: optuna.Trial, model_name: str, n_samples: int = 1000
) -> dict[str, Any]:
    """Fine-grained regularized parameter suggestion for Bayesian TPE optimization."""
    if model_name == "lightgbm":
        leaf_lo, leaf_hi = adaptive_leaf_bounds(n_samples)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 47),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", leaf_lo, leaf_hi),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "xgboost":
        cw_lo, cw_hi = adaptive_child_weight_bounds(n_samples)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", cw_lo, cw_hi),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "catboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }
    raise InvalidDataError(f"Unsupported model for tuning: '{model_name}'")


class DeepOptunaTuner:
    """Deep Bayesian hyperparameter tuner using Multivariate TPE and HyperbandPruner."""

    def __init__(self, cfg: ModelsConfig, task_type: TaskType) -> None:
        if task_type == TaskType.UNSUPERVISED:
            raise InvalidDataError("DeepOptunaTuner cannot be run on UNSUPERVISED tasks")
        self.cfg = cfg
        self.task_type = task_type

    def tune(
        self,
        models_to_tune: list[str],
        X: NDArray[np.float32],
        y: NDArray[np.float32 | np.int32],
        embargo_bars: int = 0,
    ) -> TuningResult:
        """Run deep tuning for each model in models_to_tune.

        Evaluates on walk-forward folds, applies Hyperband pruning, and refits the
        best model on the full training segment (RULE-01).
        """
        if not models_to_tune:
            raise InvalidDataError("models_to_tune cannot be empty")

        n_samples = len(X)
        if n_samples < 20:
            raise InvalidDataError(f"Insufficient samples for tuning (got {n_samples})")

        splitter = ExpandingWalkForwardSplitter(
            n_folds=self.cfg.n_wf_folds,
            min_train_ratio=0.5,
            embargo_bars=embargo_bars,
        )
        folds = list(splitter.split(n_samples))

        tuned_candidates: list[TunedCandidate] = []
        study_summaries: dict[str, Any] = {}

        for idx, model_name in enumerate(models_to_tune):
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                seed=self.cfg.random_state + idx * 500,
            )
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=self.cfg.n_wf_folds,
                reduction_factor=3,
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )

            def objective(trial: optuna.Trial) -> float:
                params = _suggest_tuning_params(trial, model_name, n_samples=n_samples)
                complexity = create_candidate(
                    name=model_name,
                    task_type=self.task_type,
                    params=params,
                    random_state=self.cfg.random_state,
                    device=self.cfg.device,
                ).complexity_score()
                sub_ics: list[float] = []
                ic_sharpe = 0.0
                objective_val = -self.cfg.complexity_penalty * complexity

                for step, (train_idx, test_idx) in enumerate(folds):
                    candidate = create_candidate(
                        name=model_name,
                        task_type=self.task_type,
                        params=params,
                        random_state=self.cfg.random_state,
                        device=self.cfg.device,
                    )
                    candidate.fit(X[train_idx], y[train_idx])
                    pred = candidate.predict(X[test_idx])
                    sub_ics.extend(fold_subwindow_ics(pred, y[test_idx]))

                    ic_sharpe = float(calc_ic_sharpe(np.array(sub_ics, dtype=np.float64)))
                    objective_val = ic_sharpe - self.cfg.complexity_penalty * complexity
                    # Report intermediate progress on the exact same scale as final objective
                    trial.report(objective_val, step=step)

                    if step >= 1 and trial.should_prune():
                        raise optuna.TrialPruned()

                trial.set_user_attr("ic_sharpe", ic_sharpe)
                trial.set_user_attr("complexity", complexity)
                return objective_val

            study.optimize(objective, n_trials=self.cfg.tuning_trials)

            best_params = study.best_params
            best_score = float(study.best_value)
            raw_ic_sharpe = float(
                study.best_trial.user_attrs.get("ic_sharpe", best_score)
            )

            # Refit on the full training segment
            refitted_model = create_candidate(
                name=model_name,
                task_type=self.task_type,
                params=best_params,
                random_state=self.cfg.random_state,
                device=self.cfg.device,
            )
            refitted_model.fit(X, y)
            final_complexity = refitted_model.complexity_score()

            tuned_candidates.append(
                TunedCandidate(
                    model_name=model_name,
                    best_params=best_params,
                    best_score=best_score,
                    best_ic_sharpe=raw_ic_sharpe,
                    complexity_score=final_complexity,
                    fitted_model=refitted_model,
                )
            )

            n_pruned = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            )
            study_summaries[model_name] = {
                "n_trials": len(study.trials),
                "n_pruned": n_pruned,
                "best_score": best_score,
                "best_params": best_params,
            }

        return TuningResult(
            tuned_candidates=tuned_candidates,
            study_summaries=study_summaries,
        )
