"""Models package: GBDT Trio tournament, decision engine, and deep tuning."""

from quantautoml.models.candidates import (
    BaseGBDCandidate,
    CatBoostCandidate,
    LightGBMCandidate,
    XGBoostCandidate,
    create_candidate,
)
from quantautoml.models.decision_engine import (
    DecisionReport,
    DecisionType,
    ModelDecisionEngine,
)
from quantautoml.models.tournament import ModelTournament, TournamentResult, TrialScore
from quantautoml.models.tuner import DeepOptunaTuner, TunedCandidate, TuningResult
from quantautoml.models.validation import ExpandingWalkForwardSplitter

__all__ = [
    "BaseGBDCandidate",
    "CatBoostCandidate",
    "DecisionReport",
    "DecisionType",
    "DeepOptunaTuner",
    "ExpandingWalkForwardSplitter",
    "LightGBMCandidate",
    "ModelDecisionEngine",
    "ModelTournament",
    "TournamentResult",
    "TrialScore",
    "TunedCandidate",
    "TuningResult",
    "XGBoostCandidate",
    "create_candidate",
]
