import pydantic
import pytest

from quantautoml.common.config import ModelsConfig, PipelineConfig, load_config
from quantautoml.common.exceptions import ModelError, QuantAutoMLError, TournamentError, TuningError


def test_model_exceptions_hierarchy() -> None:
    assert issubclass(ModelError, QuantAutoMLError)
    assert issubclass(TournamentError, ModelError)
    assert issubclass(TuningError, ModelError)


_MODELS_TOML = (
    "\n[models]\n"
    "tournament_trials = 30\n"
    "tuning_trials = 50\n"
    "n_wf_folds = 5\n"
    "diversity_threshold = 0.88\n"
    "marginal_gain_threshold = 0.015\n"
    "wf_consistency_threshold = 0.60\n"
    "complexity_penalty = 0.05\n"
    "random_state = 42\n"
    'enabled_models = ["lightgbm", "xgboost", "catboost"]\n'
)


def test_models_section_parses_defaults(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml + "\n[models]\n"))
    assert cfg.models is not None
    assert cfg.models.tournament_trials == 30
    assert cfg.models.tuning_trials == 50
    assert cfg.models.n_wf_folds == 5
    assert cfg.models.diversity_threshold == 0.88
    assert cfg.models.marginal_gain_threshold == 0.015
    assert cfg.models.wf_consistency_threshold == 0.60
    assert cfg.models.complexity_penalty == 0.05
    assert cfg.models.random_state == 42
    assert cfg.models.enabled_models == ["lightgbm", "xgboost", "catboost"]


def test_models_section_parses_custom(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml + _MODELS_TOML))
    assert cfg.models is not None
    assert cfg.models.tournament_trials == 30
    assert cfg.models.enabled_models == ["lightgbm", "xgboost", "catboost"]


def test_models_missing_is_none(write_toml, valid_toml: str) -> None:
    cfg = load_config(write_toml(valid_toml))
    assert cfg.models is None


@pytest.mark.parametrize("trials", [0, -1])
def test_invalid_trials_rejected(trials: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelsConfig(tournament_trials=trials)


@pytest.mark.parametrize("div", [0.0, 1.5, -0.1])
def test_invalid_diversity_threshold_rejected(div: float) -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelsConfig(diversity_threshold=div)


def test_invalid_enabled_models_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelsConfig(enabled_models=["random_forest"])

    with pytest.raises(pydantic.ValidationError):
        ModelsConfig(enabled_models=[])


def test_models_config_ic_window_default_and_custom(write_toml, valid_toml: str) -> None:
    assert ModelsConfig().ic_window is None
    assert ModelsConfig(ic_window=500).ic_window == 500
    cfg = load_config(write_toml(valid_toml + "\n[models]\nic_window = 250\n"))
    assert cfg.models is not None and cfg.models.ic_window == 250


@pytest.mark.parametrize("w", [19, 0, -5])
def test_models_config_ic_window_below_20_rejected(w: int) -> None:
    with pytest.raises(pydantic.ValidationError):
        ModelsConfig(ic_window=w)
