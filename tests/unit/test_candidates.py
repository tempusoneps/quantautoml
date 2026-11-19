import numpy as np
import pytest

from quantautoml.common.exceptions import InvalidDataError, ModelError
from quantautoml.common.types import TaskType
from quantautoml.models.candidates import (
    BaseGBDCandidate,
    CatBoostCandidate,
    LightGBMCandidate,
    XGBoostCandidate,
    create_candidate,
)


@pytest.fixture
def binary_data():
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    # Signal in first feature
    y = (X[:, 0] > 0.0).astype(np.int32)
    return X, y


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    # 3 classes: 0, 1, 2
    raw = X[:, 0] + X[:, 1]
    y = np.zeros(200, dtype=np.int32)
    y[raw > 0.5] = 2
    y[(raw <= 0.5) & (raw >= -0.5)] = 1
    return X, y


def test_create_candidate_factory() -> None:
    lgb = create_candidate("lightgbm", TaskType.BINARY_CLASSIFICATION)
    assert isinstance(lgb, LightGBMCandidate)

    xgb = create_candidate("xgboost", TaskType.BINARY_CLASSIFICATION)
    assert isinstance(xgb, XGBoostCandidate)

    cb = create_candidate("catboost", TaskType.BINARY_CLASSIFICATION)
    assert isinstance(cb, CatBoostCandidate)

    with pytest.raises(InvalidDataError):
        create_candidate("random_forest", TaskType.BINARY_CLASSIFICATION)

    with pytest.raises(InvalidDataError):
        create_candidate("lightgbm", TaskType.UNSUPERVISED)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_candidates_fit_predict_binary(model_name: str, binary_data) -> None:
    X, y = binary_data
    candidate = create_candidate(
        model_name,
        TaskType.BINARY_CLASSIFICATION,
        params={"n_estimators": 20},
        random_state=42,
    )
    # Calling predict before fit raises ModelError
    with pytest.raises(ModelError):
        candidate.predict(X)

    candidate.fit(X, y)
    preds = candidate.predict(X)
    probs = candidate.predict_proba(X)

    assert preds.shape == (200,)
    assert preds.dtype == np.float32
    assert probs.shape == (200, 2)
    assert probs.dtype == np.float32

    # Probs sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(200, dtype=np.float32), atol=1e-5)
    # For binary, preds match positive class probability
    np.testing.assert_allclose(preds, probs[:, 1], atol=1e-5)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_candidates_fit_predict_multiclass(model_name: str, multiclass_data) -> None:
    X, y = multiclass_data
    candidate = create_candidate(
        model_name,
        TaskType.MULTICLASS_CLASSIFICATION,
        params={"n_estimators": 20},
        random_state=42,
    )
    candidate.fit(X, y)
    preds = candidate.predict(X)
    probs = candidate.predict_proba(X)

    assert preds.shape == (200,)
    assert preds.dtype == np.float32
    assert probs.shape == (200, 3)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(200, dtype=np.float32), atol=1e-5)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_candidates_reproducibility(model_name: str, binary_data) -> None:
    X, y = binary_data
    m1 = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"n_estimators": 10}, random_state=42)
    m2 = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"n_estimators": 10}, random_state=42)

    m1.fit(X, y)
    m2.fit(X, y)

    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_complexity_score(model_name: str) -> None:
    m_simple = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"n_estimators": 10})
    m_complex = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"n_estimators": 200})

    s_simple = m_simple.complexity_score()
    s_complex = m_complex.complexity_score()

    assert 0.0 < s_simple <= 1.0
    assert 0.0 < s_complex <= 1.0
    assert s_complex > s_simple


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_complexity_score_increases_with_learning_rate(model_name: str) -> None:
    low = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"learning_rate": 0.01})
    high = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"learning_rate": 0.1})
    assert high.complexity_score() > low.complexity_score()


def test_lightgbm_complexity_respects_max_depth() -> None:
    free = create_candidate("lightgbm", TaskType.BINARY_CLASSIFICATION, params={"num_leaves": 63, "max_depth": -1})
    capped = create_candidate("lightgbm", TaskType.BINARY_CLASSIFICATION, params={"num_leaves": 63, "max_depth": 3})
    assert capped.complexity_score() < free.complexity_score()


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_multiclass_train_missing_a_class(model_name: str) -> None:
    """Early walk-forward folds may lack a class; labels {0, 2} must still fit and score."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 2, 0).astype(np.int32)
    m = create_candidate(model_name, TaskType.MULTICLASS_CLASSIFICATION, params={"n_estimators": 10})
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (150,) and pred.dtype == np.float32
    # expected class value lies in [0, 2] and tracks the signal
    assert pred.min() >= 0.0 and pred.max() <= 2.0
    assert pred[X[:, 0] > 1].mean() > pred[X[:, 0] < -1].mean()


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_single_class_train_raises_model_error(model_name: str) -> None:
    X = np.random.randn(50, 3).astype(np.float32)
    m = create_candidate(model_name, TaskType.BINARY_CLASSIFICATION, params={"n_estimators": 5})
    with pytest.raises(ModelError):
        m.fit(X, np.ones(50, dtype=np.int32))


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_candidates_accept_subsample_param(model_name: str, binary_data) -> None:
    """Tournament/tuner search spaces pass `subsample` to every GBDT (CatBoost needs Bernoulli bootstrap)."""
    X, y = binary_data
    m = create_candidate(
        model_name, TaskType.BINARY_CLASSIFICATION,
        params={"n_estimators": 10, "subsample": 0.8}, random_state=42,
    )
    m.fit(X, y)
    assert m.predict_proba(X).shape == (200, 2)


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost"])
def test_candidates_accept_subsample_param_multiclass(model_name: str, multiclass_data) -> None:
    X, y = multiclass_data
    m = create_candidate(
        model_name, TaskType.MULTICLASS_CLASSIFICATION,
        params={"n_estimators": 10, "subsample": 0.8}, random_state=42,
    )
    m.fit(X, y)
    assert m.predict_proba(X).shape == (200, 3)
