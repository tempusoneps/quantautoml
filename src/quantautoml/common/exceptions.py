"""Custom exceptions shared across the quantautoml pipeline."""


class QuantAutoMLError(Exception):
    """Base class for every error raised by quantautoml."""


class InvalidDataError(QuantAutoMLError):
    """Input data or configuration violates a pipeline data contract."""


class TimestampAlignmentError(QuantAutoMLError):
    """Timestamps are non-monotonic, duplicated or cannot be aligned."""


class DataLeakageError(QuantAutoMLError):
    """A look-ahead leakage invariant was violated (RULE-01)."""


class FeatureSelectionError(QuantAutoMLError):
    """Feature funnel cannot be fit or applied (e.g. unfitted filter, empty result)."""


class ModelError(QuantAutoMLError):
    """Base class for model training, tournament, and tuning failures."""


class TournamentError(ModelError):
    """Tournament evaluation or candidate selection failed."""


class TuningError(ModelError):
    """Hyperparameter optimization failed or produced invalid results."""


class EnsembleError(QuantAutoMLError):
    """Base class for ensemble optimization and probability calibration failures."""


class OptimizationConvergenceError(EnsembleError):
    """SLSQP convex optimization failed to converge to a feasible solution."""


class CalibrationError(EnsembleError):
    """Probability calibration failed or violated rank-invariance."""


class PackagingError(QuantAutoMLError):
    """Base class for ONNX bundle packaging failures."""


class PrecisionValidationError(PackagingError):
    """Python vs ONNX outputs differ by >= the numerical precision tolerance (RULE-06)."""


class SchemaMismatchError(PackagingError):
    """Input columns/dtype do not match the features_schema.json data contract."""


class BacktestError(QuantAutoMLError):
    """Out-of-sample backtest cannot be run on the supplied data."""
