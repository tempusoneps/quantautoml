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
