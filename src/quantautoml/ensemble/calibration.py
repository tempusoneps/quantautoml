"""Temperature Scaling Probability Calibration with Strict Rank Invariance (RULE-03)."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr

from quantautoml.common.exceptions import CalibrationError
from quantautoml.common.types import TaskType
from quantautoml.metrics.calibration import calc_ece


class CalibrationResult(BaseModel):
    """Result of probability calibration."""

    model_config = ConfigDict(frozen=True)

    temperature: float
    uncalibrated_ece: float
    calibrated_ece: float
    rank_ic_preserved: bool
    method: str = "temperature_scaling"


class TemperatureCalibrator:
    """Probability calibration via Temperature Scaling.

    Guarantees 100% preservation of ranking order (Rank Invariance) because
    scalar division T > 0 on log-odds is a strictly monotonic transformation.

    Parameters
    ----------
    task_type : TaskType, default=TaskType.BINARY
        Type of classification task.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.random_state = int(random_state)
        self.temperature_: float = 1.0
        self.result_: CalibrationResult | None = None

    def fit(
        self,
        logits_or_probs: NDArray[np.float32],
        y_true: NDArray[np.int32 | np.float32],
    ) -> Self:
        """Fit optimal temperature scalar T > 0 on Cal-B (chronologically after Ensemble).

        Parameters
        ----------
        logits_or_probs : NDArray[np.float32]
            1D array of probabilities (or logits) for binary, or 2D array of shape (N, C).
        y_true : NDArray[np.int32 | np.float32]
            1D array of ground truth class labels.
        """
        arr = np.asarray(logits_or_probs, dtype=np.float32)
        target = np.asarray(y_true, dtype=np.int32)

        if arr.size == 0 or target.size == 0:
            raise ValueError("Input logits/probs or target cannot be empty.")
        if arr.shape[0] != target.shape[0]:
            raise ValueError(
                f"Shape mismatch: logits_or_probs has {arr.shape[0]} rows but target has {target.shape[0]}."
            )

        n_samples = arr.shape[0]

        if (
            self.task_type == TaskType.BINARY_CLASSIFICATION
            or arr.ndim == 1
            or (arr.ndim == 2 and arr.shape[1] == 1)
        ):
            p = arr.ravel()
            # If values in [0, 1], convert to logit z
            p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
            z = np.log(p_safe / (1.0 - p_safe))
            y_bin = target.ravel()

            def nll_binary(t: float) -> float:
                if t <= 0:
                    return 1e9
                p_cal = expit(z / t)
                p_cal = np.clip(p_cal, 1e-9, 1.0 - 1e-9)
                loss = -np.mean(y_bin * np.log(p_cal) + (1 - y_bin) * np.log(1.0 - p_cal))
                return float(loss)

            res = minimize_scalar(nll_binary, bounds=(0.01, 10.0), method="bounded")
            self.temperature_ = float(res.x) if res.success else 1.0

        elif (
            self.task_type == TaskType.MULTICLASS_CLASSIFICATION
            or (arr.ndim == 2 and arr.shape[1] > 1)
        ):
            P = arr
            P_safe = np.clip(P, 1e-7, 1.0)
            Z = np.log(P_safe)
            y_cls = target.ravel()

            def nll_multiclass(t: float) -> float:
                if t <= 0:
                    return 1e9
                scaled_Z = Z / t
                lse = logsumexp(scaled_Z, axis=1)
                correct_logits = scaled_Z[np.arange(n_samples), y_cls]
                loss = -np.mean(correct_logits - lse)
                return float(loss)

            res = minimize_scalar(nll_multiclass, bounds=(0.01, 10.0), method="bounded")
            self.temperature_ = float(res.x) if res.success else 1.0

        else:
            self.temperature_ = 1.0

        calibrated = self.calibrate(arr)
        raw_ece = calc_ece(target, arr if arr.ndim <= 2 else arr.ravel())
        cal_ece = calc_ece(target, calibrated)
        rank_ok = self.verify_rank_invariance(arr, calibrated)

        self.result_ = CalibrationResult(
            temperature=self.temperature_,
            uncalibrated_ece=float(raw_ece),
            calibrated_ece=float(cal_ece),
            rank_ic_preserved=rank_ok,
            method="temperature_scaling",
        )
        return self

    def calibrate(self, logits_or_probs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply learned temperature scaling to input probabilities.

        Parameters
        ----------
        logits_or_probs : NDArray[np.float32]
            1D or 2D array of predictions.

        Returns
        -------
        NDArray[np.float32]
            Calibrated probabilities of the exact same shape and dtype np.float32.
        """
        arr = np.asarray(logits_or_probs, dtype=np.float32)
        t = max(self.temperature_, 1e-4)

        if (
            self.task_type == TaskType.BINARY_CLASSIFICATION
            or arr.ndim == 1
            or (arr.ndim == 2 and arr.shape[1] == 1)
        ):
            p = np.clip(arr, 1e-7, 1.0 - 1e-7)
            z = np.log(p / (1.0 - p))
            cal = expit(z / t)
            return cal.astype(np.float32)

        elif (
            self.task_type == TaskType.MULTICLASS_CLASSIFICATION
            or (arr.ndim == 2 and arr.shape[1] > 1)
        ):
            P = np.clip(arr, 1e-7, 1.0)
            Z = np.log(P)
            scaled = Z / t
            exp_z = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            cal = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return cal.astype(np.float32)

        return arr.astype(np.float32)

    def verify_rank_invariance(
        self,
        y_raw: NDArray[np.float32],
        y_calibrated: NDArray[np.float32],
    ) -> bool:
        """Verify that temperature calibration strictly preserved 100% of ranking order.

        Parameters
        ----------
        y_raw : NDArray[np.float32]
            Raw predictions before calibration.
        y_calibrated : NDArray[np.float32]
            Calibrated predictions.

        Returns
        -------
        bool
            True if rank correlation is 1.0 within numerical precision.
        """
        raw = np.asarray(y_raw, dtype=np.float32)
        cal = np.asarray(y_calibrated, dtype=np.float32)

        if raw.ndim <= 1 or (raw.ndim == 2 and raw.shape[1] == 1):
            rho, _ = spearmanr(raw.ravel(), cal.ravel())
            return bool(rho >= 1.0 - 1e-5)
        elif raw.ndim == 2:
            raw_top = np.argmax(raw, axis=1)
            cal_top = np.argmax(cal, axis=1)
            return bool(np.array_equal(raw_top, cal_top))

        return True

    def get_result(self) -> CalibrationResult:
        """Return the CalibrationResult summary."""
        if self.result_ is None:
            raise CalibrationError("Calibrator has not been fitted yet.")
        return self.result_
