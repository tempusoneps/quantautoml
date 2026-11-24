"""Friction-aware blind OOS backtesting."""

from quantautoml.backtest.buffer import HysteresisBuffer
from quantautoml.backtest.diagnostics import (
    ExecutionDiagnostic,
    SegmentDiagnostic,
    diagnose_run_dir,
    evaluate_bundle_segments,
)
from quantautoml.backtest.evaluator import OOSMetrics, evaluate_oos_performance
from quantautoml.backtest.friction import FrictionCostModel
from quantautoml.backtest.realworld_runner import RealWorldBacktestRunner, RealWorldMetrics
from quantautoml.backtest.runner import OOSBacktestRunner

__all__ = [
    "ExecutionDiagnostic",
    "FrictionCostModel",
    "HysteresisBuffer",
    "OOSBacktestRunner",
    "OOSMetrics",
    "RealWorldBacktestRunner",
    "RealWorldMetrics",
    "SegmentDiagnostic",
    "diagnose_run_dir",
    "evaluate_bundle_segments",
    "evaluate_oos_performance",
]
