"""Unit tests for BacktestConfig."""

import pytest
from pydantic import ValidationError

from quantautoml.common.config import BacktestConfig


def test_backtest_defaults_match_spec() -> None:
    cfg = BacktestConfig()
    assert cfg.price_col == "Close"
    assert cfg.fee_bps == 4.0
    assert cfg.slippage_bps == 1.0
    assert cfg.top_pct == 0.20
    assert cfg.exit_pct == 0.30
    assert cfg.min_rank_ic == 0.035
    assert cfg.min_icir == 0.40
    assert cfg.min_net_sharpe == 1.20
    assert cfg.max_mdd == 0.18
    assert cfg.min_downside_ic == 0.0


def test_exit_pct_lt_top_pct_rejected() -> None:
    with pytest.raises(ValidationError, match="exit_pct"):
        BacktestConfig(top_pct=0.30, exit_pct=0.20)


def test_negative_fee_rejected() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(fee_bps=-1.0)


def test_pct_must_be_below_half() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(top_pct=0.5, exit_pct=0.5)


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(foo=1)  # type: ignore[call-arg]
