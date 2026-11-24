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


def test_signal_ema_span_and_min_hold_bars_defaults_and_custom() -> None:
    cfg = BacktestConfig()
    assert cfg.signal_ema_span == 1
    assert cfg.min_hold_bars == 0
    custom = BacktestConfig(signal_ema_span=5, min_hold_bars=5)
    assert custom.signal_ema_span == 5
    assert custom.min_hold_bars == 5


@pytest.mark.parametrize("span", [0, -2])
def test_invalid_signal_ema_span_rejected(span: int) -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(signal_ema_span=span)


def test_negative_min_hold_bars_rejected() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(min_hold_bars=-1)
