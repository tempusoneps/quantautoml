"""Unit tests for RealWorldConfig."""

import pytest
from pydantic import ValidationError

from quantautoml.common.config import RealWorldConfig


def test_defaults_match_spec() -> None:
    cfg = RealWorldConfig()
    assert cfg.cash == 1_000_000.0
    assert cfg.commission == 0.0005
    assert cfg.margin == 1.0
    assert cfg.trade_on_close is False
    assert cfg.out_html is None
    assert cfg.out_trades_csv is None
    assert cfg.ohlcv_open_col == "Open"
    assert cfg.ohlcv_high_col == "High"
    assert cfg.ohlcv_low_col == "Low"
    assert cfg.ohlcv_close_col == "Close"
    assert cfg.ohlcv_volume_col == "Volume"


def test_cash_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(cash=0.0)


def test_negative_commission_rejected() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(commission=-0.001)


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(unknown=1)  # type: ignore[call-arg]
