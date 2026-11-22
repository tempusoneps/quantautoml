"""Unit tests for RealWorldConfig."""

import pytest
from pydantic import ValidationError

from quantautoml.common.config import RealWorldConfig


def test_defaults_match_spec() -> None:
    cfg = RealWorldConfig()
    assert cfg.cash == 1_000_000.0
    assert cfg.commission == 0.0005
    assert cfg.margin == 1.0
    assert cfg.size == "auto"
    assert cfg.trade_on_close is False
    assert cfg.out_html is None
    assert cfg.out_trades_csv is None
    assert cfg.ohlcv_open_col == "Open"
    assert cfg.ohlcv_high_col == "High"
    assert cfg.ohlcv_low_col == "Low"
    assert cfg.ohlcv_close_col == "Close"
    assert cfg.ohlcv_volume_col == "Volume"


@pytest.mark.parametrize("size", ["auto", "start_cash", "_start_cash", 1, 5, 0.5, "1", "0.25"])
def test_valid_size_modes_accepted(size: str | int | float) -> None:
    cfg = RealWorldConfig(size=size)
    assert cfg.size == size


@pytest.mark.parametrize("bad_size", ["unknown_mode", 0, -1, -0.5, 1.5, "0", "-3"])
def test_invalid_size_rejected(bad_size: str | int | float) -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(size=bad_size)


def test_cash_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(cash=0.0)


def test_negative_commission_rejected() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(commission=-0.001)


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        RealWorldConfig(unknown=1)  # type: ignore[call-arg]
