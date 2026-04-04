from __future__ import annotations

import math

import pandas as pd

from src.indicators.bollinger_bands import BollingerBandsIndicator
from src.indicators.indicator_engine import IndicatorEngine


EXPECTED_COLUMNS = [
    "bb_middle_20",
    "bb_std_20",
    "bb_upper_20_2",
    "bb_lower_20_2",
    "bb_bandwidth_20_2",
    "bb_percent_b_20_2",
]


def _build_ohlcv(close_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(close_values), freq="h"),
            "open": close_values,
            "high": close_values,
            "low": close_values,
            "close": close_values,
            "volume": [1000.0] * len(close_values),
        }
    )


def test_indicator_engine_adds_bollinger_band_columns() -> None:
    engine = IndicatorEngine()

    enriched = engine.enrich({"1m": _build_ohlcv([float(value) for value in range(1, 31)])})
    result = enriched["1m"]

    for column in EXPECTED_COLUMNS:
        assert column in result.columns


def test_bollinger_bands_respect_warmup_nan_behavior() -> None:
    indicator = BollingerBandsIndicator(period=20, std_multiplier=2.0)

    result = indicator.calculate(_build_ohlcv([float(value) for value in range(1, 25)]))

    warmup = result.iloc[:19]
    assert warmup.isna().all().all()
    assert result.iloc[19].notna().all()


def test_bollinger_bands_handle_constant_price_series_without_divide_by_zero() -> None:
    indicator = BollingerBandsIndicator(period=20, std_multiplier=2.0)

    result = indicator.calculate(_build_ohlcv([100.0] * 25))
    latest = result.iloc[-1]

    assert latest["bb_middle_20"] == 100.0
    assert latest["bb_std_20"] == 0.0
    assert latest["bb_upper_20_2"] == 100.0
    assert latest["bb_lower_20_2"] == 100.0
    assert latest["bb_bandwidth_20_2"] == 0.0
    assert latest["bb_percent_b_20_2"] == 0.5


def test_bollinger_bandwidth_is_non_zero_for_varying_series() -> None:
    indicator = BollingerBandsIndicator(period=20, std_multiplier=2.0)

    result = indicator.calculate(_build_ohlcv([float(value) for value in range(1, 31)]))
    latest = result.iloc[-1]

    assert latest["bb_std_20"] > 0.0
    assert latest["bb_bandwidth_20_2"] > 0.0
    assert not math.isnan(latest["bb_percent_b_20_2"])
