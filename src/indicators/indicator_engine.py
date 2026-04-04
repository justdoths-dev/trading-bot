"""Indicator engine for enriching multi-timeframe market data."""

from __future__ import annotations

import pandas as pd

from .atr import ATRIndicator
from .bollinger_bands import BollingerBandsIndicator
from .ema import EMAIndicator
from .macd import MACDIndicator
from .rsi import RSIIndicator


class IndicatorEngine:
    """Apply configured indicators to each timeframe DataFrame."""

    required_ohlcv_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        rsi_period: int = 14,
        ema_fast_period: int = 20,
        ema_slow_period: int = 50,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        atr_period: int = 14,
        bollinger_period: int = 20,
        bollinger_std_multiplier: float = 2.0,
    ) -> None:
        self.rsi = RSIIndicator(period=rsi_period)
        self.ema_fast = EMAIndicator(period=ema_fast_period)
        self.ema_slow = EMAIndicator(period=ema_slow_period)
        self.macd = MACDIndicator(
            fast_period=macd_fast_period,
            slow_period=macd_slow_period,
            signal_period=macd_signal_period,
        )
        self.atr = ATRIndicator(period=atr_period)
        self.bollinger_bands = BollingerBandsIndicator(
            period=bollinger_period,
            std_multiplier=bollinger_std_multiplier,
        )

    def enrich(self, multi_timeframe_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Return timeframe -> DataFrame enriched with indicator columns."""
        if not isinstance(multi_timeframe_data, dict):
            raise TypeError("multi_timeframe_data must be a dict[str, pandas.DataFrame].")

        if not multi_timeframe_data:
            raise ValueError("multi_timeframe_data is empty.")

        enriched: dict[str, pd.DataFrame] = {}

        for timeframe, df in multi_timeframe_data.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Timeframe '{timeframe}' must map to a pandas DataFrame.")

            if df.empty:
                raise ValueError(f"Timeframe '{timeframe}' DataFrame is empty.")

            self._validate_required_columns(df, timeframe)

            result = df.copy()

            rsi_series = self.rsi.calculate(result)
            ema_fast_series = self.ema_fast.calculate(result)
            ema_slow_series = self.ema_slow.calculate(result)
            macd_df = self.macd.calculate(result)
            atr_series = self.atr.calculate(result)
            bollinger_df = self.bollinger_bands.calculate(result)

            result[rsi_series.name] = rsi_series
            result[ema_fast_series.name] = ema_fast_series
            result[ema_slow_series.name] = ema_slow_series

            for column in macd_df.columns:
                result[column] = macd_df[column]

            result[atr_series.name] = atr_series

            for column in bollinger_df.columns:
                result[column] = bollinger_df[column]

            enriched[timeframe] = result

        return enriched

    def _validate_required_columns(self, df: pd.DataFrame, timeframe: str) -> None:
        """Ensure required OHLCV columns exist."""
        missing = [column for column in self.required_ohlcv_columns if column not in df.columns]

        if missing:
            raise ValueError(
                f"Timeframe '{timeframe}' is missing required OHLCV columns: {missing}"
            )
