"""Relative Strength Index (RSI) indicator."""

from __future__ import annotations

import pandas as pd

from .base_indicator import BaseIndicator


class RSIIndicator(BaseIndicator):
    """RSI implementation using Wilder smoothing via exponential averages."""

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_input(data)

        delta = data["close"].diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(
            alpha=1 / self.period,
            adjust=False,
            min_periods=self.period
        ).mean()

        avg_loss = losses.ewm(
            alpha=1 / self.period,
            adjust=False,
            min_periods=self.period
        ).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.rename(f"rsi_{self.period}")
