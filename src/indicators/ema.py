"""Exponential Moving Average (EMA) indicator."""

from __future__ import annotations

import pandas as pd

from .base_indicator import BaseIndicator


class EMAIndicator(BaseIndicator):
    """EMA implementation based on pandas exponential weighted mean."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_input(data)

        ema = data["close"].ewm(
            span=self.period,
            adjust=False,
            min_periods=self.period
        ).mean()

        return ema.rename(f"ema_{self.period}")