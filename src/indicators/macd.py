from __future__ import annotations

import pandas as pd

from .base_indicator import BaseIndicator


class MACDIndicator(BaseIndicator):
    """MACD implementation using exponential moving averages."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        if fast_period <= 0:
            raise ValueError("fast_period must be greater than 0")
        if slow_period <= 0:
            raise ValueError("slow_period must be greater than 0")
        if signal_period <= 0:
            raise ValueError("signal_period must be greater than 0")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(data)

        close = data["close"]

        ema_fast = close.ewm(
            span=self.fast_period,
            adjust=False,
            min_periods=self.fast_period,
        ).mean()

        ema_slow = close.ewm(
            span=self.slow_period,
            adjust=False,
            min_periods=self.slow_period,
        ).mean()

        macd_line = ema_fast - ema_slow

        signal_line = macd_line.ewm(
            span=self.signal_period,
            adjust=False,
            min_periods=self.signal_period,
        ).mean()

        hist = macd_line - signal_line

        suffix = f"{self.fast_period}_{self.slow_period}_{self.signal_period}"

        return pd.DataFrame(
            {
                f"macd_{suffix}": macd_line,
                f"macd_signal_{suffix}": signal_line,
                f"macd_hist_{suffix}": hist,
            },
            index=data.index,
        )