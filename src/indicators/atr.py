from __future__ import annotations

import pandas as pd

from .base_indicator import BaseIndicator


class ATRIndicator(BaseIndicator):
    """Average True Range (ATR) using Wilder smoothing."""

    required_columns: tuple[str, ...] = ("high", "low", "close")

    def __init__(self, period: int = 14) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        self.period = period

    @classmethod
    def validate_input(cls, data: pd.DataFrame) -> None:
        """Validate DataFrame input for ATR calculation."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Indicator input must be a pandas DataFrame.")

        missing = [column for column in cls.required_columns if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for column in cls.required_columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                raise TypeError(f"'{column}' column must be numeric.")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_input(data)

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / self.period,
            adjust=False,
            min_periods=self.period,
        ).mean()

        return atr.rename(f"atr_{self.period}")