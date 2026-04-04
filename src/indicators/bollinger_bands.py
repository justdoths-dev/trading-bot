"""Bollinger Bands indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base_indicator import BaseIndicator


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands based on close price rolling mean and standard deviation."""

    def __init__(self, period: int = 20, std_multiplier: float = 2.0) -> None:
        if period <= 0:
            raise ValueError("period must be greater than 0")
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be greater than 0")
        self.period = period
        self.std_multiplier = std_multiplier

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(data)

        close = data["close"]
        rolling = close.rolling(window=self.period, min_periods=self.period)

        middle = rolling.mean()
        std = rolling.std(ddof=0)
        band_distance = std * self.std_multiplier
        upper = middle + band_distance
        lower = middle - band_distance

        band_range = upper - lower
        non_zero_middle = middle.where(middle != 0)
        bandwidth = (band_range / non_zero_middle).where(middle != 0)
        bandwidth = bandwidth.mask(band_range == 0, 0.0)

        percent_b = ((close - lower) / band_range).where(band_range != 0)
        percent_b = percent_b.mask(band_range == 0, 0.5)

        multiplier_label = self._format_multiplier(self.std_multiplier)
        suffix = f"{self.period}_{multiplier_label}"

        return pd.DataFrame(
            {
                f"bb_middle_{self.period}": middle,
                f"bb_std_{self.period}": std,
                f"bb_upper_{suffix}": upper,
                f"bb_lower_{suffix}": lower,
                f"bb_bandwidth_{suffix}": bandwidth.replace([np.inf, -np.inf], np.nan),
                f"bb_percent_b_{suffix}": percent_b.replace([np.inf, -np.inf], np.nan),
            },
            index=data.index,
        )

    @staticmethod
    def _format_multiplier(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value).replace(".", "_")
