"""Base class for modular trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base class for strategy implementations."""

    strategy_name: str = "base"
    required_timeframes: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Evaluate strategy and return signal payload."""

    def _validate_data(self, data: dict[str, pd.DataFrame]) -> None:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict[str, pandas.DataFrame]")

        for timeframe in self.required_timeframes:
            if timeframe not in data:
                raise ValueError(
                    f"Missing timeframe '{timeframe}' for {self.strategy_name} strategy"
                )

            df = data[timeframe]

            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Timeframe '{timeframe}' must be a pandas DataFrame"
                )

            if df.empty:
                raise ValueError(
                    f"Timeframe '{timeframe}' DataFrame is empty"
                )

            if "close" not in df.columns:
                raise ValueError(
                    f"Timeframe '{timeframe}' missing required 'close' column"
                )

    @staticmethod
    def _momentum_score(df: pd.DataFrame) -> float:
        """Compute momentum score using log returns."""
        if len(df) < 2:
            return 0.0

        prev_close = float(df["close"].iloc[-2])
        last_close = float(df["close"].iloc[-1])

        if prev_close <= 0 or last_close <= 0:
            return 0.0

        return float(np.log(last_close / prev_close))

    @staticmethod
    def _to_signal(score: float, threshold: float = 0.0004) -> str:
        if score > threshold:
            return "long"
        if score < -threshold:
            return "short"
        return "hold"

    @staticmethod
    def _to_confidence(score: float, scale: float = 600.0) -> float:
        confidence = min(abs(score) * scale, 1.0)
        return round(confidence, 4)
