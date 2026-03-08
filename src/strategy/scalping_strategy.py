"""Scalping strategy placeholder using 1m and 5m data."""

from __future__ import annotations

from typing import Any
import pandas as pd

from .base_strategy import BaseStrategy


class ScalpingStrategy(BaseStrategy):
    """Simple momentum placeholder for short-term scalping."""

    strategy_name = "scalping"
    required_timeframes = ("1m", "5m")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        score_1m = self._momentum_score(data["1m"])
        score_5m = self._momentum_score(data["5m"])

        combined_score = (0.7 * score_1m) + (0.3 * score_5m)

        signal = self._to_signal(combined_score)
        confidence = self._to_confidence(combined_score)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
        }
