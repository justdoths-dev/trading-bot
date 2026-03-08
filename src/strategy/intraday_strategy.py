"""Intraday strategy placeholder using 5m, 15m, and 1h data."""

from __future__ import annotations

from typing import Any
import pandas as pd

from .base_strategy import BaseStrategy


class IntradayStrategy(BaseStrategy):
    """Simple momentum placeholder for intraday context."""

    strategy_name = "intraday"
    required_timeframes = ("5m", "15m", "1h")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        score_5m = self._momentum_score(data["5m"])
        score_15m = self._momentum_score(data["15m"])
        score_1h = self._momentum_score(data["1h"])

        combined_score = (
            0.4 * score_5m +
            0.35 * score_15m +
            0.25 * score_1h
        )

        signal = self._to_signal(combined_score)
        confidence = self._to_confidence(combined_score)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
        }
