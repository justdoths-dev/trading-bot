"""Swing strategy placeholder using 1h, 4h, and 1d data."""

from __future__ import annotations

from typing import Any
import pandas as pd

from .base_strategy import BaseStrategy


class SwingStrategy(BaseStrategy):
    """Momentum placeholder for higher timeframe swing trading."""

    strategy_name = "swing"
    required_timeframes = ("1h", "4h", "1d")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        score_1h = self._momentum_score(data["1h"])
        score_4h = self._momentum_score(data["4h"])
        score_1d = self._momentum_score(data["1d"])

        combined_score = (
            0.2 * score_1h +
            0.35 * score_4h +
            0.45 * score_1d
        )

        signal = self._to_signal(combined_score)
        confidence = self._to_confidence(combined_score)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
        }
