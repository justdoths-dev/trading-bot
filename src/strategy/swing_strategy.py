"""Compatibility-focused swing strategy using higher timeframe indicators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy


class SwingStrategy(BaseStrategy):
    """Higher timeframe compatibility strategy for debug and fallback views."""

    strategy_name = "swing"
    required_timeframes = ("1h", "4h", "1d")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        one_hour_state = self._build_timeframe_state(data["1h"], "1h")
        four_hour_state = self._build_timeframe_state(data["4h"], "4h")
        daily_state = self._build_timeframe_state(data["1d"], "1d")

        combined_score = (
            (0.20 * one_hour_state["score"])
            + (0.35 * four_hour_state["score"])
            + (0.45 * daily_state["score"])
        )
        signal = self._directional_signal(combined_score, threshold=0.24)
        confidence = self._bounded_confidence(combined_score, cap=0.74, floor=0.16)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
            "timeframe_summary": {
                "1h": one_hour_state,
                "4h": four_hour_state,
                "1d": daily_state,
            },
            "debug": {
                "combined_score": round(combined_score, 4),
                "score_components": {
                    "1h": one_hour_state["score"],
                    "4h": four_hour_state["score"],
                    "1d": daily_state["score"],
                },
            },
        }
