"""Compatibility-focused scalping strategy using lower timeframe indicators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy


class ScalpingStrategy(BaseStrategy):
    """Lower timeframe compatibility strategy for debug and fallback views."""

    strategy_name = "scalping"
    required_timeframes = ("1m", "5m")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        one_min_state = self._build_timeframe_state(data["1m"], "1m")
        five_min_state = self._build_timeframe_state(data["5m"], "5m")

        combined_score = (0.55 * one_min_state["score"]) + (0.45 * five_min_state["score"])
        signal = self._directional_signal(combined_score, threshold=0.2)
        confidence = self._bounded_confidence(combined_score, cap=0.64, floor=0.14)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
            "timeframe_summary": {
                "1m": one_min_state,
                "5m": five_min_state,
            },
            "debug": {
                "combined_score": round(combined_score, 4),
                "score_components": {
                    "1m": one_min_state["score"],
                    "5m": five_min_state["score"],
                },
            },
        }
