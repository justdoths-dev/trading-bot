"""Compatibility-focused intraday strategy using mid and lower timeframe indicators."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base_strategy import BaseStrategy


class IntradayStrategy(BaseStrategy):
    """Mid-timeframe compatibility strategy for debug and fallback views."""

    strategy_name = "intraday"
    required_timeframes = ("5m", "15m", "1h")

    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_data(data)

        five_min_state = self._build_timeframe_state(data["5m"], "5m")
        fifteen_min_state = self._build_timeframe_state(data["15m"], "15m")
        one_hour_state = self._build_timeframe_state(data["1h"], "1h")

        combined_score = (
            (0.25 * five_min_state["score"])
            + (0.35 * fifteen_min_state["score"])
            + (0.40 * one_hour_state["score"])
        )
        signal = self._directional_signal(combined_score, threshold=0.2)
        confidence = self._bounded_confidence(combined_score, cap=0.68, floor=0.15)

        return {
            "strategy": self.strategy_name,
            "signal": signal,
            "confidence": confidence,
            "timeframe_summary": {
                "5m": five_min_state,
                "15m": fifteen_min_state,
                "1h": one_hour_state,
            },
            "debug": {
                "combined_score": round(combined_score, 4),
                "score_components": {
                    "5m": five_min_state["score"],
                    "15m": fifteen_min_state["score"],
                    "1h": one_hour_state["score"],
                },
            },
        }
