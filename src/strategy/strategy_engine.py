"""Strategy engine orchestrating multiple modular strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .intraday_strategy import IntradayStrategy
from .scalping_strategy import ScalpingStrategy
from .swing_strategy import SwingStrategy


class StrategyEngine:
    """Run all strategies and select the best signal."""

    PRIORITY = {
        "swing": 3,
        "intraday": 2,
        "scalping": 1,
    }

    VALID_SIGNALS = {"long", "short", "hold"}

    def __init__(self) -> None:
        self.strategies = [
            ScalpingStrategy(),
            IntradayStrategy(),
            SwingStrategy(),
        ]

    def evaluate(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        raw_results = [
            strategy.evaluate(enriched_data)
            for strategy in self.strategies
        ]

        normalized_results = [
            self._normalize_result(result)
            for result in raw_results
        ]

        result_map = {
            result["strategy"]: result
            for result in normalized_results
        }

        scalping_result = result_map.get(
            "scalping",
            self._build_default_result("scalping"),
        )
        intraday_result = result_map.get(
            "intraday",
            self._build_default_result("intraday"),
        )
        swing_result = result_map.get(
            "swing",
            self._build_default_result("swing"),
        )

        all_results = [
            scalping_result,
            intraday_result,
            swing_result,
        ]

        best = self._select_best_result(all_results)

        selected_result = {
            "strategy": best["strategy"],
            "signal": best["signal"],
            "confidence": best["confidence"],
            "bias": self._signal_to_bias(best["signal"]),
            "reason": (
                f"Selected {best['strategy']} strategy "
                f"(confidence={best['confidence']:.4f})"
            ),
            "timeframe_summary": best.get("timeframe_summary", {}),
            "debug": best.get("debug", {}),
        }

        return {
            "selected_strategy": best["strategy"],
            "selected_result": selected_result,
            "scalping_result": scalping_result,
            "intraday_result": intraday_result,
            "swing_result": swing_result,
            "strategy_results": all_results,
        }

    def _select_best_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        directional = [result for result in results if result["signal"] != "hold"]
        pool = directional if directional else results

        return max(
            pool,
            key=lambda result: (
                self.PRIORITY.get(result["strategy"], 0),
                result["confidence"],
            ),
        )

    def _normalize_result(self, result: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(result, dict):
            return self._build_default_result("unknown")

        strategy = str(result.get("strategy", "unknown")).strip().lower()
        signal = str(result.get("signal", "hold")).strip().lower()

        if signal not in self.VALID_SIGNALS:
            signal = "hold"

        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(confidence, 1.0))

        normalized = {
            "strategy": strategy,
            "signal": signal,
            "confidence": confidence,
        }

        for key, value in result.items():
            if key not in normalized:
                normalized[key] = value

        return normalized

    @staticmethod
    def _build_default_result(strategy_name: str) -> dict[str, Any]:
        return {
            "strategy": strategy_name,
            "signal": "hold",
            "confidence": 0.0,
            "timeframe_summary": {},
            "debug": {},
        }

    @staticmethod
    def _signal_to_bias(signal: str) -> str:
        if signal == "long":
            return "bullish"
        if signal == "short":
            return "bearish"
        return "neutral"