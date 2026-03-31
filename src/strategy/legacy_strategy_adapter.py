"""Compatibility helpers for legacy strategy result normalization."""

from __future__ import annotations

from typing import Any


class LegacyStrategyAdapter:
    """Normalize placeholder strategy outputs without driving final selection."""

    valid_signals = {"long", "short", "hold"}

    def normalize_result(
        self,
        result: dict[str, Any] | None,
        *,
        default_strategy: str = "unknown",
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            return self.build_default_result(default_strategy)

        strategy = str(result.get("strategy", default_strategy)).strip().lower()
        signal = str(result.get("signal", "hold")).strip().lower()
        if signal not in self.valid_signals:
            signal = "hold"

        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        normalized = {
            "strategy": strategy,
            "signal": signal,
            "confidence": max(0.0, min(confidence, 1.0)),
            "timeframe_summary": result.get("timeframe_summary", {}) or {},
            "debug": result.get("debug", {}) or {},
        }

        for key, value in result.items():
            if key not in normalized:
                normalized[key] = value

        return normalized

    def build_default_result(self, strategy_name: str) -> dict[str, Any]:
        return {
            "strategy": strategy_name,
            "signal": "hold",
            "confidence": 0.0,
            "timeframe_summary": {},
            "debug": {},
        }
