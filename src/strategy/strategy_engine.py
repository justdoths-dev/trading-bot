from __future__ import annotations

from typing import Any

import pandas as pd

from .bias_detector import BiasDetector
from .setup_detector import SetupDetector
from .trigger_detector import TriggerDetector


class StrategyEngine:
    """Rule-based multi-timeframe strategy engine v1.3."""

    def __init__(self) -> None:
        self.bias_detector = BiasDetector()
        self.setup_detector = SetupDetector(
            rsi_long_threshold=50.0,
            rsi_short_threshold=50.0,
            long_recovery_floor=40.0,
            short_recovery_ceiling=60.0,
            early_long_recovery_floor=30.0,
            early_short_recovery_ceiling=70.0,
        )
        self.trigger_detector = TriggerDetector(
            rsi_long_threshold=50.0,
            rsi_short_threshold=50.0,
            max_volatility_ratio=0.003,
            improving_hist_floor=-1.0,
        )

    def evaluate(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        bias_result = self.bias_detector.detect(enriched_data)
        setup_result = self.setup_detector.detect(enriched_data)
        trigger_result = self.trigger_detector.detect(enriched_data)

        bias = bias_result["bias"]
        setup = setup_result["setup"]
        trigger = trigger_result["trigger"]

        if bias == "bullish" and setup == "long" and trigger == "long":
            signal = "long"
            reason = (
                "Higher timeframe bias is bullish, mid timeframe setup is long, "
                "and lower timeframe trigger is long."
            )
        elif bias == "bearish" and setup == "short" and trigger == "short":
            signal = "short"
            reason = (
                "Higher timeframe bias is bearish, mid timeframe setup is short, "
                "and lower timeframe trigger is short."
            )
        elif trigger in ("long", "improving_long") and setup in ("improving_long", "early_recovery_long"):
            signal = "watchlist_long"
            reason = (
                "Lower timeframe long pressure is present and mid timeframe is recovering, "
                "but higher timeframe bias is not fully aligned yet."
            )
        elif trigger in ("short", "improving_short") and setup in ("improving_short", "early_recovery_short"):
            signal = "watchlist_short"
            reason = (
                "Lower timeframe short pressure is present and mid timeframe is weakening, "
                "but higher timeframe bias is not fully aligned yet."
            )
        else:
            signal = "no_signal"
            reason = (
                f"No actionable alignment across layers: "
                f"bias={bias}, setup={setup}, trigger={trigger}"
            )

        return {
            "bias": bias,
            "signal": signal,
            "reason": reason,
            "timeframe_summary": {
                "bias_layer": bias_result,
                "setup_layer": setup_result,
                "trigger_layer": trigger_result,
            },
            "debug": {
                "bias_reason": bias_result["reason"],
                "setup_reason": setup_result["reason"],
                "trigger_reason": trigger_result["reason"],
                "setup_details": setup_result["details"],
                "trigger_details": trigger_result["details"],
            },
        }