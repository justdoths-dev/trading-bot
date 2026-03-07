from __future__ import annotations

from typing import Any

import pandas as pd

from .bias_detector import BiasDetector
from .setup_detector import SetupDetector
from .trigger_detector import TriggerDetector


class StrategyEngine:
    """Rule-based multi-timeframe strategy engine v1."""

    def __init__(self) -> None:
        self.bias_detector = BiasDetector()
        self.setup_detector = SetupDetector()
        self.trigger_detector = TriggerDetector()

    def evaluate(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        bias_result = self.bias_detector.detect(enriched_data)
        setup_result = self.setup_detector.detect(enriched_data)
        trigger_result = self.trigger_detector.detect(enriched_data)

        bias = bias_result["bias"]
        setup = setup_result["setup"]
        trigger = trigger_result["trigger"]

        if bias == "bullish" and setup == "long" and trigger == "long":
            signal = "long"
            reason = "Higher timeframe bias is bullish, mid timeframe setup is long, and lower timeframe trigger is long."
        elif bias == "bearish" and setup == "short" and trigger == "short":
            signal = "short"
            reason = "Higher timeframe bias is bearish, mid timeframe setup is short, and lower timeframe trigger is short."
        else:
            signal = "no_signal"
            reason = (
                f"No alignment across layers: "
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
        }