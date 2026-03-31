"""Strategy engine orchestrating detector-based decisions and compatibility strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .context_evaluator import ContextEvaluator
from .decision_composer import DecisionComposer
from .intraday_strategy import IntradayStrategy
from .legacy_strategy_adapter import LegacyStrategyAdapter
from .scalping_strategy import ScalpingStrategy
from .setup_detector import SetupDetector
from .swing_strategy import SwingStrategy
from .trigger_detector import TriggerDetector


class StrategyEngine:
    """Run compatibility strategies and compose the final detector-based signal."""

    def __init__(self) -> None:
        self.strategies = [
            ScalpingStrategy(),
            IntradayStrategy(),
            SwingStrategy(),
        ]
        self.context_evaluator = ContextEvaluator()
        self.setup_detector = SetupDetector()
        self.trigger_detector = TriggerDetector()
        self.decision_composer = DecisionComposer()
        self.legacy_adapter = LegacyStrategyAdapter()

    def evaluate(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        raw_results = [
            strategy.evaluate(enriched_data)
            for strategy in self.strategies
        ]
        normalized_results = [
            self.legacy_adapter.normalize_result(result)
            for result in raw_results
        ]

        result_map = {
            result["strategy"]: result
            for result in normalized_results
        }

        scalping_result = result_map.get(
            "scalping",
            self.legacy_adapter.build_default_result("scalping"),
        )
        intraday_result = result_map.get(
            "intraday",
            self.legacy_adapter.build_default_result("intraday"),
        )
        swing_result = result_map.get(
            "swing",
            self.legacy_adapter.build_default_result("swing"),
        )

        all_results = [
            scalping_result,
            intraday_result,
            swing_result,
        ]

        context_result = self._safe_context_evaluation(enriched_data)
        setup_result = self._safe_setup_detection(enriched_data)
        trigger_result = self._safe_trigger_detection(enriched_data)

        composed_decision = self.decision_composer.compose(
            context_result=context_result,
            setup_result=setup_result,
            trigger_result=trigger_result,
            legacy_results=all_results,
        )
        selected_result = composed_decision.to_selected_result()

        decision_layers = {
            "context_layer": context_result.get("layer", {}),
            "setup_layer": {
                "name": "setup",
                "state": setup_result.get("setup"),
                "bias": self._state_to_bias(str(setup_result.get("setup", "neutral"))),
                "confidence": self._layer_strength(str(setup_result.get("setup", "neutral"))),
                "reason": setup_result.get("reason"),
            },
            "trigger_layer": {
                "name": "trigger",
                "state": trigger_result.get("trigger"),
                "bias": self._state_to_bias(str(trigger_result.get("trigger", "neutral"))),
                "confidence": self._layer_strength(str(trigger_result.get("trigger", "neutral"))),
                "reason": trigger_result.get("reason"),
            },
        }

        return {
            "selected_strategy": composed_decision.selected_strategy,
            "selected_result": selected_result,
            "scalping_result": scalping_result,
            "intraday_result": intraday_result,
            "swing_result": swing_result,
            "strategy_results": all_results,
            "decision_layers": decision_layers,
            "context_result": context_result,
            "setup_result": setup_result,
            "trigger_result": trigger_result,
        }

    @staticmethod
    def _safe_detector_result(
        *,
        detector_name: str,
        state_key: str,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            state_key: "neutral",
            "bias": "neutral",
            "confidence": 0.0,
            "reason": f"{detector_name} unavailable: {error}",
            "details": {"error": str(error)},
            "layer": {
                "name": detector_name,
                "state": "neutral",
                "bias": "neutral",
                "confidence": 0.0,
                "reason": f"{detector_name} unavailable: {error}",
                "details": {"error": str(error)},
            },
        }

    def _safe_context_evaluation(
        self,
        enriched_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        try:
            return self.context_evaluator.evaluate(enriched_data)
        except Exception as error:
            return self._safe_detector_result(
                detector_name="context",
                state_key="context",
                error=error,
            )

    def _safe_setup_detection(
        self,
        enriched_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        try:
            return self.setup_detector.detect(enriched_data)
        except Exception as error:
            return self._safe_detector_result(
                detector_name="setup",
                state_key="setup",
                error=error,
            )

    def _safe_trigger_detection(
        self,
        enriched_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        try:
            return self.trigger_detector.detect(enriched_data)
        except Exception as error:
            return self._safe_detector_result(
                detector_name="trigger",
                state_key="trigger",
                error=error,
            )

    @staticmethod
    def _layer_strength(state: str) -> float:
        mapping = {
            "long": 1.0,
            "short": 1.0,
            "improving_long": 0.68,
            "improving_short": 0.68,
            "early_recovery_long": 0.58,
            "early_recovery_short": 0.58,
        }
        return mapping.get(state, 0.0)

    @staticmethod
    def _state_to_bias(state: str) -> str:
        if "long" in state:
            return "bullish"
        if "short" in state:
            return "bearish"
        return "neutral"
