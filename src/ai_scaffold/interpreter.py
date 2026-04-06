from __future__ import annotations

from abc import ABC, abstractmethod

from src.ai_scaffold.contracts import (
    AIInterpretationRequest,
    AIInterpretationResponse,
)


class AIInterpreter(ABC):
    @abstractmethod
    def interpret(
        self,
        request: AIInterpretationRequest,
    ) -> AIInterpretationResponse:
        raise NotImplementedError


class StaticMockInterpreter(AIInterpreter):
    def __init__(self, model_version: str = "static_mock_v1") -> None:
        self.model_version = model_version

    def interpret(
        self,
        request: AIInterpretationRequest,
    ) -> AIInterpretationResponse:
        timeframe_biases = self._collect_timeframe_biases(request)

        conflict = len(set(timeframe_biases)) > 1

        bias = self._resolve_bias(request, timeframe_biases, conflict)
        confidence = self._resolve_confidence(request, bias, conflict)
        regime_label = self._resolve_regime_label(request)

        caution_flags: list[str] = []

        if not request.risk_context.execution_allowed:
            caution_flags.append("execution_blocked")

        if conflict:
            caution_flags.append("timeframe_conflict")

        if request.strategy_context.setup_state not in {"confirmed", "ready"}:
            caution_flags.append("setup_not_confirmed")

        if request.risk_context.risk_reward_state not in {"acceptable", "favorable"}:
            caution_flags.append("risk_reward_unclear")

        if any(tf.volatility_state == "expanding" for tf in request.timeframes):
            caution_flags.append("elevated_volatility")

        reasoning = [
            f"strategy_bias={request.strategy_context.directional_bias}",
            f"timeframe_biases={timeframe_biases}",
            f"timeframe_conflict={str(conflict).lower()}",
            f"execution_allowed={str(request.risk_context.execution_allowed).lower()}",
            f"regime={regime_label}",
        ]

        recommended_action = self._resolve_recommended_action(
            bias=bias,
            confidence=confidence,
            execution_allowed=request.risk_context.execution_allowed,
        )

        return AIInterpretationResponse(
            bias=bias,
            confidence=confidence,
            regime_label=regime_label,
            reasoning=reasoning,
            caution_flags=caution_flags,
            recommended_action=recommended_action,
            model_version=self.model_version,
            prompt_version=request.prompt_version,
        )

    def _collect_timeframe_biases(self, request: AIInterpretationRequest) -> list[str]:
        return [
            tf.market_bias
            for tf in request.timeframes
            if tf.market_bias in {"long", "short"}
        ]

    def _resolve_bias(
        self,
        request: AIInterpretationRequest,
        timeframe_biases: list[str],
        conflict: bool,
    ) -> str:
        if not request.risk_context.execution_allowed:
            return "neutral"

        if conflict:
            return "neutral"

        if request.strategy_context.directional_bias in {"long", "short"}:
            return request.strategy_context.directional_bias

        if timeframe_biases:
            if all(b == timeframe_biases[0] for b in timeframe_biases):
                return timeframe_biases[0]

        return "neutral"

    def _resolve_confidence(
        self,
        request: AIInterpretationRequest,
        bias: str,
        conflict: bool,
    ) -> str:
        if bias == "neutral":
            return "low"

        if conflict:
            return "low"

        if request.strategy_context.alignment_state != "aligned":
            return "medium"

        if request.strategy_context.setup_state not in {"confirmed", "ready"}:
            return "medium"

        if request.risk_context.risk_reward_state not in {"acceptable", "favorable"}:
            return "medium"

        return "high"

    def _resolve_regime_label(self, request: AIInterpretationRequest) -> str:
        if any(tf.volatility_state == "expanding" for tf in request.timeframes):
            return "volatile_expansion"

        biases = {
            tf.market_bias
            for tf in request.timeframes
            if tf.market_bias in {"long", "short"}
        }

        if len(biases) == 1:
            return "directional_trend"

        if len(biases) > 1:
            return "mixed_conditions"

        return "range_bound"

    def _resolve_recommended_action(
        self,
        *,
        bias: str,
        confidence: str,
        execution_allowed: bool,
    ) -> str:
        if not execution_allowed:
            return "hold"
        if bias == "neutral":
            return "wait"
        if confidence == "high":
            return f"observe_{bias}_setup"
        return f"review_{bias}_setup"
