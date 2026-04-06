from __future__ import annotations

from typing import Protocol

from src.ai_scaffold.contracts import (
    AIInterpretationRequest,
    AIInterpretationResponse,
)


class AIInterpreter(Protocol):
    """Interface for read-only AI interpretation providers."""

    def interpret(self, request: AIInterpretationRequest) -> AIInterpretationResponse:
        ...


class StaticMockInterpreter:
    """Deterministic mock interpreter for read-only AI scaffold shadow analysis."""

    def __init__(self, model_version: str = "static_mock_v1") -> None:
        self.model_version = model_version

    def interpret(self, request: AIInterpretationRequest) -> AIInterpretationResponse:
        timeframe_biases = [tf.market_bias for tf in request.timeframes]
        conflict = self._has_timeframe_conflict(timeframe_biases)
        bias = self._resolve_bias(request, conflict)
        regime_label = self._resolve_regime_label(request, conflict)
        confidence = self._resolve_confidence(request, bias, conflict)
        reasoning = self._build_reasoning(
            request=request,
            timeframe_biases=timeframe_biases,
            conflict=conflict,
            regime_label=regime_label,
        )
        caution_flags = self._build_caution_flags(request, conflict)
        recommended_action = self._resolve_recommended_action(
            request=request,
            bias=bias,
            conflict=conflict,
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

    def _resolve_bias(
        self,
        request: AIInterpretationRequest,
        conflict: bool,
    ) -> str:
        if conflict:
            return "neutral"

        directional_bias = request.strategy_context.directional_bias
        if directional_bias in {"long", "short", "neutral"}:
            return directional_bias

        timeframe_biases = [tf.market_bias for tf in request.timeframes]
        if not timeframe_biases:
            return "neutral"

        if all(bias == "long" for bias in timeframe_biases):
            return "long"
        if all(bias == "short" for bias in timeframe_biases):
            return "short"
        return "neutral"

    def _resolve_regime_label(
        self,
        request: AIInterpretationRequest,
        conflict: bool,
    ) -> str:
        volatility_states = {tf.volatility_state for tf in request.timeframes}

        if "expanding" in volatility_states:
            return "volatile_expansion"

        if conflict:
            return "mixed_conditions"

        if request.strategy_context.directional_bias in {"long", "short"}:
            return "directional_trend"

        return "range_or_unclear"

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

        if request.strategy_context.selection_state != "selected":
            return "medium"

        if request.risk_context.execution_allowed is not True:
            return "medium"

        return "high"

    def _build_reasoning(
        self,
        *,
        request: AIInterpretationRequest,
        timeframe_biases: list[str],
        conflict: bool,
        regime_label: str,
    ) -> list[str]:
        return [
            f"strategy_bias={request.strategy_context.directional_bias}",
            f"timeframe_biases={timeframe_biases}",
            f"timeframe_conflict={'true' if conflict else 'false'}",
            f"execution_allowed={'true' if request.risk_context.execution_allowed else 'false'}",
            f"regime={regime_label}",
        ]

    def _build_caution_flags(
        self,
        request: AIInterpretationRequest,
        conflict: bool,
    ) -> list[str]:
        flags: list[str] = []

        if request.risk_context.execution_allowed is not True:
            flags.append("execution_blocked")

        if conflict:
            flags.append("timeframe_conflict")

        if request.strategy_context.setup_state != "confirmed":
            flags.append("setup_not_confirmed")

        if request.risk_context.risk_reward_state not in {"acceptable", "favorable"}:
            flags.append("risk_reward_unclear")

        if request.risk_context.volatility_risk_state in {"high", "elevated"}:
            flags.append("elevated_volatility")

        return flags

    def _resolve_recommended_action(
        self,
        *,
        request: AIInterpretationRequest,
        bias: str,
        conflict: bool,
    ) -> str:
        if request.risk_context.execution_allowed is not True:
            return "hold"

        if conflict or bias == "neutral":
            return "hold"

        if bias == "long":
            return "observe_long_setup"

        if bias == "short":
            return "observe_short_setup"

        return "hold"

    def _has_timeframe_conflict(self, timeframe_biases: list[str]) -> bool:
        normalized = {bias for bias in timeframe_biases if bias in {"long", "short"}}
        return len(normalized) > 1
