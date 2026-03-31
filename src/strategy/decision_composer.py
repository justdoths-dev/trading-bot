"""Compose the final strategy decision from detector outputs."""

from __future__ import annotations

from typing import Any

from .decision_models import ComposedDecision


class DecisionComposer:
    """Combine context, setup, and trigger layers into a final decision."""

    CONTEXT_STRATEGY_DEFAULTS = {
        "bullish_trend": "swing",
        "bearish_trend": "swing",
        "weak_bullish_context": "swing",
        "weak_bearish_context": "swing",
        "countertrend_bounce": "intraday",
        "countertrend_drop": "intraday",
        "conflicted": "intraday",
        "neutral": "intraday",
    }

    def compose(
        self,
        *,
        context_result: dict[str, Any],
        setup_result: dict[str, Any],
        trigger_result: dict[str, Any],
        legacy_results: list[dict[str, Any]] | None = None,
    ) -> ComposedDecision:
        context_state = str(context_result.get("context", "neutral"))
        context_bias = str(context_result.get("bias", "neutral"))
        context_conf = self._bounded(float(context_result.get("confidence", 0.0)))

        setup_state = str(setup_result.get("setup", "neutral"))
        trigger_state = str(trigger_result.get("trigger", "neutral"))

        setup_direction, setup_strength = self.layer_direction_strength(setup_state)
        trigger_direction, trigger_strength = self.layer_direction_strength(trigger_state)

        selected_strategy = self.compatibility_strategy_label(
            context_state=context_state,
            legacy_results=legacy_results,
        )
        signal = "hold"
        bias = context_bias if context_bias in {"bullish", "bearish"} else "neutral"
        reason = "No sufficiently aligned detector path was found."
        confidence = self._hold_confidence(context_conf, setup_strength, trigger_strength)

        if context_state == "bullish_trend":
            bias = "bullish"
            if setup_direction == "long" and trigger_direction == "long":
                signal = "long"
                selected_strategy = (
                    "swing"
                    if context_conf >= 0.6 and setup_strength >= 0.9 and trigger_strength >= 0.9
                    else "intraday"
                )
                confidence = self._directional_confidence(
                    context_conf=context_conf,
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.84,
                )
                reason = (
                    "Higher timeframes remain bullish and both setup and trigger "
                    "confirm continuation."
                )
            elif setup_direction == "short" or trigger_direction == "short":
                reason = (
                    "Bullish higher-timeframe context is being opposed by weaker "
                    "mid or lower timeframes, so the system stays in hold."
                )
            else:
                reason = (
                    "Bullish context exists, but setup and trigger are not both "
                    "confirmed enough for execution."
                )

        elif context_state == "bearish_trend":
            bias = "bearish"
            if setup_direction == "short" and trigger_direction == "short":
                signal = "short"
                selected_strategy = (
                    "swing"
                    if context_conf >= 0.6 and setup_strength >= 0.9 and trigger_strength >= 0.9
                    else "intraday"
                )
                confidence = self._directional_confidence(
                    context_conf=context_conf,
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.84,
                )
                reason = (
                    "Higher timeframes remain bearish and both setup and trigger "
                    "confirm continuation."
                )
            elif setup_direction == "long" or trigger_direction == "long":
                reason = (
                    "Bearish higher-timeframe context is being opposed by lower-frame "
                    "recovery, so the system stays in hold."
                )
            else:
                reason = (
                    "Bearish context exists, but setup and trigger are not both "
                    "confirmed enough for execution."
                )

        elif context_state == "weak_bullish_context":
            bias = "bullish"
            if setup_direction == "long" and trigger_direction == "long" and setup_strength >= 0.9 and trigger_strength >= 0.9:
                signal = "long"
                selected_strategy = "intraday"
                confidence = self._directional_confidence(
                    context_conf=max(context_conf, 0.42),
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.69,
                )
                reason = (
                    "Higher timeframes still lean bullish, but structure is weaker, "
                    "so only a fully aligned long can pass as intraday."
                )
            else:
                reason = (
                    "Higher timeframes still lean bullish, but the structure is not strong "
                    "enough to allow anything except a fully aligned continuation long."
                )

        elif context_state == "weak_bearish_context":
            bias = "bearish"
            if setup_direction == "short" and trigger_direction == "short" and setup_strength >= 0.9 and trigger_strength >= 0.9:
                signal = "short"
                selected_strategy = "intraday"
                confidence = self._directional_confidence(
                    context_conf=max(context_conf, 0.42),
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.69,
                )
                reason = (
                    "Higher timeframes still lean bearish, but structure is weaker, "
                    "so only a fully aligned short can pass as intraday."
                )
            else:
                reason = (
                    "Higher timeframes still lean bearish, but the structure is not strong "
                    "enough to allow anything except a fully aligned continuation short."
                )

        elif context_state == "countertrend_bounce":
            bias = "bearish"
            if setup_direction == "short" and trigger_direction == "short":
                signal = "short"
                selected_strategy = "intraday"
                confidence = self._directional_confidence(
                    context_conf=context_conf,
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.74,
                )
                reason = (
                    "The market is bouncing against a bearish higher-timeframe trend, "
                    "and lower layers now support short re-entry with the trend."
                )
            else:
                reason = (
                    "Countertrend bounce conditions are present, but only trend-following "
                    "short re-entry is allowed here, and that alignment is not ready."
                )

        elif context_state == "countertrend_drop":
            bias = "bullish"
            if setup_direction == "long" and trigger_direction == "long":
                signal = "long"
                selected_strategy = "intraday"
                confidence = self._directional_confidence(
                    context_conf=context_conf,
                    setup_strength=setup_strength,
                    trigger_strength=trigger_strength,
                    cap=0.74,
                )
                reason = (
                    "The market is dropping against a bullish higher-timeframe trend, "
                    "and lower layers now support long re-entry with the trend."
                )
            else:
                reason = (
                    "Countertrend drop conditions are present, but only trend-following "
                    "long re-entry is allowed here, and that alignment is not ready."
                )

        elif context_state == "conflicted":
            bias = "neutral"
            reason = (
                "Higher timeframes are conflicted, so the engine blocks directional trades "
                "until broader structure improves."
            )

        else:
            bias = "neutral"
            reason = (
                "Broader structure is neutral, so the engine blocks directional trades "
                "instead of promoting lower-timeframe signals on their own."
            )

        timeframe_summary = self._build_timeframe_summary(
            context_result=context_result,
            setup_result=setup_result,
            trigger_result=trigger_result,
        )
        debug = {
            "decision_path": {
                "context_state": context_state,
                "setup_state": setup_state,
                "trigger_state": trigger_state,
                "setup_direction": setup_direction,
                "trigger_direction": trigger_direction,
            },
            "context_details": context_result.get("details", {}),
            "setup_details": setup_result.get("details", {}),
            "trigger_details": trigger_result.get("details", {}),
            "legacy_results": legacy_results or [],
        }

        return ComposedDecision(
            selected_strategy=selected_strategy,
            signal=signal,
            bias=bias,
            confidence=round(confidence, 4),
            reason=reason,
            timeframe_summary=timeframe_summary,
            debug=debug,
        )

    def compatibility_strategy_label(
        self,
        *,
        context_state: str,
        legacy_results: list[dict[str, Any]] | None,
    ) -> str:
        if legacy_results:
            directional = [
                result for result in legacy_results
                if str(result.get("signal", "hold")) in {"long", "short"}
            ]
            pool = directional if directional else legacy_results
            best = max(
                pool,
                key=lambda result: float(result.get("confidence", 0.0)),
            )
            best_strategy = str(best.get("strategy", "")).strip().lower()
            if best_strategy in {"swing", "intraday", "scalping"}:
                return best_strategy

        return self.CONTEXT_STRATEGY_DEFAULTS.get(context_state, "intraday")

    def layer_direction_strength(self, state: str) -> tuple[str, float]:
        mapping = {
            "long": ("long", 1.0),
            "short": ("short", 1.0),
            "improving_long": ("long", 0.68),
            "improving_short": ("short", 0.68),
            "early_recovery_long": ("long", 0.58),
            "early_recovery_short": ("short", 0.58),
        }
        return mapping.get(state, ("neutral", 0.0))

    def state_to_bias(self, state: str) -> str:
        if "long" in state:
            return "bullish"
        if "short" in state:
            return "bearish"
        return "neutral"

    def _build_timeframe_summary(
        self,
        *,
        context_result: dict[str, Any],
        setup_result: dict[str, Any],
        trigger_result: dict[str, Any],
    ) -> dict[str, Any]:
        timeframe_summary: dict[str, Any] = {}
        context_timeframes = context_result.get("details", {}).get("timeframes", {}) or {}
        timeframe_summary.update(context_timeframes)

        context_layer = {
            "context": context_result.get("context"),
            "bias": context_result.get("bias"),
            "confidence": context_result.get("confidence"),
            "reason": context_result.get("reason"),
        }
        setup_layer = {
            "setup": setup_result.get("setup"),
            "bias": self.state_to_bias(str(setup_result.get("setup", "neutral"))),
            "confidence": self.layer_direction_strength(str(setup_result.get("setup", "neutral")))[1],
            "reason": setup_result.get("reason"),
        }
        trigger_layer = {
            "trigger": trigger_result.get("trigger"),
            "bias": self.state_to_bias(str(trigger_result.get("trigger", "neutral"))),
            "confidence": self.layer_direction_strength(str(trigger_result.get("trigger", "neutral")))[1],
            "reason": trigger_result.get("reason"),
        }

        timeframe_summary["context_layer"] = context_layer
        timeframe_summary["bias_layer"] = context_layer
        timeframe_summary["setup_layer"] = setup_layer
        timeframe_summary["trigger_layer"] = trigger_layer

        return timeframe_summary

    def _directional_confidence(
        self,
        *,
        context_conf: float,
        setup_strength: float,
        trigger_strength: float,
        cap: float,
    ) -> float:
        raw_score = (
            (0.42 * context_conf)
            + (0.31 * setup_strength)
            + (0.27 * trigger_strength)
        )
        return min(raw_score, cap)

    def _hold_confidence(
        self,
        context_conf: float,
        setup_strength: float,
        trigger_strength: float,
    ) -> float:
        return min(0.42, (0.18 * context_conf) + (0.1 * setup_strength) + (0.08 * trigger_strength))

    def _bounded(self, value: float) -> float:
        return max(0.0, min(value, 1.0))
