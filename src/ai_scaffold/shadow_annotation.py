from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Mapping

from src.ai_scaffold.interpreter import AIInterpreter, StaticMockInterpreter
from src.ai_scaffold.payload_builder import build_ai_interpretation_request

AI_SCAFFOLD_SHADOW_SOURCE = "ai_scaffold_static_mock"
AI_SCAFFOLD_SHADOW_MODE = "read_only_shadow"


class AIScaffoldShadowAnnotator:
    """Build read-only AI scaffold annotations for persisted trade-analysis rows."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        interpreter: AIInterpreter | None = None,
    ) -> None:
        self.enabled = enabled
        self.interpreter = interpreter or StaticMockInterpreter()

    def annotate(
        self,
        *,
        log_record: Mapping[str, Any] | None,
        edge_selection_output: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        generated_at = self._now_iso()
        request_payload: dict[str, Any] | None = None

        try:
            request = build_ai_interpretation_request(
                symbol=self._coerce_text(
                    (log_record or {}).get("symbol"),
                    fallback="unknown",
                ),
                as_of=self._coerce_optional_text((log_record or {}).get("logged_at")),
                timeframe_summaries=self._extract_timeframe_summaries(log_record),
                strategy_context=self._extract_strategy_context(
                    log_record=log_record,
                    edge_selection_output=edge_selection_output,
                ),
                risk_context=self._extract_risk_context(log_record),
            )
            request_payload = asdict(request)
            response = self.interpreter.interpret(request)

            return {
                "enabled": True,
                "source": AI_SCAFFOLD_SHADOW_SOURCE,
                "generated_at": generated_at,
                "annotation_mode": AI_SCAFFOLD_SHADOW_MODE,
                "decision_impact": False,
                "request": request_payload,
                "response": asdict(response),
                "error": None,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "source": AI_SCAFFOLD_SHADOW_SOURCE,
                "generated_at": generated_at,
                "annotation_mode": AI_SCAFFOLD_SHADOW_MODE,
                "decision_impact": False,
                "request": request_payload or {},
                "response": {},
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    def _extract_timeframe_summaries(
        self,
        log_record: Mapping[str, Any] | None,
    ) -> Mapping[str, Mapping[str, Any]]:
        raw = (log_record or {}).get("timeframe_summary")
        if isinstance(raw, Mapping):
            return raw
        return {}

    def _extract_strategy_context(
        self,
        *,
        log_record: Mapping[str, Any] | None,
        edge_selection_output: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = log_record or {}
        rule_engine = record.get("rule_engine")
        rule_engine_dict = rule_engine if isinstance(rule_engine, Mapping) else {}
        edge_output = (
            edge_selection_output if isinstance(edge_selection_output, Mapping) else {}
        )

        selection_status = self._coerce_text(
            edge_output.get("selection_status"),
            fallback="unknown",
        )

        return {
            "strategy": rule_engine_dict.get("strategy") or record.get("selected_strategy"),
            "bias": self._normalize_bias(
                rule_engine_dict.get("signal")
                or rule_engine_dict.get("bias")
                or record.get("bias")
            ),
            "setup_state": "ready" if selection_status == "selected" else "unknown",
            "selection_state": selection_status,
        }

    def _extract_risk_context(
        self,
        log_record: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = log_record or {}
        risk = record.get("risk")
        risk_dict = risk if isinstance(risk, Mapping) else {}

        risk_reward_ratio = self._safe_float(risk_dict.get("risk_reward_ratio"))
        if risk_reward_ratio is None:
            risk_reward_state = "unknown"
        elif risk_reward_ratio >= 2.0:
            risk_reward_state = "favorable"
        elif risk_reward_ratio >= 1.0:
            risk_reward_state = "acceptable"
        else:
            risk_reward_state = "unfavorable"

        execution_allowed = bool(risk_dict.get("execution_allowed", False))

        return {
            "execution_allowed": execution_allowed,
            "risk_reward_state": risk_reward_state,
            "exposure_state": "allowed" if execution_allowed else "blocked",
            "volatility_risk_state": self._coerce_text(
                risk_dict.get("volatility_state"),
                fallback="unknown",
            ),
        }

    def _normalize_bias(self, value: Any) -> str:
        text = self._coerce_text(value, fallback="neutral")
        if text in {"bullish", "long", "buy"}:
            return "long"
        if text in {"bearish", "short", "sell"}:
            return "short"
        return "neutral"

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_optional_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_text(self, value: Any, *, fallback: str) -> str:
        text = self._coerce_optional_text(value)
        if text is None:
            return fallback
        return text.lower().replace(" ", "_")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
