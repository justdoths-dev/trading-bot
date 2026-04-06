from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from src.ai_scaffold.contracts import (
    AIInterpretationRequest,
    RiskContextSummary,
    StrategyContextSummary,
    TimeframeSummary,
)

DEFAULT_PROMPT_VERSION = "ai_scaffold_prompt_v1"

_VALID_BIASES = {"long", "short", "neutral"}

# semantic timeframe ordering
_TIMEFRAME_ORDER = {
    "1m": 1,
    "3m": 2,
    "5m": 3,
    "15m": 4,
    "30m": 5,
    "1h": 6,
    "2h": 7,
    "4h": 8,
    "6h": 9,
    "12h": 10,
    "1d": 11,
}


def build_ai_interpretation_request(
    *,
    symbol: str,
    timeframe_summaries: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | None,
    strategy_context: Mapping[str, Any] | None = None,
    risk_context: Mapping[str, Any] | None = None,
    as_of: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> AIInterpretationRequest:
    return AIInterpretationRequest(
        symbol=_normalize_text(symbol, fallback="UNKNOWN"),
        as_of=_normalize_optional_text(as_of),
        timeframes=_normalize_timeframes(timeframe_summaries),
        strategy_context=_normalize_strategy_context(strategy_context),
        risk_context=_normalize_risk_context(risk_context),
        prompt_version=_normalize_text(
            prompt_version,
            fallback=DEFAULT_PROMPT_VERSION,
        ),
    )


def build_payload(
    *,
    symbol: str,
    timeframe_summaries: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | None,
    strategy_context: Mapping[str, Any] | None = None,
    risk_context: Mapping[str, Any] | None = None,
    as_of: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    request = build_ai_interpretation_request(
        symbol=symbol,
        timeframe_summaries=timeframe_summaries,
        strategy_context=strategy_context,
        risk_context=risk_context,
        as_of=as_of,
        prompt_version=prompt_version,
    )
    return asdict(request)


def _normalize_timeframes(
    timeframe_summaries: Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | None,
) -> list[TimeframeSummary]:
    if timeframe_summaries is None:
        return []

    entries: list[dict[str, Any]] = []

    if isinstance(timeframe_summaries, Mapping):
        for timeframe, raw in timeframe_summaries.items():
            if not isinstance(raw, Mapping):
                continue
            entries.append(
                {
                    "timeframe": timeframe,
                    "market_bias": raw.get("market_bias", raw.get("bias")),
                    "momentum_state": raw.get("momentum_state", raw.get("momentum")),
                    "volatility_state": raw.get("volatility_state", raw.get("volatility")),
                    "trigger_state": raw.get("trigger_state", raw.get("trigger")),
                }
            )
    else:
        for raw in timeframe_summaries:
            if not isinstance(raw, Mapping):
                continue
            entries.append(
                {
                    "timeframe": raw.get("timeframe"),
                    "market_bias": raw.get("market_bias", raw.get("bias")),
                    "momentum_state": raw.get("momentum_state", raw.get("momentum")),
                    "volatility_state": raw.get("volatility_state", raw.get("volatility")),
                    "trigger_state": raw.get("trigger_state", raw.get("trigger")),
                }
            )

    entries.sort(key=lambda x: _timeframe_sort_key(x.get("timeframe")))

    return [
        TimeframeSummary(
            timeframe=_normalize_text(entry["timeframe"], fallback="unknown"),
            market_bias=_normalize_bias(entry["market_bias"]),
            momentum_state=_normalize_text(entry["momentum_state"], fallback="unknown"),
            volatility_state=_normalize_text(entry["volatility_state"], fallback="unknown"),
            trigger_state=_normalize_text(entry["trigger_state"], fallback="unknown"),
        )
        for entry in entries
    ]


def _timeframe_sort_key(tf: Any) -> int:
    tf_norm = _normalize_text(tf, fallback="unknown")
    return _TIMEFRAME_ORDER.get(tf_norm, 999)


def _normalize_strategy_context(
    strategy_context: Mapping[str, Any] | None,
) -> StrategyContextSummary:
    raw = dict(strategy_context or {})
    return StrategyContextSummary(
        strategy_name=_normalize_text(
            raw.get("strategy_name", raw.get("strategy")),
            fallback="unknown",
        ),
        directional_bias=_normalize_bias(raw.get("directional_bias", raw.get("bias"))),
        setup_state=_normalize_text(raw.get("setup_state"), fallback="unknown"),
        selection_state=_normalize_text(raw.get("selection_state"), fallback="unknown"),
    )


def _normalize_risk_context(risk_context: Mapping[str, Any] | None) -> RiskContextSummary:
    raw = dict(risk_context or {})
    return RiskContextSummary(
        execution_allowed=bool(raw.get("execution_allowed", False)),
        risk_reward_state=_normalize_text(raw.get("risk_reward_state"), fallback="unknown"),
        exposure_state=_normalize_text(raw.get("exposure_state"), fallback="unknown"),
        volatility_risk_state=_normalize_text(
            raw.get("volatility_risk_state"),
            fallback="unknown",
        ),
    )


def _normalize_bias(value: Any) -> str:
    if value is None:
        return "neutral"

    text = str(value).strip().lower()

    if text in _VALID_BIASES:
        return text

    if "long" in text:
        return "long"
    if "short" in text:
        return "short"

    return "neutral"


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    return text


def _normalize_text(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    return text.lower().replace(" ", "_")