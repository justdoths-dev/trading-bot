from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Bias = Literal["long", "short", "neutral"]
Confidence = Literal["low", "medium", "high"]


@dataclass(slots=True)
class TimeframeSummary:
    timeframe: str
    market_bias: Bias = "neutral"
    momentum_state: str = "unknown"
    volatility_state: str = "unknown"
    trigger_state: str = "unknown"


@dataclass(slots=True)
class StrategyContextSummary:
    strategy_name: str = "unknown"
    directional_bias: Bias = "neutral"
    setup_state: str = "unknown"
    selection_state: str = "unknown"


@dataclass(slots=True)
class RiskContextSummary:
    execution_allowed: bool = False
    risk_reward_state: str = "unknown"
    exposure_state: str = "unknown"
    volatility_risk_state: str = "unknown"


@dataclass(slots=True)
class AIInterpretationRequest:
    symbol: str
    as_of: str | None = None
    timeframes: list[TimeframeSummary] = field(default_factory=list)
    strategy_context: StrategyContextSummary = field(
        default_factory=StrategyContextSummary
    )
    risk_context: RiskContextSummary = field(default_factory=RiskContextSummary)
    prompt_version: str = "ai_scaffold_prompt_v1"


@dataclass(slots=True)
class AIInterpretationResponse:
    bias: Bias
    confidence: Confidence
    regime_label: str
    reasoning: list[str] = field(default_factory=list)
    caution_flags: list[str] = field(default_factory=list)
    recommended_action: str = "wait"
    model_version: str = "static_mock_v1"
    prompt_version: str = "ai_scaffold_prompt_v1"