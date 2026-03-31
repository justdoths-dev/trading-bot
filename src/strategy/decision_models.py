"""Shared decision-layer models for detector-based strategy composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayerResult:
    """Normalized result for one decision layer."""

    name: str
    state: str
    bias: str
    confidence: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "bias": self.bias,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "details": self.details,
        }


@dataclass(slots=True)
class ComposedDecision:
    """Final strategy decision composed from context, setup, and trigger layers."""

    selected_strategy: str
    signal: str
    bias: str
    confidence: float
    reason: str
    timeframe_summary: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_selected_result(self) -> dict[str, Any]:
        return {
            "strategy": self.selected_strategy,
            "signal": self.signal,
            "confidence": round(float(self.confidence), 4),
            "bias": self.bias,
            "reason": self.reason,
            "timeframe_summary": self.timeframe_summary,
            "debug": self.debug,
        }
