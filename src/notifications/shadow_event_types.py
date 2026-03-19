from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ShadowEventType(str, Enum):
    FIRST_SELECTED_EVENT = "FIRST_SELECTED_EVENT"
    STABILITY_TRANSITION_EVENT = "STABILITY_TRANSITION_EVENT"
    SCORE_SURGE_EVENT = "SCORE_SURGE_EVENT"


@dataclass(frozen=True)
class ShadowCandidateSnapshot:
    symbol: str | None
    strategy: str | None
    horizon: str | None
    selection_score: float | None
    selection_confidence: float | None
    selected_stability_label: str | None
    score_delta: float | None
    source_preference: str | None
    reason_codes: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple[str | None, str | None, str | None]:
        return (self.symbol, self.strategy, self.horizon)

    @property
    def has_complete_identity(self) -> bool:
        return all(value is not None for value in self.identity)

    @property
    def identity_text(self) -> str:
        symbol = self.symbol or "n/a"
        strategy = self.strategy or "n/a"
        horizon = self.horizon or "n/a"
        return f"{symbol} / {strategy} / {horizon}"


@dataclass(frozen=True)
class ShadowEvent:
    event_type: ShadowEventType
    generated_at: str | None
    selection_status: str | None
    current_candidate: ShadowCandidateSnapshot | None
    previous_candidate: ShadowCandidateSnapshot | None
    score_surge_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_identity(self) -> tuple[str | None, str | None, str | None] | None:
        if self.current_candidate is None:
            return None
        return self.current_candidate.identity

    @property
    def candidate_identity_text(self) -> str:
        if self.current_candidate is None:
            return "n/a"
        return self.current_candidate.identity_text