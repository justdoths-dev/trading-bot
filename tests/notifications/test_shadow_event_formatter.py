from __future__ import annotations

from src.notifications.shadow_event_formatter import format_shadow_event
from src.notifications.shadow_event_types import (
    ShadowCandidateSnapshot,
    ShadowEvent,
    ShadowEventType,
)


def test_format_shadow_event_compacts_score_surge_message() -> None:
    event = ShadowEvent(
        event_type=ShadowEventType.SCORE_SURGE_EVENT,
        generated_at="2026-03-21T00:00:00+00:00",
        selection_status="abstain",
        current_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="1h",
            selection_score=13.2,
            selection_confidence=0.95,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=1.4,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        previous_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="1h",
            selection_score=12.2,
            selection_confidence=0.95,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=0.4,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        score_surge_threshold=1.0,
        metadata={"score_delta": 1.0},
    )

    message = format_shadow_event(event)
    lines = message.splitlines()

    assert lines[0] == "Shadow Event: Score Surge"
    assert "Candidate: BTCUSDT / swing / 1h" in message
    assert "Status: abstain" in message
    assert "Score: 13.2 | Confidence: 0.95" in message
    assert "Stability: multi_horizon_confirmed" in message
    assert "Reason: ELIGIBLE_CONSERVATIVE_PASS" in message
    assert "Trigger: score surge >= 1" in message
    assert "Generated: 2026-03-21T00:00:00+00:00" in message
    assert len(lines) <= 8


def test_format_shadow_event_omits_previous_line_when_values_match() -> None:
    candidate = ShadowCandidateSnapshot(
        symbol="ETHUSDT",
        strategy="trend",
        horizon="4h",
        selection_score=8.8,
        selection_confidence=0.91,
        selected_stability_label="single_horizon_only",
        score_delta=0.4,
        source_preference="latest",
        reason_codes=(),
    )
    event = ShadowEvent(
        event_type=ShadowEventType.STABILITY_TRANSITION_EVENT,
        generated_at="2026-03-21T00:00:00+00:00",
        selection_status="selected",
        current_candidate=candidate,
        previous_candidate=candidate,
        metadata={
            "previous_stability_label": "single_horizon_only",
            "current_stability_label": "single_horizon_only",
        },
    )

    message = format_shadow_event(event)

    assert "Previous:" not in message
