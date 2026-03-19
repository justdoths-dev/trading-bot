from __future__ import annotations

from src.notifications.shadow_event_detector import detect_shadow_events
from src.notifications.shadow_event_types import ShadowEventType


def test_detect_shadow_events_emits_first_selected_event() -> None:
    previous_record = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "ranking": [],
    }
    current_record = {
        "generated_at": "2026-03-18T00:05:00+00:00",
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
        "selected_strategy": "swing",
        "selected_horizon": "4h",
        "selection_score": 8.7,
        "selection_confidence": 0.93,
        "ranking": [
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selection_score": 8.7,
                "selection_confidence": 0.93,
                "selected_stability_label": "multi_horizon_confirmed",
                "score_delta": 0.4,
            }
        ],
    }

    events = detect_shadow_events(current_record, previous_record)

    assert [event.event_type for event in events] == [ShadowEventType.FIRST_SELECTED_EVENT]
    assert events[0].current_candidate is not None
    assert events[0].current_candidate.symbol == "BTCUSDT"


def test_detect_shadow_events_emits_stability_transition_for_same_candidate() -> None:
    previous_record = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selection_score": 6.4,
                "selection_confidence": 0.72,
                "selected_stability_label": "single_horizon_only",
                "score_delta": 0.4,
            }
        ],
    }
    current_record = {
        "generated_at": "2026-03-18T00:05:00+00:00",
        "selection_status": "selected",
        "selected_symbol": "ETHUSDT",
        "selected_strategy": "trend",
        "selected_horizon": "1h",
        "selection_score": 7.1,
        "selection_confidence": 0.88,
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selection_score": 7.1,
                "selection_confidence": 0.88,
                "selected_stability_label": "multi_horizon_confirmed",
                "score_delta": 0.7,
            }
        ],
    }

    events = detect_shadow_events(current_record, previous_record)

    event_types = [event.event_type for event in events]
    assert ShadowEventType.STABILITY_TRANSITION_EVENT in event_types


def test_detect_shadow_events_emits_score_surge_when_threshold_is_met() -> None:
    current_record = {
        "generated_at": "2026-03-18T00:05:00+00:00",
        "selection_status": "abstain",
        "ranking": [
            {
                "symbol": "SOLUSDT",
                "strategy": "breakout",
                "horizon": "15m",
                "selection_score": 4.3,
                "selection_confidence": 0.55,
                "selected_stability_label": "single_horizon_only",
                "score_delta": 1.2,
            }
        ],
    }

    events = detect_shadow_events(current_record, None, score_surge_threshold=1.0)

    assert [event.event_type for event in events] == [ShadowEventType.SCORE_SURGE_EVENT]
    assert events[0].metadata["score_delta"] == 1.2


def test_detect_shadow_events_does_not_emit_stability_transition_for_different_candidate() -> None:
    previous_record = {
        "selection_status": "abstain",
        "ranking": [
            {
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "selected_stability_label": "single_horizon_only",
            }
        ],
    }
    current_record = {
        "selection_status": "selected",
        "selected_symbol": "ETHUSDT",
        "selected_strategy": "trend",
        "selected_horizon": "1h",
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selected_stability_label": "multi_horizon_confirmed",
            }
        ],
    }

    events = detect_shadow_events(current_record, previous_record)

    assert ShadowEventType.STABILITY_TRANSITION_EVENT not in {
        event.event_type for event in events
    }


def test_detect_shadow_events_fallback_match_when_horizon_missing() -> None:
    previous_record = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "ranking": [],
    }
    current_record = {
        "generated_at": "2026-03-18T00:05:00+00:00",
        "selection_status": "selected",
        "selected_symbol": "ETHUSDT",
        "selected_strategy": "trend",
        "selected_horizon": None,
        "selection_score": 7.6,
        "selection_confidence": 0.89,
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selection_score": 7.6,
                "selection_confidence": 0.89,
                "selected_stability_label": "multi_horizon_confirmed",
                "score_delta": 1.2,
            }
        ],
    }

    events = detect_shadow_events(current_record, previous_record, score_surge_threshold=1.0)

    event_types = {event.event_type for event in events}
    assert ShadowEventType.FIRST_SELECTED_EVENT in event_types
    assert ShadowEventType.SCORE_SURGE_EVENT in event_types

    first_selected_event = next(
        event for event in events if event.event_type is ShadowEventType.FIRST_SELECTED_EVENT
    )
    assert first_selected_event.current_candidate is not None
    assert first_selected_event.current_candidate.symbol == "ETHUSDT"
    assert first_selected_event.current_candidate.strategy == "trend"
    assert first_selected_event.current_candidate.horizon == "1h"
    assert first_selected_event.current_candidate.selected_stability_label == "multi_horizon_confirmed"
    assert first_selected_event.current_candidate.score_delta == 1.2


def test_detect_shadow_events_no_fallback_when_multiple_candidates() -> None:
    previous_record = {
        "generated_at": "2026-03-18T00:00:00+00:00",
        "selection_status": "abstain",
        "ranking": [],
    }
    current_record = {
        "generated_at": "2026-03-18T00:05:00+00:00",
        "selection_status": "selected",
        "selected_symbol": "ETHUSDT",
        "selected_strategy": "trend",
        "selected_horizon": None,
        "selection_score": 7.6,
        "selection_confidence": 0.89,
        "ranking": [
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "15m",
                "selection_score": 7.4,
                "selection_confidence": 0.86,
                "selected_stability_label": "single_horizon_only",
                "score_delta": 0.7,
            },
            {
                "symbol": "ETHUSDT",
                "strategy": "trend",
                "horizon": "1h",
                "selection_score": 7.6,
                "selection_confidence": 0.89,
                "selected_stability_label": "multi_horizon_confirmed",
                "score_delta": 1.2,
            },
        ],
    }

    events = detect_shadow_events(current_record, previous_record)

    assert [event.event_type for event in events] == [ShadowEventType.FIRST_SELECTED_EVENT]
    assert events[0].current_candidate is not None
    assert events[0].current_candidate.symbol == "ETHUSDT"
    assert events[0].current_candidate.strategy == "trend"
    assert events[0].current_candidate.horizon is None
    assert events[0].current_candidate.selected_stability_label is None
    assert events[0].current_candidate.score_delta is None
