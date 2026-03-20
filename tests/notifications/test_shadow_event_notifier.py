from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import src.notifications.shadow_event_notifier as shadow_event_notifier_module
from src.notifications.shadow_event_notifier import (
    ShadowEventNotifier,
    has_incremental_score_surge,
    has_meaningful_state_change,
    is_within_cooldown,
    should_send_shadow_event,
)
from src.notifications.shadow_event_types import (
    ShadowCandidateSnapshot,
    ShadowEvent,
    ShadowEventType,
)


class FakeSender:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, message: str, parse_mode: str | None = None) -> dict[str, bool]:
        self.messages.append(message)
        return {"ok": True}


class FailingSender:
    def send_message(self, message: str, parse_mode: str | None = None) -> dict[str, bool]:
        raise RuntimeError("telegram offline")


def test_shadow_event_notifier_reads_latest_records_and_sends_events(
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(
                generated_at="2026-03-18T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="trend",
                horizon="1h",
                score=6.4,
                confidence=0.72,
                stability="single_horizon_only",
                score_delta=0.2,
            ),
            _selected_record(
                generated_at="2026-03-18T00:05:00+00:00",
                symbol="ETHUSDT",
                strategy="trend",
                horizon="1h",
                score=7.6,
                confidence=0.89,
                stability="multi_horizon_confirmed",
                score_delta=1.2,
            ),
        ],
    )
    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        score_surge_threshold=1.0,
        sender=sender,
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["detected_event_count"] == 3
    assert result["event_count"] == 1
    assert result["sent_count"] == 1
    assert result["failure_count"] == 0
    assert result["delivery_available"] is True
    assert result["suppressed_count"] == 2
    assert len(sender.messages) == 1
    assert "Shadow Event: First Selected" in sender.messages[0]


def test_shadow_event_notifier_catches_sender_failures_without_raising(
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-18T00:00:00+00:00"),
            _selected_record(
                generated_at="2026-03-18T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="single_horizon_only",
                score_delta=0.4,
            ),
        ],
    )
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        sender=FailingSender(),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 1
    assert result["sent_count"] == 0
    assert result["failure_count"] == 1
    assert result["delivery_available"] is True
    assert result["failures"] == ["telegram offline"]


def test_shadow_event_notifier_returns_reason_when_sender_configuration_is_missing(
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-18T00:00:00+00:00"),
            _selected_record(
                generated_at="2026-03-18T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="single_horizon_only",
                score_delta=0.4,
            ),
        ],
    )
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        bot_token="",
        chat_id="",
        sender=None,
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 1
    assert result["sent_count"] == 0
    assert result["failure_count"] == 0
    assert result["delivery_available"] is False
    assert "configuration missing" in result["reason"].lower()


def test_shadow_event_notifier_returns_no_event_reason_when_no_events_are_detected(
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-18T00:00:00+00:00"),
            _abstain_record(
                generated_at="2026-03-18T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="single_horizon_only",
                score_delta=0.4,
            ),
        ],
    )
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        score_surge_threshold=5.0,
        sender=FakeSender(),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 0
    assert result["sent_count"] == 0
    assert result["failure_count"] == 0
    assert result["delivery_available"] is True
    assert result["reason"] == "No shadow events detected."


def test_shadow_event_notifier_treats_missing_file_as_no_events(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing_shadow_output.jsonl"
    notifier = ShadowEventNotifier(
        shadow_output_path=missing_path,
        sender=FakeSender(),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 0
    assert result["sent_count"] == 0
    assert result["failure_count"] == 0
    assert result["delivery_available"] is True
    assert result["reason"] == "No shadow events detected."


def test_shadow_event_notifier_returns_top_level_fallback_when_reader_raises(
    monkeypatch,
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"

    def _raise_reader_error(*args, **kwargs):
        raise RuntimeError("reader exploded")

    monkeypatch.setattr(
        shadow_event_notifier_module,
        "read_edge_selection_shadow_outputs",
        _raise_reader_error,
    )

    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        sender=FakeSender(),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 0
    assert result["sent_count"] == 0
    assert result["failure_count"] == 1
    assert result["delivery_available"] is False
    assert result["failures"] == [
        "Shadow event notification flow failed before delivery."
    ]


def test_unchanged_repeated_event_is_suppressed(tmp_path: Path) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"
    identity_key = "BTCUSDT|swing|4h"

    _write_jsonl(
        shadow_output_path,
        [
            _selected_record(
                generated_at="2026-03-20T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="multi_horizon_confirmed",
                score_delta=0.5,
            ),
            _selected_record(
                generated_at="2026-03-20T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="multi_horizon_confirmed",
                score_delta=1.2,
            ),
        ],
    )
    _write_state(
        state_path,
        {
            identity_key: {
                "selection_score": 8.8,
                "selection_confidence": 0.95,
                "selection_status": "selected",
                "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
                "selected_stability_label": "multi_horizon_confirmed",
                "last_sent_at": "2026-03-20T00:04:00+00:00",
                "last_generated_at": "2026-03-20T00:00:00+00:00",
            }
        },
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        score_surge_threshold=1.0,
        cooldown_seconds=3600,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:05:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is False
    assert result["event_count"] == 0
    assert result["detected_event_count"] == 1
    assert result["suppressed_count"] == 1
    assert result["reason"] == "No notifiable shadow events after deduplication."
    assert sender.messages == []


def test_incremental_score_surge_sends_alert(tmp_path: Path) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"
    identity_key = "BTCUSDT|swing|4h"

    _write_jsonl(
        shadow_output_path,
        [
            _selected_record(
                generated_at="2026-03-20T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.8,
                confidence=0.95,
                stability="multi_horizon_confirmed",
                score_delta=0.5,
            ),
            _selected_record(
                generated_at="2026-03-20T00:10:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=10.0,
                confidence=0.96,
                stability="multi_horizon_confirmed",
                score_delta=1.2,
            ),
        ],
    )
    _write_state(
        state_path,
        {
            identity_key: {
                "selection_score": 8.8,
                "selection_confidence": 0.95,
                "selection_status": "selected",
                "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
                "selected_stability_label": "multi_horizon_confirmed",
                "last_sent_at": "2026-03-20T00:04:00+00:00",
                "last_generated_at": "2026-03-20T00:00:00+00:00",
            }
        },
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        score_surge_threshold=1.0,
        cooldown_seconds=3600,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:10:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["event_count"] == 1
    assert result["sent_count"] == 1
    assert result["suppressed_count"] == 0
    assert len(sender.messages) == 1
    assert "Shadow Event: Score Surge" in sender.messages[0]


def test_reason_or_status_transition_sends_alert(tmp_path: Path) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"
    identity_key = "BTCUSDT|swing|4h"

    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-20T00:00:00+00:00"),
            _selected_record(
                generated_at="2026-03-20T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.4,
                confidence=0.93,
                stability="single_horizon_only",
                score_delta=0.4,
                reason_codes=["STATE_TRANSITION_ALERT"],
            ),
        ],
    )
    _write_state(
        state_path,
        {
            identity_key: {
                "selection_score": 8.4,
                "selection_confidence": 0.93,
                "selection_status": "abstain",
                "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
                "selected_stability_label": "single_horizon_only",
                "last_sent_at": "2026-03-20T00:01:00+00:00",
                "last_generated_at": "2026-03-20T00:00:00+00:00",
            }
        },
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:05:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["event_count"] == 1
    assert result["sent_count"] == 1
    assert len(sender.messages) == 1
    assert "Shadow Event: First Selected" in sender.messages[0]


def test_different_horizon_counts_as_different_identity(tmp_path: Path) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"

    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-20T00:00:00+00:00"),
            _selected_record(
                generated_at="2026-03-20T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=8.2,
                confidence=0.92,
                stability="single_horizon_only",
                score_delta=0.2,
            ),
        ],
    )
    _write_state(
        state_path,
        {
            "BTCUSDT|swing|1h": {
                "selection_score": 8.2,
                "selection_confidence": 0.92,
                "selection_status": "selected",
                "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
                "selected_stability_label": "multi_horizon_confirmed",
                "last_sent_at": "2026-03-20T00:01:00+00:00",
                "last_generated_at": "2026-03-20T00:00:00+00:00",
            }
        },
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:05:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["event_count"] == 1
    assert result["sent_count"] == 1
    assert len(sender.messages) == 1
    assert "btcusdt / swing / 4h" in sender.messages[0].lower()


def test_cooldown_suppression_helper_works() -> None:
    event = ShadowEvent(
        event_type=ShadowEventType.SCORE_SURGE_EVENT,
        generated_at="2026-03-20T00:05:00+00:00",
        selection_status="selected",
        current_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="4h",
            selection_score=8.8,
            selection_confidence=0.95,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=1.2,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        previous_candidate=None,
        score_surge_threshold=1.0,
        metadata={"score_delta": 1.2},
    )
    last_state = {
        "selection_score": 8.8,
        "selection_confidence": 0.95,
        "selection_status": "selected",
        "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
        "selected_stability_label": "multi_horizon_confirmed",
        "last_sent_at": "2026-03-20T00:04:45+00:00",
    }

    current_time = _dt("2026-03-20T00:05:00+00:00")

    assert has_meaningful_state_change(
        {
            "selection_status": "selected",
            "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
            "selected_stability_label": "multi_horizon_confirmed",
        },
        last_state,
    ) is False
    assert has_incremental_score_surge(
        {"selection_score": 8.8},
        last_state,
        score_surge_threshold=1.0,
    ) is False
    assert is_within_cooldown(
        last_state,
        cooldown_seconds=3600,
        current_time=current_time,
    ) is True
    assert should_send_shadow_event(
        event,
        last_notified_state=last_state,
        score_surge_threshold=1.0,
        cooldown_seconds=3600,
        current_time=current_time,
    ) is False


def test_same_batch_duplicate_identity_is_suppressed_after_first_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"

    _write_jsonl(
        shadow_output_path,
        [
            _selected_record(
                generated_at="2026-03-20T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="1h",
                score=13.2,
                confidence=0.95,
                stability="multi_horizon_confirmed",
                score_delta=1.5,
            ),
            _selected_record(
                generated_at="2026-03-20T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="1h",
                score=13.2,
                confidence=0.95,
                stability="multi_horizon_confirmed",
                score_delta=1.5,
            ),
        ],
    )

    event = ShadowEvent(
        event_type=ShadowEventType.SCORE_SURGE_EVENT,
        generated_at="2026-03-20T00:05:00+00:00",
        selection_status="abstain",
        current_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="1h",
            selection_score=13.2,
            selection_confidence=0.95,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=1.5,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        previous_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="1h",
            selection_score=11.7,
            selection_confidence=0.93,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=0.5,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        score_surge_threshold=1.0,
        metadata={"score_delta": 1.5},
    )

    monkeypatch.setattr(
        shadow_event_notifier_module,
        "detect_shadow_events",
        lambda **kwargs: [event, event],
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        score_surge_threshold=1.0,
        cooldown_seconds=3600,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:05:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["detected_event_count"] == 2
    assert result["event_count"] == 1
    assert result["sent_count"] == 1
    assert result["suppressed_count"] == 1
    assert len(sender.messages) == 1

    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "BTCUSDT|swing|1h" in saved_state
    assert saved_state["BTCUSDT|swing|1h"]["selection_score"] == 13.2
    assert saved_state["BTCUSDT|swing|1h"]["selection_status"] == "abstain"


def test_same_state_outside_cooldown_is_still_suppressed() -> None:
    event = ShadowEvent(
        event_type=ShadowEventType.SCORE_SURGE_EVENT,
        generated_at="2026-03-20T10:05:00+00:00",
        selection_status="selected",
        current_candidate=ShadowCandidateSnapshot(
            symbol="BTCUSDT",
            strategy="swing",
            horizon="4h",
            selection_score=8.8,
            selection_confidence=0.95,
            selected_stability_label="multi_horizon_confirmed",
            score_delta=1.2,
            source_preference="latest",
            reason_codes=("ELIGIBLE_CONSERVATIVE_PASS",),
        ),
        previous_candidate=None,
        score_surge_threshold=1.0,
        metadata={"score_delta": 1.2},
    )
    last_state = {
        "selection_score": 8.8,
        "selection_confidence": 0.95,
        "selection_status": "selected",
        "reason_codes": ["ELIGIBLE_CONSERVATIVE_PASS"],
        "selected_stability_label": "multi_horizon_confirmed",
        "last_sent_at": "2026-03-20T00:00:00+00:00",
    }

    assert should_send_shadow_event(
        event,
        last_notified_state=last_state,
        score_surge_threshold=1.0,
        cooldown_seconds=3600,
        current_time=_dt("2026-03-20T10:05:00+00:00"),
    ) is False


def test_state_file_updates_after_successful_send(tmp_path: Path) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    state_path = tmp_path / "shadow_event_state.json"

    _write_jsonl(
        shadow_output_path,
        [
            _abstain_record(generated_at="2026-03-20T00:00:00+00:00"),
            _selected_record(
                generated_at="2026-03-20T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                horizon="4h",
                score=9.4,
                confidence=0.97,
                stability="multi_horizon_confirmed",
                score_delta=1.4,
            ),
        ],
    )

    sender = FakeSender()
    notifier = ShadowEventNotifier(
        shadow_output_path=shadow_output_path,
        state_path=state_path,
        score_surge_threshold=1.0,
        sender=sender,
        now_provider=lambda: _dt("2026-03-20T00:05:30+00:00"),
    )

    result = notifier.notify_latest_events()

    assert result["sent"] is True
    assert result["sent_count"] >= 1

    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "BTCUSDT|swing|4h" in saved_state
    assert saved_state["BTCUSDT|swing|4h"]["selection_score"] == 9.4
    assert saved_state["BTCUSDT|swing|4h"]["selection_status"] == "selected"
    assert saved_state["BTCUSDT|swing|4h"]["reason_codes"] == [
        "ELIGIBLE_CONSERVATIVE_PASS"
    ]


def _selected_record(
    *,
    generated_at: str,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    score: float = 8.8,
    confidence: float = 0.95,
    stability: str = "multi_horizon_confirmed",
    score_delta: float = 0.4,
    reason_codes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "selection_status": "selected",
        "selected_symbol": symbol,
        "selected_strategy": strategy,
        "selected_horizon": horizon,
        "selection_score": score,
        "selection_confidence": confidence,
        "reason_codes": reason_codes or ["ELIGIBLE_CONSERVATIVE_PASS"],
        "ranking": [
            {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "selection_score": score,
                "selection_confidence": confidence,
                "selected_stability_label": stability,
                "score_delta": score_delta,
                "reason_codes": reason_codes or ["ELIGIBLE_CONSERVATIVE_PASS"],
                "source_preference": "latest",
            }
        ],
    }


def _abstain_record(
    *,
    generated_at: str,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    score: float = 6.0,
    confidence: float = 0.65,
    stability: str = "single_horizon_only",
    score_delta: float = 0.2,
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "selection_status": "abstain",
        "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
        "ranking": [
            {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "selection_score": score,
                "selection_confidence": confidence,
                "selected_stability_label": stability,
                "score_delta": score_delta,
                "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
                "source_preference": "latest",
            }
        ],
    }


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    lines = [json.dumps(payload, ensure_ascii=False) for payload in payloads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)