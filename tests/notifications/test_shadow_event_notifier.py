from __future__ import annotations

import json
from pathlib import Path

import src.notifications.shadow_event_notifier as shadow_event_notifier_module
from src.notifications.shadow_event_notifier import ShadowEventNotifier


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
            {
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
                        "score_delta": 0.2,
                    }
                ],
            },
            {
                "generated_at": "2026-03-18T00:05:00+00:00",
                "selection_status": "selected",
                "selected_symbol": "ETHUSDT",
                "selected_strategy": "trend",
                "selected_horizon": "1h",
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
            },
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
    assert result["event_count"] == 3
    assert result["sent_count"] == 3
    assert result["failure_count"] == 0
    assert result["delivery_available"] is True
    assert len(sender.messages) == 3
    assert "Shadow Event: First Selected" in sender.messages[0]


def test_shadow_event_notifier_catches_sender_failures_without_raising(
    tmp_path: Path,
) -> None:
    shadow_output_path = tmp_path / "edge_selection_shadow.jsonl"
    _write_jsonl(
        shadow_output_path,
        [
            {
                "generated_at": "2026-03-18T00:00:00+00:00",
                "selection_status": "abstain",
                "ranking": [],
            },
            {
                "generated_at": "2026-03-18T00:05:00+00:00",
                "selection_status": "selected",
                "selected_symbol": "BTCUSDT",
                "selected_strategy": "swing",
                "selected_horizon": "4h",
                "selection_score": 8.8,
                "selection_confidence": 0.95,
                "ranking": [
                    {
                        "symbol": "BTCUSDT",
                        "strategy": "swing",
                        "horizon": "4h",
                        "selection_score": 8.8,
                        "selection_confidence": 0.95,
                        "selected_stability_label": "multi_horizon_confirmed",
                        "score_delta": 0.4,
                    }
                ],
            },
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
            {
                "generated_at": "2026-03-18T00:00:00+00:00",
                "selection_status": "abstain",
                "ranking": [],
            },
            {
                "generated_at": "2026-03-18T00:05:00+00:00",
                "selection_status": "selected",
                "selected_symbol": "BTCUSDT",
                "selected_strategy": "swing",
                "selected_horizon": "4h",
                "selection_score": 8.8,
                "selection_confidence": 0.95,
                "ranking": [
                    {
                        "symbol": "BTCUSDT",
                        "strategy": "swing",
                        "horizon": "4h",
                        "selection_score": 8.8,
                        "selection_confidence": 0.95,
                        "selected_stability_label": "multi_horizon_confirmed",
                        "score_delta": 0.4,
                    }
                ],
            },
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
            {
                "generated_at": "2026-03-18T00:00:00+00:00",
                "selection_status": "abstain",
                "ranking": [],
            },
            {
                "generated_at": "2026-03-18T00:05:00+00:00",
                "selection_status": "abstain",
                "ranking": [
                    {
                        "symbol": "BTCUSDT",
                        "strategy": "swing",
                        "horizon": "4h",
                        "selection_score": 8.8,
                        "selection_confidence": 0.95,
                        "selected_stability_label": "multi_horizon_confirmed",
                        "score_delta": 0.4,
                    }
                ],
            },
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


def _write_jsonl(path: Path, payloads: list[dict[str, object]]) -> None:
    lines = [json.dumps(payload, ensure_ascii=False) for payload in payloads]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")