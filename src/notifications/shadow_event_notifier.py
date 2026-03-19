from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.notifications.shadow_event_detector import (
    DEFAULT_SCORE_SURGE_THRESHOLD,
    detect_shadow_events,
)
from src.notifications.shadow_event_formatter import format_shadow_event
from src.notifications.shadow_event_types import ShadowEvent
from src.research.edge_selection_shadow_writer import (
    DEFAULT_SHADOW_OUTPUT_PATH,
    read_edge_selection_shadow_outputs,
)
from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)

DEFAULT_EVENT_BOT_TOKEN_ENV = "TELEGRAM_OPS_BOT_TOKEN"
DEFAULT_EVENT_CHAT_ID_ENV = "TELEGRAM_RESEARCH_CHAT_ID"
DEFAULT_SCORE_SURGE_THRESHOLD_ENV = "TELEGRAM_SHADOW_SCORE_SURGE_THRESHOLD"


class ShadowEventNotifier:
    """Read recent shadow outputs, detect observer-only events, and notify safely."""

    def __init__(
        self,
        *,
        shadow_output_path: Path | None = None,
        score_surge_threshold: float | None = None,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ) -> None:
        self._shadow_output_path = (
            Path(shadow_output_path)
            if shadow_output_path is not None
            else DEFAULT_SHADOW_OUTPUT_PATH
        )
        self._score_surge_threshold = _resolve_score_surge_threshold(
            score_surge_threshold
        )
        self._bot_token = (bot_token or os.getenv(DEFAULT_EVENT_BOT_TOKEN_ENV, "")).strip()
        self._chat_id = (chat_id or os.getenv(DEFAULT_EVENT_CHAT_ID_ENV, "")).strip()
        self._sender = sender

    def notify_latest_events(self) -> dict[str, Any]:
        try:
            current_record, previous_record = self._load_latest_records()
            events = detect_shadow_events(
                current_record=current_record,
                previous_record=previous_record,
                score_surge_threshold=self._score_surge_threshold,
            )
            return self._send_events(events)
        except Exception:
            logger.exception("Shadow event notification flow failed.")
            return {
                "sent": False,
                "event_count": 0,
                "sent_count": 0,
                "failure_count": 1,
                "delivery_available": False,
                "failures": ["Shadow event notification flow failed before delivery."],
            }

    def _load_latest_records(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        records = read_edge_selection_shadow_outputs(self._shadow_output_path)
        if not records:
            return None, None
        if len(records) == 1:
            return records[-1], None
        return records[-1], records[-2]

    def _send_events(self, events: list[ShadowEvent]) -> dict[str, Any]:
        if not events:
            return {
                "sent": False,
                "event_count": 0,
                "sent_count": 0,
                "failure_count": 0,
                "delivery_available": True,
                "failures": [],
                "reason": "No shadow events detected.",
            }

        sender = self._get_sender()
        if sender is None:
            return {
                "sent": False,
                "event_count": len(events),
                "sent_count": 0,
                "failure_count": 0,
                "delivery_available": False,
                "failures": [],
                "reason": (
                    "Shadow event notifier configuration missing. "
                    f"Set {DEFAULT_EVENT_BOT_TOKEN_ENV} and {DEFAULT_EVENT_CHAT_ID_ENV}."
                ),
            }

        sent_count = 0
        failures: list[str] = []

        for event in events:
            try:
                sender.send_message(format_shadow_event(event), parse_mode=None)
                sent_count += 1
            except Exception as exc:
                logger.exception(
                    "Failed to send shadow event notification: event_type=%s",
                    getattr(event, "event_type", "unknown"),
                )
                failures.append(str(exc))

        return {
            "sent": sent_count > 0,
            "event_count": len(events),
            "sent_count": sent_count,
            "failure_count": len(failures),
            "delivery_available": True,
            "failures": failures,
        }

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender


def _resolve_score_surge_threshold(configured_value: float | None) -> float:
    if configured_value is not None:
        return float(configured_value)

    raw_env_value = os.getenv(DEFAULT_SCORE_SURGE_THRESHOLD_ENV)
    if raw_env_value is None:
        return DEFAULT_SCORE_SURGE_THRESHOLD

    try:
        return float(raw_env_value)
    except ValueError:
        logger.warning(
            "Invalid %s value %r. Falling back to default threshold %s.",
            DEFAULT_SCORE_SURGE_THRESHOLD_ENV,
            raw_env_value,
            DEFAULT_SCORE_SURGE_THRESHOLD,
        )
        return DEFAULT_SCORE_SURGE_THRESHOLD
