from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.notifications.shadow_event_detector import (
    DEFAULT_SCORE_SURGE_THRESHOLD,
    detect_shadow_events,
)
from src.notifications.shadow_event_formatter import format_shadow_event
from src.notifications.shadow_event_types import ShadowCandidateSnapshot, ShadowEvent
from src.research.edge_selection_shadow_writer import (
    DEFAULT_SHADOW_OUTPUT_PATH,
    read_edge_selection_shadow_outputs,
)
from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)

DEFAULT_EVENT_BOT_TOKEN_ENV = "TELEGRAM_OPS_BOT_TOKEN"
DEFAULT_EVENT_CHAT_ID_ENV = "TELEGRAM_RESEARCH_CHAT_ID"
DEFAULT_SCORE_SURGE_THRESHOLD_ENV = "TELEGRAM_SHADOW_SCORE_SURGE_THRESHOLD"
DEFAULT_EVENT_COOLDOWN_SECONDS = 3600
DEFAULT_EVENT_STATE_FILENAME = "shadow_event_notify_state.json"


class ShadowEventNotifier:
    """Read recent shadow outputs, deduplicate observer-only events, and notify safely."""

    def __init__(
        self,
        *,
        shadow_output_path: Path | None = None,
        state_path: Path | None = None,
        score_surge_threshold: float | None = None,
        cooldown_seconds: int = DEFAULT_EVENT_COOLDOWN_SECONDS,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._shadow_output_path = (
            Path(shadow_output_path)
            if shadow_output_path is not None
            else DEFAULT_SHADOW_OUTPUT_PATH
        )
        self._state_path = (
            Path(state_path)
            if state_path is not None
            else self._shadow_output_path.parent / DEFAULT_EVENT_STATE_FILENAME
        )
        self._score_surge_threshold = _resolve_score_surge_threshold(
            score_surge_threshold
        )
        self._cooldown_seconds = max(int(cooldown_seconds), 0)
        self._bot_token = (bot_token or os.getenv(DEFAULT_EVENT_BOT_TOKEN_ENV, "")).strip()
        self._chat_id = (chat_id or os.getenv(DEFAULT_EVENT_CHAT_ID_ENV, "")).strip()
        self._sender = sender
        self._now_provider = now_provider or _utc_now

    def notify_latest_events(self) -> dict[str, Any]:
        try:
            current_record, previous_record = self._load_latest_records()
            detected_events = detect_shadow_events(
                current_record=current_record,
                previous_record=previous_record,
                score_surge_threshold=self._score_surge_threshold,
            )
            notifiable_events, notification_state = self._filter_events(detected_events)
            result = self._send_events(notifiable_events)
            result["detected_event_count"] = len(detected_events)
            result["suppressed_count"] = max(
                len(detected_events) - result["event_count"],
                0,
            )

            successful_events = result.pop("_successful_events", [])
            if successful_events:
                self._record_successful_notifications(
                    successful_events,
                    notification_state,
                )
                self._write_notification_state(notification_state)

            if not detected_events:
                result["reason"] = "No shadow events detected."
            elif not notifiable_events:
                result["reason"] = "No notifiable shadow events after deduplication."

            return result
        except Exception:
            logger.exception("Shadow event notification flow failed.")
            return {
                "sent": False,
                "event_count": 0,
                "sent_count": 0,
                "failure_count": 1,
                "delivery_available": False,
                "failures": ["Shadow event notification flow failed before delivery."],
                "detected_event_count": 0,
                "suppressed_count": 0,
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

    def _filter_events(
        self,
        events: list[ShadowEvent],
    ) -> tuple[list[ShadowEvent], dict[str, dict[str, Any]]]:
        notification_state = self._load_notification_state()
        working_state = dict(notification_state)
        current_time = self._now_provider()
        filtered: list[ShadowEvent] = []

        for event in events:
            identity_key = _candidate_identity_key(event.current_candidate)
            last_notified_state = (
                working_state.get(identity_key)
                if identity_key is not None
                else None
            )
            if should_send_shadow_event(
                event,
                last_notified_state=last_notified_state,
                score_surge_threshold=self._score_surge_threshold,
                cooldown_seconds=self._cooldown_seconds,
                current_time=current_time,
            ):
                filtered.append(event)
                if identity_key is not None:
                    # In-memory working state update prevents same-run duplicates
                    # from passing the filter again within the same notifier call.
                    working_state[identity_key] = _build_notified_state(
                        event,
                        sent_at=current_time,
                    )

        return filtered, notification_state

    def _record_successful_notifications(
        self,
        events: list[ShadowEvent],
        notification_state: dict[str, dict[str, Any]],
    ) -> None:
        current_time = self._now_provider()
        for event in events:
            identity_key = _candidate_identity_key(event.current_candidate)
            if identity_key is None:
                continue
            notification_state[identity_key] = _build_notified_state(
                event,
                sent_at=current_time,
            )

    def _load_notification_state(self) -> dict[str, dict[str, Any]]:
        if not self._state_path.exists():
            return {}
        if not self._state_path.is_file():
            logger.warning(
                "Shadow event state path is not a file: %s",
                self._state_path,
            )
            return {}

        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read shadow event notification state %s: %s",
                self._state_path,
                exc,
            )
            return {}

        if not isinstance(payload, dict):
            return {}

        state: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, dict):
                state[key] = value
        return state

    def _write_notification_state(
        self,
        notification_state: dict[str, dict[str, Any]],
    ) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(notification_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            logger.exception(
                "Failed to write shadow event notification state: %s",
                self._state_path,
            )
            raise

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
                "_successful_events": [],
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
                "_successful_events": [],
            }

        sent_count = 0
        failures: list[str] = []
        successful_events: list[ShadowEvent] = []

        for event in events:
            try:
                sender.send_message(build_shadow_event_message(event), parse_mode=None)
                sent_count += 1
                successful_events.append(event)
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
            "_successful_events": successful_events,
        }

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender


def should_send_shadow_event(
    event: ShadowEvent,
    *,
    last_notified_state: dict[str, Any] | None,
    score_surge_threshold: float,
    cooldown_seconds: int,
    current_time: datetime,
) -> bool:
    current_candidate = event.current_candidate
    if current_candidate is None or not current_candidate.has_complete_identity:
        return True

    if last_notified_state is None:
        return True

    current_state = _build_current_state(event)

    if has_meaningful_state_change(current_state, last_notified_state):
        return True

    if has_incremental_score_surge(
        current_state,
        last_notified_state,
        score_surge_threshold=score_surge_threshold,
    ):
        return True

    # Current policy:
    # - cooldown does not enable same-state refresh notifications
    # - unchanged state remains suppressed regardless of cooldown expiry
    if is_within_cooldown(
        last_notified_state,
        cooldown_seconds=cooldown_seconds,
        current_time=current_time,
    ):
        return False

    return False


def has_meaningful_state_change(
    current_state: dict[str, Any],
    last_notified_state: dict[str, Any],
) -> bool:
    return (
        _normalize_text(current_state.get("selection_status"))
        != _normalize_text(last_notified_state.get("selection_status"))
        or _normalize_reason_codes(current_state.get("reason_codes"))
        != _normalize_reason_codes(last_notified_state.get("reason_codes"))
        or _normalize_text(current_state.get("selected_stability_label"))
        != _normalize_text(last_notified_state.get("selected_stability_label"))
    )


def has_incremental_score_surge(
    current_state: dict[str, Any],
    last_notified_state: dict[str, Any],
    *,
    score_surge_threshold: float,
) -> bool:
    current_score = _normalize_number(current_state.get("selection_score"))
    previous_score = _normalize_number(last_notified_state.get("selection_score"))
    if current_score is None or previous_score is None:
        return False

    return (current_score - previous_score) >= float(score_surge_threshold)


def is_within_cooldown(
    last_notified_state: dict[str, Any],
    *,
    cooldown_seconds: int,
    current_time: datetime,
) -> bool:
    if cooldown_seconds <= 0:
        return False

    last_sent_at = _parse_datetime(last_notified_state.get("last_sent_at"))
    if last_sent_at is None:
        return False

    return current_time < (last_sent_at + timedelta(seconds=int(cooldown_seconds)))


def build_shadow_event_message(event: ShadowEvent) -> str:
    return format_shadow_event(event)


def _build_current_state(event: ShadowEvent) -> dict[str, Any]:
    candidate = event.current_candidate
    return {
        "selection_score": None if candidate is None else candidate.selection_score,
        "selection_confidence": (
            None if candidate is None else candidate.selection_confidence
        ),
        "selection_status": event.selection_status,
        "reason_codes": () if candidate is None else candidate.reason_codes,
        "selected_stability_label": (
            None if candidate is None else candidate.selected_stability_label
        ),
    }


def _build_notified_state(
    event: ShadowEvent,
    *,
    sent_at: datetime,
) -> dict[str, Any]:
    current_state = _build_current_state(event)
    return {
        "selection_score": current_state["selection_score"],
        "selection_confidence": current_state["selection_confidence"],
        "selection_status": current_state["selection_status"],
        "reason_codes": list(_normalize_reason_codes(current_state["reason_codes"])),
        "selected_stability_label": current_state["selected_stability_label"],
        "last_sent_at": sent_at.astimezone(UTC).isoformat(),
        "last_generated_at": _normalize_text(event.generated_at),
    }


def _candidate_identity_key(candidate: ShadowCandidateSnapshot | None) -> str | None:
    if candidate is None or not candidate.has_complete_identity:
        return None
    return "|".join(candidate.identity)


def _normalize_reason_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()

    normalized = {
        text
        for item in value
        if (text := _normalize_text(item)) is not None
    }
    return tuple(sorted(normalized))


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = _normalize_text(value)
    if text is None:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


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

