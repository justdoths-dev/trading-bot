from __future__ import annotations

from typing import Any

from src.notifications.shadow_event_types import (
    ShadowCandidateSnapshot,
    ShadowEvent,
    ShadowEventType,
)

_MAX_REASON_CODES = 5


def format_shadow_event(event: ShadowEvent) -> str:
    header = {
        ShadowEventType.FIRST_SELECTED_EVENT: "Shadow Event: First Selected",
        ShadowEventType.STABILITY_TRANSITION_EVENT: "Shadow Event: Stability Transition",
        ShadowEventType.SCORE_SURGE_EVENT: "Shadow Event: Score Surge",
    }[event.event_type]

    lines: list[str] = [header]
    lines.append(f"Generated: {event.generated_at or 'n/a'}")
    lines.append(f"Selection status: {event.selection_status or 'n/a'}")

    if event.current_candidate is not None:
        lines.append(f"Candidate: {_format_identity(event.current_candidate)}")
        lines.append(
            "Current: "
            f"score={_format_number(event.current_candidate.selection_score)}, "
            f"confidence={_format_number(event.current_candidate.selection_confidence)}, "
            f"stability={event.current_candidate.selected_stability_label or 'n/a'}, "
            f"delta={_format_signed_number(event.current_candidate.score_delta)}, "
            f"source={event.current_candidate.source_preference or 'n/a'}"
        )

        current_reason_codes = _format_reason_codes(event.current_candidate.reason_codes)
        if current_reason_codes:
            lines.append(f"Current reasons: {current_reason_codes}")

    if event.previous_candidate is not None:
        lines.append(
            "Previous: "
            f"score={_format_number(event.previous_candidate.selection_score)}, "
            f"confidence={_format_number(event.previous_candidate.selection_confidence)}, "
            f"stability={event.previous_candidate.selected_stability_label or 'n/a'}, "
            f"delta={_format_signed_number(event.previous_candidate.score_delta)}, "
            f"source={event.previous_candidate.source_preference or 'n/a'}"
        )

        previous_reason_codes = _format_reason_codes(event.previous_candidate.reason_codes)
        if previous_reason_codes:
            lines.append(f"Previous reasons: {previous_reason_codes}")

    detail_line = _build_detail_line(event)
    if detail_line:
        lines.append(detail_line)

    return "\n".join(lines).strip()


def _build_detail_line(event: ShadowEvent) -> str:
    if event.event_type is ShadowEventType.FIRST_SELECTED_EVENT:
        previous_status = event.metadata.get("previous_selection_status") or "n/a"
        return f"Transition: {previous_status} -> selected"

    if event.event_type is ShadowEventType.STABILITY_TRANSITION_EVENT:
        previous_stability = event.metadata.get("previous_stability_label") or "n/a"
        current_stability = event.metadata.get("current_stability_label") or "n/a"
        return f"Transition: {previous_stability} -> {current_stability}"

    if event.event_type is ShadowEventType.SCORE_SURGE_EVENT:
        threshold = event.score_surge_threshold
        return (
            "Trigger: "
            f"score_delta={_format_signed_number(event.metadata.get('score_delta'))} "
            f">= threshold={_format_number(threshold)}"
        )

    return ""


def _format_identity(candidate: ShadowCandidateSnapshot) -> str:
    return candidate.identity_text


def _format_reason_codes(reason_codes: tuple[str, ...]) -> str:
    if not reason_codes:
        return ""

    displayed = list(reason_codes[:_MAX_REASON_CODES])
    if len(reason_codes) > _MAX_REASON_CODES:
        displayed.append("...")

    return ", ".join(displayed)


def _format_number(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _format_signed_number(value: Any) -> str:
    number = _coerce_float(value)
    if number is None:
        return "n/a"
    return f"{number:+.2f}".rstrip("0").rstrip(".")


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None