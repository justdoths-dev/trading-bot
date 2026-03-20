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
    candidate = event.current_candidate or event.previous_candidate
    lines.append(f"Candidate: {_format_identity(candidate) if candidate else 'n/a'}")
    lines.append(f"Status: {event.selection_status or 'n/a'}")

    if event.current_candidate is not None:
        lines.append(
            "Score: "
            f"{_format_number(event.current_candidate.selection_score)} | "
            f"Confidence: {_format_number(event.current_candidate.selection_confidence)}"
        )
        lines.append(
            "Stability: "
            f"{event.current_candidate.selected_stability_label or 'n/a'}"
        )

        reason_line = _build_reason_line(event)
        if reason_line:
            lines.append(reason_line)

    previous_line = _build_previous_line(event)
    if previous_line:
        lines.append(previous_line)

    trigger_line = _build_trigger_line(event)
    if trigger_line:
        lines.append(trigger_line)

    lines.append(f"Generated: {event.generated_at or 'n/a'}")
    return "\n".join(lines).strip()


def _build_reason_line(event: ShadowEvent) -> str:
    candidate = event.current_candidate
    if candidate is None:
        return ""

    current_reason_codes = _format_reason_codes(candidate.reason_codes)
    if current_reason_codes:
        return f"Reason: {current_reason_codes}"

    if event.event_type is ShadowEventType.FIRST_SELECTED_EVENT:
        previous_status = event.metadata.get("previous_selection_status") or "n/a"
        return f"Reason: transition from {previous_status}"

    if event.event_type is ShadowEventType.STABILITY_TRANSITION_EVENT:
        previous_stability = event.metadata.get("previous_stability_label") or "n/a"
        current_stability = event.metadata.get("current_stability_label") or "n/a"
        if previous_stability != current_stability:
            return f"Reason: {previous_stability} -> {current_stability}"

    return ""


def _build_previous_line(event: ShadowEvent) -> str:
    if event.event_type is ShadowEventType.SCORE_SURGE_EVENT:
        return ""

    current = event.current_candidate
    previous = event.previous_candidate
    if current is None or previous is None:
        return ""

    differences: list[str] = []

    if _format_number(previous.selection_score) != _format_number(current.selection_score):
        differences.append(
            f"score {_format_number(previous.selection_score)} -> {_format_number(current.selection_score)}"
        )
    if _format_number(previous.selection_confidence) != _format_number(
        current.selection_confidence
    ):
        differences.append(
            "confidence "
            f"{_format_number(previous.selection_confidence)} -> "
            f"{_format_number(current.selection_confidence)}"
        )
    if (
        (previous.selected_stability_label or "n/a")
        != (current.selected_stability_label or "n/a")
    ):
        differences.append(
            "stability "
            f"{previous.selected_stability_label or 'n/a'} -> "
            f"{current.selected_stability_label or 'n/a'}"
        )

    if not differences:
        return ""

    return "Previous: " + " | ".join(differences[:2])


def _build_trigger_line(event: ShadowEvent) -> str:
    if event.event_type is ShadowEventType.FIRST_SELECTED_EVENT:
        previous_status = event.metadata.get("previous_selection_status") or "n/a"
        return f"Trigger: first selected from {previous_status}"

    if event.event_type is ShadowEventType.STABILITY_TRANSITION_EVENT:
        previous_stability = event.metadata.get("previous_stability_label") or "n/a"
        current_stability = event.metadata.get("current_stability_label") or "n/a"
        return f"Trigger: stability {previous_stability} -> {current_stability}"

    if event.event_type is ShadowEventType.SCORE_SURGE_EVENT:
        threshold = event.score_surge_threshold
        return f"Trigger: score surge >= {_format_number(threshold)}"

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
