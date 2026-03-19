from __future__ import annotations

from typing import Any

from src.notifications.shadow_event_types import (
    ShadowCandidateSnapshot,
    ShadowEvent,
    ShadowEventType,
)

DEFAULT_SCORE_SURGE_THRESHOLD = 1.0
SELECTED_STATUS = "selected"
SINGLE_HORIZON_ONLY = "single_horizon_only"
MULTI_HORIZON_CONFIRMED = "multi_horizon_confirmed"


def detect_shadow_events(
    current_record: dict[str, Any] | None,
    previous_record: dict[str, Any] | None,
    *,
    score_surge_threshold: float = DEFAULT_SCORE_SURGE_THRESHOLD,
) -> list[ShadowEvent]:
    """
    Detect observer-only shadow events from the latest and previous shadow records.

    Important design constraints:
    - Pure detection logic only
    - No file I/O
    - No notification sending
    - No pipeline side effects

    Event semantics:
    1. FIRST_SELECTED_EVENT
       Fires when selection_status changes from non-selected to selected.

    2. STABILITY_TRANSITION_EVENT
       Fires when the same candidate moves from single_horizon_only
       to multi_horizon_confirmed.

    3. SCORE_SURGE_EVENT
       Fires when the candidate's score_delta crosses the configured threshold.
       score_delta is sourced from the existing score drift pipeline, which
       represents a prior-score-to-latest-score delta for the candidate group.
    """
    current_payload = current_record if isinstance(current_record, dict) else {}
    previous_payload = previous_record if isinstance(previous_record, dict) else {}

    current_candidate = _extract_relevant_candidate(current_payload)
    previous_candidate = _extract_relevant_candidate(previous_payload)

    threshold = float(score_surge_threshold)
    events: list[ShadowEvent] = []

    if (
        current_payload.get("selection_status") == SELECTED_STATUS
        and previous_payload.get("selection_status") != SELECTED_STATUS
    ):
        events.append(
            ShadowEvent(
                event_type=ShadowEventType.FIRST_SELECTED_EVENT,
                generated_at=_as_text(current_payload.get("generated_at")),
                selection_status=_as_text(current_payload.get("selection_status")),
                current_candidate=current_candidate,
                previous_candidate=previous_candidate,
                metadata={
                    "previous_selection_status": _as_text(
                        previous_payload.get("selection_status")
                    )
                },
            )
        )

    if _same_identity(current_candidate, previous_candidate):
        previous_stability = _candidate_stability(previous_candidate)
        current_stability = _candidate_stability(current_candidate)

        if (
            previous_stability == SINGLE_HORIZON_ONLY
            and current_stability == MULTI_HORIZON_CONFIRMED
        ):
            events.append(
                ShadowEvent(
                    event_type=ShadowEventType.STABILITY_TRANSITION_EVENT,
                    generated_at=_as_text(current_payload.get("generated_at")),
                    selection_status=_as_text(current_payload.get("selection_status")),
                    current_candidate=current_candidate,
                    previous_candidate=previous_candidate,
                    metadata={
                        "previous_stability_label": previous_stability,
                        "current_stability_label": current_stability,
                    },
                )
            )

    current_score_delta = _candidate_score_delta(current_candidate)
    if (
        current_candidate is not None
        and current_score_delta is not None
        and current_score_delta >= threshold
    ):
        events.append(
            ShadowEvent(
                event_type=ShadowEventType.SCORE_SURGE_EVENT,
                generated_at=_as_text(current_payload.get("generated_at")),
                selection_status=_as_text(current_payload.get("selection_status")),
                current_candidate=current_candidate,
                previous_candidate=previous_candidate
                if _same_identity(current_candidate, previous_candidate)
                else None,
                score_surge_threshold=threshold,
                metadata={
                    "score_delta": current_score_delta,
                },
            )
        )

    return events


def _extract_relevant_candidate(
    record: dict[str, Any],
) -> ShadowCandidateSnapshot | None:
    """
    Select the candidate snapshot most relevant for event comparison.

    Selection rule:
    - If the record is selected, build a selected-candidate snapshot first,
      then enrich it with the matching ranking row when possible.
    - Otherwise, use the top-ranked candidate from the ranking list.

    Assumption:
    The ranking list is already sorted by the engine's ranking key, so the first
    valid ranking item represents the top candidate for this shadow record.
    """
    ranking_items = _ranking_items(record)

    if record.get("selection_status") == SELECTED_STATUS:
        selected_candidate = _build_selected_candidate_snapshot(record)
        if selected_candidate is not None:
            matching_candidate = _find_matching_candidate(
                selected_candidate=selected_candidate,
                ranking_items=ranking_items,
            )
            if matching_candidate is not None:
                return _merge_candidate_snapshots(selected_candidate, matching_candidate)
            return selected_candidate

    return _extract_top_ranked_candidate(ranking_items)


def _ranking_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = record.get("ranking")
    if not isinstance(ranking, list):
        return []
    return [item for item in ranking if isinstance(item, dict)]


def _extract_top_ranked_candidate(
    ranking_items: list[dict[str, Any]],
) -> ShadowCandidateSnapshot | None:
    for item in ranking_items:
        return _build_candidate_snapshot(item)
    return None


def _build_selected_candidate_snapshot(
    record: dict[str, Any],
) -> ShadowCandidateSnapshot | None:
    symbol = _as_text(record.get("selected_symbol"))
    strategy = _as_text(record.get("selected_strategy"))
    horizon = _as_text(record.get("selected_horizon"))

    if symbol is None and strategy is None and horizon is None:
        return None

    return ShadowCandidateSnapshot(
        symbol=symbol,
        strategy=strategy,
        horizon=horizon,
        selection_score=_to_float(record.get("selection_score")),
        selection_confidence=_to_float(record.get("selection_confidence")),
        selected_stability_label=_as_text(record.get("selected_stability_label")),
        score_delta=_to_float(record.get("score_delta")),
        source_preference=_as_text(record.get("source_preference")),
        reason_codes=_to_reason_codes(record.get("reason_codes")),
    )


def _build_candidate_snapshot(candidate: dict[str, Any]) -> ShadowCandidateSnapshot:
    return ShadowCandidateSnapshot(
        symbol=_as_text(candidate.get("symbol")),
        strategy=_as_text(candidate.get("strategy")),
        horizon=_as_text(candidate.get("horizon")),
        selection_score=_to_float(candidate.get("selection_score")),
        selection_confidence=_to_float(candidate.get("selection_confidence")),
        selected_stability_label=_as_text(candidate.get("selected_stability_label")),
        score_delta=_to_float(candidate.get("score_delta")),
        source_preference=_as_text(candidate.get("source_preference")),
        reason_codes=_to_reason_codes(candidate.get("reason_codes")),
    )


def _find_matching_candidate(
    *,
    selected_candidate: ShadowCandidateSnapshot,
    ranking_items: list[dict[str, Any]],
) -> ShadowCandidateSnapshot | None:
    """
    Find the ranking-row candidate that matches the selected candidate.

    Matching policy:
    1. Exact identity match on (symbol, strategy, horizon)
    2. Conservative fallback:
       If horizon is missing on the selected candidate, allow a unique
       (symbol, strategy) match only when exactly one ranking row matches.

    We deliberately avoid broad fuzzy matching because false merges are more
    dangerous than partial enrichment in this observer-only context.
    """
    exact_identity = selected_candidate.identity
    exact_matches: list[ShadowCandidateSnapshot] = []

    for item in ranking_items:
        snapshot = _build_candidate_snapshot(item)
        if snapshot.identity == exact_identity:
            exact_matches.append(snapshot)

    if exact_matches:
        return exact_matches[0]

    selected_symbol = selected_candidate.symbol
    selected_strategy = selected_candidate.strategy
    selected_horizon = selected_candidate.horizon

    if selected_symbol is None or selected_strategy is None or selected_horizon is not None:
        return None

    fallback_matches: list[ShadowCandidateSnapshot] = []
    for item in ranking_items:
        snapshot = _build_candidate_snapshot(item)
        if (
            snapshot.symbol == selected_symbol
            and snapshot.strategy == selected_strategy
        ):
            fallback_matches.append(snapshot)

    if len(fallback_matches) == 1:
        return fallback_matches[0]

    return None


def _merge_candidate_snapshots(
    primary: ShadowCandidateSnapshot,
    fallback: ShadowCandidateSnapshot,
) -> ShadowCandidateSnapshot:
    return ShadowCandidateSnapshot(
        symbol=primary.symbol or fallback.symbol,
        strategy=primary.strategy or fallback.strategy,
        horizon=primary.horizon or fallback.horizon,
        selection_score=(
            primary.selection_score
            if primary.selection_score is not None
            else fallback.selection_score
        ),
        selection_confidence=(
            primary.selection_confidence
            if primary.selection_confidence is not None
            else fallback.selection_confidence
        ),
        selected_stability_label=(
            primary.selected_stability_label or fallback.selected_stability_label
        ),
        score_delta=(
            primary.score_delta
            if primary.score_delta is not None
            else fallback.score_delta
        ),
        source_preference=primary.source_preference or fallback.source_preference,
        reason_codes=primary.reason_codes or fallback.reason_codes,
    )


def _same_identity(
    current_candidate: ShadowCandidateSnapshot | None,
    previous_candidate: ShadowCandidateSnapshot | None,
) -> bool:
    if current_candidate is None or previous_candidate is None:
        return False

    current_identity = current_candidate.identity
    previous_identity = previous_candidate.identity

    if any(value is None for value in current_identity + previous_identity):
        return False

    return current_identity == previous_identity


def _candidate_stability(candidate: ShadowCandidateSnapshot | None) -> str | None:
    return None if candidate is None else candidate.selected_stability_label


def _candidate_score_delta(candidate: ShadowCandidateSnapshot | None) -> float | None:
    return None if candidate is None else candidate.score_delta


def _to_reason_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        text
        for item in value
        if (text := _as_text(item)) is not None
    )


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
