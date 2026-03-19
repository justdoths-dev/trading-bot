"""Research-only observational notifier for score drift and score snapshots."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHANGED_GROUPS = 6
DEFAULT_SOURCE_PREFERENCE = "n/a"
DEFAULT_STABILITY_LABEL = "insufficient_data"
DEFAULT_STRENGTH_LABEL = "insufficient_data"
SUPPRESSION_MESSAGE = "Research observation suppressed: no meaningful change detected."
DEFAULT_MAX_REASON_CODES = 3


class ResearchObservationalNotifier:
    """Deliver research observations to the configured Telegram research channel."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ) -> None:
        self._bot_token = (bot_token or os.getenv("TELEGRAM_OPS_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_RESEARCH_CHAT_ID", "")).strip()
        self._sender = sender

    def send_message(self, message: str) -> dict[str, Any]:
        sender = self._get_sender()
        if sender is None:
            return {
                "sent": False,
                "reason": (
                    "Research observational notifier configuration missing. "
                    "Set TELEGRAM_OPS_BOT_TOKEN and TELEGRAM_RESEARCH_CHAT_ID."
                ),
            }

        try:
            response = sender.send_message(message, parse_mode=None)
            return {
                "sent": True,
                "reason": "Research observation sent successfully.",
                "response": response,
            }
        except Exception as exc:
            logger.exception("Failed to send research observation.")
            return {
                "sent": False,
                "reason": f"Research observation send failed: {exc}",
            }

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("Research notifier JSON file missing: %s", path)
        return None

    if not path.is_file():
        logger.warning("Research notifier JSON path is not a file: %s", path)
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Research notifier failed to read JSON file %s: %s", path, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Research notifier found invalid JSON in %s: %s", path, exc)
        return None

    return payload if isinstance(payload, dict) else None


def detect_meaningful_change(score_drift_summary: dict[str, Any] | None) -> bool:
    if not isinstance(score_drift_summary, dict):
        return False

    drift_summary = score_drift_summary.get("drift_summary")
    if isinstance(drift_summary, dict):
        if _coerce_int(drift_summary.get("increase")) > 0:
            return True
        if _coerce_int(drift_summary.get("decrease")) > 0:
            return True

    for group in _iter_drift_groups(score_drift_summary):
        if _stability_changed(group):
            return True
        if _coerce_optional_int(group.get("horizon_count_delta")) not in (None, 0):
            return True

    return False


def extract_changed_groups(
    score_drift_summary: dict[str, Any] | None,
    limit: int = DEFAULT_MAX_CHANGED_GROUPS,
) -> list[dict[str, Any]]:
    if not isinstance(score_drift_summary, dict):
        return []

    changed_groups: list[dict[str, Any]] = []
    for item in _iter_drift_groups(score_drift_summary):
        score_delta = _coerce_optional_float(item.get("score_delta"))
        horizon_count_delta = _coerce_optional_int(item.get("horizon_count_delta"))
        stability_changed = _stability_changed(item)
        drift_direction = str(item.get("drift_direction", "insufficient_history"))

        meaningful = (
            drift_direction in {"increase", "decrease"}
            or stability_changed
            or horizon_count_delta not in (None, 0)
        )
        if not meaningful:
            continue

        changed_groups.append(
            {
                "category": str(item.get("category", "n/a")),
                "group": str(item.get("group", "n/a")),
                "drift_direction": drift_direction,
                "score_delta": score_delta,
                "previous_score": _coerce_optional_float(item.get("previous_score")),
                "latest_score": _coerce_optional_float(item.get("latest_score")),
                "previous_source_preference": str(
                    item.get("previous_source_preference", DEFAULT_SOURCE_PREFERENCE)
                ),
                "latest_source_preference": str(
                    item.get("latest_source_preference", DEFAULT_SOURCE_PREFERENCE)
                ),
                "previous_selected_candidate_strength": str(
                    item.get(
                        "previous_selected_candidate_strength",
                        DEFAULT_STRENGTH_LABEL,
                    )
                ),
                "latest_selected_candidate_strength": str(
                    item.get(
                        "latest_selected_candidate_strength",
                        DEFAULT_STRENGTH_LABEL,
                    )
                ),
                "previous_stability_label": str(
                    item.get("previous_stability_label", DEFAULT_STABILITY_LABEL)
                ),
                "latest_stability_label": str(
                    item.get("latest_stability_label", DEFAULT_STABILITY_LABEL)
                ),
                "stability_transition": str(
                    item.get("stability_transition", DEFAULT_STABILITY_LABEL)
                ),
                "previous_horizons": _normalize_horizons(item.get("previous_horizons")),
                "latest_horizons": _normalize_horizons(item.get("latest_horizons")),
                "horizon_count_delta": horizon_count_delta,
            }
        )

    changed_groups.sort(key=_changed_group_sort_key, reverse=True)
    return changed_groups[: max(limit, 0)]


def build_observational_message(
    score_drift_summary: dict[str, Any] | None,
    edge_scores_summary: dict[str, Any] | None,
    comparison_summary: dict[str, Any] | None = None,
) -> str:
    changed_groups = extract_changed_groups(score_drift_summary)
    drift_summary = _safe_dict((score_drift_summary or {}).get("drift_summary"))
    snapshot_lines = _build_snapshot_lines(edge_scores_summary)
    generated_at = _first_non_empty(
        (score_drift_summary or {}).get("generated_at"),
        (edge_scores_summary or {}).get("generated_at"),
        (comparison_summary or {}).get("generated_at"),
        "n/a",
    )

    lines = ["Research Observation"]
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("Drift Summary")
    lines.append(f"- increase: {_coerce_int(drift_summary.get('increase'))}")
    lines.append(f"- decrease: {_coerce_int(drift_summary.get('decrease'))}")
    lines.append(f"- flat: {_coerce_int(drift_summary.get('flat'))}")
    lines.append(
        f"- insufficient_history: {_coerce_int(drift_summary.get('insufficient_history'))}"
    )
    lines.append("")

    lines.append("Changed Groups")
    if not changed_groups:
        lines.append("- no changed groups detected")
    else:
        for item in changed_groups:
            lines.append(_format_changed_group_line(item))
    lines.append("")

    lines.append("Current Snapshot")
    if not snapshot_lines:
        lines.append("- no current snapshot available")
    else:
        lines.extend(snapshot_lines)

    return "\n".join(lines).strip()


def build_shadow_selection_message(shadow_selection: dict[str, Any] | None) -> str:
    payload = shadow_selection if isinstance(shadow_selection, dict) else {}

    selection_status = str(payload.get("selection_status", "unknown"))
    reason_codes = _normalize_reason_codes(payload.get("reason_codes"))
    explanation = _first_non_empty(payload.get("selection_explanation"), "n/a")
    generated_at = _first_non_empty(payload.get("generated_at"), "n/a")
    candidates_considered = _coerce_optional_int(payload.get("candidates_considered"))
    ranking = payload.get("ranking") if isinstance(payload.get("ranking"), list) else []
    abstain_diagnosis = (
        payload.get("abstain_diagnosis")
        if isinstance(payload.get("abstain_diagnosis"), dict)
        else {}
    )

    top_candidate = _select_shadow_top_candidate(payload, abstain_diagnosis, ranking)
    compared_candidate = abstain_diagnosis.get("compared_candidate")
    selected_candidate = _build_selected_candidate_snapshot(payload)

    lines = ["Shadow Selection", f"Generated: {generated_at}"]
    lines.append(f"Decision: {selection_status} ({_format_reason_codes(reason_codes)})")
    lines.append(f"Summary: {explanation}")

    if candidates_considered is not None:
        lines.append(f"Candidates considered: {candidates_considered}")
    lines.append(f"Ranking depth: {len([item for item in ranking if isinstance(item, dict)])}")

    diagnosis_category = _first_non_empty(abstain_diagnosis.get("category"))
    diagnosis_summary = _first_non_empty(abstain_diagnosis.get("summary"))
    if diagnosis_category:
        lines.append(f"Diagnosis category: {diagnosis_category}")
    if diagnosis_summary:
        lines.append(f"Diagnosis: {diagnosis_summary}")

    diagnosis_counts_line = _format_abstain_diagnosis_counts(abstain_diagnosis)
    if diagnosis_counts_line:
        lines.append(diagnosis_counts_line)

    seed_summary_lines = _build_seed_diagnostics_summary_lines(abstain_diagnosis)
    if seed_summary_lines:
        lines.extend(seed_summary_lines)

    if selected_candidate is not None:
        lines.append("")
        lines.append("Selected Candidate")
        lines.extend(_build_candidate_detail_lines(selected_candidate, include_rank=False))

    if top_candidate:
        lines.append("")
        lines.append("Top Candidate")
        lines.extend(_build_candidate_detail_lines(top_candidate, include_rank=True))

    if isinstance(compared_candidate, dict) and compared_candidate:
        lines.append(
            f"Tie Peer: {_format_candidate_identity(compared_candidate)} "
            f"[{compared_candidate.get('candidate_status', 'n/a')}]"
        )
        lines.append("")
        lines.append("Compared Candidate")
        lines.extend(_build_candidate_detail_lines(compared_candidate, include_rank=False))

    return "\n".join(lines).strip()


def _select_shadow_top_candidate(
    payload: dict[str, Any],
    abstain_diagnosis: dict[str, Any],
    ranking: list[Any],
) -> dict[str, Any]:
    diagnosis_top = abstain_diagnosis.get("top_candidate")
    if isinstance(diagnosis_top, dict) and diagnosis_top:
        return diagnosis_top

    ranking_top = ranking[0] if ranking and isinstance(ranking[0], dict) else None
    if isinstance(ranking_top, dict) and ranking_top:
        return ranking_top

    selected_candidate = _build_selected_candidate_snapshot(payload)
    return selected_candidate or {}


def _build_selected_candidate_snapshot(payload: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _first_non_empty(payload.get("selected_symbol"))
    strategy = _first_non_empty(payload.get("selected_strategy"))
    horizon = _first_non_empty(payload.get("selected_horizon"))

    if not symbol and not strategy and not horizon:
        return None

    return {
        "symbol": symbol or "n/a",
        "strategy": strategy or "n/a",
        "horizon": horizon or "n/a",
        "candidate_status": "selected",
        "selection_score": payload.get("selection_score"),
        "selection_confidence": payload.get("selection_confidence"),
        "reason_codes": _normalize_reason_codes(payload.get("reason_codes")),
        "advisory_reason_codes": [],
        "gate_diagnostics": {},
    }


def _build_candidate_detail_lines(
    candidate: dict[str, Any],
    *,
    include_rank: bool,
) -> list[str]:
    lines: list[str] = []

    if include_rank and candidate.get("rank") is not None:
        lines.append(f"- rank: {candidate.get('rank')}")

    lines.append(f"- identity: {_format_candidate_identity(candidate)}")
    lines.append(f"- status: {candidate.get('candidate_status', 'n/a')}")

    score_text = _format_optional_number(candidate.get("selection_score"))
    confidence_text = _format_optional_number(candidate.get("selection_confidence"))
    lines.append(f"- score/confidence: {score_text} / {confidence_text}")

    strength = _first_non_empty(candidate.get("selected_candidate_strength"))
    stability = _first_non_empty(candidate.get("selected_stability_label"))
    drift_direction = _first_non_empty(candidate.get("drift_direction"))
    source_preference = _first_non_empty(candidate.get("source_preference"))
    visible_horizons = _format_horizon_text(candidate.get("selected_visible_horizons"))

    descriptor_parts: list[str] = []
    if strength:
        descriptor_parts.append(f"strength={strength}")
    if stability:
        descriptor_parts.append(f"stability={stability}")
    if drift_direction:
        descriptor_parts.append(f"drift={drift_direction}")
    if source_preference:
        descriptor_parts.append(f"source={source_preference}")
    if visible_horizons != "none":
        descriptor_parts.append(f"horizons={visible_horizons}")

    if descriptor_parts:
        lines.append("- " + ", ".join(descriptor_parts))

    main_reasons = _normalize_reason_codes(candidate.get("reason_codes"))
    lines.append(f"- main reasons: {_format_reason_codes(main_reasons)}")

    advisory_reasons = _normalize_reason_codes(candidate.get("advisory_reason_codes"))
    if advisory_reasons:
        lines.append(f"- advisory: {_format_reason_codes(advisory_reasons)}")

    gate_summary = _format_gate_summary(candidate.get("gate_diagnostics"))
    if gate_summary:
        lines.append(f"- gates: {gate_summary}")

    return lines


def _format_abstain_diagnosis_counts(abstain_diagnosis: dict[str, Any]) -> str | None:
    eligible_count = _coerce_optional_int(abstain_diagnosis.get("eligible_candidate_count"))
    penalized_count = _coerce_optional_int(
        abstain_diagnosis.get("penalized_candidate_count")
    )
    blocked_count = _coerce_optional_int(abstain_diagnosis.get("blocked_candidate_count"))

    if eligible_count is None and penalized_count is None and blocked_count is None:
        return None

    return (
        "Diagnosis counts: "
        f"eligible={eligible_count if eligible_count is not None else 'n/a'}, "
        f"penalized={penalized_count if penalized_count is not None else 'n/a'}, "
        f"blocked={blocked_count if blocked_count is not None else 'n/a'}"
    )


def _build_seed_diagnostics_summary_lines(abstain_diagnosis: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    candidate_seed_count = _coerce_optional_int(abstain_diagnosis.get("candidate_seed_count"))
    candidate_seed_diagnostics = (
        abstain_diagnosis.get("candidate_seed_diagnostics")
        if isinstance(abstain_diagnosis.get("candidate_seed_diagnostics"), dict)
        else {}
    )

    if candidate_seed_count is not None:
        lines.append(f"Candidate seeds: {candidate_seed_count}")

    if not candidate_seed_diagnostics:
        return lines

    total_horizons_evaluated = _coerce_optional_int(
        candidate_seed_diagnostics.get("total_horizons_evaluated")
    )
    if total_horizons_evaluated is not None:
        lines.append(f"Horizons evaluated: {total_horizons_evaluated}")

    horizons_with_seed = _format_horizon_text(
        candidate_seed_diagnostics.get("horizons_with_seed")
    )
    horizons_without_seed = _format_horizon_text(
        candidate_seed_diagnostics.get("horizons_without_seed")
    )
    lines.append(f"Horizons with seeds: {horizons_with_seed}")
    lines.append(f"Horizons without seeds: {horizons_without_seed}")

    if candidate_seed_diagnostics.get("all_horizons_insufficient_data") is True:
        lines.append("Seed diagnosis: all tracked horizons remained at insufficient_data.")

    horizon_diagnostics = candidate_seed_diagnostics.get("horizon_diagnostics")
    if isinstance(horizon_diagnostics, list) and horizon_diagnostics:
        lines.append("Seed blockers")
        for item in horizon_diagnostics:
            if not isinstance(item, dict):
                continue
            lines.append(_format_seed_diagnostic_line(item))

    return lines


def _format_seed_diagnostic_line(item: dict[str, Any]) -> str:
    horizon = _first_non_empty(item.get("horizon"), "n/a")
    seed_generated = item.get("seed_generated") is True
    latest_strength = _first_non_empty(item.get("latest_candidate_strength"), "n/a")
    cumulative_strength = _first_non_empty(item.get("cumulative_candidate_strength"), "n/a")
    latest_symbol = _first_non_empty(item.get("latest_top_symbol_group"), "n/a")
    cumulative_symbol = _first_non_empty(item.get("cumulative_top_symbol_group"), "n/a")
    latest_strategy = _first_non_empty(item.get("latest_top_strategy_group"), "n/a")
    cumulative_strategy = _first_non_empty(item.get("cumulative_top_strategy_group"), "n/a")

    blocker_reasons = item.get("blocker_reasons")
    blocker_text = _format_reason_codes(
        [str(reason).strip() for reason in blocker_reasons]
        if isinstance(blocker_reasons, list)
        else []
    )

    return (
        f"- {horizon}: seed_generated={'yes' if seed_generated else 'no'}, "
        f"strength={latest_strength}/{cumulative_strength}, "
        f"symbol={latest_symbol}/{cumulative_symbol}, "
        f"strategy={latest_strategy}/{cumulative_strategy}, "
        f"blockers={blocker_text}"
    )


def _format_candidate_identity(candidate: dict[str, Any]) -> str:
    symbol = str(candidate.get("symbol", "n/a"))
    strategy = str(candidate.get("strategy", "n/a"))
    horizon = str(candidate.get("horizon", "n/a"))
    return f"{symbol} / {strategy} / {horizon}"


def _normalize_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _format_reason_codes(reason_codes: list[str], limit: int = DEFAULT_MAX_REASON_CODES) -> str:
    if not reason_codes:
        return "n/a"
    return ", ".join(reason_codes[: max(limit, 0)])


def _format_gate_summary(value: Any) -> str:
    diagnostics = value if isinstance(value, dict) else {}
    if not diagnostics:
        return ""

    parts: list[str] = []
    for gate_name in (
        "score_gate",
        "stability_gate",
        "drift_gate",
        "eligibility_gate",
        "advisory",
    ):
        gate = diagnostics.get(gate_name) if isinstance(diagnostics.get(gate_name), dict) else {}
        if not gate:
            continue

        if gate_name == "advisory":
            reasons = _format_reason_codes(
                _normalize_reason_codes(gate.get("reason_codes"))
            )
            parts.append(f"advisory=({reasons})")
            continue

        passed = gate.get("passed")
        status = "pass" if passed is True else "fail" if passed is False else "n/a"
        reasons = _format_reason_codes(_normalize_reason_codes(gate.get("reason_codes")))
        parts.append(f"{gate_name.replace('_gate', '')}={status} ({reasons})")

    return " | ".join(parts)


def _format_optional_number(value: Any) -> str:
    parsed = _coerce_optional_float(value)
    if parsed is None:
        return "n/a"
    return _format_number(parsed)


def maybe_send_notification(
    message: str,
    *,
    dry_run: bool,
    stdout: bool,
    meaningful_change: bool,
    always_send: bool,
    notifier: ResearchObservationalNotifier | None = None,
) -> dict[str, Any]:
    suppressed = not meaningful_change and not always_send

    if suppressed:
        if stdout or dry_run:
            print(SUPPRESSION_MESSAGE)
        return {
            "sent": False,
            "suppressed": True,
            "reason": SUPPRESSION_MESSAGE,
        }

    if stdout:
        print(message)

    if dry_run:
        return {
            "sent": False,
            "suppressed": False,
            "reason": "Dry run only. Message was not sent.",
        }

    active_notifier = notifier or ResearchObservationalNotifier()
    result = active_notifier.send_message(message)
    result["suppressed"] = False
    return result


def run_research_observational_notifier(
    *,
    score_drift_summary_path: Path | None = None,
    edge_scores_summary_path: Path | None = None,
    comparison_summary_path: Path | None = None,
    dry_run: bool = False,
    always_send: bool = False,
    stdout: bool = False,
) -> dict[str, Any]:
    resolved_score_drift_summary_path = (
        score_drift_summary_path or _default_score_drift_summary_path()
    )
    resolved_edge_scores_summary_path = (
        edge_scores_summary_path or _default_edge_scores_summary_path()
    )
    resolved_comparison_summary_path = (
        comparison_summary_path or _default_comparison_summary_path()
    )

    score_drift_summary = load_json(resolved_score_drift_summary_path)
    edge_scores_summary = load_json(resolved_edge_scores_summary_path)
    comparison_summary = load_json(resolved_comparison_summary_path)

    meaningful_change = detect_meaningful_change(score_drift_summary)
    changed_groups = extract_changed_groups(score_drift_summary)
    message = build_observational_message(
        score_drift_summary=score_drift_summary,
        edge_scores_summary=edge_scores_summary,
        comparison_summary=comparison_summary,
    )
    delivery = maybe_send_notification(
        message,
        dry_run=dry_run,
        stdout=stdout,
        meaningful_change=meaningful_change,
        always_send=always_send,
    )

    return {
        "score_drift_summary_path": str(resolved_score_drift_summary_path),
        "edge_scores_summary_path": str(resolved_edge_scores_summary_path),
        "comparison_summary_path": str(resolved_comparison_summary_path),
        "score_drift_summary_found": score_drift_summary is not None,
        "edge_scores_summary_found": edge_scores_summary is not None,
        "comparison_summary_found": comparison_summary is not None,
        "meaningful_change": meaningful_change,
        "changed_group_count": len(changed_groups),
        "message": message,
        **delivery,
    }


def _iter_drift_groups(score_drift_summary: dict[str, Any]) -> list[dict[str, Any]]:
    score_drift = score_drift_summary.get("score_drift")
    if not isinstance(score_drift, list):
        return []

    return [item for item in score_drift if isinstance(item, dict)]


def _build_snapshot_lines(edge_scores_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(edge_scores_summary, dict):
        return []

    score_summary = _safe_dict(edge_scores_summary.get("score_summary"))
    score_details = _safe_dict(edge_scores_summary.get("edge_stability_scores"))
    snapshot_items = [
        (
            "strategy",
            _safe_dict(score_summary.get("top_strategy")),
            score_details.get("strategy"),
        ),
        (
            "symbol",
            _safe_dict(score_summary.get("top_symbol")),
            score_details.get("symbol"),
        ),
        (
            "alignment_state",
            _safe_dict(score_summary.get("top_alignment_state")),
            score_details.get("alignment_state"),
        ),
    ]

    lines: list[str] = []
    for category, item, category_items in snapshot_items:
        group = str(item.get("group", "n/a"))
        score = _coerce_optional_float(item.get("score"))
        source_preference = str(item.get("source_preference", DEFAULT_SOURCE_PREFERENCE))
        selected_snapshot = _select_snapshot_details(
            group=group,
            source_preference=source_preference,
            category_items=category_items,
        )
        score_text = "n/a" if score is None else _format_number(score)
        lines.append(
            f"- {category}: {group} "
            f"(score={score_text}, "
            f"source={source_preference}, "
            f"strength={selected_snapshot['selected_candidate_strength']}, "
            f"stability={selected_snapshot['selected_stability_label']}, "
            f"horizons={_format_horizon_text(selected_snapshot['selected_visible_horizons'])})"
        )

    return lines


def _select_snapshot_details(
    *,
    group: str,
    source_preference: str,
    category_items: Any,
) -> dict[str, Any]:
    if not isinstance(category_items, list):
        return {
            "selected_candidate_strength": DEFAULT_STRENGTH_LABEL,
            "selected_stability_label": DEFAULT_STABILITY_LABEL,
            "selected_visible_horizons": [],
        }

    matched_item = None
    for item in category_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("group", "")).strip() == group:
            matched_item = item
            break

    if matched_item is None:
        return {
            "selected_candidate_strength": DEFAULT_STRENGTH_LABEL,
            "selected_stability_label": DEFAULT_STABILITY_LABEL,
            "selected_visible_horizons": [],
        }

    if source_preference == "latest":
        return {
            "selected_candidate_strength": str(
                matched_item.get("latest_candidate_strength", DEFAULT_STRENGTH_LABEL)
            ),
            "selected_stability_label": str(
                matched_item.get("latest_stability_label", DEFAULT_STABILITY_LABEL)
            ),
            "selected_visible_horizons": _normalize_horizons(
                matched_item.get("latest_visible_horizons")
            ),
        }

    return {
        "selected_candidate_strength": str(
            matched_item.get("cumulative_candidate_strength", DEFAULT_STRENGTH_LABEL)
        ),
        "selected_stability_label": str(
            matched_item.get("cumulative_stability_label", DEFAULT_STABILITY_LABEL)
        ),
        "selected_visible_horizons": _normalize_horizons(
            matched_item.get("cumulative_visible_horizons")
        ),
    }


def _format_changed_group_line(item: dict[str, Any]) -> str:
    score_delta = item.get("score_delta")
    delta_text = "n/a" if score_delta is None else _format_signed_number(score_delta)
    previous_score = item.get("previous_score")
    latest_score = item.get("latest_score")
    previous_score_text = "n/a" if previous_score is None else _format_number(previous_score)
    latest_score_text = "n/a" if latest_score is None else _format_number(latest_score)
    previous_horizons_text = _format_horizon_text(item.get("previous_horizons"))
    latest_horizons_text = _format_horizon_text(item.get("latest_horizons"))
    horizon_count_delta = item.get("horizon_count_delta")
    horizon_count_delta_text = (
        "n/a" if horizon_count_delta is None else str(horizon_count_delta)
    )

    return (
        f"- {item.get('category', 'n/a')}/{item.get('group', 'n/a')}: "
        f"drift={item.get('drift_direction', 'n/a')}, "
        f"delta={delta_text}, "
        f"score={previous_score_text} -> {latest_score_text}, "
        f"strength={item.get('previous_selected_candidate_strength', DEFAULT_STRENGTH_LABEL)} "
        f"-> {item.get('latest_selected_candidate_strength', DEFAULT_STRENGTH_LABEL)}, "
        f"stability={item.get('previous_stability_label', DEFAULT_STABILITY_LABEL)} "
        f"-> {item.get('latest_stability_label', DEFAULT_STABILITY_LABEL)}, "
        f"horizons={previous_horizons_text} -> {latest_horizons_text}, "
        f"horizon_delta={horizon_count_delta_text}, "
        f"source={item.get('previous_source_preference', DEFAULT_SOURCE_PREFERENCE)} "
        f"-> {item.get('latest_source_preference', DEFAULT_SOURCE_PREFERENCE)}"
    )


def _changed_group_sort_key(item: dict[str, Any]) -> tuple[int, float, int, str, str]:
    drift_direction = str(item.get("drift_direction", "insufficient_history"))
    direction_priority = 1 if drift_direction in {"increase", "decrease"} else 0
    score_delta = abs(_coerce_optional_float(item.get("score_delta")) or 0.0)
    horizon_delta = abs(_coerce_optional_int(item.get("horizon_count_delta")) or 0)
    return (
        direction_priority,
        score_delta,
        horizon_delta,
        str(item.get("category", "")),
        str(item.get("group", "")),
    )


def _stability_changed(item: dict[str, Any]) -> bool:
    previous_label = str(item.get("previous_stability_label", DEFAULT_STABILITY_LABEL))
    latest_label = str(item.get("latest_stability_label", DEFAULT_STABILITY_LABEL))
    return previous_label != latest_label


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_horizons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        item_text = str(item).strip()
        if item_text and item_text not in normalized:
            normalized.append(item_text)
    return normalized


def _format_horizon_text(value: Any) -> str:
    horizons = _normalize_horizons(value)
    if not horizons:
        return "none"
    return ", ".join(horizons)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return ""


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _format_signed_number(value: float) -> str:
    return f"{float(value):+.2f}".rstrip("0").rstrip(".")


def _default_score_drift_summary_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "score_drift"
        / "summary.json"
    )


def _default_edge_scores_summary_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores"
        / "summary.json"
    )


def _default_comparison_summary_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "comparison"
        / "summary.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally send a research-only observational summary"
    )
    parser.add_argument(
        "--score-drift-summary",
        type=Path,
        default=_default_score_drift_summary_path(),
        help="Path to score drift summary.json",
    )
    parser.add_argument(
        "--edge-scores-summary",
        type=Path,
        default=_default_edge_scores_summary_path(),
        help="Path to edge scores summary.json",
    )
    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=_default_comparison_summary_path(),
        help="Optional path to comparison summary.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the observational message without sending it",
    )
    parser.add_argument(
        "--always-send",
        action="store_true",
        help="Send or emit the message even when no meaningful change is detected",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the message or suppression note to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_research_observational_notifier(
        score_drift_summary_path=args.score_drift_summary,
        edge_scores_summary_path=args.edge_scores_summary,
        comparison_summary_path=args.comparison_summary,
        dry_run=args.dry_run,
        always_send=args.always_send,
        stdout=args.stdout,
    )

    if not args.stdout:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()