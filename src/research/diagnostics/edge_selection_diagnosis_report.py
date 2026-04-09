from __future__ import annotations

from typing import Any


def build_edge_selection_diagnosis_report(
    research_summary_data: dict[str, Any] | None,
    edge_candidates_preview: dict[str, Any] | None,
    edge_stability_preview: dict[str, Any] | None,
    shadow_selection: dict[str, Any] | None,
) -> dict[str, Any]:
    research_summary = research_summary_data if isinstance(research_summary_data, dict) else {}
    candidates_preview = (
        edge_candidates_preview if isinstance(edge_candidates_preview, dict) else {}
    )
    stability_preview = (
        edge_stability_preview if isinstance(edge_stability_preview, dict) else {}
    )
    selection = shadow_selection if isinstance(shadow_selection, dict) else {}

    ranking = selection.get("ranking") if isinstance(selection.get("ranking"), list) else []
    ranking_items = [item for item in ranking if isinstance(item, dict)]
    abstain_diagnosis = (
        selection.get("abstain_diagnosis")
        if isinstance(selection.get("abstain_diagnosis"), dict)
        else {}
    )

    selection_status = _normalize_text(selection.get("selection_status"), "unknown")
    abstain_category = _normalize_text(abstain_diagnosis.get("category"), "n/a")
    candidates_considered = _coerce_optional_int(selection.get("candidates_considered"))
    ranking_depth = len(ranking_items)

    visible_groups = _build_visible_groups(stability_preview)
    failed_layers = _build_failed_layers(candidates_preview, stability_preview)
    stability_issues = _build_stability_issues(stability_preview, abstain_diagnosis)
    drift_issues = _build_drift_issues(selection, abstain_diagnosis, ranking_items)
    candidate_generation_state = _build_candidate_generation_state(
        selection=selection,
        ranking_items=ranking_items,
        failed_layers=failed_layers,
    )

    top_candidate = _select_top_candidate(selection, abstain_diagnosis, ranking_items)
    top_candidate_identity = _build_candidate_identity(top_candidate)

    return {
        "selection_status": selection_status,
        "abstain_category": abstain_category,
        "candidates_considered": candidates_considered,
        "ranking_depth": ranking_depth,
        "top_candidate_identity": top_candidate_identity,
        "candidate_generation_state": candidate_generation_state,
        "visible_groups": visible_groups,
        "failed_layers": failed_layers,
        "stability_issues": stability_issues,
        "drift_issues": drift_issues,
        "diagnosis_summary": _build_summary(
            candidate_generation_state=candidate_generation_state,
            failed_layers=failed_layers,
            stability_issues=stability_issues,
            drift_issues=drift_issues,
            selection=selection,
            research_summary=research_summary,
            edge_candidates_preview=candidates_preview,
            abstain_category=abstain_category,
            ranking_depth=ranking_depth,
            candidates_considered=candidates_considered,
            top_candidate_identity=top_candidate_identity,
        ),
    }


def _build_visible_groups(edge_stability_preview: dict[str, Any]) -> dict[str, dict[str, Any]]:
    visible_groups: dict[str, dict[str, Any]] = {}

    for label, entry in (
        ("strategy", edge_stability_preview.get("strategy", {})),
        ("symbol", edge_stability_preview.get("symbol", {})),
        ("alignment", edge_stability_preview.get("alignment_state", {})),
    ):
        if not isinstance(entry, dict):
            continue

        visible_groups[label] = {
            "group": entry.get("group"),
            "visible_horizons": _normalize_horizons(entry.get("visible_horizons")),
            "stability_label": _normalize_text(
                entry.get("stability_label"),
                "insufficient_data",
            ),
            "visibility_reason": _normalize_text(
                entry.get("visibility_reason"),
                "no_visible_candidates",
            ),
        }

    return visible_groups


def _build_failed_layers(
    edge_candidates_preview: dict[str, Any],
    edge_stability_preview: dict[str, Any],
) -> list[str]:
    by_horizon = edge_candidates_preview.get("by_horizon", {}) or {}
    failed: list[str] = []

    if not by_horizon:
        return ["candidate_preview_missing"]

    layer_map = {
        "strategy": "top_strategy",
        "symbol": "top_symbol",
        "alignment": "top_alignment_state",
    }

    for layer_name, candidate_key in layer_map.items():
        visible_horizons: list[str] = []
        weak_horizons: list[str] = []
        blocked_horizons: list[str] = []

        for horizon, horizon_payload in by_horizon.items():
            if not isinstance(horizon_payload, dict):
                continue

            candidate = horizon_payload.get(candidate_key)
            if not isinstance(candidate, dict):
                continue

            strength = _normalize_text(candidate.get("candidate_strength"), "insufficient_data")
            if strength == "insufficient_data":
                blocked_horizons.append(str(horizon))
                continue

            visible_horizons.append(str(horizon))
            if _normalize_text(candidate.get("quality_gate"), "failed") != "passed":
                weak_horizons.append(str(horizon))

        if not visible_horizons:
            failed.append(f"{layer_name}:no_visible_groups")
            continue

        preview_key = "alignment_state" if layer_name == "alignment" else layer_name
        stability_entry = edge_stability_preview.get(preview_key, {})
        stability_label = (
            stability_entry.get("stability_label", "insufficient_data")
            if isinstance(stability_entry, dict)
            else "insufficient_data"
        )

        if stability_label in {"single_horizon_only", "unstable", "insufficient_data"}:
            failed.append(f"{layer_name}:stability={stability_label}")

        if weak_horizons:
            failed.append(f"{layer_name}:quality_borderline@{','.join(weak_horizons)}")

        if blocked_horizons and len(blocked_horizons) == len(by_horizon):
            failed.append(f"{layer_name}:sample_insufficient")

    return failed


def _build_stability_issues(
    edge_stability_preview: dict[str, Any],
    abstain_diagnosis: dict[str, Any],
) -> list[str]:
    issues: list[str] = []

    for label, entry in (
        ("strategy", edge_stability_preview.get("strategy", {})),
        ("symbol", edge_stability_preview.get("symbol", {})),
        ("alignment", edge_stability_preview.get("alignment_state", {})),
    ):
        if not isinstance(entry, dict):
            continue

        stability_label = _normalize_text(entry.get("stability_label"), "insufficient_data")
        if stability_label == "multi_horizon_confirmed":
            continue

        visible_horizons = _normalize_horizons(entry.get("visible_horizons"))
        horizon_text = ",".join(visible_horizons) if visible_horizons else "none"
        issues.append(f"{label}:{stability_label}@{horizon_text}")

    top_candidate = abstain_diagnosis.get("top_candidate")
    if isinstance(top_candidate, dict):
        gate_diagnostics = (
            top_candidate.get("gate_diagnostics")
            if isinstance(top_candidate.get("gate_diagnostics"), dict)
            else {}
        )
        stability_gate = (
            gate_diagnostics.get("stability_gate")
            if isinstance(gate_diagnostics.get("stability_gate"), dict)
            else {}
        )
        if stability_gate.get("passed") is False:
            issues.extend(
                f"candidate:{reason}"
                for reason in _normalize_reason_codes(stability_gate.get("reason_codes"))
            )

    return _dedupe(issues)


def _build_drift_issues(
    shadow_selection: dict[str, Any],
    abstain_diagnosis: dict[str, Any],
    ranking_items: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []

    for candidate in _iter_diagnostic_candidates(
        shadow_selection,
        abstain_diagnosis,
        ranking_items,
    ):
        gate_diagnostics = (
            candidate.get("gate_diagnostics")
            if isinstance(candidate.get("gate_diagnostics"), dict)
            else {}
        )
        drift_gate = (
            gate_diagnostics.get("drift_gate")
            if isinstance(gate_diagnostics.get("drift_gate"), dict)
            else {}
        )

        if drift_gate.get("passed") is False:
            issues.extend(_normalize_reason_codes(drift_gate.get("reason_codes")))

        drift_direction = _normalize_text(candidate.get("drift_direction"))
        if drift_direction == "decrease":
            issues.append("drift_direction=decrease")

    return _dedupe(issues)


def _build_candidate_generation_state(
    *,
    selection: dict[str, Any],
    ranking_items: list[dict[str, Any]],
    failed_layers: list[str],
) -> str:
    selection_status = _normalize_text(selection.get("selection_status"), "unknown")
    candidates_considered = _coerce_optional_int(selection.get("candidates_considered"))

    if selection_status == "blocked":
        return "selection_blocked_upstream"

    if ranking_items and selection_status == "abstain":
        return "candidates_generated_but_filtered_before_final_selection"

    if candidates_considered == 0 or not ranking_items:
        if any(
            "sample_insufficient" in item or "no_visible_groups" in item
            for item in failed_layers
        ):
            return "no_visible_candidates_after_preview_filters"
        return "no_ranked_candidates_visible"

    if selection_status == "selected":
        return "final_selection_present"

    return "selection_state_unclear"


def _build_summary(
    *,
    candidate_generation_state: str,
    failed_layers: list[str],
    stability_issues: list[str],
    drift_issues: list[str],
    selection: dict[str, Any],
    research_summary: dict[str, Any],
    edge_candidates_preview: dict[str, Any],
    abstain_category: str,
    ranking_depth: int,
    candidates_considered: int | None,
    top_candidate_identity: str,
) -> str:
    selection_status = _normalize_text(selection.get("selection_status"), "unknown")
    summary_parts = [
        f"status={selection_status}",
        f"abstain_category={abstain_category}",
        f"candidates_considered={candidates_considered if candidates_considered is not None else 'n/a'}",
        f"ranking_depth={ranking_depth}",
        candidate_generation_state,
    ]

    if top_candidate_identity != "n/a":
        summary_parts.append(f"top={top_candidate_identity}")

    if failed_layers:
        summary_parts.append(f"failed_layers={', '.join(failed_layers[:3])}")

    if stability_issues:
        summary_parts.append(f"stability={', '.join(stability_issues[:3])}")

    if drift_issues:
        summary_parts.append(f"drift={', '.join(drift_issues[:3])}")

    blocked_horizons = _extract_blocked_horizons(research_summary, edge_candidates_preview)
    if blocked_horizons:
        summary_parts.append(f"blocked_horizons={', '.join(blocked_horizons)}")

    return " | ".join(summary_parts)


def _iter_diagnostic_candidates(
    shadow_selection: dict[str, Any],
    abstain_diagnosis: dict[str, Any],
    ranking_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    top_candidate = abstain_diagnosis.get("top_candidate")
    if isinstance(top_candidate, dict) and top_candidate:
        candidates.append(top_candidate)

    compared_candidate = abstain_diagnosis.get("compared_candidate")
    if isinstance(compared_candidate, dict) and compared_candidate:
        candidates.append(compared_candidate)

    candidates.extend(ranking_items[:3])

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()

    for candidate in candidates:
        identity = (
            candidate.get("symbol"),
            candidate.get("strategy"),
            candidate.get("horizon"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(candidate)

    return deduped


def _extract_blocked_horizons(
    research_summary: dict[str, Any],
    edge_candidates_preview: dict[str, Any] | None,
) -> list[str]:
    preview_root = edge_candidates_preview
    if not isinstance(preview_root, dict) or not preview_root:
        preview_root = research_summary.get("edge_candidates_preview", {})
        if not isinstance(preview_root, dict):
            preview_root = {}

    by_horizon = preview_root.get("by_horizon", {})
    if not isinstance(by_horizon, dict):
        return []

    blocked: list[str] = []
    for horizon, payload in by_horizon.items():
        if not isinstance(payload, dict):
            continue

        strengths: list[str] = []
        for key in ("top_strategy", "top_symbol", "top_alignment_state"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                strengths.append(
                    _normalize_text(candidate.get("candidate_strength"), "insufficient_data")
                )

        if strengths and all(value == "insufficient_data" for value in strengths):
            blocked.append(str(horizon))

    return blocked


def _select_top_candidate(
    selection: dict[str, Any],
    abstain_diagnosis: dict[str, Any],
    ranking_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    diagnosis_top = abstain_diagnosis.get("top_candidate")
    if isinstance(diagnosis_top, dict) and diagnosis_top:
        return diagnosis_top

    if ranking_items:
        return ranking_items[0]

    selected_symbol = _normalize_text(selection.get("selected_symbol"))
    selected_strategy = _normalize_text(selection.get("selected_strategy"))
    selected_horizon = _normalize_text(selection.get("selected_horizon"))
    if selected_symbol or selected_strategy or selected_horizon:
        return {
            "symbol": selected_symbol or "n/a",
            "strategy": selected_strategy or "n/a",
            "horizon": selected_horizon or "n/a",
        }

    return None


def _build_candidate_identity(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict) or not candidate:
        return "n/a"

    symbol = _normalize_text(candidate.get("symbol"), "n/a")
    strategy = _normalize_text(candidate.get("strategy"), "n/a")
    horizon = _normalize_text(candidate.get("horizon"), "n/a")
    return f"{symbol} / {strategy} / {horizon}"


def _normalize_horizons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    horizons: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in horizons:
            horizons.append(text)
    return horizons


def _normalize_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result