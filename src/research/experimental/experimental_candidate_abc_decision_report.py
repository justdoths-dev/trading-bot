from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_HORIZONS = ("15m", "1h", "4h")
DEFAULT_AC_REPORT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_c_intersection_comparison.json"
)
DEFAULT_BC_REPORT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_b_vs_c_intersection_comparison.json"
)
DEFAULT_AB_REPORT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_b_comparison.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_abc_decision_report.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_abc_decision_report.md"
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


def _format_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.6f}"


def _load_report(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists() or not path.is_file():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    return payload, True


def _first_available_report_type(report: dict[str, Any]) -> str | None:
    metadata = _safe_dict(report.get("metadata"))
    return _safe_text(metadata.get("report_type") or report.get("report_type"))


def _extract_nested_delta(
    section: dict[str, Any],
    horizon: str,
    primary_key: str,
    fallback_keys: tuple[str, ...] = (),
) -> float | None:
    horizon_payload = _safe_dict(section.get(horizon))
    primary_payload = _safe_dict(horizon_payload.get(primary_key))
    primary_delta = _safe_float(primary_payload.get("delta"))
    if primary_delta is not None:
        return primary_delta

    for key in fallback_keys:
        fallback_value = _safe_float(horizon_payload.get(key))
        if fallback_value is not None:
            return fallback_value
        fallback_payload = _safe_dict(horizon_payload.get(key))
        fallback_delta = _safe_float(fallback_payload.get("delta"))
        if fallback_delta is not None:
            return fallback_delta
    return None


def _extract_horizon_delta(
    section: dict[str, Any],
    key: str,
    fallback_keys: tuple[str, ...] = (),
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for horizon in TARGET_HORIZONS:
        result[horizon] = _extract_nested_delta(section, horizon, key, fallback_keys)
    return result


def _extract_ac_evidence(report: dict[str, Any], loaded: bool) -> dict[str, Any]:
    alignment = _safe_dict(report.get("alignment"))
    shared = _safe_dict(report.get("shared_row_horizon_comparison"))
    coverage = _safe_dict(report.get("coverage_structure"))
    final_diagnosis = _safe_dict(report.get("final_diagnosis"))

    flat_deltas = _extract_horizon_delta(shared, "flat_share")
    up_deltas = _extract_horizon_delta(shared, "up_share")
    down_deltas = _extract_horizon_delta(shared, "down_share")
    c_only_rows = _safe_int(alignment.get("candidate_c_only_rows")) or 0
    a_only_rows = _safe_int(alignment.get("candidate_a_only_rows")) or 0

    same_row_replacement_supported = False
    if loaded:
        negative_flat = sum(1 for value in flat_deltas.values() if value is not None and value < 0.0)
        non_negative_flat = sum(1 for value in flat_deltas.values() if value is not None and value >= 0.0)
        same_row_replacement_supported = negative_flat > non_negative_flat

    return {
        "loaded": loaded,
        "shared_rows": _safe_int(alignment.get("shared_rows")) or 0,
        "candidate_a_only_rows": a_only_rows,
        "candidate_c_only_rows": c_only_rows,
        "shared_row_ratio_from_a": _safe_float(alignment.get("shared_row_ratio_from_a")),
        "shared_row_ratio_from_c": _safe_float(alignment.get("shared_row_ratio_from_c")),
        "flat_share_delta_by_horizon": flat_deltas,
        "up_share_delta_by_horizon": up_deltas,
        "down_share_delta_by_horizon": down_deltas,
        "coverage_expansion_supported": c_only_rows > a_only_rows,
        "same_row_replacement_supported": same_row_replacement_supported,
        "final_primary_finding": _safe_text(final_diagnosis.get("primary_finding")),
        "final_secondary_finding": _safe_text(final_diagnosis.get("secondary_finding")),
        "final_summary": _safe_text(final_diagnosis.get("summary")),
        "coverage_notes": _safe_list(coverage.get("interpretation")),
    }


def _extract_bc_evidence(report: dict[str, Any], loaded: bool) -> dict[str, Any]:
    overview = _safe_dict(report.get("intersection_overview"))
    if not overview:
        overview = _safe_dict(report.get("alignment"))

    delta_root = _safe_dict(report.get("delta_on_shared_rows"))
    delta = _safe_dict(delta_root.get("by_horizon"))
    if not delta:
        delta = _safe_dict(report.get("shared_row_horizon_comparison"))

    final_diagnosis = _safe_dict(report.get("final_diagnosis"))

    coverage_recovery_horizons: list[str] = []
    flat_ratio_change_by_horizon: dict[str, float | None] = {}
    up_ratio_change_by_horizon: dict[str, float | None] = {}
    for horizon in TARGET_HORIZONS:
        flat_delta = _extract_nested_delta(
            delta,
            horizon,
            "flat_share",
            fallback_keys=("flat_ratio_change",),
        )
        up_delta = _extract_nested_delta(
            delta,
            horizon,
            "up_share",
            fallback_keys=("up_ratio_change",),
        )
        flat_ratio_change_by_horizon[horizon] = flat_delta
        up_ratio_change_by_horizon[horizon] = up_delta
        if flat_delta is not None and flat_delta < 0.0:
            coverage_recovery_horizons.append(horizon)

    primary_finding = _safe_text(final_diagnosis.get("primary_finding")) or ""
    summary_text = _safe_text(final_diagnosis.get("summary")) or ""
    recommendation_text = _safe_text(final_diagnosis.get("recommendation")) or ""
    practical_alternative_supported = loaded and (
        "seed_friendly" in primary_finding
        or "recovers_coverage" in primary_finding
        or "middle_path" in primary_finding
        or "practical" in summary_text.lower()
        or "practical" in recommendation_text.lower()
        or bool(coverage_recovery_horizons)
    )

    return {
        "loaded": loaded,
        "shared_row_ratio_vs_baseline": _safe_float(
            overview.get("shared_ratio_vs_baseline") or overview.get("shared_row_ratio_from_a")
        ),
        "shared_row_ratio_vs_experiment": _safe_float(
            overview.get("shared_ratio_vs_experiment") or overview.get("shared_row_ratio_from_c")
        ),
        "coverage_recovery_horizons": coverage_recovery_horizons,
        "flat_ratio_change_by_horizon": flat_ratio_change_by_horizon,
        "up_ratio_change_by_horizon": up_ratio_change_by_horizon,
        "practical_alternative_supported": practical_alternative_supported,
        "final_primary_finding": _safe_text(final_diagnosis.get("primary_finding")),
        "final_secondary_finding": _safe_text(final_diagnosis.get("secondary_finding")),
        "final_summary": _safe_text(final_diagnosis.get("summary")),
        "final_notes": _safe_list(final_diagnosis.get("notes")),
    }


def _extract_ab_evidence(report: dict[str, Any], loaded: bool) -> dict[str, Any]:
    final_summary = _safe_dict(report.get("final_summary"))
    final_diagnosis = _safe_dict(report.get("final_diagnosis"))
    return {
        "loaded": loaded,
        "final_primary_finding": _safe_text(
            final_summary.get("primary_finding")
            or final_diagnosis.get("primary_finding")
            or report.get("primary_finding")
        ),
        "final_secondary_finding": _safe_text(
            final_summary.get("secondary_finding")
            or final_diagnosis.get("secondary_finding")
            or report.get("secondary_finding")
        ),
        "final_summary": _safe_text(
            final_summary.get("summary")
            or final_diagnosis.get("summary")
            or report.get("summary")
        ),
    }


def _build_input_reports_section(
    ac_report: dict[str, Any],
    ac_loaded: bool,
    ac_path: Path,
    bc_report: dict[str, Any],
    bc_loaded: bool,
    bc_path: Path,
    ab_report: dict[str, Any],
    ab_loaded: bool,
    ab_path: Path,
) -> dict[str, Any]:
    return {
        "candidate_a_vs_c_intersection": {
            "path": str(ac_path),
            "loaded": ac_loaded,
            "report_type": _first_available_report_type(ac_report),
        },
        "candidate_b_vs_c_intersection": {
            "path": str(bc_path),
            "loaded": bc_loaded,
            "report_type": _first_available_report_type(bc_report),
        },
        "candidate_a_vs_b_optional": {
            "path": str(ab_path),
            "loaded": ab_loaded,
            "report_type": _first_available_report_type(ab_report),
        },
    }


def _build_candidate_positioning(
    ac_evidence: dict[str, Any],
    bc_evidence: dict[str, Any],
    ab_evidence: dict[str, Any],
) -> dict[str, Any]:
    c2_same_row_supported = bool(ac_evidence.get("same_row_replacement_supported"))
    c2_expansion_supported = bool(ac_evidence.get("coverage_expansion_supported"))
    c2_practical_vs_b = bool(bc_evidence.get("practical_alternative_supported"))

    candidate_a = {
        "role": "cleaner shared-row baseline",
        "strengths": [
            "Acts as the default benchmark for same-row label quality.",
            "Remains the cleaner replacement test for any candidate that wants to displace the baseline.",
        ],
        "weaknesses": [
            "Does not directly solve seed starvation by itself.",
            "May leave coverage on the table relative to broader-row candidates.",
        ],
        "current_interpretation": (
            "Candidate A should remain the baseline default unless another candidate clearly improves shared-row behavior without unacceptable purity give-back."
        ),
    }

    if ab_evidence.get("loaded"):
        ab_primary = _safe_text(ab_evidence.get("final_primary_finding"))
        if ab_primary:
            candidate_a["strengths"].append(
                f"A vs B reference context: {ab_primary}."
            )

    candidate_b = {
        "role": "sparse / higher-purity / lower-coverage reference",
        "strengths": [
            "Useful as the sparse reference point for higher-purity labeling behavior.",
            "Helps define the lower-coverage edge of the relabel tradeoff space.",
        ],
        "weaknesses": [
            "Can be too sparse for a seed-starved system.",
            "Practical usability falls when coverage drops faster than purity improves.",
        ],
        "current_interpretation": (
            "Candidate B should stay alive as the sparse reference candidate rather than the operational default."
        ),
    }

    if bc_evidence.get("loaded"):
        bc_primary = _safe_text(bc_evidence.get("final_primary_finding"))
        if bc_primary:
            candidate_b["weaknesses"].append(f"B vs C comparison context: {bc_primary}.")

    c2_strengths = [
        "Sits in the middle of the purity-versus-coverage tradeoff rather than at the sparse extreme.",
        "Useful for testing whether extra row coverage can relieve seed starvation.",
    ]
    if c2_expansion_supported:
        c2_strengths.append("Shows broader row coverage than Candidate A through exclusive-row expansion.")
    if c2_practical_vs_b:
        c2_strengths.append("Looks more practical than Candidate B in sparse conditions.")

    c2_weaknesses = [
        "Is not automatically better just because it changes flat share or expands coverage.",
    ]
    if not c2_same_row_supported:
        c2_weaknesses.append("Does not currently read as a clean same-row replacement for Candidate A.")
    if not c2_practical_vs_b:
        c2_weaknesses.append("Needs more proof that extra coverage survives downstream edge-selection filters.")

    c2_interpretation = (
        "Candidate C2 should be treated as a coverage-expansion candidate, not as a proven same-row upgrade over Candidate A."
    )
    if c2_practical_vs_b:
        c2_interpretation += " It remains especially relevant as a practical alternative to the overly sparse Candidate B regime."

    return {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "candidate_c2": {
            "role": "coverage-expansion middle candidate",
            "strengths": c2_strengths,
            "weaknesses": c2_weaknesses,
            "current_interpretation": c2_interpretation,
        },
    }


def _build_evidence_summary(
    ac_evidence: dict[str, Any],
    bc_evidence: dict[str, Any],
    ab_evidence: dict[str, Any],
) -> dict[str, Any]:
    ac_loaded = bool(ac_evidence.get("loaded"))
    bc_loaded = bool(bc_evidence.get("loaded"))
    ab_loaded = bool(ab_evidence.get("loaded"))

    if ac_loaded:
        ac_shared_row_result = (
            "Candidate C2 is not supported as a same-row improvement over Candidate A on the shared intersection."
            if not ac_evidence.get("same_row_replacement_supported")
            else "Candidate C2 shows some same-row support relative to Candidate A, but replacement confidence is still limited."
        )
        ac_exclusive_row_result = (
            f"Exclusive rows favor C2 expansion ({ac_evidence.get('candidate_c_only_rows', 0)} C-only vs {ac_evidence.get('candidate_a_only_rows', 0)} A-only)."
            if ac_evidence.get("coverage_expansion_supported")
            else f"Exclusive rows do not show a clear C2 expansion edge ({ac_evidence.get('candidate_c_only_rows', 0)} C-only vs {ac_evidence.get('candidate_a_only_rows', 0)} A-only)."
        )
    else:
        ac_shared_row_result = "A vs C shared-row evidence was not loaded."
        ac_exclusive_row_result = "A vs C exclusive-row evidence was not loaded."

    if bc_loaded:
        bc_tradeoff_result = (
            "B vs C supports C2 as the more practical candidate under sparse conditions."
            if bc_evidence.get("practical_alternative_supported")
            else "B vs C does not yet clearly support C2 as the more practical sparse-condition alternative."
        )
    else:
        bc_tradeoff_result = "B vs C practical tradeoff evidence was not loaded."

    if ab_loaded:
        ab_reference_result = (
            _safe_text(ab_evidence.get("final_summary"))
            or _safe_text(ab_evidence.get("final_primary_finding"))
            or "A vs B optional reference loaded."
        )
    else:
        ab_reference_result = "A vs B optional reference was not loaded."

    replacement_supported = bool(ac_evidence.get("same_row_replacement_supported"))
    practical_vs_b_supported = bool(bc_evidence.get("practical_alternative_supported"))

    return {
        "candidate_a_vs_c_shared_row_result": ac_shared_row_result,
        "candidate_a_vs_c_exclusive_row_result": ac_exclusive_row_result,
        "candidate_b_vs_c_practical_tradeoff_result": bc_tradeoff_result,
        "candidate_a_vs_b_reference_result": ab_reference_result,
        "c2_supported_as_replacement_vs_a": {
            "decision": "yes" if replacement_supported else "no",
            "rationale": (
                "Shared-row evidence supports Candidate C2 as a same-row upgrade over A."
                if replacement_supported
                else "Current shared-row evidence does not support replacing Candidate A with Candidate C2."
            ),
        },
        "c2_supported_as_practical_alternative_vs_b": {
            "decision": "yes" if practical_vs_b_supported else "no",
            "rationale": (
                "Candidate C2 appears more usable than the sparse B regime in seed-starved conditions."
                if practical_vs_b_supported
                else "The current loaded reports do not yet prove that Candidate C2 is the practical alternative to Candidate B."
            ),
        },
        "ac_numeric_snapshot": {
            "shared_rows": ac_evidence.get("shared_rows"),
            "shared_row_ratio_from_a": ac_evidence.get("shared_row_ratio_from_a"),
            "shared_row_ratio_from_c": ac_evidence.get("shared_row_ratio_from_c"),
            "flat_share_delta_by_horizon": ac_evidence.get("flat_share_delta_by_horizon"),
        },
        "bc_numeric_snapshot": {
            "shared_row_ratio_vs_baseline": bc_evidence.get("shared_row_ratio_vs_baseline"),
            "shared_row_ratio_vs_experiment": bc_evidence.get("shared_row_ratio_vs_experiment"),
            "coverage_recovery_horizons": bc_evidence.get("coverage_recovery_horizons"),
            "flat_ratio_change_by_horizon": bc_evidence.get("flat_ratio_change_by_horizon"),
        },
    }


def _build_current_decision(ac_evidence: dict[str, Any], bc_evidence: dict[str, Any]) -> dict[str, Any]:
    replace_a = bool(ac_evidence.get("same_row_replacement_supported"))
    keep_c2 = True
    keep_c2_rationale = (
        "C2 should stay alive because the coverage-expansion hypothesis is still unresolved, and the next diagnostic question is whether C2-exclusive rows improve downstream seed availability."
    )
    if replace_a:
        keep_c2_rationale = "C2 should obviously stay alive because it is already strong enough to challenge the baseline directly."
    elif bool(ac_evidence.get("coverage_expansion_supported")) or bool(bc_evidence.get("practical_alternative_supported")):
        keep_c2_rationale = "C2 still matters because it expands coverage and/or looks more practical than B under sparse conditions."

    return {
        "default_baseline_candidate": "candidate_a",
        "sparse_reference_candidate": "candidate_b",
        "coverage_expansion_candidate": "candidate_c2",
        "should_replace_a_with_c2": {
            "decision": "yes" if replace_a else "no",
            "rationale": (
                "Shared-row evidence is strong enough to treat C2 as an A replacement."
                if replace_a
                else "C2 is not clearly better than A on the shared row set, so A should stay the cleaner baseline default."
            ),
        },
        "should_keep_c2_alive_for_followup": {
            "decision": "yes" if keep_c2 else "no",
            "rationale": keep_c2_rationale,
        },
    }


def _build_next_experiment_recommendation(current_decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "keep_a_as_baseline_default": True,
        "keep_b_as_sparse_reference": True,
        "keep_c2_alive_as_coverage_expansion_experiment": (
            _safe_text(_safe_dict(current_decision.get("should_keep_c2_alive_for_followup")).get("decision")) == "yes"
        ),
        "next_priority": "diagnose_whether_c2_exclusive_rows_improve_downstream_seed_availability",
        "recommended_action": (
            "Keep A as the cleaner baseline default, keep B as the sparse reference candidate, and keep C2 alive as a coverage-expansion experiment. Next, diagnose whether C2-exclusive rows actually survive downstream edge-selection and improve seed availability."
        ),
    }


def _build_final_diagnosis(ac_evidence: dict[str, Any], bc_evidence: dict[str, Any]) -> dict[str, Any]:
    replace_a = bool(ac_evidence.get("same_row_replacement_supported"))
    practical_vs_b = bool(bc_evidence.get("practical_alternative_supported"))
    coverage_expansion = bool(ac_evidence.get("coverage_expansion_supported"))

    if replace_a:
        primary = "candidate_c2_has_enough_support_to_challenge_candidate_a"
        secondary = "shared_row_evidence_justifies_a_more_aggressive_follow_up"
        recommendation = "Promote C2 into a direct baseline challenge track and verify downstream seed-quality impact."
    else:
        primary = "candidate_c2_should_remain_live_as_coverage_expansion_but_not_replace_candidate_a"
        secondary = "candidate_a_remains_default_baseline_and_candidate_b_remains_sparse_reference"
        recommendation = "Keep A as baseline, keep B as sparse reference, and run the next diagnosis on whether C2-exclusive rows improve downstream seed availability."

    summary = (
        "Candidate A still reads as the cleaner shared-row baseline, Candidate B still defines the sparse high-purity edge, and Candidate C2 currently fits best as a coverage-expansion middle candidate rather than an A replacement."
    )
    if practical_vs_b:
        summary += " The strongest reason to keep C2 alive is that it appears more practical than B in sparse conditions."
    if coverage_expansion:
        summary += " The next question is whether C2-exclusive rows translate into downstream seed availability rather than just broader labeling coverage."
    if not practical_vs_b and not coverage_expansion and not replace_a:
        summary += " Even when the comparison set is incomplete, the next useful action is still to test whether C2-exclusive rows create usable seeds downstream."

    return {
        "primary_finding": primary,
        "secondary_finding": secondary,
        "recommendation": recommendation,
        "summary": summary,
    }


def build_experimental_candidate_abc_decision_report(
    ac_report: dict[str, Any],
    *,
    ac_loaded: bool,
    ac_path: Path,
    bc_report: dict[str, Any],
    bc_loaded: bool,
    bc_path: Path,
    ab_report: dict[str, Any],
    ab_loaded: bool,
    ab_path: Path,
) -> dict[str, Any]:
    ac_evidence = _extract_ac_evidence(ac_report, ac_loaded)
    bc_evidence = _extract_bc_evidence(bc_report, bc_loaded)
    ab_evidence = _extract_ab_evidence(ab_report, ab_loaded)

    current_decision = _build_current_decision(ac_evidence, bc_evidence)

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "experimental_candidate_abc_decision_report",
            "report_name": "candidate_abc_decision_report",
        },
        "input_reports": _build_input_reports_section(
            ac_report,
            ac_loaded,
            ac_path,
            bc_report,
            bc_loaded,
            bc_path,
            ab_report,
            ab_loaded,
            ab_path,
        ),
        "candidate_positioning": _build_candidate_positioning(ac_evidence, bc_evidence, ab_evidence),
        "evidence_summary": _build_evidence_summary(ac_evidence, bc_evidence, ab_evidence),
        "current_decision": current_decision,
        "next_experiment_recommendation": _build_next_experiment_recommendation(current_decision),
        "final_diagnosis": _build_final_diagnosis(ac_evidence, bc_evidence),
    }


def build_experimental_candidate_abc_decision_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    input_reports = _safe_dict(summary.get("input_reports"))
    positioning = _safe_dict(summary.get("candidate_positioning"))
    evidence = _safe_dict(summary.get("evidence_summary"))
    decision = _safe_dict(summary.get("current_decision"))
    next_step = _safe_dict(summary.get("next_experiment_recommendation"))
    diagnosis = _safe_dict(summary.get("final_diagnosis"))
    ac_snapshot = _safe_dict(evidence.get("ac_numeric_snapshot"))
    bc_snapshot = _safe_dict(evidence.get("bc_numeric_snapshot"))

    lines = [
        "# Candidate ABC Decision Report",
        "",
        "## Metadata",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- report_type: {metadata.get('report_type', 'n/a')}",
        f"- report_name: {metadata.get('report_name', 'n/a')}",
        "",
        "## Final Interpretation",
        f"- primary_finding: {diagnosis.get('primary_finding', 'unknown')}",
        f"- secondary_finding: {diagnosis.get('secondary_finding', 'unknown')}",
        f"- summary: {diagnosis.get('summary', 'unknown')}",
        "",
        "## Input Reports",
    ]
    for key in (
        "candidate_a_vs_c_intersection",
        "candidate_b_vs_c_intersection",
        "candidate_a_vs_b_optional",
    ):
        payload = _safe_dict(input_reports.get(key))
        lines.append(
            f"- {key}: loaded={payload.get('loaded', False)}, path={payload.get('path', 'n/a')}, report_type={payload.get('report_type', 'n/a')}"
        )

    lines.extend(["", "## Candidate Positioning"])
    for candidate_key, label in (
        ("candidate_a", "Candidate A"),
        ("candidate_b", "Candidate B"),
        ("candidate_c2", "Candidate C2"),
    ):
        payload = _safe_dict(positioning.get(candidate_key))
        strengths = "; ".join(str(item) for item in _safe_list(payload.get("strengths"))) or "n/a"
        weaknesses = "; ".join(str(item) for item in _safe_list(payload.get("weaknesses"))) or "n/a"
        lines.append(f"- {label}: role={payload.get('role', 'n/a')}")
        lines.append(f"- {label} strengths: {strengths}")
        lines.append(f"- {label} weaknesses: {weaknesses}")
        lines.append(f"- {label} interpretation: {payload.get('current_interpretation', 'n/a')}")

    lines.extend(
        [
            "",
            "## Evidence Summary",
            f"- A vs C shared-row result: {evidence.get('candidate_a_vs_c_shared_row_result', 'n/a')}",
            f"- A vs C exclusive-row result: {evidence.get('candidate_a_vs_c_exclusive_row_result', 'n/a')}",
            f"- B vs C practical tradeoff result: {evidence.get('candidate_b_vs_c_practical_tradeoff_result', 'n/a')}",
            f"- A vs B reference result: {evidence.get('candidate_a_vs_b_reference_result', 'n/a')}",
            f"- C2 replacement vs A: {_safe_dict(evidence.get('c2_supported_as_replacement_vs_a')).get('decision', 'n/a')} - {_safe_dict(evidence.get('c2_supported_as_replacement_vs_a')).get('rationale', 'n/a')}",
            f"- C2 practical alternative vs B: {_safe_dict(evidence.get('c2_supported_as_practical_alternative_vs_b')).get('decision', 'n/a')} - {_safe_dict(evidence.get('c2_supported_as_practical_alternative_vs_b')).get('rationale', 'n/a')}",
            f"- A vs C shared_rows: {_format_number(ac_snapshot.get('shared_rows'))}",
            f"- A vs C shared_row_ratio_from_a: {_format_pct(ac_snapshot.get('shared_row_ratio_from_a'))}",
            f"- A vs C shared_row_ratio_from_c: {_format_pct(ac_snapshot.get('shared_row_ratio_from_c'))}",
        ]
    )

    flat_deltas = _safe_dict(ac_snapshot.get("flat_share_delta_by_horizon"))
    if flat_deltas:
        lines.append(
            "- A vs C flat_share_delta_by_horizon: "
            + ", ".join(f"{h}={_format_pct(flat_deltas.get(h))}" for h in TARGET_HORIZONS)
        )

    lines.extend(
        [
            f"- B vs C shared_row_ratio_vs_baseline: {_format_pct(bc_snapshot.get('shared_row_ratio_vs_baseline'))}",
            f"- B vs C shared_row_ratio_vs_experiment: {_format_pct(bc_snapshot.get('shared_row_ratio_vs_experiment'))}",
            f"- B vs C coverage_recovery_horizons: {', '.join(str(item) for item in _safe_list(bc_snapshot.get('coverage_recovery_horizons'))) or 'n/a'}",
        ]
    )

    bc_flat_deltas = _safe_dict(bc_snapshot.get("flat_ratio_change_by_horizon"))
    if bc_flat_deltas:
        lines.append(
            "- B vs C flat_ratio_change_by_horizon: "
            + ", ".join(f"{h}={_format_pct(bc_flat_deltas.get(h))}" for h in TARGET_HORIZONS)
        )

    lines.extend(
        [
            "",
            "## Current Decision",
            f"- default_baseline_candidate: {decision.get('default_baseline_candidate', 'n/a')}",
            f"- sparse_reference_candidate: {decision.get('sparse_reference_candidate', 'n/a')}",
            f"- coverage_expansion_candidate: {decision.get('coverage_expansion_candidate', 'n/a')}",
            f"- should_replace_a_with_c2: {_safe_dict(decision.get('should_replace_a_with_c2')).get('decision', 'n/a')} - {_safe_dict(decision.get('should_replace_a_with_c2')).get('rationale', 'n/a')}",
            f"- should_keep_c2_alive_for_followup: {_safe_dict(decision.get('should_keep_c2_alive_for_followup')).get('decision', 'n/a')} - {_safe_dict(decision.get('should_keep_c2_alive_for_followup')).get('rationale', 'n/a')}",
            "",
            "## Next Experiment Recommendation",
            f"- next_priority: {next_step.get('next_priority', 'n/a')}",
            f"- recommended_action: {next_step.get('recommended_action', 'n/a')}",
            "",
            "## Final Diagnosis",
            f"- recommendation: {diagnosis.get('recommendation', 'unknown')}",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_abc_decision_report(
    ac_report_path: Path = DEFAULT_AC_REPORT,
    bc_report_path: Path = DEFAULT_BC_REPORT,
    ab_report_path: Path = DEFAULT_AB_REPORT,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    ac_report, ac_loaded = _load_report(ac_report_path)
    bc_report, bc_loaded = _load_report(bc_report_path)
    ab_report, ab_loaded = _load_report(ab_report_path)

    summary = build_experimental_candidate_abc_decision_report(
        ac_report,
        ac_loaded=ac_loaded,
        ac_path=ac_report_path,
        bc_report=bc_report,
        bc_loaded=bc_loaded,
        bc_path=bc_report_path,
        ab_report=ab_report,
        ab_loaded=ab_loaded,
        ab_path=ab_report_path,
    )
    markdown = build_experimental_candidate_abc_decision_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an experimental decision report for Candidates A, B, and C2"
    )
    parser.add_argument("--ac-report", type=Path, default=DEFAULT_AC_REPORT)
    parser.add_argument("--bc-report", type=Path, default=DEFAULT_BC_REPORT)
    parser.add_argument("--ab-report", type=Path, default=DEFAULT_AB_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_abc_decision_report(
        ac_report_path=args.ac_report,
        bc_report_path=args.bc_report,
        ab_report_path=args.ab_report,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()