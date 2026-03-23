from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.research.edge_selection_input_mapper import map_edge_selection_input
from src.research.edge_selection_shadow_writer import read_edge_selection_shadow_outputs
from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    TARGET_HORIZONS,
    TARGET_LABELS,
    _safe_float,
    _safe_text,
    load_jsonl_records,
)
from src.research.experimental_candidate_intersection_utils import (
    MATCH_KEY_FIELDS,
    build_intersection_datasets,
    build_row_match_key,
    filter_candidate_c_records,
)

DEFAULT_CANDIDATE_C_DATASET = Path(
    "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
)
DEFAULT_CANDIDATE_C_VARIANT = "c2_moderate"
DEFAULT_RESEARCH_REPORTS_DIR = Path("logs/research_reports")
DEFAULT_LATEST_SUMMARY = Path("logs/research_reports/latest/summary.json")
DEFAULT_SHADOW_LOG = Path("logs/edge_selection_shadow/edge_selection_shadow.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/candidate_c_downstream_survival_diagnosis.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/candidate_c_downstream_survival_diagnosis.md"
)
TOP_DISTRIBUTION_LIMIT = 10
LOW_SURVIVAL_THRESHOLD = 0.20
MODERATE_SURVIVAL_THRESHOLD = 0.40
HIGH_CONCENTRATION_THRESHOLD = 0.60
MODERATE_CONCENTRATION_THRESHOLD = 0.40


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


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


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists() or not path.is_file():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    if not isinstance(payload, dict):
        return {}, False
    return payload, True


def _valid_label(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_LABELS:
        return text
    return None


def _symbol_value(row: dict[str, Any]) -> str:
    return _safe_text(row.get("symbol")) or "unknown"


def _strategy_value(row: dict[str, Any]) -> str:
    return _safe_text(row.get("selected_strategy") or row.get("strategy")) or "unknown"


def _build_exclusive_rows(
    source_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_counts: dict[tuple[str, ...], int] = {}
    shared_counts: dict[tuple[str, ...], int] = {}

    for row in source_rows:
        key = build_row_match_key(row)
        source_counts[key] = source_counts.get(key, 0) + 1
    for row in shared_rows:
        key = build_row_match_key(row)
        shared_counts[key] = shared_counts.get(key, 0) + 1

    emitted_counts: dict[tuple[str, ...], int] = {}
    exclusive_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = build_row_match_key(row)
        allowed = source_counts.get(key, 0) - shared_counts.get(key, 0)
        emitted = emitted_counts.get(key, 0)
        if emitted >= allowed:
            continue
        exclusive_rows.append(row)
        emitted_counts[key] = emitted + 1

    return exclusive_rows


def _build_directional_observations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        symbol = _symbol_value(row)
        strategy = _strategy_value(row)
        row_key = tuple(build_row_match_key(row))
        for horizon in TARGET_HORIZONS:
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if label is None:
                continue
            observations.append(
                {
                    "row_key": row_key,
                    "symbol": symbol,
                    "strategy": strategy,
                    "horizon": horizon,
                    "label": label,
                    "is_directional": label in {"up", "down"},
                }
            )
    return observations


def _identity_tuple(symbol: str | None, strategy: str | None, horizon: str | None) -> tuple[str, str, str]:
    return (
        _safe_text(symbol) or "unknown",
        _safe_text(strategy) or "unknown",
        _safe_text(horizon) or "unknown",
    )


def _top_distribution(counter: Counter[str], total: int) -> dict[str, Any]:
    rows = []
    for name, count in counter.most_common(TOP_DISTRIBUTION_LIMIT):
        rows.append({"name": name, "count": count, "ratio": _safe_ratio(count, total)})
    top_ratio = _safe_float(rows[0]["ratio"]) if rows else None
    return {"unique_count": len(counter), "top": rows, "top_ratio": top_ratio}


def _distribution_for_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    symbol_counter = Counter(obs["symbol"] for obs in observations)
    strategy_counter = Counter(obs["strategy"] for obs in observations)
    horizon_counter = Counter(obs["horizon"] for obs in observations)
    return {
        "symbol_distribution": _top_distribution(symbol_counter, len(observations)),
        "strategy_distribution": _top_distribution(strategy_counter, len(observations)),
        "horizon_distribution": _top_distribution(horizon_counter, len(observations)),
    }


def _load_mapper_result(research_reports_dir: Path) -> dict[str, Any]:
    try:
        result = map_edge_selection_input(research_reports_dir)
    except Exception as exc:
        return {
            "available": False,
            "ok": False,
            "errors": [f"mapper_execution_failed:{exc}"],
            "warnings": [],
            "candidates": [],
            "candidate_identities": set(),
            "candidate_seed_count": None,
            "candidate_seed_diagnostics": {},
        }

    candidates = [candidate for candidate in _safe_list(result.get("candidates")) if isinstance(candidate, dict)]
    identities = {
        _identity_tuple(candidate.get("symbol"), candidate.get("strategy"), candidate.get("horizon"))
        for candidate in candidates
    }
    return {
        "available": True,
        "ok": bool(result.get("ok")),
        "errors": _safe_list(result.get("errors")),
        "warnings": _safe_list(result.get("warnings")),
        "candidates": candidates,
        "candidate_identities": identities,
        "candidate_seed_count": result.get("candidate_seed_count"),
        "candidate_seed_diagnostics": _safe_dict(result.get("candidate_seed_diagnostics")),
    }


def _load_preview_snapshot(latest_summary_path: Path) -> dict[str, Any]:
    summary, loaded = _load_json(latest_summary_path)
    if not loaded:
        return {"available": False, "loaded": False, "by_horizon": {}}

    preview_root = _safe_dict(summary.get("edge_candidates_preview"))
    by_horizon = _safe_dict(preview_root.get("by_horizon"))
    parsed: dict[str, Any] = {}

    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(by_horizon.get(horizon))
        top_symbol = _safe_dict(horizon_payload.get("top_symbol"))
        top_strategy = _safe_dict(horizon_payload.get("top_strategy"))
        parsed[horizon] = {
            "top_symbol": top_symbol,
            "top_strategy": top_strategy,
            "top_symbol_group": _safe_text(top_symbol.get("group")),
            "top_strategy_group": _safe_text(top_strategy.get("group")),
            "top_symbol_candidate_strength": _safe_text(top_symbol.get("candidate_strength")),
            "top_strategy_candidate_strength": _safe_text(top_strategy.get("candidate_strength")),
            "top_symbol_quality_gate": _safe_text(top_symbol.get("quality_gate")),
            "top_strategy_quality_gate": _safe_text(top_strategy.get("quality_gate")),
        }

    return {"available": True, "loaded": True, "by_horizon": parsed}


def _preview_component_state(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    group = _safe_text(candidate_payload.get("group")) or "unknown"
    candidate_strength = _safe_text(candidate_payload.get("candidate_strength")) or "insufficient_data"
    quality_gate = _safe_text(candidate_payload.get("quality_gate")) or "failed"
    preview_visible = candidate_strength != "insufficient_data"
    failure_reasons: list[str] = []
    if group == "unknown":
        failure_reasons.append("group_not_visible_in_preview")
    if group != "unknown" and not preview_visible:
        failure_reasons.append("candidate_not_visible_in_preview")
    if group != "unknown" and preview_visible and quality_gate != "passed":
        failure_reasons.append("quality_gate_not_passed")
    return {
        "group": group,
        "preview_visible": preview_visible,
        "quality_gate_passed": quality_gate == "passed",
        "failure_reasons": failure_reasons,
    }


def _build_preview_observation_index(preview_snapshot: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not preview_snapshot.get("available"):
        return {}

    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(_safe_dict(preview_snapshot.get("by_horizon")).get(horizon))
        symbol_candidate = _safe_dict(horizon_payload.get("top_symbol"))
        strategy_candidate = _safe_dict(horizon_payload.get("top_strategy"))

        symbol_state = _preview_component_state(symbol_candidate)
        strategy_state = _preview_component_state(strategy_candidate)

        symbol_group = symbol_state["group"]
        strategy_group = strategy_state["group"]
        if symbol_group == "unknown" or strategy_group == "unknown":
            continue

        key = _identity_tuple(symbol_group, strategy_group, horizon)
        index[key] = {
            "included": symbol_state["preview_visible"] and strategy_state["preview_visible"],
            "passed": symbol_state["quality_gate_passed"] and strategy_state["quality_gate_passed"],
            "failure_reasons": sorted(
                set(symbol_state["failure_reasons"] + strategy_state["failure_reasons"])
            ),
        }
    return index


def _load_shadow_snapshot(shadow_log_path: Path) -> dict[str, Any]:
    if not shadow_log_path.exists() or not shadow_log_path.is_file():
        return {
            "available": False,
            "loaded": False,
            "runs": [],
            "identity_stats": {},
            "selection_identities": set(),
            "ranking_run_count": 0,
            "errors": [],
        }

    try:
        runs = read_edge_selection_shadow_outputs(shadow_log_path)
    except Exception as exc:
        return {
            "available": False,
            "loaded": False,
            "runs": [],
            "identity_stats": {},
            "selection_identities": set(),
            "ranking_run_count": 0,
            "errors": [f"shadow_read_failed:{exc}"],
        }

    identity_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    selection_identities: set[tuple[str, str, str]] = set()
    ranking_run_count = 0

    for run in runs:
        ranking = [item for item in _safe_list(run.get("ranking")) if isinstance(item, dict)]
        if ranking:
            ranking_run_count += 1

        if _safe_text(run.get("selection_status")) == "selected":
            selection_identities.add(
                _identity_tuple(
                    run.get("selected_symbol"),
                    run.get("selected_strategy"),
                    run.get("selected_horizon"),
                )
            )

        for item in ranking:
            key = _identity_tuple(item.get("symbol"), item.get("strategy"), item.get("horizon"))
            stats = identity_stats.setdefault(
                key,
                {
                    "ranking_count": 0,
                    "eligible_count": 0,
                    "rejected_count": 0,
                    "candidate_status_counts": Counter(),
                    "reason_codes": Counter(),
                    "eligibility_fail_reasons": Counter(),
                    "gate_fail_reasons": Counter(),
                },
            )
            stats["ranking_count"] += 1
            candidate_status = _safe_text(item.get("candidate_status")) or "unknown"
            stats["candidate_status_counts"][candidate_status] += 1
            if candidate_status == "eligible":
                stats["eligible_count"] += 1
            else:
                stats["rejected_count"] += 1

            for reason in _safe_list(item.get("reason_codes")):
                reason_text = _safe_text(reason)
                if reason_text is not None:
                    stats["reason_codes"][reason_text] += 1

            gate_diagnostics = _safe_dict(item.get("gate_diagnostics"))
            eligibility_gate = _safe_dict(gate_diagnostics.get("eligibility_gate"))
            if eligibility_gate.get("passed") is False:
                for reason in _safe_list(eligibility_gate.get("reason_codes")):
                    reason_text = _safe_text(reason)
                    if reason_text is not None:
                        stats["eligibility_fail_reasons"][reason_text] += 1

            for gate_name, gate_value in sorted(gate_diagnostics.items()):
                gate_payload = _safe_dict(gate_value)
                if gate_payload.get("passed") is False:
                    for reason in _safe_list(gate_payload.get("reason_codes")):
                        reason_text = _safe_text(reason)
                        if reason_text is not None:
                            stats["gate_fail_reasons"][f"{gate_name}:{reason_text}"] += 1

    return {
        "available": True,
        "loaded": True,
        "runs": runs,
        "identity_stats": identity_stats,
        "selection_identities": selection_identities,
        "ranking_run_count": ranking_run_count,
        "errors": [],
    }


def _aggregate_stage_metrics(
    observations: list[dict[str, Any]],
    *,
    mapper_snapshot: dict[str, Any],
    preview_snapshot: dict[str, Any],
    shadow_snapshot: dict[str, Any],
) -> dict[str, Any]:
    directional_observations = [obs for obs in observations if obs.get("is_directional")]
    total_directional = len(directional_observations)
    preview_index = _build_preview_observation_index(preview_snapshot)
    shadow_stats = (
        shadow_snapshot.get("identity_stats", {})
        if isinstance(shadow_snapshot.get("identity_stats"), dict)
        else {}
    )
    mapper_identities = (
        mapper_snapshot.get("candidate_identities", set())
        if isinstance(mapper_snapshot.get("candidate_identities"), set)
        else set()
    )
    selection_identities = (
        shadow_snapshot.get("selection_identities", set())
        if isinstance(shadow_snapshot.get("selection_identities"), set)
        else set()
    )

    summary = {
        "totals": {
            "exclusive_rows": len({obs["row_key"] for obs in observations}),
            "labeled_observations": len(observations),
            "directional_observations": total_directional,
        },
        "survival_rates": {},
        "failure_breakdown": {
            "preview_gate_fail_reasons": [],
            "eligibility_fail_reasons": [],
        },
        "horizon_level_survival": {h: {} for h in TARGET_HORIZONS},
        "distribution_analysis": {},
    }

    stage_buckets = {
        "mapper_included": [],
        "mapper_excluded": [],
        "preview_passed": [],
        "preview_failed": [],
        "eligible": [],
        "rejected": [],
        "selected": [],
        "not_selected": [],
    }
    preview_fail_counter: Counter[str] = Counter()
    eligibility_fail_counter: Counter[str] = Counter()

    for horizon in TARGET_HORIZONS:
        horizon_obs = [obs for obs in directional_observations if obs["horizon"] == horizon]
        mapper_included = 0
        preview_pass = 0
        eligible = 0
        selected = 0

        for obs in horizon_obs:
            identity = _identity_tuple(obs["symbol"], obs["strategy"], obs["horizon"])
            in_mapper = identity in mapper_identities
            if in_mapper:
                mapper_included += 1
                stage_buckets["mapper_included"].append(obs)
            else:
                stage_buckets["mapper_excluded"].append(obs)

            if preview_snapshot.get("available"):
                preview_state = preview_index.get(identity)
                if preview_state and preview_state.get("passed"):
                    preview_pass += 1
                    stage_buckets["preview_passed"].append(obs)
                else:
                    stage_buckets["preview_failed"].append(obs)
                    reasons = (
                        preview_state.get("failure_reasons", ["identity_not_visible_in_preview"])
                        if preview_state
                        else ["identity_not_visible_in_preview"]
                    )
                    for reason in reasons:
                        preview_fail_counter[str(reason)] += 1

            shadow_stage_available = bool(shadow_snapshot.get("available")) and int(
                shadow_snapshot.get("ranking_run_count", 0)
            ) > 0
            if shadow_stage_available:
                shadow_state = _safe_dict(shadow_stats.get(identity))
                if int(shadow_state.get("eligible_count", 0)) > 0:
                    eligible += 1
                    stage_buckets["eligible"].append(obs)
                else:
                    stage_buckets["rejected"].append(obs)
                    reason_counter = (
                        shadow_state.get("eligibility_fail_reasons")
                        or shadow_state.get("reason_codes")
                        or {}
                    )
                    if isinstance(reason_counter, Counter):
                        for reason, count in reason_counter.items():
                            eligibility_fail_counter[str(reason)] += int(count)

                if identity in selection_identities:
                    selected += 1
                    stage_buckets["selected"].append(obs)
                else:
                    stage_buckets["not_selected"].append(obs)

        shadow_stage_available = bool(shadow_snapshot.get("available")) and int(
            shadow_snapshot.get("ranking_run_count", 0)
        ) > 0
        summary["horizon_level_survival"][horizon] = {
            "directional_observations": len(horizon_obs),
            "mapper_included_count": mapper_included,
            "mapper_inclusion_rate": _safe_ratio(mapper_included, len(horizon_obs)),
            "preview_pass_count": preview_pass if preview_snapshot.get("available") else None,
            "preview_pass_rate": (
                _safe_ratio(preview_pass, mapper_included)
                if preview_snapshot.get("available") and mapper_included > 0
                else None
            ),
            "eligible_count": eligible if shadow_stage_available else None,
            "eligibility_rate": (
                _safe_ratio(eligible, len(horizon_obs)) if shadow_stage_available else None
            ),
            "final_selection_count": selected if shadow_stage_available else None,
            "final_selection_rate": (
                _safe_ratio(selected, len(horizon_obs)) if shadow_stage_available else None
            ),
        }

    mapper_included_total = len(stage_buckets["mapper_included"])
    shadow_stage_available = bool(shadow_snapshot.get("available")) and int(
        shadow_snapshot.get("ranking_run_count", 0)
    ) > 0

    summary["survival_rates"] = {
        "tracking_unit": "directional_c2_exclusive_row_observation_by_symbol_strategy_horizon",
        "mapper_stage_available": mapper_snapshot.get("available", False) and mapper_snapshot.get("ok", False),
        "preview_stage_available": preview_snapshot.get("available", False),
        "eligibility_stage_available": shadow_stage_available,
        "mapper_inclusion_rate": _safe_ratio(mapper_included_total, total_directional),
        "preview_pass_rate": (
            _safe_ratio(len(stage_buckets["preview_passed"]), mapper_included_total)
            if preview_snapshot.get("available") and mapper_included_total > 0
            else None
        ),
        "eligibility_rate": (
            _safe_ratio(len(stage_buckets["eligible"]), total_directional)
            if shadow_stage_available
            else None
        ),
        "final_selection_rate": (
            _safe_ratio(len(stage_buckets["selected"]), total_directional)
            if shadow_stage_available
            else None
        ),
        "mapper_included_count": mapper_included_total,
        "preview_pass_count": len(stage_buckets["preview_passed"]) if preview_snapshot.get("available") else None,
        "eligible_count": len(stage_buckets["eligible"]) if shadow_stage_available else None,
        "final_selection_count": len(stage_buckets["selected"]) if shadow_stage_available else None,
    }

    summary["failure_breakdown"] = {
        "preview_gate_fail_reasons": [
            {"reason": reason, "count": count}
            for reason, count in preview_fail_counter.most_common(TOP_DISTRIBUTION_LIMIT)
        ],
        "eligibility_fail_reasons": [
            {"reason": reason, "count": count}
            for reason, count in eligibility_fail_counter.most_common(TOP_DISTRIBUTION_LIMIT)
        ],
    }

    summary["distribution_analysis"] = {
        "mapper_included": _distribution_for_observations(stage_buckets["mapper_included"]),
        "mapper_excluded": _distribution_for_observations(stage_buckets["mapper_excluded"]),
        "preview_passed": _distribution_for_observations(stage_buckets["preview_passed"]),
        "preview_failed": _distribution_for_observations(stage_buckets["preview_failed"]),
        "eligible": _distribution_for_observations(stage_buckets["eligible"]),
        "rejected": _distribution_for_observations(stage_buckets["rejected"]),
        "selected": _distribution_for_observations(stage_buckets["selected"]),
        "not_selected": _distribution_for_observations(stage_buckets["not_selected"]),
    }
    return summary


def _build_final_diagnosis(survival_summary: dict[str, Any]) -> dict[str, Any]:
    rates = _safe_dict(survival_summary.get("survival_rates"))
    totals = _safe_dict(survival_summary.get("totals"))
    mapper_available = bool(rates.get("mapper_stage_available"))
    preview_available = bool(rates.get("preview_stage_available"))
    eligibility_available = bool(rates.get("eligibility_stage_available"))
    directional_observations = int(totals.get("directional_observations", 0))

    if directional_observations <= 0:
        return {
            "primary_finding": "c2_exclusive_rows_not_present_in_loaded_inputs",
            "secondary_finding": "downstream_survival_is_blocked_before_stage_tracing",
            "recommendation": "Re-run this diagnosis against the actual Candidate A and Candidate C2 relabel outputs so exclusive identities exist before evaluating downstream survival.",
            "summary": "No directional C2-exclusive observations were available in the loaded relabel datasets, so downstream survival could not be assessed from data.",
        }

    if not mapper_available and not preview_available and not eligibility_available:
        return {
            "primary_finding": "downstream_survival_cannot_be_assessed_with_current_outputs",
            "secondary_finding": "mapper_preview_and_shadow_evidence_are_missing_or_unusable",
            "recommendation": "Re-run the experimental research pipeline until mapper, preview, and shadow-selection artifacts are available, then rerun this diagnosis.",
            "summary": "The downstream survival question is currently blocked because the available experimental outputs do not expose usable mapper, preview, or ranking evidence for C2-exclusive identities.",
        }

    mapper_rate = _safe_float(rates.get("mapper_inclusion_rate"))
    eligibility_rate = _safe_float(rates.get("eligibility_rate"))
    final_selection_rate = _safe_float(rates.get("final_selection_rate"))

    eligible_distribution = _safe_dict(_safe_dict(survival_summary.get("distribution_analysis")).get("eligible"))
    top_strategy = _safe_dict(
        (_safe_list(_safe_dict(eligible_distribution.get("strategy_distribution")).get("top")) or [{}])[0]
    )
    top_strategy_ratio = _safe_float(top_strategy.get("ratio")) or 0.0

    if eligibility_available and eligibility_rate is not None and eligibility_rate < LOW_SURVIVAL_THRESHOLD:
        primary = "c2_exclusive_rows_mostly_die_before_becoming_eligible_candidates"
        secondary = "coverage_expansion_is_not_translating_into_usable_downstream_supply"
        recommendation = "Treat current C2 coverage expansion as low value for seed starvation and consider redesigning the candidate or trying a C3 variant."
    elif eligibility_available and eligibility_rate is not None and eligibility_rate >= MODERATE_SURVIVAL_THRESHOLD:
        primary = "c2_exclusive_rows_do_reach_the_candidate_pool_in_meaningful_volume"
        secondary = "coverage_expansion_is_translating_into_downstream_candidate_supply"
        recommendation = "Keep C2 active in follow-up experiments and measure whether the surviving exclusive identities improve final seed availability over time."
    elif mapper_available and mapper_rate is not None and mapper_rate < LOW_SURVIVAL_THRESHOLD:
        primary = "c2_exclusive_rows_are_filtered_out_very_early"
        secondary = "coverage_expansion_breaks_down_before_candidate_generation"
        recommendation = "Focus next on why C2-exclusive identities are not surviving mapper inclusion before spending more effort on later selection stages."
    else:
        primary = "c2_exclusive_rows_have_mixed_downstream_survival"
        secondary = "coverage_expansion_needs_stage_by_stage_follow_up"
        recommendation = "Keep C2 exploratory, and inspect the earliest stage with material attrition before deciding whether to redesign or specialize it."

    if top_strategy_ratio >= HIGH_CONCENTRATION_THRESHOLD:
        secondary = "survival_is_highly_concentrated_in_one_strategy"
        recommendation += " The surviving slice looks specialized, so treat C2 as a strategy-specific candidate rather than a general replacement."
    elif top_strategy_ratio >= MODERATE_CONCENTRATION_THRESHOLD:
        secondary = "survival_is_moderately_concentrated"

    summary = (
        "This diagnosis asks whether C2-exclusive identities survive mapper, preview, eligibility, and final-selection stages strongly enough to matter for downstream candidate availability."
    )
    if eligibility_available and eligibility_rate is not None:
        summary += f" Eligibility survival is {_format_pct(eligibility_rate)} of directional exclusive observations."
    elif mapper_available and mapper_rate is not None:
        summary += f" Mapper inclusion is {_format_pct(mapper_rate)} of directional exclusive observations, but later-stage evidence is still incomplete."
    if final_selection_rate is not None:
        summary += f" Final selection is {_format_pct(final_selection_rate)}."

    return {
        "primary_finding": primary,
        "secondary_finding": secondary,
        "recommendation": recommendation,
        "summary": summary,
    }


def build_experimental_candidate_c_downstream_survival_diagnosis(
    candidate_a_records: list[dict[str, Any]],
    candidate_c_records: list[dict[str, Any]],
    *,
    candidate_a_path: Path,
    candidate_c_path: Path,
    candidate_a_instrumentation: dict[str, int] | None = None,
    candidate_c_instrumentation: dict[str, int] | None = None,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
    research_reports_dir: Path = DEFAULT_RESEARCH_REPORTS_DIR,
    latest_summary_path: Path = DEFAULT_LATEST_SUMMARY,
    shadow_log_path: Path = DEFAULT_SHADOW_LOG,
) -> dict[str, Any]:
    filtered_candidate_c_records = filter_candidate_c_records(
        candidate_c_records,
        variant_name=candidate_c_variant,
    )
    _, candidate_c_shared_rows, intersection_overview = build_intersection_datasets(
        candidate_a_records,
        filtered_candidate_c_records,
    )
    candidate_c_only_rows = _build_exclusive_rows(filtered_candidate_c_records, candidate_c_shared_rows)
    observations = _build_directional_observations(candidate_c_only_rows)

    mapper_snapshot = _load_mapper_result(research_reports_dir)
    preview_snapshot = _load_preview_snapshot(latest_summary_path)
    shadow_snapshot = _load_shadow_snapshot(shadow_log_path)
    survival_summary = _aggregate_stage_metrics(
        observations,
        mapper_snapshot=mapper_snapshot,
        preview_snapshot=preview_snapshot,
        shadow_snapshot=shadow_snapshot,
    )
    final_diagnosis = _build_final_diagnosis(survival_summary)

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "experimental_candidate_c_downstream_survival_diagnosis",
            "report_name": "candidate_c_downstream_survival_diagnosis",
        },
        "inputs": {
            "candidate_a_path": str(candidate_a_path),
            "candidate_c_path": str(candidate_c_path),
            "candidate_c_variant": candidate_c_variant,
            "candidate_a_parser_instrumentation": candidate_a_instrumentation or {},
            "candidate_c_parser_instrumentation": candidate_c_instrumentation or {},
            "candidate_a_raw_total_rows": len(candidate_a_records),
            "candidate_c_raw_total_rows": len(candidate_c_records),
            "candidate_c_filtered_row_count": len(filtered_candidate_c_records),
            "research_reports_dir": str(research_reports_dir),
            "latest_summary_path": str(latest_summary_path),
            "shadow_log_path": str(shadow_log_path),
            "row_key_definition_summary": (
                "Exclusive rows are derived with the existing experimental intersection key: "
                + ", ".join(MATCH_KEY_FIELDS)
                + "."
            ),
            "tracking_unit_definition_summary": (
                "Downstream survival is tracked at the directional C2-exclusive observation level using the exact identity tuple (symbol, strategy, horizon), because downstream research artifacts do not preserve raw relabel row keys."
            ),
            "downstream_stage_availability": {
                "mapper_available": mapper_snapshot.get("available", False),
                "mapper_ok": mapper_snapshot.get("ok", False),
                "preview_available": preview_snapshot.get("available", False),
                "shadow_available": shadow_snapshot.get("available", False),
                "shadow_ranking_run_count": shadow_snapshot.get("ranking_run_count", 0),
                "mapper_errors": mapper_snapshot.get("errors", []),
                "mapper_warnings": mapper_snapshot.get("warnings", []),
                "shadow_errors": shadow_snapshot.get("errors", []),
            },
        },
        "exclusive_row_count": {
            "candidate_a_total_rows": int(intersection_overview.get("baseline_total_rows", 0)),
            "candidate_c_total_rows": int(intersection_overview.get("experiment_total_rows", 0)),
            "shared_rows": int(intersection_overview.get("shared_row_count", 0)),
            "candidate_a_only_rows": int(intersection_overview.get("baseline_only_row_count", 0)),
            "candidate_c_only_rows": int(intersection_overview.get("experiment_only_row_count", 0)),
            "shared_row_ratio_from_a": _safe_float(intersection_overview.get("shared_ratio_vs_baseline")),
            "shared_row_ratio_from_c": _safe_float(intersection_overview.get("shared_ratio_vs_experiment")),
            "labeled_observation_count": _safe_dict(survival_summary.get("totals")).get("labeled_observations"),
            "directional_observation_count": _safe_dict(survival_summary.get("totals")).get("directional_observations"),
        },
        "survival_metrics": survival_summary.get("survival_rates", {}),
        "failure_breakdown": survival_summary.get("failure_breakdown", {}),
        "distribution_analysis": {
            **_safe_dict(survival_summary.get("distribution_analysis")),
            "horizon_level_survival": survival_summary.get("horizon_level_survival", {}),
        },
        "final_diagnosis": final_diagnosis,
    }


def build_experimental_candidate_c_downstream_survival_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    inputs = _safe_dict(summary.get("inputs"))
    counts = _safe_dict(summary.get("exclusive_row_count"))
    metrics = _safe_dict(summary.get("survival_metrics"))
    failures = _safe_dict(summary.get("failure_breakdown"))
    distribution = _safe_dict(summary.get("distribution_analysis"))
    horizons = _safe_dict(distribution.get("horizon_level_survival"))
    diagnosis = _safe_dict(summary.get("final_diagnosis"))
    stage_availability = _safe_dict(inputs.get("downstream_stage_availability"))

    lines = [
        "# Candidate C2 Downstream Survival Diagnosis",
        "",
        "## Metadata",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- report_type: {metadata.get('report_type', 'n/a')}",
        f"- report_name: {metadata.get('report_name', 'n/a')}",
        "",
        "## Inputs",
        f"- candidate_a_path: {inputs.get('candidate_a_path', 'n/a')}",
        f"- candidate_c_path: {inputs.get('candidate_c_path', 'n/a')}",
        f"- candidate_c_variant: {inputs.get('candidate_c_variant', 'n/a')}",
        f"- candidate_a_parser_instrumentation: {json.dumps(inputs.get('candidate_a_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_c_parser_instrumentation: {json.dumps(inputs.get('candidate_c_parser_instrumentation', {}), ensure_ascii=False, sort_keys=True)}",
        f"- candidate_a_raw_total_rows: {inputs.get('candidate_a_raw_total_rows', 0)}",
        f"- candidate_c_raw_total_rows: {inputs.get('candidate_c_raw_total_rows', 0)}",
        f"- candidate_c_filtered_row_count: {inputs.get('candidate_c_filtered_row_count', 0)}",
        f"- research_reports_dir: {inputs.get('research_reports_dir', 'n/a')}",
        f"- latest_summary_path: {inputs.get('latest_summary_path', 'n/a')}",
        f"- shadow_log_path: {inputs.get('shadow_log_path', 'n/a')}",
        f"- row_key_definition_summary: {inputs.get('row_key_definition_summary', 'n/a')}",
        f"- tracking_unit_definition_summary: {inputs.get('tracking_unit_definition_summary', 'n/a')}",
        "",
        "## Downstream Stage Availability",
        f"- mapper_available: {stage_availability.get('mapper_available', False)}",
        f"- mapper_ok: {stage_availability.get('mapper_ok', False)}",
        f"- preview_available: {stage_availability.get('preview_available', False)}",
        f"- shadow_available: {stage_availability.get('shadow_available', False)}",
        f"- shadow_ranking_run_count: {stage_availability.get('shadow_ranking_run_count', 0)}",
        f"- mapper_errors: {json.dumps(stage_availability.get('mapper_errors', []), ensure_ascii=False)}",
        f"- mapper_warnings: {json.dumps(stage_availability.get('mapper_warnings', []), ensure_ascii=False)}",
        f"- shadow_errors: {json.dumps(stage_availability.get('shadow_errors', []), ensure_ascii=False)}",
        "",
        "## Exclusive Row Count",
        f"- candidate_a_total_rows: {counts.get('candidate_a_total_rows', 0)}",
        f"- candidate_c_total_rows: {counts.get('candidate_c_total_rows', 0)}",
        f"- shared_rows: {counts.get('shared_rows', 0)}",
        f"- candidate_a_only_rows: {counts.get('candidate_a_only_rows', 0)}",
        f"- candidate_c_only_rows: {counts.get('candidate_c_only_rows', 0)}",
        f"- shared_row_ratio_from_a: {_format_pct(counts.get('shared_row_ratio_from_a'))}",
        f"- shared_row_ratio_from_c: {_format_pct(counts.get('shared_row_ratio_from_c'))}",
        f"- labeled_observation_count: {counts.get('labeled_observation_count', 0)}",
        f"- directional_observation_count: {counts.get('directional_observation_count', 0)}",
        "",
        "## Survival Metrics",
        f"- tracking_unit: {metrics.get('tracking_unit', 'n/a')}",
        f"- mapper_inclusion_rate: {_format_pct(metrics.get('mapper_inclusion_rate'))}",
        f"- preview_pass_rate: {_format_pct(metrics.get('preview_pass_rate'))}",
        f"- eligibility_rate: {_format_pct(metrics.get('eligibility_rate'))}",
        f"- final_selection_rate: {_format_pct(metrics.get('final_selection_rate'))}",
        f"- mapper_included_count: {metrics.get('mapper_included_count', 'n/a')}",
        f"- preview_pass_count: {metrics.get('preview_pass_count', 'n/a')}",
        f"- eligible_count: {metrics.get('eligible_count', 'n/a')}",
        f"- final_selection_count: {metrics.get('final_selection_count', 'n/a')}",
        f"- mapper_stage_available: {metrics.get('mapper_stage_available', False)}",
        f"- preview_stage_available: {metrics.get('preview_stage_available', False)}",
        f"- eligibility_stage_available: {metrics.get('eligibility_stage_available', False)}",
        "",
        "## Failure Breakdown",
    ]

    preview_rows = _safe_list(failures.get("preview_gate_fail_reasons"))
    if preview_rows:
        lines.append(
            "- preview_gate_fail_reasons: "
            + "; ".join(
                f"{_safe_dict(row).get('reason', 'unknown')}={_safe_dict(row).get('count', 0)}"
                for row in preview_rows
            )
        )
    else:
        lines.append("- preview_gate_fail_reasons: none_or_unavailable")

    eligibility_rows = _safe_list(failures.get("eligibility_fail_reasons"))
    if eligibility_rows:
        lines.append(
            "- eligibility_fail_reasons: "
            + "; ".join(
                f"{_safe_dict(row).get('reason', 'unknown')}={_safe_dict(row).get('count', 0)}"
                for row in eligibility_rows
            )
        )
    else:
        lines.append("- eligibility_fail_reasons: none_or_unavailable")

    lines.extend(["", "## Horizon-Level Survival"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(horizons.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"directional={payload.get('directional_observations', 0)}, "
            f"mapper_rate={_format_pct(payload.get('mapper_inclusion_rate'))}, "
            f"preview_rate={_format_pct(payload.get('preview_pass_rate'))}, "
            f"eligibility_rate={_format_pct(payload.get('eligibility_rate'))}, "
            f"final_selection_rate={_format_pct(payload.get('final_selection_rate'))}"
        )

    lines.extend(["", "## Distribution Analysis"])
    for section_key, label in (
        ("mapper_included", "mapper included"),
        ("mapper_excluded", "mapper excluded"),
        ("preview_passed", "preview passed"),
        ("preview_failed", "preview failed"),
        ("eligible", "eligible"),
        ("rejected", "rejected"),
        ("selected", "selected"),
        ("not_selected", "not selected"),
    ):
        payload = _safe_dict(distribution.get(section_key))
        symbol_preview = "; ".join(
            f"{_safe_dict(row).get('name', 'unknown')}={_safe_dict(row).get('count', 0)} ({_format_pct(_safe_dict(row).get('ratio'))})"
            for row in _safe_list(_safe_dict(payload.get("symbol_distribution")).get("top"))[:5]
        )
        strategy_preview = "; ".join(
            f"{_safe_dict(row).get('name', 'unknown')}={_safe_dict(row).get('count', 0)} ({_format_pct(_safe_dict(row).get('ratio'))})"
            for row in _safe_list(_safe_dict(payload.get("strategy_distribution")).get("top"))[:5]
        )
        horizon_preview = "; ".join(
            f"{_safe_dict(row).get('name', 'unknown')}={_safe_dict(row).get('count', 0)} ({_format_pct(_safe_dict(row).get('ratio'))})"
            for row in _safe_list(_safe_dict(payload.get("horizon_distribution")).get("top"))[:5]
        )
        lines.append(f"- {label} symbols: {symbol_preview or 'none'}")
        lines.append(f"- {label} strategies: {strategy_preview or 'none'}")
        lines.append(f"- {label} horizons: {horizon_preview or 'none'}")

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {diagnosis.get('secondary_finding', 'unknown')}",
            f"- recommendation: {diagnosis.get('recommendation', 'unknown')}",
            f"- summary: {diagnosis.get('summary', 'unknown')}",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_c_downstream_survival_diagnosis(
    candidate_a_path: Path = CANDIDATE_A_DEFAULT_PATH,
    candidate_c_path: Path = DEFAULT_CANDIDATE_C_DATASET,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
    research_reports_dir: Path = DEFAULT_RESEARCH_REPORTS_DIR,
    latest_summary_path: Path = DEFAULT_LATEST_SUMMARY,
    shadow_log_path: Path = DEFAULT_SHADOW_LOG,
) -> dict[str, Any]:
    candidate_a_records, candidate_a_instrumentation = load_jsonl_records(candidate_a_path)
    candidate_c_records, candidate_c_instrumentation = load_jsonl_records(candidate_c_path)

    summary = build_experimental_candidate_c_downstream_survival_diagnosis(
        candidate_a_records,
        candidate_c_records,
        candidate_a_path=candidate_a_path,
        candidate_c_path=candidate_c_path,
        candidate_a_instrumentation=candidate_a_instrumentation,
        candidate_c_instrumentation=candidate_c_instrumentation,
        candidate_c_variant=candidate_c_variant,
        research_reports_dir=research_reports_dir,
        latest_summary_path=latest_summary_path,
        shadow_log_path=shadow_log_path,
    )
    markdown = build_experimental_candidate_c_downstream_survival_markdown(summary)

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
        description="Diagnose downstream survival of Candidate C2-exclusive rows"
    )
    parser.add_argument("--candidate-a-path", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--candidate-c-path", type=Path, default=DEFAULT_CANDIDATE_C_DATASET)
    parser.add_argument("--candidate-c-variant", default=DEFAULT_CANDIDATE_C_VARIANT)
    parser.add_argument("--research-reports-dir", type=Path, default=DEFAULT_RESEARCH_REPORTS_DIR)
    parser.add_argument("--latest-summary-path", type=Path, default=DEFAULT_LATEST_SUMMARY)
    parser.add_argument("--shadow-log-path", type=Path, default=DEFAULT_SHADOW_LOG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_c_downstream_survival_diagnosis(
        candidate_a_path=args.candidate_a_path,
        candidate_c_path=args.candidate_c_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
        candidate_c_variant=args.candidate_c_variant,
        research_reports_dir=args.research_reports_dir,
        latest_summary_path=args.latest_summary_path,
        shadow_log_path=args.shadow_log_path,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()