from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from src.research.research_analyzer import (
    EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT,
    EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
    EDGE_EARLY_MODERATE_ROBUSTNESS_PCT,
    EDGE_EARLY_MODERATE_SAMPLE_COUNT,
    EDGE_MODERATE_MEDIAN_RETURN_PCT,
    EDGE_MODERATE_SAMPLE_COUNT,
    HORIZONS,
    MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
    MODERATE_MIN_AGGREGATE_SCORE,
    MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE,
    MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE,
    MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE,
    POSITIVE_RATE_MINIMUM_FLOOR_PCT,
    THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT,
    THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT,
)


REPORT_TYPE = "analyzer_counterfactual_strictness_report"
REPORT_TITLE = "Analyzer Counterfactual Strictness Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_SUMMARY_PATH = Path("logs/research_reports/latest/summary.json")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

KNOWN_HORIZON_ORDER = tuple(str(item) for item in HORIZONS)
_KNOWN_HORIZON_INDEX = {
    horizon: index for index, horizon in enumerate(KNOWN_HORIZON_ORDER)
}

UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE = "unsupported_strategy_or_horizon_noise"
NEAR_MISS_SAMPLE_FLOOR = "near_miss_sample_floor"
NEAR_MISS_SAMPLE_MODERATION = "near_miss_sample_moderation"
NEAR_MISS_POSITIVE_RATE = "near_miss_positive_rate"
NEAR_MISS_OTHER_NARROW = "near_miss_other_narrow"
STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING = (
    "structurally_weak_or_still_unconvincing"
)
DIAGNOSIS_BUCKET_ORDER = (
    UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE,
    NEAR_MISS_SAMPLE_FLOOR,
    NEAR_MISS_SAMPLE_MODERATION,
    NEAR_MISS_POSITIVE_RATE,
    NEAR_MISS_OTHER_NARROW,
    STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING,
)
NEAR_MISS_BUCKETS = frozenset(
    {
        NEAR_MISS_SAMPLE_FLOOR,
        NEAR_MISS_SAMPLE_MODERATION,
        NEAR_MISS_POSITIVE_RATE,
        NEAR_MISS_OTHER_NARROW,
    }
)

SCENARIO_ORDER = (
    "baseline_current_rules",
    "baseline_excluding_incompatible_noise",
    "narrow_sample_floor_relief",
    "narrow_sample_moderation_relief",
    "narrow_positive_rate_near_threshold_relief",
    "combined_narrow_relief",
)
SCENARIO_DESCRIPTIONS = {
    "baseline_current_rules": "Reference-only current analyzer behavior.",
    "baseline_excluding_incompatible_noise": (
        "Reference-only view after removing unsupported strategy/horizon noise."
    ),
    "narrow_sample_floor_relief": (
        "Diagnosis-only rescue for rows just below the absolute sample floor."
    ),
    "narrow_sample_moderation_relief": (
        "Diagnosis-only rescue for rows just below the moderate-sample boundary."
    ),
    "narrow_positive_rate_near_threshold_relief": (
        "Diagnosis-only rescue for rows narrowly below positive-rate thresholds."
    ),
    "combined_narrow_relief": (
        "Union of the conservative diagnosis-only near-miss rescue buckets."
    ),
}

DIAGNOSIS_SAMPLE_FLOOR_MARGIN = 2
DIAGNOSIS_SAMPLE_MODERATION_MARGIN = 3
DIAGNOSIS_POSITIVE_RATE_MARGIN_PCT = 2.0
DIAGNOSIS_AGGREGATE_SCORE_MARGIN = 3.0

AGGREGATE_THRESHOLD_BY_REASON = {
    "aggregate_below_moderate_threshold": MODERATE_MIN_AGGREGATE_SCORE,
    "one_supporting_deficit_but_aggregate_too_low": (
        MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE
    ),
    "two_supporting_deficits_but_aggregate_too_low": (
        MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE
    ),
    "three_supporting_deficits_but_aggregate_too_low": (
        MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only counterfactual strictness report from a single "
            "analyzer summary.json artifact."
        )
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--write-latest-copy", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path, resolution = resolve_summary_path(args.summary_path)
    report = build_report(
        summary_path=summary_path,
        summary_path_resolution=resolution,
    )

    written_paths: dict[str, str] = {}
    if args.write_latest_copy:
        written_paths = write_report_files(report, args.output_dir)

    combined = _safe_dict(report["counterfactual_scenarios"]).get(
        "combined_narrow_relief",
        {},
    )
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "summary_path": report["inputs"]["summary_path"],
                "summary_path_resolution": report["inputs"]["summary_path_resolution"],
                "summary_file_exists": report["artifact_presence"]["summary_file_exists"],
                "preview_block_exists": report["analyzer_preview"]["preview_block_exists"],
                "joined_row_block_exists": report["joined_row_artifact"]["joined_row_block_exists"],
                "eligible_joined_row_count": report["joined_row_artifact"]["eligible_joined_row_count"],
                "rejected_joined_row_count": report["rejected_row_diagnostics"]["normalized_rejected_row_count"],
                "combined_narrow_relief_rescued_rejected_row_count": combined.get(
                    "rescued_rejected_row_count",
                    0,
                ),
                "final_assessment": report["final_assessment"]["assessment"],
                "written_paths": written_paths,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_summary_path(explicit_path: Path | None) -> tuple[Path, str]:
    candidate = explicit_path if explicit_path is not None else DEFAULT_SUMMARY_PATH
    resolution = "explicit" if explicit_path is not None else "default"
    resolved = candidate.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve(), resolution


def build_report(
    *,
    summary_path: Path,
    summary_path_resolution: str = "explicit",
) -> dict[str, Any]:
    loaded = load_summary_artifact(summary_path)
    payload = loaded["payload"]
    preview = _build_preview_section(payload)
    joined = _build_joined_row_section(payload)
    threshold_context = _build_threshold_context()
    rejected_rows = _collect_rejected_row_details(
        joined_block=_safe_dict(payload.get("edge_candidate_rows")),
        threshold_context=threshold_context,
    )
    rejected = _build_rejected_row_diagnostics(rejected_rows, joined)
    scenarios = _build_counterfactual_scenarios(rejected_rows, joined)
    final_assessment = _build_final_assessment(
        loaded=loaded,
        preview=preview,
        joined=joined,
        rejected=rejected,
        scenarios=scenarios,
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "inputs": {
            "summary_path": str(summary_path),
            "summary_path_resolution": summary_path_resolution,
        },
        "artifact_presence": {
            "summary_file_exists": loaded["exists"],
            "summary_path_is_file": loaded["is_file"],
            "summary_json_loaded": loaded["json_loaded"],
            "summary_payload_is_object": loaded["payload_is_object"],
            "load_error": loaded["load_error"],
            "top_level_keys": loaded["top_level_keys"],
        },
        "analyzer_preview": preview,
        "joined_row_artifact": joined,
        "threshold_context": threshold_context,
        "rejected_row_diagnostics": rejected,
        "counterfactual_scenarios": scenarios,
        "final_assessment": final_assessment,
        "rejected_row_details": rejected_rows,
        "assumptions": [
            "This report reads one analyzer summary artifact only.",
            "Counterfactual scenarios are diagnosis-only and do not change production behavior.",
            "Trade-analysis JSONL, mapper, and engine are out of scope for this report.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    joined = _safe_dict(report.get("joined_row_artifact"))
    rejected = _safe_dict(report.get("rejected_row_diagnostics"))
    combined = _safe_dict(
        _safe_dict(report.get("counterfactual_scenarios")).get("combined_narrow_relief")
    )
    final_assessment = _safe_dict(report.get("final_assessment"))
    return "\n".join(
        [
            f"# {REPORT_TITLE}",
            "",
            f"- eligible_joined_row_count: {joined.get('eligible_joined_row_count')}",
            f"- rejected_joined_row_count: {rejected.get('normalized_rejected_row_count')}",
            f"- diagnosis_bucket_counts: {json.dumps(rejected.get('diagnosis_bucket_counts', {}), ensure_ascii=False)}",
            f"- combined_narrow_relief_rescued_rejected_row_count: {combined.get('rescued_rejected_row_count', 0)}",
            f"- final_assessment: {final_assessment.get('assessment', 'mixed_or_inconclusive')}",
            "",
        ]
    )


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    md_path = output_dir / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json_report": str(json_path), "markdown_report": str(md_path)}


def load_summary_artifact(path: Path) -> dict[str, Any]:
    result = {
        "exists": path.exists(),
        "is_file": path.is_file(),
        "json_loaded": False,
        "payload_is_object": False,
        "load_error": None,
        "payload": {},
        "top_level_keys": [],
    }
    if not result["exists"]:
        result["load_error"] = "summary_path_missing"
        return result
    if not result["is_file"]:
        result["load_error"] = "summary_path_not_file"
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        result["load_error"] = "summary_path_unreadable"
        return result
    except json.JSONDecodeError:
        result["load_error"] = "summary_json_invalid"
        return result
    result["json_loaded"] = True
    if not isinstance(payload, dict):
        result["load_error"] = "summary_payload_not_object"
        return result
    result["payload_is_object"] = True
    result["payload"] = payload
    result["top_level_keys"] = sorted(str(key) for key in payload)
    return result


def _build_preview_section(summary: dict[str, Any]) -> dict[str, Any]:
    preview_value = summary.get("edge_candidates_preview")
    by_horizon = _safe_dict(_safe_dict(preview_value).get("by_horizon"))
    horizons = _sort_horizon_strings(list(by_horizon.keys()))
    return {
        "preview_block_exists": isinstance(preview_value, dict),
        "preview_by_horizon_present": isinstance(
            _safe_dict(preview_value).get("by_horizon"),
            dict,
        ),
        "preview_horizons_present": horizons,
        "preview_data_present": bool(horizons),
    }


def _build_joined_row_section(summary: dict[str, Any]) -> dict[str, Any]:
    joined_value = summary.get("edge_candidate_rows")
    joined_block = _safe_dict(joined_value)
    rows = _safe_dict_list(joined_block.get("rows"))
    diagnostic_rows = _safe_dict_list(joined_block.get("diagnostic_rows"))
    empty_reason = _safe_dict(joined_block.get("empty_reason_summary"))

    eligible = _safe_int(joined_block.get("row_count"))
    rejected = _safe_int(joined_block.get("diagnostic_row_count"))
    eligible = len(rows) if eligible is None else eligible
    rejected = len(diagnostic_rows) if rejected is None else rejected

    rejection_reason_counts = _safe_counter_dict(
        empty_reason.get("diagnostic_rejection_reason_counts")
    ) or _ordered_counter(
        Counter(
            _text(row.get("rejection_reason")) or "unknown"
            for row in diagnostic_rows
        )
    )
    diagnostic_category_counts = _safe_counter_dict(
        empty_reason.get("diagnostic_category_counts")
    ) or _ordered_counter(
        Counter(
            _text(row.get("diagnostic_category")) or "other_rejection"
            for row in diagnostic_rows
        )
    )

    return {
        "joined_row_block_exists": isinstance(joined_value, dict),
        "eligible_joined_row_count": eligible,
        "rejected_joined_row_count": rejected,
        "joined_row_total_count": eligible + rejected,
        "dropped_row_count": _safe_int(joined_block.get("dropped_row_count")),
        "identity_horizon_evaluation_count": len(
            _safe_dict_list(joined_block.get("identity_horizon_evaluations"))
        ),
        "has_eligible_rows": _safe_bool(empty_reason.get("has_eligible_rows")),
        "empty_state_category": _text(empty_reason.get("empty_state_category")),
        "dominant_rejection_reason": _text(empty_reason.get("dominant_rejection_reason")),
        "rejection_reason_counts": rejection_reason_counts,
        "diagnostic_category_counts": diagnostic_category_counts,
        "identities_blocked_only_by_incompatibility_count": len(
            _normalize_string_list(
                empty_reason.get("identities_blocked_only_by_incompatibility")
            )
        ),
        "strategies_without_analyzer_compatible_horizons_count": len(
            _normalize_string_list(
                empty_reason.get("strategies_without_analyzer_compatible_horizons")
            )
        ),
    }


def _build_threshold_context() -> dict[str, Any]:
    return {
        "source": "src.research.research_analyzer constants",
        "absolute_minimum_sample_count": MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
        "emerging_moderate_sample_count": EDGE_EARLY_MODERATE_SAMPLE_COUNT,
        "moderate_sample_count": EDGE_MODERATE_SAMPLE_COUNT,
        "emerging_moderate_median_return_pct": EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT,
        "moderate_median_return_pct": EDGE_MODERATE_MEDIAN_RETURN_PCT,
        "positive_rate_minimum_floor_pct": POSITIVE_RATE_MINIMUM_FLOOR_PCT,
        "emerging_moderate_positive_rate_pct": EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
        "three_supporting_deficits_min_positive_rate_pct": (
            THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT
        ),
        "emerging_moderate_robustness_pct": EDGE_EARLY_MODERATE_ROBUSTNESS_PCT,
        "three_supporting_deficits_min_robustness_pct": (
            THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT
        ),
        "moderate_min_aggregate_score": MODERATE_MIN_AGGREGATE_SCORE,
        "moderate_with_one_supporting_deficit_min_score": (
            MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE
        ),
        "moderate_with_two_supporting_deficit_min_score": (
            MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE
        ),
        "moderate_with_three_supporting_deficit_min_score": (
            MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE
        ),
        "diagnosis_only_margins": {
            "sample_floor_margin": DIAGNOSIS_SAMPLE_FLOOR_MARGIN,
            "sample_moderation_margin": DIAGNOSIS_SAMPLE_MODERATION_MARGIN,
            "positive_rate_margin_pct": DIAGNOSIS_POSITIVE_RATE_MARGIN_PCT,
            "aggregate_score_margin": DIAGNOSIS_AGGREGATE_SCORE_MARGIN,
        },
    }


def _collect_rejected_row_details(
    *,
    joined_block: dict[str, Any],
    threshold_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}

    for identity in _safe_dict_list(joined_block.get("identity_horizon_evaluations")):
        compatible_horizons = _sort_horizon_strings(
            _normalize_string_list(identity.get("strategy_compatible_horizons"))
        )
        for horizon, raw_evaluation in _safe_dict(identity.get("horizon_evaluations")).items():
            evaluation = _safe_dict(raw_evaluation)
            if _text(evaluation.get("status")) == "selected":
                continue
            diagnostics = _safe_dict(evaluation.get("candidate_strength_diagnostics"))
            metrics = _safe_dict(evaluation.get("metrics"))
            row = {
                "row_id": _row_id(
                    evaluation.get("symbol"),
                    evaluation.get("strategy"),
                    evaluation.get("horizon") or horizon,
                ),
                "symbol": _text(evaluation.get("symbol")),
                "strategy": _text(evaluation.get("strategy")),
                "horizon": _text(evaluation.get("horizon")) or _text(horizon),
                "source_kind": "identity_horizon_evaluation",
                "diagnostic_category": _diagnostic_category_from_fields(
                    _text(evaluation.get("rejection_reason")),
                    _text(evaluation.get("candidate_strength")),
                ),
                "strategy_horizon_compatible": _safe_bool(
                    evaluation.get("strategy_horizon_compatible")
                ),
                "rejection_reason": _text(evaluation.get("rejection_reason")),
                "rejection_reasons": _normalize_string_list(
                    evaluation.get("rejection_reasons")
                ),
                "candidate_strength": _text(evaluation.get("candidate_strength")),
                "classification_reason": _text(diagnostics.get("classification_reason")),
                "aggregate_score": _safe_float(
                    evaluation.get("aggregate_score", diagnostics.get("aggregate_score"))
                ),
                "sample_count": _safe_int(metrics.get("sample_count")),
                "labeled_count": _safe_int(metrics.get("labeled_count")),
                "median_future_return_pct": _safe_float(
                    metrics.get("median_future_return_pct")
                ),
                "positive_rate_pct": _safe_float(metrics.get("positive_rate_pct")),
                "robustness_signal_pct": _safe_float(
                    metrics.get("robustness_signal_pct")
                ),
                "analyzer_compatible_horizons": compatible_horizons,
            }
            _upsert_rejected_row(rows_by_key, row)

    for diagnostic in _safe_dict_list(joined_block.get("diagnostic_rows")):
        row = {
            "row_id": _row_id(
                diagnostic.get("symbol"),
                diagnostic.get("strategy"),
                diagnostic.get("horizon"),
            ),
            "symbol": _text(diagnostic.get("symbol")),
            "strategy": _text(diagnostic.get("strategy")),
            "horizon": _text(diagnostic.get("horizon")),
            "source_kind": "diagnostic_row",
            "diagnostic_category": _text(diagnostic.get("diagnostic_category"))
            or _diagnostic_category_from_fields(
                _text(diagnostic.get("rejection_reason")),
                _text(diagnostic.get("candidate_strength")),
            ),
            "strategy_horizon_compatible": _safe_bool(
                diagnostic.get("strategy_horizon_compatible")
            ),
            "rejection_reason": _text(diagnostic.get("rejection_reason")),
            "rejection_reasons": _normalize_string_list(diagnostic.get("rejection_reasons")),
            "candidate_strength": _text(diagnostic.get("candidate_strength")),
            "classification_reason": _text(diagnostic.get("classification_reason")),
            "aggregate_score": _safe_float(diagnostic.get("aggregate_score")),
            "sample_count": _safe_int(diagnostic.get("sample_count")),
            "labeled_count": _safe_int(diagnostic.get("labeled_count")),
            "median_future_return_pct": _safe_float(
                diagnostic.get("median_future_return_pct")
            ),
            "positive_rate_pct": _safe_float(diagnostic.get("positive_rate_pct")),
            "robustness_signal_pct": _safe_float(
                diagnostic.get("robustness_signal_pct")
            ),
            "analyzer_compatible_horizons": _sort_horizon_strings(
                _normalize_string_list(diagnostic.get("analyzer_compatible_horizons"))
            ),
        }
        _upsert_rejected_row(rows_by_key, row)

    rows = list(rows_by_key.values())
    for row in rows:
        bucket, reason = _diagnose_bucket(row, threshold_context)
        row["diagnosis_bucket"] = bucket
        row["diagnosis_bucket_reason"] = reason
        row["scenario_survival_flags"] = _build_scenario_flags(bucket)
    rows.sort(key=_row_sort_key)
    return rows


def _upsert_rejected_row(
    rows_by_key: dict[str, dict[str, Any]],
    incoming_row: dict[str, Any],
) -> None:
    row_id = _text(incoming_row.get("row_id")) or _row_id(
        incoming_row.get("symbol"),
        incoming_row.get("strategy"),
        incoming_row.get("horizon"),
    )
    incoming_row["row_id"] = row_id
    existing = rows_by_key.get(row_id)
    if existing is None:
        rows_by_key[row_id] = incoming_row
        return
    rows_by_key[row_id] = _merge_rejected_row(existing, incoming_row)


def _merge_rejected_row(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)

    merged["rejection_reasons"] = _merge_string_lists(
        existing.get("rejection_reasons"),
        incoming.get("rejection_reasons"),
    )
    merged["analyzer_compatible_horizons"] = _sort_horizon_strings(
        _merge_string_lists(
            existing.get("analyzer_compatible_horizons"),
            incoming.get("analyzer_compatible_horizons"),
        )
    )

    existing_kind = _text(existing.get("source_kind"))
    incoming_kind = _text(incoming.get("source_kind"))
    if existing_kind and incoming_kind and existing_kind != incoming_kind:
        merged["source_kind"] = "merged_identity_horizon_evaluation_and_diagnostic_row"

    for key, value in incoming.items():
        if key in {"row_id", "rejection_reasons", "analyzer_compatible_horizons"}:
            continue
        if _is_missing_value(merged.get(key)) and not _is_missing_value(value):
            merged[key] = value

    return merged


def _merge_string_lists(existing: Any, incoming: Any) -> list[str]:
    merged: list[str] = []
    for item in _normalize_string_list(existing):
        if item not in merged:
            merged.append(item)
    for item in _normalize_string_list(incoming):
        if item not in merged:
            merged.append(item)
    return merged


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def _diagnostic_category_from_fields(
    rejection_reason: str | None,
    candidate_strength: str | None,
) -> str:
    if rejection_reason == "strategy_horizon_incompatible":
        return "incompatibility"
    if rejection_reason in {
        "sample_count_zero",
        "no_labeled_rows_for_horizon",
        "missing_median_future_return",
        "failed_absolute_minimum_gate",
    }:
        return "insufficient_data"
    if candidate_strength == "weak" or rejection_reason == "candidate_strength_weak":
        return "quality_rejected"
    return "other_rejection"


def _diagnose_bucket(
    row: dict[str, Any],
    threshold_context: dict[str, Any],
) -> tuple[str, str]:
    if _is_incompatible_noise(row):
        return (
            UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE,
            "Unsupported strategy/horizon noise rather than analyzer-quality evidence.",
        )
    if _is_near_sample_floor(row, threshold_context):
        return (
            NEAR_MISS_SAMPLE_FLOOR,
            "Just below the absolute sample floor with label support and positive median still present.",
        )
    if _is_near_sample_moderation(row, threshold_context):
        return (
            NEAR_MISS_SAMPLE_MODERATION,
            "Misses the moderate-sample boundary by only a small margin.",
        )
    if _is_near_positive_rate(row, threshold_context):
        return (
            NEAR_MISS_POSITIVE_RATE,
            "Narrowly below a positive-rate threshold while other persisted metrics stay credible.",
        )
    if _is_near_other_narrow(row):
        return (
            NEAR_MISS_OTHER_NARROW,
            "Aggregate score is narrowly below the current threshold without broader structural weakness.",
        )
    return (
        STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING,
        "No conservative near-threshold rescue matched the persisted evidence.",
    )


def _is_incompatible_noise(row: dict[str, Any]) -> bool:
    return (
        row.get("strategy_horizon_compatible") is False
        or _text(row.get("rejection_reason")) == "strategy_horizon_incompatible"
        or _text(row.get("diagnostic_category")) == "incompatibility"
    )


def _is_near_sample_floor(
    row: dict[str, Any],
    threshold_context: dict[str, Any],
) -> bool:
    sample_count = _safe_float(row.get("sample_count"))
    labeled_count = _safe_float(row.get("labeled_count"))
    median_return = _safe_float(row.get("median_future_return_pct"))
    positive_rate = _safe_float(row.get("positive_rate_pct"))
    robustness = _safe_float(row.get("robustness_signal_pct"))
    absolute_floor = _safe_float(threshold_context.get("absolute_minimum_sample_count"))
    if _text(row.get("rejection_reason")) != "failed_absolute_minimum_gate":
        return False
    if sample_count is None or absolute_floor is None:
        return False
    if sample_count >= absolute_floor or sample_count < absolute_floor - DIAGNOSIS_SAMPLE_FLOOR_MARGIN:
        return False
    if labeled_count is None or labeled_count <= 0 or median_return is None or median_return <= 0:
        return False
    if positive_rate is not None and positive_rate < POSITIVE_RATE_MINIMUM_FLOOR_PCT:
        return False
    if robustness is not None and robustness < THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT:
        return False
    reasons = set(_normalize_string_list(row.get("rejection_reasons")))
    return (not reasons) or reasons.issubset(
        {"failed_absolute_minimum_gate", "sample_count_below_absolute_floor"}
    )


def _is_near_sample_moderation(
    row: dict[str, Any],
    threshold_context: dict[str, Any],
) -> bool:
    sample_count = _safe_float(row.get("sample_count"))
    moderate_floor = _safe_float(threshold_context.get("moderate_sample_count"))
    return bool(
        _text(row.get("rejection_reason")) == "candidate_strength_weak"
        and _text(row.get("classification_reason"))
        in {
            "two_supporting_deficits_but_sample_not_moderate",
            "three_supporting_deficits_but_sample_not_moderate",
        }
        and sample_count is not None
        and moderate_floor is not None
        and moderate_floor - DIAGNOSIS_SAMPLE_MODERATION_MARGIN <= sample_count < moderate_floor
    )


def _is_near_positive_rate(
    row: dict[str, Any],
    threshold_context: dict[str, Any],
) -> bool:
    reason = _text(row.get("classification_reason"))
    sample_count = _safe_float(row.get("sample_count"))
    positive_rate = _safe_float(row.get("positive_rate_pct"))
    moderate_sample = _safe_float(threshold_context.get("moderate_sample_count"))
    robustness = _safe_float(row.get("robustness_signal_pct"))
    if _text(row.get("rejection_reason")) != "candidate_strength_weak":
        return False
    if sample_count is None or positive_rate is None or moderate_sample is None:
        return False
    if reason == "two_supporting_deficits_but_positive_rate_below_floor":
        return (
            sample_count >= moderate_sample
            and POSITIVE_RATE_MINIMUM_FLOOR_PCT - DIAGNOSIS_POSITIVE_RATE_MARGIN_PCT
            <= positive_rate
            < POSITIVE_RATE_MINIMUM_FLOOR_PCT
        )
    if reason == "three_supporting_deficits_but_positive_rate_too_low":
        return (
            sample_count >= moderate_sample
            and THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT
            - DIAGNOSIS_POSITIVE_RATE_MARGIN_PCT
            <= positive_rate
            < THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT
            and (
                robustness is None
                or robustness >= THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT
            )
        )
    return False


def _is_near_other_narrow(row: dict[str, Any]) -> bool:
    threshold = AGGREGATE_THRESHOLD_BY_REASON.get(
        _text(row.get("classification_reason")) or ""
    )
    aggregate = _safe_float(row.get("aggregate_score"))
    sample_count = _safe_float(row.get("sample_count"))
    median_return = _safe_float(row.get("median_future_return_pct"))
    positive_rate = _safe_float(row.get("positive_rate_pct"))
    return bool(
        _text(row.get("rejection_reason")) == "candidate_strength_weak"
        and threshold is not None
        and aggregate is not None
        and threshold - DIAGNOSIS_AGGREGATE_SCORE_MARGIN <= aggregate < threshold
        and (sample_count is None or sample_count >= EDGE_EARLY_MODERATE_SAMPLE_COUNT)
        and (
            median_return is None
            or median_return >= EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT
        )
        and (
            positive_rate is None
            or positive_rate >= EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT
        )
    )


def _build_scenario_flags(bucket: str) -> dict[str, bool]:
    return {
        "baseline_current_rules": False,
        "baseline_excluding_incompatible_noise": (
            bucket != UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE
        ),
        "narrow_sample_floor_relief": bucket == NEAR_MISS_SAMPLE_FLOOR,
        "narrow_sample_moderation_relief": bucket == NEAR_MISS_SAMPLE_MODERATION,
        "narrow_positive_rate_near_threshold_relief": bucket == NEAR_MISS_POSITIVE_RATE,
        "combined_narrow_relief": bucket in NEAR_MISS_BUCKETS,
    }


def _build_rejected_row_diagnostics(
    rejected_rows: list[dict[str, Any]],
    joined: dict[str, Any],
) -> dict[str, Any]:
    bucket_counts = Counter(
        _text(row.get("diagnosis_bucket")) or STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING
        for row in rejected_rows
    )
    rejection_reason_counts = Counter(
        _text(row.get("rejection_reason")) or "unknown"
        for row in rejected_rows
    )
    classification_reason_counts = Counter(
        _text(row.get("classification_reason")) or "unavailable"
        for row in rejected_rows
    )
    diagnostic_category_counts = Counter(
        _text(row.get("diagnostic_category")) or "other_rejection"
        for row in rejected_rows
    )
    artifact_count = _safe_int(joined.get("rejected_joined_row_count"))
    return {
        "normalized_rejected_row_count": len(rejected_rows),
        "artifact_diagnostic_row_count": artifact_count,
        "diagnostic_row_count_gap_vs_artifact": (
            len(rejected_rows) - artifact_count if artifact_count is not None else None
        ),
        "diagnosis_bucket_counts": _ordered_counter(
            bucket_counts,
            DIAGNOSIS_BUCKET_ORDER,
        ),
        "rejection_reason_counts": _ordered_counter(rejection_reason_counts),
        "classification_reason_counts": _ordered_counter(classification_reason_counts),
        "diagnostic_category_counts": _ordered_counter(diagnostic_category_counts),
        "unsupported_strategy_or_horizon_noise_count": bucket_counts[
            UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE
        ],
        "near_miss_row_count": sum(
            bucket_counts[item]
            for item in DIAGNOSIS_BUCKET_ORDER
            if item in NEAR_MISS_BUCKETS
        ),
        "structurally_weak_or_still_unconvincing_count": bucket_counts[
            STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING
        ],
        "non_noise_rejected_row_count": len(rejected_rows)
        - bucket_counts[UNSUPPORTED_STRATEGY_OR_HORIZON_NOISE],
    }


def _build_counterfactual_scenarios(
    rejected_rows: list[dict[str, Any]],
    joined: dict[str, Any],
) -> dict[str, Any]:
    eligible = _safe_int(joined.get("eligible_joined_row_count")) or 0
    non_noise_rows = [
        row
        for row in rejected_rows
        if _safe_dict(row.get("scenario_survival_flags")).get(
            "baseline_excluding_incompatible_noise"
        )
    ]
    scenarios: dict[str, Any] = {}
    for name in SCENARIO_ORDER:
        rescued = []
        considered = rejected_rows if name == "baseline_current_rules" else non_noise_rows
        if name not in {"baseline_current_rules", "baseline_excluding_incompatible_noise"}:
            rescued = [
                row
                for row in rejected_rows
                if _safe_dict(row.get("scenario_survival_flags")).get(name)
            ]
        scenarios[name] = {
            "description": SCENARIO_DESCRIPTIONS[name],
            "diagnosis_only": True,
            "considered_rejected_row_count": len(considered),
            "excluded_incompatible_noise_count": len(rejected_rows) - len(considered),
            "rescued_rejected_row_count": len(rescued),
            "rescued_rejected_row_share_of_all_rejected": _ratio(
                len(rescued),
                len(rejected_rows),
            ),
            "rescued_rejected_row_share_of_non_noise_rejected": _ratio(
                len(rescued),
                len(non_noise_rows),
            ),
            "eligible_joined_row_count_after_scenario": eligible + len(rescued),
            "quality_profile": _build_quality_profile(rescued),
        }
    return scenarios


def _build_quality_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "bucket_counts": _ordered_counter(
            Counter(
                _text(row.get("diagnosis_bucket"))
                or STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING
                for row in rows
            ),
            DIAGNOSIS_BUCKET_ORDER,
        ),
        "sample_count_median": _median_from_rows(rows, "sample_count"),
        "aggregate_score_median": _median_from_rows(rows, "aggregate_score"),
        "median_future_return_pct_median": _median_from_rows(
            rows,
            "median_future_return_pct",
        ),
        "positive_rate_pct_median": _median_from_rows(rows, "positive_rate_pct"),
        "rows_with_positive_median_return_count": sum(
            1
            for row in rows
            if (_safe_float(row.get("median_future_return_pct")) or 0.0) > 0.0
        ),
        "rows_at_or_above_coinflip_positive_rate_count": sum(
            1
            for row in rows
            if (_safe_float(row.get("positive_rate_pct")) or -1.0) >= 50.0
        ),
    }


def _build_final_assessment(
    *,
    loaded: dict[str, Any],
    preview: dict[str, Any],
    joined: dict[str, Any],
    rejected: dict[str, Any],
    scenarios: dict[str, Any],
) -> dict[str, Any]:
    sufficient = bool(
        loaded["payload_is_object"]
        and preview.get("preview_block_exists")
        and joined.get("joined_row_block_exists")
    )
    facts = []
    inference_notes = []
    uncertainty_notes = [
        "Heuristic only; this report does not recommend production threshold changes.",
        "Trade-analysis JSONL, mapper, and engine are outside this report scope.",
    ]
    if not loaded["exists"]:
        return {
            "artifact_sufficient_for_counterfactual_diagnosis": False,
            "assessment": "mixed_or_inconclusive",
            "facts": ["Analyzer summary artifact is missing."],
            "inference_notes": [],
            "uncertainty_notes": uncertainty_notes,
        }
    if not loaded["json_loaded"] or not loaded["payload_is_object"]:
        return {
            "artifact_sufficient_for_counterfactual_diagnosis": False,
            "assessment": "mixed_or_inconclusive",
            "facts": [f"Summary artifact could not be loaded cleanly: {loaded['load_error']}."],
            "inference_notes": [],
            "uncertainty_notes": uncertainty_notes,
        }
    if not sufficient:
        return {
            "artifact_sufficient_for_counterfactual_diagnosis": False,
            "assessment": "mixed_or_inconclusive",
            "facts": ["Expected preview/joined-row blocks are missing from the artifact."],
            "inference_notes": [],
            "uncertainty_notes": uncertainty_notes,
        }

    combined = _safe_dict(scenarios.get("combined_narrow_relief"))
    combined_quality = _safe_dict(combined.get("quality_profile"))
    combined_bucket_counts = _safe_dict(combined_quality.get("bucket_counts"))
    rejected_count = _safe_int(rejected.get("normalized_rejected_row_count")) or 0
    non_noise = _safe_int(rejected.get("non_noise_rejected_row_count")) or 0
    rescue_count = _safe_int(combined.get("rescued_rejected_row_count")) or 0
    rescue_ratio = _safe_float(
        combined.get("rescued_rejected_row_share_of_non_noise_rejected")
    )
    non_floor_rescues = (
        _safe_int(combined_bucket_counts.get(NEAR_MISS_SAMPLE_MODERATION)) or 0
    ) + (_safe_int(combined_bucket_counts.get(NEAR_MISS_POSITIVE_RATE)) or 0) + (
        _safe_int(combined_bucket_counts.get(NEAR_MISS_OTHER_NARROW)) or 0
    )

    facts.append(
        f"Joined-row counts: eligible={joined.get('eligible_joined_row_count')}, rejected={rejected_count}."
    )
    facts.append(
        "Diagnosis bucket counts: "
        f"{json.dumps(rejected.get('diagnosis_bucket_counts', {}), ensure_ascii=False)}."
    )
    facts.append(f"Combined narrow relief rescues {rescue_count} rejected rows.")

    if rejected_count == 0:
        assessment = "mixed_or_inconclusive"
        inference_notes.append(
            "No rejected joined rows were available for strictness analysis."
        )
    elif non_noise == 0 or rescue_count == 0:
        assessment = "strict_but_probably_justified"
        inference_notes.append(
            "The rejected set is dominated by unsupported noise or rows that do not survive any conservative rescue."
        )
    elif (
        rescue_count >= 3
        and rescue_ratio is not None
        and rescue_ratio >= 0.35
        and non_floor_rescues >= 2
        and _safe_int(combined_quality.get("rows_with_positive_median_return_count"))
        == rescue_count
    ):
        assessment = "potential_lossiness_detected"
        inference_notes.append(
            "A material share of non-noise rejected rows survives only under narrow counterfactual relief."
        )
    elif rescue_ratio is not None and rescue_ratio >= 0.15:
        assessment = "mixed_or_inconclusive"
        inference_notes.append(
            "Some near-miss evidence exists, but it is not broad enough to call confirmed lossiness."
        )
    else:
        assessment = "strict_but_probably_justified"
        inference_notes.append(
            "Recovered rows are too few relative to the remaining non-noise rejected set."
        )

    return {
        "artifact_sufficient_for_counterfactual_diagnosis": sufficient,
        "assessment": assessment,
        "combined_narrow_relief_rescued_rejected_row_count": rescue_count,
        "combined_narrow_relief_rescue_share_of_non_noise_rejected": rescue_ratio,
        "facts": facts,
        "inference_notes": inference_notes,
        "uncertainty_notes": uncertainty_notes,
    }


def _row_id(symbol: Any, strategy: Any, horizon: Any) -> str:
    return ":".join(
        [
            _text(symbol) or "unknown_symbol",
            _text(strategy) or "unknown_strategy",
            _text(horizon) or "unknown_horizon",
        ]
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str, str]:
    bucket = _text(row.get("diagnosis_bucket")) or STRUCTURALLY_WEAK_OR_STILL_UNCONVINCING
    horizon = _text(row.get("horizon")) or ""
    return (
        DIAGNOSIS_BUCKET_ORDER.index(bucket)
        if bucket in DIAGNOSIS_BUCKET_ORDER
        else len(DIAGNOSIS_BUCKET_ORDER),
        _KNOWN_HORIZON_INDEX.get(horizon, len(_KNOWN_HORIZON_INDEX)),
        _text(row.get("symbol")) or "",
        _text(row.get("strategy")) or "",
        _text(row.get("rejection_reason")) or "",
    )


def _median_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return round(float(median(values)), 4) if values else None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
        return int(numeric) if numeric.is_integer() else None
    return None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_string_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else []
    normalized: list[str] = []
    for item in items:
        text = _text(item)
        if text is not None and text not in normalized:
            normalized.append(text)
    return normalized


def _sort_horizon_strings(values: list[str]) -> list[str]:
    return sorted(
        _normalize_string_list(values),
        key=lambda item: (
            _KNOWN_HORIZON_INDEX.get(item, len(_KNOWN_HORIZON_INDEX)),
            item,
        ),
    )


def _safe_counter_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    parsed = {
        _text(key): _safe_int(raw)
        for key, raw in value.items()
        if _text(key) is not None and _safe_int(raw) is not None
    }
    return _ordered_counter(parsed)


def _ordered_counter(
    counter_like: Counter[str] | dict[str, int],
    preferred: tuple[str, ...] | None = None,
) -> dict[str, int]:
    counter = dict(counter_like)
    ordered: dict[str, int] = {}
    if preferred is not None:
        for key in preferred:
            value = counter.pop(key, 0)
            if value:
                ordered[key] = value
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if value:
            ordered[key] = value
    return ordered


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator > 0 else None