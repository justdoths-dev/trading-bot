from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_fully_aligned_final_hold_split_diagnosis_report as final_split_module,
)
from src.research.diagnostics import (
    selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report as threshold_module,
)

REPORT_TYPE = "selected_strategy_rule_engine_confidence_contributor_diagnosis_report"
REPORT_TITLE = "Selected Strategy Rule Engine Confidence Contributor Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = final_split_module.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = final_split_module.DEFAULT_OUTPUT_DIR
DEFAULT_MIN_SYMBOL_SUPPORT = final_split_module.DEFAULT_MIN_SYMBOL_SUPPORT

DiagnosisConfiguration = final_split_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = final_split_module.DEFAULT_CONFIGURATIONS

_COMPARISON_GROUP_PRESERVED = final_split_module._COMPARISON_GROUP_PRESERVED
_COMPARISON_GROUP_COLLAPSED = final_split_module._COMPARISON_GROUP_COLLAPSED

_RULE_ENGINE_CONFIDENCE_FIELD = "rule_engine_confidence"
_CONTRIBUTOR_FIELD_NAMES = (
    "setup_layer_confidence",
    "context_layer_confidence",
    "bias_layer_confidence",
    "selected_strategy_confidence",
    "trigger_layer_confidence",
)
_CONTRIBUTOR_FIELD_SPECS = tuple(
    spec
    for spec in final_split_module._NUMERIC_FIELD_SPECS
    if spec[0] in _CONTRIBUTOR_FIELD_NAMES
)
_CONTRIBUTOR_FIELD_ORDER = {
    name: index for index, (name, _, _) in enumerate(_CONTRIBUTOR_FIELD_SPECS)
}
_CONTRIBUTOR_ROLE_MAP = {
    "setup_layer_confidence": "setup contributor",
    "context_layer_confidence": "context contributor",
    "bias_layer_confidence": "bias contributor",
    "selected_strategy_confidence": "selected-strategy contributor",
    "trigger_layer_confidence": "trigger negative control",
}
_NON_TRIGGER_CONTRIBUTOR_FAMILY_MAP = {
    "setup_layer_confidence": "setup",
    "context_layer_confidence": "context_and_bias",
    "bias_layer_confidence": "context_and_bias",
    "selected_strategy_confidence": "selected_strategy",
}
_NON_TRIGGER_CONTRIBUTOR_FIELDS = {
    "setup_layer_confidence",
    "context_layer_confidence",
    "bias_layer_confidence",
    "selected_strategy_confidence",
}
_TRACKING_ALIGNMENT_ORDER = {
    "clean_group_separation": 0,
    "group_gap_present_with_overlap": 1,
    "missingness_only": 2,
    "mixed_or_inverse": 3,
    "insufficient_data": 4,
}
_CONTRIBUTOR_COMPARISON_STATUS_ORDER = {
    "higher_on_preserved": 0,
    "no_clear_separation": 1,
    "higher_on_collapsed": 2,
    "missing_on_collapsed_only": 3,
    "missing_on_preserved_only": 4,
    "all_missing": 5,
}
_RANGE_OVERLAP_ORDER = {
    "collapsed_below_preserved": 0,
    "overlapping": 1,
    "preserved_below_collapsed": 2,
    "incomplete": 3,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the final fully aligned, rule-bias-aligned "
            "slice and determine whether low rule_engine_confidence in collapsed rows "
            "is already explained by persisted contributor weakness or still suggests "
            "an additional hidden aggregate or penalty rule."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Window/max_rows pair in the form WINDOW_HOURS/MAX_ROWS. Repeatable.",
    )
    parser.add_argument(
        "--min-symbol-support",
        type=int,
        default=DEFAULT_MIN_SYMBOL_SUPPORT,
        help=(
            "Retained for architectural parity with sibling reports. "
            "Contributor comparison itself uses the final preserved-vs-collapsed slice."
        ),
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = _resolve_path(args.input)
    output_dir = _resolve_path(args.output_dir)
    configurations = _parse_configuration_values(args.config)

    result = run_selected_strategy_rule_engine_confidence_contributor_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_assessment = _safe_dict(report.get("final_assessment"))
    strongest_contributor = _safe_dict(
        _safe_dict(report.get("contributor_tracking")).get(
            "strongest_tracking_contributor"
        )
    )
    final_interpretation = _safe_dict(report.get("final_interpretation"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    final_assessment.get("widest_configuration")
                ).get("display_name"),
                "final_rule_bias_aligned_row_count": summary.get(
                    "final_rule_bias_aligned_row_count",
                    0,
                ),
                "preserved_final_directional_outcome_row_count": summary.get(
                    "preserved_final_directional_outcome_row_count",
                    0,
                ),
                "collapsed_final_hold_outcome_row_count": summary.get(
                    "collapsed_final_hold_outcome_row_count",
                    0,
                ),
                "comparison_support_status": summary.get(
                    "comparison_support_status",
                    "unknown",
                ),
                "strongest_contributor_field": strongest_contributor.get("field"),
                "interpretation_status": final_interpretation.get(
                    "interpretation_status"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_contributor_diagnosis_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
    min_symbol_support: int = DEFAULT_MIN_SYMBOL_SUPPORT,
    write_report_copies: bool = False,
) -> dict[str, Any]:
    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=min_symbol_support,
    )
    written_paths: dict[str, str] = {}
    if write_report_copies:
        written_paths = write_report_files(report, output_dir)
    return {
        "input_path": report["input_path"],
        "output_dir": report["output_dir"],
        "written_paths": written_paths,
        "report": report,
        "markdown": render_markdown(report),
    }


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
    min_symbol_support: int = DEFAULT_MIN_SYMBOL_SUPPORT,
) -> dict[str, Any]:
    resolved_input = _resolve_path(input_path)
    resolved_output = _resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)
    effective_min_symbol_support = max(1, int(min_symbol_support))

    configuration_summaries: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        run_output_dir = resolved_output / f"_{REPORT_TYPE}" / configuration.slug
        effective_input_path, raw_records, source_metadata = (
            _materialize_configuration_input(
                input_path=resolved_input,
                run_output_dir=run_output_dir,
                latest_window_hours=configuration.latest_window_hours,
                latest_max_rows=configuration.latest_max_rows,
            )
        )
        configuration_summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                effective_input_path=effective_input_path,
                run_output_dir=run_output_dir,
                raw_records=raw_records,
                source_metadata=source_metadata,
                min_symbol_support=effective_min_symbol_support,
            )
        )

    widest_summary = _widest_configuration_summary(configuration_summaries)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output),
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
        },
        "min_symbol_support": effective_min_symbol_support,
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary.get("headline")) for summary in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": _safe_dict(widest_summary.get("summary")),
        "rule_engine_confidence_context": _safe_dict(
            widest_summary.get("rule_engine_confidence_context")
        ),
        "contributor_comparison": _safe_dict(
            widest_summary.get("contributor_comparison")
        ),
        "contributor_tracking": _safe_dict(widest_summary.get("contributor_tracking")),
        "tertiary_reason_context": _safe_dict(
            widest_summary.get("tertiary_reason_context")
        ),
        "final_interpretation": _safe_dict(widest_summary.get("final_interpretation")),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned slice definition from the neighboring final hold split report.",
            "Only already-persisted confidence-like contributor fields are inspected here; no production decision logic, mapper logic, or execution gate behavior is modified.",
            "Rule_engine_confidence is the target metric being explained, while trigger_layer_confidence is carried as a negative control rather than being promoted automatically into the main cause.",
            "Pairwise preserved-vs-collapsed ordering and row-level correlation to rule_engine_confidence are descriptive helpers only and are not causal proof.",
            "Combined reason bucket remains tertiary context only and is surfaced as supporting evidence rather than as a primary differentiator.",
        ],
    }


def build_configuration_summary(
    *,
    configuration: DiagnosisConfiguration,
    input_path: Path,
    effective_input_path: Path,
    run_output_dir: Path,
    raw_records: Sequence[dict[str, Any]],
    source_metadata: dict[str, Any],
    min_symbol_support: int,
) -> dict[str, Any]:
    stage_rows = [
        _build_stage_row(raw_record)
        for raw_record in raw_records
        if isinstance(raw_record, dict)
    ]
    actionable_rows = [
        _build_activation_gap_row(row)
        for row in stage_rows
        if row.get("selected_strategy_result_signal_state")
        in final_split_module.fully_aligned_module._ACTIONABLE_SIGNAL_STATES
    ]
    fully_aligned_rows = [
        fully_aligned_row
        for fully_aligned_row in (
            _build_fully_aligned_row(row) for row in actionable_rows
        )
        if fully_aligned_row is not None
    ]
    final_split_rows = [
        final_split_row
        for final_split_row in (
            _build_final_split_row(row) for row in fully_aligned_rows
        )
        if final_split_row is not None
    ]
    comparison_rows = [
        row
        for row in final_split_rows
        if row.get("comparison_group")
        in {
            _COMPARISON_GROUP_PRESERVED,
            _COMPARISON_GROUP_COLLAPSED,
        }
    ]

    summary = final_split_module.build_summary(
        actionable_rows=actionable_rows,
        fully_aligned_rows=fully_aligned_rows,
        final_split_rows=final_split_rows,
        min_symbol_support=min_symbol_support,
    )
    rule_engine_confidence_context = build_rule_engine_confidence_context(
        comparison_rows=comparison_rows
    )
    contributor_comparison = build_contributor_comparison(
        comparison_rows=comparison_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    contributor_tracking = build_contributor_tracking(
        comparison_rows=comparison_rows,
        contributor_comparison=contributor_comparison,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    tertiary_reason_context = build_tertiary_reason_context(
        comparison_rows=comparison_rows
    )
    final_interpretation = build_final_interpretation(
        summary=summary,
        contributor_tracking=contributor_tracking,
    )
    key_observations = build_key_observations(
        summary=summary,
        rule_engine_confidence_context=rule_engine_confidence_context,
        contributor_tracking=contributor_tracking,
        final_interpretation=final_interpretation,
        tertiary_reason_context=tertiary_reason_context,
    )

    strongest_contributor = _safe_dict(
        contributor_tracking.get("strongest_tracking_contributor")
    )
    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "effective_input_path": str(effective_input_path),
        "run_output_dir": str(run_output_dir),
        "source_metadata": {
            "input_path": source_metadata.get("input_path", str(input_path)),
            "rotation_aware": bool(source_metadata.get("rotation_aware", False)),
            "source_files": _safe_list(source_metadata.get("source_files")),
            "source_file_count": int(source_metadata.get("source_file_count", 0) or 0),
            "raw_record_count": int(
                source_metadata.get("raw_record_count", len(raw_records)) or 0
            ),
            "windowed_record_count": int(
                source_metadata.get("windowed_record_count", len(raw_records)) or 0
            ),
            "effective_input_path": source_metadata.get(
                "effective_input_path",
                str(effective_input_path),
            ),
            "effective_input_record_count": int(
                source_metadata.get("effective_input_record_count", len(raw_records))
                or 0
            ),
            "effective_input_materialized": bool(
                source_metadata.get("effective_input_materialized", True)
            ),
        },
        "headline": {
            "display_name": configuration.display_name,
            "latest_window_hours": configuration.latest_window_hours,
            "latest_max_rows": configuration.latest_max_rows,
            "actionable_selected_strategy_row_count": summary[
                "actionable_selected_strategy_row_count"
            ],
            "fully_aligned_row_count": summary["fully_aligned_row_count"],
            "final_rule_bias_aligned_row_count": summary[
                "final_rule_bias_aligned_row_count"
            ],
            "preserved_final_directional_outcome_row_count": summary[
                "preserved_final_directional_outcome_row_count"
            ],
            "collapsed_final_hold_outcome_row_count": summary[
                "collapsed_final_hold_outcome_row_count"
            ],
            "comparison_support_status": summary["comparison_support_status"],
            "strongest_contributor_field": strongest_contributor.get("field"),
            "interpretation_status": final_interpretation.get("interpretation_status"),
            "trigger_negative_control_status": contributor_tracking.get(
                "trigger_negative_control_status"
            ),
        },
        "summary": summary,
        "rule_engine_confidence_context": rule_engine_confidence_context,
        "contributor_comparison": contributor_comparison,
        "contributor_tracking": contributor_tracking,
        "tertiary_reason_context": tertiary_reason_context,
        "final_interpretation": final_interpretation,
        "key_observations": key_observations,
    }


def build_rule_engine_confidence_context(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    distribution = threshold_module.build_rule_engine_confidence_distribution(
        comparison_rows=comparison_rows
    )
    range_overlap = threshold_module.build_rule_engine_confidence_overlap(
        distribution=distribution
    )
    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "support_status": distribution.get("support_status", "limited_support"),
        "distribution": distribution,
        "range_overlap": range_overlap,
        "median_difference_preserved_minus_collapsed": distribution.get(
            "median_difference_preserved_minus_collapsed"
        ),
        "mean_difference_preserved_minus_collapsed": distribution.get(
            "mean_difference_preserved_minus_collapsed"
        ),
    }


def build_contributor_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    comparison = threshold_module.build_numeric_field_comparison_for_fields(
        comparison_rows=comparison_rows,
        field_specs=_CONTRIBUTOR_FIELD_SPECS,
    )
    rule_engine_median_gap = _safe_abs_float(
        rule_engine_confidence_context.get("median_difference_preserved_minus_collapsed")
    )

    field_rows: list[dict[str, Any]] = []
    for base_row in _safe_list(comparison.get("field_comparisons")):
        row = _safe_dict(base_row)
        field = str(row.get("field") or "")
        preserved_summary = _safe_dict(row.get(_COMPARISON_GROUP_PRESERVED))
        collapsed_summary = _safe_dict(row.get(_COMPARISON_GROUP_COLLAPSED))
        median_gap = _safe_abs_float(
            row.get("median_difference_preserved_minus_collapsed")
        )
        range_overlap = _numeric_range_overlap(
            preserved_summary=preserved_summary,
            collapsed_summary=collapsed_summary,
        )
        field_rows.append(
            {
                **row,
                "contributor_role": _CONTRIBUTOR_ROLE_MAP.get(field, field),
                "range_overlap": range_overlap,
                "group_gap_summary": {
                    "median_difference_preserved_minus_collapsed": row.get(
                        "median_difference_preserved_minus_collapsed"
                    ),
                    "mean_difference_preserved_minus_collapsed": row.get(
                        "mean_difference_preserved_minus_collapsed"
                    ),
                    "median_gap_share_of_rule_engine_confidence": _gap_share(
                        contributor_gap=median_gap,
                        rule_engine_gap=rule_engine_median_gap,
                    ),
                },
            }
        )

    field_rows.sort(key=_contributor_comparison_sort_key)
    leading_contributor = _safe_dict(field_rows[0] if field_rows else {})
    return {
        "support_status": comparison.get("support_status", "limited_support"),
        "preserved_final_directional_outcome_row_count": comparison.get(
            "preserved_final_directional_outcome_row_count",
            0,
        ),
        "collapsed_final_hold_outcome_row_count": comparison.get(
            "collapsed_final_hold_outcome_row_count",
            0,
        ),
        "comparison_sort_method": (
            "Fields are ordered by comparison_status, then by range_overlap order, "
            "then by absolute preserved-vs-collapsed median gap."
        ),
        "field_comparisons": field_rows,
        "leading_contributor_differentiator": leading_contributor,
        "non_differentiating_fields": _safe_list(
            comparison.get("non_differentiating_fields")
        ),
        "unresolved_fields": _safe_list(comparison.get("unresolved_fields")),
    }


def build_contributor_tracking(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    contributor_comparison: dict[str, Any],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    rule_engine_reference = _safe_dict(
        rule_engine_confidence_context.get("distribution")
    )
    rule_engine_range_overlap = _safe_dict(
        rule_engine_confidence_context.get("range_overlap")
    )
    rule_engine_median_gap = _safe_abs_float(
        rule_engine_confidence_context.get("median_difference_preserved_minus_collapsed")
    )

    ranking_rows: list[dict[str, Any]] = []
    for base_row in _safe_list(contributor_comparison.get("field_comparisons")):
        row = _safe_dict(base_row)
        field = str(row.get("field") or "")
        pairwise_group_ordering = _pairwise_group_ordering(
            comparison_rows=comparison_rows,
            field=field,
        )
        rule_engine_correlation = _correlation_with_rule_engine(
            comparison_rows=comparison_rows,
            contributor_field=field,
        )
        ranking_rows.append(
            {
                "field": field,
                "field_label": row.get("field_label"),
                "contributor_role": row.get("contributor_role"),
                "comparison_status": row.get("comparison_status"),
                "tracking_alignment": _tracking_alignment(row),
                "median_difference_preserved_minus_collapsed": row.get(
                    "median_difference_preserved_minus_collapsed"
                ),
                "median_gap_share_of_rule_engine_confidence": _gap_share(
                    contributor_gap=_safe_abs_float(
                        row.get("median_difference_preserved_minus_collapsed")
                    ),
                    rule_engine_gap=rule_engine_median_gap,
                ),
                "range_overlap": _safe_dict(row.get("range_overlap")),
                "pairwise_group_ordering": pairwise_group_ordering,
                "rule_engine_correlation": rule_engine_correlation,
            }
        )

    ranking_rows.sort(key=_tracking_sort_key)
    strongest_tracking_contributor = _safe_dict(ranking_rows[0] if ranking_rows else {})

    return {
        "support_status": contributor_comparison.get("support_status", "limited_support"),
        "tracking_method": (
            "Contributors are ordered by preserved-vs-collapsed group direction first, "
            "then by pairwise preserved>collapsed rate, row-level correlation to "
            "rule_engine_confidence, and contributor median-gap share of the "
            "rule_engine_confidence median gap."
        ),
        "rule_engine_reference": {
            "field": _RULE_ENGINE_CONFIDENCE_FIELD,
            "comparison_status": rule_engine_reference.get("comparison_status"),
            "median_difference_preserved_minus_collapsed": rule_engine_reference.get(
                "median_difference_preserved_minus_collapsed"
            ),
            "range_order": rule_engine_range_overlap.get("range_order"),
            "ranges_overlap": rule_engine_range_overlap.get("ranges_overlap"),
        },
        "ranked_contributors": ranking_rows,
        "strongest_tracking_contributor": strongest_tracking_contributor,
        "trigger_negative_control_status": _trigger_negative_control_status(
            ranking_rows=ranking_rows
        ),
        "ordered_contributor_fields": [row.get("field") for row in ranking_rows],
    }


def build_tertiary_reason_context(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    comparison = final_split_module.build_tertiary_reason_comparison(
        comparison_rows=comparison_rows
    )
    return {
        **comparison,
        "contextual_only": True,
        "used_for_contributor_interpretation_only": True,
        "note": (
            "Combined reason bucket remains tertiary context only and is not treated "
            "as stronger evidence than the persisted contributor confidences."
        ),
    }


def build_final_interpretation(
    *,
    summary: dict[str, Any],
    contributor_tracking: dict[str, Any],
) -> dict[str, Any]:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    ranking_rows = [
        _safe_dict(row)
        for row in contributor_tracking.get("ranked_contributors", [])
    ]
    trigger_negative_control_status = str(
        contributor_tracking.get("trigger_negative_control_status")
        or "trigger_negative_control_unknown"
    )

    if final_slice_row_count <= 0:
        return {
            "support_status": "unsupported",
            "interpretation_status": "comparison_unsupported",
            "strong_aligned_non_trigger_contributors": [],
            "aligned_non_trigger_contributors": [],
            "strong_aligned_non_trigger_contributor_families": [],
            "aligned_non_trigger_contributor_families": [],
            "weak_or_mixed_contributors": list(_CONTRIBUTOR_FIELD_NAMES),
            "trigger_negative_control_status": trigger_negative_control_status,
            "explanation": (
                "No rows reached the final fully aligned plus rule-bias-aligned slice, "
                "so contributor diagnosis is unsupported."
            ),
        }
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return {
            "support_status": "unsupported",
            "interpretation_status": "comparison_unsupported",
            "strong_aligned_non_trigger_contributors": [],
            "aligned_non_trigger_contributors": [],
            "strong_aligned_non_trigger_contributor_families": [],
            "aligned_non_trigger_contributor_families": [],
            "weak_or_mixed_contributors": list(_CONTRIBUTOR_FIELD_NAMES),
            "trigger_negative_control_status": trigger_negative_control_status,
            "explanation": (
                "The final fully aligned plus rule-bias-aligned slice exists, but one "
                "side of the preserved-vs-collapsed comparison is missing, so the "
                "current run cannot safely judge whether persisted contributor weakness "
                "is sufficient on its own."
            ),
        }

    aligned_non_trigger_contributors = [
        row["field"]
        for row in ranking_rows
        if row.get("field") in _NON_TRIGGER_CONTRIBUTOR_FIELDS
        and row.get("tracking_alignment")
        in {"clean_group_separation", "group_gap_present_with_overlap"}
    ]
    strong_aligned_non_trigger_contributors = [
        row["field"]
        for row in ranking_rows
        if row.get("field") in _NON_TRIGGER_CONTRIBUTOR_FIELDS
        and row.get("tracking_alignment") == "clean_group_separation"
    ]
    aligned_non_trigger_contributor_families = _non_trigger_contributor_families(
        aligned_non_trigger_contributors
    )
    strong_aligned_non_trigger_contributor_families = _non_trigger_contributor_families(
        strong_aligned_non_trigger_contributors
    )
    weak_or_mixed_contributors = [
        row["field"]
        for row in ranking_rows
        if row.get("tracking_alignment")
        not in {"clean_group_separation", "group_gap_present_with_overlap"}
    ]

    interpretation_status = (
        "persisted_surface_still_suggests_hidden_aggregate_or_penalty_rule"
    )
    explanation = (
        "The persisted non-trigger contributor surface remains too mixed, missing, or "
        "weak relative to the preserved-vs-collapsed rule_engine_confidence split, so "
        "the current surface still suggests an additional hidden aggregate or penalty rule."
    )

    has_strong_context_bias_family = (
        "context_and_bias" in strong_aligned_non_trigger_contributor_families
    )
    has_strong_independent_family = bool(
        {"setup", "selected_strategy"}.intersection(
            strong_aligned_non_trigger_contributor_families
        )
    )
    has_aligned_context_bias_family = (
        "context_and_bias" in aligned_non_trigger_contributor_families
    )
    has_aligned_independent_family = bool(
        {"setup", "selected_strategy"}.intersection(
            aligned_non_trigger_contributor_families
        )
    )

    if (
        trigger_negative_control_status == "trigger_remains_negative_control"
        and has_strong_context_bias_family
        and has_strong_independent_family
    ):
        interpretation_status = "persisted_contributors_appear_sufficient"
        explanation = (
            "The persisted surface shows clean separation for the context/bias family "
            "and for at least one additional independent non-trigger family while "
            "trigger still behaves like a negative control, so the visible contributor "
            "structure appears sufficient to explain most of the low "
            "rule_engine_confidence seen in collapsed rows."
        )
    elif (
        trigger_negative_control_status == "trigger_remains_negative_control"
        and has_aligned_context_bias_family
        and has_aligned_independent_family
    ):
        interpretation_status = "mixed_persisted_contributor_surface"
        explanation = (
            "Persisted contributor weakness spans both the context/bias family and at "
            "least one additional independent non-trigger family, but at least one of "
            "those families still overlaps rather than separating cleanly, so a hidden "
            "aggregate or penalty rule remains plausible."
        )
    elif aligned_non_trigger_contributors:
        interpretation_status = "mixed_persisted_contributor_surface"
        explanation = (
            "Persisted contributor weakness is visible in part of the final slice, but "
            "the aligned evidence does not yet span both the context/bias family and a "
            "second independent non-trigger family with enough clarity to treat the "
            "current surface as sufficient on its own, so an additional hidden "
            "aggregate or penalty rule remains plausible."
        )

    return {
        "support_status": summary.get("comparison_support_status", "limited_support"),
        "interpretation_status": interpretation_status,
        "strong_aligned_non_trigger_contributors": strong_aligned_non_trigger_contributors,
        "aligned_non_trigger_contributors": aligned_non_trigger_contributors,
        "strong_aligned_non_trigger_contributor_families": (
            strong_aligned_non_trigger_contributor_families
        ),
        "aligned_non_trigger_contributor_families": (
            aligned_non_trigger_contributor_families
        ),
        "weak_or_mixed_contributors": weak_or_mixed_contributors,
        "trigger_negative_control_status": trigger_negative_control_status,
        "explanation": explanation,
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    rule_engine_confidence_context: dict[str, Any],
    contributor_tracking: dict[str, Any],
    final_interpretation: dict[str, Any],
    tertiary_reason_context: dict[str, Any],
) -> dict[str, list[str]]:
    rule_engine_distribution = _safe_dict(
        rule_engine_confidence_context.get("distribution")
    )
    rule_engine_overlap = _safe_dict(rule_engine_confidence_context.get("range_overlap"))
    preserved_rule_engine = _safe_dict(
        rule_engine_distribution.get(_COMPARISON_GROUP_PRESERVED)
    )
    collapsed_rule_engine = _safe_dict(
        rule_engine_distribution.get(_COMPARISON_GROUP_COLLAPSED)
    )
    strongest_contributor = _safe_dict(
        contributor_tracking.get("strongest_tracking_contributor")
    )

    facts = [
        (
            "Actionable selected-strategy rows="
            f"{summary.get('actionable_selected_strategy_row_count', 0)}; "
            f"fully_aligned_row_count={summary.get('fully_aligned_row_count', 0)}; "
            "final_rule_bias_aligned_row_count="
            f"{summary.get('final_rule_bias_aligned_row_count', 0)}."
        ),
        (
            "Final rule-bias-aligned comparison groups: "
            "preserved_final_directional_outcome="
            f"{summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            "collapsed_final_hold_outcome="
            f"{summary.get('collapsed_final_hold_outcome_row_count', 0)}, "
            "comparison_support_status="
            f"{summary.get('comparison_support_status', 'unknown')}."
        ),
        (
            "Rule_engine_confidence preserved median="
            f"{preserved_rule_engine.get('median')}, collapsed median="
            f"{collapsed_rule_engine.get('median')}, range_order="
            f"{rule_engine_overlap.get('range_order')}."
        ),
    ]

    if strongest_contributor:
        pairwise_group_ordering = _safe_dict(
            strongest_contributor.get("pairwise_group_ordering")
        )
        rule_engine_correlation = _safe_dict(
            strongest_contributor.get("rule_engine_correlation")
        )
        facts.append(
            "Strongest contributor field="
            f"{strongest_contributor.get('field')}; tracking_alignment="
            f"{strongest_contributor.get('tracking_alignment')}; preserved_greater_pair_rate="
            f"{pairwise_group_ordering.get('preserved_greater_rate')}; "
            "rule_engine_correlation="
            f"{rule_engine_correlation.get('pearson_correlation_with_rule_engine_confidence')}."
        )

    facts.append(
        "Trigger negative control status="
        f"{contributor_tracking.get('trigger_negative_control_status', 'unknown')}."
    )
    facts.append(str(tertiary_reason_context.get("note") or ""))

    inferences = [
        (
            "Final interpretation: "
            f"{final_interpretation.get('interpretation_status')} "
            f"({final_interpretation.get('explanation')})."
        )
    ]

    strong_aligned = _safe_list(
        final_interpretation.get("strong_aligned_non_trigger_contributors")
    )
    if strong_aligned:
        inferences.append(
            "Strong aligned non-trigger contributors: "
            + ", ".join(str(strong) for strong in strong_aligned)
            + "."
        )
    strong_family_aligned = _safe_list(
        final_interpretation.get("strong_aligned_non_trigger_contributor_families")
    )
    if strong_family_aligned:
        inferences.append(
            "Strong aligned contributor families: "
            + ", ".join(str(family) for family in strong_family_aligned)
            + "."
        )
    aligned = _safe_list(final_interpretation.get("aligned_non_trigger_contributors"))
    if aligned and aligned != strong_aligned:
        inferences.append(
            "Additional aligned non-trigger contributors: "
            + ", ".join(str(value) for value in aligned if value not in strong_aligned)
            + "."
        )

    uncertainties = [
        "Contributor ordering is descriptive only and cannot prove the internal aggregate formula or penalty rule inside the hidden decision layer.",
        "Rule_engine_confidence remains the target surface being explained, not a proved causal gate by itself.",
        "Final outcome fields and reason buckets remain contextual evidence only and are not treated as stronger proof than the persisted contributor fields.",
    ]
    if summary.get("comparison_support_status") != "supported":
        uncertainties.append(
            "Because the preserved-vs-collapsed comparison is still limited in this synthetic-sized run, contributor rankings should be treated as directional evidence rather than settled proof."
        )

    return {
        "facts": [item for item in facts if item],
        "inferences": [item for item in inferences if item],
        "uncertainties": uncertainties,
        "ordered_contributor_fields": _safe_list(
            contributor_tracking.get("ordered_contributor_fields")
        ),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    observations = _safe_dict(widest.get("key_observations"))
    final_interpretation = _safe_dict(widest.get("final_interpretation"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "interpretation_status": final_interpretation.get("interpretation_status"),
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
        "contributor_ranking": _safe_list(observations.get("ordered_contributor_fields")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            final_interpretation=final_interpretation,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {REPORT_TITLE}", ""]
    lines.append("## Configurations")
    lines.append("")
    for configuration in _safe_list(report.get("configurations_evaluated")):
        config = _safe_dict(configuration)
        lines.append(
            f"- {config.get('display_name')}: latest_window_hours={config.get('latest_window_hours')}, latest_max_rows={config.get('latest_max_rows')}"
        )
    lines.append("")

    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        headline = _safe_dict(_safe_dict(summary).get("headline"))
        observations = _safe_dict(_safe_dict(summary).get("key_observations"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(
            "- final_rule_bias_aligned_row_count: "
            f"{headline.get('final_rule_bias_aligned_row_count', 0)}"
        )
        lines.append(
            "- preserved_final_directional_outcome_row_count: "
            f"{headline.get('preserved_final_directional_outcome_row_count', 0)}"
        )
        lines.append(
            "- collapsed_final_hold_outcome_row_count: "
            f"{headline.get('collapsed_final_hold_outcome_row_count', 0)}"
        )
        lines.append(
            "- comparison_support_status: "
            f"{headline.get('comparison_support_status', 'unknown')}"
        )
        lines.append(
            "- strongest_contributor_field: "
            f"{headline.get('strongest_contributor_field', 'none')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{headline.get('interpretation_status', 'unknown')}"
        )
        lines.append(
            "- trigger_negative_control_status: "
            f"{headline.get('trigger_negative_control_status', 'unknown')}"
        )
        for fact in _safe_list(observations.get("facts"))[:6]:
            lines.append(f"- fact: {fact}")
        for inference in _safe_list(observations.get("inferences"))[:4]:
            lines.append(f"- inference: {inference}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        f"- interpretation_status: {final_assessment.get('interpretation_status', 'unknown')}"
    )
    for item in _safe_list(final_assessment.get("contributor_ranking"))[:5]:
        lines.append(f"- contributor_rank: {item}")
    for item in _safe_list(final_assessment.get("remains_unproven"))[:5]:
        lines.append(f"- unproven: {item}")
    return "\n".join(lines) + "\n"


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output_dir = _resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / REPORT_JSON_NAME
    md_path = resolved_output_dir / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def _tracking_alignment(row: dict[str, Any]) -> str:
    comparison_status = str(row.get("comparison_status") or "")
    range_order = _safe_dict(row.get("range_overlap")).get("range_order")
    if comparison_status == "all_missing":
        return "insufficient_data"
    if comparison_status in {"missing_on_collapsed_only", "missing_on_preserved_only"}:
        return "missingness_only"
    if comparison_status == "higher_on_preserved":
        if range_order == "collapsed_below_preserved":
            return "clean_group_separation"
        return "group_gap_present_with_overlap"
    return "mixed_or_inverse"


def _tracking_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    pairwise_group_ordering = _safe_dict(row.get("pairwise_group_ordering"))
    correlation = _safe_dict(row.get("rule_engine_correlation")).get(
        "pearson_correlation_with_rule_engine_confidence"
    )
    correlation_value = _to_float(correlation, default=-2.0)
    gap_share = _to_float(
        row.get("median_gap_share_of_rule_engine_confidence"),
        default=-1.0,
    )
    return (
        _TRACKING_ALIGNMENT_ORDER.get(str(row.get("tracking_alignment") or ""), 99),
        -_to_float(
            pairwise_group_ordering.get("preserved_greater_rate"),
            default=0.0,
        ),
        -correlation_value,
        -gap_share,
        _CONTRIBUTOR_FIELD_ORDER.get(str(row.get("field") or ""), 99),
    )


def _contributor_comparison_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    range_overlap = _safe_dict(row.get("range_overlap"))
    return (
        _CONTRIBUTOR_COMPARISON_STATUS_ORDER.get(
            str(row.get("comparison_status") or ""),
            99,
        ),
        _RANGE_OVERLAP_ORDER.get(str(range_overlap.get("range_order") or ""), 99),
        -abs(
            _to_float(
                row.get("median_difference_preserved_minus_collapsed"),
                default=0.0,
            )
        ),
        _CONTRIBUTOR_FIELD_ORDER.get(str(row.get("field") or ""), 99),
    )


def _non_trigger_contributor_families(fields: Sequence[str]) -> list[str]:
    ordered_families: list[str] = []
    seen_families: set[str] = set()
    for field in fields:
        family = _NON_TRIGGER_CONTRIBUTOR_FAMILY_MAP.get(str(field))
        if family is None or family in seen_families:
            continue
        seen_families.add(family)
        ordered_families.append(family)
    return ordered_families


def _trigger_negative_control_status(
    *,
    ranking_rows: Sequence[dict[str, Any]],
) -> str:
    trigger_row = next(
        (
            _safe_dict(row)
            for row in ranking_rows
            if _safe_dict(row).get("field") == "trigger_layer_confidence"
        ),
        {},
    )
    if not trigger_row:
        return "trigger_negative_control_missing"

    trigger_alignment = str(trigger_row.get("tracking_alignment") or "")
    non_trigger_aligned_count = sum(
        1
        for row in ranking_rows
        if _safe_dict(row).get("field") in _NON_TRIGGER_CONTRIBUTOR_FIELDS
        and _safe_dict(row).get("tracking_alignment")
        in {"clean_group_separation", "group_gap_present_with_overlap"}
    )
    if trigger_alignment in {"mixed_or_inverse", "insufficient_data", "missingness_only"}:
        return "trigger_remains_negative_control"
    if trigger_alignment == "clean_group_separation" and non_trigger_aligned_count <= 0:
        return "trigger_looks_stronger_than_expected"
    return "trigger_separates_but_is_not_primary"


def _numeric_range_overlap(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
) -> dict[str, Any]:
    preserved_min = preserved_summary.get("min")
    preserved_max = preserved_summary.get("max")
    collapsed_min = collapsed_summary.get("min")
    collapsed_max = collapsed_summary.get("max")

    if (
        preserved_min is None
        or preserved_max is None
        or collapsed_min is None
        or collapsed_max is None
    ):
        return {
            "support_status": "insufficient_data",
            "ranges_overlap": None,
            "range_order": "incomplete",
            "non_overlap_gap": None,
        }

    preserved_min_value = _to_float(preserved_min, default=0.0)
    preserved_max_value = _to_float(preserved_max, default=0.0)
    collapsed_min_value = _to_float(collapsed_min, default=0.0)
    collapsed_max_value = _to_float(collapsed_max, default=0.0)

    ranges_overlap = not (
        preserved_min_value > collapsed_max_value
        or collapsed_min_value > preserved_max_value
    )
    if ranges_overlap:
        return {
            "support_status": "supported",
            "ranges_overlap": True,
            "range_order": "overlapping",
            "non_overlap_gap": 0.0,
        }

    if preserved_min_value > collapsed_max_value:
        return {
            "support_status": "supported",
            "ranges_overlap": False,
            "range_order": "collapsed_below_preserved",
            "non_overlap_gap": round(preserved_min_value - collapsed_max_value, 6),
        }
    return {
        "support_status": "supported",
        "ranges_overlap": False,
        "range_order": "preserved_below_collapsed",
        "non_overlap_gap": round(collapsed_min_value - preserved_max_value, 6),
    }


def _pairwise_group_ordering(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    preserved_values = [
        float(row[field])
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
        and row.get(field) is not None
    ]
    collapsed_values = [
        float(row[field])
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
        and row.get(field) is not None
    ]

    if not preserved_values or not collapsed_values:
        return {
            "support_status": "insufficient_data",
            "comparable_pair_count": 0,
            "preserved_greater_pair_count": 0,
            "collapsed_greater_pair_count": 0,
            "tie_pair_count": 0,
            "preserved_greater_rate": 0.0,
            "collapsed_greater_rate": 0.0,
            "tie_rate": 0.0,
            "dominant_relation": "incomplete",
        }

    preserved_greater_pair_count = 0
    collapsed_greater_pair_count = 0
    tie_pair_count = 0
    for preserved_value in preserved_values:
        for collapsed_value in collapsed_values:
            if preserved_value > collapsed_value:
                preserved_greater_pair_count += 1
            elif collapsed_value > preserved_value:
                collapsed_greater_pair_count += 1
            else:
                tie_pair_count += 1

    comparable_pair_count = (
        preserved_greater_pair_count + collapsed_greater_pair_count + tie_pair_count
    )
    dominant_relation = "tied"
    if preserved_greater_pair_count > collapsed_greater_pair_count:
        dominant_relation = "preserved_greater"
    elif collapsed_greater_pair_count > preserved_greater_pair_count:
        dominant_relation = "collapsed_greater"

    return {
        "support_status": "supported",
        "comparable_pair_count": comparable_pair_count,
        "preserved_greater_pair_count": preserved_greater_pair_count,
        "collapsed_greater_pair_count": collapsed_greater_pair_count,
        "tie_pair_count": tie_pair_count,
        "preserved_greater_rate": _safe_ratio(
            preserved_greater_pair_count,
            comparable_pair_count,
        ),
        "collapsed_greater_rate": _safe_ratio(
            collapsed_greater_pair_count,
            comparable_pair_count,
        ),
        "tie_rate": _safe_ratio(tie_pair_count, comparable_pair_count),
        "dominant_relation": dominant_relation,
    }


def _correlation_with_rule_engine(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    contributor_field: str,
) -> dict[str, Any]:
    pairs = [
        (
            float(row[contributor_field]),
            float(row[_RULE_ENGINE_CONFIDENCE_FIELD]),
        )
        for row in comparison_rows
        if row.get(contributor_field) is not None
        and row.get(_RULE_ENGINE_CONFIDENCE_FIELD) is not None
    ]
    if len(pairs) < 2:
        return {
            "paired_row_count": len(pairs),
            "pearson_correlation_with_rule_engine_confidence": None,
        }

    contributor_values = [pair[0] for pair in pairs]
    rule_engine_values = [pair[1] for pair in pairs]
    contributor_mean = sum(contributor_values) / len(contributor_values)
    rule_engine_mean = sum(rule_engine_values) / len(rule_engine_values)
    numerator = sum(
        (contributor_value - contributor_mean)
        * (rule_engine_value - rule_engine_mean)
        for contributor_value, rule_engine_value in pairs
    )
    contributor_variance = sum(
        (value - contributor_mean) ** 2 for value in contributor_values
    )
    rule_engine_variance = sum(
        (value - rule_engine_mean) ** 2 for value in rule_engine_values
    )
    if contributor_variance <= 0 or rule_engine_variance <= 0:
        return {
            "paired_row_count": len(pairs),
            "pearson_correlation_with_rule_engine_confidence": None,
        }

    correlation = numerator / math.sqrt(contributor_variance * rule_engine_variance)
    return {
        "paired_row_count": len(pairs),
        "pearson_correlation_with_rule_engine_confidence": round(correlation, 6),
    }


def _gap_share(
    *,
    contributor_gap: float | None,
    rule_engine_gap: float | None,
) -> float | None:
    if contributor_gap is None or rule_engine_gap is None or rule_engine_gap <= 0:
        return None
    return round(float(contributor_gap) / float(rule_engine_gap), 6)


def _safe_abs_float(value: Any) -> float | None:
    converted = _to_float(value, default=None)
    if converted is None:
        return None
    return abs(converted)


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    final_interpretation: dict[str, Any],
) -> str:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    interpretation_status = str(
        final_interpretation.get("interpretation_status") or "comparison_unsupported"
    )

    if final_slice_row_count <= 0:
        return (
            "No rows reached the final fully aligned plus rule-bias-aligned slice in "
            "the widest configuration, so contributor diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one side "
            "of the preserved-vs-collapsed comparison is missing, so contributor "
            "diagnosis remains incomplete."
        )
    if interpretation_status == "persisted_contributors_appear_sufficient":
        return (
            "The widest configuration indicates that low rule_engine_confidence in "
            "collapsed final-slice rows is mainly explainable by already-persisted "
            "non-trigger contributor weakness rather than requiring a stronger hidden "
            "aggregate or penalty rule."
        )
    if interpretation_status == "mixed_persisted_contributor_surface":
        return (
            "The widest configuration shows persisted contributor weakness inside the "
            "final slice, but that surface is still more overlapped than the "
            "rule_engine_confidence split, so a hidden aggregate or penalty rule "
            "remains plausible."
        )
    return (
        "The widest configuration still suggests an additional hidden aggregate or "
        "penalty rule: persisted contributors remain too mixed, missing, or weak to "
        "fully account for the low rule_engine_confidence visible in collapsed rows."
    )


def _resolve_path(path: Path) -> Path:
    return final_split_module._resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return final_split_module._parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return final_split_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    return final_split_module._build_stage_row(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    return final_split_module._build_activation_gap_row(row)


def _build_fully_aligned_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return final_split_module._build_fully_aligned_row(row)


def _build_final_split_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return final_split_module._build_final_split_row(row)


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return final_split_module._widest_configuration_summary(configuration_summaries)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _to_float(value: Any, *, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


if __name__ == "__main__":
    main()
