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

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Aggregate Shape Diagnosis Report"
)
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
_CONTEXT_BIAS_FAMILY_FIELD = "context_bias_family_mean"
_SURFACE_COMPARISON_TOLERANCE = 1e-6

AGGREGATE_COMPONENT_FIELDS = (
    "setup_layer_confidence",
    "context_layer_confidence",
    "bias_layer_confidence",
    "selected_strategy_confidence",
)

_SINGLE_CONTRIBUTOR_FIELD_SPECS = tuple(
    spec
    for spec in final_split_module._NUMERIC_FIELD_SPECS
    if spec[0] in AGGREGATE_COMPONENT_FIELDS
)
_SINGLE_CONTRIBUTOR_FIELD_ORDER = {
    name: index for index, (name, _, _) in enumerate(_SINGLE_CONTRIBUTOR_FIELD_SPECS)
}

_AGGREGATE_CANDIDATE_SPECS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        _CONTEXT_BIAS_FAMILY_FIELD,
        "Context/bias family mean",
        "mean(context_layer_confidence, bias_layer_confidence)",
        ("context_layer_confidence", "bias_layer_confidence"),
    ),
    (
        "min_of_three",
        "Min of three",
        "min(setup_layer_confidence, context_bias_family_mean, selected_strategy_confidence)",
        (
            "setup_layer_confidence",
            _CONTEXT_BIAS_FAMILY_FIELD,
            "selected_strategy_confidence",
        ),
    ),
    (
        "mean_of_three",
        "Mean of three",
        "mean(setup_layer_confidence, context_bias_family_mean, selected_strategy_confidence)",
        (
            "setup_layer_confidence",
            _CONTEXT_BIAS_FAMILY_FIELD,
            "selected_strategy_confidence",
        ),
    ),
    (
        "weighted_mean_setup_emphasis",
        "Weighted mean with setup emphasis",
        "0.50*setup_layer_confidence + 0.25*context_bias_family_mean + 0.25*selected_strategy_confidence",
        (
            "setup_layer_confidence",
            _CONTEXT_BIAS_FAMILY_FIELD,
            "selected_strategy_confidence",
        ),
    ),
    (
        "second_lowest_of_three",
        "Second lowest of three",
        "second_lowest(setup_layer_confidence, context_bias_family_mean, selected_strategy_confidence)",
        (
            "setup_layer_confidence",
            _CONTEXT_BIAS_FAMILY_FIELD,
            "selected_strategy_confidence",
        ),
    ),
)

AGGREGATE_CANDIDATE_FIELDS = tuple(
    field for field, _, _, _ in _AGGREGATE_CANDIDATE_SPECS
)
_AGGREGATE_FIELD_ORDER = {
    field: index for index, field in enumerate(AGGREGATE_CANDIDATE_FIELDS)
}
_AGGREGATE_FORMULA_MAP = {
    field: formula for field, _, formula, _ in _AGGREGATE_CANDIDATE_SPECS
}
_AGGREGATE_COMPONENT_MAP = {
    field: component_fields
    for field, _, _, component_fields in _AGGREGATE_CANDIDATE_SPECS
}
_AGGREGATE_FIELD_LABEL_MAP = {
    field: label for field, label, _, _ in _AGGREGATE_CANDIDATE_SPECS
}

_COMPARISON_STATUS_ORDER = {
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
_RANGE_DIRECTIONAL_SEPARATION_SCORE = {
    "preserved_below_collapsed": -1,
    "overlapping": 0,
    "collapsed_below_preserved": 1,
}

_AGGREGATE_COMPARISON_FIELD_SPECS = tuple(
    (
        field,
        label,
        lambda row, *, _field=field: _to_float(row.get(_field), default=None),
    )
    for field, label, _, _ in _AGGREGATE_CANDIDATE_SPECS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the final fully aligned, "
            "rule-bias-aligned preserved-vs-collapsed slice and test whether a small "
            "transparent aggregate of persisted contributor fields can reproduce the "
            "observed rule_engine_confidence split."
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
            "Retained for architectural parity with sibling reports. Aggregate-shape "
            "comparison itself uses only the final preserved-vs-collapsed slice."
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

    result = run_selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_interpretation = _safe_dict(report.get("final_interpretation"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
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
                "best_aggregate_candidate": final_interpretation.get(
                    "best_aggregate_candidate"
                ),
                "interpretation_status": final_interpretation.get(
                    "interpretation_status"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report(
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
        "single_contributor_reference": _safe_dict(
            widest_summary.get("single_contributor_reference")
        ),
        "aggregate_shape_comparison": _safe_dict(
            widest_summary.get("aggregate_shape_comparison")
        ),
        "final_interpretation": _safe_dict(widest_summary.get("final_interpretation")),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the neighboring sibling reports.",
            "Only persisted contributor-confidence fields are used to build the transparent aggregate candidates here; no production decision logic, mapper logic, or execution-gate behavior is modified.",
            "Trigger remains excluded from the aggregate candidates and is preserved as a negative control by omission rather than being folded into any synthetic score.",
            "Range overlap, preserved-greater pair rate, and row-level correlation to rule_engine_confidence are descriptive helpers only and are not causal proof.",
            "A simple aggregate is treated as mostly reproducible only when its tracked descriptive surfaces match actual persisted rule_engine_confidence within a very small tolerance; otherwise the result remains unproven rather than promoted to a positive conclusion.",
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
    aggregate_comparison_rows = [
        {
            **row,
            **build_aggregate_candidate_values(row),
        }
        for row in comparison_rows
    ]

    summary = final_split_module.build_summary(
        actionable_rows=actionable_rows,
        fully_aligned_rows=fully_aligned_rows,
        final_split_rows=final_split_rows,
        min_symbol_support=min_symbol_support,
    )
    rule_engine_confidence_context = build_rule_engine_confidence_context(
        comparison_rows=aggregate_comparison_rows
    )
    single_contributor_reference = build_single_contributor_reference(
        comparison_rows=aggregate_comparison_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    aggregate_shape_comparison = build_aggregate_shape_comparison(
        comparison_rows=aggregate_comparison_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
        single_contributor_reference=single_contributor_reference,
    )
    final_interpretation = build_final_interpretation(
        summary=summary,
        aggregate_shape_comparison=aggregate_shape_comparison,
    )
    key_observations = build_key_observations(
        summary=summary,
        rule_engine_confidence_context=rule_engine_confidence_context,
        single_contributor_reference=single_contributor_reference,
        aggregate_shape_comparison=aggregate_shape_comparison,
        final_interpretation=final_interpretation,
    )

    best_aggregate = _safe_dict(aggregate_shape_comparison.get("best_aggregate_candidate"))
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
            "best_aggregate_candidate": best_aggregate.get("field"),
            "best_aggregate_is_sharper_than_any_single_contributor": (
                final_interpretation.get(
                    "best_aggregate_is_sharper_than_any_single_contributor"
                )
            ),
            "best_aggregate_still_weaker_than_actual_rule_engine_confidence": (
                final_interpretation.get(
                    "best_aggregate_still_weaker_than_actual_rule_engine_confidence"
                )
            ),
            "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": (
                final_interpretation.get(
                    "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance"
                )
            ),
            "interpretation_status": final_interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "rule_engine_confidence_context": rule_engine_confidence_context,
        "single_contributor_reference": single_contributor_reference,
        "aggregate_shape_comparison": aggregate_shape_comparison,
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


def build_aggregate_candidate_values(row: dict[str, Any]) -> dict[str, float | None]:
    context_confidence = _to_float(
        row.get("context_layer_confidence"),
        default=None,
    )
    bias_confidence = _to_float(
        row.get("bias_layer_confidence"),
        default=None,
    )
    setup_confidence = _to_float(
        row.get("setup_layer_confidence"),
        default=None,
    )
    selected_strategy_confidence = _to_float(
        row.get("selected_strategy_confidence"),
        default=None,
    )

    context_bias_family_mean = _rounded_mean(
        [context_confidence, bias_confidence]
    )
    trio_values = _complete_values(
        setup_confidence,
        context_bias_family_mean,
        selected_strategy_confidence,
    )
    if trio_values is None:
        return {
            _CONTEXT_BIAS_FAMILY_FIELD: context_bias_family_mean,
            "min_of_three": None,
            "mean_of_three": None,
            "weighted_mean_setup_emphasis": None,
            "second_lowest_of_three": None,
        }

    ordered_values = sorted(trio_values)
    return {
        _CONTEXT_BIAS_FAMILY_FIELD: context_bias_family_mean,
        "min_of_three": round(min(trio_values), 6),
        "mean_of_three": _rounded_mean(trio_values),
        "weighted_mean_setup_emphasis": round(
            (0.50 * trio_values[0]) + (0.25 * trio_values[1]) + (0.25 * trio_values[2]),
            6,
        ),
        "second_lowest_of_three": round(ordered_values[1], 6),
    }


def build_single_contributor_reference(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    comparison = threshold_module.build_numeric_field_comparison_for_fields(
        comparison_rows=comparison_rows,
        field_specs=_SINGLE_CONTRIBUTOR_FIELD_SPECS,
    )
    rule_engine_median_gap = _safe_abs_float(
        rule_engine_confidence_context.get("median_difference_preserved_minus_collapsed")
    )

    field_rows = [
        _build_metric_row(
            base_row=_safe_dict(base_row),
            comparison_rows=comparison_rows,
            formula=f"persisted::{_safe_dict(base_row).get('field')}",
            component_fields=(str(_safe_dict(base_row).get("field") or ""),),
            actual_rule_engine_median_gap=rule_engine_median_gap,
        )
        for base_row in _safe_list(comparison.get("field_comparisons"))
    ]
    field_rows.sort(
        key=lambda row: _shape_sort_key(
            row,
            field_order=_SINGLE_CONTRIBUTOR_FIELD_ORDER,
        )
    )
    strongest_single_contributor = _safe_dict(field_rows[0] if field_rows else {})
    return {
        "support_status": comparison.get("support_status", "limited_support"),
        "field_sort_method": (
            "Single contributors are ordered by preserved-vs-collapsed direction, "
            "range separation, preserved-greater pair rate, absolute median gap, and "
            "correlation to persisted rule_engine_confidence."
        ),
        "field_comparisons": field_rows,
        "strongest_single_contributor": strongest_single_contributor,
        "ordered_single_contributor_fields": [
            row.get("field") for row in field_rows if row.get("field")
        ],
    }


def build_aggregate_shape_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
    single_contributor_reference: dict[str, Any],
) -> dict[str, Any]:
    comparison = threshold_module.build_numeric_field_comparison_for_fields(
        comparison_rows=comparison_rows,
        field_specs=_AGGREGATE_COMPARISON_FIELD_SPECS,
    )
    strongest_single_contributor = _safe_dict(
        single_contributor_reference.get("strongest_single_contributor")
    )
    actual_rule_engine_reference = build_rule_engine_confidence_reference(
        comparison_rows=comparison_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    rule_engine_median_gap = _safe_abs_float(
        rule_engine_confidence_context.get("median_difference_preserved_minus_collapsed")
    )

    candidate_rows: list[dict[str, Any]] = []
    for base_row in _safe_list(comparison.get("field_comparisons")):
        row = _build_metric_row(
            base_row=_safe_dict(base_row),
            comparison_rows=comparison_rows,
            formula=_AGGREGATE_FORMULA_MAP.get(str(_safe_dict(base_row).get("field"))),
            component_fields=_AGGREGATE_COMPONENT_MAP.get(
                str(_safe_dict(base_row).get("field")),
                (),
            ),
            actual_rule_engine_median_gap=rule_engine_median_gap,
        )
        row["sharper_than_any_single_contributor_field"] = bool(
            strongest_single_contributor
            and _shape_relation(row, strongest_single_contributor) == "sharper"
        )
        actual_surface_comparison = _compare_candidate_to_actual_rule_engine_confidence(
            candidate_row=row,
            actual_row=actual_rule_engine_reference,
        )
        row["comparison_to_actual_rule_engine_confidence"] = _safe_dict(
            actual_surface_comparison.get("summary")
        )
        row["residual_gap_vs_actual_rule_engine_confidence"] = _safe_dict(
            actual_surface_comparison.get("difference_summary")
        )
        candidate_rows.append(row)

    candidate_rows.sort(
        key=lambda row: _shape_sort_key(
            row,
            field_order=_AGGREGATE_FIELD_ORDER,
        )
    )
    for index, row in enumerate(candidate_rows, start=1):
        row["candidate_rank"] = index

    best_aggregate_candidate = _safe_dict(candidate_rows[0] if candidate_rows else {})
    best_actual_surface_comparison = _safe_dict(
        best_aggregate_candidate.get("comparison_to_actual_rule_engine_confidence")
    )
    best_vs_single = (
        strongest_single_contributor
        and _shape_relation(best_aggregate_candidate, strongest_single_contributor)
        == "sharper"
    )

    return {
        "support_status": comparison.get("support_status", "limited_support"),
        "trigger_excluded_from_aggregate_candidates": True,
        "aggregate_component_fields": list(AGGREGATE_COMPONENT_FIELDS),
        "candidate_sort_method": (
            "Aggregate candidates are ordered by preserved-vs-collapsed direction, "
            "range separation, preserved-greater pair rate, absolute median gap, and "
            "correlation to persisted rule_engine_confidence."
        ),
        "candidate_formulas": [
            {
                "field": field,
                "field_label": label,
                "formula": formula,
                "component_fields": list(component_fields),
            }
            for field, label, formula, component_fields in _AGGREGATE_CANDIDATE_SPECS
        ],
        "candidate_comparisons": candidate_rows,
        "best_aggregate_candidate": best_aggregate_candidate,
        "strongest_single_contributor_reference": strongest_single_contributor,
        "actual_rule_engine_confidence_reference": actual_rule_engine_reference,
        "best_aggregate_is_sharper_than_any_single_contributor": bool(best_vs_single),
        "best_aggregate_has_supported_actual_surface_comparison": (
            best_actual_surface_comparison.get("support_status") == "supported"
        ),
        "best_aggregate_still_weaker_than_actual_rule_engine_confidence": (
            best_actual_surface_comparison.get("materially_weaker_on_tracked_surfaces")
        ),
        "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": (
            best_actual_surface_comparison.get(
                "matches_actual_on_tracked_surfaces_within_tolerance"
            )
        ),
    }


def build_rule_engine_confidence_reference(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    distribution = _safe_dict(rule_engine_confidence_context.get("distribution"))
    rule_engine_median_gap = _safe_abs_float(
        distribution.get("median_difference_preserved_minus_collapsed")
    )
    reference = _build_metric_row(
        base_row={
            "field": _RULE_ENGINE_CONFIDENCE_FIELD,
            "field_label": "Rule engine confidence",
            _COMPARISON_GROUP_PRESERVED: _safe_dict(
                distribution.get(_COMPARISON_GROUP_PRESERVED)
            ),
            _COMPARISON_GROUP_COLLAPSED: _safe_dict(
                distribution.get(_COMPARISON_GROUP_COLLAPSED)
            ),
            "comparison_status": distribution.get("comparison_status", "all_missing"),
            "median_difference_preserved_minus_collapsed": distribution.get(
                "median_difference_preserved_minus_collapsed"
            ),
            "mean_difference_preserved_minus_collapsed": distribution.get(
                "mean_difference_preserved_minus_collapsed"
            ),
        },
        comparison_rows=comparison_rows,
        formula="persisted::rule_engine_confidence",
        component_fields=(_RULE_ENGINE_CONFIDENCE_FIELD,),
        actual_rule_engine_median_gap=rule_engine_median_gap,
    )
    reference["range_overlap"] = _safe_dict(
        rule_engine_confidence_context.get("range_overlap")
    )
    reference["median_gap_share_of_actual_rule_engine_confidence"] = (
        1.0 if rule_engine_median_gap is not None else None
    )
    return reference


def build_final_interpretation(
    *,
    summary: dict[str, Any],
    aggregate_shape_comparison: dict[str, Any],
) -> dict[str, Any]:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )

    if final_slice_row_count <= 0:
        return {
            "support_status": "unsupported",
            "best_aggregate_candidate": None,
            "best_aggregate_is_sharper_than_any_single_contributor": False,
            "best_aggregate_still_weaker_than_actual_rule_engine_confidence": None,
            "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": None,
            "actual_split_mostly_reproducible_by_simple_aggregate": False,
            "interpretation_status": "comparison_unsupported",
            "explanation": (
                "No rows reached the final fully aligned plus rule-bias-aligned slice, "
                "so aggregate-shape diagnosis is unsupported."
            ),
        }

    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return {
            "support_status": "unsupported",
            "best_aggregate_candidate": None,
            "best_aggregate_is_sharper_than_any_single_contributor": False,
            "best_aggregate_still_weaker_than_actual_rule_engine_confidence": None,
            "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": None,
            "actual_split_mostly_reproducible_by_simple_aggregate": False,
            "interpretation_status": "comparison_unsupported",
            "explanation": (
                "The final fully aligned plus rule-bias-aligned slice exists, but one "
                "side of the preserved-vs-collapsed comparison is missing, so this run "
                "cannot safely judge whether a simple persisted aggregate reproduces the "
                "actual rule_engine_confidence split."
            ),
        }

    best_aggregate_candidate = _safe_dict(
        aggregate_shape_comparison.get("best_aggregate_candidate")
    )
    best_actual_surface_comparison = _safe_dict(
        best_aggregate_candidate.get("comparison_to_actual_rule_engine_confidence")
    )
    if (
        aggregate_shape_comparison.get("support_status") != "supported"
        or not best_aggregate_candidate
        or best_actual_surface_comparison.get("support_status") != "supported"
    ):
        return {
            "support_status": "unsupported",
            "best_aggregate_candidate": best_aggregate_candidate.get("field"),
            "best_aggregate_is_sharper_than_any_single_contributor": bool(
                aggregate_shape_comparison.get(
                    "best_aggregate_is_sharper_than_any_single_contributor"
                )
            ),
            "best_aggregate_still_weaker_than_actual_rule_engine_confidence": None,
            "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": None,
            "actual_split_mostly_reproducible_by_simple_aggregate": False,
            "interpretation_status": "comparison_unsupported",
            "explanation": (
                "The final preserved-vs-collapsed slice exists, but the best aggregate "
                "candidate does not have a fully supported tracked-surface comparison "
                "against actual rule_engine_confidence, so this run remains "
                "interpretation-unsupported."
            ),
        }

    best_aggregate_field = best_aggregate_candidate.get("field")
    best_is_sharper_than_any_single_contributor = bool(
        aggregate_shape_comparison.get(
            "best_aggregate_is_sharper_than_any_single_contributor"
        )
    )
    best_still_weaker_than_actual = bool(
        best_actual_surface_comparison.get("materially_weaker_on_tracked_surfaces")
    )
    best_matches_actual_on_tracked_surfaces = bool(
        best_actual_surface_comparison.get(
            "matches_actual_on_tracked_surfaces_within_tolerance"
        )
    )
    actual_split_mostly_reproducible_by_simple_aggregate = bool(
        best_aggregate_field
        and best_is_sharper_than_any_single_contributor
        and best_matches_actual_on_tracked_surfaces
    )

    interpretation_status = "simple_aggregate_does_not_beat_single_contributors"
    explanation = (
        "The best transparent aggregate candidate does not outrank the strongest single "
        "persisted contributor field on the final preserved-vs-collapsed slice, so the "
        "current evidence does not support the view that a simple persisted aggregate is "
        "the main reason actual rule_engine_confidence is sharper."
    )

    if actual_split_mostly_reproducible_by_simple_aggregate:
        interpretation_status = "simple_aggregate_mostly_reproduces_actual_split"
        explanation = (
            "The best transparent aggregate candidate outranks every single persisted "
            "contributor field and matches the tracked descriptive surfaces of actual "
            "persisted rule_engine_confidence within the report tolerance, so the "
            "actual split looks mostly reproducible by a simple persisted aggregate on "
            "this run."
        )
    elif best_is_sharper_than_any_single_contributor:
        if best_still_weaker_than_actual:
            interpretation_status = (
                "aggregate_improves_on_single_contributors_but_remains_weaker_than_actual_rule_engine_confidence"
            )
            explanation = (
                "The best transparent aggregate candidate is sharper than any single "
                "persisted contributor field, but it remains materially weaker than "
                "actual persisted rule_engine_confidence on the tracked descriptive "
                "surfaces, so the aggregate shape helps yet remains an incomplete "
                "reproduction of the observed split."
            )
        else:
            interpretation_status = (
                "aggregate_improves_on_single_contributors_but_reproducibility_remains_unproven"
            )
            explanation = (
                "The best transparent aggregate candidate is sharper than any single "
                "persisted contributor field, but it does not match actual persisted "
                "rule_engine_confidence closely enough on the tracked descriptive "
                "surfaces to justify a positive reproducibility conclusion."
            )

    return {
        "support_status": summary.get("comparison_support_status", "limited_support"),
        "best_aggregate_candidate": best_aggregate_field,
        "best_aggregate_is_sharper_than_any_single_contributor": (
            best_is_sharper_than_any_single_contributor
        ),
        "best_aggregate_still_weaker_than_actual_rule_engine_confidence": (
            best_still_weaker_than_actual
        ),
        "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": (
            best_matches_actual_on_tracked_surfaces
        ),
        "actual_split_mostly_reproducible_by_simple_aggregate": (
            actual_split_mostly_reproducible_by_simple_aggregate
        ),
        "interpretation_status": interpretation_status,
        "explanation": explanation,
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    rule_engine_confidence_context: dict[str, Any],
    single_contributor_reference: dict[str, Any],
    aggregate_shape_comparison: dict[str, Any],
    final_interpretation: dict[str, Any],
) -> dict[str, list[str]]:
    actual_rule_engine_reference = _safe_dict(
        aggregate_shape_comparison.get("actual_rule_engine_confidence_reference")
    )
    strongest_single_contributor = _safe_dict(
        single_contributor_reference.get("strongest_single_contributor")
    )
    best_aggregate_candidate = _safe_dict(
        aggregate_shape_comparison.get("best_aggregate_candidate")
    )

    actual_pairwise = _safe_dict(actual_rule_engine_reference.get("pairwise_group_ordering"))
    best_aggregate_pairwise = _safe_dict(
        best_aggregate_candidate.get("pairwise_group_ordering")
    )
    strongest_single_pairwise = _safe_dict(
        strongest_single_contributor.get("pairwise_group_ordering")
    )
    best_actual_surface_comparison = _safe_dict(
        best_aggregate_candidate.get("comparison_to_actual_rule_engine_confidence")
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
            "Actual persisted rule_engine_confidence: preserved median="
            f"{_safe_dict(actual_rule_engine_reference.get(_COMPARISON_GROUP_PRESERVED)).get('median')}, "
            "collapsed median="
            f"{_safe_dict(actual_rule_engine_reference.get(_COMPARISON_GROUP_COLLAPSED)).get('median')}, "
            "range_order="
            f"{_safe_dict(actual_rule_engine_reference.get('range_overlap')).get('range_order')}, "
            "preserved_greater_pair_rate="
            f"{actual_pairwise.get('preserved_greater_rate')}."
        ),
        (
            "Strongest single contributor field="
            f"{strongest_single_contributor.get('field')}; preserved median="
            f"{_safe_dict(strongest_single_contributor.get(_COMPARISON_GROUP_PRESERVED)).get('median')}; "
            "collapsed median="
            f"{_safe_dict(strongest_single_contributor.get(_COMPARISON_GROUP_COLLAPSED)).get('median')}; "
            "preserved_greater_pair_rate="
            f"{strongest_single_pairwise.get('preserved_greater_rate')}."
        ),
        (
            "Best aggregate candidate="
            f"{best_aggregate_candidate.get('field')}; formula="
            f"{best_aggregate_candidate.get('formula')}; preserved median="
            f"{_safe_dict(best_aggregate_candidate.get(_COMPARISON_GROUP_PRESERVED)).get('median')}; "
            "collapsed median="
            f"{_safe_dict(best_aggregate_candidate.get(_COMPARISON_GROUP_COLLAPSED)).get('median')}; "
            "preserved_greater_pair_rate="
            f"{best_aggregate_pairwise.get('preserved_greater_rate')}."
        ),
        (
            "Best aggregate vs actual tracked surfaces: support_status="
            f"{best_actual_surface_comparison.get('support_status')}, "
            "materially_weaker_on_tracked_surfaces="
            f"{best_actual_surface_comparison.get('materially_weaker_on_tracked_surfaces')}, "
            "matches_actual_on_tracked_surfaces_within_tolerance="
            f"{best_actual_surface_comparison.get('matches_actual_on_tracked_surfaces_within_tolerance')}."
        ),
        (
            "Trigger remains excluded from every aggregate candidate; aggregate_component_fields="
            f"{', '.join(AGGREGATE_COMPONENT_FIELDS)}."
        ),
    ]

    inferences: list[str] = []
    if final_interpretation.get("actual_split_mostly_reproducible_by_simple_aggregate"):
        inferences.append(
            "On this run, the best simple aggregate matches the report's tracked descriptive surfaces of actual persisted rule_engine_confidence closely enough to count as mostly reproducible."
        )
    elif final_interpretation.get(
        "best_aggregate_is_sharper_than_any_single_contributor"
    ):
        if final_interpretation.get(
            "best_aggregate_still_weaker_than_actual_rule_engine_confidence"
        ):
            inferences.append(
                "A simple transparent aggregate improves on every single persisted contributor field, but actual rule_engine_confidence still remains materially sharper on the tracked descriptive surfaces."
            )
        else:
            inferences.append(
                "A simple transparent aggregate improves on every single persisted contributor field, but tracked-surface reproducibility against actual rule_engine_confidence remains unproven."
            )
    else:
        inferences.append(
            "The tested simple aggregates do not outrank the strongest single persisted contributor field, so the current aggregate shape remains insufficient as the main explanation."
        )

    strongest_single_field = strongest_single_contributor.get("field")
    best_aggregate_field = best_aggregate_candidate.get("field")
    if strongest_single_field and best_aggregate_field:
        inferences.append(
            "The best aggregate candidate is "
            f"{best_aggregate_field}, while the strongest single-contributor benchmark is "
            f"{strongest_single_field}."
        )

    uncertainties = [
        "These aggregates are built only from persisted contributor fields and do not prove that the hidden production rule, if any, uses the same shape.",
        "Pairwise ordering, range overlap, and row-level correlation are descriptive diagnostics rather than causal proof.",
        "Final outcome fields remain slice labels only; they are not treated here as evidence of the literal causal decision rule.",
    ]
    if summary.get("comparison_support_status") != "supported":
        uncertainties.append(
            "Because one or both comparison groups remain below the usual support threshold on this run, aggregate rankings should be treated as directional rather than settled."
        )
    if best_actual_surface_comparison.get("support_status") != "supported":
        uncertainties.append(
            "The best aggregate candidate does not have a fully supported tracked-surface comparison to actual persisted rule_engine_confidence on this run, so the aggregate-shape conclusion remains limited."
        )

    return {
        "facts": [item for item in facts if item],
        "inferences": [item for item in inferences if item],
        "uncertainties": uncertainties,
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    observations = _safe_dict(widest.get("key_observations"))
    final_interpretation = _safe_dict(widest.get("final_interpretation"))
    aggregate_shape_comparison = _safe_dict(widest.get("aggregate_shape_comparison"))
    best_aggregate_candidate = _safe_dict(
        aggregate_shape_comparison.get("best_aggregate_candidate")
    )

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "best_aggregate_candidate": best_aggregate_candidate.get("field"),
        "best_aggregate_is_sharper_than_any_single_contributor": (
            final_interpretation.get(
                "best_aggregate_is_sharper_than_any_single_contributor"
            )
        ),
        "best_aggregate_still_weaker_than_actual_rule_engine_confidence": (
            final_interpretation.get(
                "best_aggregate_still_weaker_than_actual_rule_engine_confidence"
            )
        ),
        "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance": (
            final_interpretation.get(
                "best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance"
            )
        ),
        "actual_split_mostly_reproducible_by_simple_aggregate": (
            final_interpretation.get(
                "actual_split_mostly_reproducible_by_simple_aggregate"
            )
        ),
        "interpretation_status": final_interpretation.get("interpretation_status"),
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
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
            "- best_aggregate_candidate: "
            f"{headline.get('best_aggregate_candidate', 'none')}"
        )
        lines.append(
            "- best_aggregate_is_sharper_than_any_single_contributor: "
            f"{headline.get('best_aggregate_is_sharper_than_any_single_contributor', False)}"
        )
        lines.append(
            "- best_aggregate_still_weaker_than_actual_rule_engine_confidence: "
            f"{headline.get('best_aggregate_still_weaker_than_actual_rule_engine_confidence')}"
        )
        lines.append(
            "- best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance: "
            f"{headline.get('best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{headline.get('interpretation_status', 'unknown')}"
        )
        for fact in _safe_list(observations.get("facts"))[:5]:
            lines.append(f"- fact: {fact}")
        for inference in _safe_list(observations.get("inferences"))[:3]:
            lines.append(f"- inference: {inference}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        "- best_aggregate_candidate: "
        f"{final_assessment.get('best_aggregate_candidate', 'none')}"
    )
    lines.append(
        "- best_aggregate_is_sharper_than_any_single_contributor: "
        f"{final_assessment.get('best_aggregate_is_sharper_than_any_single_contributor', False)}"
    )
    lines.append(
        "- best_aggregate_still_weaker_than_actual_rule_engine_confidence: "
        f"{final_assessment.get('best_aggregate_still_weaker_than_actual_rule_engine_confidence')}"
    )
    lines.append(
        "- best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance: "
        f"{final_assessment.get('best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance')}"
    )
    lines.append(
        "- actual_split_mostly_reproducible_by_simple_aggregate: "
        f"{final_assessment.get('actual_split_mostly_reproducible_by_simple_aggregate', False)}"
    )
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


def _build_metric_row(
    *,
    base_row: dict[str, Any],
    comparison_rows: Sequence[dict[str, Any]],
    formula: str | None,
    component_fields: Sequence[str],
    actual_rule_engine_median_gap: float | None,
) -> dict[str, Any]:
    field = str(base_row.get("field") or "")
    preserved_summary = _safe_dict(base_row.get(_COMPARISON_GROUP_PRESERVED))
    collapsed_summary = _safe_dict(base_row.get(_COMPARISON_GROUP_COLLAPSED))
    median_gap = _safe_abs_float(base_row.get("median_difference_preserved_minus_collapsed"))

    return {
        **base_row,
        "formula": formula,
        "component_fields": list(component_fields),
        "range_overlap": _numeric_range_overlap(
            preserved_summary=preserved_summary,
            collapsed_summary=collapsed_summary,
        ),
        "pairwise_group_ordering": _pairwise_group_ordering(
            comparison_rows=comparison_rows,
            field=field,
        ),
        "rule_engine_correlation": _correlation_with_rule_engine(
            comparison_rows=comparison_rows,
            contributor_field=field,
        ),
        "median_gap_share_of_actual_rule_engine_confidence": _gap_share(
            candidate_gap=median_gap,
            actual_gap=actual_rule_engine_median_gap,
        ),
    }


def _shape_strength_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    pairwise_group_ordering = _safe_dict(row.get("pairwise_group_ordering"))
    correlation = _safe_dict(row.get("rule_engine_correlation")).get(
        "pearson_correlation_with_rule_engine_confidence"
    )
    return (
        _COMPARISON_STATUS_ORDER.get(str(row.get("comparison_status") or ""), 99),
        _RANGE_OVERLAP_ORDER.get(
            str(_safe_dict(row.get("range_overlap")).get("range_order") or ""),
            99,
        ),
        -_to_float(pairwise_group_ordering.get("preserved_greater_rate"), default=0.0),
        -abs(
            _to_float(
                row.get("median_difference_preserved_minus_collapsed"),
                default=0.0,
            )
        ),
        -abs(_to_float(correlation, default=0.0)),
    )


def _shape_sort_key(
    row: dict[str, Any],
    *,
    field_order: dict[str, int],
) -> tuple[Any, ...]:
    return _shape_strength_tuple(row) + (
        field_order.get(str(row.get("field") or ""), 99),
    )


def _shape_relation(left: dict[str, Any], right: dict[str, Any]) -> str:
    if not left or not right:
        return "incomparable"
    left_strength = _shape_strength_tuple(left)
    right_strength = _shape_strength_tuple(right)
    if left_strength < right_strength:
        return "sharper"
    if left_strength > right_strength:
        return "weaker"
    return "equal"


def _compare_candidate_to_actual_rule_engine_confidence(
    *,
    candidate_row: dict[str, Any],
    actual_row: dict[str, Any],
) -> dict[str, Any]:
    difference_summary = _residual_gap_summary(
        candidate_row,
        actual_row,
    )
    if not _has_supported_tracked_surface_comparison(
        candidate_row=candidate_row,
        actual_row=actual_row,
        difference_summary=difference_summary,
    ):
        return {
            "summary": {
                "support_status": "insufficient_data",
                "materially_weaker_on_tracked_surfaces": None,
                "materially_stronger_on_tracked_surfaces": None,
                "matches_actual_on_tracked_surfaces_within_tolerance": None,
                "surface_relation_to_actual_rule_engine_confidence": "incomplete",
                "tolerance": _SURFACE_COMPARISON_TOLERANCE,
                "weakening_signals": [],
                "strengthening_signals": [],
            },
            "difference_summary": difference_summary,
        }

    weakening_signals: list[str] = []
    strengthening_signals: list[str] = []

    median_gap_shortfall = _to_float(
        difference_summary.get("median_gap_shortfall_vs_actual_rule_engine_confidence"),
        default=None,
    )
    pair_rate_shortfall = _to_float(
        difference_summary.get(
            "preserved_greater_pair_rate_shortfall_vs_actual_rule_engine_confidence"
        ),
        default=None,
    )
    non_overlap_gap_shortfall = _to_float(
        difference_summary.get(
            "non_overlap_gap_shortfall_vs_actual_rule_engine_confidence"
        ),
        default=None,
    )

    if median_gap_shortfall is not None:
        if median_gap_shortfall > _SURFACE_COMPARISON_TOLERANCE:
            weakening_signals.append("smaller_median_gap_than_actual")
        elif median_gap_shortfall < -_SURFACE_COMPARISON_TOLERANCE:
            strengthening_signals.append("larger_median_gap_than_actual")

    if pair_rate_shortfall is not None:
        if pair_rate_shortfall > _SURFACE_COMPARISON_TOLERANCE:
            weakening_signals.append("lower_preserved_greater_pair_rate_than_actual")
        elif pair_rate_shortfall < -_SURFACE_COMPARISON_TOLERANCE:
            strengthening_signals.append("higher_preserved_greater_pair_rate_than_actual")

    if non_overlap_gap_shortfall is not None:
        if non_overlap_gap_shortfall > _SURFACE_COMPARISON_TOLERANCE:
            weakening_signals.append("smaller_non_overlap_gap_than_actual")
        elif non_overlap_gap_shortfall < -_SURFACE_COMPARISON_TOLERANCE:
            strengthening_signals.append("larger_non_overlap_gap_than_actual")

    candidate_range_order = str(
        difference_summary.get("candidate_range_order") or "incomplete"
    )
    actual_range_order = str(
        difference_summary.get("actual_range_order") or "incomplete"
    )
    if candidate_range_order != actual_range_order:
        candidate_range_score = _RANGE_DIRECTIONAL_SEPARATION_SCORE.get(
            candidate_range_order
        )
        actual_range_score = _RANGE_DIRECTIONAL_SEPARATION_SCORE.get(actual_range_order)
        if (
            candidate_range_score is not None
            and actual_range_score is not None
            and candidate_range_score < actual_range_score
        ):
            weakening_signals.append("weaker_range_order_than_actual")
        elif (
            candidate_range_score is not None
            and actual_range_score is not None
            and candidate_range_score > actual_range_score
        ):
            strengthening_signals.append("stronger_range_order_than_actual")
        else:
            weakening_signals.append("range_order_mismatch_to_actual")

    materially_weaker = bool(weakening_signals)
    materially_stronger = bool(strengthening_signals)
    matches_actual = not materially_weaker and not materially_stronger

    relation = "matched_on_tracked_surfaces"
    if materially_weaker and materially_stronger:
        relation = "mixed_surface_difference"
    elif materially_weaker:
        relation = "weaker_on_tracked_surfaces"
    elif materially_stronger:
        relation = "stronger_on_tracked_surfaces"

    return {
        "summary": {
            "support_status": "supported",
            "materially_weaker_on_tracked_surfaces": materially_weaker,
            "materially_stronger_on_tracked_surfaces": materially_stronger,
            "matches_actual_on_tracked_surfaces_within_tolerance": matches_actual,
            "surface_relation_to_actual_rule_engine_confidence": relation,
            "tolerance": _SURFACE_COMPARISON_TOLERANCE,
            "weakening_signals": weakening_signals,
            "strengthening_signals": strengthening_signals,
        },
        "difference_summary": difference_summary,
    }


def _has_supported_tracked_surface_comparison(
    *,
    candidate_row: dict[str, Any],
    actual_row: dict[str, Any],
    difference_summary: dict[str, Any],
) -> bool:
    candidate_preserved = _safe_dict(candidate_row.get(_COMPARISON_GROUP_PRESERVED))
    candidate_collapsed = _safe_dict(candidate_row.get(_COMPARISON_GROUP_COLLAPSED))
    actual_preserved = _safe_dict(actual_row.get(_COMPARISON_GROUP_PRESERVED))
    actual_collapsed = _safe_dict(actual_row.get(_COMPARISON_GROUP_COLLAPSED))
    candidate_pairwise = _safe_dict(candidate_row.get("pairwise_group_ordering"))
    actual_pairwise = _safe_dict(actual_row.get("pairwise_group_ordering"))
    candidate_range = _safe_dict(candidate_row.get("range_overlap"))
    actual_range = _safe_dict(actual_row.get("range_overlap"))

    required_values = [
        candidate_preserved.get("median"),
        candidate_collapsed.get("median"),
        actual_preserved.get("median"),
        actual_collapsed.get("median"),
        candidate_pairwise.get("preserved_greater_rate"),
        actual_pairwise.get("preserved_greater_rate"),
        difference_summary.get("candidate_range_order"),
        difference_summary.get("actual_range_order"),
    ]
    if any(value is None for value in required_values):
        return False

    if candidate_range.get("support_status") != "supported":
        return False
    if actual_range.get("support_status") != "supported":
        return False
    if candidate_pairwise.get("support_status") != "supported":
        return False
    if actual_pairwise.get("support_status") != "supported":
        return False
    return True


def _residual_gap_summary(
    candidate_row: dict[str, Any],
    actual_row: dict[str, Any],
) -> dict[str, Any]:
    candidate_pairwise = _safe_dict(candidate_row.get("pairwise_group_ordering"))
    actual_pairwise = _safe_dict(actual_row.get("pairwise_group_ordering"))
    candidate_range = _safe_dict(candidate_row.get("range_overlap"))
    actual_range = _safe_dict(actual_row.get("range_overlap"))

    candidate_median_gap = _safe_abs_float(
        candidate_row.get("median_difference_preserved_minus_collapsed")
    )
    actual_median_gap = _safe_abs_float(
        actual_row.get("median_difference_preserved_minus_collapsed")
    )
    candidate_non_overlap_gap = _to_float(
        candidate_range.get("non_overlap_gap"),
        default=None,
    )
    actual_non_overlap_gap = _to_float(
        actual_range.get("non_overlap_gap"),
        default=None,
    )

    return {
        "median_gap_shortfall_vs_actual_rule_engine_confidence": _difference_or_none(
            actual_median_gap,
            candidate_median_gap,
        ),
        "preserved_greater_pair_rate_shortfall_vs_actual_rule_engine_confidence": (
            _difference_or_none(
                _to_float(
                    actual_pairwise.get("preserved_greater_rate"),
                    default=None,
                ),
                _to_float(
                    candidate_pairwise.get("preserved_greater_rate"),
                    default=None,
                ),
            )
        ),
        "range_order_matches_actual_rule_engine_confidence": (
            candidate_range.get("range_order") == actual_range.get("range_order")
        ),
        "candidate_range_order": candidate_range.get("range_order"),
        "actual_range_order": actual_range.get("range_order"),
        "candidate_non_overlap_gap": candidate_non_overlap_gap,
        "actual_non_overlap_gap": actual_non_overlap_gap,
        "non_overlap_gap_shortfall_vs_actual_rule_engine_confidence": (
            _difference_or_none(
                actual_non_overlap_gap,
                candidate_non_overlap_gap,
            )
        ),
    }


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
    candidate_gap: float | None,
    actual_gap: float | None,
) -> float | None:
    if candidate_gap is None or actual_gap is None or actual_gap <= 0:
        return None
    return round(float(candidate_gap) / float(actual_gap), 6)


def _difference_or_none(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _complete_values(*values: float | None) -> list[float] | None:
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _rounded_mean(values: Sequence[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if len(numeric_values) != len(values) or not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


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
            "the widest configuration, so aggregate-shape diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one side "
            "of the preserved-vs-collapsed comparison is missing, so aggregate-shape "
            "diagnosis remains incomplete."
        )
    if interpretation_status == "simple_aggregate_mostly_reproduces_actual_split":
        return (
            "The widest configuration suggests that a small transparent aggregate over "
            "persisted contributor fields can mostly reproduce the observed final-slice "
            "rule_engine_confidence split."
        )
    if (
        interpretation_status
        == "aggregate_improves_on_single_contributors_but_remains_weaker_than_actual_rule_engine_confidence"
    ):
        return (
            "The widest configuration suggests that a simple transparent aggregate helps "
            "explain the final-slice rule_engine_confidence split, but actual persisted "
            "rule_engine_confidence still remains sharper than that aggregate."
        )
    if (
        interpretation_status
        == "aggregate_improves_on_single_contributors_but_reproducibility_remains_unproven"
    ):
        return (
            "The widest configuration suggests that a simple transparent aggregate can "
            "improve on single-contributor surfaces, but reproducibility against actual "
            "persisted rule_engine_confidence remains unproven on the tracked "
            "descriptive surfaces."
        )
    return (
        "The widest configuration suggests that the tested simple transparent "
        "aggregates remain too weak relative to the strongest single contributor "
        "baseline or to actual persisted rule_engine_confidence to fully reproduce the "
        "final-slice split."
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