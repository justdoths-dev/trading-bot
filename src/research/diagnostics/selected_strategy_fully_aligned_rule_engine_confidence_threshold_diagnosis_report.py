from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_fully_aligned_final_hold_split_diagnosis_report as final_split_module,
)

REPORT_TYPE = (
    "selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Fully Aligned Rule Engine Confidence Threshold Diagnosis Report"
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
_THRESHOLD_CLASSIFICATION_BASIS = (
    "rule_engine_confidence_exact_ranges_and_fixed_bands"
)

_THRESHOLD_CLASSIFICATIONS = {
    "no_rows",
    "insufficient_support",
    "clean_non_overlapping_split",
    "mostly_separated_with_mixed_band",
    "broadly_mixed",
    "inconclusive",
}

_BAND_DEFINITIONS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<= 0.25", None, 0.25),
    ("(0.25, 0.40]", 0.25, 0.40),
    ("(0.40, 0.60]", 0.40, 0.60),
    ("(0.60, 0.80]", 0.60, 0.80),
    ("> 0.80", 0.80, None),
)

_SECONDARY_NUMERIC_FIELDS = (
    "setup_layer_confidence",
    "context_layer_confidence",
    "bias_layer_confidence",
    "selected_strategy_confidence",
    "trigger_layer_confidence",
)
_SECONDARY_NUMERIC_FIELD_SPECS = tuple(
    spec
    for spec in final_split_module._NUMERIC_FIELD_SPECS
    if spec[0] in _SECONDARY_NUMERIC_FIELDS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the final preserved-vs-collapsed "
            "split inside the fully aligned, rule-bias-aligned slice and characterize "
            "whether rule_engine_confidence behaves like a clean threshold, a mostly "
            "separated banded split, or a mixed distribution."
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
            "Retained for architectural parity with neighboring reports. "
            "Threshold classification itself does not depend on symbol-level summaries."
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

    result = run_selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_assessment = _safe_dict(report.get("final_assessment"))
    threshold_likeness = _safe_dict(report.get("threshold_likeness"))
    overlap = _safe_dict(report.get("rule_engine_confidence_overlap"))
    band_summary = _safe_dict(report.get("confidence_band_summary"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    _safe_dict(report.get("final_assessment")).get(
                        "widest_configuration"
                    )
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
                "threshold_classification": threshold_likeness.get("classification"),
                "ranges_overlap": overlap.get("ranges_overlap"),
                "mixed_band_labels": band_summary.get("mixed_band_labels", []),
                "missing_rule_engine_confidence_row_count": band_summary.get(
                    "missing_rule_engine_confidence_row_count",
                    0,
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report(
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
        "rule_engine_confidence_distribution": _safe_dict(
            widest_summary.get("rule_engine_confidence_distribution")
        ),
        "rule_engine_confidence_overlap": _safe_dict(
            widest_summary.get("rule_engine_confidence_overlap")
        ),
        "confidence_band_summary": _safe_dict(
            widest_summary.get("confidence_band_summary")
        ),
        "threshold_likeness": _safe_dict(widest_summary.get("threshold_likeness")),
        "mixed_band_secondary_comparison": _safe_dict(
            widest_summary.get("mixed_band_secondary_comparison")
        ),
        "final_outcome_context": _safe_dict(widest_summary.get("final_outcome_context")),
        "missingness_context": _safe_dict(widest_summary.get("missingness_context")),
        "tertiary_reason_context": _safe_dict(
            widest_summary.get("tertiary_reason_context")
        ),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the already-established final rule-bias-aligned slice extraction from the neighboring fully aligned final hold split report.",
            "Threshold classification is intentionally restricted to persisted rule_engine_confidence structure inside that final slice; final outcome fields remain contextual end-state evidence only.",
            "Fixed confidence bands are descriptive helpers, not production thresholds, and coarse-band mixing is treated conservatively even when exact ranges are close.",
            "Reason-bucket comparisons remain tertiary evidence only and are explicitly excluded from the threshold classifier.",
            "Missing rule_engine_confidence values are surfaced explicitly and degrade the threshold judgment instead of being silently imputed or ignored.",
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
    rule_engine_confidence_distribution = build_rule_engine_confidence_distribution(
        comparison_rows=comparison_rows
    )
    rule_engine_confidence_overlap = build_rule_engine_confidence_overlap(
        distribution=rule_engine_confidence_distribution
    )
    confidence_band_summary = build_confidence_band_summary(
        comparison_rows=comparison_rows
    )
    mixed_band_secondary_comparison = build_mixed_band_secondary_comparison(
        comparison_rows=comparison_rows,
        confidence_band_summary=confidence_band_summary,
    )
    final_outcome_context = build_final_outcome_context(comparison_rows=comparison_rows)
    missingness_context = build_missingness_context(comparison_rows=comparison_rows)
    tertiary_reason_context = build_tertiary_reason_context(
        comparison_rows=comparison_rows
    )
    threshold_likeness = build_threshold_likeness(
        summary=summary,
        distribution=rule_engine_confidence_distribution,
        overlap=rule_engine_confidence_overlap,
        confidence_band_summary=confidence_band_summary,
        mixed_band_secondary_comparison=mixed_band_secondary_comparison,
        final_outcome_context=final_outcome_context,
        missingness_context=missingness_context,
        tertiary_reason_context=tertiary_reason_context,
    )
    key_observations = build_key_observations(
        summary=summary,
        distribution=rule_engine_confidence_distribution,
        overlap=rule_engine_confidence_overlap,
        confidence_band_summary=confidence_band_summary,
        threshold_likeness=threshold_likeness,
        final_outcome_context=final_outcome_context,
        missingness_context=missingness_context,
        tertiary_reason_context=tertiary_reason_context,
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
            "threshold_classification": threshold_likeness.get("classification"),
            "ranges_overlap": rule_engine_confidence_overlap.get("ranges_overlap"),
            "mixed_band_count": len(
                _safe_list(confidence_band_summary.get("mixed_band_labels"))
            ),
            "mixed_band_labels": _safe_list(
                confidence_band_summary.get("mixed_band_labels")
            ),
            "missing_rule_engine_confidence_row_count": confidence_band_summary.get(
                "missing_rule_engine_confidence_row_count",
                0,
            ),
        },
        "summary": summary,
        "rule_engine_confidence_distribution": rule_engine_confidence_distribution,
        "rule_engine_confidence_overlap": rule_engine_confidence_overlap,
        "confidence_band_summary": confidence_band_summary,
        "threshold_likeness": threshold_likeness,
        "mixed_band_secondary_comparison": mixed_band_secondary_comparison,
        "final_outcome_context": final_outcome_context,
        "missingness_context": missingness_context,
        "tertiary_reason_context": tertiary_reason_context,
        "key_observations": key_observations,
    }


def build_rule_engine_confidence_distribution(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    comparison = build_numeric_field_comparison_for_fields(
        comparison_rows=comparison_rows,
        field_specs=(
            (
                _RULE_ENGINE_CONFIDENCE_FIELD,
                "Rule engine confidence",
                final_split_module._rule_engine_confidence,
            ),
        ),
    )
    field_row = _safe_dict(
        _safe_list(comparison.get("field_comparisons"))[0]
        if _safe_list(comparison.get("field_comparisons"))
        else {}
    )
    preserved_summary = _safe_dict(field_row.get(_COMPARISON_GROUP_PRESERVED))
    collapsed_summary = _safe_dict(field_row.get(_COMPARISON_GROUP_COLLAPSED))
    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "field_label": "Rule engine confidence",
        "support_status": comparison.get("support_status", "limited_support"),
        _COMPARISON_GROUP_PRESERVED: preserved_summary,
        _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
        "comparison_status": field_row.get("comparison_status", "all_missing"),
        "median_difference_preserved_minus_collapsed": field_row.get(
            "median_difference_preserved_minus_collapsed"
        ),
        "mean_difference_preserved_minus_collapsed": field_row.get(
            "mean_difference_preserved_minus_collapsed"
        ),
        "present_row_count": int(
            preserved_summary.get("present_row_count", 0) or 0
        )
        + int(collapsed_summary.get("present_row_count", 0) or 0),
        "missing_rule_engine_confidence_row_count": int(
            preserved_summary.get("missing_row_count", 0) or 0
        )
        + int(collapsed_summary.get("missing_row_count", 0) or 0),
    }


def build_rule_engine_confidence_overlap(
    *,
    distribution: dict[str, Any],
) -> dict[str, Any]:
    preserved_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_PRESERVED))
    collapsed_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_COLLAPSED))

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
            "preserved_min": preserved_min,
            "preserved_max": preserved_max,
            "collapsed_min": collapsed_min,
            "collapsed_max": collapsed_max,
            "ranges_overlap": None,
            "overlap_interval": {},
            "non_overlap_gap": None,
            "range_order": "incomplete",
            "interpretation": (
                "At least one comparison group lacks present rule_engine_confidence "
                "values, so exact range overlap cannot be characterized safely."
            ),
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
        overlap_start = round(
            max(preserved_min_value, collapsed_min_value),
            6,
        )
        overlap_end = round(
            min(preserved_max_value, collapsed_max_value),
            6,
        )
        overlap_width = round(overlap_end - overlap_start, 6)
        return {
            "support_status": "supported",
            "preserved_min": preserved_min,
            "preserved_max": preserved_max,
            "collapsed_min": collapsed_min,
            "collapsed_max": collapsed_max,
            "ranges_overlap": True,
            "overlap_interval": {
                "min": overlap_start,
                "max": overlap_end,
                "width": overlap_width,
            },
            "non_overlap_gap": 0.0,
            "range_order": "overlapping",
            "interpretation": (
                "The present preserved and collapsed rule_engine_confidence ranges "
                "overlap, so the split is not a clean exact-range threshold on this run."
            ),
        }

    if preserved_min_value > collapsed_max_value:
        gap = round(preserved_min_value - collapsed_max_value, 6)
        range_order = "collapsed_below_preserved"
        interpretation = (
            "All present collapsed rows sit below all present preserved rows on "
            "rule_engine_confidence."
        )
    else:
        gap = round(collapsed_min_value - preserved_max_value, 6)
        range_order = "preserved_below_collapsed"
        interpretation = (
            "All present preserved rows sit below all present collapsed rows on "
            "rule_engine_confidence."
        )

    return {
        "support_status": "supported",
        "preserved_min": preserved_min,
        "preserved_max": preserved_max,
        "collapsed_min": collapsed_min,
        "collapsed_max": collapsed_max,
        "ranges_overlap": False,
        "overlap_interval": {},
        "non_overlap_gap": gap,
        "range_order": range_order,
        "interpretation": interpretation,
    }


def build_confidence_band_summary(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = _comparison_group_rows(
        comparison_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_rows = _comparison_group_rows(
        comparison_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    band_rows: list[dict[str, Any]] = []
    mixed_band_labels: list[str] = []
    non_empty_band_labels: list[str] = []

    for band_label, lower_bound_exclusive, upper_bound_inclusive in _BAND_DEFINITIONS:
        band_comparison_rows = [
            row
            for row in comparison_rows
            if _band_label_for_value(row.get(_RULE_ENGINE_CONFIDENCE_FIELD))
            == band_label
        ]
        preserved_band_rows = _comparison_group_rows(
            band_comparison_rows,
            _COMPARISON_GROUP_PRESERVED,
        )
        collapsed_band_rows = _comparison_group_rows(
            band_comparison_rows,
            _COMPARISON_GROUP_COLLAPSED,
        )
        total_row_count = len(band_comparison_rows)
        preserved_row_count = len(preserved_band_rows)
        collapsed_row_count = len(collapsed_band_rows)
        band_mix_status = _band_mix_status(
            preserved_row_count=preserved_row_count,
            collapsed_row_count=collapsed_row_count,
        )
        if total_row_count > 0:
            non_empty_band_labels.append(band_label)
        if band_mix_status == "mixed":
            mixed_band_labels.append(band_label)

        band_rows.append(
            {
                "band_label": band_label,
                "lower_bound_exclusive": lower_bound_exclusive,
                "upper_bound_inclusive": upper_bound_inclusive,
                "total_row_count": total_row_count,
                "preserved_row_count": preserved_row_count,
                "collapsed_row_count": collapsed_row_count,
                "preserved_rate_within_band": _safe_ratio(
                    preserved_row_count,
                    total_row_count,
                ),
                "collapsed_rate_within_band": _safe_ratio(
                    collapsed_row_count,
                    total_row_count,
                ),
                "band_mix_status": band_mix_status,
                "support_status": _comparison_support_status(
                    baseline_row_count=preserved_row_count,
                    collapsed_row_count=collapsed_row_count,
                ),
            }
        )

    total_comparison_rows = len(comparison_rows)
    preserved_missing = sum(
        1
        for row in preserved_rows
        if row.get(_RULE_ENGINE_CONFIDENCE_FIELD) is None
    )
    collapsed_missing = sum(
        1
        for row in collapsed_rows
        if row.get(_RULE_ENGINE_CONFIDENCE_FIELD) is None
    )
    missing_total = preserved_missing + collapsed_missing

    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "bands": band_rows,
        "mixed_band_labels": mixed_band_labels,
        "non_empty_band_labels": non_empty_band_labels,
        "missing_rule_engine_confidence_row_count": missing_total,
        "missing_rule_engine_confidence_by_group": {
            _COMPARISON_GROUP_PRESERVED: preserved_missing,
            _COMPARISON_GROUP_COLLAPSED: collapsed_missing,
        },
        "missing_rule_engine_confidence_rate_within_comparison_rows": _safe_ratio(
            missing_total,
            total_comparison_rows,
        ),
    }


def build_threshold_likeness(
    *,
    summary: dict[str, Any],
    distribution: dict[str, Any],
    overlap: dict[str, Any],
    confidence_band_summary: dict[str, Any],
    mixed_band_secondary_comparison: dict[str, Any],
    final_outcome_context: dict[str, Any],
    missingness_context: dict[str, Any],
    tertiary_reason_context: dict[str, Any],
) -> dict[str, Any]:
    preserved_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_PRESERVED))
    collapsed_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_COLLAPSED))
    bands = _safe_list(confidence_band_summary.get("bands"))
    mixed_band_labels = [
        str(label) for label in confidence_band_summary.get("mixed_band_labels", [])
    ]
    non_empty_bands = [
        _safe_dict(band)
        for band in bands
        if int(_safe_dict(band).get("total_row_count", 0) or 0) > 0
    ]
    one_sided_band_count = sum(
        1
        for band in non_empty_bands
        if _safe_dict(band).get("band_mix_status")
        in {"preserved_only", "collapsed_only"}
    )

    preserved_present = int(preserved_summary.get("present_row_count", 0) or 0)
    collapsed_present = int(collapsed_summary.get("present_row_count", 0) or 0)
    total_present = preserved_present + collapsed_present
    missing_rule_engine_confidence_row_count = int(
        confidence_band_summary.get("missing_rule_engine_confidence_row_count", 0) or 0
    )
    ranges_overlap = overlap.get("ranges_overlap")
    overlap_interval = _safe_dict(overlap.get("overlap_interval"))

    classification = "inconclusive"
    explanation = (
        "The current run does not support a cleaner threshold characterization."
    )

    if total_present <= 0:
        classification = "no_rows"
        explanation = (
            "No preserved or collapsed rows in the final rule-bias-aligned slice "
            "carry rule_engine_confidence, so there is no threshold surface to inspect."
        )
    elif preserved_present <= 0 or collapsed_present <= 0:
        classification = "insufficient_support"
        explanation = (
            "Only one side of the preserved-vs-collapsed comparison has present "
            "rule_engine_confidence values, so exact threshold structure is incomplete."
        )
    elif missing_rule_engine_confidence_row_count > 0:
        classification = "inconclusive"
        explanation = (
            "Some final-slice rows are missing rule_engine_confidence, so threshold "
            "classification remains conservative even though present rows still carry "
            "descriptive signal."
        )
    elif ranges_overlap is False and not mixed_band_labels:
        classification = "clean_non_overlapping_split"
        explanation = (
            "Present preserved and collapsed rule_engine_confidence ranges do not "
            "overlap, and every populated fixed band is one-sided."
        )
    elif (
        len(mixed_band_labels) == 1
        and one_sided_band_count >= 1
        and len(non_empty_bands) >= 2
    ):
        classification = "mostly_separated_with_mixed_band"
        explanation = (
            "Most populated bands are one-sided, but one fixed confidence band still "
            "contains both preserved and collapsed rows."
        )
    elif ranges_overlap is True and mixed_band_labels and one_sided_band_count == 0:
        classification = "broadly_mixed"
        explanation = (
            "Present preserved and collapsed exact ranges overlap, and every populated "
            "fixed confidence band that contains rows is mixed."
        )
    elif len(mixed_band_labels) >= 2 or (
        ranges_overlap is True and len(non_empty_bands) >= 2
    ):
        classification = "broadly_mixed"
        explanation = (
            "Preserved and collapsed rows overlap materially across present ranges or "
            "across multiple populated bands, so a clean threshold-like split is not supported."
        )
    elif ranges_overlap is False and len(mixed_band_labels) == 1:
        classification = "mostly_separated_with_mixed_band"
        explanation = (
            "Exact present ranges do not overlap, but the fixed banding still produces "
            "one mixed band, so the pattern is only mostly separated at the report's band granularity."
        )

    confirmed_threshold_evidence: list[str] = []
    mixed_or_overlap_evidence: list[str] = []

    if ranges_overlap is False:
        if overlap.get("range_order") == "collapsed_below_preserved":
            confirmed_threshold_evidence.append(
                "Present collapsed max rule_engine_confidence is below present preserved min."
            )
        elif overlap.get("range_order") == "preserved_below_collapsed":
            confirmed_threshold_evidence.append(
                "Present preserved max rule_engine_confidence is below present collapsed min."
            )
    elif ranges_overlap is True:
        mixed_or_overlap_evidence.append(
            "Present preserved and collapsed rule_engine_confidence ranges overlap."
        )
        if overlap_interval:
            mixed_or_overlap_evidence.append(
                "Overlap interval="
                f"[{overlap_interval.get('min')}, {overlap_interval.get('max')}] "
                f"(width={overlap_interval.get('width')})."
            )

    if mixed_band_labels:
        mixed_or_overlap_evidence.append(
            "Mixed fixed bands: " + ", ".join(mixed_band_labels) + "."
        )
    elif total_present > 0:
        confirmed_threshold_evidence.append(
            "Every populated fixed confidence band is one-sided."
        )

    if missing_rule_engine_confidence_row_count > 0:
        mixed_or_overlap_evidence.append(
            "Missing rule_engine_confidence rows="
            f"{missing_rule_engine_confidence_row_count}."
        )

    remains_unproven = [
        "Final outcome fields still show where the split becomes visible, but they are contextual end-state facts rather than proof of the hidden causal merge rule.",
        "Rule_engine_confidence remains descriptive evidence only; even a clean non-overlapping split does not prove a universal internal cutoff.",
        "Reason buckets remain tertiary evidence and are not used to classify threshold structure.",
    ]
    if (
        str(missingness_context.get("explanatory_status") or "")
        != "not_explanatory_in_this_slice"
    ):
        remains_unproven.append(
            "This run contains explicit missingness in tracked fields, so missingness is surfaced as a caution rather than promoted into a root-cause claim."
        )
    if _safe_list(mixed_band_secondary_comparison.get("bands")):
        remains_unproven.append(
            "Secondary numeric differences inside mixed bands remain descriptive and do not, by themselves, prove a single universal tie-breaker."
        )

    if str(tertiary_reason_context.get("comparison_status") or "") == "non_differentiating":
        confirmed_threshold_evidence.append(
            "Reason-bucket parity does not block threshold classification."
        )

    outcome_fields = _safe_list(final_outcome_context.get("confirmed_differentiators"))
    if outcome_fields:
        mixed_or_overlap_evidence.append(
            "Final outcome context still separates groups on "
            + ", ".join(str(field) for field in outcome_fields)
            + ", but those fields are excluded from the threshold classifier."
        )

    return {
        "classification": classification,
        "support_status": distribution.get("support_status", "limited_support"),
        "classification_basis": _THRESHOLD_CLASSIFICATION_BASIS,
        "allowed_classifications": sorted(_THRESHOLD_CLASSIFICATIONS),
        "primary_field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "used_for_threshold_classification": [_RULE_ENGINE_CONFIDENCE_FIELD],
        "excluded_context_only_fields": [
            "rule_signal_state",
            "execution_signal",
            "execution_action",
            "execution_allowed",
            "combined_reason_bucket",
        ],
        "ranges_overlap": ranges_overlap,
        "mixed_band_count": len(mixed_band_labels),
        "mixed_band_labels": mixed_band_labels,
        "missing_rule_engine_confidence_row_count": missing_rule_engine_confidence_row_count,
        "preserved_present_row_count": preserved_present,
        "collapsed_present_row_count": collapsed_present,
        "preserved_total_row_count": int(
            summary.get("preserved_final_directional_outcome_row_count", 0) or 0
        ),
        "collapsed_total_row_count": int(
            summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
        ),
        "explanation": explanation,
        "confirmed_threshold_evidence": confirmed_threshold_evidence,
        "mixed_or_overlap_evidence": mixed_or_overlap_evidence,
        "remains_unproven": remains_unproven,
    }


def build_mixed_band_secondary_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    confidence_band_summary: dict[str, Any],
) -> dict[str, Any]:
    mixed_band_labels = [
        str(label) for label in confidence_band_summary.get("mixed_band_labels", [])
    ]
    band_rows: list[dict[str, Any]] = []

    for band_label in mixed_band_labels:
        band_comparison_rows = [
            row
            for row in comparison_rows
            if _band_label_for_value(row.get(_RULE_ENGINE_CONFIDENCE_FIELD))
            == band_label
        ]
        comparison = build_numeric_field_comparison_for_fields(
            comparison_rows=band_comparison_rows,
            field_specs=_SECONDARY_NUMERIC_FIELD_SPECS,
        )
        band_rows.append(
            {
                "band_label": band_label,
                "total_row_count": len(band_comparison_rows),
                "preserved_row_count": len(
                    _comparison_group_rows(
                        band_comparison_rows,
                        _COMPARISON_GROUP_PRESERVED,
                    )
                ),
                "collapsed_row_count": len(
                    _comparison_group_rows(
                        band_comparison_rows,
                        _COMPARISON_GROUP_COLLAPSED,
                    )
                ),
                "field_comparisons": _safe_list(comparison.get("field_comparisons")),
                "strongest_secondary_differentiator": _safe_dict(
                    comparison.get("strongest_numeric_differentiator")
                ),
                "non_differentiating_fields": _safe_list(
                    comparison.get("non_differentiating_fields")
                ),
                "unresolved_fields": _safe_list(comparison.get("unresolved_fields")),
            }
        )

    return {
        "mixed_band_count": len(band_rows),
        "mixed_band_labels": mixed_band_labels,
        "secondary_fields": list(_SECONDARY_NUMERIC_FIELDS),
        "bands": band_rows,
    }


def build_final_outcome_context(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    comparison = final_split_module.build_final_outcome_field_comparison(
        comparison_rows=comparison_rows
    )
    return {
        **comparison,
        "contextual_only": True,
        "used_for_threshold_classification": False,
        "note": (
            "Final outcome fields already show the preserved-vs-collapsed end-state split, "
            "but they are not used as the threshold classifier because they are visible outputs, "
            "not proof of the hidden merge rule."
        ),
    }


def build_missingness_context(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    comparison = final_split_module.build_missingness_comparison(
        comparison_rows=comparison_rows
    )
    rule_engine_missingness = _field_row(
        _safe_list(comparison.get("field_comparisons")),
        _RULE_ENGINE_CONFIDENCE_FIELD,
    )
    explanatory_status = "not_explanatory_in_this_slice"
    note = (
        "Tracked final outcome and numeric fields are fully present across preserved "
        "and collapsed rows in this slice, so missingness is not explanatory here."
    )
    if _safe_list(comparison.get("confirmed_missingness_differentiators")):
        explanatory_status = "explicit_missingness_requires_caution"
        note = (
            "This run contains explicit tracked-field missingness, so threshold "
            "classification stays conservative and missingness is surfaced as a caveat rather than a root-cause proof."
        )
    return {
        **comparison,
        "contextual_only": True,
        "used_for_threshold_classification": False,
        "explanatory_status": explanatory_status,
        "rule_engine_confidence_missingness": rule_engine_missingness,
        "note": note,
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
        "used_for_threshold_classification": False,
        "note": (
            "Combined reason bucket remains tertiary evidence only and is not required "
            "to classify the rule_engine_confidence threshold structure."
        ),
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    distribution: dict[str, Any],
    overlap: dict[str, Any],
    confidence_band_summary: dict[str, Any],
    threshold_likeness: dict[str, Any],
    final_outcome_context: dict[str, Any],
    missingness_context: dict[str, Any],
    tertiary_reason_context: dict[str, Any],
) -> dict[str, list[str]]:
    preserved_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_PRESERVED))
    collapsed_summary = _safe_dict(distribution.get(_COMPARISON_GROUP_COLLAPSED))
    mixed_band_labels = _safe_list(confidence_band_summary.get("mixed_band_labels"))

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
            "other_rule_bias_aligned_final_outcome="
            f"{summary.get('other_rule_bias_aligned_final_outcome_row_count', 0)}."
        ),
        (
            "Rule_engine_confidence preserved distribution: "
            f"min={preserved_summary.get('min')}, "
            f"median={preserved_summary.get('median')}, "
            f"mean={preserved_summary.get('mean')}, "
            f"max={preserved_summary.get('max')}."
        ),
        (
            "Rule_engine_confidence collapsed distribution: "
            f"min={collapsed_summary.get('min')}, "
            f"median={collapsed_summary.get('median')}, "
            f"mean={collapsed_summary.get('mean')}, "
            f"max={collapsed_summary.get('max')}."
        ),
        (
            "Exact range overlap="
            f"{overlap.get('ranges_overlap')}; "
            f"mixed_band_labels={mixed_band_labels}; "
            "missing_rule_engine_confidence_row_count="
            f"{confidence_band_summary.get('missing_rule_engine_confidence_row_count', 0)}."
        ),
        str(final_outcome_context.get("note") or ""),
        str(tertiary_reason_context.get("note") or ""),
        str(missingness_context.get("note") or ""),
    ]

    inferences = [
        (
            "Threshold classification: "
            f"{threshold_likeness.get('classification')} "
            f"({threshold_likeness.get('explanation')})."
        )
    ]
    if threshold_likeness.get("classification") == "clean_non_overlapping_split":
        inferences.append(
            "This run is consistent with a threshold-like descriptive split on rule_engine_confidence inside the final slice."
        )
    elif threshold_likeness.get("classification") == "mostly_separated_with_mixed_band":
        inferences.append(
            "Most of the current structure is one-sided, but at least one populated confidence band remains mixed, so the threshold claim stays partial."
        )
    elif threshold_likeness.get("classification") == "broadly_mixed":
        inferences.append(
            "The present rule_engine_confidence structure is too mixed to support a clean threshold-like reading."
        )
    elif threshold_likeness.get("classification") == "inconclusive":
        inferences.append(
            "The present data still carries descriptive signal, but overlap or missingness prevents a cleaner threshold judgment."
        )

    uncertainties = _safe_list(threshold_likeness.get("remains_unproven"))

    return {
        "facts": [item for item in facts if item],
        "inferences": [item for item in inferences if item],
        "uncertainties": uncertainties,
        "confirmed_threshold_evidence": _safe_list(
            threshold_likeness.get("confirmed_threshold_evidence")
        ),
        "mixed_or_overlap_evidence": _safe_list(
            threshold_likeness.get("mixed_or_overlap_evidence")
        ),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    threshold_likeness = _safe_dict(widest.get("threshold_likeness"))
    observations = _safe_dict(widest.get("key_observations"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "threshold_classification": threshold_likeness.get("classification"),
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "confirmed_threshold_evidence": _safe_list(
            observations.get("confirmed_threshold_evidence")
        ),
        "mixed_or_overlap_evidence": _safe_list(
            observations.get("mixed_or_overlap_evidence")
        ),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            threshold_likeness=threshold_likeness,
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
            "- actionable_selected_strategy_row_count: "
            f"{headline.get('actionable_selected_strategy_row_count', 0)}"
        )
        lines.append(
            "- fully_aligned_row_count: "
            f"{headline.get('fully_aligned_row_count', 0)}"
        )
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
            "- threshold_classification: "
            f"{headline.get('threshold_classification', 'n/a')}"
        )
        lines.append(f"- ranges_overlap: {headline.get('ranges_overlap', 'n/a')}")
        lines.append(
            "- mixed_band_labels: "
            f"{headline.get('mixed_band_labels', [])}"
        )
        lines.append(
            "- missing_rule_engine_confidence_row_count: "
            f"{headline.get('missing_rule_engine_confidence_row_count', 0)}"
        )
        for fact in _safe_list(observations.get("facts"))[:6]:
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
        f"- threshold_classification: {final_assessment.get('threshold_classification', 'n/a')}"
    )
    for item in _safe_list(final_assessment.get("confirmed_threshold_evidence"))[:6]:
        lines.append(f"- confirmed_threshold_evidence: {item}")
    for item in _safe_list(final_assessment.get("mixed_or_overlap_evidence"))[:6]:
        lines.append(f"- mixed_or_overlap_evidence: {item}")
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


def build_numeric_field_comparison_for_fields(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field_specs: Sequence[tuple[str, str, Any]],
) -> dict[str, Any]:
    preserved_rows = _comparison_group_rows(
        comparison_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_rows = _comparison_group_rows(
        comparison_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    field_rows: list[dict[str, Any]] = []
    for field, label, _ in field_specs:
        preserved_summary = final_split_module._numeric_field_summary(
            preserved_rows,
            field,
        )
        collapsed_summary = final_split_module._numeric_field_summary(
            collapsed_rows,
            field,
        )
        field_rows.append(
            {
                "field": field,
                "field_label": label,
                _COMPARISON_GROUP_PRESERVED: preserved_summary,
                _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
                "comparison_status": final_split_module._numeric_comparison_status(
                    preserved_summary=preserved_summary,
                    collapsed_summary=collapsed_summary,
                ),
                "median_difference_preserved_minus_collapsed": _difference_or_none(
                    preserved_summary.get("median"),
                    collapsed_summary.get("median"),
                ),
                "mean_difference_preserved_minus_collapsed": _difference_or_none(
                    preserved_summary.get("mean"),
                    collapsed_summary.get("mean"),
                ),
            }
        )

    field_rows.sort(
        key=lambda item: (
            final_split_module._NUMERIC_STATUS_ORDER.get(
                str(item.get("comparison_status") or ""),
                99,
            ),
            -abs(
                _to_float(
                    item.get("median_difference_preserved_minus_collapsed"),
                    default=0.0,
                )
            ),
            final_split_module._NUMERIC_FIELD_ORDER.get(
                str(item.get("field") or ""),
                99,
            ),
        )
    )
    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "field_comparisons": field_rows,
        "strongest_numeric_differentiator": final_split_module._strongest_numeric_differentiator(
            field_rows
        ),
        "non_differentiating_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "no_clear_separation"
        ],
        "unresolved_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "all_missing"
        ],
    }


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    threshold_likeness: dict[str, Any],
) -> str:
    final_split_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    classification = str(threshold_likeness.get("classification") or "inconclusive")

    if final_split_row_count <= 0:
        return (
            "No rows reached the final rule-bias-aligned slice in the widest configuration, so there is no final threshold surface to characterize."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final rule-bias-aligned slice exists, but one side of the preserved-vs-collapsed comparison is missing, so threshold characterization remains incomplete."
        )
    if classification == "clean_non_overlapping_split":
        return (
            "The widest configuration is consistent with a clean threshold-like descriptive split on persisted rule_engine_confidence inside the final fully aligned residual: present collapsed rows stay entirely on one side of present preserved rows, although that still does not prove the hidden causal merge rule."
        )
    if classification == "mostly_separated_with_mixed_band":
        return (
            "The widest configuration is mostly separated on persisted rule_engine_confidence, but at least one fixed confidence band remains mixed, so the current evidence supports only a partial threshold-like reading rather than a clean cutoff claim."
        )
    if classification == "broadly_mixed":
        return (
            "The widest configuration remains materially mixed on persisted rule_engine_confidence, so the final fully aligned residual cannot currently be described as a clean threshold-like split."
        )
    if classification == "insufficient_support":
        return (
            "The final fully aligned residual is present, but one side lacks enough present rule_engine_confidence values for a reliable threshold characterization."
        )
    return (
        "The widest configuration still points to rule_engine_confidence as a descriptive differentiator inside the final fully aligned residual, but overlap or missingness keeps the threshold interpretation inconclusive."
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
    summaries = list(configuration_summaries)
    if not summaries:
        return {}
    return max(
        summaries,
        key=lambda item: (
            int(
                _safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0
            ),
            int(_safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0),
        ),
    )


def _comparison_group_rows(
    comparison_rows: Sequence[dict[str, Any]],
    comparison_group: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in comparison_rows
        if str(row.get("comparison_group") or "") == comparison_group
    ]


def _field_row(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    for row in rows:
        item = _safe_dict(row)
        if str(item.get("field") or "") == field:
            return item
    return {}


def _band_label_for_value(value: Any) -> str | None:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return None
    for band_label, lower_bound_exclusive, upper_bound_inclusive in _BAND_DEFINITIONS:
        if lower_bound_exclusive is not None and numeric_value <= lower_bound_exclusive:
            continue
        if upper_bound_inclusive is not None and numeric_value > upper_bound_inclusive:
            continue
        return band_label
    return None


def _band_mix_status(
    *,
    preserved_row_count: int,
    collapsed_row_count: int,
) -> str:
    if preserved_row_count > 0 and collapsed_row_count > 0:
        return "mixed"
    if preserved_row_count > 0:
        return "preserved_only"
    if collapsed_row_count > 0:
        return "collapsed_only"
    return "no_rows"


def _comparison_support_status(
    *,
    baseline_row_count: int,
    collapsed_row_count: int,
) -> str:
    return str(
        final_split_module._comparison_support_status(
            baseline_row_count=baseline_row_count,
            collapsed_row_count=collapsed_row_count,
        )
    )


def _difference_or_none(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(_to_float(left, default=0.0) - _to_float(right, default=0.0), 6)


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