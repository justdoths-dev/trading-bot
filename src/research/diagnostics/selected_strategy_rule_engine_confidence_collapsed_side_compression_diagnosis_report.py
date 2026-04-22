from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_fully_aligned_final_hold_split_diagnosis_report as final_split_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report as aggregate_shape_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report as residual_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Collapsed-Side Compression Diagnosis "
    "Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = residual_module.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = residual_module.DEFAULT_OUTPUT_DIR
DEFAULT_MIN_SYMBOL_SUPPORT = residual_module.DEFAULT_MIN_SYMBOL_SUPPORT

DiagnosisConfiguration = residual_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = residual_module.DEFAULT_CONFIGURATIONS

_COMPARISON_GROUP_PRESERVED = residual_module._COMPARISON_GROUP_PRESERVED
_COMPARISON_GROUP_COLLAPSED = residual_module._COMPARISON_GROUP_COLLAPSED

_RULE_ENGINE_CONFIDENCE_FIELD = residual_module._RULE_ENGINE_CONFIDENCE_FIELD
_BASELINE_NAME = residual_module._BASELINE_NAME
_BASELINE_LABEL = residual_module._BASELINE_LABEL
_BASELINE_FORMULA = residual_module._BASELINE_FORMULA
_RESIDUAL_FIELD = residual_module._RESIDUAL_FIELD

_BUCKET_WIDTH = 0.05
_BUCKET_UPPER_BOUNDS = tuple(round(index * _BUCKET_WIDTH, 2) for index in range(1, 21))
_BUCKET_DEFINITIONS: tuple[tuple[str, float | None, float | None], ...] = tuple(
    ("<= 0.05", None, 0.05)
    if index == 0
    else (
        f"({round(upper - _BUCKET_WIDTH, 2):.2f}, {upper:.2f}]",
        round(upper - _BUCKET_WIDTH, 2),
        upper,
    )
    for index, upper in enumerate(_BUCKET_UPPER_BOUNDS)
)
_LOW_BAND_THRESHOLDS: tuple[float, ...] = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)
_LOW_REGION_MAX_THRESHOLD = 0.35
_LOW_EDGE_THRESHOLD = 0.10
_TOP_BUCKET_COUNT = 3
_TOP_EXACT_VALUE_COUNT = 3
_MIN_BUCKET_REGION_ROW_COUNT = 3
_MIN_BUCKET_REGION_COLLAPSED_ROW_COUNT = 3

_LOW_BAND_STATUS_ORDER = {
    "sharp_low_band": 0,
    "collapsed_leaning_low_band": 1,
    "mixed_or_weak": 2,
    "insufficient_data": 3,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the final fully aligned, "
            "rule-bias-aligned preserved-vs-collapsed slice and test whether the "
            "sharp split in actual rule_engine_confidence is better explained by "
            "collapsed-side output-shape compression, low-band packing, or repeated "
            "bucketed levels than by another raw-surface gate."
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
            "Retained for parity with sibling reports. The collapsed-side "
            "compression diagnosis itself reuses only the final preserved-vs-"
            "collapsed slice."
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

    result = (
        run_selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report(
            input_path=input_path,
            output_dir=output_dir,
            configurations=configurations,
            min_symbol_support=args.min_symbol_support,
            write_report_copies=args.write_latest_copy,
        )
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    best_low_band = _safe_dict(
        _safe_dict(report.get("low_band_occupancy_profile")).get(
            "best_low_band_profile"
        )
    )
    signature = _safe_dict(report.get("collapsed_side_compression_signature"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    report.get("widest_configuration")
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
                "best_low_band": {
                    "threshold_label": best_low_band.get("threshold_label"),
                    "collapsed_capture_rate": best_low_band.get(
                        "collapsed_capture_rate"
                    ),
                    "preserved_capture_rate": best_low_band.get(
                        "preserved_capture_rate"
                    ),
                    "profile_strength_status": best_low_band.get(
                        "profile_strength_status"
                    ),
                },
                "strongest_candidate_pattern": signature.get(
                    "strongest_candidate_pattern"
                ),
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report(
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
    summary = _safe_dict(widest_summary.get("summary"))
    interpretation = _safe_dict(widest_summary.get("interpretation"))

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
        "widest_configuration": _safe_dict(widest_summary.get("configuration")),
        "min_symbol_support": effective_min_symbol_support,
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
        "baseline_name": _BASELINE_NAME,
        "baseline_label": _BASELINE_LABEL,
        "baseline_formula": _BASELINE_FORMULA,
        "residual_bucket_behavior_definition": {
            "field": _RESIDUAL_FIELD,
            "baseline_name": _BASELINE_NAME,
            "formula": f"{_RULE_ENGINE_CONFIDENCE_FIELD} - {_BASELINE_NAME}",
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary_row.get("headline"))
            for summary_row in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": summary,
        "actual_rule_engine_confidence_reference": _safe_dict(
            widest_summary.get("actual_rule_engine_confidence_reference")
        ),
        "rule_engine_confidence_overlap": _safe_dict(
            widest_summary.get("rule_engine_confidence_overlap")
        ),
        "confidence_bucket_distribution": _safe_dict(
            widest_summary.get("confidence_bucket_distribution")
        ),
        "low_band_occupancy_profile": _safe_dict(
            widest_summary.get("low_band_occupancy_profile")
        ),
        "repeated_value_concentration": _safe_dict(
            widest_summary.get("repeated_value_concentration")
        ),
        "bucket_concentration_profile": _safe_dict(
            widest_summary.get("bucket_concentration_profile")
        ),
        "bucket_residual_profile": _safe_dict(
            widest_summary.get("bucket_residual_profile")
        ),
        "collapsed_side_compression_signature": _safe_dict(
            widest_summary.get("collapsed_side_compression_signature")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis chain.",
            "The report intentionally shifts away from raw persisted input surfaces and inspects only the actual output shape of persisted rule_engine_confidence inside that settled final slice.",
            "Fixed low-band thresholds, exact repeated-value counts, and 0.05-wide confidence buckets are transparent descriptive helpers only and are not promoted into production thresholds or engine rules.",
            "Bucket-level residual summaries keep the already-tested weighted_mean_setup_emphasis baseline only as carried-forward context for describing output-shape behavior, not as a new production proposal.",
            "No mapper logic, engine logic, candidate-quality-gate logic, execution-gate logic, or production defaults are modified by this report.",
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
    residual_rows = [residual_module.build_residual_row(row) for row in comparison_rows]

    summary = final_split_module.build_summary(
        actionable_rows=actionable_rows,
        fully_aligned_rows=fully_aligned_rows,
        final_split_rows=final_split_rows,
        min_symbol_support=min_symbol_support,
    )
    rule_engine_confidence_context = aggregate_shape_module.build_rule_engine_confidence_context(
        comparison_rows=residual_rows
    )
    actual_rule_engine_confidence_reference = (
        residual_module.build_actual_rule_engine_confidence_reference(
            comparison_rows=residual_rows,
            rule_engine_confidence_context=rule_engine_confidence_context,
        )
    )
    rule_engine_confidence_overlap = _safe_dict(
        rule_engine_confidence_context.get("range_overlap")
    )
    confidence_bucket_distribution = build_confidence_bucket_distribution(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    low_band_occupancy_profile = build_low_band_occupancy_profile(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    repeated_value_concentration = build_repeated_value_concentration(
        comparison_rows=residual_rows
    )
    bucket_concentration_profile = build_bucket_concentration_profile(
        confidence_bucket_distribution=confidence_bucket_distribution
    )
    bucket_residual_profile = build_bucket_residual_profile(
        comparison_rows=residual_rows
    )
    collapsed_side_compression_signature = build_collapsed_side_compression_signature(
        summary=summary,
        low_band_occupancy_profile=low_band_occupancy_profile,
        repeated_value_concentration=repeated_value_concentration,
        bucket_concentration_profile=bucket_concentration_profile,
        bucket_residual_profile=bucket_residual_profile,
    )
    interpretation = build_interpretation(
        summary=summary,
        actual_rule_engine_confidence_reference=actual_rule_engine_confidence_reference,
        rule_engine_confidence_overlap=rule_engine_confidence_overlap,
        low_band_occupancy_profile=low_band_occupancy_profile,
        repeated_value_concentration=repeated_value_concentration,
        bucket_concentration_profile=bucket_concentration_profile,
        bucket_residual_profile=bucket_residual_profile,
        collapsed_side_compression_signature=collapsed_side_compression_signature,
    )
    limitations = build_limitations(
        summary=summary,
        repeated_value_concentration=repeated_value_concentration,
        bucket_concentration_profile=bucket_concentration_profile,
        collapsed_side_compression_signature=collapsed_side_compression_signature,
    )

    best_low_band = _safe_dict(low_band_occupancy_profile.get("best_low_band_profile"))
    strongest_bucket = _safe_dict(bucket_residual_profile.get("strongest_collapsed_bucket"))
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
            "best_low_band_threshold_label": best_low_band.get("threshold_label"),
            "strongest_collapsed_bucket": strongest_bucket.get("bucket_label"),
            "strongest_candidate_pattern": collapsed_side_compression_signature.get(
                "strongest_candidate_pattern"
            ),
            "interpretation_status": interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "actual_rule_engine_confidence_reference": actual_rule_engine_confidence_reference,
        "rule_engine_confidence_overlap": rule_engine_confidence_overlap,
        "confidence_bucket_distribution": confidence_bucket_distribution,
        "low_band_occupancy_profile": low_band_occupancy_profile,
        "repeated_value_concentration": repeated_value_concentration,
        "bucket_concentration_profile": bucket_concentration_profile,
        "bucket_residual_profile": bucket_residual_profile,
        "collapsed_side_compression_signature": collapsed_side_compression_signature,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_confidence_bucket_distribution(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    distribution = _safe_dict(rule_engine_confidence_context.get("distribution"))
    preserved_present = int(
        _safe_dict(distribution.get(_COMPARISON_GROUP_PRESERVED)).get(
            "present_row_count",
            0,
        )
        or 0
    )
    collapsed_present = int(
        _safe_dict(distribution.get(_COMPARISON_GROUP_COLLAPSED)).get(
            "present_row_count",
            0,
        )
        or 0
    )
    bucket_rows: list[dict[str, Any]] = []
    for bucket_label, lower_bound_exclusive, upper_bound_inclusive in _BUCKET_DEFINITIONS:
        rows_in_bucket = _rows_in_band(
            rows=comparison_rows,
            field=_RULE_ENGINE_CONFIDENCE_FIELD,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
        preserved_rows = _comparison_group_rows(rows_in_bucket, _COMPARISON_GROUP_PRESERVED)
        collapsed_rows = _comparison_group_rows(rows_in_bucket, _COMPARISON_GROUP_COLLAPSED)
        total_row_count = len(rows_in_bucket)
        preserved_row_count = len(preserved_rows)
        collapsed_row_count = len(collapsed_rows)
        bucket_rows.append(
            {
                "bucket_label": bucket_label,
                "lower_bound_exclusive": lower_bound_exclusive,
                "upper_bound_inclusive": upper_bound_inclusive,
                "total_row_count": total_row_count,
                "preserved_row_count": preserved_row_count,
                "collapsed_row_count": collapsed_row_count,
                "preserved_rate_within_bucket": _safe_ratio(
                    preserved_row_count,
                    total_row_count,
                ),
                "collapsed_rate_within_bucket": _safe_ratio(
                    collapsed_row_count,
                    total_row_count,
                ),
                "preserved_capture_rate": _safe_ratio(
                    preserved_row_count,
                    preserved_present,
                ),
                "collapsed_capture_rate": _safe_ratio(
                    collapsed_row_count,
                    collapsed_present,
                ),
                "collapsed_minus_preserved_capture_rate": round(
                    _safe_ratio(collapsed_row_count, collapsed_present)
                    - _safe_ratio(preserved_row_count, preserved_present),
                    6,
                ),
                "bucket_mix_status": _bucket_mix_status(
                    preserved_row_count=preserved_row_count,
                    collapsed_row_count=collapsed_row_count,
                ),
                "support_status": _comparison_support_status(
                    baseline_row_count=preserved_row_count,
                    collapsed_row_count=collapsed_row_count,
                ),
            }
        )

    low_bucket_rows = [
        row
        for row in bucket_rows
        if _to_float(row.get("upper_bound_inclusive"), default=1.0)
        <= _LOW_REGION_MAX_THRESHOLD
    ]
    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "bucket_width": _BUCKET_WIDTH,
        "support_status": distribution.get("support_status", "limited_support"),
        "present_row_count": preserved_present + collapsed_present,
        "missing_rule_engine_confidence_row_count": int(
            distribution.get("missing_rule_engine_confidence_row_count", 0) or 0
        ),
        "low_region_upper_bound": _LOW_REGION_MAX_THRESHOLD,
        "preserved_low_region_capture_rate": round(
            sum(
                _to_float(row.get("preserved_capture_rate"), default=0.0)
                for row in low_bucket_rows
            ),
            6,
        ),
        "collapsed_low_region_capture_rate": round(
            sum(
                _to_float(row.get("collapsed_capture_rate"), default=0.0)
                for row in low_bucket_rows
            ),
            6,
        ),
        "dominant_collapsed_buckets": sorted(
            [row for row in bucket_rows if int(row.get("collapsed_row_count", 0) or 0) > 0],
            key=_collapsed_bucket_sort_key,
        )[:_TOP_BUCKET_COUNT],
        "buckets": bucket_rows,
    }


def build_low_band_occupancy_profile(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    distribution = _safe_dict(rule_engine_confidence_context.get("distribution"))
    preserved_present = int(
        _safe_dict(distribution.get(_COMPARISON_GROUP_PRESERVED)).get(
            "present_row_count",
            0,
        )
        or 0
    )
    collapsed_present = int(
        _safe_dict(distribution.get(_COMPARISON_GROUP_COLLAPSED)).get(
            "present_row_count",
            0,
        )
        or 0
    )

    threshold_profiles: list[dict[str, Any]] = []
    for threshold in _LOW_BAND_THRESHOLDS:
        rows_in_band = [
            row
            for row in comparison_rows
            if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
            and _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=0.0)
            <= threshold
        ]
        preserved_rows = _comparison_group_rows(rows_in_band, _COMPARISON_GROUP_PRESERVED)
        collapsed_rows = _comparison_group_rows(rows_in_band, _COMPARISON_GROUP_COLLAPSED)
        preserved_capture_rate = _safe_ratio(len(preserved_rows), preserved_present)
        collapsed_capture_rate = _safe_ratio(len(collapsed_rows), collapsed_present)
        gap = round(collapsed_capture_rate - preserved_capture_rate, 6)
        threshold_profiles.append(
            {
                "threshold": threshold,
                "threshold_label": f"{_RULE_ENGINE_CONFIDENCE_FIELD} <= {threshold:.2f}",
                "row_count": len(rows_in_band),
                "preserved_row_count": len(preserved_rows),
                "collapsed_row_count": len(collapsed_rows),
                "preserved_capture_rate": preserved_capture_rate,
                "collapsed_capture_rate": collapsed_capture_rate,
                "collapsed_minus_preserved_capture_rate": gap,
                "collapsed_rate_within_low_band": _safe_ratio(
                    len(collapsed_rows),
                    len(rows_in_band),
                ),
                "profile_shape": (
                    "narrow_low_band"
                    if threshold <= _LOW_REGION_MAX_THRESHOLD
                    else "broad_low_region"
                ),
                "profile_strength_status": _low_band_strength_status(
                    threshold=threshold,
                    preserved_capture_rate=preserved_capture_rate,
                    collapsed_capture_rate=collapsed_capture_rate,
                ),
            }
        )

    best_low_band_profile = (
        sorted(threshold_profiles, key=_low_band_sort_key)[0]
        if threshold_profiles
        else {}
    )
    interpretation_status = str(
        _safe_dict(best_low_band_profile).get("profile_strength_status")
        or "insufficient_data"
    )
    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "support_status": distribution.get("support_status", "limited_support"),
        "threshold_profiles": threshold_profiles,
        "best_low_band_profile": best_low_band_profile,
        "interpretation_status": interpretation_status,
        "interpretation": _low_band_interpretation(_safe_dict(best_low_band_profile)),
    }


def build_repeated_value_concentration(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = _comparison_group_rows(comparison_rows, _COMPARISON_GROUP_PRESERVED)
    collapsed_rows = _comparison_group_rows(comparison_rows, _COMPARISON_GROUP_COLLAPSED)

    preserved_summary = _exact_value_summary(preserved_rows)
    collapsed_summary = _exact_value_summary(collapsed_rows)

    top_3_gap = round(
        _to_float(collapsed_summary.get("top_3_value_share"), default=0.0)
        - _to_float(preserved_summary.get("top_3_value_share"), default=0.0),
        6,
    )
    repeated_row_gap = round(
        _to_float(collapsed_summary.get("repeated_value_row_share"), default=0.0)
        - _to_float(preserved_summary.get("repeated_value_row_share"), default=0.0),
        6,
    )
    support_status = _comparison_support_status(
        baseline_row_count=int(preserved_summary.get("present_row_count", 0) or 0),
        collapsed_row_count=int(collapsed_summary.get("present_row_count", 0) or 0),
    )
    concentration_status = _exact_value_concentration_status(
        preserved_summary=preserved_summary,
        collapsed_summary=collapsed_summary,
        top_3_gap=top_3_gap,
    )

    collapsed_common_value = collapsed_summary.get("most_common_value")
    preserved_share_for_collapsed_common_value = _share_for_exact_value(
        rows=preserved_rows,
        value=collapsed_common_value,
    )
    low_level_pattern_status = "no_low_level_pattern_signal"
    if support_status != "supported":
        low_level_pattern_status = "insufficient_data"
    elif (
        collapsed_common_value is not None
        and _to_float(collapsed_common_value, default=1.0) <= _LOW_EDGE_THRESHOLD
        and _to_float(collapsed_summary.get("most_common_value_share"), default=0.0)
        >= 0.25
        and preserved_share_for_collapsed_common_value <= 0.05
    ):
        low_level_pattern_status = "collapsed_flooring_or_quantization_candidate"
    elif (
        concentration_status == "collapsed_exact_values_concentrated"
        and _to_float(collapsed_common_value, default=1.0) <= _LOW_REGION_MAX_THRESHOLD
    ):
        low_level_pattern_status = "collapsed_low_level_quantization_candidate"

    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "support_status": support_status,
        "precision_decimals": None,
        _COMPARISON_GROUP_PRESERVED: preserved_summary,
        _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
        "top_3_value_share_gap_collapsed_minus_preserved": top_3_gap,
        "repeated_value_row_share_gap_collapsed_minus_preserved": repeated_row_gap,
        "concentration_status": concentration_status,
        "low_level_pattern_status": low_level_pattern_status,
        "interpretation": _exact_value_interpretation(
            concentration_status=concentration_status,
            top_3_gap=top_3_gap,
            low_level_pattern_status=low_level_pattern_status,
        ),
    }


def build_bucket_concentration_profile(
    *,
    confidence_bucket_distribution: dict[str, Any],
) -> dict[str, Any]:
    bucket_rows = _safe_list(confidence_bucket_distribution.get("buckets"))
    preserved_summary = _bucket_group_summary(
        bucket_rows=bucket_rows,
        count_key="preserved_row_count",
    )
    collapsed_summary = _bucket_group_summary(
        bucket_rows=bucket_rows,
        count_key="collapsed_row_count",
    )
    top_3_gap = round(
        _to_float(collapsed_summary.get("top_3_bucket_share"), default=0.0)
        - _to_float(preserved_summary.get("top_3_bucket_share"), default=0.0),
        6,
    )
    concentration_status = _bucket_concentration_status(
        preserved_summary=preserved_summary,
        collapsed_summary=collapsed_summary,
        top_3_gap=top_3_gap,
    )
    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "bucket_width": _BUCKET_WIDTH,
        "support_status": confidence_bucket_distribution.get(
            "support_status",
            "limited_support",
        ),
        _COMPARISON_GROUP_PRESERVED: preserved_summary,
        _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
        "top_3_bucket_share_gap_collapsed_minus_preserved": top_3_gap,
        "concentration_status": concentration_status,
        "interpretation": _bucket_concentration_interpretation(
            concentration_status=concentration_status,
            top_3_gap=top_3_gap,
        ),
    }


def build_bucket_residual_profile(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = _comparison_group_rows(comparison_rows, _COMPARISON_GROUP_PRESERVED)
    collapsed_rows = _comparison_group_rows(comparison_rows, _COMPARISON_GROUP_COLLAPSED)
    preserved_present = sum(
        1
        for row in preserved_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    )
    collapsed_present = sum(
        1
        for row in collapsed_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    )

    bucket_profiles: list[dict[str, Any]] = []
    for bucket_label, lower_bound_exclusive, upper_bound_inclusive in _BUCKET_DEFINITIONS:
        rows_in_bucket = _rows_in_band(
            rows=comparison_rows,
            field=_RULE_ENGINE_CONFIDENCE_FIELD,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
        if not rows_in_bucket:
            continue
        preserved_bucket_rows = _comparison_group_rows(
            rows_in_bucket,
            _COMPARISON_GROUP_PRESERVED,
        )
        collapsed_bucket_rows = _comparison_group_rows(
            rows_in_bucket,
            _COMPARISON_GROUP_COLLAPSED,
        )
        bucket_profiles.append(
            {
                "bucket_label": bucket_label,
                "lower_bound_exclusive": lower_bound_exclusive,
                "upper_bound_inclusive": upper_bound_inclusive,
                "total_row_count": len(rows_in_bucket),
                "preserved_row_count": len(preserved_bucket_rows),
                "collapsed_row_count": len(collapsed_bucket_rows),
                "preserved_capture_rate": _safe_ratio(
                    len(preserved_bucket_rows),
                    preserved_present,
                ),
                "collapsed_capture_rate": _safe_ratio(
                    len(collapsed_bucket_rows),
                    collapsed_present,
                ),
                "collapsed_rate_within_bucket": _safe_ratio(
                    len(collapsed_bucket_rows),
                    len(rows_in_bucket),
                ),
                "support_status": _bucket_profile_support_status(
                    total_row_count=len(rows_in_bucket),
                    collapsed_row_count=len(collapsed_bucket_rows),
                ),
                "residual_overall_summary": final_split_module._numeric_field_summary(
                    rows_in_bucket,
                    _RESIDUAL_FIELD,
                ),
                _COMPARISON_GROUP_PRESERVED: final_split_module._numeric_field_summary(
                    preserved_bucket_rows,
                    _RESIDUAL_FIELD,
                ),
                _COMPARISON_GROUP_COLLAPSED: final_split_module._numeric_field_summary(
                    collapsed_bucket_rows,
                    _RESIDUAL_FIELD,
                ),
            }
        )

    dense_low_buckets = [
        bucket
        for bucket in bucket_profiles
        if _to_float(bucket.get("upper_bound_inclusive"), default=1.0)
        <= _LOW_REGION_MAX_THRESHOLD
        and _to_float(bucket.get("collapsed_rate_within_bucket"), default=0.0) >= 0.75
        and int(bucket.get("collapsed_row_count", 0) or 0)
        >= _MIN_BUCKET_REGION_COLLAPSED_ROW_COUNT
        and int(bucket.get("total_row_count", 0) or 0) >= _MIN_BUCKET_REGION_ROW_COUNT
    ]
    dense_bucket_labels = {str(bucket.get("bucket_label") or "") for bucket in dense_low_buckets}
    dense_region_rows = [
        row
        for row in comparison_rows
        if _bucket_label_for_value(row.get(_RULE_ENGINE_CONFIDENCE_FIELD)) in dense_bucket_labels
    ]
    dense_region = {
        "bucket_labels": [bucket.get("bucket_label") for bucket in dense_low_buckets],
        "bucket_count": len(dense_low_buckets),
        "collapsed_capture_rate": _safe_ratio(
            sum(int(bucket.get("collapsed_row_count", 0) or 0) for bucket in dense_low_buckets),
            collapsed_present,
        ),
        "preserved_leakage_rate": _safe_ratio(
            sum(int(bucket.get("preserved_row_count", 0) or 0) for bucket in dense_low_buckets),
            preserved_present,
        ),
        "residual_overall_summary": final_split_module._numeric_field_summary(
            dense_region_rows,
            _RESIDUAL_FIELD,
        ),
        _COMPARISON_GROUP_PRESERVED: final_split_module._numeric_field_summary(
            _comparison_group_rows(dense_region_rows, _COMPARISON_GROUP_PRESERVED),
            _RESIDUAL_FIELD,
        ),
        _COMPARISON_GROUP_COLLAPSED: final_split_module._numeric_field_summary(
            _comparison_group_rows(dense_region_rows, _COMPARISON_GROUP_COLLAPSED),
            _RESIDUAL_FIELD,
        ),
    }
    dense_region["concentration_status"] = _dense_low_bucket_region_status(dense_region)
    dense_region["interpretation"] = _dense_low_bucket_region_interpretation(
        dense_region
    )
    strongest_collapsed_bucket = (
        sorted(bucket_profiles, key=_collapsed_bucket_sort_key)[0] if bucket_profiles else {}
    )

    return {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "residual_field": _RESIDUAL_FIELD,
        "baseline_name": _BASELINE_NAME,
        "support_status": _comparison_support_status(
            baseline_row_count=preserved_present,
            collapsed_row_count=collapsed_present,
        ),
        "bucket_profiles": bucket_profiles,
        "strongest_collapsed_bucket": strongest_collapsed_bucket,
        "collapsed_dense_low_bucket_region": dense_region,
    }


def build_collapsed_side_compression_signature(
    *,
    summary: dict[str, Any],
    low_band_occupancy_profile: dict[str, Any],
    repeated_value_concentration: dict[str, Any],
    bucket_concentration_profile: dict[str, Any],
    bucket_residual_profile: dict[str, Any],
) -> dict[str, Any]:
    comparison_support_status = str(
        summary.get("comparison_support_status") or "limited_support"
    )
    best_low_band = _safe_dict(low_band_occupancy_profile.get("best_low_band_profile"))
    low_band_status = str(
        low_band_occupancy_profile.get("interpretation_status") or "insufficient_data"
    )
    exact_value_status = str(
        repeated_value_concentration.get("concentration_status") or "insufficient_data"
    )
    bucket_status = str(
        bucket_concentration_profile.get("concentration_status") or "insufficient_data"
    )
    dense_region = _safe_dict(
        bucket_residual_profile.get("collapsed_dense_low_bucket_region")
    )
    dense_region_status = str(
        dense_region.get("concentration_status") or "insufficient_data"
    )

    interpretation_status = "comparison_unsupported"
    strongest_candidate_pattern = "comparison_unsupported"
    explanation = (
        "The final preserved-vs-collapsed comparison is below the family's normal "
        "support threshold, so collapsed-side compression claims stay withheld."
    )

    if comparison_support_status == "supported":
        if low_band_status == "sharp_low_band" and (
            bucket_status == "collapsed_bucket_concentrated"
            or exact_value_status == "collapsed_exact_values_concentrated"
            or dense_region_status == "collapsed_dense_low_bucket_region"
        ):
            interpretation_status = (
                "collapsed_side_low_band_and_bucket_concentration_supported"
            )
            strongest_candidate_pattern = (
                "collapsed_low_band_with_discrete_bucket_concentration"
            )
            explanation = (
                "Collapsed rows pack into a low rule_engine_confidence output region "
                "and also concentrate into a small set of buckets or repeated levels "
                "that preserved rows mostly avoid."
            )
        elif bucket_status == "collapsed_bucket_concentrated" or exact_value_status == (
            "collapsed_exact_values_concentrated"
        ):
            interpretation_status = (
                "collapsed_side_discrete_bucket_concentration_supported"
            )
            strongest_candidate_pattern = (
                "collapsed_side_discrete_levels_or_buckets"
            )
            explanation = (
                "Collapsed rows are disproportionately concentrated in a few actual "
                "rule_engine_confidence output levels or fixed buckets, even without "
                "a uniquely strong cumulative low-band cutoff."
            )
        elif low_band_status == "sharp_low_band":
            interpretation_status = "collapsed_side_low_band_supported"
            strongest_candidate_pattern = "collapsed_low_confidence_region_packing"
            explanation = (
                "Collapsed rows mainly pack into a low rule_engine_confidence region "
                "that preserved rows largely avoid, while discrete repeated-level "
                "evidence remains weaker."
            )
        else:
            interpretation_status = "collapsed_side_compression_inconclusive"
            strongest_candidate_pattern = "mixed_or_weak"
            explanation = (
                "Collapsed-side output-shape evidence remains mixed: neither a low-band "
                "packing pattern nor a few repeated levels/buckets clears a "
                "conservative support threshold."
            )

    return {
        "support_status": comparison_support_status,
        "strongest_candidate_pattern": strongest_candidate_pattern,
        "interpretation_status": interpretation_status,
        "collapsed_rows_pack_into_low_band": low_band_status in {
            "sharp_low_band",
            "collapsed_leaning_low_band",
        },
        "collapsed_rows_pack_into_few_fixed_buckets": bucket_status in {
            "collapsed_bucket_concentrated",
            "collapsed_bucket_leaning",
        },
        "collapsed_rows_pack_into_repeated_levels": exact_value_status in {
            "collapsed_exact_values_concentrated",
            "collapsed_exact_values_leaning",
        },
        "collapsed_rows_show_low_level_flooring_or_quantization_candidate": bool(
            repeated_value_concentration.get("low_level_pattern_status")
            in {
                "collapsed_flooring_or_quantization_candidate",
                "collapsed_low_level_quantization_candidate",
            }
        ),
        "preserved_rows_mostly_avoid_collapsed_dense_buckets": bool(
            _to_float(dense_region.get("collapsed_capture_rate"), default=0.0) >= 0.60
            and _to_float(dense_region.get("preserved_leakage_rate"), default=1.0)
            <= 0.20
        ),
        "best_low_band_profile": best_low_band,
        "dense_low_bucket_region": dense_region,
        "explanation": explanation,
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    actual_rule_engine_confidence_reference: dict[str, Any],
    rule_engine_confidence_overlap: dict[str, Any],
    low_band_occupancy_profile: dict[str, Any],
    repeated_value_concentration: dict[str, Any],
    bucket_concentration_profile: dict[str, Any],
    bucket_residual_profile: dict[str, Any],
    collapsed_side_compression_signature: dict[str, Any],
) -> dict[str, Any]:
    actual_preserved = _safe_dict(
        actual_rule_engine_confidence_reference.get(_COMPARISON_GROUP_PRESERVED)
    )
    actual_collapsed = _safe_dict(
        actual_rule_engine_confidence_reference.get(_COMPARISON_GROUP_COLLAPSED)
    )
    best_low_band = _safe_dict(low_band_occupancy_profile.get("best_low_band_profile"))
    repeated_collapsed = _safe_dict(
        repeated_value_concentration.get(_COMPARISON_GROUP_COLLAPSED)
    )
    repeated_preserved = _safe_dict(
        repeated_value_concentration.get(_COMPARISON_GROUP_PRESERVED)
    )
    bucket_collapsed = _safe_dict(
        bucket_concentration_profile.get(_COMPARISON_GROUP_COLLAPSED)
    )
    bucket_preserved = _safe_dict(
        bucket_concentration_profile.get(_COMPARISON_GROUP_PRESERVED)
    )
    strongest_bucket = _safe_dict(bucket_residual_profile.get("strongest_collapsed_bucket"))
    dense_region = _safe_dict(
        bucket_residual_profile.get("collapsed_dense_low_bucket_region")
    )
    signature = _safe_dict(collapsed_side_compression_signature)

    facts = [
        (
            "Final rule-bias-aligned comparison groups: preserved_final_directional_"
            "outcome="
            f"{summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            "collapsed_final_hold_outcome="
            f"{summary.get('collapsed_final_hold_outcome_row_count', 0)}, "
            "comparison_support_status="
            f"{summary.get('comparison_support_status', 'unknown')}."
        ),
        (
            "Actual rule_engine_confidence medians: preserved="
            f"{actual_preserved.get('median')}, collapsed={actual_collapsed.get('median')}."
        ),
        (
            "Actual rule_engine_confidence range overlap: "
            f"{rule_engine_confidence_overlap.get('interpretation')}."
        ),
    ]
    if best_low_band:
        facts.append(
            "Best cumulative low-band profile: "
            f"{best_low_band.get('threshold_label')} with "
            f"collapsed_capture_rate={best_low_band.get('collapsed_capture_rate')}, "
            f"preserved_capture_rate={best_low_band.get('preserved_capture_rate')}, "
            "collapsed_rate_within_low_band="
            f"{best_low_band.get('collapsed_rate_within_low_band')}."
        )
    if repeated_collapsed and repeated_preserved:
        facts.append(
            "Exact repeated-value concentration: collapsed top_3_value_share="
            f"{repeated_collapsed.get('top_3_value_share')}, preserved="
            f"{repeated_preserved.get('top_3_value_share')}, status="
            f"{repeated_value_concentration.get('concentration_status')}."
        )
    if bucket_collapsed and bucket_preserved:
        facts.append(
            "Fixed-bucket concentration: collapsed top_3_bucket_share="
            f"{bucket_collapsed.get('top_3_bucket_share')}, preserved="
            f"{bucket_preserved.get('top_3_bucket_share')}, status="
            f"{bucket_concentration_profile.get('concentration_status')}."
        )
    if strongest_bucket:
        facts.append(
            "Strongest collapsed fixed bucket: "
            f"{strongest_bucket.get('bucket_label')} with "
            f"collapsed_capture_rate={strongest_bucket.get('collapsed_capture_rate')} "
            "and collapsed_rate_within_bucket="
            f"{strongest_bucket.get('collapsed_rate_within_bucket')}."
        )
    if dense_region:
        facts.append(
            "Collapsed-dense low-bucket region: labels="
            f"{dense_region.get('bucket_labels', [])}, collapsed_capture_rate="
            f"{dense_region.get('collapsed_capture_rate')}, preserved_leakage_rate="
            f"{dense_region.get('preserved_leakage_rate')}, status="
            f"{dense_region.get('concentration_status')}."
        )

    observed_output_shape_behavior = [
        item
        for item in [
            _safe_dict(low_band_occupancy_profile).get("interpretation"),
            _safe_dict(repeated_value_concentration).get("interpretation"),
            _safe_dict(bucket_concentration_profile).get("interpretation"),
            _safe_dict(dense_region).get("interpretation"),
        ]
        if item
    ]

    interpretation_status = str(
        signature.get("interpretation_status") or "comparison_unsupported"
    )
    strongest_pattern = [str(signature.get("explanation") or "")]
    uncertainty = [
        "This report remains descriptive: low-band packing, repeated exact levels, or fixed-bucket concentration do not prove the literal hidden mapper or merge function.",
        "Exact repeated-value evidence can be weak when persisted floats are mostly unique, so fixed buckets remain the primary fallback surface in that case.",
        "Bucket-level residual summaries retain the already-tested weighted baseline only as context for describing output-shape behavior and do not reopen raw-surface gate tuning.",
    ]
    if interpretation_status != (
        "collapsed_side_low_band_and_bucket_concentration_supported"
    ):
        uncertainty.append(
            "Because the output-shape evidence is not jointly strong on every surface, the best reading should stay at the level of descriptive compression rather than a settled production mechanism claim."
        )

    return {
        "interpretation_status": interpretation_status,
        "facts": [item for item in facts if item],
        "observed_output_shape_behavior": observed_output_shape_behavior,
        "strongest_candidate_explanatory_pattern": [
            item for item in strongest_pattern if item
        ],
        "inference": [item for item in strongest_pattern if item],
        "uncertainty": uncertainty,
    }


def build_limitations(
    *,
    summary: dict[str, Any],
    repeated_value_concentration: dict[str, Any],
    bucket_concentration_profile: dict[str, Any],
    collapsed_side_compression_signature: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact profiles only the already-settled final preserved-vs-collapsed slice and does not reopen mapper, engine, candidate gate, or execution gate logic.",
        "Only actual persisted rule_engine_confidence output shape is profiled here; no new raw-surface threshold search is added.",
        "Repeated-value inspection stays intentionally simple and exact, so continuous-value runs may lean more heavily on fixed bucket concentration than on literal repeated floats.",
    ]
    if summary.get("comparison_support_status") != "supported":
        limitations.append(
            "The final preserved-vs-collapsed comparison is below the family's normal supported threshold, so compression claims remain provisional."
        )
    if repeated_value_concentration.get("concentration_status") == "mostly_unique":
        limitations.append(
            "Exact repeated-value concentration is weak on this run, so quantization evidence remains mostly bucket-based rather than value-level."
        )
    if bucket_concentration_profile.get("concentration_status") not in {
        "collapsed_bucket_concentrated",
        "collapsed_bucket_leaning",
    } and collapsed_side_compression_signature.get("interpretation_status") not in {
        "comparison_unsupported",
    }:
        limitations.append(
            "Fixed-bucket concentration does not isolate a strongly collapsed-dense pocket on this run, which keeps the compression read conservative."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    signature = _safe_dict(widest.get("collapsed_side_compression_signature"))
    low_band_occupancy_profile = _safe_dict(widest.get("low_band_occupancy_profile"))
    bucket_residual_profile = _safe_dict(widest.get("bucket_residual_profile"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "baseline_name": _BASELINE_NAME,
        "interpretation_status": interpretation.get("interpretation_status"),
        "strongest_candidate_pattern": signature.get("strongest_candidate_pattern"),
        "best_low_band_profile": _safe_dict(
            low_band_occupancy_profile.get("best_low_band_profile")
        ),
        "strongest_collapsed_bucket": _safe_dict(
            bucket_residual_profile.get("strongest_collapsed_bucket")
        ),
        "collapsed_dense_low_bucket_region": _safe_dict(
            bucket_residual_profile.get("collapsed_dense_low_bucket_region")
        ),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            signature=signature,
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
        interpretation = _safe_dict(_safe_dict(summary).get("interpretation"))
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
            "- best_low_band_threshold_label: "
            f"{headline.get('best_low_band_threshold_label', 'none')}"
        )
        lines.append(
            "- strongest_candidate_pattern: "
            f"{headline.get('strongest_candidate_pattern', 'unknown')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{headline.get('interpretation_status', 'unknown')}"
        )
        for fact in _safe_list(interpretation.get("facts"))[:5]:
            lines.append(f"- fact: {fact}")
        for item in _safe_list(
            interpretation.get("strongest_candidate_explanatory_pattern")
        )[:3]:
            lines.append(f"- inference: {item}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    best_low_band = _safe_dict(final_assessment.get("best_low_band_profile"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        "- strongest_candidate_pattern: "
        f"{final_assessment.get('strongest_candidate_pattern', 'unknown')}"
    )
    lines.append(
        "- best_low_band_profile: "
        f"{best_low_band.get('threshold_label', 'none')}"
    )
    for item in _safe_list(final_assessment.get("observed"))[:5]:
        lines.append(f"- observed: {item}")
    for item in _safe_list(final_assessment.get("inference"))[:3]:
        lines.append(f"- inference: {item}")
    for item in _safe_list(final_assessment.get("remains_unproven"))[:4]:
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


def _exact_value_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row[_RULE_ENGINE_CONFIDENCE_FIELD])
        for row in rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    ]
    counter = Counter(values)
    top_values = [
        {
            "value": value,
            "row_count": row_count,
            "share_within_group": _safe_ratio(row_count, len(values)),
        }
        for value, row_count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :_TOP_EXACT_VALUE_COUNT
        ]
    ]
    repeated_value_row_count = sum(row_count for row_count in counter.values() if row_count >= 2)
    repeated_value_count = sum(1 for row_count in counter.values() if row_count >= 2)
    low_value_repeated_row_count = sum(
        row_count
        for value, row_count in counter.items()
        if row_count >= 2 and value <= _LOW_REGION_MAX_THRESHOLD
    )
    most_common_value = top_values[0]["value"] if top_values else None
    most_common_value_row_count = top_values[0]["row_count"] if top_values else 0
    return {
        "present_row_count": len(values),
        "unique_value_count": len(counter),
        "unique_value_rate": _safe_ratio(len(counter), len(values)),
        "repeated_value_count": repeated_value_count,
        "repeated_value_row_count": repeated_value_row_count,
        "repeated_value_row_share": _safe_ratio(repeated_value_row_count, len(values)),
        "low_value_repeated_row_share": _safe_ratio(
            low_value_repeated_row_count,
            len(values),
        ),
        "most_common_value": most_common_value,
        "most_common_value_row_count": most_common_value_row_count,
        "most_common_value_share": _safe_ratio(most_common_value_row_count, len(values)),
        "top_3_value_share": _safe_ratio(
            sum(item["row_count"] for item in top_values),
            len(values),
        ),
        "top_values": top_values,
    }


def _bucket_group_summary(
    *,
    bucket_rows: Sequence[dict[str, Any]],
    count_key: str,
) -> dict[str, Any]:
    populated_rows = [
        {
            "bucket_label": str(_safe_dict(row).get("bucket_label") or ""),
            "row_count": int(_safe_dict(row).get(count_key, 0) or 0),
        }
        for row in bucket_rows
        if int(_safe_dict(row).get(count_key, 0) or 0) > 0
    ]
    populated_rows.sort(key=lambda row: (-row["row_count"], row["bucket_label"]))
    present_row_count = sum(row["row_count"] for row in populated_rows)
    top_buckets = [
        {
            **row,
            "share_within_group": _safe_ratio(row["row_count"], present_row_count),
        }
        for row in populated_rows[:_TOP_BUCKET_COUNT]
    ]
    return {
        "present_row_count": present_row_count,
        "non_empty_bucket_count": len(populated_rows),
        "most_common_bucket_label": top_buckets[0]["bucket_label"] if top_buckets else None,
        "most_common_bucket_row_count": top_buckets[0]["row_count"] if top_buckets else 0,
        "most_common_bucket_share": (
            top_buckets[0]["share_within_group"] if top_buckets else 0.0
        ),
        "top_3_bucket_share": _safe_ratio(
            sum(bucket["row_count"] for bucket in top_buckets),
            present_row_count,
        ),
        "top_buckets": top_buckets,
    }


def _low_band_strength_status(
    *,
    threshold: float,
    preserved_capture_rate: float,
    collapsed_capture_rate: float,
) -> str:
    gap = round(collapsed_capture_rate - preserved_capture_rate, 6)
    if collapsed_capture_rate <= 0 and preserved_capture_rate <= 0:
        return "insufficient_data"
    if (
        collapsed_capture_rate >= 0.75
        and preserved_capture_rate <= 0.30
        and gap >= 0.45
        and threshold <= 0.45
    ):
        return "sharp_low_band"
    if (
        collapsed_capture_rate >= 0.60
        and preserved_capture_rate <= 0.45
        and gap >= 0.25
    ):
        return "collapsed_leaning_low_band"
    return "mixed_or_weak"


def _exact_value_concentration_status(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
    top_3_gap: float,
) -> str:
    preserved_present = int(preserved_summary.get("present_row_count", 0) or 0)
    collapsed_present = int(collapsed_summary.get("present_row_count", 0) or 0)
    if preserved_present <= 0 or collapsed_present <= 0:
        return "insufficient_data"
    if (
        int(collapsed_summary.get("unique_value_count", 0) or 0) <= 3
        and _to_float(collapsed_summary.get("top_3_value_share"), default=0.0) >= 0.75
        and (
            top_3_gap >= 0.20
            or int(preserved_summary.get("unique_value_count", 0) or 0)
            >= int(collapsed_summary.get("unique_value_count", 0) or 0) + 3
        )
    ):
        return "collapsed_exact_values_concentrated"
    if (
        top_3_gap >= 0.15
        and int(collapsed_summary.get("unique_value_count", 0) or 0)
        < int(preserved_summary.get("unique_value_count", 0) or 0)
    ):
        return "collapsed_exact_values_leaning"
    if (
        _to_float(preserved_summary.get("repeated_value_row_share"), default=0.0) == 0.0
        and _to_float(collapsed_summary.get("repeated_value_row_share"), default=0.0)
        == 0.0
    ):
        return "mostly_unique"
    return "mixed"


def _bucket_concentration_status(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
    top_3_gap: float,
) -> str:
    preserved_present = int(preserved_summary.get("present_row_count", 0) or 0)
    collapsed_present = int(collapsed_summary.get("present_row_count", 0) or 0)
    if preserved_present <= 0 or collapsed_present <= 0:
        return "insufficient_data"
    if (
        int(collapsed_summary.get("non_empty_bucket_count", 0) or 0) <= 3
        and _to_float(collapsed_summary.get("top_3_bucket_share"), default=0.0) >= 0.75
        and (
            top_3_gap >= 0.20
            or int(preserved_summary.get("non_empty_bucket_count", 0) or 0)
            >= int(collapsed_summary.get("non_empty_bucket_count", 0) or 0) + 2
        )
    ):
        return "collapsed_bucket_concentrated"
    if (
        top_3_gap >= 0.15
        and int(collapsed_summary.get("non_empty_bucket_count", 0) or 0)
        < int(preserved_summary.get("non_empty_bucket_count", 0) or 0)
    ):
        return "collapsed_bucket_leaning"
    return "mixed"


def _dense_low_bucket_region_status(dense_region: dict[str, Any]) -> str:
    bucket_count = int(dense_region.get("bucket_count", 0) or 0)
    if bucket_count <= 0:
        return "absent"
    collapsed_capture_rate = _to_float(
        dense_region.get("collapsed_capture_rate"),
        default=0.0,
    )
    preserved_leakage_rate = _to_float(
        dense_region.get("preserved_leakage_rate"),
        default=1.0,
    )
    if collapsed_capture_rate >= 0.60 and preserved_leakage_rate <= 0.20:
        return "collapsed_dense_low_bucket_region"
    if collapsed_capture_rate - preserved_leakage_rate >= 0.20:
        return "collapsed_low_bucket_leaning"
    return "mixed"


def _low_band_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _LOW_BAND_STATUS_ORDER.get(str(row.get("profile_strength_status") or ""), 99),
        -_to_float(row.get("collapsed_minus_preserved_capture_rate"), default=0.0),
        -_to_float(row.get("collapsed_capture_rate"), default=0.0),
        _to_float(row.get("preserved_capture_rate"), default=1.0),
        _to_float(row.get("threshold"), default=99.0),
    )


def _collapsed_bucket_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    row_dict = _safe_dict(row)
    return (
        -_to_float(row_dict.get("collapsed_capture_rate"), default=0.0),
        -_to_float(row_dict.get("collapsed_rate_within_bucket"), default=0.0),
        -int(row_dict.get("collapsed_row_count", 0) or 0),
        _to_float(row_dict.get("upper_bound_inclusive"), default=99.0),
        str(row_dict.get("bucket_label") or ""),
    )


def _low_band_interpretation(best_low_band_profile: dict[str, Any]) -> str:
    if not best_low_band_profile:
        return "No cumulative low-band profile is available."
    return (
        f"{best_low_band_profile.get('threshold_label')} captures "
        f"{best_low_band_profile.get('collapsed_capture_rate')} of collapsed rows and "
        f"{best_low_band_profile.get('preserved_capture_rate')} of preserved rows "
        f"({best_low_band_profile.get('profile_strength_status')})."
    )


def _exact_value_interpretation(
    *,
    concentration_status: str,
    top_3_gap: float,
    low_level_pattern_status: str,
) -> str:
    return (
        "Exact repeated-value concentration status="
        f"{concentration_status}, top_3_gap={top_3_gap}, "
        f"low_level_pattern_status={low_level_pattern_status}."
    )


def _bucket_concentration_interpretation(
    *,
    concentration_status: str,
    top_3_gap: float,
) -> str:
    return (
        "Fixed-bucket concentration status="
        f"{concentration_status}, top_3_gap={top_3_gap}."
    )


def _dense_low_bucket_region_interpretation(dense_region: dict[str, Any]) -> str:
    status = str(dense_region.get("concentration_status") or "absent")
    if status == "absent":
        return "No low fixed bucket is both collapsed-dense and non-trivial on this run."
    return (
        "Low fixed bucket region status="
        f"{status}, labels={dense_region.get('bucket_labels', [])}, "
        "collapsed_capture_rate="
        f"{dense_region.get('collapsed_capture_rate')}, preserved_leakage_rate="
        f"{dense_region.get('preserved_leakage_rate')}."
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    signature: dict[str, Any],
) -> str:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    interpretation_status = str(
        signature.get("interpretation_status") or "comparison_unsupported"
    )

    if final_slice_row_count <= 0:
        return (
            "No rows reached the final fully aligned plus rule-bias-aligned slice, so "
            "collapsed-side compression diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final preserved-vs-collapsed slice exists, but one side is missing, "
            "so collapsed-side compression diagnosis remains incomplete."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but the preserved-vs-collapsed comparison does "
            "not clear the family's normal support threshold, so output-shape claims "
            "stay withheld."
        )
    if interpretation_status == (
        "collapsed_side_low_band_and_bucket_concentration_supported"
    ):
        return (
            "The strongest current reading is collapsed-side output compression: "
            "collapsed rows pack into a low rule_engine_confidence region and a few "
            "discrete buckets or repeated levels that preserved rows mostly avoid."
        )
    if interpretation_status == (
        "collapsed_side_discrete_bucket_concentration_supported"
    ):
        return (
            "The strongest current reading is discrete collapsed-side bucket or level "
            "concentration in actual rule_engine_confidence, even without a single "
            "dominant cumulative low-band cutoff."
        )
    if interpretation_status == "collapsed_side_low_band_supported":
        return (
            "The strongest current reading is low-band packing on the collapsed side, "
            "but repeated-level and bucket-level concentration evidence stays weaker."
        )
    return (
        "Actual rule_engine_confidence still shows some output-shape structure, but "
        "collapsed-side compression evidence remains too mixed to promote a single "
        "supported mechanism claim."
    )


def _share_for_exact_value(*, rows: Sequence[dict[str, Any]], value: Any) -> float:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return 0.0
    exact_values = [
        float(row[_RULE_ENGINE_CONFIDENCE_FIELD])
        for row in rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    ]
    return _safe_ratio(
        sum(1 for exact_value in exact_values if exact_value == numeric_value),
        len(exact_values),
    )


def _bucket_label_for_value(value: Any) -> str | None:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return None
    for bucket_label, lower_bound_exclusive, upper_bound_inclusive in _BUCKET_DEFINITIONS:
        if _value_in_band(
            value=numeric_value,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        ):
            return bucket_label
    return None


def _rows_in_band(
    *,
    rows: Sequence[dict[str, Any]],
    field: str,
    lower_bound_exclusive: float | None,
    upper_bound_inclusive: float | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _value_in_band(
            value=row.get(field),
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
    ]


def _value_in_band(
    *,
    value: Any,
    lower_bound_exclusive: float | None,
    upper_bound_inclusive: float | None,
) -> bool:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return False
    if lower_bound_exclusive is not None and numeric_value <= lower_bound_exclusive:
        return False
    if upper_bound_inclusive is not None and numeric_value > upper_bound_inclusive:
        return False
    return True


def _bucket_mix_status(
    *,
    preserved_row_count: int,
    collapsed_row_count: int,
) -> str:
    if preserved_row_count <= 0 and collapsed_row_count <= 0:
        return "empty"
    if preserved_row_count > 0 and collapsed_row_count > 0:
        return "mixed"
    if preserved_row_count > 0:
        return "preserved_only"
    return "collapsed_only"


def _bucket_profile_support_status(
    *,
    total_row_count: int,
    collapsed_row_count: int,
) -> str:
    if (
        total_row_count >= _MIN_BUCKET_REGION_ROW_COUNT
        and collapsed_row_count >= _MIN_BUCKET_REGION_COLLAPSED_ROW_COUNT
    ):
        return "supported"
    if total_row_count > 0:
        return "limited_support"
    return "insufficient_data"


def _resolve_path(path: Path) -> Path:
    return residual_module._resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return residual_module._parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return residual_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    return residual_module._build_stage_row(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    return residual_module._build_activation_gap_row(row)


def _build_fully_aligned_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return residual_module._build_fully_aligned_row(row)


def _build_final_split_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return residual_module._build_final_split_row(row)


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return residual_module._widest_configuration_summary(configuration_summaries)


def _comparison_group_rows(
    rows: Sequence[dict[str, Any]],
    comparison_group: str,
) -> list[dict[str, Any]]:
    return residual_module._comparison_group_rows(rows, comparison_group)


def _comparison_support_status(
    *,
    baseline_row_count: int,
    collapsed_row_count: int,
) -> str:
    return residual_module._comparison_support_status(
        baseline_row_count=baseline_row_count,
        collapsed_row_count=collapsed_row_count,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return residual_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return residual_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return residual_module._safe_list(value)


if __name__ == "__main__":
    main()
