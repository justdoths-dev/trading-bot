from __future__ import annotations

import argparse
import json
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
    "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Weighted Aggregate Residual Threshold Profile Diagnosis Report"
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
_BASELINE_COMPONENT_FIELDS = residual_module._BASELINE_COMPONENT_FIELDS
_CONTEXT_BIAS_FAMILY_FIELD = residual_module._CONTEXT_BIAS_FAMILY_FIELD
_RESIDUAL_FIELD = residual_module._RESIDUAL_FIELD

_SHARPLY_NEGATIVE_RESIDUAL_THRESHOLD = (
    residual_module._STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD
)
_MIN_PROFILE_ROW_COUNT = residual_module._MIN_REGIME_ROW_COUNT
_MIN_PROFILE_COLLAPSED_ROW_COUNT = residual_module._MIN_REGIME_COLLAPSED_ROW_COUNT

_RAW_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
)
_SHORTFALL_THRESHOLDS: tuple[float, ...] = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
)

_RAW_CONFIDENCE_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<= 0.45", None, 0.45),
    ("(0.45, 0.55]", 0.45, 0.55),
    ("(0.55, 0.65]", 0.55, 0.65),
    ("(0.65, 0.75]", 0.65, 0.75),
    ("> 0.75", 0.75, None),
)
_SHORTFALL_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<= 0.20", None, 0.20),
    ("(0.20, 0.30]", 0.20, 0.30),
    ("(0.30, 0.40]", 0.30, 0.40),
    ("(0.40, 0.50]", 0.40, 0.50),
    ("> 0.50", 0.50, None),
)

_AXIS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "field": _CONTEXT_BIAS_FAMILY_FIELD,
        "field_label": "Context/bias family mean",
        "transform": "raw_confidence",
        "threshold_operator": "<=",
        "threshold_grid": _RAW_CONFIDENCE_THRESHOLDS,
        "band_definitions": _RAW_CONFIDENCE_BANDS,
        "extreme_band_side": "low",
    },
    {
        "field": "setup_layer_confidence",
        "field_label": "Setup layer confidence",
        "transform": "raw_confidence",
        "threshold_operator": "<=",
        "threshold_grid": _RAW_CONFIDENCE_THRESHOLDS,
        "band_definitions": _RAW_CONFIDENCE_BANDS,
        "extreme_band_side": "low",
    },
    {
        "field": "selected_strategy_confidence",
        "field_label": "Selected-strategy confidence",
        "transform": "raw_confidence",
        "threshold_operator": "<=",
        "threshold_grid": _RAW_CONFIDENCE_THRESHOLDS,
        "band_definitions": _RAW_CONFIDENCE_BANDS,
        "extreme_band_side": "low",
    },
    {
        "field": "context_bias_family_shortfall",
        "field_label": "Context/bias family shortfall",
        "transform": "shortfall",
        "threshold_operator": ">=",
        "threshold_grid": _SHORTFALL_THRESHOLDS,
        "band_definitions": _SHORTFALL_BANDS,
        "extreme_band_side": "high",
    },
    {
        "field": "setup_shortfall",
        "field_label": "Setup shortfall",
        "transform": "shortfall",
        "threshold_operator": ">=",
        "threshold_grid": _SHORTFALL_THRESHOLDS,
        "band_definitions": _SHORTFALL_BANDS,
        "extreme_band_side": "high",
    },
    {
        "field": "selected_strategy_shortfall",
        "field_label": "Selected-strategy shortfall",
        "transform": "shortfall",
        "threshold_operator": ">=",
        "threshold_grid": _SHORTFALL_THRESHOLDS,
        "band_definitions": _SHORTFALL_BANDS,
        "extreme_band_side": "high",
    },
)

_AXIS_ORDER = {
    str(spec["field"]): index for index, spec in enumerate(_AXIS_SPECS)
}
_PROFILE_SUPPORT_ORDER = {
    "strong_support": 0,
    "weak_support": 1,
    "limited_support": 2,
    "insufficient_data": 3,
}
_SAMPLE_SUPPORT_ORDER = {
    "supported": 0,
    "limited_support": 1,
    "insufficient_data": 2,
}
_TOP_AXIS_BAND_PROFILE_COUNT = 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only threshold-profile report for the final fully "
            "aligned, rule-bias-aligned preserved-vs-collapsed slice using the "
            "confirmed weighted_mean_setup_emphasis residual baseline and a fixed "
            "sharply negative residual pocket definition."
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
            "Retained for architectural parity with sibling reports. Threshold "
            "profiling itself uses only the final preserved-vs-collapsed slice."
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
        run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report(
            input_path=input_path,
            output_dir=output_dir,
            configurations=configurations,
            min_symbol_support=args.min_symbol_support,
            write_report_copies=args.write_latest_copy,
        )
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    best_axis = _safe_dict(report.get("best_threshold_axis"))
    best_profile = _safe_dict(report.get("best_threshold_profile"))

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
                "baseline_name": report.get("baseline_name"),
                "residual_pocket_definition": report.get("residual_pocket_definition"),
                "best_threshold_axis": {
                    "field": best_axis.get("field"),
                    "transform": best_axis.get("transform"),
                    "threshold": best_profile.get("threshold"),
                    "threshold_label": best_profile.get("threshold_label"),
                    "profile_strength_status": best_axis.get(
                        "profile_strength_status"
                    ),
                },
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report(
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
        "baseline_name": _BASELINE_NAME,
        "baseline_label": _BASELINE_LABEL,
        "baseline_formula": _BASELINE_FORMULA,
        "residual_pocket_definition": {
            "field": _RESIDUAL_FIELD,
            "operator": "<=",
            "threshold": _SHARPLY_NEGATIVE_RESIDUAL_THRESHOLD,
            "label": "sharply_negative_residual_pocket",
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary.get("headline")) for summary in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": _safe_dict(widest_summary.get("summary")),
        "baseline_reference": _safe_dict(widest_summary.get("baseline_reference")),
        "actual_rule_engine_confidence_reference": _safe_dict(
            widest_summary.get("actual_rule_engine_confidence_reference")
        ),
        "residual_class_comparison": _safe_dict(
            widest_summary.get("residual_class_comparison")
        ),
        "residual_sign_distribution": _safe_dict(
            widest_summary.get("residual_sign_distribution")
        ),
        "threshold_profile_axes": _safe_list(
            widest_summary.get("threshold_profile_axes")
        ),
        "best_threshold_axis": _safe_dict(widest_summary.get("best_threshold_axis")),
        "best_threshold_profile": _safe_dict(
            widest_summary.get("best_threshold_profile")
        ),
        "axis_rankings": _safe_list(widest_summary.get("axis_rankings")),
        "top_axis_band_profiles": _safe_list(
            widest_summary.get("top_axis_band_profiles")
        ),
        "selected_strategy_value_check": _safe_dict(
            widest_summary.get("selected_strategy_value_check")
        ),
        "gate_vs_severity_check": _safe_dict(
            widest_summary.get("gate_vs_severity_check")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis family.",
            "Residual remains fixed to rule_engine_confidence minus weighted_mean_setup_emphasis so the threshold-profile work starts from the already-confirmed weighted aggregate residual context rather than reopening aggregate-family exploration.",
            "The sharply negative residual pocket stays fixed at residual <= -0.15 for continuity with the previous diagnosis stage.",
            "Single-axis threshold grids and ordered bands are intentionally fixed and transparent rather than optimized for this run.",
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
    rule_engine_confidence_context = (
        aggregate_shape_module.build_rule_engine_confidence_context(
            comparison_rows=residual_rows
        )
    )
    actual_rule_engine_confidence_reference = (
        residual_module.build_actual_rule_engine_confidence_reference(
            comparison_rows=residual_rows,
            rule_engine_confidence_context=rule_engine_confidence_context,
        )
    )
    baseline_reference = residual_module.build_baseline_reference(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
        actual_rule_engine_confidence_reference=actual_rule_engine_confidence_reference,
    )
    residual_class_comparison = residual_module.build_residual_class_comparison(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    residual_sign_distribution = residual_module.build_residual_sign_distribution(
        comparison_rows=residual_rows
    )
    threshold_profile_axes = build_threshold_profile_axes(comparison_rows=residual_rows)
    axis_rankings = build_axis_rankings(threshold_profile_axes=threshold_profile_axes)
    best_threshold_axis = _safe_dict(axis_rankings[0] if axis_rankings else {})
    best_threshold_profile = _safe_dict(best_threshold_axis.get("best_threshold_profile"))
    top_axis_band_profiles = build_top_axis_band_profiles(
        comparison_rows=residual_rows,
        axis_rankings=axis_rankings,
    )
    selected_strategy_value_check = build_selected_strategy_value_check(
        axis_rankings=axis_rankings
    )
    gate_vs_severity_check = build_gate_vs_severity_check(
        comparison_rows=residual_rows,
        axis_rankings=axis_rankings,
    )
    interpretation = build_interpretation(
        summary=summary,
        baseline_reference=baseline_reference,
        residual_class_comparison=residual_class_comparison,
        residual_sign_distribution=residual_sign_distribution,
        best_threshold_axis=best_threshold_axis,
        best_threshold_profile=best_threshold_profile,
        top_axis_band_profiles=top_axis_band_profiles,
        selected_strategy_value_check=selected_strategy_value_check,
        gate_vs_severity_check=gate_vs_severity_check,
    )
    limitations = build_limitations(
        summary=summary,
        best_threshold_axis=best_threshold_axis,
        top_axis_band_profiles=top_axis_band_profiles,
        gate_vs_severity_check=gate_vs_severity_check,
    )

    best_band_profile = _safe_dict(
        next(
            (
                row
                for row in top_axis_band_profiles
                if str(row.get("field") or "") == str(best_threshold_axis.get("field") or "")
            ),
            {},
        )
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
            "best_threshold_axis": best_threshold_axis.get("field"),
            "best_threshold_transform": best_threshold_axis.get("transform"),
            "best_threshold_label": best_threshold_profile.get("threshold_label"),
            "best_threshold_profile_strength_status": best_threshold_axis.get(
                "profile_strength_status"
            ),
            "best_axis_band_profile_shape": best_band_profile.get("profile_shape"),
            "interpretation_status": interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "baseline_reference": baseline_reference,
        "actual_rule_engine_confidence_reference": actual_rule_engine_confidence_reference,
        "residual_class_comparison": residual_class_comparison,
        "residual_sign_distribution": residual_sign_distribution,
        "threshold_profile_axes": threshold_profile_axes,
        "best_threshold_axis": best_threshold_axis,
        "best_threshold_profile": best_threshold_profile,
        "axis_rankings": axis_rankings,
        "top_axis_band_profiles": top_axis_band_profiles,
        "selected_strategy_value_check": selected_strategy_value_check,
        "gate_vs_severity_check": gate_vs_severity_check,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_threshold_profile_axes(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    axis_rows: list[dict[str, Any]] = []
    for axis_spec in _AXIS_SPECS:
        field = str(axis_spec["field"])
        axis_rows.append(
            build_axis_threshold_profile(
                comparison_rows=comparison_rows,
                field=field,
                field_label=str(axis_spec["field_label"]),
                transform=str(axis_spec["transform"]),
                threshold_operator=str(axis_spec["threshold_operator"]),
                threshold_grid=tuple(axis_spec["threshold_grid"]),
            )
        )
    return axis_rows


def build_axis_threshold_profile(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field: str,
    field_label: str,
    transform: str,
    threshold_operator: str,
    threshold_grid: Sequence[float],
) -> dict[str, Any]:
    present_rows = _axis_present_rows(comparison_rows=comparison_rows, field=field)
    preserved_present_rows = _comparison_group_rows(
        present_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_present_rows = _comparison_group_rows(
        present_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    threshold_profile_rows: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        threshold_profile_rows.append(
            build_threshold_profile_row(
                present_rows=present_rows,
                field=field,
                field_label=field_label,
                transform=transform,
                threshold_operator=threshold_operator,
                threshold=threshold,
                preserved_present_rows=preserved_present_rows,
                collapsed_present_rows=collapsed_present_rows,
            )
        )

    ranked_threshold_rows = sorted(
        threshold_profile_rows,
        key=lambda row: _threshold_profile_sort_key(
            row=row,
            threshold_operator=threshold_operator,
        ),
    )
    best_threshold_profile = _safe_dict(
        ranked_threshold_rows[0] if ranked_threshold_rows else {}
    )
    return {
        "field": field,
        "field_label": field_label,
        "transform": transform,
        "threshold_operator": threshold_operator,
        "threshold_grid": [round(float(value), 2) for value in threshold_grid],
        "present_row_count": len(present_rows),
        "missing_row_count": len(comparison_rows) - len(present_rows),
        "preserved_present_row_count": len(preserved_present_rows),
        "collapsed_present_row_count": len(collapsed_present_rows),
        "support_status": best_threshold_profile.get("support_status", "insufficient_data"),
        "profile_strength_status": best_threshold_profile.get(
            "profile_strength_status",
            "insufficient_data",
        ),
        "best_threshold_profile": best_threshold_profile,
        "threshold_profile_rows": threshold_profile_rows,
    }


def build_threshold_profile_row(
    *,
    present_rows: Sequence[dict[str, Any]],
    field: str,
    field_label: str,
    transform: str,
    threshold_operator: str,
    threshold: float,
    preserved_present_rows: Sequence[dict[str, Any]],
    collapsed_present_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    inside_rows = [
        row
        for row in present_rows
        if _matches_threshold(
            value=row.get(field),
            threshold_operator=threshold_operator,
            threshold=threshold,
        )
    ]
    outside_rows = [
        row
        for row in present_rows
        if not _matches_threshold(
            value=row.get(field),
            threshold_operator=threshold_operator,
            threshold=threshold,
        )
    ]
    metrics = _build_profile_metrics(
        inside_rows=inside_rows,
        outside_rows=outside_rows,
        preserved_present_rows=preserved_present_rows,
        collapsed_present_rows=collapsed_present_rows,
    )
    return {
        "field": field,
        "field_label": field_label,
        "transform": transform,
        "threshold_operator": threshold_operator,
        "threshold": round(float(threshold), 2),
        "threshold_label": (
            f"{field} {threshold_operator} {float(threshold):.2f}"
        ),
        **metrics,
    }


def build_axis_rankings(
    *,
    threshold_profile_axes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranking_rows: list[dict[str, Any]] = []
    for axis_row in threshold_profile_axes:
        axis = _safe_dict(axis_row)
        best_threshold_profile = _safe_dict(axis.get("best_threshold_profile"))
        ranking_rows.append(
            {
                "field": axis.get("field"),
                "field_label": axis.get("field_label"),
                "transform": axis.get("transform"),
                "threshold_operator": axis.get("threshold_operator"),
                "present_row_count": axis.get("present_row_count", 0),
                "missing_row_count": axis.get("missing_row_count", 0),
                "preserved_present_row_count": axis.get(
                    "preserved_present_row_count",
                    0,
                ),
                "collapsed_present_row_count": axis.get(
                    "collapsed_present_row_count",
                    0,
                ),
                "support_status": axis.get("support_status", "insufficient_data"),
                "profile_strength_status": axis.get(
                    "profile_strength_status",
                    "insufficient_data",
                ),
                "best_threshold_value": best_threshold_profile.get("threshold"),
                "best_threshold_label": best_threshold_profile.get("threshold_label"),
                "collapsed_capture_rate": best_threshold_profile.get(
                    "collapsed_capture_rate"
                ),
                "preserved_leakage_rate": best_threshold_profile.get(
                    "preserved_leakage_rate"
                ),
                "capture_minus_leakage": best_threshold_profile.get(
                    "capture_minus_leakage"
                ),
                "collapsed_pocket_capture_rate": best_threshold_profile.get(
                    "collapsed_pocket_capture_rate"
                ),
                "inside_pocket_concentration_gap": best_threshold_profile.get(
                    "inside_pocket_concentration_gap"
                ),
                "collapsed_pocket_lift": best_threshold_profile.get(
                    "collapsed_pocket_lift"
                ),
                "best_threshold_profile": best_threshold_profile,
            }
        )

    ranking_rows.sort(key=_axis_ranking_sort_key)
    return ranking_rows


def build_top_axis_band_profiles(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    axis_rankings: Sequence[dict[str, Any]],
    count: int = _TOP_AXIS_BAND_PROFILE_COUNT,
) -> list[dict[str, Any]]:
    top_rankings = list(axis_rankings)[: max(1, int(count))]
    band_profiles: list[dict[str, Any]] = []
    for ranking in top_rankings:
        field = str(_safe_dict(ranking).get("field") or "")
        axis_spec = _axis_spec(field)
        if not axis_spec:
            continue
        band_profiles.append(
            build_axis_band_profile(
                comparison_rows=comparison_rows,
                field=field,
                field_label=str(axis_spec["field_label"]),
                transform=str(axis_spec["transform"]),
                threshold_operator=str(axis_spec["threshold_operator"]),
                band_definitions=tuple(axis_spec["band_definitions"]),
                extreme_band_side=str(axis_spec["extreme_band_side"]),
            )
        )
    return band_profiles


def build_axis_band_profile(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field: str,
    field_label: str,
    transform: str,
    threshold_operator: str,
    band_definitions: Sequence[tuple[str, float | None, float | None]],
    extreme_band_side: str,
) -> dict[str, Any]:
    present_rows = _axis_present_rows(comparison_rows=comparison_rows, field=field)
    preserved_present_rows = _comparison_group_rows(
        present_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_present_rows = _comparison_group_rows(
        present_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    band_rows: list[dict[str, Any]] = []
    for band_label, lower_bound_exclusive, upper_bound_inclusive in band_definitions:
        inside_rows = [
            row
            for row in present_rows
            if _value_in_band(
                value=row.get(field),
                lower_bound_exclusive=lower_bound_exclusive,
                upper_bound_inclusive=upper_bound_inclusive,
            )
        ]
        outside_rows = [
            row
            for row in present_rows
            if not _value_in_band(
                value=row.get(field),
                lower_bound_exclusive=lower_bound_exclusive,
                upper_bound_inclusive=upper_bound_inclusive,
            )
        ]
        metrics = _build_profile_metrics(
            inside_rows=inside_rows,
            outside_rows=outside_rows,
            preserved_present_rows=preserved_present_rows,
            collapsed_present_rows=collapsed_present_rows,
        )
        band_rows.append(
            {
                "band_label": band_label,
                "lower_bound_exclusive": lower_bound_exclusive,
                "upper_bound_inclusive": upper_bound_inclusive,
                **metrics,
            }
        )

    populated_band_rows = [
        row for row in band_rows if int(row.get("inside_row_count", 0) or 0) > 0
    ]
    strongest_band = _safe_dict(
        sorted(populated_band_rows, key=_band_profile_sort_key)[0]
        if populated_band_rows
        else {}
    )
    profile_shape, shape_reason = _classify_band_profile(
        band_rows=band_rows,
        extreme_band_side=extreme_band_side,
    )
    return {
        "field": field,
        "field_label": field_label,
        "transform": transform,
        "threshold_operator": threshold_operator,
        "support_status": strongest_band.get("support_status", "insufficient_data"),
        "profile_shape": profile_shape,
        "shape_reason": shape_reason,
        "strongest_band": strongest_band,
        "ordered_band_rows": band_rows,
    }


def build_selected_strategy_value_check(
    *,
    axis_rankings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    best_axis = _safe_dict(axis_rankings[0] if axis_rankings else {})
    selected_strategy_axes = [
        row
        for row in axis_rankings
        if str(_safe_dict(row).get("field") or "")
        in {
            "selected_strategy_confidence",
            "selected_strategy_shortfall",
        }
    ]
    best_selected_strategy_axis = _safe_dict(
        selected_strategy_axes[0] if selected_strategy_axes else {}
    )
    if not best_axis or not best_selected_strategy_axis:
        return {
            "support_status": "insufficient_data",
            "adds_meaningful_value_beyond_best_axis": None,
            "interpretation": (
                "Selected-strategy threshold-profile value is unavailable because no "
                "eligible best-axis comparison could be formed."
            ),
        }

    best_profile = _safe_dict(best_axis.get("best_threshold_profile"))
    selected_profile = _safe_dict(best_selected_strategy_axis.get("best_threshold_profile"))

    capture_shortfall = _difference_or_none(
        best_profile.get("collapsed_capture_rate"),
        selected_profile.get("collapsed_capture_rate"),
    )
    leakage_excess = _difference_or_none(
        selected_profile.get("preserved_leakage_rate"),
        best_profile.get("preserved_leakage_rate"),
    )
    pocket_gap_shortfall = _difference_or_none(
        best_profile.get("inside_pocket_concentration_gap"),
        selected_profile.get("inside_pocket_concentration_gap"),
    )

    adds_meaningful_value: bool | None
    interpretation: str
    if str(best_axis.get("field") or "").startswith("selected_strategy"):
        adds_meaningful_value = True
        interpretation = (
            "A selected-strategy axis itself ranks as the strongest single-axis "
            "threshold-profile candidate in this run."
        )
    elif (
        best_selected_strategy_axis.get("profile_strength_status") == "strong_support"
        and _to_float(capture_shortfall, default=0.0) <= 0.05
        and _to_float(leakage_excess, default=0.0) <= 0.05
        and _to_float(pocket_gap_shortfall, default=0.0) <= 0.10
    ):
        adds_meaningful_value = True
        interpretation = (
            "Selected-strategy threshold profiling remains materially competitive with "
            "the strongest single-axis gate candidate on the tracked capture, leakage, "
            "and pocket-contrast surfaces."
        )
    else:
        adds_meaningful_value = False
        interpretation = (
            "Selected-strategy threshold profiling stays materially weaker than the "
            "strongest single-axis gate candidate on at least one of capture, "
            "leakage, or sharply-negative-pocket contrast."
        )

    return {
        "support_status": best_selected_strategy_axis.get(
            "profile_strength_status",
            "insufficient_data",
        ),
        "best_selected_strategy_axis": best_selected_strategy_axis,
        "relative_to_best_axis": {
            "best_axis_field": best_axis.get("field"),
            "collapsed_capture_rate_shortfall": capture_shortfall,
            "preserved_leakage_rate_excess": leakage_excess,
            "inside_pocket_concentration_gap_shortfall": pocket_gap_shortfall,
        },
        "adds_meaningful_value_beyond_best_axis": adds_meaningful_value,
        "interpretation": interpretation,
    }


def build_gate_vs_severity_check(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    axis_rankings: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    best_gate_axis = _safe_dict(axis_rankings[0] if axis_rankings else {})
    setup_axes = [
        row
        for row in axis_rankings
        if str(_safe_dict(row).get("field") or "")
        in {
            "setup_layer_confidence",
            "setup_shortfall",
        }
    ]
    best_setup_axis = _safe_dict(setup_axes[0] if setup_axes else {})
    best_gate_profile = _safe_dict(best_gate_axis.get("best_threshold_profile"))

    if not best_gate_axis or not best_setup_axis or not best_gate_profile:
        return {
            "support_status": "insufficient_data",
            "interpretation_status": "gate_vs_severity_unavailable",
            "interpretation": (
                "Gate-vs-severity comparison is unavailable because either the best "
                "gate axis or the best setup-side axis is missing."
            ),
        }

    if str(best_gate_axis.get("field") or "") in {
        "setup_layer_confidence",
        "setup_shortfall",
    }:
        return {
            "support_status": best_gate_axis.get(
                "profile_strength_status",
                "insufficient_data",
            ),
            "best_gate_axis": best_gate_axis,
            "setup_axis": best_setup_axis,
            "interpretation_status": "best_gate_already_setup_side",
            "interpretation": (
                "The strongest gate candidate is already the setup-side axis, so this "
                "report does not claim a separate setup severity role inside a "
                "different gate."
            ),
        }

    gate_rows = [
        row
        for row in comparison_rows
        if _matches_threshold(
            value=row.get(str(best_gate_axis.get("field") or "")),
            threshold_operator=str(best_gate_axis.get("threshold_operator") or ""),
            threshold=_to_float(best_gate_profile.get("threshold"), default=0.0) or 0.0,
        )
    ]
    preserved_gate_rows = _comparison_group_rows(
        gate_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_gate_rows = _comparison_group_rows(
        gate_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )
    setup_axis_spec = _axis_spec(str(best_setup_axis.get("field") or ""))
    if not setup_axis_spec:
        return {
            "support_status": "insufficient_data",
            "interpretation_status": "gate_vs_severity_unavailable",
            "interpretation": (
                "Gate-vs-severity comparison is unavailable because the setup-side "
                "axis specification could not be resolved."
            ),
        }

    conditional_profile = build_axis_threshold_profile(
        comparison_rows=gate_rows,
        field=str(setup_axis_spec["field"]),
        field_label=str(setup_axis_spec["field_label"]),
        transform=str(setup_axis_spec["transform"]),
        threshold_operator=str(setup_axis_spec["threshold_operator"]),
        threshold_grid=tuple(setup_axis_spec["threshold_grid"]),
    )
    best_conditional_profile = _safe_dict(
        conditional_profile.get("best_threshold_profile")
    )

    overall_gate_advantage = _difference_or_none(
        best_gate_profile.get("capture_minus_leakage"),
        _safe_dict(best_setup_axis.get("best_threshold_profile")).get(
            "capture_minus_leakage"
        ),
    )
    collapsed_median_delta = _to_float(
        best_conditional_profile.get("collapsed_inside_vs_outside_residual_median_delta"),
        default=None,
    )
    collapsed_pocket_lift = _to_float(
        best_conditional_profile.get("collapsed_pocket_lift"),
        default=None,
    )

    interpretation_status = "setup_axis_role_unclear"
    interpretation = (
        "The best setup-side axis can be profiled inside the strongest gate, but its "
        "role stays descriptive rather than decisive."
    )
    if (
        best_conditional_profile.get("profile_strength_status") in {"strong_support", "weak_support"}
        and _to_float(overall_gate_advantage, default=0.0) >= 0.15
        and (
            (collapsed_median_delta is not None and collapsed_median_delta <= -0.08)
            or (collapsed_pocket_lift is not None and collapsed_pocket_lift >= 0.20)
        )
    ):
        interpretation_status = "setup_axis_behaves_more_like_severity_inside_gate"
        interpretation = (
            "The strongest gate remains outside the setup axis, but inside that gate "
            "the best setup-side threshold still lines up with more negative "
            "collapsed residual severity, so setup looks more like a severity axis "
            "than a cleaner top-level gate."
        )
    elif (
        best_setup_axis.get("profile_strength_status") == "strong_support"
        and _to_float(overall_gate_advantage, default=0.0) <= 0.05
    ):
        interpretation_status = "setup_axis_competes_as_gate_candidate"
        interpretation = (
            "The best setup-side axis remains close enough to the strongest gate "
            "candidate that a clean gate-vs-severity split is not supported."
        )

    return {
        "support_status": residual_module._regime_support_status(
            total_row_count=len(gate_rows),
            collapsed_row_count=len(collapsed_gate_rows),
        ),
        "best_gate_axis": best_gate_axis,
        "gate_profile": best_gate_profile,
        "gate_row_count": len(gate_rows),
        "gate_preserved_row_count": len(preserved_gate_rows),
        "gate_collapsed_row_count": len(collapsed_gate_rows),
        "setup_axis": best_setup_axis,
        "conditional_setup_profile": conditional_profile,
        "best_conditional_setup_profile": best_conditional_profile,
        "overall_gate_advantage_vs_best_setup_axis": overall_gate_advantage,
        "interpretation_status": interpretation_status,
        "interpretation": interpretation,
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    baseline_reference: dict[str, Any],
    residual_class_comparison: dict[str, Any],
    residual_sign_distribution: dict[str, Any],
    best_threshold_axis: dict[str, Any],
    best_threshold_profile: dict[str, Any],
    top_axis_band_profiles: Sequence[dict[str, Any]],
    selected_strategy_value_check: dict[str, Any],
    gate_vs_severity_check: dict[str, Any],
) -> dict[str, Any]:
    baseline_preserved = _safe_dict(
        baseline_reference.get(_COMPARISON_GROUP_PRESERVED)
    )
    baseline_collapsed = _safe_dict(
        baseline_reference.get(_COMPARISON_GROUP_COLLAPSED)
    )
    residual_preserved = _safe_dict(
        residual_class_comparison.get(_COMPARISON_GROUP_PRESERVED)
    )
    residual_collapsed = _safe_dict(
        residual_class_comparison.get(_COMPARISON_GROUP_COLLAPSED)
    )
    negative_pocket = _safe_dict(
        residual_sign_distribution.get("sharply_negative_pocket")
    )
    best_band_profile = _safe_dict(
        next(
            (
                row
                for row in top_axis_band_profiles
                if str(_safe_dict(row).get("field") or "")
                == str(best_threshold_axis.get("field") or "")
            ),
            {},
        )
    )

    facts = [
        (
            "Final rule-bias-aligned comparison groups: preserved_final_directional_outcome="
            f"{summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            "collapsed_final_hold_outcome="
            f"{summary.get('collapsed_final_hold_outcome_row_count', 0)}, "
            "comparison_support_status="
            f"{summary.get('comparison_support_status', 'unknown')}."
        ),
        (
            "Baseline weighted aggregate medians: preserved="
            f"{baseline_preserved.get('median')}, collapsed={baseline_collapsed.get('median')}."
        ),
        (
            "Residual medians after subtracting the weighted baseline: preserved="
            f"{residual_preserved.get('median')}, collapsed={residual_collapsed.get('median')}."
        ),
        (
            "Sharply negative residual pocket uses residual <= "
            f"{_SHARPLY_NEGATIVE_RESIDUAL_THRESHOLD} and currently shows preserved_rate="
            f"{negative_pocket.get('preserved_rate')}, collapsed_rate={negative_pocket.get('collapsed_rate')}."
        ),
    ]
    if best_threshold_axis and best_threshold_profile:
        facts.append(
            "Best single-axis threshold profile: "
            f"{best_threshold_axis.get('field')} "
            f"at {best_threshold_profile.get('threshold_label')} with "
            f"collapsed_capture_rate={best_threshold_profile.get('collapsed_capture_rate')}, "
            f"preserved_leakage_rate={best_threshold_profile.get('preserved_leakage_rate')}, "
            "inside_pocket_concentration_gap="
            f"{best_threshold_profile.get('inside_pocket_concentration_gap')}."
        )
    if best_band_profile:
        facts.append(
            "Ordered band shape for the best axis is "
            f"{best_band_profile.get('profile_shape')} "
            f"({best_band_profile.get('shape_reason')})."
        )

    comparison_supported = summary.get("comparison_support_status") == "supported"
    interpretation_status = "comparison_unsupported"
    inference: list[str] = []

    if not comparison_supported:
        interpretation_status = "comparison_unsupported"
        inference.append(
            "The final preserved-vs-collapsed slice does not clear the family's "
            "supported-comparison threshold, so ordered threshold-profile claims stay "
            "withheld."
        )
    else:
        if best_threshold_axis.get("profile_strength_status") == "strong_support":
            interpretation_status = "threshold_profile_supported"
            if str(best_threshold_axis.get("field") or "") == _CONTEXT_BIAS_FAMILY_FIELD:
                inference.append(
                    "Ordered fixed-grid profiling keeps context_bias_family_mean as the "
                    "strongest single-axis gate candidate for the sharply negative "
                    "residual pocket."
                )
            else:
                inference.append(
                    "Ordered fixed-grid profiling moves the strongest single-axis gate "
                    f"candidate to {best_threshold_axis.get('field')} rather than "
                    "context_bias_family_mean."
                )
        elif best_threshold_axis:
            interpretation_status = "threshold_profile_inconclusive"
            inference.append(
                "A best observed axis can still be ranked descriptively, but no single "
                "axis clears the report's conservative strong-support threshold."
            )

        if best_band_profile:
            inference.append(
                "The best-axis band profile is most consistent with "
                f"{best_band_profile.get('profile_shape')} rather than a fully unstructured mix."
            )
        if selected_strategy_value_check:
            inference.append(
                str(selected_strategy_value_check.get("interpretation") or "")
            )
        if gate_vs_severity_check:
            inference.append(str(gate_vs_severity_check.get("interpretation") or ""))

    uncertainty = [
        "This report remains descriptive: a strong single-axis threshold profile does not prove the literal hidden production merge rule.",
        "The threshold grids and bands are fixed helpers only and are not optimized or promoted to production thresholds.",
        "Gate-vs-severity profiling is intentionally subordinate and does not reopen a broader interaction-effect search.",
    ]
    if best_threshold_axis.get("profile_strength_status") != "strong_support":
        uncertainty.append(
            "Because no axis clears the report's strong-support threshold, the best-axis ranking should be read as an ordered descriptive summary rather than a settled mechanism claim."
        )

    return {
        "interpretation_status": interpretation_status,
        "facts": [item for item in facts if item],
        "inference": [item for item in inference if item],
        "uncertainty": uncertainty,
    }


def build_limitations(
    *,
    summary: dict[str, Any],
    best_threshold_axis: dict[str, Any],
    top_axis_band_profiles: Sequence[dict[str, Any]],
    gate_vs_severity_check: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact profiles only the already-confirmed weighted aggregate residual slice and does not revisit earlier bottlenecks or production gate logic.",
        "Single-axis threshold profiling is intentionally conservative and can miss wider multi-axis shapes that still exist descriptively.",
    ]
    if summary.get("comparison_support_status") != "supported":
        limitations.append(
            "The final preserved-vs-collapsed comparison is below the family's normal supported threshold, so all threshold-profile conclusions remain provisional."
        )
    best_band_profile = _safe_dict(
        next(
            (
                row
                for row in top_axis_band_profiles
                if str(_safe_dict(row).get("field") or "")
                == str(best_threshold_axis.get("field") or "")
            ),
            {},
        )
    )
    if best_band_profile.get("profile_shape") in {"mixed_or_weak", "insufficient_data"}:
        limitations.append(
            "Ordered band profiling does not isolate a clean cutoff-or-band shape on the leading axis, so the shape reading remains limited."
        )
    if str(gate_vs_severity_check.get("interpretation_status") or "") in {
        "setup_axis_role_unclear",
        "gate_vs_severity_unavailable",
    }:
        limitations.append(
            "The subordinate gate-vs-severity check remains descriptive and does not independently settle setup's role inside the best gate."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    best_threshold_axis = _safe_dict(widest.get("best_threshold_axis"))
    best_threshold_profile = _safe_dict(widest.get("best_threshold_profile"))
    selected_strategy_value_check = _safe_dict(
        widest.get("selected_strategy_value_check")
    )
    gate_vs_severity_check = _safe_dict(widest.get("gate_vs_severity_check"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "baseline_name": _BASELINE_NAME,
        "interpretation_status": interpretation.get("interpretation_status"),
        "best_threshold_axis": best_threshold_axis,
        "best_threshold_profile": best_threshold_profile,
        "selected_strategy_value_check": selected_strategy_value_check,
        "gate_vs_severity_check": gate_vs_severity_check,
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            interpretation=interpretation,
            best_threshold_axis=best_threshold_axis,
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
            "- comparison_support_status: "
            f"{headline.get('comparison_support_status', 'unknown')}"
        )
        lines.append(
            "- best_threshold_axis: "
            f"{headline.get('best_threshold_axis', 'none')}"
        )
        lines.append(
            "- best_threshold_label: "
            f"{headline.get('best_threshold_label', 'none')}"
        )
        lines.append(
            "- best_axis_band_profile_shape: "
            f"{headline.get('best_axis_band_profile_shape', 'n/a')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{headline.get('interpretation_status', 'unknown')}"
        )
        for fact in _safe_list(interpretation.get("facts"))[:5]:
            lines.append(f"- fact: {fact}")
        for item in _safe_list(interpretation.get("inference"))[:4]:
            lines.append(f"- inference: {item}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    best_axis = _safe_dict(final_assessment.get("best_threshold_axis"))
    best_profile = _safe_dict(final_assessment.get("best_threshold_profile"))
    lines.append(
        "- best_threshold_axis: "
        f"{best_axis.get('field', 'none')} @ {best_profile.get('threshold_label', 'n/a')}"
    )
    for item in _safe_list(final_assessment.get("observed"))[:5]:
        lines.append(f"- observed: {item}")
    for item in _safe_list(final_assessment.get("inference"))[:4]:
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


def _build_profile_metrics(
    *,
    inside_rows: Sequence[dict[str, Any]],
    outside_rows: Sequence[dict[str, Any]],
    preserved_present_rows: Sequence[dict[str, Any]],
    collapsed_present_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    inside_preserved_rows = _comparison_group_rows(
        inside_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    inside_collapsed_rows = _comparison_group_rows(
        inside_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )
    outside_preserved_rows = _comparison_group_rows(
        outside_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    outside_collapsed_rows = _comparison_group_rows(
        outside_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    inside_preserved_summary = final_split_module._numeric_field_summary(
        inside_preserved_rows,
        _RESIDUAL_FIELD,
    )
    inside_collapsed_summary = final_split_module._numeric_field_summary(
        inside_collapsed_rows,
        _RESIDUAL_FIELD,
    )
    outside_preserved_summary = final_split_module._numeric_field_summary(
        outside_preserved_rows,
        _RESIDUAL_FIELD,
    )
    outside_collapsed_summary = final_split_module._numeric_field_summary(
        outside_collapsed_rows,
        _RESIDUAL_FIELD,
    )
    inside_preserved_bucket = residual_module._residual_bucket_summary(
        inside_preserved_rows
    )
    inside_collapsed_bucket = residual_module._residual_bucket_summary(
        inside_collapsed_rows
    )
    outside_preserved_bucket = residual_module._residual_bucket_summary(
        outside_preserved_rows
    )
    outside_collapsed_bucket = residual_module._residual_bucket_summary(
        outside_collapsed_rows
    )

    inside_preserved_pocket_rate = _to_float(
        inside_preserved_bucket.get("strongly_negative_rate"),
        default=0.0,
    )
    inside_collapsed_pocket_rate = _to_float(
        inside_collapsed_bucket.get("strongly_negative_rate"),
        default=0.0,
    )
    outside_preserved_pocket_rate = _to_float(
        outside_preserved_bucket.get("strongly_negative_rate"),
        default=0.0,
    )
    outside_collapsed_pocket_rate = _to_float(
        outside_collapsed_bucket.get("strongly_negative_rate"),
        default=0.0,
    )
    total_preserved_pocket_count = int(
        residual_module._residual_bucket_summary(preserved_present_rows).get(
            "strongly_negative_count",
            0,
        )
        or 0
    )
    total_collapsed_pocket_count = int(
        residual_module._residual_bucket_summary(collapsed_present_rows).get(
            "strongly_negative_count",
            0,
        )
        or 0
    )

    support_status = _profile_sample_support_status(
        inside_row_count=len(inside_rows),
        inside_collapsed_row_count=len(inside_collapsed_rows),
    )
    capture_minus_leakage = round(
        _safe_ratio(len(inside_collapsed_rows), len(collapsed_present_rows))
        - _safe_ratio(len(inside_preserved_rows), len(preserved_present_rows)),
        6,
    )
    inside_pocket_concentration_gap = round(
        inside_collapsed_pocket_rate - inside_preserved_pocket_rate,
        6,
    )
    outside_pocket_concentration_gap = round(
        outside_collapsed_pocket_rate - outside_preserved_pocket_rate,
        6,
    )
    collapsed_pocket_lift = round(
        inside_collapsed_pocket_rate - outside_collapsed_pocket_rate,
        6,
    )

    profile_strength_status = _profile_strength_status(
        support_status=support_status,
        collapsed_capture_rate=_safe_ratio(
            len(inside_collapsed_rows),
            len(collapsed_present_rows),
        ),
        preserved_leakage_rate=_safe_ratio(
            len(inside_preserved_rows),
            len(preserved_present_rows),
        ),
        inside_pocket_concentration_gap=inside_pocket_concentration_gap,
        collapsed_pocket_lift=collapsed_pocket_lift,
    )

    return {
        "support_status": support_status,
        "profile_strength_status": profile_strength_status,
        "inside_row_count": len(inside_rows),
        "inside_preserved_row_count": len(inside_preserved_rows),
        "inside_collapsed_row_count": len(inside_collapsed_rows),
        "outside_row_count": len(outside_rows),
        "outside_preserved_row_count": len(outside_preserved_rows),
        "outside_collapsed_row_count": len(outside_collapsed_rows),
        "collapsed_capture_rate": _safe_ratio(
            len(inside_collapsed_rows),
            len(collapsed_present_rows),
        ),
        "preserved_leakage_rate": _safe_ratio(
            len(inside_preserved_rows),
            len(preserved_present_rows),
        ),
        "capture_minus_leakage": capture_minus_leakage,
        "collapsed_pocket_capture_rate": _safe_ratio(
            int(inside_collapsed_bucket.get("strongly_negative_count", 0) or 0),
            total_collapsed_pocket_count,
        ),
        "preserved_pocket_capture_rate": _safe_ratio(
            int(inside_preserved_bucket.get("strongly_negative_count", 0) or 0),
            total_preserved_pocket_count,
        ),
        "inside_sharply_negative_residual_pocket_rate_by_class": {
            _COMPARISON_GROUP_PRESERVED: inside_preserved_pocket_rate,
            _COMPARISON_GROUP_COLLAPSED: inside_collapsed_pocket_rate,
        },
        "outside_sharply_negative_residual_pocket_rate_by_class": {
            _COMPARISON_GROUP_PRESERVED: outside_preserved_pocket_rate,
            _COMPARISON_GROUP_COLLAPSED: outside_collapsed_pocket_rate,
        },
        "inside_pocket_concentration_gap": inside_pocket_concentration_gap,
        "outside_pocket_concentration_gap": outside_pocket_concentration_gap,
        "collapsed_pocket_lift": collapsed_pocket_lift,
        "inside_residual_median_by_class": {
            _COMPARISON_GROUP_PRESERVED: inside_preserved_summary.get("median"),
            _COMPARISON_GROUP_COLLAPSED: inside_collapsed_summary.get("median"),
        },
        "outside_residual_median_by_class": {
            _COMPARISON_GROUP_PRESERVED: outside_preserved_summary.get("median"),
            _COMPARISON_GROUP_COLLAPSED: outside_collapsed_summary.get("median"),
        },
        "inside_residual_summary_by_class": {
            _COMPARISON_GROUP_PRESERVED: inside_preserved_summary,
            _COMPARISON_GROUP_COLLAPSED: inside_collapsed_summary,
        },
        "outside_residual_summary_by_class": {
            _COMPARISON_GROUP_PRESERVED: outside_preserved_summary,
            _COMPARISON_GROUP_COLLAPSED: outside_collapsed_summary,
        },
        "collapsed_inside_vs_outside_residual_median_delta": _difference_or_none(
            inside_collapsed_summary.get("median"),
            outside_collapsed_summary.get("median"),
        ),
        "preserved_inside_vs_outside_residual_median_delta": _difference_or_none(
            inside_preserved_summary.get("median"),
            outside_preserved_summary.get("median"),
        ),
    }


def _axis_present_rows(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in comparison_rows
        if row.get(_RESIDUAL_FIELD) is not None
        and _to_float(row.get(field), default=None) is not None
    ]


def _profile_sample_support_status(
    *,
    inside_row_count: int,
    inside_collapsed_row_count: int,
) -> str:
    if (
        inside_row_count >= _MIN_PROFILE_ROW_COUNT
        and inside_collapsed_row_count >= _MIN_PROFILE_COLLAPSED_ROW_COUNT
    ):
        return "supported"
    if inside_row_count > 0:
        return "limited_support"
    return "insufficient_data"


def _profile_strength_status(
    *,
    support_status: str,
    collapsed_capture_rate: float,
    preserved_leakage_rate: float,
    inside_pocket_concentration_gap: float,
    collapsed_pocket_lift: float,
) -> str:
    if support_status != "supported":
        return support_status
    if (
        collapsed_capture_rate >= 0.60
        and preserved_leakage_rate <= 0.25
        and inside_pocket_concentration_gap >= 0.50
        and collapsed_pocket_lift >= 0.20
    ):
        return "strong_support"
    return "weak_support"


def _threshold_profile_sort_key(
    *,
    row: dict[str, Any],
    threshold_operator: str,
) -> tuple[Any, ...]:
    threshold = _to_float(row.get("threshold"), default=0.0) or 0.0
    conservative_threshold_key = threshold
    if threshold_operator == ">=":
        conservative_threshold_key = -threshold
    return (
        _PROFILE_SUPPORT_ORDER.get(
            str(row.get("profile_strength_status") or ""),
            99,
        ),
        -_to_float(row.get("collapsed_capture_rate"), default=0.0),
        _to_float(row.get("preserved_leakage_rate"), default=1.0),
        -_to_float(row.get("inside_pocket_concentration_gap"), default=0.0),
        -_to_float(row.get("collapsed_pocket_lift"), default=0.0),
        -_to_float(row.get("capture_minus_leakage"), default=0.0),
        -int(row.get("inside_row_count", 0) or 0),
        conservative_threshold_key,
    )


def _axis_ranking_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _PROFILE_SUPPORT_ORDER.get(
            str(row.get("profile_strength_status") or ""),
            99,
        ),
        -_to_float(row.get("collapsed_capture_rate"), default=0.0),
        _to_float(row.get("preserved_leakage_rate"), default=1.0),
        -_to_float(row.get("inside_pocket_concentration_gap"), default=0.0),
        -_to_float(row.get("collapsed_pocket_lift"), default=0.0),
        -_to_float(row.get("capture_minus_leakage"), default=0.0),
        _AXIS_ORDER.get(str(row.get("field") or ""), 99),
    )


def _band_profile_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _PROFILE_SUPPORT_ORDER.get(
            str(row.get("profile_strength_status") or ""),
            99,
        ),
        -_to_float(row.get("inside_pocket_concentration_gap"), default=0.0),
        -_to_float(row.get("collapsed_capture_rate"), default=0.0),
        _to_float(row.get("preserved_leakage_rate"), default=1.0),
        -_to_float(row.get("collapsed_pocket_lift"), default=0.0),
        -int(row.get("inside_row_count", 0) or 0),
    )


def _classify_band_profile(
    *,
    band_rows: Sequence[dict[str, Any]],
    extreme_band_side: str,
) -> tuple[str, str]:
    populated_band_rows = [
        row for row in band_rows if int(_safe_dict(row).get("inside_row_count", 0) or 0) > 0
    ]
    if not populated_band_rows:
        return (
            "insufficient_data",
            "No populated ordered bands were available for this axis.",
        )

    positive_indexes = [
        index
        for index, row in enumerate(band_rows)
        if _is_meaningful_band(_safe_dict(row))
    ]
    if not positive_indexes:
        return (
            "mixed_or_weak",
            "No ordered band clears the report's conservative pocket-contrast helper.",
        )

    strongest_band_index = min(
        range(len(band_rows)),
        key=lambda index: _band_profile_sort_key(_safe_dict(band_rows[index])),
    )
    if strongest_band_index not in {
        0,
        len(band_rows) - 1,
    }:
        return (
            "narrow_band",
            "The strongest ordered band is interior rather than sitting on the "
            "extreme side of the axis, so a narrow band looks more plausible than a "
            "simple one-sided cutoff.",
        )

    expected_extreme_indexes = _expected_extreme_indexes(
        band_count=len(band_rows),
        extreme_band_side=extreme_band_side,
        run_length=len(positive_indexes),
    )
    if positive_indexes == expected_extreme_indexes and len(positive_indexes) == 1:
        return (
            "simple_cutoff",
            "Only the extreme-side band stays meaningfully concentrated, which fits a "
            "simple cutoff better than a broad helper regime.",
        )
    if positive_indexes == expected_extreme_indexes:
        return (
            "broad_helper_regime",
            "Multiple contiguous extreme-side bands remain meaningfully concentrated, "
            "which fits a broad low-confidence/high-shortfall helper regime better "
            "than a narrow band.",
        )
    return (
        "mixed_or_weak",
        "Ordered band support is present but not cleanly contiguous from the extreme "
        "side, so the band shape remains mixed.",
    )


def _is_meaningful_band(row: dict[str, Any]) -> bool:
    return bool(
        _to_float(row.get("collapsed_capture_rate"), default=0.0) >= 0.20
        and _to_float(row.get("preserved_leakage_rate"), default=1.0) <= 0.25
        and _to_float(row.get("inside_pocket_concentration_gap"), default=0.0) >= 0.50
    )


def _expected_extreme_indexes(
    *,
    band_count: int,
    extreme_band_side: str,
    run_length: int,
) -> list[int]:
    if extreme_band_side == "high":
        return list(range(max(0, band_count - run_length), band_count))
    return list(range(0, min(band_count, run_length)))


def _matches_threshold(
    *,
    value: Any,
    threshold_operator: str,
    threshold: float,
) -> bool:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return False
    if threshold_operator == "<=":
        return bool(numeric_value <= threshold)
    if threshold_operator == ">=":
        return bool(numeric_value >= threshold)
    return False


def _value_in_band(
    *,
    value: Any,
    lower_bound_exclusive: float | None,
    upper_bound_inclusive: float | None,
) -> bool:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return False
    if (
        lower_bound_exclusive is not None
        and numeric_value <= lower_bound_exclusive
    ):
        return False
    if (
        upper_bound_inclusive is not None
        and numeric_value > upper_bound_inclusive
    ):
        return False
    return True


def _axis_spec(field: str) -> dict[str, Any]:
    return next(
        (
            spec
            for spec in _AXIS_SPECS
            if str(spec.get("field") or "") == field
        ),
        {},
    )


def _comparison_group_rows(
    rows: Sequence[dict[str, Any]],
    comparison_group: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("comparison_group") or "") == comparison_group
    ]


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    interpretation: dict[str, Any],
    best_threshold_axis: dict[str, Any],
) -> str:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    interpretation_status = str(
        interpretation.get("interpretation_status") or "comparison_unsupported"
    )

    if final_slice_row_count <= 0:
        return (
            "No rows reached the final fully aligned plus rule-bias-aligned slice in "
            "the widest configuration, so threshold-profile diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one "
            "side of the preserved-vs-collapsed comparison is missing, so "
            "threshold-profile diagnosis remains incomplete."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but ordered threshold-profile claims stay "
            "withheld because the preserved-vs-collapsed comparison does not clear "
            "the family's normal support threshold."
        )
    if interpretation_status == "threshold_profile_supported":
        return (
            "The widest configuration supports a single-axis threshold-profile "
            f"reading led by {best_threshold_axis.get('field')}, while still keeping "
            "that evidence descriptive rather than causal proof of the hidden "
            "production rule."
        )
    return (
        "The widest configuration still allows a descriptive best-axis ranking, but "
        "no single-axis threshold profile clears the report's conservative strong-"
        "support threshold."
    )


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _difference_or_none(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(_to_float(left, default=0.0) - _to_float(right, default=0.0), 6)


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