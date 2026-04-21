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
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report as aggregate_shape_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Weighted Aggregate Residual Diagnosis Report"
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
_BASELINE_NAME = "weighted_mean_setup_emphasis"
_BASELINE_LABEL = aggregate_shape_module._AGGREGATE_FIELD_LABEL_MAP.get(
    _BASELINE_NAME,
    "Weighted mean with setup emphasis",
)
_BASELINE_FORMULA = aggregate_shape_module._AGGREGATE_FORMULA_MAP.get(
    _BASELINE_NAME,
    (
        "0.50*setup_layer_confidence + 0.25*context_bias_family_mean + "
        "0.25*selected_strategy_confidence"
    ),
)
_BASELINE_COMPONENT_FIELDS = aggregate_shape_module._AGGREGATE_COMPONENT_MAP.get(
    _BASELINE_NAME,
    (
        "setup_layer_confidence",
        "context_bias_family_mean",
        "selected_strategy_confidence",
    ),
)
_CONTEXT_BIAS_FAMILY_FIELD = "context_bias_family_mean"
_RESIDUAL_FIELD = "weighted_aggregate_residual"

_LOW_CONFIDENCE_THRESHOLD = 0.65
_NEAR_ZERO_RESIDUAL_ABS_THRESHOLD = 0.05
_STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD = -0.15
_MIN_REGIME_ROW_COUNT = 10
_MIN_REGIME_COLLAPSED_ROW_COUNT = 5

_EXPLANATION_CLASS_ORDER = {
    "threshold": 0,
    "interaction_effect": 1,
    "piecewise_compression": 2,
    "penalty": 3,
    "unclear": 4,
}
_REGIME_SUPPORT_ORDER = {
    "supported": 0,
    "limited_support": 1,
    "insufficient_data": 2,
}
_SURFACE_FIELD_ORDER = {
    "setup_layer_confidence": 0,
    _CONTEXT_BIAS_FAMILY_FIELD: 1,
    "selected_strategy_confidence": 2,
}
_RAW_SURFACE_SPECS = (
    ("setup_layer_confidence", "Setup layer confidence"),
    (_CONTEXT_BIAS_FAMILY_FIELD, "Context/bias family mean"),
    ("selected_strategy_confidence", "Selected-strategy confidence"),
)
_SHORTFALL_SPECS = (
    ("setup_shortfall", "Setup shortfall"),
    ("context_bias_family_shortfall", "Context/bias family shortfall"),
    ("selected_strategy_shortfall", "Selected-strategy shortfall"),
)
_JOINT_SHORTFALL_REGIME_SPECS = (
    (tuple(), "no_low_confidence_surface"),
    (("setup_layer_confidence",), "low_setup_only"),
    ((_CONTEXT_BIAS_FAMILY_FIELD,), "low_context_bias_family_only"),
    (("selected_strategy_confidence",), "low_selected_strategy_only"),
    (
        ("setup_layer_confidence", _CONTEXT_BIAS_FAMILY_FIELD),
        "low_setup_and_low_context_bias_family",
    ),
    (
        ("setup_layer_confidence", "selected_strategy_confidence"),
        "low_setup_and_low_selected_strategy",
    ),
    (
        (_CONTEXT_BIAS_FAMILY_FIELD, "selected_strategy_confidence"),
        "low_context_bias_family_and_low_selected_strategy",
    ),
    (
        (
            "setup_layer_confidence",
            _CONTEXT_BIAS_FAMILY_FIELD,
            "selected_strategy_confidence",
        ),
        "all_three_low",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the final fully aligned, "
            "rule-bias-aligned preserved-vs-collapsed slice and characterize the "
            "residual structure that remains after applying the already-tested "
            "weighted_mean_setup_emphasis aggregate as the simple baseline."
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
            "Retained for architectural parity with sibling reports. Residual "
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

    result = (
        run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report(
            input_path=input_path,
            output_dir=output_dir,
            configurations=configurations,
            min_symbol_support=args.min_symbol_support,
            write_report_copies=args.write_latest_copy,
        )
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_assessment = _safe_dict(report.get("final_assessment"))

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
                "baseline_name": report.get("baseline_name"),
                "comparison_support_status": summary.get(
                    "comparison_support_status",
                    "unknown",
                ),
                "strongest_candidate_explanation_class": final_assessment.get(
                    "strongest_candidate_explanation_class"
                ),
                "interpretation_status": final_assessment.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report(
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
        "baseline_name": _BASELINE_NAME,
        "baseline_label": _BASELINE_LABEL,
        "baseline_formula": _BASELINE_FORMULA,
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
        "residual_surface_correlations": _safe_dict(
            widest_summary.get("residual_surface_correlations")
        ),
        "residual_low_confidence_regimes": _safe_dict(
            widest_summary.get("residual_low_confidence_regimes")
        ),
        "joint_shortfall_regimes": _safe_dict(
            widest_summary.get("joint_shortfall_regimes")
        ),
        "residual_step_profile": _safe_dict(
            widest_summary.get("residual_step_profile")
        ),
        "residual_behavior_classification": _safe_dict(
            widest_summary.get("residual_behavior_classification")
        ),
        "interpretation_status": _safe_dict(
            widest_summary.get("residual_behavior_classification")
        ).get("interpretation_status"),
        "interpretation": _safe_dict(widest_summary.get("interpretation")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis reports.",
            "The simple baseline is fixed to the already-tested weighted_mean_setup_emphasis aggregate and is computed through the sibling aggregate-shape module so the baseline formula does not drift here.",
            "Residual is defined descriptively as rule_engine_confidence minus weighted_mean_setup_emphasis; negative residual therefore means actual persisted rule_engine_confidence is more severe than the simple weighted aggregate baseline predicted.",
            "Fixed low-confidence and sharply-negative-residual thresholds in this report are descriptive regime helpers only and are not promoted to production thresholds or causal proof.",
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
    residual_rows = [build_residual_row(row) for row in comparison_rows]

    summary = final_split_module.build_summary(
        actionable_rows=actionable_rows,
        fully_aligned_rows=fully_aligned_rows,
        final_split_rows=final_split_rows,
        min_symbol_support=min_symbol_support,
    )
    rule_engine_confidence_context = aggregate_shape_module.build_rule_engine_confidence_context(
        comparison_rows=residual_rows
    )
    actual_rule_engine_confidence_reference = build_actual_rule_engine_confidence_reference(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    baseline_reference = build_baseline_reference(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
        actual_rule_engine_confidence_reference=actual_rule_engine_confidence_reference,
    )
    residual_class_comparison = build_residual_class_comparison(
        comparison_rows=residual_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    residual_sign_distribution = build_residual_sign_distribution(
        comparison_rows=residual_rows
    )
    residual_surface_correlations = build_residual_surface_correlations(
        comparison_rows=residual_rows
    )
    residual_low_confidence_regimes = build_residual_low_confidence_regimes(
        comparison_rows=residual_rows
    )
    joint_shortfall_regimes = build_joint_shortfall_regimes(comparison_rows=residual_rows)
    residual_step_profile = build_residual_step_profile(comparison_rows=residual_rows)
    residual_behavior_classification = build_residual_behavior_classification(
        summary=summary,
        residual_class_comparison=residual_class_comparison,
        residual_sign_distribution=residual_sign_distribution,
        residual_low_confidence_regimes=residual_low_confidence_regimes,
        joint_shortfall_regimes=joint_shortfall_regimes,
        residual_step_profile=residual_step_profile,
    )
    interpretation = build_interpretation(
        summary=summary,
        baseline_reference=baseline_reference,
        actual_rule_engine_confidence_reference=actual_rule_engine_confidence_reference,
        residual_class_comparison=residual_class_comparison,
        residual_sign_distribution=residual_sign_distribution,
        residual_surface_correlations=residual_surface_correlations,
        residual_low_confidence_regimes=residual_low_confidence_regimes,
        joint_shortfall_regimes=joint_shortfall_regimes,
        residual_step_profile=residual_step_profile,
        residual_behavior_classification=residual_behavior_classification,
    )

    strongest_joint_regime = _safe_dict(
        joint_shortfall_regimes.get("strongest_joint_regime")
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
            "baseline_name": _BASELINE_NAME,
            "strongest_candidate_explanation_class": (
                residual_behavior_classification.get(
                    "strongest_candidate_explanation_class"
                )
            ),
            "strongest_joint_regime": strongest_joint_regime.get("regime_label") or "none",
            "interpretation_status": residual_behavior_classification.get(
                "interpretation_status"
            ),
        },
        "summary": summary,
        "baseline_reference": baseline_reference,
        "actual_rule_engine_confidence_reference": actual_rule_engine_confidence_reference,
        "residual_class_comparison": residual_class_comparison,
        "residual_sign_distribution": residual_sign_distribution,
        "residual_surface_correlations": residual_surface_correlations,
        "residual_low_confidence_regimes": residual_low_confidence_regimes,
        "joint_shortfall_regimes": joint_shortfall_regimes,
        "residual_step_profile": residual_step_profile,
        "residual_behavior_classification": residual_behavior_classification,
        "interpretation": interpretation,
    }


def build_residual_row(row: dict[str, Any]) -> dict[str, Any]:
    aggregate_candidate_values = aggregate_shape_module.build_aggregate_candidate_values(
        row
    )
    rule_engine_confidence = _to_float(
        row.get(_RULE_ENGINE_CONFIDENCE_FIELD),
        default=None,
    )
    baseline_value = _to_float(
        aggregate_candidate_values.get(_BASELINE_NAME),
        default=None,
    )
    setup_confidence = _to_float(
        row.get("setup_layer_confidence"),
        default=None,
    )
    context_bias_family_mean = _to_float(
        aggregate_candidate_values.get(_CONTEXT_BIAS_FAMILY_FIELD),
        default=None,
    )
    selected_strategy_confidence = _to_float(
        row.get("selected_strategy_confidence"),
        default=None,
    )
    setup_shortfall = _shortfall(setup_confidence)
    context_bias_family_shortfall = _shortfall(context_bias_family_mean)
    selected_strategy_shortfall = _shortfall(selected_strategy_confidence)

    low_setup_confidence = _low_confidence_flag(setup_confidence)
    low_context_bias_family = _low_confidence_flag(context_bias_family_mean)
    low_selected_strategy = _low_confidence_flag(selected_strategy_confidence)
    low_flags = (
        low_setup_confidence,
        low_context_bias_family,
        low_selected_strategy,
    )

    return {
        **row,
        **aggregate_candidate_values,
        _RESIDUAL_FIELD: _difference_or_none(rule_engine_confidence, baseline_value),
        "setup_shortfall": setup_shortfall,
        "context_bias_family_shortfall": context_bias_family_shortfall,
        "selected_strategy_shortfall": selected_strategy_shortfall,
        "low_setup_confidence_regime": low_setup_confidence,
        "low_context_bias_family_regime": low_context_bias_family,
        "low_selected_strategy_confidence_regime": low_selected_strategy,
        "low_confidence_surface_count": _low_confidence_surface_count(low_flags),
        "exact_low_confidence_fields": _exact_low_confidence_fields(
            low_setup_confidence=low_setup_confidence,
            low_context_bias_family=low_context_bias_family,
            low_selected_strategy=low_selected_strategy,
        ),
    }


def build_actual_rule_engine_confidence_reference(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    reference = aggregate_shape_module.build_rule_engine_confidence_reference(
        comparison_rows=comparison_rows,
        rule_engine_confidence_context=rule_engine_confidence_context,
    )
    return {
        **reference,
        "support_status": rule_engine_confidence_context.get(
            "support_status",
            "limited_support",
        ),
    }


def build_baseline_reference(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
    actual_rule_engine_confidence_reference: dict[str, Any],
) -> dict[str, Any]:
    reference = _build_metric_row(
        comparison_rows=comparison_rows,
        field=_BASELINE_NAME,
        field_label=_BASELINE_LABEL,
        formula=_BASELINE_FORMULA,
        component_fields=_BASELINE_COMPONENT_FIELDS,
        actual_rule_engine_median_gap=_safe_abs_float(
            rule_engine_confidence_context.get(
                "median_difference_preserved_minus_collapsed"
            )
        ),
    )
    comparison_to_actual = (
        aggregate_shape_module._compare_candidate_to_actual_rule_engine_confidence(
            candidate_row=reference,
            actual_row=actual_rule_engine_confidence_reference,
        )
    )
    return {
        **reference,
        "comparison_to_actual_rule_engine_confidence": _safe_dict(
            comparison_to_actual.get("summary")
        ),
        "residual_gap_vs_actual_rule_engine_confidence": _safe_dict(
            comparison_to_actual.get("difference_summary")
        ),
    }


def build_residual_class_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    rule_engine_confidence_context: dict[str, Any],
) -> dict[str, Any]:
    row = _build_metric_row(
        comparison_rows=comparison_rows,
        field=_RESIDUAL_FIELD,
        field_label=(
            "Residual (rule_engine_confidence - weighted_mean_setup_emphasis)"
        ),
        formula=f"{_RULE_ENGINE_CONFIDENCE_FIELD} - {_BASELINE_NAME}",
        component_fields=(_RULE_ENGINE_CONFIDENCE_FIELD, _BASELINE_NAME),
        actual_rule_engine_median_gap=_safe_abs_float(
            rule_engine_confidence_context.get(
                "median_difference_preserved_minus_collapsed"
            )
        ),
    )
    return {
        **row,
        "baseline_name": _BASELINE_NAME,
        "baseline_formula": _BASELINE_FORMULA,
        "directional_pair_rate_label": "preserved_greater_pair_rate",
    }


def build_residual_sign_distribution(
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
    preserved_summary = _residual_bucket_summary(preserved_rows)
    collapsed_summary = _residual_bucket_summary(collapsed_rows)

    preserved_sharply_negative_rate = _to_float(
        preserved_summary.get("strongly_negative_rate"),
        default=0.0,
    )
    collapsed_sharply_negative_rate = _to_float(
        collapsed_summary.get("strongly_negative_rate"),
        default=0.0,
    )
    sharply_negative_gap = round(
        collapsed_sharply_negative_rate - preserved_sharply_negative_rate,
        6,
    )
    concentration_status = "mixed"
    if collapsed_summary.get("present_row_count", 0) <= 0:
        concentration_status = "insufficient_data"
    elif (
        collapsed_sharply_negative_rate >= 0.6
        and preserved_sharply_negative_rate <= 0.25
        and sharply_negative_gap >= 0.5
    ):
        concentration_status = "collapsed_concentrated"
    elif collapsed_sharply_negative_rate <= 0 and preserved_sharply_negative_rate <= 0:
        concentration_status = "absent"
    elif collapsed_sharply_negative_rate > preserved_sharply_negative_rate:
        concentration_status = "collapsed_leaning"

    support_status = _comparison_support_status(
        baseline_row_count=int(preserved_summary.get("present_row_count", 0) or 0),
        collapsed_row_count=int(collapsed_summary.get("present_row_count", 0) or 0),
    )
    if (
        int(preserved_summary.get("present_row_count", 0) or 0) <= 0
        or int(collapsed_summary.get("present_row_count", 0) or 0) <= 0
    ):
        support_status = "insufficient_data"

    return {
        "support_status": support_status,
        "thresholds": {
            "near_zero_abs_threshold": _NEAR_ZERO_RESIDUAL_ABS_THRESHOLD,
            "sharply_negative_threshold": _STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD,
        },
        _COMPARISON_GROUP_PRESERVED: preserved_summary,
        _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
        "sharply_negative_pocket": {
            "threshold": _STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD,
            "preserved_rate": preserved_summary.get("strongly_negative_rate"),
            "collapsed_rate": collapsed_summary.get("strongly_negative_rate"),
            "collapsed_minus_preserved_rate": sharply_negative_gap,
            "concentration_status": concentration_status,
            "support_status": support_status,
        },
    }


def build_residual_surface_correlations(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_surface_rows: list[dict[str, Any]] = []
    for field, label in _RAW_SURFACE_SPECS:
        correlation = _correlation_between_fields(
            rows=comparison_rows,
            left_field=field,
            right_field=_RESIDUAL_FIELD,
        )
        value = _to_float(correlation.get("pearson_correlation"), default=None)
        raw_surface_rows.append(
            {
                "field": field,
                "field_label": label,
                "transform": "raw_confidence",
                "paired_row_count": correlation.get("paired_row_count", 0),
                "pearson_correlation_to_residual": value,
                "alignment_to_negative_residual": (
                    "aligned" if value is not None and value > 0 else "not_aligned"
                ),
                "alignment_strength": (
                    round(value, 6) if value is not None and value > 0 else 0.0
                ),
            }
        )

    shortfall_rows: list[dict[str, Any]] = []
    for field, label in _SHORTFALL_SPECS:
        correlation = _correlation_between_fields(
            rows=comparison_rows,
            left_field=field,
            right_field=_RESIDUAL_FIELD,
        )
        value = _to_float(correlation.get("pearson_correlation"), default=None)
        shortfall_rows.append(
            {
                "field": field,
                "field_label": label,
                "transform": "shortfall",
                "paired_row_count": correlation.get("paired_row_count", 0),
                "pearson_correlation_to_residual": value,
                "alignment_to_negative_residual": (
                    "aligned" if value is not None and value < 0 else "not_aligned"
                ),
                "alignment_strength": (
                    round(abs(value), 6)
                    if value is not None and value < 0
                    else 0.0
                ),
            }
        )

    raw_surface_rows.sort(key=_correlation_sort_key)
    shortfall_rows.sort(key=_correlation_sort_key)
    strongest_raw = _safe_dict(raw_surface_rows[0] if raw_surface_rows else {})
    strongest_shortfall = _safe_dict(shortfall_rows[0] if shortfall_rows else {})
    support_status = "insufficient_data"
    if raw_surface_rows or shortfall_rows:
        support_status = "limited_support"
    if (
        strongest_raw.get("pearson_correlation_to_residual") is not None
        or strongest_shortfall.get("pearson_correlation_to_residual") is not None
    ):
        support_status = "supported"

    return {
        "support_status": support_status,
        "raw_surface_correlations": raw_surface_rows,
        "shortfall_transform_correlations": shortfall_rows,
        "strongest_raw_surface_alignment": strongest_raw,
        "strongest_shortfall_alignment": strongest_shortfall,
    }


def build_residual_low_confidence_regimes(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    regime_rows: list[dict[str, Any]] = []
    for field, label in _RAW_SURFACE_SPECS:
        regime_rows_for_field = [
            row
            for row in comparison_rows
            if row.get(_RESIDUAL_FIELD) is not None
            and _to_float(row.get(field), default=None) is not None
            and _to_float(row.get(field), default=0.0) <= _LOW_CONFIDENCE_THRESHOLD
        ]
        outside_rows = [
            row
            for row in comparison_rows
            if row.get(_RESIDUAL_FIELD) is not None
            and _to_float(row.get(field), default=None) is not None
            and _to_float(row.get(field), default=0.0) > _LOW_CONFIDENCE_THRESHOLD
        ]
        row_summary = _build_regime_summary(
            regime_label=f"low_{field}",
            regime_rows=regime_rows_for_field,
            low_surface_fields=(field,),
        )
        row_summary.update(
            {
                "field": field,
                "field_label": label,
                "threshold": _LOW_CONFIDENCE_THRESHOLD,
                "outside_regime_residual_summary": final_split_module._numeric_field_summary(
                    outside_rows,
                    _RESIDUAL_FIELD,
                ),
                "outside_regime_sharply_negative_rate": _residual_bucket_summary(
                    outside_rows
                ).get("strongly_negative_rate"),
            }
        )
        regime_rows.append(row_summary)

    regime_rows.sort(key=_single_regime_sort_key)
    populated_regime_rows = [
        row for row in regime_rows if int(row.get("row_count", 0) or 0) > 0
    ]
    strongest_single_surface_regime = _safe_dict(
        populated_regime_rows[0] if populated_regime_rows else {}
    )
    support_status = str(
        strongest_single_surface_regime.get("support_status") or "insufficient_data"
    )

    return {
        "support_status": support_status,
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "regime_summaries": regime_rows,
        "strongest_single_surface_regime": strongest_single_surface_regime,
    }


def build_joint_shortfall_regimes(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    complete_rows = [
        row
        for row in comparison_rows
        if isinstance(row.get("exact_low_confidence_fields"), tuple)
    ]
    regime_rows: list[dict[str, Any]] = []
    for exact_low_fields, regime_label in _JOINT_SHORTFALL_REGIME_SPECS:
        regime_rows_for_fields = [
            row
            for row in complete_rows
            if row.get(_RESIDUAL_FIELD) is not None
            and tuple(row.get("exact_low_confidence_fields") or ()) == exact_low_fields
        ]
        row_summary = _build_regime_summary(
            regime_label=regime_label,
            regime_rows=regime_rows_for_fields,
            low_surface_fields=exact_low_fields,
        )
        row_summary["low_surface_count"] = len(exact_low_fields)
        regime_rows.append(row_summary)

    regime_rows.sort(key=_joint_regime_sort_key)

    populated_true_joint_regimes = [
        row
        for row in regime_rows
        if int(_safe_dict(row).get("low_surface_count", 0) or 0) >= 2
        and int(_safe_dict(row).get("row_count", 0) or 0) > 0
    ]
    strongest_joint_regime = _safe_dict(
        populated_true_joint_regimes[0] if populated_true_joint_regimes else {}
    )

    reference_no_low_confidence_surface = _safe_dict(
        next(
            (
                row
                for row in regime_rows
                if _safe_dict(row).get("low_surface_count", 0) == 0
            ),
            {},
        )
    )

    support_status = str(
        strongest_joint_regime.get("support_status") or "insufficient_data"
    )

    return {
        "support_status": support_status,
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "regime_summaries": regime_rows,
        "strongest_joint_regime": strongest_joint_regime,
        "reference_no_low_confidence_surface_regime": (
            reference_no_low_confidence_surface
        ),
    }


def build_residual_step_profile(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows_with_complete_low_surface_counts = [
        row
        for row in comparison_rows
        if row.get(_RESIDUAL_FIELD) is not None
        and row.get("low_confidence_surface_count") is not None
    ]

    count_rows: list[dict[str, Any]] = []
    for low_surface_count in range(4):
        count_bucket_rows = [
            row
            for row in rows_with_complete_low_surface_counts
            if int(row.get("low_confidence_surface_count", -1) or -1)
            == low_surface_count
        ]
        count_rows.append(
            {
                "low_confidence_surface_count": low_surface_count,
                "row_count": len(count_bucket_rows),
                "overall_residual_summary": final_split_module._numeric_field_summary(
                    count_bucket_rows,
                    _RESIDUAL_FIELD,
                ),
                _COMPARISON_GROUP_PRESERVED: final_split_module._numeric_field_summary(
                    _comparison_group_rows(
                        count_bucket_rows,
                        _COMPARISON_GROUP_PRESERVED,
                    ),
                    _RESIDUAL_FIELD,
                ),
                _COMPARISON_GROUP_COLLAPSED: final_split_module._numeric_field_summary(
                    _comparison_group_rows(
                        count_bucket_rows,
                        _COMPARISON_GROUP_COLLAPSED,
                    ),
                    _RESIDUAL_FIELD,
                ),
            }
        )

    populated_rows = [
        row
        for row in count_rows
        if int(_safe_dict(row.get("overall_residual_summary")).get("present_row_count", 0))
        > 0
    ]
    support_status = "insufficient_data"
    if len(populated_rows) >= 2:
        support_status = "limited_support"
    if len(populated_rows) >= 3:
        support_status = "supported"

    overall_medians = [
        _to_float(
            _safe_dict(row.get("overall_residual_summary")).get("median"),
            default=None,
        )
        for row in populated_rows
    ]
    monotonic_non_increasing = _is_non_increasing(overall_medians)
    largest_adjacent_drop = _largest_adjacent_drop(populated_rows)
    total_drop = _total_drop(overall_medians)

    profile_shape = "mixed"
    if support_status == "insufficient_data":
        profile_shape = "insufficient_data"
    elif (
        monotonic_non_increasing
        and largest_adjacent_drop.get("median_drop", 0.0) >= 0.12
        and _to_float(total_drop, default=0.0) >= 0.12
        and _to_float(largest_adjacent_drop.get("median_drop"), default=0.0)
        >= 0.6 * _to_float(total_drop, default=0.0)
    ):
        profile_shape = "step_like_drop"
    elif monotonic_non_increasing and _to_float(total_drop, default=0.0) >= 0.12:
        profile_shape = "continuous_deterioration"

    return {
        "support_status": support_status,
        "dimension": "count_of_low_confidence_surfaces_at_or_below_threshold",
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "count_summaries": count_rows,
        "monotonic_non_increasing_overall_medians": monotonic_non_increasing,
        "largest_adjacent_drop": largest_adjacent_drop,
        "total_drop_from_zero_to_highest_count": total_drop,
        "profile_shape": profile_shape,
    }


def build_residual_behavior_classification(
    *,
    summary: dict[str, Any],
    residual_class_comparison: dict[str, Any],
    residual_sign_distribution: dict[str, Any],
    residual_low_confidence_regimes: dict[str, Any],
    joint_shortfall_regimes: dict[str, Any],
    residual_step_profile: dict[str, Any],
) -> dict[str, Any]:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )

    if final_slice_row_count <= 0 or preserved_row_count <= 0 or collapsed_row_count <= 0:
        return {
            "support_status": "unsupported",
            "explanation_class_support_status": "unsupported",
            "strongest_candidate_explanation_class": "unclear",
            "interpretation_status": "comparison_unsupported",
            "strongest_aligned_regime_label": None,
            "unsupported_explanation_classes": [
                "threshold",
                "interaction_effect",
                "piecewise_compression",
                "penalty",
            ],
            "classification_reason": (
                "The final preserved-vs-collapsed slice is incomplete, so residual "
                "mechanism classification is unsupported."
            ),
        }

    if (
        summary.get("comparison_support_status") != "supported"
        or residual_class_comparison.get("support_status") != "supported"
    ):
        return {
            "support_status": "unsupported",
            "explanation_class_support_status": "unsupported",
            "strongest_candidate_explanation_class": "unclear",
            "interpretation_status": "comparison_unsupported",
            "strongest_aligned_regime_label": None,
            "unsupported_explanation_classes": [
                "threshold",
                "interaction_effect",
                "piecewise_compression",
                "penalty",
            ],
            "classification_reason": (
                "Residual class comparison does not clear the family's supported "
                "comparison threshold, so explanation-class claims stay withheld."
            ),
        }

    preserved_summary = _safe_dict(
        residual_class_comparison.get(_COMPARISON_GROUP_PRESERVED)
    )
    collapsed_summary = _safe_dict(
        residual_class_comparison.get(_COMPARISON_GROUP_COLLAPSED)
    )
    preserved_median = _to_float(preserved_summary.get("median"), default=None)
    collapsed_median = _to_float(collapsed_summary.get("median"), default=None)
    pairwise = _safe_dict(residual_class_comparison.get("pairwise_group_ordering"))
    preserved_greater_rate = _to_float(
        pairwise.get("preserved_greater_rate"),
        default=None,
    )
    negative_pocket = _safe_dict(
        residual_sign_distribution.get("sharply_negative_pocket")
    )
    strongest_single_regime = _safe_dict(
        residual_low_confidence_regimes.get("strongest_single_surface_regime")
    )
    strongest_joint_regime = _safe_dict(
        joint_shortfall_regimes.get("strongest_joint_regime")
    )
    best_single_gap = _to_float(
        strongest_single_regime.get("collapsed_sharply_negative_rate_minus_preserved"),
        default=None,
    )
    best_joint_gap = _to_float(
        strongest_joint_regime.get("collapsed_sharply_negative_rate_minus_preserved"),
        default=None,
    )

    collapse_heavy_residual = bool(
        preserved_median is not None
        and collapsed_median is not None
        and preserved_median >= -_NEAR_ZERO_RESIDUAL_ABS_THRESHOLD
        and collapsed_median <= _STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD
        and preserved_greater_rate is not None
        and preserved_greater_rate >= 0.75
    )
    threshold_supported = (
        negative_pocket.get("support_status") == "supported"
        and negative_pocket.get("concentration_status") == "collapsed_concentrated"
    )
    interaction_supported = bool(
        strongest_joint_regime.get("support_status") == "supported"
        and int(strongest_joint_regime.get("low_surface_count", 0) or 0) >= 2
        and best_joint_gap is not None
        and best_joint_gap >= 0.45
        and (
            best_single_gap is None
            or best_joint_gap - best_single_gap >= 0.15
        )
    )
    piecewise_supported = bool(
        residual_step_profile.get("support_status") == "supported"
        and residual_step_profile.get("profile_shape") == "step_like_drop"
    )

    strongest_candidate_explanation_class = "unclear"
    strongest_aligned_regime_label = None
    classification_reason = (
        "Residual comparison is supported, but no threshold, interaction, piecewise "
        "compression, or general penalty reading clears the report's conservative "
        "classification rules."
    )

    if interaction_supported:
        strongest_candidate_explanation_class = "interaction_effect"
        strongest_aligned_regime_label = strongest_joint_regime.get("regime_label")
        classification_reason = (
            "The strongest collapsed-side negative residual concentration sits in a "
            "multi-surface low-confidence regime and materially exceeds the strongest "
            "single-surface regime, so an interaction effect is the best-supported "
            "remaining explanation class."
        )
    elif threshold_supported:
        strongest_candidate_explanation_class = "threshold"
        strongest_aligned_regime_label = "sharply_negative_residual_pocket"
        classification_reason = (
            "Collapsed rows cluster in the sharply negative residual pocket while "
            "preserved rows mostly avoid it, so the remaining structure looks more "
            "threshold-like than uniformly continuous."
        )
    elif piecewise_supported:
        strongest_candidate_explanation_class = "piecewise_compression"
        strongest_aligned_regime_label = "low_confidence_surface_count_profile"
        classification_reason = (
            "Residual medians deteriorate monotonically across ordered low-confidence "
            "count bins with a large higher-count step, so a piecewise-compression "
            "reading is better supported than a single flat penalty."
        )
    elif collapse_heavy_residual:
        strongest_candidate_explanation_class = "penalty"
        strongest_aligned_regime_label = strongest_single_regime.get("regime_label")
        classification_reason = (
            "Preserved residual stays near zero while collapsed residual is strongly "
            "negative, but the remaining evidence is not clean enough to elevate a "
            "threshold, piecewise, or interaction class above a simpler collapse-side "
            "penalty reading."
        )

    unsupported_explanation_classes = [
        explanation_class
        for explanation_class in (
            "threshold",
            "interaction_effect",
            "piecewise_compression",
            "penalty",
        )
        if explanation_class != strongest_candidate_explanation_class
    ]
    if strongest_candidate_explanation_class == "unclear":
        unsupported_explanation_classes = [
            "threshold",
            "interaction_effect",
            "piecewise_compression",
            "penalty",
        ]

    interpretation_status = "no_clear_explanation_class_supported"
    if strongest_candidate_explanation_class == "threshold":
        interpretation_status = "threshold_explanation_supported"
    elif strongest_candidate_explanation_class == "interaction_effect":
        interpretation_status = "interaction_effect_supported"
    elif strongest_candidate_explanation_class == "piecewise_compression":
        interpretation_status = "piecewise_compression_supported"
    elif strongest_candidate_explanation_class == "penalty":
        interpretation_status = "penalty_explanation_supported"

    return {
        "support_status": "supported",
        "explanation_class_support_status": (
            "supported"
            if strongest_candidate_explanation_class != "unclear"
            else "unsupported"
        ),
        "strongest_candidate_explanation_class": strongest_candidate_explanation_class,
        "interpretation_status": interpretation_status,
        "strongest_aligned_regime_label": strongest_aligned_regime_label,
        "collapse_heavy_residual_supported": collapse_heavy_residual,
        "threshold_supported": threshold_supported,
        "interaction_effect_supported": interaction_supported,
        "piecewise_compression_supported": piecewise_supported,
        "unsupported_explanation_classes": unsupported_explanation_classes,
        "classification_reason": classification_reason,
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    baseline_reference: dict[str, Any],
    actual_rule_engine_confidence_reference: dict[str, Any],
    residual_class_comparison: dict[str, Any],
    residual_sign_distribution: dict[str, Any],
    residual_surface_correlations: dict[str, Any],
    residual_low_confidence_regimes: dict[str, Any],
    joint_shortfall_regimes: dict[str, Any],
    residual_step_profile: dict[str, Any],
    residual_behavior_classification: dict[str, Any],
) -> dict[str, Any]:
    actual_preserved = _safe_dict(
        actual_rule_engine_confidence_reference.get(_COMPARISON_GROUP_PRESERVED)
    )
    actual_collapsed = _safe_dict(
        actual_rule_engine_confidence_reference.get(_COMPARISON_GROUP_COLLAPSED)
    )
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
    strongest_raw_surface = _safe_dict(
        residual_surface_correlations.get("strongest_raw_surface_alignment")
    )
    strongest_shortfall = _safe_dict(
        residual_surface_correlations.get("strongest_shortfall_alignment")
    )
    strongest_single_regime = _safe_dict(
        residual_low_confidence_regimes.get("strongest_single_surface_regime")
    )
    strongest_joint_regime = _safe_dict(
        joint_shortfall_regimes.get("strongest_joint_regime")
    )
    largest_adjacent_drop = _safe_dict(
        residual_step_profile.get("largest_adjacent_drop")
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
            "Actual persisted rule_engine_confidence medians: preserved="
            f"{actual_preserved.get('median')}, collapsed={actual_collapsed.get('median')}."
        ),
        (
            "Residual medians after subtracting the weighted baseline: preserved="
            f"{residual_preserved.get('median')}, collapsed={residual_collapsed.get('median')}, "
            "preserved_greater_pair_rate="
            f"{_safe_dict(residual_class_comparison.get('pairwise_group_ordering')).get('preserved_greater_rate')}."
        ),
        (
            "Sharply negative residual pocket rate: preserved="
            f"{negative_pocket.get('preserved_rate')}, collapsed={negative_pocket.get('collapsed_rate')}, "
            "gap="
            f"{negative_pocket.get('collapsed_minus_preserved_rate')}."
        ),
    ]

    supporting_descriptive_evidence: list[str] = []
    if strongest_raw_surface.get("field"):
        supporting_descriptive_evidence.append(
            "Strongest raw-surface residual alignment="
            f"{strongest_raw_surface.get('field')} "
            f"(correlation={strongest_raw_surface.get('pearson_correlation_to_residual')})."
        )
    if strongest_shortfall.get("field"):
        supporting_descriptive_evidence.append(
            "Strongest shortfall residual alignment="
            f"{strongest_shortfall.get('field')} "
            f"(correlation={strongest_shortfall.get('pearson_correlation_to_residual')})."
        )
    if strongest_single_regime.get("regime_label"):
        supporting_descriptive_evidence.append(
            "Strongest single low-confidence regime="
            f"{strongest_single_regime.get('regime_label')} "
            "with collapsed_minus_preserved_sharply_negative_rate="
            f"{strongest_single_regime.get('collapsed_sharply_negative_rate_minus_preserved')}."
        )
    if strongest_joint_regime.get("regime_label"):
        supporting_descriptive_evidence.append(
            "Strongest joint shortfall regime="
            f"{strongest_joint_regime.get('regime_label')} "
            "with collapsed_minus_preserved_sharply_negative_rate="
            f"{strongest_joint_regime.get('collapsed_sharply_negative_rate_minus_preserved')}."
        )
    if residual_step_profile.get("profile_shape"):
        supporting_descriptive_evidence.append(
            "Ordered low-confidence-count profile shape="
            f"{residual_step_profile.get('profile_shape')}, largest_adjacent_drop="
            f"{largest_adjacent_drop.get('median_drop')}."
        )

    strongest_candidate_explanation_class = str(
        residual_behavior_classification.get(
            "strongest_candidate_explanation_class"
        )
        or "unclear"
    )
    inference = [
        residual_behavior_classification.get("classification_reason"),
    ]
    if strongest_candidate_explanation_class == "unclear":
        inference.append(
            "Residual remains a real collapse-heavy diagnostic target only if the "
            "supported class comparison itself is strong; otherwise the mechanism "
            "should stay open rather than being promoted from one descriptive surface."
        )

    uncertainties = [
        "This report remains descriptive: negative residual means actual persisted rule_engine_confidence is more severe than the weighted baseline predicted, but that still does not prove the literal hidden production rule shape.",
        "Fixed low-confidence and sharply-negative-residual thresholds are only report heuristics for regime grouping.",
        "Correlations and grouped residual summaries indicate alignment, not causal direction.",
    ]
    if summary.get("comparison_support_status") != "supported":
        uncertainties.append(
            "Because the final preserved-vs-collapsed slice is below the family's normal support threshold, explanation-class claims stay withheld."
        )
    if residual_behavior_classification.get("strongest_candidate_explanation_class") == "unclear":
        uncertainties.append(
            "No explanation class clears the report's conservative thresholds on this run, so threshold, interaction, piecewise, and generic penalty readings all remain unproven."
        )

    return {
        "observed_residual_behavior": facts,
        "supporting_descriptive_evidence": supporting_descriptive_evidence,
        "strongest_candidate_explanation_class": strongest_candidate_explanation_class,
        "unsupported_explanation_classes": _safe_list(
            residual_behavior_classification.get("unsupported_explanation_classes")
        ),
        "facts": [item for item in facts if item],
        "inference": [item for item in inference if item],
        "uncertainty": uncertainties,
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    classification = _safe_dict(widest.get("residual_behavior_classification"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "baseline_name": _BASELINE_NAME,
        "strongest_candidate_explanation_class": classification.get(
            "strongest_candidate_explanation_class"
        ),
        "interpretation_status": classification.get("interpretation_status"),
        "observed": _safe_list(interpretation.get("facts")),
        "strongly_suggested": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            residual_behavior_classification=classification,
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
            f"- baseline_name: {headline.get('baseline_name', _BASELINE_NAME)}"
        )
        lines.append(
            "- strongest_candidate_explanation_class: "
            f"{headline.get('strongest_candidate_explanation_class', 'unclear')}"
        )
        lines.append(
            "- strongest_joint_regime: "
            f"{headline.get('strongest_joint_regime', 'none')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{headline.get('interpretation_status', 'unknown')}"
        )
        for fact in _safe_list(interpretation.get("facts"))[:5]:
            lines.append(f"- fact: {fact}")
        for item in _safe_list(interpretation.get("inference"))[:3]:
            lines.append(f"- inference: {item}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        "- strongest_candidate_explanation_class: "
        f"{final_assessment.get('strongest_candidate_explanation_class', 'unclear')}"
    )
    lines.append(
        f"- interpretation_status: {final_assessment.get('interpretation_status', 'unknown')}"
    )
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


def _build_metric_row(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    field: str,
    field_label: str,
    formula: str | None,
    component_fields: Sequence[str],
    actual_rule_engine_median_gap: float | None,
) -> dict[str, Any]:
    comparison = threshold_module.build_numeric_field_comparison_for_fields(
        comparison_rows=comparison_rows,
        field_specs=(
            (
                field,
                field_label,
                lambda row, *, _field=field: _to_float(row.get(_field), default=None),
            ),
        ),
    )
    base_row = _safe_dict(
        _safe_list(comparison.get("field_comparisons"))[0]
        if _safe_list(comparison.get("field_comparisons"))
        else {}
    )
    metric_row = aggregate_shape_module._build_metric_row(
        base_row=base_row,
        comparison_rows=comparison_rows,
        formula=formula,
        component_fields=component_fields,
        actual_rule_engine_median_gap=actual_rule_engine_median_gap,
    )
    return {
        **metric_row,
        "support_status": comparison.get("support_status", "limited_support"),
    }


def _build_regime_summary(
    *,
    regime_label: str,
    regime_rows: Sequence[dict[str, Any]],
    low_surface_fields: Sequence[str],
) -> dict[str, Any]:
    preserved_rows = _comparison_group_rows(
        regime_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_rows = _comparison_group_rows(
        regime_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )
    overall_summary = final_split_module._numeric_field_summary(
        regime_rows,
        _RESIDUAL_FIELD,
    )
    preserved_summary = final_split_module._numeric_field_summary(
        preserved_rows,
        _RESIDUAL_FIELD,
    )
    collapsed_summary = final_split_module._numeric_field_summary(
        collapsed_rows,
        _RESIDUAL_FIELD,
    )
    sign_distribution = _residual_sign_distribution_for_rows(regime_rows)

    return {
        "regime_label": regime_label,
        "low_surface_fields": list(low_surface_fields),
        "support_status": _regime_support_status(
            total_row_count=len(regime_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "row_count": len(regime_rows),
        "preserved_row_count": len(preserved_rows),
        "collapsed_row_count": len(collapsed_rows),
        "collapsed_rate_within_regime": _safe_ratio(len(collapsed_rows), len(regime_rows)),
        "residual_overall_summary": overall_summary,
        _COMPARISON_GROUP_PRESERVED: preserved_summary,
        _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
        "pairwise_group_ordering": aggregate_shape_module._pairwise_group_ordering(
            comparison_rows=regime_rows,
            field=_RESIDUAL_FIELD,
        ),
        "sharply_negative_distribution": sign_distribution,
        "collapsed_sharply_negative_rate": _safe_dict(
            sign_distribution.get(_COMPARISON_GROUP_COLLAPSED)
        ).get("strongly_negative_rate"),
        "preserved_sharply_negative_rate": _safe_dict(
            sign_distribution.get(_COMPARISON_GROUP_PRESERVED)
        ).get("strongly_negative_rate"),
        "collapsed_sharply_negative_rate_minus_preserved": round(
            _to_float(
                _safe_dict(sign_distribution.get(_COMPARISON_GROUP_COLLAPSED)).get(
                    "strongly_negative_rate"
                ),
                default=0.0,
            )
            - _to_float(
                _safe_dict(sign_distribution.get(_COMPARISON_GROUP_PRESERVED)).get(
                    "strongly_negative_rate"
                ),
                default=0.0,
            ),
            6,
        ),
    }


def _residual_bucket_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row[_RESIDUAL_FIELD])
        for row in rows
        if row.get(_RESIDUAL_FIELD) is not None
    ]
    row_count = len(rows)
    present_row_count = len(values)
    missing_row_count = row_count - present_row_count
    strongly_negative_count = sum(
        1 for value in values if value <= _STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD
    )
    negative_count = sum(
        1
        for value in values
        if _STRONGLY_NEGATIVE_RESIDUAL_THRESHOLD < value < -_NEAR_ZERO_RESIDUAL_ABS_THRESHOLD
    )
    near_zero_count = sum(
        1
        for value in values
        if abs(value) <= _NEAR_ZERO_RESIDUAL_ABS_THRESHOLD
    )
    positive_count = sum(
        1 for value in values if value > _NEAR_ZERO_RESIDUAL_ABS_THRESHOLD
    )
    return {
        "row_count": row_count,
        "present_row_count": present_row_count,
        "missing_row_count": missing_row_count,
        "present_rate": _safe_ratio(present_row_count, row_count),
        "strongly_negative_count": strongly_negative_count,
        "strongly_negative_rate": _safe_ratio(strongly_negative_count, present_row_count),
        "negative_count": negative_count,
        "negative_rate": _safe_ratio(negative_count, present_row_count),
        "near_zero_count": near_zero_count,
        "near_zero_rate": _safe_ratio(near_zero_count, present_row_count),
        "positive_count": positive_count,
        "positive_rate": _safe_ratio(positive_count, present_row_count),
    }


def _residual_sign_distribution_for_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        _COMPARISON_GROUP_PRESERVED: _residual_bucket_summary(
            _comparison_group_rows(rows, _COMPARISON_GROUP_PRESERVED)
        ),
        _COMPARISON_GROUP_COLLAPSED: _residual_bucket_summary(
            _comparison_group_rows(rows, _COMPARISON_GROUP_COLLAPSED)
        ),
    }


def _correlation_between_fields(
    *,
    rows: Sequence[dict[str, Any]],
    left_field: str,
    right_field: str,
) -> dict[str, Any]:
    pairs = [
        (
            float(row[left_field]),
            float(row[right_field]),
        )
        for row in rows
        if row.get(left_field) is not None and row.get(right_field) is not None
    ]
    if len(pairs) < 2:
        return {
            "paired_row_count": len(pairs),
            "pearson_correlation": None,
        }

    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in pairs
    )
    left_variance = sum((value - left_mean) ** 2 for value in left_values)
    right_variance = sum((value - right_mean) ** 2 for value in right_values)
    if left_variance <= 0 or right_variance <= 0:
        return {
            "paired_row_count": len(pairs),
            "pearson_correlation": None,
        }

    correlation = numerator / math.sqrt(left_variance * right_variance)
    return {
        "paired_row_count": len(pairs),
        "pearson_correlation": round(correlation, 6),
    }


def _comparison_group_rows(
    rows: Sequence[dict[str, Any]],
    comparison_group: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("comparison_group") or "") == comparison_group
    ]


def _correlation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("alignment_to_negative_residual") == "aligned" else 1,
        -_to_float(row.get("alignment_strength"), default=0.0),
        _SURFACE_FIELD_ORDER.get(str(row.get("field") or ""), 99),
        str(row.get("field") or ""),
    )


def _single_regime_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    collapsed_summary = _safe_dict(row.get(_COMPARISON_GROUP_COLLAPSED))
    return (
        _REGIME_SUPPORT_ORDER.get(str(row.get("support_status") or ""), 99),
        -_to_float(
            row.get("collapsed_sharply_negative_rate_minus_preserved"),
            default=-1.0,
        ),
        -abs(_to_float(collapsed_summary.get("median"), default=0.0)),
        _SURFACE_FIELD_ORDER.get(str(row.get("field") or ""), 99),
    )


def _joint_regime_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    collapsed_summary = _safe_dict(row.get(_COMPARISON_GROUP_COLLAPSED))
    return (
        _REGIME_SUPPORT_ORDER.get(str(row.get("support_status") or ""), 99),
        -_to_float(
            row.get("collapsed_sharply_negative_rate_minus_preserved"),
            default=-1.0,
        ),
        -int(row.get("low_surface_count", 0) or 0),
        -abs(_to_float(collapsed_summary.get("median"), default=0.0)),
        str(row.get("regime_label") or ""),
    )


def _is_non_increasing(values: Sequence[float | None]) -> bool:
    numeric_values = [value for value in values if value is not None]
    if len(numeric_values) < 2:
        return False
    previous_value = numeric_values[0]
    for value in numeric_values[1:]:
        if value > previous_value + 1e-9:
            return False
        previous_value = value
    return True


def _largest_adjacent_drop(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best_row = {
        "from_low_confidence_surface_count": None,
        "to_low_confidence_surface_count": None,
        "median_drop": 0.0,
    }
    for left_row, right_row in zip(rows, rows[1:]):
        left_median = _to_float(
            _safe_dict(left_row.get("overall_residual_summary")).get("median"),
            default=None,
        )
        right_median = _to_float(
            _safe_dict(right_row.get("overall_residual_summary")).get("median"),
            default=None,
        )
        if left_median is None or right_median is None:
            continue
        drop = round(left_median - right_median, 6)
        if drop > _to_float(best_row.get("median_drop"), default=0.0):
            best_row = {
                "from_low_confidence_surface_count": left_row.get(
                    "low_confidence_surface_count"
                ),
                "to_low_confidence_surface_count": right_row.get(
                    "low_confidence_surface_count"
                ),
                "median_drop": drop,
            }
    return best_row


def _total_drop(values: Sequence[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if len(numeric_values) < 2:
        return None
    return round(numeric_values[0] - numeric_values[-1], 6)


def _shortfall(value: float | None) -> float | None:
    if value is None:
        return None
    return round(1.0 - float(value), 6)


def _low_confidence_flag(value: float | None) -> bool | None:
    if value is None:
        return None
    return bool(value <= _LOW_CONFIDENCE_THRESHOLD)


def _low_confidence_surface_count(
    low_flags: Sequence[bool | None],
) -> int | None:
    if any(flag is None for flag in low_flags):
        return None
    return sum(1 for flag in low_flags if flag)


def _exact_low_confidence_fields(
    *,
    low_setup_confidence: bool | None,
    low_context_bias_family: bool | None,
    low_selected_strategy: bool | None,
) -> tuple[str, ...] | None:
    flags = (
        ("setup_layer_confidence", low_setup_confidence),
        (_CONTEXT_BIAS_FAMILY_FIELD, low_context_bias_family),
        ("selected_strategy_confidence", low_selected_strategy),
    )
    if any(flag is None for _, flag in flags):
        return None
    return tuple(field for field, flag in flags if flag)


def _regime_support_status(
    *,
    total_row_count: int,
    collapsed_row_count: int,
) -> str:
    if total_row_count >= _MIN_REGIME_ROW_COUNT and collapsed_row_count >= _MIN_REGIME_COLLAPSED_ROW_COUNT:
        return "supported"
    if total_row_count > 0:
        return "limited_support"
    return "insufficient_data"


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    residual_behavior_classification: dict[str, Any],
) -> str:
    final_slice_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    interpretation_status = str(
        residual_behavior_classification.get("interpretation_status")
        or "comparison_unsupported"
    )

    if final_slice_row_count <= 0:
        return (
            "No rows reached the final fully aligned plus rule-bias-aligned slice in "
            "the widest configuration, so weighted-aggregate residual diagnosis is "
            "unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one side "
            "of the preserved-vs-collapsed comparison is missing, so weighted-aggregate "
            "residual diagnosis remains incomplete."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but residual mechanism claims stay withheld "
            "because the preserved-vs-collapsed residual comparison does not clear the "
            "family's normal support threshold."
        )
    if interpretation_status == "interaction_effect_supported":
        return (
            "The widest configuration suggests that the residual remaining after the "
            "weighted baseline is most consistent with a joint shortfall interaction "
            "rather than a single-surface weakness alone."
        )
    if interpretation_status == "threshold_explanation_supported":
        return (
            "The widest configuration suggests that the residual remaining after the "
            "weighted baseline clusters in a sharply negative pocket that preserved rows "
            "mostly avoid, so the remaining structure looks threshold-like."
        )
    if interpretation_status == "piecewise_compression_supported":
        return (
            "The widest configuration suggests that the residual remaining after the "
            "weighted baseline deteriorates in ordered low-confidence-count steps, so a "
            "piecewise compression is the best-supported class."
        )
    if interpretation_status == "penalty_explanation_supported":
        return (
            "The widest configuration suggests that residual remains mostly a "
            "collapse-heavy penalty layered on top of the weighted baseline, without "
            "stronger support for a threshold, piecewise, or joint interaction class."
        )
    return (
        "The widest configuration confirms that residual remains after the weighted "
        "baseline, but the remaining penalty/compression/interaction structure stays "
        "too ambiguous to promote a single explanation class."
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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _difference_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _safe_abs_float(value: Any) -> float | None:
    converted = _to_float(value, default=None)
    if converted is None:
        return None
    return abs(converted)


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
