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
    selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report as origin_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report as residual_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_signature_conditioned_pocket_"
    "severity_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Signature-Conditioned Pocket "
    "Severity Diagnosis Report"
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
_CONTEXT_BIAS_FAMILY_FIELD = residual_module._CONTEXT_BIAS_FAMILY_FIELD
_LOW_CONFIDENCE_THRESHOLD = residual_module._LOW_CONFIDENCE_THRESHOLD

_POCKET_MAX_THRESHOLD = 0.25
_MIN_GROUP_SUPPORT_ROWS = 10
_MATERIAL_SEVERITY_GAP = 0.05

_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE = (
    _CONTEXT_BIAS_FAMILY_FIELD,
    "selected_strategy_confidence",
)
_DOMINANT_SIGNATURE_LABEL = " + ".join(_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE)
_LOW_SURFACE_COUNT_ANCHOR = 2

_SIGNATURE_CONDITIONED_POCKET_GROUP_LABEL = (
    "signature_conditioned_compressed_low_band_pocket"
)
_SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL = (
    "signature_conditioned_preserved_outside_compressed_low_band_pocket"
)
_SIGNATURE_CONDITIONED_OUTSIDE_GROUP_LABEL = (
    "signature_conditioned_all_outside_compressed_low_band_pocket"
)

_FIELD_SPECS: tuple[dict[str, Any], ...] = (
    origin_module._FIELD_SPEC_MAP[_RULE_ENGINE_CONFIDENCE_FIELD],
    origin_module._FIELD_SPEC_MAP[_BASELINE_NAME],
    origin_module._FIELD_SPEC_MAP[_RESIDUAL_FIELD],
    origin_module._FIELD_SPEC_MAP[_CONTEXT_BIAS_FAMILY_FIELD],
    origin_module._FIELD_SPEC_MAP["selected_strategy_confidence"],
    origin_module._FIELD_SPEC_MAP["setup_layer_confidence"],
    origin_module._FIELD_SPEC_MAP["trigger_layer_confidence"],
)
_FIELD_SPEC_MAP = {str(spec["field"]): dict(spec) for spec in _FIELD_SPECS}

_PRIMARY_SEVERITY_FIELDS = (
    _CONTEXT_BIAS_FAMILY_FIELD,
    "selected_strategy_confidence",
    _RESIDUAL_FIELD,
)
_SIGNATURE_DEPTH_FIELDS = (
    _CONTEXT_BIAS_FAMILY_FIELD,
    "selected_strategy_confidence",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report that holds the observed dominant exact "
            "low-confidence signature fixed and tests whether compressed-pocket "
            "membership is better explained by severity depth inside that signature."
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
            "Retained for parity with sibling reports. This diagnosis reuses only "
            "the final preserved-vs-collapsed slice."
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

    result = run_selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    signature_summary = _safe_dict(report.get("signature_conditioning_summary"))
    severity_assessment = _safe_dict(report.get("severity_depth_assessment"))
    setup_reading = _safe_dict(report.get("setup_secondary_reading"))
    trigger_reading = _safe_dict(report.get("trigger_negative_control_reading"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    report.get("widest_configuration")
                ).get("display_name"),
                "final_rule_bias_aligned_row_count": report.get(
                    "final_rule_bias_aligned_row_count",
                    0,
                ),
                "signature_conditioned_row_count": signature_summary.get(
                    "signature_conditioned_row_count",
                    0,
                ),
                "signature_conditioned_pocket_row_count": signature_summary.get(
                    "signature_conditioned_pocket_row_count",
                    0,
                ),
                "signature_conditioned_preserved_outside_row_count": (
                    signature_summary.get(
                        "signature_conditioned_preserved_outside_pocket_row_count",
                        0,
                    )
                ),
                "selected_reference_group_label": severity_assessment.get(
                    "selected_reference_group_label"
                ),
                "severity_depth_status": severity_assessment.get(
                    "severity_depth_status"
                ),
                "setup_separation_status": setup_reading.get(
                    "setup_separation_status"
                ),
                "trigger_negative_control_status": trigger_reading.get(
                    "trigger_negative_control_status"
                ),
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report(
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
        "pocket_definition": {
            "field": _RULE_ENGINE_CONFIDENCE_FIELD,
            "operator": "<=",
            "threshold": _POCKET_MAX_THRESHOLD,
            "label": "compressed_low_band_pocket",
        },
        "dominant_signature_definition": _dominant_signature_definition(),
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
        "baseline_reference": _safe_dict(widest_summary.get("baseline_reference")),
        "signature_conditioning_summary": _safe_dict(
            widest_summary.get("signature_conditioning_summary")
        ),
        "conditioned_slice_field_summaries": _safe_dict(
            widest_summary.get("conditioned_slice_field_summaries")
        ),
        "pocket_vs_preserved_outside_conditioned_comparison": _safe_dict(
            widest_summary.get("pocket_vs_preserved_outside_conditioned_comparison")
        ),
        "pocket_vs_all_outside_conditioned_comparison": _safe_dict(
            widest_summary.get("pocket_vs_all_outside_conditioned_comparison")
        ),
        "reference_selection": _safe_dict(widest_summary.get("reference_selection")),
        "selected_conditioned_comparison": _safe_dict(
            widest_summary.get("selected_conditioned_comparison")
        ),
        "contributor_family_conditioned_comparison": _safe_dict(
            widest_summary.get("contributor_family_conditioned_comparison")
        ),
        "severity_depth_assessment": _safe_dict(
            widest_summary.get("severity_depth_assessment")
        ),
        "setup_secondary_reading": _safe_dict(
            widest_summary.get("setup_secondary_reading")
        ),
        "trigger_negative_control_reading": _safe_dict(
            widest_summary.get("trigger_negative_control_reading")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis chain.",
            "The compressed pocket is fixed to actual rule_engine_confidence <= 0.25.",
            "The dominant exact low-confidence signature is fixed to context_bias_family_mean + selected_strategy_confidence with low_surface_count=2.",
            "Residual remains fixed to rule_engine_confidence minus weighted_mean_setup_emphasis.",
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
    residual_rows = [
        origin_module._prepare_origin_residual_row(
            residual_module.build_residual_row(row)
        )
        for row in comparison_rows
    ]

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

    row_sets = build_signature_conditioned_row_sets(residual_rows)
    signature_conditioning_summary = build_signature_conditioning_summary(
        row_sets=row_sets
    )
    conditioned_slice_field_summaries = build_conditioned_slice_field_summaries(
        row_sets=row_sets
    )
    pocket_vs_preserved_outside = build_conditioned_group_comparison(
        left_rows=row_sets["signature_conditioned_pocket_rows"],
        right_rows=row_sets[
            "signature_conditioned_preserved_outside_pocket_rows"
        ],
        left_group_label=_SIGNATURE_CONDITIONED_POCKET_GROUP_LABEL,
        right_group_label=_SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL,
    )
    pocket_vs_all_outside = build_conditioned_group_comparison(
        left_rows=row_sets["signature_conditioned_pocket_rows"],
        right_rows=row_sets["signature_conditioned_outside_pocket_rows"],
        left_group_label=_SIGNATURE_CONDITIONED_POCKET_GROUP_LABEL,
        right_group_label=_SIGNATURE_CONDITIONED_OUTSIDE_GROUP_LABEL,
    )
    reference_selection = build_reference_selection(
        primary_comparison=pocket_vs_preserved_outside,
        fallback_comparison=pocket_vs_all_outside,
    )
    selected_comparison = _safe_dict(
        reference_selection.get("selected_comparison")
    )
    contributor_family_comparison = origin_module.build_contributor_family_summary(
        left_rows=row_sets["signature_conditioned_pocket_rows"],
        right_rows=_reference_rows_for_selection(
            row_sets=row_sets,
            reference_selection=reference_selection,
        ),
        left_group_label=_SIGNATURE_CONDITIONED_POCKET_GROUP_LABEL,
        right_group_label=str(
            reference_selection.get("selected_reference_group_label") or ""
        ),
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    setup_secondary_reading = build_setup_secondary_reading(
        selected_comparison=selected_comparison
    )
    trigger_negative_control_reading = build_trigger_negative_control_reading(
        contributor_family_comparison=contributor_family_comparison
    )
    severity_depth_assessment = build_severity_depth_assessment(
        signature_conditioning_summary=signature_conditioning_summary,
        reference_selection=reference_selection,
        selected_comparison=selected_comparison,
        setup_secondary_reading=setup_secondary_reading,
        trigger_negative_control_reading=trigger_negative_control_reading,
    )
    interpretation = build_interpretation(
        summary=summary,
        signature_conditioning_summary=signature_conditioning_summary,
        reference_selection=reference_selection,
        severity_depth_assessment=severity_depth_assessment,
        setup_secondary_reading=setup_secondary_reading,
        trigger_negative_control_reading=trigger_negative_control_reading,
    )
    limitations = build_limitations(
        summary=summary,
        signature_conditioning_summary=signature_conditioning_summary,
        reference_selection=reference_selection,
        severity_depth_assessment=severity_depth_assessment,
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
            "signature_conditioned_row_count": signature_conditioning_summary.get(
                "signature_conditioned_row_count"
            ),
            "signature_conditioned_pocket_row_count": (
                signature_conditioning_summary.get(
                    "signature_conditioned_pocket_row_count"
                )
            ),
            "signature_conditioned_preserved_outside_pocket_row_count": (
                signature_conditioning_summary.get(
                    "signature_conditioned_preserved_outside_pocket_row_count"
                )
            ),
            "selected_reference_group_label": severity_depth_assessment.get(
                "selected_reference_group_label"
            ),
            "severity_depth_status": severity_depth_assessment.get(
                "severity_depth_status"
            ),
            "interpretation_status": interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "actual_rule_engine_confidence_reference": actual_rule_engine_confidence_reference,
        "baseline_reference": baseline_reference,
        "signature_conditioning_summary": signature_conditioning_summary,
        "conditioned_slice_field_summaries": conditioned_slice_field_summaries,
        "pocket_vs_preserved_outside_conditioned_comparison": (
            pocket_vs_preserved_outside
        ),
        "pocket_vs_all_outside_conditioned_comparison": pocket_vs_all_outside,
        "reference_selection": {
            key: value
            for key, value in reference_selection.items()
            if key != "selected_comparison"
        },
        "selected_conditioned_comparison": selected_comparison,
        "contributor_family_conditioned_comparison": (
            contributor_family_comparison
        ),
        "severity_depth_assessment": severity_depth_assessment,
        "setup_secondary_reading": setup_secondary_reading,
        "trigger_negative_control_reading": trigger_negative_control_reading,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_signature_conditioned_row_sets(
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible_rows = [
        row
        for row in comparison_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    ]
    complete_signature_rows = origin_module._rows_with_complete_joint_signature(
        eligible_rows
    )
    signature_conditioned_rows = [
        row for row in complete_signature_rows if _has_fixed_dominant_signature(row)
    ]
    signature_conditioned_pocket_rows = [
        row
        for row in signature_conditioned_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=99.0)
        <= _POCKET_MAX_THRESHOLD
    ]
    signature_conditioned_outside_pocket_rows = [
        row
        for row in signature_conditioned_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=99.0)
        > _POCKET_MAX_THRESHOLD
    ]
    signature_conditioned_preserved_outside_pocket_rows = _comparison_group_rows(
        signature_conditioned_outside_pocket_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    signature_conditioned_collapsed_outside_pocket_rows = _comparison_group_rows(
        signature_conditioned_outside_pocket_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    return {
        "eligible_rows": eligible_rows,
        "missing_rule_engine_confidence_row_count": len(comparison_rows)
        - len(eligible_rows),
        "complete_signature_rows": complete_signature_rows,
        "signature_conditioned_rows": signature_conditioned_rows,
        "signature_conditioned_pocket_rows": signature_conditioned_pocket_rows,
        "signature_conditioned_outside_pocket_rows": (
            signature_conditioned_outside_pocket_rows
        ),
        "signature_conditioned_preserved_outside_pocket_rows": (
            signature_conditioned_preserved_outside_pocket_rows
        ),
        "signature_conditioned_collapsed_outside_pocket_rows": (
            signature_conditioned_collapsed_outside_pocket_rows
        ),
    }


def build_signature_conditioning_summary(
    *,
    row_sets: dict[str, Any],
) -> dict[str, Any]:
    eligible_rows = _safe_list(row_sets.get("eligible_rows"))
    complete_signature_rows = _safe_list(row_sets.get("complete_signature_rows"))
    signature_conditioned_rows = _safe_list(
        row_sets.get("signature_conditioned_rows")
    )
    pocket_rows = _safe_list(row_sets.get("signature_conditioned_pocket_rows"))
    outside_rows = _safe_list(
        row_sets.get("signature_conditioned_outside_pocket_rows")
    )
    preserved_outside_rows = _safe_list(
        row_sets.get("signature_conditioned_preserved_outside_pocket_rows")
    )
    collapsed_outside_rows = _safe_list(
        row_sets.get("signature_conditioned_collapsed_outside_pocket_rows")
    )
    return {
        "dominant_signature": _dominant_signature_definition(),
        "eligible_rule_engine_confidence_row_count": len(eligible_rows),
        "missing_rule_engine_confidence_row_count": int(
            row_sets.get("missing_rule_engine_confidence_row_count", 0) or 0
        ),
        "complete_signature_row_count": len(complete_signature_rows),
        "signature_conditioned_row_count": len(signature_conditioned_rows),
        "signature_conditioned_rate_within_complete_rows": _safe_ratio(
            len(signature_conditioned_rows),
            len(complete_signature_rows),
        ),
        "signature_conditioned_pocket_row_count": len(pocket_rows),
        "signature_conditioned_outside_pocket_row_count": len(outside_rows),
        "signature_conditioned_preserved_outside_pocket_row_count": len(
            preserved_outside_rows
        ),
        "signature_conditioned_collapsed_outside_pocket_row_count": len(
            collapsed_outside_rows
        ),
        "pocket_rate_within_signature_conditioned_rows": _safe_ratio(
            len(pocket_rows),
            len(signature_conditioned_rows),
        ),
        "preserved_outside_reference_support_status": _group_support_status(
            left_row_count=len(pocket_rows),
            right_row_count=len(preserved_outside_rows),
            min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
        ),
        "all_outside_reference_support_status": _group_support_status(
            left_row_count=len(pocket_rows),
            right_row_count=len(outside_rows),
            min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
        ),
        "signature_presence_status": "held_constant_by_design",
    }


def build_conditioned_slice_field_summaries(
    *,
    row_sets: dict[str, Any],
) -> dict[str, Any]:
    groups = {
        "signature_conditioned_all_rows": _safe_list(
            row_sets.get("signature_conditioned_rows")
        ),
        _SIGNATURE_CONDITIONED_POCKET_GROUP_LABEL: _safe_list(
            row_sets.get("signature_conditioned_pocket_rows")
        ),
        _SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL: _safe_list(
            row_sets.get("signature_conditioned_preserved_outside_pocket_rows")
        ),
        _SIGNATURE_CONDITIONED_OUTSIDE_GROUP_LABEL: _safe_list(
            row_sets.get("signature_conditioned_outside_pocket_rows")
        ),
    }
    return {
        group_label: {
            "row_count": len(rows),
            "field_summaries": {
                str(spec["field"]): final_split_module._numeric_field_summary(
                    rows,
                    str(spec["field"]),
                )
                for spec in _FIELD_SPECS
            },
        }
        for group_label, rows in groups.items()
    }


def build_conditioned_group_comparison(
    *,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    left_group_label: str,
    right_group_label: str,
) -> dict[str, Any]:
    comparison = origin_module.build_group_field_comparison(
        left_rows=left_rows,
        right_rows=right_rows,
        left_group_label=left_group_label,
        right_group_label=right_group_label,
        field_specs=_FIELD_SPECS,
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    field_rows = _safe_list(comparison.get("field_comparisons"))
    material_fields = [
        str(_safe_dict(row).get("field") or "")
        for row in field_rows
        if _field_row_has_material_left_severity(_safe_dict(row))
    ]
    primary_material_fields = [
        field for field in material_fields if field in _PRIMARY_SEVERITY_FIELDS
    ]
    return {
        **comparison,
        "material_severity_gap_threshold": _MATERIAL_SEVERITY_GAP,
        "material_left_severity_fields": material_fields,
        "primary_material_left_severity_fields": primary_material_fields,
        "signature_depth_material_field_count": sum(
            1 for field in _SIGNATURE_DEPTH_FIELDS if field in material_fields
        ),
        "primary_material_field_count": len(primary_material_fields),
        "signature_presence_status": "held_constant_by_design",
    }


def build_reference_selection(
    *,
    primary_comparison: dict[str, Any],
    fallback_comparison: dict[str, Any],
) -> dict[str, Any]:
    primary_status = str(primary_comparison.get("support_status") or "")
    fallback_status = str(fallback_comparison.get("support_status") or "")
    if primary_status == "supported":
        selected_comparison = primary_comparison
        selected_reference_group_label = _SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL
        selected_reference_role = "preserved_outside_conditioned_primary"
        fallback_used = False
        conservative_reference_note = (
            "Primary preserved-outside conditioned reference is supported."
        )
    elif fallback_status == "supported":
        selected_comparison = fallback_comparison
        selected_reference_group_label = _SIGNATURE_CONDITIONED_OUTSIDE_GROUP_LABEL
        selected_reference_role = "all_outside_conditioned_fallback"
        fallback_used = True
        conservative_reference_note = (
            "Primary preserved-outside conditioned reference is absent or below "
            "support; all outside-pocket conditioned rows are used only as a "
            "conservative fallback."
        )
    else:
        selected_comparison = (
            primary_comparison
            if int(primary_comparison.get("right_row_count", 0) or 0) > 0
            else fallback_comparison
        )
        selected_reference_group_label = str(
            selected_comparison.get("right_group_label") or ""
        )
        selected_reference_role = "conditioned_reference_insufficient"
        fallback_used = selected_comparison is fallback_comparison
        conservative_reference_note = (
            "No conditioned reference group clears support, so severity-depth "
            "claims are withheld."
        )

    return {
        "primary_reference_group_label": (
            _SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL
        ),
        "fallback_reference_group_label": _SIGNATURE_CONDITIONED_OUTSIDE_GROUP_LABEL,
        "primary_support_status": primary_status,
        "fallback_support_status": fallback_status,
        "selected_reference_group_label": selected_reference_group_label,
        "selected_reference_role": selected_reference_role,
        "fallback_used": fallback_used,
        "conservative_reference_note": conservative_reference_note,
        "selected_comparison": selected_comparison,
    }


def build_setup_secondary_reading(
    *,
    selected_comparison: dict[str, Any],
) -> dict[str, Any]:
    setup_row = _comparison_row_for_field(
        field_comparison=selected_comparison,
        field="setup_layer_confidence",
    )
    context_row = _comparison_row_for_field(
        field_comparison=selected_comparison,
        field=_CONTEXT_BIAS_FAMILY_FIELD,
    )
    selected_strategy_row = _comparison_row_for_field(
        field_comparison=selected_comparison,
        field="selected_strategy_confidence",
    )
    setup_gap = _to_float(setup_row.get("left_group_severity_gap"), default=None)
    signature_gaps = [
        _to_float(context_row.get("left_group_severity_gap"), default=0.0),
        _to_float(selected_strategy_row.get("left_group_severity_gap"), default=0.0),
    ]
    strongest_signature_gap = max(signature_gaps, default=0.0)

    if str(selected_comparison.get("support_status") or "") != "supported":
        status = "insufficient_data"
        reading = (
            "Setup separation is not interpreted because the selected conditioned "
            "reference does not clear support."
        )
    elif not _field_row_has_material_left_severity(setup_row):
        status = "setup_stays_secondary_or_flat"
        reading = (
            "Once the fixed signature is held constant, setup does not add a "
            "material pocket-vs-reference severity split."
        )
    elif setup_gap is not None and setup_gap >= max(
        _MATERIAL_SEVERITY_GAP,
        strongest_signature_gap * 0.75,
    ):
        status = "setup_competes_with_signature_depth"
        reading = (
            "Setup adds material separation large enough to compete with the fixed "
            "signature depth fields."
        )
    else:
        status = "setup_adds_secondary_separation"
        reading = (
            "Setup moves in the pocket-weaker direction, but remains secondary to "
            "the fixed signature depth fields."
        )

    return {
        "setup_separation_status": status,
        "setup_field_row": setup_row,
        "setup_severity_gap": setup_gap,
        "strongest_signature_depth_gap": strongest_signature_gap,
        "material_severity_gap_threshold": _MATERIAL_SEVERITY_GAP,
        "reading": reading,
    }


def build_trigger_negative_control_reading(
    *,
    contributor_family_comparison: dict[str, Any],
) -> dict[str, Any]:
    trigger_row = _safe_dict(
        next(
            (
                row
                for row in _safe_list(
                    contributor_family_comparison.get("family_summaries")
                )
                if str(_safe_dict(row).get("family") or "")
                == "trigger_negative_control"
            ),
            {},
        )
    )
    status = str(
        contributor_family_comparison.get("trigger_negative_control_status")
        or "trigger_negative_control_missing"
    )
    if str(contributor_family_comparison.get("support_status") or "") != "supported":
        reading = (
            "Trigger is reported but not interpreted because the selected conditioned "
            "reference does not clear support."
        )
    elif status == "trigger_remains_negative_control":
        reading = (
            "Trigger remains a negative control inside the conditioned slice."
        )
    elif status == "trigger_moves_but_remains_secondary":
        reading = (
            "Trigger moves in the pocket-weaker direction, but remains secondary to "
            "non-trigger fields."
        )
    else:
        reading = (
            "Trigger no longer behaves as a clean negative control inside this "
            "conditioned comparison."
        )
    return {
        "trigger_negative_control_status": status,
        "trigger_field_row": trigger_row,
        "reading": reading,
    }


def build_severity_depth_assessment(
    *,
    signature_conditioning_summary: dict[str, Any],
    reference_selection: dict[str, Any],
    selected_comparison: dict[str, Any],
    setup_secondary_reading: dict[str, Any],
    trigger_negative_control_reading: dict[str, Any],
) -> dict[str, Any]:
    support_status = str(selected_comparison.get("support_status") or "")
    material_fields = _safe_list(
        selected_comparison.get("material_left_severity_fields")
    )
    primary_material_fields = _safe_list(
        selected_comparison.get("primary_material_left_severity_fields")
    )
    signature_depth_material_field_count = int(
        selected_comparison.get("signature_depth_material_field_count", 0) or 0
    )
    residual_depth_supported = _RESIDUAL_FIELD in material_fields
    selected_reference_role = str(reference_selection.get("selected_reference_role"))
    fallback_used = bool(reference_selection.get("fallback_used", False))

    if support_status != "supported":
        severity_depth_status = "conditioned_reference_insufficient"
        reading = (
            "The fixed dominant signature can be isolated, but the conditioned "
            "reference group is absent or too small, so severity-depth diagnosis "
            "stays conservative."
        )
    elif (
        signature_depth_material_field_count == len(_SIGNATURE_DEPTH_FIELDS)
        and residual_depth_supported
        and not fallback_used
    ):
        severity_depth_status = "severity_conditioned_split_supported"
        reading = (
            "With the dominant signature held constant, pocket rows are materially "
            "deeper on both fixed signature fields and on residual depth versus the "
            "weighted setup-emphasis baseline."
        )
    elif (
        signature_depth_material_field_count == len(_SIGNATURE_DEPTH_FIELDS)
        and residual_depth_supported
        and fallback_used
    ):
        severity_depth_status = "severity_conditioned_split_fallback_leaning"
        reading = (
            "The all-outside conditioned fallback points toward severity depth, but "
            "the preserved-outside conditioned reference is not supported, so the "
            "report does not promote a full support claim."
        )
    elif len(primary_material_fields) >= 2 and not fallback_used:
        severity_depth_status = "severity_conditioned_split_leaning"
        reading = (
            "The fixed-signature comparison leans toward severity depth, but not all "
            "primary depth checks clear the material-gap threshold."
        )
    elif len(primary_material_fields) >= 2 and fallback_used:
        severity_depth_status = "severity_conditioned_split_fallback_leaning"
        reading = (
            "Fallback comparison leans toward severity depth, but primary conditioned "
            "preserved-outside support is missing or too small."
        )
    else:
        severity_depth_status = "severity_conditioned_split_inconclusive"
        reading = (
            "Once the dominant signature is held constant, the pocket-vs-reference "
            "severity depths are too similar or too mixed to support a severity "
            "split."
        )

    setup_status = str(
        setup_secondary_reading.get("setup_separation_status") or ""
    )
    trigger_status = str(
        trigger_negative_control_reading.get("trigger_negative_control_status") or ""
    )
    setup_stays_secondary = setup_status in {
        "setup_stays_secondary_or_flat",
        "setup_adds_secondary_separation",
        "insufficient_data",
    }
    trigger_remains_negative_control = trigger_status in {
        "trigger_remains_negative_control",
        "trigger_moves_but_remains_secondary",
    }

    return {
        "selected_reference_group_label": reference_selection.get(
            "selected_reference_group_label"
        ),
        "selected_reference_role": selected_reference_role,
        "fallback_used": fallback_used,
        "support_status": support_status,
        "signature_presence_status": signature_conditioning_summary.get(
            "signature_presence_status"
        ),
        "material_severity_gap_threshold": _MATERIAL_SEVERITY_GAP,
        "material_left_severity_fields": material_fields,
        "primary_material_left_severity_fields": primary_material_fields,
        "signature_depth_material_field_count": signature_depth_material_field_count,
        "residual_depth_supported": residual_depth_supported,
        "setup_stays_secondary": setup_stays_secondary,
        "trigger_remains_negative_control": trigger_remains_negative_control,
        "severity_depth_status": severity_depth_status,
        "reading": reading,
        "edge_selection_absence_reading": _edge_selection_absence_reading(
            severity_depth_status=severity_depth_status,
            fallback_used=fallback_used,
        ),
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    signature_conditioning_summary: dict[str, Any],
    reference_selection: dict[str, Any],
    severity_depth_assessment: dict[str, Any],
    setup_secondary_reading: dict[str, Any],
    trigger_negative_control_reading: dict[str, Any],
) -> dict[str, Any]:
    facts = [
        (
            "Final rule-bias-aligned slice support: "
            f"{summary.get('comparison_support_status', 'unknown')} "
            f"(preserved={summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            f"collapsed={summary.get('collapsed_final_hold_outcome_row_count', 0)})."
        ),
        (
            "Fixed dominant signature: "
            f"{_DOMINANT_SIGNATURE_LABEL}; low_surface_count="
            f"{_LOW_SURFACE_COUNT_ANCHOR}; low_threshold="
            f"{_LOW_CONFIDENCE_THRESHOLD}."
        ),
        (
            "Signature-conditioned counts: all="
            f"{signature_conditioning_summary.get('signature_conditioned_row_count', 0)}, "
            "pocket="
            f"{signature_conditioning_summary.get('signature_conditioned_pocket_row_count', 0)}, "
            "preserved_outside="
            f"{signature_conditioning_summary.get('signature_conditioned_preserved_outside_pocket_row_count', 0)}, "
            "all_outside="
            f"{signature_conditioning_summary.get('signature_conditioned_outside_pocket_row_count', 0)}."
        ),
        (
            "Selected reference role="
            f"{reference_selection.get('selected_reference_role')}; "
            "primary_support="
            f"{reference_selection.get('primary_support_status')}; "
            "fallback_support="
            f"{reference_selection.get('fallback_support_status')}."
        ),
        (
            "Severity depth status="
            f"{severity_depth_assessment.get('severity_depth_status')}; "
            "material_fields="
            f"{severity_depth_assessment.get('material_left_severity_fields', [])}."
        ),
        (
            "Setup status="
            f"{setup_secondary_reading.get('setup_separation_status')}; "
            "trigger status="
            f"{trigger_negative_control_reading.get('trigger_negative_control_status')}."
        ),
    ]

    if summary.get("comparison_support_status") != "supported":
        interpretation_status = "comparison_unsupported"
        strongest_pattern = [
            "The final preserved-vs-collapsed slice does not clear the family's normal support threshold, so signature-conditioned severity claims stay withheld."
        ]
    else:
        severity_status = str(
            severity_depth_assessment.get("severity_depth_status") or ""
        )
        if severity_status == "severity_conditioned_split_supported":
            interpretation_status = "signature_conditioned_severity_depth_supported"
            strongest_pattern = [
                "The missing edge-selection outcome is best explained as deeper compression inside the already-shared weak signature, not as the mere presence of that signature."
            ]
        elif severity_status == "severity_conditioned_split_fallback_leaning":
            interpretation_status = (
                "signature_conditioned_severity_depth_fallback_leaning"
            )
            strongest_pattern = [
                "Severity depth is visible in the conditioned fallback comparison, but preserved-outside conditioned support is absent or too small, so the claim stays conservative."
            ]
        elif severity_status == "severity_conditioned_split_leaning":
            interpretation_status = "signature_conditioned_severity_depth_leaning"
            strongest_pattern = [
                "The conditioned comparison leans toward continuous severity depth, but not every primary depth check clears the material threshold."
            ]
        elif severity_status == "conditioned_reference_insufficient":
            interpretation_status = "conditioned_reference_insufficient"
            strongest_pattern = [
                "The fixed signature exists, but the conditioned reference group is absent or too small to confirm a severity-depth split."
            ]
        else:
            interpretation_status = "signature_conditioned_severity_depth_inconclusive"
            strongest_pattern = [
                "Holding the fixed signature constant does not isolate enough continuous depth difference to explain pocket membership on this run."
            ]

    uncertainty = [
        "This report is descriptive and does not infer or recommend production edge-selection recovery logic.",
        "The pocket threshold and fixed low-confidence signature are diagnosis anchors only.",
        "Because signature presence is held constant by design, the report can only speak to depth inside that signature, not to a new signature class.",
    ]
    if bool(reference_selection.get("fallback_used", False)):
        uncertainty.append(
            "The selected reference is an all-outside conditioned fallback, so the final reading remains weaker than a direct preserved-outside conditioned comparison."
        )

    return {
        "interpretation_status": interpretation_status,
        "facts": [item for item in facts if item],
        "strongest_candidate_explanatory_pattern": strongest_pattern,
        "inference": strongest_pattern,
        "uncertainty": uncertainty,
    }


def build_limitations(
    *,
    summary: dict[str, Any],
    signature_conditioning_summary: dict[str, Any],
    reference_selection: dict[str, Any],
    severity_depth_assessment: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact stays inside the already-confirmed final slice and does not reopen earlier bottlenecks, mapper changes, engine changes, or execution-gate changes.",
        "The dominant exact low-confidence signature is fixed before comparison, so this is not a broad signature search.",
        "The compressed pocket is fixed to actual rule_engine_confidence <= 0.25, so this is not a threshold search.",
    ]
    if summary.get("comparison_support_status") != "supported":
        limitations.append(
            "The final preserved-vs-collapsed comparison is below the family's normal supported threshold, so conditioned severity claims remain withheld."
        )
    if (
        signature_conditioning_summary.get("preserved_outside_reference_support_status")
        != "supported"
    ):
        limitations.append(
            "The conditioned preserved-outside reference is absent or below support, so fallback readings remain conservative."
        )
    if (
        severity_depth_assessment.get("severity_depth_status")
        != "severity_conditioned_split_supported"
    ):
        limitations.append(
            "The report does not fully confirm severity-conditioned separation on this run."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    severity_depth_assessment = _safe_dict(widest.get("severity_depth_assessment"))
    signature_conditioning_summary = _safe_dict(
        widest.get("signature_conditioning_summary")
    )
    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "dominant_signature_definition": _dominant_signature_definition(),
        "signature_conditioning_summary": signature_conditioning_summary,
        "severity_depth_status": severity_depth_assessment.get(
            "severity_depth_status"
        ),
        "selected_reference_group_label": severity_depth_assessment.get(
            "selected_reference_group_label"
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            interpretation=interpretation,
            severity_depth_assessment=severity_depth_assessment,
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
            "- signature_conditioned_row_count: "
            f"{headline.get('signature_conditioned_row_count', 0)}"
        )
        lines.append(
            "- signature_conditioned_pocket_row_count: "
            f"{headline.get('signature_conditioned_pocket_row_count', 0)}"
        )
        lines.append(
            "- signature_conditioned_preserved_outside_pocket_row_count: "
            f"{headline.get('signature_conditioned_preserved_outside_pocket_row_count', 0)}"
        )
        lines.append(
            "- selected_reference_group_label: "
            f"{headline.get('selected_reference_group_label', 'unknown')}"
        )
        lines.append(
            "- severity_depth_status: "
            f"{headline.get('severity_depth_status', 'unknown')}"
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
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        "- severity_depth_status: "
        f"{final_assessment.get('severity_depth_status', 'unknown')}"
    )
    lines.append(
        "- selected_reference_group_label: "
        f"{final_assessment.get('selected_reference_group_label', 'unknown')}"
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


def _dominant_signature_definition() -> dict[str, Any]:
    return {
        "exact_low_confidence_fields": list(
            _DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
        ),
        "signature_label": _DOMINANT_SIGNATURE_LABEL,
        "low_surface_count": _LOW_SURFACE_COUNT_ANCHOR,
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "status": "fixed_diagnosis_anchor",
    }


def _reference_rows_for_selection(
    *,
    row_sets: dict[str, Any],
    reference_selection: dict[str, Any],
) -> list[dict[str, Any]]:
    if (
        reference_selection.get("selected_reference_group_label")
        == _SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL
    ):
        return _safe_list(
            row_sets.get("signature_conditioned_preserved_outside_pocket_rows")
        )
    return _safe_list(row_sets.get("signature_conditioned_outside_pocket_rows"))


def _field_row_has_material_left_severity(row: dict[str, Any]) -> bool:
    return bool(
        row.get("left_group_more_severe_by_orientation") is True
        and (
            _to_float(row.get("left_group_severity_gap"), default=0.0)
            >= _MATERIAL_SEVERITY_GAP
        )
    )


def _comparison_row_for_field(
    *,
    field_comparison: Any,
    field: str,
) -> dict[str, Any]:
    return origin_module._comparison_row_for_field(
        field_comparison=field_comparison,
        field=field,
    )


def _edge_selection_absence_reading(
    *,
    severity_depth_status: str,
    fallback_used: bool,
) -> str:
    if severity_depth_status == "severity_conditioned_split_supported":
        return (
            "supports_deeper_compression_within_already_shared_weak_signature"
        )
    if severity_depth_status == "severity_conditioned_split_fallback_leaning":
        return (
            "fallback_leans_deeper_compression_but_preserved_reference_missing"
        )
    if severity_depth_status in {
        "severity_conditioned_split_leaning",
    }:
        return "leans_deeper_compression_within_shared_signature"
    if severity_depth_status == "conditioned_reference_insufficient":
        return "insufficient_conditioned_reference"
    if fallback_used:
        return "fallback_inconclusive"
    return "inconclusive"


def _has_fixed_dominant_signature(row: dict[str, Any]) -> bool:
    return (
        tuple(row.get("exact_low_confidence_fields") or ())
        == _DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
    )


def _group_support_status(
    *,
    left_row_count: int,
    right_row_count: int,
    min_support_rows: int,
) -> str:
    return origin_module._group_support_status(
        left_row_count=left_row_count,
        right_row_count=right_row_count,
        min_support_rows=min_support_rows,
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    interpretation: dict[str, Any],
    severity_depth_assessment: dict[str, Any],
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
    severity_depth_status = str(
        severity_depth_assessment.get("severity_depth_status") or ""
    )

    if final_slice_row_count <= 0:
        return (
            "No rows reached the final fully aligned plus rule-bias-aligned slice, "
            "so signature-conditioned severity diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one "
            "side of the preserved-vs-collapsed comparison is missing."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but conditioned severity claims stay withheld "
            "because the preserved-vs-collapsed comparison does not clear support."
        )
    if severity_depth_status == "severity_conditioned_split_supported":
        return (
            "The widest configuration supports the final confirmation reading: "
            "pocket rows differ by deeper severity inside the same fixed weak "
            "signature, not by signature presence alone."
        )
    if severity_depth_status == "severity_conditioned_split_fallback_leaning":
        return (
            "The widest configuration leans toward deeper severity inside the fixed "
            "signature, but only through the all-outside conditioned fallback."
        )
    if severity_depth_status == "conditioned_reference_insufficient":
        return (
            "The fixed signature is present, but the conditioned reference is absent "
            "or too small for final confirmation."
        )
    return (
        "The widest configuration does not confirm a severity-depth split once the "
        "dominant weak signature is held constant."
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


def _comparison_group_rows(
    rows: Sequence[dict[str, Any]],
    comparison_group: str,
) -> list[dict[str, Any]]:
    return residual_module._comparison_group_rows(rows, comparison_group)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return origin_module._safe_ratio(numerator, denominator)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return residual_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return residual_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return residual_module._safe_list(value)


if __name__ == "__main__":
    main()
