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
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report as threshold_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_"
    "context_band_conditional_setup_diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Weighted Aggregate Residual "
    "Context Band Conditional Setup Diagnosis Report"
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

_CONTEXT_MEAN_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<= 0.45", None, 0.45),
    ("(0.45, 0.50]", 0.45, 0.50),
    ("(0.50, 0.55]", 0.50, 0.55),
    ("(0.55, 0.60]", 0.55, 0.60),
    ("(0.60, 0.65]", 0.60, 0.65),
    ("> 0.65", 0.65, None),
)
_CONTEXT_SHORTFALL_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("<= 0.35", None, 0.35),
    ("(0.35, 0.40]", 0.35, 0.40),
    ("(0.40, 0.45]", 0.40, 0.45),
    ("(0.45, 0.50]", 0.45, 0.50),
    ("(0.50, 0.55]", 0.50, 0.55),
    ("> 0.55", 0.55, None),
)
_CONTEXT_AXIS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "field": _CONTEXT_BIAS_FAMILY_FIELD,
        "field_label": "Context/bias family mean",
        "transform": "raw_confidence",
        "threshold_operator": "<=",
        "reference_threshold": 0.55,
        "reference_threshold_label": (
            "context_bias_family_mean <= 0.55"
        ),
        "band_definitions": _CONTEXT_MEAN_BANDS,
        "extreme_band_side": "low",
    },
    {
        "field": "context_bias_family_shortfall",
        "field_label": "Context/bias family shortfall",
        "transform": "shortfall",
        "threshold_operator": ">=",
        "reference_threshold": 0.45,
        "reference_threshold_label": (
            "context_bias_family_shortfall >= 0.45"
        ),
        "band_definitions": _CONTEXT_SHORTFALL_BANDS,
        "extreme_band_side": "high",
    },
)
_SETUP_AXIS_SPECS: tuple[dict[str, Any], ...] = (
    {
        "field": "setup_layer_confidence",
        "field_label": "Setup layer confidence",
        "transform": "raw_confidence",
        "threshold_operator": "<=",
        "threshold_grid": (
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
        ),
    },
    {
        "field": "setup_shortfall",
        "field_label": "Setup shortfall",
        "transform": "shortfall",
        "threshold_operator": ">=",
        "threshold_grid": (
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
        ),
    },
)
_SETUP_AXIS_ORDER = {
    str(spec["field"]): index for index, spec in enumerate(_SETUP_AXIS_SPECS)
}
_CONDITIONAL_STATUS_ORDER = {
    "sharpens_collapsed_severity_without_replacing_gate": 0,
    "sharpens_collapsed_severity_but_competes_as_gate": 1,
    "unclear": 2,
    "insufficient_data": 3,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only context-band plus conditional-setup report for "
            "the final fully aligned, rule-bias-aligned preserved-vs-collapsed "
            "slice using the confirmed weighted_mean_setup_emphasis residual "
            "baseline and a fixed sharply negative residual pocket definition."
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
            "Retained for architectural parity with sibling reports. This context-"
            "band diagnosis itself reuses only the final preserved-vs-collapsed "
            "slice."
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
        run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_context_band_conditional_setup_diagnosis_report(
            input_path=input_path,
            output_dir=output_dir,
            configurations=configurations,
            min_symbol_support=args.min_symbol_support,
            write_report_copies=args.write_latest_copy,
        )
    )
    report = result["report"]
    best_context_band = _safe_dict(report.get("best_context_band"))
    best_conditional = _safe_dict(report.get("best_conditional_setup_profile"))

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
                "preserved_final_directional_outcome_row_count": report.get(
                    "preserved_final_directional_outcome_row_count",
                    0,
                ),
                "collapsed_final_hold_outcome_row_count": report.get(
                    "collapsed_final_hold_outcome_row_count",
                    0,
                ),
                "baseline_name": report.get("baseline_name"),
                "residual_pocket_definition": report.get("residual_pocket_definition"),
                "best_context_band": {
                    "field": best_context_band.get("field"),
                    "band_label": best_context_band.get("band_label"),
                    "profile_shape": best_context_band.get("profile_shape"),
                },
                "best_conditional_setup_profile": {
                    "field": best_conditional.get("field"),
                    "threshold_label": _safe_dict(
                        best_conditional.get("best_conditional_threshold_profile")
                    ).get("threshold_label"),
                    "conditional_severity_status": best_conditional.get(
                        "conditional_severity_status"
                    ),
                },
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_weighted_aggregate_residual_context_band_conditional_setup_diagnosis_report(
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
            _safe_dict(summary_row.get("headline"))
            for summary_row in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": summary,
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
        "context_band_profiles": _safe_list(
            widest_summary.get("context_band_profiles")
        ),
        "best_context_band": _safe_dict(widest_summary.get("best_context_band")),
        "context_band_interpretation": _safe_dict(
            widest_summary.get("context_band_interpretation")
        ),
        "conditional_setup_profiles": _safe_dict(
            widest_summary.get("conditional_setup_profiles")
        ),
        "best_conditional_setup_profile": _safe_dict(
            widest_summary.get("best_conditional_setup_profile")
        ),
        "context_band_vs_single_cutoff_comparison": _safe_dict(
            widest_summary.get("context_band_vs_single_cutoff_comparison")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis family.",
            "Residual remains fixed to rule_engine_confidence minus weighted_mean_setup_emphasis so the diagnosis continues from the already-confirmed weighted aggregate residual baseline.",
            "The sharply negative residual pocket stays fixed at residual <= -0.15 for continuity with the previous diagnosis stage.",
            "Context-band grids and setup-side threshold grids are fixed transparent helpers rather than optimized cutoffs.",
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
    context_band_profiles = build_context_band_profiles(comparison_rows=residual_rows)
    best_context_band = build_best_context_band(
        context_band_profiles=context_band_profiles
    )
    context_band_vs_single_cutoff_comparison = (
        build_context_band_vs_single_cutoff_comparison(
            best_context_band=best_context_band,
            context_band_profiles=context_band_profiles,
        )
    )
    context_band_interpretation = build_context_band_interpretation(
        summary=summary,
        context_band_profiles=context_band_profiles,
        best_context_band=best_context_band,
        context_band_vs_single_cutoff_comparison=(
            context_band_vs_single_cutoff_comparison
        ),
    )
    conditional_setup_profiles = build_conditional_setup_profiles(
        comparison_rows=residual_rows,
        best_context_band=best_context_band,
    )
    best_conditional_setup_profile = _safe_dict(
        conditional_setup_profiles.get("best_conditional_setup_profile")
    )
    interpretation = build_interpretation(
        summary=summary,
        baseline_reference=baseline_reference,
        residual_class_comparison=residual_class_comparison,
        residual_sign_distribution=residual_sign_distribution,
        context_band_profiles=context_band_profiles,
        best_context_band=best_context_band,
        context_band_interpretation=context_band_interpretation,
        conditional_setup_profiles=conditional_setup_profiles,
        best_conditional_setup_profile=best_conditional_setup_profile,
        context_band_vs_single_cutoff_comparison=(
            context_band_vs_single_cutoff_comparison
        ),
    )
    limitations = build_limitations(
        summary=summary,
        context_band_interpretation=context_band_interpretation,
        conditional_setup_profiles=conditional_setup_profiles,
    )

    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "effective_input_path": str(effective_input_path),
        "run_output_dir": str(run_output_dir),
        "source_metadata": source_metadata,
        "headline": {
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
            "best_context_band": (
                f"{best_context_band.get('field')} @ "
                f"{best_context_band.get('band_label')}"
                if best_context_band
                else "none"
            ),
            "best_conditional_setup_axis": best_conditional_setup_profile.get("field"),
            "interpretation_status": interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "baseline_reference": baseline_reference,
        "actual_rule_engine_confidence_reference": (
            actual_rule_engine_confidence_reference
        ),
        "residual_class_comparison": residual_class_comparison,
        "residual_sign_distribution": residual_sign_distribution,
        "context_band_profiles": context_band_profiles,
        "best_context_band": best_context_band,
        "context_band_interpretation": context_band_interpretation,
        "conditional_setup_profiles": conditional_setup_profiles,
        "best_conditional_setup_profile": best_conditional_setup_profile,
        "context_band_vs_single_cutoff_comparison": (
            context_band_vs_single_cutoff_comparison
        ),
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_context_band_profiles(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_spec in _CONTEXT_AXIS_SPECS:
        field = str(axis_spec["field"])
        band_profile = threshold_module.build_axis_band_profile(
            comparison_rows=comparison_rows,
            field=field,
            field_label=str(axis_spec["field_label"]),
            transform=str(axis_spec["transform"]),
            threshold_operator=str(axis_spec["threshold_operator"]),
            band_definitions=tuple(axis_spec["band_definitions"]),
            extreme_band_side=str(axis_spec["extreme_band_side"]),
        )
        reference_profile = threshold_module.build_axis_threshold_profile(
            comparison_rows=comparison_rows,
            field=field,
            field_label=str(axis_spec["field_label"]),
            transform=str(axis_spec["transform"]),
            threshold_operator=str(axis_spec["threshold_operator"]),
            threshold_grid=(float(axis_spec["reference_threshold"]),),
        )
        rows.append(
            {
                **band_profile,
                "reference_threshold_profile": _safe_dict(
                    reference_profile.get("best_threshold_profile")
                ),
                "reference_threshold_label": axis_spec.get(
                    "reference_threshold_label"
                ),
            }
        )
    return rows


def build_best_context_band(
    *,
    context_band_profiles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mean_profile = _context_band_profile_for_field(
        context_band_profiles=context_band_profiles,
        field=_CONTEXT_BIAS_FAMILY_FIELD,
    )
    strongest_band = _safe_dict(mean_profile.get("strongest_band"))
    if not mean_profile or not strongest_band:
        return {}
    return {
        "field": mean_profile.get("field"),
        "field_label": mean_profile.get("field_label"),
        "transform": mean_profile.get("transform"),
        "profile_shape": mean_profile.get("profile_shape"),
        "shape_reason": mean_profile.get("shape_reason"),
        "reference_threshold_profile": _safe_dict(
            mean_profile.get("reference_threshold_profile")
        ),
        **strongest_band,
    }


def build_context_band_vs_single_cutoff_comparison(
    *,
    best_context_band: dict[str, Any],
    context_band_profiles: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reference_profile = _safe_dict(
        _context_band_profile_for_field(
            context_band_profiles=context_band_profiles,
            field=_CONTEXT_BIAS_FAMILY_FIELD,
        ).get("reference_threshold_profile")
    )
    if not best_context_band or not reference_profile:
        return {
            "support_status": "insufficient_data",
            "comparison_reading": "comparison_unavailable",
            "interpretation": (
                "Band-versus-single-cutoff comparison is unavailable because the "
                "reference context cutoff or strongest context band is missing."
            ),
        }

    leakage_improvement = _difference_or_none(
        reference_profile.get("preserved_leakage_rate"),
        best_context_band.get("preserved_leakage_rate"),
    )
    collapsed_capture_change = _difference_or_none(
        best_context_band.get("collapsed_capture_rate"),
        reference_profile.get("collapsed_capture_rate"),
    )
    collapsed_pocket_capture_change = _difference_or_none(
        best_context_band.get("collapsed_pocket_capture_rate"),
        reference_profile.get("collapsed_pocket_capture_rate"),
    )
    pocket_gap_change = _difference_or_none(
        best_context_band.get("inside_pocket_concentration_gap"),
        reference_profile.get("inside_pocket_concentration_gap"),
    )
    capture_minus_leakage_change = _difference_or_none(
        best_context_band.get("capture_minus_leakage"),
        reference_profile.get("capture_minus_leakage"),
    )

    materially_reduces_preserved_leakage = bool(
        leakage_improvement is not None and leakage_improvement >= 0.10
    )
    keeps_most_collapsed_pocket_membership = bool(
        _to_float(
            best_context_band.get("collapsed_pocket_capture_rate"),
            default=0.0,
        )
        >= max(
            0.60,
            _to_float(
                reference_profile.get("collapsed_pocket_capture_rate"),
                default=0.0,
            )
            - 0.15,
        )
    )
    materially_improves_capture_minus_leakage = bool(
        capture_minus_leakage_change is not None and capture_minus_leakage_change >= 0.05
    )

    comparison_reading = "mixed_or_inconclusive"
    interpretation = (
        "The strongest context band can be compared with the carried-forward "
        "context_bias_family_mean <= 0.55 cutoff, but the band does not clear this "
        "report's conservative criteria for a clean upgrade."
    )
    if (
        str(best_context_band.get("profile_shape") or "") == "narrow_band"
        and materially_reduces_preserved_leakage
        and keeps_most_collapsed_pocket_membership
        and _to_float(capture_minus_leakage_change, default=0.0) >= 0.0
        and _to_float(pocket_gap_change, default=0.0) >= -0.05
    ):
        comparison_reading = "narrow_band_better_than_single_cutoff"
        interpretation = (
            "The strongest context band reduces preserved leakage materially relative "
            "to the coarse <= 0.55 cutoff while still keeping most collapsed pocket "
            "membership, so a narrow neighborhood explains the residual pocket "
            "better than the plain single cutoff on this run."
        )
    elif str(best_context_band.get("profile_shape") or "") in {
        "simple_cutoff",
        "broad_helper_regime",
    }:
        comparison_reading = "plain_single_cutoff_or_broad_helper_regime_more_plausible"
        interpretation = (
            "The ordered context bands behave more like an extreme-side cutoff or a "
            "contiguous broad helper regime than an interior neighborhood, so the "
            "band idea does not outrun the plain cutoff reading here."
        )

    return {
        "support_status": best_context_band.get(
            "support_status",
            "insufficient_data",
        ),
        "reference_single_cutoff_profile": reference_profile,
        "best_context_band": best_context_band,
        "preserved_leakage_improvement_vs_single_cutoff": leakage_improvement,
        "collapsed_capture_change_vs_single_cutoff": collapsed_capture_change,
        "collapsed_pocket_capture_change_vs_single_cutoff": (
            collapsed_pocket_capture_change
        ),
        "pocket_concentration_gap_change_vs_single_cutoff": pocket_gap_change,
        "capture_minus_leakage_change_vs_single_cutoff": (
            capture_minus_leakage_change
        ),
        "materially_reduces_preserved_leakage": (
            materially_reduces_preserved_leakage
        ),
        "keeps_most_collapsed_pocket_membership": (
            keeps_most_collapsed_pocket_membership
        ),
        "materially_improves_capture_minus_leakage": (
            materially_improves_capture_minus_leakage
        ),
        "comparison_reading": comparison_reading,
        "interpretation": interpretation,
    }


def build_context_band_interpretation(
    *,
    summary: dict[str, Any],
    context_band_profiles: Sequence[dict[str, Any]],
    best_context_band: dict[str, Any],
    context_band_vs_single_cutoff_comparison: dict[str, Any],
) -> dict[str, Any]:
    if summary.get("comparison_support_status") != "supported":
        return {
            "support_status": summary.get(
                "comparison_support_status",
                "insufficient_data",
            ),
            "best_context_band_reading": "comparison_unsupported",
            "interpretation": (
                "Context-band interpretation is withheld because the final "
                "preserved-vs-collapsed comparison does not clear the family's "
                "supported threshold."
            ),
        }

    mean_profile = _context_band_profile_for_field(
        context_band_profiles=context_band_profiles,
        field=_CONTEXT_BIAS_FAMILY_FIELD,
    )
    shortfall_profile = _context_band_profile_for_field(
        context_band_profiles=context_band_profiles,
        field="context_bias_family_shortfall",
    )
    comparison_reading = str(
        context_band_vs_single_cutoff_comparison.get("comparison_reading") or ""
    )
    band_reading = "mixed_or_weak"
    if comparison_reading == "narrow_band_better_than_single_cutoff":
        band_reading = "narrow_band"
    elif str(best_context_band.get("profile_shape") or "") in {
        "simple_cutoff",
        "broad_helper_regime",
    }:
        band_reading = str(best_context_band.get("profile_shape"))

    interpretation = (
        "Context profiling stays mixed: context_bias_family_mean remains the lead "
        "descriptive axis, but the ordered bands do not yet clear a narrow-band "
        "upgrade over the coarse threshold reference."
    )
    if band_reading == "narrow_band":
        interpretation = (
            "Context profiling is most consistent with a narrow neighborhood around "
            "the carried-forward 0.55 region rather than a single one-sided cutoff."
        )
    elif band_reading in {"simple_cutoff", "broad_helper_regime"}:
        interpretation = (
            "Context profiling is better read as an extreme-side cutoff or broad low-"
            "context helper regime than as a narrow interior neighborhood."
        )

    return {
        "support_status": mean_profile.get("support_status", "insufficient_data"),
        "best_context_axis": _CONTEXT_BIAS_FAMILY_FIELD,
        "best_context_band_label": best_context_band.get("band_label"),
        "best_context_band_reading": band_reading,
        "best_context_band_profile_shape": best_context_band.get("profile_shape"),
        "shortfall_transform_profile_shape": shortfall_profile.get("profile_shape"),
        "relative_to_single_cutoff": comparison_reading,
        "interpretation": interpretation,
    }


def build_conditional_setup_profiles(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    best_context_band: dict[str, Any],
) -> dict[str, Any]:
    if not best_context_band:
        return {
            "support_status": "insufficient_data",
            "interpretation_status": "conditional_setup_unavailable",
            "interpretation": (
                "Conditional setup profiling is unavailable because no context band "
                "candidate was identified."
            ),
            "setup_axis_profiles": [],
            "best_conditional_setup_profile": {},
        }

    context_band_rows = _rows_in_band(
        rows=comparison_rows,
        field=str(best_context_band.get("field") or ""),
        lower_bound_exclusive=_to_float(
            best_context_band.get("lower_bound_exclusive"),
            default=None,
        ),
        upper_bound_inclusive=_to_float(
            best_context_band.get("upper_bound_inclusive"),
            default=None,
        ),
    )
    context_band_present_rows = [
        row for row in context_band_rows if row.get(_RESIDUAL_FIELD) is not None
    ]
    context_band_preserved_rows = _comparison_group_rows(
        context_band_present_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    context_band_collapsed_rows = _comparison_group_rows(
        context_band_present_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )

    if not context_band_present_rows:
        return {
            "support_status": "insufficient_data",
            "interpretation_status": "conditional_setup_unavailable",
            "interpretation": (
                "Conditional setup profiling is unavailable because the selected "
                "context band contains no rows with a residual value."
            ),
            "context_band_row_count": 0,
            "context_band_preserved_row_count": 0,
            "context_band_collapsed_row_count": 0,
            "setup_axis_profiles": [],
            "best_conditional_setup_profile": {},
        }

    setup_axis_profiles: list[dict[str, Any]] = []
    for axis_spec in _SETUP_AXIS_SPECS:
        field = str(axis_spec["field"])
        full_slice_profile = threshold_module.build_axis_threshold_profile(
            comparison_rows=comparison_rows,
            field=field,
            field_label=str(axis_spec["field_label"]),
            transform=str(axis_spec["transform"]),
            threshold_operator=str(axis_spec["threshold_operator"]),
            threshold_grid=tuple(axis_spec["threshold_grid"]),
        )
        conditional_profile = threshold_module.build_axis_threshold_profile(
            comparison_rows=context_band_present_rows,
            field=field,
            field_label=str(axis_spec["field_label"]),
            transform=str(axis_spec["transform"]),
            threshold_operator=str(axis_spec["threshold_operator"]),
            threshold_grid=tuple(axis_spec["threshold_grid"]),
        )
        best_full_slice_threshold = _safe_dict(
            full_slice_profile.get("best_threshold_profile")
        )
        best_conditional_threshold = _safe_dict(
            conditional_profile.get("best_threshold_profile")
        )
        overall_gate_advantage = _difference_or_none(
            best_context_band.get("capture_minus_leakage"),
            best_full_slice_threshold.get("capture_minus_leakage"),
        )
        collapsed_delta = _to_float(
            best_conditional_threshold.get(
                "collapsed_inside_vs_outside_residual_median_delta"
            ),
            default=None,
        )
        collapsed_pocket_lift = _to_float(
            best_conditional_threshold.get("collapsed_pocket_lift"),
            default=None,
        )
        preserved_leakage_impact = _to_float(
            best_conditional_threshold.get("preserved_leakage_rate"),
            default=None,
        )
        severity_signal = bool(
            best_conditional_threshold
            and best_conditional_threshold.get("profile_strength_status")
            in {"strong_support", "weak_support"}
            and (
                (collapsed_delta is not None and collapsed_delta <= -0.08)
                or (collapsed_pocket_lift is not None and collapsed_pocket_lift >= 0.20)
            )
        )

        conditional_severity_status = "unclear"
        interpretation = (
            "Inside the selected context band, this setup-side axis does not yet "
            "clear the report's conservative conditional-severity threshold."
        )
        if not best_conditional_threshold:
            conditional_severity_status = "insufficient_data"
            interpretation = (
                "Inside the selected context band, this setup-side axis has no "
                "eligible threshold profile."
            )
        elif severity_signal and _to_float(overall_gate_advantage, default=0.0) >= 0.10:
            conditional_severity_status = (
                "sharpens_collapsed_severity_without_replacing_gate"
            )
            interpretation = (
                "Inside the selected context band, this setup-side axis aligns with "
                "deeper collapsed residual severity, while the context band still "
                "remains the stronger top-level gate candidate."
            )
        elif severity_signal:
            conditional_severity_status = (
                "sharpens_collapsed_severity_but_competes_as_gate"
            )
            interpretation = (
                "Inside the selected context band, this setup-side axis sharpens "
                "collapsed residual severity, but it also stays close enough as a "
                "top-level gate candidate that a clean gate-versus-severity split is "
                "not supported."
            )

        setup_axis_profiles.append(
            {
                "field": field,
                "field_label": axis_spec.get("field_label"),
                "transform": axis_spec.get("transform"),
                "context_band_label": best_context_band.get("band_label"),
                "context_band_row_count": len(context_band_present_rows),
                "context_band_preserved_row_count": len(context_band_preserved_rows),
                "context_band_collapsed_row_count": len(context_band_collapsed_rows),
                "top_level_best_threshold_profile": best_full_slice_threshold,
                "conditional_threshold_profile": conditional_profile,
                "best_conditional_threshold_profile": best_conditional_threshold,
                "overall_gate_advantage_vs_this_setup_axis": overall_gate_advantage,
                "collapsed_inside_vs_outside_residual_median_delta": collapsed_delta,
                "collapsed_pocket_lift_inside_context_band": collapsed_pocket_lift,
                "preserved_leakage_impact_inside_context_band": (
                    preserved_leakage_impact
                ),
                "conditional_severity_status": conditional_severity_status,
                "interpretation": interpretation,
            }
        )

    setup_axis_profiles.sort(key=_conditional_setup_sort_key)
    best_profile = _safe_dict(setup_axis_profiles[0] if setup_axis_profiles else {})

    interpretation_status = "conditional_setup_inconclusive"
    interpretation = (
        "Inside the selected context band, setup-side profiling remains descriptive "
        "but does not yet cleanly sharpen collapsed severity."
    )
    if (
        best_profile.get("conditional_severity_status")
        == "sharpens_collapsed_severity_without_replacing_gate"
    ):
        interpretation_status = "conditional_setup_supported"
        interpretation = (
            "Inside the selected context band, the strongest setup-side axis sharpens "
            "collapsed residual severity without overtaking context as the stronger "
            "top-level gate candidate."
        )
    elif (
        best_profile.get("conditional_severity_status")
        == "sharpens_collapsed_severity_but_competes_as_gate"
    ):
        interpretation_status = "setup_axis_competes_as_top_level_gate"
        interpretation = (
            "Inside the selected context band, setup-side severity is visible, but it "
            "stays too competitive as a top-level gate to support a clean subordinate "
            "severity-only reading."
        )
    elif best_profile.get("conditional_severity_status") == "insufficient_data":
        interpretation_status = "conditional_setup_unavailable"
        interpretation = (
            "Inside the selected context band, setup-side profiling remains too thin "
            "for a reliable conditional-severity reading."
        )

    return {
        "support_status": residual_module._regime_support_status(
            total_row_count=len(context_band_present_rows),
            collapsed_row_count=len(context_band_collapsed_rows),
        ),
        "context_band_label": best_context_band.get("band_label"),
        "context_band_row_count": len(context_band_present_rows),
        "context_band_preserved_row_count": len(context_band_preserved_rows),
        "context_band_collapsed_row_count": len(context_band_collapsed_rows),
        "setup_axis_profiles": setup_axis_profiles,
        "best_conditional_setup_profile": best_profile,
        "interpretation_status": interpretation_status,
        "interpretation": interpretation,
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    baseline_reference: dict[str, Any],
    residual_class_comparison: dict[str, Any],
    residual_sign_distribution: dict[str, Any],
    context_band_profiles: Sequence[dict[str, Any]],
    best_context_band: dict[str, Any],
    context_band_interpretation: dict[str, Any],
    conditional_setup_profiles: dict[str, Any],
    best_conditional_setup_profile: dict[str, Any],
    context_band_vs_single_cutoff_comparison: dict[str, Any],
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
    shortfall_profile = _context_band_profile_for_field(
        context_band_profiles=context_band_profiles,
        field="context_bias_family_shortfall",
    )
    single_cutoff_profile = _safe_dict(
        context_band_vs_single_cutoff_comparison.get("reference_single_cutoff_profile")
    )
    best_conditional_threshold = _safe_dict(
        best_conditional_setup_profile.get("best_conditional_threshold_profile")
    )

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
            "Baseline weighted aggregate medians: preserved="
            f"{baseline_preserved.get('median')}, "
            f"collapsed={baseline_collapsed.get('median')}."
        ),
        (
            "Residual medians after subtracting the weighted baseline: preserved="
            f"{residual_preserved.get('median')}, "
            f"collapsed={residual_collapsed.get('median')}."
        ),
        (
            "Sharply negative residual pocket uses residual <= "
            f"{_SHARPLY_NEGATIVE_RESIDUAL_THRESHOLD} and currently shows "
            f"preserved_rate={negative_pocket.get('preserved_rate')}, "
            f"collapsed_rate={negative_pocket.get('collapsed_rate')}."
        ),
    ]
    if single_cutoff_profile:
        facts.append(
            "Reference single cutoff on context_bias_family_mean <= 0.55: "
            f"collapsed_capture_rate={single_cutoff_profile.get('collapsed_capture_rate')}, "
            f"preserved_leakage_rate={single_cutoff_profile.get('preserved_leakage_rate')}, "
            "collapsed_pocket_capture_rate="
            f"{single_cutoff_profile.get('collapsed_pocket_capture_rate')}."
        )
    if best_context_band:
        facts.append(
            "Strongest context band candidate on context_bias_family_mean: "
            f"{best_context_band.get('band_label')} with "
            f"collapsed_capture_rate={best_context_band.get('collapsed_capture_rate')}, "
            f"preserved_leakage_rate={best_context_band.get('preserved_leakage_rate')}, "
            "collapsed_pocket_capture_rate="
            f"{best_context_band.get('collapsed_pocket_capture_rate')}, "
            f"profile_shape={best_context_band.get('profile_shape')}."
        )
    if shortfall_profile:
        facts.append(
            "Supporting shortfall transform profile shape: "
            f"{shortfall_profile.get('profile_shape')}."
        )
    if best_conditional_setup_profile and best_conditional_threshold:
        facts.append(
            "Inside the strongest context band, best setup-side conditional profile: "
            f"{best_conditional_setup_profile.get('field')} at "
            f"{best_conditional_threshold.get('threshold_label')} with "
            "collapsed_inside_vs_outside_residual_median_delta="
            f"{best_conditional_setup_profile.get('collapsed_inside_vs_outside_residual_median_delta')}, "
            "collapsed_pocket_lift="
            f"{best_conditional_setup_profile.get('collapsed_pocket_lift_inside_context_band')}, "
            "preserved_leakage_impact="
            f"{best_conditional_setup_profile.get('preserved_leakage_impact_inside_context_band')}."
        )

    observed_context_band_behavior: list[str] = []
    if best_context_band:
        observed_context_band_behavior.append(
            "The strongest ordered context band is "
            f"{best_context_band.get('band_label')} with band shape "
            f"{best_context_band.get('profile_shape')}."
        )
    if context_band_vs_single_cutoff_comparison:
        observed_context_band_behavior.append(
            str(context_band_vs_single_cutoff_comparison.get("interpretation") or "")
        )
    if shortfall_profile:
        observed_context_band_behavior.append(
            "The supporting shortfall transform reads as "
            f"{shortfall_profile.get('profile_shape')}."
        )

    observed_conditional_setup_behavior: list[str] = []
    if best_conditional_setup_profile:
        observed_conditional_setup_behavior.append(
            str(best_conditional_setup_profile.get("interpretation") or "")
        )
        observed_conditional_setup_behavior.append(
            "Conditional setup interpretation status="
            f"{conditional_setup_profiles.get('interpretation_status')}."
        )

    interpretation_status = "comparison_unsupported"
    strongest_pattern: list[str] = []
    unsupported_paths: list[str] = []

    if summary.get("comparison_support_status") != "supported":
        strongest_pattern.append(
            "The final preserved-vs-collapsed slice does not clear the family's "
            "supported-comparison threshold, so context-band and conditional-setup "
            "claims stay withheld."
        )
        unsupported_paths.extend(
            [
                "settled_narrow_context_band_mechanism",
                "settled_context_band_plus_conditional_setup_mechanism",
            ]
        )
    else:
        comparison_reading = str(
            context_band_vs_single_cutoff_comparison.get("comparison_reading") or ""
        )
        conditional_status = str(
            conditional_setup_profiles.get("interpretation_status") or ""
        )

        if (
            comparison_reading == "narrow_band_better_than_single_cutoff"
            and conditional_status == "conditional_setup_supported"
        ):
            interpretation_status = "context_band_with_conditional_setup_supported"
            strongest_pattern.append(
                "The remaining residual is better explained by a narrow context/bias "
                "band around the carried-forward threshold region, with setup-side "
                "severity sharpening collapsed residual depth inside that band."
            )
            unsupported_paths.extend(
                [
                    "plain_single_context_cutoff_as_complete_explanation",
                    "setup_side_as_stronger_top_level_gate_than_context",
                ]
            )
        elif comparison_reading == "narrow_band_better_than_single_cutoff":
            interpretation_status = "narrow_context_band_supported_setup_inconclusive"
            strongest_pattern.append(
                "The remaining residual is better aligned with a narrow context/bias "
                "band than with the coarse <= 0.55 cutoff, but the subordinate "
                "setup-side severity split remains incomplete."
            )
            unsupported_paths.extend(
                [
                    "plain_single_context_cutoff_as_complete_explanation",
                    "settled_context_band_plus_conditional_setup_mechanism",
                ]
            )
        elif comparison_reading == (
            "plain_single_cutoff_or_broad_helper_regime_more_plausible"
        ):
            interpretation_status = "broad_context_helper_regime_supported"
            strongest_pattern.append(
                "Context still looks like the leading descriptive axis, but the "
                "remaining residual reads more like an extreme-side cutoff or broad "
                "helper regime than a narrow interior band."
            )
            unsupported_paths.extend(
                [
                    "settled_narrow_context_band_mechanism",
                    "setup_side_as_stronger_top_level_gate_than_context",
                ]
            )
        else:
            interpretation_status = "context_band_conditional_setup_inconclusive"
            strongest_pattern.append(
                "Context/bias remains the leading descriptive axis, but neither a "
                "narrow context band nor a subordinate setup-side severity split "
                "clears a conservative support threshold on this run."
            )
            unsupported_paths.extend(
                [
                    "settled_narrow_context_band_mechanism",
                    "settled_context_band_plus_conditional_setup_mechanism",
                ]
            )

        if conditional_status == "setup_axis_competes_as_top_level_gate":
            unsupported_paths.append(
                "clean_gate_vs_conditional_setup_severity_split"
            )

    uncertainty = [
        "This report remains descriptive: a context band or setup-side severity split does not prove the literal hidden production merge rule.",
        "The fixed bands and thresholds are transparent report helpers only and are not proposed as production settings.",
        "The setup-side check stays subordinate to the context-band question and does not reopen a broader interaction-model search.",
    ]
    if interpretation_status not in {
        "context_band_with_conditional_setup_supported",
        "broad_context_helper_regime_supported",
    }:
        uncertainty.append(
            "Because the context-band evidence remains mixed or conditional, the best reading should be treated as an ordered descriptive summary rather than a settled mechanism claim."
        )

    return {
        "interpretation_status": interpretation_status,
        "facts": [item for item in facts if item],
        "observed_context_band_behavior": [
            item for item in observed_context_band_behavior if item
        ],
        "observed_conditional_setup_behavior": [
            item for item in observed_conditional_setup_behavior if item
        ],
        "strongest_candidate_explanatory_pattern": [
            item for item in strongest_pattern if item
        ],
        "unsupported_explanation_paths": [
            item for item in unsupported_paths if item
        ],
        "inference": [item for item in strongest_pattern if item],
        "uncertainty": uncertainty,
    }


def build_limitations(
    *,
    summary: dict[str, Any],
    context_band_interpretation: dict[str, Any],
    conditional_setup_profiles: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact profiles only the already-confirmed weighted aggregate residual slice and does not revisit earlier bottlenecks or production gate logic.",
        "Context profiling stays focused on context_bias_family_mean and its shortfall transform rather than reopening broad multi-axis exploration.",
        "The setup-side check is conditional only and is not promoted to a broader interaction search or top-level gate search.",
    ]
    if summary.get("comparison_support_status") != "supported":
        limitations.append(
            "The final preserved-vs-collapsed comparison is below the family's normal supported threshold, so context-band and conditional-setup claims remain provisional."
        )
    if str(context_band_interpretation.get("best_context_band_reading") or "") not in {
        "narrow_band",
        "broad_helper_regime",
        "simple_cutoff",
    }:
        limitations.append(
            "Ordered context-band profiling does not isolate a clean narrow-band or broad-helper shape on this run."
        )
    if str(conditional_setup_profiles.get("interpretation_status") or "") not in {
        "conditional_setup_supported",
    }:
        limitations.append(
            "Inside the selected context band, setup-side severity remains descriptive and does not independently settle the remaining residual."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    best_context_band = _safe_dict(widest.get("best_context_band"))
    best_conditional_setup_profile = _safe_dict(
        widest.get("best_conditional_setup_profile")
    )

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "baseline_name": _BASELINE_NAME,
        "interpretation_status": interpretation.get("interpretation_status"),
        "best_context_band": best_context_band,
        "best_conditional_setup_profile": best_conditional_setup_profile,
        "context_band_vs_single_cutoff_comparison": _safe_dict(
            widest.get("context_band_vs_single_cutoff_comparison")
        ),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            interpretation=interpretation,
            best_context_band=best_context_band,
            best_conditional_setup_profile=best_conditional_setup_profile,
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
            "- best_context_band: "
            f"{headline.get('best_context_band', 'none')}"
        )
        lines.append(
            "- best_conditional_setup_axis: "
            f"{headline.get('best_conditional_setup_axis', 'none')}"
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
    best_context_band = _safe_dict(final_assessment.get("best_context_band"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append(
        "- best_context_band: "
        f"{best_context_band.get('field', 'none')} @ "
        f"{best_context_band.get('band_label', 'n/a')}"
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


def _context_band_profile_for_field(
    *,
    context_band_profiles: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    return next(
        (
            _safe_dict(profile)
            for profile in context_band_profiles
            if str(_safe_dict(profile).get("field") or "") == field
        ),
        {},
    )


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


def _conditional_setup_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    best_threshold = _safe_dict(row.get("best_conditional_threshold_profile"))
    collapsed_delta = _to_float(
        row.get("collapsed_inside_vs_outside_residual_median_delta"),
        default=None,
    )
    return (
        _CONDITIONAL_STATUS_ORDER.get(
            str(row.get("conditional_severity_status") or ""),
            99,
        ),
        threshold_module._PROFILE_SUPPORT_ORDER.get(
            str(best_threshold.get("profile_strength_status") or ""),
            99,
        ),
        collapsed_delta if collapsed_delta is not None else 999.0,
        -_to_float(
            row.get("collapsed_pocket_lift_inside_context_band"),
            default=0.0,
        ),
        _to_float(
            row.get("preserved_leakage_impact_inside_context_band"),
            default=1.0,
        ),
        -_to_float(
            row.get("overall_gate_advantage_vs_this_setup_axis"),
            default=0.0,
        ),
        _SETUP_AXIS_ORDER.get(str(row.get("field") or ""), 99),
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    interpretation: dict[str, Any],
    best_context_band: dict[str, Any],
    best_conditional_setup_profile: dict[str, Any],
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
            "the widest configuration, so context-band diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one "
            "side of the preserved-vs-collapsed comparison is missing, so context-"
            "band diagnosis remains incomplete."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but context-band and conditional-setup claims "
            "stay withheld because the preserved-vs-collapsed comparison does not "
            "clear the family's normal support threshold."
        )
    if interpretation_status == "context_band_with_conditional_setup_supported":
        return (
            "The widest configuration supports a context band near "
            f"{best_context_band.get('band_label')} as the better residual gate "
            "description, with setup-side severity sharpening collapsed residual "
            "depth inside that band."
        )
    if interpretation_status == "narrow_context_band_supported_setup_inconclusive":
        return (
            "The widest configuration supports a narrow context band near "
            f"{best_context_band.get('band_label')} more than the coarse single "
            "cutoff, but the subordinate setup-side severity split remains "
            "inconclusive."
        )
    if interpretation_status == "broad_context_helper_regime_supported":
        return (
            "The widest configuration keeps context as the leading descriptive axis, "
            "but the ordered bands read more like a broad helper regime or simple "
            "cutoff than a narrow interior neighborhood."
        )
    if best_conditional_setup_profile:
        return (
            "The widest configuration still suggests some conditional setup-side "
            "behavior inside the selected context band, but the overall context-band "
            "mechanism remains too mixed to settle."
        )
    return (
        "The widest configuration keeps context/bias as the leading descriptive "
        "axis, but neither a narrow context band nor a conditional setup-side split "
        "is supported strongly enough to settle the remaining residual."
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
    return threshold_module._comparison_group_rows(rows, comparison_group)


def _value_in_band(
    *,
    value: Any,
    lower_bound_exclusive: float | None,
    upper_bound_inclusive: float | None,
) -> bool:
    return threshold_module._value_in_band(
        value=value,
        lower_bound_exclusive=lower_bound_exclusive,
        upper_bound_inclusive=upper_bound_inclusive,
    )


def _difference_or_none(left: Any, right: Any) -> float | None:
    return threshold_module._difference_or_none(left, right)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return threshold_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return threshold_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return threshold_module._safe_list(value)


if __name__ == "__main__":
    main()
