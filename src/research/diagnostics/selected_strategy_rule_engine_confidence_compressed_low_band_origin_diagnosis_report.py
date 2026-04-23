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
    "selected_strategy_rule_engine_confidence_compressed_low_band_origin_"
    "diagnosis_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Compressed Low-Band Origin "
    "Diagnosis Report"
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
_POCKET_GROUP_LABEL = "compressed_low_band_pocket"
_OUTSIDE_POCKET_GROUP_LABEL = "outside_compressed_low_band_pocket"
_PRESERVED_OUTSIDE_POCKET_GROUP_LABEL = (
    "preserved_outside_compressed_low_band_pocket"
)
_DENSE_BUCKET_DEFINITIONS: tuple[tuple[str, float, float], ...] = (
    ("(0.15, 0.20]", 0.15, 0.20),
    ("(0.20, 0.25]", 0.20, 0.25),
)
_MIN_GROUP_SUPPORT_ROWS = 10
_MIN_DENSE_BUCKET_SUPPORT_ROWS = 5
_PIECEWISE_RESIDUAL_MEDIAN_DIFF_THRESHOLD = 0.05
_PIECEWISE_CONTRIBUTOR_MEDIAN_DIFF_THRESHOLD = 0.08
_COMPARABLE_RESIDUAL_MEDIAN_DELTA = 0.03

_FIELD_SPECS: tuple[dict[str, Any], ...] = (
    {
        "field": _RULE_ENGINE_CONFIDENCE_FIELD,
        "field_label": "Rule engine confidence",
        "family": "actual_output",
        "orientation": "lower_is_weaker",
    },
    {
        "field": _BASELINE_NAME,
        "field_label": _BASELINE_LABEL,
        "family": "baseline",
        "orientation": "lower_is_weaker",
    },
    {
        "field": _RESIDUAL_FIELD,
        "field_label": "Residual vs weighted setup emphasis baseline",
        "family": "residual",
        "orientation": "more_negative_is_weaker",
    },
    {
        "field": "setup_layer_confidence",
        "field_label": "Setup layer confidence",
        "family": "setup",
        "orientation": "lower_is_weaker",
    },
    {
        "field": "context_layer_confidence",
        "field_label": "Context layer confidence",
        "family": "context_and_bias",
        "orientation": "lower_is_weaker",
    },
    {
        "field": "bias_layer_confidence",
        "field_label": "Bias layer confidence",
        "family": "context_and_bias",
        "orientation": "lower_is_weaker",
    },
    {
        "field": _CONTEXT_BIAS_FAMILY_FIELD,
        "field_label": "Context/bias family mean",
        "family": "context_and_bias",
        "orientation": "lower_is_weaker",
    },
    {
        "field": "selected_strategy_confidence",
        "field_label": "Selected-strategy confidence",
        "family": "selected_strategy",
        "orientation": "lower_is_weaker",
    },
    {
        "field": "trigger_layer_confidence",
        "field_label": "Trigger layer confidence",
        "family": "trigger_negative_control",
        "orientation": "lower_is_weaker",
    },
    {
        "field": "setup_shortfall",
        "field_label": "Setup shortfall",
        "family": "setup",
        "orientation": "higher_is_weaker",
    },
    {
        "field": "context_bias_family_shortfall",
        "field_label": "Context/bias family shortfall",
        "family": "context_and_bias",
        "orientation": "higher_is_weaker",
    },
    {
        "field": "selected_strategy_shortfall",
        "field_label": "Selected-strategy shortfall",
        "family": "selected_strategy",
        "orientation": "higher_is_weaker",
    },
)
_FIELD_ORDER = {
    str(spec["field"]): index for index, spec in enumerate(_FIELD_SPECS)
}
_FIELD_SPEC_MAP = {
    str(spec["field"]): dict(spec) for spec in _FIELD_SPECS
}
_RESIDUAL_FIELD_NAMES = {
    _RULE_ENGINE_CONFIDENCE_FIELD,
    _BASELINE_NAME,
    _RESIDUAL_FIELD,
}
_PRIMARY_CONTRIBUTOR_FAMILY_SPECS: tuple[dict[str, Any], ...] = (
    {
        "family": "setup",
        "field": "setup_layer_confidence",
        "field_label": "Setup layer confidence",
        "shortfall_field": "setup_shortfall",
        "shortfall_label": "Setup shortfall",
        "threshold": _LOW_CONFIDENCE_THRESHOLD,
    },
    {
        "family": "context_and_bias",
        "field": _CONTEXT_BIAS_FAMILY_FIELD,
        "field_label": "Context/bias family mean",
        "shortfall_field": "context_bias_family_shortfall",
        "shortfall_label": "Context/bias family shortfall",
        "threshold": _LOW_CONFIDENCE_THRESHOLD,
    },
    {
        "family": "selected_strategy",
        "field": "selected_strategy_confidence",
        "field_label": "Selected-strategy confidence",
        "shortfall_field": "selected_strategy_shortfall",
        "shortfall_label": "Selected-strategy shortfall",
        "threshold": _LOW_CONFIDENCE_THRESHOLD,
    },
    {
        "family": "trigger_negative_control",
        "field": "trigger_layer_confidence",
        "field_label": "Trigger layer confidence",
        "shortfall_field": None,
        "shortfall_label": None,
        "threshold": _LOW_CONFIDENCE_THRESHOLD,
    },
)
_PRIMARY_CONTRIBUTOR_FIELD_NAMES = {
    str(spec["field"]) for spec in _PRIMARY_CONTRIBUTOR_FAMILY_SPECS
}
_LOW_FLAG_FIELD_MAP = {
    "setup_layer_confidence": "low_setup_confidence_regime",
    _CONTEXT_BIAS_FAMILY_FIELD: "low_context_bias_family_regime",
    "selected_strategy_confidence": "low_selected_strategy_confidence_regime",
}
_FAMILY_ORDER = {
    "setup": 0,
    "context_and_bias": 1,
    "selected_strategy": 2,
    "trigger_negative_control": 3,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report that explains the origin of the "
            "already-supported compressed low-confidence pocket in actual "
            "rule_engine_confidence inside the final fully aligned plus "
            "rule-bias-aligned slice."
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
            "Retained for parity with sibling reports. This origin diagnosis itself "
            "reuses only the final preserved-vs-collapsed slice."
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
        run_selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report(
            input_path=input_path,
            output_dir=output_dir,
            configurations=configurations,
            min_symbol_support=args.min_symbol_support,
            write_report_copies=args.write_latest_copy,
        )
    )
    report = result["report"]
    pocket_summary = _safe_dict(report.get("pocket_summary"))
    dense_bucket_comparison = _safe_dict(report.get("dense_bucket_comparison"))
    joint_weakness_signature = _safe_dict(report.get("joint_weakness_signature"))

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
                "pocket_row_count": pocket_summary.get("pocket_row_count", 0),
                "pocket_collapsed_row_count": pocket_summary.get(
                    "pocket_collapsed_row_count",
                    0,
                ),
                "collapsed_rows_outside_pocket_row_count": pocket_summary.get(
                    "collapsed_rows_outside_pocket_row_count",
                    0,
                ),
                "joint_weakness_status": joint_weakness_signature.get(
                    "signature_status"
                ),
                "dense_bucket_piecewise_status": dense_bucket_comparison.get(
                    "piecewise_status"
                ),
                "stronger_downward_residual_bucket_label": (
                    dense_bucket_comparison.get(
                        "stronger_downward_residual_bucket_label"
                    )
                ),
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report(
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
        "dense_bucket_definitions": [
            {
                "bucket_label": label,
                "lower_bound_exclusive": lower_bound_exclusive,
                "upper_bound_inclusive": upper_bound_inclusive,
            }
            for label, lower_bound_exclusive, upper_bound_inclusive in (
                _DENSE_BUCKET_DEFINITIONS
            )
        ],
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
        "pocket_summary": _safe_dict(widest_summary.get("pocket_summary")),
        "pocket_vs_other_comparison": _safe_dict(
            widest_summary.get("pocket_vs_other_comparison")
        ),
        "pocket_vs_preserved_outside_comparison": _safe_dict(
            widest_summary.get("pocket_vs_preserved_outside_comparison")
        ),
        "residual_by_pocket_membership": _safe_dict(
            widest_summary.get("residual_by_pocket_membership")
        ),
        "contributor_family_by_pocket_membership": _safe_dict(
            widest_summary.get("contributor_family_by_pocket_membership")
        ),
        "dense_bucket_summary": _safe_dict(
            widest_summary.get("dense_bucket_summary")
        ),
        "dense_bucket_comparison": _safe_dict(
            widest_summary.get("dense_bucket_comparison")
        ),
        "contributor_family_by_dense_bucket": _safe_dict(
            widest_summary.get("contributor_family_by_dense_bucket")
        ),
        "joint_weakness_signature": _safe_dict(
            widest_summary.get("joint_weakness_signature")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and reuses the exact final fully aligned plus rule-bias-aligned preserved-vs-collapsed slice definition from the sibling diagnosis chain.",
            "The compressed pocket is fixed to actual rule_engine_confidence <= 0.25, with dense buckets fixed to (0.15, 0.20] and (0.20, 0.25].",
            "Residual remains fixed to rule_engine_confidence minus weighted_mean_setup_emphasis so the report can describe whether dense buckets differ by downward residual versus the carried-forward baseline.",
            "Joint weakness signatures and shortfall helpers are recomputed locally from base contributor confidence fields inside this report so the artifact does not rely on upstream residual helper-field availability.",
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
        _prepare_origin_residual_row(residual_module.build_residual_row(row))
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

    row_sets = _build_pocket_row_sets(residual_rows)
    pocket_summary = build_pocket_summary(row_sets=row_sets)
    pocket_vs_other_comparison = build_group_field_comparison(
        left_rows=row_sets["pocket_rows"],
        right_rows=row_sets["outside_pocket_rows"],
        left_group_label=_POCKET_GROUP_LABEL,
        right_group_label=_OUTSIDE_POCKET_GROUP_LABEL,
        field_specs=_FIELD_SPECS,
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    pocket_vs_preserved_outside_comparison = build_group_field_comparison(
        left_rows=row_sets["pocket_rows"],
        right_rows=row_sets["preserved_outside_pocket_rows"],
        left_group_label=_POCKET_GROUP_LABEL,
        right_group_label=_PRESERVED_OUTSIDE_POCKET_GROUP_LABEL,
        field_specs=_FIELD_SPECS,
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    residual_by_pocket_membership = build_group_field_comparison(
        left_rows=row_sets["pocket_rows"],
        right_rows=row_sets["outside_pocket_rows"],
        left_group_label=_POCKET_GROUP_LABEL,
        right_group_label=_OUTSIDE_POCKET_GROUP_LABEL,
        field_specs=[
            _FIELD_SPEC_MAP[_RULE_ENGINE_CONFIDENCE_FIELD],
            _FIELD_SPEC_MAP[_BASELINE_NAME],
            _FIELD_SPEC_MAP[_RESIDUAL_FIELD],
        ],
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    contributor_family_by_pocket_membership = build_contributor_family_summary(
        left_rows=row_sets["pocket_rows"],
        right_rows=row_sets["preserved_outside_pocket_rows"],
        left_group_label=_POCKET_GROUP_LABEL,
        right_group_label=_PRESERVED_OUTSIDE_POCKET_GROUP_LABEL,
        min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
    )
    dense_bucket_summary = build_dense_bucket_summary(
        collapsed_dense_bucket_rows_by_label=row_sets[
            "collapsed_dense_bucket_rows_by_label"
        ],
        preserved_dense_bucket_rows_by_label=row_sets[
            "preserved_dense_bucket_rows_by_label"
        ],
    )
    dense_bucket_comparison = build_dense_bucket_comparison(
        collapsed_dense_bucket_rows_by_label=row_sets[
            "collapsed_dense_bucket_rows_by_label"
        ],
        preserved_dense_bucket_rows_by_label=row_sets[
            "preserved_dense_bucket_rows_by_label"
        ],
    )
    left_dense_bucket_label = _DENSE_BUCKET_DEFINITIONS[0][0]
    right_dense_bucket_label = _DENSE_BUCKET_DEFINITIONS[1][0]
    contributor_family_by_dense_bucket = build_contributor_family_summary(
        left_rows=_safe_dict(row_sets["collapsed_dense_bucket_rows_by_label"]).get(
            left_dense_bucket_label,
            [],
        ),
        right_rows=_safe_dict(row_sets["collapsed_dense_bucket_rows_by_label"]).get(
            right_dense_bucket_label,
            [],
        ),
        left_group_label=left_dense_bucket_label,
        right_group_label=right_dense_bucket_label,
        min_support_rows=_MIN_DENSE_BUCKET_SUPPORT_ROWS,
    )
    joint_weakness_signature = build_joint_weakness_signature(
        pocket_rows=row_sets["pocket_rows"],
        preserved_outside_pocket_rows=row_sets["preserved_outside_pocket_rows"],
        outside_pocket_rows=row_sets["outside_pocket_rows"],
    )
    interpretation = build_interpretation(
        summary=summary,
        pocket_summary=pocket_summary,
        pocket_vs_other_comparison=pocket_vs_other_comparison,
        pocket_vs_preserved_outside_comparison=pocket_vs_preserved_outside_comparison,
        contributor_family_by_pocket_membership=(
            contributor_family_by_pocket_membership
        ),
        dense_bucket_summary=dense_bucket_summary,
        dense_bucket_comparison=dense_bucket_comparison,
        joint_weakness_signature=joint_weakness_signature,
    )
    limitations = build_limitations(
        summary=summary,
        pocket_summary=pocket_summary,
        dense_bucket_summary=dense_bucket_summary,
        dense_bucket_comparison=dense_bucket_comparison,
        joint_weakness_signature=joint_weakness_signature,
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
            "pocket_row_count": pocket_summary.get("pocket_row_count"),
            "collapsed_rows_outside_pocket_row_count": pocket_summary.get(
                "collapsed_rows_outside_pocket_row_count"
            ),
            "joint_weakness_status": joint_weakness_signature.get(
                "signature_status"
            ),
            "dense_bucket_piecewise_status": dense_bucket_comparison.get(
                "piecewise_status"
            ),
            "interpretation_status": interpretation.get("interpretation_status"),
        },
        "summary": summary,
        "actual_rule_engine_confidence_reference": actual_rule_engine_confidence_reference,
        "baseline_reference": baseline_reference,
        "pocket_summary": pocket_summary,
        "pocket_vs_other_comparison": pocket_vs_other_comparison,
        "pocket_vs_preserved_outside_comparison": (
            pocket_vs_preserved_outside_comparison
        ),
        "residual_by_pocket_membership": residual_by_pocket_membership,
        "contributor_family_by_pocket_membership": (
            contributor_family_by_pocket_membership
        ),
        "dense_bucket_summary": dense_bucket_summary,
        "dense_bucket_comparison": dense_bucket_comparison,
        "contributor_family_by_dense_bucket": contributor_family_by_dense_bucket,
        "joint_weakness_signature": joint_weakness_signature,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_pocket_summary(
    *,
    row_sets: dict[str, Any],
) -> dict[str, Any]:
    eligible_rows = _safe_list(row_sets.get("eligible_rows"))
    pocket_rows = _safe_list(row_sets.get("pocket_rows"))
    outside_pocket_rows = _safe_list(row_sets.get("outside_pocket_rows"))
    pocket_preserved_rows = _safe_list(row_sets.get("pocket_preserved_rows"))
    pocket_collapsed_rows = _safe_list(row_sets.get("pocket_collapsed_rows"))
    preserved_outside_pocket_rows = _safe_list(
        row_sets.get("preserved_outside_pocket_rows")
    )
    collapsed_outside_pocket_rows = _safe_list(
        row_sets.get("collapsed_outside_pocket_rows")
    )
    collapsed_dense_bucket_rows_by_label = _safe_dict(
        row_sets.get("collapsed_dense_bucket_rows_by_label")
    )
    preserved_dense_bucket_rows_by_label = _safe_dict(
        row_sets.get("preserved_dense_bucket_rows_by_label")
    )
    dense_bucket_collapsed_row_count = sum(
        len(_safe_list(rows)) for rows in collapsed_dense_bucket_rows_by_label.values()
    )
    dense_bucket_preserved_row_count = sum(
        len(_safe_list(rows)) for rows in preserved_dense_bucket_rows_by_label.values()
    )
    dense_bucket_total_row_count = (
        dense_bucket_collapsed_row_count + dense_bucket_preserved_row_count
    )

    return {
        "eligible_rule_engine_confidence_row_count": len(eligible_rows),
        "missing_rule_engine_confidence_row_count": int(
            row_sets.get("missing_rule_engine_confidence_row_count", 0) or 0
        ),
        "pocket_row_count": len(pocket_rows),
        "outside_pocket_row_count": len(outside_pocket_rows),
        "pocket_rate_within_present_rows": _safe_ratio(
            len(pocket_rows),
            len(eligible_rows),
        ),
        "pocket_preserved_row_count": len(pocket_preserved_rows),
        "pocket_collapsed_row_count": len(pocket_collapsed_rows),
        "outside_pocket_preserved_row_count": len(preserved_outside_pocket_rows),
        "collapsed_rows_outside_pocket_row_count": len(collapsed_outside_pocket_rows),
        "zero_collapsed_outside_pocket": len(collapsed_outside_pocket_rows) == 0,
        "pocket_vs_other_support_status": _group_support_status(
            left_row_count=len(pocket_rows),
            right_row_count=len(outside_pocket_rows),
            min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
        ),
        "pocket_vs_preserved_outside_support_status": _group_support_status(
            left_row_count=len(pocket_rows),
            right_row_count=len(preserved_outside_pocket_rows),
            min_support_rows=_MIN_GROUP_SUPPORT_ROWS,
        ),
        "dense_bucket_row_count": dense_bucket_total_row_count,
        "dense_bucket_collapsed_row_count": dense_bucket_collapsed_row_count,
        "dense_bucket_preserved_row_count": dense_bucket_preserved_row_count,
        "dense_bucket_preserved_contamination_present": (
            dense_bucket_preserved_row_count > 0
        ),
        "dense_bucket_labels_present": [
            label
            for label, rows in collapsed_dense_bucket_rows_by_label.items()
            if len(_safe_list(rows)) > 0
        ],
    }


def build_group_field_comparison(
    *,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    left_group_label: str,
    right_group_label: str,
    field_specs: Sequence[dict[str, Any]],
    min_support_rows: int,
) -> dict[str, Any]:
    field_rows = [
        _build_group_field_row(
            left_rows=left_rows,
            right_rows=right_rows,
            left_group_label=left_group_label,
            right_group_label=right_group_label,
            field_spec=field_spec,
        )
        for field_spec in field_specs
    ]
    strongest_left_group_severity_fields = [
        row["field"]
        for row in sorted(field_rows, key=_left_group_severity_sort_key)
        if _safe_dict(row).get("left_group_more_severe_by_orientation") is True
    ][:5]

    return {
        "support_status": _group_support_status(
            left_row_count=len(left_rows),
            right_row_count=len(right_rows),
            min_support_rows=min_support_rows,
        ),
        "left_group_label": left_group_label,
        "right_group_label": right_group_label,
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "field_comparisons": field_rows,
        "strongest_left_group_severity_fields": strongest_left_group_severity_fields,
    }


def build_contributor_family_summary(
    *,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    left_group_label: str,
    right_group_label: str,
    min_support_rows: int,
) -> dict[str, Any]:
    family_rows: list[dict[str, Any]] = []
    for family_spec in _PRIMARY_CONTRIBUTOR_FAMILY_SPECS:
        field = str(family_spec["field"])
        shortfall_field = family_spec.get("shortfall_field")
        threshold = _to_float(family_spec.get("threshold"), default=None)
        left_summary = final_split_module._numeric_field_summary(left_rows, field)
        right_summary = final_split_module._numeric_field_summary(right_rows, field)
        left_shortfall_summary = (
            final_split_module._numeric_field_summary(left_rows, str(shortfall_field))
            if shortfall_field
            else {}
        )
        right_shortfall_summary = (
            final_split_module._numeric_field_summary(right_rows, str(shortfall_field))
            if shortfall_field
            else {}
        )
        left_median = _to_float(left_summary.get("median"), default=None)
        right_median = _to_float(right_summary.get("median"), default=None)
        primary_gap = _difference_or_none(left_median, right_median)
        family_rows.append(
            {
                "family": family_spec["family"],
                "field": field,
                "field_label": family_spec["field_label"],
                "shortfall_field": shortfall_field,
                "shortfall_label": family_spec.get("shortfall_label"),
                "left_group_summary": left_summary,
                "right_group_summary": right_summary,
                "left_group_shortfall_summary": left_shortfall_summary,
                "right_group_shortfall_summary": right_shortfall_summary,
                "primary_median_difference_left_minus_right": primary_gap,
                "primary_abs_median_difference": _safe_abs_float(primary_gap),
                "left_group_more_severe_by_orientation": _left_group_more_severe(
                    orientation="lower_is_weaker",
                    left_median=left_median,
                    right_median=right_median,
                ),
                "severity_gap_in_left_direction": _severity_gap_in_left_direction(
                    orientation="lower_is_weaker",
                    left_median=left_median,
                    right_median=right_median,
                ),
                "low_threshold": threshold,
                "left_group_low_rate": _low_rate_for_field(
                    rows=left_rows,
                    field=field,
                    threshold=threshold,
                ),
                "right_group_low_rate": _low_rate_for_field(
                    rows=right_rows,
                    field=field,
                    threshold=threshold,
                ),
            }
        )

    for row in family_rows:
        row["low_rate_gap_left_minus_right"] = round(
            _to_float(row.get("left_group_low_rate"), default=0.0)
            - _to_float(row.get("right_group_low_rate"), default=0.0),
            6,
        )

    family_rows.sort(
        key=lambda row: (
            _FAMILY_ORDER.get(str(row.get("family") or ""), 99),
            -_to_float(row.get("severity_gap_in_left_direction"), default=-999.0),
        )
    )
    strongest_non_trigger_family = _safe_dict(
        next(
            (
                row
                for row in sorted(family_rows, key=_family_severity_sort_key)
                if str(_safe_dict(row).get("family") or "")
                != "trigger_negative_control"
                and _safe_dict(row).get("left_group_more_severe_by_orientation")
                is True
            ),
            {},
        )
    )

    return {
        "support_status": _group_support_status(
            left_row_count=len(left_rows),
            right_row_count=len(right_rows),
            min_support_rows=min_support_rows,
        ),
        "left_group_label": left_group_label,
        "right_group_label": right_group_label,
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "family_summaries": family_rows,
        "strongest_non_trigger_family": strongest_non_trigger_family,
        "trigger_negative_control_status": _trigger_negative_control_status(
            family_rows=family_rows
        ),
    }


def build_dense_bucket_summary(
    *,
    collapsed_dense_bucket_rows_by_label: dict[str, Sequence[dict[str, Any]]],
    preserved_dense_bucket_rows_by_label: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    bucket_rows = [
        _build_dense_bucket_row(
            bucket_label=label,
            collapsed_rows=_safe_list(
                collapsed_dense_bucket_rows_by_label.get(label)
            ),
            preserved_rows=_safe_list(
                preserved_dense_bucket_rows_by_label.get(label)
            ),
        )
        for label, _, _ in _DENSE_BUCKET_DEFINITIONS
    ]
    present_bucket_labels = [
        row["bucket_label"]
        for row in bucket_rows
        if int(row["collapsed_origin_row_count"]) > 0
    ]
    preserved_contamination_row_count = sum(
        int(_safe_dict(row).get("preserved_row_count", 0) or 0)
        for row in bucket_rows
    )
    return {
        "bucket_summaries": bucket_rows,
        "present_bucket_labels": present_bucket_labels,
        "present_bucket_count": len(present_bucket_labels),
        "collapsed_origin_row_count": sum(
            int(_safe_dict(row).get("collapsed_origin_row_count", 0) or 0)
            for row in bucket_rows
        ),
        "preserved_contamination_row_count": preserved_contamination_row_count,
        "preserved_contamination_present": preserved_contamination_row_count > 0,
        "comparison_population": "collapsed_only_dense_bucket_rows",
    }


def build_dense_bucket_comparison(
    *,
    collapsed_dense_bucket_rows_by_label: dict[str, Sequence[dict[str, Any]]],
    preserved_dense_bucket_rows_by_label: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    left_bucket_label = _DENSE_BUCKET_DEFINITIONS[0][0]
    right_bucket_label = _DENSE_BUCKET_DEFINITIONS[1][0]
    left_rows = _safe_list(collapsed_dense_bucket_rows_by_label.get(left_bucket_label))
    right_rows = _safe_list(collapsed_dense_bucket_rows_by_label.get(right_bucket_label))
    left_preserved_contamination_rows = _safe_list(
        preserved_dense_bucket_rows_by_label.get(left_bucket_label)
    )
    right_preserved_contamination_rows = _safe_list(
        preserved_dense_bucket_rows_by_label.get(right_bucket_label)
    )
    field_comparison = build_group_field_comparison(
        left_rows=left_rows,
        right_rows=right_rows,
        left_group_label=left_bucket_label,
        right_group_label=right_bucket_label,
        field_specs=[
            _FIELD_SPEC_MAP[_RULE_ENGINE_CONFIDENCE_FIELD],
            _FIELD_SPEC_MAP[_BASELINE_NAME],
            _FIELD_SPEC_MAP[_RESIDUAL_FIELD],
            _FIELD_SPEC_MAP["setup_layer_confidence"],
            _FIELD_SPEC_MAP[_CONTEXT_BIAS_FAMILY_FIELD],
            _FIELD_SPEC_MAP["selected_strategy_confidence"],
            _FIELD_SPEC_MAP["trigger_layer_confidence"],
            _FIELD_SPEC_MAP["setup_shortfall"],
            _FIELD_SPEC_MAP["context_bias_family_shortfall"],
            _FIELD_SPEC_MAP["selected_strategy_shortfall"],
        ],
        min_support_rows=_MIN_DENSE_BUCKET_SUPPORT_ROWS,
    )
    residual_row = _comparison_row_for_field(
        field_comparison=field_comparison,
        field=_RESIDUAL_FIELD,
    )
    left_residual_median = _to_float(
        _safe_dict(residual_row.get("left_group_summary")).get("median"),
        default=None,
    )
    right_residual_median = _to_float(
        _safe_dict(residual_row.get("right_group_summary")).get("median"),
        default=None,
    )
    residual_median_difference_left_minus_right = _difference_or_none(
        left_residual_median,
        right_residual_median,
    )
    stronger_downward_residual_bucket_label = _stronger_downward_residual_bucket_label(
        left_bucket_label=left_bucket_label,
        right_bucket_label=right_bucket_label,
        left_residual_median=left_residual_median,
        right_residual_median=right_residual_median,
    )

    material_contributor_shifts = [
        row["field"]
        for row in _safe_list(field_comparison.get("field_comparisons"))
        if str(_safe_dict(row).get("field") or "") in _PRIMARY_CONTRIBUTOR_FIELD_NAMES
        and (
            _safe_abs_float(_safe_dict(row).get("left_minus_right_median")) or 0.0
        )
        >= _PIECEWISE_CONTRIBUTOR_MEDIAN_DIFF_THRESHOLD
    ]
    left_dominant_signature = _dominant_exact_low_confidence_signature(left_rows)
    right_dominant_signature = _dominant_exact_low_confidence_signature(right_rows)
    dominant_signature_shift = (
        bool(left_dominant_signature)
        and bool(right_dominant_signature)
        and str(left_dominant_signature.get("signature_label") or "")
        != str(right_dominant_signature.get("signature_label") or "")
        and max(
            _to_float(left_dominant_signature.get("share"), default=0.0),
            _to_float(right_dominant_signature.get("share"), default=0.0),
        )
        >= 0.30
    )

    piecewise_status = _dense_bucket_piecewise_status(
        left_row_count=len(left_rows),
        right_row_count=len(right_rows),
        residual_median_difference_left_minus_right=(
            residual_median_difference_left_minus_right
        ),
        material_contributor_shift_count=len(material_contributor_shifts),
        dominant_signature_shift=dominant_signature_shift,
    )
    if piecewise_status == "dense_bucket_comparison_unavailable":
        piecewise_reading = (
            "At least one dense bucket is absent, so the report keeps the dense-pocket "
            "origin reading descriptive rather than claiming a piecewise split."
        )
    elif piecewise_status == "bucket_distinct_piecewise_regimes":
        piecewise_reading = (
            "The two dense buckets differ in both residual depth and contributor mix, "
            "so the compressed pocket behaves like piecewise sub-regimes rather than "
            "one homogeneous bucket cluster."
        )
    elif piecewise_status == "same_regime_with_ordered_severity":
        piecewise_reading = (
            "The dense buckets still look related, but at least one ordered severity "
            "step is visible inside the pocket."
        )
    else:
        piecewise_reading = (
            "The dense buckets look like the same broad regime or only a shallow step "
            "change on this run."
        )

    preserved_contamination_present = bool(
        left_preserved_contamination_rows or right_preserved_contamination_rows
    )
    if preserved_contamination_present:
        piecewise_reading += (
            " Preserved contamination inside dense buckets is reported separately, and "
            "the dense-bucket origin comparison itself is evaluated on collapsed rows only."
        )

    return {
        "support_status": field_comparison.get("support_status"),
        "left_bucket_label": left_bucket_label,
        "right_bucket_label": right_bucket_label,
        "left_row_count": len(left_rows),
        "right_row_count": len(right_rows),
        "left_preserved_contamination_row_count": len(
            left_preserved_contamination_rows
        ),
        "right_preserved_contamination_row_count": len(
            right_preserved_contamination_rows
        ),
        "preserved_contamination_present": preserved_contamination_present,
        "comparison_population": "collapsed_only_dense_bucket_rows",
        "field_comparison": field_comparison,
        "residual_median_difference_left_minus_right": (
            residual_median_difference_left_minus_right
        ),
        "stronger_downward_residual_bucket_label": (
            stronger_downward_residual_bucket_label
        ),
        "material_contributor_shift_fields": material_contributor_shifts,
        "left_dominant_exact_low_confidence_signature": left_dominant_signature,
        "right_dominant_exact_low_confidence_signature": right_dominant_signature,
        "piecewise_status": piecewise_status,
        "piecewise_reading": piecewise_reading,
    }


def build_joint_weakness_signature(
    *,
    pocket_rows: Sequence[dict[str, Any]],
    preserved_outside_pocket_rows: Sequence[dict[str, Any]],
    outside_pocket_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    pocket_complete_rows = _rows_with_complete_joint_signature(pocket_rows)
    reference_rows = (
        _rows_with_complete_joint_signature(preserved_outside_pocket_rows)
        if preserved_outside_pocket_rows
        else _rows_with_complete_joint_signature(outside_pocket_rows)
    )
    reference_group_label = (
        _PRESERVED_OUTSIDE_POCKET_GROUP_LABEL
        if preserved_outside_pocket_rows
        else _OUTSIDE_POCKET_GROUP_LABEL
    )

    low_surface_count_profiles = [
        _low_surface_count_profile(
            pocket_rows=pocket_complete_rows,
            reference_rows=reference_rows,
            low_surface_count=low_surface_count,
        )
        for low_surface_count in range(4)
    ]
    two_or_more_low_surface_profile = _combined_low_surface_profile(
        pocket_rows=pocket_complete_rows,
        reference_rows=reference_rows,
        minimum_low_surface_count=2,
    )
    three_low_surface_profile = _combined_low_surface_profile(
        pocket_rows=pocket_complete_rows,
        reference_rows=reference_rows,
        minimum_low_surface_count=3,
    )

    single_family_low_rates = []
    for family_spec in _PRIMARY_CONTRIBUTOR_FAMILY_SPECS:
        field = str(family_spec["field"])
        if field not in _LOW_FLAG_FIELD_MAP:
            continue
        flag_field = _LOW_FLAG_FIELD_MAP[field]
        pocket_rate = _rate_for_boolean_field(
            rows=pocket_complete_rows,
            field=flag_field,
            expected_value=True,
        )
        reference_rate = _rate_for_boolean_field(
            rows=reference_rows,
            field=flag_field,
            expected_value=True,
        )
        single_family_low_rates.append(
            {
                "family": family_spec["family"],
                "field": field,
                "field_label": family_spec["field_label"],
                "pocket_low_rate": pocket_rate,
                "reference_low_rate": reference_rate,
                "rate_gap": round(pocket_rate - reference_rate, 6),
            }
        )
    single_family_low_rates.sort(
        key=lambda row: (
            -_to_float(row.get("rate_gap"), default=0.0),
            _FAMILY_ORDER.get(str(row.get("family") or ""), 99),
        )
    )
    best_single_family_low_rate = _safe_dict(
        single_family_low_rates[0] if single_family_low_rates else {}
    )

    exact_signature_rows = _exact_low_confidence_signature_comparison_rows(
        pocket_rows=pocket_complete_rows,
        reference_rows=reference_rows,
    )
    dominant_pocket_joint_signature = _safe_dict(
        next(
            (
                row
                for row in exact_signature_rows
                if int(_safe_dict(row).get("low_surface_count", 0) or 0) >= 2
                and int(_safe_dict(row).get("pocket_row_count", 0) or 0) > 0
            ),
            {},
        )
    )

    joint_gap = _to_float(
        two_or_more_low_surface_profile.get("rate_gap"),
        default=0.0,
    )
    best_single_gap = _to_float(best_single_family_low_rate.get("rate_gap"), default=0.0)
    dominant_signature_gap = _to_float(
        dominant_pocket_joint_signature.get("rate_gap"),
        default=0.0,
    )
    dominant_signature_pocket_rate = _to_float(
        dominant_pocket_joint_signature.get("pocket_rate"),
        default=0.0,
    )
    dominant_signature_reference_rate = _to_float(
        dominant_pocket_joint_signature.get("reference_rate"),
        default=0.0,
    )
    stable_signature_visible = (
        dominant_signature_pocket_rate >= 0.30
        and dominant_signature_reference_rate <= 0.10
    )

    if not pocket_complete_rows:
        signature_status = "insufficient_data"
        reading = (
            "Pocket rows do not retain enough complete contributor signatures to judge "
            "whether joint weakness is visible."
        )
    elif not reference_rows:
        signature_status = "reference_group_absent"
        reading = (
            "A non-pocket reference group with complete contributor signatures is "
            "absent, so joint weakness remains descriptive only."
        )
    elif (
        _to_float(two_or_more_low_surface_profile.get("pocket_rate"), default=0.0)
        >= 0.75
        and _to_float(
            two_or_more_low_surface_profile.get("reference_rate"),
            default=1.0,
        )
        <= 0.25
        and (
            joint_gap >= best_single_gap + 0.10
            or (stable_signature_visible and dominant_signature_gap >= 0.20)
        )
    ):
        signature_status = "joint_weakness_signature_supported"
        reading = (
            "Pocket rows are better described by a multi-family weakness signature "
            "than by any one carried-forward single-family helper cutoff."
        )
    elif (
        _to_float(two_or_more_low_surface_profile.get("pocket_rate"), default=0.0)
        > _to_float(
            two_or_more_low_surface_profile.get("reference_rate"),
            default=0.0,
        )
        and joint_gap >= best_single_gap - 0.05
    ):
        signature_status = "joint_weakness_signature_leaning"
        reading = (
            "Pocket rows lean toward a joint weakness pattern, but the margin over the "
            "best single-family helper view stays modest."
        )
    elif best_single_gap >= joint_gap + 0.10:
        signature_status = "single_family_cutoff_leaning"
        reading = (
            "One carried-forward family-specific helper cutoff still looks at least as "
            "sharp as the multi-family signature on this run."
        )
    else:
        signature_status = "mixed"
        reading = (
            "Joint weakness is visible in places, but the signature stays too mixed to "
            "treat as the cleanest pocket-origin description."
        )

    return {
        "reference_group_label": reference_group_label,
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "pocket_complete_row_count": len(pocket_complete_rows),
        "reference_complete_row_count": len(reference_rows),
        "low_surface_count_profiles": low_surface_count_profiles,
        "two_or_more_low_surface_profile": two_or_more_low_surface_profile,
        "three_low_surface_profile": three_low_surface_profile,
        "single_family_low_rates": single_family_low_rates,
        "best_single_family_low_rate": best_single_family_low_rate,
        "exact_signature_comparison": exact_signature_rows,
        "dominant_pocket_joint_signature": dominant_pocket_joint_signature,
        "stable_signature_visible": stable_signature_visible,
        "signature_status": signature_status,
        "reading": reading,
    }


def build_interpretation(
    *,
    summary: dict[str, Any],
    pocket_summary: dict[str, Any],
    pocket_vs_other_comparison: dict[str, Any],
    pocket_vs_preserved_outside_comparison: dict[str, Any],
    contributor_family_by_pocket_membership: dict[str, Any],
    dense_bucket_summary: dict[str, Any],
    dense_bucket_comparison: dict[str, Any],
    joint_weakness_signature: dict[str, Any],
) -> dict[str, Any]:
    strongest_non_trigger_family = _safe_dict(
        contributor_family_by_pocket_membership.get("strongest_non_trigger_family")
    )
    best_single_family = _safe_dict(
        joint_weakness_signature.get("best_single_family_low_rate")
    )
    dense_bucket_row = _safe_dict(
        _comparison_row_for_field(
            field_comparison=dense_bucket_comparison.get("field_comparison"),
            field=_RESIDUAL_FIELD,
        )
    )

    facts = [
        (
            "Final rule-bias-aligned slice support: "
            f"{summary.get('comparison_support_status', 'unknown')} "
            f"(preserved={summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            f"collapsed={summary.get('collapsed_final_hold_outcome_row_count', 0)})."
        ),
        (
            "Compressed pocket counts: "
            f"pocket={pocket_summary.get('pocket_row_count', 0)}, "
            f"outside_pocket={pocket_summary.get('outside_pocket_row_count', 0)}, "
            f"pocket_collapsed={pocket_summary.get('pocket_collapsed_row_count', 0)}, "
            "collapsed_outside_pocket="
            f"{pocket_summary.get('collapsed_rows_outside_pocket_row_count', 0)}."
        ),
        (
            "Joint weakness status="
            f"{joint_weakness_signature.get('signature_status', 'unknown')}; "
            "two_or_more_low_surface_rate="
            f"{_safe_dict(joint_weakness_signature.get('two_or_more_low_surface_profile')).get('pocket_rate')} "
            "vs reference="
            f"{_safe_dict(joint_weakness_signature.get('two_or_more_low_surface_profile')).get('reference_rate')}."
        ),
        (
            "Dense bucket piecewise status="
            f"{dense_bucket_comparison.get('piecewise_status', 'unknown')}; "
            "stronger_downward_residual_bucket="
            f"{dense_bucket_comparison.get('stronger_downward_residual_bucket_label')}."
        ),
    ]
    if pocket_summary.get("zero_collapsed_outside_pocket") is True:
        facts.append(
            "No collapsed rows remain outside the fixed low-band pocket on this run, "
            "so the live control is preserved rows outside the pocket rather than a "
            "matched collapsed-outside group."
        )
    if dense_bucket_summary.get("preserved_contamination_present") is True:
        facts.append(
            "Preserved rows appear inside one or more dense buckets, so dense-bucket "
            "origin comparison is evaluated on collapsed rows only and contamination "
            "is reported separately."
        )
    if strongest_non_trigger_family:
        facts.append(
            "Strongest pocket-vs-preserved-outside non-trigger family="
            f"{strongest_non_trigger_family.get('family')} "
            f"(field={strongest_non_trigger_family.get('field')}, "
            "severity_gap="
            f"{strongest_non_trigger_family.get('severity_gap_in_left_direction')})."
        )
    if best_single_family:
        facts.append(
            "Best single-family helper rate gap="
            f"{best_single_family.get('field')} "
            f"({best_single_family.get('rate_gap')})."
        )
    if dense_bucket_row:
        facts.append(
            "Dense bucket residual medians: left="
            f"{_safe_dict(dense_bucket_row.get('left_group_summary')).get('median')}, "
            "right="
            f"{_safe_dict(dense_bucket_row.get('right_group_summary')).get('median')}."
        )

    interpretation_status = "comparison_unsupported"
    strongest_pattern: list[str] = []

    if summary.get("comparison_support_status") != "supported":
        strongest_pattern.append(
            "The final preserved-vs-collapsed slice exists, but it does not clear the "
            "family's normal supported-comparison threshold, so pocket-origin claims "
            "stay withheld."
        )
    else:
        joint_status = str(joint_weakness_signature.get("signature_status") or "")
        piecewise_status = str(dense_bucket_comparison.get("piecewise_status") or "")

        if (
            joint_status == "joint_weakness_signature_supported"
            and piecewise_status == "bucket_distinct_piecewise_regimes"
        ):
            interpretation_status = (
                "compressed_low_band_origin_joint_weakness_with_piecewise_dense_"
                "buckets_supported"
            )
            strongest_pattern.append(
                "Pocket membership is best described as a multi-family weakness "
                "signature, and the two dense buckets behave like piecewise "
                "sub-regimes with different residual-origin mixes."
            )
        elif joint_status == "joint_weakness_signature_supported":
            interpretation_status = (
                "compressed_low_band_origin_joint_weakness_supported"
            )
            strongest_pattern.append(
                "Pocket membership is better described by joint weakness across "
                "multiple contributor families than by any one carried-forward "
                "single-family helper cutoff."
            )
        elif piecewise_status == "bucket_distinct_piecewise_regimes":
            interpretation_status = (
                "compressed_low_band_origin_piecewise_dense_buckets_supported_"
                "joint_signature_mixed"
            )
            strongest_pattern.append(
                "Dense buckets show a piecewise split in origin behavior, but the "
                "broader joint weakness signature remains mixed."
            )
        elif joint_status == "joint_weakness_signature_leaning":
            interpretation_status = (
                "compressed_low_band_origin_joint_weakness_leaning"
            )
            strongest_pattern.append(
                "The compressed pocket still leans toward a joint weakness origin, "
                "but the margin over the strongest single-family helper remains "
                "conservative rather than decisive."
            )
        elif joint_status == "single_family_cutoff_leaning":
            interpretation_status = (
                "compressed_low_band_origin_single_family_helper_leaning"
            )
            strongest_pattern.append(
                "A single carried-forward family-specific helper view remains as sharp "
                "as the broader joint signature, so the report does not overclaim a "
                "uniquely multi-family origin."
            )
        else:
            interpretation_status = "compressed_low_band_origin_inconclusive"
            strongest_pattern.append(
                "Pocket-origin evidence remains descriptive but mixed: no single "
                "joint-signature or dense-bucket reading clears a conservative support "
                "threshold on this run."
            )

    uncertainty = [
        "This report remains descriptive: pocket-vs-non-pocket comparisons cannot prove the exact hidden merge rule inside the production decision layer.",
        "The fixed low-band pocket and dense buckets are diagnosis anchors only and are not production thresholds or engine recommendations.",
        "Joint weakness signatures are recomputed locally from base contributor confidence fields so the artifact stays robust without reopening a new raw-surface threshold search.",
    ]
    if pocket_summary.get("zero_collapsed_outside_pocket") is True:
        uncertainty.append(
            "Because no collapsed rows remain outside the pocket, the most useful live control is preserved rows outside the pocket; that asymmetry helps description but does not establish a symmetric failure boundary."
        )
    if dense_bucket_comparison.get("piecewise_status") != "bucket_distinct_piecewise_regimes":
        uncertainty.append(
            "Dense-bucket behavior may still be gradual rather than piecewise on this run, so bucket-level mechanism language should stay conservative unless the piecewise reading is explicit."
        )

    return {
        "interpretation_status": interpretation_status,
        "facts": [item for item in facts if item],
        "strongest_candidate_explanatory_pattern": [
            item for item in strongest_pattern if item
        ],
        "inference": [item for item in strongest_pattern if item],
        "uncertainty": uncertainty,
    }


def build_limitations(
    *,
    summary: dict[str, Any],
    pocket_summary: dict[str, Any],
    dense_bucket_summary: dict[str, Any],
    dense_bucket_comparison: dict[str, Any],
    joint_weakness_signature: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact stays inside the already-confirmed final slice and does not reopen earlier bottlenecks, mapper changes, engine changes, or execution-gate changes.",
        "Pocket membership is fixed to actual rule_engine_confidence <= 0.25, and dense buckets are fixed to (0.15, 0.20] and (0.20, 0.25], so the report does not run another threshold hunt.",
        "Joint weakness signatures and shortfall helpers are recomputed locally from base contributor confidence fields so the report does not depend on upstream residual helper-field contracts.",
    ]
    if summary.get("comparison_support_status") != "supported":
        limitations.append(
            "The final preserved-vs-collapsed comparison is below the family's normal supported threshold, so any pocket-origin reading remains provisional."
        )
    if pocket_summary.get("zero_collapsed_outside_pocket") is True:
        limitations.append(
            "Collapsed rows do not appear outside the pocket on this run, so the strongest non-pocket control is preserved-only rather than a balanced preserved-vs-collapsed outside comparison."
        )
    if dense_bucket_summary.get("preserved_contamination_present") is True:
        limitations.append(
            "Preserved contamination inside dense buckets is reported separately, and dense-bucket origin comparisons are collapsed-only to avoid mixed-population distortion."
        )
    if dense_bucket_comparison.get("piecewise_status") == "dense_bucket_comparison_unavailable":
        limitations.append(
            "At least one dense bucket is absent, so the report avoids a piecewise dense-bucket claim."
        )
    if joint_weakness_signature.get("signature_status") not in {
        "joint_weakness_signature_supported",
    }:
        limitations.append(
            "The joint weakness signature remains descriptive and does not by itself settle whether the hidden compression rule is additive, piecewise, or another merge shape."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    interpretation = _safe_dict(widest.get("interpretation"))
    pocket_summary = _safe_dict(widest.get("pocket_summary"))
    dense_bucket_comparison = _safe_dict(widest.get("dense_bucket_comparison"))
    joint_weakness_signature = _safe_dict(widest.get("joint_weakness_signature"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "pocket_summary": pocket_summary,
        "joint_weakness_status": joint_weakness_signature.get("signature_status"),
        "dense_bucket_piecewise_status": dense_bucket_comparison.get(
            "piecewise_status"
        ),
        "stronger_downward_residual_bucket_label": dense_bucket_comparison.get(
            "stronger_downward_residual_bucket_label"
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            pocket_summary=pocket_summary,
            interpretation=interpretation,
            dense_bucket_comparison=dense_bucket_comparison,
            joint_weakness_signature=joint_weakness_signature,
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
            "- pocket_row_count: "
            f"{headline.get('pocket_row_count', 0)}"
        )
        lines.append(
            "- collapsed_rows_outside_pocket_row_count: "
            f"{headline.get('collapsed_rows_outside_pocket_row_count', 0)}"
        )
        lines.append(
            "- joint_weakness_status: "
            f"{headline.get('joint_weakness_status', 'unknown')}"
        )
        lines.append(
            "- dense_bucket_piecewise_status: "
            f"{headline.get('dense_bucket_piecewise_status', 'unknown')}"
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
        "- joint_weakness_status: "
        f"{final_assessment.get('joint_weakness_status', 'unknown')}"
    )
    lines.append(
        "- dense_bucket_piecewise_status: "
        f"{final_assessment.get('dense_bucket_piecewise_status', 'unknown')}"
    )
    lines.append(
        "- stronger_downward_residual_bucket_label: "
        f"{final_assessment.get('stronger_downward_residual_bucket_label', 'n/a')}"
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


def _prepare_origin_residual_row(row: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    context_bias_family_mean = _to_float(
        prepared.get(_CONTEXT_BIAS_FAMILY_FIELD),
        default=None,
    )
    if context_bias_family_mean is None:
        context_bias_family_mean = _context_bias_family_mean_from_row(prepared)
        if context_bias_family_mean is not None:
            prepared[_CONTEXT_BIAS_FAMILY_FIELD] = context_bias_family_mean

    low_confidence_fields: list[str] = []
    for field, flag_field, shortfall_field in (
        (
            "setup_layer_confidence",
            "low_setup_confidence_regime",
            "setup_shortfall",
        ),
        (
            _CONTEXT_BIAS_FAMILY_FIELD,
            "low_context_bias_family_regime",
            "context_bias_family_shortfall",
        ),
        (
            "selected_strategy_confidence",
            "low_selected_strategy_confidence_regime",
            "selected_strategy_shortfall",
        ),
    ):
        confidence = _to_float(prepared.get(field), default=None)
        is_low = bool(
            confidence is not None
            and _LOW_CONFIDENCE_THRESHOLD is not None
            and confidence <= _LOW_CONFIDENCE_THRESHOLD
        )
        prepared[flag_field] = is_low
        prepared[shortfall_field] = _shortfall_for_threshold(
            value=confidence,
            threshold=_LOW_CONFIDENCE_THRESHOLD,
        )
        if is_low:
            low_confidence_fields.append(field)

    prepared["low_confidence_surface_count"] = len(low_confidence_fields)
    prepared["exact_low_confidence_fields"] = tuple(low_confidence_fields)
    return prepared


def _context_bias_family_mean_from_row(row: dict[str, Any]) -> float | None:
    context_confidence = _to_float(row.get("context_layer_confidence"), default=None)
    bias_confidence = _to_float(row.get("bias_layer_confidence"), default=None)
    available_values = [
        value for value in (context_confidence, bias_confidence) if value is not None
    ]
    if not available_values:
        return None
    return round(sum(available_values) / len(available_values), 6)


def _shortfall_for_threshold(
    *,
    value: float | None,
    threshold: float | None,
) -> float | None:
    if value is None or threshold is None:
        return None
    return round(max(0.0, threshold - value), 6)


def _build_pocket_row_sets(
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible_rows = [
        row
        for row in comparison_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=None) is not None
    ]
    pocket_rows = [
        row
        for row in eligible_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=99.0)
        <= _POCKET_MAX_THRESHOLD
    ]
    outside_pocket_rows = [
        row
        for row in eligible_rows
        if _to_float(row.get(_RULE_ENGINE_CONFIDENCE_FIELD), default=99.0)
        > _POCKET_MAX_THRESHOLD
    ]
    pocket_preserved_rows = _comparison_group_rows(
        pocket_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    pocket_collapsed_rows = _comparison_group_rows(
        pocket_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )
    preserved_outside_pocket_rows = _comparison_group_rows(
        outside_pocket_rows,
        _COMPARISON_GROUP_PRESERVED,
    )
    collapsed_outside_pocket_rows = _comparison_group_rows(
        outside_pocket_rows,
        _COMPARISON_GROUP_COLLAPSED,
    )
    dense_bucket_rows_by_label = {
        label: _rows_in_band(
            rows=eligible_rows,
            field=_RULE_ENGINE_CONFIDENCE_FIELD,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
        for label, lower_bound_exclusive, upper_bound_inclusive in (
            _DENSE_BUCKET_DEFINITIONS
        )
    }
    collapsed_dense_bucket_rows_by_label = {
        label: _rows_in_band(
            rows=pocket_collapsed_rows,
            field=_RULE_ENGINE_CONFIDENCE_FIELD,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
        for label, lower_bound_exclusive, upper_bound_inclusive in (
            _DENSE_BUCKET_DEFINITIONS
        )
    }
    preserved_dense_bucket_rows_by_label = {
        label: _rows_in_band(
            rows=pocket_preserved_rows,
            field=_RULE_ENGINE_CONFIDENCE_FIELD,
            lower_bound_exclusive=lower_bound_exclusive,
            upper_bound_inclusive=upper_bound_inclusive,
        )
        for label, lower_bound_exclusive, upper_bound_inclusive in (
            _DENSE_BUCKET_DEFINITIONS
        )
    }
    return {
        "eligible_rows": eligible_rows,
        "missing_rule_engine_confidence_row_count": len(comparison_rows)
        - len(eligible_rows),
        "pocket_rows": pocket_rows,
        "outside_pocket_rows": outside_pocket_rows,
        "pocket_preserved_rows": pocket_preserved_rows,
        "pocket_collapsed_rows": pocket_collapsed_rows,
        "preserved_outside_pocket_rows": preserved_outside_pocket_rows,
        "collapsed_outside_pocket_rows": collapsed_outside_pocket_rows,
        "dense_bucket_rows_by_label": dense_bucket_rows_by_label,
        "collapsed_dense_bucket_rows_by_label": collapsed_dense_bucket_rows_by_label,
        "preserved_dense_bucket_rows_by_label": preserved_dense_bucket_rows_by_label,
    }


def _build_group_field_row(
    *,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    left_group_label: str,
    right_group_label: str,
    field_spec: dict[str, Any],
) -> dict[str, Any]:
    field = str(field_spec["field"])
    left_summary = final_split_module._numeric_field_summary(left_rows, field)
    right_summary = final_split_module._numeric_field_summary(right_rows, field)
    left_median = _to_float(left_summary.get("median"), default=None)
    right_median = _to_float(right_summary.get("median"), default=None)
    orientation = str(field_spec.get("orientation") or "lower_is_weaker")
    return {
        "field": field,
        "field_label": field_spec.get("field_label", field),
        "family": field_spec.get("family"),
        "orientation": orientation,
        "left_group_label": left_group_label,
        "right_group_label": right_group_label,
        "left_group_summary": left_summary,
        "right_group_summary": right_summary,
        "comparison_status": _group_median_comparison_status(
            left_summary=left_summary,
            right_summary=right_summary,
        ),
        "left_minus_right_median": _difference_or_none(left_median, right_median),
        "left_group_more_severe_by_orientation": _left_group_more_severe(
            orientation=orientation,
            left_median=left_median,
            right_median=right_median,
        ),
        "left_group_severity_gap": _severity_gap_in_left_direction(
            orientation=orientation,
            left_median=left_median,
            right_median=right_median,
        ),
        "pairwise_relation": _pairwise_relation_summary(
            left_rows=left_rows,
            right_rows=right_rows,
            field=field,
        ),
        "range_overlap": _range_overlap_summary(
            left_summary=left_summary,
            right_summary=right_summary,
        ),
    }


def _build_dense_bucket_row(
    *,
    bucket_label: str,
    collapsed_rows: Sequence[dict[str, Any]],
    preserved_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_row_count = len(collapsed_rows) + len(preserved_rows)
    return {
        "bucket_label": bucket_label,
        "row_count": len(collapsed_rows),
        "collapsed_origin_row_count": len(collapsed_rows),
        "total_row_count": total_row_count,
        "preserved_row_count": len(preserved_rows),
        "preserved_contamination_rate": _safe_ratio(
            len(preserved_rows),
            total_row_count,
        ),
        "comparison_population": "collapsed_only_origin_rows",
        "rule_engine_confidence_summary": final_split_module._numeric_field_summary(
            collapsed_rows,
            _RULE_ENGINE_CONFIDENCE_FIELD,
        ),
        "weighted_mean_setup_emphasis_summary": (
            final_split_module._numeric_field_summary(collapsed_rows, _BASELINE_NAME)
        ),
        "residual_summary": final_split_module._numeric_field_summary(
            collapsed_rows,
            _RESIDUAL_FIELD,
        ),
        "low_surface_count_distribution": [
            _low_surface_count_distribution_row(
                rows=collapsed_rows,
                low_surface_count=value,
            )
            for value in range(4)
        ],
        "dominant_exact_low_confidence_signature": (
            _dominant_exact_low_confidence_signature(collapsed_rows)
        ),
    }


def _rows_with_complete_joint_signature(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("low_confidence_surface_count") is not None
        and isinstance(row.get("exact_low_confidence_fields"), tuple)
    ]


def _comparison_row_for_field(
    *,
    field_comparison: Any,
    field: str,
) -> dict[str, Any]:
    comparison_dict = _safe_dict(field_comparison)
    return next(
        (
            _safe_dict(row)
            for row in _safe_list(comparison_dict.get("field_comparisons"))
            if str(_safe_dict(row).get("field") or "") == field
        ),
        {},
    )


def _low_surface_count_profile(
    *,
    pocket_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    low_surface_count: int,
) -> dict[str, Any]:
    pocket_row_count = sum(
        1
        for row in pocket_rows
        if int(row.get("low_confidence_surface_count", -1) or -1) == low_surface_count
    )
    reference_row_count = sum(
        1
        for row in reference_rows
        if int(row.get("low_confidence_surface_count", -1) or -1) == low_surface_count
    )
    pocket_rate = _safe_ratio(pocket_row_count, len(pocket_rows))
    reference_rate = _safe_ratio(reference_row_count, len(reference_rows))
    return {
        "low_surface_count": low_surface_count,
        "pocket_row_count": pocket_row_count,
        "reference_row_count": reference_row_count,
        "pocket_rate": pocket_rate,
        "reference_rate": reference_rate,
        "rate_gap": round(pocket_rate - reference_rate, 6),
    }


def _combined_low_surface_profile(
    *,
    pocket_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    minimum_low_surface_count: int,
) -> dict[str, Any]:
    pocket_row_count = sum(
        1
        for row in pocket_rows
        if int(row.get("low_confidence_surface_count", -1) or -1)
        >= minimum_low_surface_count
    )
    reference_row_count = sum(
        1
        for row in reference_rows
        if int(row.get("low_confidence_surface_count", -1) or -1)
        >= minimum_low_surface_count
    )
    pocket_rate = _safe_ratio(pocket_row_count, len(pocket_rows))
    reference_rate = _safe_ratio(reference_row_count, len(reference_rows))
    return {
        "minimum_low_surface_count": minimum_low_surface_count,
        "pocket_row_count": pocket_row_count,
        "reference_row_count": reference_row_count,
        "pocket_rate": pocket_rate,
        "reference_rate": reference_rate,
        "rate_gap": round(pocket_rate - reference_rate, 6),
    }


def _exact_low_confidence_signature_comparison_rows(
    *,
    pocket_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pocket_counter = Counter(
        tuple(row.get("exact_low_confidence_fields") or ()) for row in pocket_rows
    )
    reference_counter = Counter(
        tuple(row.get("exact_low_confidence_fields") or ()) for row in reference_rows
    )
    all_signatures = sorted(
        set(pocket_counter).union(reference_counter),
        key=lambda fields: (
            -len(fields),
            _signature_label(fields),
        ),
    )
    rows = []
    for fields in all_signatures:
        pocket_row_count = int(pocket_counter.get(fields, 0) or 0)
        reference_row_count = int(reference_counter.get(fields, 0) or 0)
        pocket_rate = _safe_ratio(pocket_row_count, len(pocket_rows))
        reference_rate = _safe_ratio(reference_row_count, len(reference_rows))
        rows.append(
            {
                "exact_low_confidence_fields": list(fields),
                "signature_label": _signature_label(fields),
                "low_surface_count": len(fields),
                "pocket_row_count": pocket_row_count,
                "reference_row_count": reference_row_count,
                "pocket_rate": pocket_rate,
                "reference_rate": reference_rate,
                "rate_gap": round(pocket_rate - reference_rate, 6),
            }
        )
    rows.sort(
        key=lambda row: (
            -_to_float(row.get("rate_gap"), default=0.0),
            -int(row.get("low_surface_count", 0) or 0),
            -_to_float(row.get("pocket_rate"), default=0.0),
            str(row.get("signature_label") or ""),
        )
    )
    return rows


def _dominant_exact_low_confidence_signature(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows_with_signature = [
        tuple(row.get("exact_low_confidence_fields") or ())
        for row in rows
        if isinstance(row.get("exact_low_confidence_fields"), tuple)
    ]
    if not rows_with_signature:
        return {}
    counter = Counter(rows_with_signature)
    fields, row_count = sorted(
        counter.items(),
        key=lambda item: (-item[1], -len(item[0]), _signature_label(item[0])),
    )[0]
    return {
        "exact_low_confidence_fields": list(fields),
        "signature_label": _signature_label(fields),
        "low_surface_count": len(fields),
        "row_count": row_count,
        "share": _safe_ratio(row_count, len(rows_with_signature)),
    }


def _low_surface_count_distribution_row(
    *,
    rows: Sequence[dict[str, Any]],
    low_surface_count: int,
) -> dict[str, Any]:
    qualifying_rows = [
        row
        for row in rows
        if int(row.get("low_confidence_surface_count", -1) or -1) == low_surface_count
    ]
    return {
        "low_surface_count": low_surface_count,
        "row_count": len(qualifying_rows),
        "share": _safe_ratio(len(qualifying_rows), len(rows)),
    }


def _group_support_status(
    *,
    left_row_count: int,
    right_row_count: int,
    min_support_rows: int,
) -> str:
    if left_row_count <= 0 or right_row_count <= 0:
        return "insufficient_data"
    if left_row_count >= min_support_rows and right_row_count >= min_support_rows:
        return "supported"
    return "limited_support"


def _group_median_comparison_status(
    *,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> str:
    left_present = int(left_summary.get("present_row_count", 0) or 0)
    right_present = int(right_summary.get("present_row_count", 0) or 0)
    if left_present <= 0 and right_present <= 0:
        return "all_missing"
    if left_present > 0 and right_present <= 0:
        return "missing_on_right_only"
    if right_present > 0 and left_present <= 0:
        return "missing_on_left_only"
    left_median = _to_float(left_summary.get("median"), default=0.0)
    right_median = _to_float(right_summary.get("median"), default=0.0)
    if left_median < right_median:
        return "lower_on_left"
    if left_median > right_median:
        return "higher_on_left"
    return "no_clear_separation"


def _left_group_more_severe(
    *,
    orientation: str,
    left_median: float | None,
    right_median: float | None,
) -> bool | None:
    if left_median is None or right_median is None:
        return None
    if orientation in {"lower_is_weaker", "more_negative_is_weaker"}:
        return bool(left_median < right_median)
    if orientation == "higher_is_weaker":
        return bool(left_median > right_median)
    return None


def _severity_gap_in_left_direction(
    *,
    orientation: str,
    left_median: float | None,
    right_median: float | None,
) -> float | None:
    if left_median is None or right_median is None:
        return None
    if orientation in {"lower_is_weaker", "more_negative_is_weaker"}:
        return round(right_median - left_median, 6)
    if orientation == "higher_is_weaker":
        return round(left_median - right_median, 6)
    return None


def _pairwise_relation_summary(
    *,
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    left_values = [
        float(row[field])
        for row in left_rows
        if _to_float(row.get(field), default=None) is not None
    ]
    right_values = [
        float(row[field])
        for row in right_rows
        if _to_float(row.get(field), default=None) is not None
    ]
    if not left_values or not right_values:
        return {
            "support_status": "insufficient_data",
            "comparable_pair_count": 0,
            "left_lower_pair_count": 0,
            "left_greater_pair_count": 0,
            "tie_pair_count": 0,
            "left_lower_rate": 0.0,
            "left_greater_rate": 0.0,
            "tie_rate": 0.0,
            "dominant_relation": "incomplete",
        }

    left_lower_pair_count = 0
    left_greater_pair_count = 0
    tie_pair_count = 0
    for left_value in left_values:
        for right_value in right_values:
            if left_value < right_value:
                left_lower_pair_count += 1
            elif left_value > right_value:
                left_greater_pair_count += 1
            else:
                tie_pair_count += 1
    comparable_pair_count = len(left_values) * len(right_values)
    left_lower_rate = _safe_ratio(left_lower_pair_count, comparable_pair_count)
    left_greater_rate = _safe_ratio(left_greater_pair_count, comparable_pair_count)
    tie_rate = _safe_ratio(tie_pair_count, comparable_pair_count)
    if left_lower_rate == 1.0:
        dominant_relation = "left_strictly_lower"
    elif left_greater_rate == 1.0:
        dominant_relation = "left_strictly_higher"
    elif left_lower_rate >= 0.75:
        dominant_relation = "left_mostly_lower"
    elif left_greater_rate >= 0.75:
        dominant_relation = "left_mostly_higher"
    else:
        dominant_relation = "mixed"
    return {
        "support_status": "supported",
        "comparable_pair_count": comparable_pair_count,
        "left_lower_pair_count": left_lower_pair_count,
        "left_greater_pair_count": left_greater_pair_count,
        "tie_pair_count": tie_pair_count,
        "left_lower_rate": left_lower_rate,
        "left_greater_rate": left_greater_rate,
        "tie_rate": tie_rate,
        "dominant_relation": dominant_relation,
    }


def _range_overlap_summary(
    *,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> dict[str, Any]:
    left_min = _to_float(left_summary.get("min"), default=None)
    left_max = _to_float(left_summary.get("max"), default=None)
    right_min = _to_float(right_summary.get("min"), default=None)
    right_max = _to_float(right_summary.get("max"), default=None)
    if None in {left_min, left_max, right_min, right_max}:
        return {
            "support_status": "insufficient_data",
            "ranges_overlap": False,
            "range_order": "incomplete",
        }
    if left_max < right_min:
        range_order = "left_below_right"
        ranges_overlap = False
    elif right_max < left_min:
        range_order = "right_below_left"
        ranges_overlap = False
    else:
        range_order = "overlap_or_touching"
        ranges_overlap = True
    return {
        "support_status": "supported",
        "ranges_overlap": ranges_overlap,
        "range_order": range_order,
        "left_min": left_min,
        "left_max": left_max,
        "right_min": right_min,
        "right_max": right_max,
    }


def _low_rate_for_field(
    *,
    rows: Sequence[dict[str, Any]],
    field: str,
    threshold: float | None,
) -> float:
    if threshold is None:
        return 0.0
    values = [
        float(row[field])
        for row in rows
        if _to_float(row.get(field), default=None) is not None
    ]
    return _safe_ratio(sum(1 for value in values if value <= threshold), len(values))


def _rate_for_boolean_field(
    *,
    rows: Sequence[dict[str, Any]],
    field: str,
    expected_value: bool,
) -> float:
    values = [
        bool(row[field])
        for row in rows
        if isinstance(row.get(field), bool)
    ]
    return _safe_ratio(
        sum(1 for value in values if value is expected_value),
        len(values),
    )


def _trigger_negative_control_status(
    *,
    family_rows: Sequence[dict[str, Any]],
) -> str:
    trigger_row = _safe_dict(
        next(
            (
                row
                for row in family_rows
                if str(_safe_dict(row).get("family") or "")
                == "trigger_negative_control"
            ),
            {},
        )
    )
    if not trigger_row:
        return "trigger_negative_control_missing"

    trigger_gap = _to_float(
        trigger_row.get("severity_gap_in_left_direction"),
        default=0.0,
    )
    best_non_trigger_gap = max(
        (
            _to_float(
                _safe_dict(row).get("severity_gap_in_left_direction"),
                default=0.0,
            )
            for row in family_rows
            if str(_safe_dict(row).get("family") or "")
            != "trigger_negative_control"
        ),
        default=0.0,
    )
    if trigger_gap <= 0.0:
        return "trigger_remains_negative_control"
    if best_non_trigger_gap <= 0.0:
        return "trigger_unopposed"
    if trigger_gap <= best_non_trigger_gap * 0.50:
        return "trigger_remains_negative_control"
    if trigger_gap < best_non_trigger_gap:
        return "trigger_moves_but_remains_secondary"
    return "trigger_competes_with_non_trigger"


def _dense_bucket_piecewise_status(
    *,
    left_row_count: int,
    right_row_count: int,
    residual_median_difference_left_minus_right: float | None,
    material_contributor_shift_count: int,
    dominant_signature_shift: bool,
) -> str:
    if left_row_count <= 0 or right_row_count <= 0:
        return "dense_bucket_comparison_unavailable"
    residual_shift_visible = (
        _safe_abs_float(residual_median_difference_left_minus_right) or 0.0
    ) >= _PIECEWISE_RESIDUAL_MEDIAN_DIFF_THRESHOLD
    if (
        left_row_count >= _MIN_DENSE_BUCKET_SUPPORT_ROWS
        and right_row_count >= _MIN_DENSE_BUCKET_SUPPORT_ROWS
        and residual_shift_visible
        and (material_contributor_shift_count >= 2 or dominant_signature_shift)
    ):
        return "bucket_distinct_piecewise_regimes"
    if residual_shift_visible or material_contributor_shift_count > 0 or dominant_signature_shift:
        return "same_regime_with_ordered_severity"
    return "same_regime_or_gradual_scale_change"


def _stronger_downward_residual_bucket_label(
    *,
    left_bucket_label: str,
    right_bucket_label: str,
    left_residual_median: float | None,
    right_residual_median: float | None,
) -> str | None:
    if left_residual_median is None or right_residual_median is None:
        return None
    if abs(left_residual_median - right_residual_median) < _COMPARABLE_RESIDUAL_MEDIAN_DELTA:
        return "comparable"
    if left_residual_median < right_residual_median:
        return left_bucket_label
    return right_bucket_label


def _signature_label(fields: Sequence[str]) -> str:
    if not fields:
        return "no_low_confidence_fields"
    return " + ".join(str(field) for field in fields)


def _left_group_severity_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    row_dict = _safe_dict(row)
    left_more_severe = row_dict.get("left_group_more_severe_by_orientation")
    return (
        0 if left_more_severe is True else 1,
        -_to_float(row_dict.get("left_group_severity_gap"), default=0.0),
        _FIELD_ORDER.get(str(row_dict.get("field") or ""), 99),
    )


def _family_severity_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    row_dict = _safe_dict(row)
    return (
        -_to_float(row_dict.get("severity_gap_in_left_direction"), default=0.0),
        _FAMILY_ORDER.get(str(row_dict.get("family") or ""), 99),
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    pocket_summary: dict[str, Any],
    interpretation: dict[str, Any],
    dense_bucket_comparison: dict[str, Any],
    joint_weakness_signature: dict[str, Any],
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
            "No rows reached the final fully aligned plus rule-bias-aligned slice, so "
            "compressed low-band origin diagnosis is unsupported."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final fully aligned plus rule-bias-aligned slice exists, but one "
            "side of the preserved-vs-collapsed comparison is missing, so compressed "
            "low-band origin diagnosis remains incomplete."
        )
    if interpretation_status == "comparison_unsupported":
        return (
            "The final slice exists, but pocket-origin claims stay withheld because "
            "the family-level preserved-vs-collapsed comparison does not clear the "
            "normal support threshold."
        )
    if interpretation_status == (
        "compressed_low_band_origin_joint_weakness_with_piecewise_dense_"
        "buckets_supported"
    ):
        return (
            "The compressed low-confidence pocket is best described as a joint "
            "weakness origin across multiple contributor families, with the two dense "
            "buckets behaving like piecewise sub-regimes rather than one homogeneous "
            "cluster."
        )
    if interpretation_status == "compressed_low_band_origin_joint_weakness_supported":
        return (
            "The compressed low-confidence pocket is best described as a joint "
            "weakness origin across multiple contributor families, while the dense "
            "bucket split remains secondary."
        )
    if interpretation_status == (
        "compressed_low_band_origin_piecewise_dense_buckets_supported_"
        "joint_signature_mixed"
    ):
        return (
            "The dense buckets show a meaningful piecewise split in residual origin, "
            "but the broader joint weakness signature remains mixed."
        )
    if interpretation_status == "compressed_low_band_origin_joint_weakness_leaning":
        return (
            "The compressed pocket still leans toward a joint weakness origin, but "
            "the margin over the strongest single-family helper view remains "
            "conservative."
        )
    if interpretation_status == (
        "compressed_low_band_origin_single_family_helper_leaning"
    ):
        return (
            "A single carried-forward family-specific helper view remains about as "
            "sharp as the broader joint signature, so the report does not promote a "
            "uniquely multi-family origin claim."
        )
    return (
        "The fixed low-band pocket is real, but the current origin evidence remains "
        "mixed: pocket rows share weakness and residual compression, yet the cleanest "
        "joint signature or dense-bucket regime split is not fully settled."
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
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _difference_or_none(left: Any, right: Any) -> float | None:
    left_float = _to_float(left, default=None)
    right_float = _to_float(right, default=None)
    if left_float is None or right_float is None:
        return None
    return round(left_float - right_float, 6)


def _safe_abs_float(value: Any) -> float | None:
    numeric_value = _to_float(value, default=None)
    if numeric_value is None:
        return None
    return round(abs(numeric_value), 6)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return residual_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return residual_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return residual_module._safe_list(value)


if __name__ == "__main__":
    main()
