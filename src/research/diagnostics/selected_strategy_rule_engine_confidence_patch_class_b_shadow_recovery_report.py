from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report as patch_class_a_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report as signature_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Patch Class B Shadow Recovery Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = patch_class_a_module.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = patch_class_a_module.DEFAULT_OUTPUT_DIR
DEFAULT_MIN_SYMBOL_SUPPORT = patch_class_a_module.DEFAULT_MIN_SYMBOL_SUPPORT

DiagnosisConfiguration = patch_class_a_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = patch_class_a_module.DEFAULT_CONFIGURATIONS

_COMPARISON_GROUP_PRESERVED = patch_class_a_module._COMPARISON_GROUP_PRESERVED
_COMPARISON_GROUP_COLLAPSED = patch_class_a_module._COMPARISON_GROUP_COLLAPSED
_RULE_ENGINE_CONFIDENCE_FIELD = patch_class_a_module._RULE_ENGINE_CONFIDENCE_FIELD
_BASELINE_NAME = patch_class_a_module._BASELINE_NAME
_BASELINE_LABEL = patch_class_a_module._BASELINE_LABEL
_BASELINE_FORMULA = patch_class_a_module._BASELINE_FORMULA
_RESIDUAL_FIELD = patch_class_a_module._RESIDUAL_FIELD
_CONTEXT_BIAS_FAMILY_FIELD = patch_class_a_module._CONTEXT_BIAS_FAMILY_FIELD
_LOW_CONFIDENCE_THRESHOLD = patch_class_a_module._LOW_CONFIDENCE_THRESHOLD
_POCKET_MAX_THRESHOLD = patch_class_a_module._POCKET_MAX_THRESHOLD
_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE = (
    patch_class_a_module._DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
)
_LOW_SURFACE_COUNT_ANCHOR = patch_class_a_module._LOW_SURFACE_COUNT_ANCHOR
_DIRECT_EDGE_SELECTION_AVAILABLE = patch_class_a_module._DIRECT_EDGE_SELECTION_AVAILABLE
_DIRECT_EDGE_SELECTION_UNAVAILABLE = (
    patch_class_a_module._DIRECT_EDGE_SELECTION_UNAVAILABLE
)
_NONTRIVIAL_RESCUE_MIN_ROWS = patch_class_a_module._NONTRIVIAL_RESCUE_MIN_ROWS
_NONTRIVIAL_RESCUE_MIN_RATE = patch_class_a_module._NONTRIVIAL_RESCUE_MIN_RATE
_SOURCE_ROW_INDEX = patch_class_a_module._SOURCE_ROW_INDEX

_PATCH_CLASS_A_CANDIDATE_ID = patch_class_a_module._PATCH_CLASS_A_CANDIDATE_ID
_PATCH_CLASS_A_CANDIDATE_LABEL = patch_class_a_module._PATCH_CLASS_A_CANDIDATE_LABEL
_PATCH_CLASS_B_CANDIDATE_ID = "patch_class_b_shadow_b1"
_PATCH_CLASS_B_CANDIDATE_LABEL = "Patch Class B Shadow B1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only shadow report that applies one narrow "
            "Patch Class B joint setup-vs-context de-compression candidate "
            "inside the already-known compressed pocket."
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
            "Retained for parity with sibling reports. The shadow report reuses "
            "only the final preserved-vs-collapsed slice."
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

    result = run_selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    shadow_summary = _safe_dict(report.get("shadow_summary"))
    comparator_summary = _safe_dict(report.get("patch_class_a_comparator_summary"))
    direct_summary = _safe_dict(report.get("direct_edge_selection_summary"))
    patch_class_a_shadow_summary = _safe_dict(
        comparator_summary.get("patch_class_a_shadow_summary")
    )
    interpretation = _safe_dict(report.get("interpretation"))

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
                "baseline_pocket_row_count": shadow_summary.get(
                    "baseline_pocket_row_count",
                    0,
                ),
                "shadow_pocket_row_count": shadow_summary.get(
                    "shadow_pocket_row_count",
                    0,
                ),
                "rescued_from_pocket_row_count": shadow_summary.get(
                    "rescued_from_pocket_row_count",
                    0,
                ),
                "patch_class_a_rescued_from_pocket_row_count": (
                    patch_class_a_shadow_summary.get("rescued_from_pocket_row_count", 0)
                ),
                "scope_relation_status": comparator_summary.get("scope_relation_status"),
                "joint_selectivity_status": interpretation.get(
                    "joint_selectivity_status"
                ),
                "direct_edge_selection_status": direct_summary.get("status"),
                "baseline_edge_selection_count": direct_summary.get(
                    "baseline_edge_selection_count"
                ),
                "shadow_edge_selection_count": direct_summary.get(
                    "shadow_edge_selection_count"
                ),
                "net_edge_selection_change": direct_summary.get("net_change"),
                "concentration_status": shadow_summary.get("concentration_status"),
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report(
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
    base_summary = _safe_dict(widest_summary.get("summary"))
    shadow_summary = _safe_dict(widest_summary.get("shadow_summary"))
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
        "final_rule_bias_aligned_row_count": base_summary.get(
            "final_rule_bias_aligned_row_count",
            0,
        ),
        "preserved_final_directional_outcome_row_count": base_summary.get(
            "preserved_final_directional_outcome_row_count",
            0,
        ),
        "collapsed_final_hold_outcome_row_count": base_summary.get(
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
        "shadow_patch_candidate": _patch_class_b_candidate_definition(),
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary_row.get("headline"))
            for summary_row in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": base_summary,
        "signature_conditioning_summary": _safe_dict(
            widest_summary.get("signature_conditioning_summary")
        ),
        "setup_secondary_reading": _safe_dict(
            widest_summary.get("setup_secondary_reading")
        ),
        "trigger_negative_control_reading": _safe_dict(
            widest_summary.get("trigger_negative_control_reading")
        ),
        "shadow_summary": shadow_summary,
        "patch_class_a_comparator_summary": _safe_dict(
            widest_summary.get("patch_class_a_comparator_summary")
        ),
        "direct_edge_selection_summary": _safe_dict(
            widest_summary.get("direct_edge_selection_summary")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and does not patch production logic, mapper logic, engine logic, candidate-quality-gate logic, or execution-gate logic.",
            "Patch Class B shadow uplift is applied only when the final fully aligned plus rule-bias-aligned row sits inside the fixed compressed pocket, inside the fixed dominant exact low-confidence signature, belongs to the collapsed outcome class, and has setup_margin > context_bias_shortfall.",
            "The fixed pocket anchor remains actual rule_engine_confidence <= 0.25.",
            "The fixed dominant exact low-confidence signature remains context_bias_family_mean + selected_strategy_confidence with low_surface_count=2.",
            "Only the B1 fixed candidate is evaluated here; A1 is replayed only as a narrow same-slice comparator so the report can test whether B1 remains more selective than setup-only relief.",
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
    base_configuration_summary = signature_module.build_configuration_summary(
        configuration=configuration,
        input_path=input_path,
        effective_input_path=effective_input_path,
        run_output_dir=run_output_dir,
        raw_records=raw_records,
        source_metadata=source_metadata,
        min_symbol_support=min_symbol_support,
    )
    comparison_rows = patch_class_a_module._prepare_comparison_rows(raw_records)
    patch_class_a_shadow_rows = patch_class_a_module._apply_patch_class_a_shadow_candidate(
        comparison_rows
    )
    shadow_rows = _apply_patch_class_b_shadow_candidate(comparison_rows)
    shadow_summary = build_shadow_summary(
        comparison_rows=comparison_rows,
        shadow_rows=shadow_rows,
    )
    patch_class_a_comparator_summary = build_patch_class_a_comparator_summary(
        comparison_rows=comparison_rows,
        patch_class_a_shadow_rows=patch_class_a_shadow_rows,
        shadow_rows=shadow_rows,
    )
    direct_edge_selection_summary = _build_direct_edge_selection_summary(
        raw_records=raw_records,
        shadow_rows=shadow_rows,
        run_output_dir=run_output_dir,
    )
    interpretation = build_interpretation(
        base_configuration_summary=base_configuration_summary,
        shadow_summary=shadow_summary,
        patch_class_a_comparator_summary=patch_class_a_comparator_summary,
        direct_edge_selection_summary=direct_edge_selection_summary,
    )
    limitations = build_limitations(
        base_configuration_summary=base_configuration_summary,
        shadow_summary=shadow_summary,
        patch_class_a_comparator_summary=patch_class_a_comparator_summary,
        direct_edge_selection_summary=direct_edge_selection_summary,
    )

    base_headline = _safe_dict(base_configuration_summary.get("headline"))
    shadow_headline = {
        "display_name": configuration.display_name,
        "latest_window_hours": configuration.latest_window_hours,
        "latest_max_rows": configuration.latest_max_rows,
        "final_rule_bias_aligned_row_count": _safe_dict(
            base_configuration_summary.get("summary")
        ).get("final_rule_bias_aligned_row_count"),
        "baseline_pocket_row_count": shadow_summary.get("baseline_pocket_row_count"),
        "shadow_pocket_row_count": shadow_summary.get("shadow_pocket_row_count"),
        "rescued_from_pocket_row_count": shadow_summary.get(
            "rescued_from_pocket_row_count"
        ),
        "scope_relation_status": patch_class_a_comparator_summary.get(
            "scope_relation_status"
        ),
        "direct_edge_selection_status": direct_edge_selection_summary.get("status"),
        "net_edge_selection_change": direct_edge_selection_summary.get("net_change"),
        "interpretation_status": interpretation.get("interpretation_status"),
    }

    return {
        **base_configuration_summary,
        "headline": {**base_headline, **shadow_headline},
        "shadow_patch_candidate": _patch_class_b_candidate_definition(),
        "shadow_summary": shadow_summary,
        "patch_class_a_comparator_summary": patch_class_a_comparator_summary,
        "direct_edge_selection_summary": direct_edge_selection_summary,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_shadow_summary(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    shadow_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_pocket_rows = [
        row for row in comparison_rows if _row_is_in_pocket(row, use_shadow=False)
    ]
    shadow_pocket_rows = [
        row for row in shadow_rows if _row_is_in_pocket(row, use_shadow=True)
    ]
    eligible_rows = [
        row for row in shadow_rows if bool(row.get("patch_class_b_shadow_eligible"))
    ]
    eligible_pocket_rows = [
        row
        for row in eligible_rows
        if _to_float(row.get("actual_rule_engine_confidence"), default=99.0)
        <= _POCKET_MAX_THRESHOLD
    ]
    changed_rows = [
        row
        for row in shadow_rows
        if _to_float(row.get("shadow_uplift"), default=0.0) > 0.0
    ]
    rescued_rows = [row for row in shadow_rows if bool(row.get("rescued_from_pocket"))]
    rescued_outside_target_class_rows = [
        row for row in rescued_rows if not bool(row.get("patch_class_b_shadow_eligible"))
    ]
    changed_outside_target_class_rows = [
        row
        for row in changed_rows
        if not bool(row.get("patch_class_b_shadow_eligible"))
    ]
    rescued_outcome_class_counts = patch_class_a_module._comparison_group_counts(
        rescued_rows
    )
    rescued_outcome_count_map = {
        str(_safe_dict(item).get("comparison_group") or ""): int(
            _safe_dict(item).get("row_count", 0) or 0
        )
        for item in rescued_outcome_class_counts
    }
    concentration_status = patch_class_a_module._concentration_status(
        changed_outside_target_class_row_count=len(changed_outside_target_class_rows),
        rescued_outside_target_class_row_count=len(rescued_outside_target_class_rows),
        rescued_from_pocket_row_count=len(rescued_rows),
    )

    return {
        "candidate_id": _PATCH_CLASS_B_CANDIDATE_ID,
        "candidate_label": _PATCH_CLASS_B_CANDIDATE_LABEL,
        "low_confidence_threshold": _LOW_CONFIDENCE_THRESHOLD,
        "pocket_threshold": _POCKET_MAX_THRESHOLD,
        "baseline_pocket_row_count": len(baseline_pocket_rows),
        "shadow_pocket_row_count": len(shadow_pocket_rows),
        "pocket_row_count_delta": len(shadow_pocket_rows) - len(baseline_pocket_rows),
        "rescued_from_pocket_row_count": len(rescued_rows),
        "rescued_from_pocket_rate_within_baseline_pocket": _safe_ratio(
            len(rescued_rows),
            len(baseline_pocket_rows),
        ),
        "eligible_row_count": len(eligible_rows),
        "eligible_pocket_row_count": len(eligible_pocket_rows),
        "uplift_applied_row_count": len(changed_rows),
        "uplift_applied_rate_within_eligible_rows": _safe_ratio(
            len(changed_rows),
            len(eligible_rows),
        ),
        "changed_outside_target_class_row_count": len(
            changed_outside_target_class_rows
        ),
        "rescued_outside_target_class_row_count": len(
            rescued_outside_target_class_rows
        ),
        "eligibility_breakdown": _eligibility_breakdown(shadow_rows),
        "rescued_outcome_class_counts": rescued_outcome_class_counts,
        "rescued_collapsed_row_count": rescued_outcome_count_map.get(
            _COMPARISON_GROUP_COLLAPSED,
            0,
        ),
        "rescued_preserved_row_count": rescued_outcome_count_map.get(
            _COMPARISON_GROUP_PRESERVED,
            0,
        ),
        "rescued_row_summaries": _build_rescued_row_summaries(rescued_rows),
        "confidence_shift_summaries": _build_confidence_shift_summaries(rescued_rows),
        "concentration_status": concentration_status,
        "concentration_explanation": patch_class_a_module._concentration_explanation(
            concentration_status=concentration_status,
            rescued_from_pocket_row_count=len(rescued_rows),
            changed_outside_target_class_row_count=len(
                changed_outside_target_class_rows
            ),
        ),
    }


def build_patch_class_a_comparator_summary(
    *,
    comparison_rows: Sequence[dict[str, Any]],
    patch_class_a_shadow_rows: Sequence[dict[str, Any]],
    shadow_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    patch_class_a_shadow_summary = patch_class_a_module.build_shadow_summary(
        comparison_rows=comparison_rows,
        shadow_rows=patch_class_a_shadow_rows,
    )
    patch_class_a_eligible_indices = _source_row_indices(
        patch_class_a_shadow_rows,
        indicator_field="patch_class_a_shadow_eligible",
    )
    patch_class_b_eligible_indices = _source_row_indices(
        shadow_rows,
        indicator_field="patch_class_b_shadow_eligible",
    )
    patch_class_a_rescued_indices = _source_row_indices(
        patch_class_a_shadow_rows,
        indicator_field="rescued_from_pocket",
    )
    patch_class_b_rescued_indices = _source_row_indices(
        shadow_rows,
        indicator_field="rescued_from_pocket",
    )
    scope_relation_status = _scope_relation_status(
        patch_class_a_eligible_indices=patch_class_a_eligible_indices,
        patch_class_b_eligible_indices=patch_class_b_eligible_indices,
        patch_class_a_rescued_indices=patch_class_a_rescued_indices,
        patch_class_b_rescued_indices=patch_class_b_rescued_indices,
    )
    narrower_or_equal = scope_relation_status in {
        "matches_patch_class_a_scope",
        "strict_subset_of_patch_class_a",
    }
    strict_subset = scope_relation_status == "strict_subset_of_patch_class_a"
    rescued_overlap_count = len(
        patch_class_a_rescued_indices & patch_class_b_rescued_indices
    )

    return {
        "patch_class_a_candidate_id": _PATCH_CLASS_A_CANDIDATE_ID,
        "patch_class_a_candidate_label": _PATCH_CLASS_A_CANDIDATE_LABEL,
        "patch_class_a_shadow_summary": patch_class_a_shadow_summary,
        "patch_class_a_eligible_row_count": len(patch_class_a_eligible_indices),
        "patch_class_b_eligible_row_count": len(patch_class_b_eligible_indices),
        "patch_class_a_rescued_from_pocket_row_count": len(
            patch_class_a_rescued_indices
        ),
        "patch_class_b_rescued_from_pocket_row_count": len(
            patch_class_b_rescued_indices
        ),
        "eligible_overlap_row_count": len(
            patch_class_a_eligible_indices & patch_class_b_eligible_indices
        ),
        "rescued_overlap_row_count": rescued_overlap_count,
        "eligible_only_in_patch_class_a_row_count": len(
            patch_class_a_eligible_indices - patch_class_b_eligible_indices
        ),
        "eligible_only_in_patch_class_b_row_count": len(
            patch_class_b_eligible_indices - patch_class_a_eligible_indices
        ),
        "rescued_only_in_patch_class_a_row_count": len(
            patch_class_a_rescued_indices - patch_class_b_rescued_indices
        ),
        "rescued_only_in_patch_class_b_row_count": len(
            patch_class_b_rescued_indices - patch_class_a_rescued_indices
        ),
        "rescue_rate_within_patch_class_a_rescue": _safe_ratio(
            rescued_overlap_count,
            len(patch_class_a_rescued_indices),
        ),
        "scope_relation_status": scope_relation_status,
        "scope_relation_explanation": _scope_relation_explanation(
            scope_relation_status=scope_relation_status
        ),
        "is_narrower_or_equal_than_patch_class_a": narrower_or_equal,
        "strictly_narrower_than_patch_class_a": strict_subset,
        "more_selective_subset_observed": strict_subset
        and len(patch_class_b_rescued_indices) > 0,
    }


def build_interpretation(
    *,
    base_configuration_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    patch_class_a_comparator_summary: dict[str, Any],
    direct_edge_selection_summary: dict[str, Any],
) -> dict[str, Any]:
    base_summary = _safe_dict(base_configuration_summary.get("summary"))
    trigger_reading = _safe_dict(
        base_configuration_summary.get("trigger_negative_control_reading")
    )
    setup_reading = _safe_dict(base_configuration_summary.get("setup_secondary_reading"))

    rescued_count = int(shadow_summary.get("rescued_from_pocket_row_count", 0) or 0)
    eligible_pocket_row_count = int(
        shadow_summary.get("eligible_pocket_row_count", 0) or 0
    )
    nontrivial_rescue_threshold = _nontrivial_rescue_threshold(
        eligible_pocket_row_count
    )

    concentration_status = str(shadow_summary.get("concentration_status") or "")
    direct_status = str(direct_edge_selection_summary.get("status") or "")
    net_change = int(direct_edge_selection_summary.get("net_change", 0) or 0)

    trigger_status = str(
        trigger_reading.get("trigger_negative_control_status") or ""
    )
    trigger_ok = trigger_status in {
        "trigger_remains_negative_control",
        "trigger_moves_but_remains_secondary",
    }

    base_support_status = str(base_summary.get("comparison_support_status") or "")
    base_support_ok = base_support_status == "supported"

    preserved_leakage_ok = _preserved_leakage_status(shadow_summary=shadow_summary)
    direct_edge_selection_supported = (
        direct_status == _DIRECT_EDGE_SELECTION_AVAILABLE and net_change > 0
    )

    nontrivial_pocket_rescue = rescued_count >= nontrivial_rescue_threshold

    joint_selectivity_status = _joint_selectivity_status(
        patch_class_a_comparator_summary=patch_class_a_comparator_summary,
        shadow_summary=shadow_summary,
        direct_edge_selection_summary=direct_edge_selection_summary,
    )

    if not base_support_ok:
        interpretation_status = "patch_class_b_shadow_inconclusive"
    elif (
        nontrivial_pocket_rescue
        and concentration_status == "recovery_tightly_concentrated"
        and trigger_ok
        and preserved_leakage_ok == "no_obvious_preserved_leakage_explosion"
        and joint_selectivity_status
        == "joint_selectivity_supported_with_direct_edge_improvement"
    ):
        interpretation_status = "patch_class_b_shadow_supported"
    elif (
        nontrivial_pocket_rescue
        and concentration_status == "recovery_tightly_concentrated"
        and trigger_ok
        and preserved_leakage_ok == "no_obvious_preserved_leakage_explosion"
        and joint_selectivity_status
        == "joint_selectivity_narrows_rescue_but_direct_edge_unproven"
    ):
        interpretation_status = "patch_class_b_shadow_leaning"
    else:
        interpretation_status = "patch_class_b_shadow_inconclusive"

    facts = [
        (
            "Final slice support: "
            f"{base_support_status} "
            f"(preserved={base_summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            f"collapsed={base_summary.get('collapsed_final_hold_outcome_row_count', 0)})."
        ),
        (
            "Patch Class B gate: final fully aligned + rule-bias-aligned collapsed rows, "
            f"actual rule_engine_confidence <= {_POCKET_MAX_THRESHOLD}, "
            f"exact_low_confidence_fields={_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE}, "
            f"low_confidence_surface_count={_LOW_SURFACE_COUNT_ANCHOR}, trigger not below "
            f"{_LOW_CONFIDENCE_THRESHOLD}, setup present, context_bias_family_mean present, "
            "and setup_margin > context_bias_shortfall."
        ),
        (
            "Pocket rescue: baseline="
            f"{shadow_summary.get('baseline_pocket_row_count', 0)}, "
            "shadow="
            f"{shadow_summary.get('shadow_pocket_row_count', 0)}, "
            "rescued="
            f"{rescued_count}."
        ),
        (
            "Concentration: "
            f"{shadow_summary.get('concentration_status')}; "
            "changed_outside_target_class="
            f"{shadow_summary.get('changed_outside_target_class_row_count', 0)}."
        ),
        (
            "Rescued outcome split: collapsed="
            f"{shadow_summary.get('rescued_collapsed_row_count', 0)}, "
            "preserved="
            f"{shadow_summary.get('rescued_preserved_row_count', 0)}."
        ),
        (
            "Patch Class A comparator: rescue_a1="
            f"{patch_class_a_comparator_summary.get('patch_class_a_rescued_from_pocket_row_count', 0)}, "
            "rescue_b1="
            f"{patch_class_a_comparator_summary.get('patch_class_b_rescued_from_pocket_row_count', 0)}, "
            "scope_relation="
            f"{patch_class_a_comparator_summary.get('scope_relation_status')}."
        ),
        (
            "Joint selectivity status="
            f"{joint_selectivity_status}."
        ),
        (
            "Setup status="
            f"{setup_reading.get('setup_separation_status')}; "
            "trigger status="
            f"{trigger_status}."
        ),
        (
            "Direct edge-selection status="
            f"{direct_status}; net_change={net_change}."
        ),
    ]

    if interpretation_status == "patch_class_b_shadow_supported":
        inference = [
            "The fixed B1 shadow candidate rescues a nontrivial, strictly narrower subset than A1, stays tightly concentrated without preserved leakage, and improves direct edge-selection count."
        ]
    elif interpretation_status == "patch_class_b_shadow_leaning":
        inference = [
            "The fixed B1 shadow candidate rescues a strictly narrower subset than A1 while staying tightly concentrated, but direct edge restoration is still not proven."
        ]
    elif not base_support_ok:
        inference = [
            "The underlying preserved-vs-collapsed slice is not strong enough to support a conservative Patch Class B claim on this run."
        ]
    else:
        inference = [
            "B1 rescue is measurable, but it is not selective enough versus A1 or does not improve direct edge-selection count strongly enough to support a conservative B-class claim."
        ]

    uncertainty = [
        "This artifact is descriptive and replay-only; it does not change production thresholds or production action logic.",
        "Only the B1 fixed candidate is evaluated here; the report does not broaden into a B-class sweep.",
    ]

    if direct_status != _DIRECT_EDGE_SELECTION_AVAILABLE:
        uncertainty.append(
            "Direct edge-selection count could not be exercised safely from the diagnosis path on this run, so the report falls back to pocket-rescue evidence and the A1-vs-B1 comparator."
        )
    elif net_change <= 0:
        uncertainty.append(
            "Direct edge-selection count stayed unchanged, so B1 remains short of a full support reading even if rescue stays concentrated."
        )

    if joint_selectivity_status == "joint_selectivity_matches_patch_class_a_scope":
        uncertainty.append(
            "B1 matches A1 scope on this run, so the result does not demonstrate additional joint selectivity."
        )
    elif joint_selectivity_status == "joint_selectivity_broader_than_patch_class_a":
        uncertainty.append(
            "B1 is broader than A1 on this run, so the joint-selectivity claim remains conservative."
        )
    elif joint_selectivity_status == "joint_selectivity_mixed_vs_patch_class_a":
        uncertainty.append(
            "B1 and A1 overlap, but B1 does not remain a clean narrower subset on this run."
        )

    if preserved_leakage_ok != "no_obvious_preserved_leakage_explosion":
        uncertainty.append(
            "Non-collapsed rescue or another leakage-like pattern is present, so the report does not promote a stronger claim."
        )

    return {
        "interpretation_status": interpretation_status,
        "nontrivial_rescue_threshold": nontrivial_rescue_threshold,
        "facts": facts,
        "inference": inference,
        "uncertainty": uncertainty,
        "direct_edge_selection_status": direct_status,
        "preserved_leakage_status": preserved_leakage_ok,
        "scope_relation_status": patch_class_a_comparator_summary.get(
            "scope_relation_status"
        ),
        "joint_selectivity_status": joint_selectivity_status,
    }


def build_limitations(
    *,
    base_configuration_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    patch_class_a_comparator_summary: dict[str, Any],
    direct_edge_selection_summary: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact stays inside the already-established compressed pocket diagnosis and does not reopen earlier diagnosis branches.",
        "Patch Class B applies only one fixed B1 candidate and does not run a parameter sweep, a context-band search, a signature search, or a broader family sweep.",
        "Only rule_engine_confidence is shadowed; mapper, engine, candidate-quality gate, execution gate, and production defaults remain untouched.",
    ]
    if shadow_summary.get("rescued_from_pocket_row_count", 0) <= 0:
        limitations.append(
            "No pocket rows were rescued above the pocket threshold in this configuration."
        )
    if shadow_summary.get("rescued_preserved_row_count", 0) > 0:
        limitations.append(
            "At least one non-collapsed row was rescued, so the reading stays conservative."
        )
    if shadow_summary.get("changed_outside_target_class_row_count", 0) > 0:
        limitations.append(
            "At least one changed row fell outside the intended target class."
        )
    if not bool(
        patch_class_a_comparator_summary.get("is_narrower_or_equal_than_patch_class_a")
    ):
        limitations.append(
            "B1 does not remain narrower or equal than A1 on this run, so the joint-selectivity claim remains conservative."
        )
    if (
        direct_edge_selection_summary.get("status")
        != _DIRECT_EDGE_SELECTION_AVAILABLE
    ):
        limitations.append(
            "Direct edge-selection count was unavailable, so the report is limited to pocket rescue and A1-vs-B1 comparator evidence."
        )
    if _safe_dict(base_configuration_summary.get("summary")).get(
        "comparison_support_status"
    ) != "supported":
        limitations.append(
            "The base preserved-vs-collapsed slice is below the family's normal support threshold, so interpretation remains conservative."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    interpretation = _safe_dict(widest.get("interpretation"))
    shadow_summary = _safe_dict(widest.get("shadow_summary"))
    direct_summary = _safe_dict(widest.get("direct_edge_selection_summary"))
    comparator_summary = _safe_dict(widest.get("patch_class_a_comparator_summary"))
    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": _safe_dict(widest.get("summary")),
        "shadow_patch_candidate": _patch_class_b_candidate_definition(),
        "shadow_summary": shadow_summary,
        "patch_class_a_comparator_summary": comparator_summary,
        "direct_edge_selection_summary": direct_summary,
        "interpretation_status": interpretation.get("interpretation_status"),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            interpretation=interpretation,
            shadow_summary=shadow_summary,
            patch_class_a_comparator_summary=comparator_summary,
            direct_edge_selection_summary=direct_summary,
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
        shadow_summary = _safe_dict(_safe_dict(summary).get("shadow_summary"))
        comparator_summary = _safe_dict(
            _safe_dict(summary).get("patch_class_a_comparator_summary")
        )
        direct_summary = _safe_dict(
            _safe_dict(summary).get("direct_edge_selection_summary")
        )
        interpretation = _safe_dict(_safe_dict(summary).get("interpretation"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(
            "- baseline_pocket_row_count: "
            f"{shadow_summary.get('baseline_pocket_row_count', 0)}"
        )
        lines.append(
            "- shadow_pocket_row_count: "
            f"{shadow_summary.get('shadow_pocket_row_count', 0)}"
        )
        lines.append(
            "- rescued_from_pocket_row_count: "
            f"{shadow_summary.get('rescued_from_pocket_row_count', 0)}"
        )
        lines.append(
            "- patch_class_a_rescued_from_pocket_row_count: "
            f"{comparator_summary.get('patch_class_a_rescued_from_pocket_row_count', 0)}"
        )
        lines.append(
            "- scope_relation_status: "
            f"{comparator_summary.get('scope_relation_status')}"
        )
        lines.append(
            "- rescued_preserved_row_count: "
            f"{shadow_summary.get('rescued_preserved_row_count', 0)}"
        )
        lines.append(
            "- concentration_status: "
            f"{shadow_summary.get('concentration_status')}"
        )
        lines.append(
            "- direct_edge_selection_status: "
            f"{direct_summary.get('status')}"
        )
        if direct_summary.get("status") == _DIRECT_EDGE_SELECTION_AVAILABLE:
            lines.append(
                "- direct_edge_selection_counts: "
                f"baseline={direct_summary.get('baseline_edge_selection_count')}, "
                f"shadow={direct_summary.get('shadow_edge_selection_count')}, "
                f"net_change={direct_summary.get('net_change')}"
            )
        lines.append(
            "- joint_selectivity_status: "
            f"{interpretation.get('joint_selectivity_status')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{interpretation.get('interpretation_status')}"
        )
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append(
        "- interpretation_status: "
        f"{final_assessment.get('interpretation_status')}"
    )
    lines.append(
        "- overall_conclusion: "
        f"{final_assessment.get('overall_conclusion')}"
    )
    lines.append("")

    lines.append("## Evidence")
    lines.append("")
    for item in _safe_list(_safe_dict(report.get("interpretation")).get("facts")):
        lines.append(f"- {item}")
    if not _safe_list(_safe_dict(report.get("interpretation")).get("facts")):
        lines.append("- none")
    lines.append("")

    lines.append("## Remaining Uncertainty")
    lines.append("")
    for item in _safe_list(_safe_dict(report.get("interpretation")).get("uncertainty")):
        lines.append(f"- {item}")
    if not _safe_list(_safe_dict(report.get("interpretation")).get("uncertainty")):
        lines.append("- none")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output = _resolve_path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output / REPORT_JSON_NAME
    md_path = resolved_output / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def _apply_patch_class_b_shadow_candidate(
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    shadow_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        actual_rule_engine_confidence = _to_float(
            row.get(_RULE_ENGINE_CONFIDENCE_FIELD),
            default=None,
        )
        actual_weighted_aggregate_residual = _to_float(
            row.get(_RESIDUAL_FIELD),
            default=None,
        )
        (
            setup_confidence,
            context_bias_family_mean,
            setup_margin,
            context_bias_shortfall,
            joint_net_margin,
        ) = _build_margin_components(row)
        block_reason = _patch_class_b_block_reason(row)
        eligible = block_reason is None
        shadow_uplift = 0.0
        shadow_rule_engine_confidence = actual_rule_engine_confidence
        if (
            eligible
            and actual_rule_engine_confidence is not None
            and setup_confidence is not None
            and context_bias_family_mean is not None
            and joint_net_margin is not None
        ):
            shadow_uplift = round(min(0.12, 2.0 * joint_net_margin), 6)
            shadow_rule_engine_confidence = round(
                min(1.0, actual_rule_engine_confidence + shadow_uplift),
                6,
            )

        shadow_input = dict(row)
        shadow_input[_RULE_ENGINE_CONFIDENCE_FIELD] = shadow_rule_engine_confidence
        rebuilt_shadow_row = patch_class_a_module.origin_module._prepare_origin_residual_row(
            patch_class_a_module.residual_module.build_residual_row(shadow_input)
        )
        rebuilt_shadow_row = _carry_source_metadata(rebuilt_shadow_row, row)
        rebuilt_shadow_row["actual_rule_engine_confidence"] = (
            actual_rule_engine_confidence
        )
        rebuilt_shadow_row["actual_weighted_aggregate_residual"] = (
            actual_weighted_aggregate_residual
        )
        rebuilt_shadow_row["shadow_rule_engine_confidence"] = (
            shadow_rule_engine_confidence
        )
        rebuilt_shadow_row["shadow_weighted_aggregate_residual"] = _to_float(
            rebuilt_shadow_row.get(_RESIDUAL_FIELD),
            default=None,
        )
        rebuilt_shadow_row["setup_margin"] = 0.0 if setup_margin is None else setup_margin
        rebuilt_shadow_row["context_bias_shortfall"] = (
            0.0 if context_bias_shortfall is None else context_bias_shortfall
        )
        rebuilt_shadow_row["joint_net_margin"] = (
            0.0 if joint_net_margin is None else joint_net_margin
        )
        rebuilt_shadow_row["shadow_uplift"] = shadow_uplift
        rebuilt_shadow_row["patch_class_b_shadow_eligible"] = eligible
        rebuilt_shadow_row["patch_class_b_shadow_applied"] = shadow_uplift > 0.0
        rebuilt_shadow_row["patch_class_b_shadow_block_reason"] = block_reason
        rebuilt_shadow_row["rescued_from_pocket"] = bool(
            actual_rule_engine_confidence is not None
            and actual_rule_engine_confidence <= _POCKET_MAX_THRESHOLD
            and shadow_rule_engine_confidence is not None
            and shadow_rule_engine_confidence > _POCKET_MAX_THRESHOLD
        )
        shadow_rows.append(rebuilt_shadow_row)
    return shadow_rows


def _build_margin_components(
    row: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    setup_confidence = _to_float(row.get("setup_layer_confidence"), default=None)
    context_bias_family_mean = _to_float(
        row.get(_CONTEXT_BIAS_FAMILY_FIELD),
        default=None,
    )
    if setup_confidence is None:
        setup_margin = None
    else:
        setup_margin = round(
            max(0.0, setup_confidence - float(_LOW_CONFIDENCE_THRESHOLD)),
            6,
        )
    if context_bias_family_mean is None:
        context_bias_shortfall = None
    else:
        context_bias_shortfall = round(
            max(0.0, float(_LOW_CONFIDENCE_THRESHOLD) - context_bias_family_mean),
            6,
        )
    if setup_margin is None or context_bias_shortfall is None:
        joint_net_margin = None
    else:
        joint_net_margin = round(
            max(0.0, setup_margin - context_bias_shortfall),
            6,
        )
    return (
        setup_confidence,
        context_bias_family_mean,
        setup_margin,
        context_bias_shortfall,
        joint_net_margin,
    )


def _patch_class_b_block_reason(row: dict[str, Any]) -> str | None:
    if str(row.get("comparison_group") or "") != _COMPARISON_GROUP_COLLAPSED:
        return "comparison_group_not_collapsed"
    actual_rule_engine_confidence = _to_float(
        row.get(_RULE_ENGINE_CONFIDENCE_FIELD),
        default=None,
    )
    if actual_rule_engine_confidence is None:
        return "missing_rule_engine_confidence"
    if actual_rule_engine_confidence > _POCKET_MAX_THRESHOLD:
        return "outside_pocket"
    if tuple(row.get("exact_low_confidence_fields") or ()) != (
        _DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
    ):
        return "signature_mismatch"
    if int(row.get("low_confidence_surface_count", -1) or -1) != (
        _LOW_SURFACE_COUNT_ANCHOR
    ):
        return "low_surface_count_mismatch"
    trigger_layer_confidence = _to_float(
        row.get("trigger_layer_confidence"),
        default=None,
    )
    if (
        trigger_layer_confidence is None
        or trigger_layer_confidence < _LOW_CONFIDENCE_THRESHOLD
    ):
        return "trigger_below_low_confidence_threshold"
    (
        _setup_confidence,
        _context_bias_family_mean,
        setup_margin,
        context_bias_shortfall,
        _joint_net_margin,
    ) = _build_margin_components(row)
    if setup_margin is None:
        return "missing_setup_layer_confidence"
    if context_bias_shortfall is None:
        return "missing_context_bias_family_mean"
    if setup_margin <= context_bias_shortfall:
        return "setup_margin_not_above_context_bias_shortfall"
    return None


def _build_rescued_row_summaries(
    rescued_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    static_fields = (
        "setup_layer_confidence",
        _CONTEXT_BIAS_FAMILY_FIELD,
        "selected_strategy_confidence",
        "trigger_layer_confidence",
        _BASELINE_NAME,
    )
    summaries = {
        field: patch_class_a_module.final_split_module._numeric_field_summary(
            rescued_rows,
            field,
        )
        for field in static_fields
    }
    summaries[_RESIDUAL_FIELD] = {
        "baseline": patch_class_a_module.final_split_module._numeric_field_summary(
            rescued_rows,
            "actual_weighted_aggregate_residual",
        ),
        "shadow": patch_class_a_module.final_split_module._numeric_field_summary(
            rescued_rows,
            "shadow_weighted_aggregate_residual",
        ),
    }
    return summaries


def _build_confidence_shift_summaries(
    rescued_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "rule_engine_confidence": {
            "baseline": patch_class_a_module.final_split_module._numeric_field_summary(
                rescued_rows,
                "actual_rule_engine_confidence",
            ),
            "shadow": patch_class_a_module.final_split_module._numeric_field_summary(
                rescued_rows,
                "shadow_rule_engine_confidence",
            ),
        },
        "shadow_uplift": patch_class_a_module.final_split_module._numeric_field_summary(
            rescued_rows,
            "shadow_uplift",
        ),
        "setup_margin": patch_class_a_module.final_split_module._numeric_field_summary(
            rescued_rows,
            "setup_margin",
        ),
        "context_bias_shortfall": (
            patch_class_a_module.final_split_module._numeric_field_summary(
                rescued_rows,
                "context_bias_shortfall",
            )
        ),
        "joint_net_margin": (
            patch_class_a_module.final_split_module._numeric_field_summary(
                rescued_rows,
                "joint_net_margin",
            )
        ),
    }


def _eligibility_breakdown(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("patch_class_b_shadow_block_reason") or "eligible")
        counts[label] = counts.get(label, 0) + 1
    total = len(rows)
    return [
        {
            "status": label,
            "row_count": count,
            "row_rate": _safe_ratio(count, total),
        }
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _source_row_indices(
    rows: Sequence[dict[str, Any]],
    *,
    indicator_field: str,
) -> set[int]:
    source_indices: set[int] = set()
    for row in rows:
        if not bool(row.get(indicator_field)):
            continue
        source_row_index = row.get(_SOURCE_ROW_INDEX)
        if isinstance(source_row_index, int):
            source_indices.add(source_row_index)
    return source_indices


def _scope_relation_status(
    *,
    patch_class_a_eligible_indices: set[int],
    patch_class_b_eligible_indices: set[int],
    patch_class_a_rescued_indices: set[int],
    patch_class_b_rescued_indices: set[int],
) -> str:
    if (
        patch_class_a_eligible_indices == patch_class_b_eligible_indices
        and patch_class_a_rescued_indices == patch_class_b_rescued_indices
    ):
        return "matches_patch_class_a_scope"
    if (
        patch_class_b_eligible_indices.issubset(patch_class_a_eligible_indices)
        and patch_class_b_rescued_indices.issubset(patch_class_a_rescued_indices)
    ):
        return "strict_subset_of_patch_class_a"
    if (
        patch_class_a_eligible_indices.issubset(patch_class_b_eligible_indices)
        and patch_class_a_rescued_indices.issubset(patch_class_b_rescued_indices)
    ):
        return "broader_than_patch_class_a"
    return "mixed_scope_vs_patch_class_a"


def _scope_relation_explanation(*, scope_relation_status: str) -> str:
    if scope_relation_status == "matches_patch_class_a_scope":
        return "B1 changes the same eligible/rescued row set as A1 on this run."
    if scope_relation_status == "strict_subset_of_patch_class_a":
        return "B1 remains narrower than A1 by keeping both the eligible set and the rescued set inside the corresponding A1 sets."
    if scope_relation_status == "broader_than_patch_class_a":
        return "B1 reaches at least one eligible or rescued row outside the corresponding A1 set, so it is broader on this run."
    return "B1 and A1 overlap, but neither candidate stays cleanly inside the other's eligible/rescued sets on this run."


def _joint_selectivity_status(
    *,
    patch_class_a_comparator_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    direct_edge_selection_summary: dict[str, Any],
) -> str:
    rescued_count = int(shadow_summary.get("rescued_from_pocket_row_count", 0) or 0)
    direct_status = str(direct_edge_selection_summary.get("status") or "")
    net_change = int(direct_edge_selection_summary.get("net_change", 0) or 0)
    if rescued_count <= 0:
        return "no_joint_selectivity_recovery_observed"
    if bool(patch_class_a_comparator_summary.get("more_selective_subset_observed")):
        if direct_status == _DIRECT_EDGE_SELECTION_AVAILABLE and net_change > 0:
            return "joint_selectivity_supported_with_direct_edge_improvement"
        return "joint_selectivity_narrows_rescue_but_direct_edge_unproven"
    if bool(
        patch_class_a_comparator_summary.get("is_narrower_or_equal_than_patch_class_a")
    ):
        return "joint_selectivity_matches_patch_class_a_scope"
    scope_relation_status = str(
        patch_class_a_comparator_summary.get("scope_relation_status") or ""
    )
    if scope_relation_status == "broader_than_patch_class_a":
        return "joint_selectivity_broader_than_patch_class_a"
    return "joint_selectivity_mixed_vs_patch_class_a"


def _build_direct_edge_selection_summary(
    *,
    raw_records: Sequence[dict[str, Any]],
    shadow_rows: Sequence[dict[str, Any]],
    run_output_dir: Path,
) -> dict[str, Any]:
    return patch_class_a_module._build_direct_edge_selection_summary(
        raw_records=raw_records,
        shadow_rows=shadow_rows,
        run_output_dir=run_output_dir,
    )


def _row_is_in_pocket(row: dict[str, Any], *, use_shadow: bool) -> bool:
    field_name = (
        "shadow_rule_engine_confidence"
        if use_shadow
        else _RULE_ENGINE_CONFIDENCE_FIELD
    )
    confidence = _to_float(row.get(field_name), default=None)
    return bool(confidence is not None and confidence <= _POCKET_MAX_THRESHOLD)


def _preserved_leakage_status(
    *,
    shadow_summary: dict[str, Any],
) -> str:
    return patch_class_a_module._preserved_leakage_status(
        shadow_summary=shadow_summary
    )


def _nontrivial_rescue_threshold(eligible_pocket_row_count: int) -> int:
    rate_based_threshold = int(
        max(
            1,
            round(float(eligible_pocket_row_count) * _NONTRIVIAL_RESCUE_MIN_RATE),
        )
    )
    return max(
        1,
        min(
            eligible_pocket_row_count,
            max(_NONTRIVIAL_RESCUE_MIN_ROWS, rate_based_threshold),
        ),
    )


def _overall_conclusion(
    *,
    interpretation: dict[str, Any],
    shadow_summary: dict[str, Any],
    patch_class_a_comparator_summary: dict[str, Any],
    direct_edge_selection_summary: dict[str, Any],
) -> str:
    interpretation_status = str(
        interpretation.get("interpretation_status")
        or "patch_class_b_shadow_inconclusive"
    )
    if interpretation_status == "patch_class_b_shadow_supported":
        return (
            "The fixed B1 shadow candidate rescues a nontrivial subset of the targeted "
            "compressed pocket, stays tightly concentrated inside the intended collapsed "
            "class, remains no broader than A1, and improves direct edge-selection count "
            "on the snapshot path."
        )
    if interpretation_status == "patch_class_b_shadow_leaning":
        if bool(patch_class_a_comparator_summary.get("more_selective_subset_observed")):
            return (
                "The fixed B1 shadow candidate rescues a narrower subset than A1 without "
                "visible broad non-target change, but direct edge restoration remains "
                "unproven."
            )
        return (
            "The fixed B1 shadow candidate rescues targeted pocket rows without visible "
            "broad non-target change, but the remaining evidence is not strong enough "
            "to promote a full support claim."
        )
    if int(shadow_summary.get("rescued_from_pocket_row_count", 0) or 0) <= 0:
        return (
            "The fixed B1 shadow candidate does not rescue the targeted compressed pocket "
            "on this run, so Patch Class B remains unsupported here."
        )
    if not bool(
        patch_class_a_comparator_summary.get("is_narrower_or_equal_than_patch_class_a")
    ):
        return (
            "B1 rescue is measurable, but it does not remain narrower or equal than A1, "
            "so the joint-selectivity reading remains inconclusive."
        )
    if direct_edge_selection_summary.get("status") != _DIRECT_EDGE_SELECTION_AVAILABLE:
        return (
            "Pocket rescue is measurable and may be more selective than A1, but direct "
            "edge-selection count is unavailable, so the report remains inconclusive."
        )
    return (
        "Patch Class B shadow evidence remains mixed or too small to promote beyond an "
        "inconclusive reading."
    )


def _patch_class_b_candidate_definition() -> dict[str, Any]:
    return {
        "candidate_id": _PATCH_CLASS_B_CANDIDATE_ID,
        "candidate_label": _PATCH_CLASS_B_CANDIDATE_LABEL,
        "eligibility_gate": {
            "final_slice_required": True,
            "comparison_group_required": _COMPARISON_GROUP_COLLAPSED,
            "rule_engine_confidence_operator": "<=",
            "rule_engine_confidence_threshold": _POCKET_MAX_THRESHOLD,
            "exact_low_confidence_fields": (
                _DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
            ),
            "low_confidence_surface_count": _LOW_SURFACE_COUNT_ANCHOR,
            "trigger_not_below_threshold": _LOW_CONFIDENCE_THRESHOLD,
            "setup_layer_confidence_required": True,
            "context_bias_family_mean_required": True,
            "joint_selectivity_condition": "setup_margin > context_bias_shortfall",
        },
        "formula": {
            "threshold_symbol": "T",
            "T": _LOW_CONFIDENCE_THRESHOLD,
            "setup_margin": "max(0.0, setup_layer_confidence - T)",
            "context_bias_shortfall": "max(0.0, T - context_bias_family_mean)",
            "joint_net_margin": "max(0.0, setup_margin - context_bias_shortfall)",
            "shadow_uplift": "min(0.12, 2.0 * joint_net_margin)",
            "shadow_rule_engine_confidence": (
                "min(1.0, actual_rule_engine_confidence + shadow_uplift)"
            ),
        },
    }


def _dominant_signature_definition() -> dict[str, Any]:
    return patch_class_a_module._dominant_signature_definition()


def _carry_source_metadata(
    new_row: dict[str, Any],
    source_row: dict[str, Any],
) -> dict[str, Any]:
    return patch_class_a_module._carry_source_metadata(new_row, source_row)


def _resolve_path(path: Path) -> Path:
    return patch_class_a_module._resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return patch_class_a_module._parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return patch_class_a_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return patch_class_a_module._widest_configuration_summary(configuration_summaries)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return patch_class_a_module._safe_ratio(numerator, denominator)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return patch_class_a_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return patch_class_a_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return patch_class_a_module._safe_list(value)


if __name__ == "__main__":
    main()
