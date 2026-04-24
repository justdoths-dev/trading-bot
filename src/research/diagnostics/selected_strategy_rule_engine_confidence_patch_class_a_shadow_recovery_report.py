from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_fully_aligned_final_hold_split_diagnosis_report as final_split_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report as origin_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report as signature_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report as residual_module,
)

REPORT_TYPE = (
    "selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report"
)
REPORT_TITLE = (
    "Selected Strategy Rule Engine Confidence Patch Class A Shadow Recovery Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = signature_module.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = signature_module.DEFAULT_OUTPUT_DIR
DEFAULT_MIN_SYMBOL_SUPPORT = signature_module.DEFAULT_MIN_SYMBOL_SUPPORT

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
_POCKET_MAX_THRESHOLD = signature_module._POCKET_MAX_THRESHOLD
_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE = (
    signature_module._DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE
)
_LOW_SURFACE_COUNT_ANCHOR = signature_module._LOW_SURFACE_COUNT_ANCHOR
_ACTIONABLE_SIGNAL_STATES = (
    final_split_module.fully_aligned_module._ACTIONABLE_SIGNAL_STATES
)

_PATCH_CLASS_A_CANDIDATE_ID = "patch_class_a_shadow_a1"
_PATCH_CLASS_A_CANDIDATE_LABEL = "Patch Class A Shadow A1"

_DIRECT_EDGE_SELECTION_AVAILABLE = "edge_selection_direct_count_available"
_DIRECT_EDGE_SELECTION_UNAVAILABLE = "edge_selection_direct_count_unavailable"

_NONTRIVIAL_RESCUE_MIN_ROWS = 2
_NONTRIVIAL_RESCUE_MIN_RATE = 0.15

_SOURCE_ROW_INDEX = "_source_row_index"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only shadow report that applies one tightly gated "
            "Patch Class A setup-aware de-compression candidate inside the already "
            "known compressed pocket."
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

    result = run_selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    shadow_summary = _safe_dict(report.get("shadow_summary"))
    direct_summary = _safe_dict(report.get("direct_edge_selection_summary"))

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
                "direct_edge_selection_status": direct_summary.get("status"),
                "baseline_edge_selection_count": direct_summary.get(
                    "baseline_edge_selection_count"
                ),
                "shadow_edge_selection_count": direct_summary.get(
                    "shadow_edge_selection_count"
                ),
                "net_edge_selection_change": direct_summary.get("net_change"),
                "setup_separation_status": _safe_dict(
                    report.get("setup_secondary_reading")
                ).get("setup_separation_status"),
                "trigger_negative_control_status": _safe_dict(
                    report.get("trigger_negative_control_reading")
                ).get("trigger_negative_control_status"),
                "interpretation_status": report.get("interpretation_status"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report(
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
        "shadow_patch_candidate": _patch_class_a_candidate_definition(),
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
        "direct_edge_selection_summary": _safe_dict(
            widest_summary.get("direct_edge_selection_summary")
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and does not patch production logic, mapper logic, engine logic, candidate-quality-gate logic, or execution-gate logic.",
            "Patch Class A shadow uplift is applied only when the final fully aligned plus rule-bias-aligned row sits inside the fixed compressed pocket, inside the fixed dominant exact low-confidence signature, and belongs to the collapsed outcome class.",
            "The fixed pocket anchor remains actual rule_engine_confidence <= 0.25.",
            "The fixed dominant exact low-confidence signature remains context_bias_family_mean + selected_strategy_confidence with low_surface_count=2.",
            "Only the A1 candidate is evaluated here to keep the shadow comparator narrow and deterministic.",
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
    comparison_rows = _prepare_comparison_rows(raw_records)
    shadow_rows = _apply_patch_class_a_shadow_candidate(comparison_rows)
    shadow_summary = build_shadow_summary(
        comparison_rows=comparison_rows,
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
        direct_edge_selection_summary=direct_edge_selection_summary,
    )
    limitations = build_limitations(
        base_configuration_summary=base_configuration_summary,
        shadow_summary=shadow_summary,
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
        "direct_edge_selection_status": direct_edge_selection_summary.get("status"),
        "net_edge_selection_change": direct_edge_selection_summary.get("net_change"),
        "interpretation_status": interpretation.get("interpretation_status"),
    }

    return {
        **base_configuration_summary,
        "headline": {**base_headline, **shadow_headline},
        "shadow_patch_candidate": _patch_class_a_candidate_definition(),
        "shadow_summary": shadow_summary,
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
        row for row in shadow_rows if bool(row.get("patch_class_a_shadow_eligible"))
    ]
    eligible_pocket_rows = [
        row
        for row in eligible_rows
        if _to_float(row.get("actual_rule_engine_confidence"), default=99.0)
        <= _POCKET_MAX_THRESHOLD
    ]
    changed_rows = [
        row for row in shadow_rows if _to_float(row.get("shadow_uplift"), default=0.0) > 0.0
    ]
    rescued_rows = [
        row for row in shadow_rows if bool(row.get("rescued_from_pocket"))
    ]
    rescued_outside_target_class_rows = [
        row for row in rescued_rows if not bool(row.get("patch_class_a_shadow_eligible"))
    ]
    changed_outside_target_class_rows = [
        row
        for row in changed_rows
        if not bool(row.get("patch_class_a_shadow_eligible"))
    ]
    rescued_outcome_class_counts = _comparison_group_counts(rescued_rows)
    rescued_outcome_count_map = {
        str(_safe_dict(item).get("comparison_group") or ""): int(
            _safe_dict(item).get("row_count", 0) or 0
        )
        for item in rescued_outcome_class_counts
    }

    concentration_status = _concentration_status(
        changed_outside_target_class_row_count=len(changed_outside_target_class_rows),
        rescued_outside_target_class_row_count=len(rescued_outside_target_class_rows),
        rescued_from_pocket_row_count=len(rescued_rows),
    )

    return {
        "candidate_id": _PATCH_CLASS_A_CANDIDATE_ID,
        "candidate_label": _PATCH_CLASS_A_CANDIDATE_LABEL,
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
        "concentration_explanation": _concentration_explanation(
            concentration_status=concentration_status,
            rescued_from_pocket_row_count=len(rescued_rows),
            changed_outside_target_class_row_count=len(
                changed_outside_target_class_rows
            ),
        ),
    }


def build_interpretation(
    *,
    base_configuration_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
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

    direct_edge_selection_supported = (
        direct_status == _DIRECT_EDGE_SELECTION_AVAILABLE and net_change > 0
    )
    preserved_leakage_ok = _preserved_leakage_status(
        shadow_summary=shadow_summary,
    )
    nontrivial_pocket_rescue = rescued_count >= nontrivial_rescue_threshold

    if (
        nontrivial_pocket_rescue
        and concentration_status == "recovery_tightly_concentrated"
        and direct_edge_selection_supported
        and trigger_ok
        and preserved_leakage_ok == "no_obvious_preserved_leakage_explosion"
    ):
        interpretation_status = "patch_class_a_shadow_supported"
    elif (
        rescued_count > 0
        and concentration_status == "recovery_tightly_concentrated"
        and trigger_ok
    ):
        interpretation_status = "patch_class_a_shadow_leaning"
    else:
        interpretation_status = "patch_class_a_shadow_inconclusive"

    facts = [
        (
            "Final slice support: "
            f"{base_summary.get('comparison_support_status')} "
            f"(preserved={base_summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            f"collapsed={base_summary.get('collapsed_final_hold_outcome_row_count', 0)})."
        ),
        (
            "Patch Class A gate: final fully aligned + rule-bias-aligned collapsed rows, "
            f"actual rule_engine_confidence <= {_POCKET_MAX_THRESHOLD}, "
            f"exact_low_confidence_fields={_DOMINANT_EXACT_LOW_CONFIDENCE_SIGNATURE}, "
            f"low_confidence_surface_count={_LOW_SURFACE_COUNT_ANCHOR}, trigger not below "
            f"{_LOW_CONFIDENCE_THRESHOLD}, setup present."
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

    if interpretation_status == "patch_class_a_shadow_supported":
        inference = [
            "A tightly gated setup-aware de-compression rescues a nontrivial subset of the fixed compressed pocket and also improves direct edge-selection count without observable non-collapsed rescue or broad non-target changes."
        ]
    elif interpretation_status == "patch_class_a_shadow_leaning":
        inference = [
            "The fixed A1 shadow candidate rescues targeted pocket rows and stays concentrated inside the intended class, but the evidence remains short of a full support claim."
        ]
    else:
        inference = [
            "Either the rescue is too small, too indirect, or too weakly connected to direct edge-selection counts to promote Patch Class A beyond an inconclusive reading."
        ]

    uncertainty = [
        "This artifact is descriptive and replay-only; it does not change production thresholds or production action logic.",
        "Only the A1 fixed candidate is evaluated here; no sibling sweep or broad search is performed.",
    ]
    if direct_status != _DIRECT_EDGE_SELECTION_AVAILABLE:
        uncertainty.append(
            "Direct edge-selection count could not be exercised safely from the diagnosis path on this run, so the report falls back to pocket-rescue evidence and confidence-shift summaries."
        )
    if preserved_leakage_ok != "no_obvious_preserved_leakage_explosion":
        uncertainty.append(
            "Non-collapsed rescue or another leakage-like pattern is present, so the report does not promote a full support claim."
        )

    return {
        "interpretation_status": interpretation_status,
        "nontrivial_rescue_threshold": nontrivial_rescue_threshold,
        "facts": facts,
        "inference": inference,
        "uncertainty": uncertainty,
        "direct_edge_selection_status": direct_status,
        "preserved_leakage_status": preserved_leakage_ok,
    }


def build_limitations(
    *,
    base_configuration_summary: dict[str, Any],
    shadow_summary: dict[str, Any],
    direct_edge_selection_summary: dict[str, Any],
) -> list[str]:
    limitations = [
        "This artifact stays inside the already-established compressed pocket diagnosis and does not reopen earlier diagnosis branches.",
        "Patch Class A applies only one fixed A1 candidate and does not run a parameter sweep or broaden the eligibility gate.",
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
    if (
        direct_edge_selection_summary.get("status")
        != _DIRECT_EDGE_SELECTION_AVAILABLE
    ):
        limitations.append(
            "Direct edge-selection count was unavailable, so the report is limited to pocket rescue and confidence-shift evidence."
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
    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": _safe_dict(widest.get("summary")),
        "shadow_patch_candidate": _patch_class_a_candidate_definition(),
        "shadow_summary": shadow_summary,
        "direct_edge_selection_summary": direct_summary,
        "interpretation_status": interpretation.get("interpretation_status"),
        "observed": _safe_list(interpretation.get("facts")),
        "inference": _safe_list(interpretation.get("inference")),
        "remains_unproven": _safe_list(interpretation.get("uncertainty")),
        "overall_conclusion": _overall_conclusion(
            interpretation=interpretation,
            shadow_summary=shadow_summary,
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
            "- rescued_collapsed_row_count: "
            f"{shadow_summary.get('rescued_collapsed_row_count', 0)}"
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


def _prepare_comparison_rows(
    raw_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage_rows: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            continue
        stage_row = _build_stage_row(raw_record)
        stage_row[_SOURCE_ROW_INDEX] = index
        stage_rows.append(stage_row)

    actionable_rows: list[dict[str, Any]] = []
    for row in stage_rows:
        if row.get("selected_strategy_result_signal_state") not in (
            _ACTIONABLE_SIGNAL_STATES
        ):
            continue
        actionable_row = _carry_source_metadata(_build_activation_gap_row(row), row)
        actionable_rows.append(actionable_row)

    fully_aligned_rows: list[dict[str, Any]] = []
    for row in actionable_rows:
        fully_aligned_row = _build_fully_aligned_row(row)
        if fully_aligned_row is None:
            continue
        fully_aligned_rows.append(_carry_source_metadata(fully_aligned_row, row))

    final_split_rows: list[dict[str, Any]] = []
    for row in fully_aligned_rows:
        final_split_row = _build_final_split_row(row)
        if final_split_row is None:
            continue
        final_split_rows.append(_carry_source_metadata(final_split_row, row))

    comparison_rows = [
        row
        for row in final_split_rows
        if row.get("comparison_group")
        in {
            _COMPARISON_GROUP_PRESERVED,
            _COMPARISON_GROUP_COLLAPSED,
        }
    ]
    residual_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        residual_row = origin_module._prepare_origin_residual_row(
            residual_module.build_residual_row(row)
        )
        residual_rows.append(_carry_source_metadata(residual_row, row))
    return residual_rows


def _apply_patch_class_a_shadow_candidate(
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
        block_reason = _patch_class_a_block_reason(row)
        eligible = block_reason is None
        setup_confidence = _to_float(row.get("setup_layer_confidence"), default=None)
        setup_margin = 0.0
        shadow_uplift = 0.0
        shadow_rule_engine_confidence = actual_rule_engine_confidence
        if (
            eligible
            and actual_rule_engine_confidence is not None
            and setup_confidence is not None
        ):
            setup_margin = round(
                max(0.0, setup_confidence - float(_LOW_CONFIDENCE_THRESHOLD)),
                6,
            )
            shadow_uplift = round(min(0.12, 1.5 * setup_margin), 6)
            shadow_rule_engine_confidence = round(
                min(1.0, actual_rule_engine_confidence + shadow_uplift),
                6,
            )

        shadow_input = dict(row)
        shadow_input[_RULE_ENGINE_CONFIDENCE_FIELD] = shadow_rule_engine_confidence
        rebuilt_shadow_row = origin_module._prepare_origin_residual_row(
            residual_module.build_residual_row(shadow_input)
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
        rebuilt_shadow_row["setup_margin"] = setup_margin
        rebuilt_shadow_row["shadow_uplift"] = shadow_uplift
        rebuilt_shadow_row["patch_class_a_shadow_eligible"] = eligible
        rebuilt_shadow_row["patch_class_a_shadow_applied"] = shadow_uplift > 0.0
        rebuilt_shadow_row["patch_class_a_shadow_block_reason"] = block_reason
        rebuilt_shadow_row["rescued_from_pocket"] = bool(
            actual_rule_engine_confidence is not None
            and actual_rule_engine_confidence <= _POCKET_MAX_THRESHOLD
            and shadow_rule_engine_confidence is not None
            and shadow_rule_engine_confidence > _POCKET_MAX_THRESHOLD
        )
        shadow_rows.append(rebuilt_shadow_row)
    return shadow_rows


def _build_direct_edge_selection_summary(
    *,
    raw_records: Sequence[dict[str, Any]],
    shadow_rows: Sequence[dict[str, Any]],
    run_output_dir: Path,
) -> dict[str, Any]:
    try:
        baseline_records = [copy.deepcopy(row) for row in raw_records]
        shadow_records = _build_shadow_raw_records(raw_records, shadow_rows)
        baseline_result = _run_direct_edge_selection_snapshot(
            raw_records=baseline_records,
            workspace_root=run_output_dir / "_direct_edge_selection" / "baseline",
        )
        shadow_result = _run_direct_edge_selection_snapshot(
            raw_records=shadow_records,
            workspace_root=run_output_dir / "_direct_edge_selection" / "shadow",
        )
    except Exception as exc:
        return {
            "status": _DIRECT_EDGE_SELECTION_UNAVAILABLE,
            "reason": str(exc),
        }

    if not bool(baseline_result.get("available")):
        return {
            "status": _DIRECT_EDGE_SELECTION_UNAVAILABLE,
            "reason": baseline_result.get("reason"),
        }
    if not bool(shadow_result.get("available")):
        return {
            "status": _DIRECT_EDGE_SELECTION_UNAVAILABLE,
            "reason": shadow_result.get("reason"),
        }

    baseline_edge_selection_count = int(
        baseline_result.get("edge_selection_count", 0) or 0
    )
    shadow_edge_selection_count = int(
        shadow_result.get("edge_selection_count", 0) or 0
    )
    return {
        "status": _DIRECT_EDGE_SELECTION_AVAILABLE,
        "count_method": "single_snapshot_mapper_engine_execution",
        "baseline_edge_selection_count": baseline_edge_selection_count,
        "shadow_edge_selection_count": shadow_edge_selection_count,
        "net_change": shadow_edge_selection_count - baseline_edge_selection_count,
        "baseline_selection_status": baseline_result.get("selection_status"),
        "shadow_selection_status": shadow_result.get("selection_status"),
        "baseline_selected_symbol": baseline_result.get("selected_symbol"),
        "shadow_selected_symbol": shadow_result.get("selected_symbol"),
        "imports": {
            "mapper": baseline_result.get("mapper_import"),
            "engine": baseline_result.get("engine_import"),
            "comparison_pipeline": baseline_result.get("comparison_pipeline_import"),
        },
    }


def _run_direct_edge_selection_snapshot(
    *,
    raw_records: Sequence[dict[str, Any]],
    workspace_root: Path,
) -> dict[str, Any]:
    if not raw_records:
        return {
            "available": False,
            "reason": "no_input_records",
        }

    workspace_root.mkdir(parents=True, exist_ok=True)
    logs_dir = workspace_root / "logs"
    latest_output_dir = logs_dir / "research_reports" / "latest"
    latest_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = logs_dir / "trade_analysis.jsonl"
    _write_jsonl(input_path, raw_records)

    try:
        historical_module, run_research_analyzer = (
            _load_direct_edge_selection_dependencies()
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"import_resolution_failed: {exc}",
        }

    try:
        mapper_module_name, mapper_attr, mapper_func = (
            historical_module.resolve_mapper_callable()
        )
        engine_module_name, engine_attr, engine_func = (
            historical_module.resolve_engine_callable()
        )
        comparison_module_name, comparison_attr, comparison_pipeline_func = (
            historical_module.resolve_comparison_pipeline_callable()
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"import_resolution_failed: {exc}",
        }

    try:
        run_research_analyzer(
            input_path=input_path,
            output_dir=latest_output_dir,
        )
        with historical_module.pushd(workspace_root):
            historical_module.run_comparison_pipeline_step(
                comparison_pipeline_func=comparison_pipeline_func,
                workspace_root=workspace_root,
            )
            mapper_payload = historical_module.run_mapper(
                mapper_func=mapper_func,
                workspace_root=workspace_root,
            )
            engine_output = historical_module.run_engine(
                engine_func=engine_func,
                mapper_payload=mapper_payload,
            )
        selection_fields = historical_module.extract_selection_fields(engine_output)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"snapshot_execution_failed: {exc}",
        }

    return {
        "available": True,
        "edge_selection_count": 1
        if selection_fields.get("selection_status") == "selected"
        else 0,
        "selection_status": selection_fields.get("selection_status"),
        "selected_symbol": selection_fields.get("selected_symbol"),
        "selected_strategy": selection_fields.get("selected_strategy"),
        "selected_horizon": selection_fields.get("selected_horizon"),
        "mapper_import": f"{mapper_module_name}.{mapper_attr}",
        "engine_import": f"{engine_module_name}.{engine_attr}",
        "comparison_pipeline_import": (
            f"{comparison_module_name}.{comparison_attr}"
        ),
    }


def _build_shadow_raw_records(
    raw_records: Sequence[dict[str, Any]],
    shadow_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    copied_records = [copy.deepcopy(row) for row in raw_records]
    for row in shadow_rows:
        source_row_index = row.get(_SOURCE_ROW_INDEX)
        if not isinstance(source_row_index, int):
            continue
        if source_row_index < 0 or source_row_index >= len(copied_records):
            continue
        shadow_rule_engine_confidence = _to_float(
            row.get("shadow_rule_engine_confidence"),
            default=None,
        )
        if shadow_rule_engine_confidence is None:
            continue
        record = copied_records[source_row_index]
        if not isinstance(record, dict):
            continue
        rule_engine = dict(record.get("rule_engine") or {})
        rule_engine["confidence"] = shadow_rule_engine_confidence
        record["rule_engine"] = rule_engine
        copied_records[source_row_index] = record
    return copied_records


def _patch_class_a_block_reason(row: dict[str, Any]) -> str | None:
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
    if _to_float(row.get("setup_layer_confidence"), default=None) is None:
        return "missing_setup_layer_confidence"
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
        field: final_split_module._numeric_field_summary(rescued_rows, field)
        for field in static_fields
    }
    summaries[_RESIDUAL_FIELD] = {
        "baseline": final_split_module._numeric_field_summary(
            rescued_rows,
            "actual_weighted_aggregate_residual",
        ),
        "shadow": final_split_module._numeric_field_summary(
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
            "baseline": final_split_module._numeric_field_summary(
                rescued_rows,
                "actual_rule_engine_confidence",
            ),
            "shadow": final_split_module._numeric_field_summary(
                rescued_rows,
                "shadow_rule_engine_confidence",
            ),
        },
        "shadow_uplift": final_split_module._numeric_field_summary(
            rescued_rows,
            "shadow_uplift",
        ),
        "setup_margin": final_split_module._numeric_field_summary(
            rescued_rows,
            "setup_margin",
        ),
    }


def _comparison_group_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {
        _COMPARISON_GROUP_PRESERVED: 0,
        _COMPARISON_GROUP_COLLAPSED: 0,
    }
    for row in rows:
        label = str(row.get("comparison_group") or "")
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    return [
        {
            "comparison_group": label,
            "row_count": row_count,
            "row_rate": _safe_ratio(row_count, total),
        }
        for label, row_count in counts.items()
    ]


def _eligibility_breakdown(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(
            row.get("patch_class_a_shadow_block_reason")
            or "eligible"
        )
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


def _row_is_in_pocket(row: dict[str, Any], *, use_shadow: bool) -> bool:
    field_name = "shadow_rule_engine_confidence" if use_shadow else _RULE_ENGINE_CONFIDENCE_FIELD
    confidence = _to_float(row.get(field_name), default=None)
    return bool(confidence is not None and confidence <= _POCKET_MAX_THRESHOLD)


def _concentration_status(
    *,
    changed_outside_target_class_row_count: int,
    rescued_outside_target_class_row_count: int,
    rescued_from_pocket_row_count: int,
) -> str:
    if (
        rescued_from_pocket_row_count > 0
        and changed_outside_target_class_row_count == 0
        and rescued_outside_target_class_row_count == 0
    ):
        return "recovery_tightly_concentrated"
    if rescued_from_pocket_row_count > 0:
        return "recovery_partially_concentrated"
    return "no_recovery_observed"


def _concentration_explanation(
    *,
    concentration_status: str,
    rescued_from_pocket_row_count: int,
    changed_outside_target_class_row_count: int,
) -> str:
    if concentration_status == "recovery_tightly_concentrated":
        return (
            "All observed uplift stays inside the intended fixed pocket/signature class, "
            "and rescued rows leave the pocket without any changed rows outside that class."
        )
    if concentration_status == "recovery_partially_concentrated":
        return (
            "Some pocket rescue is present, but at least one changed row falls outside the "
            "intended class, so the reading remains conservative."
        )
    if rescued_from_pocket_row_count <= 0:
        return "No eligible rows were rescued above the pocket threshold."
    return "Recovery remains too mixed to call tightly concentrated."


def _preserved_leakage_status(
    *,
    shadow_summary: dict[str, Any],
) -> str:
    if int(shadow_summary.get("changed_outside_target_class_row_count", 0) or 0) > 0:
        return "possible_non_target_change_growth"
    if int(shadow_summary.get("rescued_preserved_row_count", 0) or 0) > 0:
        return "noncollapsed_rescue_observed"
    if int(shadow_summary.get("rescued_from_pocket_row_count", 0) or 0) > 0:
        return "no_obvious_preserved_leakage_explosion"
    return "no_recovery_observed"


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
    direct_edge_selection_summary: dict[str, Any],
) -> str:
    interpretation_status = str(
        interpretation.get("interpretation_status") or "patch_class_a_shadow_inconclusive"
    )
    if interpretation_status == "patch_class_a_shadow_supported":
        return (
            "The fixed A1 shadow candidate rescues a nontrivial subset of the targeted "
            "compressed pocket, stays tightly concentrated inside the intended collapsed "
            "class, and improves direct edge-selection count on the snapshot path."
        )
    if interpretation_status == "patch_class_a_shadow_leaning":
        return (
            "The fixed A1 shadow candidate rescues targeted pocket rows without visible "
            "broad non-target change, but the remaining evidence is not strong enough "
            "to promote a full support claim."
        )
    if int(shadow_summary.get("rescued_from_pocket_row_count", 0) or 0) <= 0:
        return (
            "The fixed A1 shadow candidate does not rescue the targeted compressed pocket "
            "on this run, so Patch Class A remains unsupported here."
        )
    if direct_edge_selection_summary.get("status") != _DIRECT_EDGE_SELECTION_AVAILABLE:
        return (
            "Pocket rescue is measurable, but direct edge-selection count is unavailable, "
            "so the report remains inconclusive."
        )
    return (
        "Patch Class A shadow evidence remains mixed or too small to promote beyond an "
        "inconclusive reading."
    )


def _patch_class_a_candidate_definition() -> dict[str, Any]:
    return {
        "candidate_id": _PATCH_CLASS_A_CANDIDATE_ID,
        "candidate_label": _PATCH_CLASS_A_CANDIDATE_LABEL,
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
        },
        "formula": {
            "threshold_symbol": "T",
            "T": _LOW_CONFIDENCE_THRESHOLD,
            "setup_margin": "max(0.0, setup_layer_confidence - T)",
            "shadow_uplift": "min(0.12, 1.5 * setup_margin)",
            "shadow_rule_engine_confidence": (
                "min(1.0, actual_rule_engine_confidence + shadow_uplift)"
            ),
        },
    }


def _dominant_signature_definition() -> dict[str, Any]:
    return signature_module._dominant_signature_definition()


def _carry_source_metadata(
    new_row: dict[str, Any],
    source_row: dict[str, Any],
) -> dict[str, Any]:
    if _SOURCE_ROW_INDEX in source_row and _SOURCE_ROW_INDEX not in new_row:
        new_row[_SOURCE_ROW_INDEX] = source_row[_SOURCE_ROW_INDEX]
    return new_row


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_direct_edge_selection_dependencies() -> tuple[Any, Any]:
    from src.research.diagnostics.historical import (
        historical_direct_edge_selection_diagnosis as historical_module,
    )
    from src.research.research_analyzer import run_research_analyzer

    return historical_module, run_research_analyzer


def _resolve_path(path: Path) -> Path:
    return signature_module._resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return signature_module._parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return signature_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    return signature_module._build_stage_row(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    return signature_module._build_activation_gap_row(row)


def _build_fully_aligned_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return signature_module._build_fully_aligned_row(row)


def _build_final_split_row(row: dict[str, Any]) -> dict[str, Any] | None:
    return signature_module._build_final_split_row(row)


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return signature_module._widest_configuration_summary(configuration_summaries)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return signature_module._safe_ratio(numerator, denominator)


def _to_float(value: Any, *, default: float | None) -> float | None:
    return signature_module._to_float(value, default=default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return signature_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return signature_module._safe_list(value)


if __name__ == "__main__":
    main()