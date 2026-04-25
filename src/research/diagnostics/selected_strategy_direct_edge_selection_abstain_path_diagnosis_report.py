from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report as patch_class_a_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report as patch_class_b_module,
)

REPORT_TYPE = "selected_strategy_direct_edge_selection_abstain_path_diagnosis_report"
REPORT_TITLE = "Selected Strategy Direct Edge Selection Abstain Path Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = patch_class_b_module.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = patch_class_b_module.DEFAULT_OUTPUT_DIR
DEFAULT_MIN_SYMBOL_SUPPORT = patch_class_b_module.DEFAULT_MIN_SYMBOL_SUPPORT

DiagnosisConfiguration = patch_class_b_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = patch_class_b_module.DEFAULT_CONFIGURATIONS

_DIRECT_EDGE_SELECTION_AVAILABLE = "edge_selection_direct_abstain_path_available"
_DIRECT_EDGE_SELECTION_PARTIAL = "edge_selection_direct_abstain_path_partially_available"
_DIRECT_EDGE_SELECTION_UNAVAILABLE = "edge_selection_direct_abstain_path_unavailable"

_SNAPSHOT_BASELINE = "baseline"
_SNAPSHOT_PATCH_CLASS_A = "patch_class_a_shadow_a1"
_SNAPSHOT_PATCH_CLASS_B = "patch_class_b_shadow_b1"
_SNAPSHOT_ORDER = (
    _SNAPSHOT_BASELINE,
    _SNAPSHOT_PATCH_CLASS_A,
    _SNAPSHOT_PATCH_CLASS_B,
)
_SNAPSHOT_LABELS = {
    _SNAPSHOT_BASELINE: "Baseline",
    _SNAPSHOT_PATCH_CLASS_A: "Patch Class A Shadow A1",
    _SNAPSHOT_PATCH_CLASS_B: "Patch Class B Shadow B1",
}

_TOP_N_RANKING_SNAPSHOT = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for why direct edge-selection remains "
            "abstain across baseline, Patch Class A A1, and Patch Class B B1."
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
            "Retained for parity with sibling reports. This report diagnoses the "
            "direct edge-selection abstain path after the fixed A1/B1 shadows."
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

    result = run_selected_strategy_direct_edge_selection_abstain_path_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("direct_edge_selection_abstain_path_summary"))
    snapshots = _safe_dict(summary.get("snapshots"))
    baseline = _safe_dict(snapshots.get(_SNAPSHOT_BASELINE))
    patch_class_a = _safe_dict(snapshots.get(_SNAPSHOT_PATCH_CLASS_A))
    patch_class_b = _safe_dict(snapshots.get(_SNAPSHOT_PATCH_CLASS_B))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    report.get("widest_configuration")
                ).get("display_name"),
                "summary_status": summary.get("status"),
                "primary_abstain_path_classification": summary.get(
                    "primary_abstain_path_classification"
                ),
                "interpretation_status": report.get("interpretation_status"),
                "baseline_selection_status": baseline.get("selection_status"),
                "patch_class_a_selection_status": patch_class_a.get(
                    "selection_status"
                ),
                "patch_class_b_selection_status": patch_class_b.get(
                    "selection_status"
                ),
                "baseline_candidate_count": baseline.get("candidate_count"),
                "patch_class_a_candidate_count": patch_class_a.get("candidate_count"),
                "patch_class_b_candidate_count": patch_class_b.get("candidate_count"),
                "baseline_top_candidate": baseline.get("top_candidate_identity"),
                "patch_class_a_top_candidate": patch_class_a.get(
                    "top_candidate_identity"
                ),
                "patch_class_b_top_candidate": patch_class_b.get(
                    "top_candidate_identity"
                ),
                "persistent_reason_codes": _safe_dict(
                    summary.get("persistent_abstain_reason")
                ).get("persistent_reason_codes"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_direct_edge_selection_abstain_path_diagnosis_report(
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
        effective_input_path, raw_records, source_metadata = _materialize_configuration_input(
            input_path=resolved_input,
            run_output_dir=run_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
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
    direct_summary = _safe_dict(
        widest_summary.get("direct_edge_selection_abstain_path_summary")
    )
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
        "snapshot_order": list(_SNAPSHOT_ORDER),
        "snapshot_labels": dict(_SNAPSHOT_LABELS),
        "patch_class_a_shadow_candidate": (
            patch_class_a_module._patch_class_a_candidate_definition()
        ),
        "patch_class_b_shadow_candidate": (
            patch_class_b_module._patch_class_b_candidate_definition()
        ),
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary_row.get("headline"))
            for summary_row in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "direct_edge_selection_abstain_path_summary": direct_summary,
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
        "limitations": _safe_list(widest_summary.get("limitations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This artifact is diagnosis-only and does not patch production logic, mapper logic, engine logic, candidate-quality-gate logic, or execution-gate logic.",
            "Baseline, A1, and B1 are treated as the only fixed comparator snapshots.",
            "Patch Class A and Patch Class B rows are produced by the existing sibling shadow helpers; no new recovery candidate is introduced.",
            "If the current direct edge-selection path does not expose a field, this report records it as unavailable instead of inventing a surrogate.",
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
    comparison_rows = patch_class_a_module._prepare_comparison_rows(raw_records)
    patch_class_a_shadow_rows = patch_class_a_module._apply_patch_class_a_shadow_candidate(
        comparison_rows
    )
    patch_class_b_shadow_rows = patch_class_b_module._apply_patch_class_b_shadow_candidate(
        comparison_rows
    )

    patch_class_a_shadow_summary = patch_class_a_module.build_shadow_summary(
        comparison_rows=comparison_rows,
        shadow_rows=patch_class_a_shadow_rows,
    )
    patch_class_b_shadow_summary = patch_class_b_module.build_shadow_summary(
        comparison_rows=comparison_rows,
        shadow_rows=patch_class_b_shadow_rows,
    )
    direct_summary = _build_direct_edge_selection_abstain_path_summary(
        raw_records=raw_records,
        patch_class_a_shadow_rows=patch_class_a_shadow_rows,
        patch_class_b_shadow_rows=patch_class_b_shadow_rows,
        run_output_dir=run_output_dir,
    )
    interpretation = build_interpretation(direct_summary)
    limitations = build_limitations(direct_summary)

    headline = {
        "display_name": configuration.display_name,
        "latest_window_hours": configuration.latest_window_hours,
        "latest_max_rows": configuration.latest_max_rows,
        "summary_status": direct_summary.get("status"),
        "primary_abstain_path_classification": direct_summary.get(
            "primary_abstain_path_classification"
        ),
        "baseline_selection_status": _snapshot_value(
            direct_summary,
            _SNAPSHOT_BASELINE,
            "selection_status",
        ),
        "patch_class_a_selection_status": _snapshot_value(
            direct_summary,
            _SNAPSHOT_PATCH_CLASS_A,
            "selection_status",
        ),
        "patch_class_b_selection_status": _snapshot_value(
            direct_summary,
            _SNAPSHOT_PATCH_CLASS_B,
            "selection_status",
        ),
        "baseline_candidate_count": _snapshot_value(
            direct_summary,
            _SNAPSHOT_BASELINE,
            "candidate_count",
        ),
        "patch_class_a_candidate_count": _snapshot_value(
            direct_summary,
            _SNAPSHOT_PATCH_CLASS_A,
            "candidate_count",
        ),
        "patch_class_b_candidate_count": _snapshot_value(
            direct_summary,
            _SNAPSHOT_PATCH_CLASS_B,
            "candidate_count",
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
    }

    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "effective_input_path": str(effective_input_path),
        "output_dir": str(run_output_dir),
        "source_metadata": source_metadata,
        "min_symbol_support": min_symbol_support,
        "headline": headline,
        "patch_class_a_shadow_summary": patch_class_a_shadow_summary,
        "patch_class_b_shadow_summary": patch_class_b_shadow_summary,
        "direct_edge_selection_abstain_path_summary": direct_summary,
        "interpretation": interpretation,
        "limitations": limitations,
    }


def build_abstain_path_summary(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_snapshots = {
        snapshot_name: _normalize_direct_snapshot(
            snapshot_name,
            _safe_dict(snapshots.get(snapshot_name)),
        )
        for snapshot_name in _SNAPSHOT_ORDER
    }
    available_count = sum(
        1 for snapshot in normalized_snapshots.values() if snapshot["available"]
    )
    if available_count == len(_SNAPSHOT_ORDER):
        status = _DIRECT_EDGE_SELECTION_AVAILABLE
    elif available_count > 0:
        status = _DIRECT_EDGE_SELECTION_PARTIAL
    else:
        status = _DIRECT_EDGE_SELECTION_UNAVAILABLE

    comparisons = {
        "baseline_vs_patch_class_a": _compare_snapshots(
            normalized_snapshots[_SNAPSHOT_BASELINE],
            normalized_snapshots[_SNAPSHOT_PATCH_CLASS_A],
        ),
        "baseline_vs_patch_class_b": _compare_snapshots(
            normalized_snapshots[_SNAPSHOT_BASELINE],
            normalized_snapshots[_SNAPSHOT_PATCH_CLASS_B],
        ),
        "patch_class_a_vs_patch_class_b": _compare_snapshots(
            normalized_snapshots[_SNAPSHOT_PATCH_CLASS_A],
            normalized_snapshots[_SNAPSHOT_PATCH_CLASS_B],
        ),
    }
    persistent_reason = _build_persistent_abstain_reason(normalized_snapshots)
    shared_top_candidate = _build_shared_top_candidate_summary(normalized_snapshots)
    primary_classification = _primary_abstain_path_classification(
        normalized_snapshots
    )
    interpretation = build_interpretation(
        {
            "status": status,
            "snapshots": normalized_snapshots,
            "primary_abstain_path_classification": primary_classification,
            "persistent_abstain_reason": persistent_reason,
        }
    )

    return {
        "status": status,
        "snapshot_order": list(_SNAPSHOT_ORDER),
        "snapshots": normalized_snapshots,
        "comparisons": comparisons,
        "same_top_candidate_across_snapshots": bool(shared_top_candidate),
        "shared_top_candidate": shared_top_candidate,
        "persistent_abstain_reason": persistent_reason,
        "remaining_acceptance_deficit": _remaining_acceptance_deficit(
            normalized_snapshots=normalized_snapshots,
            persistent_reason=persistent_reason,
            shared_top_candidate=shared_top_candidate,
        ),
        "primary_abstain_path_classification": primary_classification,
        "minimum_question_answers": _build_minimum_question_answers(
            normalized_snapshots=normalized_snapshots,
            comparisons=comparisons,
            primary_classification=primary_classification,
        ),
        "interpretation_status": interpretation.get("interpretation_status"),
        "interpretation": interpretation,
    }


def build_interpretation(direct_summary: Mapping[str, Any]) -> dict[str, Any]:
    summary = _safe_dict(direct_summary)
    snapshots = _safe_dict(summary.get("snapshots"))
    normalized_snapshots = {
        snapshot_name: _safe_dict(snapshots.get(snapshot_name))
        for snapshot_name in _SNAPSHOT_ORDER
    }
    available_snapshots = [
        snapshot
        for snapshot in normalized_snapshots.values()
        if bool(snapshot.get("available"))
    ]
    statuses = [str(snapshot.get("selection_status") or "") for snapshot in available_snapshots]
    all_available = len(available_snapshots) == len(_SNAPSHOT_ORDER)
    all_abstain = all_available and all(status == "abstain" for status in statuses)
    any_selected = any(status == "selected" for status in statuses)
    primary_classification = str(
        summary.get("primary_abstain_path_classification") or "unavailable"
    )
    persistent_reason = _safe_dict(summary.get("persistent_abstain_reason"))

    if not all_available:
        interpretation_status = "direct_edge_selection_abstain_path_incomplete"
    elif all_abstain and primary_classification != "mixed_abstain_path":
        interpretation_status = "direct_edge_selection_abstain_path_explained"
    elif all_abstain:
        interpretation_status = "direct_edge_selection_abstain_path_explained_mixed"
    elif any_selected:
        interpretation_status = "direct_edge_selection_not_persistently_abstaining"
    else:
        interpretation_status = "direct_edge_selection_abstain_path_inconclusive"

    facts = [
        (
            "Snapshot availability: "
            f"{len(available_snapshots)}/{len(_SNAPSHOT_ORDER)} direct snapshots."
        ),
        (
            "Snapshot statuses: "
            + ", ".join(
                f"{snapshot_name}={_safe_dict(snapshots.get(snapshot_name)).get('selection_status')}"
                for snapshot_name in _SNAPSHOT_ORDER
            )
            + "."
        ),
        (
            "Primary abstain path classification: "
            f"{primary_classification}."
        ),
    ]
    if persistent_reason.get("persistent_reason_codes"):
        facts.append(
            "Persistent engine reason codes: "
            f"{persistent_reason.get('persistent_reason_codes')}."
        )

    return {
        "interpretation_status": interpretation_status,
        "primary_abstain_path_classification": primary_classification,
        "all_snapshots_available": all_available,
        "all_snapshots_abstain": all_abstain,
        "any_snapshot_selected": any_selected,
        "facts": facts,
        "conclusion": _interpretation_conclusion(
            interpretation_status=interpretation_status,
            primary_classification=primary_classification,
        ),
    }


def build_limitations(direct_summary: Mapping[str, Any]) -> list[str]:
    summary = _safe_dict(direct_summary)
    limitations = [
        "This artifact diagnoses only the direct edge-selection abstain path and does not introduce a recovery patch class.",
        "Only baseline, A1, and B1 fixed snapshots are compared; no parameter sweep or threshold search is performed.",
        "Mapper, engine, candidate-quality gate, execution gate, and production latest-window defaults are not modified.",
        "Candidate and rejection fields are limited to values already exposed by the current mapper and engine outputs.",
    ]
    if summary.get("status") != _DIRECT_EDGE_SELECTION_AVAILABLE:
        limitations.append(
            "At least one direct snapshot was unavailable, so path classification is partial."
        )
    return limitations


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest_summary = _widest_configuration_summary(configuration_summaries)
    interpretation = _safe_dict(widest_summary.get("interpretation"))
    direct_summary = _safe_dict(
        widest_summary.get("direct_edge_selection_abstain_path_summary")
    )
    return {
        "interpretation_status": interpretation.get("interpretation_status"),
        "primary_abstain_path_classification": direct_summary.get(
            "primary_abstain_path_classification"
        ),
        "persistent_abstain_reason": _safe_dict(
            direct_summary.get("persistent_abstain_reason")
        ),
        "remaining_acceptance_deficit": direct_summary.get(
            "remaining_acceptance_deficit"
        ),
        "overall_conclusion": interpretation.get("conclusion"),
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
        item = _safe_dict(summary)
        config = _safe_dict(item.get("configuration"))
        direct_summary = _safe_dict(
            item.get("direct_edge_selection_abstain_path_summary")
        )
        snapshots = _safe_dict(direct_summary.get("snapshots"))
        comparisons = _safe_dict(direct_summary.get("comparisons"))
        interpretation = _safe_dict(item.get("interpretation"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(f"- summary_status: {direct_summary.get('status')}")
        lines.append(
            "- primary_abstain_path_classification: "
            f"{direct_summary.get('primary_abstain_path_classification')}"
        )
        lines.append(
            "- interpretation_status: "
            f"{interpretation.get('interpretation_status')}"
        )
        for snapshot_name in _SNAPSHOT_ORDER:
            snapshot = _safe_dict(snapshots.get(snapshot_name))
            lines.append(
                f"- {snapshot_name}: status={snapshot.get('selection_status')}, "
                f"reason_codes={snapshot.get('reason_codes')}, "
                f"candidate_count={snapshot.get('candidate_count')}, "
                f"top_candidate={snapshot.get('top_candidate_identity')}, "
                f"path={snapshot.get('abstain_path_classification')}"
            )
        for comparison_name, comparison in comparisons.items():
            delta = _safe_dict(comparison)
            lines.append(
                f"- {comparison_name}: candidate_count_delta={delta.get('candidate_count_delta')}, "
                f"top_changed={delta.get('top_candidate_identity_changed')}, "
                f"ordering_changed={delta.get('candidate_ordering_changed')}, "
                f"reason_changed={delta.get('abstain_reason_changed')}"
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
        "- primary_abstain_path_classification: "
        f"{final_assessment.get('primary_abstain_path_classification')}"
    )
    lines.append(
        "- overall_conclusion: "
        f"{final_assessment.get('overall_conclusion')}"
    )
    lines.append("")
    return "\n".join(lines)


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


def _build_direct_edge_selection_abstain_path_summary(
    *,
    raw_records: Sequence[dict[str, Any]],
    patch_class_a_shadow_rows: Sequence[dict[str, Any]],
    patch_class_b_shadow_rows: Sequence[dict[str, Any]],
    run_output_dir: Path,
) -> dict[str, Any]:
    baseline_records = [copy.deepcopy(row) for row in raw_records]
    patch_class_a_records = patch_class_a_module._build_shadow_raw_records(
        raw_records,
        patch_class_a_shadow_rows,
    )
    patch_class_b_records = patch_class_a_module._build_shadow_raw_records(
        raw_records,
        patch_class_b_shadow_rows,
    )
    snapshots = {
        _SNAPSHOT_BASELINE: _run_direct_edge_selection_snapshot(
            raw_records=baseline_records,
            workspace_root=run_output_dir
            / "_direct_edge_selection_abstain_path"
            / _SNAPSHOT_BASELINE,
            snapshot_name=_SNAPSHOT_BASELINE,
            snapshot_label=_SNAPSHOT_LABELS[_SNAPSHOT_BASELINE],
        ),
        _SNAPSHOT_PATCH_CLASS_A: _run_direct_edge_selection_snapshot(
            raw_records=patch_class_a_records,
            workspace_root=run_output_dir
            / "_direct_edge_selection_abstain_path"
            / _SNAPSHOT_PATCH_CLASS_A,
            snapshot_name=_SNAPSHOT_PATCH_CLASS_A,
            snapshot_label=_SNAPSHOT_LABELS[_SNAPSHOT_PATCH_CLASS_A],
        ),
        _SNAPSHOT_PATCH_CLASS_B: _run_direct_edge_selection_snapshot(
            raw_records=patch_class_b_records,
            workspace_root=run_output_dir
            / "_direct_edge_selection_abstain_path"
            / _SNAPSHOT_PATCH_CLASS_B,
            snapshot_name=_SNAPSHOT_PATCH_CLASS_B,
            snapshot_label=_SNAPSHOT_LABELS[_SNAPSHOT_PATCH_CLASS_B],
        ),
    }
    return build_abstain_path_summary(snapshots)


def _run_direct_edge_selection_snapshot(
    *,
    raw_records: Sequence[dict[str, Any]],
    workspace_root: Path,
    snapshot_name: str,
    snapshot_label: str,
) -> dict[str, Any]:
    if not raw_records:
        return {
            "available": False,
            "snapshot_name": snapshot_name,
            "snapshot_label": snapshot_label,
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
            patch_class_a_module._load_direct_edge_selection_dependencies()
        )
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
            "snapshot_name": snapshot_name,
            "snapshot_label": snapshot_label,
            "reason": f"import_resolution_failed: {exc}",
        }

    try:
        run_research_analyzer(input_path=input_path, output_dir=latest_output_dir)
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
        mapper_data = historical_module.to_plain_data(mapper_payload)
        engine_data = historical_module.to_plain_data(engine_output)
        candidate_seed_count, horizons_with_seed, horizons_without_seed = (
            historical_module.extract_candidate_seed_info(mapper_payload)
        )
    except Exception as exc:
        return {
            "available": False,
            "snapshot_name": snapshot_name,
            "snapshot_label": snapshot_label,
            "reason": f"snapshot_execution_failed: {exc}",
        }

    mapper_dict = _safe_dict(mapper_data)
    engine_dict = _safe_dict(engine_data)
    candidates = mapper_dict.get("candidates")
    ranking = _safe_ranking(
        engine_dict.get("ranking") or engine_dict.get("ranked_candidates")
    )

    return {
        "available": True,
        "snapshot_name": snapshot_name,
        "snapshot_label": snapshot_label,
        "selection_status": selection_fields.get("selection_status"),
        "reason": selection_fields.get("reason"),
        "reason_codes": _safe_str_list(engine_dict.get("reason_codes")),
        "selection_explanation": engine_dict.get("selection_explanation"),
        "selected_symbol": selection_fields.get("selected_symbol"),
        "selected_strategy": selection_fields.get("selected_strategy"),
        "selected_horizon": selection_fields.get("selected_horizon"),
        "selected_score": engine_dict.get("selection_score"),
        "selected_confidence": engine_dict.get("selection_confidence"),
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
        "candidates_considered": engine_dict.get("candidates_considered"),
        "candidate_seed_count": candidate_seed_count,
        "candidate_seed_diagnostics": mapper_dict.get("candidate_seed_diagnostics"),
        "horizons_with_seed": horizons_with_seed,
        "horizons_without_seed": horizons_without_seed,
        "ranking_count": len(ranking),
        "ranking": ranking,
        "abstain_diagnosis": engine_dict.get("abstain_diagnosis"),
        "imports": {
            "mapper": f"{mapper_module_name}.{mapper_attr}",
            "engine": f"{engine_module_name}.{engine_attr}",
            "comparison_pipeline": f"{comparison_module_name}.{comparison_attr}",
        },
    }


def _normalize_direct_snapshot(
    snapshot_name: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    available = bool(snapshot.get("available"))
    ranking = _safe_ranking(snapshot.get("ranking"))
    abstain_diagnosis = _safe_dict(snapshot.get("abstain_diagnosis"))
    top_candidate = _extract_top_candidate(snapshot=snapshot, ranking=ranking)
    top_candidate_snapshot = _candidate_snapshot(top_candidate)
    reason_codes = _safe_str_list(snapshot.get("reason_codes"))
    if not reason_codes:
        reason_text = snapshot.get("reason")
        if isinstance(reason_text, str) and reason_text:
            reason_codes = [reason_text]

    candidate_status_counts = _candidate_status_counts(ranking)
    eligible_count = _coerce_int(
        snapshot.get("eligible_candidate_count"),
        default=_coerce_int(
            abstain_diagnosis.get("eligible_candidate_count"),
            default=candidate_status_counts.get("eligible", 0),
        ),
    )
    penalized_count = _coerce_int(
        snapshot.get("penalized_candidate_count"),
        default=_coerce_int(
            abstain_diagnosis.get("penalized_candidate_count"),
            default=candidate_status_counts.get("penalized", 0),
        ),
    )
    blocked_count = _coerce_int(
        snapshot.get("blocked_candidate_count"),
        default=_coerce_int(
            abstain_diagnosis.get("blocked_candidate_count"),
            default=candidate_status_counts.get("blocked", 0),
        ),
    )
    candidate_count = _coerce_int(
        snapshot.get("candidate_count"),
        default=_coerce_int(snapshot.get("candidates_considered"), default=len(ranking)),
    )
    ranking_count = _coerce_int(snapshot.get("ranking_count"), default=len(ranking))
    selection_status = str(snapshot.get("selection_status") or "unavailable")
    abstain_category = str(
        snapshot.get("abstain_category")
        or abstain_diagnosis.get("category")
        or ""
    )
    classification = _classify_abstain_path(
        available=available,
        selection_status=selection_status,
        reason_codes=reason_codes,
        abstain_category=abstain_category,
        candidate_count=candidate_count,
        ranking_count=ranking_count,
        eligible_count=eligible_count,
    )

    return {
        "available": available,
        "snapshot_name": snapshot_name,
        "snapshot_label": snapshot.get("snapshot_label")
        or _SNAPSHOT_LABELS.get(snapshot_name, snapshot_name),
        "selection_status": selection_status,
        "final_selection_status": selection_status,
        "reason": snapshot.get("reason"),
        "reason_codes": reason_codes,
        "selection_explanation": snapshot.get("selection_explanation"),
        "abstain_category": abstain_category or None,
        "abstain_path_classification": classification,
        "candidate_count": candidate_count,
        "candidate_seed_count": _coerce_int(
            snapshot.get("candidate_seed_count"),
            default=None,
        ),
        "candidate_seed_diagnostics": _safe_dict(
            snapshot.get("candidate_seed_diagnostics")
        ),
        "candidate_presence": candidate_count > 0 or ranking_count > 0,
        "ranking_count": ranking_count,
        "ranking_identity_sequence": [
            _candidate_identity(item) for item in ranking if _candidate_identity(item)
        ],
        "ranking_top_candidates": [
            _candidate_snapshot(item) for item in ranking[:_TOP_N_RANKING_SNAPSHOT]
        ],
        "eligible_candidate_count": eligible_count,
        "penalized_candidate_count": penalized_count,
        "blocked_candidate_count": blocked_count,
        "candidate_status_counts": candidate_status_counts,
        "top_candidate": top_candidate_snapshot,
        "top_candidate_identity": _candidate_identity(top_candidate_snapshot),
        "top_candidate_status": _safe_dict(top_candidate_snapshot).get(
            "candidate_status"
        ),
        "top_candidate_score": _safe_dict(top_candidate_snapshot).get(
            "selection_score"
        ),
        "top_candidate_confidence": _safe_dict(top_candidate_snapshot).get(
            "selection_confidence"
        ),
        "selected_candidate_identity": _selected_candidate_identity(snapshot),
        "horizons_with_seed": _safe_str_list(snapshot.get("horizons_with_seed")),
        "horizons_without_seed": _safe_str_list(snapshot.get("horizons_without_seed")),
        "imports": _safe_dict(snapshot.get("imports")),
        "unavailable_reason": None if available else snapshot.get("reason"),
    }


def _compare_snapshots(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_snapshot = _safe_dict(before)
    after_snapshot = _safe_dict(after)
    if not before_snapshot.get("available") or not after_snapshot.get("available"):
        return {
            "comparison_status": "comparison_unavailable",
            "reason": "one_or_both_snapshots_unavailable",
        }

    before_sequence = _safe_str_list(before_snapshot.get("ranking_identity_sequence"))
    after_sequence = _safe_str_list(after_snapshot.get("ranking_identity_sequence"))
    before_top = before_snapshot.get("top_candidate_identity")
    after_top = after_snapshot.get("top_candidate_identity")

    return {
        "comparison_status": "comparison_available",
        "before_snapshot": before_snapshot.get("snapshot_name"),
        "after_snapshot": after_snapshot.get("snapshot_name"),
        "candidate_count_before": before_snapshot.get("candidate_count"),
        "candidate_count_after": after_snapshot.get("candidate_count"),
        "candidate_count_delta": _numeric_delta(
            before_snapshot.get("candidate_count"),
            after_snapshot.get("candidate_count"),
        ),
        "ranking_count_before": before_snapshot.get("ranking_count"),
        "ranking_count_after": after_snapshot.get("ranking_count"),
        "ranking_count_delta": _numeric_delta(
            before_snapshot.get("ranking_count"),
            after_snapshot.get("ranking_count"),
        ),
        "eligible_candidate_count_delta": _numeric_delta(
            before_snapshot.get("eligible_candidate_count"),
            after_snapshot.get("eligible_candidate_count"),
        ),
        "top_candidate_before": before_top,
        "top_candidate_after": after_top,
        "top_candidate_identity_changed": before_top != after_top,
        "top_candidate_status_before": before_snapshot.get("top_candidate_status"),
        "top_candidate_status_after": after_snapshot.get("top_candidate_status"),
        "top_candidate_score_delta": _numeric_delta(
            before_snapshot.get("top_candidate_score"),
            after_snapshot.get("top_candidate_score"),
        ),
        "top_candidate_confidence_delta": _numeric_delta(
            before_snapshot.get("top_candidate_confidence"),
            after_snapshot.get("top_candidate_confidence"),
        ),
        "candidate_ordering_changed": before_sequence != after_sequence,
        "selection_status_before": before_snapshot.get("selection_status"),
        "selection_status_after": after_snapshot.get("selection_status"),
        "selection_status_changed": before_snapshot.get("selection_status")
        != after_snapshot.get("selection_status"),
        "abstain_path_before": before_snapshot.get("abstain_path_classification"),
        "abstain_path_after": after_snapshot.get("abstain_path_classification"),
        "abstain_path_changed": before_snapshot.get("abstain_path_classification")
        != after_snapshot.get("abstain_path_classification"),
        "reason_codes_before": before_snapshot.get("reason_codes"),
        "reason_codes_after": after_snapshot.get("reason_codes"),
        "abstain_reason_changed": before_snapshot.get("reason_codes")
        != after_snapshot.get("reason_codes")
        or before_snapshot.get("abstain_category")
        != after_snapshot.get("abstain_category"),
    }


def _build_persistent_abstain_reason(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    available_snapshots = {
        name: _safe_dict(snapshot)
        for name, snapshot in snapshots.items()
        if _safe_dict(snapshot).get("available")
    }
    all_available = len(available_snapshots) == len(_SNAPSHOT_ORDER)
    all_abstain = all(
        str(snapshot.get("selection_status") or "") == "abstain"
        for snapshot in available_snapshots.values()
    )
    reason_codes_by_snapshot = {
        name: _safe_str_list(snapshot.get("reason_codes"))
        for name, snapshot in available_snapshots.items()
    }
    categories_by_snapshot = {
        name: snapshot.get("abstain_category")
        for name, snapshot in available_snapshots.items()
    }
    reason_tuples = {tuple(value) for value in reason_codes_by_snapshot.values()}
    categories = {value for value in categories_by_snapshot.values()}

    return {
        "all_snapshots_available": all_available,
        "all_available_snapshots_abstain": all_abstain,
        "reason_codes_by_snapshot": reason_codes_by_snapshot,
        "abstain_category_by_snapshot": categories_by_snapshot,
        "persistent_reason_codes": list(next(iter(reason_tuples)))
        if all_available and all_abstain and len(reason_tuples) == 1 and reason_tuples
        else None,
        "persistent_abstain_category": next(iter(categories))
        if all_available and all_abstain and len(categories) == 1
        else None,
    }


def _build_shared_top_candidate_summary(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    top_identities = [
        _safe_dict(snapshot).get("top_candidate_identity")
        for snapshot in snapshots.values()
        if _safe_dict(snapshot).get("available")
    ]
    if len(top_identities) != len(_SNAPSHOT_ORDER):
        return None
    if any(identity is None for identity in top_identities):
        return None
    if len(set(top_identities)) != 1:
        return None

    identity = str(top_identities[0])
    return {
        "identity": identity,
        "by_snapshot": {
            snapshot_name: {
                "candidate_status": _safe_dict(snapshot).get("top_candidate_status"),
                "selection_score": _safe_dict(snapshot).get("top_candidate_score"),
                "selection_confidence": _safe_dict(snapshot).get(
                    "top_candidate_confidence"
                ),
                "reason_codes": _safe_dict(
                    _safe_dict(snapshot).get("top_candidate")
                ).get("reason_codes"),
                "gate_diagnostics": _safe_dict(
                    _safe_dict(snapshot).get("top_candidate")
                ).get("gate_diagnostics"),
            }
            for snapshot_name, snapshot in snapshots.items()
        },
    }


def _remaining_acceptance_deficit(
    *,
    normalized_snapshots: Mapping[str, Mapping[str, Any]],
    persistent_reason: Mapping[str, Any],
    shared_top_candidate: Mapping[str, Any] | None,
) -> str | None:
    if not shared_top_candidate:
        return None

    persistent_reason_codes = persistent_reason.get("persistent_reason_codes")
    if persistent_reason_codes:
        return f"persistent_engine_reason_codes={persistent_reason_codes}"

    top_reason_sets = {
        tuple(
            _safe_str_list(
                _safe_dict(_safe_dict(snapshot).get("top_candidate")).get(
                    "reason_codes"
                )
            )
        )
        for snapshot in normalized_snapshots.values()
    }
    if len(top_reason_sets) == 1:
        reason_codes = list(next(iter(top_reason_sets)))
        if reason_codes:
            return f"persistent_top_candidate_reason_codes={reason_codes}"

    top_statuses = {
        _safe_dict(snapshot).get("top_candidate_status")
        for snapshot in normalized_snapshots.values()
    }
    if len(top_statuses) == 1:
        status = next(iter(top_statuses))
        if status and status != "eligible":
            return f"persistent_top_candidate_status={status}"
        if status == "eligible":
            return "top_candidate_eligible_but_final_engine_abstain_reason_remains"

    return "shared_top_candidate_present_but_deficit_fields_are_mixed"


def _primary_abstain_path_classification(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> str:
    classifications = [
        str(_safe_dict(snapshot).get("abstain_path_classification") or "unavailable")
        for snapshot in snapshots.values()
        if _safe_dict(snapshot).get("available")
    ]
    if not classifications:
        return "unavailable"
    unique = set(classifications)
    if len(unique) == 1:
        return classifications[0]
    return "mixed_abstain_path"


def _build_minimum_question_answers(
    *,
    normalized_snapshots: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    primary_classification: str,
) -> dict[str, Any]:
    return {
        "snapshot_answers": {
            snapshot_name: _snapshot_question_answer(snapshot)
            for snapshot_name, snapshot in normalized_snapshots.items()
        },
        "candidate_count_deltas": {
            comparison_name: _safe_dict(comparison).get("candidate_count_delta")
            for comparison_name, comparison in comparisons.items()
        },
        "top_candidate_identity_changes": {
            comparison_name: _safe_dict(comparison).get(
                "top_candidate_identity_changed"
            )
            for comparison_name, comparison in comparisons.items()
        },
        "top_candidate_score_deltas": {
            comparison_name: _safe_dict(comparison).get("top_candidate_score_delta")
            for comparison_name, comparison in comparisons.items()
        },
        "candidate_ordering_changes": {
            comparison_name: _safe_dict(comparison).get("candidate_ordering_changed")
            for comparison_name, comparison in comparisons.items()
        },
        "acceptance_reason_changes": {
            comparison_name: _safe_dict(comparison).get("abstain_reason_changed")
            for comparison_name, comparison in comparisons.items()
        },
        "primary_abstain_path": primary_classification,
    }


def _snapshot_question_answer(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    item = _safe_dict(snapshot)
    classification = str(item.get("abstain_path_classification") or "unavailable")
    return {
        "selection_status": item.get("selection_status"),
        "candidate_count": item.get("candidate_count"),
        "ranking_count": item.get("ranking_count"),
        "candidate_presence": item.get("candidate_presence"),
        "top_candidate_identity": item.get("top_candidate_identity"),
        "top_candidate_status": item.get("top_candidate_status"),
        "top_candidate_score": item.get("top_candidate_score"),
        "top_candidate_confidence": item.get("top_candidate_confidence"),
        "reason_codes": item.get("reason_codes"),
        "abstain_category": item.get("abstain_category"),
        "candidate_formation_scarcity": classification
        == "candidate_formation_scarcity",
        "eligibility_or_acceptance_rejection": classification
        == "eligibility_or_acceptance_rejection",
        "selection_competition": classification == "selection_competition",
        "final_acceptance_boundary_or_guard": classification
        == "final_acceptance_boundary_or_guard",
        "explicit_engine_abstain_reason": classification,
    }


def _classify_abstain_path(
    *,
    available: bool,
    selection_status: str,
    reason_codes: Sequence[str],
    abstain_category: str,
    candidate_count: int,
    ranking_count: int,
    eligible_count: int,
) -> str:
    if not available:
        return "snapshot_unavailable"
    if selection_status == "selected":
        return "selected"
    if selection_status == "blocked":
        return "upstream_or_engine_blocked"
    if selection_status != "abstain":
        return "explicit_engine_non_abstain_status"

    reason_set = {str(reason) for reason in reason_codes}
    if (
        candidate_count <= 0
        and ranking_count <= 0
        or "NO_CANDIDATES_AVAILABLE" in reason_set
        or abstain_category == "no_candidates_available"
    ):
        return "candidate_formation_scarcity"
    if (
        "ALL_CANDIDATES_BLOCKED" in reason_set
        or "NO_ELIGIBLE_CANDIDATES" in reason_set
        or abstain_category
        in {"all_candidates_blocked", "no_eligible_candidates"}
        or eligible_count <= 0
    ):
        return "eligibility_or_acceptance_rejection"
    if (
        "TOP_CANDIDATES_TIED" in reason_set
        or abstain_category == "tied_top_candidates"
    ):
        return "selection_competition"
    if eligible_count > 0:
        return "final_acceptance_boundary_or_guard"
    return "explicit_engine_abstain_reason"


def _candidate_snapshot(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None
    item = _safe_dict(candidate)
    fields = (
        "rank",
        "symbol",
        "strategy",
        "horizon",
        "candidate_status",
        "selection_score",
        "selection_confidence",
        "reason_codes",
        "advisory_reason_codes",
        "selected_candidate_strength",
        "selected_stability_label",
        "drift_direction",
        "score_delta",
        "source_preference",
        "edge_stability_score",
        "latest_sample_size",
        "cumulative_sample_size",
        "symbol_cumulative_support",
        "strategy_cumulative_support",
        "aggregate_score",
        "sample_count",
        "median_future_return_pct",
        "positive_rate_pct",
        "robustness_signal_pct",
        "supporting_major_deficit_count",
        "ranking_signals",
        "gate_diagnostics",
    )
    return {field: item.get(field) for field in fields if field in item}


def _extract_top_candidate(
    *,
    snapshot: Mapping[str, Any],
    ranking: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if ranking:
        return ranking[0]
    abstain_diagnosis = _safe_dict(snapshot.get("abstain_diagnosis"))
    top_candidate = abstain_diagnosis.get("top_candidate")
    if isinstance(top_candidate, dict):
        return top_candidate
    for key in ("selected_candidate", "selected", "winner"):
        value = snapshot.get(key)
        if isinstance(value, dict):
            return value
    if snapshot.get("selected_symbol") or snapshot.get("selected_strategy"):
        return {
            "symbol": snapshot.get("selected_symbol"),
            "strategy": snapshot.get("selected_strategy"),
            "horizon": snapshot.get("selected_horizon"),
            "selection_score": snapshot.get("selected_score"),
            "selection_confidence": snapshot.get("selected_confidence"),
            "candidate_status": "selected"
            if snapshot.get("selection_status") == "selected"
            else None,
        }
    return None


def _candidate_identity(candidate: Mapping[str, Any] | None) -> str | None:
    if not isinstance(candidate, Mapping):
        return None
    symbol = candidate.get("symbol")
    strategy = candidate.get("strategy")
    horizon = candidate.get("horizon") or candidate.get("timeframe")
    parts = [str(value) for value in (symbol, strategy, horizon) if value]
    return " / ".join(parts) if parts else None


def _selected_candidate_identity(snapshot: Mapping[str, Any]) -> str | None:
    return _candidate_identity(
        {
            "symbol": snapshot.get("selected_symbol"),
            "strategy": snapshot.get("selected_strategy"),
            "horizon": snapshot.get("selected_horizon"),
        }
    )


def _candidate_status_counts(ranking: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {"eligible": 0, "penalized": 0, "blocked": 0}
    for item in ranking:
        status = item.get("candidate_status")
        if status in counts:
            counts[status] += 1
    return counts


def _snapshot_value(
    direct_summary: Mapping[str, Any],
    snapshot_name: str,
    field_name: str,
) -> Any:
    return _safe_dict(_safe_dict(direct_summary.get("snapshots")).get(snapshot_name)).get(
        field_name
    )


def _interpretation_conclusion(
    *,
    interpretation_status: str,
    primary_classification: str,
) -> str:
    if interpretation_status == "direct_edge_selection_abstain_path_explained":
        return (
            "Baseline, A1, and B1 remain abstain through the same exposed direct "
            f"edge-selection path: {primary_classification}."
        )
    if interpretation_status == "direct_edge_selection_abstain_path_explained_mixed":
        return (
            "Baseline, A1, and B1 all remain abstain, but exposed path classes differ "
            "across snapshots, so the report should be read as a mixed abstain path."
        )
    if interpretation_status == "direct_edge_selection_not_persistently_abstaining":
        return (
            "At least one snapshot selected a candidate, so the latest evidence does "
            "not represent a persistent baseline/A1/B1 abstain path."
        )
    if interpretation_status == "direct_edge_selection_abstain_path_incomplete":
        return (
            "At least one direct edge-selection snapshot was unavailable, so no full "
            "baseline/A1/B1 abstain-path conclusion is confirmed."
        )
    return "The exposed direct edge-selection fields were insufficient for a clean conclusion."


def _safe_ranking(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _safe_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _numeric_delta(before: Any, after: Any) -> float | int | None:
    if not isinstance(before, (int, float)) or isinstance(before, bool):
        return None
    if not isinstance(after, (int, float)) or isinstance(after, bool):
        return None
    delta = after - before
    return round(delta, 6) if isinstance(delta, float) else delta


def _coerce_int(value: Any, *, default: int | None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_path(path: Path) -> Path:
    return patch_class_b_module._resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return patch_class_b_module._parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return patch_class_b_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return patch_class_b_module._widest_configuration_summary(configuration_summaries)


def _safe_dict(value: Any) -> dict[str, Any]:
    return patch_class_b_module._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return patch_class_b_module._safe_list(value)


if __name__ == "__main__":
    main()
