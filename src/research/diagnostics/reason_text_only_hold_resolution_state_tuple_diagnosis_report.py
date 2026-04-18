from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_hold_resolution_reason_diagnosis_report as hold_reason_module,
)

REPORT_TYPE = "reason_text_only_hold_resolution_state_tuple_diagnosis_report"
REPORT_TITLE = "Reason Text Only Hold Resolution State Tuple Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
_MISSING_LABEL = "(missing)"

_PRIMARY_FACTOR_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_BREAKDOWN_SUPPORT_ROWS = 10

_REASON_TEXT_ONLY_EVIDENCE_SOURCE = "reason_text_only"

_NEUTRAL_OR_MISSING_ALIGNMENTS = {"neutral_or_missing_direction", "missing"}
_SUPPORTISH_ALIGNMENTS = {"confirmed", "same_direction_but_not_fully_confirmed"}

_CONTEXT_RELATION_ORDER = {
    "supports_selected_signal": 0,
    "neutral_like": 1,
    "unknown": 2,
    "conflicted": 3,
    "opposes_selected_signal": 4,
    _MISSING_LABEL: 9,
}
_ALIGNMENT_ORDER = {
    "confirmed": 0,
    "same_direction_but_not_fully_confirmed": 1,
    "neutral_or_missing_direction": 2,
    "missing": 3,
    "opposed": 4,
    _MISSING_LABEL: 9,
}
_REASON_CATEGORY_ORDER = {
    "confirmation_gap": 0,
    "directional_opposition": 1,
    "neutrality": 2,
    "conflict": 3,
    "multiple_reason_categories": 4,
    "no_classified_reason_category": 5,
}
_STATE_TUPLE_FAMILY_ORDER = {
    "dual_confirmation_absence": 0,
    "partial_confirmation_only": 1,
    "dual_confirmation_present_but_unpromoted": 2,
    "directional_resistance_present": 3,
    "mixed_alignment": 4,
    "no_reason_text_only_rows": 9,
    "insufficient_support": 10,
}

_CONFLICT_MARKERS = ("conflict", "conflicted", "conflicting", "mixed", "disagree")
_NEUTRALITY_MARKERS = ("neutral", "flat", "sideways", "range", "hold", "no_trade", "no-trade")


DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for rows that remain "
            "reason_text_only inside selected-strategy hold-resolution targeting, "
            "using state tuples built from context/setup/trigger alignments."
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
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = hold_reason_module.resolve_path(args.input)
    output_dir = hold_reason_module.resolve_path(args.output_dir)
    configurations = hold_reason_module.parse_configuration_values(args.config)

    result = run_reason_text_only_hold_resolution_state_tuple_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    final_assessment = _safe_dict(report.get("final_assessment"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["inputs"]["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    final_assessment.get("widest_configuration")
                ).get("display_name"),
                "primary_reason_text_only_state_tuple_family": final_assessment.get(
                    "primary_reason_text_only_state_tuple_family"
                ),
                "dominant_reason_text_only_state_tuple_family": final_assessment.get(
                    "dominant_reason_text_only_state_tuple_family"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_reason_text_only_hold_resolution_state_tuple_diagnosis_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
    write_report_copies: bool = False,
) -> dict[str, Any]:
    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    written_paths: dict[str, str] = {}
    if write_report_copies:
        written_paths = write_report_files(report, output_dir)
    return {
        "input_path": report["inputs"]["input_path"],
        "output_dir": report["inputs"]["output_dir"],
        "written_paths": written_paths,
        "report": report,
        "markdown": render_markdown(report),
    }


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
) -> dict[str, Any]:
    resolved_input = hold_reason_module.resolve_path(input_path)
    resolved_output = hold_reason_module.resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)

    configuration_summaries: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        run_output_dir = resolved_output / f"_{REPORT_TYPE}" / configuration.slug
        effective_input_path, raw_records, source_metadata = (
            hold_reason_module._materialize_configuration_input(
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
            )
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary.get("headline")) for summary in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the same effective input snapshot materialization as the hold-resolution reason report so row targeting stays consistent.",
            "Target rows are limited to rows already classified as selected-strategy hold-resolution targets whose hold-resolution evidence source is reason_text_only.",
            "State tuples are descriptive, not causal: they summarize observable context/setup/trigger relations present on reason_text_only rows without claiming the exact internal mechanism.",
            "The report keeps strategy-level support thresholds and conservative primary-family rules so fragmented tuple distributions remain mixed_or_inconclusive instead of being overclaimed.",
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
) -> dict[str, Any]:
    stage_rows = [
        hold_reason_module._build_stage_row(raw_record)
        for raw_record in raw_records
        if isinstance(raw_record, dict)
    ]
    reason_text_only_rows = [
        _build_reason_text_only_row(row)
        for row in stage_rows
        if row.get("selected_strategy_hold_resolution_target_row") is True
        and row.get("hold_resolution_evidence_source") == _REASON_TEXT_ONLY_EVIDENCE_SOURCE
    ]

    reason_text_only_targeting = build_reason_text_only_targeting(
        stage_rows=stage_rows,
        reason_text_only_rows=reason_text_only_rows,
    )
    observable_decision_layer_inventory = (
        hold_reason_module.build_observable_decision_layer_inventory(reason_text_only_rows)
    )
    reason_text_only_state_tuple_summary = build_state_tuple_summary(
        rows=reason_text_only_rows,
        support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
    )
    reason_text_only_by_strategy = build_reason_text_only_breakdown(
        rows=reason_text_only_rows,
        group_fields=("strategy",),
        support_threshold=_MIN_BREAKDOWN_SUPPORT_ROWS,
    )
    reason_text_only_by_strategy_symbol = build_reason_text_only_breakdown(
        rows=reason_text_only_rows,
        group_fields=("strategy", "symbol"),
        support_threshold=_MIN_BREAKDOWN_SUPPORT_ROWS,
    )
    state_tuple_family_reason_category_count_rows = build_state_tuple_family_reason_category_rows(
        rows=reason_text_only_rows
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
            "transition_row_count": len(stage_rows),
            "reason_text_only_target_row_count": reason_text_only_targeting[
                "reason_text_only_target_row_count"
            ],
            "reason_text_only_share_of_hold_resolution_target_rows": reason_text_only_targeting[
                "reason_text_only_share_of_hold_resolution_target_rows"
            ],
            "dominant_reason_text_only_state_tuple_family": reason_text_only_state_tuple_summary[
                "dominant_reason_text_only_state_tuple_family"
            ],
            "primary_reason_text_only_state_tuple_family": reason_text_only_state_tuple_summary[
                "primary_reason_text_only_state_tuple_family"
            ],
            "dominant_reason_text_category_label": reason_text_only_state_tuple_summary[
                "dominant_reason_text_category_label"
            ],
        },
        "reason_text_only_targeting": reason_text_only_targeting,
        "observable_decision_layer_inventory": observable_decision_layer_inventory,
        "reason_text_only_state_tuple_summary": reason_text_only_state_tuple_summary,
        "reason_text_only_by_strategy": reason_text_only_by_strategy,
        "reason_text_only_by_strategy_symbol": reason_text_only_by_strategy_symbol,
        "state_tuple_family_reason_category_count_rows": (
            state_tuple_family_reason_category_count_rows
        ),
        "confirmed_observations": _build_configuration_confirmed_observations(
            configuration=configuration,
            reason_text_only_targeting=reason_text_only_targeting,
            reason_text_only_state_tuple_summary=reason_text_only_state_tuple_summary,
            reason_text_only_by_strategy=reason_text_only_by_strategy,
        ),
        "evidence_backed_inferences": _build_configuration_evidence_backed_inferences(
            reason_text_only_targeting=reason_text_only_targeting,
            reason_text_only_state_tuple_summary=reason_text_only_state_tuple_summary,
            reason_text_only_by_strategy=reason_text_only_by_strategy,
        ),
        "unresolved_uncertainties": _build_configuration_unresolved_uncertainties(
            reason_text_only_state_tuple_summary=reason_text_only_state_tuple_summary,
        ),
    }


def build_reason_text_only_targeting(
    *,
    stage_rows: Sequence[dict[str, Any]],
    reason_text_only_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    hold_resolution_target_rows = 0
    structured_state_backed_hold_resolution_rows = 0
    reason_text_only_hold_resolution_rows = 0

    reason_category_counter: Counter[str] = Counter()

    for row in stage_rows:
        if row.get("selected_strategy_hold_resolution_target_row") is not True:
            continue
        hold_resolution_target_rows += 1

        evidence_source = row.get("hold_resolution_evidence_source")
        if evidence_source == _REASON_TEXT_ONLY_EVIDENCE_SOURCE:
            reason_text_only_hold_resolution_rows += 1
        elif evidence_source in {"structured_state_only", "structured_state_and_reason_text"}:
            structured_state_backed_hold_resolution_rows += 1

    for row in reason_text_only_rows:
        reason_category_counter[str(row.get("reason_text_category_label") or "no_classified_reason_category")] += 1

    return {
        "transition_row_count": len(stage_rows),
        "hold_resolution_target_row_count": hold_resolution_target_rows,
        "structured_state_backed_hold_resolution_row_count": (
            structured_state_backed_hold_resolution_rows
        ),
        "reason_text_only_target_row_count": len(reason_text_only_rows),
        "reason_text_only_share_of_hold_resolution_target_rows": _safe_ratio(
            len(reason_text_only_rows),
            hold_resolution_target_rows,
        ),
        "reason_text_category_label_count_rows": _counter_rows(
            reason_category_counter,
            key_name="reason_text_category_label",
            total=len(reason_text_only_rows),
            include_share=True,
            order_map=_REASON_CATEGORY_ORDER,
        ),
    }


def build_state_tuple_summary(
    *,
    rows: Sequence[dict[str, Any]],
    support_threshold: int,
) -> dict[str, Any]:
    state_tuple_counter: Counter[str] = Counter()
    state_tuple_family_counter: Counter[str] = Counter()
    context_relation_counter: Counter[str] = Counter()
    setup_alignment_counter: Counter[str] = Counter()
    trigger_alignment_counter: Counter[str] = Counter()
    reason_category_counter: Counter[str] = Counter()

    for row in rows:
        state_tuple_counter[str(row.get("state_tuple") or _MISSING_LABEL)] += 1
        state_tuple_family_counter[str(row.get("state_tuple_family") or "mixed_alignment")] += 1
        context_relation_counter[str(row.get("context_relation") or _MISSING_LABEL)] += 1
        setup_alignment_counter[str(row.get("setup_alignment") or _MISSING_LABEL)] += 1
        trigger_alignment_counter[str(row.get("trigger_alignment") or _MISSING_LABEL)] += 1
        reason_category_counter[str(row.get("reason_text_category_label") or "no_classified_reason_category")] += 1

    row_count = len(rows)
    return {
        "reason_text_only_target_row_count": row_count,
        "state_tuple_count_rows": _counter_rows(
            state_tuple_counter,
            key_name="state_tuple",
            total=row_count,
            include_share=True,
        ),
        "state_tuple_family_count_rows": _counter_rows(
            state_tuple_family_counter,
            key_name="state_tuple_family",
            total=row_count,
            include_share=True,
            order_map=_STATE_TUPLE_FAMILY_ORDER,
        ),
        "context_relation_count_rows": _counter_rows(
            context_relation_counter,
            key_name="context_relation",
            total=row_count,
            include_share=True,
            order_map=_CONTEXT_RELATION_ORDER,
        ),
        "setup_alignment_count_rows": _counter_rows(
            setup_alignment_counter,
            key_name="setup_alignment",
            total=row_count,
            include_share=True,
            order_map=_ALIGNMENT_ORDER,
        ),
        "trigger_alignment_count_rows": _counter_rows(
            trigger_alignment_counter,
            key_name="trigger_alignment",
            total=row_count,
            include_share=True,
            order_map=_ALIGNMENT_ORDER,
        ),
        "reason_text_category_label_count_rows": _counter_rows(
            reason_category_counter,
            key_name="reason_text_category_label",
            total=row_count,
            include_share=True,
            order_map=_REASON_CATEGORY_ORDER,
        ),
        "dominant_reason_text_only_state_tuple": _dominant_label(
            state_tuple_counter,
            empty="none",
        ),
        "primary_reason_text_only_state_tuple": _primary_label(
            state_tuple_counter,
            support_threshold=support_threshold,
            empty="no_reason_text_only_rows",
        ),
        "dominant_reason_text_only_state_tuple_family": _dominant_label(
            state_tuple_family_counter,
            empty="none",
            order_map=_STATE_TUPLE_FAMILY_ORDER,
        ),
        "primary_reason_text_only_state_tuple_family": _primary_label(
            state_tuple_family_counter,
            support_threshold=support_threshold,
            empty="no_reason_text_only_rows",
            order_map=_STATE_TUPLE_FAMILY_ORDER,
        ),
        "dominant_reason_text_category_label": _dominant_label(
            reason_category_counter,
            empty="none",
            order_map=_REASON_CATEGORY_ORDER,
        ),
        "support_status": (
            "supported" if row_count >= support_threshold else "limited_support"
        ),
    }


def build_reason_text_only_breakdown(
    *,
    rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    support_threshold: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_values = [_group_value(row, field) for field in group_fields]
        if any(value is None for value in group_values):
            continue
        grouped.setdefault(tuple(str(value) for value in group_values), []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        summary = build_state_tuple_summary(
            rows=grouped_rows,
            support_threshold=support_threshold,
        )
        output_rows.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "reason_text_only_target_row_count": summary[
                    "reason_text_only_target_row_count"
                ],
                "dominant_reason_text_only_state_tuple_family": summary[
                    "dominant_reason_text_only_state_tuple_family"
                ],
                "primary_reason_text_only_state_tuple_family": summary[
                    "primary_reason_text_only_state_tuple_family"
                ],
                "dominant_reason_text_category_label": summary[
                    "dominant_reason_text_category_label"
                ],
                "support_status": summary["support_status"],
                "state_tuple_family_count_rows": summary["state_tuple_family_count_rows"],
            }
        )

    output_rows.sort(
        key=lambda item: (
            -int(item.get("reason_text_only_target_row_count", 0) or 0),
            str(item.get("strategy") or ""),
            str(item.get("symbol") or ""),
        )
    )
    return output_rows


def build_state_tuple_family_reason_category_rows(
    *,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        family = str(row.get("state_tuple_family") or "mixed_alignment")
        category = str(
            row.get("reason_text_category_label") or "no_classified_reason_category"
        )
        counter[(family, category)] += 1

    total = len(rows)
    rendered: list[dict[str, Any]] = []
    for (family, category), count in counter.items():
        rendered.append(
            {
                "state_tuple_family": family,
                "reason_text_category_label": category,
                "count": count,
                "share": _safe_ratio(count, total),
            }
        )
    rendered.sort(
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            _STATE_TUPLE_FAMILY_ORDER.get(str(item.get("state_tuple_family")), 99),
            _REASON_CATEGORY_ORDER.get(str(item.get("reason_text_category_label")), 99),
            str(item.get("state_tuple_family") or ""),
            str(item.get("reason_text_category_label") or ""),
        )
    )
    return rendered


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "primary_reason_text_only_state_tuple_family": "none",
            "dominant_reason_text_only_state_tuple_family": "none",
            "widest_configuration": None,
            "supported_strategy_primary_state_tuple_families": {},
            "strategy_level_consistency": "unknown",
            "confirmed_observations": [],
            "evidence_backed_inferences": [],
            "unresolved_uncertainties": [],
            "overall_conclusion": "No configurations were evaluated.",
        }

    widest = max(
        summaries,
        key=lambda item: (
            int(_safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0),
            int(_safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0),
        ),
    )
    tuple_summary = _safe_dict(widest.get("reason_text_only_state_tuple_summary"))
    strategy_map = _supported_strategy_primary_state_tuple_families(
        widest.get("reason_text_only_by_strategy")
    )
    row_count = int(tuple_summary.get("reason_text_only_target_row_count", 0) or 0)
    primary_family = str(
        tuple_summary.get("primary_reason_text_only_state_tuple_family") or "none"
    )
    dominant_family = str(
        tuple_summary.get("dominant_reason_text_only_state_tuple_family") or "none"
    )

    return {
        "assessment": _overall_assessment_label(
            primary_family=primary_family,
            row_count=row_count,
        ),
        "primary_reason_text_only_state_tuple_family": primary_family,
        "dominant_reason_text_only_state_tuple_family": dominant_family,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "supported_strategy_primary_state_tuple_families": strategy_map,
        "strategy_level_consistency": _strategy_level_consistency(strategy_map),
        "confirmed_observations": _build_final_confirmed_observations(
            widest=widest,
            tuple_summary=tuple_summary,
            strategy_map=strategy_map,
        ),
        "evidence_backed_inferences": _build_final_evidence_backed_inferences(
            tuple_summary=tuple_summary,
            strategy_map=strategy_map,
        ),
        "unresolved_uncertainties": _build_final_unresolved_uncertainties(
            tuple_summary=tuple_summary,
        ),
        "overall_conclusion": _build_overall_conclusion(
            widest=widest,
            tuple_summary=tuple_summary,
            strategy_map=strategy_map,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('report_title', REPORT_TITLE)}",
        "",
        f"- Generated at: {report.get('generated_at', _MISSING_LABEL)}",
        f"- Input path: {_safe_dict(report.get('inputs')).get('input_path', _MISSING_LABEL)}",
        f"- Output dir: {_safe_dict(report.get('inputs')).get('output_dir', _MISSING_LABEL)}",
        "",
        "## Final Assessment",
    ]

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.extend(
        [
            f"- Assessment: {final_assessment.get('assessment', 'none')}",
            (
                "- Primary reason-text-only state tuple family: "
                f"{final_assessment.get('primary_reason_text_only_state_tuple_family', 'none')}"
            ),
            (
                "- Dominant reason-text-only state tuple family: "
                f"{final_assessment.get('dominant_reason_text_only_state_tuple_family', 'none')}"
            ),
            (
                "- Strategy-level consistency: "
                f"{final_assessment.get('strategy_level_consistency', 'unknown')}"
            ),
            "",
            "### Confirmed Observations",
        ]
    )
    for item in _safe_list(final_assessment.get("confirmed_observations")):
        lines.append(f"- {item}")

    lines.extend(["", "### Evidence-Backed Inferences"])
    for item in _safe_list(final_assessment.get("evidence_backed_inferences")):
        lines.append(f"- {item}")

    lines.extend(["", "### Unresolved Uncertainties"])
    for item in _safe_list(final_assessment.get("unresolved_uncertainties")):
        lines.append(f"- {item}")

    for summary in _safe_list(report.get("configuration_summaries")):
        summary_dict = _safe_dict(summary)
        configuration = _safe_dict(summary_dict.get("configuration"))
        tuple_summary = _safe_dict(summary_dict.get("reason_text_only_state_tuple_summary"))
        lines.extend(
            [
                "",
                f"## Configuration: {configuration.get('display_name', _MISSING_LABEL)}",
                (
                    f"- Reason-text-only target rows: "
                    f"{tuple_summary.get('reason_text_only_target_row_count', 0)}"
                ),
                (
                    "- Dominant state tuple family: "
                    f"{tuple_summary.get('dominant_reason_text_only_state_tuple_family', 'none')}"
                ),
                (
                    "- Primary state tuple family: "
                    f"{tuple_summary.get('primary_reason_text_only_state_tuple_family', 'none')}"
                ),
                (
                    "- Dominant reason category label: "
                    f"{tuple_summary.get('dominant_reason_text_category_label', 'none')}"
                ),
                (
                    "- State tuple family rows: "
                    f"{_format_counter_rows(tuple_summary.get('state_tuple_family_count_rows'), 'state_tuple_family')}"
                ),
                (
                    "- Strategy breakdown: "
                    f"{_format_breakdown_rows(summary_dict.get('reason_text_only_by_strategy'))}"
                ),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    md_path = output_dir / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def _build_reason_text_only_row(row: dict[str, Any]) -> dict[str, Any]:
    selected_signal = str(row.get("selected_strategy_result_signal_state") or "")
    context_relation = _context_relation(
        context_state=row.get("context_state"),
        context_bias=row.get("context_bias"),
        rule_bias=row.get("rule_bias"),
        selected_signal=selected_signal,
    )
    setup_alignment = hold_reason_module._layer_alignment(
        row.get("setup_state"),
        selected_signal,
    )
    trigger_alignment = hold_reason_module._layer_alignment(
        row.get("trigger_state"),
        selected_signal,
    )
    reason_text_category_label = _reason_text_category_label(
        row.get("hold_resolution_reason_text_categories")
    )
    state_tuple = (
        f"context={context_relation}|setup={setup_alignment}|trigger={trigger_alignment}"
    )
    state_tuple_family = _state_tuple_family(
        context_relation=context_relation,
        setup_alignment=setup_alignment,
        trigger_alignment=trigger_alignment,
    )
    return {
        **row,
        "context_relation": context_relation,
        "setup_alignment": setup_alignment,
        "trigger_alignment": trigger_alignment,
        "reason_text_category_label": reason_text_category_label,
        "state_tuple": state_tuple,
        "state_tuple_family": state_tuple_family,
    }


def _context_relation(
    *,
    context_state: Any,
    context_bias: Any,
    rule_bias: Any,
    selected_signal: str,
) -> str:
    context_state_text = _normalize_text(context_state)
    if _text_contains_any(context_state_text, _CONFLICT_MARKERS) or _normalize_text(rule_bias) == "neutral_conflict":
        return "conflicted"
    if _text_contains_any(context_state_text, _NEUTRALITY_MARKERS) or (
        context_state_text is None and _normalize_text(rule_bias) == "neutral"
    ):
        return "neutral_like"

    direction = hold_reason_module._direction_from_bias(_normalize_text(context_bias))
    if direction == selected_signal:
        return "supports_selected_signal"
    if direction is not None and direction != selected_signal:
        return "opposes_selected_signal"
    return "unknown"


def _reason_text_category_label(value: Any) -> str:
    categories = [
        item for item in _safe_list(value) if isinstance(item, str) and item.strip()
    ]
    if not categories:
        return "no_classified_reason_category"
    unique_categories = sorted(set(categories), key=lambda x: (_REASON_CATEGORY_ORDER.get(x, 99), x))
    if len(unique_categories) == 1:
        return unique_categories[0]
    return "multiple_reason_categories"


def _state_tuple_family(
    *,
    context_relation: str,
    setup_alignment: str,
    trigger_alignment: str,
) -> str:
    if (
        context_relation in {"conflicted", "opposes_selected_signal"}
        or setup_alignment == "opposed"
        or trigger_alignment == "opposed"
    ):
        return "directional_resistance_present"

    if (
        setup_alignment in _NEUTRAL_OR_MISSING_ALIGNMENTS
        and trigger_alignment in _NEUTRAL_OR_MISSING_ALIGNMENTS
    ):
        return "dual_confirmation_absence"

    if (
        setup_alignment in _SUPPORTISH_ALIGNMENTS
        and trigger_alignment in _SUPPORTISH_ALIGNMENTS
    ):
        return "dual_confirmation_present_but_unpromoted"

    if (
        setup_alignment in _SUPPORTISH_ALIGNMENTS
        and trigger_alignment in _NEUTRAL_OR_MISSING_ALIGNMENTS
    ) or (
        trigger_alignment in _SUPPORTISH_ALIGNMENTS
        and setup_alignment in _NEUTRAL_OR_MISSING_ALIGNMENTS
    ):
        return "partial_confirmation_only"

    return "mixed_alignment"


def _build_configuration_confirmed_observations(
    *,
    configuration: DiagnosisConfiguration,
    reason_text_only_targeting: dict[str, Any],
    reason_text_only_state_tuple_summary: dict[str, Any],
    reason_text_only_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    row_count = int(
        reason_text_only_state_tuple_summary.get("reason_text_only_target_row_count", 0)
        or 0
    )
    dominant_family = str(
        reason_text_only_state_tuple_summary.get("dominant_reason_text_only_state_tuple_family")
        or "none"
    )
    dominant_reason = str(
        reason_text_only_state_tuple_summary.get("dominant_reason_text_category_label")
        or "none"
    )
    reason_text_only_share = _to_float(
        reason_text_only_targeting.get("reason_text_only_share_of_hold_resolution_target_rows"),
        default=0.0,
    )

    observations = [
        (
            f"At {configuration.display_name}, {row_count} rows remained reason_text_only "
            "inside hold-resolution targeting."
        ),
        (
            "These rows represented "
            f"{reason_text_only_share:.2%} of all selected-strategy hold-resolution target rows."
        ),
        (
            f"The dominant reason-text-only state tuple family was {dominant_family}, "
            f"and the dominant reason category label was {dominant_reason}."
        ),
    ]

    strategy_map = _supported_strategy_primary_state_tuple_families(reason_text_only_by_strategy)
    if strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        observations.append(
            "Supported strategy-level primary tuple families were " + rendered + "."
        )

    return observations


def _build_configuration_evidence_backed_inferences(
    *,
    reason_text_only_targeting: dict[str, Any],
    reason_text_only_state_tuple_summary: dict[str, Any],
    reason_text_only_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    row_count = int(
        reason_text_only_state_tuple_summary.get("reason_text_only_target_row_count", 0)
        or 0
    )
    if row_count <= 0:
        return [
            "No reason_text_only rows were observed, so no state-tuple inference is available for this configuration."
        ]

    primary_family = str(
        reason_text_only_state_tuple_summary.get("primary_reason_text_only_state_tuple_family")
        or "no_reason_text_only_rows"
    )
    dominant_reason = str(
        reason_text_only_state_tuple_summary.get("dominant_reason_text_category_label")
        or "none"
    )
    strategy_map = _supported_strategy_primary_state_tuple_families(reason_text_only_by_strategy)

    inferences = [
        "Because these rows are already reason_text_only rows inside hold-resolution targeting, this report focuses on why structured state failed to promote them into explicit hold-resolution buckets.",
    ]
    if primary_family not in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        inferences.append(
            f"The most consistent descriptive state-tuple family for reason_text_only rows was {primary_family}."
        )
    if dominant_reason != "none":
        inferences.append(
            f"The dominant classified reason-text category among reason_text_only rows was {dominant_reason}."
        )
    if _strategy_level_consistency(strategy_map) == "split":
        inferences.append(
            "Strategy-level primary tuple families split across strategies, so reason_text_only rows should not yet be reduced to one universal structured-state failure pattern."
        )
    return inferences


def _build_configuration_unresolved_uncertainties(
    *,
    reason_text_only_state_tuple_summary: dict[str, Any],
) -> list[str]:
    row_count = int(
        reason_text_only_state_tuple_summary.get("reason_text_only_target_row_count", 0)
        or 0
    )
    uncertainties = [
        "This report does not prove whether reason-text-only rows reflect missing persisted structure, intentionally coarse persisted states, or a genuine mismatch between templated reasons and structured state layers.",
        "The state tuple families are descriptive summaries of observed context/setup/trigger relations, not proofs of the exact internal decision branch.",
    ]
    if row_count > 0:
        uncertainties.append(
            f"{row_count} reason_text_only rows still require deeper explanation if the goal is to isolate the exact internal hold-resolution mechanism."
        )
    return uncertainties


def _build_final_confirmed_observations(
    *,
    widest: dict[str, Any],
    tuple_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> list[str]:
    configuration = _safe_dict(widest.get("configuration"))
    row_count = int(tuple_summary.get("reason_text_only_target_row_count", 0) or 0)
    dominant_family = str(
        tuple_summary.get("dominant_reason_text_only_state_tuple_family") or "none"
    )
    dominant_reason = str(
        tuple_summary.get("dominant_reason_text_category_label") or "none"
    )
    observations = [
        (
            f"The widest supported configuration {configuration.get('display_name', _MISSING_LABEL)} "
            f"contained {row_count} reason_text_only rows inside hold-resolution targeting."
        ),
        (
            f"The dominant reason-text-only state tuple family was {dominant_family}, "
            f"while the dominant reason category label was {dominant_reason}."
        ),
    ]
    if strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        observations.append(
            "Supported strategy-level primary tuple families were " + rendered + "."
        )
    return observations


def _build_final_evidence_backed_inferences(
    *,
    tuple_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> list[str]:
    row_count = int(tuple_summary.get("reason_text_only_target_row_count", 0) or 0)
    if row_count <= 0:
        return [
            "No reason_text_only rows were present at the widest configuration, so no final state-tuple inference is available."
        ]

    primary_family = str(
        tuple_summary.get("primary_reason_text_only_state_tuple_family")
        or "no_reason_text_only_rows"
    )
    inferences = [
        "The reason_text_only subset can be separated from structured-state-backed rows and analyzed without changing the original hold-resolution target definition.",
    ]
    if primary_family not in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        inferences.append(
            f"The widest-configuration reason_text_only rows were most consistently described by the state tuple family {primary_family}."
        )
    if _strategy_level_consistency(strategy_map) == "split":
        inferences.append(
            "Strategy-level supported tuple families remain split, so the reason_text_only subset still does not support a single universal state-tuple explanation."
        )
    return inferences


def _build_final_unresolved_uncertainties(
    *,
    tuple_summary: dict[str, Any],
) -> list[str]:
    row_count = int(tuple_summary.get("reason_text_only_target_row_count", 0) or 0)
    uncertainties = [
        "This report still cannot prove whether reason-text-only rows are caused by missing persistence, coarse state serialization, or a mismatch between human-readable reason text and structured decision-layer states.",
        "The state tuple families do not prove whether the underlying live decision logic behaved as intended conservatism or unintended neutralization.",
    ]
    if row_count > 0:
        uncertainties.append(
            f"{row_count} reason_text_only rows still remain outside structured hold-resolution buckets and therefore warrant deeper diagnosis."
        )
    return uncertainties


def _build_overall_conclusion(
    *,
    widest: dict[str, Any],
    tuple_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> str:
    configuration = _safe_dict(widest.get("configuration"))
    row_count = int(tuple_summary.get("reason_text_only_target_row_count", 0) or 0)
    primary_family = str(
        tuple_summary.get("primary_reason_text_only_state_tuple_family")
        or "no_reason_text_only_rows"
    )
    strategy_consistency = _strategy_level_consistency(strategy_map)

    if row_count <= 0:
        return (
            "No reason_text_only rows were observed, so this artifact cannot add further state-tuple diagnosis beyond the structured hold-resolution report."
        )

    if primary_family in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            "reason_text_only rows remained too fragmented for a single dominant state-tuple family, "
            "so the descriptive explanation stayed mixed_or_inconclusive."
        )

    if strategy_consistency == "split" and strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            f"reason_text_only rows most often matched {primary_family}, but supported strategy slices remained split "
            f"({rendered}), so one universal structured-state explanation is still not proven."
        )

    return (
        f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
        f"reason_text_only rows were most consistently described by the state tuple family {primary_family}."
    )


def _dominant_label(
    counter: Counter[str],
    *,
    empty: str,
    order_map: dict[str, int] | None = None,
) -> str:
    if not counter:
        return empty
    return min(
        counter.items(),
        key=lambda item: (
            -item[1],
            (order_map or {}).get(item[0], 99),
            item[0],
        ),
    )[0]


def _primary_label(
    counter: Counter[str],
    *,
    support_threshold: int,
    empty: str,
    order_map: dict[str, int] | None = None,
) -> str:
    total = sum(counter.values())
    if total == 0:
        return empty
    if total < support_threshold:
        return "insufficient_support"
    dominant_count = max(counter.values())
    dominant = [key for key, value in counter.items() if value == dominant_count]
    if len(dominant) > 1 or _safe_ratio(dominant_count, total) < _PRIMARY_FACTOR_THRESHOLD:
        return "mixed_or_inconclusive"
    dominant.sort(key=lambda value: ((order_map or {}).get(value, 99), value))
    return dominant[0]


def _overall_assessment_label(*, primary_family: str, row_count: int) -> str:
    if row_count < _MIN_PRIMARY_SUPPORT_ROWS:
        return "insufficient_support"
    if primary_family in {"mixed_or_inconclusive", "insufficient_support", "no_reason_text_only_rows", "none"}:
        return primary_family
    return f"{primary_family}_primary"


def _supported_strategy_primary_state_tuple_families(rows: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _safe_list(rows):
        row_dict = _safe_dict(row)
        strategy = _normalize_strategy(row_dict.get("strategy"))
        if strategy is None:
            continue
        if row_dict.get("support_status") != "supported":
            continue
        family = str(row_dict.get("primary_reason_text_only_state_tuple_family") or "").strip()
        if family in {"", "mixed_or_inconclusive", "insufficient_support", "no_reason_text_only_rows"}:
            continue
        result[strategy] = family
    return result


def _strategy_level_consistency(strategy_map: dict[str, str]) -> str:
    if not strategy_map:
        return "unknown"
    unique_values = {value for value in strategy_map.values() if value}
    if len(unique_values) <= 1:
        return "aligned"
    return "split"


def _group_value(row: dict[str, Any], field: str) -> str | None:
    if field == "strategy":
        return _normalize_strategy(row.get("strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    raise ValueError(f"Unsupported group field: {field}")


def _text_contains_any(text: str | None, markers: Sequence[str]) -> bool:
    if text is None:
        return False
    return any(marker in text for marker in markers)


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _normalize_strategy(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _normalize_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _counter_rows(
    counter: Counter[str],
    *,
    key_name: str,
    total: int | None = None,
    include_share: bool = False,
    order_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        {
            key_name: key,
            "count": value,
            **(
                {"share": _safe_ratio(value, total)}
                if include_share and isinstance(total, int) and total > 0
                else {}
            ),
        }
        for key, value in counter.items()
    ]
    rows.sort(
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            (order_map or {}).get(str(item.get(key_name)), 99),
            str(item.get(key_name) or ""),
        )
    )
    return rows


def _format_counter_rows(value: Any, key_name: str) -> str:
    parts: list[str] = []
    for item in _safe_list(value):
        row = _safe_dict(item)
        label = row.get(key_name)
        count = row.get("count")
        share = row.get("share")
        if share is None:
            parts.append(f"{label}={count}")
        else:
            parts.append(f"{label}={count} ({share:.2%})")
    return ", ".join(parts) if parts else "none"


def _format_breakdown_rows(value: Any) -> str:
    parts: list[str] = []
    for item in _safe_list(value)[:5]:
        row = _safe_dict(item)
        strategy = row.get("strategy")
        symbol = row.get("symbol")
        family = row.get("primary_reason_text_only_state_tuple_family")
        count = row.get("reason_text_only_target_row_count")
        label_parts = [str(strategy)]
        if symbol is not None:
            label_parts.append(str(symbol))
        parts.append(f"{'/'.join(label_parts)}={family} ({count})")
    return ", ".join(parts) if parts else "none"


if __name__ == "__main__":
    main()