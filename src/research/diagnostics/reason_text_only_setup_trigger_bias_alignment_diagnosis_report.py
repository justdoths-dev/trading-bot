from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_hold_resolution_reason_diagnosis_report as hold_reason_module,
)

REPORT_TYPE = "reason_text_only_setup_trigger_bias_alignment_diagnosis_report"
REPORT_TITLE = "Reason Text Only Setup Trigger Bias Alignment Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
_REASON_TEXT_ONLY_EVIDENCE_SOURCE = "reason_text_only"
_MISSING_LABEL = "(missing)"

_PRIMARY_FACTOR_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_BREAKDOWN_SUPPORT_ROWS = 10

_RELATION_ORDER = {
    "supports_selected_signal": 0,
    "neutral_or_missing": 1,
    "unknown": 2,
    "opposes_selected_signal": 3,
    _MISSING_LABEL: 9,
}
_TEXT_CLASS_ORDER = {
    "missing": 0,
    "neutral_or_missing_direction": 1,
    "supports_selected_signal": 2,
    "opposes_selected_signal": 3,
    "other_directional_or_mixed": 4,
    _MISSING_LABEL: 9,
}
_FAMILY_ORDER = {
    "both_bias_support_selected_signal_but_state_text_non_directional": 0,
    "one_bias_supports_selected_signal_other_neutral_or_missing": 1,
    "both_bias_neutral_or_missing": 2,
    "one_or_more_biases_oppose_selected_signal": 3,
    "bias_state_mixed_or_unclassified": 4,
    "insufficient_support": 9,
    "no_reason_text_only_rows": 10,
}
_REASON_CATEGORY_ORDER = {
    "confirmation_gap": 0,
    "multiple_reason_categories": 1,
    "directional_opposition": 2,
    "neutrality": 3,
    "conflict": 4,
    "no_classified_reason_category": 5,
}

DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for reason_text_only hold-resolution rows "
            "by comparing setup/trigger raw bias fields against setup/trigger state text."
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

    result = run_reason_text_only_setup_trigger_bias_alignment_diagnosis_report(
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
                "primary_bias_alignment_family": final_assessment.get(
                    "primary_bias_alignment_family"
                ),
                "dominant_bias_alignment_family": final_assessment.get(
                    "dominant_bias_alignment_family"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_reason_text_only_setup_trigger_bias_alignment_diagnosis_report(
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
            "This report is diagnosis-only and reuses the same effective input snapshot materialization as the hold-resolution reports so row targeting stays consistent.",
            "Target rows are limited to rows already classified as selected-strategy hold-resolution targets whose hold-resolution evidence source is reason_text_only.",
            "The report compares setup_layer.bias / trigger_layer.bias against setup_layer.setup / trigger_layer.trigger text alignment without changing the original hold-resolution semantics.",
            "Bias alignment families are descriptive only and do not prove the exact internal live decision branch.",
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
        _build_reason_text_only_bias_row(row)
        for row in stage_rows
        if row.get("selected_strategy_hold_resolution_target_row") is True
        and row.get("hold_resolution_evidence_source") == _REASON_TEXT_ONLY_EVIDENCE_SOURCE
    ]

    reason_text_only_targeting = build_reason_text_only_targeting(
        stage_rows=stage_rows,
        rows=reason_text_only_rows,
    )
    bias_alignment_summary = build_bias_alignment_summary(
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
            "reason_text_only_target_row_count": reason_text_only_targeting[
                "reason_text_only_target_row_count"
            ],
            "reason_text_only_share_of_hold_resolution_target_rows": reason_text_only_targeting[
                "reason_text_only_share_of_hold_resolution_target_rows"
            ],
            "dominant_bias_alignment_family": bias_alignment_summary[
                "dominant_bias_alignment_family"
            ],
            "primary_bias_alignment_family": bias_alignment_summary[
                "primary_bias_alignment_family"
            ],
            "dominant_reason_text_category_label": bias_alignment_summary[
                "dominant_reason_text_category_label"
            ],
        },
        "reason_text_only_targeting": reason_text_only_targeting,
        "bias_alignment_summary": bias_alignment_summary,
        "reason_text_only_by_strategy": reason_text_only_by_strategy,
        "reason_text_only_by_strategy_symbol": reason_text_only_by_strategy_symbol,
        "confirmed_observations": _build_configuration_confirmed_observations(
            configuration=configuration,
            reason_text_only_targeting=reason_text_only_targeting,
            bias_alignment_summary=bias_alignment_summary,
            reason_text_only_by_strategy=reason_text_only_by_strategy,
        ),
        "evidence_backed_inferences": _build_configuration_evidence_backed_inferences(
            bias_alignment_summary=bias_alignment_summary,
            reason_text_only_by_strategy=reason_text_only_by_strategy,
        ),
        "unresolved_uncertainties": _build_configuration_unresolved_uncertainties(
            bias_alignment_summary=bias_alignment_summary,
        ),
    }


def build_reason_text_only_targeting(
    *,
    stage_rows: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    hold_resolution_target_rows = 0
    for row in stage_rows:
        if row.get("selected_strategy_hold_resolution_target_row") is True:
            hold_resolution_target_rows += 1

    reason_category_counter: Counter[str] = Counter()
    for row in rows:
        reason_category_counter[str(row.get("reason_text_category_label") or "no_classified_reason_category")] += 1

    return {
        "transition_row_count": len(stage_rows),
        "hold_resolution_target_row_count": hold_resolution_target_rows,
        "reason_text_only_target_row_count": len(rows),
        "reason_text_only_share_of_hold_resolution_target_rows": _safe_ratio(
            len(rows),
            hold_resolution_target_rows,
        ),
        "reason_text_category_label_count_rows": _counter_rows(
            reason_category_counter,
            key_name="reason_text_category_label",
            total=len(rows),
            include_share=True,
            order_map=_REASON_CATEGORY_ORDER,
        ),
    }


def build_bias_alignment_summary(
    *,
    rows: Sequence[dict[str, Any]],
    support_threshold: int,
) -> dict[str, Any]:
    family_counter: Counter[str] = Counter()
    setup_bias_relation_counter: Counter[str] = Counter()
    trigger_bias_relation_counter: Counter[str] = Counter()
    setup_text_class_counter: Counter[str] = Counter()
    trigger_text_class_counter: Counter[str] = Counter()
    reason_category_counter: Counter[str] = Counter()
    pair_counter: Counter[str] = Counter()

    for row in rows:
        family_counter[str(row.get("bias_alignment_family") or "bias_state_mixed_or_unclassified")] += 1
        setup_bias_relation_counter[str(row.get("setup_bias_relation") or _MISSING_LABEL)] += 1
        trigger_bias_relation_counter[str(row.get("trigger_bias_relation") or _MISSING_LABEL)] += 1
        setup_text_class_counter[str(row.get("setup_text_class") or _MISSING_LABEL)] += 1
        trigger_text_class_counter[str(row.get("trigger_text_class") or _MISSING_LABEL)] += 1
        reason_category_counter[str(row.get("reason_text_category_label") or "no_classified_reason_category")] += 1
        pair_counter[
            f"setup_bias={row.get('setup_bias_relation') or _MISSING_LABEL}"
            f"|trigger_bias={row.get('trigger_bias_relation') or _MISSING_LABEL}"
            f"|setup_text={row.get('setup_text_class') or _MISSING_LABEL}"
            f"|trigger_text={row.get('trigger_text_class') or _MISSING_LABEL}"
        ] += 1

    row_count = len(rows)
    return {
        "reason_text_only_target_row_count": row_count,
        "bias_alignment_family_count_rows": _counter_rows(
            family_counter,
            key_name="bias_alignment_family",
            total=row_count,
            include_share=True,
            order_map=_FAMILY_ORDER,
        ),
        "setup_bias_relation_count_rows": _counter_rows(
            setup_bias_relation_counter,
            key_name="setup_bias_relation",
            total=row_count,
            include_share=True,
            order_map=_RELATION_ORDER,
        ),
        "trigger_bias_relation_count_rows": _counter_rows(
            trigger_bias_relation_counter,
            key_name="trigger_bias_relation",
            total=row_count,
            include_share=True,
            order_map=_RELATION_ORDER,
        ),
        "setup_text_class_count_rows": _counter_rows(
            setup_text_class_counter,
            key_name="setup_text_class",
            total=row_count,
            include_share=True,
            order_map=_TEXT_CLASS_ORDER,
        ),
        "trigger_text_class_count_rows": _counter_rows(
            trigger_text_class_counter,
            key_name="trigger_text_class",
            total=row_count,
            include_share=True,
            order_map=_TEXT_CLASS_ORDER,
        ),
        "reason_text_category_label_count_rows": _counter_rows(
            reason_category_counter,
            key_name="reason_text_category_label",
            total=row_count,
            include_share=True,
            order_map=_REASON_CATEGORY_ORDER,
        ),
        "bias_text_pair_count_rows": _counter_rows(
            pair_counter,
            key_name="bias_text_pair",
            total=row_count,
            include_share=True,
        ),
        "dominant_bias_alignment_family": _dominant_label(
            family_counter,
            empty="none",
            order_map=_FAMILY_ORDER,
        ),
        "primary_bias_alignment_family": _primary_label(
            family_counter,
            support_threshold=support_threshold,
            empty="no_reason_text_only_rows",
            order_map=_FAMILY_ORDER,
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
        summary = build_bias_alignment_summary(
            rows=grouped_rows,
            support_threshold=support_threshold,
        )
        output_rows.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "reason_text_only_target_row_count": summary[
                    "reason_text_only_target_row_count"
                ],
                "dominant_bias_alignment_family": summary[
                    "dominant_bias_alignment_family"
                ],
                "primary_bias_alignment_family": summary[
                    "primary_bias_alignment_family"
                ],
                "dominant_reason_text_category_label": summary[
                    "dominant_reason_text_category_label"
                ],
                "support_status": summary["support_status"],
                "bias_alignment_family_count_rows": summary[
                    "bias_alignment_family_count_rows"
                ],
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


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "primary_bias_alignment_family": "none",
            "dominant_bias_alignment_family": "none",
            "widest_configuration": None,
            "supported_strategy_primary_bias_alignment_families": {},
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
    summary = _safe_dict(widest.get("bias_alignment_summary"))
    strategy_map = _supported_strategy_primary_bias_alignment_families(
        widest.get("reason_text_only_by_strategy")
    )
    row_count = int(summary.get("reason_text_only_target_row_count", 0) or 0)
    primary_family = str(summary.get("primary_bias_alignment_family") or "none")
    dominant_family = str(summary.get("dominant_bias_alignment_family") or "none")

    return {
        "assessment": _overall_assessment_label(
            primary_family=primary_family,
            row_count=row_count,
        ),
        "primary_bias_alignment_family": primary_family,
        "dominant_bias_alignment_family": dominant_family,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "supported_strategy_primary_bias_alignment_families": strategy_map,
        "strategy_level_consistency": _strategy_level_consistency(strategy_map),
        "confirmed_observations": _build_final_confirmed_observations(
            widest=widest,
            bias_alignment_summary=summary,
            strategy_map=strategy_map,
        ),
        "evidence_backed_inferences": _build_final_evidence_backed_inferences(
            bias_alignment_summary=summary,
            strategy_map=strategy_map,
        ),
        "unresolved_uncertainties": _build_final_unresolved_uncertainties(
            bias_alignment_summary=summary,
        ),
        "overall_conclusion": _build_overall_conclusion(
            widest=widest,
            bias_alignment_summary=summary,
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
                "- Primary bias-alignment family: "
                f"{final_assessment.get('primary_bias_alignment_family', 'none')}"
            ),
            (
                "- Dominant bias-alignment family: "
                f"{final_assessment.get('dominant_bias_alignment_family', 'none')}"
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
        bias_summary = _safe_dict(summary_dict.get("bias_alignment_summary"))
        lines.extend(
            [
                "",
                f"## Configuration: {configuration.get('display_name', _MISSING_LABEL)}",
                (
                    f"- Reason-text-only target rows: "
                    f"{bias_summary.get('reason_text_only_target_row_count', 0)}"
                ),
                (
                    "- Dominant bias-alignment family: "
                    f"{bias_summary.get('dominant_bias_alignment_family', 'none')}"
                ),
                (
                    "- Primary bias-alignment family: "
                    f"{bias_summary.get('primary_bias_alignment_family', 'none')}"
                ),
                (
                    "- Dominant reason category label: "
                    f"{bias_summary.get('dominant_reason_text_category_label', 'none')}"
                ),
                (
                    "- Bias-alignment family rows: "
                    f"{_format_counter_rows(bias_summary.get('bias_alignment_family_count_rows'), 'bias_alignment_family')}"
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


def _build_reason_text_only_bias_row(row: dict[str, Any]) -> dict[str, Any]:
    selected_signal = str(row.get("selected_strategy_result_signal_state") or "")
    setup_bias = _normalize_text(
        _safe_dict(row.get("setup_layer_payload")).get("bias")
    )
    trigger_bias = _normalize_text(
        _safe_dict(row.get("trigger_layer_payload")).get("bias")
    )

    setup_bias_relation = _bias_relation(setup_bias, selected_signal)
    trigger_bias_relation = _bias_relation(trigger_bias, selected_signal)
    setup_text_class = _text_alignment_class(row.get("setup_state"), selected_signal)
    trigger_text_class = _text_alignment_class(row.get("trigger_state"), selected_signal)

    reason_text_category_label = _reason_text_category_label(
        row.get("hold_resolution_reason_text_categories")
    )

    bias_alignment_family = _bias_alignment_family(
        setup_bias_relation=setup_bias_relation,
        trigger_bias_relation=trigger_bias_relation,
        setup_text_class=setup_text_class,
        trigger_text_class=trigger_text_class,
    )

    return {
        **row,
        "setup_bias_relation": setup_bias_relation,
        "trigger_bias_relation": trigger_bias_relation,
        "setup_text_class": setup_text_class,
        "trigger_text_class": trigger_text_class,
        "reason_text_category_label": reason_text_category_label,
        "bias_alignment_family": bias_alignment_family,
    }


def _bias_relation(value: str | None, selected_signal: str) -> str:
    direction = hold_reason_module._direction_from_bias(value)
    if direction is None:
        return "neutral_or_missing"
    if direction == selected_signal:
        return "supports_selected_signal"
    return "opposes_selected_signal"


def _text_alignment_class(value: Any, selected_signal: str) -> str:
    alignment = hold_reason_module._layer_alignment(value, selected_signal)
    if alignment == "confirmed":
        return "supports_selected_signal"
    if alignment == "opposed":
        return "opposes_selected_signal"
    if alignment in {"neutral_or_missing_direction", "missing"}:
        return "neutral_or_missing_direction"
    if alignment == "same_direction_but_not_fully_confirmed":
        return "other_directional_or_mixed"
    return "other_directional_or_mixed"


def _reason_text_category_label(value: Any) -> str:
    categories = [
        item for item in _safe_list(value) if isinstance(item, str) and item.strip()
    ]
    if not categories:
        return "no_classified_reason_category"
    unique_categories = sorted(
        set(categories),
        key=lambda x: (_REASON_CATEGORY_ORDER.get(x, 99), x),
    )
    if len(unique_categories) == 1:
        return unique_categories[0]
    return "multiple_reason_categories"


def _bias_alignment_family(
    *,
    setup_bias_relation: str,
    trigger_bias_relation: str,
    setup_text_class: str,
    trigger_text_class: str,
) -> str:
    bias_relations = {setup_bias_relation, trigger_bias_relation}
    text_classes = {setup_text_class, trigger_text_class}

    if "opposes_selected_signal" in bias_relations:
        return "one_or_more_biases_oppose_selected_signal"

    if (
        setup_bias_relation == "supports_selected_signal"
        and trigger_bias_relation == "supports_selected_signal"
        and setup_text_class == "neutral_or_missing_direction"
        and trigger_text_class == "neutral_or_missing_direction"
    ):
        return "both_bias_support_selected_signal_but_state_text_non_directional"

    if (
        "supports_selected_signal" in bias_relations
        and "neutral_or_missing" in bias_relations
        and text_classes == {"neutral_or_missing_direction"}
    ):
        return "one_bias_supports_selected_signal_other_neutral_or_missing"

    if (
        setup_bias_relation == "neutral_or_missing"
        and trigger_bias_relation == "neutral_or_missing"
        and text_classes == {"neutral_or_missing_direction"}
    ):
        return "both_bias_neutral_or_missing"

    return "bias_state_mixed_or_unclassified"


def _build_configuration_confirmed_observations(
    *,
    configuration: DiagnosisConfiguration,
    reason_text_only_targeting: dict[str, Any],
    bias_alignment_summary: dict[str, Any],
    reason_text_only_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    dominant_family = str(bias_alignment_summary.get("dominant_bias_alignment_family") or "none")
    dominant_reason = str(
        bias_alignment_summary.get("dominant_reason_text_category_label") or "none"
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
            f"These rows represented {reason_text_only_share:.2%} of all selected-strategy hold-resolution target rows."
        ),
        (
            f"The dominant bias-alignment family was {dominant_family}, and the dominant reason category label was {dominant_reason}."
        ),
    ]

    strategy_map = _supported_strategy_primary_bias_alignment_families(reason_text_only_by_strategy)
    if strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        observations.append(
            "Supported strategy-level primary bias-alignment families were " + rendered + "."
        )
    return observations


def _build_configuration_evidence_backed_inferences(
    *,
    bias_alignment_summary: dict[str, Any],
    reason_text_only_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    if row_count <= 0:
        return [
            "No reason_text_only rows were observed, so no setup-trigger bias alignment inference is available for this configuration."
        ]

    primary_family = str(bias_alignment_summary.get("primary_bias_alignment_family") or "no_reason_text_only_rows")
    strategy_map = _supported_strategy_primary_bias_alignment_families(reason_text_only_by_strategy)
    inferences = [
        "Because these rows are already reason_text_only rows inside hold-resolution targeting, this report focuses on whether setup/trigger bias fields preserve direction that the setup/trigger text fields fail to carry.",
    ]
    if primary_family not in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        inferences.append(
            f"The most consistent descriptive bias-vs-text family for reason_text_only rows was {primary_family}."
        )
    if _strategy_level_consistency(strategy_map) == "split":
        inferences.append(
            "Strategy-level primary bias-alignment families split across strategies, so the reason_text_only subset still does not support one universal persistence explanation."
        )
    return inferences


def _build_configuration_unresolved_uncertainties(
    *,
    bias_alignment_summary: dict[str, Any],
) -> list[str]:
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    uncertainties = [
        "This report does not prove whether the observed bias-vs-text pattern is caused by intended coarse serialization, missing persistence, or a genuine decision-layer mismatch.",
        "The bias-alignment families are descriptive summaries of persisted setup/trigger fields, not proofs of the exact internal live decision branch.",
    ]
    if row_count > 0:
        uncertainties.append(
            f"{row_count} reason_text_only rows still require deeper diagnosis if the goal is to isolate the exact internal hold-resolution mechanism."
        )
    return uncertainties


def _build_final_confirmed_observations(
    *,
    widest: dict[str, Any],
    bias_alignment_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> list[str]:
    configuration = _safe_dict(widest.get("configuration"))
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    dominant_family = str(bias_alignment_summary.get("dominant_bias_alignment_family") or "none")
    dominant_reason = str(
        bias_alignment_summary.get("dominant_reason_text_category_label") or "none"
    )
    observations = [
        (
            f"The widest supported configuration {configuration.get('display_name', _MISSING_LABEL)} "
            f"contained {row_count} reason_text_only rows inside hold-resolution targeting."
        ),
        (
            f"The dominant bias-alignment family was {dominant_family}, while the dominant reason category label was {dominant_reason}."
        ),
    ]
    if strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        observations.append(
            "Supported strategy-level primary bias-alignment families were " + rendered + "."
        )
    return observations


def _build_final_evidence_backed_inferences(
    *,
    bias_alignment_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> list[str]:
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    if row_count <= 0:
        return [
            "No reason_text_only rows were present at the widest configuration, so no final setup-trigger bias alignment inference is available."
        ]

    primary_family = str(
        bias_alignment_summary.get("primary_bias_alignment_family") or "no_reason_text_only_rows"
    )
    inferences = [
        "The reason_text_only subset can be separated from structured-state-backed rows and examined specifically for setup/trigger bias-vs-text mismatch without changing the original hold-resolution target definition.",
    ]
    if primary_family not in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        inferences.append(
            f"The widest-configuration reason_text_only rows were most consistently described by the bias-alignment family {primary_family}."
        )
    if _strategy_level_consistency(strategy_map) == "split":
        inferences.append(
            "Strategy-level supported families remain split, so the reason_text_only subset still does not support a single universal persistence explanation."
        )
    return inferences


def _build_final_unresolved_uncertainties(
    *,
    bias_alignment_summary: dict[str, Any],
) -> list[str]:
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    uncertainties = [
        "This report still cannot prove whether reason_text_only rows are caused by missing persistence, intentionally coarse state serialization, or a deeper mismatch between human-readable reasons and structured setup/trigger states.",
        "The bias-alignment families do not prove whether the underlying live decision logic behaved as intended conservatism or unintended neutralization.",
    ]
    if row_count > 0:
        uncertainties.append(
            f"{row_count} reason_text_only rows remain outside structured hold-resolution buckets and therefore warrant deeper diagnosis."
        )
    return uncertainties


def _build_overall_conclusion(
    *,
    widest: dict[str, Any],
    bias_alignment_summary: dict[str, Any],
    strategy_map: dict[str, str],
) -> str:
    configuration = _safe_dict(widest.get("configuration"))
    row_count = int(bias_alignment_summary.get("reason_text_only_target_row_count", 0) or 0)
    primary_family = str(
        bias_alignment_summary.get("primary_bias_alignment_family") or "no_reason_text_only_rows"
    )
    strategy_consistency = _strategy_level_consistency(strategy_map)

    if row_count <= 0:
        return (
            "No reason_text_only rows were observed, so this artifact cannot add further setup-trigger bias alignment diagnosis."
        )

    if primary_family in {"insufficient_support", "mixed_or_inconclusive", "no_reason_text_only_rows", "none"}:
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            "reason_text_only rows remained too fragmented for a single dominant setup-trigger bias-alignment family."
        )

    if strategy_consistency == "split" and strategy_map:
        rendered = ", ".join(
            f"{strategy}={family}" for strategy, family in sorted(strategy_map.items())
        )
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            f"reason_text_only rows most often matched {primary_family}, but supported strategy slices remained split "
            f"({rendered}), so one universal persistence explanation is still not proven."
        )

    return (
        f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
        f"reason_text_only rows were most consistently described by the setup-trigger bias-alignment family {primary_family}."
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


def _supported_strategy_primary_bias_alignment_families(rows: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _safe_list(rows):
        row_dict = _safe_dict(row)
        strategy = _normalize_text(row_dict.get("strategy"))
        if strategy is None:
            continue
        if row_dict.get("support_status") != "supported":
            continue
        family = str(row_dict.get("primary_bias_alignment_family") or "").strip()
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
        return _normalize_text(row.get("strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    raise ValueError(f"Unsupported group field: {field}")


def _normalize_text(value: Any) -> str | None:
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


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        family = row.get("primary_bias_alignment_family")
        count = row.get("reason_text_only_target_row_count")
        label_parts = [str(strategy)]
        if symbol is not None:
            label_parts.append(str(symbol))
        parts.append(f"{'/'.join(label_parts)}={family} ({count})")
    return ", ".join(parts) if parts else "none"


if __name__ == "__main__":
    main()