from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_hold_resolution_reason_diagnosis_report as hold_reason_module,
)

REPORT_TYPE = "selected_strategy_setup_trigger_activation_gap_diagnosis_report"
REPORT_TITLE = "Selected Strategy Setup Trigger Activation Gap Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_MIN_SYMBOL_SUPPORT = 10

_ACTIONABLE_SIGNAL_STATES = {"long", "short"}
_MISSING_LABEL = "(missing)"

_GROUP_ALL_ACTIONABLE = "all_actionable_selected_strategy_rows"
_GROUP_PRESERVED = "preserved_aligned_actionable"
_GROUP_COLLAPSED = "collapsed_to_hold"
_GROUP_OTHER_ACTIONABLE = "other_actionable_rule_outcome"
_GROUP_OTHER_NON_ACTIONABLE = "other_non_actionable_rule_outcome"

_GROUP_ORDER = {
    _GROUP_ALL_ACTIONABLE: 0,
    _GROUP_PRESERVED: 1,
    _GROUP_COLLAPSED: 2,
    _GROUP_OTHER_ACTIONABLE: 3,
    _GROUP_OTHER_NON_ACTIONABLE: 4,
}
_ACTIVATION_PATTERN_ORDER = {
    "dual_aligned_with_selected": 0,
    "setup_only_aligned": 1,
    "trigger_only_aligned": 2,
    "dual_neutral_or_missing": 3,
    "any_opposition": 4,
    "mixed_or_inconclusive": 5,
    "insufficient_support": 9,
    "none": 10,
}

_PRIMARY_PATTERN_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_STRATEGY_SUPPORT_ROWS = 10

_NEUTRAL_STATE_TOKENS = (
    "neutral",
    "missing",
    "none",
    "unknown",
    "unavailable",
    "not_ready",
    "not-ready",
    "no_signal",
    "no-signal",
    "no_confirmation",
    "no-confirmation",
    "hold",
)
_MIXED_STATE_TOKENS = (
    "mixed",
    "inconclusive",
    "conflict",
    "conflicting",
    "divergent",
    "both",
)
_DIRECTION_TOKENS = {
    "long": ("long", "bullish", "buy"),
    "short": ("short", "bearish", "sell"),
}
_NEUTRAL_BIAS_TOKENS = {
    "",
    "neutral",
    "none",
    "missing",
    "unknown",
    "n/a",
    "na",
}

DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report comparing setup/trigger activation on "
            "actionable selected-strategy rows across preserved actionable, "
            "collapsed-to-hold, and other actionable rule outcomes."
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
        help="Minimum rows required for strategy+symbol+comparison_group summaries.",
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

    result = run_selected_strategy_setup_trigger_activation_gap_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    final_assessment = _safe_dict(report.get("final_assessment"))
    summary = _safe_dict(report.get("summary"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    final_assessment.get("widest_configuration")
                ).get("display_name"),
                "actionable_selected_strategy_row_count": summary.get(
                    "actionable_selected_strategy_row_count",
                    0,
                ),
                "preserved_aligned_actionable_row_count": summary.get(
                    "preserved_aligned_actionable_row_count",
                    0,
                ),
                "collapsed_to_hold_row_count": summary.get(
                    "collapsed_to_hold_row_count",
                    0,
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_setup_trigger_activation_gap_diagnosis_report(
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
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary.get("headline")) for summary in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "summary": _safe_dict(widest_summary.get("summary")),
        "overall_group_summaries": _safe_list(
            widest_summary.get("overall_group_summaries")
        ),
        "strategy_group_summaries": _safe_list(
            widest_summary.get("strategy_group_summaries")
        ),
        "symbol_group_summaries": _safe_list(
            widest_summary.get("symbol_group_summaries")
        ),
        "activation_pattern_group_summaries": _safe_list(
            widest_summary.get("activation_pattern_group_summaries")
        ),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the same effective input snapshot materialization and stage-row extraction interface used by the nearby hold-resolution reports, while keeping those dependencies behind local wrappers.",
            "Activation is classified conservatively from setup/trigger state first; directional bias without supportive setup/trigger state is retained as mixed_or_inconclusive instead of being promoted to aligned_with_selected.",
            "Preserved aligned actionable rows and collapsed-to-hold rows are the primary comparison groups; other actionable and other non-actionable rule outcomes are tracked separately so they do not contaminate the preserved-vs-collapsed comparison.",
            "The evidence-source split in this report is setup/trigger-centered: structured_state_backed requires observable setup/trigger state or setup/trigger bias, not context-layer structure alone.",
            "The configurable symbol support threshold is a reporting guardrail for diagnosis output only and is not a production recommendation.",
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
        if row.get("selected_strategy_result_signal_state") in _ACTIONABLE_SIGNAL_STATES
    ]
    overall_group_summaries = build_overall_group_summaries(actionable_rows)
    strategy_group_summaries = build_group_summaries(
        actionable_rows,
        group_fields=("strategy", "comparison_group"),
        support_threshold=_MIN_STRATEGY_SUPPORT_ROWS,
    )
    symbol_group_summaries = build_group_summaries(
        actionable_rows,
        group_fields=("symbol", "strategy", "comparison_group"),
        support_threshold=max(1, min_symbol_support),
        min_row_count=max(1, min_symbol_support),
    )
    activation_pattern_group_summaries = build_group_summaries(
        actionable_rows,
        group_fields=("comparison_group", "activation_pattern"),
        support_threshold=1,
    )
    summary = build_summary(
        actionable_rows=actionable_rows,
        overall_group_summaries=overall_group_summaries,
        min_symbol_support=min_symbol_support,
    )
    key_observations = build_key_observations(
        summary=summary,
        overall_group_summaries=overall_group_summaries,
        strategy_group_summaries=strategy_group_summaries,
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
            "preserved_aligned_actionable_row_count": summary[
                "preserved_aligned_actionable_row_count"
            ],
            "collapsed_to_hold_row_count": summary["collapsed_to_hold_row_count"],
            "other_actionable_rule_outcome_row_count": summary[
                "other_actionable_rule_outcome_row_count"
            ],
            "other_non_actionable_rule_outcome_row_count": summary[
                "other_non_actionable_rule_outcome_row_count"
            ],
            "preserved_dual_activation_rate": _safe_dict(
                _group_map(overall_group_summaries).get(_GROUP_PRESERVED)
            ).get("dual_activation_rate", 0.0),
            "collapsed_dual_activation_rate": _safe_dict(
                _group_map(overall_group_summaries).get(_GROUP_COLLAPSED)
            ).get("dual_activation_rate", 0.0),
            "collapsed_dual_neutral_or_missing_rate": _safe_dict(
                _group_map(overall_group_summaries).get(_GROUP_COLLAPSED)
            ).get("dual_neutral_or_missing_rate", 0.0),
        },
        "summary": summary,
        "overall_group_summaries": overall_group_summaries,
        "strategy_group_summaries": strategy_group_summaries,
        "symbol_group_summaries": symbol_group_summaries,
        "activation_pattern_group_summaries": activation_pattern_group_summaries,
        "key_observations": key_observations,
    }


def build_overall_group_summaries(
    actionable_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_rows = [
        {"comparison_group": _GROUP_ALL_ACTIONABLE, **_summarize_rows(actionable_rows)},
    ]
    for comparison_group in (
        _GROUP_PRESERVED,
        _GROUP_COLLAPSED,
        _GROUP_OTHER_ACTIONABLE,
        _GROUP_OTHER_NON_ACTIONABLE,
    ):
        subset = [
            row
            for row in actionable_rows
            if row.get("comparison_group") == comparison_group
        ]
        group_rows.append(
            {
                "comparison_group": comparison_group,
                **_summarize_rows(subset),
            }
        )
    return group_rows


def build_group_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
    support_threshold: int,
    min_row_count: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_group_value(row, field) for field in group_fields)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        if len(items) < min_row_count:
            continue
        summary_row = {field: key[index] for index, field in enumerate(group_fields)}
        summary_row.update(
            _summarize_rows(items, support_threshold=max(1, int(support_threshold)))
        )
        summary_rows.append(summary_row)

    summary_rows.sort(
        key=lambda item: tuple(
            _sort_group_value(field, item.get(field))
            for field in group_fields
        )
        + (-int(item.get("row_count", 0) or 0),)
    )
    return summary_rows


def build_summary(
    *,
    actionable_rows: Sequence[dict[str, Any]],
    overall_group_summaries: Sequence[dict[str, Any]],
    min_symbol_support: int,
) -> dict[str, Any]:
    group_map = _group_map(overall_group_summaries)
    preserved = _safe_dict(group_map.get(_GROUP_PRESERVED))
    collapsed = _safe_dict(group_map.get(_GROUP_COLLAPSED))
    other_actionable = _safe_dict(group_map.get(_GROUP_OTHER_ACTIONABLE))
    other_non_actionable = _safe_dict(group_map.get(_GROUP_OTHER_NON_ACTIONABLE))

    evidence_source_counter = Counter(
        str(row.get("activation_evidence_source") or "other_or_missing")
        for row in actionable_rows
    )

    return {
        "actionable_selected_strategy_row_count": len(actionable_rows),
        "preserved_aligned_actionable_row_count": int(
            preserved.get("row_count", 0) or 0
        ),
        "collapsed_to_hold_row_count": int(collapsed.get("row_count", 0) or 0),
        "other_actionable_rule_outcome_row_count": int(
            other_actionable.get("row_count", 0) or 0
        ),
        "other_non_actionable_rule_outcome_row_count": int(
            other_non_actionable.get("row_count", 0) or 0
        ),
        "min_symbol_support": max(1, int(min_symbol_support)),
        "evidence_source_counts": dict(evidence_source_counter),
        "preserved_aligned_actionable_evidence_source_counts": _safe_dict(
            preserved.get("evidence_source_counts")
        ),
        "collapsed_to_hold_evidence_source_counts": _safe_dict(
            collapsed.get("evidence_source_counts")
        ),
        "other_actionable_rule_outcome_evidence_source_counts": _safe_dict(
            other_actionable.get("evidence_source_counts")
        ),
        "preserved_vs_collapsed_rate_gap": {
            "setup_activation_rate_gap": round(
                _to_float(preserved.get("setup_activation_rate"), default=0.0)
                - _to_float(collapsed.get("setup_activation_rate"), default=0.0),
                6,
            ),
            "trigger_activation_rate_gap": round(
                _to_float(preserved.get("trigger_activation_rate"), default=0.0)
                - _to_float(collapsed.get("trigger_activation_rate"), default=0.0),
                6,
            ),
            "dual_activation_rate_gap": round(
                _to_float(preserved.get("dual_activation_rate"), default=0.0)
                - _to_float(collapsed.get("dual_activation_rate"), default=0.0),
                6,
            ),
            "dual_neutral_or_missing_rate_gap": round(
                _to_float(
                    collapsed.get("dual_neutral_or_missing_rate"),
                    default=0.0,
                )
                - _to_float(
                    preserved.get("dual_neutral_or_missing_rate"),
                    default=0.0,
                ),
                6,
            ),
        },
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    overall_group_summaries: Sequence[dict[str, Any]],
    strategy_group_summaries: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    group_map = _group_map(overall_group_summaries)
    preserved = _safe_dict(group_map.get(_GROUP_PRESERVED))
    collapsed = _safe_dict(group_map.get(_GROUP_COLLAPSED))
    other_actionable = _safe_dict(group_map.get(_GROUP_OTHER_ACTIONABLE))
    primary_comparison_supported = _primary_comparison_supported(
        overall_group_summaries
    )

    facts = [
        (
            "Actionable selected-strategy rows="
            f"{summary.get('actionable_selected_strategy_row_count', 0)}; "
            f"preserved_aligned_actionable={summary.get('preserved_aligned_actionable_row_count', 0)}; "
            f"collapsed_to_hold={summary.get('collapsed_to_hold_row_count', 0)}; "
            f"other_actionable_rule_outcome={summary.get('other_actionable_rule_outcome_row_count', 0)}."
        ),
        (
            "Primary comparison support: "
            f"preserved_aligned_actionable={preserved.get('support_status', 'unknown')} "
            f"(n={int(preserved.get('row_count', 0) or 0)}), "
            f"collapsed_to_hold={collapsed.get('support_status', 'unknown')} "
            f"(n={int(collapsed.get('row_count', 0) or 0)})."
        ),
        (
            "Preserved aligned actionable rows: "
            f"setup_activation_rate={_to_float(preserved.get('setup_activation_rate'), default=0.0):.2%}, "
            f"trigger_activation_rate={_to_float(preserved.get('trigger_activation_rate'), default=0.0):.2%}, "
            f"dual_activation_rate={_to_float(preserved.get('dual_activation_rate'), default=0.0):.2%}, "
            "dual_neutral_or_missing_rate="
            f"{_to_float(preserved.get('dual_neutral_or_missing_rate'), default=0.0):.2%}."
        ),
        (
            "Collapsed-to-hold rows: "
            f"setup_activation_rate={_to_float(collapsed.get('setup_activation_rate'), default=0.0):.2%}, "
            f"trigger_activation_rate={_to_float(collapsed.get('trigger_activation_rate'), default=0.0):.2%}, "
            f"dual_activation_rate={_to_float(collapsed.get('dual_activation_rate'), default=0.0):.2%}, "
            "dual_neutral_or_missing_rate="
            f"{_to_float(collapsed.get('dual_neutral_or_missing_rate'), default=0.0):.2%}."
        ),
    ]

    if int(other_actionable.get("row_count", 0) or 0) > 0:
        facts.append(
            "Other actionable rule outcomes are retained separately: "
            f"{int(other_actionable.get('row_count', 0) or 0)} rows."
        )

    strongest_strategy_gap = _strongest_strategy_gap(strategy_group_summaries)
    if strongest_strategy_gap:
        facts.append(strongest_strategy_gap)

    inferences: list[str] = []
    if primary_comparison_supported:
        if _to_float(
            preserved.get("dual_activation_rate"), default=0.0
        ) > _to_float(
            collapsed.get("dual_activation_rate"),
            default=0.0,
        ):
            inferences.append(
                "The supported preserved-vs-collapsed comparison is consistent with a setup/trigger activation gap: rows preserved in the selected direction retain higher dual activation than rows that collapse to hold."
            )
        if _to_float(
            collapsed.get("dual_neutral_or_missing_rate"), default=0.0
        ) > _to_float(
            preserved.get("dual_neutral_or_missing_rate"),
            default=0.0,
        ):
            inferences.append(
                "Within the supported primary comparison, collapsed hold rows are more concentrated in dual neutral/missing setup+trigger states than preserved aligned actionable rows."
            )
    else:
        inferences.append(
            "The primary preserved-vs-collapsed comparison still has limited support in this slice, so apparent rate gaps should be treated as provisional rather than as stable evidence."
        )

    collapsed_reason_text_only = int(
        _safe_dict(summary.get("collapsed_to_hold_evidence_source_counts")).get(
            "reason_text_only",
            0,
        )
        or 0
    )
    collapsed_structured = int(
        _safe_dict(summary.get("collapsed_to_hold_evidence_source_counts")).get(
            "structured_state_backed",
            0,
        )
        or 0
    )
    if collapsed_reason_text_only > 0 and collapsed_structured > 0:
        inferences.append(
            "Collapsed rows still span both reason_text_only and structured_state_backed setup/trigger evidence sources, so the observed activation-gap pattern should not be reduced to a single evidence subset."
        )

    uncertainties = [
        "This report compares observable setup/trigger activation patterns; it does not prove which internal rule, threshold, or merge path caused a hold resolution.",
        "Bias-only directional hints without supportive setup/trigger state are classified conservatively as mixed_or_inconclusive rather than activated, so these rates may undercount any purely bias-driven internal mechanism.",
        "Rows that end in watchlist, no_signal, unknown, other, or missing rule outcomes are tracked separately and are not used as preserved-vs-collapsed evidence.",
        "The activation_evidence_source split in this report is setup/trigger-centered rather than full-pipeline-centered, so it should be interpreted as an evidence-shape distinction, not as proof of the complete internal decision path.",
    ]

    return {
        "facts": facts,
        "inferences": inferences,
        "uncertainties": uncertainties,
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    observations = _safe_dict(widest.get("key_observations"))
    overall_group_summaries = _safe_list(widest.get("overall_group_summaries"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "facts": _safe_list(observations.get("facts")),
        "inferences": _safe_list(observations.get("inferences")),
        "uncertainties": _safe_list(observations.get("uncertainties")),
        "overall_conclusion": _overall_conclusion(summary, overall_group_summaries),
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
        key_observations = _safe_dict(_safe_dict(summary).get("key_observations"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(
            "- actionable_selected_strategy_row_count: "
            f"{headline.get('actionable_selected_strategy_row_count', 0)}"
        )
        lines.append(
            "- preserved_aligned_actionable_row_count: "
            f"{headline.get('preserved_aligned_actionable_row_count', 0)}"
        )
        lines.append(
            "- collapsed_to_hold_row_count: "
            f"{headline.get('collapsed_to_hold_row_count', 0)}"
        )
        lines.append(
            "- other_actionable_rule_outcome_row_count: "
            f"{headline.get('other_actionable_rule_outcome_row_count', 0)}"
        )
        lines.append(
            "- preserved_dual_activation_rate: "
            f"{_to_float(headline.get('preserved_dual_activation_rate'), default=0.0):.2%}"
        )
        lines.append(
            "- collapsed_dual_activation_rate: "
            f"{_to_float(headline.get('collapsed_dual_activation_rate'), default=0.0):.2%}"
        )
        lines.append(
            "- collapsed_dual_neutral_or_missing_rate: "
            f"{_to_float(headline.get('collapsed_dual_neutral_or_missing_rate'), default=0.0):.2%}"
        )
        for fact in _safe_list(key_observations.get("facts"))[:4]:
            lines.append(f"- fact: {fact}")
        for inference in _safe_list(key_observations.get("inferences"))[:2]:
            lines.append(f"- inference: {inference}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    for fact in _safe_list(final_assessment.get("facts"))[:4]:
        lines.append(f"- fact: {fact}")
    for inference in _safe_list(final_assessment.get("inferences"))[:2]:
        lines.append(f"- inference: {inference}")
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


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    selected_signal = str(row.get("selected_strategy_result_signal_state") or "")
    setup_bias = _safe_dict(row.get("setup_layer_payload")).get("bias")
    trigger_bias = _safe_dict(row.get("trigger_layer_payload")).get("bias")

    setup_relation = _activation_relation(
        state_value=row.get("setup_state"),
        bias_value=setup_bias,
        selected_signal=selected_signal,
    )
    trigger_relation = _activation_relation(
        state_value=row.get("trigger_state"),
        bias_value=trigger_bias,
        selected_signal=selected_signal,
    )

    return {
        **row,
        "comparison_group": _comparison_group(
            selected_signal=selected_signal,
            rule_signal=str(row.get("rule_signal_state") or ""),
        ),
        "setup_relation_to_selected": setup_relation,
        "trigger_relation_to_selected": trigger_relation,
        "activation_pattern": _activation_pattern(
            setup_relation=setup_relation,
            trigger_relation=trigger_relation,
        ),
        "activation_evidence_source": _activation_evidence_source(row),
    }


def _comparison_group(*, selected_signal: str, rule_signal: str) -> str:
    if rule_signal == selected_signal:
        return _GROUP_PRESERVED
    if rule_signal == "hold":
        return _GROUP_COLLAPSED
    if rule_signal in _ACTIONABLE_SIGNAL_STATES:
        return _GROUP_OTHER_ACTIONABLE
    return _GROUP_OTHER_NON_ACTIONABLE


def _activation_relation(
    *,
    state_value: Any,
    bias_value: Any,
    selected_signal: str,
) -> str:
    state_relation = _state_relation_to_selected(
        state_value=state_value,
        selected_signal=selected_signal,
    )
    bias_relation = _bias_relation(bias_value, selected_signal)

    if state_relation == "aligned_with_selected":
        if bias_relation == "opposite_to_selected":
            return "mixed_or_inconclusive"
        return "aligned_with_selected"

    if state_relation == "opposite_to_selected":
        if bias_relation == "aligned_with_selected":
            return "mixed_or_inconclusive"
        return "opposite_to_selected"

    if state_relation == "mixed_or_inconclusive":
        return "mixed_or_inconclusive"

    if bias_relation in {"aligned_with_selected", "opposite_to_selected"}:
        return "mixed_or_inconclusive"

    return "neutral_or_missing"


def _state_relation_to_selected(*, state_value: Any, selected_signal: str) -> str:
    normalized = _normalize_text(state_value)
    if not normalized:
        return "neutral_or_missing"

    if any(token in normalized for token in _MIXED_STATE_TOKENS):
        return "mixed_or_inconclusive"

    if any(token in normalized for token in _NEUTRAL_STATE_TOKENS):
        return "neutral_or_missing"

    selected_tokens = _signal_tokens(selected_signal)
    opposite_tokens = _signal_tokens(_opposite_signal(selected_signal))

    has_selected = any(token in normalized for token in selected_tokens)
    has_opposite = any(token in normalized for token in opposite_tokens)

    if has_selected and has_opposite:
        return "mixed_or_inconclusive"
    if has_selected:
        return "aligned_with_selected"
    if has_opposite:
        return "opposite_to_selected"

    fallback = _external_layer_alignment_fallback(
        state_value=state_value,
        selected_signal=selected_signal,
    )
    if fallback is not None:
        return fallback

    return "neutral_or_missing"


def _external_layer_alignment_fallback(
    *,
    state_value: Any,
    selected_signal: str,
) -> str | None:
    layer_alignment = getattr(hold_reason_module, "_layer_alignment", None)
    if not callable(layer_alignment):
        return None

    alignment = str(layer_alignment(state_value, selected_signal) or "")
    if alignment in {"confirmed", "same_direction_but_not_fully_confirmed"}:
        return "aligned_with_selected"
    if alignment == "opposed":
        return "opposite_to_selected"
    if alignment in {"mixed_or_inconclusive", "conflicted"}:
        return "mixed_or_inconclusive"
    if alignment in {"neutral_or_missing", "missing", "none"}:
        return "neutral_or_missing"
    return None


def _bias_relation(value: Any, selected_signal: str) -> str:
    direction = _direction_from_bias(value)
    if direction is None:
        return "neutral_or_missing"
    if direction == selected_signal:
        return "aligned_with_selected"
    return "opposite_to_selected"


def _direction_from_bias(value: Any) -> str | None:
    normalized = _normalize_text(value)
    if normalized in _NEUTRAL_BIAS_TOKENS:
        return None

    if any(token in normalized for token in _DIRECTION_TOKENS["long"]):
        if any(token in normalized for token in _DIRECTION_TOKENS["short"]):
            return None
        return "long"

    if any(token in normalized for token in _DIRECTION_TOKENS["short"]):
        return "short"

    return None


def _activation_pattern(*, setup_relation: str, trigger_relation: str) -> str:
    relations = {setup_relation, trigger_relation}
    if "opposite_to_selected" in relations:
        return "any_opposition"
    if (
        setup_relation == "aligned_with_selected"
        and trigger_relation == "aligned_with_selected"
    ):
        return "dual_aligned_with_selected"
    if (
        setup_relation == "aligned_with_selected"
        and trigger_relation == "neutral_or_missing"
    ):
        return "setup_only_aligned"
    if (
        setup_relation == "neutral_or_missing"
        and trigger_relation == "aligned_with_selected"
    ):
        return "trigger_only_aligned"
    if (
        setup_relation == "neutral_or_missing"
        and trigger_relation == "neutral_or_missing"
    ):
        return "dual_neutral_or_missing"
    return "mixed_or_inconclusive"


def _activation_evidence_source(row: dict[str, Any]) -> str:
    setup_trigger_values = (
        row.get("setup_state"),
        row.get("trigger_state"),
        _safe_dict(row.get("setup_layer_payload")).get("bias"),
        _safe_dict(row.get("trigger_layer_payload")).get("bias"),
    )
    has_setup_trigger_structured = any(
        _has_meaningful_value(value) for value in setup_trigger_values
    )
    has_reason = _has_meaningful_value(row.get("rule_reason_text")) or _has_meaningful_value(
        row.get("root_reason_text")
    )

    if has_setup_trigger_structured:
        return "structured_state_backed"
    if has_reason:
        return "reason_text_only"
    return "other_or_missing"


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    support_threshold: int = _MIN_PRIMARY_SUPPORT_ROWS,
) -> dict[str, Any]:
    row_count = len(rows)
    setup_counter = Counter(
        str(row.get("setup_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    trigger_counter = Counter(
        str(row.get("trigger_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    pattern_counter = Counter(
        str(row.get("activation_pattern") or "mixed_or_inconclusive")
        for row in rows
    )
    evidence_counter = Counter(
        str(row.get("activation_evidence_source") or "other_or_missing")
        for row in rows
    )

    dominant_pattern = _dominant_value(
        pattern_counter,
        empty="none",
        order_map=_ACTIVATION_PATTERN_ORDER,
    )
    return {
        "row_count": row_count,
        "support_status": (
            "supported"
            if row_count >= max(1, int(support_threshold))
            else "limited_support"
        ),
        "setup_activation_rate": _safe_ratio(
            setup_counter["aligned_with_selected"],
            row_count,
        ),
        "trigger_activation_rate": _safe_ratio(
            trigger_counter["aligned_with_selected"],
            row_count,
        ),
        "dual_activation_rate": _safe_ratio(
            pattern_counter["dual_aligned_with_selected"],
            row_count,
        ),
        "dual_neutral_or_missing_rate": _safe_ratio(
            pattern_counter["dual_neutral_or_missing"],
            row_count,
        ),
        "setup_relation_counts": dict(setup_counter),
        "trigger_relation_counts": dict(trigger_counter),
        "activation_pattern_counts": dict(pattern_counter),
        "evidence_source_counts": dict(evidence_counter),
        "dominant_activation_pattern": dominant_pattern,
        "primary_activation_pattern": _primary_pattern(
            pattern_counter=pattern_counter,
            row_count=row_count,
            support_threshold=max(1, int(support_threshold)),
        ),
    }


def _primary_pattern(
    *,
    pattern_counter: Counter[str],
    row_count: int,
    support_threshold: int,
) -> str:
    if row_count <= 0:
        return "none"
    if row_count < support_threshold:
        return "insufficient_support"
    dominant = _dominant_value(
        pattern_counter,
        empty="none",
        order_map=_ACTIVATION_PATTERN_ORDER,
    )
    if _safe_ratio(pattern_counter.get(dominant, 0), row_count) >= _PRIMARY_PATTERN_THRESHOLD:
        return dominant
    return "mixed_or_inconclusive"


def _dominant_value(
    counter: Counter[str],
    *,
    empty: str,
    order_map: dict[str, int],
) -> str:
    if not counter:
        return empty
    return sorted(
        counter.items(),
        key=lambda item: (
            -int(item[1] or 0),
            order_map.get(item[0], 99),
            item[0],
        ),
    )[0][0]


def _strongest_strategy_gap(
    strategy_group_summaries: Sequence[dict[str, Any]],
) -> str | None:
    by_strategy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in strategy_group_summaries:
        strategy = str(row.get("strategy") or _MISSING_LABEL)
        comparison_group = str(row.get("comparison_group") or "")
        by_strategy[strategy][comparison_group] = _safe_dict(row)

    best_text: str | None = None
    best_gap = 0.0
    for strategy, strategy_rows in by_strategy.items():
        preserved = _safe_dict(strategy_rows.get(_GROUP_PRESERVED))
        collapsed = _safe_dict(strategy_rows.get(_GROUP_COLLAPSED))
        if not _group_is_supported(preserved) or not _group_is_supported(collapsed):
            continue

        gap = _to_float(
            preserved.get("dual_activation_rate"),
            default=0.0,
        ) - _to_float(
            collapsed.get("dual_activation_rate"),
            default=0.0,
        )
        if gap <= best_gap:
            continue

        best_gap = gap
        best_text = (
            f"Strategy comparison highlight: {strategy} preserved dual activation="
            f"{_to_float(preserved.get('dual_activation_rate'), default=0.0):.2%} "
            f"vs collapsed dual activation={_to_float(collapsed.get('dual_activation_rate'), default=0.0):.2%}."
        )
    return best_text


def _overall_conclusion(
    summary: dict[str, Any],
    overall_group_summaries: Sequence[dict[str, Any]],
) -> str:
    group_map = _group_map(overall_group_summaries)
    preserved = _safe_dict(group_map.get(_GROUP_PRESERVED))
    collapsed = _safe_dict(group_map.get(_GROUP_COLLAPSED))

    if not _group_is_supported(preserved) or not _group_is_supported(collapsed):
        return (
            "The preserved-vs-collapsed comparison currently has limited support in this slice, "
            "so any apparent setup/trigger activation-gap pattern should be treated as provisional rather than as a stable conclusion."
        )

    gap = _safe_dict(summary.get("preserved_vs_collapsed_rate_gap"))
    dual_gap = _to_float(gap.get("dual_activation_rate_gap"), default=0.0)
    neutral_gap = _to_float(gap.get("dual_neutral_or_missing_rate_gap"), default=0.0)

    if dual_gap > 0 or neutral_gap > 0:
        return (
            "The supported preserved-vs-collapsed comparison supports an activation-gap framing: "
            "collapsed hold rows show weaker setup/trigger co-activation and/or stronger dual neutrality than preserved aligned actionable rows."
        )

    return (
        "With supported preserved-vs-collapsed groups, this slice does not show a stronger setup/trigger activation gap in collapsed hold rows than in preserved aligned actionable rows."
    )


def _widest_configuration_summary(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {}
    return max(
        summaries,
        key=lambda item: (
            int(
                _safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0
            ),
            int(
                _safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0
            ),
        ),
    )


def _group_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("comparison_group") or ""): row
        for row in rows
        if isinstance(row, dict)
    }


def _group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        text = value.strip()
        return text or _MISSING_LABEL
    return _MISSING_LABEL


def _sort_group_value(field: str, value: Any) -> Any:
    text = str(value or _MISSING_LABEL)
    if field == "comparison_group":
        return _GROUP_ORDER.get(text, 99)
    if field == "activation_pattern":
        return _ACTIVATION_PATTERN_ORDER.get(text, 99)
    return text


def _group_is_supported(row: dict[str, Any]) -> bool:
    return (
        bool(row)
        and str(row.get("support_status") or "") == "supported"
        and int(row.get("row_count", 0) or 0) > 0
    )


def _primary_comparison_supported(
    overall_group_summaries: Sequence[dict[str, Any]],
) -> bool:
    group_map = _group_map(overall_group_summaries)
    return _group_is_supported(_safe_dict(group_map.get(_GROUP_PRESERVED))) and _group_is_supported(
        _safe_dict(group_map.get(_GROUP_COLLAPSED))
    )


def _resolve_path(path: Path) -> Path:
    return hold_reason_module.resolve_path(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    return hold_reason_module.parse_configuration_values(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    materializer = getattr(hold_reason_module, "_materialize_configuration_input", None)
    if not callable(materializer):
        raise RuntimeError(
            "selected_strategy_setup_trigger_activation_gap_diagnosis_report requires "
            "hold-resolution materialization support, but _materialize_configuration_input is unavailable."
        )
    return materializer(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(hold_reason_module, "_build_stage_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_setup_trigger_activation_gap_diagnosis_report requires "
            "hold-resolution stage-row extraction support, but _build_stage_row is unavailable."
        )
    return builder(raw_record)


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _signal_tokens(signal: str | None) -> tuple[str, ...]:
    if not signal:
        return ()
    return _DIRECTION_TOKENS.get(signal, (signal,))


def _opposite_signal(signal: str) -> str | None:
    if signal == "long":
        return "short"
    if signal == "short":
        return "long"
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _to_float(value: Any, *, default: float) -> float:
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