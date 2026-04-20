from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Sequence

from src.research.diagnostics import (
    selected_strategy_fully_aligned_hold_residual_diagnosis_report as fully_aligned_module,
)
from src.research.diagnostics import (
    selected_strategy_hold_resolution_reason_diagnosis_report as hold_reason_module,
)

REPORT_TYPE = "selected_strategy_fully_aligned_final_hold_split_diagnosis_report"
REPORT_TITLE = "Selected Strategy Fully Aligned Final Hold Split Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_MIN_SYMBOL_SUPPORT = 10

_MISSING_LABEL = "(missing)"
_PRIMARY_FACTOR_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_STRATEGY_SUPPORT_ROWS = 10

_COMPARISON_GROUP_PRESERVED = "preserved_final_directional_outcome"
_COMPARISON_GROUP_COLLAPSED = "collapsed_final_hold_outcome"
_COMPARISON_GROUP_OTHER = "other_rule_bias_aligned_final_outcome"

_COMPARISON_GROUP_ORDER = {
    _COMPARISON_GROUP_PRESERVED: 0,
    _COMPARISON_GROUP_COLLAPSED: 1,
    _COMPARISON_GROUP_OTHER: 2,
}
_FIELD_STATUS_ORDER = {
    "separates_groups": 0,
    "missingness_separates_groups": 1,
    "non_differentiating": 2,
    "insufficient_data": 3,
    "all_missing": 4,
}
_NUMERIC_STATUS_ORDER = {
    "higher_on_preserved": 0,
    "higher_on_collapsed": 1,
    "missing_on_collapsed_only": 2,
    "missing_on_preserved_only": 3,
    "no_clear_separation": 4,
    "all_missing": 5,
}
_MISSINGNESS_PATTERN_ORDER = {
    "missing_on_collapsed_only": 0,
    "missing_on_preserved_only": 1,
    "more_missing_on_collapsed": 2,
    "more_missing_on_preserved": 3,
    "equally_missing": 4,
    "fully_present_on_both": 5,
    "all_missing": 6,
}
_REASON_BUCKET_ORDER = dict(
    getattr(
        fully_aligned_module,
        "_REASON_BUCKET_ORDER",
        {
            "confirmation_not_ready_or_missing": 0,
            "conflict_or_disagreement": 1,
            "risk_or_filter_rejection": 2,
            "context_not_supportive": 3,
            "opposition_or_invalidated": 4,
            "insufficient_explanation": 5,
            "other": 6,
            "mixed_or_inconclusive": 7,
            "insufficient_support": 8,
            "no_rows": 9,
        },
    )
)

DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS


def _rule_signal_state(row: dict[str, Any]) -> Any:
    return _first_present_value(
        row.get("rule_signal_state"),
        row.get("rule_signal"),
        _safe_dict(row.get("rule_engine_payload")).get("signal"),
        _safe_dict(row.get("rule_engine")).get("signal"),
    )


def _execution_signal(row: dict[str, Any]) -> Any:
    execution_payload = _safe_dict(row.get("execution_payload"))
    execution = _safe_dict(row.get("execution"))
    return _first_present_value(
        row.get("execution_signal"),
        execution_payload.get("signal"),
        execution.get("signal"),
    )


def _execution_action(row: dict[str, Any]) -> Any:
    execution_payload = _safe_dict(row.get("execution_payload"))
    execution = _safe_dict(row.get("execution"))
    return _first_present_value(
        row.get("execution_action"),
        execution_payload.get("action"),
        execution.get("action"),
    )


def _execution_allowed(row: dict[str, Any]) -> Any:
    execution_payload = _safe_dict(row.get("execution_payload"))
    execution = _safe_dict(row.get("execution"))
    return _first_present_value(
        row.get("execution_allowed"),
        execution_payload.get("execution_allowed"),
        execution_payload.get("allowed"),
        execution.get("execution_allowed"),
        execution.get("allowed"),
    )


_FINAL_OUTCOME_FIELD_SPECS: tuple[
    tuple[str, str, Callable[[dict[str, Any]], Any]],
    ...,
] = (
    ("rule_signal_state", "Final rule signal state", _rule_signal_state),
    ("execution_signal", "Execution signal", _execution_signal),
    ("execution_action", "Execution action", _execution_action),
    ("execution_allowed", "Execution allowed", _execution_allowed),
)
_FINAL_OUTCOME_FIELD_ORDER = {
    name: index for index, (name, _, _) in enumerate(_FINAL_OUTCOME_FIELD_SPECS)
}


def _selected_strategy_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("selected_strategy_confidence"),
            _safe_dict(row.get("selected_strategy_payload")).get("confidence"),
        ),
        default=None,
    )


def _rule_engine_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("rule_engine_confidence"),
            _safe_dict(row.get("rule_engine_payload")).get("confidence"),
            _safe_dict(row.get("rule_engine")).get("confidence"),
        ),
        default=None,
    )


def _context_layer_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("context_layer_confidence"),
            _safe_dict(row.get("context_layer_payload")).get("confidence"),
        ),
        default=None,
    )


def _bias_layer_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("bias_layer_confidence"),
            _safe_dict(row.get("bias_layer_payload")).get("confidence"),
        ),
        default=None,
    )


def _setup_layer_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("setup_layer_confidence"),
            _safe_dict(row.get("setup_layer_payload")).get("confidence"),
        ),
        default=None,
    )


def _trigger_layer_confidence(row: dict[str, Any]) -> float | None:
    return _to_float(
        _first_present_value(
            row.get("trigger_layer_confidence"),
            _safe_dict(row.get("trigger_layer_payload")).get("confidence"),
        ),
        default=None,
    )


_NUMERIC_FIELD_SPECS: tuple[
    tuple[str, str, Callable[[dict[str, Any]], float | None]],
    ...,
] = (
    (
        "selected_strategy_confidence",
        "Selected strategy confidence",
        _selected_strategy_confidence,
    ),
    ("rule_engine_confidence", "Rule engine confidence", _rule_engine_confidence),
    ("context_layer_confidence", "Context layer confidence", _context_layer_confidence),
    ("bias_layer_confidence", "Bias layer confidence", _bias_layer_confidence),
    ("setup_layer_confidence", "Setup layer confidence", _setup_layer_confidence),
    ("trigger_layer_confidence", "Trigger layer confidence", _trigger_layer_confidence),
)
_NUMERIC_FIELD_ORDER = {
    name: index for index, (name, _, _) in enumerate(_NUMERIC_FIELD_SPECS)
}
_FIELD_KIND_MAP = {
    **{name: "categorical" for name, _, _ in _FINAL_OUTCOME_FIELD_SPECS},
    **{name: "numeric" for name, _, _ in _NUMERIC_FIELD_SPECS},
}
_FIELD_LABEL_MAP = {
    **{name: label for name, label, _ in _FINAL_OUTCOME_FIELD_SPECS},
    **{name: label for name, label, _ in _NUMERIC_FIELD_SPECS},
}
_MISSINGNESS_FIELD_ORDER = {
    **_FINAL_OUTCOME_FIELD_ORDER,
    **{
        name: len(_FINAL_OUTCOME_FIELD_SPECS) + index
        for index, (name, _, _) in enumerate(_NUMERIC_FIELD_SPECS)
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the last clean residual: "
            "actionable selected-strategy rows whose setup, trigger, context_state, "
            "context_bias, and rule_bias all align with the selected direction, "
            "compared between preserved directional outcomes and collapsed hold outcomes."
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
        help="Minimum rows required for symbol+comparison_group summaries.",
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

    result = run_selected_strategy_fully_aligned_final_hold_split_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_assessment = _safe_dict(report.get("final_assessment"))
    strongest_numeric = _safe_dict(
        _safe_dict(report.get("numeric_field_comparison")).get(
            "strongest_numeric_differentiator"
        )
    )

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
                "comparison_support_status": summary.get(
                    "comparison_support_status",
                    "unknown",
                ),
                "strongest_numeric_field": strongest_numeric.get("field"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_fully_aligned_final_hold_split_diagnosis_report(
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
        "comparison_group_summaries": _safe_list(
            widest_summary.get("comparison_group_summaries")
        ),
        "strategy_final_split_summaries": _safe_list(
            widest_summary.get("strategy_final_split_summaries")
        ),
        "symbol_final_split_summaries": _safe_list(
            widest_summary.get("symbol_final_split_summaries")
        ),
        "final_outcome_field_comparison": _safe_dict(
            widest_summary.get("final_outcome_field_comparison")
        ),
        "numeric_field_comparison": _safe_dict(
            widest_summary.get("numeric_field_comparison")
        ),
        "missingness_comparison": _safe_dict(
            widest_summary.get("missingness_comparison")
        ),
        "tertiary_reason_comparison": _safe_dict(
            widest_summary.get("tertiary_reason_comparison")
        ),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the existing stage-row extraction, setup/trigger activation mapping, and fully aligned row construction helpers instead of modifying production decision logic.",
            "The target population is intentionally narrower than the neighboring fully aligned report: only rows where rule_bias is already aligned_with_selected remain in scope for the final hold split comparison.",
            "Final rule signal state and execution fields are treated as persisted end-state descriptors; they expose the last visible split but do not, by themselves, prove the upstream merge branch or threshold that produced it.",
            "Numeric summaries include only already-persisted confidence-like fields, with explicit missingness and without invented defaults.",
            "Strategy, symbol, and reason-bucket summaries remain secondary or tertiary evidence only; they are reported transparently but are not promoted into a root-cause claim on their own.",
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
        in fully_aligned_module._ACTIONABLE_SIGNAL_STATES
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

    summary = build_summary(
        actionable_rows=actionable_rows,
        fully_aligned_rows=fully_aligned_rows,
        final_split_rows=final_split_rows,
        min_symbol_support=min_symbol_support,
    )
    comparison_group_summaries = build_comparison_group_summaries(
        comparison_rows=final_split_rows,
    )
    strategy_final_split_summaries = build_group_summaries(
        final_split_rows,
        group_fields=("strategy", "comparison_group"),
        support_threshold=_MIN_STRATEGY_SUPPORT_ROWS,
    )
    symbol_final_split_summaries = build_group_summaries(
        final_split_rows,
        group_fields=("symbol", "comparison_group"),
        support_threshold=max(1, min_symbol_support),
        min_row_count=max(1, min_symbol_support),
    )
    final_outcome_field_comparison = build_final_outcome_field_comparison(
        comparison_rows=final_split_rows,
    )
    numeric_field_comparison = build_numeric_field_comparison(
        comparison_rows=final_split_rows,
    )
    missingness_comparison = build_missingness_comparison(
        comparison_rows=final_split_rows,
    )
    tertiary_reason_comparison = build_tertiary_reason_comparison(
        comparison_rows=final_split_rows,
    )
    key_observations = build_key_observations(
        summary=summary,
        comparison_group_summaries=comparison_group_summaries,
        strategy_final_split_summaries=strategy_final_split_summaries,
        final_outcome_field_comparison=final_outcome_field_comparison,
        numeric_field_comparison=numeric_field_comparison,
        missingness_comparison=missingness_comparison,
        tertiary_reason_comparison=tertiary_reason_comparison,
    )

    comparison_group_map = _comparison_group_map(comparison_group_summaries)
    strongest_numeric = _safe_dict(
        numeric_field_comparison.get("strongest_numeric_differentiator")
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
            "collapsed_dominant_rule_signal_state": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_COLLAPSED)
            ).get("dominant_rule_signal_state", "none"),
            "collapsed_dominant_execution_action": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_COLLAPSED)
            ).get("dominant_execution_action", "none"),
            "collapsed_primary_reason_bucket": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_COLLAPSED)
            ).get("primary_reason_bucket", "no_rows"),
            "strongest_numeric_differentiator": strongest_numeric.get("field"),
        },
        "summary": summary,
        "comparison_group_summaries": comparison_group_summaries,
        "strategy_final_split_summaries": strategy_final_split_summaries,
        "symbol_final_split_summaries": symbol_final_split_summaries,
        "final_outcome_field_comparison": final_outcome_field_comparison,
        "numeric_field_comparison": numeric_field_comparison,
        "missingness_comparison": missingness_comparison,
        "tertiary_reason_comparison": tertiary_reason_comparison,
        "key_observations": key_observations,
    }


def build_summary(
    *,
    actionable_rows: Sequence[dict[str, Any]],
    fully_aligned_rows: Sequence[dict[str, Any]],
    final_split_rows: Sequence[dict[str, Any]],
    min_symbol_support: int,
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in final_split_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in final_split_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]
    other_rows = [
        row
        for row in final_split_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_OTHER
    ]

    return {
        "actionable_selected_strategy_row_count": len(actionable_rows),
        "fully_aligned_row_count": len(fully_aligned_rows),
        "final_rule_bias_aligned_row_count": len(final_split_rows),
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "other_rule_bias_aligned_final_outcome_row_count": len(other_rows),
        "final_rule_bias_aligned_share_within_fully_aligned": _safe_ratio(
            len(final_split_rows),
            len(fully_aligned_rows),
        ),
        "final_rule_bias_aligned_share_within_actionable_selected_strategy": _safe_ratio(
            len(final_split_rows),
            len(actionable_rows),
        ),
        "comparison_support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "min_symbol_support": max(1, int(min_symbol_support)),
    }


def build_comparison_group_summaries(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[str(row.get("comparison_group") or _COMPARISON_GROUP_OTHER)].append(row)

    group_names = [
        _COMPARISON_GROUP_PRESERVED,
        _COMPARISON_GROUP_COLLAPSED,
    ]
    if grouped.get(_COMPARISON_GROUP_OTHER):
        group_names.append(_COMPARISON_GROUP_OTHER)

    rows: list[dict[str, Any]] = []
    for group_name in group_names:
        row_summary = _summarize_rows(
            grouped.get(group_name, []),
            support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
        )
        row_summary["comparison_group"] = group_name
        rows.append(row_summary)

    rows.sort(
        key=lambda item: _sort_group_value(
            "comparison_group",
            item.get("comparison_group"),
        )
    )
    return rows


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
    for key, grouped_rows in grouped.items():
        if len(grouped_rows) < max(1, int(min_row_count)):
            continue
        summary = _summarize_rows(
            grouped_rows,
            support_threshold=max(1, int(support_threshold)),
        )
        row = {field: key[index] for index, field in enumerate(group_fields)}
        row.update(summary)
        summary_rows.append(row)

    summary_rows.sort(
        key=lambda item: tuple(
            _sort_group_value(field, item.get(field)) for field in group_fields
        )
        + (-int(item.get("row_count", 0) or 0),)
    )
    return summary_rows


def build_final_outcome_field_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]

    field_rows: list[dict[str, Any]] = []
    for field, label, _ in _FINAL_OUTCOME_FIELD_SPECS:
        preserved_summary = _categorical_field_summary(preserved_rows, field)
        collapsed_summary = _categorical_field_summary(collapsed_rows, field)
        field_rows.append(
            {
                "field": field,
                "field_label": label,
                _COMPARISON_GROUP_PRESERVED: preserved_summary,
                _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
                "comparison_status": _categorical_comparison_status(
                    preserved_summary=preserved_summary,
                    collapsed_summary=collapsed_summary,
                ),
            }
        )

    field_rows.sort(
        key=lambda item: (
            _FIELD_STATUS_ORDER.get(str(item.get("comparison_status") or ""), 99),
            _FINAL_OUTCOME_FIELD_ORDER.get(str(item.get("field") or ""), 99),
        )
    )
    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "field_comparisons": field_rows,
        "confirmed_differentiators": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status")
            in {"separates_groups", "missingness_separates_groups"}
        ],
        "non_differentiating_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "non_differentiating"
        ],
        "unresolved_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "all_missing"
        ],
    }


def build_numeric_field_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]

    field_rows: list[dict[str, Any]] = []
    for field, label, _ in _NUMERIC_FIELD_SPECS:
        preserved_summary = _numeric_field_summary(preserved_rows, field)
        collapsed_summary = _numeric_field_summary(collapsed_rows, field)
        median_difference = _difference_or_none(
            preserved_summary.get("median"),
            collapsed_summary.get("median"),
        )
        mean_difference = _difference_or_none(
            preserved_summary.get("mean"),
            collapsed_summary.get("mean"),
        )
        field_rows.append(
            {
                "field": field,
                "field_label": label,
                _COMPARISON_GROUP_PRESERVED: preserved_summary,
                _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
                "comparison_status": _numeric_comparison_status(
                    preserved_summary=preserved_summary,
                    collapsed_summary=collapsed_summary,
                ),
                "median_difference_preserved_minus_collapsed": median_difference,
                "mean_difference_preserved_minus_collapsed": mean_difference,
            }
        )

    field_rows.sort(
        key=lambda item: (
            _NUMERIC_STATUS_ORDER.get(str(item.get("comparison_status") or ""), 99),
            -abs(
                _to_float(
                    item.get("median_difference_preserved_minus_collapsed"),
                    default=0.0,
                )
            ),
            _NUMERIC_FIELD_ORDER.get(str(item.get("field") or ""), 99),
        )
    )
    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "field_comparisons": field_rows,
        "strongest_numeric_differentiator": _strongest_numeric_differentiator(
            field_rows
        ),
        "non_differentiating_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "no_clear_separation"
        ],
        "unresolved_fields": [
            row["field"]
            for row in field_rows
            if row.get("comparison_status") == "all_missing"
        ],
    }


def build_missingness_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]

    field_rows: list[dict[str, Any]] = []
    for field in _iter_missingness_fields():
        preserved_summary = _missingness_field_summary(preserved_rows, field)
        collapsed_summary = _missingness_field_summary(collapsed_rows, field)
        field_rows.append(
            {
                "field": field,
                "field_label": _FIELD_LABEL_MAP.get(field, field),
                "field_kind": _FIELD_KIND_MAP.get(field, "unknown"),
                _COMPARISON_GROUP_PRESERVED: preserved_summary,
                _COMPARISON_GROUP_COLLAPSED: collapsed_summary,
                "missingness_pattern": _missingness_pattern(
                    preserved_summary=preserved_summary,
                    collapsed_summary=collapsed_summary,
                ),
            }
        )

    field_rows.sort(
        key=lambda item: (
            _MISSINGNESS_PATTERN_ORDER.get(
                str(item.get("missingness_pattern") or ""),
                99,
            ),
            _MISSINGNESS_FIELD_ORDER.get(str(item.get("field") or ""), 99),
        )
    )
    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "field_comparisons": field_rows,
        "confirmed_missingness_differentiators": [
            row["field"]
            for row in field_rows
            if row.get("missingness_pattern")
            in {
                "missing_on_collapsed_only",
                "missing_on_preserved_only",
                "more_missing_on_collapsed",
                "more_missing_on_preserved",
            }
        ],
        "unresolved_fields": [
            row["field"]
            for row in field_rows
            if row.get("missingness_pattern") == "all_missing"
        ],
    }


def build_tertiary_reason_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]

    preserved_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in preserved_rows
    )
    collapsed_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in collapsed_rows
    )
    preserved_rate_map = _rate_map(preserved_counter, len(preserved_rows))
    collapsed_rate_map = _rate_map(collapsed_counter, len(collapsed_rows))
    comparison_status = (
        "non_differentiating"
        if _rate_maps_equal(preserved_rate_map, collapsed_rate_map)
        else "separates_groups"
    )
    if not preserved_rows or not collapsed_rows:
        comparison_status = "insufficient_data"

    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(preserved_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "comparison_status": comparison_status,
        "preserved_final_directional_outcome_row_count": len(preserved_rows),
        "collapsed_final_hold_outcome_row_count": len(collapsed_rows),
        "reason_bucket_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(preserved_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_counter),
        },
        "reason_bucket_rates": {
            _COMPARISON_GROUP_PRESERVED: preserved_rate_map,
            _COMPARISON_GROUP_COLLAPSED: collapsed_rate_map,
        },
        "primary_reason_bucket": {
            _COMPARISON_GROUP_PRESERVED: _primary_reason_bucket(
                preserved_counter,
                row_count=len(preserved_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
            _COMPARISON_GROUP_COLLAPSED: _primary_reason_bucket(
                collapsed_counter,
                row_count=len(collapsed_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
        },
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    comparison_group_summaries: Sequence[dict[str, Any]],
    strategy_final_split_summaries: Sequence[dict[str, Any]],
    final_outcome_field_comparison: dict[str, Any],
    numeric_field_comparison: dict[str, Any],
    missingness_comparison: dict[str, Any],
    tertiary_reason_comparison: dict[str, Any],
) -> dict[str, list[str]]:
    group_map = _comparison_group_map(comparison_group_summaries)
    collapsed = _safe_dict(group_map.get(_COMPARISON_GROUP_COLLAPSED))
    strongest_numeric = _safe_dict(
        numeric_field_comparison.get("strongest_numeric_differentiator")
    )
    strongest_strategy_slice = _strongest_supported_collapsed_strategy_slice(
        strategy_final_split_summaries
    )

    facts = [
        (
            "Actionable selected-strategy rows="
            f"{summary.get('actionable_selected_strategy_row_count', 0)}; "
            f"fully_aligned_row_count={summary.get('fully_aligned_row_count', 0)}; "
            "final_rule_bias_aligned_row_count="
            f"{summary.get('final_rule_bias_aligned_row_count', 0)}."
        ),
        (
            "Final rule-bias-aligned comparison groups: "
            "preserved_final_directional_outcome="
            f"{summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            "collapsed_final_hold_outcome="
            f"{summary.get('collapsed_final_hold_outcome_row_count', 0)}, "
            "other_rule_bias_aligned_final_outcome="
            f"{summary.get('other_rule_bias_aligned_final_outcome_row_count', 0)}."
        ),
        (
            "Final split comparison support: "
            f"{summary.get('comparison_support_status', 'unknown')} "
            f"(preserved={summary.get('preserved_final_directional_outcome_row_count', 0)}, "
            f"collapsed={summary.get('collapsed_final_hold_outcome_row_count', 0)})."
        ),
    ]

    if int(collapsed.get("row_count", 0) or 0) > 0:
        facts.append(
            "Collapsed final hold dominant_rule_signal_state="
            f"{collapsed.get('dominant_rule_signal_state', 'none')}; "
            "dominant_execution_action="
            f"{collapsed.get('dominant_execution_action', 'none')}; "
            "primary_reason_bucket="
            f"{collapsed.get('primary_reason_bucket', 'no_rows')}."
        )
    if strongest_numeric:
        facts.append(
            f"Strongest numeric split: {strongest_numeric.get('field')} "
            f"(preserved_median={_safe_dict(strongest_numeric.get(_COMPARISON_GROUP_PRESERVED)).get('median')}, "
            f"collapsed_median={_safe_dict(strongest_numeric.get(_COMPARISON_GROUP_COLLAPSED)).get('median')}, "
            f"status={strongest_numeric.get('comparison_status')})."
        )
    if strongest_strategy_slice:
        facts.append(strongest_strategy_slice)

    confirmed_differentiators = _confirmed_differentiators(
        final_outcome_field_comparison=final_outcome_field_comparison,
        numeric_field_comparison=numeric_field_comparison,
        missingness_comparison=missingness_comparison,
    )
    non_differentiating_fields = _non_differentiating_fields(
        final_outcome_field_comparison=final_outcome_field_comparison,
        numeric_field_comparison=numeric_field_comparison,
        tertiary_reason_comparison=tertiary_reason_comparison,
    )
    unresolved_fields = _unresolved_fields(
        final_outcome_field_comparison=final_outcome_field_comparison,
        numeric_field_comparison=numeric_field_comparison,
        missingness_comparison=missingness_comparison,
    )

    inferences: list[str] = []
    if _has_outcome_field_separator(final_outcome_field_comparison):
        inferences.append(
            "Final rule signal state and execution-layer fields expose the last persisted branch inside this rule-bias-aligned slice: preserved rows remain directional while collapsed rows end in hold, which confirms a real final decision split but does not, by itself, reveal the hidden merge rule that caused it."
        )
    if strongest_numeric:
        inferences.append(
            f"{strongest_numeric.get('field')} shows the clearest preserved-vs-collapsed numeric separation inside the final slice, which makes it a focused descriptive differentiator to inspect rather than proof that confidence is the universal root cause."
        )
    elif _safe_list(missingness_comparison.get("confirmed_missingness_differentiators")):
        inferences.append(
            "The final slice shows explicit field-presence asymmetry even where numeric medians do not separate cleanly, which suggests the last visible split may be partly exposed by missing final evidence rather than by a single stronger confidence threshold."
        )
    else:
        inferences.append(
            "Beyond the direct final outcome fields, the currently persisted numeric and missingness comparisons do not expose a stronger universal separator inside this final slice."
        )
    if tertiary_reason_comparison.get("comparison_status") == "non_differentiating":
        inferences.append(
            "Reason-bucket parity across preserved and collapsed rows is consistent with the current source of truth: reason text remains tertiary evidence here and is not required to classify the final split."
        )

    uncertainties = [
        "Final outcome fields show where the split becomes visible, but they do not prove the exact internal merge rule, filter, or threshold that resolved aligned rows to hold.",
        "Numeric confidence summaries are descriptive and intentionally conservative; a lower collapsed median can be a useful clue without being stable proof of causal priority.",
        "Strategy-level and symbol-level concentrations are secondary evidence only and should not be promoted into a root-cause claim on their own.",
    ]
    if summary.get("comparison_support_status") != "supported":
        uncertainties.append(
            "Because the primary preserved-vs-collapsed comparison is still limited in this slice, any differentiator should be treated as focused directional evidence rather than settled proof."
        )

    return {
        "facts": facts,
        "inferences": inferences,
        "uncertainties": uncertainties,
        "confirmed_differentiators": confirmed_differentiators,
        "non_differentiating_fields": non_differentiating_fields,
        "unresolved_fields": unresolved_fields,
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    widest = _widest_configuration_summary(configuration_summaries)
    summary = _safe_dict(widest.get("summary"))
    observations = _safe_dict(widest.get("key_observations"))
    final_outcome_field_comparison = _safe_dict(
        widest.get("final_outcome_field_comparison")
    )
    numeric_field_comparison = _safe_dict(widest.get("numeric_field_comparison"))
    missingness_comparison = _safe_dict(widest.get("missingness_comparison"))
    tertiary_reason_comparison = _safe_dict(widest.get("tertiary_reason_comparison"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
        "confirmed_differentiators": _safe_list(
            observations.get("confirmed_differentiators")
        ),
        "non_differentiating_fields": _safe_list(
            observations.get("non_differentiating_fields")
        ),
        "unresolved_fields": _safe_list(observations.get("unresolved_fields")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            final_outcome_field_comparison=final_outcome_field_comparison,
            numeric_field_comparison=numeric_field_comparison,
            missingness_comparison=missingness_comparison,
            tertiary_reason_comparison=tertiary_reason_comparison,
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
        observations = _safe_dict(_safe_dict(summary).get("key_observations"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(
            "- actionable_selected_strategy_row_count: "
            f"{headline.get('actionable_selected_strategy_row_count', 0)}"
        )
        lines.append(
            "- fully_aligned_row_count: "
            f"{headline.get('fully_aligned_row_count', 0)}"
        )
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
            "- comparison_support_status: "
            f"{headline.get('comparison_support_status', 'unknown')}"
        )
        lines.append(
            "- collapsed_dominant_rule_signal_state: "
            f"{headline.get('collapsed_dominant_rule_signal_state', 'none')}"
        )
        lines.append(
            "- collapsed_dominant_execution_action: "
            f"{headline.get('collapsed_dominant_execution_action', 'none')}"
        )
        lines.append(
            "- strongest_numeric_differentiator: "
            f"{headline.get('strongest_numeric_differentiator', 'none')}"
        )
        for fact in _safe_list(observations.get("facts"))[:5]:
            lines.append(f"- fact: {fact}")
        for inference in _safe_list(observations.get("inferences"))[:3]:
            lines.append(f"- inference: {inference}")
        lines.append("")

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append("## Final Assessment")
    lines.append("")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    for item in _safe_list(final_assessment.get("observed"))[:5]:
        lines.append(f"- observed: {item}")
    for item in _safe_list(final_assessment.get("confirmed_differentiators"))[:6]:
        lines.append(f"- confirmed_differentiator: {item}")
    for item in _safe_list(final_assessment.get("non_differentiating_fields"))[:6]:
        lines.append(f"- non_differentiating: {item}")
    for item in _safe_list(final_assessment.get("unresolved_fields"))[:6]:
        lines.append(f"- unresolved: {item}")
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


def _build_final_split_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("rule_bias_relation_to_selected") or "") != "aligned_with_selected":
        return None

    final_outcome_values = {
        field: extractor(row) for field, _, extractor in _FINAL_OUTCOME_FIELD_SPECS
    }
    numeric_values = {
        field: extractor(row) for field, _, extractor in _NUMERIC_FIELD_SPECS
    }
    return {
        **row,
        "fully_aligned_comparison_group": row.get("comparison_group"),
        "comparison_group": _final_split_comparison_group(
            fully_aligned_comparison_group=str(row.get("comparison_group") or "")
        ),
        **final_outcome_values,
        **numeric_values,
    }


def _final_split_comparison_group(*, fully_aligned_comparison_group: str) -> str:
    if fully_aligned_comparison_group == fully_aligned_module._COMPARISON_GROUP_PRESERVED:
        return _COMPARISON_GROUP_PRESERVED
    if fully_aligned_comparison_group == fully_aligned_module._COMPARISON_GROUP_COLLAPSED:
        return _COMPARISON_GROUP_COLLAPSED
    return _COMPARISON_GROUP_OTHER


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    support_threshold: int,
) -> dict[str, Any]:
    row_count = len(rows)
    rule_signal_counter = _categorical_counter(rows, "rule_signal_state")
    execution_signal_counter = _categorical_counter(rows, "execution_signal")
    execution_action_counter = _categorical_counter(rows, "execution_action")
    execution_allowed_counter = _categorical_counter(rows, "execution_allowed")
    reason_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in rows
    )

    return {
        "row_count": row_count,
        "support_status": (
            "supported"
            if row_count >= max(1, int(support_threshold))
            else "limited_support"
        ),
        "rule_signal_state_counts": dict(rule_signal_counter),
        "rule_signal_state_rates": _rate_map(rule_signal_counter, row_count),
        "execution_signal_counts": dict(execution_signal_counter),
        "execution_signal_rates": _rate_map(execution_signal_counter, row_count),
        "execution_action_counts": dict(execution_action_counter),
        "execution_action_rates": _rate_map(execution_action_counter, row_count),
        "execution_allowed_counts": dict(execution_allowed_counter),
        "execution_allowed_rates": _rate_map(execution_allowed_counter, row_count),
        "reason_bucket_counts": dict(reason_counter),
        "reason_bucket_rates": _rate_map(reason_counter, row_count),
        "dominant_rule_signal_state": _dominant_value(
            rule_signal_counter,
            empty="none",
            order_map={},
        ),
        "dominant_execution_signal": _dominant_value(
            execution_signal_counter,
            empty="none",
            order_map={},
        ),
        "dominant_execution_action": _dominant_value(
            execution_action_counter,
            empty="none",
            order_map={},
        ),
        "dominant_execution_allowed": _dominant_value(
            execution_allowed_counter,
            empty="none",
            order_map={},
        ),
        "primary_reason_bucket": _primary_reason_bucket(
            reason_counter,
            row_count=row_count,
            support_threshold=max(1, int(support_threshold)),
        ),
    }


def _categorical_counter(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = _categorical_counter_value(row.get(field))
        if value is not None:
            counter[value] += 1
    return counter


def _categorical_field_summary(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    counter = _categorical_counter(rows, field)
    row_count = len(rows)
    present_row_count = sum(counter.values())
    missing_row_count = row_count - present_row_count
    return {
        "row_count": row_count,
        "present_row_count": present_row_count,
        "missing_row_count": missing_row_count,
        "present_rate": _safe_ratio(present_row_count, row_count),
        "missing_rate": _safe_ratio(missing_row_count, row_count),
        "value_counts": dict(counter),
        "value_rates": _rate_map(counter, row_count),
        "dominant_value": _dominant_value(counter, empty="none", order_map={}),
    }


def _categorical_comparison_status(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
) -> str:
    preserved_present = int(preserved_summary.get("present_row_count", 0) or 0)
    collapsed_present = int(collapsed_summary.get("present_row_count", 0) or 0)

    if preserved_present <= 0 and collapsed_present <= 0:
        return "all_missing"
    if preserved_present <= 0 or collapsed_present <= 0:
        return "missingness_separates_groups"
    if int(preserved_summary.get("missing_row_count", 0) or 0) != int(
        collapsed_summary.get("missing_row_count", 0) or 0
    ):
        if _rate_maps_equal(
            _safe_dict(preserved_summary.get("value_rates")),
            _safe_dict(collapsed_summary.get("value_rates")),
        ):
            return "missingness_separates_groups"
        return "separates_groups"
    if not _rate_maps_equal(
        _safe_dict(preserved_summary.get("value_rates")),
        _safe_dict(collapsed_summary.get("value_rates")),
    ):
        return "separates_groups"
    return "non_differentiating"


def _numeric_field_summary(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        float(value)
        for value in (row.get(field) for row in rows)
        if value is not None
    ]
    row_count = len(rows)
    present_row_count = len(values)
    missing_row_count = row_count - present_row_count

    if not values:
        return {
            "row_count": row_count,
            "present_row_count": 0,
            "missing_row_count": missing_row_count,
            "present_rate": _safe_ratio(0, row_count),
            "missing_rate": _safe_ratio(missing_row_count, row_count),
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }

    return {
        "row_count": row_count,
        "present_row_count": present_row_count,
        "missing_row_count": missing_row_count,
        "present_rate": _safe_ratio(present_row_count, row_count),
        "missing_rate": _safe_ratio(missing_row_count, row_count),
        "min": round(min(values), 6),
        "median": round(float(median(values)), 6),
        "mean": round(float(mean(values)), 6),
        "max": round(max(values), 6),
    }


def _numeric_comparison_status(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
) -> str:
    preserved_present = int(preserved_summary.get("present_row_count", 0) or 0)
    collapsed_present = int(collapsed_summary.get("present_row_count", 0) or 0)
    if preserved_present <= 0 and collapsed_present <= 0:
        return "all_missing"
    if preserved_present > 0 and collapsed_present <= 0:
        return "missing_on_collapsed_only"
    if collapsed_present > 0 and preserved_present <= 0:
        return "missing_on_preserved_only"

    preserved_median = _to_float(preserved_summary.get("median"), default=0.0)
    collapsed_median = _to_float(collapsed_summary.get("median"), default=0.0)
    if preserved_median > collapsed_median:
        return "higher_on_preserved"
    if collapsed_median > preserved_median:
        return "higher_on_collapsed"
    return "no_clear_separation"


def _strongest_numeric_differentiator(
    field_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    differentiators = [
        _safe_dict(row)
        for row in field_rows
        if _safe_dict(row).get("comparison_status")
        in {"higher_on_preserved", "higher_on_collapsed"}
    ]
    if not differentiators:
        return {}
    differentiators.sort(
        key=lambda item: (
            -abs(
                _to_float(
                    item.get("median_difference_preserved_minus_collapsed"),
                    default=0.0,
                )
            ),
            _NUMERIC_FIELD_ORDER.get(str(item.get("field") or ""), 99),
        )
    )
    return differentiators[0]


def _missingness_field_summary(
    rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    row_count = len(rows)
    missing_row_count = sum(1 for row in rows if _is_missing_value(row.get(field)))
    present_row_count = row_count - missing_row_count
    return {
        "row_count": row_count,
        "present_row_count": present_row_count,
        "missing_row_count": missing_row_count,
        "present_rate": _safe_ratio(present_row_count, row_count),
        "missing_rate": _safe_ratio(missing_row_count, row_count),
    }


def _missingness_pattern(
    *,
    preserved_summary: dict[str, Any],
    collapsed_summary: dict[str, Any],
) -> str:
    preserved_missing = int(preserved_summary.get("missing_row_count", 0) or 0)
    collapsed_missing = int(collapsed_summary.get("missing_row_count", 0) or 0)
    preserved_rows = int(preserved_summary.get("row_count", 0) or 0)
    collapsed_rows = int(collapsed_summary.get("row_count", 0) or 0)

    if preserved_missing >= preserved_rows and collapsed_missing >= collapsed_rows:
        return "all_missing"
    if preserved_missing <= 0 and collapsed_missing <= 0:
        return "fully_present_on_both"
    if preserved_missing <= 0 and collapsed_missing > 0:
        return "missing_on_collapsed_only"
    if collapsed_missing <= 0 and preserved_missing > 0:
        return "missing_on_preserved_only"

    preserved_rate = _to_float(preserved_summary.get("missing_rate"), default=0.0)
    collapsed_rate = _to_float(collapsed_summary.get("missing_rate"), default=0.0)
    if collapsed_rate > preserved_rate:
        return "more_missing_on_collapsed"
    if preserved_rate > collapsed_rate:
        return "more_missing_on_preserved"
    return "equally_missing"


def _confirmed_differentiators(
    *,
    final_outcome_field_comparison: dict[str, Any],
    numeric_field_comparison: dict[str, Any],
    missingness_comparison: dict[str, Any],
) -> list[str]:
    rows: list[str] = []
    for field_row in _safe_list(final_outcome_field_comparison.get("field_comparisons")):
        row = _safe_dict(field_row)
        if row.get("comparison_status") in {
            "separates_groups",
            "missingness_separates_groups",
        }:
            rows.append(f"{row.get('field')}: {row.get('comparison_status')}")

    for field_row in _safe_list(numeric_field_comparison.get("field_comparisons")):
        row = _safe_dict(field_row)
        if row.get("comparison_status") not in {
            "higher_on_preserved",
            "higher_on_collapsed",
        }:
            continue
        preserved_summary = _safe_dict(row.get(_COMPARISON_GROUP_PRESERVED))
        collapsed_summary = _safe_dict(row.get(_COMPARISON_GROUP_COLLAPSED))
        rows.append(
            f"{row.get('field')}: {row.get('comparison_status')} "
            f"(preserved_median={preserved_summary.get('median')}, "
            f"collapsed_median={collapsed_summary.get('median')})"
        )

    for field_row in _safe_list(missingness_comparison.get("field_comparisons")):
        row = _safe_dict(field_row)
        pattern = str(row.get("missingness_pattern") or "")
        if pattern in {
            "missing_on_collapsed_only",
            "missing_on_preserved_only",
            "more_missing_on_collapsed",
            "more_missing_on_preserved",
        }:
            rows.append(f"{row.get('field')}: missingness={pattern}")
    return rows


def _non_differentiating_fields(
    *,
    final_outcome_field_comparison: dict[str, Any],
    numeric_field_comparison: dict[str, Any],
    tertiary_reason_comparison: dict[str, Any],
) -> list[str]:
    rows: list[str] = []
    rows.extend(
        str(field)
        for field in _safe_list(
            final_outcome_field_comparison.get("non_differentiating_fields")
        )
    )
    rows.extend(
        str(field)
        for field in _safe_list(numeric_field_comparison.get("non_differentiating_fields"))
    )
    if tertiary_reason_comparison.get("comparison_status") == "non_differentiating":
        rows.append("combined_reason_bucket")
    return rows


def _unresolved_fields(
    *,
    final_outcome_field_comparison: dict[str, Any],
    numeric_field_comparison: dict[str, Any],
    missingness_comparison: dict[str, Any],
) -> list[str]:
    rows: list[str] = []
    rows.extend(
        str(field)
        for field in _safe_list(final_outcome_field_comparison.get("unresolved_fields"))
    )
    rows.extend(
        str(field)
        for field in _safe_list(numeric_field_comparison.get("unresolved_fields"))
    )
    rows.extend(
        str(field)
        for field in _safe_list(missingness_comparison.get("unresolved_fields"))
        if str(field) not in rows
    )
    return rows


def _has_outcome_field_separator(
    final_outcome_field_comparison: dict[str, Any],
) -> bool:
    return any(
        _safe_dict(row).get("comparison_status")
        in {"separates_groups", "missingness_separates_groups"}
        for row in _safe_list(final_outcome_field_comparison.get("field_comparisons"))
    )


def _strongest_supported_collapsed_strategy_slice(
    strategy_final_split_summaries: Sequence[dict[str, Any]],
) -> str | None:
    collapsed_rows = [
        _safe_dict(row)
        for row in strategy_final_split_summaries
        if _safe_dict(row).get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
        and _row_is_supported(_safe_dict(row))
    ]
    if not collapsed_rows:
        return None
    collapsed_rows.sort(
        key=lambda item: (
            -int(item.get("row_count", 0) or 0),
            str(item.get("strategy") or ""),
        )
    )
    row = collapsed_rows[0]
    return (
        "Largest supported collapsed final-split strategy slice: "
        f"{row.get('strategy', _MISSING_LABEL)} "
        f"(n={int(row.get('row_count', 0) or 0)})."
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    final_outcome_field_comparison: dict[str, Any],
    numeric_field_comparison: dict[str, Any],
    missingness_comparison: dict[str, Any],
    tertiary_reason_comparison: dict[str, Any],
) -> str:
    final_split_row_count = int(summary.get("final_rule_bias_aligned_row_count", 0) or 0)
    preserved_row_count = int(
        summary.get("preserved_final_directional_outcome_row_count", 0) or 0
    )
    collapsed_row_count = int(
        summary.get("collapsed_final_hold_outcome_row_count", 0) or 0
    )
    comparison_supported = str(summary.get("comparison_support_status") or "") == "supported"

    if final_split_row_count <= 0:
        return (
            "No rows reached the final rule-bias-aligned slice in the widest configuration, so this diagnosis has no final hold split target to compare."
        )
    if preserved_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "The final rule-bias-aligned slice exists, but one side of the preserved-vs-collapsed comparison is missing in the widest configuration, so the last visible split remains incomplete."
        )

    strongest_numeric = _safe_dict(
        numeric_field_comparison.get("strongest_numeric_differentiator")
    )
    missingness_fields = _safe_list(
        missingness_comparison.get("confirmed_missingness_differentiators")
    )

    if not comparison_supported:
        return (
            "The widest configuration shows a real final rule-bias-aligned split, and the last persisted branch is visible in the rule/execution outcome fields, but the preserved-vs-collapsed comparison still has limited support, so any deeper interpretation should remain focused evidence rather than stable proof."
        )

    if strongest_numeric:
        return (
            "The widest configuration supports a real final rule-bias-aligned split: final rule/execution fields expose the last persisted branch between directional preservation and collapsed hold, and at least one numeric confidence field still separates the groups, although that numeric gap remains descriptive evidence rather than proof of the hidden final decision rule."
        )

    if missingness_fields:
        return (
            "The widest configuration supports a real final rule-bias-aligned split: the last persisted branch is visible in final outcome fields, and explicit field-presence asymmetry remains visible, but the currently persisted numeric values still do not isolate a single universal threshold that explains the hold collapse."
        )

    if tertiary_reason_comparison.get("comparison_status") == "non_differentiating":
        return (
            "The widest configuration supports a real final rule-bias-aligned split, but beyond the direct final rule/execution outcome fields the currently persisted numeric, missingness, and reason-bucket surfaces do not expose a stronger preserved-vs-collapsed separator inside this smallest residual slice."
        )

    return (
        "The widest configuration supports a real final rule-bias-aligned split, and the last visible branch is clear in the persisted final outcome fields, but the deeper separator inside this smallest residual slice remains only partially explained."
    )


def _primary_reason_bucket(
    counter: Counter[str],
    *,
    row_count: int,
    support_threshold: int,
) -> str:
    if row_count <= 0:
        return "no_rows"
    if row_count < support_threshold:
        return "insufficient_support"

    dominant = _dominant_value(
        counter,
        empty="no_rows",
        order_map=_REASON_BUCKET_ORDER,
    )
    if _safe_ratio(counter.get(dominant, 0), row_count) >= _PRIMARY_FACTOR_THRESHOLD:
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


def _comparison_group_map(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(_safe_dict(row).get("comparison_group") or ""): _safe_dict(row)
        for row in rows
    }


def _comparison_support_status(
    *,
    baseline_row_count: int,
    collapsed_row_count: int,
) -> str:
    return str(
        fully_aligned_module._comparison_support_status(
            baseline_row_count=baseline_row_count,
            collapsed_row_count=collapsed_row_count,
        )
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


def _group_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        text = value.strip()
        return text or _MISSING_LABEL
    if isinstance(value, bool):
        return "true" if value else "false"
    return _MISSING_LABEL


def _sort_group_value(field: str, value: Any) -> Any:
    text = str(value or _MISSING_LABEL)
    if field == "comparison_group":
        return _COMPARISON_GROUP_ORDER.get(text, 99)
    return text


def _row_is_supported(row: dict[str, Any] | None) -> bool:
    item = _safe_dict(row)
    return (
        bool(item)
        and str(item.get("support_status") or "") == "supported"
        and int(item.get("row_count", 0) or 0) > 0
    )


def _iter_missingness_fields() -> Sequence[str]:
    return tuple(_MISSINGNESS_FIELD_ORDER.keys())


def _categorical_counter_value(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _first_present_value(*values: Any) -> Any:
    for value in values:
        if not _is_missing_value(value):
            return value
    return None


def _rate_map(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: _safe_ratio(value, total) for key, value in counter.items()}


def _rate_maps_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    keys = sorted({*left.keys(), *right.keys()})
    return all(
        _to_float(left.get(key), default=0.0) == _to_float(right.get(key), default=0.0)
        for key in keys
    )


def _difference_or_none(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(_to_float(left, default=0.0) - _to_float(right, default=0.0), 6)


def _resolve_path(path: Path) -> Path:
    resolver = getattr(fully_aligned_module, "_resolve_path", None)
    if not callable(resolver):
        resolver = getattr(hold_reason_module, "resolve_path", None)
    if not callable(resolver):
        raise RuntimeError(
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report "
            "requires path resolution support, but no resolver is available."
        )
    return resolver(path)


def _parse_configuration_values(
    raw_values: Sequence[str] | None,
) -> Sequence[DiagnosisConfiguration]:
    parser = getattr(fully_aligned_module, "_parse_configuration_values", None)
    if not callable(parser):
        parser = getattr(hold_reason_module, "parse_configuration_values", None)
    if not callable(parser):
        raise RuntimeError(
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report "
            "requires configuration parsing support, but no parser is available."
        )
    return parser(raw_values)


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    return hold_reason_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(fully_aligned_module, "_build_stage_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report "
            "requires _build_stage_row support."
        )
    return builder(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(fully_aligned_module, "_build_activation_gap_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report "
            "requires _build_activation_gap_row support."
        )
    return builder(row)


def _build_fully_aligned_row(row: dict[str, Any]) -> dict[str, Any] | None:
    builder = getattr(fully_aligned_module, "_build_fully_aligned_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report "
            "requires _build_fully_aligned_row support."
        )
    return builder(row)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


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
