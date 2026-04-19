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
from src.research.diagnostics import (
    selected_strategy_post_confirmation_hold_residual_diagnosis_report as residual_module,
)
from src.research.diagnostics import (
    selected_strategy_setup_trigger_activation_gap_diagnosis_report as activation_gap_module,
)

REPORT_TYPE = "selected_strategy_dual_aligned_context_hold_residual_diagnosis_report"
REPORT_TITLE = "Selected Strategy Dual Aligned Context Hold Residual Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_MIN_SYMBOL_SUPPORT = 10

_MISSING_LABEL = "(missing)"
_PRIMARY_FACTOR_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_STRATEGY_SUPPORT_ROWS = 10

_ACTIONABLE_SIGNAL_STATES = frozenset(
    str(value)
    for value in getattr(
        activation_gap_module,
        "_ACTIONABLE_SIGNAL_STATES",
        {"long", "short"},
    )
)

_COMPARISON_GROUP_PRESERVED = "preserved_dual_aligned_baseline"
_COMPARISON_GROUP_COLLAPSED = "collapsed_dual_aligned"
_COMPARISON_GROUP_OTHER = "other_dual_aligned_rule_outcome"

_COMPARISON_GROUP_ORDER = {
    _COMPARISON_GROUP_PRESERVED: 0,
    _COMPARISON_GROUP_COLLAPSED: 1,
    _COMPARISON_GROUP_OTHER: 2,
}
_RELATION_ORDER = {
    "aligned_with_selected": 0,
    "neutral_or_missing": 1,
    "opposite_to_selected": 2,
    "mixed_or_inconclusive": 3,
    "none": 9,
}
_CONTEXT_FAMILY_ORDER = {
    "dual_context_aligned": 0,
    "context_state_only_aligned": 1,
    "context_bias_only_aligned": 2,
    "dual_context_neutral_or_missing": 3,
    "any_context_opposition": 4,
    "context_mixed_or_inconclusive": 5,
    "none": 9,
}
_REASON_BUCKET_ORDER = {
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
}

DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for the cleanest post-confirmation "
            "residual subset: actionable selected-strategy rows with dual-aligned "
            "setup+trigger, compared between preserved actionable rows and "
            "collapsed-to-hold rows through a deeper context-layer decomposition."
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
        help="Minimum rows required for strategy+symbol+comparison_group+context_family summaries.",
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

    result = run_selected_strategy_dual_aligned_context_hold_residual_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        min_symbol_support=args.min_symbol_support,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    summary = _safe_dict(report.get("summary"))
    final_assessment = _safe_dict(report.get("final_assessment"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    final_assessment.get("widest_configuration")
                ).get("display_name"),
                "dual_aligned_row_count": summary.get("dual_aligned_row_count", 0),
                "preserved_dual_aligned_baseline_row_count": summary.get(
                    "preserved_dual_aligned_baseline_row_count",
                    0,
                ),
                "collapsed_dual_aligned_row_count": summary.get(
                    "collapsed_dual_aligned_row_count",
                    0,
                ),
                "comparison_support_status": summary.get(
                    "comparison_support_status",
                    "unknown",
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_dual_aligned_context_hold_residual_diagnosis_report(
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
        "strategy_context_summaries": _safe_list(
            widest_summary.get("strategy_context_summaries")
        ),
        "symbol_context_summaries": _safe_list(
            widest_summary.get("symbol_context_summaries")
        ),
        "context_family_comparison": _safe_dict(
            widest_summary.get("context_family_comparison")
        ),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the existing hold-resolution stage-row extraction, effective input slicing, and activation-gap row semantics instead of modifying production logic.",
            "The primary population stays narrow: actionable selected-strategy rows with dual-aligned setup+trigger, split into preserved_dual_aligned_baseline and collapsed_dual_aligned, while any other dual-aligned rule outcome is retained only as reference context.",
            "Context decomposition is stronger evidence than persisted reason text here: state, bias, combined relation, and context_family are derived from structured fields, while reason buckets remain descriptive heuristics only.",
            "Combined context relation allows state-only or bias-only directional support when there is no observable opposition, but any explicit opposition or mixed state keeps the result conservative.",
            "Strategy-level and symbol-level summaries are reporting aids only; their support thresholds are not production recommendations.",
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
    dual_aligned_rows = [
        row
        for row in actionable_rows
        if str(row.get("activation_pattern") or "") == "dual_aligned_with_selected"
    ]
    comparison_rows = [
        comparison_row
        for comparison_row in (
            _build_dual_aligned_context_row(row) for row in dual_aligned_rows
        )
        if comparison_row is not None
    ]

    summary = build_summary(
        actionable_rows=actionable_rows,
        dual_aligned_rows=comparison_rows,
        min_symbol_support=min_symbol_support,
    )
    comparison_group_summaries = build_comparison_group_summaries(
        comparison_rows=comparison_rows,
    )
    strategy_context_summaries = build_group_summaries(
        comparison_rows,
        group_fields=("strategy", "comparison_group", "context_family"),
        support_threshold=_MIN_STRATEGY_SUPPORT_ROWS,
    )
    symbol_context_summaries = build_group_summaries(
        comparison_rows,
        group_fields=("symbol", "strategy", "comparison_group", "context_family"),
        support_threshold=max(1, min_symbol_support),
        min_row_count=max(1, min_symbol_support),
    )
    context_family_comparison = build_context_family_comparison(
        comparison_rows=comparison_rows,
    )
    key_observations = build_key_observations(
        summary=summary,
        comparison_group_summaries=comparison_group_summaries,
        strategy_context_summaries=strategy_context_summaries,
        comparison=context_family_comparison,
    )

    comparison_group_map = _comparison_group_map(comparison_group_summaries)
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
            "dual_aligned_row_count": summary["dual_aligned_row_count"],
            "preserved_dual_aligned_baseline_row_count": summary[
                "preserved_dual_aligned_baseline_row_count"
            ],
            "collapsed_dual_aligned_row_count": summary[
                "collapsed_dual_aligned_row_count"
            ],
            "comparison_support_status": summary["comparison_support_status"],
            "collapsed_dominant_context_family": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_COLLAPSED)
            ).get("dominant_context_family", "none"),
            "preserved_dominant_context_family": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_PRESERVED)
            ).get("dominant_context_family", "none"),
            "collapsed_primary_reason_bucket": _safe_dict(
                comparison_group_map.get(_COMPARISON_GROUP_COLLAPSED)
            ).get("primary_reason_bucket", "no_rows"),
        },
        "summary": summary,
        "comparison_group_summaries": comparison_group_summaries,
        "strategy_context_summaries": strategy_context_summaries,
        "symbol_context_summaries": symbol_context_summaries,
        "context_family_comparison": context_family_comparison,
        "key_observations": key_observations,
    }


def build_summary(
    *,
    actionable_rows: Sequence[dict[str, Any]],
    dual_aligned_rows: Sequence[dict[str, Any]],
    min_symbol_support: int,
) -> dict[str, Any]:
    preserved_rows = [
        row
        for row in dual_aligned_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in dual_aligned_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]
    other_rows = [
        row
        for row in dual_aligned_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_OTHER
    ]

    return {
        "actionable_selected_strategy_row_count": len(actionable_rows),
        "dual_aligned_row_count": len(dual_aligned_rows),
        "preserved_dual_aligned_baseline_row_count": len(preserved_rows),
        "collapsed_dual_aligned_row_count": len(collapsed_rows),
        "other_dual_aligned_rule_outcome_row_count": len(other_rows),
        "dual_aligned_share_within_actionable_selected_strategy": _safe_ratio(
            len(dual_aligned_rows),
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


def build_context_family_comparison(
    *,
    comparison_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_PRESERVED
    ]
    collapsed_rows = [
        row
        for row in comparison_rows
        if row.get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
    ]

    baseline_state_counter = Counter(
        str(row.get("context_state_relation_to_selected") or "neutral_or_missing")
        for row in baseline_rows
    )
    collapsed_state_counter = Counter(
        str(row.get("context_state_relation_to_selected") or "neutral_or_missing")
        for row in collapsed_rows
    )
    baseline_bias_counter = Counter(
        str(row.get("context_bias_relation_to_selected") or "neutral_or_missing")
        for row in baseline_rows
    )
    collapsed_bias_counter = Counter(
        str(row.get("context_bias_relation_to_selected") or "neutral_or_missing")
        for row in collapsed_rows
    )
    baseline_combined_counter = Counter(
        str(row.get("combined_context_relation_to_selected") or "neutral_or_missing")
        for row in baseline_rows
    )
    collapsed_combined_counter = Counter(
        str(row.get("combined_context_relation_to_selected") or "neutral_or_missing")
        for row in collapsed_rows
    )
    baseline_family_counter = Counter(
        str(row.get("context_family") or "context_mixed_or_inconclusive")
        for row in baseline_rows
    )
    collapsed_family_counter = Counter(
        str(row.get("context_family") or "context_mixed_or_inconclusive")
        for row in collapsed_rows
    )
    baseline_reason_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in baseline_rows
    )
    collapsed_reason_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in collapsed_rows
    )

    return {
        "support_status": _comparison_support_status(
            baseline_row_count=len(baseline_rows),
            collapsed_row_count=len(collapsed_rows),
        ),
        "preserved_dual_aligned_baseline_row_count": len(baseline_rows),
        "collapsed_dual_aligned_row_count": len(collapsed_rows),
        "context_state_relation_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(baseline_state_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_state_counter),
        },
        "context_state_relation_rates": {
            _COMPARISON_GROUP_PRESERVED: _rate_map(
                baseline_state_counter,
                len(baseline_rows),
            ),
            _COMPARISON_GROUP_COLLAPSED: _rate_map(
                collapsed_state_counter,
                len(collapsed_rows),
            ),
        },
        "context_bias_relation_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(baseline_bias_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_bias_counter),
        },
        "context_bias_relation_rates": {
            _COMPARISON_GROUP_PRESERVED: _rate_map(
                baseline_bias_counter,
                len(baseline_rows),
            ),
            _COMPARISON_GROUP_COLLAPSED: _rate_map(
                collapsed_bias_counter,
                len(collapsed_rows),
            ),
        },
        "combined_context_relation_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(baseline_combined_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_combined_counter),
        },
        "combined_context_relation_rates": {
            _COMPARISON_GROUP_PRESERVED: _rate_map(
                baseline_combined_counter,
                len(baseline_rows),
            ),
            _COMPARISON_GROUP_COLLAPSED: _rate_map(
                collapsed_combined_counter,
                len(collapsed_rows),
            ),
        },
        "context_family_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(baseline_family_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_family_counter),
        },
        "context_family_rates": {
            _COMPARISON_GROUP_PRESERVED: _rate_map(
                baseline_family_counter,
                len(baseline_rows),
            ),
            _COMPARISON_GROUP_COLLAPSED: _rate_map(
                collapsed_family_counter,
                len(collapsed_rows),
            ),
        },
        "reason_bucket_counts": {
            _COMPARISON_GROUP_PRESERVED: dict(baseline_reason_counter),
            _COMPARISON_GROUP_COLLAPSED: dict(collapsed_reason_counter),
        },
        "reason_bucket_rates": {
            _COMPARISON_GROUP_PRESERVED: _rate_map(
                baseline_reason_counter,
                len(baseline_rows),
            ),
            _COMPARISON_GROUP_COLLAPSED: _rate_map(
                collapsed_reason_counter,
                len(collapsed_rows),
            ),
        },
        "dominant_context_family": {
            _COMPARISON_GROUP_PRESERVED: _dominant_value(
                baseline_family_counter,
                empty="none",
                order_map=_CONTEXT_FAMILY_ORDER,
            ),
            _COMPARISON_GROUP_COLLAPSED: _dominant_value(
                collapsed_family_counter,
                empty="none",
                order_map=_CONTEXT_FAMILY_ORDER,
            ),
        },
        "primary_reason_bucket": {
            _COMPARISON_GROUP_PRESERVED: _primary_reason_bucket(
                baseline_reason_counter,
                row_count=len(baseline_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
            _COMPARISON_GROUP_COLLAPSED: _primary_reason_bucket(
                collapsed_reason_counter,
                row_count=len(collapsed_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
        },
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    comparison_group_summaries: Sequence[dict[str, Any]],
    strategy_context_summaries: Sequence[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, list[str]]:
    group_map = _comparison_group_map(comparison_group_summaries)
    baseline = _safe_dict(group_map.get(_COMPARISON_GROUP_PRESERVED))
    collapsed = _safe_dict(group_map.get(_COMPARISON_GROUP_COLLAPSED))
    comparison_supported = str(comparison.get("support_status") or "") == "supported"

    facts = [
        (
            "Actionable selected-strategy rows="
            f"{summary.get('actionable_selected_strategy_row_count', 0)}; "
            f"dual_aligned_row_count={summary.get('dual_aligned_row_count', 0)}."
        ),
        (
            "Dual-aligned comparison groups: "
            f"preserved_dual_aligned_baseline={summary.get('preserved_dual_aligned_baseline_row_count', 0)}, "
            f"collapsed_dual_aligned={summary.get('collapsed_dual_aligned_row_count', 0)}, "
            f"other_dual_aligned_rule_outcome={summary.get('other_dual_aligned_rule_outcome_row_count', 0)}."
        ),
        (
            "Dual-aligned comparison support: "
            f"{summary.get('comparison_support_status', 'unknown')} "
            f"(baseline={summary.get('preserved_dual_aligned_baseline_row_count', 0)}, "
            f"collapsed={summary.get('collapsed_dual_aligned_row_count', 0)})."
        ),
    ]

    if int(baseline.get("row_count", 0) or 0) > 0:
        facts.append(
            "Preserved dual-aligned baseline dominant_context_family="
            f"{baseline.get('dominant_context_family', 'none')}; "
            "primary_reason_bucket="
            f"{baseline.get('primary_reason_bucket', 'no_rows')}."
        )
    if int(collapsed.get("row_count", 0) or 0) > 0:
        facts.append(
            "Collapsed dual-aligned dominant_context_family="
            f"{collapsed.get('dominant_context_family', 'none')}; "
            "primary_reason_bucket="
            f"{collapsed.get('primary_reason_bucket', 'no_rows')}."
        )

    strongest_strategy_slice = _strongest_supported_collapsed_strategy_slice(
        strategy_context_summaries
    )
    if strongest_strategy_slice:
        facts.append(strongest_strategy_slice)

    inferences: list[str] = []
    collapsed_family_rates = _safe_dict(comparison.get("context_family_rates")).get(
        _COMPARISON_GROUP_COLLAPSED,
        {},
    )
    baseline_family_rates = _safe_dict(comparison.get("context_family_rates")).get(
        _COMPARISON_GROUP_PRESERVED,
        {},
    )
    collapsed_combined_rates = _safe_dict(
        comparison.get("combined_context_relation_rates")
    ).get(_COMPARISON_GROUP_COLLAPSED, {})
    baseline_combined_rates = _safe_dict(
        comparison.get("combined_context_relation_rates")
    ).get(_COMPARISON_GROUP_PRESERVED, {})

    if comparison_supported:
        if _to_float(
            collapsed_family_rates.get("dual_context_neutral_or_missing"),
            default=0.0,
        ) > _to_float(
            baseline_family_rates.get("dual_context_neutral_or_missing"),
            default=0.0,
        ):
            inferences.append(
                "Within supported dual-aligned rows, collapsed hold outcomes are more concentrated in dual_context_neutral_or_missing than the preserved baseline, which is consistent with missing higher-level context support remaining a residual differentiator after setup+trigger co-activation."
            )
        if _to_float(
            collapsed_family_rates.get("any_context_opposition"),
            default=0.0,
        ) > _to_float(
            baseline_family_rates.get("any_context_opposition"),
            default=0.0,
        ):
            inferences.append(
                "Within supported dual-aligned rows, collapsed hold outcomes show more any_context_opposition than the preserved baseline, which is consistent with residual context-layer invalidation or disagreement still mattering after lower-layer confirmation is already present."
            )
        if _to_float(
            collapsed_family_rates.get("dual_context_aligned"),
            default=0.0,
        ) < _to_float(
            baseline_family_rates.get("dual_context_aligned"),
            default=0.0,
        ):
            inferences.append(
                "Collapsed dual-aligned rows retain less fully aligned context family support than preserved baseline rows, which suggests the remaining hold mechanism is more context-sensitive than the setup/trigger activation layer alone."
            )
        if _to_float(
            collapsed_combined_rates.get("aligned_with_selected"),
            default=0.0,
        ) < _to_float(
            baseline_combined_rates.get("aligned_with_selected"),
            default=0.0,
        ):
            inferences.append(
                "The combined context relation is less often aligned_with_selected in collapsed dual-aligned rows than in the preserved baseline, which is consistent with a higher-level context merge still differentiating final hold outcomes."
            )
    else:
        inferences.append(
            "The preserved-vs-collapsed dual-aligned comparison still has limited support in this slice, so context-family gaps should be treated as directional evidence rather than proof."
        )

    collapsed_reason_rates = _safe_dict(comparison.get("reason_bucket_rates")).get(
        _COMPARISON_GROUP_COLLAPSED,
        {},
    )
    if comparison_supported and any(
        _to_float(collapsed_reason_rates.get(bucket), default=0.0) > 0.0
        for bucket in (
            "context_not_supportive",
            "opposition_or_invalidated",
        )
    ):
        inferences.append(
            "Persisted reason text in the collapsed dual-aligned subset is directionally consistent with the structured context-family decomposition, but it remains supporting evidence rather than literal proof of the final internal merge branch."
        )

    uncertainties = [
        "This report does not prove the exact internal merge rule, threshold, or branch that converts dual-aligned selected-strategy rows into final hold outcomes.",
        "Reason buckets are transparent text-pattern heuristics over persisted explanations and should not be treated as exact pipeline traces.",
        "Strategy-level and symbol-level context-family slices are descriptive only; they do not prove that the same residual mechanism is stable across every strategy or symbol.",
    ]
    if not comparison_supported:
        uncertainties.append(
            "Because the primary preserved-vs-collapsed comparison has limited support in this slice, the context-layer gap should be read as a focused residual hypothesis rather than as a fully stable conclusion."
        )

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
    comparison = _safe_dict(widest.get("context_family_comparison"))
    observations = _safe_dict(widest.get("key_observations"))

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
        "overall_conclusion": _overall_conclusion(
            summary=summary,
            comparison=comparison,
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
            "- dual_aligned_row_count: "
            f"{headline.get('dual_aligned_row_count', 0)}"
        )
        lines.append(
            "- preserved_dual_aligned_baseline_row_count: "
            f"{headline.get('preserved_dual_aligned_baseline_row_count', 0)}"
        )
        lines.append(
            "- collapsed_dual_aligned_row_count: "
            f"{headline.get('collapsed_dual_aligned_row_count', 0)}"
        )
        lines.append(
            "- comparison_support_status: "
            f"{headline.get('comparison_support_status', 'unknown')}"
        )
        lines.append(
            "- preserved_dominant_context_family: "
            f"{headline.get('preserved_dominant_context_family', 'none')}"
        )
        lines.append(
            "- collapsed_dominant_context_family: "
            f"{headline.get('collapsed_dominant_context_family', 'none')}"
        )
        lines.append(
            "- collapsed_primary_reason_bucket: "
            f"{headline.get('collapsed_primary_reason_bucket', 'no_rows')}"
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
    for item in _safe_list(final_assessment.get("strongly_suggested"))[:3]:
        lines.append(f"- suggested: {item}")
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


def _build_dual_aligned_context_row(row: dict[str, Any]) -> dict[str, Any]:
    selected_signal = str(row.get("selected_strategy_result_signal_state") or "")
    context_state_relation = _context_state_relation_to_selected(
        context_state=row.get("context_state"),
        selected_signal=selected_signal,
    )
    context_bias_relation = _context_bias_relation_to_selected(
        context_bias=row.get("context_bias"),
        selected_signal=selected_signal,
    )
    combined_context_relation = _combined_context_relation_to_selected(
        context_state_relation=context_state_relation,
        context_bias_relation=context_bias_relation,
    )
    context_family = _context_family(
        context_state_relation=context_state_relation,
        context_bias_relation=context_bias_relation,
    )
    rule_reason_bucket = _bucket_reason_text(row.get("rule_reason_text"))
    root_reason_bucket = _bucket_reason_text(row.get("root_reason_text"))
    combined_reason_bucket, combined_reason_source = _combined_reason_bucket(
        rule_reason_bucket=rule_reason_bucket,
        root_reason_bucket=root_reason_bucket,
    )

    return {
        **row,
        "activation_gap_comparison_group": row.get("comparison_group"),
        "comparison_group": _dual_aligned_comparison_group(
            activation_gap_comparison_group=str(row.get("comparison_group") or ""),
        ),
        "context_state_relation_to_selected": context_state_relation,
        "context_bias_relation_to_selected": context_bias_relation,
        "combined_context_relation_to_selected": combined_context_relation,
        "context_family": context_family,
        "rule_reason_bucket": rule_reason_bucket,
        "root_reason_bucket": root_reason_bucket,
        "combined_reason_bucket": combined_reason_bucket,
        "combined_reason_bucket_source": combined_reason_source,
    }


def _dual_aligned_comparison_group(*, activation_gap_comparison_group: str) -> str:
    if activation_gap_comparison_group == activation_gap_module._GROUP_PRESERVED:
        return _COMPARISON_GROUP_PRESERVED
    if activation_gap_comparison_group == activation_gap_module._GROUP_COLLAPSED:
        return _COMPARISON_GROUP_COLLAPSED
    return _COMPARISON_GROUP_OTHER


def _context_state_relation_to_selected(
    *,
    context_state: Any,
    selected_signal: str,
) -> str:
    return _state_relation_to_selected(
        state_value=context_state,
        selected_signal=selected_signal,
    )


def _context_bias_relation_to_selected(
    *,
    context_bias: Any,
    selected_signal: str,
) -> str:
    return _bias_relation(context_bias, selected_signal)


def _combined_context_relation_to_selected(
    *,
    context_state_relation: str,
    context_bias_relation: str,
) -> str:
    if context_state_relation == "mixed_or_inconclusive":
        return "mixed_or_inconclusive"

    if context_state_relation == "aligned_with_selected":
        if context_bias_relation == "opposite_to_selected":
            return "mixed_or_inconclusive"
        return "aligned_with_selected"

    if context_state_relation == "opposite_to_selected":
        if context_bias_relation == "aligned_with_selected":
            return "mixed_or_inconclusive"
        return "opposite_to_selected"

    if context_bias_relation in {"aligned_with_selected", "opposite_to_selected"}:
        return context_bias_relation

    return "neutral_or_missing"


def _context_family(
    *,
    context_state_relation: str,
    context_bias_relation: str,
) -> str:
    if (
        context_state_relation == "aligned_with_selected"
        and context_bias_relation == "aligned_with_selected"
    ):
        return "dual_context_aligned"
    if (
        context_state_relation == "aligned_with_selected"
        and context_bias_relation == "neutral_or_missing"
    ):
        return "context_state_only_aligned"
    if (
        context_state_relation == "neutral_or_missing"
        and context_bias_relation == "aligned_with_selected"
    ):
        return "context_bias_only_aligned"
    if (
        context_state_relation == "neutral_or_missing"
        and context_bias_relation == "neutral_or_missing"
    ):
        return "dual_context_neutral_or_missing"
    if "mixed_or_inconclusive" in {
        context_state_relation,
        context_bias_relation,
    }:
        return "context_mixed_or_inconclusive"
    if "opposite_to_selected" in {
        context_state_relation,
        context_bias_relation,
    }:
        return "any_context_opposition"
    return "context_mixed_or_inconclusive"


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    support_threshold: int,
) -> dict[str, Any]:
    row_count = len(rows)
    context_state_counter = Counter(
        str(row.get("context_state_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    context_bias_counter = Counter(
        str(row.get("context_bias_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    combined_context_counter = Counter(
        str(row.get("combined_context_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    context_family_counter = Counter(
        str(row.get("context_family") or "context_mixed_or_inconclusive")
        for row in rows
    )
    combined_reason_counter = Counter(
        str(row.get("combined_reason_bucket") or "insufficient_explanation")
        for row in rows
    )
    rule_reason_counter = Counter(
        str(row.get("rule_reason_bucket") or "insufficient_explanation")
        for row in rows
    )
    root_reason_counter = Counter(
        str(row.get("root_reason_bucket") or "insufficient_explanation")
        for row in rows
    )
    reason_source_counter = Counter(
        str(row.get("combined_reason_bucket_source") or "no_reason_text")
        for row in rows
    )

    return {
        "row_count": row_count,
        "support_status": (
            "supported"
            if row_count >= max(1, int(support_threshold))
            else "limited_support"
        ),
        "context_state_relation_counts": dict(context_state_counter),
        "context_state_relation_rates": _rate_map(context_state_counter, row_count),
        "context_bias_relation_counts": dict(context_bias_counter),
        "context_bias_relation_rates": _rate_map(context_bias_counter, row_count),
        "combined_context_relation_counts": dict(combined_context_counter),
        "combined_context_relation_rates": _rate_map(
            combined_context_counter,
            row_count,
        ),
        "context_family_counts": dict(context_family_counter),
        "context_family_rates": _rate_map(context_family_counter, row_count),
        "reason_bucket_counts": dict(combined_reason_counter),
        "reason_bucket_rates": _rate_map(combined_reason_counter, row_count),
        "rule_reason_bucket_counts": dict(rule_reason_counter),
        "root_reason_bucket_counts": dict(root_reason_counter),
        "reason_bucket_source_counts": dict(reason_source_counter),
        "dominant_context_family": _dominant_value(
            context_family_counter,
            empty="none",
            order_map=_CONTEXT_FAMILY_ORDER,
        ),
        "primary_reason_bucket": _primary_reason_bucket(
            combined_reason_counter,
            row_count=row_count,
            support_threshold=max(1, int(support_threshold)),
        ),
    }


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


def _strongest_supported_collapsed_strategy_slice(
    strategy_context_summaries: Sequence[dict[str, Any]],
) -> str | None:
    collapsed_rows = [
        _safe_dict(row)
        for row in strategy_context_summaries
        if _safe_dict(row).get("comparison_group") == _COMPARISON_GROUP_COLLAPSED
        and _row_is_supported(_safe_dict(row))
    ]
    if not collapsed_rows:
        return None
    collapsed_rows.sort(
        key=lambda item: (
            -int(item.get("row_count", 0) or 0),
            str(item.get("strategy") or ""),
            _CONTEXT_FAMILY_ORDER.get(str(item.get("context_family") or ""), 99),
        )
    )
    row = collapsed_rows[0]
    return (
        "Largest supported collapsed strategy/context_family slice: "
        f"{row.get('strategy', _MISSING_LABEL)} / "
        f"{row.get('context_family', _MISSING_LABEL)} "
        f"(n={int(row.get('row_count', 0) or 0)})."
    )


def _overall_conclusion(
    *,
    summary: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    dual_aligned_row_count = int(summary.get("dual_aligned_row_count", 0) or 0)
    baseline_row_count = int(
        summary.get("preserved_dual_aligned_baseline_row_count", 0) or 0
    )
    collapsed_row_count = int(summary.get("collapsed_dual_aligned_row_count", 0) or 0)
    comparison_supported = str(comparison.get("support_status") or "") == "supported"

    if dual_aligned_row_count <= 0:
        return (
            "No actionable selected-strategy rows with dual-aligned setup+trigger were observed in the widest configuration, so this focused residual diagnosis has no target comparison slice."
        )
    if baseline_row_count <= 0 or collapsed_row_count <= 0:
        return (
            "Dual-aligned rows were observed, but one side of the preserved-vs-collapsed comparison is missing in the widest configuration, so the residual context-layer comparison remains incomplete."
        )
    if not comparison_supported:
        return (
            "The widest configuration shows a real dual-aligned residual slice, but the preserved-vs-collapsed comparison still has limited support, so context-layer differentiators should be treated as focused directional evidence rather than as stable proof."
        )

    collapsed_family_rates = _safe_dict(comparison.get("context_family_rates")).get(
        _COMPARISON_GROUP_COLLAPSED,
        {},
    )
    baseline_family_rates = _safe_dict(comparison.get("context_family_rates")).get(
        _COMPARISON_GROUP_PRESERVED,
        {},
    )

    if _to_float(
        collapsed_family_rates.get("dual_context_neutral_or_missing"),
        default=0.0,
    ) > _to_float(
        baseline_family_rates.get("dual_context_neutral_or_missing"),
        default=0.0,
    ) or _to_float(
        collapsed_family_rates.get("any_context_opposition"),
        default=0.0,
    ) > _to_float(
        baseline_family_rates.get("any_context_opposition"),
        default=0.0,
    ):
        return (
            "The widest configuration supports a clean dual-aligned residual comparison: once setup+trigger are already aligned, collapsed hold rows differ from the preserved baseline mainly through weaker or opposing context-family support, which strongly suggests a higher-level context-sensitive conservative layer still matters after lower-layer confirmation is present."
        )

    return (
        "The widest configuration supports the clean dual-aligned residual comparison, but the structured context-family gap is not stronger in collapsed hold rows than in the preserved baseline for this slice."
    )


def _comparison_support_status(
    *,
    baseline_row_count: int,
    collapsed_row_count: int,
) -> str:
    if (
        baseline_row_count >= _MIN_PRIMARY_SUPPORT_ROWS
        and collapsed_row_count >= _MIN_PRIMARY_SUPPORT_ROWS
    ):
        return "supported"
    return "limited_support"


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
    return _MISSING_LABEL


def _sort_group_value(field: str, value: Any) -> Any:
    text = str(value or _MISSING_LABEL)
    if field == "comparison_group":
        return _COMPARISON_GROUP_ORDER.get(text, 99)
    if field == "context_family":
        return _CONTEXT_FAMILY_ORDER.get(text, 99)
    return text


def _row_is_supported(row: dict[str, Any] | None) -> bool:
    item = _safe_dict(row)
    return (
        bool(item)
        and str(item.get("support_status") or "") == "supported"
        and int(item.get("row_count", 0) or 0) > 0
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
    return hold_reason_module._materialize_configuration_input(
        input_path=input_path,
        run_output_dir=run_output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(hold_reason_module, "_build_stage_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_dual_aligned_context_hold_residual_diagnosis_report requires "
            "hold-resolution stage-row extraction support, but _build_stage_row is unavailable."
        )
    return builder(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(activation_gap_module, "_build_activation_gap_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_dual_aligned_context_hold_residual_diagnosis_report requires "
            "activation-gap row construction support, but _build_activation_gap_row is unavailable."
        )
    return builder(row)


def _state_relation_to_selected(*, state_value: Any, selected_signal: str) -> str:
    helper = getattr(activation_gap_module, "_state_relation_to_selected", None)
    if not callable(helper):
        return "neutral_or_missing"
    return str(
        helper(state_value=state_value, selected_signal=selected_signal)
        or "neutral_or_missing"
    )


def _bias_relation(value: Any, selected_signal: str) -> str:
    helper = getattr(activation_gap_module, "_bias_relation", None)
    if not callable(helper):
        return "neutral_or_missing"
    return str(helper(value, selected_signal) or "neutral_or_missing")


def _bucket_reason_text(text: Any) -> str:
    helper = getattr(residual_module, "_bucket_reason_text", None)
    if not callable(helper):
        return "insufficient_explanation"
    return str(helper(text) or "insufficient_explanation")


def _combined_reason_bucket(
    *,
    rule_reason_bucket: str,
    root_reason_bucket: str,
) -> tuple[str, str]:
    helper = getattr(residual_module, "_combined_reason_bucket", None)
    if not callable(helper):
        if rule_reason_bucket != "insufficient_explanation":
            return rule_reason_bucket, "rule_reason_text"
        if root_reason_bucket != "insufficient_explanation":
            return root_reason_bucket, "root_reason_text"
        return "insufficient_explanation", "no_reason_text"

    bucket, source = helper(
        rule_reason_bucket=rule_reason_bucket,
        root_reason_bucket=root_reason_bucket,
    )
    return (
        str(bucket or "insufficient_explanation"),
        str(source or "no_reason_text"),
    )


def _rate_map(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: _safe_ratio(value, total) for key, value in counter.items()}


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