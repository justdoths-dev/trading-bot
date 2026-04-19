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
    selected_strategy_setup_trigger_activation_gap_diagnosis_report as activation_gap_module,
)

REPORT_TYPE = "selected_strategy_post_confirmation_hold_residual_diagnosis_report"
REPORT_TITLE = "Selected Strategy Post Confirmation Hold Residual Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_MIN_SYMBOL_SUPPORT = 10

_MISSING_LABEL = "(missing)"
_PRIMARY_FACTOR_THRESHOLD = 0.60
_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_STRATEGY_SUPPORT_ROWS = 10

_RESIDUAL_GROUP_PRESERVED_BASELINE = "preserved_dual_aligned_baseline"
_RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED = "collapsed_dual_aligned"
_RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY = "collapsed_setup_only"
_RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY = "collapsed_trigger_only"
_RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION = "collapsed_any_opposition"
_RESIDUAL_GROUP_COLLAPSED_MIXED = "collapsed_mixed_or_inconclusive"
_RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL = "collapsed_dual_neutral_or_missing"

_TARGET_RESIDUAL_GROUPS = {
    _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED,
    _RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY,
    _RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY,
    _RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION,
    _RESIDUAL_GROUP_COLLAPSED_MIXED,
}
_REFERENCE_RESIDUAL_GROUPS = {_RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL}
_BASELINE_RESIDUAL_GROUPS = {_RESIDUAL_GROUP_PRESERVED_BASELINE}
_RESIDUAL_GROUP_ORDER = {
    _RESIDUAL_GROUP_PRESERVED_BASELINE: 0,
    _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: 1,
    _RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY: 2,
    _RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY: 3,
    _RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION: 4,
    _RESIDUAL_GROUP_COLLAPSED_MIXED: 5,
    _RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL: 6,
}
_GROUP_ROLE_ORDER = {
    "baseline": 0,
    "target": 1,
    "reference": 2,
}
_RELATION_ORDER = {
    "aligned_with_selected": 0,
    "neutral_or_missing": 1,
    "mixed_or_inconclusive": 2,
    "opposite_to_selected": 3,
    "none": 9,
}

_REASON_BUCKET_CONFIRMATION = "confirmation_not_ready_or_missing"
_REASON_BUCKET_CONFLICT = "conflict_or_disagreement"
_REASON_BUCKET_RISK = "risk_or_filter_rejection"
_REASON_BUCKET_CONTEXT = "context_not_supportive"
_REASON_BUCKET_OPPOSITION = "opposition_or_invalidated"
_REASON_BUCKET_INSUFFICIENT = "insufficient_explanation"
_REASON_BUCKET_OTHER = "other"
_PRIMARY_REASON_BUCKET_MIXED = "mixed_or_inconclusive"
_PRIMARY_REASON_BUCKET_INSUFFICIENT_SUPPORT = "insufficient_support"
_PRIMARY_REASON_BUCKET_NO_ROWS = "no_rows"

_REASON_BUCKET_ORDER = {
    _REASON_BUCKET_CONFIRMATION: 0,
    _REASON_BUCKET_CONFLICT: 1,
    _REASON_BUCKET_RISK: 2,
    _REASON_BUCKET_CONTEXT: 3,
    _REASON_BUCKET_OPPOSITION: 4,
    _REASON_BUCKET_INSUFFICIENT: 5,
    _REASON_BUCKET_OTHER: 6,
    _PRIMARY_REASON_BUCKET_MIXED: 7,
    _PRIMARY_REASON_BUCKET_INSUFFICIENT_SUPPORT: 8,
    _PRIMARY_REASON_BUCKET_NO_ROWS: 9,
}

_REASON_CONFIRMATION_MARKERS = (
    "confirmation",
    "confirmations",
    "not confirmed",
    "not ready",
    "await",
    "waiting",
    "pending",
    "setup",
    "trigger",
    "activation gap",
    "activation",
    "not both",
    "not enough",
    "insufficient confirmation",
    "needs confirmation",
    "requires confirmation",
)
_REASON_CONFLICT_MARKERS = (
    "conflict",
    "conflicted",
    "conflicting",
    "disagree",
    "disagreement",
    "mixed",
    "inconclusive",
    "misalign",
    "misaligned",
    "divergent",
)
_REASON_RISK_MARKERS = (
    "risk",
    "filter",
    "filtered",
    "volatility",
    "spread",
    "slippage",
    "liquidity",
    "cooldown",
    "exposure",
    "max position",
    "max_positions",
    "reward",
    "rr ",
    "r:r",
    "guard",
    "cap",
    "threshold",
    "too extended",
)
_REASON_CONTEXT_MARKERS = (
    "context",
    "broader",
    "macro",
    "structure",
    "regime",
    "trend",
    "background",
    "market state",
)
_REASON_CONTEXT_NEGATIVE_MARKERS = (
    "neutral",
    "sideways",
    "flat",
    "not supportive",
    "unsupportive",
    "not favorable",
    "not favourable",
    "weak context",
    "range-bound",
    "range bound",
)
_REASON_OPPOSITION_MARKERS = (
    "oppose",
    "opposed",
    "opposition",
    "against",
    "counter",
    "opposite",
    "invalid",
    "invalidate",
    "invalidated",
    "rejected",
    "rejection",
)

DiagnosisConfiguration = hold_reason_module.DiagnosisConfiguration
DEFAULT_CONFIGURATIONS = hold_reason_module.DEFAULT_CONFIGURATIONS
_ACTIONABLE_SIGNAL_STATES = frozenset(
    str(value)
    for value in getattr(
        activation_gap_module,
        "_ACTIONABLE_SIGNAL_STATES",
        {"long", "short"},
    )
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for actionable selected-strategy rows "
            "that still collapse to hold after setup/trigger confirmation activity "
            "becomes visible."
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
        help="Minimum rows required for strategy+symbol+residual_group summaries.",
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

    result = run_selected_strategy_post_confirmation_hold_residual_diagnosis_report(
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
                "collapsed_to_hold_row_count": summary.get(
                    "collapsed_to_hold_row_count",
                    0,
                ),
                "residual_target_row_count": summary.get(
                    "residual_target_row_count",
                    0,
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run_selected_strategy_post_confirmation_hold_residual_diagnosis_report(
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
        "residual_group_summaries": _safe_list(
            widest_summary.get("residual_group_summaries")
        ),
        "strategy_residual_summaries": _safe_list(
            widest_summary.get("strategy_residual_summaries")
        ),
        "symbol_residual_summaries": _safe_list(
            widest_summary.get("symbol_residual_summaries")
        ),
        "preserved_vs_collapsed_dual_aligned_comparison": _safe_dict(
            widest_summary.get("preserved_vs_collapsed_dual_aligned_comparison")
        ),
        "key_observations": _safe_dict(widest_summary.get("key_observations")),
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses the hold-resolution stage-row extraction plus activation-gap row semantics so the new residual split stays aligned with the prior diagnosis artifacts.",
            "Residual targeting stays within actionable selected-strategy rows and does not reopen mapper, engine, execution, or candidate-quality production logic.",
            "Context relation uses the same conservative state-first relation vocabulary as setup/trigger, but it is applied to context_state/context_bias only rather than to any inferred hidden pipeline state.",
            "Reason buckets are transparent string-pattern heuristics over persisted reason text; they are descriptive diagnostics rather than proof of the exact internal live-rule branch.",
            "Residual target rows exclude collapsed dual_neutral_or_missing rows so the report focuses on the post-confirmation subset that remains after the activation-gap framing is already established.",
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
    residual_rows = [
        residual_row
        for residual_row in (_build_residual_row(row) for row in actionable_rows)
        if residual_row is not None
    ]
    overall_group_summaries = activation_gap_module.build_overall_group_summaries(
        actionable_rows
    )
    summary = build_summary(
        actionable_rows=actionable_rows,
        residual_rows=residual_rows,
        overall_group_summaries=overall_group_summaries,
        min_symbol_support=min_symbol_support,
    )
    residual_group_summaries = build_residual_group_summaries(
        residual_rows=residual_rows,
        summary=summary,
    )
    strategy_residual_summaries = build_group_summaries(
        residual_rows,
        group_fields=("strategy", "residual_group"),
        support_threshold=_MIN_STRATEGY_SUPPORT_ROWS,
    )
    symbol_residual_summaries = build_group_summaries(
        residual_rows,
        group_fields=("symbol", "strategy", "residual_group"),
        support_threshold=max(1, min_symbol_support),
        min_row_count=max(1, min_symbol_support),
    )
    preserved_vs_collapsed_dual_aligned_comparison = (
        build_preserved_vs_collapsed_dual_aligned_comparison(residual_rows)
    )
    key_observations = build_key_observations(
        summary=summary,
        residual_group_summaries=residual_group_summaries,
        strategy_residual_summaries=strategy_residual_summaries,
        comparison=preserved_vs_collapsed_dual_aligned_comparison,
    )

    residual_group_map = _residual_group_map(residual_group_summaries)
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
            "collapsed_to_hold_row_count": summary["collapsed_to_hold_row_count"],
            "residual_target_row_count": summary["residual_target_row_count"],
            "collapsed_dual_neutral_or_missing_row_count": summary[
                "collapsed_dual_neutral_or_missing_row_count"
            ],
            "dominant_residual_group": _dominant_residual_group_label(
                residual_group_summaries
            ),
            "collapsed_dual_aligned_row_count": int(
                _safe_dict(
                    residual_group_map.get(_RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED)
                ).get("row_count", 0)
                or 0
            ),
            "collapsed_any_opposition_row_count": int(
                _safe_dict(
                    residual_group_map.get(_RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION)
                ).get("row_count", 0)
                or 0
            ),
        },
        "summary": summary,
        "residual_group_summaries": residual_group_summaries,
        "strategy_residual_summaries": strategy_residual_summaries,
        "symbol_residual_summaries": symbol_residual_summaries,
        "preserved_vs_collapsed_dual_aligned_comparison": (
            preserved_vs_collapsed_dual_aligned_comparison
        ),
        "key_observations": key_observations,
    }


def build_summary(
    *,
    actionable_rows: Sequence[dict[str, Any]],
    residual_rows: Sequence[dict[str, Any]],
    overall_group_summaries: Sequence[dict[str, Any]],
    min_symbol_support: int,
) -> dict[str, Any]:
    group_map = activation_gap_module._group_map(overall_group_summaries)
    preserved = _safe_dict(group_map.get(activation_gap_module._GROUP_PRESERVED))
    collapsed = _safe_dict(group_map.get(activation_gap_module._GROUP_COLLAPSED))

    residual_target_row_count = sum(
        1 for row in residual_rows if row.get("residual_group_role") == "target"
    )
    collapsed_dual_neutral_or_missing_row_count = sum(
        1
        for row in residual_rows
        if row.get("residual_group") == _RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL
    )
    collapsed_mixed_or_inconclusive_row_count = sum(
        1
        for row in residual_rows
        if row.get("residual_group") == _RESIDUAL_GROUP_COLLAPSED_MIXED
    )

    return {
        "actionable_selected_strategy_row_count": len(actionable_rows),
        "preserved_aligned_actionable_row_count": int(
            preserved.get("row_count", 0) or 0
        ),
        "collapsed_to_hold_row_count": int(collapsed.get("row_count", 0) or 0),
        "residual_target_row_count": residual_target_row_count,
        "collapsed_dual_neutral_or_missing_row_count": (
            collapsed_dual_neutral_or_missing_row_count
        ),
        "collapsed_mixed_or_inconclusive_row_count": (
            collapsed_mixed_or_inconclusive_row_count
        ),
        "residual_target_share_within_collapsed_to_hold": _safe_ratio(
            residual_target_row_count,
            int(collapsed.get("row_count", 0) or 0),
        ),
        "min_symbol_support": max(1, int(min_symbol_support)),
    }


def build_residual_group_summaries(
    *,
    residual_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    collapsed_total = int(summary.get("collapsed_to_hold_row_count", 0) or 0)
    residual_target_total = int(summary.get("residual_target_row_count", 0) or 0)
    preserved_total = int(summary.get("preserved_aligned_actionable_row_count", 0) or 0)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        grouped[str(row.get("residual_group"))].append(row)

    group_names = [
        _RESIDUAL_GROUP_PRESERVED_BASELINE,
        _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED,
        _RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY,
        _RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY,
        _RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION,
    ]
    if grouped.get(_RESIDUAL_GROUP_COLLAPSED_MIXED):
        group_names.append(_RESIDUAL_GROUP_COLLAPSED_MIXED)
    group_names.append(_RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL)

    summaries: list[dict[str, Any]] = []
    for residual_group in group_names:
        rows = grouped.get(residual_group, [])
        role = _residual_group_role(residual_group)
        row_summary = _summarize_rows(
            rows,
            support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
        )
        row_summary.update(
            {
                "residual_group": residual_group,
                "group_role": role,
                "is_residual_target_group": role == "target",
                "collapsed_share": (
                    _safe_ratio(len(rows), collapsed_total)
                    if role in {"target", "reference"}
                    else 0.0
                ),
                "residual_target_share": (
                    _safe_ratio(len(rows), residual_target_total)
                    if role == "target"
                    else 0.0
                ),
                "preserved_share": (
                    _safe_ratio(len(rows), preserved_total)
                    if role == "baseline"
                    else 0.0
                ),
            }
        )
        summaries.append(row_summary)

    summaries.sort(
        key=lambda item: _sort_group_value("residual_group", item.get("residual_group"))
    )
    return summaries


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

    summaries: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        if len(grouped_rows) < max(1, int(min_row_count)):
            continue
        summary = _summarize_rows(
            grouped_rows,
            support_threshold=max(1, int(support_threshold)),
        )
        row = {field: key[index] for index, field in enumerate(group_fields)}
        if "residual_group" in group_fields:
            row["group_role"] = _residual_group_role(str(row.get("residual_group")))
        row.update(summary)
        summaries.append(row)

    summaries.sort(
        key=lambda item: tuple(
            _sort_group_value(field, item.get(field)) for field in group_fields
        )
        + (-int(item.get("row_count", 0) or 0),)
    )
    return summaries


def build_preserved_vs_collapsed_dual_aligned_comparison(
    residual_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_rows = [
        row
        for row in residual_rows
        if row.get("residual_group") == _RESIDUAL_GROUP_PRESERVED_BASELINE
    ]
    collapsed_rows = [
        row
        for row in residual_rows
        if row.get("residual_group") == _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED
    ]

    baseline_context_counter = Counter(
        str(row.get("context_relation_to_selected") or "neutral_or_missing")
        for row in baseline_rows
    )
    collapsed_context_counter = Counter(
        str(row.get("context_relation_to_selected") or "neutral_or_missing")
        for row in collapsed_rows
    )
    baseline_reason_counter = Counter(
        str(row.get("reason_bucket") or _REASON_BUCKET_INSUFFICIENT)
        for row in baseline_rows
    )
    collapsed_reason_counter = Counter(
        str(row.get("reason_bucket") or _REASON_BUCKET_INSUFFICIENT)
        for row in collapsed_rows
    )

    return {
        "support_status": (
            "supported"
            if len(baseline_rows) >= _MIN_PRIMARY_SUPPORT_ROWS
            and len(collapsed_rows) >= _MIN_PRIMARY_SUPPORT_ROWS
            else "limited_support"
        ),
        "preserved_dual_aligned_baseline_row_count": len(baseline_rows),
        "collapsed_dual_aligned_row_count": len(collapsed_rows),
        "context_relation_counts": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: dict(baseline_context_counter),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: dict(collapsed_context_counter),
        },
        "context_relation_rates": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: _rate_map(
                baseline_context_counter,
                len(baseline_rows),
            ),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: _rate_map(
                collapsed_context_counter,
                len(collapsed_rows),
            ),
        },
        "reason_bucket_counts": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: dict(baseline_reason_counter),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: dict(collapsed_reason_counter),
        },
        "reason_bucket_rates": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: _rate_map(
                baseline_reason_counter,
                len(baseline_rows),
            ),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: _rate_map(
                collapsed_reason_counter,
                len(collapsed_rows),
            ),
        },
        "dominant_context_relation": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: _dominant_value(
                baseline_context_counter,
                empty="none",
                order_map=_RELATION_ORDER,
            ),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: _dominant_value(
                collapsed_context_counter,
                empty="none",
                order_map=_RELATION_ORDER,
            ),
        },
        "primary_reason_bucket": {
            _RESIDUAL_GROUP_PRESERVED_BASELINE: _primary_reason_bucket(
                baseline_reason_counter,
                row_count=len(baseline_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
            _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED: _primary_reason_bucket(
                collapsed_reason_counter,
                row_count=len(collapsed_rows),
                support_threshold=_MIN_PRIMARY_SUPPORT_ROWS,
            ),
        },
    }


def build_key_observations(
    *,
    summary: dict[str, Any],
    residual_group_summaries: Sequence[dict[str, Any]],
    strategy_residual_summaries: Sequence[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, list[str]]:
    residual_map = _residual_group_map(residual_group_summaries)
    top_target = _top_target_residual_group(residual_group_summaries)
    collapsed_dual = _safe_dict(
        residual_map.get(_RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED)
    )
    any_opposition = _safe_dict(
        residual_map.get(_RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION)
    )
    comparison_supported = str(comparison.get("support_status") or "") == "supported"
    top_target_supported = _row_is_supported(top_target)

    facts = [
        (
            "Actionable selected-strategy rows="
            f"{summary.get('actionable_selected_strategy_row_count', 0)}; "
            f"preserved_aligned_actionable={summary.get('preserved_aligned_actionable_row_count', 0)}; "
            f"collapsed_to_hold={summary.get('collapsed_to_hold_row_count', 0)}."
        ),
        (
            "Residual target rows="
            f"{summary.get('residual_target_row_count', 0)} after excluding "
            "collapsed_dual_neutral_or_missing="
            f"{summary.get('collapsed_dual_neutral_or_missing_row_count', 0)}."
        ),
        (
            "Preserved-vs-collapsed dual-aligned comparison support: "
            f"{comparison.get('support_status', 'unknown')} "
            f"(baseline={int(comparison.get('preserved_dual_aligned_baseline_row_count', 0) or 0)}, "
            f"collapsed={int(comparison.get('collapsed_dual_aligned_row_count', 0) or 0)})."
        ),
    ]
    if top_target is not None:
        facts.append(
            "Largest residual target group: "
            f"{top_target.get('residual_group')} "
            f"(n={int(top_target.get('row_count', 0) or 0)}, "
            f"support_status={top_target.get('support_status', 'unknown')}, "
            f"collapsed_share={_to_float(top_target.get('collapsed_share'), default=0.0):.2%}, "
            f"primary_reason_bucket={top_target.get('primary_reason_bucket', _PRIMARY_REASON_BUCKET_NO_ROWS)})."
        )
    if int(collapsed_dual.get("row_count", 0) or 0) > 0:
        facts.append(
            "Collapsed dual-aligned rows remained observable after activation was present: "
            f"n={int(collapsed_dual.get('row_count', 0) or 0)}, "
            f"support_status={collapsed_dual.get('support_status', 'unknown')}, "
            "dominant_context_relation="
            f"{collapsed_dual.get('dominant_context_relation', 'none')}, "
            "primary_reason_bucket="
            f"{collapsed_dual.get('primary_reason_bucket', _PRIMARY_REASON_BUCKET_NO_ROWS)}."
        )
    if int(any_opposition.get("row_count", 0) or 0) > 0:
        facts.append(
            "Collapsed rows with setup/trigger opposition remained present: "
            f"n={int(any_opposition.get('row_count', 0) or 0)}, "
            f"support_status={any_opposition.get('support_status', 'unknown')}."
        )

    strongest_strategy = _strongest_strategy_slice(strategy_residual_summaries)
    if strongest_strategy:
        facts.append(strongest_strategy)

    inferences: list[str] = []
    residual_target_row_count = int(summary.get("residual_target_row_count", 0) or 0)
    if residual_target_row_count >= _MIN_PRIMARY_SUPPORT_ROWS:
        inferences.append(
            "Because residual_target rows exclude the pure dual_neutral_or_missing subset, the remaining collapsed rows are consistent with post-confirmation hold mechanisms rather than with activation-gap absence alone."
        )
    elif residual_target_row_count > 0:
        inferences.append(
            "A post-confirmation residual subset is observable, but its total support is still limited, so any residual-mechanism interpretation should be treated as provisional."
        )

    if top_target is not None and top_target_supported:
        residual_group = str(top_target.get("residual_group") or "")
        if residual_group in {
            _RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY,
            _RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY,
        }:
            inferences.append(
                "The largest supported residual target group is a partial-confirmation family, which suggests one-sided setup/trigger alignment often remains insufficient for rule preservation even after downstream activity appears."
            )
        elif residual_group == _RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION:
            inferences.append(
                "Residual opposition remains a supported hold pathway, which suggests some post-confirmation collapses are better explained by disagreement or invalidation than by missing activation."
            )
        elif residual_group == _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED:
            inferences.append(
                "Supported dual-aligned rows still collapsing to hold suggest a residual mechanism beyond raw setup/trigger activation, such as context mismatch, risk/filter conservatism, or a higher-level merge rule."
            )

    baseline_context = _safe_dict(comparison.get("context_relation_rates")).get(
        _RESIDUAL_GROUP_PRESERVED_BASELINE,
        {},
    )
    collapsed_context = _safe_dict(comparison.get("context_relation_rates")).get(
        _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED,
        {},
    )
    if comparison_supported and baseline_context and collapsed_context:
        if _to_float(
            collapsed_context.get("aligned_with_selected"),
            default=0.0,
        ) < _to_float(
            baseline_context.get("aligned_with_selected"),
            default=0.0,
        ):
            inferences.append(
                "Within supported dual-aligned rows, preserved baseline rows retain more context alignment than collapsed dual-aligned rows, which is consistent with context support still mattering after setup+trigger activation is present."
            )

    collapsed_reason_rates = _safe_dict(comparison.get("reason_bucket_rates")).get(
        _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED,
        {},
    )
    if comparison_supported and any(
        _to_float(collapsed_reason_rates.get(bucket), default=0.0) > 0.0
        for bucket in (
            _REASON_BUCKET_RISK,
            _REASON_BUCKET_CONTEXT,
            _REASON_BUCKET_OPPOSITION,
        )
    ):
        inferences.append(
            "The supported collapsed dual-aligned subset is consistent with residual conservatism beyond missing confirmation because its persisted reasons extend into context, opposition, or risk/filter buckets."
        )

    uncertainties = [
        "This report does not prove the exact internal rule, merge branch, or threshold that resolved the residual rows to hold.",
        "Reason buckets are heuristic summaries over persisted text and may understate nuanced or templated explanations.",
        "Symbol-level slices use a reporting support threshold only and should not be treated as production recommendations.",
    ]
    if not comparison_supported:
        uncertainties.append(
            "The preserved-vs-collapsed dual-aligned comparison has limited support in this slice, so any context or reason-bucket gap within that comparison should be treated as directional evidence rather than proof."
        )
    if top_target is not None and not top_target_supported:
        uncertainties.append(
            "The largest residual target group still has limited support in this slice, so group-specific mechanism claims should be treated as provisional."
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
    observations = _safe_dict(widest.get("key_observations"))
    comparison = _safe_dict(
        widest.get("preserved_vs_collapsed_dual_aligned_comparison")
    )

    return {
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "summary": summary,
        "observed": _safe_list(observations.get("facts")),
        "strongly_suggested": _safe_list(observations.get("inferences")),
        "remains_unproven": _safe_list(observations.get("uncertainties")),
        "overall_conclusion": _overall_conclusion(summary, comparison),
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
            "- collapsed_to_hold_row_count: "
            f"{headline.get('collapsed_to_hold_row_count', 0)}"
        )
        lines.append(
            "- residual_target_row_count: "
            f"{headline.get('residual_target_row_count', 0)}"
        )
        lines.append(
            "- collapsed_dual_neutral_or_missing_row_count: "
            f"{headline.get('collapsed_dual_neutral_or_missing_row_count', 0)}"
        )
        lines.append(
            "- dominant_residual_group: "
            f"{headline.get('dominant_residual_group', 'none')}"
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


def _build_residual_row(row: dict[str, Any]) -> dict[str, Any] | None:
    residual_group = _classify_residual_group(
        comparison_group=str(row.get("comparison_group") or ""),
        activation_pattern=str(row.get("activation_pattern") or ""),
    )
    if residual_group is None:
        return None

    context_relation = _context_relation_to_selected(
        context_state=row.get("context_state"),
        context_bias=row.get("context_bias"),
        selected_signal=str(row.get("selected_strategy_result_signal_state") or ""),
    )
    rule_reason_bucket = _bucket_reason_text(row.get("rule_reason_text"))
    root_reason_bucket = _bucket_reason_text(row.get("root_reason_text"))
    reason_bucket, reason_bucket_source = _combined_reason_bucket(
        rule_reason_bucket=rule_reason_bucket,
        root_reason_bucket=root_reason_bucket,
    )

    return {
        **row,
        "residual_group": residual_group,
        "residual_group_role": _residual_group_role(residual_group),
        "context_relation_to_selected": context_relation,
        "rule_reason_bucket": rule_reason_bucket,
        "root_reason_bucket": root_reason_bucket,
        "reason_bucket": reason_bucket,
        "reason_bucket_source": reason_bucket_source,
    }


def _classify_residual_group(
    *,
    comparison_group: str,
    activation_pattern: str,
) -> str | None:
    if (
        comparison_group == activation_gap_module._GROUP_PRESERVED
        and activation_pattern == "dual_aligned_with_selected"
    ):
        return _RESIDUAL_GROUP_PRESERVED_BASELINE

    if comparison_group != activation_gap_module._GROUP_COLLAPSED:
        return None

    if activation_pattern == "dual_aligned_with_selected":
        return _RESIDUAL_GROUP_COLLAPSED_DUAL_ALIGNED
    if activation_pattern == "setup_only_aligned":
        return _RESIDUAL_GROUP_COLLAPSED_SETUP_ONLY
    if activation_pattern == "trigger_only_aligned":
        return _RESIDUAL_GROUP_COLLAPSED_TRIGGER_ONLY
    if activation_pattern == "any_opposition":
        return _RESIDUAL_GROUP_COLLAPSED_ANY_OPPOSITION
    if activation_pattern == "mixed_or_inconclusive":
        return _RESIDUAL_GROUP_COLLAPSED_MIXED
    if activation_pattern == "dual_neutral_or_missing":
        return _RESIDUAL_GROUP_COLLAPSED_DUAL_NEUTRAL

    return _RESIDUAL_GROUP_COLLAPSED_MIXED


def _context_relation_to_selected(
    *,
    context_state: Any,
    context_bias: Any,
    selected_signal: str,
) -> str:
    state_relation = _state_relation_to_selected(
        state_value=context_state,
        selected_signal=selected_signal,
    )
    bias_relation = _bias_relation(
        context_bias,
        selected_signal,
    )

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

    return "neutral_or_missing"


def _bucket_reason_text(text: Any) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return _REASON_BUCKET_INSUFFICIENT

    if any(marker in normalized for marker in _REASON_RISK_MARKERS):
        return _REASON_BUCKET_RISK
    if any(marker in normalized for marker in _REASON_OPPOSITION_MARKERS):
        return _REASON_BUCKET_OPPOSITION
    if any(marker in normalized for marker in _REASON_CONFLICT_MARKERS):
        return _REASON_BUCKET_CONFLICT

    reason_categories = _classify_reason_text(normalized)
    if (
        any(marker in normalized for marker in _REASON_CONTEXT_MARKERS)
        and (
            any(marker in normalized for marker in _REASON_CONTEXT_NEGATIVE_MARKERS)
            or "neutrality" in reason_categories
        )
    ):
        return _REASON_BUCKET_CONTEXT
    if "directional_opposition" in reason_categories:
        return _REASON_BUCKET_OPPOSITION
    if "conflict" in reason_categories:
        return _REASON_BUCKET_CONFLICT
    if "confirmation_gap" in reason_categories or any(
        marker in normalized for marker in _REASON_CONFIRMATION_MARKERS
    ):
        return _REASON_BUCKET_CONFIRMATION
    if "neutrality" in reason_categories:
        return _REASON_BUCKET_CONTEXT
    return _REASON_BUCKET_OTHER


def _combined_reason_bucket(
    *,
    rule_reason_bucket: str,
    root_reason_bucket: str,
) -> tuple[str, str]:
    if rule_reason_bucket != _REASON_BUCKET_INSUFFICIENT:
        return rule_reason_bucket, "rule_reason_text"
    if root_reason_bucket != _REASON_BUCKET_INSUFFICIENT:
        return root_reason_bucket, "root_reason_text"
    return _REASON_BUCKET_INSUFFICIENT, "no_reason_text"


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    support_threshold: int,
) -> dict[str, Any]:
    row_count = len(rows)
    context_counter = Counter(
        str(row.get("context_relation_to_selected") or "neutral_or_missing")
        for row in rows
    )
    reason_counter = Counter(
        str(row.get("reason_bucket") or _REASON_BUCKET_INSUFFICIENT)
        for row in rows
    )
    rule_reason_counter = Counter(
        str(row.get("rule_reason_bucket") or _REASON_BUCKET_INSUFFICIENT)
        for row in rows
    )
    root_reason_counter = Counter(
        str(row.get("root_reason_bucket") or _REASON_BUCKET_INSUFFICIENT)
        for row in rows
    )
    reason_source_counter = Counter(
        str(row.get("reason_bucket_source") or "no_reason_text")
        for row in rows
    )

    return {
        "row_count": row_count,
        "support_status": (
            "supported"
            if row_count >= max(1, int(support_threshold))
            else "limited_support"
        ),
        "dominant_context_relation": _dominant_value(
            context_counter,
            empty="none",
            order_map=_RELATION_ORDER,
        ),
        "primary_reason_bucket": _primary_reason_bucket(
            reason_counter,
            row_count=row_count,
            support_threshold=max(1, int(support_threshold)),
        ),
        "context_relation_counts": dict(context_counter),
        "reason_bucket_counts": dict(reason_counter),
        "rule_reason_bucket_counts": dict(rule_reason_counter),
        "root_reason_bucket_counts": dict(root_reason_counter),
        "reason_bucket_source_counts": dict(reason_source_counter),
    }


def _primary_reason_bucket(
    counter: Counter[str],
    *,
    row_count: int,
    support_threshold: int,
) -> str:
    if row_count <= 0:
        return _PRIMARY_REASON_BUCKET_NO_ROWS
    if row_count < support_threshold:
        return _PRIMARY_REASON_BUCKET_INSUFFICIENT_SUPPORT

    dominant = _dominant_value(
        counter,
        empty=_PRIMARY_REASON_BUCKET_NO_ROWS,
        order_map=_REASON_BUCKET_ORDER,
    )
    if _safe_ratio(counter.get(dominant, 0), row_count) >= _PRIMARY_FACTOR_THRESHOLD:
        return dominant
    return _PRIMARY_REASON_BUCKET_MIXED


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


def _dominant_residual_group_label(
    residual_group_summaries: Sequence[dict[str, Any]],
) -> str:
    target_rows = [
        row
        for row in residual_group_summaries
        if _safe_dict(row).get("group_role") == "target"
        and int(_safe_dict(row).get("row_count", 0) or 0) > 0
    ]
    if not target_rows:
        return "none"
    target_rows.sort(
        key=lambda item: (
            -int(item.get("row_count", 0) or 0),
            _RESIDUAL_GROUP_ORDER.get(str(item.get("residual_group") or ""), 99),
            str(item.get("residual_group") or ""),
        )
    )
    return str(target_rows[0].get("residual_group") or "none")


def _top_target_residual_group(
    residual_group_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    target_rows = [
        _safe_dict(row)
        for row in residual_group_summaries
        if _safe_dict(row).get("group_role") == "target"
        and int(_safe_dict(row).get("row_count", 0) or 0) > 0
    ]
    if not target_rows:
        return None
    target_rows.sort(
        key=lambda item: (
            -int(item.get("row_count", 0) or 0),
            _RESIDUAL_GROUP_ORDER.get(str(item.get("residual_group") or ""), 99),
            str(item.get("residual_group") or ""),
        )
    )
    return target_rows[0]


def _strongest_strategy_slice(
    strategy_residual_summaries: Sequence[dict[str, Any]],
) -> str | None:
    target_rows = [
        _safe_dict(row)
        for row in strategy_residual_summaries
        if _safe_dict(row).get("group_role") == "target"
        and _row_is_supported(_safe_dict(row))
    ]
    if not target_rows:
        return None
    target_rows.sort(
        key=lambda item: (
            -int(item.get("row_count", 0) or 0),
            str(item.get("strategy") or ""),
            _RESIDUAL_GROUP_ORDER.get(str(item.get("residual_group") or ""), 99),
        )
    )
    row = target_rows[0]
    return (
        "Largest supported strategy residual slice: "
        f"{row.get('strategy', _MISSING_LABEL)} / "
        f"{row.get('residual_group', _MISSING_LABEL)} "
        f"(n={int(row.get('row_count', 0) or 0)})."
    )


def _overall_conclusion(summary: dict[str, Any], comparison: dict[str, Any]) -> str:
    collapsed_rows = int(summary.get("collapsed_to_hold_row_count", 0) or 0)
    residual_target_rows = int(summary.get("residual_target_row_count", 0) or 0)
    reference_rows = int(
        summary.get("collapsed_dual_neutral_or_missing_row_count", 0) or 0
    )
    collapsed_dual_aligned_rows = int(
        comparison.get("collapsed_dual_aligned_row_count", 0) or 0
    )
    comparison_supported = str(comparison.get("support_status") or "") == "supported"

    if collapsed_rows <= 0:
        return (
            "No collapsed-to-hold rows were observed in the evaluated widest configuration, "
            "so the post-confirmation residual diagnosis has no target slice to analyze."
        )
    if residual_target_rows <= 0:
        return (
            "Collapsed hold rows in the widest configuration were entirely explained by the "
            "dual_neutral_or_missing reference subset, so no post-confirmation residual slice "
            "remained for this artifact."
        )
    if residual_target_rows < _MIN_PRIMARY_SUPPORT_ROWS:
        return (
            "A post-confirmation residual subset is observable, but it currently has limited support in the widest configuration, "
            "so residual-mechanism conclusions should be treated as provisional."
        )

    if collapsed_dual_aligned_rows > 0 and comparison_supported:
        return (
            "The widest configuration shows a supported post-confirmation residual subset: collapsed hold rows "
            f"still include {residual_target_rows} non-dual-neutral rows beyond the {reference_rows} "
            "dual_neutral_or_missing reference rows, including supported dual-aligned collapses. "
            "That pattern strongly suggests the remaining hold outcomes are not explained by activation gap alone, "
            "while still leaving the exact higher-level context, risk/filter, or merge mechanism unproven."
        )

    if collapsed_dual_aligned_rows > 0:
        return (
            "The widest configuration shows a real post-confirmation residual subset: collapsed hold rows "
            f"still include {residual_target_rows} non-dual-neutral rows beyond the {reference_rows} "
            "dual_neutral_or_missing reference rows, including dual-aligned collapses. "
            "That pattern suggests the remaining hold outcomes are not explained by activation gap alone, "
            "but the preserved-vs-collapsed dual-aligned comparison still has limited support, so mechanism-specific interpretation should remain provisional."
        )

    return (
        "The widest configuration shows a real post-confirmation residual subset: collapsed hold rows "
        f"still include {residual_target_rows} non-dual-neutral rows beyond the {reference_rows} "
        "dual_neutral_or_missing reference rows. That pattern suggests partial confirmation, opposition, "
        "and/or residual context-filter conservatism still contribute after the main activation gap is established, "
        "but the exact internal rule mechanism remains unproven."
    )


def _residual_group_role(residual_group: str) -> str:
    if residual_group in _BASELINE_RESIDUAL_GROUPS:
        return "baseline"
    if residual_group in _REFERENCE_RESIDUAL_GROUPS:
        return "reference"
    return "target"


def _residual_group_map(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(_safe_dict(row).get("residual_group") or ""): _safe_dict(row)
        for row in rows
    }


def _rate_map(counter: Counter[str], total: int) -> dict[str, float]:
    return {key: _safe_ratio(value, total) for key, value in counter.items()}


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
    if field == "residual_group":
        return _RESIDUAL_GROUP_ORDER.get(text, 99)
    if field == "group_role":
        return _GROUP_ROLE_ORDER.get(text, 99)
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
            "selected_strategy_post_confirmation_hold_residual_diagnosis_report requires "
            "hold-resolution stage-row extraction support, but _build_stage_row is unavailable."
        )
    return builder(raw_record)


def _build_activation_gap_row(row: dict[str, Any]) -> dict[str, Any]:
    builder = getattr(activation_gap_module, "_build_activation_gap_row", None)
    if not callable(builder):
        raise RuntimeError(
            "selected_strategy_post_confirmation_hold_residual_diagnosis_report requires "
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


def _classify_reason_text(normalized_text: str) -> set[str]:
    classifier = getattr(hold_reason_module, "_classify_reason_text", None)
    if not callable(classifier):
        return set()

    result = classifier(normalized_text)
    if isinstance(result, (set, list, tuple)):
        return {str(item) for item in result}
    return set()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


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