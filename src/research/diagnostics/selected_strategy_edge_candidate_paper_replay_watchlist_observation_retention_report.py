from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_watchlist_report as watchlist_report,
)

REPORT_TYPE = (
    "selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report"
)
REPORT_TITLE = (
    "Selected Strategy Edge Candidate Paper Replay Watchlist Observation "
    "Retention Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

RETENTION_CONTRACT_VERSION = "paper_replay_watchlist_observation_retention_v1"
RETENTION_MODE = "aggregate_metric_watchlist_observation_retention"
RETAINED_PROMISING_TIER = "paper_replay_retained_promising_observation"
RETAINED_STANDARD_TIER = "paper_replay_retained_standard_observation"
SINGLE_WINDOW_PROMISING_TIER = "paper_replay_single_window_promising_observation"
SINGLE_WINDOW_STANDARD_TIER = "paper_replay_single_window_standard_observation"

DEFAULT_INPUT_PATH = watchlist_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = watchlist_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = watchlist_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = watchlist_report.SupportWindowConfiguration

SOURCE_REPORT_TYPE = watchlist_report.REPORT_TYPE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only observation retention report from paper replay "
            "watchlist rows. Retention rows are non-live aggregate-only "
            "observations, not trades, orders, fills, prices, or PnL."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Support window in WINDOW_HOURS/MAX_ROWS form. Repeatable.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        configurations=parse_configuration_values(args.config),
        write_report_copies=args.write_latest_copy,
    )
    final = _safe_dict(result["report"].get("final_assessment"))
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "recommended_next_stage": final.get("recommended_next_stage"),
                "retention_rows_present": final.get("retention_rows_present"),
                "retention_row_count": final.get("retention_row_count"),
                "retained_observation_count": final.get(
                    "retained_observation_count"
                ),
                "single_window_observation_count": final.get(
                    "single_window_observation_count"
                ),
                "production_candidate_count": final.get(
                    "production_candidate_count"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return watchlist_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return watchlist_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[SupportWindowConfiguration] | None = None,
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
    configurations: Sequence[SupportWindowConfiguration] | None = None,
) -> dict[str, Any]:
    source_watchlist_report = watchlist_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(source_watchlist_report.get("configuration_summaries"))
        if isinstance(summary, dict)
    ]
    retention_rows = build_retention_rows(
        source_watchlist_configuration_summaries=source_summaries
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_watchlist_report.get("inputs")),
        "diagnostic_only": True,
        "retention_contract": {
            "retention_contract_version": RETENTION_CONTRACT_VERSION,
            "retention_mode": RETENTION_MODE,
            "source_report_type": SOURCE_REPORT_TYPE,
            "source_watchlist_contract_version": (
                watchlist_report.WATCHLIST_CONTRACT_VERSION
            ),
            "retention_rows_are_trades": False,
            "retention_rows_are_orders": False,
            "retention_rows_are_fills": False,
            "retention_rows_enter_live_mapper_or_engine": False,
            "retention_rows_are_live_edge_selection": False,
            "aggregate_metrics_are_exact_future_outcomes": False,
            "synthetic_prices_orders_fills_or_pnl_created": False,
        },
        "source_watchlist_final_assessment": _safe_dict(
            source_watchlist_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_watchlist_report.get("configurations_evaluated")
        ),
        "configuration_summaries": source_summaries,
        "source_watchlist_configuration_summaries": source_summaries,
        "retention_rows": retention_rows,
        "best_retention_row": _first(retention_rows),
        "final_assessment": build_final_assessment(
            retention_rows=retention_rows,
            source_watchlist_configuration_summaries=source_summaries,
        ),
        "assumptions": [
            "Retention rows are report-only records derived from paper replay watchlist rows.",
            "Observation retention is grouped only by symbol, strategy, and horizon.",
            "Support-window-specific source IDs are preserved as lineage but are not used as retention grouping keys.",
            "Aggregate metrics remain aggregate-only evidence, not exact future outcomes, trades, fills, or PnL.",
            "Existing candidate quality gates, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        ],
    }


def build_retention_rows(
    *,
    source_watchlist_configuration_summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for summary in source_watchlist_configuration_summaries:
        configuration = _safe_dict(summary.get("configuration"))
        for row in _safe_list(summary.get("watchlist_rows")):
            if not isinstance(row, dict):
                continue
            source_row = dict(row)
            source_row["_source_configuration"] = configuration
            grouped_rows[observation_key_parts(source_row)].append(source_row)

    retention_rows = [
        build_retention_row(source_watchlist_rows=rows)
        for _, rows in sorted(grouped_rows.items(), key=lambda item: item[0])
    ]
    retention_rows = deduplicate_retention_ids(retention_rows)
    retention_rows.sort(key=_retention_sort_key)
    return retention_rows


def build_retention_row(
    *,
    source_watchlist_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in source_watchlist_rows if isinstance(row, dict)]
    if not rows:
        raise ValueError("source_watchlist_rows must contain at least one row")

    symbol, strategy, horizon = observation_key_parts(rows[0])
    observed_configurations = sorted(
        (
            _clean_configuration(row.get("_source_configuration"))
            for row in rows
            if isinstance(row.get("_source_configuration"), dict)
        ),
        key=_configuration_sort_key,
    )
    observed_configurations = _dedupe_configurations(observed_configurations)
    observed_configuration_count = len(observed_configurations)
    has_promising_source = any(
        row.get("watchlist_tier") == watchlist_report.PROMISING_WATCHLIST_TIER
        for row in rows
    )
    retention_tier, retention_priority = retention_tier_and_priority(
        observed_configuration_count=observed_configuration_count,
        has_promising_source=has_promising_source,
    )

    aggregate_score_values = _sorted_unique_numbers(
        row.get("aggregate_score") for row in rows
    )
    aggregate_sample_count_values = _sorted_unique_ints(
        row.get("aggregate_sample_count") for row in rows
    )
    aggregate_labeled_count_values = _sorted_unique_ints(
        row.get("aggregate_labeled_count") for row in rows
    )
    source_watchlist_tiers = [
        str(row.get("watchlist_tier") or "") for row in rows if row.get("watchlist_tier")
    ]
    source_watchlist_priorities = [
        str(row.get("watchlist_priority") or "")
        for row in rows
        if row.get("watchlist_priority")
    ]

    return {
        "paper_replay_watchlist_observation_retention_id": build_retention_id(
            symbol=symbol,
            strategy=strategy,
            horizon=horizon,
        ),
        "retention_contract_version": RETENTION_CONTRACT_VERSION,
        "retention_mode": RETENTION_MODE,
        "source_report_type": SOURCE_REPORT_TYPE,
        "retention_observation_key": build_retention_observation_key(
            symbol=symbol,
            strategy=strategy,
            horizon=horizon,
        ),
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "observed_configuration_count": observed_configuration_count,
        "observed_configurations": observed_configurations,
        "source_watchlist_row_count": len(rows),
        "source_paper_replay_watchlist_ids": _dedupe_sorted_non_empty(
            row.get("paper_replay_watchlist_id") for row in rows
        ),
        "source_aggregate_metric_evaluation_ids": _dedupe_sorted_non_empty(
            row.get("aggregate_metric_evaluation_id") for row in rows
        ),
        "source_outcome_attachment_source_ids": _dedupe_sorted_non_empty(
            row.get("source_outcome_attachment_source_id") for row in rows
        ),
        "source_outcome_tracking_ids": _dedupe_sorted_non_empty(
            row.get("source_outcome_tracking_id") for row in rows
        ),
        "source_journal_entry_ids": _dedupe_sorted_non_empty(
            row.get("source_journal_entry_id") for row in rows
        ),
        "source_paper_replay_candidate_ids": _dedupe_sorted_non_empty(
            row.get("source_paper_replay_candidate_id") for row in rows
        ),
        "best_watchlist_tier": best_watchlist_tier(source_watchlist_tiers),
        "best_watchlist_priority": best_watchlist_priority(
            source_watchlist_priorities
        ),
        "retained_promising_count": (
            1 if retention_tier == RETAINED_PROMISING_TIER else 0
        ),
        "retained_standard_count": (
            1 if retention_tier == RETAINED_STANDARD_TIER else 0
        ),
        "single_window_promising_count": (
            1 if retention_tier == SINGLE_WINDOW_PROMISING_TIER else 0
        ),
        "single_window_standard_count": (
            1 if retention_tier == SINGLE_WINDOW_STANDARD_TIER else 0
        ),
        "max_aggregate_score": _max_or_none(aggregate_score_values),
        "max_aggregate_sample_count": _max_or_none(aggregate_sample_count_values),
        "max_aggregate_labeled_count": _max_or_none(aggregate_labeled_count_values),
        "aggregate_score_values": aggregate_score_values,
        "aggregate_sample_count_values": aggregate_sample_count_values,
        "aggregate_labeled_count_values": aggregate_labeled_count_values,
        "evaluation_buckets": _dedupe_sorted_non_empty(
            row.get("evaluation_bucket") for row in rows
        ),
        "retention_tier": retention_tier,
        "retention_priority": retention_priority,
        "retention_reason": retention_reason(
            observed_configuration_count=observed_configuration_count,
            has_promising_source=has_promising_source,
        ),
        "production_live_selection_allowed": False,
        "mapper_live_path_allowed": False,
        "engine_live_path_allowed": False,
        "no_order_execution": True,
        "no_synthetic_fill": True,
        "no_pnl_claim": True,
        "order_id": None,
        "fill_id": None,
        "entry_price": None,
        "exit_price": None,
        "realized_pnl": None,
        "unrealized_pnl": None,
        "exact_outcome_used": False,
        "aggregate_metric_only": True,
        "retention_is_live_edge_selection": False,
    }


def observation_key_parts(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _clean_text(row.get("symbol")),
        _clean_text(row.get("strategy")),
        _clean_text(row.get("horizon")),
    )


def build_retention_observation_key(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
) -> str:
    return f"{symbol}:{strategy}:{horizon}"


def build_retention_id(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
) -> str:
    return (
        f"{RETENTION_CONTRACT_VERSION}:"
        f"{_id_part(symbol)}:{_id_part(strategy)}:{_id_part(horizon)}"
    )


def retention_tier_and_priority(
    *,
    observed_configuration_count: int,
    has_promising_source: bool,
) -> tuple[str, str]:
    if observed_configuration_count >= 2 and has_promising_source:
        return RETAINED_PROMISING_TIER, "high"
    if observed_configuration_count >= 2:
        return RETAINED_STANDARD_TIER, "medium"
    if has_promising_source:
        return SINGLE_WINDOW_PROMISING_TIER, "medium"
    return SINGLE_WINDOW_STANDARD_TIER, "low"


def retention_reason(
    *,
    observed_configuration_count: int,
    has_promising_source: bool,
) -> str:
    if observed_configuration_count >= 2 and has_promising_source:
        return "Observation appears in multiple support windows with at least one promising watchlist source row."
    if observed_configuration_count >= 2:
        return "Observation appears in multiple support windows with standard watchlist source rows only."
    if has_promising_source:
        return "Observation appears in a single support window with a promising watchlist source row."
    return "Observation appears in a single support window with a standard watchlist source row."


def best_watchlist_tier(source_watchlist_tiers: Sequence[str]) -> str | None:
    if watchlist_report.PROMISING_WATCHLIST_TIER in source_watchlist_tiers:
        return watchlist_report.PROMISING_WATCHLIST_TIER
    if watchlist_report.STANDARD_WATCHLIST_TIER in source_watchlist_tiers:
        return watchlist_report.STANDARD_WATCHLIST_TIER
    return _first_value(source_watchlist_tiers)


def best_watchlist_priority(source_watchlist_priorities: Sequence[str]) -> str | None:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    priorities = [
        priority
        for priority in source_watchlist_priorities
        if priority in priority_order
    ]
    if not priorities:
        return _first_value(source_watchlist_priorities)
    return sorted(priorities, key=lambda priority: priority_order[priority])[0]


def deduplicate_retention_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = str(
            item.get("paper_replay_watchlist_observation_retention_id") or ""
        )
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["paper_replay_watchlist_observation_retention_id"] = (
                f"{base_id}:dup_{seen[base_id]}"
            )
        deduplicated.append(item)
    return deduplicated


def build_retention_safety_invariant_summary(
    retention_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in retention_rows if isinstance(row, dict)]
    retention_ids = [
        row.get("paper_replay_watchlist_observation_retention_id") for row in rows
    ]
    return {
        "all_rows_production_live_selection_allowed_is_false": all(
            row.get("production_live_selection_allowed") is False for row in rows
        ),
        "all_rows_mapper_live_path_allowed_is_false": all(
            row.get("mapper_live_path_allowed") is False for row in rows
        ),
        "all_rows_engine_live_path_allowed_is_false": all(
            row.get("engine_live_path_allowed") is False for row in rows
        ),
        "all_rows_no_order_execution_is_true": all(
            row.get("no_order_execution") is True for row in rows
        ),
        "all_rows_no_synthetic_fill_is_true": all(
            row.get("no_synthetic_fill") is True for row in rows
        ),
        "all_rows_no_pnl_claim_is_true": all(
            row.get("no_pnl_claim") is True for row in rows
        ),
        "no_order_or_fill_ids_are_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in rows
        ),
        "no_price_or_pnl_fields_are_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in rows
        ),
        "exact_outcomes_are_not_used": all(
            row.get("exact_outcome_used") is False for row in rows
        ),
        "aggregate_metrics_remain_aggregate_only": all(
            row.get("aggregate_metric_only") is True for row in rows
        ),
        "retention_rows_are_not_live_edge_selections": all(
            row.get("retention_is_live_edge_selection") is False for row in rows
        ),
        "all_retention_rows_have_non_empty_observation_keys": all(
            _is_non_empty_value(row.get("retention_observation_key"))
            for row in rows
        ),
        "all_retention_rows_preserve_source_paper_replay_watchlist_ids": all(
            len(_safe_list(row.get("source_paper_replay_watchlist_ids"))) > 0
            for row in rows
        ),
        "all_retention_rows_preserve_source_aggregate_metric_evaluation_ids": all(
            len(_safe_list(row.get("source_aggregate_metric_evaluation_ids"))) > 0
            for row in rows
        ),
        "all_retention_rows_preserve_source_outcome_attachment_source_ids": all(
            len(_safe_list(row.get("source_outcome_attachment_source_ids"))) > 0
            for row in rows
        ),
        "all_retention_rows_preserve_source_outcome_tracking_ids": all(
            len(_safe_list(row.get("source_outcome_tracking_ids"))) > 0
            for row in rows
        ),
        "all_retention_rows_preserve_source_journal_entry_ids": all(
            len(_safe_list(row.get("source_journal_entry_ids"))) > 0
            for row in rows
        ),
        "all_retention_rows_preserve_source_paper_replay_candidate_ids": all(
            len(_safe_list(row.get("source_paper_replay_candidate_ids"))) > 0
            for row in rows
        ),
        "retention_ids_are_unique": len(retention_ids) == len(set(retention_ids)),
    }


def build_final_assessment(
    *,
    retention_rows: Sequence[dict[str, Any]],
    source_watchlist_configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()

    for summary in source_watchlist_configuration_summaries:
        counts["watchlist_row"] += _safe_int(summary.get("watchlist_row_count"))
        counts["promising_watchlist"] += _safe_int(
            summary.get("promising_watchlist_count")
        )
        counts["standard_watchlist"] += _safe_int(
            summary.get("standard_watchlist_count")
        )
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )

    for row in retention_rows:
        counts["retention_row"] += 1
        counts["retained_promising"] += _safe_int(row.get("retained_promising_count"))
        counts["retained_standard"] += _safe_int(row.get("retained_standard_count"))
        counts["single_window_promising"] += _safe_int(
            row.get("single_window_promising_count")
        )
        counts["single_window_standard"] += _safe_int(
            row.get("single_window_standard_count")
        )

    retained_observation_count = (
        counts["retained_promising"] + counts["retained_standard"]
    )
    single_window_observation_count = (
        counts["single_window_promising"] + counts["single_window_standard"]
    )

    if retained_observation_count > 0:
        recommended_next_stage = "design_retained_watchlist_review_packet_report"
    elif counts["retention_row"] > 0:
        recommended_next_stage = "continue_watchlist_observation"
    else:
        recommended_next_stage = "continue_aggregate_observation"

    return {
        "retention_rows_present": counts["retention_row"] > 0,
        "retention_row_count": counts["retention_row"],
        "retained_observation_count": retained_observation_count,
        "single_window_observation_count": single_window_observation_count,
        "retained_promising_count": counts["retained_promising"],
        "retained_standard_count": counts["retained_standard"],
        "single_window_promising_count": counts["single_window_promising"],
        "single_window_standard_count": counts["single_window_standard"],
        "production_candidate_count": counts["production_candidate"],
        "candidate_counts": dict(counts),
        "retention_safety_invariant_summary": (
            build_retention_safety_invariant_summary(retention_rows)
        ),
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
        "recommended_next_stage": recommended_next_stage,
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_retained_watchlist_review_packet_report":
        return "Stop here. The next stage should design a report-only retained watchlist review packet report."
    if recommended_next_stage == "continue_watchlist_observation":
        return "Stop here. Continue observing single-window aggregate-only watchlist rows before retained review design."
    if recommended_next_stage == "continue_aggregate_observation":
        return "Stop here. Continue collecting aggregate-only observations before watchlist retention review."
    return "Stop here. This report does not implement paper trading execution."


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output = resolve_path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output / REPORT_JSON_NAME
    md_path = resolved_output / REPORT_MD_NAME
    _write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown(report: dict[str, Any]) -> str:
    final = _safe_dict(report.get("final_assessment"))
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Final Assessment",
        f"- retention_rows_present: {final.get('retention_rows_present')}",
        f"- retention_row_count: {final.get('retention_row_count')}",
        f"- retained_observation_count: {final.get('retained_observation_count')}",
        f"- single_window_observation_count: {final.get('single_window_observation_count')}",
        f"- retained_promising_count: {final.get('retained_promising_count')}",
        f"- retained_standard_count: {final.get('retained_standard_count')}",
        f"- single_window_promising_count: {final.get('single_window_promising_count')}",
        f"- single_window_standard_count: {final.get('single_window_standard_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- retention_contract_version: {RETENTION_CONTRACT_VERSION}",
        f"- retention_mode: {RETENTION_MODE}",
        f"- source_report_type: {SOURCE_REPORT_TYPE}",
        "- Retention rows are report-only aggregate metric watchlist observations, not trades, orders, fills, or live engine candidates.",
        "- Retention grouping uses symbol, strategy, and horizon rather than support-window-specific source IDs.",
        "- Candidate quality gate, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- retention_safety_invariant_summary: {final.get('retention_safety_invariant_summary')}",
        "",
        "## Retention Rows",
    ]

    for row in _safe_list(report.get("retention_rows")):
        item = _safe_dict(row)
        lines.extend(
            [
                "",
                f"### {item.get('retention_observation_key')}",
                f"- retention_tier: {item.get('retention_tier')}",
                f"- retention_priority: {item.get('retention_priority')}",
                f"- observed_configuration_count: {item.get('observed_configuration_count')}",
                f"- source_watchlist_row_count: {item.get('source_watchlist_row_count')}",
                f"- max_aggregate_score: {item.get('max_aggregate_score')}",
                f"- max_aggregate_sample_count: {item.get('max_aggregate_sample_count')}",
                f"- evaluation_buckets: {item.get('evaluation_buckets')}",
                f"- retention_reason: {item.get('retention_reason')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _retention_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str, str]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return (
        priority_order.get(str(row.get("retention_priority") or ""), 9),
        -float(row.get("max_aggregate_score") or 0.0),
        -int(row.get("max_aggregate_sample_count") or 0),
        str(row.get("retention_observation_key") or ""),
        str(row.get("paper_replay_watchlist_observation_retention_id") or ""),
    )


def _configuration_sort_key(configuration: dict[str, Any]) -> tuple[int, int, str]:
    return (
        _safe_int(configuration.get("window_hours")),
        _safe_int(configuration.get("max_rows")),
        str(configuration.get("display_name") or ""),
    )


def _clean_configuration(configuration: Any) -> dict[str, Any]:
    item = _safe_dict(configuration)
    parsed = _parse_configuration_display_name(item.get("display_name"))
    window_hours = item.get("window_hours")
    max_rows = item.get("max_rows")
    if window_hours is None:
        window_hours = parsed.get("window_hours")
    if max_rows is None:
        max_rows = parsed.get("max_rows")
    return {
        "window_hours": window_hours,
        "max_rows": max_rows,
        "display_name": item.get("display_name"),
    }


def _parse_configuration_display_name(display_name: Any) -> dict[str, int | None]:
    clean = _clean_text(display_name)
    match = re.fullmatch(r"(\d+)\s*h\s*(?:/|_)\s*(\d+)", clean)
    if not match:
        return {"window_hours": None, "max_rows": None}
    return {
        "window_hours": int(match.group(1)),
        "max_rows": int(match.group(2)),
    }


def _dedupe_configurations(
    configurations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    result: list[dict[str, Any]] = []
    for configuration in configurations:
        key = _configuration_sort_key(configuration)
        if key in seen:
            continue
        seen.add(key)
        result.append(configuration)
    return result


def _dedupe_sorted_non_empty(values: Any) -> list[str]:
    return sorted({_clean_text(value) for value in values if _is_non_empty_value(value)})


def _sorted_unique_numbers(values: Any) -> list[float]:
    result: set[float] = set()
    for value in values:
        parsed = _safe_float(value)
        if parsed is None:
            continue
        result.add(parsed)
    return sorted(result)


def _sorted_unique_ints(values: Any) -> list[int]:
    result: set[int] = set()
    for value in values:
        parsed = _safe_optional_int(value)
        if parsed is None:
            continue
        result.add(parsed)
    return sorted(result)


def _max_or_none(values: Sequence[Any]) -> Any:
    if not values:
        return None
    return max(values)


def _first(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    return first if isinstance(first, dict) else None


def _first_value(values: Sequence[str]) -> str | None:
    for value in values:
        clean = _clean_text(value)
        if clean:
            return clean
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _id_part(value: Any) -> str:
    clean = _clean_text(value).lower().replace(" ", "_")
    return clean if clean else "unknown"


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_dict(value: Any) -> dict[str, Any]:
    return watchlist_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return watchlist_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return watchlist_report._safe_int(value)


def _safe_optional_int(value: Any) -> int | None:
    return watchlist_report._safe_optional_int(value)


def _safe_float(value: Any) -> float | None:
    return watchlist_report._safe_float(value)


if __name__ == "__main__":
    main()
