from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report as aggregate_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_paper_replay_watchlist_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Paper Replay Watchlist Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

WATCHLIST_CONTRACT_VERSION = "paper_replay_watchlist_v1"
WATCHLIST_MODE = "aggregate_metric_observation_watchlist"
PROMISING_WATCHLIST_TIER = "paper_replay_promising_watchlist"
STANDARD_WATCHLIST_TIER = "paper_replay_standard_watchlist"

DEFAULT_INPUT_PATH = aggregate_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = aggregate_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = aggregate_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = aggregate_report.SupportWindowConfiguration

WATCHLIST_SOURCE_BUCKETS = {
    aggregate_report.PROMISING_BUCKET,
    aggregate_report.WATCHLIST_BUCKET,
}

REQUIRED_SOURCE_ID_FIELDS = (
    "source_outcome_attachment_source_id",
    "source_outcome_tracking_id",
    "source_journal_entry_id",
    "source_paper_replay_candidate_id",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only paper replay watchlist from aggregate metric "
            "paper replay evaluation rows. Watchlist rows are non-live "
            "aggregate-only observations, not trades, orders, fills, or PnL."
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
    result = run_selected_strategy_edge_candidate_paper_replay_watchlist_report(
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
                "watchlist_rows_present": final.get("watchlist_rows_present"),
                "watchlist_row_count": final.get("watchlist_row_count"),
                "promising_watchlist_count": final.get(
                    "promising_watchlist_count"
                ),
                "standard_watchlist_count": final.get("standard_watchlist_count"),
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
    return aggregate_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return aggregate_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_replay_watchlist_report(
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
    source_aggregate_report = aggregate_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(
            source_aggregate_report.get("configuration_summaries")
        )
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_aggregate_evaluation_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_aggregate_report.get("inputs")),
        "diagnostic_only": True,
        "watchlist_contract": {
            "watchlist_contract_version": WATCHLIST_CONTRACT_VERSION,
            "watchlist_mode": WATCHLIST_MODE,
            "source_report_type": aggregate_report.REPORT_TYPE,
            "source_aggregate_metric_paper_replay_evaluation_contract_version": (
                aggregate_report.EVALUATION_CONTRACT_VERSION
            ),
            "watchlist_rows_are_trades": False,
            "watchlist_rows_are_orders": False,
            "watchlist_rows_are_fills": False,
            "watchlist_rows_enter_live_mapper_or_engine": False,
            "watchlist_rows_are_live_edge_selection": False,
            "aggregate_metrics_are_exact_future_outcomes": False,
            "synthetic_prices_orders_fills_or_pnl_created": False,
        },
        "source_aggregate_metric_evaluation_final_assessment": _safe_dict(
            source_aggregate_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_aggregate_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Watchlist rows are report-only records derived from aggregate metric evaluation rows.",
            "Only promising and watchlist aggregate metric observations become paper replay watchlist rows.",
            "Weak and unavailable aggregate metric observations are excluded from watchlist rows.",
            "Malformed watchlist-eligible source rows without aggregate_metric_evaluation_id are not converted into watchlist rows.",
            "Aggregate metrics remain aggregate-only evidence, not exact future outcomes, trades, fills, or PnL.",
            "Existing candidate quality gates, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_aggregate_evaluation_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_aggregate_evaluation_summary)
    configuration = _safe_dict(source_summary.get("configuration"))

    aggregate_evaluation_rows = [
        row
        for row in _safe_list(source_summary.get("aggregate_evaluation_rows"))
        if isinstance(row, dict)
    ]
    watchlist_source_rows = [
        row for row in aggregate_evaluation_rows if is_watchlist_source_row(row)
    ]
    malformed_watchlist_source_rows = [
        row
        for row in watchlist_source_rows
        if not _has_non_empty_value(row, "aggregate_metric_evaluation_id")
    ]
    valid_watchlist_source_rows = [
        row
        for row in watchlist_source_rows
        if _has_non_empty_value(row, "aggregate_metric_evaluation_id")
    ]

    watchlist_rows = [
        build_watchlist_row(aggregate_evaluation_row=row)
        for row in valid_watchlist_source_rows
    ]
    watchlist_rows = deduplicate_watchlist_ids(watchlist_rows)
    watchlist_rows.sort(key=_watchlist_sort_key)

    tier_counts = Counter(
        str(row.get("watchlist_tier") or "") for row in watchlist_rows
    )
    summary = {
        "configuration": configuration,
        "source_aggregate_evaluation_summary": source_summary,
        "aggregate_evaluation_row_count": _safe_int(
            source_summary.get("aggregate_evaluation_row_count")
        ),
        "watchlist_source_row_count": len(watchlist_source_rows),
        "malformed_watchlist_source_row_count": len(
            malformed_watchlist_source_rows
        ),
        "watchlist_row_count": len(watchlist_rows),
        "promising_watchlist_count": tier_counts[PROMISING_WATCHLIST_TIER],
        "standard_watchlist_count": tier_counts[STANDARD_WATCHLIST_TIER],
        "paper_replay_candidate_count": _safe_int(
            source_summary.get("paper_replay_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("production_candidate_count")
        ),
        "watchlist_rows": watchlist_rows,
        "best_watchlist_row": _first(watchlist_rows),
        "watchlist_tier_summary": {
            "watchlist_source_row_count": len(watchlist_source_rows),
            "malformed_watchlist_source_row_count": len(
                malformed_watchlist_source_rows
            ),
            "watchlist_row_count": len(watchlist_rows),
            "promising_watchlist_count": tier_counts[PROMISING_WATCHLIST_TIER],
            "standard_watchlist_count": tier_counts[STANDARD_WATCHLIST_TIER],
            "tier_counts": dict(tier_counts),
        },
    }
    summary["watchlist_safety_invariants"] = build_watchlist_safety_invariants(
        summary
    )
    return summary


def is_watchlist_source_row(row: dict[str, Any]) -> bool:
    return row.get("evaluation_bucket") in WATCHLIST_SOURCE_BUCKETS


def build_watchlist_row(
    *,
    aggregate_evaluation_row: dict[str, Any],
) -> dict[str, Any]:
    aggregate_metric_evaluation_id = _require_non_empty_string(
        aggregate_evaluation_row.get("aggregate_metric_evaluation_id"),
        field_name="aggregate_metric_evaluation_id",
    )
    watchlist_tier, watchlist_priority = watchlist_tier_and_priority(
        aggregate_evaluation_row.get("evaluation_bucket")
    )

    safety_notes = _normalize_string_list(
        aggregate_evaluation_row.get("safety_notes")
    )
    safety_notes.extend(
        [
            "Paper replay watchlist row is report-only and is not a trade.",
            "Aggregate metrics remain aggregate-only evidence and are not exact outcomes, trade results, fill results, or PnL.",
            "No order, fill, synthetic price, realized PnL, or unrealized PnL is permitted.",
            "The row must not be fed into the live mapper, engine, or production selection path.",
        ]
    )

    source_outcome_attachment_source_id = _first_non_empty_value(
        aggregate_evaluation_row,
        "source_outcome_attachment_source_id",
        "outcome_attachment_source_id",
        "source_attachment_source_id",
        "attachment_source_id",
    )

    return {
        "paper_replay_watchlist_id": build_watchlist_id(
            aggregate_metric_evaluation_id=aggregate_metric_evaluation_id
        ),
        "watchlist_contract_version": WATCHLIST_CONTRACT_VERSION,
        "watchlist_mode": WATCHLIST_MODE,
        "source_report_type": aggregate_report.REPORT_TYPE,
        "aggregate_metric_evaluation_id": aggregate_metric_evaluation_id,
        "source_outcome_attachment_source_id": (
            source_outcome_attachment_source_id
        ),
        "source_outcome_tracking_id": aggregate_evaluation_row.get(
            "source_outcome_tracking_id"
        ),
        "source_journal_entry_id": aggregate_evaluation_row.get(
            "source_journal_entry_id"
        ),
        "source_paper_replay_candidate_id": aggregate_evaluation_row.get(
            "source_paper_replay_candidate_id"
        ),
        "symbol": aggregate_evaluation_row.get("symbol"),
        "strategy": aggregate_evaluation_row.get("strategy"),
        "horizon": aggregate_evaluation_row.get("horizon"),
        "aggregate_sample_count": _safe_optional_int(
            aggregate_evaluation_row.get("aggregate_sample_count")
        ),
        "aggregate_labeled_count": _safe_optional_int(
            aggregate_evaluation_row.get("aggregate_labeled_count")
        ),
        "aggregate_median_future_return_pct": _safe_float(
            aggregate_evaluation_row.get("aggregate_median_future_return_pct")
        ),
        "aggregate_positive_rate_pct": _safe_float(
            aggregate_evaluation_row.get("aggregate_positive_rate_pct")
        ),
        "aggregate_robustness_signal": aggregate_evaluation_row.get(
            "aggregate_robustness_signal"
        ),
        "aggregate_robustness_signal_pct": _safe_float(
            aggregate_evaluation_row.get("aggregate_robustness_signal_pct")
        ),
        "aggregate_score": _safe_float(
            aggregate_evaluation_row.get("aggregate_score")
        ),
        "evaluation_bucket": aggregate_evaluation_row.get("evaluation_bucket"),
        "evaluation_reason": aggregate_evaluation_row.get("evaluation_reason"),
        "watchlist_tier": watchlist_tier,
        "watchlist_priority": watchlist_priority,
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
        "safety_notes": _dedupe_preserve_order(safety_notes),
    }


def watchlist_tier_and_priority(evaluation_bucket: Any) -> tuple[str, str]:
    if evaluation_bucket == aggregate_report.PROMISING_BUCKET:
        return PROMISING_WATCHLIST_TIER, "high"
    return STANDARD_WATCHLIST_TIER, "medium"


def build_watchlist_id(*, aggregate_metric_evaluation_id: str) -> str:
    clean_id = _require_non_empty_string(
        aggregate_metric_evaluation_id,
        field_name="aggregate_metric_evaluation_id",
    )
    return f"{WATCHLIST_CONTRACT_VERSION}:{clean_id}"


def deduplicate_watchlist_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = str(item.get("paper_replay_watchlist_id") or "")
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["paper_replay_watchlist_id"] = f"{base_id}:dup_{seen[base_id]}"
        deduplicated.append(item)
    return deduplicated


def build_watchlist_safety_invariants(summary: dict[str, Any]) -> dict[str, Any]:
    watchlist_rows = [
        row
        for row in _safe_list(summary.get("watchlist_rows"))
        if isinstance(row, dict)
    ]
    watchlist_ids = [row.get("paper_replay_watchlist_id") for row in watchlist_rows]

    return {
        "all_watchlist_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False
            for row in watchlist_rows
        ),
        "all_watchlist_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in watchlist_rows
        ),
        "all_watchlist_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in watchlist_rows
        ),
        "all_watchlist_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in watchlist_rows
        ),
        "all_watchlist_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in watchlist_rows
        ),
        "all_watchlist_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in watchlist_rows
        ),
        "no_order_or_fill_identifiers_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in watchlist_rows
        ),
        "no_price_or_pnl_fields_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in watchlist_rows
        ),
        "exact_outcomes_not_used": all(
            row.get("exact_outcome_used") is False for row in watchlist_rows
        ),
        "aggregate_metrics_remain_aggregate_only": all(
            row.get("aggregate_metric_only") is True for row in watchlist_rows
        ),
        "watchlist_rows_sourced_only_from_promising_or_watchlist_aggregate_observations": all(
            row.get("evaluation_bucket") in WATCHLIST_SOURCE_BUCKETS
            for row in watchlist_rows
        ),
        "all_watchlist_rows_have_non_empty_aggregate_metric_evaluation_id": all(
            _has_non_empty_value(row, "aggregate_metric_evaluation_id")
            for row in watchlist_rows
        ),
        "all_watchlist_rows_have_required_source_ids": all(
            all(_has_non_empty_value(row, field) for field in REQUIRED_SOURCE_ID_FIELDS)
            for row in watchlist_rows
        ),
        "watchlist_ids_are_unique": len(watchlist_ids) == len(set(watchlist_ids)),
        "watchlist_row_count": len(watchlist_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invariant_summaries = []
    for summary in configuration_summaries:
        counts["watchlist_source_row"] += _safe_int(
            summary.get("watchlist_source_row_count")
        )
        counts["malformed_watchlist_source_row"] += _safe_int(
            summary.get("malformed_watchlist_source_row_count")
        )
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
        counts["aggregate_evaluation_row"] += _safe_int(
            summary.get("aggregate_evaluation_row_count")
        )
        invariant_summaries.append(
            _safe_dict(summary.get("watchlist_safety_invariants"))
        )

    recommended_next_stage = (
        "design_watchlist_observation_retention_report"
        if counts["watchlist_row"] > 0
        else "continue_aggregate_observation"
    )
    invariant_keys = [
        "all_watchlist_rows_production_live_selection_disallowed",
        "all_watchlist_rows_mapper_live_path_disallowed",
        "all_watchlist_rows_engine_live_path_disallowed",
        "all_watchlist_rows_no_order_execution",
        "all_watchlist_rows_no_synthetic_fill",
        "all_watchlist_rows_no_pnl_claim",
        "no_order_or_fill_identifiers_present",
        "no_price_or_pnl_fields_present",
        "exact_outcomes_not_used",
        "aggregate_metrics_remain_aggregate_only",
        "watchlist_rows_sourced_only_from_promising_or_watchlist_aggregate_observations",
        "all_watchlist_rows_have_non_empty_aggregate_metric_evaluation_id",
        "all_watchlist_rows_have_required_source_ids",
        "watchlist_ids_are_unique",
    ]

    return {
        "watchlist_rows_present": counts["watchlist_row"] > 0,
        "watchlist_row_count": counts["watchlist_row"],
        "promising_watchlist_count": counts["promising_watchlist"],
        "standard_watchlist_count": counts["standard_watchlist"],
        "production_candidate_count": counts["production_candidate"],
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "watchlist_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in invariant_summaries)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_watchlist_observation_retention_report":
        return "Stop here. The next stage should design a report-only watchlist observation retention report."
    if recommended_next_stage == "continue_aggregate_observation":
        return "Stop here. Continue collecting aggregate-only observations before watchlist retention design."
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
        f"- watchlist_rows_present: {final.get('watchlist_rows_present')}",
        f"- watchlist_row_count: {final.get('watchlist_row_count')}",
        f"- promising_watchlist_count: {final.get('promising_watchlist_count')}",
        f"- standard_watchlist_count: {final.get('standard_watchlist_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- watchlist_contract_version: {WATCHLIST_CONTRACT_VERSION}",
        f"- watchlist_mode: {WATCHLIST_MODE}",
        f"- source_report_type: {aggregate_report.REPORT_TYPE}",
        "- Watchlist rows are report-only aggregate metric observations, not trades, orders, fills, or live engine candidates.",
        "- Only promising and watchlist aggregate metric observations become watchlist rows.",
        "- Weak and unavailable aggregate metric observations are excluded from watchlist rows.",
        "- Candidate quality gate, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- watchlist_safety_invariant_summary: {final.get('watchlist_safety_invariant_summary')}",
        "",
        "## Configuration Summaries",
    ]

    for summary in _safe_list(report.get("configuration_summaries")):
        item = _safe_dict(summary)
        configuration = _safe_dict(item.get("configuration"))
        lines.extend(
            [
                "",
                f"### {configuration.get('display_name')}",
                f"- watchlist_source_row_count: {item.get('watchlist_source_row_count')}",
                f"- malformed_watchlist_source_row_count: {item.get('malformed_watchlist_source_row_count')}",
                f"- watchlist_row_count: {item.get('watchlist_row_count')}",
                f"- promising_watchlist_count: {item.get('promising_watchlist_count')}",
                f"- standard_watchlist_count: {item.get('standard_watchlist_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- watchlist_tier_summary: {item.get('watchlist_tier_summary')}",
                f"- watchlist_safety_invariants: {item.get('watchlist_safety_invariants')}",
                f"- best_watchlist_row: {item.get('best_watchlist_row')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _watchlist_sort_key(
    row: dict[str, Any],
) -> tuple[int, float, int, str, str, str, str]:
    priority_order = {"high": 0, "medium": 1}
    return (
        priority_order.get(str(row.get("watchlist_priority") or ""), 9),
        -float(row.get("aggregate_score") or 0.0),
        -int(row.get("aggregate_sample_count") or 0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("paper_replay_watchlist_id") or ""),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _first(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    return first if isinstance(first, dict) else None


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            clean = str(item).strip()
            if clean:
                result.append(clean)
        return result
    clean = str(value).strip()
    return [clean] if clean else []


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _first_non_empty_value(row: dict[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        value = row.get(field_name)
        if _is_non_empty_value(value):
            return value
    return None


def _has_non_empty_value(row: dict[str, Any], field_name: str) -> bool:
    return _is_non_empty_value(row.get(field_name))


def _is_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{field_name} must be non-empty")
    return clean


def _safe_dict(value: Any) -> dict[str, Any]:
    return aggregate_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return aggregate_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return aggregate_report._safe_int(value)


def _safe_optional_int(value: Any) -> int | None:
    return aggregate_report._safe_optional_int(value)


def _safe_float(value: Any) -> float | None:
    return aggregate_report._safe_float(value)


if __name__ == "__main__":
    main()
