from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report as retention_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_retained_watchlist_review_packet_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Retained Watchlist Review Packet Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

REVIEW_PACKET_CONTRACT_VERSION = "retained_watchlist_review_packet_v1"
REVIEW_PACKET_MODE = "retained_aggregate_metric_watchlist_human_review_packet"
RETAINED_PROMISING_REVIEW_PACKET_TIER = "retained_promising_review_packet"
RETAINED_STANDARD_REVIEW_PACKET_TIER = "retained_standard_review_packet"

DEFAULT_INPUT_PATH = retention_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = retention_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = retention_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = retention_report.SupportWindowConfiguration

RETAINED_SOURCE_TIERS = {
    retention_report.RETAINED_PROMISING_TIER,
    retention_report.RETAINED_STANDARD_TIER,
}
SINGLE_WINDOW_SOURCE_TIERS = {
    retention_report.SINGLE_WINDOW_PROMISING_TIER,
    retention_report.SINGLE_WINDOW_STANDARD_TIER,
}

ALLOWED_NEXT_ACTIONS = [
    "review_evidence_packet",
    "define_manual_review_criteria",
    "continue_non_live_observation",
    "collect_exact_outcome_labels_without_live_execution",
]
FORBIDDEN_NEXT_ACTIONS = [
    "relax_live_candidate_gate",
    "modify_mapper_live_path",
    "modify_engine_live_path",
    "modify_execution_gate",
    "route_to_live_mapper",
    "route_to_live_engine",
    "place_order",
    "create_synthetic_fill",
    "claim_realized_pnl",
    "claim_unrealized_pnl",
    "treat_aggregate_metrics_as_exact_outcomes",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only human review packet from retained paper replay "
            "watchlist observations. Review packets are non-live aggregate-only "
            "evidence packets, not trades, orders, fills, prices, or PnL."
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
    result = run_selected_strategy_edge_candidate_retained_watchlist_review_packet_report(
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
                "review_packets_present": final.get("review_packets_present"),
                "review_packet_count": final.get("review_packet_count"),
                "retained_source_row_count": final.get("retained_source_row_count"),
                "excluded_single_window_observation_count": final.get(
                    "excluded_single_window_observation_count"
                ),
                "production_candidate_count": final.get("production_candidate_count"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return retention_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return retention_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_retained_watchlist_review_packet_report(
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
    source_retention_report = retention_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_retention_rows = [
        row
        for row in _safe_list(source_retention_report.get("retention_rows"))
        if isinstance(row, dict)
    ]
    review_packet_rows = build_review_packet_rows(
        source_retention_rows=source_retention_rows
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_retention_report.get("inputs")),
        "diagnostic_only": True,
        "review_packet_contract": {
            "review_packet_contract_version": REVIEW_PACKET_CONTRACT_VERSION,
            "review_packet_mode": REVIEW_PACKET_MODE,
            "source_retention_report_type": retention_report.REPORT_TYPE,
            "source_retention_contract_version": (
                retention_report.RETENTION_CONTRACT_VERSION
            ),
            "review_packet_rows_are_trades": False,
            "review_packet_rows_are_orders": False,
            "review_packet_rows_are_fills": False,
            "review_packet_rows_enter_live_mapper_or_engine": False,
            "review_packet_rows_are_live_edge_selection": False,
            "aggregate_metrics_are_exact_future_outcomes": False,
            "synthetic_prices_orders_fills_or_pnl_created": False,
            "live_change_allowed_by_this_report": False,
        },
        "source_retention_final_assessment": _safe_dict(
            source_retention_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_retention_report.get("configurations_evaluated")
        ),
        "source_retention_rows": source_retention_rows,
        "review_packet_rows": review_packet_rows,
        "best_review_packet_row": _first(review_packet_rows),
        "final_assessment": build_final_assessment(
            review_packet_rows=review_packet_rows,
            source_retention_rows=source_retention_rows,
            source_retention_final_assessment=_safe_dict(
                source_retention_report.get("final_assessment")
            ),
        ),
        "assumptions": [
            "Review packet rows are report-only records derived from retained paper replay watchlist observations.",
            "Single-window observations are excluded from review packet rows and remain observation-only evidence.",
            "Aggregate metrics remain aggregate-only evidence, not exact future outcomes, trades, fills, or PnL.",
            "A human review is required before any separate live-path change proposal.",
            "This report does not relax live candidate gates, route rows to the mapper or engine, place orders, create fills, or claim PnL.",
        ],
    }


def build_review_packet_rows(
    *,
    source_retention_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    retained_rows = [
        row
        for row in source_retention_rows
        if _clean_text(row.get("retention_tier")) in RETAINED_SOURCE_TIERS
    ]
    review_packet_rows = [
        build_review_packet_row(source_retention_row=row) for row in retained_rows
    ]
    review_packet_rows = deduplicate_review_packet_ids(review_packet_rows)
    review_packet_rows.sort(key=_review_packet_sort_key)
    return review_packet_rows


def build_review_packet_row(
    *,
    source_retention_row: dict[str, Any],
) -> dict[str, Any]:
    source_row = dict(source_retention_row)
    source_retention_tier = _clean_text(source_row.get("retention_tier"))
    review_packet_tier, review_packet_priority = review_packet_tier_and_priority(
        source_retention_tier=source_retention_tier
    )
    source_retention_id = _clean_text(
        source_row.get("paper_replay_watchlist_observation_retention_id")
    )

    return {
        "retained_watchlist_review_packet_id": build_review_packet_id(
            source_retention_id=source_retention_id
        ),
        "review_packet_contract_version": REVIEW_PACKET_CONTRACT_VERSION,
        "review_packet_mode": REVIEW_PACKET_MODE,
        "source_report_type": source_row.get("source_report_type"),
        "source_retention_report_type": retention_report.REPORT_TYPE,
        "source_retention_id": source_retention_id,
        "retention_observation_key": source_row.get("retention_observation_key"),
        "symbol": source_row.get("symbol"),
        "strategy": source_row.get("strategy"),
        "horizon": source_row.get("horizon"),
        "source_retention_tier": source_retention_tier,
        "source_retention_priority": source_row.get("retention_priority"),
        "observed_configuration_count": source_row.get(
            "observed_configuration_count"
        ),
        "observed_configurations": _safe_list(
            source_row.get("observed_configurations")
        ),
        "source_watchlist_row_count": source_row.get("source_watchlist_row_count"),
        "source_paper_replay_watchlist_ids": _safe_list(
            source_row.get("source_paper_replay_watchlist_ids")
        ),
        "source_aggregate_metric_evaluation_ids": _safe_list(
            source_row.get("source_aggregate_metric_evaluation_ids")
        ),
        "source_outcome_attachment_source_ids": _safe_list(
            source_row.get("source_outcome_attachment_source_ids")
        ),
        "source_outcome_tracking_ids": _safe_list(
            source_row.get("source_outcome_tracking_ids")
        ),
        "source_journal_entry_ids": _safe_list(
            source_row.get("source_journal_entry_ids")
        ),
        "source_paper_replay_candidate_ids": _safe_list(
            source_row.get("source_paper_replay_candidate_ids")
        ),
        "max_aggregate_score": source_row.get("max_aggregate_score"),
        "max_aggregate_sample_count": source_row.get("max_aggregate_sample_count"),
        "max_aggregate_labeled_count": source_row.get("max_aggregate_labeled_count"),
        "aggregate_score_values": _safe_list(source_row.get("aggregate_score_values")),
        "aggregate_sample_count_values": _safe_list(
            source_row.get("aggregate_sample_count_values")
        ),
        "aggregate_labeled_count_values": _safe_list(
            source_row.get("aggregate_labeled_count_values")
        ),
        "evaluation_buckets": _safe_list(source_row.get("evaluation_buckets")),
        "best_watchlist_tier": source_row.get("best_watchlist_tier"),
        "best_watchlist_priority": source_row.get("best_watchlist_priority"),
        "review_packet_tier": review_packet_tier,
        "review_packet_priority": review_packet_priority,
        "review_status": "pending_human_review",
        "review_required_before_live_change": True,
        "review_decision_status": "not_reviewed",
        "reviewer": None,
        "reviewed_at": None,
        "approved_for_live_change": False,
        "live_change_requires_separate_pr": True,
        "live_change_allowed_by_this_report": False,
        "review_reason": review_reason(source_retention_tier),
        "allowed_next_actions": list(ALLOWED_NEXT_ACTIONS),
        "forbidden_next_actions": list(FORBIDDEN_NEXT_ACTIONS),
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
        "review_packet_is_live_edge_selection": False,
    }


def review_packet_tier_and_priority(
    *,
    source_retention_tier: str,
) -> tuple[str, str]:
    if source_retention_tier == retention_report.RETAINED_PROMISING_TIER:
        return RETAINED_PROMISING_REVIEW_PACKET_TIER, "high"
    if source_retention_tier == retention_report.RETAINED_STANDARD_TIER:
        return RETAINED_STANDARD_REVIEW_PACKET_TIER, "medium"
    raise ValueError(f"Unsupported retained source tier: {source_retention_tier}")


def review_reason(source_retention_tier: str) -> str:
    if source_retention_tier == retention_report.RETAINED_PROMISING_TIER:
        return "Retained promising aggregate-only watchlist observation requires human evidence review before any separate live-path proposal."
    if source_retention_tier == retention_report.RETAINED_STANDARD_TIER:
        return "Retained standard aggregate-only watchlist observation requires human evidence review before any separate live-path proposal."
    return "Unsupported retained source tier."


def build_review_packet_id(*, source_retention_id: str) -> str:
    return f"{REVIEW_PACKET_CONTRACT_VERSION}:{source_retention_id}"


def deduplicate_review_packet_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = _clean_text(item.get("retained_watchlist_review_packet_id"))
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["retained_watchlist_review_packet_id"] = (
                f"{base_id}:dup_{seen[base_id]}"
            )
        deduplicated.append(item)
    return deduplicated


def build_review_packet_safety_invariant_summary(
    review_packet_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in review_packet_rows if isinstance(row, dict)]
    review_packet_ids = [
        row.get("retained_watchlist_review_packet_id") for row in rows
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
        "review_packets_are_not_live_edge_selections": all(
            row.get("review_packet_is_live_edge_selection") is False
            for row in rows
        ),
        "review_is_required_before_live_change": all(
            row.get("review_required_before_live_change") is True for row in rows
        ),
        "live_change_is_not_allowed_by_this_report": all(
            row.get("live_change_allowed_by_this_report") is False
            and row.get("approved_for_live_change") is False
            for row in rows
        ),
        "all_packets_have_non_empty_source_retention_id": all(
            _is_non_empty_value(row.get("source_retention_id")) for row in rows
        ),
        "all_packets_have_non_empty_retention_observation_key": all(
            _is_non_empty_value(row.get("retention_observation_key"))
            for row in rows
        ),
        "all_packets_preserve_at_least_one_source_paper_replay_watchlist_id": all(
            len(_safe_list(row.get("source_paper_replay_watchlist_ids"))) > 0
            for row in rows
        ),
        "all_packets_preserve_source_aggregate_metric_evaluation_ids": all(
            len(_safe_list(row.get("source_aggregate_metric_evaluation_ids"))) > 0
            for row in rows
        ),
        "all_packets_preserve_source_outcome_attachment_source_ids": all(
            len(_safe_list(row.get("source_outcome_attachment_source_ids"))) > 0
            for row in rows
        ),
        "all_packets_preserve_source_outcome_tracking_ids": all(
            len(_safe_list(row.get("source_outcome_tracking_ids"))) > 0
            for row in rows
        ),
        "all_packets_preserve_source_journal_entry_ids": all(
            len(_safe_list(row.get("source_journal_entry_ids"))) > 0
            for row in rows
        ),
        "all_packets_preserve_source_paper_replay_candidate_ids": all(
            len(_safe_list(row.get("source_paper_replay_candidate_ids"))) > 0
            for row in rows
        ),
        "review_packet_ids_are_unique": len(review_packet_ids)
        == len(set(review_packet_ids)),
    }


def build_final_assessment(
    *,
    review_packet_rows: Sequence[dict[str, Any]],
    source_retention_rows: Sequence[dict[str, Any]],
    source_retention_final_assessment: dict[str, Any],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    source_candidate_counts = _safe_dict(
        source_retention_final_assessment.get("candidate_counts")
    )
    for key, value in source_candidate_counts.items():
        counts[str(key)] += _safe_int(value)

    for row in source_retention_rows:
        tier = _clean_text(row.get("retention_tier"))
        if tier in RETAINED_SOURCE_TIERS:
            counts["retained_source_row"] += 1
        if tier in SINGLE_WINDOW_SOURCE_TIERS:
            counts["excluded_single_window_observation"] += 1

    for row in review_packet_rows:
        counts["review_packet"] += 1
        if (
            row.get("review_packet_tier")
            == RETAINED_PROMISING_REVIEW_PACKET_TIER
        ):
            counts["retained_promising_review_packet"] += 1
        if row.get("review_packet_tier") == RETAINED_STANDARD_REVIEW_PACKET_TIER:
            counts["retained_standard_review_packet"] += 1

    production_candidate_count = _safe_int(
        source_retention_final_assessment.get("production_candidate_count")
    )
    if production_candidate_count == 0:
        production_candidate_count = counts["production_candidate"]

    if counts["review_packet"] > 0:
        recommended_next_stage = "human_review_retained_watchlist_packet"
    elif counts["excluded_single_window_observation"] > 0:
        recommended_next_stage = "continue_watchlist_retention_observation"
    else:
        recommended_next_stage = "continue_aggregate_observation"

    return {
        "review_packets_present": counts["review_packet"] > 0,
        "review_packet_count": counts["review_packet"],
        "retained_source_row_count": counts["retained_source_row"],
        "retained_promising_review_packet_count": counts[
            "retained_promising_review_packet"
        ],
        "retained_standard_review_packet_count": counts[
            "retained_standard_review_packet"
        ],
        "excluded_single_window_observation_count": counts[
            "excluded_single_window_observation"
        ],
        "production_candidate_count": production_candidate_count,
        "candidate_counts": dict(counts),
        "review_packet_safety_invariant_summary": (
            build_review_packet_safety_invariant_summary(review_packet_rows)
        ),
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
        "recommended_next_stage": recommended_next_stage,
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "human_review_retained_watchlist_packet":
        return "Stop here. Human review is required before any separate live-path change proposal."
    if recommended_next_stage == "continue_watchlist_retention_observation":
        return "Stop here. Continue observing single-window aggregate-only watchlist rows before review packet creation."
    if recommended_next_stage == "continue_aggregate_observation":
        return "Stop here. Continue collecting aggregate-only observations before retained review packet creation."
    return "Stop here. This report does not implement paper trading or live execution."


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
        f"- review_packets_present: {final.get('review_packets_present')}",
        f"- review_packet_count: {final.get('review_packet_count')}",
        f"- retained_source_row_count: {final.get('retained_source_row_count')}",
        f"- retained_promising_review_packet_count: {final.get('retained_promising_review_packet_count')}",
        f"- retained_standard_review_packet_count: {final.get('retained_standard_review_packet_count')}",
        f"- excluded_single_window_observation_count: {final.get('excluded_single_window_observation_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- review_packet_contract_version: {REVIEW_PACKET_CONTRACT_VERSION}",
        f"- review_packet_mode: {REVIEW_PACKET_MODE}",
        f"- source_retention_report_type: {retention_report.REPORT_TYPE}",
        "- Review packet rows are human-review evidence packets, not trades, orders, fills, or live engine candidates.",
        "- Aggregate metrics remain aggregate-only and are not treated as exact outcomes.",
        "- Candidate quality gate, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- review_packet_safety_invariant_summary: {final.get('review_packet_safety_invariant_summary')}",
        "",
        "## Review Packet Rows",
    ]

    for row in _safe_list(report.get("review_packet_rows")):
        item = _safe_dict(row)
        lines.extend(
            [
                "",
                f"### {item.get('retention_observation_key')}",
                f"- review_packet_tier: {item.get('review_packet_tier')}",
                f"- review_packet_priority: {item.get('review_packet_priority')}",
                f"- source_retention_tier: {item.get('source_retention_tier')}",
                f"- source_retention_priority: {item.get('source_retention_priority')}",
                f"- observed_configuration_count: {item.get('observed_configuration_count')}",
                f"- source_watchlist_row_count: {item.get('source_watchlist_row_count')}",
                f"- max_aggregate_score: {item.get('max_aggregate_score')}",
                f"- max_aggregate_sample_count: {item.get('max_aggregate_sample_count')}",
                f"- review_status: {item.get('review_status')}",
                f"- review_reason: {item.get('review_reason')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _review_packet_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str, str]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return (
        priority_order.get(str(row.get("review_packet_priority") or ""), 9),
        -float(row.get("max_aggregate_score") or 0.0),
        -int(row.get("max_aggregate_sample_count") or 0),
        str(row.get("retention_observation_key") or ""),
        str(row.get("retained_watchlist_review_packet_id") or ""),
    )


def _first(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    return first if isinstance(first, dict) else None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
    return retention_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return retention_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return retention_report._safe_int(value)


if __name__ == "__main__":
    main()
