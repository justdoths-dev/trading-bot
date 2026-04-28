from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_retained_watchlist_review_packet_report as review_packet_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_retained_watchlist_review_checklist_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Retained Watchlist Review Checklist Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

REVIEW_CHECKLIST_CONTRACT_VERSION = "retained_watchlist_review_checklist_v1"
REVIEW_CHECKLIST_MODE = "non_live_human_review_readiness_checklist"
EXACT_OUTCOME_LABEL_COLLECTION_READY_TIER = "exact_outcome_label_collection_ready"
REVIEW_FURTHER_TIER = "review_further_before_exact_outcome_label_collection"
REVIEW_CHECKLIST_BLOCKED_TIER = "review_checklist_blocked"

DEFAULT_INPUT_PATH = review_packet_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = review_packet_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = review_packet_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = review_packet_report.SupportWindowConfiguration

SUPPORTED_REVIEW_PACKET_TIERS = {
    review_packet_report.RETAINED_PROMISING_REVIEW_PACKET_TIER,
    review_packet_report.RETAINED_STANDARD_REVIEW_PACKET_TIER,
}
SUPPORTED_RETENTION_TIERS = set(review_packet_report.RETAINED_SOURCE_TIERS)

ALLOWED_NEXT_ACTIONS = [
    "review_checklist_row",
    "define_manual_review_criteria",
    "collect_exact_outcome_labels_without_live_execution",
    "continue_non_live_observation",
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
    "record_human_approval_without_external_review",
    "apply_live_change_without_separate_pr",
]
SOURCE_ALLOWED_NON_LIVE_ACTIONS = set(ALLOWED_NEXT_ACTIONS) | set(
    review_packet_report.ALLOWED_NEXT_ACTIONS
)
SOURCE_REQUIRED_FORBIDDEN_LIVE_PATH_ACTIONS = set(
    review_packet_report.FORBIDDEN_NEXT_ACTIONS
)
REQUIRED_LINEAGE_LIST_FIELDS = [
    "source_paper_replay_watchlist_ids",
    "source_aggregate_metric_evaluation_ids",
    "source_outcome_attachment_source_ids",
    "source_outcome_tracking_ids",
    "source_journal_entry_ids",
    "source_paper_replay_candidate_ids",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only retained watchlist review checklist. Checklist "
            "rows are non-live readiness records for exact outcome label "
            "collection and do not approve live changes."
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
    result = run_selected_strategy_edge_candidate_retained_watchlist_review_checklist_report(
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
                "review_checklist_rows_present": final.get(
                    "review_checklist_rows_present"
                ),
                "review_checklist_row_count": final.get(
                    "review_checklist_row_count"
                ),
                "exact_outcome_label_collection_ready_count": final.get(
                    "exact_outcome_label_collection_ready_count"
                ),
                "review_further_count": final.get("review_further_count"),
                "blocked_count": final.get("blocked_count"),
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
    return review_packet_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return review_packet_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_retained_watchlist_review_checklist_report(
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
    source_review_packet_report = review_packet_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_review_packet_rows = [
        row
        for row in _safe_list(source_review_packet_report.get("review_packet_rows"))
        if isinstance(row, dict)
    ]
    review_checklist_rows = build_review_checklist_rows(
        source_review_packet_rows=source_review_packet_rows
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_review_packet_report.get("inputs")),
        "diagnostic_only": True,
        "review_checklist_contract": {
            "review_checklist_contract_version": (
                REVIEW_CHECKLIST_CONTRACT_VERSION
            ),
            "review_checklist_mode": REVIEW_CHECKLIST_MODE,
            "source_review_packet_report_type": review_packet_report.REPORT_TYPE,
            "source_review_packet_contract_version": (
                review_packet_report.REVIEW_PACKET_CONTRACT_VERSION
            ),
            "review_checklist_rows_are_trades": False,
            "review_checklist_rows_are_orders": False,
            "review_checklist_rows_are_fills": False,
            "review_checklist_rows_enter_live_mapper_or_engine": False,
            "review_checklist_rows_are_live_edge_selection": False,
            "aggregate_metrics_are_exact_future_outcomes": False,
            "human_approval_recorded_by_this_report": False,
            "live_change_allowed_by_this_report": False,
        },
        "source_review_packet_final_assessment": _safe_dict(
            source_review_packet_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_review_packet_report.get("configurations_evaluated")
        ),
        "source_review_packet_rows": source_review_packet_rows,
        "review_checklist_rows": review_checklist_rows,
        "best_review_checklist_row": _first(review_checklist_rows),
        "final_assessment": build_final_assessment(
            review_checklist_rows=review_checklist_rows,
            source_review_packet_final_assessment=_safe_dict(
                source_review_packet_report.get("final_assessment")
            ),
        ),
        "assumptions": [
            "Checklist rows are report-only readiness records derived from retained watchlist review packets.",
            "This report does not perform or record human review and does not approve live changes.",
            "The only ready next step is non-live exact outcome label collection without order execution.",
            "Aggregate metrics remain aggregate-only evidence until exact labels are collected by a separate report.",
            "Candidate quality gate, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        ],
    }


def build_review_checklist_rows(
    *,
    source_review_packet_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        build_review_checklist_row(source_review_packet_row=row)
        for row in source_review_packet_rows
        if isinstance(row, dict)
    ]
    rows = deduplicate_review_checklist_ids(rows)
    rows.sort(key=_review_checklist_sort_key)
    return rows


def build_review_checklist_row(
    *,
    source_review_packet_row: dict[str, Any],
) -> dict[str, Any]:
    source_row = dict(source_review_packet_row)
    source_review_packet_id = _clean_text(
        source_row.get("retained_watchlist_review_packet_id")
    )
    source_retention_id = _clean_text(source_row.get("source_retention_id"))
    retention_observation_key = _clean_text(
        source_row.get("retention_observation_key")
    )
    source_review_packet_tier = _clean_text(source_row.get("review_packet_tier"))
    source_review_packet_priority = _clean_text(
        source_row.get("review_packet_priority")
    )
    source_retention_tier = _clean_text(source_row.get("source_retention_tier"))
    observed_configuration_count = _safe_int(
        source_row.get("observed_configuration_count")
    )
    source_watchlist_row_count = _safe_int(
        source_row.get("source_watchlist_row_count")
    )

    booleans = checklist_booleans(
        source_row=source_row,
        source_review_packet_id=source_review_packet_id,
        source_retention_id=source_retention_id,
        retention_observation_key=retention_observation_key,
        source_review_packet_tier=source_review_packet_tier,
        source_retention_tier=source_retention_tier,
        observed_configuration_count=observed_configuration_count,
        source_watchlist_row_count=source_watchlist_row_count,
    )
    tier, priority, recommended_action, reason, passed = classify_checklist_row(
        source_review_packet_tier=source_review_packet_tier,
        source_review_packet_priority=source_review_packet_priority,
        observed_configuration_count=observed_configuration_count,
        source_watchlist_row_count=source_watchlist_row_count,
        booleans=booleans,
    )

    return {
        "retained_watchlist_review_checklist_id": build_review_checklist_id(
            source_review_packet_id=source_review_packet_id
        ),
        "review_checklist_contract_version": REVIEW_CHECKLIST_CONTRACT_VERSION,
        "review_checklist_mode": REVIEW_CHECKLIST_MODE,
        "source_report_type": source_row.get("source_report_type"),
        "source_review_packet_report_type": review_packet_report.REPORT_TYPE,
        "source_review_packet_id": source_review_packet_id,
        "source_retention_id": source_retention_id,
        "retention_observation_key": retention_observation_key,
        "symbol": source_row.get("symbol"),
        "strategy": source_row.get("strategy"),
        "horizon": source_row.get("horizon"),
        "source_review_packet_tier": source_review_packet_tier,
        "source_review_packet_priority": source_review_packet_priority,
        "source_retention_tier": source_retention_tier,
        "source_retention_priority": source_row.get("source_retention_priority"),
        "observed_configuration_count": observed_configuration_count,
        "observed_configurations": _safe_list(
            source_row.get("observed_configurations")
        ),
        "source_watchlist_row_count": source_watchlist_row_count,
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
        "source_review_status": source_row.get("review_status"),
        "source_review_decision_status": source_row.get("review_decision_status"),
        "source_review_required_before_live_change": source_row.get(
            "review_required_before_live_change"
        ),
        "source_approved_for_live_change": source_row.get(
            "approved_for_live_change"
        ),
        "source_live_change_allowed_by_this_report": source_row.get(
            "live_change_allowed_by_this_report"
        ),
        "source_live_change_requires_separate_pr": source_row.get(
            "live_change_requires_separate_pr"
        ),
        "source_allowed_next_actions": _safe_list(
            source_row.get("allowed_next_actions")
        ),
        "source_forbidden_next_actions": _safe_list(
            source_row.get("forbidden_next_actions")
        ),
        **booleans,
        "review_checklist_tier": tier,
        "review_checklist_priority": priority,
        "recommended_non_live_next_action": recommended_action,
        "checklist_reason": reason,
        "checklist_passed": passed,
        "this_report_records_human_approval": False,
        "human_review_completed": False,
        "human_reviewer": None,
        "human_reviewed_at": None,
        "approved_for_live_change": False,
        "live_change_requires_separate_pr": True,
        "live_change_allowed_by_this_report": False,
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
        "review_checklist_is_live_edge_selection": False,
    }


def checklist_booleans(
    *,
    source_row: dict[str, Any],
    source_review_packet_id: str,
    source_retention_id: str,
    retention_observation_key: str,
    source_review_packet_tier: str,
    source_retention_tier: str,
    observed_configuration_count: int,
    source_watchlist_row_count: int,
) -> dict[str, bool]:
    source_allowed_actions = {
        _clean_text(action)
        for action in _safe_list(source_row.get("allowed_next_actions"))
        if _clean_text(action)
    }
    source_forbidden_actions = {
        _clean_text(action)
        for action in _safe_list(source_row.get("forbidden_next_actions"))
        if _clean_text(action)
    }
    return {
        "has_source_review_packet_id": bool(source_review_packet_id),
        "has_source_retention_id": bool(source_retention_id),
        "has_retention_observation_key": bool(retention_observation_key),
        "has_supported_review_packet_tier": (
            source_review_packet_tier in SUPPORTED_REVIEW_PACKET_TIERS
        ),
        "has_supported_retention_tier": (
            source_retention_tier in SUPPORTED_RETENTION_TIERS
        ),
        "observed_in_multiple_configurations": observed_configuration_count >= 2,
        "has_multiple_source_watchlist_rows": source_watchlist_row_count >= 2,
        "has_aggregate_score_evidence": has_numeric_evidence(
            source_row.get("max_aggregate_score"),
            source_row.get("aggregate_score_values"),
        ),
        "has_aggregate_sample_evidence": has_numeric_evidence(
            source_row.get("max_aggregate_sample_count"),
            source_row.get("aggregate_sample_count_values"),
        ),
        "has_all_required_lineage_lists": all(
            len(_safe_list(source_row.get(field))) > 0
            for field in REQUIRED_LINEAGE_LIST_FIELDS
        ),
        "source_review_is_pending": (
            source_row.get("review_status") == "pending_human_review"
            and source_row.get("review_decision_status") == "not_reviewed"
        ),
        "source_review_has_not_approved_live_change": (
            source_row.get("approved_for_live_change") is False
        ),
        "source_requires_human_review_before_live_change": (
            source_row.get("review_required_before_live_change") is True
        ),
        "source_live_change_disallowed_by_report": (
            source_row.get("live_change_allowed_by_this_report") is False
            and source_row.get("live_change_requires_separate_pr") is True
        ),
        "source_allowed_actions_are_non_live": (
            len(source_allowed_actions) > 0
            and source_allowed_actions <= SOURCE_ALLOWED_NON_LIVE_ACTIONS
        ),
        "source_forbidden_actions_cover_live_paths": (
            SOURCE_REQUIRED_FORBIDDEN_LIVE_PATH_ACTIONS
            <= source_forbidden_actions
        ),
        "non_live_safety_fields_intact": source_non_live_safety_fields_intact(
            source_row
        ),
    }


def has_numeric_evidence(value: Any, values: Any) -> bool:
    if _is_numeric_value(value):
        return True
    return any(_is_numeric_value(item) for item in _safe_list(values))


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def source_non_live_safety_fields_intact(source_row: dict[str, Any]) -> bool:
    return (
        source_row.get("production_live_selection_allowed") is False
        and source_row.get("mapper_live_path_allowed") is False
        and source_row.get("engine_live_path_allowed") is False
        and source_row.get("no_order_execution") is True
        and source_row.get("no_synthetic_fill") is True
        and source_row.get("no_pnl_claim") is True
        and source_row.get("order_id") is None
        and source_row.get("fill_id") is None
        and source_row.get("entry_price") is None
        and source_row.get("exit_price") is None
        and source_row.get("realized_pnl") is None
        and source_row.get("unrealized_pnl") is None
        and source_row.get("exact_outcome_used") is False
        and source_row.get("aggregate_metric_only") is True
        and source_row.get("review_packet_is_live_edge_selection") is False
    )


def classify_checklist_row(
    *,
    source_review_packet_tier: str,
    source_review_packet_priority: str,
    observed_configuration_count: int,
    source_watchlist_row_count: int,
    booleans: dict[str, bool],
) -> tuple[str, str, str, str, bool]:
    blocking_checks = [
        "has_source_review_packet_id",
        "has_source_retention_id",
        "has_retention_observation_key",
        "has_supported_review_packet_tier",
        "has_supported_retention_tier",
        "has_all_required_lineage_lists",
        "source_review_is_pending",
        "source_review_has_not_approved_live_change",
        "source_requires_human_review_before_live_change",
        "source_live_change_disallowed_by_report",
        "source_allowed_actions_are_non_live",
        "source_forbidden_actions_cover_live_paths",
        "non_live_safety_fields_intact",
    ]
    failed_checks = [check for check in blocking_checks if not booleans.get(check)]
    if failed_checks:
        return (
            REVIEW_CHECKLIST_BLOCKED_TIER,
            "blocked",
            "fix_review_packet_contract_before_review",
            "Blocked because required checklist checks failed: "
            + ", ".join(failed_checks),
            False,
        )

    if (
        source_review_packet_tier
        == review_packet_report.RETAINED_PROMISING_REVIEW_PACKET_TIER
        and source_review_packet_priority == "high"
        and observed_configuration_count >= 2
        and source_watchlist_row_count >= 2
        and booleans.get("has_aggregate_score_evidence") is True
        and booleans.get("has_aggregate_sample_evidence") is True
    ):
        return (
            EXACT_OUTCOME_LABEL_COLLECTION_READY_TIER,
            "high",
            "collect_exact_outcome_labels_without_live_execution",
            "Retained promising high-priority packet is complete enough for non-live exact outcome label collection.",
            True,
        )

    return (
        REVIEW_FURTHER_TIER,
        "medium",
        "define_manual_review_criteria",
        "Packet is safe and internally complete but does not meet the conservative ready criteria.",
        False,
    )


def build_review_checklist_id(*, source_review_packet_id: str) -> str:
    return f"{REVIEW_CHECKLIST_CONTRACT_VERSION}:{source_review_packet_id}"


def deduplicate_review_checklist_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = _clean_text(item.get("retained_watchlist_review_checklist_id"))
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["retained_watchlist_review_checklist_id"] = (
                f"{base_id}:dup_{seen[base_id]}"
            )
        deduplicated.append(item)
    return deduplicated


def build_review_checklist_safety_invariant_summary(
    review_checklist_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in review_checklist_rows if isinstance(row, dict)]
    checklist_ids = [
        row.get("retained_watchlist_review_checklist_id") for row in rows
    ]
    expected_allowed = set(ALLOWED_NEXT_ACTIONS)
    required_forbidden = set(FORBIDDEN_NEXT_ACTIONS)
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
        "review_checklist_rows_are_not_live_edge_selections": all(
            row.get("review_checklist_is_live_edge_selection") is False
            for row in rows
        ),
        "this_report_records_no_human_approval": all(
            row.get("this_report_records_human_approval") is False for row in rows
        ),
        "human_review_is_not_marked_completed_by_this_report": all(
            row.get("human_review_completed") is False
            and row.get("human_reviewer") is None
            and row.get("human_reviewed_at") is None
            for row in rows
        ),
        "live_change_is_not_allowed_by_this_report": all(
            row.get("live_change_allowed_by_this_report") is False
            and row.get("approved_for_live_change") is False
            for row in rows
        ),
        "live_change_requires_separate_pr": all(
            row.get("live_change_requires_separate_pr") is True for row in rows
        ),
        "all_rows_have_non_empty_source_review_packet_id": all(
            _is_non_empty_value(row.get("source_review_packet_id")) for row in rows
        ),
        "all_rows_have_non_empty_source_retention_id": all(
            _is_non_empty_value(row.get("source_retention_id")) for row in rows
        ),
        "all_rows_have_non_empty_retention_observation_key": all(
            _is_non_empty_value(row.get("retention_observation_key"))
            for row in rows
        ),
        "all_rows_preserve_at_least_one_source_paper_replay_watchlist_id": all(
            len(_safe_list(row.get("source_paper_replay_watchlist_ids"))) > 0
            for row in rows
        ),
        "all_rows_preserve_source_aggregate_metric_evaluation_ids": all(
            len(_safe_list(row.get("source_aggregate_metric_evaluation_ids"))) > 0
            for row in rows
        ),
        "all_rows_preserve_source_outcome_attachment_source_ids": all(
            len(_safe_list(row.get("source_outcome_attachment_source_ids"))) > 0
            for row in rows
        ),
        "all_rows_preserve_source_outcome_tracking_ids": all(
            len(_safe_list(row.get("source_outcome_tracking_ids"))) > 0
            for row in rows
        ),
        "all_rows_preserve_source_journal_entry_ids": all(
            len(_safe_list(row.get("source_journal_entry_ids"))) > 0
            for row in rows
        ),
        "all_rows_preserve_source_paper_replay_candidate_ids": all(
            len(_safe_list(row.get("source_paper_replay_candidate_ids"))) > 0
            for row in rows
        ),
        "allowed_next_actions_are_exactly_the_expected_non_live_whitelist": all(
            set(_safe_list(row.get("allowed_next_actions"))) == expected_allowed
            for row in rows
        ),
        "forbidden_next_actions_include_required_live_path_prohibitions": all(
            required_forbidden <= set(_safe_list(row.get("forbidden_next_actions")))
            for row in rows
        ),
        "review_checklist_ids_are_unique": len(checklist_ids)
        == len(set(checklist_ids)),
    }


def build_final_assessment(
    *,
    review_checklist_rows: Sequence[dict[str, Any]],
    source_review_packet_final_assessment: dict[str, Any],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    source_candidate_counts = _safe_dict(
        source_review_packet_final_assessment.get("candidate_counts")
    )
    for key, value in source_candidate_counts.items():
        counts[str(key)] += _safe_int(value)

    for row in review_checklist_rows:
        counts["review_checklist_row"] += 1
        tier = row.get("review_checklist_tier")
        if tier == EXACT_OUTCOME_LABEL_COLLECTION_READY_TIER:
            counts["exact_outcome_label_collection_ready"] += 1
        elif tier == REVIEW_FURTHER_TIER:
            counts["review_further"] += 1
        elif tier == REVIEW_CHECKLIST_BLOCKED_TIER:
            counts["blocked"] += 1

    production_candidate_count = _safe_int(
        source_review_packet_final_assessment.get("production_candidate_count")
    )
    if production_candidate_count == 0:
        production_candidate_count = counts["production_candidate"]

    recommended_next_stage = recommended_next_stage_for_counts(counts)

    return {
        "review_checklist_rows_present": counts["review_checklist_row"] > 0,
        "review_checklist_row_count": counts["review_checklist_row"],
        "exact_outcome_label_collection_ready_count": counts[
            "exact_outcome_label_collection_ready"
        ],
        "review_further_count": counts["review_further"],
        "blocked_count": counts["blocked"],
        "production_candidate_count": production_candidate_count,
        "candidate_counts": dict(counts),
        "review_checklist_safety_invariant_summary": (
            build_review_checklist_safety_invariant_summary(review_checklist_rows)
        ),
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
        "recommended_next_stage": recommended_next_stage,
    }


def recommended_next_stage_for_counts(counts: Counter[str]) -> str:
    if counts["exact_outcome_label_collection_ready"] > 0:
        return "design_exact_outcome_label_collection_report"
    if counts["review_further"] > 0:
        return "continue_manual_review_without_live_change"
    if counts["blocked"] > 0:
        return "fix_review_packet_contract_before_review"
    return "continue_watchlist_retention_observation"


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_exact_outcome_label_collection_report":
        return "Stop here. The next stage should design a report-only exact outcome label collection report without live execution."
    if recommended_next_stage == "continue_manual_review_without_live_change":
        return "Stop here. Define manual review criteria without approving live changes."
    if recommended_next_stage == "fix_review_packet_contract_before_review":
        return "Stop here. Fix review packet lineage or safety contract issues before further review."
    if recommended_next_stage == "continue_watchlist_retention_observation":
        return "Stop here. Continue non-live retained watchlist observation before checklist review."
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
        f"- review_checklist_rows_present: {final.get('review_checklist_rows_present')}",
        f"- review_checklist_row_count: {final.get('review_checklist_row_count')}",
        f"- exact_outcome_label_collection_ready_count: {final.get('exact_outcome_label_collection_ready_count')}",
        f"- review_further_count: {final.get('review_further_count')}",
        f"- blocked_count: {final.get('blocked_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- review_checklist_contract_version: {REVIEW_CHECKLIST_CONTRACT_VERSION}",
        f"- review_checklist_mode: {REVIEW_CHECKLIST_MODE}",
        f"- source_review_packet_report_type: {review_packet_report.REPORT_TYPE}",
        "- Checklist rows are non-live readiness records, not trades, orders, fills, or live engine candidates.",
        "- This report records no human approval and allows no live change.",
        "- The only ready next action is report-only exact outcome label collection without live execution.",
        "",
        "## Safety Invariants",
        f"- review_checklist_safety_invariant_summary: {final.get('review_checklist_safety_invariant_summary')}",
        "",
        "## Review Checklist Rows",
    ]

    for row in _safe_list(report.get("review_checklist_rows")):
        item = _safe_dict(row)
        lines.extend(
            [
                "",
                f"### {item.get('retention_observation_key')}",
                f"- review_checklist_tier: {item.get('review_checklist_tier')}",
                f"- review_checklist_priority: {item.get('review_checklist_priority')}",
                f"- recommended_non_live_next_action: {item.get('recommended_non_live_next_action')}",
                f"- source_review_packet_tier: {item.get('source_review_packet_tier')}",
                f"- source_review_packet_priority: {item.get('source_review_packet_priority')}",
                f"- observed_configuration_count: {item.get('observed_configuration_count')}",
                f"- source_watchlist_row_count: {item.get('source_watchlist_row_count')}",
                f"- max_aggregate_score: {item.get('max_aggregate_score')}",
                f"- max_aggregate_sample_count: {item.get('max_aggregate_sample_count')}",
                f"- checklist_passed: {item.get('checklist_passed')}",
                f"- checklist_reason: {item.get('checklist_reason')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _review_checklist_sort_key(row: dict[str, Any]) -> tuple[int, float, int, str, str]:
    priority_order = {"high": 0, "medium": 1, "blocked": 2, "low": 3}
    return (
        priority_order.get(str(row.get("review_checklist_priority") or ""), 9),
        -float(row.get("max_aggregate_score") or 0.0),
        -int(row.get("max_aggregate_sample_count") or 0),
        str(row.get("retention_observation_key") or ""),
        str(row.get("retained_watchlist_review_checklist_id") or ""),
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
    return review_packet_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return review_packet_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return review_packet_report._safe_int(value)


if __name__ == "__main__":
    main()
