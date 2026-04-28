from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_retained_watchlist_review_checklist_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_retained_watchlist_review_checklist_report as report_module,
)


def _review_packet_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "retained_watchlist_review_packet_id": (
            "retained_watchlist_review_packet_v1:"
            "paper_replay_watchlist_observation_retention_v1:btcusdt:swing:4h"
        ),
        "review_packet_contract_version": "retained_watchlist_review_packet_v1",
        "review_packet_mode": "retained_aggregate_metric_watchlist_human_review_packet",
        "source_report_type": (
            "selected_strategy_edge_candidate_paper_replay_watchlist_report"
        ),
        "source_retention_report_type": (
            "selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report"
        ),
        "source_retention_id": (
            "paper_replay_watchlist_observation_retention_v1:btcusdt:swing:4h"
        ),
        "retention_observation_key": "BTCUSDT:swing:4h",
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "horizon": "4h",
        "source_retention_tier": (
            "paper_replay_retained_promising_observation"
        ),
        "source_retention_priority": "high",
        "observed_configuration_count": 2,
        "observed_configurations": [
            {"window_hours": 144, "max_rows": 5000, "display_name": "144h/5000"},
            {"window_hours": 336, "max_rows": 10000, "display_name": "336h/10000"},
        ],
        "source_watchlist_row_count": 2,
        "source_paper_replay_watchlist_ids": ["watchlist:144", "watchlist:336"],
        "source_aggregate_metric_evaluation_ids": [
            "evaluation:144",
            "evaluation:336",
        ],
        "source_outcome_attachment_source_ids": [
            "attachment:144",
            "attachment:336",
        ],
        "source_outcome_tracking_ids": ["tracking:144", "tracking:336"],
        "source_journal_entry_ids": ["journal:144", "journal:336"],
        "source_paper_replay_candidate_ids": ["candidate:144", "candidate:336"],
        "max_aggregate_score": 62.75,
        "max_aggregate_sample_count": 82,
        "max_aggregate_labeled_count": 70,
        "aggregate_score_values": [60.0, 62.75],
        "aggregate_sample_count_values": [80, 82],
        "aggregate_labeled_count_values": [68, 70],
        "evaluation_buckets": ["aggregate_metric_promising_observation"],
        "best_watchlist_tier": "paper_replay_promising_watchlist",
        "best_watchlist_priority": "high",
        "review_packet_tier": "retained_promising_review_packet",
        "review_packet_priority": "high",
        "review_status": "pending_human_review",
        "review_required_before_live_change": True,
        "review_decision_status": "not_reviewed",
        "reviewer": None,
        "reviewed_at": None,
        "approved_for_live_change": False,
        "live_change_requires_separate_pr": True,
        "live_change_allowed_by_this_report": False,
        "review_reason": "Needs human review.",
        "allowed_next_actions": [
            "review_evidence_packet",
            "define_manual_review_criteria",
            "continue_non_live_observation",
            "collect_exact_outcome_labels_without_live_execution",
        ],
        "forbidden_next_actions": [
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
        ],
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
    row.update(overrides)
    return row


def _standard_review_packet_row(**overrides: Any) -> dict[str, Any]:
    return _review_packet_row(
        retained_watchlist_review_packet_id=(
            "retained_watchlist_review_packet_v1:"
            "paper_replay_watchlist_observation_retention_v1:ethusdt:swing:4h"
        ),
        source_retention_id=(
            "paper_replay_watchlist_observation_retention_v1:ethusdt:swing:4h"
        ),
        retention_observation_key="ETHUSDT:swing:4h",
        symbol="ETHUSDT",
        source_retention_tier="paper_replay_retained_standard_observation",
        source_retention_priority="medium",
        best_watchlist_tier="paper_replay_standard_watchlist",
        best_watchlist_priority="medium",
        review_packet_tier="retained_standard_review_packet",
        review_packet_priority="medium",
        max_aggregate_score=55.0,
        **overrides,
    )


def test_promising_high_priority_complete_packet_is_ready_for_exact_labels() -> None:
    rows = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_review_packet_row()]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["review_checklist_tier"] == (
        report_module.EXACT_OUTCOME_LABEL_COLLECTION_READY_TIER
    )
    assert row["review_checklist_priority"] == "high"
    assert row["recommended_non_live_next_action"] == (
        "collect_exact_outcome_labels_without_live_execution"
    )
    assert row["checklist_passed"] is True


def test_retained_standard_review_packet_requires_further_review() -> None:
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_standard_review_packet_row()]
    )[0]

    assert row["review_checklist_tier"] == report_module.REVIEW_FURTHER_TIER
    assert row["review_checklist_priority"] == "medium"
    assert row["recommended_non_live_next_action"] == "define_manual_review_criteria"
    assert row["checklist_passed"] is False


def test_missing_lineage_list_blocks_checklist() -> None:
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[
            _review_packet_row(source_outcome_tracking_ids=[])
        ]
    )[0]

    assert row["review_checklist_tier"] == (
        report_module.REVIEW_CHECKLIST_BLOCKED_TIER
    )
    assert row["review_checklist_priority"] == "blocked"
    assert row["has_all_required_lineage_lists"] is False


def test_source_live_change_approval_or_allowed_blocks_checklist() -> None:
    approved = report_module.build_review_checklist_rows(
        source_review_packet_rows=[
            _review_packet_row(approved_for_live_change=True)
        ]
    )[0]
    allowed = report_module.build_review_checklist_rows(
        source_review_packet_rows=[
            _review_packet_row(live_change_allowed_by_this_report=True)
        ]
    )[0]

    assert approved["review_checklist_tier"] == (
        report_module.REVIEW_CHECKLIST_BLOCKED_TIER
    )
    assert allowed["review_checklist_tier"] == (
        report_module.REVIEW_CHECKLIST_BLOCKED_TIER
    )
    assert approved["source_review_has_not_approved_live_change"] is False
    assert allowed["source_live_change_disallowed_by_report"] is False


def test_unsafe_allowed_next_actions_blocks_checklist() -> None:
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[
            _review_packet_row(allowed_next_actions=["place_order"])
        ]
    )[0]

    assert row["review_checklist_tier"] == (
        report_module.REVIEW_CHECKLIST_BLOCKED_TIER
    )
    assert row["source_allowed_actions_are_non_live"] is False


def test_missing_required_forbidden_next_actions_blocks_checklist() -> None:
    forbidden = [
        action
        for action in _review_packet_row()["forbidden_next_actions"]
        if action != "place_order"
    ]
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[
            _review_packet_row(forbidden_next_actions=forbidden)
        ]
    )[0]

    assert row["review_checklist_tier"] == (
        report_module.REVIEW_CHECKLIST_BLOCKED_TIER
    )
    assert row["source_forbidden_actions_cover_live_paths"] is False


def test_source_identity_configurations_and_lineage_lists_are_preserved() -> None:
    source = _review_packet_row()
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[source]
    )[0]

    assert row["source_review_packet_id"] == source[
        "retained_watchlist_review_packet_id"
    ]
    assert row["source_retention_id"] == source["source_retention_id"]
    assert row["retention_observation_key"] == source["retention_observation_key"]
    assert row["observed_configurations"] == source["observed_configurations"]
    assert row["source_paper_replay_watchlist_ids"] == [
        "watchlist:144",
        "watchlist:336",
    ]
    assert row["source_aggregate_metric_evaluation_ids"] == [
        "evaluation:144",
        "evaluation:336",
    ]
    assert row["source_outcome_attachment_source_ids"] == [
        "attachment:144",
        "attachment:336",
    ]
    assert row["source_outcome_tracking_ids"] == ["tracking:144", "tracking:336"]
    assert row["source_journal_entry_ids"] == ["journal:144", "journal:336"]
    assert row["source_paper_replay_candidate_ids"] == [
        "candidate:144",
        "candidate:336",
    ]


def test_report_does_not_record_human_approval() -> None:
    row = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_review_packet_row()]
    )[0]

    assert row["this_report_records_human_approval"] is False
    assert row["human_review_completed"] is False
    assert row["human_reviewer"] is None
    assert row["human_reviewed_at"] is None
    assert row["approved_for_live_change"] is False


def test_checklist_rows_are_strictly_non_live_and_do_not_enter_runtime_paths() -> None:
    rows = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_review_packet_row()]
    )
    row = rows[0]
    invariants = report_module.build_review_checklist_safety_invariant_summary(rows)

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    assert row["review_checklist_is_live_edge_selection"] is False
    assert set(row["allowed_next_actions"]) == set(report_module.ALLOWED_NEXT_ACTIONS)
    assert set(report_module.FORBIDDEN_NEXT_ACTIONS).issubset(
        set(row["forbidden_next_actions"])
    )
    assert invariants["all_rows_production_live_selection_allowed_is_false"] is True
    assert invariants["all_rows_mapper_live_path_allowed_is_false"] is True
    assert invariants["all_rows_engine_live_path_allowed_is_false"] is True
    assert (
        invariants["review_checklist_rows_are_not_live_edge_selections"] is True
    )
    assert (
        invariants[
            "allowed_next_actions_are_exactly_the_expected_non_live_whitelist"
        ]
        is True
    )
    assert (
        invariants[
            "forbidden_next_actions_include_required_live_path_prohibitions"
        ]
        is True
    )


def test_no_order_fill_price_or_pnl_fields_are_populated() -> None:
    rows = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_review_packet_row()]
    )
    row = rows[0]
    invariants = report_module.build_review_checklist_safety_invariant_summary(rows)

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert row["order_id"] is None
    assert row["fill_id"] is None
    assert row["entry_price"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["unrealized_pnl"] is None
    assert row["exact_outcome_used"] is False
    assert row["aggregate_metric_only"] is True
    assert invariants["all_rows_no_order_execution_is_true"] is True
    assert invariants["all_rows_no_synthetic_fill_is_true"] is True
    assert invariants["all_rows_no_pnl_claim_is_true"] is True
    assert invariants["no_order_or_fill_ids_are_present"] is True
    assert invariants["no_price_or_pnl_fields_are_present"] is True
    assert invariants["exact_outcomes_are_not_used"] is True
    assert invariants["aggregate_metrics_remain_aggregate_only"] is True


def test_final_assessment_recommends_exact_label_report_when_ready_rows_exist() -> None:
    rows = report_module.build_review_checklist_rows(
        source_review_packet_rows=[_review_packet_row(), _standard_review_packet_row()]
    )
    final = report_module.build_final_assessment(
        review_checklist_rows=rows,
        source_review_packet_final_assessment={
            "production_candidate_count": 0,
            "candidate_counts": {"production_candidate": 0},
        },
    )

    assert final["review_checklist_rows_present"] is True
    assert final["review_checklist_row_count"] == 2
    assert final["exact_outcome_label_collection_ready_count"] == 1
    assert final["review_further_count"] == 1
    assert final["blocked_count"] == 0
    assert final["production_candidate_count"] == 0
    assert final["recommended_next_stage"] == (
        "design_exact_outcome_label_collection_report"
    )
    invariant_summary = final["review_checklist_safety_invariant_summary"]
    assert invariant_summary["review_checklist_ids_are_unique"] is True
    assert (
        invariant_summary[
            "all_rows_preserve_at_least_one_source_paper_replay_watchlist_id"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_rows_preserve_source_aggregate_metric_evaluation_ids"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_rows_preserve_source_outcome_attachment_source_ids"
        ]
        is True
    )
    assert invariant_summary["all_rows_preserve_source_outcome_tracking_ids"] is True
    assert invariant_summary["all_rows_preserve_source_journal_entry_ids"] is True
    assert (
        invariant_summary["all_rows_preserve_source_paper_replay_candidate_ids"]
        is True
    )
    assert invariant_summary["this_report_records_no_human_approval"] is True
    assert (
        invariant_summary[
            "human_review_is_not_marked_completed_by_this_report"
        ]
        is True
    )
    assert invariant_summary["live_change_is_not_allowed_by_this_report"] is True
    assert invariant_summary["live_change_requires_separate_pr"] is True


def test_build_report_uses_source_review_packet_builder(monkeypatch: Any) -> None:
    source_report = {
        "inputs": {"input_path": "input.jsonl", "output_dir": "out"},
        "final_assessment": {
            "production_candidate_count": 0,
            "candidate_counts": {"production_candidate": 0},
        },
        "configurations_evaluated": [],
        "review_packet_rows": [_review_packet_row()],
    }
    calls: list[dict[str, Any]] = []

    def fake_build_report(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return source_report

    monkeypatch.setattr(
        report_module.review_packet_report,
        "build_report",
        fake_build_report,
    )

    report = report_module.build_report(
        input_path=Path("source.jsonl"),
        output_dir=Path("reports"),
        configurations=None,
    )

    assert calls == [
        {
            "input_path": Path("source.jsonl"),
            "output_dir": Path("reports"),
            "configurations": None,
        }
    ]
    assert report["report_type"] == report_module.REPORT_TYPE
    assert report["review_checklist_rows"][0][
        "source_review_packet_report_type"
    ] == (
        "selected_strategy_edge_candidate_retained_watchlist_review_packet_report"
    )


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_retained_watchlist_review_checklist_report
        is report_module.run_selected_strategy_edge_candidate_retained_watchlist_review_checklist_report
    )
