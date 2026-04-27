from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_retained_watchlist_review_packet_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_retained_watchlist_review_packet_report as report_module,
)


def _retention_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_replay_watchlist_observation_retention_id": (
            "paper_replay_watchlist_observation_retention_v1:"
            "btcusdt:swing:4h"
        ),
        "retention_contract_version": (
            "paper_replay_watchlist_observation_retention_v1"
        ),
        "retention_mode": "aggregate_metric_watchlist_observation_retention",
        "source_report_type": (
            "selected_strategy_edge_candidate_paper_replay_watchlist_report"
        ),
        "retention_observation_key": "BTCUSDT:swing:4h",
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "horizon": "4h",
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
        "best_watchlist_tier": "paper_replay_promising_watchlist",
        "best_watchlist_priority": "high",
        "retained_promising_count": 1,
        "retained_standard_count": 0,
        "single_window_promising_count": 0,
        "single_window_standard_count": 0,
        "max_aggregate_score": 62.75,
        "max_aggregate_sample_count": 120,
        "max_aggregate_labeled_count": 100,
        "aggregate_score_values": [60.0, 62.75],
        "aggregate_sample_count_values": [80, 120],
        "aggregate_labeled_count_values": [70, 100],
        "evaluation_buckets": ["aggregate_metric_promising_observation"],
        "retention_tier": (
            "paper_replay_retained_promising_observation"
        ),
        "retention_priority": "high",
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
    row.update(overrides)
    return row


def _standard_retention_row(**overrides: Any) -> dict[str, Any]:
    return _retention_row(
        paper_replay_watchlist_observation_retention_id=(
            "paper_replay_watchlist_observation_retention_v1:"
            "ethusdt:swing:4h"
        ),
        retention_observation_key="ETHUSDT:swing:4h",
        symbol="ETHUSDT",
        retained_promising_count=0,
        retained_standard_count=1,
        best_watchlist_tier="paper_replay_standard_watchlist",
        best_watchlist_priority="medium",
        retention_tier="paper_replay_retained_standard_observation",
        retention_priority="medium",
        max_aggregate_score=55.0,
        **overrides,
    )


def _single_window_retention_row(**overrides: Any) -> dict[str, Any]:
    return _retention_row(
        paper_replay_watchlist_observation_retention_id=(
            "paper_replay_watchlist_observation_retention_v1:"
            "solusdt:scalp:1h"
        ),
        retention_observation_key="SOLUSDT:scalp:1h",
        symbol="SOLUSDT",
        strategy="scalp",
        horizon="1h",
        observed_configuration_count=1,
        source_watchlist_row_count=1,
        retained_promising_count=0,
        retained_standard_count=0,
        single_window_promising_count=1,
        retention_tier="paper_replay_single_window_promising_observation",
        retention_priority="medium",
        **overrides,
    )


def test_retained_promising_observation_becomes_high_priority_review_packet() -> None:
    rows = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )

    assert len(rows) == 1
    row = rows[0]
    assert (
        row["review_packet_tier"]
        == report_module.RETAINED_PROMISING_REVIEW_PACKET_TIER
    )
    assert row["review_packet_priority"] == "high"


def test_retained_standard_observation_becomes_medium_priority_review_packet() -> None:
    rows = report_module.build_review_packet_rows(
        source_retention_rows=[_standard_retention_row()]
    )

    assert len(rows) == 1
    row = rows[0]
    assert (
        row["review_packet_tier"]
        == report_module.RETAINED_STANDARD_REVIEW_PACKET_TIER
    )
    assert row["review_packet_priority"] == "medium"


def test_single_window_retention_rows_are_excluded_and_counted() -> None:
    source_rows = [_retention_row(), _single_window_retention_row()]
    packet_rows = report_module.build_review_packet_rows(
        source_retention_rows=source_rows
    )
    final = report_module.build_final_assessment(
        review_packet_rows=packet_rows,
        source_retention_rows=source_rows,
        source_retention_final_assessment={"production_candidate_count": 0},
    )

    assert len(packet_rows) == 1
    assert final["review_packet_count"] == 1
    assert final["retained_source_row_count"] == 1
    assert final["excluded_single_window_observation_count"] == 1


def test_source_identity_observed_configurations_and_lineage_lists_are_preserved() -> None:
    source = _retention_row()
    row = report_module.build_review_packet_rows(source_retention_rows=[source])[0]

    assert row["source_retention_id"] == source[
        "paper_replay_watchlist_observation_retention_id"
    ]
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


def test_review_fields_are_pending_not_reviewed_and_do_not_approve_live_changes() -> None:
    row = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )[0]

    assert row["review_status"] == "pending_human_review"
    assert row["review_required_before_live_change"] is True
    assert row["review_decision_status"] == "not_reviewed"
    assert row["reviewer"] is None
    assert row["reviewed_at"] is None
    assert row["approved_for_live_change"] is False
    assert row["live_change_requires_separate_pr"] is True
    assert row["live_change_allowed_by_this_report"] is False


def test_allowed_next_actions_are_non_live_report_only_actions() -> None:
    row = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )[0]
    allowed_actions = set(row["allowed_next_actions"])

    assert allowed_actions == set(report_module.ALLOWED_NEXT_ACTIONS)
    assert "place_order" not in allowed_actions
    assert "route_to_live_mapper" not in allowed_actions
    assert "route_to_live_engine" not in allowed_actions
    assert "claim_realized_pnl" not in allowed_actions


def test_forbidden_next_actions_include_live_path_order_and_pnl_prohibitions() -> None:
    row = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )[0]

    assert set(report_module.FORBIDDEN_NEXT_ACTIONS).issubset(
        set(row["forbidden_next_actions"])
    )


def test_review_packet_rows_are_strictly_non_live() -> None:
    rows = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )
    row = rows[0]
    invariants = report_module.build_review_packet_safety_invariant_summary(rows)

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    assert row["review_packet_is_live_edge_selection"] is False
    assert invariants["all_rows_production_live_selection_allowed_is_false"] is True
    assert invariants["all_rows_mapper_live_path_allowed_is_false"] is True
    assert invariants["all_rows_engine_live_path_allowed_is_false"] is True
    assert invariants["review_packets_are_not_live_edge_selections"] is True
    assert invariants["review_is_required_before_live_change"] is True
    assert invariants["live_change_is_not_allowed_by_this_report"] is True


def test_no_order_fill_price_or_pnl_fields_are_populated() -> None:
    rows = report_module.build_review_packet_rows(
        source_retention_rows=[_retention_row()]
    )
    row = rows[0]
    invariants = report_module.build_review_packet_safety_invariant_summary(rows)

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


def test_final_assessment_recommends_human_review_when_packets_exist() -> None:
    source_rows = [_retention_row(), _standard_retention_row()]
    packet_rows = report_module.build_review_packet_rows(
        source_retention_rows=source_rows
    )

    final = report_module.build_final_assessment(
        review_packet_rows=packet_rows,
        source_retention_rows=source_rows,
        source_retention_final_assessment={
            "production_candidate_count": 0,
            "candidate_counts": {"production_candidate": 0},
        },
    )

    assert final["review_packets_present"] is True
    assert final["review_packet_count"] == 2
    assert final["retained_promising_review_packet_count"] == 1
    assert final["retained_standard_review_packet_count"] == 1
    assert final["production_candidate_count"] == 0
    assert final["recommended_next_stage"] == (
        "human_review_retained_watchlist_packet"
    )
    invariant_summary = final["review_packet_safety_invariant_summary"]
    assert invariant_summary["review_packet_ids_are_unique"] is True
    assert (
        invariant_summary[
            "all_packets_preserve_at_least_one_source_paper_replay_watchlist_id"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_packets_preserve_source_aggregate_metric_evaluation_ids"
        ]
        is True
    )
    assert (
        invariant_summary[
            "all_packets_preserve_source_outcome_attachment_source_ids"
        ]
        is True
    )
    assert (
        invariant_summary["all_packets_preserve_source_outcome_tracking_ids"]
        is True
    )
    assert invariant_summary["all_packets_preserve_source_journal_entry_ids"] is True
    assert (
        invariant_summary["all_packets_preserve_source_paper_replay_candidate_ids"]
        is True
    )


def test_build_report_uses_source_retention_builder(monkeypatch: Any) -> None:
    source_report = {
        "inputs": {"input_path": "input.jsonl", "output_dir": "out"},
        "final_assessment": {
            "production_candidate_count": 0,
            "candidate_counts": {"production_candidate": 0},
        },
        "configurations_evaluated": [],
        "retention_rows": [_retention_row()],
    }
    calls: list[dict[str, Any]] = []

    def fake_build_report(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return source_report

    monkeypatch.setattr(
        report_module.retention_report,
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
    assert report["review_packet_rows"][0]["source_retention_report_type"] == (
        "selected_strategy_edge_candidate_paper_replay_watchlist_observation_retention_report"
    )


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_retained_watchlist_review_packet_report
        is report_module.run_selected_strategy_edge_candidate_retained_watchlist_review_packet_report
    )
