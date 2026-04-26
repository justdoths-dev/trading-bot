from __future__ import annotations

from typing import Any

from src.research import (
    selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report as report_module,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report as attachment_report,
)


def _attachment_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome_attachment_source_id": (
            "paper_replay_outcome_attachment_source_v1:"
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
        ),
        "source_outcome_tracking_id": (
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
        ),
        "source_journal_entry_id": (
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
        ),
        "source_paper_replay_candidate_id": (
            "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
        ),
        "symbol": "ETHUSDT",
        "strategy": "swing",
        "horizon": "4h",
        "source_policy_class": "paper_only_candidate",
        "attachment_source_status": "aggregate_diagnostic_metrics_only",
        "aggregate_metric_attachment_available": True,
        "aggregate_sample_count": 46,
        "aggregate_labeled_count": 46,
        "aggregate_median_future_return_pct": 0.174932,
        "aggregate_positive_rate_pct": 52.17,
        "aggregate_robustness_signal": "up_rate_pct",
        "aggregate_robustness_signal_pct": 52.17,
        "aggregate_score": 63.6,
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
        "exact_outcome_attachment_available": False,
        "exact_attached_future_label": None,
        "exact_attached_future_return_pct": None,
        "safety_notes": ["source row is report-only"],
    }
    row.update(overrides)
    return row


def _source_attachment_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "configuration": {
            "window_hours": 336,
            "max_rows": 10000,
            "display_name": "336h_10000",
        },
        "attachment_source_row_count": len(rows),
        "paper_replay_candidate_count": len(rows),
        "production_candidate_count": 0,
        "attachment_source_rows": rows,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_attachment_summary=_source_attachment_summary(rows)
    )


def test_aggregate_metric_rows_become_evaluation_rows() -> None:
    summary = _summary([_attachment_row()])

    assert summary["aggregate_evaluation_row_count"] == 1
    row = summary["aggregate_evaluation_rows"][0]
    assert (
        row["aggregate_metric_evaluation_id"]
        == "aggregate_metric_paper_replay_evaluation_v1:"
        + row["source_outcome_attachment_source_id"]
    )
    assert row["evaluation_contract_version"] == (
        "aggregate_metric_paper_replay_evaluation_v1"
    )
    assert row["evaluation_mode"] == "aggregate_metric_only_evaluation"
    assert row["source_outcome_tracking_id"].startswith(
        "paper_replay_outcome_tracking_v1:"
    )
    assert row["source_journal_entry_id"].startswith(
        "paper_replay_observation_journal_v1:"
    )
    assert row["source_paper_replay_candidate_id"].startswith(
        "paper_only_replay_v1:"
    )
    assert row["symbol"] == "ETHUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["source_policy_class"] == "paper_only_candidate"
    assert row["attachment_source_status"] == "aggregate_diagnostic_metrics_only"
    assert row["aggregate_sample_count"] == 46
    assert row["aggregate_labeled_count"] == 46
    assert row["aggregate_median_future_return_pct"] == 0.174932
    assert row["aggregate_positive_rate_pct"] == 52.17
    assert row["aggregate_robustness_signal"] == "up_rate_pct"
    assert row["aggregate_robustness_signal_pct"] == 52.17
    assert row["aggregate_score"] == 63.6
    assert row["exact_outcome_used"] is False
    assert row["aggregate_metric_only"] is True


def test_promising_bucket_classification() -> None:
    summary = _summary(
        [
            _attachment_row(
                aggregate_sample_count=30,
                aggregate_labeled_count=30,
                aggregate_median_future_return_pct=0.01,
                aggregate_positive_rate_pct=45,
                aggregate_score=60,
            )
        ]
    )
    row = summary["aggregate_evaluation_rows"][0]
    final = report_module.build_final_assessment([summary])

    assert row["evaluation_bucket"] == "aggregate_metric_promising_observation"
    assert final["promising_observation_count"] == 1
    assert final["recommended_next_stage"] == "design_paper_replay_watchlist_report"


def test_watchlist_bucket_classification() -> None:
    summary = _summary(
        [
            _attachment_row(
                aggregate_sample_count=30,
                aggregate_labeled_count=30,
                aggregate_median_future_return_pct=0.01,
                aggregate_positive_rate_pct=40,
                aggregate_score=50,
            )
        ]
    )
    row = summary["aggregate_evaluation_rows"][0]
    final = report_module.build_final_assessment([summary])

    assert row["evaluation_bucket"] == "aggregate_metric_watchlist_observation"
    assert final["watchlist_observation_count"] == 1
    assert final["recommended_next_stage"] == "continue_aggregate_observation"


def test_weak_bucket_classification() -> None:
    summary = _summary(
        [
            _attachment_row(
                aggregate_sample_count=30,
                aggregate_labeled_count=30,
                aggregate_median_future_return_pct=0.01,
                aggregate_positive_rate_pct=39.9,
                aggregate_score=49.9,
            )
        ]
    )
    row = summary["aggregate_evaluation_rows"][0]
    final = report_module.build_final_assessment([summary])

    assert row["evaluation_bucket"] == "aggregate_metric_weak_observation"
    assert final["weak_observation_count"] == 1
    assert final["recommended_next_stage"] == "collect_more_data"


def test_missing_aggregate_metric_fields_become_unavailable() -> None:
    summary = _summary([_attachment_row(aggregate_score=None)])
    row = summary["aggregate_evaluation_rows"][0]
    final = report_module.build_final_assessment([summary])

    assert row["evaluation_bucket"] == "aggregate_metric_unavailable"
    assert "aggregate_score" in row["evaluation_reason"]
    assert final["unavailable_count"] == 1
    assert final["recommended_next_stage"] == "collect_more_data"


def test_labeled_count_greater_than_sample_count_becomes_unavailable() -> None:
    summary = _summary(
        [
            _attachment_row(
                aggregate_sample_count=30,
                aggregate_labeled_count=31,
                aggregate_median_future_return_pct=0.01,
                aggregate_positive_rate_pct=45,
                aggregate_score=60,
            )
        ]
    )
    row = summary["aggregate_evaluation_rows"][0]
    final = report_module.build_final_assessment([summary])

    assert row["evaluation_bucket"] == "aggregate_metric_unavailable"
    assert "aggregate_labeled_count:greater_than_sample_count" in row[
        "aggregate_metric_validation_errors"
    ]
    assert final["unavailable_count"] == 1


def test_non_live_no_mapper_no_engine_no_order_no_fill_no_pnl_invariants_hold() -> None:
    summary = _summary([_attachment_row()])
    row = summary["aggregate_evaluation_rows"][0]

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
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

    invariants = summary["aggregate_metric_evaluation_safety_invariants"]
    assert (
        invariants["all_evaluation_rows_production_live_selection_disallowed"]
        is True
    )
    assert invariants["all_evaluation_rows_mapper_live_path_disallowed"] is True
    assert invariants["all_evaluation_rows_engine_live_path_disallowed"] is True
    assert invariants["all_evaluation_rows_no_order_execution"] is True
    assert invariants["all_evaluation_rows_no_synthetic_fill"] is True
    assert invariants["all_evaluation_rows_no_pnl_claim"] is True
    assert invariants["no_order_or_fill_identifiers_present"] is True
    assert invariants["no_price_or_pnl_fields_present"] is True
    assert invariants["exact_outcomes_not_used"] is True
    assert invariants["aggregate_metrics_remain_aggregate_only"] is True


def test_deterministic_and_unique_ids_with_duplicate_source_rows() -> None:
    rows = [
        _attachment_row(aggregate_score=63.6),
        _attachment_row(aggregate_score=62.6),
    ]
    summary_one = _summary(rows)
    summary_two = _summary(rows)

    ids_one = [
        row["aggregate_metric_evaluation_id"]
        for row in summary_one["aggregate_evaluation_rows"]
    ]
    ids_two = [
        row["aggregate_metric_evaluation_id"]
        for row in summary_two["aggregate_evaluation_rows"]
    ]
    expected_base_id = (
        "aggregate_metric_paper_replay_evaluation_v1:"
        "paper_replay_outcome_attachment_source_v1:"
        "paper_replay_outcome_tracking_v1:"
        "paper_replay_observation_journal_v1:"
        "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    )
    assert ids_one == ids_two
    assert ids_one == [expected_base_id, f"{expected_base_id}:dup_2"]
    assert len(ids_one) == len(set(ids_one))
    assert summary_one["aggregate_metric_evaluation_safety_invariants"][
        "aggregate_metric_evaluation_ids_are_unique"
    ] is True


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert wrapper.attachment_source_report is attachment_report
    assert (
        wrapper.run_selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report
        is report_module.run_selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report
    )
