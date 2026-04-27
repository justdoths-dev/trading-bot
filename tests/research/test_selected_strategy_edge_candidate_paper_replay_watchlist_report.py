from __future__ import annotations

from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_replay_watchlist_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report as aggregate_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_watchlist_report as report_module,
)


def _attachment_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outcome_attachment_source_id": (
            "paper_replay_outcome_attachment_source_v1:"
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_outcome_tracking_id": (
            "paper_replay_outcome_tracking_v1:"
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_journal_entry_id": (
            "paper_replay_observation_journal_v1:"
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "source_paper_replay_candidate_id": (
            "paper_only_replay_v1:336h_10000:btcusdt:intraday:1h:paper_only_candidate"
        ),
        "symbol": "BTCUSDT",
        "strategy": "intraday",
        "horizon": "1h",
        "source_policy_class": "paper_only_candidate",
        "attachment_source_status": "aggregate_diagnostic_metrics_only",
        "aggregate_metric_attachment_available": True,
        "aggregate_sample_count": 80,
        "aggregate_labeled_count": 80,
        "aggregate_median_future_return_pct": 0.084126,
        "aggregate_positive_rate_pct": 48.75,
        "aggregate_robustness_signal": "up_rate_pct",
        "aggregate_robustness_signal_pct": 41.25,
        "aggregate_score": 62.75,
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


def _source_attachment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_report.build_configuration_summary(
        source_attachment_summary=_source_attachment_summary(rows)
    )


def _watchlist_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_aggregate_evaluation_summary=_aggregate_summary(rows)
    )


def test_promising_aggregate_row_becomes_high_priority_promising_watchlist_row() -> None:
    summary = _watchlist_summary([_attachment_row()])

    assert summary["watchlist_row_count"] == 1
    assert summary["watchlist_source_row_count"] == 1
    assert summary["malformed_watchlist_source_row_count"] == 0

    row = summary["watchlist_rows"][0]
    assert row["paper_replay_watchlist_id"] == (
        "paper_replay_watchlist_v1:"
        + row["aggregate_metric_evaluation_id"]
    )
    assert row["watchlist_contract_version"] == "paper_replay_watchlist_v1"
    assert row["watchlist_mode"] == "aggregate_metric_observation_watchlist"
    assert row["evaluation_bucket"] == "aggregate_metric_promising_observation"
    assert row["watchlist_tier"] == "paper_replay_promising_watchlist"
    assert row["watchlist_priority"] == "high"
    assert row["symbol"] == "BTCUSDT"
    assert row["strategy"] == "intraday"
    assert row["horizon"] == "1h"
    assert row["aggregate_sample_count"] == 80
    assert row["aggregate_labeled_count"] == 80
    assert row["aggregate_median_future_return_pct"] == 0.084126
    assert row["aggregate_positive_rate_pct"] == 48.75
    assert row["aggregate_robustness_signal"] == "up_rate_pct"
    assert row["aggregate_robustness_signal_pct"] == 41.25
    assert row["aggregate_score"] == 62.75
    assert row["source_outcome_attachment_source_id"].startswith(
        "paper_replay_outcome_attachment_source_v1:"
    )
    assert row["source_outcome_tracking_id"].startswith(
        "paper_replay_outcome_tracking_v1:"
    )
    assert row["source_journal_entry_id"].startswith(
        "paper_replay_observation_journal_v1:"
    )
    assert row["source_paper_replay_candidate_id"].startswith(
        "paper_only_replay_v1:"
    )


def test_watchlist_aggregate_row_becomes_medium_priority_standard_watchlist_row() -> None:
    summary = _watchlist_summary(
        [
            _attachment_row(
                aggregate_positive_rate_pct=40,
                aggregate_robustness_signal_pct=40,
                aggregate_score=50,
            )
        ]
    )

    row = summary["watchlist_rows"][0]
    assert row["evaluation_bucket"] == "aggregate_metric_watchlist_observation"
    assert row["watchlist_tier"] == "paper_replay_standard_watchlist"
    assert row["watchlist_priority"] == "medium"
    assert summary["promising_watchlist_count"] == 0
    assert summary["standard_watchlist_count"] == 1


def test_weak_and_unavailable_rows_do_not_become_watchlist_rows() -> None:
    summary = _watchlist_summary(
        [
            _attachment_row(
                aggregate_positive_rate_pct=39.9,
                aggregate_robustness_signal_pct=39.9,
                aggregate_score=49.9,
            ),
            _attachment_row(
                outcome_attachment_source_id="unavailable-source",
                aggregate_score=None,
            ),
        ]
    )

    assert summary["watchlist_row_count"] == 0
    assert summary["watchlist_rows"] == []
    final = report_module.build_final_assessment([summary])
    assert final["watchlist_rows_present"] is False
    assert final["recommended_next_stage"] == "continue_aggregate_observation"


def test_watchlist_rows_are_non_live_and_cannot_enter_mapper_or_engine_paths() -> None:
    summary = _watchlist_summary([_attachment_row()])
    row = summary["watchlist_rows"][0]

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    assert row["exact_outcome_used"] is False
    assert row["aggregate_metric_only"] is True
    invariants = summary["watchlist_safety_invariants"]
    assert (
        invariants["all_watchlist_rows_production_live_selection_disallowed"]
        is True
    )
    assert invariants["all_watchlist_rows_mapper_live_path_disallowed"] is True
    assert invariants["all_watchlist_rows_engine_live_path_disallowed"] is True
    assert invariants["exact_outcomes_not_used"] is True
    assert invariants["aggregate_metrics_remain_aggregate_only"] is True


def test_no_order_fill_price_or_pnl_fields_are_populated() -> None:
    summary = _watchlist_summary([_attachment_row()])
    row = summary["watchlist_rows"][0]

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert row["order_id"] is None
    assert row["fill_id"] is None
    assert row["entry_price"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["unrealized_pnl"] is None
    invariants = summary["watchlist_safety_invariants"]
    assert invariants["all_watchlist_rows_no_order_execution"] is True
    assert invariants["all_watchlist_rows_no_synthetic_fill"] is True
    assert invariants["all_watchlist_rows_no_pnl_claim"] is True
    assert invariants["no_order_or_fill_identifiers_present"] is True
    assert invariants["no_price_or_pnl_fields_present"] is True


def test_required_source_ids_are_preserved_and_non_empty() -> None:
    summary = _watchlist_summary([_attachment_row()])
    row = summary["watchlist_rows"][0]

    assert row["aggregate_metric_evaluation_id"]
    assert row["source_outcome_attachment_source_id"]
    assert row["source_outcome_tracking_id"]
    assert row["source_journal_entry_id"]
    assert row["source_paper_replay_candidate_id"]

    invariants = summary["watchlist_safety_invariants"]
    assert (
        invariants["all_watchlist_rows_have_non_empty_aggregate_metric_evaluation_id"]
        is True
    )
    assert invariants["all_watchlist_rows_have_required_source_ids"] is True


def test_watchlist_ids_are_deterministic_and_unique_with_duplicate_source_rows() -> None:
    aggregate_row = aggregate_report.build_aggregate_metric_evaluation_row(
        attachment_source_row=_attachment_row()
    )
    source_summary = {
        "configuration": {
            "window_hours": 336,
            "max_rows": 10000,
            "display_name": "336h_10000",
        },
        "aggregate_evaluation_row_count": 2,
        "paper_replay_candidate_count": 2,
        "production_candidate_count": 0,
        "aggregate_evaluation_rows": [aggregate_row, dict(aggregate_row)],
    }

    summary_one = report_module.build_configuration_summary(
        source_aggregate_evaluation_summary=source_summary
    )
    summary_two = report_module.build_configuration_summary(
        source_aggregate_evaluation_summary=source_summary
    )
    ids_one = [
        row["paper_replay_watchlist_id"] for row in summary_one["watchlist_rows"]
    ]
    ids_two = [
        row["paper_replay_watchlist_id"] for row in summary_two["watchlist_rows"]
    ]
    expected_base_id = (
        "paper_replay_watchlist_v1:"
        + aggregate_row["aggregate_metric_evaluation_id"]
    )

    assert ids_one == ids_two
    assert ids_one == [expected_base_id, f"{expected_base_id}:dup_2"]
    assert len(ids_one) == len(set(ids_one))
    assert summary_one["watchlist_safety_invariants"]["watchlist_ids_are_unique"] is True


def test_malformed_watchlist_source_row_without_aggregate_id_is_not_converted() -> None:
    aggregate_row = aggregate_report.build_aggregate_metric_evaluation_row(
        attachment_source_row=_attachment_row()
    )
    malformed_row = dict(aggregate_row)
    malformed_row["aggregate_metric_evaluation_id"] = ""

    source_summary = {
        "configuration": {
            "window_hours": 336,
            "max_rows": 10000,
            "display_name": "336h_10000",
        },
        "aggregate_evaluation_row_count": 1,
        "paper_replay_candidate_count": 1,
        "production_candidate_count": 0,
        "aggregate_evaluation_rows": [malformed_row],
    }

    summary = report_module.build_configuration_summary(
        source_aggregate_evaluation_summary=source_summary
    )

    assert summary["watchlist_source_row_count"] == 1
    assert summary["malformed_watchlist_source_row_count"] == 1
    assert summary["watchlist_row_count"] == 0
    assert summary["watchlist_rows"] == []


def test_final_assessment_recommends_retention_report_when_rows_exist() -> None:
    summary = _watchlist_summary([_attachment_row()])
    final = report_module.build_final_assessment([summary])

    assert final["watchlist_rows_present"] is True
    assert final["watchlist_row_count"] == 1
    assert final["promising_watchlist_count"] == 1
    assert final["standard_watchlist_count"] == 0
    assert final["production_candidate_count"] == 0
    assert (
        final["recommended_next_stage"]
        == "design_watchlist_observation_retention_report"
    )
    assert final["candidate_counts"]["watchlist_row"] == 1
    assert final["watchlist_safety_invariant_summary"][
        "watchlist_rows_sourced_only_from_promising_or_watchlist_aggregate_observations"
    ] is True
    assert final["watchlist_safety_invariant_summary"][
        "all_watchlist_rows_have_required_source_ids"
    ] is True


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_replay_watchlist_report
        is report_module.run_selected_strategy_edge_candidate_paper_replay_watchlist_report
    )