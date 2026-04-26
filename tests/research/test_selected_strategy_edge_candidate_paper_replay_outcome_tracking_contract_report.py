from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as replay_contract_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_observation_journal_report as journal_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report as report_module,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_policy_split_report as policy_split_report,
)


def _diagnostic_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": "ETHUSDT",
        "strategy": "swing",
        "horizon": "4h",
        "diagnostic_category": "quality_rejected",
        "rejection_reason": "candidate_strength_weak",
        "rejection_reasons": [
            "candidate_strength_weak",
            "one_supporting_deficit_but_aggregate_too_low",
        ],
        "candidate_strength": "weak",
        "sample_count": 46,
        "labeled_count": 46,
        "median_future_return_pct": 0.174932,
        "positive_rate_pct": 52.17,
        "robustness_signal": "up_rate_pct",
        "robustness_signal_pct": 52.17,
        "aggregate_score": 63.6,
        "near_miss_classification": "quality_weak_near_miss",
        "suggested_next_policy_bucket": "paper_only_candidate_review",
    }
    row.update(overrides)
    return row


def _source_policy_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return policy_split_report.build_configuration_summary(
        configuration=policy_split_report.SupportWindowConfiguration(336, 10000),
        input_path=Path("/tmp/input.jsonl"),
        analyzer_output_dir=Path("/tmp/report"),
        analyzer_metrics={
            "edge_candidate_rows": {
                "row_count": 0,
                "rows": [],
                "diagnostic_row_count": len(diagnostics),
                "diagnostic_rows": diagnostics,
                "empty_reason_summary": {},
            }
        },
        source_metadata={"raw_record_count": 1, "windowed_record_count": 1},
        raw_record_count=1,
        labelable_count=1,
    )


def _journal_summary(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    replay_summary = replay_contract_report.build_configuration_summary(
        source_policy_split_summary=_source_policy_summary(diagnostics)
    )
    return journal_report.build_configuration_summary(
        source_replay_contract_summary=replay_summary
    )


def _outcome_summary(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_journal_summary=_journal_summary(diagnostics)
    )


def test_journal_rows_become_outcome_tracking_rows() -> None:
    summary = _outcome_summary([_diagnostic_row()])

    assert summary["outcome_tracking_row_count"] == 1
    row = summary["outcome_rows"][0]
    assert (
        row["outcome_tracking_contract_version"]
        == "paper_replay_outcome_tracking_v1"
    )
    assert row["tracking_mode"] == "label_only_observation"
    assert row["source_report_type"] == journal_report.REPORT_TYPE
    assert row["source_policy_class"] == "paper_only_candidate"
    assert row["symbol"] == "ETHUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["tracked_horizon"] == "4h"
    assert row["observation_status"] == "open_observation"
    assert (
        row["observation_lifecycle_state"]
        == "created_from_report_only_contract"
    )
    assert row["source_rejection_reason"] == "candidate_strength_weak"
    assert row["source_rejection_reasons"] == [
        "candidate_strength_weak",
        "one_supporting_deficit_but_aggregate_too_low",
    ]
    assert row["near_miss_classification"] == "quality_weak_near_miss"
    assert row["suggested_next_policy_bucket"] == "paper_only_candidate_review"


def test_outcome_rows_are_non_live_and_cannot_enter_mapper_or_engine_path() -> None:
    summary = _outcome_summary([_diagnostic_row()])
    row = summary["outcome_rows"][0]

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    invariants = summary["outcome_tracking_safety_invariants"]
    assert (
        invariants["all_outcome_rows_production_live_selection_disallowed"]
        is True
    )
    assert invariants["all_outcome_rows_mapper_live_path_disallowed"] is True
    assert invariants["all_outcome_rows_engine_live_path_disallowed"] is True


def test_no_order_fill_price_or_pnl_fields_are_populated() -> None:
    summary = _outcome_summary([_diagnostic_row()])
    row = summary["outcome_rows"][0]

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert row["order_id"] is None
    assert row["fill_id"] is None
    assert row["entry_price"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["unrealized_pnl"] is None
    invariants = summary["outcome_tracking_safety_invariants"]
    assert invariants["all_outcome_rows_no_order_execution"] is True
    assert invariants["all_outcome_rows_no_synthetic_fill"] is True
    assert invariants["all_outcome_rows_no_pnl_claim"] is True
    assert invariants["no_order_or_fill_identifiers_present"] is True
    assert invariants["no_price_or_pnl_fields_present"] is True


def test_outcome_tracking_ids_are_deterministic_and_unique() -> None:
    journal_summary = _journal_summary(
        [
            _diagnostic_row(aggregate_score=63.6),
            _diagnostic_row(aggregate_score=62.6),
        ]
    )
    summary_one = report_module.build_configuration_summary(
        source_journal_summary=journal_summary
    )
    summary_two = report_module.build_configuration_summary(
        source_journal_summary=journal_summary
    )

    ids_one = [row["outcome_tracking_id"] for row in summary_one["outcome_rows"]]
    ids_two = [row["outcome_tracking_id"] for row in summary_two["outcome_rows"]]
    expected_base_id = (
        "paper_replay_outcome_tracking_v1:"
        "paper_replay_observation_journal_v1:"
        "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    )
    assert ids_one == ids_two
    assert ids_one == [expected_base_id, f"{expected_base_id}:dup_2"]
    assert len(ids_one) == len(set(ids_one))
    assert summary_one["outcome_tracking_safety_invariants"][
        "outcome_tracking_ids_are_unique"
    ] is True


def test_missing_future_labels_and_returns_are_unavailable_not_synthesized() -> None:
    summary = _outcome_summary([_diagnostic_row()])
    row = summary["outcome_rows"][0]

    assert row["tracked_future_label"] is None
    assert row["tracked_future_return_pct"] is None
    assert row["outcome_label_available"] is False
    assert row["outcome_return_available"] is False
    assert row["outcome_observation_available"] is False
    assert summary["outcome_availability_summary"] == {
        "outcome_tracking_row_count": 1,
        "outcome_observation_available_count": 0,
        "outcome_observation_unavailable_count": 1,
        "outcome_label_available_count": 0,
        "outcome_label_unavailable_count": 1,
        "outcome_return_available_count": 0,
        "outcome_return_unavailable_count": 1,
    }


def test_available_outcome_uses_matching_horizon_fields_only() -> None:
    source_summary = _journal_summary([_diagnostic_row(horizon="1h")])
    source_summary["journal_rows"][0].update(
        {
            "future_label_15m": "down",
            "future_return_15m": -0.8,
            "future_label_1h": "up",
            "future_return_1h": 0.7,
            "future_label_4h": "flat",
            "future_return_4h": 0.0,
        }
    )
    summary = report_module.build_configuration_summary(
        source_journal_summary=source_summary
    )
    row = summary["outcome_rows"][0]

    assert row["tracked_horizon"] == "1h"
    assert row["tracked_future_label"] == "up"
    assert row["tracked_future_return_pct"] == 0.7
    assert row["outcome_label_available"] is True
    assert row["outcome_return_available"] is True
    assert row["outcome_observation_available"] is True


def test_final_assessment_recommends_attachment_contract_when_outcomes_unavailable() -> None:
    summary = _outcome_summary([_diagnostic_row()])
    final = report_module.build_final_assessment([summary])

    assert final["outcome_tracking_rows_present"] is True
    assert final["outcome_tracking_contract_supported"] is True
    assert final["outcome_tracking_row_count"] == 1
    assert final["outcome_observation_available_count"] == 0
    assert final["outcome_observation_unavailable_count"] == 1
    assert (
        final["recommended_next_stage"]
        == "design_outcome_attachment_source_contract"
    )


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report
        is report_module.run_selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report
    )
