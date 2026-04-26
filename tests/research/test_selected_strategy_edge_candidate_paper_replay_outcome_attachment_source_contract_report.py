from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as replay_contract_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_observation_journal_report as journal_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report as report_module,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report as outcome_tracking_report,
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
    return outcome_tracking_report.build_configuration_summary(
        source_journal_summary=_journal_summary(diagnostics)
    )


def _attachment_summary(
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_outcome_tracking_summary=_outcome_summary(diagnostics)
    )


def test_outcome_tracking_rows_become_attachment_source_rows() -> None:
    summary = _attachment_summary([_diagnostic_row()])

    assert summary["attachment_source_row_count"] == 1
    row = summary["attachment_source_rows"][0]
    assert (
        row["outcome_attachment_source_contract_version"]
        == "paper_replay_outcome_attachment_source_v1"
    )
    assert row["attachment_mode"] == "attachment_source_contract_only"
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


def test_exact_future_label_return_attach_only_from_matching_horizon_fields() -> None:
    source_summary = _outcome_summary([_diagnostic_row(horizon="1h")])
    source_summary["source_journal_summary"]["journal_rows"][0].update(
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
        source_outcome_tracking_summary=source_summary
    )
    row = summary["attachment_source_rows"][0]

    assert (
        row["attachment_source_status"]
        == "exact_horizon_future_label_return_available"
    )
    assert row["exact_outcome_attachment_available"] is True
    assert row["exact_attached_future_label"] == "up"
    assert row["exact_attached_future_return_pct"] == 0.7
    assert row["aggregate_metric_attachment_available"] is False
    assert row["exact_source_label_field"] == "future_label_1h"
    assert row["exact_source_return_field"] == "future_return_1h"


def test_missing_exact_future_label_return_does_not_synthesize_exact_outcome() -> None:
    summary = _attachment_summary([_diagnostic_row()])
    row = summary["attachment_source_rows"][0]

    assert row["exact_outcome_attachment_available"] is False
    assert row["exact_attached_future_label"] is None
    assert row["exact_attached_future_return_pct"] is None
    assert row["exact_source_fields_present"] is False
    assert summary["attachment_source_safety_invariants"][
        "exact_outcomes_not_synthesized"
    ] is True


def test_aggregate_diagnostic_metrics_attach_as_aggregate_only_evidence() -> None:
    summary = _attachment_summary([_diagnostic_row()])
    row = summary["attachment_source_rows"][0]

    assert row["attachment_source_status"] == "aggregate_diagnostic_metrics_only"
    assert row["aggregate_metric_attachment_available"] is True
    assert row["aggregate_sample_count"] == 46
    assert row["aggregate_labeled_count"] == 46
    assert row["aggregate_median_future_return_pct"] == 0.174932
    assert row["aggregate_positive_rate_pct"] == 52.17
    assert row["aggregate_robustness_signal"] == "up_rate_pct"
    assert row["aggregate_robustness_signal_pct"] == 52.17
    assert row["aggregate_score"] == 63.6

    final = report_module.build_final_assessment([summary])
    assert final["attachment_source_rows_present"] is True
    assert final["attachment_source_row_count"] == 1
    assert final["exact_outcome_attachment_available_count"] == 0
    assert final["exact_outcome_attachment_unavailable_count"] == 1
    assert final["aggregate_metric_attachment_available_count"] == 1
    assert final["aggregate_metric_only_count"] == 1
    assert final["outcome_source_unavailable_count"] == 0
    assert (
        final["recommended_next_stage"]
        == "design_aggregate_metric_paper_replay_evaluation_report"
    )


def test_aggregate_metrics_are_not_marked_as_exact_outcomes() -> None:
    summary = _attachment_summary([_diagnostic_row()])
    row = summary["attachment_source_rows"][0]

    assert row["attachment_source_status"] == "aggregate_diagnostic_metrics_only"
    assert row["aggregate_metric_attachment_available"] is True
    assert row["exact_outcome_attachment_available"] is False
    assert row["exact_attached_future_label"] is None
    assert row["exact_attached_future_return_pct"] is None
    assert summary["attachment_source_safety_invariants"][
        "aggregate_metrics_not_marked_as_exact_outcomes"
    ] is True


def test_non_live_no_order_no_fill_no_price_no_pnl_invariants_hold() -> None:
    summary = _attachment_summary([_diagnostic_row()])
    row = summary["attachment_source_rows"][0]

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

    invariants = summary["attachment_source_safety_invariants"]
    assert (
        invariants["all_attachment_rows_production_live_selection_disallowed"]
        is True
    )
    assert invariants["all_attachment_rows_mapper_live_path_disallowed"] is True
    assert invariants["all_attachment_rows_engine_live_path_disallowed"] is True
    assert invariants["all_attachment_rows_no_order_execution"] is True
    assert invariants["all_attachment_rows_no_synthetic_fill"] is True
    assert invariants["all_attachment_rows_no_pnl_claim"] is True
    assert invariants["no_order_or_fill_identifiers_present"] is True
    assert invariants["no_price_or_pnl_fields_present"] is True


def test_attachment_source_ids_are_deterministic_and_unique() -> None:
    source_summary = _outcome_summary(
        [
            _diagnostic_row(aggregate_score=63.6),
            _diagnostic_row(aggregate_score=62.6),
        ]
    )
    summary_one = report_module.build_configuration_summary(
        source_outcome_tracking_summary=source_summary
    )
    summary_two = report_module.build_configuration_summary(
        source_outcome_tracking_summary=source_summary
    )

    ids_one = [
        row["outcome_attachment_source_id"]
        for row in summary_one["attachment_source_rows"]
    ]
    ids_two = [
        row["outcome_attachment_source_id"]
        for row in summary_two["attachment_source_rows"]
    ]
    expected_base_id = (
        "paper_replay_outcome_attachment_source_v1:"
        "paper_replay_outcome_tracking_v1:"
        "paper_replay_observation_journal_v1:"
        "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    )
    assert ids_one == ids_two
    assert ids_one == [expected_base_id, f"{expected_base_id}:dup_2"]
    assert len(ids_one) == len(set(ids_one))
    assert summary_one["attachment_source_safety_invariants"][
        "attachment_source_ids_are_unique"
    ] is True


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report
        is report_module.run_selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report
    )
