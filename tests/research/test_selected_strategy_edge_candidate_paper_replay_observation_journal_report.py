from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_replay_observation_journal_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as replay_contract_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_observation_journal_report as report_module,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_policy_split_report as policy_split_report,
)


def _eligible_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "strategy": "intraday",
        "horizon": "1h",
        "selected_candidate_strength": "strong",
        "sample_count": 120,
        "labeled_count": 118,
        "median_future_return_pct": 0.24,
        "positive_rate_pct": 57.0,
        "robustness_signal": "up_rate_pct",
        "robustness_signal_pct": 53.0,
        "aggregate_score": 78.0,
    }
    row.update(overrides)
    return row


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
    }
    row.update(overrides)
    return row


def _source_policy_summary(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return policy_split_report.build_configuration_summary(
        configuration=policy_split_report.SupportWindowConfiguration(336, 10000),
        input_path=Path("/tmp/input.jsonl"),
        analyzer_output_dir=Path("/tmp/report"),
        analyzer_metrics={
            "edge_candidate_rows": {
                "row_count": len(rows),
                "rows": rows,
                "diagnostic_row_count": len(diagnostics),
                "diagnostic_rows": diagnostics,
                "empty_reason_summary": {},
            }
        },
        source_metadata={"raw_record_count": 1, "windowed_record_count": 1},
        raw_record_count=1,
        labelable_count=1,
    )


def _replay_contract_summary(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return replay_contract_report.build_configuration_summary(
        source_policy_split_summary=_source_policy_summary(rows, diagnostics)
    )


def _journal_summary(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_replay_contract_summary=_replay_contract_summary(rows, diagnostics)
    )


def test_paper_replay_rows_become_observation_journal_rows() -> None:
    summary = _journal_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )

    assert summary["journal_entry_count"] == 1
    row = summary["journal_rows"][0]
    assert row["journal_contract_version"] == "paper_replay_observation_journal_v1"
    assert row["replay_contract_version"] == "paper_only_replay_v1"
    assert row["replay_mode"] == "observation_only"
    assert (
        row["source_report_type"]
        == replay_contract_report.REPORT_TYPE
    )
    assert row["source_policy_class"] == "paper_only_candidate"
    assert row["observation_status"] == "open_observation"
    assert (
        row["observation_lifecycle_state"]
        == "created_from_report_only_contract"
    )
    assert row["symbol"] == "ETHUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["sample_count"] == 46
    assert row["labeled_count"] == 46
    assert row["median_future_return_pct"] == 0.174932
    assert row["positive_rate_pct"] == 52.17
    assert row["robustness_signal"] == "up_rate_pct"
    assert row["robustness_signal_pct"] == 52.17
    assert row["aggregate_score"] == 63.6
    assert row["source_rejection_reason"] == "candidate_strength_weak"
    assert row["near_miss_classification"] == "quality_weak_near_miss"
    assert row["suggested_next_policy_bucket"] == "paper_only_candidate_review"


def test_journal_rows_are_non_live_and_cannot_enter_mapper_or_engine_path() -> None:
    summary = _journal_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    row = summary["journal_rows"][0]

    assert row["production_live_selection_allowed"] is False
    assert row["mapper_live_path_allowed"] is False
    assert row["engine_live_path_allowed"] is False
    invariants = summary["journal_safety_invariants"]
    assert (
        invariants["all_journal_rows_production_live_selection_disallowed"]
        is True
    )
    assert invariants["all_journal_rows_mapper_live_path_disallowed"] is True
    assert invariants["all_journal_rows_engine_live_path_disallowed"] is True


def test_no_order_fill_pnl_or_price_fields_are_populated() -> None:
    summary = _journal_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    row = summary["journal_rows"][0]

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert row["order_id"] is None
    assert row["fill_id"] is None
    assert row["entry_price"] is None
    assert row["exit_price"] is None
    assert row["realized_pnl"] is None
    assert row["unrealized_pnl"] is None
    invariants = summary["journal_safety_invariants"]
    assert invariants["all_journal_rows_no_order_execution"] is True
    assert invariants["all_journal_rows_no_synthetic_fill"] is True
    assert invariants["all_journal_rows_no_pnl_claim"] is True
    assert invariants["no_order_or_fill_identifiers_present"] is True
    assert invariants["no_price_or_pnl_fields_present"] is True


def test_journal_entry_ids_are_deterministic_and_unique() -> None:
    source_summary = _replay_contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review",
                aggregate_score=63.6,
            ),
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review",
                aggregate_score=62.6,
            ),
        ],
    )
    summary_one = report_module.build_configuration_summary(
        source_replay_contract_summary=source_summary
    )
    summary_two = report_module.build_configuration_summary(
        source_replay_contract_summary=source_summary
    )

    ids_one = [row["journal_entry_id"] for row in summary_one["journal_rows"]]
    ids_two = [row["journal_entry_id"] for row in summary_two["journal_rows"]]
    expected_base_id = (
        "paper_replay_observation_journal_v1:"
        "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    )
    assert ids_one == ids_two
    assert ids_one == [expected_base_id, f"{expected_base_id}:dup_2"]
    assert len(ids_one) == len(set(ids_one))
    assert summary_one["journal_safety_invariants"][
        "journal_entry_ids_are_unique"
    ] is True


def test_non_paper_rows_do_not_become_journal_rows() -> None:
    source_summary = _replay_contract_summary(
        [_eligible_row()],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    source_summary["paper_replay_rows"].extend(
        [
            {
                "paper_replay_candidate_id": "hard-block-id",
                "source_policy_class": "hard_block",
                "symbol": "BNBUSDT",
            },
            {
                "paper_replay_candidate_id": "human-review-id",
                "source_policy_class": "human_review_candidate",
                "symbol": "SOLUSDT",
            },
        ]
    )

    summary = report_module.build_configuration_summary(
        source_replay_contract_summary=source_summary
    )

    assert [row["symbol"] for row in summary["journal_rows"]] == ["ETHUSDT"]
    assert summary["journal_safety_invariants"][
        "journal_rows_sourced_only_from_paper_replay_candidates"
    ] is True


def test_final_assessment_recommends_outcome_tracking_when_journal_rows_exist() -> None:
    summary = _journal_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    final = report_module.build_final_assessment([summary])

    assert final["journal_entries_present"] is True
    assert final["journal_contract_supported"] is True
    assert final["journal_entry_count"] == 1
    assert (
        final["recommended_next_stage"]
        == "design_paper_replay_outcome_tracking_contract"
    )


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_replay_observation_journal_report
        is report_module.run_selected_strategy_edge_candidate_paper_replay_observation_journal_report
    )
