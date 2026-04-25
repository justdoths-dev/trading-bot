from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as report_module,
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


def _contract_summary(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        source_policy_split_summary=_source_policy_summary(rows, diagnostics)
    )


def test_paper_only_candidate_policy_rows_become_paper_replay_rows() -> None:
    summary = _contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )

    assert summary["paper_replay_candidate_count"] == 1
    row = summary["paper_replay_rows"][0]
    assert row["source_policy_class"] == "paper_only_candidate"
    assert row["source_row_type"] == "rejected_diagnostic_near_miss_row"
    assert row["replay_mode"] == "observation_only"
    assert row["paper_replay_allowed"] is True
    assert row["symbol"] == "ETHUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["sample_count"] == 46
    assert row["labeled_count"] == 46
    assert row["median_future_return_pct"] == 0.174932
    assert row["positive_rate_pct"] == 52.17
    assert row["robustness_signal_pct"] == 52.17
    assert row["aggregate_score"] == 63.6


def test_production_live_selection_is_always_false_for_paper_replay_rows() -> None:
    summary = _contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )

    assert all(
        row["production_live_selection_allowed"] is False
        for row in summary["paper_replay_rows"]
    )
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_production_live_selection_disallowed"
    ] is True


def test_mapper_and_engine_live_paths_are_always_false() -> None:
    summary = _contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )

    assert all(
        row["mapper_live_path_allowed"] is False
        and row["engine_live_path_allowed"] is False
        for row in summary["paper_replay_rows"]
    )
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_mapper_live_path_disallowed"
    ] is True
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_engine_live_path_disallowed"
    ] is True


def test_non_paper_policy_rows_do_not_become_paper_replay_rows() -> None:
    summary = _contract_summary(
        [_eligible_row()],
        [
            _diagnostic_row(
                symbol="BNBUSDT",
                suggested_next_policy_bucket="hard_block",
                rejection_reason="median_future_return_non_positive",
                rejection_reasons=["median_future_return_non_positive"],
                median_future_return_pct=-0.1,
            ),
            _diagnostic_row(
                symbol="ADAUSDT",
                suggested_next_policy_bucket="collect_more_data",
                rejection_reason="sample_count_below_absolute_floor",
                rejection_reasons=["sample_count_below_absolute_floor"],
                sample_count=12,
                labeled_count=12,
            ),
            _diagnostic_row(
                symbol="SOLUSDT",
                suggested_next_policy_bucket="human_review_candidate_review",
                positive_rate_pct=35.0,
            ),
            _diagnostic_row(
                symbol="ETHUSDT",
                suggested_next_policy_bucket="paper_only_candidate_review",
            ),
        ],
    )

    assert [row["symbol"] for row in summary["paper_replay_rows"]] == ["ETHUSDT"]
    invariants = summary["replay_safety_invariants"]
    assert invariants["no_hard_block_rows_in_paper_replay_rows"] is True
    assert invariants["no_collect_more_data_rows_in_paper_replay_rows"] is True
    assert invariants["no_human_review_candidate_rows_in_paper_replay_rows"] is True
    assert (
        invariants["paper_replay_rows_sourced_only_from_paper_only_candidate_policy"]
        is True
    )


def test_paper_replay_candidate_id_is_deterministic_and_stable() -> None:
    source_summary = _source_policy_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    summary_one = report_module.build_configuration_summary(
        source_policy_split_summary=source_summary
    )
    summary_two = report_module.build_configuration_summary(
        source_policy_split_summary=source_summary
    )

    first_id = summary_one["paper_replay_rows"][0]["paper_replay_candidate_id"]
    second_id = summary_two["paper_replay_rows"][0]["paper_replay_candidate_id"]
    base_id = summary_one["paper_replay_rows"][0]["paper_replay_candidate_base_id"]
    assert first_id == second_id
    assert base_id == first_id
    assert (
        first_id
        == "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    )


def test_duplicate_paper_only_source_rows_receive_unique_candidate_ids() -> None:
    summary = _contract_summary(
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
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review",
                aggregate_score=61.6,
            ),
        ],
    )

    base_id = "paper_only_replay_v1:336h_10000:ethusdt:swing:4h:paper_only_candidate"
    ids = [row["paper_replay_candidate_id"] for row in summary["paper_replay_rows"]]
    base_ids = [
        row["paper_replay_candidate_base_id"] for row in summary["paper_replay_rows"]
    ]

    assert ids == [base_id, f"{base_id}:dup_2", f"{base_id}:dup_3"]
    assert base_ids == [base_id, base_id, base_id]
    assert len(ids) == len(set(ids))
    assert summary["replay_safety_invariants"][
        "paper_replay_candidate_ids_are_unique"
    ] is True


def test_final_assessment_recommends_observation_journal_when_paper_rows_exist() -> None:
    summary = _contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    final = report_module.build_final_assessment([summary])

    assert final["paper_replay_candidates_present"] is True
    assert final["replay_contract_supported"] is True
    assert final["recommended_next_stage"] == "design_paper_replay_observation_journal"


def test_no_order_fill_or_pnl_claim_is_allowed() -> None:
    summary = _contract_summary(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    row = summary["paper_replay_rows"][0]

    assert row["no_order_execution"] is True
    assert row["no_synthetic_fill"] is True
    assert row["no_pnl_claim"] is True
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_no_order_execution"
    ] is True
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_no_synthetic_fill"
    ] is True
    assert summary["replay_safety_invariants"][
        "all_paper_replay_rows_no_pnl_claim"
    ] is True


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_paper_only_replay_contract_report
        is report_module.run_selected_strategy_edge_candidate_paper_only_replay_contract_report
    )
