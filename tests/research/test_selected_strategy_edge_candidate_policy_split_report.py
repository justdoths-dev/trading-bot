from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import selected_strategy_edge_candidate_policy_split_report as wrapper
from src.research.diagnostics import (
    selected_strategy_edge_candidate_policy_split_report as report_module,
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
        "symbol": "BTCUSDT",
        "strategy": "intraday",
        "horizon": "1h",
        "diagnostic_category": "quality_rejected",
        "rejection_reason": "candidate_strength_weak",
        "rejection_reasons": ["candidate_strength_weak"],
        "candidate_strength": "weak",
        "sample_count": 81,
        "labeled_count": 80,
        "median_future_return_pct": 0.080669,
        "positive_rate_pct": 47.5,
        "robustness_signal": "up_rate_pct",
        "robustness_signal_pct": 41.25,
        "aggregate_score": 62.75,
    }
    row.update(overrides)
    return row


def _summary_for(rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return report_module.build_configuration_summary(
        configuration=report_module.SupportWindowConfiguration(336, 10000),
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


def test_eligible_rows_are_only_production_live_allowed() -> None:
    summary = _summary_for([_eligible_row()], [_diagnostic_row()])
    live_rows = [
        row
        for row in summary["policy_rows"]
        if row["production_live_selection_allowed"] is True
    ]

    assert summary["eligible_production_candidate_count"] == 1
    assert [row["policy_class"] for row in live_rows] == [
        "production_eligible_candidate"
    ]
    assert all(
        row["source_row_type"] == "eligible_edge_candidate_row" for row in live_rows
    )
    assert summary["safety_invariants"][
        "diagnostic_rows_live_selection_allowed"
    ] is True


def test_paper_only_near_miss_is_non_live_and_paper_replay_allowed() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(suggested_next_policy_bucket="paper_only_candidate_review")
    )

    assert row["policy_class"] == "paper_only_candidate"
    assert row["production_live_selection_allowed"] is False
    assert row["paper_replay_allowed"] is True


def test_pre_tagged_paper_only_with_non_positive_reason_is_hard_block() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            rejection_reason="median_future_return_non_positive",
            rejection_reasons=["median_future_return_non_positive"],
            suggested_next_policy_bucket="paper_only_candidate_review",
        )
    )

    assert row["policy_class"] == "hard_block"
    assert row["production_live_selection_allowed"] is False
    assert row["paper_replay_allowed"] is False
    assert any("non-positive median return" in note for note in row["safety_notes"])


def test_pre_tagged_paper_only_with_non_positive_return_value_is_hard_block() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            median_future_return_pct=0,
            suggested_next_policy_bucket="paper_only_candidate_review",
        )
    )

    assert row["policy_class"] == "hard_block"
    assert row["production_live_selection_allowed"] is False
    assert row["paper_replay_allowed"] is False
    assert any("non-positive median return" in note for note in row["safety_notes"])


def test_pre_tagged_paper_only_with_sample_floor_reason_collects_more_data() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            rejection_reason="sample_count_below_absolute_floor",
            rejection_reasons=["sample_count_below_absolute_floor"],
            sample_count=12,
            labeled_count=12,
            suggested_next_policy_bucket="paper_only_candidate_review",
        )
    )

    assert row["policy_class"] == "collect_more_data"
    assert row["production_live_selection_allowed"] is False
    assert row["paper_replay_allowed"] is False
    assert any("sample-limited diagnostic row" in note for note in row["safety_notes"])


def test_human_review_near_miss_is_non_live_and_human_review_allowed() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            suggested_next_policy_bucket="human_review_candidate_review",
            positive_rate_pct=35.0,
        )
    )

    assert row["policy_class"] == "human_review_candidate"
    assert row["production_live_selection_allowed"] is False
    assert row["human_review_allowed"] is True


def test_sample_limited_rows_collect_more_data() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            diagnostic_category="insufficient_data",
            rejection_reason="failed_absolute_minimum_gate",
            rejection_reasons=[
                "failed_absolute_minimum_gate",
                "sample_count_below_absolute_floor",
            ],
            sample_count=12,
            labeled_count=12,
            aggregate_score=None,
            suggested_next_policy_bucket="collect_more_data",
        )
    )

    assert row["policy_class"] == "collect_more_data"
    assert row["production_live_selection_allowed"] is False
    assert row["paper_replay_allowed"] is False


def test_incompatible_and_negative_return_rows_hard_block() -> None:
    incompatible = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            diagnostic_category="incompatibility",
            rejection_reason="strategy_horizon_incompatible",
            rejection_reasons=["strategy_horizon_incompatible"],
            suggested_next_policy_bucket="hard_block",
        )
    )
    negative = report_module.build_diagnostic_policy_row(
        _diagnostic_row(
            rejection_reason="median_future_return_non_positive",
            rejection_reasons=["median_future_return_non_positive"],
            median_future_return_pct=-0.03,
            suggested_next_policy_bucket="hard_block",
        )
    )

    assert incompatible["policy_class"] == "hard_block"
    assert negative["policy_class"] == "hard_block"
    assert incompatible["production_live_selection_allowed"] is False
    assert negative["production_live_selection_allowed"] is False


def test_unknown_bucket_never_becomes_production_live_allowed() -> None:
    row = report_module.build_diagnostic_policy_row(
        _diagnostic_row(suggested_next_policy_bucket="new_future_bucket")
    )

    assert row["policy_class"] == "hard_block"
    assert row["production_live_selection_allowed"] is False


def test_final_assessment_recommends_paper_only_contract_when_present() -> None:
    summary = _summary_for(
        [],
        [
            _diagnostic_row(
                suggested_next_policy_bucket="paper_only_candidate_review"
            )
        ],
    )
    final = report_module.build_final_assessment([summary])

    assert final["paper_only_candidates_present"] is True
    assert final["policy_split_supported"] is True
    assert final["recommended_next_stage"] == "design_paper_only_replay_contract"


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_policy_split_report
        is report_module.run_selected_strategy_edge_candidate_policy_split_report
    )
