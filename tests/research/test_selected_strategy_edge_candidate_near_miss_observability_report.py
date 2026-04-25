from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import (
    selected_strategy_edge_candidate_near_miss_observability_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_near_miss_observability_report as report_module,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "horizon": "4h",
        "status": "rejected",
        "diagnostic_category": "quality_rejected",
        "rejection_reason": "candidate_strength_weak",
        "rejection_reasons": [
            "candidate_strength_weak",
            "three_supporting_deficits_but_aggregate_too_low",
        ],
        "candidate_strength": "weak",
        "sample_count": 55,
        "labeled_count": 55,
        "coverage_pct": 100.0,
        "median_future_return_pct": 0.18,
        "positive_rate_pct": 49.0,
        "robustness_signal": "up_rate_pct",
        "robustness_signal_pct": 45.0,
        "aggregate_score": 59.1,
        "chosen_metric_summary": "sample=55 median=0.18 positive_rate=49 aggregate=59.1",
        "visibility_reason": "candidate_strength_weak",
    }
    row.update(overrides)
    return row


def test_sample_limited_rows_are_collect_more_data() -> None:
    result = report_module.build_near_miss_row(
        _row(
            diagnostic_category="insufficient_data",
            rejection_reason="failed_absolute_minimum_gate",
            rejection_reasons=[
                "failed_absolute_minimum_gate",
                "sample_count_below_absolute_floor",
            ],
            candidate_strength="insufficient_data",
            sample_count=12,
            labeled_count=12,
            median_future_return_pct=0.21,
            aggregate_score=None,
        )
    )

    assert result["near_miss_classification"] == "insufficient_sample"
    assert result["suggested_next_policy_bucket"] == "collect_more_data"


def test_candidate_strength_weak_positive_usable_sample_is_quality_near_miss() -> None:
    result = report_module.build_near_miss_row(
        _row(
            rejection_reasons=["candidate_strength_weak"],
            aggregate_score=57.5,
        )
    )

    assert result["near_miss_classification"] == "quality_weak_near_miss"
    assert result["suggested_next_policy_bucket"] == "paper_only_candidate_review"


def test_negative_median_rows_are_not_promoted_to_paper_review() -> None:
    result = report_module.build_near_miss_row(
        _row(
            rejection_reason="failed_absolute_minimum_gate",
            rejection_reasons=[
                "failed_absolute_minimum_gate",
                "median_future_return_non_positive",
            ],
            candidate_strength="insufficient_data",
            sample_count=45,
            median_future_return_pct=-0.03,
            aggregate_score=None,
        )
    )

    assert result["near_miss_classification"] == "negative_return_blocked"
    assert result["suggested_next_policy_bucket"] == "hard_block"


def test_strategy_horizon_incompatible_rows_are_hard_blocked() -> None:
    result = report_module.build_near_miss_row(
        _row(
            diagnostic_category="incompatibility",
            rejection_reason="strategy_horizon_incompatible",
            rejection_reasons=["strategy_horizon_incompatible"],
            candidate_strength="incompatible",
            sample_count=0,
            labeled_count=0,
            median_future_return_pct=None,
            positive_rate_pct=None,
            aggregate_score=None,
        )
    )

    assert result["near_miss_classification"] == "hard_blocked_incompatible"
    assert result["suggested_next_policy_bucket"] == "hard_block"


def test_final_assessment_identifies_policy_split_review_for_quality_near_misses() -> None:
    summary = report_module.build_configuration_summary(
        configuration=report_module.SupportWindowConfiguration(336, 10000),
        input_path=Path("/tmp/input.jsonl"),
        analyzer_output_dir=Path("/tmp/report"),
        analyzer_metrics={
            "edge_candidate_rows": {
                "row_count": 0,
                "diagnostic_rows": [
                    _row(
                        rejection_reasons=["candidate_strength_weak"],
                        aggregate_score=57.5,
                    )
                ],
                "empty_reason_summary": {},
            }
        },
        source_metadata={"raw_record_count": 1, "windowed_record_count": 1},
        raw_record_count=1,
        labelable_count=1,
    )

    final = report_module.build_final_assessment([summary])

    assert summary["window_suggestion"] == "policy_split_review"
    assert final["classification"] == "paper_only_review_candidates_present"
    assert final["quality_near_misses_present"] is True


def test_wrapper_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_near_miss_observability_report
        is report_module.run_selected_strategy_edge_candidate_near_miss_observability_report
    )
