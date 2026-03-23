from __future__ import annotations

from typing import Any

from src.research import experimental_candidate_c_topn_weak_strength_diagnosis_report as diagnosis_report
from src.research import research_analyzer


def _candidate_detail(
    *,
    group: str,
    sample_count: int,
    median_future_return_pct: float,
    positive_rate_pct: float | None,
    robustness_value: float | None,
) -> dict[str, Any]:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=sample_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_value=robustness_value,
    )

    return {
        "group": group,
        "symbol": group,
        "strategy": "intraday",
        "horizon": "15m",
        "support_category": "symbol",
        "support_group": group,
        "sample_count": sample_count,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_value": robustness_value,
        "aggregate_score": diagnostics["aggregate_score"],
        "classification_reason": diagnostics["classification_reason"],
        "final_classification": diagnostics["final_classification"],
        "hard_blockers": diagnostics["hard_blockers"],
        "soft_penalties": diagnostics["soft_penalties"],
        "major_deficits": diagnostics["major_deficits"],
        "major_deficit_breakdown": diagnostics["major_deficit_breakdown"],
        "robustness_signal_pct": robustness_value,
        "candidate_strength": diagnostics["final_classification"],
        "candidate_strength_diagnostics": diagnostics,
    }


def test_aggregate_score_summary_reports_count_min_max_avg_and_bucket_counts() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_a",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=45.0,
        ),
        _candidate_detail(
            group="candidate_b",
            sample_count=55,
            median_future_return_pct=0.35,
            positive_rate_pct=56.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_c",
            sample_count=42,
            median_future_return_pct=0.12,
            positive_rate_pct=48.0,
            robustness_value=57.0,
        ),
    ]

    summary = diagnosis_report._aggregate_score_summary(candidates)

    assert summary["available_count"] == 3
    assert isinstance(summary["min"], float)
    assert isinstance(summary["max"], float)
    assert isinstance(summary["avg"], float)
    assert isinstance(summary["bucket_counts"], list)
    assert len(summary["bucket_counts"]) >= 1


def test_aggregate_consistency_debug_reports_threshold_count_and_fields() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_above_threshold",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=45.0,
        ),
        _candidate_detail(
            group="candidate_below_threshold",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=44.0,
        ),
    ]

    debug = diagnosis_report._aggregate_consistency_debug(candidates)

    assert debug["count_above_59_5"] == 1
    assert len(debug["candidates"]) == 2

    first = debug["candidates"][0]
    assert first["sample_count"] == 55
    assert first["final_classification"] == "moderate"
    assert first["classification_reason"] == "cleared_weighted_moderate_profile_with_three_supporting_deficits"
    assert first["support_category"] == "symbol"
    assert first["support_group"] == "candidate_above_threshold"
    assert "major_deficits" in first
    assert "major_deficit_breakdown" in first
    assert "hard_blockers" in first
    assert "soft_penalties" in first


def test_candidate_strength_summary_uses_v5_2_scoring_model() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    assert diagnostics["scoring_model"] == "banded_weighted_v5_2"


def test_candidate_strength_summary_recovers_two_supporting_deficits_as_moderate() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_two_supporting_deficits"
    )


def test_candidate_strength_summary_two_supporting_deficits_recover_at_v5_4_positive_rate_floor() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=40.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_two_supporting_deficits"
    )


def test_candidate_strength_summary_two_supporting_deficits_below_v5_4_positive_rate_floor_remain_weak() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=39.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "two_supporting_deficits_but_positive_rate_below_floor"
    )


def test_candidate_strength_summary_three_supporting_deficits_recover_under_v5_3_threshold() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_three_supporting_deficits"
    )


def test_candidate_strength_summary_three_supporting_deficits_with_lower_positive_rate_still_fail_on_positive_rate_guard() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=48.0,
        robustness_value=45.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "three_supporting_deficits_but_positive_rate_too_low"
    )


def test_candidate_strength_summary_three_supporting_deficits_with_lower_robustness_still_fail_on_aggregate_first() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=44.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "three_supporting_deficits_but_aggregate_too_low"
    )


def test_candidate_strength_summary_three_supporting_deficits_reason_distribution_can_be_counted() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_moderate_two",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_moderate_three",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=45.0,
        ),
        _candidate_detail(
            group="candidate_weak_three_positive_guard",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=48.0,
            robustness_value=45.0,
        ),
        _candidate_detail(
            group="candidate_moderate_two_floor40",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=40.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_weak_two_below_floor40",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=39.0,
            robustness_value=53.0,
        ),
    ]

    count_map: dict[str, int] = {}
    for item in candidates:
        diagnostics = item["candidate_strength_diagnostics"]
        reason = str(diagnostics["classification_reason"])
        count_map[reason] = count_map.get(reason, 0) + 1

    assert (
        count_map["cleared_weighted_moderate_profile_with_two_supporting_deficits"] == 2
    )
    assert count_map["cleared_weighted_moderate_profile_with_three_supporting_deficits"] == 1
    assert count_map["three_supporting_deficits_but_positive_rate_too_low"] == 1
    assert count_map["two_supporting_deficits_but_positive_rate_below_floor"] == 1


def test_strength_counts_split_weak_and_non_weak_correctly() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_moderate_two",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_moderate_three",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=45.0,
        ),
        _candidate_detail(
            group="candidate_moderate_two_floor40",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=40.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_weak",
            sample_count=42,
            median_future_return_pct=0.12,
            positive_rate_pct=48.0,
            robustness_value=57.0,
        ),
    ]

    weak_count = sum(1 for item in candidates if item["candidate_strength"] == "weak")
    moderate_count = sum(
        1 for item in candidates if item["candidate_strength"] == "moderate"
    )
    strong_count = sum(1 for item in candidates if item["candidate_strength"] == "strong")
    non_weak_count = sum(
        1 for item in candidates if item["candidate_strength"] in {"moderate", "strong"}
    )

    assert weak_count == 1
    assert moderate_count == 3
    assert strong_count == 0
    assert non_weak_count == 3