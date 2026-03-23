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
        "sample_count": sample_count,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
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
            robustness_value=53.0,
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


def test_candidate_strength_summary_uses_v5_1_scoring_model() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=53.0,
    )

    assert diagnostics["scoring_model"] == "banded_weighted_v5_1"


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


def test_candidate_strength_summary_blocks_two_supporting_deficits_when_sample_not_moderate() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=42,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "two_supporting_deficits_but_sample_not_moderate"
    )


def test_candidate_strength_summary_blocks_two_supporting_deficits_when_positive_rate_below_floor() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=47.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "two_supporting_deficits_but_positive_rate_below_floor"
    )


def test_candidate_strength_summary_blocks_three_supporting_deficits() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert (
        diagnostics["classification_reason"]
        == "three_or_more_supporting_deficits_present"
    )


def test_classification_reason_distribution_can_be_counted_from_candidate_details() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_moderate",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_weak_sample",
            sample_count=42,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_weak_three_deficits",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=45.0,
        ),
    ]

    count_map: dict[str, int] = {}
    for item in candidates:
        diagnostics = item["candidate_strength_diagnostics"]
        reason = str(diagnostics["classification_reason"])
        count_map[reason] = count_map.get(reason, 0) + 1

    assert (
        count_map["cleared_weighted_moderate_profile_with_two_supporting_deficits"] == 1
    )
    assert count_map["two_supporting_deficits_but_sample_not_moderate"] == 1
    assert count_map["three_or_more_supporting_deficits_present"] == 1


def test_strength_counts_split_weak_and_non_weak_correctly() -> None:
    candidates = [
        _candidate_detail(
            group="candidate_moderate_a",
            sample_count=55,
            median_future_return_pct=0.17,
            positive_rate_pct=49.0,
            robustness_value=53.0,
        ),
        _candidate_detail(
            group="candidate_moderate_b",
            sample_count=55,
            median_future_return_pct=0.35,
            positive_rate_pct=56.0,
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
    assert moderate_count == 2
    assert strong_count == 0
    assert non_weak_count == 2