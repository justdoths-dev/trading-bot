from __future__ import annotations

from src.research.experimental_candidate_c_topn_weak_strength_diagnosis_report import (
    build_experimental_candidate_c_topn_weak_strength_diagnosis_summary,
    render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown,
)


def _diagnostics(
    *,
    classification: str,
    reason: str,
    aggregate_score: float,
    hard_blockers: list[str] | None = None,
    soft_penalties: list[str] | None = None,
    major_deficits: list[str] | None = None,
    critical: list[str] | None = None,
    supporting: list[str] | None = None,
    sample_band: str = "moderate",
    median_band: str = "moderate",
    positive_band: str = "moderate",
    robustness_band: str = "moderate",
) -> dict:
    return {
        "final_classification": classification,
        "classification_reason": reason,
        "aggregate_score": aggregate_score,
        "hard_blockers": hard_blockers or [],
        "soft_penalties": soft_penalties or [],
        "major_deficits": major_deficits or [],
        "major_deficit_breakdown": {
            "critical": critical or [],
            "supporting": supporting or [],
        },
        "component_scores": {
            "sample_count": {"band": sample_band},
            "median_future_return_pct": {"band": median_band},
            "positive_rate_pct": {"band": positive_band},
            "robustness_value": {"band": robustness_band},
        },
    }


def _candidate(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    candidate_strength: str,
    quality_gate: str,
    diagnostics: dict,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_strength": candidate_strength,
        "quality_gate": quality_gate,
        "sample_count": 60,
        "median_future_return_pct": 0.35,
        "positive_rate_pct": 54.0,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_value": 50.0,
        "candidate_strength_diagnostics": diagnostics,
        "final_classification": diagnostics["final_classification"],
        "classification_reason": diagnostics["classification_reason"],
        "aggregate_score": diagnostics["aggregate_score"],
        "hard_blockers": diagnostics["hard_blockers"],
        "soft_penalties": diagnostics["soft_penalties"],
        "major_deficits": diagnostics["major_deficits"],
        "major_deficit_breakdown": diagnostics["major_deficit_breakdown"],
        "component_scores": diagnostics["component_scores"],
    }


def _mixed_inputs() -> list[dict]:
    return [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            candidate_strength="weak",
            quality_gate="borderline",
            diagnostics=_diagnostics(
                classification="weak",
                reason="fell_short_of_weighted_moderate_profile",
                aggregate_score=58.0,
                soft_penalties=["subscale_positive_rate"],
                major_deficits=["positive_rate_below_emerging_moderate"],
                supporting=["positive_rate_below_emerging_moderate"],
                positive_band="thin",
            ),
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            candidate_strength="weak",
            quality_gate="borderline",
            diagnostics=_diagnostics(
                classification="weak",
                reason="fell_short_of_weighted_moderate_profile",
                aggregate_score=57.0,
                soft_penalties=["subscale_positive_rate", "subscale_robustness"],
                major_deficits=[
                    "positive_rate_below_emerging_moderate",
                    "robustness_below_emerging_moderate",
                ],
                supporting=[
                    "positive_rate_below_emerging_moderate",
                    "robustness_below_emerging_moderate",
                ],
                positive_band="thin",
                robustness_band="thin",
            ),
        ),
        _candidate(
            symbol="ETHUSDT",
            strategy="swing",
            horizon="1h",
            candidate_strength="moderate",
            quality_gate="passed",
            diagnostics=_diagnostics(
                classification="moderate",
                reason="cleared_weighted_moderate_profile",
                aggregate_score=68.0,
                soft_penalties=["subscale_positive_rate"],
                positive_band="emerging",
                robustness_band="moderate",
            ),
        ),
    ]


def _single_metric_inputs() -> list[dict]:
    return [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            candidate_strength="weak",
            quality_gate="borderline",
            diagnostics=_diagnostics(
                classification="weak",
                reason="fell_short_of_weighted_moderate_profile",
                aggregate_score=59.0,
                soft_penalties=["subscale_positive_rate"],
                major_deficits=[],
                positive_band="thin",
            ),
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            candidate_strength="weak",
            quality_gate="borderline",
            diagnostics=_diagnostics(
                classification="weak",
                reason="fell_short_of_weighted_moderate_profile",
                aggregate_score=58.5,
                soft_penalties=["subscale_positive_rate"],
                major_deficits=[],
                positive_band="thin",
            ),
        ),
    ]


def test_classification_reason_and_major_deficit_counts_are_correct() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 9},
    )

    reason_counts = {
        item["classification_reason"]: item["count"]
        for item in summary["classification_reason_counts"]
    }
    major_deficit_counts = {
        item["major_deficit"]: item["count"]
        for item in summary["major_deficit_counts"]
    }

    assert summary["weak_count"] == 2
    assert summary["moderate_count"] == 1
    assert summary["strong_count"] == 0
    assert summary["non_weak_count"] == 1
    assert reason_counts["fell_short_of_weighted_moderate_profile"] == 2
    assert reason_counts["cleared_weighted_moderate_profile"] == 1
    assert major_deficit_counts["positive_rate_below_emerging_moderate"] == 2
    assert major_deficit_counts["robustness_below_emerging_moderate"] == 1


def test_penalty_and_component_band_aggregates_are_grouped_correctly() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 9},
    )

    soft_penalties = {
        item["soft_penalty"]: item["count"] for item in summary["soft_penalty_counts"]
    }
    positive_bands = {
        item["band"]: item["count"]
        for item in summary["component_band_counts"]["positive_rate_pct"]
    }

    assert soft_penalties["subscale_positive_rate"] == 3
    assert soft_penalties["subscale_robustness"] == 1
    assert positive_bands["thin"] == 2
    assert positive_bands["emerging"] == 1
    assert summary["aggregate_score_summary"]["available_count"] == 3


def test_root_assessment_recommends_threshold_adjustment_when_one_metric_pattern_dominates() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_single_metric_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 9},
    )

    assert summary["root_assessment"]["dominant_classification_reason"] == "fell_short_of_weighted_moderate_profile"
    assert summary["root_assessment"]["dominant_metric_pattern"] == "positive_rate_pct"
    assert summary["root_assessment"]["recommended_next_change"] == "candidate strength threshold adjustment"


def test_supporting_major_deficits_follow_analyzer_v3_contract() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 9},
    )

    critical_counts = {
        item["major_deficit"]: item["count"]
        for item in summary["critical_major_deficit_counts"]
    }
    supporting_counts = {
        item["major_deficit"]: item["count"]
        for item in summary["supporting_major_deficit_counts"]
    }

    assert "positive_rate_below_emerging_moderate" not in critical_counts
    assert supporting_counts["positive_rate_below_emerging_moderate"] == 2
    assert supporting_counts["robustness_below_emerging_moderate"] == 1


def test_hard_blocker_counts_are_exposed_in_summary() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=[
            _candidate(
                symbol="BTCUSDT",
                strategy="intraday",
                horizon="15m",
                candidate_strength="weak",
                quality_gate="borderline",
                diagnostics=_diagnostics(
                    classification="weak",
                    reason="failed_absolute_floor",
                    aggregate_score=0.0,
                    hard_blockers=["median_future_return_pct_non_positive"],
                    major_deficits=[],
                    sample_band="moderate",
                    median_band="below_floor",
                ),
            )
        ],
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 1},
    )

    hard_blockers = {
        item["hard_blocker"]: item["count"]
        for item in summary["hard_blocker_counts"]
    }

    assert hard_blockers["median_future_return_pct_non_positive"] == 1
    assert summary["root_assessment"]["hard_blocker_count"] == 1


def test_markdown_mentions_new_diagnostics_based_summary() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={"newly_visible_identity_count": 9},
    )

    markdown = render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown(summary)

    assert "Config: top_n_symbols=2, top_n_strategies=2" in markdown
    assert "Newly visible identities: 9" in markdown
    assert "Weak count: 2" in markdown
    assert "Moderate count: 1" in markdown
    assert "Dominant classification reason: fell_short_of_weighted_moderate_profile" in markdown
    assert "Supporting Major Deficits" in markdown