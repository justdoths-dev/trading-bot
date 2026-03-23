from __future__ import annotations

from src.research.experimental_candidate_c_topn_weak_strength_diagnosis_report import (
    build_experimental_candidate_c_topn_weak_strength_diagnosis_summary,
    render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown,
)


def _candidate(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    sample_count: int,
    median_future_return_pct: float,
    positive_rate_pct: float | None,
    robustness_value: float | None,
    candidate_strength: str = "weak",
    weak_drivers: list[str] | None = None,
) -> dict:
    drivers = weak_drivers or []
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "sample_count": sample_count,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct" if robustness_value is not None else "n/a",
        "robustness_value": robustness_value,
        "candidate_strength": candidate_strength,
        "quality_gate": "borderline" if candidate_strength == "weak" else "passed",
        "weak_drivers": drivers,
        "weak_driver_combination": (
            "+".join(sorted(drivers)) if drivers else "meets_moderate_thresholds"
        ),
    }


def _dominant_positive_rate_inputs() -> list[dict]:
    return [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            sample_count=60,
            median_future_return_pct=0.42,
            positive_rate_pct=52.0,
            robustness_value=58.0,
            weak_drivers=["low_positive_rate"],
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            sample_count=62,
            median_future_return_pct=0.39,
            positive_rate_pct=53.0,
            robustness_value=57.0,
            weak_drivers=["low_positive_rate"],
        ),
        _candidate(
            symbol="XRPUSDT",
            strategy="reversal",
            horizon="4h",
            sample_count=58,
            median_future_return_pct=0.36,
            positive_rate_pct=54.0,
            robustness_value=56.0,
            weak_drivers=["low_positive_rate"],
        ),
    ]


def _dominant_robustness_inputs() -> list[dict]:
    return [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            sample_count=80,
            median_future_return_pct=0.42,
            positive_rate_pct=61.0,
            robustness_value=40.0,
            weak_drivers=["low_robustness_value"],
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            sample_count=82,
            median_future_return_pct=0.39,
            positive_rate_pct=62.0,
            robustness_value=41.0,
            weak_drivers=["low_robustness_value"],
        ),
        _candidate(
            symbol="XRPUSDT",
            strategy="reversal",
            horizon="4h",
            sample_count=78,
            median_future_return_pct=0.36,
            positive_rate_pct=63.0,
            robustness_value=42.0,
            weak_drivers=["low_robustness_value"],
        ),
    ]


def _mixed_driver_inputs() -> list[dict]:
    return [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            sample_count=44,
            median_future_return_pct=0.28,
            positive_rate_pct=57.0,
            robustness_value=58.0,
            weak_drivers=["low_sample_count", "low_median_return"],
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            sample_count=66,
            median_future_return_pct=0.35,
            positive_rate_pct=52.0,
            robustness_value=50.0,
            weak_drivers=["low_positive_rate", "low_robustness_value"],
        ),
        _candidate(
            symbol="ETHUSDT",
            strategy="swing",
            horizon="1h",
            sample_count=72,
            median_future_return_pct=0.33,
            positive_rate_pct=54.0,
            robustness_value=None,
            weak_drivers=["low_positive_rate"],
        ),
    ]


def test_weak_driver_classification_counts_individual_drivers() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_driver_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={
            "baseline_top1_visible_count": 3,
            "topn_visible_count": 12,
            "newly_visible_identity_count": 9,
        },
    )

    individual = {
        item["weak_driver"]: item["count"]
        for item in summary["weak_driver_counts"]["individual"]
    }

    assert individual["low_positive_rate"] == 2
    assert individual["low_sample_count"] == 1
    assert individual["low_median_return"] == 1
    assert individual["low_robustness_value"] == 1


def test_aggregate_weak_driver_breakdowns_are_grouped_correctly() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_mixed_driver_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={
            "baseline_top1_visible_count": 3,
            "topn_visible_count": 12,
            "newly_visible_identity_count": 9,
        },
    )

    by_horizon = {
        item["horizon"]: item for item in summary["weak_driver_breakdown_by_horizon"]
    }
    by_strategy = {
        item["strategy"]: item for item in summary["weak_driver_breakdown_by_strategy"]
    }

    assert by_horizon["1h"]["candidate_count"] == 2
    assert by_horizon["15m"]["candidate_count"] == 1
    assert by_strategy["trend"]["weak_driver_counts"][0]["weak_driver"] == "low_positive_rate"
    assert summary["weak_count"] == 3
    assert summary["non_weak_count"] == 0


def test_root_assessment_recommends_threshold_adjustment_when_single_metric_dominates() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_dominant_positive_rate_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={
            "baseline_top1_visible_count": 3,
            "topn_visible_count": 12,
            "newly_visible_identity_count": 9,
        },
    )

    assert summary["root_assessment"]["dominant_weak_driver"] == "low_positive_rate"
    assert summary["root_assessment"]["single_metric_dominates"] is True
    assert (
        summary["root_assessment"]["recommended_next_change"]
        == "candidate strength threshold adjustment"
    )


def test_root_assessment_recommends_formula_redesign_for_robustness_dominance() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_dominant_robustness_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={
            "baseline_top1_visible_count": 3,
            "topn_visible_count": 12,
            "newly_visible_identity_count": 9,
        },
    )

    assert summary["root_assessment"]["dominant_weak_driver"] == "low_robustness_value"
    assert summary["root_assessment"]["single_metric_dominates"] is True
    assert (
        summary["root_assessment"]["recommended_next_change"]
        == "candidate strength formula redesign"
    )


def test_markdown_mentions_dominant_driver_and_recommendation() -> None:
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_dominant_positive_rate_inputs(),
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
        visibility_context={
            "baseline_top1_visible_count": 3,
            "topn_visible_count": 12,
            "newly_visible_identity_count": 9,
        },
    )

    markdown = render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown(summary)

    assert "Config: top_n_symbols=2, top_n_strategies=2" in markdown
    assert "Newly visible identities: 9" in markdown
    assert "Weak count: 3" in markdown
    assert "Dominant weak driver: low_positive_rate" in markdown
    assert "Recommended next change: candidate strength threshold adjustment" in markdown
