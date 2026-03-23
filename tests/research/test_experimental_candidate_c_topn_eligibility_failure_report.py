from __future__ import annotations

from src.research.experimental_candidate_c_topn_eligibility_failure_report import (
    build_experimental_candidate_c_topn_eligibility_failure_summary,
    render_experimental_candidate_c_topn_eligibility_failure_markdown,
)


def _candidate(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    strength: str,
    stability: str,
    score: float,
    sample_size: int,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_candidate_strength": strength,
        "selected_stability_label": stability,
        "edge_stability_score": score,
        "latest_sample_size": sample_size,
        "selected_visible_horizons": [horizon],
        "source_preference": "symbol",
    }


def _ranked(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    status: str,
    reasons: list[str],
    strength: str,
    stability: str,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_status": status,
        "reason_codes": reasons,
        "selected_candidate_strength": strength,
        "selected_stability_label": stability,
        "gate_diagnostics": {
            "blocked_reasons": reasons if status == "blocked" else [],
            "penalty_reasons": reasons if status == "penalized" else [],
        },
    }


def _inputs() -> tuple[list[dict], list[dict], dict]:
    restored_candidates = [
        _candidate(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            strength="moderate",
            stability="multi_horizon_confirmed",
            score=4.2,
            sample_size=44,
        ),
        _candidate(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            strength="moderate",
            stability="single_horizon_only",
            score=3.4,
            sample_size=38,
        ),
        _candidate(
            symbol="XRPUSDT",
            strategy="reversal",
            horizon="1h",
            strength="weak",
            stability="single_horizon_only",
            score=2.7,
            sample_size=36,
        ),
    ]
    restored_ranking = [
        _ranked(
            symbol="SOLUSDT",
            strategy="breakout",
            horizon="15m",
            status="blocked",
            reasons=["CANDIDATE_SYMBOL_SUPPORT_TOO_LOW"],
            strength="moderate",
            stability="multi_horizon_confirmed",
        ),
        _ranked(
            symbol="ADAUSDT",
            strategy="trend",
            horizon="1h",
            status="penalized",
            reasons=["CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY"],
            strength="moderate",
            stability="single_horizon_only",
        ),
        _ranked(
            symbol="XRPUSDT",
            strategy="reversal",
            horizon="1h",
            status="penalized",
            reasons=["CANDIDATE_STRENGTH_WEAK", "CANDIDATE_EDGE_STABILITY_SCORE_LOW"],
            strength="weak",
            stability="single_horizon_only",
        ),
    ]
    visibility_context = {
        "baseline_top1_visible_count": 3,
        "topn_visible_count": 12,
        "newly_visible_identity_count": 9,
    }
    return restored_candidates, restored_ranking, visibility_context


def test_blocked_restored_candidates_are_counted_correctly() -> None:
    restored_candidates, restored_ranking, visibility_context = _inputs()

    summary = build_experimental_candidate_c_topn_eligibility_failure_summary(
        restored_candidates=restored_candidates,
        restored_ranking=restored_ranking,
        visibility_context=visibility_context,
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
    )

    assert summary["restored_candidate_count"] == 3
    assert summary["blocked_count"] == 1
    assert summary["penalized_count"] == 2
    assert summary["eligibility_passed_count"] == 0


def test_blocker_breakdowns_group_reasons_correctly() -> None:
    restored_candidates, restored_ranking, visibility_context = _inputs()

    summary = build_experimental_candidate_c_topn_eligibility_failure_summary(
        restored_candidates=restored_candidates,
        restored_ranking=restored_ranking,
        visibility_context=visibility_context,
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
    )

    horizon_breakdown = {item["horizon"]: item for item in summary["blocker_breakdown_by_horizon"]}
    strategy_breakdown = {item["strategy"]: item for item in summary["blocker_breakdown_by_strategy"]}
    stability_breakdown = {
        item["stability_label"]: item for item in summary["blocker_breakdown_by_stability_label"]
    }

    assert horizon_breakdown["1h"]["candidate_count"] == 2
    assert horizon_breakdown["15m"]["candidate_count"] == 1
    assert strategy_breakdown["reversal"]["blocker_counts"][0]["reason_code"] == "CANDIDATE_EDGE_STABILITY_SCORE_LOW"
    assert stability_breakdown["single_horizon_only"]["candidate_count"] == 2


def test_summary_distinguishes_visibility_restoration_from_readiness_failure() -> None:
    restored_candidates, restored_ranking, visibility_context = _inputs()

    summary = build_experimental_candidate_c_topn_eligibility_failure_summary(
        restored_candidates=restored_candidates,
        restored_ranking=restored_ranking,
        visibility_context=visibility_context,
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
    )

    assert summary["root_assessment"]["visibility_restoration_confirmed"] is True
    assert summary["root_assessment"]["newly_visible_identity_count"] == 9
    assert summary["root_assessment"]["restored_candidates_failed_readiness"] is True
    assert summary["root_assessment"]["single_most_common_failure_reason"] == "CANDIDATE_EDGE_STABILITY_SCORE_LOW"
    assert summary["root_assessment"]["recommended_next_change"] == "readiness scoring redesign"


def test_markdown_mentions_recommended_next_change() -> None:
    restored_candidates, restored_ranking, visibility_context = _inputs()
    summary = build_experimental_candidate_c_topn_eligibility_failure_summary(
        restored_candidates=restored_candidates,
        restored_ranking=restored_ranking,
        visibility_context=visibility_context,
        experimental_config={"top_n_symbols": 2, "top_n_strategies": 2},
    )

    markdown = render_experimental_candidate_c_topn_eligibility_failure_markdown(summary)

    assert "Restored candidate count: 3" in markdown
    assert "Most common failure reason: CANDIDATE_EDGE_STABILITY_SCORE_LOW" in markdown
    assert "Dominant failure family: weak_candidate_quality_or_low_confidence" in markdown
    assert "Recommended next change: readiness scoring redesign" in markdown