from __future__ import annotations

from src.research.experimental_candidate_c_topn_preservation_report import (
    build_experimental_candidate_c_topn_preservation_summary,
    render_experimental_candidate_c_topn_preservation_markdown,
)


def _row(
    *,
    logged_at: str,
    symbol: str,
    strategy: str,
    label_15m: str = "flat",
    label_1h: str = "flat",
    label_4h: str = "flat",
) -> dict:
    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "future_return_15m": 0.4 if label_15m == "up" else -0.2 if label_15m == "down" else 0.0,
        "future_return_1h": 0.5 if label_1h == "up" else -0.3 if label_1h == "down" else 0.0,
        "future_return_4h": 0.6 if label_4h == "up" else -0.4 if label_4h == "down" else 0.0,
        "future_label_15m": label_15m,
        "future_label_1h": label_1h,
        "future_label_4h": label_4h,
    }


def _ranking_entry(*, group: str, rank: int, score: float, sample_count: int) -> dict:
    return {
        "group": group,
        "rank": rank,
        "score": score,
        "metrics": {
            "sample_count": sample_count,
            "labeled_count": sample_count,
            "coverage_pct": 100.0,
            "median_future_return_pct": 0.7,
            "positive_rate_pct": 62.0,
            "robust_positive_rate_pct": 58.0,
        },
    }


def _experiment_summary() -> dict:
    return {
        "strategy_lab": {
            "ranking": {
                "15m": {
                    "by_symbol": {
                        "ranked_groups": [
                            _ranking_entry(group="BTCUSDT", rank=1, score=5.8, sample_count=90),
                            _ranking_entry(group="SOLUSDT", rank=2, score=4.9, sample_count=72),
                        ]
                    },
                    "by_strategy": {
                        "ranked_groups": [
                            _ranking_entry(group="swing", rank=1, score=5.7, sample_count=88),
                            _ranking_entry(group="breakout", rank=2, score=4.7, sample_count=68),
                        ]
                    },
                },
                "1h": {
                    "by_symbol": {
                        "ranked_groups": [
                            _ranking_entry(group="BTCUSDT", rank=1, score=5.4, sample_count=84),
                        ]
                    },
                    "by_strategy": {
                        "ranked_groups": [
                            _ranking_entry(group="swing", rank=1, score=5.3, sample_count=82),
                        ]
                    },
                },
                "4h": {
                    "by_symbol": {"ranked_groups": []},
                    "by_strategy": {"ranked_groups": []},
                },
            }
        }
    }


def _rows() -> tuple[list[dict], list[dict]]:
    baseline_rows = [
        _row(
            logged_at="2026-03-20T00:00:00+00:00",
            symbol="BTCUSDT",
            strategy="swing",
            label_15m="up",
            label_1h="up",
        )
    ]
    experiment_rows = baseline_rows + [
        _row(
            logged_at="2026-03-20T01:00:00+00:00",
            symbol="SOLUSDT",
            strategy="breakout",
            label_15m="up",
        ),
        _row(
            logged_at="2026-03-20T02:00:00+00:00",
            symbol="SOLUSDT",
            strategy="breakout",
            label_15m="down",
        ),
    ]
    return baseline_rows, experiment_rows


def test_topn_increases_visible_identity_coverage_vs_top1() -> None:
    baseline_rows, experiment_rows = _rows()

    summary = build_experimental_candidate_c_topn_preservation_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=_experiment_summary(),
        top_n_symbols=2,
        top_n_strategies=2,
    )

    assert summary["baseline_top1_summary"]["visible_c2_exclusive_identity_count"] == 0
    assert summary["experimental_topn_summary"]["visible_c2_exclusive_identity_count"] == 1
    assert summary["coverage_gain_summary"]["newly_visible_identity_count"] == 1
    assert summary["root_assessment"]["topn_meaningfully_restores_identity_visibility"] is True


def test_topn_preserves_additional_groups_without_touching_top1_baseline() -> None:
    baseline_rows, experiment_rows = _rows()

    summary = build_experimental_candidate_c_topn_preservation_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=_experiment_summary(),
        top_n_symbols=2,
        top_n_strategies=2,
    )

    horizon_summary = summary["per_horizon_preserved_groups"]["15m"]

    assert horizon_summary["top1_symbol_groups"] == ["BTCUSDT"]
    assert horizon_summary["top1_strategy_groups"] == ["swing"]
    assert horizon_summary["topn_symbol_groups"] == ["BTCUSDT", "SOLUSDT"]
    assert horizon_summary["topn_strategy_groups"] == ["swing", "breakout"]
    assert summary["experimental_config"] == {"top_n_symbols": 2, "top_n_strategies": 2}


def test_summary_distinguishes_visibility_restoration_from_eligibility_failure() -> None:
    baseline_rows, experiment_rows = _rows()

    summary = build_experimental_candidate_c_topn_preservation_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=_experiment_summary(),
        top_n_symbols=2,
        top_n_strategies=2,
    )

    assert summary["coverage_gain_summary"]["newly_visible_identity_count"] == 1
    assert summary["experimental_topn_summary"]["visible_c2_exclusive_identity_count"] == 1
    assert summary["experimental_topn_summary"]["mapper_emitted_visible_identity_count"] == 1
    assert summary["experimental_topn_summary"]["mapper_emitted_candidate_count"] == 1

    assert (
        0
        <= summary["experimental_topn_summary"]["eligibility_passed_visible_identity_count"]
        <= summary["experimental_topn_summary"]["visible_c2_exclusive_identity_count"]
    )
    assert summary["eligibility_delta"]["delta"] in {0, 1}
    assert isinstance(
        summary["root_assessment"]["eligibility_next_dominant_blocker_after_restoration"],
        bool,
    )


def test_markdown_mentions_visibility_and_eligibility_split() -> None:
    markdown = render_experimental_candidate_c_topn_preservation_markdown(
        {
            "baseline_top1_summary": {
                "visible_c2_exclusive_identity_count": 0,
                "mapper_emitted_visible_identity_count": 0,
                "eligibility_passed_visible_identity_count": 0,
            },
            "experimental_topn_summary": {
                "visible_c2_exclusive_identity_count": 1,
                "mapper_emitted_visible_identity_count": 1,
                "eligibility_passed_visible_identity_count": 0,
            },
            "coverage_gain_summary": {
                "newly_visible_identity_count": 1,
                "top1_invisible_count": 1,
            },
            "mapper_emission_delta": {"delta": 1},
            "eligibility_delta": {"delta": 0},
            "final_selection_delta": {"delta": 0},
            "root_assessment": {
                "summary": (
                    "Top-N restoration improves visibility, but eligibility still blocks "
                    "downstream survival."
                )
            },
            "experimental_config": {
                "top_n_symbols": 2,
                "top_n_strategies": 2,
            },
        }
    )

    assert "Newly visible identities: 1" in markdown
    assert "Top-1 invisible identities: 1" in markdown
    assert "Experimental mapper-emitted visible identities: 1" in markdown
    assert "Experimental eligibility-passed visible identities: 0" in markdown