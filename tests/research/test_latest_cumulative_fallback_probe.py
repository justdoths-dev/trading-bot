from __future__ import annotations

import json
from pathlib import Path

from src.research.latest_cumulative_fallback_probe import (
    render_latest_cumulative_fallback_probe_markdown,
    run_latest_cumulative_fallback_probe,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _comparison_summary(
    *,
    latest_strength_15m: str = "weak",
    cumulative_strength_15m: str = "weak",
    latest_strength_1h: str = "insufficient_data",
    cumulative_strength_1h: str = "weak",
    latest_strength_4h: str = "insufficient_data",
    cumulative_strength_4h: str = "insufficient_data",
) -> dict:
    return {
        "edge_candidates_comparison": {
            "15m": {
                "latest_candidate_strength": latest_strength_15m,
                "cumulative_candidate_strength": cumulative_strength_15m,
                "latest_top_strategy_group": "swing",
                "cumulative_top_strategy_group": "swing",
                "latest_top_symbol_group": "BTCUSDT",
                "cumulative_top_symbol_group": "BTCUSDT",
                "latest_top_alignment_state_group": "aligned",
                "cumulative_top_alignment_state_group": "aligned",
            },
            "1h": {
                "latest_candidate_strength": latest_strength_1h,
                "cumulative_candidate_strength": cumulative_strength_1h,
                "latest_top_strategy_group": "n/a",
                "cumulative_top_strategy_group": "trend",
                "latest_top_symbol_group": "n/a",
                "cumulative_top_symbol_group": "ETHUSDT",
                "latest_top_alignment_state_group": "n/a",
                "cumulative_top_alignment_state_group": "mixed",
            },
            "4h": {
                "latest_candidate_strength": latest_strength_4h,
                "cumulative_candidate_strength": cumulative_strength_4h,
                "latest_top_strategy_group": "n/a",
                "cumulative_top_strategy_group": "n/a",
                "latest_top_symbol_group": "n/a",
                "cumulative_top_symbol_group": "n/a",
                "latest_top_alignment_state_group": "n/a",
                "cumulative_top_alignment_state_group": "n/a",
            },
        }
    }


def _summary(
    candidate_strength_15m: str,
    candidate_strength_1h: str,
    candidate_strength_4h: str,
    *,
    median_15m: float | None = 0.2,
    median_1h: float | None = -0.1,
    median_4h: float | None = None,
) -> dict:
    return {
        "edge_candidates_preview": {
            "by_horizon": {
                "15m": {
                    "top_strategy": {"group": "swing", "candidate_strength": candidate_strength_15m},
                    "top_symbol": {"group": "BTCUSDT", "candidate_strength": candidate_strength_15m},
                    "top_alignment_state": {"group": "aligned", "candidate_strength": candidate_strength_15m},
                },
                "1h": {
                    "top_strategy": {"group": "trend", "candidate_strength": candidate_strength_1h},
                    "top_symbol": {"group": "ETHUSDT", "candidate_strength": candidate_strength_1h},
                    "top_alignment_state": {"group": "mixed", "candidate_strength": candidate_strength_1h},
                },
                "4h": {
                    "top_strategy": {"group": "macro", "candidate_strength": candidate_strength_4h},
                    "top_symbol": {"group": "SOLUSDT", "candidate_strength": candidate_strength_4h},
                    "top_alignment_state": {"group": "aligned", "candidate_strength": candidate_strength_4h},
                },
            }
        },
        "strategy_lab": {
            "ranking": {
                "15m": {
                    "by_strategy": {
                        "ranked_groups": [
                            {"group": "swing", "metrics": {"median_future_return_pct": median_15m, "sample_count": 60, "labeled_count": 50}}
                        ]
                    },
                    "by_symbol": {
                        "ranked_groups": [
                            {"group": "BTCUSDT", "metrics": {"median_future_return_pct": median_15m, "sample_count": 60, "labeled_count": 50}}
                        ]
                    },
                    "by_alignment_state": {
                        "ranked_groups": [
                            {"group": "aligned", "metrics": {"median_future_return_pct": median_15m, "sample_count": 60, "labeled_count": 50}}
                        ]
                    },
                },
                "1h": {
                    "by_strategy": {
                        "ranked_groups": [
                            {"group": "trend", "metrics": {"median_future_return_pct": median_1h, "sample_count": 42, "labeled_count": 23}}
                        ]
                    },
                    "by_symbol": {
                        "ranked_groups": [
                            {"group": "ETHUSDT", "metrics": {"median_future_return_pct": median_1h, "sample_count": 42, "labeled_count": 23}}
                        ]
                    },
                    "by_alignment_state": {
                        "ranked_groups": [
                            {"group": "mixed", "metrics": {"median_future_return_pct": median_1h, "sample_count": 42, "labeled_count": 23}}
                        ]
                    },
                },
                "4h": {
                    "by_strategy": {
                        "ranked_groups": (
                            []
                            if median_4h is None
                            else [{"group": "macro", "metrics": {"median_future_return_pct": median_4h, "sample_count": 55, "labeled_count": 40}}]
                        )
                    },
                    "by_symbol": {
                        "ranked_groups": (
                            []
                            if median_4h is None
                            else [{"group": "SOLUSDT", "metrics": {"median_future_return_pct": median_4h, "sample_count": 55, "labeled_count": 40}}]
                        )
                    },
                    "by_alignment_state": {
                        "ranked_groups": (
                            []
                            if median_4h is None
                            else [{"group": "aligned", "metrics": {"median_future_return_pct": median_4h, "sample_count": 55, "labeled_count": 40}}]
                        )
                    },
                },
            }
        },
    }


def _score_drift(
    *,
    strategy_direction: str = "flat",
    symbol_direction: str = "flat",
    alignment_direction: str = "flat",
) -> dict:
    return {
        "score_drift": [
            {"category": "strategy", "group": "trend", "drift_direction": strategy_direction},
            {"category": "symbol", "group": "ETHUSDT", "drift_direction": symbol_direction},
            {"category": "alignment_state", "group": "mixed", "drift_direction": alignment_direction},
        ]
    }


def test_missing_input_files(tmp_path: Path) -> None:
    result = run_latest_cumulative_fallback_probe(
        comparison_summary_path=tmp_path / "missing_comparison.json",
        score_drift_summary_path=tmp_path / "missing_drift.json",
        latest_summary_path=tmp_path / "missing_latest.json",
        cumulative_summary_path=tmp_path / "missing_cumulative.json",
        output_dir=tmp_path / "out",
    )

    assert result["summary"]["metadata"]["total_category_pairs_evaluated"] == 9
    assert result["summary"]["final_diagnosis"]["fallback_justification"] == "rarely_justified"
    assert Path(result["summary_json"]).exists()
    assert Path(result["summary_md"]).exists()


def test_basic_healthy_input_where_latest_and_cumulative_are_both_visible(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_strength_1h="weak",
            cumulative_strength_1h="weak",
        ),
    )
    _write_json(drift, _score_drift())
    _write_json(latest, _summary("weak", "weak", "insufficient_data", median_1h=0.11))
    _write_json(cumulative, _summary("weak", "weak", "insufficient_data", median_1h=0.11))

    result = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )
    summary = result["summary"]

    assert summary["fallback_eligibility_analysis"]["latest_failed_cumulative_visible"] == 0
    assert summary["final_diagnosis"]["fallback_justification"] == "rarely_justified"


def test_latest_failed_non_positive_median_and_drift_not_decreasing_is_eligible(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(comparison, _comparison_summary())
    _write_json(drift, _score_drift(strategy_direction="flat", symbol_direction="increase", alignment_direction="flat"))
    _write_json(latest, _summary("weak", "insufficient_data", "insufficient_data", median_1h=-0.1))
    _write_json(cumulative, _summary("weak", "weak", "insufficient_data", median_1h=0.2))

    result = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )
    summary = result["summary"]

    assert summary["fallback_eligibility_analysis"]["latest_failed_due_to_non_positive_median_and_drift_not_decreasing"] == 3
    assert summary["horizon_breakdown"]["1h"]["fallback_eligible_count"] == 3
    assert summary["horizon_breakdown"]["1h"]["non_positive_median_count"] == 3


def test_latest_failed_non_positive_median_and_drift_decreasing_is_not_eligible(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(comparison, _comparison_summary())
    _write_json(drift, _score_drift(strategy_direction="decrease", symbol_direction="decrease", alignment_direction="decrease"))
    _write_json(latest, _summary("weak", "insufficient_data", "insufficient_data", median_1h=-0.2))
    _write_json(cumulative, _summary("weak", "weak", "insufficient_data", median_1h=0.2))

    result = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )
    summary = result["summary"]

    assert summary["fallback_eligibility_analysis"]["latest_failed_due_to_non_positive_median_and_drift_decreasing"] == 3
    assert summary["horizon_breakdown"]["1h"]["fallback_ineligible_count"] == 3


def test_unknown_drift_keeps_non_positive_median_visible_as_separate_reason(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(comparison, _comparison_summary())
    _write_json(drift, {"score_drift": []})
    _write_json(latest, _summary("weak", "insufficient_data", "insufficient_data", median_1h=-0.3))
    _write_json(cumulative, _summary("weak", "weak", "insufficient_data", median_1h=0.2))

    result = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )
    summary = result["summary"]

    assert summary["fallback_eligibility_analysis"]["latest_failed_due_to_non_positive_median"] == 3
    assert summary["fallback_eligibility_analysis"]["latest_failed_due_to_non_positive_median_and_drift_not_decreasing"] == 0
    assert summary["horizon_breakdown"]["1h"]["drift_direction_distribution"]["unknown"] == 3


def test_malformed_json_input_is_handled_safely(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_raw(comparison, "{not-valid-json")
    _write_raw(drift, "{not-valid-json")
    _write_raw(latest, "{not-valid-json")
    _write_raw(cumulative, "{not-valid-json")

    result = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )

    assert result["summary"]["metadata"]["total_category_pairs_evaluated"] == 9
    assert result["summary"]["final_diagnosis"]["fallback_justification"] == "rarely_justified"


def test_mixed_horizon_breakdown_correctness(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_strength_15m="weak",
            cumulative_strength_15m="weak",
            latest_strength_1h="insufficient_data",
            cumulative_strength_1h="weak",
            latest_strength_4h="insufficient_data",
            cumulative_strength_4h="weak",
        ),
    )
    _write_json(
        drift,
        {
            "score_drift": [
                {"category": "strategy", "group": "trend", "drift_direction": "flat"},
                {"category": "symbol", "group": "ETHUSDT", "drift_direction": "flat"},
                {"category": "alignment_state", "group": "mixed", "drift_direction": "flat"},
                {"category": "strategy", "group": "macro", "drift_direction": "decrease"},
                {"category": "symbol", "group": "SOLUSDT", "drift_direction": "decrease"},
                {"category": "alignment_state", "group": "aligned", "drift_direction": "decrease"},
            ]
        },
    )
    _write_json(
        latest,
        _summary(
            "weak",
            "insufficient_data",
            "insufficient_data",
            median_1h=-0.2,
            median_4h=-0.15,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "weak",
            "weak",
            "weak",
            median_1h=0.1,
            median_4h=0.1,
        ),
    )

    summary = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]

    assert summary["horizon_breakdown"]["15m"]["fallback_eligible_count"] == 0
    assert summary["horizon_breakdown"]["1h"]["fallback_eligible_count"] == 3
    assert summary["horizon_breakdown"]["4h"]["fallback_ineligible_count"] == 3


def test_markdown_rendering_contains_final_diagnosis_section(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(comparison, _comparison_summary())
    _write_json(drift, _score_drift())
    _write_json(latest, _summary("weak", "insufficient_data", "insufficient_data", median_1h=-0.1))
    _write_json(cumulative, _summary("weak", "weak", "insufficient_data", median_1h=0.2))

    summary = run_latest_cumulative_fallback_probe(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]
    markdown = render_latest_cumulative_fallback_probe_markdown(summary)

    assert "Final Diagnosis" in markdown
    assert "fallback_justification" in markdown
    assert "median=" in markdown
