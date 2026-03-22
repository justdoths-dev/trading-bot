from __future__ import annotations

from pathlib import Path

from src.research.experimental_relabel_comparison_report import (
    _build_edge_candidates_preview_comparison,
    _build_edge_stability_preview_comparison,
    _build_final_diagnosis,
    _build_group_comparison,
    _build_horizon_comparison,
    _normalize_metric_payload,
    build_experimental_relabel_comparison_report,
)


def _make_summary() -> dict[str, object]:
    return {
        "horizon_summary": {
            "15m": {
                "labeled_records": 100,
                "label_distribution": {"up": 20, "down": 20, "flat": 60},
                "avg_future_return_pct": 0.01,
                "median_future_return_pct": -0.04,
                "positive_rate_pct": 20.0,
                "negative_rate_pct": 20.0,
                "flat_rate_pct": 60.0,
            },
            "1h": {
                "labeled_records": 80,
                "label_distribution": {"up": 22, "down": 18, "flat": 40},
                "avg_future_return_pct": 0.03,
                "median_future_return_pct": -0.02,
                "positive_rate_pct": 27.5,
                "negative_rate_pct": 22.5,
                "flat_rate_pct": 50.0,
            },
            "4h": {
                "labeled_records": 60,
                "label_distribution": {"up": 20, "down": 15, "flat": 25},
                "avg_future_return_pct": 0.05,
                "median_future_return_pct": 0.01,
                "positive_rate_pct": 33.3,
                "negative_rate_pct": 25.0,
                "flat_rate_pct": 41.7,
            },
        },
        "by_symbol": {
            "BTCUSDT": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 40,
                        "label_distribution": {"up": 8, "down": 10, "flat": 22},
                        "median_future_return_pct": -0.03,
                        "positive_rate_pct": 20.0,
                        "flat_rate_pct": 55.0,
                    }
                }
            },
            "ETHUSDT": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 35,
                        "label_distribution": {"up": 7, "down": 8, "flat": 20},
                        "median_future_return_pct": -0.04,
                        "positive_rate_pct": 20.0,
                        "flat_rate_pct": 57.0,
                    }
                }
            },
        },
        "by_strategy": {
            "swing": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 50,
                        "label_distribution": {"up": 10, "down": 10, "flat": 30},
                        "median_future_return_pct": -0.05,
                        "positive_rate_pct": 20.0,
                        "flat_rate_pct": 60.0,
                    }
                }
            }
        },
        "edge_candidates_preview": {
            "by_horizon": {
                "15m": {
                    "sample_gate": "passed",
                    "quality_gate": "failed",
                    "candidate_strength": "weak",
                    "visibility_reason": "limited_quality",
                    "top_strategy": {
                        "sample_gate": "passed",
                        "quality_gate": "failed",
                        "candidate_strength": "weak",
                        "visibility_reason": "limited_quality",
                        "group": "swing",
                        "median_future_return_pct": -0.05,
                        "positive_rate_pct": 20.0,
                    },
                    "top_symbol": {
                        "sample_gate": "passed",
                        "quality_gate": "failed",
                        "candidate_strength": "weak",
                        "visibility_reason": "limited_quality",
                        "group": "BTCUSDT",
                        "median_future_return_pct": -0.03,
                        "positive_rate_pct": 20.0,
                    },
                    "top_alignment_state": {
                        "sample_gate": "passed",
                        "quality_gate": "failed",
                        "candidate_strength": "weak",
                        "visibility_reason": "limited_quality",
                        "group": "aligned",
                        "median_future_return_pct": -0.01,
                        "positive_rate_pct": 22.0,
                    },
                }
            }
        },
        "edge_stability_preview": {
            "strategy": {
                "group": "swing",
                "visible_horizons": ["15m"],
                "stability_label": "single_horizon_only",
                "stability_score": 0.32,
                "visibility_reason": "limited_quality",
            },
            "symbol": {
                "group": "BTCUSDT",
                "visible_horizons": ["15m"],
                "stability_label": "single_horizon_only",
                "stability_score": 0.28,
                "visibility_reason": "limited_quality",
            },
            "alignment_state": {
                "group": "aligned",
                "visible_horizons": ["15m"],
                "stability_label": "single_horizon_only",
                "stability_score": 0.35,
                "visibility_reason": "limited_quality",
            },
        },
    }


def test_normalize_metric_payload_preserves_zero_labeled_records() -> None:
    payload = {
        "labeled_records": 0,
        "labeled_count": 12,
        "sample_count": 20,
        "label_distribution": {"up": 0, "down": 0, "flat": 0},
        "positive_rate_pct": 0.0,
        "negative_rate_pct": 0.0,
        "flat_rate_pct": 0.0,
    }

    result = _normalize_metric_payload(payload)

    assert result["labeled_records"] == 0


def test_horizon_delta_calculation() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["horizon_summary"]["15m"]["flat_rate_pct"] = 48.0
    experiment["horizon_summary"]["15m"]["median_future_return_pct"] = 0.03
    experiment["horizon_summary"]["15m"]["labeled_records"] = 110

    result = _build_horizon_comparison(baseline, experiment)

    assert result["15m"]["flat_rate_pct"] == {
        "baseline": 60.0,
        "experiment": 48.0,
        "delta": -12.0,
    }
    assert result["15m"]["median_future_return_pct"] == {
        "baseline": -0.04,
        "experiment": 0.03,
        "delta": 0.07,
    }
    assert result["15m"]["labeled_records"] == {
        "baseline": 100.0,
        "experiment": 110.0,
        "delta": 10.0,
    }


def test_missing_key_safe_comparison_behavior() -> None:
    baseline = {"horizon_summary": {"15m": {"flat_rate_pct": 60.0}}}
    experiment = {"horizon_summary": {"15m": {}}}

    result = _build_horizon_comparison(baseline, experiment)

    assert result["15m"]["flat_rate_pct"] == {
        "baseline": 60.0,
        "experiment": None,
        "delta": None,
    }
    assert result["15m"]["median_future_return_pct"] == {
        "baseline": None,
        "experiment": None,
        "delta": None,
    }


def test_final_diagnosis_marks_improved_but_not_selection_ready() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["horizon_summary"]["15m"]["flat_rate_pct"] = 45.0
    experiment["horizon_summary"]["15m"]["median_future_return_pct"] = 0.03
    experiment["horizon_summary"]["1h"]["median_future_return_pct"] = 0.04
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["median_future_return_pct"] = 0.02
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["flat_rate_pct"] = 42.0
    experiment["by_strategy"]["swing"]["horizon_summary"]["15m"]["median_future_return_pct"] = 0.01
    experiment["by_strategy"]["swing"]["horizon_summary"]["15m"]["flat_rate_pct"] = 44.0

    horizon_comparison = _build_horizon_comparison(baseline, experiment)
    by_symbol_comparison = _build_group_comparison(
        baseline, experiment, category="symbol"
    )
    by_strategy_comparison = _build_group_comparison(
        baseline, experiment, category="strategy"
    )
    edge_preview_comparison = _build_edge_candidates_preview_comparison(
        baseline, experiment
    )
    stability_comparison = build_experimental_relabel_comparison_report(
        baseline,
        experiment,
        baseline_path=Path("baseline.json"),
        experiment_path=Path("experiment.json"),
    )["edge_stability_preview_comparison"]

    diagnosis = _build_final_diagnosis(
        horizon_comparison=horizon_comparison,
        by_symbol_comparison=by_symbol_comparison,
        by_strategy_comparison=by_strategy_comparison,
        edge_preview_comparison=edge_preview_comparison,
        stability_comparison=stability_comparison,
    )

    assert diagnosis["primary_finding"] == "experiment_improved_distribution_but_not_selection_ready"
    assert "meaningful_flat_suppression_reduction_detected" in diagnosis["diagnosis_labels"]
    assert "median_recovery_detected_across_horizons" in diagnosis["diagnosis_labels"]
    assert "edge_strength_still_weak" in diagnosis["diagnosis_labels"]
    assert "experiment_improved_distribution_but_not_selection_ready" in diagnosis["diagnosis_labels"]


def test_symbol_comparison_correctness() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["median_future_return_pct"] = 0.01
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["flat_rate_pct"] = 47.0

    result = _build_group_comparison(baseline, experiment, category="symbol")
    btc_row = result["15m"][0]

    assert btc_row["group"] == "BTCUSDT"
    assert btc_row["median_future_return_pct"] == {
        "baseline": -0.03,
        "experiment": 0.01,
        "delta": 0.04,
    }
    assert btc_row["flat_rate_pct"] == {
        "baseline": 55.0,
        "experiment": 47.0,
        "delta": -8.0,
    }


def test_strategy_comparison_correctness() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["by_strategy"]["swing"]["horizon_summary"]["15m"]["positive_rate_pct"] = 28.0
    experiment["by_strategy"]["swing"]["horizon_summary"]["15m"]["flat_rate_pct"] = 46.0

    result = _build_group_comparison(baseline, experiment, category="strategy")
    swing_row = result["15m"][0]

    assert swing_row["group"] == "swing"
    assert swing_row["positive_rate_pct"] == {
        "baseline": 20.0,
        "experiment": 28.0,
        "delta": 8.0,
    }
    assert swing_row["flat_rate_pct"] == {
        "baseline": 60.0,
        "experiment": 46.0,
        "delta": -14.0,
    }


def test_edge_preview_comparison_correctness() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["top_symbol"]["group"] = "ETHUSDT"
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["top_symbol"]["median_future_return_pct"] = 0.02
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["top_symbol"]["positive_rate_pct"] = 31.0

    result = _build_edge_candidates_preview_comparison(baseline, experiment)
    top_symbol = result["15m"]["top_symbol"]

    assert top_symbol["group"] == {
        "baseline": "BTCUSDT",
        "experiment": "ETHUSDT",
        "changed": True,
    }
    assert top_symbol["median_future_return_pct"] == {
        "baseline": -0.03,
        "experiment": 0.02,
        "delta": 0.05,
    }
    assert top_symbol["positive_rate_pct"] == {
        "baseline": 20.0,
        "experiment": 31.0,
        "delta": 11.0,
    }


def test_horizon_level_edge_preview_comparison_correctness() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["sample_gate"] = "failed"
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["quality_gate"] = "passed"
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["candidate_strength"] = "moderate"
    experiment["edge_candidates_preview"]["by_horizon"]["15m"]["visibility_reason"] = "improved_distribution"

    result = _build_edge_candidates_preview_comparison(baseline, experiment)
    horizon_payload = result["15m"]

    assert horizon_payload["sample_gate"] == {
        "baseline": "passed",
        "experiment": "failed",
        "changed": True,
    }
    assert horizon_payload["quality_gate"] == {
        "baseline": "failed",
        "experiment": "passed",
        "changed": True,
    }
    assert horizon_payload["candidate_strength"] == {
        "baseline": "weak",
        "experiment": "moderate",
        "changed": True,
    }
    assert horizon_payload["visibility_reason"] == {
        "baseline": "limited_quality",
        "experiment": "improved_distribution",
        "changed": True,
    }


def test_stability_comparison_correctness() -> None:
    baseline = _make_summary()
    experiment = _make_summary()
    experiment["edge_stability_preview"]["symbol"]["group"] = "ETHUSDT"
    experiment["edge_stability_preview"]["symbol"]["visible_horizons"] = ["15m", "1h"]
    experiment["edge_stability_preview"]["symbol"]["stability_label"] = "multi_horizon_confirmed"
    experiment["edge_stability_preview"]["symbol"]["stability_score"] = 0.51
    experiment["edge_stability_preview"]["symbol"]["visibility_reason"] = "recovered_visibility"

    result = _build_edge_stability_preview_comparison(baseline, experiment)
    symbol_payload = result["symbol"]

    assert symbol_payload["group"] == {
        "baseline": "BTCUSDT",
        "experiment": "ETHUSDT",
        "changed": True,
    }
    assert symbol_payload["visible_horizons"] == {
        "baseline": ["15m"],
        "experiment": ["15m", "1h"],
        "changed": True,
    }
    assert symbol_payload["stability_label"] == {
        "baseline": "single_horizon_only",
        "experiment": "multi_horizon_confirmed",
        "changed": True,
    }
    assert symbol_payload["stability_score"] == {
        "baseline": 0.28,
        "experiment": 0.51,
        "delta": 0.23,
    }
    assert symbol_payload["visibility_reason"] == {
        "baseline": "limited_quality",
        "experiment": "recovered_visibility",
        "changed": True,
    }


def test_build_report_tracks_loaded_flags() -> None:
    baseline = _make_summary()
    experiment = _make_summary()

    result = build_experimental_relabel_comparison_report(
        baseline,
        experiment,
        baseline_path=Path("baseline.json"),
        experiment_path=Path("experiment.json"),
        baseline_loaded=False,
        experiment_loaded=True,
    )

    assert result["metadata"]["baseline_loaded"] is False
    assert result["metadata"]["experiment_loaded"] is True
