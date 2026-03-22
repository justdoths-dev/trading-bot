from __future__ import annotations

import json
from pathlib import Path

from src.research.positive_rate_bottleneck_diagnosis_report import (
    _build_final_diagnosis,
    _build_group_examples,
    _build_horizon_rows,
    build_positive_rate_bottleneck_diagnosis_summary,
    load_summary_json,
    run_positive_rate_bottleneck_diagnosis_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary_payload() -> dict[str, object]:
    return {
        "horizon_summary": {
            "15m": {
                "labeled_records": 100,
                "label_distribution": {"up": 20, "down": 15, "flat": 65},
                "avg_future_return_pct": 0.01,
                "median_future_return_pct": -0.04,
                "positive_rate_pct": 20.0,
                "negative_rate_pct": 15.0,
                "flat_rate_pct": 65.0,
            },
            "1h": {
                "labeled_records": 80,
                "label_distribution": {"up": 28, "down": 18, "flat": 34},
                "avg_future_return_pct": 0.02,
                "median_future_return_pct": -0.01,
                "positive_rate_pct": 35.0,
                "negative_rate_pct": 22.5,
                "flat_rate_pct": 42.5,
            },
            "4h": {
                "labeled_records": 70,
                "label_distribution": {"up": 32, "down": 15, "flat": 23},
                "avg_future_return_pct": 0.03,
                "median_future_return_pct": 0.01,
                "positive_rate_pct": 45.0,
                "negative_rate_pct": 21.428571,
                "flat_rate_pct": 32.857143,
            },
        },
        "by_symbol": {
            "BTCUSDT": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 40,
                        "label_distribution": {"up": 10, "down": 7, "flat": 23},
                        "median_future_return_pct": -0.02,
                        "positive_rate_pct": 25.0,
                        "flat_rate_pct": 57.5,
                    }
                }
            },
            "ETHUSDT": {
                "horizon_summary": {
                    "4h": {
                        "labeled_records": 30,
                        "label_distribution": {"up": 15, "down": 5, "flat": 10},
                        "median_future_return_pct": 0.02,
                        "positive_rate_pct": 50.0,
                        "flat_rate_pct": 33.333333,
                    }
                }
            },
            "ADAUSDT": {
                "horizon_summary": {
                    "1h": {
                        "labeled_records": 20,
                        "label_distribution": {"up": 7, "down": 4, "flat": 9},
                        "median_future_return_pct": -0.01,
                        "positive_rate_pct": 35.0,
                        "flat_rate_pct": 45.0,
                    }
                }
            },
        },
        "by_strategy": {
            "swing": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 50,
                        "label_distribution": {"up": 12, "down": 8, "flat": 30},
                        "median_future_return_pct": -0.03,
                        "positive_rate_pct": 24.0,
                        "flat_rate_pct": 60.0,
                    }
                }
            },
            "intraday": {
                "horizon_summary": {
                    "4h": {
                        "labeled_records": 35,
                        "label_distribution": {"up": 18, "down": 7, "flat": 10},
                        "median_future_return_pct": 0.03,
                        "positive_rate_pct": 51.428571,
                        "flat_rate_pct": 28.571429,
                    }
                }
            },
        },
    }


def _comparison_payload() -> dict[str, object]:
    return {
        "final_diagnosis": {
            "primary_finding": "experiment_improved_distribution_but_not_selection_ready",
            "diagnosis_labels": [
                "meaningful_flat_suppression_reduction_detected",
                "positive_rate_still_below_selection_grade",
            ],
        }
    }


def test_missing_file_case(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    loaded = load_summary_json(missing)
    result = build_positive_rate_bottleneck_diagnosis_summary(
        loaded,
        {},
        {},
        baseline_path=missing,
        experiment_path=missing,
        comparison_path=missing,
    )

    assert loaded == {}
    assert result["parser_instrumentation"]["baseline_loaded"] is False
    assert result["final_diagnosis"]["primary_finding"] in {
        "positive_rate_bottleneck_source_not_found",
        "positive_rate_remains_below_selection_grade_across_horizons",
    }


def test_horizon_rows_compute_selection_gap() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["positive_rate_pct"] = 31.0
    experiment["horizon_summary"]["1h"]["positive_rate_pct"] = 44.0
    experiment["horizon_summary"]["4h"]["positive_rate_pct"] = 49.0

    rows = _build_horizon_rows(baseline, experiment)

    assert rows[0]["selection_gap_pct"] == 24.0
    assert rows[0]["positive_rate_pct"]["delta"] == 11.0
    assert rows[2]["selection_gap_pct"] == 6.0


def test_final_diagnosis_marks_short_horizon_bottleneck() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["positive_rate_pct"] = 28.0
    experiment["horizon_summary"]["1h"]["positive_rate_pct"] = 47.0
    experiment["horizon_summary"]["4h"]["positive_rate_pct"] = 52.0

    horizon_rows = _build_horizon_rows(baseline, experiment)
    symbol_closest = _build_group_examples(
        baseline, experiment, category="symbol", limit=6, closest=True
    )
    symbol_blockers = _build_group_examples(
        baseline, experiment, category="symbol", limit=6, closest=False
    )
    strategy_closest = _build_group_examples(
        baseline, experiment, category="strategy", limit=6, closest=True
    )
    strategy_blockers = _build_group_examples(
        baseline, experiment, category="strategy", limit=6, closest=False
    )

    diagnosis = _build_final_diagnosis(
        horizon_rows,
        symbol_closest,
        symbol_blockers,
        strategy_closest,
        strategy_blockers,
        _comparison_payload(),
    )

    assert diagnosis["primary_finding"] == (
        "positive_rate_remains_below_selection_grade_across_horizons"
    )
    assert "short_horizon_positive_rate_is_primary_bottleneck" in diagnosis["diagnosis_labels"]
    assert diagnosis["secondary_finding"] == "experiment_improved_distribution_but_not_selection_ready"


def test_symbol_prioritization_and_deterministic_ordering() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["positive_rate_pct"] = 27.0
    experiment["by_symbol"]["ETHUSDT"]["horizon_summary"]["4h"]["positive_rate_pct"] = 52.0
    experiment["by_symbol"]["ADAUSDT"]["horizon_summary"]["1h"]["positive_rate_pct"] = 36.0

    closest = _build_group_examples(
        baseline, experiment, category="symbol", limit=6, closest=True
    )
    blockers = _build_group_examples(
        baseline, experiment, category="symbol", limit=6, closest=False
    )

    assert [row["group"] for row in closest[:2]] == ["BTCUSDT", "ETHUSDT"]
    assert blockers[0]["group"] == "BTCUSDT"


def test_run_report_writes_outputs(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    experiment_path = tmp_path / "experiment.json"
    comparison_path = tmp_path / "comparison.json"
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["positive_rate_pct"] = 29.0
    experiment["horizon_summary"]["1h"]["positive_rate_pct"] = 43.0
    experiment["horizon_summary"]["4h"]["positive_rate_pct"] = 50.0
    _write_json(baseline_path, baseline)
    _write_json(experiment_path, experiment)
    _write_json(comparison_path, _comparison_payload())

    result = run_positive_rate_bottleneck_diagnosis_report(
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        comparison_path=comparison_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["final_diagnosis"]["primary_finding"] == (
        "positive_rate_remains_below_selection_grade_across_horizons"
    )
    assert json_output.exists()
    assert markdown_output.exists()
