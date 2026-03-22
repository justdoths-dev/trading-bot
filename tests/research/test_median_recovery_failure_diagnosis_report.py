from __future__ import annotations

import json
from pathlib import Path

from src.research.median_recovery_failure_diagnosis_report import (
    _build_final_diagnosis,
    _build_group_examples,
    _build_horizon_diagnosis_rows,
    build_median_recovery_failure_diagnosis_summary,
    load_summary_json,
    run_median_recovery_failure_diagnosis_report,
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
                "label_distribution": {"up": 24, "down": 18, "flat": 38},
                "avg_future_return_pct": 0.02,
                "median_future_return_pct": -0.03,
                "positive_rate_pct": 30.0,
                "negative_rate_pct": 22.5,
                "flat_rate_pct": 47.5,
            },
            "4h": {
                "labeled_records": 70,
                "label_distribution": {"up": 28, "down": 16, "flat": 26},
                "avg_future_return_pct": 0.04,
                "median_future_return_pct": 0.0,
                "positive_rate_pct": 40.0,
                "negative_rate_pct": 22.857143,
                "flat_rate_pct": 37.142857,
            },
        },
        "by_symbol": {
            "BTCUSDT": {
                "horizon_summary": {
                    "15m": {
                        "labeled_records": 40,
                        "label_distribution": {"up": 9, "down": 7, "flat": 24},
                        "avg_future_return_pct": 0.015,
                        "median_future_return_pct": -0.03,
                        "positive_rate_pct": 22.5,
                        "negative_rate_pct": 17.5,
                        "flat_rate_pct": 60.0,
                    }
                }
            },
            "ETHUSDT": {
                "horizon_summary": {
                    "1h": {
                        "labeled_records": 30,
                        "label_distribution": {"up": 9, "down": 8, "flat": 13},
                        "avg_future_return_pct": 0.01,
                        "median_future_return_pct": -0.02,
                        "positive_rate_pct": 30.0,
                        "negative_rate_pct": 26.666667,
                        "flat_rate_pct": 43.333333,
                    }
                }
            },
            "ADAUSDT": {
                "horizon_summary": {
                    "4h": {
                        "labeled_records": 20,
                        "label_distribution": {"up": 10, "down": 3, "flat": 7},
                        "avg_future_return_pct": 0.02,
                        "median_future_return_pct": 0.01,
                        "positive_rate_pct": 50.0,
                        "negative_rate_pct": 15.0,
                        "flat_rate_pct": 35.0,
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
                        "avg_future_return_pct": 0.02,
                        "median_future_return_pct": -0.03,
                        "positive_rate_pct": 24.0,
                        "negative_rate_pct": 16.0,
                        "flat_rate_pct": 60.0,
                    }
                }
            },
            "intraday": {
                "horizon_summary": {
                    "1h": {
                        "labeled_records": 35,
                        "label_distribution": {"up": 11, "down": 10, "flat": 14},
                        "avg_future_return_pct": 0.01,
                        "median_future_return_pct": -0.01,
                        "positive_rate_pct": 31.428571,
                        "negative_rate_pct": 28.571429,
                        "flat_rate_pct": 40.0,
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
                "edge_strength_still_weak",
            ],
        }
    }


def test_missing_file_case(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    loaded = load_summary_json(missing)
    result = build_median_recovery_failure_diagnosis_summary(
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
        "median_recovery_failure_source_not_found",
        "recovery_is_not_broad_across_horizons",
    }


def test_horizon_logic_identifies_downside_replacement() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["avg_future_return_pct"] = 0.03
    experiment["horizon_summary"]["15m"]["median_future_return_pct"] = -0.035
    experiment["horizon_summary"]["15m"]["positive_rate_pct"] = 23.0
    experiment["horizon_summary"]["15m"]["negative_rate_pct"] = 24.0
    experiment["horizon_summary"]["15m"]["flat_rate_pct"] = 53.0
    experiment["horizon_summary"]["15m"]["label_distribution"]["down"] = 24
    experiment["horizon_summary"]["15m"]["label_distribution"]["flat"] = 53

    rows = _build_horizon_diagnosis_rows(baseline, experiment, {})
    row_15m = rows[0]

    assert row_15m["diagnosis_reason"] == "downside_replacement_is_suppressing_median"
    assert "flat_reduction_without_meaningful_median_recovery" in row_15m["diagnosis_labels"]
    assert "mean_recovery_without_median_recovery" in row_15m["diagnosis_labels"]


def test_final_diagnosis_marks_recovery_as_not_broad() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["flat_rate_pct"] = 52.0
    experiment["horizon_summary"]["15m"]["median_future_return_pct"] = -0.03
    experiment["horizon_summary"]["15m"]["avg_future_return_pct"] = 0.03
    experiment["horizon_summary"]["15m"]["negative_rate_pct"] = 25.0
    experiment["horizon_summary"]["1h"]["flat_rate_pct"] = 41.0
    experiment["horizon_summary"]["1h"]["median_future_return_pct"] = -0.01
    experiment["horizon_summary"]["4h"]["flat_rate_pct"] = 31.0
    experiment["horizon_summary"]["4h"]["median_future_return_pct"] = 0.03

    horizon_rows = _build_horizon_diagnosis_rows(baseline, experiment, {})
    symbol_examples = _build_group_examples(baseline, experiment, category="symbol", limit=6)
    strategy_examples = _build_group_examples(
        baseline,
        experiment,
        category="strategy",
        limit=6,
    )

    diagnosis = _build_final_diagnosis(
        horizon_rows,
        symbol_examples,
        strategy_examples,
        _comparison_payload(),
    )

    assert diagnosis["primary_finding"] == "flat_reduction_without_meaningful_median_recovery"
    assert "recovery_is_not_broad_across_horizons" in diagnosis["diagnosis_labels"]
    assert diagnosis["secondary_finding"] == "experiment_improved_distribution_but_not_selection_ready"


def test_symbol_examples_prioritize_btc_eth_and_keep_horizon_order() -> None:
    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["flat_rate_pct"] = 50.0
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["negative_rate_pct"] = 24.0
    experiment["by_symbol"]["BTCUSDT"]["horizon_summary"]["15m"]["median_future_return_pct"] = -0.028
    experiment["by_symbol"]["ETHUSDT"]["horizon_summary"]["1h"]["flat_rate_pct"] = 35.0
    experiment["by_symbol"]["ETHUSDT"]["horizon_summary"]["1h"]["negative_rate_pct"] = 31.0
    experiment["by_symbol"]["ETHUSDT"]["horizon_summary"]["1h"]["median_future_return_pct"] = -0.019
    experiment["by_symbol"]["ADAUSDT"]["horizon_summary"]["4h"]["flat_rate_pct"] = 28.0
    experiment["by_symbol"]["ADAUSDT"]["horizon_summary"]["4h"]["median_future_return_pct"] = 0.012

    examples = _build_group_examples(baseline, experiment, category="symbol", limit=6)

    assert [row["group"] for row in examples[:2]] == ["BTCUSDT", "ETHUSDT"]
    assert [row["horizon"] for row in examples] == ["15m", "1h", "4h"]


def test_run_report_writes_outputs(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    experiment_path = tmp_path / "experiment.json"
    comparison_path = tmp_path / "comparison.json"
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    baseline = _summary_payload()
    experiment = _summary_payload()
    experiment["horizon_summary"]["15m"]["flat_rate_pct"] = 54.0
    experiment["horizon_summary"]["15m"]["negative_rate_pct"] = 22.0
    experiment["horizon_summary"]["15m"]["avg_future_return_pct"] = 0.025
    experiment["horizon_summary"]["15m"]["median_future_return_pct"] = -0.032
    _write_json(baseline_path, baseline)
    _write_json(experiment_path, experiment)
    _write_json(comparison_path, _comparison_payload())

    result = run_median_recovery_failure_diagnosis_report(
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        comparison_path=comparison_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["final_diagnosis"]["primary_finding"] == (
        "flat_reduction_without_meaningful_median_recovery"
    )
    assert json_output.exists()
    assert markdown_output.exists()
