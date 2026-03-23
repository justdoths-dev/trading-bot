from __future__ import annotations

import json
from pathlib import Path

from src.research.experimental_candidate_ac_comparison_report import (
    build_experimental_candidate_ac_comparison_report,
    run_experimental_candidate_ac_comparison_report,
)


def _make_summary(
    *,
    total_records: int,
    dataset_rows: int,
    flat_15m: int,
    up_15m: int,
    down_15m: int,
    positive_15m: float,
    median_15m: float,
) -> dict[str, object]:
    return {
        "dataset_overview": {"total_records": total_records},
        "schema_validation": {"valid_records": total_records, "invalid_records": 0},
        "strategy_lab": {
            "dataset_rows": dataset_rows,
            "performance": {
                "15m": {
                    "labeled_records": dataset_rows,
                    "label_distribution": {
                        "up": up_15m,
                        "down": down_15m,
                        "flat": flat_15m,
                    },
                    "median_future_return_pct": median_15m,
                    "positive_rate_pct": positive_15m,
                    "negative_rate_pct": 100.0 - positive_15m,
                    "flat_rate_pct": round(flat_15m / dataset_rows * 100.0, 6),
                },
                "1h": {
                    "labeled_records": dataset_rows,
                    "label_distribution": {"up": 4, "down": 3, "flat": 3},
                    "median_future_return_pct": 0.02,
                    "positive_rate_pct": 53.0,
                    "negative_rate_pct": 47.0,
                    "flat_rate_pct": 30.0,
                },
                "4h": {
                    "labeled_records": dataset_rows,
                    "label_distribution": {"up": 5, "down": 2, "flat": 3},
                    "median_future_return_pct": 0.03,
                    "positive_rate_pct": 56.0,
                    "negative_rate_pct": 44.0,
                    "flat_rate_pct": 30.0,
                },
            },
            "comparison": {
                "15m": {
                    "by_strategy": {
                        "groups": {
                            "swing": {
                                "labeled_records": dataset_rows,
                                "label_distribution": {
                                    "up": up_15m,
                                    "down": down_15m,
                                    "flat": flat_15m,
                                },
                                "median_future_return_pct": median_15m,
                                "positive_rate_pct": positive_15m,
                                "flat_rate_pct": round(flat_15m / dataset_rows * 100.0, 6),
                            }
                        }
                    },
                    "by_symbol": {
                        "groups": {
                            "BTCUSDT": {
                                "labeled_records": dataset_rows,
                                "label_distribution": {
                                    "up": up_15m,
                                    "down": down_15m,
                                    "flat": flat_15m,
                                },
                                "median_future_return_pct": median_15m,
                                "positive_rate_pct": positive_15m,
                                "flat_rate_pct": round(flat_15m / dataset_rows * 100.0, 6),
                            }
                        }
                    },
                }
            },
        },
    }


def test_ac_comparison_report_structure_and_delta_calculation() -> None:
    candidate_a = _make_summary(
        total_records=100,
        dataset_rows=10,
        flat_15m=2,
        up_15m=4,
        down_15m=4,
        positive_15m=50.0,
        median_15m=0.01,
    )
    candidate_c = _make_summary(
        total_records=100,
        dataset_rows=10,
        flat_15m=3,
        up_15m=4,
        down_15m=3,
        positive_15m=56.0,
        median_15m=0.04,
    )

    summary = build_experimental_candidate_ac_comparison_report(
        candidate_a,
        candidate_c,
        baseline_path=Path("candidate_a_summary.json"),
        experiment_path=Path("candidate_c_summary.json"),
    )

    assert summary["metadata"]["baseline_name"] == "candidate_a"
    assert summary["metadata"]["experiment_name"] == "candidate_c"
    assert summary["row_count_comparison"]["total_records"]["delta"] == 0.0
    assert summary["label_distribution_by_horizon"]["15m"]["flat"]["ratio"]["delta"] == 0.1
    assert summary["metric_comparison_by_horizon"]["15m"]["positive_rate_pct"]["delta"] == 6.0
    assert summary["metric_comparison_by_horizon"]["15m"]["median_future_return_pct"]["delta"] == 0.03
    assert summary["final_diagnosis"]["primary_finding"] in {
        "candidate_c_trades_some_coverage_for_purity_vs_candidate_a",
        "candidate_c_looks_more_balanced_than_candidate_a",
        "candidate_c_remains_mixed_vs_candidate_a",
    }


def test_ac_runner_writes_outputs_and_handles_missing_optional_metrics(tmp_path: Path) -> None:
    baseline_path = tmp_path / "candidate_a_summary.json"
    experiment_path = tmp_path / "candidate_c_summary.json"
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"

    baseline_path.write_text(
        json.dumps({"dataset_overview": {"total_records": 12}, "schema_validation": {"valid_records": 12}}),
        encoding="utf-8",
    )
    experiment_path.write_text(
        json.dumps({"dataset_overview": {"total_records": 12}, "schema_validation": {"valid_records": 12}}),
        encoding="utf-8",
    )

    result = run_experimental_candidate_ac_comparison_report(
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["metric_comparison_by_horizon"]["15m"]["positive_rate_pct"]["baseline"] is None
    assert result["summary_json"] == str(json_output)
    assert result["summary_md"] == str(markdown_output)
    assert json_output.exists()
    assert markdown_output.exists()
