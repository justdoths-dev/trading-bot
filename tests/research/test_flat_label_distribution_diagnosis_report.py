from __future__ import annotations

import json
from pathlib import Path

from src.research.flat_label_distribution_diagnosis_report import (
    build_flat_label_distribution_diagnosis_markdown,
    build_flat_label_distribution_diagnosis_report,
    write_flat_label_distribution_diagnosis_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_flat_label_distribution_report_computes_overall_and_thresholds(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"

    _write_json(
        latest_dir / "summary.json",
        {
            "by_strategy": {
                "swing": {
                    "horizon_summary": {
                        "15m": {
                            "median_future_return_pct": -0.01,
                            "avg_future_return_pct": -0.005,
                            "positive_rate_pct": 15.0,
                            "flat_rate_pct": 70.0,
                        },
                        "1h": {
                            "median_future_return_pct": 0.02,
                            "avg_future_return_pct": 0.03,
                            "positive_rate_pct": 55.0,
                            "flat_rate_pct": 25.0,
                        },
                    }
                }
            },
            "by_symbol": {
                "BTCUSDT": {
                    "horizon_summary": {
                        "15m": {
                            "median_future_return_pct": -0.02,
                            "avg_future_return_pct": -0.01,
                            "positive_rate_pct": 10.0,
                            "flat_rate_pct": 75.0,
                        }
                    }
                }
            },
        },
    )

    summary = build_flat_label_distribution_diagnosis_report(latest_dir)

    overall = summary["overall_flat_distribution_overview"]
    assert overall["total_evaluated_rows"] == 3
    assert overall["flat_dominant_count"] == 2
    assert overall["flat_dominant_ratio"] == 0.6667

    horizon = summary["horizon_breakdown"]
    assert horizon["15m"]["flat_dominant_count"] == 2
    assert horizon["15m"]["threshold_buckets"]["flat_rate_ge_70"]["count"] == 2
    assert horizon["1h"]["flat_dominant_count"] == 0

    category = summary["category_breakdown"]
    assert category["strategy"]["evaluated_rows"] == 2
    assert category["symbol"]["evaluated_rows"] == 1

    interactions = summary["interaction_breakdown"]
    assert interactions["flat_dominant_and_non_positive_median_count"] == 2
    assert interactions["flat_dominant_and_avg_gt_zero_count"] == 0

    examples = summary["representative_examples"]
    assert len(examples) == 2
    assert examples[0]["origin_file"] == "summary.json"


def test_build_flat_label_distribution_report_detects_conflict_case(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"

    _write_json(
        latest_dir / "summary.json",
        {
            "by_symbol": {
                "ETHUSDT": {
                    "horizon_summary": {
                        "4h": {
                            "median_future_return_pct": -0.01,
                            "avg_future_return_pct": 0.02,
                            "positive_rate_pct": 52.0,
                            "flat_rate_pct": 53.0,
                        }
                    }
                }
            }
        },
    )

    summary = build_flat_label_distribution_diagnosis_report(latest_dir)
    interactions = summary["interaction_breakdown"]

    assert interactions["flat_dominant_and_non_positive_median_count"] == 1
    assert interactions["flat_dominant_and_avg_gt_zero_count"] == 1
    assert interactions["flat_dominant_and_positive_rate_ge_50_count"] == 1


def test_markdown_and_write_outputs(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"

    _write_json(
        latest_dir / "summary.json",
        {
            "by_strategy": {
                "scalping": {
                    "horizon_summary": {
                        "15m": {
                            "median_future_return_pct": -0.30,
                            "avg_future_return_pct": -0.20,
                            "positive_rate_pct": 5.0,
                            "flat_rate_pct": 80.0,
                        }
                    }
                }
            }
        },
    )

    summary = build_flat_label_distribution_diagnosis_report(latest_dir)
    markdown = build_flat_label_distribution_diagnosis_markdown(summary)

    assert "Flat Label Distribution Diagnosis" in markdown
    assert "Overall Flat Distribution Overview" in markdown
    assert "Final Diagnosis" in markdown

    outputs = write_flat_label_distribution_diagnosis_report(
        summary=summary,
        json_output_path=latest_dir / "out" / "summary.json",
        markdown_output_path=latest_dir / "out" / "summary.md",
    )

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["summary_md"]).exists()

    written_json = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    written_md = Path(outputs["summary_md"]).read_text(encoding="utf-8")

    assert written_json["source_targeting"]["main_rows_count"] == 1
    assert "Representative Examples" in written_md