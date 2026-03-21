from __future__ import annotations

import json
from pathlib import Path

from src.research.non_positive_median_diagnosis_report import (
    build_non_positive_median_diagnosis_markdown,
    build_non_positive_median_diagnosis_report,
    load_normalized_rows,
    write_non_positive_median_diagnosis_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_normalized_rows_defaults_summary_json_in_latest_dir_to_latest_source(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"
    _write_json(
        latest_dir / "summary.json",
        {
            "15m": {
                "strategy": [
                    {
                        "group": "swing",
                        "median_future_return_pct": -0.10,
                        "avg_future_return_pct": 0.02,
                        "positive_rate_pct": 52.0,
                        "flat_rate_pct": 40.0,
                        "labeled_count": 12,
                    }
                ]
            }
        },
    )

    rows, instrumentation = load_normalized_rows(latest_dir)

    assert len(rows) == 1
    assert rows[0].source == "latest"
    assert rows[0].horizon == "15m"
    assert rows[0].category == "strategy"
    assert rows[0].group == "swing"
    assert rows[0].rank == 1
    assert instrumentation["latest_rows_count"] == 1
    assert instrumentation["normalized_rows_count"] == 1


def test_load_normalized_rows_supports_prefixed_latest_and_cumulative_comparison_rows(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"
    _write_json(
        latest_dir / "latest_cumulative_fallback_probe_summary.json",
        {
            "cases": [
                {
                    "horizon": "1h",
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "latest_top_median_future_return_pct": -0.05,
                    "latest_top_avg_future_return_pct": 0.01,
                    "latest_top_positive_rate_pct": 55.0,
                    "latest_top_flat_rate_pct": 35.0,
                    "cumulative_top_median_future_return_pct": 0.12,
                    "cumulative_top_avg_future_return_pct": 0.16,
                    "cumulative_top_positive_rate_pct": 58.0,
                    "cumulative_top_flat_rate_pct": 22.0,
                }
            ]
        },
    )

    rows, instrumentation = load_normalized_rows(latest_dir)

    assert len(rows) == 2
    assert {row.source for row in rows} == {"latest", "cumulative"}
    assert instrumentation["latest_rows_count"] == 1
    assert instrumentation["cumulative_rows_count"] == 1


def test_build_non_positive_median_diagnosis_report_computes_required_sections(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"

    _write_json(
        latest_dir / "summary.json",
        {
            "latest_candidate_rankings": {
                "15m": {
                    "strategy": [
                        {
                            "group": "swing",
                            "median_future_return_pct": -0.10,
                            "avg_future_return_pct": 0.20,
                            "positive_rate_pct": 55.0,
                            "flat_rate_pct": 35.0,
                            "labeled_count": 10,
                        },
                        {
                            "group": "intraday",
                            "median_future_return_pct": -0.04,
                            "avg_future_return_pct": -0.01,
                            "positive_rate_pct": 41.0,
                            "flat_rate_pct": 47.0,
                            "labeled_count": 8,
                        },
                    ]
                },
                "1h": {
                    "symbol": [
                        {
                            "group": "BTCUSDT",
                            "median_future_return_pct": -0.20,
                            "avg_future_return_pct": 0.10,
                            "positive_rate_pct": 52.0,
                            "flat_rate_pct": 38.0,
                            "labeled_count": 7,
                        }
                    ]
                },
                "4h": {
                    "alignment_state": [
                        {
                            "group": "aligned",
                            "median_future_return_pct": 0.30,
                            "avg_future_return_pct": 0.35,
                            "positive_rate_pct": 65.0,
                            "flat_rate_pct": 20.0,
                            "labeled_count": 20,
                        }
                    ]
                },
            }
        },
    )

    _write_json(
        latest_dir / "latest_cumulative_fallback_probe_summary.json",
        {
            "cases": [
                {
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "swing",
                    "latest_top_median_future_return_pct": -0.10,
                    "latest_top_avg_future_return_pct": 0.20,
                    "latest_top_positive_rate_pct": 55.0,
                    "latest_top_flat_rate_pct": 35.0,
                    "cumulative_top_median_future_return_pct": 0.25,
                    "cumulative_top_avg_future_return_pct": 0.40,
                    "cumulative_top_positive_rate_pct": 62.0,
                    "cumulative_top_flat_rate_pct": 18.0,
                },
                {
                    "horizon": "1h",
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "latest_top_median_future_return_pct": -0.20,
                    "latest_top_avg_future_return_pct": 0.10,
                    "latest_top_positive_rate_pct": 52.0,
                    "latest_top_flat_rate_pct": 38.0,
                    "cumulative_top_median_future_return_pct": 0.15,
                    "cumulative_top_avg_future_return_pct": 0.22,
                    "cumulative_top_positive_rate_pct": 57.0,
                    "cumulative_top_flat_rate_pct": 28.0,
                },
            ]
        },
    )

    summary = build_non_positive_median_diagnosis_report(latest_dir)

    overall = summary["overall_median_blocker_overview"]
    assert overall["total_normalized_rows"] >= 6
    assert overall["total_evaluated_rows"] >= 4
    assert overall["non_positive_median_count"] >= 3
    assert overall["positive_median_count"] >= 1

    horizon_breakdown = summary["horizon_breakdown"]
    assert horizon_breakdown["15m"]["non_positive_median_count"] >= 2
    assert horizon_breakdown["1h"]["non_positive_median_count"] >= 1
    assert horizon_breakdown["4h"]["positive_median_count"] >= 1

    category_breakdown = summary["category_breakdown"]
    assert category_breakdown["strategy"]["non_positive_median_count"] >= 2
    assert category_breakdown["symbol"]["non_positive_median_count"] >= 1
    assert category_breakdown["alignment_state"]["positive_median_count"] >= 1

    metric_interactions = summary["metric_interaction_breakdown"]
    assert metric_interactions["median_le_zero_and_avg_gt_zero_count"] >= 2
    assert metric_interactions["median_le_zero_and_positive_rate_ge_50_count"] >= 2
    assert metric_interactions["median_le_zero_and_labeled_count_sufficient_count"] >= 3

    latest_vs_cumulative = summary["latest_vs_cumulative_summary"]
    assert latest_vs_cumulative["pair_count"] >= 2
    assert latest_vs_cumulative["latest_non_positive_while_cumulative_positive_count"] >= 2

    parser_stats = summary["parser_instrumentation"]
    assert parser_stats["normalized_rows_count"] == overall["total_normalized_rows"]
    assert parser_stats["latest_rows_count"] == overall["total_evaluated_rows"]

    final_diagnosis = summary["final_diagnosis"]
    assert final_diagnosis["primary_finding"] != "normalization_failure_or_schema_mismatch"


def test_markdown_and_write_outputs_include_parser_stats(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"
    _write_json(
        latest_dir / "summary.json",
        {
            "4h": {
                "alignment_state": [
                    {
                        "group": "aligned",
                        "median_future_return_pct": -0.01,
                        "avg_future_return_pct": 0.02,
                        "positive_rate_pct": 51.0,
                        "flat_rate_pct": 33.0,
                        "labeled_count": 9,
                    }
                ]
            }
        },
    )

    summary = build_non_positive_median_diagnosis_report(latest_dir)
    markdown = build_non_positive_median_diagnosis_markdown(summary)

    assert "Parser Instrumentation" in markdown
    assert "Overall Median Blocker Overview" in markdown
    assert "Final Diagnosis" in markdown

    outputs = write_non_positive_median_diagnosis_report(
        summary=summary,
        json_output_path=latest_dir / "out" / "summary.json",
        markdown_output_path=latest_dir / "out" / "summary.md",
    )

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["summary_md"]).exists()

    written_json = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    written_md = Path(outputs["summary_md"]).read_text(encoding="utf-8")

    assert written_json["parser_instrumentation"]["normalized_rows_count"] >= 1
    assert "Representative Examples" in written_md