from __future__ import annotations

import json
from pathlib import Path

from src.research.non_positive_median_diagnosis_report import (
    build_non_positive_median_diagnosis_markdown,
    build_non_positive_median_diagnosis_report,
    load_main_metric_rows,
    load_probe_pair_rows,
    write_non_positive_median_diagnosis_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_main_metric_rows_reads_only_metric_complete_summary_rows(tmp_path: Path) -> None:
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
                            "avg_future_return_pct": 0.03,
                            "positive_rate_pct": 51.0,
                            "flat_rate_pct": 44.0,
                            "labeled_count": 10,
                        },
                        {
                            "group": "intraday",
                            "median_future_return_pct": -0.02,
                            "labeled_count": 7,
                        },
                    ]
                }
            }
        },
    )

    rows, stats = load_main_metric_rows(latest_dir)

    assert len(rows) == 1
    assert rows[0].source == "latest"
    assert rows[0].origin_file == "summary.json"
    assert rows[0].horizon == "15m"
    assert rows[0].category == "strategy"
    assert rows[0].group == "swing"
    assert rows[0].rank == 1
    assert stats["main_rows_count"] == 1
    assert stats["metric_complete_rows_count"] == 1


def test_load_probe_pair_rows_reads_probe_only_as_auxiliary_pairs(tmp_path: Path) -> None:
    latest_dir = tmp_path / "latest"

    _write_json(
        latest_dir / "latest_cumulative_fallback_probe_summary.json",
        {
            "representative_examples": {
                "1h": [
                    {
                        "category": "symbol",
                        "group": "BTCUSDT",
                        "latest_top_median_future_return_pct": -0.05,
                        "cumulative_top_median_future_return_pct": 0.12,
                        "latest_top_avg_future_return_pct": 0.01,
                        "cumulative_top_avg_future_return_pct": 0.16,
                    }
                ]
            }
        },
    )

    rows, stats = load_probe_pair_rows(latest_dir)

    assert len(rows) == 1
    assert rows[0].horizon == "1h"
    assert rows[0].category == "symbol"
    assert rows[0].group == "BTCUSDT"
    assert rows[0].latest_median_future_return_pct == -0.05
    assert rows[0].cumulative_median_future_return_pct == 0.12
    assert stats["auxiliary_probe_rows_count"] == 1
    assert stats["pair_rows_count"] == 1


def test_build_report_uses_summary_for_main_diagnosis_and_probe_for_pairs(tmp_path: Path) -> None:
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
                            "median_future_return_pct": -0.03,
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
            "representative_examples": {
                "15m": [
                    {
                        "category": "strategy",
                        "group": "swing",
                        "latest_top_median_future_return_pct": -0.10,
                        "cumulative_top_median_future_return_pct": 0.25,
                        "latest_top_avg_future_return_pct": 0.20,
                        "cumulative_top_avg_future_return_pct": 0.40,
                        "latest_top_positive_rate_pct": 55.0,
                        "cumulative_top_positive_rate_pct": 62.0,
                        "latest_top_flat_rate_pct": 35.0,
                        "cumulative_top_flat_rate_pct": 18.0,
                    }
                ],
                "1h": [
                    {
                        "category": "symbol",
                        "group": "BTCUSDT",
                        "latest_top_median_future_return_pct": -0.20,
                        "cumulative_top_median_future_return_pct": 0.15,
                        "latest_top_avg_future_return_pct": 0.10,
                        "cumulative_top_avg_future_return_pct": 0.22,
                    }
                ],
            }
        },
    )

    summary = build_non_positive_median_diagnosis_report(latest_dir)

    source_targeting = summary["source_targeting"]
    assert source_targeting["main_diagnosis_source"] == "summary.json"
    assert source_targeting["auxiliary_pair_source"] == "latest_cumulative_fallback_probe_summary.json"
    assert source_targeting["main_rows_count"] == 4
    assert source_targeting["metric_complete_rows_count"] == 4
    assert source_targeting["pair_rows_count"] == 2

    overall = summary["overall_median_blocker_overview"]
    assert overall["total_evaluated_rows"] == 4
    assert overall["non_positive_median_count"] == 3
    assert overall["positive_median_count"] == 1
    assert overall["non_positive_median_ratio"] == 0.75

    metric_interactions = summary["metric_interaction_breakdown"]
    assert metric_interactions["median_le_zero_and_avg_gt_zero_count"] == 2
    assert metric_interactions["median_le_zero_and_positive_rate_ge_50_count"] == 2
    assert metric_interactions["median_le_zero_and_flat_rate_dominant_count"] == 1
    assert metric_interactions["median_le_zero_and_labeled_count_sufficient_count"] == 3

    latest_vs_cumulative = summary["latest_vs_cumulative_summary"]
    assert latest_vs_cumulative["pair_count"] == 2
    assert latest_vs_cumulative["latest_non_positive_while_cumulative_positive_count"] == 2
    assert latest_vs_cumulative["latest_non_positive_while_cumulative_positive_ratio"] == 1.0

    examples = summary["representative_examples"]
    assert len(examples) == 3
    assert all(example["origin_file"] == "summary.json" for example in examples)

    final_diagnosis = summary["final_diagnosis"]
    assert "non_positive_median_is_broad_across_rankings" in final_diagnosis["diagnosis_labels"]
    assert "mean_positive_but_median_non_positive_conflict_exists" in final_diagnosis["diagnosis_labels"]
    assert "positive_rate_strength_conflicts_with_non_positive_median" in final_diagnosis["diagnosis_labels"]
    assert "latest_window_noise_dominates_median_signal" in final_diagnosis["diagnosis_labels"]


def test_markdown_and_write_outputs_include_source_targeting(tmp_path: Path) -> None:
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

    assert "Source Targeting" in markdown
    assert "main_diagnosis_source: summary.json" in markdown
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

    assert written_json["source_targeting"]["main_rows_count"] == 1
    assert "Representative Examples" in written_md