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


def test_load_normalized_rows_recovers_latest_and_cumulative_rows(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "latest_candidate_metrics.json",
        {
            "rows": [
                {
                    "source": "latest",
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "swing",
                    "rank": 1,
                    "median_future_return_pct": -0.2,
                    "avg_future_return_pct": 0.3,
                    "positive_rate_pct": 55.0,
                    "flat_rate_pct": 35.0,
                    "labeled_count": 12,
                }
            ]
        },
    )
    _write_json(
        tmp_path / "cumulative_candidate_metrics.json",
        {
            "rows": [
                {
                    "source": "cumulative",
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "swing",
                    "rank": 1,
                    "median_future_return_pct": 0.4,
                    "avg_future_return_pct": 0.5,
                    "positive_rate_pct": 60.0,
                    "flat_rate_pct": 20.0,
                    "labeled_count": 50,
                }
            ]
        },
    )

    rows = load_normalized_rows(tmp_path)

    assert len(rows) == 2
    assert {row.source for row in rows} == {"latest", "cumulative"}
    assert {row.horizon for row in rows} == {"15m"}
    assert {row.category for row in rows} == {"strategy"}


def test_build_non_positive_median_diagnosis_report_computes_required_sections(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "latest_report.json",
        {
            "comparison_rows": [
                {
                    "source": "latest",
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "swing",
                    "rank": 1,
                    "median_future_return_pct": -0.10,
                    "avg_future_return_pct": 0.20,
                    "positive_rate_pct": 55.0,
                    "flat_rate_pct": 30.0,
                    "labeled_count": 10,
                },
                {
                    "source": "latest",
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "intraday",
                    "rank": 2,
                    "median_future_return_pct": -0.05,
                    "avg_future_return_pct": -0.01,
                    "positive_rate_pct": 40.0,
                    "flat_rate_pct": 45.0,
                    "labeled_count": 7,
                },
                {
                    "source": "latest",
                    "horizon": "1h",
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "rank": 1,
                    "median_future_return_pct": -0.20,
                    "avg_future_return_pct": 0.10,
                    "positive_rate_pct": 52.0,
                    "flat_rate_pct": 38.0,
                    "labeled_count": 8,
                },
                {
                    "source": "latest",
                    "horizon": "4h",
                    "category": "alignment_state",
                    "group": "aligned",
                    "rank": 1,
                    "median_future_return_pct": 0.30,
                    "avg_future_return_pct": 0.35,
                    "positive_rate_pct": 65.0,
                    "flat_rate_pct": 20.0,
                    "labeled_count": 20,
                },
            ]
        },
    )
    _write_json(
        tmp_path / "cumulative_report.json",
        {
            "comparison_rows": [
                {
                    "source": "cumulative",
                    "horizon": "15m",
                    "category": "strategy",
                    "group": "swing",
                    "rank": 1,
                    "median_future_return_pct": 0.25,
                    "avg_future_return_pct": 0.40,
                    "positive_rate_pct": 62.0,
                    "flat_rate_pct": 18.0,
                    "labeled_count": 100,
                },
                {
                    "source": "cumulative",
                    "horizon": "1h",
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "rank": 1,
                    "median_future_return_pct": 0.15,
                    "avg_future_return_pct": 0.22,
                    "positive_rate_pct": 57.0,
                    "flat_rate_pct": 28.0,
                    "labeled_count": 80,
                },
            ]
        },
    )

    summary = build_non_positive_median_diagnosis_report(tmp_path)

    overall = summary["overall_median_blocker_overview"]
    assert overall["total_evaluated_rows"] == 4
    assert overall["non_positive_median_count"] == 3
    assert overall["positive_median_count"] == 1
    assert overall["non_positive_median_ratio"] == 0.75

    horizon_breakdown = summary["horizon_breakdown"]
    assert horizon_breakdown["15m"]["non_positive_median_count"] == 2
    assert horizon_breakdown["1h"]["non_positive_median_count"] == 1
    assert horizon_breakdown["4h"]["positive_median_count"] == 1

    category_breakdown = summary["category_breakdown"]
    assert category_breakdown["strategy"]["non_positive_median_count"] == 2
    assert category_breakdown["symbol"]["non_positive_median_count"] == 1
    assert category_breakdown["alignment_state"]["positive_median_count"] == 1

    metric_interactions = summary["metric_interaction_breakdown"]
    assert metric_interactions["median_le_zero_and_avg_gt_zero_count"] == 2
    assert metric_interactions["median_le_zero_and_positive_rate_ge_50_count"] == 2
    assert metric_interactions["median_le_zero_and_flat_rate_dominant_count"] == 1
    assert metric_interactions["median_le_zero_and_labeled_count_sufficient_count"] == 3

    latest_vs_cumulative = summary["latest_vs_cumulative_summary"]
    assert latest_vs_cumulative["pair_count"] == 2
    assert latest_vs_cumulative["latest_non_positive_while_cumulative_positive_count"] == 2
    assert latest_vs_cumulative["latest_non_positive_while_cumulative_positive_ratio"] == 1.0

    rank_scope = summary["rank_scope_breakdown"]
    assert rank_scope["top1"]["non_positive_median_count"] == 2
    assert rank_scope["top3"]["non_positive_median_count"] == 3

    final_diagnosis = summary["final_diagnosis"]
    assert final_diagnosis["worst_horizon"] == "15m"
    assert final_diagnosis["worst_category"] == "strategy"
    assert "mean_positive_but_median_non_positive_conflict_exists" in final_diagnosis["diagnosis_labels"]
    assert "latest_window_noise_dominates_median_signal" in final_diagnosis["diagnosis_labels"]


def test_markdown_and_write_outputs(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "latest_rows.json",
        {
            "rows": [
                {
                    "source": "latest",
                    "horizon": "4h",
                    "category": "alignment_state",
                    "group": "aligned",
                    "rank": 1,
                    "median_future_return_pct": -0.01,
                    "avg_future_return_pct": 0.02,
                    "positive_rate_pct": 51.0,
                    "flat_rate_pct": 33.0,
                    "labeled_count": 9,
                }
            ]
        },
    )

    summary = build_non_positive_median_diagnosis_report(tmp_path)
    markdown = build_non_positive_median_diagnosis_markdown(summary)

    assert "Non-Positive Median Diagnosis" in markdown
    assert "Overall Median Blocker Overview" in markdown
    assert "Final Diagnosis" in markdown

    outputs = write_non_positive_median_diagnosis_report(
        summary=summary,
        json_output_path=tmp_path / "out" / "summary.json",
        markdown_output_path=tmp_path / "out" / "summary.md",
    )

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["summary_md"]).exists()

    written_json = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    written_md = Path(outputs["summary_md"]).read_text(encoding="utf-8")

    assert written_json["overall_median_blocker_overview"]["non_positive_median_count"] == 1
    assert "Representative Examples" in written_md