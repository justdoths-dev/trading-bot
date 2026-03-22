from __future__ import annotations

import json
from pathlib import Path

from src.research.experimental_candidate_comparison_matrix import (
    build_experimental_candidate_comparison_matrix,
    load_jsonl_records,
    run_experimental_candidate_comparison_matrix,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _candidate_a_rows() -> list[dict]:
    return [
        {
            "symbol": "BTCUSDT",
            "selected_strategy": "swing",
            "future_return_15m": 0.06,
            "future_return_1h": -0.02,
            "future_return_4h": 0.20,
            "future_label_15m": "up",
            "future_label_1h": "flat",
            "future_label_4h": "up",
        },
        {
            "symbol": "ETHUSDT",
            "selected_strategy": "swing",
            "future_return_15m": -0.08,
            "future_return_1h": 0.10,
            "future_return_4h": -0.30,
            "future_label_15m": "down",
            "future_label_1h": "up",
            "future_label_4h": "down",
        },
        {
            "symbol": "ADAUSDT",
            "selected_strategy": "scalp",
            "future_return_15m": 0.01,
            "future_return_1h": 0.00,
            "future_return_4h": 0.02,
            "future_label_15m": "flat",
            "future_label_1h": "flat",
            "future_label_4h": "flat",
        },
    ]


def _candidate_b_rows() -> list[dict]:
    return [
        {
            "symbol": "BTCUSDT",
            "selected_strategy": "swing",
            "future_return_15m": 0.06,
            "future_return_1h": -0.02,
            "future_return_4h": 0.20,
            "future_label_15m": "up",
            "future_label_1h": "flat",
            "future_label_4h": "flat",
            "experimental_labeling": {
                "labeling_method": "candidate_b_volatility_adjusted_v1",
                "used_fallback_atr_pct": False,
                "thresholds": {"15m": 0.05, "1h": 0.10, "4h": 0.25},
            },
        },
        {
            "symbol": "ETHUSDT",
            "selected_strategy": "swing",
            "future_return_15m": -0.08,
            "future_return_1h": 0.10,
            "future_return_4h": -0.30,
            "future_label_15m": "down",
            "future_label_1h": "flat",
            "future_label_4h": "down",
            "experimental_labeling": {
                "labeling_method": "candidate_b_volatility_adjusted_v1",
                "used_fallback_atr_pct": True,
                "thresholds": {"15m": 0.05, "1h": 0.12, "4h": 0.15},
            },
        },
        {
            "symbol": "ADAUSDT",
            "selected_strategy": "scalp",
            "future_return_15m": 0.01,
            "future_return_1h": 0.00,
            "future_return_4h": 0.02,
            "future_label_15m": "up",
            "future_label_1h": "flat",
            "future_label_4h": "flat",
            "experimental_labeling": {
                "labeling_method": "candidate_b_volatility_adjusted_v1",
                "used_fallback_atr_pct": False,
                "thresholds": {"15m": 0.01, "1h": 0.10, "4h": 0.15},
            },
        },
        {
            "symbol": "XRPUSDT",
            "selected_strategy": "intraday",
            "future_return_15m": 0.20,
            "future_return_1h": 0.30,
            "future_return_4h": 0.40,
            "future_label_15m": "up",
            "future_label_1h": "up",
            "future_label_4h": "up",
            "experimental_labeling": {
                "labeling_method": "some_other_method",
                "used_fallback_atr_pct": False,
                "thresholds": {"15m": 9.99, "1h": 9.99, "4h": 9.99},
            },
        },
    ]


def test_load_jsonl_records_ignores_blank_invalid_and_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "candidate.jsonl"
    path.write_text(
        '{"symbol": "BTCUSDT"}\n\nnot-json\n[]\n{"symbol": "ETHUSDT"}\n',
        encoding="utf-8",
    )

    records, instrumentation = load_jsonl_records(path)

    assert records == [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}]
    assert instrumentation == {
        "blank_line_count": 1,
        "invalid_json_line_count": 1,
        "non_object_line_count": 1,
    }


def test_comparison_builds_label_distribution_positive_rate_and_medians() -> None:
    summary = build_experimental_candidate_comparison_matrix(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["candidate_a"]["label_distribution_by_horizon"]["15m"]["flat"] == {
        "count": 1,
        "ratio": 0.333333,
    }
    assert summary["candidate_b"]["positive_rate_by_horizon"]["4h"] == {
        "count": 2,
        "ratio": 0.666667,
        "numeric_row_count": 3,
    }
    assert summary["candidate_a"]["label_conditional_median_future_return_by_horizon"]["1h"] == {
        "up": 0.1,
        "down": None,
        "flat": -0.01,
    }


def test_delta_and_final_summary_are_computed() -> None:
    summary = build_experimental_candidate_comparison_matrix(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["delta_a_to_b"]["by_horizon"]["15m"]["flat_ratio_change"] == -0.333333
    assert summary["delta_a_to_b"]["by_horizon"]["15m"]["up_ratio_change"] == 0.333334
    assert summary["delta_a_to_b"]["by_horizon"]["1h"]["positive_rate_change"] == 0.0
    assert summary["final_summary"]["primary_finding"] in {
        "candidate_b_looks_structurally_better_than_candidate_a",
        "candidate_b_results_are_mixed_vs_candidate_a",
        "candidate_b_looks_structurally_worse_than_candidate_a",
    }
    assert summary["final_summary"]["notes"]


def test_candidate_b_volatility_metadata_is_summarized() -> None:
    summary = build_experimental_candidate_comparison_matrix(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["candidate_b"]["volatility_metadata"] == {
        "metadata_row_count": 3,
        "fallback_row_count": 1,
        "fallback_row_ratio": 0.333333,
        "threshold_statistics_by_horizon": {
            "15m": {"min": 0.01, "max": 0.05, "mean": 0.036667, "median": 0.05},
            "1h": {"min": 0.1, "max": 0.12, "mean": 0.106667, "median": 0.1},
            "4h": {"min": 0.15, "max": 0.25, "mean": 0.183333, "median": 0.15},
        },
    }


def test_group_metrics_include_strategy_and_symbol_views() -> None:
    summary = build_experimental_candidate_comparison_matrix(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["candidate_a"]["positive_rate_by_strategy"]["swing"]["by_horizon"]["15m"] == {
        "positive_rate": 0.5,
        "median_future_return": -0.01,
        "numeric_row_count": 2,
    }
    assert summary["candidate_b"]["positive_rate_by_symbol"]["BTCUSDT"]["by_horizon"]["4h"] == {
        "positive_rate": 1.0,
        "median_future_return": 0.2,
        "numeric_row_count": 1,
    }


def test_runner_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    candidate_a_path = tmp_path / "candidate_a.jsonl"
    candidate_b_path = tmp_path / "candidate_b.jsonl"
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"

    _write_jsonl(candidate_a_path, _candidate_a_rows())
    _write_jsonl(candidate_b_path, _candidate_b_rows())

    result = run_experimental_candidate_comparison_matrix(
        candidate_a_path=candidate_a_path,
        candidate_b_path=candidate_b_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["inputs"]["candidate_a_path"] == str(candidate_a_path)
    assert result["summary"]["inputs"]["candidate_b_filtered_row_count"] == 3
    assert json_output.exists()
    assert markdown_output.exists()

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["candidate_b"]["dataset_overview"]["total_row_count"] == 3