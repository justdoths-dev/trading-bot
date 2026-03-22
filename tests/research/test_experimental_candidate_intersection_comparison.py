from __future__ import annotations

import json
from pathlib import Path

from src.research.experimental_candidate_intersection_comparison import (
    build_experimental_candidate_intersection_comparison,
    build_intersection_datasets,
    build_row_match_key,
    run_experimental_candidate_intersection_comparison,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _candidate_a_rows() -> list[dict]:
    return [
        {
            "logged_at": "2026-03-20T00:00:00Z",
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
            "logged_at": "2026-03-20T00:05:00Z",
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
            "logged_at": "2026-03-20T00:10:00Z",
            "symbol": "ADAUSDT",
            "selected_strategy": "scalp",
            "future_return_15m": 0.01,
            "future_return_1h": 0.00,
            "future_return_4h": 0.02,
            "future_label_15m": "flat",
            "future_label_1h": "flat",
            "future_label_4h": "flat",
        },
        {
            "logged_at": "2026-03-20T00:15:00Z",
            "symbol": "SOLUSDT",
            "selected_strategy": "trend",
            "future_return_15m": 0.03,
            "future_return_1h": 0.05,
            "future_return_4h": 0.10,
            "future_label_15m": "up",
            "future_label_1h": "up",
            "future_label_4h": "up",
        },
    ]


def _candidate_b_rows() -> list[dict]:
    return [
        {
            "logged_at": "2026-03-20T00:00:00Z",
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
            "logged_at": "2026-03-20T00:05:00Z",
            "symbol": "ETHUSDT",
            "selected_strategy": "swing",
            "future_return_15m": -0.08,
            "future_return_1h": 0.10,
            "future_return_4h": -0.30,
            "future_label_15m": "up",
            "future_label_1h": "flat",
            "future_label_4h": "down",
            "experimental_labeling": {
                "labeling_method": "candidate_b_volatility_adjusted_v1",
                "used_fallback_atr_pct": True,
                "thresholds": {"15m": 0.05, "1h": 0.12, "4h": 0.15},
            },
        },
        {
            "logged_at": "2026-03-20T00:10:00Z",
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
            "logged_at": "2026-03-20T00:20:00Z",
            "symbol": "XRPUSDT",
            "selected_strategy": "intraday",
            "future_return_15m": 0.20,
            "future_return_1h": 0.30,
            "future_return_4h": 0.40,
            "future_label_15m": "up",
            "future_label_1h": "up",
            "future_label_4h": "up",
            "experimental_labeling": {
                "labeling_method": "candidate_b_volatility_adjusted_v1",
                "used_fallback_atr_pct": False,
                "thresholds": {"15m": 0.20, "1h": 0.30, "4h": 0.40},
            },
        },
        {
            "logged_at": "2026-03-20T00:15:00Z",
            "symbol": "SOLUSDT",
            "selected_strategy": "trend",
            "future_return_15m": 0.03,
            "future_return_1h": 0.05,
            "future_return_4h": 0.10,
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


def test_row_key_generation_is_stable_and_uses_fallback_strategy_field() -> None:
    row = {
        "logged_at": "2026-03-20T00:00:00Z",
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "future_return_15m": "0.0600",
        "future_return_1h": -0.02,
        "future_return_4h": None,
    }

    assert build_row_match_key(row) == (
        "2026-03-20T00:00:00Z",
        "BTCUSDT",
        "swing",
        "0.06",
        "-0.02",
        "__missing__",
    )


def test_safe_matching_with_identical_rows_uses_one_to_one_alignment() -> None:
    duplicate = {
        "logged_at": "2026-03-20T00:00:00Z",
        "symbol": "BTCUSDT",
        "selected_strategy": "swing",
        "future_return_15m": 0.06,
        "future_return_1h": -0.02,
        "future_return_4h": 0.20,
    }

    a_rows = [duplicate, duplicate]
    b_rows = [duplicate, duplicate, duplicate]

    a_intersection, b_intersection, summary = build_intersection_datasets(a_rows, b_rows)

    assert len(a_intersection) == 2
    assert len(b_intersection) == 2
    assert summary["candidate_a_only_count"] == 0
    assert summary["candidate_b_only_count"] == 1


def test_intersection_only_filtering_tracks_a_only_and_b_only_counts() -> None:
    summary = build_experimental_candidate_intersection_comparison(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["intersection_summary"] == {
        "candidate_a_total_rows": 4,
        "candidate_b_total_rows": 4,
        "candidate_a_intersection_count": 3,
        "candidate_b_intersection_count": 3,
        "candidate_a_only_count": 1,
        "candidate_b_only_count": 1,
    }
    assert summary["candidate_a_intersection"]["dataset_overview"]["total_row_count"] == 3
    assert summary["candidate_b_intersection"]["dataset_overview"]["total_row_count"] == 3


def test_candidate_b_method_filtering_is_applied_before_intersection() -> None:
    summary = build_experimental_candidate_intersection_comparison(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert summary["inputs"]["candidate_b_raw_total_rows"] == 5
    assert summary["inputs"]["candidate_b_filtered_row_count"] == 4
    assert summary["inputs"]["candidate_b_required_labeling_method"] == (
        "candidate_b_volatility_adjusted_v1"
    )


def test_delta_calculation_uses_intersection_rows_only() -> None:
    summary = build_experimental_candidate_intersection_comparison(
        _candidate_a_rows(),
        _candidate_b_rows(),
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    delta = summary["delta_a_to_b_on_intersection"]["by_horizon"]["15m"]

    assert delta["flat_ratio_change"] == -0.333333
    assert delta["positive_rate_change"] == 0.0
    assert delta["overall_median_future_return_change"] == 0.0
    assert delta["up_ratio_change"] == 0.666667
    assert delta["down_ratio_change"] == -0.333333
    assert delta["up_bucket_median_change"] == -0.05
    assert delta["down_bucket_median_change"] is None
    assert delta["flat_bucket_median_change"] is None
    assert delta["up_bucket_positive_rate_change"] == -0.333333
    assert delta["down_bucket_positive_rate_change"] == 0.0
    assert delta["flat_bucket_positive_rate_change"] == -1.0


def test_output_generation_writes_json_and_markdown(tmp_path: Path) -> None:
    candidate_a_path = tmp_path / "candidate_a.jsonl"
    candidate_b_path = tmp_path / "candidate_b.jsonl"
    json_output = tmp_path / "comparison.json"
    markdown_output = tmp_path / "comparison.md"

    _write_jsonl(candidate_a_path, _candidate_a_rows())
    _write_jsonl(candidate_b_path, _candidate_b_rows())

    result = run_experimental_candidate_intersection_comparison(
        candidate_a_path=candidate_a_path,
        candidate_b_path=candidate_b_path,
        json_output_path=json_output,
        markdown_output_path=markdown_output,
    )

    assert result["summary"]["intersection_summary"]["candidate_a_intersection_count"] == 3
    assert json_output.exists()
    assert markdown_output.exists()

    loaded = json.loads(json_output.read_text(encoding="utf-8"))
    assert loaded["candidate_b_intersection"]["volatility_metadata"]["fallback_row_count"] == 1
    assert "Candidate A vs Candidate B Intersection Comparison" in markdown_output.read_text(
        encoding="utf-8"
    )


def test_intersection_build_is_deterministic() -> None:
    candidate_a_rows = list(reversed(_candidate_a_rows()))
    candidate_b_rows = list(reversed(_candidate_b_rows()))

    first = build_experimental_candidate_intersection_comparison(
        candidate_a_rows,
        candidate_b_rows,
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )
    second = build_experimental_candidate_intersection_comparison(
        candidate_a_rows,
        candidate_b_rows,
        candidate_a_path=Path("candidate_a.jsonl"),
        candidate_b_path=Path("candidate_b.jsonl"),
    )

    assert first["intersection_summary"] == second["intersection_summary"]
    assert first["delta_a_to_b_on_intersection"] == second["delta_a_to_b_on_intersection"]
    assert first["highlights"] == second["highlights"]


