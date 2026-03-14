from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.research import research_analyzer


def _stub_strategy_lab_metrics(dataset_rows: int = 0) -> dict[str, Any]:
    return {
        "dataset_rows": dataset_rows,
        "performance": {},
        "comparison": {},
        "ranking": {},
        "edge": {},
        "segment": {},
    }


def _candidate(group: str, strength: str = "moderate") -> dict[str, Any]:
    return {
        "group": group,
        "sample_count": 60,
        "labeled_count": 60,
        "coverage_pct": 100.0,
        "median_future_return_pct": 0.45,
        "positive_rate_pct": 58.0,
        "signal_match_rate_pct": 57.0,
        "bias_match_rate_pct": 56.0,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": 57.0,
        "sample_gate": "passed",
        "quality_gate": "passed" if strength in {"moderate", "strong"} else "borderline",
        "candidate_strength": strength,
        "visibility_reason": "passed_sample_and_quality_gate",
        "chosen_metric_summary": "sample=60, median=0.45, positive_rate=58.0",
    }


def _insufficient_candidate() -> dict[str, Any]:
    return {
        "group": "insufficient_data",
        "sample_count": 0,
        "labeled_count": 0,
        "coverage_pct": None,
        "median_future_return_pct": None,
        "positive_rate_pct": None,
        "signal_match_rate_pct": None,
        "bias_match_rate_pct": None,
        "robustness_signal": "n/a",
        "robustness_signal_pct": None,
        "sample_gate": "failed",
        "quality_gate": "failed",
        "candidate_strength": "insufficient_data",
        "visibility_reason": "failed_absolute_minimum_gate",
        "chosen_metric_summary": "insufficient_data",
    }


def _edge_candidates_preview(by_horizon: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_sample_count": 30,
        "strength_thresholds": {
            "weak": {
                "sample_count": 30,
                "median_future_return_pct_gt": 0,
                "positive_rate_pct": 50,
            },
            "moderate": {
                "sample_count": 50,
                "median_future_return_pct": 0.30,
                "positive_rate_pct": 55.0,
                "robustness_pct": 52.0,
            },
            "strong": {
                "sample_count": 80,
                "median_future_return_pct": 0.50,
                "positive_rate_pct": 58.0,
                "robustness_pct": 55.0,
            },
        },
        "by_horizon": by_horizon,
    }


def test_run_research_analyzer_handles_valid_records(
    monkeypatch,
    tmp_path: Path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    second_record = deepcopy(valid_research_record)
    second_record["symbol"] = "ETHUSDT"
    second_record["selected_strategy"] = "mean_reversion"
    second_record["future_label_15m"] = "down"
    second_record["future_return_15m"] = -0.4

    input_path = write_jsonl([valid_research_record, second_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: _stub_strategy_lab_metrics(dataset_rows=2),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["dataset_overview"]["total_records"] == 2
    assert result["schema_validation"]["valid_records"] == 2
    assert result["schema_validation"]["invalid_records"] == 0
    assert result["strategy_lab"]["dataset_rows"] == 2
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()


def test_run_research_analyzer_skips_invalid_records_without_crashing(
    monkeypatch,
    tmp_path: Path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    invalid_record = deepcopy(valid_research_record)
    invalid_record["risk"] = "invalid"

    input_path = write_jsonl([valid_research_record, invalid_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: (_ for _ in ()).throw(AssertionError("strategy lab should be skipped")),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["dataset_overview"]["total_records"] == 1
    assert result["schema_validation"]["total_records"] == 2
    assert result["schema_validation"]["valid_records"] == 1
    assert result["schema_validation"]["invalid_records"] == 1
    assert len(result["schema_validation"]["invalid_examples"]) == 1
    assert result["strategy_lab"]["dataset_rows"] == 0


def test_edge_stability_preview_returns_insufficient_data_without_visible_candidates() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "insufficient_data"
    assert result["strategy"]["stability_score"] == 0


def test_edge_stability_preview_returns_single_horizon_only_for_one_visible_group() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_strategy": _candidate("swing"),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "single_horizon_only"
    assert result["strategy"]["stability_score"] == 1


def test_edge_stability_preview_returns_multi_horizon_confirmed_for_repeated_group() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {"top_strategy": _candidate("swing")},
            "1h": {"top_strategy": _candidate("swing")},
            "4h": {"top_strategy": _insufficient_candidate()},
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "multi_horizon_confirmed"
    assert result["strategy"]["stability_score"] == 2


def test_edge_stability_preview_returns_unstable_for_different_visible_groups() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {"top_strategy": _candidate("alpha")},
            "1h": {"top_strategy": _candidate("beta")},
            "4h": {"top_strategy": _insufficient_candidate()},
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "unstable"
    assert result["strategy"]["stability_score"] == 1
