from __future__ import annotations

import json
from pathlib import Path

from src.research.edge_selection_input_mapper import map_edge_selection_input


def test_valid_upstream_reports_produce_normalized_payload(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["latest_window_record_count"] == 124
    assert payload["cumulative_record_count"] == 3578
    assert payload["history_line_count"] is None
    assert payload["warnings"] == [
        f"Optional upstream report is missing: {base_dir / 'edge_scores_history.jsonl'}"
    ]
    assert payload["candidates"] == [
        {
            "symbol": "BTCUSDT",
            "strategy": "swing",
            "horizon": "4h",
            "selected_candidate_strength": "moderate",
            "selected_stability_label": "single_horizon_only",
            "source_preference": "latest",
            "edge_stability_score": 3.2,
            "drift_direction": "decrease",
            "score_delta": -0.4,
        }
    ]


def test_missing_required_report_returns_failure(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)
    (base_dir / "comparison" / "summary.json").unlink()

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is False
    assert payload["candidates"] == []
    assert any("Missing required upstream report" in error for error in payload["errors"])


def test_malformed_json_returns_failure(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)
    (base_dir / "score_drift" / "summary.json").write_text("{not-json", encoding="utf-8")

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is False
    assert payload["candidates"] == []
    assert any("not valid JSON" in error for error in payload["errors"])


def test_optional_history_file_missing_is_still_valid(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert "candidates" in payload
    assert len(payload["warnings"]) == 1
    assert payload["history_line_count"] is None


def test_optional_history_file_present_sets_line_count(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)
    history_path = base_dir / "edge_scores_history.jsonl"
    history_path.write_text('{"score": 1}\n\n{"score": 2}\n', encoding="utf-8")

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert payload["history_line_count"] == 2


def test_normalized_payload_always_contains_candidates(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(
        tmp_path,
        comparison_summary={
            "dataset_overview_comparison": {
                "latest_total_records": 124,
                "cumulative_total_records": 3578,
            },
            "edge_candidates_comparison": {},
        },
    )

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["candidates"] == []


def test_payload_metadata_fields_present(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(tmp_path)

    payload = map_edge_selection_input(base_dir)

    assert isinstance(payload["generated_at"], str)
    assert payload["latest_window_record_count"] == 124
    assert payload["cumulative_record_count"] == 3578
    assert "errors" in payload
    assert "warnings" in payload
    assert "history_line_count" in payload


def test_source_preference_na_uses_available_values_without_inventing(tmp_path: Path) -> None:
    base_dir = _write_valid_reports(
        tmp_path,
        edge_scores_summary={
            "generated_at": "2026-03-15T00:02:00+00:00",
            "edge_stability_scores": {
                "symbol": [
                    {
                        "group": "BTCUSDT",
                        "score": 3.2,
                        "latest_stability_label": "single_horizon_only",
                        "cumulative_stability_label": "unstable",
                        "latest_candidate_strength": "moderate",
                        "cumulative_candidate_strength": "weak",
                        "source_preference": "n/a",
                    }
                ],
                "strategy": [],
                "alignment_state": [],
            },
        },
    )

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert payload["candidates"] == [
        {
            "symbol": "BTCUSDT",
            "strategy": "swing",
            "horizon": "4h",
            "selected_candidate_strength": "moderate",
            "selected_stability_label": "single_horizon_only",
            "source_preference": "n/a",
            "edge_stability_score": 3.2,
            "drift_direction": "decrease",
            "score_delta": -0.4,
        }
    ]


def _write_valid_reports(
    tmp_path: Path,
    *,
    latest_summary: dict | None = None,
    comparison_summary: dict | None = None,
    edge_scores_summary: dict | None = None,
    score_drift_summary: dict | None = None,
) -> Path:
    base_dir = tmp_path / "logs" / "research_reports"
    (base_dir / "latest").mkdir(parents=True, exist_ok=True)
    (base_dir / "comparison").mkdir(parents=True, exist_ok=True)
    (base_dir / "edge_scores").mkdir(parents=True, exist_ok=True)
    (base_dir / "score_drift").mkdir(parents=True, exist_ok=True)

    _write_json(
        base_dir / "latest" / "summary.json",
        latest_summary
        or {
            "generated_at": "2026-03-15T00:00:00+00:00",
            "dataset_overview": {"total_records": 124},
        },
    )
    _write_json(
        base_dir / "comparison" / "summary.json",
        comparison_summary
        or {
            "generated_at": "2026-03-15T00:01:00+00:00",
            "dataset_overview_comparison": {
                "latest_total_records": 124,
                "cumulative_total_records": 3578,
            },
            "edge_candidates_comparison": {
                "4h": {
                    "latest_candidate_strength": "moderate",
                    "cumulative_candidate_strength": "weak",
                    "latest_top_symbol_group": "BTCUSDT",
                    "cumulative_top_symbol_group": "BTCUSDT",
                    "latest_top_strategy_group": "swing",
                    "cumulative_top_strategy_group": "swing",
                }
            },
        },
    )
    _write_json(
        base_dir / "edge_scores" / "summary.json",
        edge_scores_summary
        or {
            "generated_at": "2026-03-15T00:02:00+00:00",
            "edge_stability_scores": {
                "symbol": [
                    {
                        "group": "BTCUSDT",
                        "score": 3.2,
                        "latest_stability_label": "single_horizon_only",
                        "cumulative_stability_label": "unstable",
                        "latest_candidate_strength": "moderate",
                        "cumulative_candidate_strength": "weak",
                        "source_preference": "latest",
                    }
                ],
                "strategy": [
                    {
                        "group": "swing",
                        "score": 3.0,
                        "latest_stability_label": "single_horizon_only",
                        "cumulative_stability_label": "single_horizon_only",
                        "latest_candidate_strength": "moderate",
                        "cumulative_candidate_strength": "moderate",
                        "source_preference": "latest",
                    }
                ],
                "alignment_state": [],
            },
        },
    )
    _write_json(
        base_dir / "score_drift" / "summary.json",
        score_drift_summary
        or {
            "generated_at": "2026-03-15T00:03:00+00:00",
            "score_drift": [
                {
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "drift_direction": "decrease",
                    "score_delta": -0.4,
                }
            ],
        },
    )

    return base_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
