from __future__ import annotations

import json
from pathlib import Path

from src.research.preview_gate_failure_diagnosis_report import (
    build_preview_gate_failure_diagnosis_summary,
    load_summary_json,
    render_preview_gate_failure_diagnosis_markdown,
    run_preview_gate_failure_diagnosis_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ranked_group(
    *,
    group: str,
    rank: int = 1,
    score: float = 10.0,
    sample_count: int = 100,
    labeled_count: int = 80,
    coverage_pct: float = 80.0,
    median_future_return_pct: float | None = 0.5,
    up_rate_pct: float | None = 55.0,
    signal_match_rate_pct: float | None = 60.0,
    bias_match_rate_pct: float | None = 58.0,
) -> dict:
    return {
        "group": group,
        "rank": rank,
        "score": score,
        "metrics": {
            "sample_count": sample_count,
            "labeled_count": labeled_count,
            "coverage_pct": coverage_pct,
            "median_future_return_pct": median_future_return_pct,
            "up_rate_pct": up_rate_pct,
            "signal_match_rate_pct": signal_match_rate_pct,
            "bias_match_rate_pct": bias_match_rate_pct,
        },
    }


def _candidate_preview(
    *,
    group: str = "swing",
    strength: str = "weak",
    sample_gate: str = "passed",
    quality_gate: str = "borderline",
) -> dict:
    return {
        "group": group,
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": strength,
        "visibility_reason": (
            "passed_sample_and_quality_gate"
            if quality_gate == "passed"
            else "passed_sample_gate_only"
        ),
    }


def _ranking_wrapper(rows: list[dict]) -> dict:
    return {
        "ranked_groups": rows,
    }


def _edge_wrapper(count: int) -> dict:
    return {
        "edge_findings": [{} for _ in range(count)],
    }


def _summary_payload(
    *,
    preview_by_horizon: dict,
    ranking_by_horizon: dict,
    edge_by_horizon: dict,
) -> dict:
    return {
        "schema_validation": {
            "input_path": "logs/trade_analysis.jsonl",
            "rotation_aware": True,
            "source_file_count": 3,
            "raw_record_count": 1000,
            "windowed_record_count": 800,
            "total_records": 800,
            "valid_records": 800,
            "invalid_records": 0,
        },
        "edge_candidates_preview": {
            "by_horizon": preview_by_horizon,
        },
        "strategy_lab": {
            "ranking": ranking_by_horizon,
            "edge": edge_by_horizon,
        },
    }


def test_normal_dataset_case(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"

    payload = _summary_payload(
        preview_by_horizon={
            "15m": {
                "top_strategy": _candidate_preview(group="swing", strength="weak", quality_gate="borderline"),
                "top_symbol": _candidate_preview(group="btcusdt", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_alignment_state": _candidate_preview(group="aligned", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            },
            "1h": {
                "top_strategy": _candidate_preview(group="swing", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_symbol": _candidate_preview(group="btcusdt", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_alignment_state": _candidate_preview(group="aligned", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            },
            "4h": {
                "top_strategy": _candidate_preview(group="intraday", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_symbol": _candidate_preview(group="ethusdt", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_alignment_state": _candidate_preview(group="misaligned", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            },
        },
        ranking_by_horizon={
            "15m": {
                "by_strategy": _ranking_wrapper([
                    _ranked_group(group="swing", median_future_return_pct=0.2, up_rate_pct=45.0),
                ]),
                "by_symbol": _ranking_wrapper([
                    _ranked_group(group="btcusdt", median_future_return_pct=None, up_rate_pct=None),
                ]),
                "by_alignment_state": _ranking_wrapper([
                    _ranked_group(group="aligned", sample_count=10, median_future_return_pct=0.2, up_rate_pct=60.0),
                ]),
            },
            "1h": {
                "by_strategy": _ranking_wrapper([
                    _ranked_group(group="swing", median_future_return_pct=0.3, up_rate_pct=40.0),
                ]),
                "by_symbol": _ranking_wrapper([
                    _ranked_group(group="btcusdt", median_future_return_pct=0.1, up_rate_pct=44.0),
                ]),
                "by_alignment_state": _ranking_wrapper([
                    _ranked_group(group="aligned", median_future_return_pct=0.05, up_rate_pct=42.0),
                ]),
            },
            "4h": {
                "by_strategy": _ranking_wrapper([
                    _ranked_group(group="intraday", median_future_return_pct=0.4, up_rate_pct=46.0),
                ]),
                "by_symbol": _ranking_wrapper([
                    _ranked_group(group="ethusdt", median_future_return_pct=0.6, up_rate_pct=49.0),
                ]),
                "by_alignment_state": _ranking_wrapper([
                    _ranked_group(group="misaligned", median_future_return_pct=0.8, up_rate_pct=47.0),
                ]),
            },
        },
        edge_by_horizon={
            "15m": {
                "by_symbol": _edge_wrapper(0),
                "by_strategy": _edge_wrapper(2),
                "by_alignment_state": _edge_wrapper(1),
                "by_ai_execution_state": _edge_wrapper(0),
            },
            "1h": {
                "by_symbol": _edge_wrapper(0),
                "by_strategy": _edge_wrapper(2),
                "by_alignment_state": _edge_wrapper(0),
                "by_ai_execution_state": _edge_wrapper(0),
            },
            "4h": {
                "by_symbol": _edge_wrapper(0),
                "by_strategy": _edge_wrapper(1),
                "by_alignment_state": _edge_wrapper(1),
                "by_ai_execution_state": _edge_wrapper(0),
            },
        },
    )

    _write_json(path, payload)

    result = run_preview_gate_failure_diagnosis_report(path)
    summary = result["summary"]
    markdown = result["markdown"]

    assert summary["preview_overview"]["total_candidate_slots"] == 9
    assert summary["preview_overview"]["visible_candidate_slots"] == 1
    assert "Executive Summary" in markdown
    assert "Final Diagnosis" in markdown

    overall_reasons = {
        row["value"]: row["count"]
        for row in summary["gate_failures"]["overall_reason_distribution"]
    }
    assert overall_reasons["positive_rate_below_minimum_or_missing"] >= 1
    assert overall_reasons["sample_count_below_minimum"] >= 1
    assert overall_reasons["median_future_return_non_positive_or_missing"] >= 1


def test_empty_file_case(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")

    loaded = load_summary_json(path)
    summary = build_preview_gate_failure_diagnosis_summary(loaded, path)

    assert loaded == {}
    assert summary["preview_overview"]["total_candidate_slots"] == 9
    assert summary["preview_overview"]["visible_candidate_slots"] == 0


def test_missing_file_case(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    result = run_preview_gate_failure_diagnosis_report(path)
    summary = result["summary"]

    assert summary["preview_overview"]["visible_candidate_slots"] == 0
    assert summary["diagnosis"]["primary_issue"] in {
        "preview_gate_blocks_all_candidate_slots",
        "preview_gate_blocks_visibility_despite_detected_edge_activity",
    }


def test_positive_rate_dominant_case(tmp_path: Path) -> None:
    path = tmp_path / "positive_rate_case.json"

    ranking_by_horizon = {}
    preview_by_horizon = {}
    edge_by_horizon = {}

    for horizon in ("15m", "1h", "4h"):
        preview_by_horizon[horizon] = {
            "top_strategy": _candidate_preview(group="swing", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            "top_symbol": _candidate_preview(group="btcusdt", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            "top_alignment_state": _candidate_preview(group="aligned", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
        }
        ranking_by_horizon[horizon] = {
            "by_strategy": _ranking_wrapper([
                _ranked_group(group="swing", median_future_return_pct=0.2, up_rate_pct=40.0),
            ]),
            "by_symbol": _ranking_wrapper([
                _ranked_group(group="btcusdt", median_future_return_pct=0.3, up_rate_pct=45.0),
            ]),
            "by_alignment_state": _ranking_wrapper([
                _ranked_group(group="aligned", median_future_return_pct=0.4, up_rate_pct=49.0),
            ]),
        }
        edge_by_horizon[horizon] = {
            "by_symbol": _edge_wrapper(0),
            "by_strategy": _edge_wrapper(1),
            "by_alignment_state": _edge_wrapper(1),
            "by_ai_execution_state": _edge_wrapper(0),
        }

    _write_json(
        path,
        _summary_payload(
            preview_by_horizon=preview_by_horizon,
            ranking_by_horizon=ranking_by_horizon,
            edge_by_horizon=edge_by_horizon,
        ),
    )

    summary = run_preview_gate_failure_diagnosis_report(path)["summary"]
    assert (
        summary["diagnosis"]["primary_issue"]
        == "positive_rate_absolute_minimum_is_dominant_preview_blocker"
    )


def test_visible_candidate_case(tmp_path: Path) -> None:
    path = tmp_path / "visible_case.json"

    payload = _summary_payload(
        preview_by_horizon={
            "15m": {
                "top_strategy": _candidate_preview(group="swing", strength="moderate", quality_gate="passed"),
                "top_symbol": _candidate_preview(group="btcusdt", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
                "top_alignment_state": _candidate_preview(group="aligned", strength="insufficient_data", sample_gate="failed", quality_gate="failed"),
            },
            "1h": {},
            "4h": {},
        },
        ranking_by_horizon={
            "15m": {
                "by_strategy": _ranking_wrapper([
                    _ranked_group(group="swing", median_future_return_pct=0.6, up_rate_pct=60.0),
                ]),
                "by_symbol": _ranking_wrapper([]),
                "by_alignment_state": _ranking_wrapper([]),
            },
            "1h": {},
            "4h": {},
        },
        edge_by_horizon={
            "15m": {
                "by_symbol": _edge_wrapper(0),
                "by_strategy": _edge_wrapper(1),
                "by_alignment_state": _edge_wrapper(0),
                "by_ai_execution_state": _edge_wrapper(0),
            },
            "1h": {},
            "4h": {},
        },
    )

    _write_json(path, payload)

    summary = run_preview_gate_failure_diagnosis_report(path)["summary"]
    assert summary["preview_overview"]["visible_candidate_slots"] == 1
    assert summary["preview_overview"]["insufficient_candidate_slots"] == 8


def test_markdown_render_case(tmp_path: Path) -> None:
    path = tmp_path / "markdown_case.json"

    payload = _summary_payload(
        preview_by_horizon={},
        ranking_by_horizon={},
        edge_by_horizon={},
    )
    _write_json(path, payload)

    summary = run_preview_gate_failure_diagnosis_report(path)["summary"]
    markdown = render_preview_gate_failure_diagnosis_markdown(summary)

    assert "Preview Gate Failure Diagnosis Report" in markdown
    assert "Overall Gate Failure Reasons" in markdown