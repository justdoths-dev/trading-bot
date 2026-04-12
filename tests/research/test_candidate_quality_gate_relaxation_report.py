from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.candidate_quality_gate import apply_candidate_quality_gate
from src.research.diagnostics import candidate_quality_gate_relaxation_report


def _candidate(
    *,
    symbol: str,
    strategy: str = "swing",
    horizon: str = "4h",
) -> dict[str, str]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
    }


def _selected_record(
    *,
    selected_symbol: str,
    selected_strategy: str = "swing",
    selected_horizon: str = "4h",
    return_value: float | int | str | None,
    return_field_horizon: str = "4h",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "future_return_15m": None,
        "future_return_1h": None,
        "future_return_4h": None,
        "edge_selection_output": {
            "selection_status": "selected",
            "selected_symbol": selected_symbol,
            "selected_strategy": selected_strategy,
            "selected_horizon": selected_horizon,
        },
    }
    payload[f"future_return_{return_field_horizon}"] = return_value
    return payload


def _analysis_row(
    *,
    candidates: list[dict[str, str]] | None = None,
    gate_block: dict[str, object] | None = None,
) -> dict[str, object]:
    mapper_payload: dict[str, object] = {}
    if candidates is not None:
        mapper_payload["candidates"] = candidates
    if gate_block is not None:
        mapper_payload["candidate_quality_gate"] = gate_block
    return {"edge_selection_mapper_payload": mapper_payload}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _write_comparison_summary(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _gate_snapshot(
    candidates: list[dict[str, str]],
    trade_analysis_path: Path,
) -> dict[str, object]:
    result = apply_candidate_quality_gate(
        candidates,
        trade_analysis_path=trade_analysis_path,
    )
    return {
        "input_path_used": result["input_path_used"],
        "total_candidates": result["total_candidates"],
        "strict_kept_count": result["strict_kept_count"],
        "strict_kept_candidates": result["strict_kept_candidates"],
        "strict_dropped_count": result["strict_dropped_count"],
        "strict_dropped_candidates": result["strict_dropped_candidates"],
        "fallback_applied": result["fallback_applied"],
        "fallback_restored_count": result["fallback_restored_count"],
        "fallback_restored_candidates": result["fallback_restored_candidates"],
        "final_kept_count": result["final_kept_count"],
        "final_kept_candidates": result["final_kept_candidates"],
    }


def test_build_report_aggregates_counts_and_recovers_gate_block_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trade_analysis_path = tmp_path / "trade_analysis.jsonl"
    comparison_summary_path = tmp_path / "comparison_summary.json"

    strong_candidate = _candidate(symbol="BTCUSDT")
    sample_near_miss_candidate = _candidate(symbol="ETHUSDT")
    positive_rate_near_miss_candidate = _candidate(symbol="SOLUSDT")
    kept_median_near_miss_candidate = _candidate(symbol="BNBUSDT")

    evidence_rows: list[dict[str, object]] = []
    evidence_rows.extend(
        _selected_record(selected_symbol="BTCUSDT", return_value=0.6) for _ in range(30)
    )
    evidence_rows.extend(
        _selected_record(selected_symbol="ETHUSDT", return_value=0.5) for _ in range(25)
    )
    evidence_rows.extend(
        _selected_record(selected_symbol="SOLUSDT", return_value=1.0) for _ in range(13)
    )
    evidence_rows.extend(
        _selected_record(selected_symbol="SOLUSDT", return_value=0.0) for _ in range(17)
    )
    evidence_rows.extend(
        _selected_record(selected_symbol="BNBUSDT", return_value=0.0) for _ in range(15)
    )
    evidence_rows.extend(
        _selected_record(selected_symbol="BNBUSDT", return_value=0.2) for _ in range(15)
    )

    _write_jsonl(trade_analysis_path, evidence_rows)

    gate_block = _gate_snapshot(
        [sample_near_miss_candidate, positive_rate_near_miss_candidate],
        trade_analysis_path,
    )

    full_rows = list(evidence_rows)
    full_rows.append(
        _analysis_row(
            candidates=[
                strong_candidate,
                positive_rate_near_miss_candidate,
                kept_median_near_miss_candidate,
            ]
        )
    )
    full_rows.append(_analysis_row(gate_block=gate_block))
    _write_jsonl(trade_analysis_path, full_rows)

    _write_comparison_summary(
        comparison_summary_path,
        {
            "drift_notes": [
                "15m: candidate_visibility_increased",
                "strategy: stability_strengthened",
            ]
        },
    )

    monkeypatch.setattr(
        candidate_quality_gate_relaxation_report,
        "_utc_now_iso",
        lambda: "2026-04-10T00:00:00+00:00",
    )

    report = candidate_quality_gate_relaxation_report.build_candidate_quality_gate_relaxation_report(
        trade_analysis_path=trade_analysis_path,
        comparison_summary_path=comparison_summary_path,
    )

    assert report["generated_at"] == "2026-04-10T00:00:00+00:00"
    assert report["summary"]["candidate_sets_analyzed"] == 2
    assert report["summary"]["input_candidate_count"] == 5
    assert report["summary"]["strict_kept_count"] == 3
    assert report["summary"]["strict_dropped_count"] == 2
    assert report["summary"]["fallback_applied"] is False
    assert report["summary"]["fallback_applied_row_count"] == 0
    assert report["summary"]["fallback_restored_candidate_count"] == 0
    assert report["summary"]["drop_reason_counts"] == {
        "positive_rate_pct_below_minimum": 2,
    }
    assert report["summary"]["near_miss_counts"] == {
        "positive_rate_pct_40_00_to_49_99": 2,
        "sample_count_20_29": 1,
        "median_return_pct_0_00_to_0_24": 3,
    }
    assert report["summary"]["comparison_drift_note_counts"] == {
        "15m: candidate_visibility_increased": 1,
        "strategy: stability_strengthened": 1,
    }
    assert report["data_quality"]["candidate_row_source_counts"] == {
        "candidate_quality_gate": 1,
        "mapper_candidates": 1,
    }
    assert any(
        "comparison_drift_note_counts was derived from comparison_summary.drift_notes; downstream outcome fields were not available."
        == note
        for note in report["summary"]["notes"]
    )
    assert any(
        "near_miss_counts are counted per candidate occurrence and are not deduplicated by candidate identity."
        == note
        for note in report["summary"]["notes"]
    )


def test_build_report_handles_missing_comparison_summary(tmp_path: Path) -> None:
    trade_analysis_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _selected_record(selected_symbol="BTCUSDT", return_value=0.6),
        _analysis_row(candidates=[_candidate(symbol="BTCUSDT")]),
    ]
    _write_jsonl(trade_analysis_path, rows)

    report = candidate_quality_gate_relaxation_report.build_candidate_quality_gate_relaxation_report(
        trade_analysis_path=trade_analysis_path,
    )

    assert report["summary"]["comparison_drift_note_counts"] == {}
    assert any(
        "No comparison summary was provided" in note
        for note in report["summary"]["notes"]
    )


def test_build_report_handles_partial_comparison_summary(tmp_path: Path) -> None:
    trade_analysis_path = tmp_path / "trade_analysis.jsonl"
    comparison_summary_path = tmp_path / "comparison_summary.json"

    rows = [
        _selected_record(selected_symbol="BTCUSDT", return_value=0.6),
        _analysis_row(candidates=[_candidate(symbol="BTCUSDT")]),
    ]
    _write_jsonl(trade_analysis_path, rows)
    _write_comparison_summary(comparison_summary_path, {"comparison_summary": {}})

    report = candidate_quality_gate_relaxation_report.build_candidate_quality_gate_relaxation_report(
        trade_analysis_path=trade_analysis_path,
        comparison_summary_path=comparison_summary_path,
    )

    assert report["summary"]["comparison_drift_note_counts"] == {}
    assert any(
        "drift_notes is unavailable" in note for note in report["summary"]["notes"]
    )


def test_build_report_requires_exact_trade_analysis_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_trade_analysis.jsonl"

    with pytest.raises(FileNotFoundError):
        candidate_quality_gate_relaxation_report.build_candidate_quality_gate_relaxation_report(
            trade_analysis_path=missing_path,
        )


def test_write_report_files_uses_deterministic_names(tmp_path: Path, monkeypatch) -> None:
    trade_analysis_path = tmp_path / "trade_analysis.jsonl"
    output_dir = tmp_path / "reports"

    rows = [
        _selected_record(selected_symbol="BTCUSDT", return_value=0.6),
        _analysis_row(candidates=[_candidate(symbol="BTCUSDT")]),
    ]
    _write_jsonl(trade_analysis_path, rows)

    monkeypatch.setattr(
        candidate_quality_gate_relaxation_report,
        "_utc_now_iso",
        lambda: "2026-04-10T00:00:00+00:00",
    )

    report = candidate_quality_gate_relaxation_report.build_candidate_quality_gate_relaxation_report(
        trade_analysis_path=trade_analysis_path,
    )
    written_paths = candidate_quality_gate_relaxation_report.write_candidate_quality_gate_relaxation_report(
        report,
        output_dir,
    )

    json_path = output_dir / "candidate_quality_gate_relaxation_report.json"
    md_path = output_dir / "candidate_quality_gate_relaxation_report.md"

    assert written_paths == {
        "json_report": str(json_path.resolve()),
        "markdown_report": str(md_path.resolve()),
    }
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Candidate Quality Gate Relaxation Report" in markdown
    assert "- input_candidate_count: 1" in markdown


def test_main_writes_report_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    trade_analysis_path = tmp_path / "trade_analysis.jsonl"
    output_dir = tmp_path / "reports"

    rows = [
        _selected_record(selected_symbol="BTCUSDT", return_value=0.6),
        _analysis_row(candidates=[_candidate(symbol="BTCUSDT")]),
    ]
    _write_jsonl(trade_analysis_path, rows)

    monkeypatch.setattr(
        candidate_quality_gate_relaxation_report,
        "_utc_now_iso",
        lambda: "2026-04-10T00:00:00+00:00",
    )

    candidate_quality_gate_relaxation_report.main(
        [
            "--trade-analysis",
            str(trade_analysis_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)

    assert payload["report_type"] == "candidate_quality_gate_relaxation_report"
    assert payload["candidate_sets_analyzed"] == 1
    assert payload["input_candidate_count"] == 1
    assert payload["fallback_applied"] is True
    assert Path(payload["written_paths"]["json_report"]).exists()
    assert Path(payload["written_paths"]["markdown_report"]).exists()