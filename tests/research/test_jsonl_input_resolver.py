from __future__ import annotations

import json
from pathlib import Path

from src.research.inputs.jsonl_input_resolver import (
    discover_jsonl_layout,
    render_resolution_text,
    resolve_jsonl_inputs,
)
from src.research.inputs.jsonl_snapshot_manifest import create_jsonl_snapshot_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_discovery_and_default_resolution(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    _write_jsonl(logs_root / "trade_analysis.jsonl", [{"id": 1}])
    _write_jsonl(logs_root / "trade_analysis_cumulative.jsonl", [{"id": 1}, {"id": 2}])
    _write_jsonl(logs_root / "archive" / "trade_analysis_older.jsonl", [{"id": 3}, {"id": 4}])
    _write_jsonl(logs_root / "backups" / "trade_analysis_backup.jsonl", [{"id": 5}])
    _write_jsonl(logs_root / "experiments" / "candidate_a.jsonl", [{"id": 6}])
    _write_jsonl(logs_root / "research_reports" / "edge_scores_history.jsonl", [{"id": 7}])
    _write_jsonl(logs_root / "edge_selection_shadow" / "edge_selection_shadow.jsonl", [{"id": 8}])

    discovery = discover_jsonl_layout(logs_root)
    categories = {row["relative_path"]: row["category"] for row in discovery["files"]}
    line_counts = {row["relative_path"]: row["line_count"] for row in discovery["files"]}

    assert categories["trade_analysis.jsonl"] == "primary_production_current"
    assert categories["trade_analysis_cumulative.jsonl"] == "primary_production_cumulative"
    assert categories["archive/trade_analysis_older.jsonl"] == "archive_historical_candidate"
    assert categories["backups/trade_analysis_backup.jsonl"] == "backup_snapshot_candidate"
    assert categories["experiments/candidate_a.jsonl"] == "experiment_input"
    assert categories["research_reports/edge_scores_history.jsonl"] == "derived_report_artifact"
    assert categories["edge_selection_shadow/edge_selection_shadow.jsonl"] == "derived_shadow_output"
    assert line_counts["archive/trade_analysis_older.jsonl"] == 2

    current = resolve_jsonl_inputs("current", logs_root=logs_root)
    historical = resolve_jsonl_inputs("historical", logs_root=logs_root)
    experiment = resolve_jsonl_inputs("experiment", logs_root=logs_root)

    assert current["included_relative_paths"] == ["trade_analysis.jsonl"]
    assert historical["included_relative_paths"] == ["trade_analysis_cumulative.jsonl"]
    assert historical["record_ordering"] == "file_order_only"
    assert experiment["included_relative_paths"] == ["experiments/candidate_a.jsonl"]

    rendered = render_resolution_text(historical)
    assert "trade_analysis_cumulative.jsonl" in rendered
    assert "archive/trade_analysis_older.jsonl" in rendered
    assert "excluded by default because overlap with cumulative is assumed but not yet validated" in rendered


def test_archive_and_current_historical_opt_in(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    _write_jsonl(logs_root / "trade_analysis.jsonl", [{"id": 1}])
    _write_jsonl(logs_root / "trade_analysis_cumulative.jsonl", [{"id": 2}])
    _write_jsonl(logs_root / "archive" / "trade_analysis_older.jsonl", [{"id": 3}])

    historical_with_archive = resolve_jsonl_inputs(
        "historical",
        logs_root=logs_root,
        include_archive_candidates=True,
    )
    assert historical_with_archive["included_relative_paths"] == [
        "trade_analysis_cumulative.jsonl",
        "archive/trade_analysis_older.jsonl",
    ]

    historical_with_current = resolve_jsonl_inputs(
        "historical",
        logs_root=logs_root,
        include_current_with_historical=True,
    )
    assert historical_with_current["include_current_with_historical"] is True
    assert historical_with_current["included_relative_paths"] == [
        "trade_analysis_cumulative.jsonl",
        "trade_analysis.jsonl",
    ]


def test_manifest_counts_only_included_files(tmp_path: Path) -> None:
    logs_root = tmp_path / "logs"
    _write_jsonl(logs_root / "trade_analysis.jsonl", [{"id": 1}])
    _write_jsonl(logs_root / "trade_analysis_cumulative.jsonl", [{"id": 1}, {"id": 2}, {"id": 3}])
    _write_jsonl(logs_root / "backups" / "trade_analysis_backup.jsonl", [{"id": 4}, {"id": 5}])

    result = create_jsonl_snapshot_manifest(
        "historical",
        logs_root=logs_root,
        output_path=tmp_path / "manifest.json",
    )

    manifest = result["manifest"]
    assert manifest["mode"] == "historical"
    assert manifest["file_count"] == 1
    assert manifest["include_archive_candidates"] is False
    assert manifest["include_current_with_historical"] is False
    assert manifest["included_relative_paths"] == ["trade_analysis_cumulative.jsonl"]
    assert manifest["file_list"] == [str(logs_root / "trade_analysis_cumulative.jsonl")]
    assert manifest["files"][0]["line_count"] == 3
    assert manifest["total_lines"] == 3
    assert manifest["decision_summary"] == result["selection"]["decision_summary"]
    assert Path(result["output_path"]).exists() is True
