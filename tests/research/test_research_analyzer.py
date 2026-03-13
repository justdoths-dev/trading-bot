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


def test_run_research_analyzer_handles_empty_input_safely(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: (_ for _ in ()).throw(AssertionError("strategy lab should be skipped")),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["dataset_overview"]["total_records"] == 0
    assert result["schema_validation"]["total_records"] == 0
    assert result["schema_validation"]["valid_records"] == 0
    assert result["schema_validation"]["invalid_records"] == 0
    assert result["strategy_lab"]["dataset_rows"] == 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()


def test_run_research_analyzer_returns_structurally_valid_summary(
    monkeypatch,
    tmp_path: Path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    input_path = write_jsonl([valid_research_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: _stub_strategy_lab_metrics(dataset_rows=1),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)
    written_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    expected_top_level_keys = {
        "dataset_overview",
        "horizon_summary",
        "by_symbol",
        "by_strategy",
        "schema_validation",
        "strategy_lab",
    }

    assert expected_top_level_keys.issubset(result.keys())
    assert expected_top_level_keys.issubset(written_summary.keys())
    assert set(result["schema_validation"].keys()) == {
        "input_path",
        "total_records",
        "valid_records",
        "invalid_records",
        "error_count",
        "warning_count",
        "invalid_examples",
    }
