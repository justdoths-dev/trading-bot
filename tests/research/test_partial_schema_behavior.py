from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.research import research_analyzer


def test_partial_schema_invalid_but_analyzer_runs(
    monkeypatch,
    tmp_path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    invalid_record = deepcopy(valid_research_record)
    invalid_record["future_return_1h"] = "invalid"

    input_path = write_jsonl([valid_research_record, invalid_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: (_ for _ in ()).throw(
            AssertionError("strategy lab should be skipped for mixed-invalid datasets")
        ),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["schema_validation"]["total_records"] == 2
    assert result["schema_validation"]["valid_records"] == 1
    assert result["schema_validation"]["invalid_records"] == 1
    assert result["dataset_overview"]["total_records"] == 1
    assert result["strategy_lab"]["dataset_rows"] == 0
    assert len(result["schema_validation"]["invalid_examples"]) == 1