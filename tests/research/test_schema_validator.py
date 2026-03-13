from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.research.schema_validator import validate_jsonl_file, validate_record


def test_validate_record_accepts_valid_record(valid_research_record: dict[str, object]) -> None:
    result = validate_record(valid_research_record, line_number=1)

    assert result["is_valid"] is True
    assert result["errors"] == []
    assert result["warnings"] == []


def test_validate_record_fails_for_missing_required_top_level_field(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record.pop("risk")

    result = validate_record(record, line_number=3)

    assert result["is_valid"] is False
    assert "Line 3: Missing required field: risk" in result["errors"]


def test_validate_record_fails_for_missing_required_nested_field(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["rule_engine"].pop("signal")

    result = validate_record(record, line_number=4)

    assert result["is_valid"] is False
    assert "Line 4: Missing required field: rule_engine.signal" in result["errors"]


def test_validate_record_fails_for_invalid_datetime(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["logged_at"] = "not-a-datetime"

    result = validate_record(record)

    assert result["is_valid"] is False
    assert "logged_at must be a parseable datetime string" in result["errors"]


def test_validate_record_fails_for_invalid_symbol(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["symbol"] = "   "

    result = validate_record(record)

    assert result["is_valid"] is False
    assert "symbol must be a non-empty string" in result["errors"]


def test_validate_record_fails_for_invalid_bool_numeric_and_label_types(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["risk"]["execution_allowed"] = "yes"
    record["future_return_15m"] = "0.8"
    record["future_label_1h"] = "sideways"

    result = validate_record(record, line_number=8)

    assert result["is_valid"] is False
    assert "Line 8: risk.execution_allowed must be a bool" in result["errors"]
    assert "Line 8: future_return_15m must be an int or float when present" in result["errors"]
    assert "Line 8: future_label_1h must be one of: up, down, flat" in result["errors"]


def test_validate_jsonl_file_summarizes_invalid_examples(
    valid_research_record: dict[str, object],
    write_jsonl,
) -> None:
    invalid_record = deepcopy(valid_research_record)
    invalid_record.pop("execution")

    input_path = write_jsonl([valid_research_record, invalid_record])

    result = validate_jsonl_file(Path(input_path))

    assert result["total_records"] == 2
    assert result["valid_records"] == 1
    assert result["invalid_records"] == 1
    assert result["error_count"] >= 1
    assert len(result["invalid_examples"]) == 1
