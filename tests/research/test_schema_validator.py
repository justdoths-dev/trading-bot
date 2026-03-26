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


def test_validate_record_accepts_historical_row_without_replay_fields(
    valid_research_record: dict[str, object],
) -> None:
    result = validate_record(valid_research_record, line_number=10)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_validate_record_accepts_success_replay_row(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["edge_selection_mapper_payload"] = {
        "generated_at": "2026-03-26T00:00:00+00:00",
        "ranked_candidates": [{"symbol": "BTCUSDT", "score": 0.91}],
    }
    record["edge_selection_output"] = {
        "selection_status": "selected",
        "selected_symbol": "BTCUSDT",
    }
    record["edge_selection_metadata"] = {
        "mapper_version": "edge_selection_input_mapper_v1",
        "engine_version": "edge_selection_engine_v1",
        "replay_ready": True,
        "shadow_status": "success",
        "trigger_symbol": "BTCUSDT",
        "reports_dir": "logs/research_reports",
        "shadow_output_path": "logs/research_reports/edge_selection_shadow.json",
        "error_type": None,
        "error_message": None,
        "selection_status": "selected",
        "mapper_generated_at": "2026-03-26T00:00:00+00:00",
    }

    result = validate_record(record, line_number=11)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_validate_record_accepts_failed_replay_row(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["edge_selection_mapper_payload"] = None
    record["edge_selection_output"] = None
    record["edge_selection_metadata"] = {
        "mapper_version": "edge_selection_input_mapper_v1",
        "engine_version": "edge_selection_engine_v1",
        "replay_ready": False,
        "shadow_status": "failed",
        "trigger_symbol": "BTCUSDT",
        "reports_dir": "logs/research_reports",
        "shadow_output_path": None,
        "error_type": "RuntimeError",
        "error_message": "Forced shadow observation failure",
        "selection_status": None,
        "mapper_generated_at": None,
    }

    result = validate_record(record, line_number=12)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_validate_record_rejects_invalid_replay_field_types(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["edge_selection_mapper_payload"] = []
    record["edge_selection_output"] = "selected"

    result = validate_record(record, line_number=13)

    assert result["is_valid"] is False
    assert "Line 13: edge_selection_mapper_payload must be a dict or null when present" in result["errors"]
    assert "Line 13: edge_selection_output must be a dict or null when present" in result["errors"]


def test_validate_record_rejects_invalid_replay_ready_type(
    valid_research_record: dict[str, object],
) -> None:
    record = deepcopy(valid_research_record)
    record["edge_selection_metadata"] = {
        "replay_ready": "yes",
        "shadow_status": "success",
    }

    result = validate_record(record, line_number=14)

    assert result["is_valid"] is False
    assert "Line 14: edge_selection_metadata.replay_ready must be a bool or null" in result["errors"]