from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL_FIELDS = (
    "logged_at",
    "symbol",
    "rule_engine",
    "risk",
    "execution",
)

FUTURE_RETURN_FIELDS = (
    "future_return_15m",
    "future_return_1h",
    "future_return_4h",
)

FUTURE_LABEL_FIELDS = (
    "future_label_15m",
    "future_label_1h",
    "future_label_4h",
)

VALID_FUTURE_LABELS = {"up", "down", "flat"}
MAX_INVALID_EXAMPLES = 5


def validate_record(record: dict[str, Any], line_number: int | None = None) -> dict[str, Any]:
    """Validate a single trade analysis record."""
    errors: list[str] = []
    warnings: list[str] = []
    prefix = _line_prefix(line_number)

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in record:
            errors.append(f"{prefix}Missing required field: {field}")

    logged_at = record.get("logged_at")
    if "logged_at" in record and not _is_parseable_datetime(logged_at):
        errors.append(f"{prefix}logged_at must be a parseable datetime string")

    symbol = record.get("symbol")
    if "symbol" in record and not _is_non_empty_string(symbol):
        errors.append(f"{prefix}symbol must be a non-empty string")

    parent_objects: dict[str, bool] = {}
    for field in ("rule_engine", "risk", "execution"):
        value = record.get(field)
        is_valid_mapping = isinstance(value, dict)
        parent_objects[field] = is_valid_mapping

        if field in record and not is_valid_mapping:
            errors.append(f"{prefix}{field} must be a dict")

    if parent_objects.get("rule_engine"):
        _validate_required_string(record, "rule_engine.bias", errors, prefix)
        _validate_required_string(record, "rule_engine.signal", errors, prefix)

    if parent_objects.get("risk"):
        risk_execution_allowed = _get_nested_value(record, "risk.execution_allowed")
        if risk_execution_allowed is _MISSING:
            errors.append(f"{prefix}Missing required field: risk.execution_allowed")
        elif not isinstance(risk_execution_allowed, bool):
            errors.append(f"{prefix}risk.execution_allowed must be a bool")

    if parent_objects.get("execution"):
        _validate_required_string(record, "execution.action", errors, prefix)

    ai_output = record.get("ai_output")
    if "ai_output" in record and ai_output is not None and not isinstance(ai_output, dict):
        warnings.append(f"{prefix}ai_output should be a dict when present")

    for field in FUTURE_RETURN_FIELDS:
        if field not in record:
            continue

        value = record.get(field)
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"{prefix}{field} should be an int or float when present")

    for field in FUTURE_LABEL_FIELDS:
        if field not in record:
            continue

        value = record.get(field)
        if value is None:
            continue

        if not isinstance(value, str) or value.strip().lower() not in VALID_FUTURE_LABELS:
            warnings.append(f"{prefix}{field} should be one of: up, down, flat")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def validate_jsonl_file(input_path: Path) -> dict[str, Any]:
    """Load and validate a JSONL file containing trade analysis records."""
    summary = {
        "input_path": str(input_path),
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "error_count": 0,
        "warning_count": 0,
        "invalid_examples": [],
    }

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            content = line.strip()
            if not content:
                continue

            summary["total_records"] += 1

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                error_message = f"Line {line_number}: malformed JSON - {exc.msg}"
                summary["invalid_records"] += 1
                summary["error_count"] += 1
                _append_invalid_example(
                    summary["invalid_examples"],
                    {
                        "line_number": line_number,
                        "errors": [error_message],
                        "warnings": [],
                    },
                )
                continue

            if not isinstance(parsed, dict):
                error_message = (
                    f"Line {line_number}: expected JSON object, got {type(parsed).__name__}"
                )
                summary["invalid_records"] += 1
                summary["error_count"] += 1
                _append_invalid_example(
                    summary["invalid_examples"],
                    {
                        "line_number": line_number,
                        "errors": [error_message],
                        "warnings": [],
                    },
                )
                continue

            result = validate_record(parsed, line_number=line_number)
            summary["error_count"] += len(result["errors"])
            summary["warning_count"] += len(result["warnings"])

            if result["is_valid"]:
                summary["valid_records"] += 1
            else:
                summary["invalid_records"] += 1
                _append_invalid_example(
                    summary["invalid_examples"],
                    {
                        "line_number": line_number,
                        "errors": result["errors"],
                        "warnings": result["warnings"],
                    },
                )

    return summary


def _validate_required_string(
    record: dict[str, Any],
    path: str,
    errors: list[str],
    prefix: str,
) -> None:
    value = _get_nested_value(record, path)
    if value is _MISSING:
        errors.append(f"{prefix}Missing required field: {path}")
        return

    if not _is_non_empty_string(value):
        errors.append(f"{prefix}{path} must be a non-empty string")


def _get_nested_value(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_parseable_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False

    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        datetime.fromisoformat(candidate)
        return True
    except ValueError:
        return False


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _append_invalid_example(examples: list[dict[str, Any]], example: dict[str, Any]) -> None:
    if len(examples) >= MAX_INVALID_EXAMPLES:
        return
    examples.append(example)


def _line_prefix(line_number: int | None) -> str:
    if line_number is None:
        return ""
    return f"Line {line_number}: "


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate logs/trade_analysis.jsonl before research analysis runs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to the trade analysis JSONL file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = validate_jsonl_file(args.input)
    print(json.dumps(summary, indent=2))


class _MissingValue:
    pass


_MISSING = _MissingValue()


if __name__ == "__main__":
    main()
