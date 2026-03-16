from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.research.edge_selection_schema_validator import validate_shadow_output

DEFAULT_SHADOW_OUTPUT_PATH = Path(
    "logs/edge_selection_shadow/edge_selection_shadow.jsonl"
)


def write_edge_selection_shadow_output(
    payload: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Validate and append a shadow selection payload to a JSONL audit log."""
    final_path = Path(output_path) if output_path is not None else DEFAULT_SHADOW_OUTPUT_PATH

    validation_result = validate_shadow_output(payload)
    if not validation_result.is_valid:
        joined_errors = "; ".join(validation_result.errors)
        raise ValueError(f"Invalid shadow output payload: {joined_errors}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    with final_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    return final_path


def read_edge_selection_shadow_outputs(path: Path) -> list[dict[str, Any]]:
    """Read JSONL shadow outputs from disk, skipping blank lines."""
    records: list[dict[str, Any]] = []
    final_path = Path(path)

    if not final_path.exists():
        return records

    if not final_path.is_file():
        raise ValueError(f"Shadow output path is not a file: {final_path}")

    with final_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Shadow output JSONL line {line_number} is not valid JSON: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Shadow output JSONL line {line_number} must contain a JSON object."
                )

            records.append(payload)

    return records
