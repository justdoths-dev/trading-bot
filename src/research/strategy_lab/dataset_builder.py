from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.research.inputs.current_trade_analysis_resolver import (
    discover_current_trade_analysis_files,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "logs" / "trade_analysis.jsonl"

DEFAULT_LATEST_WINDOW_HOURS = 36
DEFAULT_LATEST_MAX_ROWS = 2500


def load_jsonl_records(
    path: str | Path | None = None,
    *,
    rotation_aware: bool | None = None,
    max_age_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> list[dict[str, Any]]:
    records, _ = load_jsonl_records_with_metadata(
        path=path,
        rotation_aware=rotation_aware,
        max_age_hours=max_age_hours,
        max_rows=max_rows,
    )
    return records


def load_jsonl_records_with_metadata(
    path: str | Path | None = None,
    *,
    rotation_aware: bool | None = None,
    max_age_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = Path(path) if path else DEFAULT_DATASET_PATH

    source_files = resolve_source_files(
        input_path,
        rotation_aware=rotation_aware,
    )

    raw_records: list[dict[str, Any]] = []
    per_file_row_counts: dict[str, int] = {}

    for source_file in source_files:
        rows = _read_jsonl_file(source_file)
        raw_records.extend(rows)
        per_file_row_counts[str(source_file)] = len(rows)

    rotation_enabled = _should_use_rotation_aware(input_path, rotation_aware)

    windowed_records = _apply_recent_window(
        raw_records,
        max_age_hours=max_age_hours if rotation_enabled else None,
        max_rows=max_rows if rotation_enabled else None,
    )

    metadata = {
        "input_path": str(input_path),
        "rotation_aware": rotation_enabled,
        "source_files": [str(path) for path in source_files],
        "source_file_count": len(source_files),
        "source_row_counts": per_file_row_counts,
        "max_age_hours": max_age_hours if rotation_enabled else None,
        "max_rows": max_rows if rotation_enabled else None,
        "raw_record_count": len(raw_records),
        "windowed_record_count": len(windowed_records),
    }

    return windowed_records, metadata


def resolve_source_files(
    input_path: Path,
    *,
    rotation_aware: bool | None = None,
) -> list[Path]:
    if not _should_use_rotation_aware(input_path, rotation_aware):
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")
        return [input_path]

    log_dir = input_path.parent
    if not log_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {log_dir}")

    candidates = discover_current_trade_analysis_files(
        log_dir,
        include_rotated_base=True,
    )

    if not candidates:
        raise FileNotFoundError(f"No dataset files found for rotation-aware input: {input_path}")

    return candidates


def build_dataset(
    path: str | Path | None = None,
    *,
    rotation_aware: bool | None = None,
    max_age_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> list[dict[str, Any]]:
    raw_records = load_jsonl_records(
        path=path,
        rotation_aware=rotation_aware,
        max_age_hours=max_age_hours,
        max_rows=max_rows,
    )

    dataset: list[dict[str, Any]] = []
    for record in raw_records:
        normalized = normalize_record(record)
        if _is_research_labelable_record(normalized):
            dataset.append(normalized)

    return dataset


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    rule_engine = _as_dict(record.get("rule_engine"))
    execution = _as_dict(record.get("execution"))
    risk = _as_dict(record.get("risk"))
    ai = _as_dict(record.get("ai"))
    alignment = record.get("alignment")

    scalping = _as_dict(record.get("scalping_result"))
    intraday = _as_dict(record.get("intraday_result"))
    swing = _as_dict(record.get("swing_result"))

    logged_at = _parse_datetime(record.get("logged_at"))

    return {
        "logged_at": logged_at,
        "symbol": _to_str(record.get("symbol")),
        "selected_strategy": _first_text(
            record.get("selected_strategy"),
            rule_engine.get("strategy"),
        ),
        "bias": _to_str(record.get("bias") or rule_engine.get("bias")),
        "rule_signal": _to_str(rule_engine.get("signal")),
        "execution_signal": _to_str(execution.get("signal")),
        "execution_action": _to_str(execution.get("action")),
        "ai_source": _to_str(ai.get("source")),
        "ai_model": _to_str(ai.get("model")),
        "ai_final_stance": _to_str(ai.get("final_stance")),
        "alignment_state": _extract_alignment_state(alignment),
        "execution_allowed": _to_bool(
            execution.get("execution_allowed", risk.get("execution_allowed"))
        ),
        "entry_price": _to_float(
            execution.get("entry_price", risk.get("entry_price"))
        ),
        "stop_loss": _to_float(execution.get("stop_loss", risk.get("stop_loss"))),
        "take_profit": _to_float(execution.get("take_profit", risk.get("take_profit"))),
        "volatility_state": _to_str(risk.get("volatility_state")),
        "scalping_confidence": _to_float(scalping.get("confidence")),
        "intraday_confidence": _to_float(intraday.get("confidence")),
        "swing_confidence": _to_float(swing.get("confidence")),
        "future_return_15m": _to_float(record.get("future_return_15m")),
        "future_return_1h": _to_float(record.get("future_return_1h")),
        "future_return_4h": _to_float(record.get("future_return_4h")),
        "future_label_15m": _to_str(record.get("future_label_15m")),
        "future_label_1h": _to_str(record.get("future_label_1h")),
        "future_label_4h": _to_str(record.get("future_label_4h")),
    }


def _is_research_labelable_record(record: dict[str, Any]) -> bool:
    """
    Decide whether a normalized row is eligible for the strategy-lab dataset.

    Current conservative rule:
    - must have logged_at
    - must have symbol
    - must have selected_strategy
    - must have a positive entry_price

    Rationale:
    The raw trade_analysis stream contains many hold/blocked observation rows with
    no executable entry context. Including those rows inflates the research dataset
    denominator and makes future-return label coverage appear much worse than it is.
    """
    if record.get("logged_at") is None:
        return False

    if not record.get("symbol"):
        return False

    if not record.get("selected_strategy"):
        return False

    entry_price = record.get("entry_price")
    if entry_price is None:
        return False

    try:
        return float(entry_price) > 0
    except (TypeError, ValueError):
        return False


def _should_use_rotation_aware(
    input_path: Path,
    rotation_aware: bool | None,
) -> bool:
    if rotation_aware is not None:
        return rotation_aware

    return input_path.name == "trade_analysis.jsonl"


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path} line {line_no}: {exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_no}")

            records.append(parsed)

    return records


def _apply_recent_window(
    records: list[dict[str, Any]],
    *,
    max_age_hours: int | None,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if not records:
        return []

    filtered = records

    if max_age_hours is not None and max_age_hours > 0:
        timestamps = [
            _parse_datetime(record.get("logged_at"))
            for record in records
        ]
        valid_timestamps = [ts for ts in timestamps if ts is not None]

        if valid_timestamps:
            latest_ts = max(valid_timestamps)
            cutoff = latest_ts - timedelta(hours=max_age_hours)

            kept: list[dict[str, Any]] = []
            for record in records:
                ts = _parse_datetime(record.get("logged_at"))
                if ts is None:
                    continue
                if ts >= cutoff:
                    kept.append(record)
            filtered = kept

    if max_rows is not None and max_rows > 0 and len(filtered) > max_rows:
        filtered = filtered[-max_rows:]

    return filtered


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _to_str(v: Any) -> str | None:
    if v is None:
        return None
    text = str(v).strip()
    return text if text else None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _to_str(value)
        if text is not None:
            return text
    return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v

    if v is None:
        return None

    text = str(v).lower()

    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False

    return None


def _parse_datetime(v: Any) -> datetime | None:
    if v is None:
        return None

    text = str(v).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def _extract_alignment_state(v: Any) -> str | None:
    if isinstance(v, str):
        return v.lower()

    if isinstance(v, dict):
        if "is_aligned" in v:
            return "aligned" if v["is_aligned"] else "misaligned"

        if "state" in v:
            return str(v["state"]).lower()

    return None
