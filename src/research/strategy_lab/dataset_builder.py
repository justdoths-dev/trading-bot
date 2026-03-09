from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, UTC
from typing import Any


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "logs" / "trade_analysis.jsonl"


def load_jsonl_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    input_path = Path(path) if path else DEFAULT_DATASET_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            content = line.strip()

            if not content:
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {input_path} line {line_no}: {exc.msg}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in {input_path} line {line_no}"
                )

            records.append(parsed)

    return records


def build_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    raw_records = load_jsonl_records(path)
    return [normalize_record(r) for r in raw_records]


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
        "selected_strategy": _to_str(record.get("selected_strategy")),

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


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _to_str(v: Any) -> str | None:
    if v is None:
        return None
    text = str(v).strip()
    return text if text else None


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

    text = str(v)

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
