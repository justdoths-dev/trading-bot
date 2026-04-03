from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

DEFAULT_PRIMARY_TRADE_ANALYSIS_PATH = Path("logs/trade_analysis_cumulative.jsonl")
DEFAULT_FALLBACK_TRADE_ANALYSIS_PATH = Path("logs/trade_analysis.jsonl")

HORIZON_RETURN_FIELDS = {
    "15m": "future_return_15m",
    "1h": "future_return_1h",
    "4h": "future_return_4h",
}

MIN_POSITIVE_RATE_PCT = 45.0
MIN_SAMPLE_COUNT = 30


def apply_candidate_quality_gate(
    candidates: list[dict[str, Any]],
    trade_analysis_path: Path | None = None,
) -> dict[str, Any]:
    """
    Filter edge-selection candidates using downstream quality metrics derived
    from trade-analysis logs.

    The gate evaluates each candidate by exact selected identity:
    (selected_symbol, selected_strategy, selected_horizon)

    Fallback behavior:
    - If every candidate is dropped, restore the original candidates so the
      pipeline does not re-enter candidate starvation.
    """
    normalized_candidates = [candidate for candidate in candidates if isinstance(candidate, dict)]
    resolved_path = resolve_trade_analysis_path(trade_analysis_path)
    records = load_trade_analysis_records(resolved_path)

    kept_candidates: list[dict[str, Any]] = []
    dropped_candidates: list[dict[str, Any]] = []

    for candidate in normalized_candidates:
        metrics = compute_candidate_metrics(candidate, records)
        drop_reason = determine_drop_reason(metrics)

        if drop_reason is None:
            kept_candidates.append(candidate)
            continue

        dropped_candidates.append(
            {
                "candidate": candidate,
                "reason": drop_reason,
                "metrics": metrics,
            }
        )

    fallback_applied = False
    if normalized_candidates and not kept_candidates:
        kept_candidates = list(normalized_candidates)
        fallback_applied = True

    return {
        "kept_candidates": kept_candidates,
        "dropped_candidates": dropped_candidates,
        "input_path_used": str(resolved_path),
        "total_candidates": len(normalized_candidates),
        "kept_count": len(kept_candidates),
        "dropped_count": len(dropped_candidates),
        "fallback_applied": fallback_applied,
    }


def resolve_trade_analysis_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        final_path = explicit_path.expanduser()
        if not final_path.is_absolute():
            final_path = Path.cwd() / final_path
        final_path = final_path.resolve()
        if not final_path.exists():
            raise FileNotFoundError(f"Trade-analysis input path does not exist: {final_path}")
        return final_path

    for candidate in (DEFAULT_PRIMARY_TRADE_ANALYSIS_PATH, DEFAULT_FALLBACK_TRADE_ANALYSIS_PATH):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find trade-analysis JSONL input. Checked: "
        f"{DEFAULT_PRIMARY_TRADE_ANALYSIS_PATH} and {DEFAULT_FALLBACK_TRADE_ANALYSIS_PATH}"
    )


def load_trade_analysis_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                records.append(payload)

    return records


def compute_candidate_metrics(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """
    Compute downstream quality metrics for a single candidate.

    Only rows that represent an actual selected candidate are eligible.
    Identity match must be exact on:
    - symbol
    - strategy
    - horizon
    """
    candidate_identity = extract_candidate_identity(candidate)
    if candidate_identity is None:
        return {
            "positive_rate_pct": None,
            "median_return_pct": None,
            "sample_count": 0,
        }

    _, _, candidate_horizon = candidate_identity
    return_field = HORIZON_RETURN_FIELDS[candidate_horizon]

    matching_returns: list[float] = []
    positive_count = 0

    for record in records:
        if not is_selected_record(record):
            continue

        if extract_selected_candidate_identity(record) != candidate_identity:
            continue

        raw_return = coerce_float(record.get(return_field))
        if raw_return is None:
            continue

        matching_returns.append(raw_return)
        if raw_return > 0:
            positive_count += 1

    sample_count = len(matching_returns)
    if sample_count == 0:
        return {
            "positive_rate_pct": None,
            "median_return_pct": None,
            "sample_count": 0,
        }

    return {
        "positive_rate_pct": round((positive_count / sample_count) * 100.0, 6),
        "median_return_pct": round(float(median(matching_returns)), 6),
        "sample_count": sample_count,
    }


def determine_drop_reason(metrics: dict[str, float | int | None]) -> str | None:
    median_return_pct = coerce_float(metrics.get("median_return_pct"))
    positive_rate_pct = coerce_float(metrics.get("positive_rate_pct"))
    sample_count = coerce_int(metrics.get("sample_count"))

    if sample_count is None or sample_count < MIN_SAMPLE_COUNT:
        return "sample_count_below_minimum"
    if median_return_pct is None or median_return_pct < 0:
        return "median_return_pct_negative"
    if positive_rate_pct is None or positive_rate_pct < MIN_POSITIVE_RATE_PCT:
        return "positive_rate_pct_below_minimum"
    return None


def extract_candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str] | None:
    symbol = normalize_symbol(candidate.get("symbol"))
    strategy = normalize_strategy(candidate.get("strategy"))
    horizon = normalize_horizon(candidate.get("horizon"))

    if symbol is None or strategy is None or horizon is None:
        return None

    return (symbol, strategy, horizon)


def is_selected_record(record: dict[str, Any]) -> bool:
    """
    Decide whether a trade-analysis row should count as a selected-candidate row.

    Accepted cases:
    1. Modern enriched rows with edge_selection_output.selection_status == "selected"
    2. Legacy-compatible rows with complete top-level selected_* identity fields

    Rejected cases:
    - explicit non-selected statuses such as abstain
    - rows without a complete selected candidate identity
    """
    edge_selection_output = record.get("edge_selection_output")
    if isinstance(edge_selection_output, dict):
        status = normalize_status(edge_selection_output.get("selection_status"))
        if status is not None:
            if status != "selected":
                return False
            return extract_selected_candidate_identity(record) is not None

    return (
        normalize_symbol(record.get("selected_symbol")) is not None
        and normalize_strategy(record.get("selected_strategy")) is not None
        and normalize_horizon(record.get("selected_horizon")) is not None
    )


def extract_selected_candidate_identity(record: dict[str, Any]) -> tuple[str, str, str] | None:
    edge_selection_output = record.get("edge_selection_output")
    selected_payload = edge_selection_output if isinstance(edge_selection_output, dict) else {}

    symbol = first_non_empty_normalized_symbol(
        selected_payload.get("selected_symbol"),
        record.get("selected_symbol"),
    )
    strategy = first_non_empty_normalized_strategy(
        selected_payload.get("selected_strategy"),
        record.get("selected_strategy"),
    )
    horizon = first_non_empty_normalized_horizon(
        selected_payload.get("selected_horizon"),
        record.get("selected_horizon"),
    )

    if symbol is None or strategy is None or horizon is None:
        return None

    return (symbol, strategy, horizon)


def normalize_status(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def first_non_empty_normalized_symbol(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_symbol(value)
        if normalized is not None:
            return normalized
    return None


def first_non_empty_normalized_strategy(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_strategy(value)
        if normalized is not None:
            return normalized
    return None


def first_non_empty_normalized_horizon(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_horizon(value)
        if normalized is not None:
            return normalized
    return None


def normalize_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def normalize_strategy(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def normalize_horizon(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text in HORIZON_RETURN_FIELDS else None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None