from __future__ import annotations

from statistics import median
from typing import Any

from src.research.research_metrics import HORIZONS

from .dataset_builder import build_dataset
from .filters import (
    filter_by_symbol,
    filter_by_strategy,
    filter_labeled_only,
)

_VALID_HORIZONS = set(HORIZONS)
_VALID_LABELS = {"up", "down", "flat"}


def generate_performance_report(
    symbol: str | None = None,
    strategy: str | None = None,
    horizon: str = "15m",
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Generate strategy performance metrics from normalized research rows."""
    horizon_key = _normalize_horizon(horizon)

    normalized_symbol = _normalize_optional_text(symbol)
    normalized_strategy = _normalize_optional_text(strategy)

    rows = build_dataset(path=dataset_path)

    if normalized_symbol is not None:
        rows = filter_by_symbol(rows, normalized_symbol)

    if normalized_strategy is not None:
        rows = filter_by_strategy(rows, normalized_strategy)

    return _build_performance_from_rows(
        rows=rows,
        horizon=horizon_key,
        symbol=normalized_symbol,
        strategy=normalized_strategy,
    )


def _build_performance_from_rows(
    *,
    rows: list[dict[str, Any]],
    horizon: str,
    symbol: str | None,
    strategy: str | None,
) -> dict[str, Any]:
    sample_count = len(rows)

    labeled_rows = filter_labeled_only(rows, horizon)
    labeled_count = len(labeled_rows)

    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"

    label_counts = {"up": 0, "down": 0, "flat": 0}
    returns: list[float] = []

    bias_known = 0
    bias_match = 0

    signal_known = 0
    signal_match = 0

    for row in labeled_rows:
        label = _normalize_label(row.get(label_key))
        if label not in _VALID_LABELS:
            continue

        label_counts[label] += 1

        future_return = _to_float(row.get(return_key))
        if future_return is not None:
            returns.append(future_return)

        bias_direction = _bias_to_direction(row.get("bias"))
        if bias_direction in _VALID_LABELS:
            bias_known += 1
            if bias_direction == label:
                bias_match += 1

        signal_direction = _signal_to_direction(_extract_signal(row))
        if signal_direction in _VALID_LABELS:
            signal_known += 1
            if signal_direction == label:
                signal_match += 1

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": _pct(labeled_count, sample_count),
        "label_distribution": label_counts,
        "up_rate_pct": _pct(label_counts["up"], labeled_count),
        "down_rate_pct": _pct(label_counts["down"], labeled_count),
        "flat_rate_pct": _pct(label_counts["flat"], labeled_count),
        "bias_match_rate_pct": _pct(bias_match, bias_known),
        "signal_match_rate_pct": _pct(signal_match, signal_known),
        "avg_future_return_pct": round(sum(returns) / len(returns), 6) if returns else None,
        "median_future_return_pct": round(median(returns), 6) if returns else None,
    }


def _normalize_horizon(horizon: str) -> str:
    value = str(horizon or "").strip().lower()
    if value not in _VALID_HORIZONS:
        raise ValueError(f"horizon must be one of: {sorted(_VALID_HORIZONS)}")
    return value


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_LABELS:
        return text
    return "unknown"


def _extract_signal(row: dict[str, Any]) -> Any:
    for key in ("rule_signal", "execution_signal", "execution_action"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _bias_to_direction(value: Any) -> str:
    text = str(value or "").strip().lower()

    if text in ("bullish", "long", "buy", "up"):
        return "up"
    if text in ("bearish", "short", "sell", "down"):
        return "down"
    if text in ("neutral", "hold", "flat", "no_trade"):
        return "flat"
    return "unknown"


def _signal_to_direction(value: Any) -> str:
    text = str(value or "").strip().lower()

    if text in ("long", "buy", "up"):
        return "up"
    if text in ("short", "sell", "down"):
        return "down"
    if text in ("hold", "neutral", "flat", "no_trade"):
        return "flat"
    return "unknown"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100, 2)
