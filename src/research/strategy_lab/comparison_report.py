from __future__ import annotations

from statistics import median
from typing import Any, Callable

from .dataset_builder import build_dataset
from .filters import filter_labeled_only

_VALID_HORIZONS = {"15m", "1h", "4h"}
_VALID_LABELS = {"up", "down", "flat"}


def compare_by_symbol(horizon: str = "15m", dataset_path: str | None = None) -> dict[str, Any]:
    """Compare performance across symbols for a selected horizon."""
    rows = build_dataset(path=dataset_path)
    return _compare_by_group(
        rows=rows,
        horizon=horizon,
        comparison_type="symbol",
        group_extractor=_extract_symbol,
    )


def compare_by_strategy(horizon: str = "15m", dataset_path: str | None = None) -> dict[str, Any]:
    """Compare performance across selected strategies for a selected horizon."""
    rows = build_dataset(path=dataset_path)
    return _compare_by_group(
        rows=rows,
        horizon=horizon,
        comparison_type="strategy",
        group_extractor=_extract_strategy,
    )


def compare_by_alignment_state(horizon: str = "15m", dataset_path: str | None = None) -> dict[str, Any]:
    """Compare performance across alignment states for a selected horizon."""
    rows = build_dataset(path=dataset_path)
    return _compare_by_group(
        rows=rows,
        horizon=horizon,
        comparison_type="alignment_state",
        group_extractor=_extract_alignment_state,
    )


def compare_by_ai_execution_state(horizon: str = "15m", dataset_path: str | None = None) -> dict[str, Any]:
    """Compare performance across AI execution states for a selected horizon."""
    rows = build_dataset(path=dataset_path)
    return _compare_by_group(
        rows=rows,
        horizon=horizon,
        comparison_type="ai_execution_state",
        group_extractor=_extract_ai_execution_state,
    )


def _compare_by_group(
    rows: list[dict[str, Any]],
    horizon: str,
    comparison_type: str,
    group_extractor: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    horizon_key = _normalize_horizon(horizon)

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_value = group_extractor(row)
        if group_value is None:
            continue
        grouped_rows.setdefault(group_value, []).append(row)

    groups = {
        group_name: _build_performance_from_rows(group_rows, horizon_key)
        for group_name, group_rows in sorted(grouped_rows.items(), key=lambda item: item[0])
    }

    return {
        "comparison_type": comparison_type,
        "horizon": horizon_key,
        "group_count": len(groups),
        "groups": groups,
    }


def _build_performance_from_rows(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
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
        "symbol": None,
        "strategy": None,
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


def _extract_symbol(row: dict[str, Any]) -> str | None:
    return _clean_text(row.get("symbol"))


def _extract_strategy(row: dict[str, Any]) -> str | None:
    return _clean_text(row.get("selected_strategy"))


def _extract_alignment_state(row: dict[str, Any]) -> str | None:
    text = _clean_text(row.get("alignment_state"))
    if text in ("aligned", "misaligned", "unknown"):
        return text
    return None


def _extract_ai_execution_state(row: dict[str, Any]) -> str | None:
    candidates = (
        row.get("ai_execution_state"),
        row.get("execution_source"),
        row.get("ai_decision_source"),
    )

    for value in candidates:
        text = _clean_text(value)
        if text in ("executed", "reused", "unknown"):
            return text

    return None


def _extract_signal(row: dict[str, Any]) -> Any:
    for key in ("rule_signal", "execution_signal", "execution_action"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _normalize_horizon(horizon: str) -> str:
    value = str(horizon or "").strip().lower()
    if value not in _VALID_HORIZONS:
        raise ValueError("horizon must be one of: 15m, 1h, 4h")
    return value


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _VALID_LABELS:
        return text
    return "unknown"


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


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


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