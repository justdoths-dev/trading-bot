from __future__ import annotations

"""Segment-based performance reports for strategy lab research rows."""

from datetime import datetime
from statistics import median
from typing import Any, Callable, Iterable

from src.research.research_metrics import HORIZONS

_VALID_HORIZONS = set(HORIZONS)
_VALID_LABELS = {"up", "down", "flat"}
_DAY_ORDER = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEK_PART_ORDER = ("weekday", "weekend")


def build_hour_of_day_segment_report(
    rows: list[dict[str, Any]],
    horizon: str = "15m",
    min_samples: int = 10,
) -> dict[str, Any]:

    return _build_segment_report(
        rows,
        horizon,
        min_samples,
        "hour_of_day",
        _extract_hour_of_day_segment,
        _hour_of_day_sort_key,
    )


def build_day_of_week_segment_report(
    rows: list[dict[str, Any]],
    horizon: str = "15m",
    min_samples: int = 10,
) -> dict[str, Any]:

    return _build_segment_report(
        rows,
        horizon,
        min_samples,
        "day_of_week",
        _extract_day_of_week_segment,
        _day_of_week_sort_key,
    )


def build_week_part_segment_report(
    rows: list[dict[str, Any]],
    horizon: str = "15m",
    min_samples: int = 10,
) -> dict[str, Any]:

    return _build_segment_report(
        rows,
        horizon,
        min_samples,
        "week_part",
        _extract_week_part_segment,
        _week_part_sort_key,
    )


def build_segment_reports(
    rows: list[dict[str, Any]],
    horizons: tuple[str, ...] = ("15m", "1h", "4h"),
    min_samples: int = 10,
) -> dict[str, Any]:

    safe_rows = _coerce_rows(rows)
    valid_rows = [r for r in safe_rows if _extract_timestamp(r) is not None]

    safe_min_samples = _normalize_min_samples(min_samples)
    normalized_horizons = tuple(_normalize_horizon(h) for h in horizons)

    reports: dict[str, dict[str, Any]] = {}

    for horizon in normalized_horizons:
        reports[horizon] = {
            "hour_of_day": build_hour_of_day_segment_report(valid_rows, horizon, safe_min_samples),
            "day_of_week": build_day_of_week_segment_report(valid_rows, horizon, safe_min_samples),
            "week_part": build_week_part_segment_report(valid_rows, horizon, safe_min_samples),
        }

    return {
        "report_type": "segment_reports",
        "total_rows": len(valid_rows),
        "horizons": normalized_horizons,
        "min_samples": safe_min_samples,
        "reports": reports,
    }


def _build_segment_report(
    rows: list[dict[str, Any]],
    horizon: str,
    min_samples: int,
    segment_type: str,
    segment_extractor: Callable[[dict[str, Any]], str | None],
    sort_key: Callable[[str], Any],
) -> dict[str, Any]:

    horizon_key = _normalize_horizon(horizon)
    safe_min_samples = _normalize_min_samples(min_samples)

    valid_rows = [r for r in rows if _extract_timestamp(r) is not None]

    grouped_rows: dict[str, list[dict[str, Any]]] = {}

    for row in valid_rows:

        segment = segment_extractor(row)
        if segment is None:
            continue

        grouped_rows.setdefault(segment, []).append(row)

    segments: list[dict[str, Any]] = []

    for segment, group_rows in grouped_rows.items():

        metrics = _build_metrics_for_rows(group_rows, horizon_key)

        if metrics["sample_count"] < safe_min_samples:
            continue

        segments.append(
            {
                "segment": segment,
                **metrics,
            }
        )

    segments.sort(key=lambda item: sort_key(str(item.get("segment") or "")))

    return {
        "segment_type": segment_type,
        "horizon": horizon_key,
        "total_rows": len(valid_rows),
        "qualified_segments": len(segments),
        "segments": segments,
    }


def _build_metrics_for_rows(rows: list[dict[str, Any]], horizon: str) -> dict[str, Any]:

    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"

    sample_count = len(rows)

    labeled_count = 0
    returns: list[float] = []

    label_distribution = {"up": 0, "down": 0, "flat": 0}

    signal_known = 0
    signal_match = 0

    bias_known = 0
    bias_match = 0

    for row in rows:

        label = _normalize_label(row.get(label_key))
        future_return = _to_float(row.get(return_key))

        if label not in _VALID_LABELS or future_return is None:
            continue

        labeled_count += 1
        returns.append(future_return)

        label_distribution[label] += 1

        signal_direction = _signal_to_direction(_extract_signal(row))

        if signal_direction in _VALID_LABELS:
            signal_known += 1
            if signal_direction == label:
                signal_match += 1

        bias_direction = _bias_to_direction(row.get("bias"))

        if bias_direction in _VALID_LABELS:
            bias_known += 1
            if bias_direction == label:
                bias_match += 1

    return {
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": _pct(labeled_count, sample_count),
        "signal_match_rate": _ratio(signal_match, signal_known),
        "bias_match_rate": _ratio(bias_match, bias_known),
        "avg_future_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
        "median_future_return_pct": round(median(returns), 6) if returns else 0.0,
        "label_distribution": label_distribution,
    }


def _extract_hour_of_day_segment(row: dict[str, Any]) -> str | None:

    dt = _extract_timestamp(row)
    if dt is None:
        return None

    return f"{dt.hour:02d}"


def _extract_day_of_week_segment(row: dict[str, Any]) -> str | None:

    dt = _extract_timestamp(row)
    if dt is None:
        return None

    return _DAY_ORDER[dt.weekday()]


def _extract_week_part_segment(row: dict[str, Any]) -> str | None:

    dt = _extract_timestamp(row)
    if dt is None:
        return None

    return "weekend" if dt.weekday() >= 5 else "weekday"


def _extract_timestamp(row: dict[str, Any]) -> datetime | None:

    for key in ("logged_at", "timestamp", "loggedAt", "time"):
        value = row.get(key)
        dt = _parse_timestamp(value)
        if dt is not None:
            return dt

    return None


def _parse_timestamp(value: Any) -> datetime | None:

    if isinstance(value, datetime):
        return value

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _extract_signal(row: dict[str, Any]) -> Any:

    for key in ("rule_signal", "execution_signal", "execution_action", "signal"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value

    return None


def _normalize_horizon(horizon: str) -> str:

    value = str(horizon or "").strip().lower()

    if value not in _VALID_HORIZONS:
        raise ValueError(f"horizon must be one of: {sorted(_VALID_HORIZONS)}")

    return value


def _normalize_min_samples(min_samples: int) -> int:

    normalized = int(min_samples)

    if normalized < 1:
        raise ValueError("min_samples must be >= 1")

    return normalized


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


def _to_float(value: Any) -> float | None:

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(numerator: int, denominator: int) -> float:

    if denominator <= 0:
        return 0.0

    return round((numerator / denominator) * 100.0, 2)


def _ratio(numerator: int, denominator: int) -> float:

    if denominator <= 0:
        return 0.0

    return round(numerator / denominator, 4)


def _coerce_rows(rows: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:

    if rows is None:
        return []

    return [row for row in rows if isinstance(row, dict)]


def _hour_of_day_sort_key(value: str) -> tuple[int, str]:

    try:
        return (int(value), value)
    except ValueError:
        return (99, value)


def _day_of_week_sort_key(value: str) -> tuple[int, str]:

    try:
        return (_DAY_ORDER.index(value), value)
    except ValueError:
        return (99, value)


def _week_part_sort_key(value: str) -> tuple[int, str]:

    try:
        return (_WEEK_PART_ORDER.index(value), value)
    except ValueError:
        return (99, value)
