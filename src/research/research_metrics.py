from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import median
from typing import Any, Callable

HORIZONS: tuple[str, ...] = ("15m", "1h", "4h")
KNOWN_LABELS: set[str] = {"up", "down", "flat"}


def calculate_research_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate full research metrics payload for trade-analysis records."""
    overview = _build_dataset_overview(records)
    horizons = {
        horizon: _build_horizon_summary(records, horizon)
        for horizon in HORIZONS
    }

    by_symbol = {
        symbol: _build_group_summary(group_records)
        for symbol, group_records in _group_by_key(records, "symbol").items()
    }

    by_strategy = {
        strategy: _build_group_summary(group_records)
        for strategy, group_records in _group_by_key(records, "selected_strategy").items()
    }

    return {
        "dataset_overview": overview,
        "horizon_summary": horizons,
        "by_symbol": by_symbol,
        "by_strategy": by_strategy,
    }


def _build_dataset_overview(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    records_with_any_future_label = sum(1 for record in records if _has_any_future_label(record))

    return {
        "total_records": total_records,
        "records_with_any_future_label": records_with_any_future_label,
        "label_coverage_any_horizon_pct": _pct(records_with_any_future_label, total_records),
        "symbols_distribution": _distribution(records, lambda record: _str_or_unknown(record.get("symbol"))),
        "selected_strategies_distribution": _distribution(
            records,
            lambda record: _str_or_unknown(record.get("selected_strategy")),
        ),
        "bias_distribution": _distribution(records, _extract_bias),
        "ai_execution_distribution": _distribution(records, _extract_ai_execution_state),
        "alignment_distribution": _distribution(records, _extract_alignment_state),
        "date_range": _build_date_range(records),
    }


def _build_group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    labeled_any = sum(1 for record in records if _has_any_future_label(record))

    return {
        "total_records": total_records,
        "records_with_any_future_label": labeled_any,
        "label_coverage_any_horizon_pct": _pct(labeled_any, total_records),
        "horizon_summary": {
            horizon: _build_horizon_summary(records, horizon)
            for horizon in HORIZONS
        },
    }


def _build_horizon_summary(records: list[dict[str, Any]], horizon: str) -> dict[str, Any]:
    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"

    labeled_count = 0
    label_counter: Counter[str] = Counter()
    returns: list[float] = []

    bias_match = 0
    bias_mismatch = 0
    bias_unknown = 0

    signal_match = 0
    signal_mismatch = 0
    signal_unknown = 0

    for record in records:
        label = _normalize_label(record.get(label_key))
        if label not in KNOWN_LABELS:
            continue

        labeled_count += 1
        label_counter[label] += 1

        future_return = _to_float(record.get(return_key))
        if future_return is not None:
            returns.append(future_return)

        bias_direction = _bias_to_direction(_extract_bias(record))
        signal_direction = _signal_to_direction(_extract_signal(record))

        if bias_direction == "unknown":
            bias_unknown += 1
        elif bias_direction == label:
            bias_match += 1
        else:
            bias_mismatch += 1

        if signal_direction == "unknown":
            signal_unknown += 1
        elif signal_direction == label:
            signal_match += 1
        else:
            signal_mismatch += 1

    return {
        "labeled_records": labeled_count,
        "label_distribution": {
            "up": label_counter.get("up", 0),
            "down": label_counter.get("down", 0),
            "flat": label_counter.get("flat", 0),
        },
        "avg_future_return_pct": round(sum(returns) / len(returns), 6) if returns else None,
        "median_future_return_pct": round(median(returns), 6) if returns else None,
        "positive_rate_pct": _pct(label_counter.get("up", 0), labeled_count),
        "negative_rate_pct": _pct(label_counter.get("down", 0), labeled_count),
        "flat_rate_pct": _pct(label_counter.get("flat", 0), labeled_count),
        "bias_vs_label": {
            "match": bias_match,
            "mismatch": bias_mismatch,
            "unknown": bias_unknown,
            "match_rate_pct": _pct(bias_match, bias_match + bias_mismatch),
        },
        "signal_vs_label": {
            "match": signal_match,
            "mismatch": signal_mismatch,
            "unknown": signal_unknown,
            "match_rate_pct": _pct(signal_match, signal_match + signal_mismatch),
        },
    }


def _group_by_key(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_str_or_unknown(record.get(key))].append(record)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _distribution(
    records: list[dict[str, Any]],
    key_builder: Callable[[dict[str, Any]], str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[key_builder(record)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _build_date_range(records: list[dict[str, Any]]) -> dict[str, str]:
    parsed_dates = [_parse_datetime(record.get("logged_at")) for record in records]
    valid_dates = [value for value in parsed_dates if value is not None]

    if not valid_dates:
        return {"start": "unknown", "end": "unknown"}

    return {
        "start": min(valid_dates).isoformat(),
        "end": max(valid_dates).isoformat(),
    }


def _has_any_future_label(record: dict[str, Any]) -> bool:
    for horizon in HORIZONS:
        if _normalize_label(record.get(f"future_label_{horizon}")) in KNOWN_LABELS:
            return True
    return False


def _extract_bias(record: dict[str, Any]) -> str:
    top_level = _str_or_unknown(record.get("bias"))
    if top_level != "unknown":
        return top_level

    rule_engine = record.get("rule_engine") or {}
    return _str_or_unknown(rule_engine.get("bias"))


def _extract_signal(record: dict[str, Any]) -> str:
    execution = record.get("execution") or {}
    rule_engine = record.get("rule_engine") or {}

    for candidate in (
        rule_engine.get("signal"),
        execution.get("signal"),
        execution.get("action"),
    ):
        value = _str_or_unknown(candidate)
        if value != "unknown":
            return value

    return "unknown"


def _extract_ai_execution_state(record: dict[str, Any]) -> str:
    """
    Infer AI execution state from whatever AI metadata exists.

    Preferred buckets:
    - executed
    - reused
    - skipped
    - unknown
    """
    ai = record.get("ai")
    if not isinstance(ai, dict) or not ai:
        return "unknown"

    # 1. Explicit boolean fields if present
    executed_flag = ai.get("executed")
    reused_flag = ai.get("reused")
    skipped_flag = ai.get("skipped")

    if reused_flag is True:
        return "reused"
    if skipped_flag is True:
        return "skipped"
    if executed_flag is True:
        return "executed"

    # 2. Source-based interpretation
    source = _str_or_unknown(ai.get("source")).lower()
    if source in {"cache", "cached", "reused", "reuse"}:
        return "reused"
    if source in {"scheduler_cache", "skip", "skipped"}:
        return "skipped"
    if source in {"live", "fresh", "executed", "generated"}:
        return "executed"

    # 3. generated_at heuristic
    generated_at = ai.get("generated_at")
    logged_at_dt = _parse_datetime(record.get("logged_at"))
    ai_generated_dt = _parse_generated_at(generated_at)

    if logged_at_dt is not None and ai_generated_dt is not None:
        lag_seconds = (logged_at_dt - ai_generated_dt).total_seconds()

        # 오래된 값이면 재사용 가능성이 큼
        if lag_seconds > 120:
            return "reused"

        # 거의 같은 시점이면 해당 cycle에서 생성된 값으로 간주
        if lag_seconds >= -5:
            return "executed"

    # 4. 최소한 AI 결과가 존재하되 명확한 실행 상태를 모르겠는 경우
    if any(key in ai for key in ("final_stance", "analysis", "response", "model")):
        return "executed"

    return "unknown"


def _extract_alignment_state(record: dict[str, Any]) -> str:
    alignment = record.get("alignment")

    if isinstance(alignment, str):
        return _normalize_alignment_state(alignment)

    if isinstance(alignment, dict):
        if "is_aligned" in alignment:
            value = alignment.get("is_aligned")
            if value is True:
                return "aligned"
            if value is False:
                return "misaligned"

        if "state" in alignment:
            return _normalize_alignment_state(alignment.get("state"))

        if "alignment" in alignment:
            return _normalize_alignment_state(alignment.get("alignment"))

    return "unknown"


def _normalize_alignment_state(value: Any) -> str:
    text = _str_or_unknown(value).strip().lower()
    if text == "aligned":
        return "aligned"
    if text in {"misaligned", "not_aligned"}:
        return "misaligned"
    return "unknown"


def _bias_to_direction(raw_bias: str) -> str:
    bias = raw_bias.strip().lower()

    if bias in ("bullish", "long", "buy", "up"):
        return "up"
    if bias in ("bearish", "short", "sell", "down"):
        return "down"
    if bias in ("neutral", "hold", "flat", "no_trade"):
        return "flat"
    return "unknown"


def _signal_to_direction(raw_signal: str) -> str:
    signal = raw_signal.strip().lower()

    if signal in ("long", "buy", "up"):
        return "up"
    if signal in ("short", "sell", "down"):
        return "down"
    if signal in ("hold", "neutral", "no_trade", "flat"):
        return "flat"
    return "unknown"


def _normalize_label(raw_label: Any) -> str:
    if raw_label is None:
        return "unknown"

    label = str(raw_label).strip().lower()
    if label in KNOWN_LABELS:
        return label
    return "unknown"


def _parse_datetime(raw_value: Any) -> datetime | None:
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def _parse_generated_at(raw_value: Any) -> datetime | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value <= 0:
            return None

        # millisecond timestamp
        if value > 1_000_000_000_000:
            value = value / 1000.0

        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(raw_value, str):
        return _parse_datetime(raw_value)

    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"

    text = str(value).strip()
    return text if text else "unknown"


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100, 2)