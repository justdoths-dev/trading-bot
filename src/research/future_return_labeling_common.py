from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Mapping


HORIZONS: dict[str, timedelta] = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}

DEFAULT_LABEL_THRESHOLDS_PCT: dict[str, float] = {
    "15m": 0.05,
    "1h": 0.10,
    "4h": 0.15,
}


def parse_logged_at_to_utc(value: Any) -> datetime | None:
    """Parse an ISO-like timestamp into an aware UTC datetime."""
    if value is None:
        return None

    text = str(value).strip()
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


def iter_supported_horizons() -> tuple[str, ...]:
    """Return supported horizons in deterministic order."""
    return tuple(HORIZONS.keys())


def horizon_to_timedelta(horizon: str) -> timedelta:
    """Map a supported horizon string to its timedelta."""
    try:
        return HORIZONS[horizon]
    except KeyError as exc:
        raise ValueError(f"Unsupported horizon: {horizon}") from exc


def is_mature_for_horizon(
    row_ts: datetime | None,
    latest_ts: datetime | None,
    horizon: str,
) -> bool:
    """Return True when the row is mature relative to the provided latest timestamp."""
    if row_ts is None or latest_ts is None:
        return False

    return row_ts + horizon_to_timedelta(horizon) <= latest_ts


def has_future_fields_for_horizon(row: Mapping[str, Any], horizon: str) -> bool:
    """
    Treat a horizon as labeled only when BOTH future_return and future_label
    are present with non-empty values.
    """
    return _has_nonempty_value(row.get(f"future_return_{horizon}")) and _has_nonempty_value(
        row.get(f"future_label_{horizon}")
    )


def build_future_update_for_row(
    *,
    row: Mapping[str, Any],
    future_price: float,
    horizon: str,
    label_thresholds_pct: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """
    Build a future update for a row using an externally obtained production-equivalent
    future market price.
    """
    entry_price = extract_entry_price(row)
    if entry_price is None or entry_price <= 0:
        return None

    return build_future_update_from_prices(
        entry_price=entry_price,
        future_price=future_price,
        horizon=horizon,
        label_thresholds_pct=label_thresholds_pct,
    )


def build_future_update_from_prices(
    *,
    entry_price: float,
    future_price: float,
    horizon: str,
    label_thresholds_pct: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Build future_return_* and future_label_* using the existing label formula."""
    if entry_price <= 0 or future_price <= 0:
        return None

    future_return = ((future_price - entry_price) / entry_price) * 100.0

    resolved_thresholds = (
        label_thresholds_pct
        if label_thresholds_pct is not None
        else DEFAULT_LABEL_THRESHOLDS_PCT
    )

    return {
        f"future_return_{horizon}": round(future_return, 6),
        f"future_label_{horizon}": _to_label(
            future_return=future_return,
            horizon=horizon,
            label_thresholds_pct=resolved_thresholds,
        ),
    }


def apply_future_update(
    row: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a shallow-copied row with update fields applied."""
    merged = dict(row)
    merged.update(update)
    return merged


def normalize_symbol(raw_symbol: Any) -> str | None:
    """Normalize symbol values to the slash format expected by ccxt."""
    if raw_symbol is None:
        return None

    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        return None

    if "/" in symbol:
        return symbol

    if symbol.endswith("USDT") and len(symbol) > 4:
        base = symbol[:-4]
        return f"{base}/USDT"

    return symbol


def extract_entry_price(record: Mapping[str, Any]) -> float | None:
    """Extract the entry price from supported record locations."""
    execution = record.get("execution")
    risk = record.get("risk")

    candidate_values = [
        record.get("entry_price"),
        execution.get("entry_price") if isinstance(execution, Mapping) else None,
        risk.get("entry_price") if isinstance(risk, Mapping) else None,
    ]

    for value in candidate_values:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _to_label(
    *,
    future_return: float,
    horizon: str,
    label_thresholds_pct: Mapping[str, float],
) -> str:
    threshold_pct = float(label_thresholds_pct[horizon])

    if future_return > threshold_pct:
        return "up"

    if future_return < -threshold_pct:
        return "down"

    return "flat"


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True
