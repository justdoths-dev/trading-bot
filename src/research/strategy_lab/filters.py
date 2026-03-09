from __future__ import annotations

from datetime import datetime
from typing import Any

from src.research.research_metrics import HORIZONS


def filter_by_symbol(rows: list[dict[str, Any]], symbol: str):

    target = symbol.lower()

    return [
        r for r in rows
        if str(r.get("symbol") or "").lower() == target
    ]


def filter_by_strategy(rows: list[dict[str, Any]], strategy: str):

    target = strategy.lower()

    return [
        r for r in rows
        if str(r.get("selected_strategy") or "").lower() == target
    ]


def filter_labeled_only(rows: list[dict[str, Any]], horizon: str):

    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")

    label_key = f"future_label_{horizon}"

    valid = {"up", "down", "flat"}

    return [
        r for r in rows
        if str(r.get(label_key) or "").lower() in valid
    ]


def filter_by_date_range(
    rows: list[dict[str, Any]],
    start: datetime | None,
    end: datetime | None,
):

    if start is None and end is None:
        return rows

    out = []

    for r in rows:

        ts = r.get("logged_at")

        if not isinstance(ts, datetime):
            continue

        if start and ts < start:
            continue

        if end and ts > end:
            continue

        out.append(r)

    return out
