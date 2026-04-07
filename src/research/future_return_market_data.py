from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Mapping

from src.research.future_return_labeling_common import (
    horizon_to_timedelta,
    iter_supported_horizons,
)

LOGGER = logging.getLogger(__name__)


def fetch_horizon_prices(
    *,
    exchange: Any,
    symbol: str,
    logged_at: datetime,
    now_utc: datetime | None = None,
) -> dict[str, float] | None:
    """
    Fetch production-equivalent future prices using 1m OHLCV close prices
    at each supported horizon.
    """
    resolved_now_utc = now_utc or datetime.now(UTC)
    prices: dict[str, float] = {}

    for horizon in iter_supported_horizons():
        target_dt = logged_at + horizon_to_timedelta(horizon)

        if target_dt > resolved_now_utc:
            return None

        since_ms = int(target_dt.timestamp() * 1000)

        try:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe="1m",
                since=since_ms,
                limit=1,
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed to fetch OHLCV for %s horizon=%s: %s",
                symbol,
                horizon,
                exc,
            )
            return None

        if not candles:
            return None

        candle = candles[0]
        if not isinstance(candle, (list, tuple)) or len(candle) < 5:
            LOGGER.warning(
                "Unexpected OHLCV payload for %s horizon=%s: %r",
                symbol,
                horizon,
                candle,
            )
            return None

        try:
            close_price = float(candle[4])
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid OHLCV close price for %s horizon=%s: %r",
                symbol,
                horizon,
                candle[4],
            )
            return None

        if close_price <= 0:
            LOGGER.warning(
                "Non-positive OHLCV close price for %s horizon=%s: %s",
                symbol,
                horizon,
                close_price,
            )
            return None

        prices[horizon] = close_price

    return prices