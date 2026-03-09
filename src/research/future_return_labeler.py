from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ccxt

LOGGER = logging.getLogger(__name__)

HORIZONS: dict[str, timedelta] = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


def label_dataset() -> None:
    """Add future return labels to unlabeled records in logs/trade_analysis.jsonl."""
    log_path = Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"

    if not log_path.exists():
        LOGGER.info("Log file not found: %s", log_path)
        return

    exchange = ccxt.binance({"enableRateLimit": True})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []

    total_records = 0
    updated_records = 0

    for line in lines:
        if not line.strip():
            continue

        total_records += 1
        record = json.loads(line)

        if _is_fully_labeled(record):
            updated_lines.append(json.dumps(record, ensure_ascii=False))
            continue

        symbol = _normalize_symbol(record.get("symbol"))
        logged_at = _parse_logged_at(record.get("logged_at"))
        entry_price = _extract_entry_price(record)

        if not symbol or logged_at is None or entry_price is None or entry_price <= 0:
            updated_lines.append(json.dumps(record, ensure_ascii=False))
            continue

        horizon_prices = _fetch_horizon_prices(
            exchange=exchange,
            symbol=symbol,
            logged_at=logged_at,
        )

        if horizon_prices is None:
            updated_lines.append(json.dumps(record, ensure_ascii=False))
            continue

        for horizon, future_price in horizon_prices.items():
            future_return = ((future_price - entry_price) / entry_price) * 100
            record[f"future_return_{horizon}"] = round(future_return, 6)
            record[f"future_label_{horizon}"] = _to_label(future_return)

        updated_records += 1
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    tmp_path = log_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
    tmp_path.replace(log_path)

    LOGGER.info("Labeling complete: total=%s updated=%s", total_records, updated_records)


def _is_fully_labeled(record: dict[str, Any]) -> bool:
    required_keys = (
        "future_return_15m",
        "future_return_1h",
        "future_return_4h",
        "future_label_15m",
        "future_label_1h",
        "future_label_4h",
    )
    return all(key in record for key in required_keys)


def _normalize_symbol(raw_symbol: Any) -> str | None:
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


def _parse_logged_at(raw_logged_at: Any) -> datetime | None:
    if raw_logged_at is None:
        return None

    text = str(raw_logged_at).strip()
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


def _extract_entry_price(record: dict[str, Any]) -> float | None:
    candidate_values = [
        record.get("entry_price"),
        (record.get("execution") or {}).get("entry_price"),
        (record.get("risk") or {}).get("entry_price"),
    ]

    for value in candidate_values:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _fetch_horizon_prices(
    exchange: ccxt.binance,
    symbol: str,
    logged_at: datetime,
) -> dict[str, float] | None:
    now_utc = datetime.now(UTC)
    prices: dict[str, float] = {}

    for horizon, delta in HORIZONS.items():
        target_dt = logged_at + delta

        if target_dt > now_utc:
            return None

        since_ms = int(target_dt.timestamp() * 1000)

        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since_ms, limit=1)
        except Exception as exc:
            LOGGER.warning("Failed to fetch OHLCV for %s horizon=%s: %s", symbol, horizon, exc)
            return None

        if not candles:
            return None

        candle = candles[0]
        close_price = float(candle[4])
        prices[horizon] = close_price

    return prices


def _to_label(future_return: float) -> str:
    if future_return > 0.2:
        return "up"
    if future_return < -0.2:
        return "down"
    return "flat"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    label_dataset()
