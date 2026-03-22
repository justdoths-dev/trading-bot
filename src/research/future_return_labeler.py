from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import ccxt

from src.services.cron_health import CronHealthReporter

LOGGER = logging.getLogger(__name__)

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


def label_dataset(
    *,
    log_path: Path | None = None,
    force_relabel: bool = False,
    label_thresholds_pct: dict[str, float] | None = None,
) -> dict[str, int]:
    """Label or relabel future returns for the research dataset."""
    resolved_log_path = (
        log_path
        if log_path is not None
        else Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"
    )

    if not resolved_log_path.exists():
        LOGGER.info("Log file not found: %s", resolved_log_path)
        return {
            "total_records": 0,
            "updated_records": 0,
            "skipped_records": 0,
            "force_relabel": force_relabel,
            "thresholds_pct": label_thresholds_pct or DEFAULT_LABEL_THRESHOLDS_PCT,
        }

    resolved_thresholds = _resolve_label_thresholds(label_thresholds_pct)

    exchange = ccxt.binance({"enableRateLimit": True})

    lines = resolved_log_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []

    total_records = 0
    updated_records = 0

    for line in lines:
        if not line.strip():
            continue

        total_records += 1
        record = json.loads(line)

        if _is_fully_labeled(record) and not force_relabel:
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
            record[f"future_label_{horizon}"] = _to_label(
                future_return=future_return,
                horizon=horizon,
                label_thresholds_pct=resolved_thresholds,
            )

        updated_records += 1
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    tmp_path = resolved_log_path.with_suffix(".jsonl.tmp")

    tmp_path.write_text(
        "\n".join(updated_lines) + ("\n" if updated_lines else ""),
        encoding="utf-8",
    )

    tmp_path.replace(resolved_log_path)

    skipped_records = total_records - updated_records

    LOGGER.info(
        "Labeling complete: total=%s updated=%s skipped=%s force_relabel=%s thresholds=%s",
        total_records,
        updated_records,
        skipped_records,
        force_relabel,
        resolved_thresholds,
    )

    return {
        "total_records": total_records,
        "updated_records": updated_records,
        "skipped_records": skipped_records,
        "force_relabel": force_relabel,
        "thresholds_pct": resolved_thresholds,
    }


def _resolve_label_thresholds(
    provided: dict[str, float] | None,
) -> dict[str, float]:
    """Resolve per-horizon label thresholds with validation."""
    thresholds = dict(DEFAULT_LABEL_THRESHOLDS_PCT)

    if provided is not None:
        thresholds.update(provided)

    for horizon in HORIZONS:
        value = thresholds.get(horizon)
        if value is None:
            raise ValueError(f"Missing threshold for horizon: {horizon}")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold for horizon {horizon}: {value}") from exc

        if numeric_value < 0:
            raise ValueError(
                f"Threshold must be non-negative for horizon {horizon}: {numeric_value}"
            )

        thresholds[horizon] = numeric_value

    return thresholds


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
        close_price = float(candle[4])

        prices[horizon] = close_price

    return prices


def _to_label(
    *,
    future_return: float,
    horizon: str,
    label_thresholds_pct: dict[str, float],
) -> str:
    """Assign up/down/flat label using horizon-specific absolute return threshold."""
    threshold_pct = label_thresholds_pct[horizon]

    if future_return > threshold_pct:
        return "up"

    if future_return < -threshold_pct:
        return "down"

    return "flat"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label or relabel future returns in trade_analysis.jsonl"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl",
        help="Path to the trade analysis JSONL file.",
    )
    parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Recompute future returns and relabel rows even if labels already exist.",
    )
    parser.add_argument(
        "--threshold-15m",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["15m"],
        help="Absolute return threshold percentage for 15m flat labeling.",
    )
    parser.add_argument(
        "--threshold-1h",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["1h"],
        help="Absolute return threshold percentage for 1h flat labeling.",
    )
    parser.add_argument(
        "--threshold-4h",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["4h"],
        help="Absolute return threshold percentage for 4h flat labeling.",
    )
    return parser.parse_args()


def main() -> None:
    reporter = CronHealthReporter("future_return_labeler")
    args = _parse_args()

    try:
        result = label_dataset(
            log_path=args.input_path,
            force_relabel=args.force_relabel,
            label_thresholds_pct={
                "15m": args.threshold_15m,
                "1h": args.threshold_1h,
                "4h": args.threshold_4h,
            },
        )
        reporter.success(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        reporter.failure(
            error=exc,
            message="Future return labeling failed",
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    main()