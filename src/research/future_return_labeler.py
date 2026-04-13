from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import ccxt

from src.research.future_return_labeling_common import (
    DEFAULT_LABEL_THRESHOLDS_PCT,
    HORIZONS,
    apply_future_update,
    build_future_update_from_prices,
    extract_entry_price,
    has_future_fields_for_horizon,
    horizon_to_timedelta,
    normalize_symbol,
    parse_logged_at_to_utc,
)
from src.research.inputs.current_trade_analysis_resolver import (
    BASE_TRADE_ANALYSIS_FILENAME,
    discover_current_trade_analysis_files,
)
from src.research.future_return_market_data import fetch_horizon_prices
from src.services.cron_health import CronHealthReporter

LOGGER = logging.getLogger(__name__)


def label_dataset(
    *,
    log_path: Path | None = None,
    force_relabel: bool = False,
    label_thresholds_pct: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Label or relabel future returns for the research dataset."""
    resolved_log_path = (
        log_path
        if log_path is not None
        else Path(__file__).resolve().parents[2] / "logs" / BASE_TRADE_ANALYSIS_FILENAME
    )
    target_paths = _resolve_target_paths(resolved_log_path)

    if not target_paths:
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

    total_records = 0
    updated_records = 0

    for target_path in target_paths:
        file_counts = _label_single_file(
            log_path=target_path,
            exchange=exchange,
            force_relabel=force_relabel,
            label_thresholds_pct=resolved_thresholds,
        )
        total_records += file_counts["total_records"]
        updated_records += file_counts["updated_records"]

    skipped_records = total_records - updated_records

    LOGGER.info(
        "Labeling complete: files=%s total=%s updated=%s skipped=%s force_relabel=%s thresholds=%s",
        len(target_paths),
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


def _resolve_target_paths(log_path: Path) -> list[Path]:
    if log_path.name != BASE_TRADE_ANALYSIS_FILENAME:
        return [log_path] if log_path.exists() else []

    return discover_current_trade_analysis_files(
        log_path.parent,
        include_rotated_base=False,
    )


def _label_single_file(
    *,
    log_path: Path,
    exchange: Any,
    force_relabel: bool,
    label_thresholds_pct: dict[str, float],
) -> dict[str, int]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {
            "total_records": 0,
            "updated_records": 0,
        }

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
            update = build_future_update_from_prices(
                entry_price=entry_price,
                future_price=future_price,
                horizon=horizon,
                label_thresholds_pct=label_thresholds_pct,
            )
            if update is None:
                continue
            record = apply_future_update(record, update)

        updated_records += 1
        updated_lines.append(json.dumps(record, ensure_ascii=False))

    tmp_path = log_path.with_suffix(".jsonl.tmp")
    tmp_path.write_text(
        "\n".join(updated_lines) + ("\n" if updated_lines else ""),
        encoding="utf-8",
    )
    tmp_path.replace(log_path)

    return {
        "total_records": total_records,
        "updated_records": updated_records,
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
    return all(
        has_future_fields_for_horizon(record, horizon)
        for horizon in HORIZONS.keys()
    )


def _normalize_symbol(raw_symbol: Any) -> str | None:
    return normalize_symbol(raw_symbol)


def _parse_logged_at(raw_logged_at: Any):
    return parse_logged_at_to_utc(raw_logged_at)


def _extract_entry_price(record: dict[str, Any]) -> float | None:
    return extract_entry_price(record)


def _fetch_horizon_prices(
    *,
    exchange: Any,
    symbol: str,
    logged_at,
) -> dict[str, float] | None:
    return fetch_horizon_prices(
        exchange=exchange,
        symbol=symbol,
        logged_at=logged_at,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label or relabel future returns in trade_analysis.jsonl"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "logs" / BASE_TRADE_ANALYSIS_FILENAME,
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
