from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import ccxt

from src.research.future_return_labeling_common import (
    apply_future_update,
    build_future_update_for_row,
    extract_entry_price,
    has_future_fields_for_horizon,
    is_mature_for_horizon,
    iter_supported_horizons,
    normalize_symbol,
    parse_logged_at_to_utc,
)
from src.research.future_return_market_data import fetch_horizon_prices
from src.research.strategy_lab.dataset_builder import (
    DEFAULT_LATEST_MAX_ROWS,
    DEFAULT_LATEST_WINDOW_HOURS,
    load_jsonl_records_with_metadata,
)

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_PATH = (
    Path("logs/research_reports/latest")
    / "recent_window_relabeled_trade_analysis.jsonl"
)


def materialize_recent_window_labels(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_age_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> dict[str, Any]:
    rows, metadata = load_jsonl_records_with_metadata(
        path=input_path,
        max_age_hours=max_age_hours,
        max_rows=max_rows,
    )

    source_files = [Path(value) for value in metadata.get("source_files", [])]
    _validate_output_path(output_path=output_path, source_files=source_files)

    latest_ts = _latest_logged_at(rows)
    exchange = ccxt.binance({"enableRateLimit": True})
    price_cache: dict[tuple[str, str], dict[str, float] | None] = {}

    per_horizon = {
        horizon: {
            "mature_unlabeled_seen": 0,
            "updated": 0,
            "still_missing_after_attempt": 0,
        }
        for horizon in iter_supported_horizons()
    }

    rows_updated = 0
    mature_unlabeled_seen = 0
    mature_unlabeled_horizon_count = 0
    rows_still_missing_after_attempt = 0
    invalid_rows_skipped_for_update = 0
    fetch_failures = 0
    updated_rows: list[dict[str, Any]] = []

    for row in rows:
        updated_row = dict(row)
        row_ts = parse_logged_at_to_utc(row.get("logged_at"))
        row_changed = False
        mature_unlabeled_horizons: list[str] = []

        for horizon in iter_supported_horizons():
            if not is_mature_for_horizon(row_ts, latest_ts, horizon):
                continue
            if has_future_fields_for_horizon(updated_row, horizon):
                continue
            mature_unlabeled_horizons.append(horizon)
            per_horizon[horizon]["mature_unlabeled_seen"] += 1
            mature_unlabeled_horizon_count += 1

        if mature_unlabeled_horizons:
            mature_unlabeled_seen += 1

        unresolved_horizons = list(mature_unlabeled_horizons)

        symbol = normalize_symbol(updated_row.get("symbol"))
        entry_price = extract_entry_price(updated_row)

        if mature_unlabeled_horizons:
            if symbol is None or row_ts is None or entry_price is None or entry_price <= 0:
                invalid_rows_skipped_for_update += 1
            else:
                cache_key = (symbol, row_ts.isoformat())
                if cache_key not in price_cache:
                    price_cache[cache_key] = fetch_horizon_prices(
                        exchange=exchange,
                        symbol=symbol,
                        logged_at=row_ts,
                    )

                horizon_prices = price_cache[cache_key]
                if horizon_prices is None:
                    fetch_failures += 1
                else:
                    unresolved_horizons = []

                    for horizon in mature_unlabeled_horizons:
                        future_price = horizon_prices.get(horizon)
                        if future_price is None:
                            unresolved_horizons.append(horizon)
                            continue

                        update = build_future_update_for_row(
                            row=updated_row,
                            future_price=future_price,
                            horizon=horizon,
                        )
                        if update is None:
                            unresolved_horizons.append(horizon)
                            continue

                        updated_row = apply_future_update(updated_row, update)
                        per_horizon[horizon]["updated"] += 1
                        row_changed = True

        for horizon in unresolved_horizons:
            per_horizon[horizon]["still_missing_after_attempt"] += 1

        if unresolved_horizons:
            rows_still_missing_after_attempt += 1

        if row_changed:
            rows_updated += 1

        updated_rows.append(updated_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in updated_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_file_count": len(source_files),
        "total_rows": len(updated_rows),
        "rows_updated": rows_updated,
        "rows_preserved": len(updated_rows) - rows_updated,
        "mature_unlabeled_seen": mature_unlabeled_seen,
        "mature_unlabeled_horizon_count": mature_unlabeled_horizon_count,
        "rows_still_missing_after_attempt": rows_still_missing_after_attempt,
        "invalid_rows_skipped_for_update": invalid_rows_skipped_for_update,
        "fetch_failures": fetch_failures,
        "per_horizon": per_horizon,
    }


def _latest_logged_at(rows: list[dict[str, Any]]) -> datetime | None:
    timestamps = [
        parsed
        for row in rows
        if (parsed := parse_logged_at_to_utc(row.get("logged_at"))) is not None
    ]
    if not timestamps:
        return None
    return max(timestamps)


def _validate_output_path(*, output_path: Path, source_files: list[Path]) -> None:
    resolved_output = output_path.resolve()
    resolved_inputs = {path.resolve() for path in source_files}

    if resolved_output in resolved_inputs:
        raise ValueError(
            "Output path must not overlap with any input source file: "
            f"{output_path}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a recent-window trade_analysis JSONL dataset with "
            "missing mature future-return labels filled using production-equivalent "
            "market price fetches"
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Base path for trade_analysis JSONL input. Default: logs/trade_analysis.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Output path for the materialized recent-window relabeled JSONL dataset. "
            "Default: logs/research_reports/latest/recent_window_relabeled_trade_analysis.jsonl"
        ),
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=DEFAULT_LATEST_WINDOW_HOURS,
        help="Maximum recent-window age in hours. Default: 36",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_LATEST_MAX_ROWS,
        help="Maximum recent-window row count. Default: 2500",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = materialize_recent_window_labels(
        input_path=args.input_path,
        output_path=args.output_path,
        max_age_hours=args.max_age_hours,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()