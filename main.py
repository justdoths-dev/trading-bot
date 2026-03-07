from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import settings
from src.data.multi_timeframe_loader import MultiTimeframeLoader
from src.exchange.binance_client import BinanceMarketDataClient
from src.indicators.indicator_engine import IndicatorEngine


@dataclass
class TimeframeConfig:
    """Configuration for one timeframe data request."""
    timeframe: str
    limit: int


def print_timeframe_preview(
    title: str,
    data: dict[str, pd.DataFrame],
    tail_size: int = 5,
) -> None:
    """Print the last rows for each timeframe DataFrame."""
    for timeframe, df in data.items():
        print(f"\n=== {title} | {timeframe} ===")

        if df.empty:
            print("DataFrame is empty.")
            continue

        print(df.tail(tail_size))


def build_timeframe_configs() -> list[TimeframeConfig]:
    """Return the default multi-timeframe configuration set."""
    return [
        TimeframeConfig(timeframe="1m", limit=100),
        TimeframeConfig(timeframe="5m", limit=100),
        TimeframeConfig(timeframe="15m", limit=100),
        TimeframeConfig(timeframe="1h", limit=100),
        TimeframeConfig(timeframe="4h", limit=100),
        TimeframeConfig(timeframe="1d", limit=100),
    ]


def main() -> None:
    api_key = settings.binance_api_key
    api_secret = settings.binance_api_secret

    if not api_key or not api_secret:
        print(
            "BINANCE_API_KEY or BINANCE_API_SECRET is missing. "
            "Public market data requests can still work."
        )

    client = BinanceMarketDataClient()
    loader = MultiTimeframeLoader(client=client)

    configs = build_timeframe_configs()

    multi_timeframe_data = loader.load(
        symbol="BTCUSDT",
        configs=configs,
    )

    print_timeframe_preview("RAW DATA", multi_timeframe_data)

    indicator_engine = IndicatorEngine(
        rsi_period=14,
        ema_fast_period=20,
        ema_slow_period=50,
    )

    enriched_data = indicator_engine.enrich(multi_timeframe_data)

    print_timeframe_preview("ENRICHED DATA", enriched_data)


if __name__ == "__main__":
    main()