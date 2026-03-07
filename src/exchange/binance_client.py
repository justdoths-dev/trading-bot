import logging
import os
import time
from collections.abc import Callable
from typing import Any

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceMarketDataClient:
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if not api_key or not api_secret:
            logger.warning(
                "BINANCE_API_KEY or BINANCE_API_SECRET is missing. "
                "Public market data requests can still work."
            )

        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )

    def get_latest_price(self, symbol: str) -> float:
        ticker = self._fetch_with_retry(
            operation_name=f"ticker:{symbol}",
            fetch_func=lambda: self.exchange.fetch_ticker(symbol),
        )

        last_price = ticker.get("last")
        if last_price is None:
            raise ValueError("Ticker response does not contain 'last' price.")

        price = float(last_price)
        logger.info("Fetched %s latest price: %s", symbol, price)
        return price

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for reusable timeframes such as 1m, 5m, 15m, 1h, 4h, 1d."""
        ohlcv = self._fetch_with_retry(
            operation_name=f"ohlcv:{symbol}:{timeframe}:{limit}",
            fetch_func=lambda: self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
            ),
        )

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)

        logger.info(
            "Fetched %s OHLCV rows for %s (timeframe=%s, limit=%s)",
            len(df),
            symbol,
            timeframe,
            limit,
        )
        return df

    def _fetch_with_retry(self, operation_name: str, fetch_func: Callable[[], Any]) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Fetching %s (attempt %s/%s)",
                    operation_name,
                    attempt,
                    self.max_retries,
                )
                return fetch_func()

            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                logger.warning(
                    "Attempt %s failed for %s: %s",
                    attempt,
                    operation_name,
                    exc,
                )

                if attempt == self.max_retries:
                    raise

                sleep_seconds = self.retry_delay_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)
