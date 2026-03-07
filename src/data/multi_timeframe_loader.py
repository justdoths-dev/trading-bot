from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exchange.binance_client import BinanceMarketDataClient


@dataclass
class TimeframeConfig:
    timeframe: str
    limit: int


class MultiTimeframeLoader:
    def __init__(self, client: BinanceMarketDataClient) -> None:
        self.client = client

    def load(
        self,
        symbol: str,
        configs: list[TimeframeConfig],
    ) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}

        for config in configs:
            df = self.client.get_ohlcv(
                symbol=symbol,
                timeframe=config.timeframe,
                limit=config.limit,
            )
            data[config.timeframe] = df

        return data