from src.data.multi_timeframe_loader import MultiTimeframeLoader, TimeframeConfig
from src.exchange.binance_client import BinanceMarketDataClient


def main():
    client = BinanceMarketDataClient()
    loader = MultiTimeframeLoader(client)

    configs = [
        TimeframeConfig("1m", 10),
        TimeframeConfig("5m", 10),
        TimeframeConfig("15m", 10),
        TimeframeConfig("1h", 10),
        TimeframeConfig("4h", 10),
        TimeframeConfig("1d", 10),
    ]

    multi_tf_data = loader.load("BTC/USDT", configs)

    for timeframe, df in multi_tf_data.items():
        print(f"\n=== {timeframe} ===")
        print(df.tail())


if __name__ == "__main__":
    main()