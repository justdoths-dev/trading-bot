from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import settings
from src.data.multi_timeframe_loader import MultiTimeframeLoader
from src.execution.execution_engine import ExecutionEngine
from src.exchange.binance_client import BinanceMarketDataClient
from src.indicators.indicator_engine import IndicatorEngine
from src.risk.risk_manager import RiskManager
from src.strategy.strategy_engine import StrategyEngine


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


def print_strategy_result(result: dict) -> None:
    """Print strategy evaluation result."""
    print("\n=== STRATEGY RESULT ===")
    print(f"Bias   : {result['bias']}")
    print(f"Signal : {result['signal']}")
    print(f"Reason : {result['reason']}")

    print("\n=== STRATEGY LAYERS ===")
    print("Bias Layer   :", result["timeframe_summary"]["bias_layer"]["reason"])
    print("Setup Layer  :", result["timeframe_summary"]["setup_layer"]["reason"])
    print("Trigger Layer:", result["timeframe_summary"]["trigger_layer"]["reason"])

    setup_details = result["debug"]["setup_details"]
    trigger_details = result["debug"]["trigger_details"]

    print("\n=== SETUP DEBUG ===")
    for timeframe in ("1h", "15m"):
        info = setup_details[timeframe]
        print(
            f"[{timeframe}] RSI={info['rsi_14']:.2f}, "
            f"HIST={info['macd_hist_12_26_9']:.6f}, "
            f"PREV_HIST={info['macd_hist_prev']:.6f}"
        )
        print(f"  LONG                -> {info['long_check']}")
        print(f"  SHORT               -> {info['short_check']}")
        print(f"  IMPROVING_LONG      -> {info['improving_long_check']}")
        print(f"  IMPROVING_SHORT     -> {info['improving_short_check']}")
        print(f"  EARLY_RECOVERY_LONG -> {info['early_recovery_long_check']}")
        print(f"  EARLY_RECOVERY_SHORT-> {info['early_recovery_short_check']}")

    print("\n=== TRIGGER DEBUG ===")
    for timeframe in ("5m", "1m"):
        info = trigger_details[timeframe]
        print(
            f"[{timeframe}] RSI={info['rsi_14']:.2f}, "
            f"HIST={info['macd_hist_12_26_9']:.6f}, "
            f"PREV_HIST={info['macd_hist_prev']:.6f}, "
            f"ATR={info['atr_14']:.2f}"
        )
        print(f"  LONG            -> {info['long_check']}")
        print(f"  SHORT           -> {info['short_check']}")
        print(f"  IMPROVING_LONG  -> {info['improving_long_check']}")
        print(f"  IMPROVING_SHORT -> {info['improving_short_check']}")


def print_risk_result(result: dict) -> None:
    """Print risk evaluation result."""
    print("\n=== RISK RESULT ===")
    print(f"Execution Allowed : {result['execution_allowed']}")
    print(f"Entry Price       : {result['entry_price']}")
    print(f"Stop Loss         : {result['stop_loss']}")
    print(f"Take Profit       : {result['take_profit']}")
    print(f"Risk Per Unit     : {result['risk_per_unit']}")
    print(f"Reward Per Unit   : {result['reward_per_unit']}")
    print(f"RR Ratio          : {result['risk_reward_ratio']}")
    print(f"ATR Value         : {result['atr_value']}")
    print(f"Volatility State  : {result['volatility_state']}")
    print(f"Reason            : {result['reason']}")


def print_execution_result(result: dict) -> None:
    """Print execution plan result."""
    print("\n=== EXECUTION RESULT ===")
    print(f"Action            : {result['action']}")
    print(f"Symbol            : {result['symbol']}")
    print(f"Execution Mode    : {result['execution_mode']}")
    print(f"Signal            : {result['signal']}")
    print(f"Bias              : {result['bias']}")
    print(f"Execution Allowed : {result['execution_allowed']}")
    print(f"Entry Price       : {result['entry_price']}")
    print(f"Stop Loss         : {result['stop_loss']}")
    print(f"Take Profit       : {result['take_profit']}")
    print(f"Reason            : {result['reason']}")


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
        macd_fast_period=12,
        macd_slow_period=26,
        macd_signal_period=9,
        atr_period=14,
    )

    enriched_data = indicator_engine.enrich(multi_timeframe_data)

    print_timeframe_preview("ENRICHED DATA", enriched_data)

    strategy_engine = StrategyEngine()
    strategy_result = strategy_engine.evaluate(enriched_data)

    print_strategy_result(strategy_result)

    risk_manager = RiskManager(
        entry_timeframe="5m",
        atr_column="atr_14",
        stop_atr_multiplier=1.5,
        take_profit_atr_multiplier=2.0,
        min_risk_reward_ratio=1.0,
    )
    risk_result = risk_manager.evaluate(strategy_result, enriched_data)

    print_risk_result(risk_result)

    execution_engine = ExecutionEngine(
        symbol="BTCUSDT",
        execution_mode="paper",
    )
    execution_result = execution_engine.create_plan(strategy_result, risk_result)

    print_execution_result(execution_result)


if __name__ == "__main__":
    main()