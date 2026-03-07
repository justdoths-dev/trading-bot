from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ai.ai_service import AIService, AIServiceConfig
from src.config.settings import settings
from src.data.multi_timeframe_loader import MultiTimeframeLoader
from src.execution.execution_engine import ExecutionEngine
from src.exchange.binance_client import BinanceMarketDataClient
from src.indicators.indicator_engine import IndicatorEngine
from src.risk.risk_manager import RiskManager
from src.storage.trade_analysis_logger import (
    TradeAnalysisLogger,
    TradeAnalysisLoggerConfig,
)
from src.strategy.strategy_engine import StrategyEngine
from src.telegram.telegram_formatter import TelegramFormatter


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
    for timeframe, df in data.items():
        print(f"\n=== {title} | {timeframe} ===")

        if df.empty:
            print("DataFrame is empty.")
            continue

        print(df.tail(tail_size))


def build_timeframe_configs() -> list[TimeframeConfig]:
    return [
        TimeframeConfig(timeframe="1m", limit=100),
        TimeframeConfig(timeframe="5m", limit=100),
        TimeframeConfig(timeframe="15m", limit=100),
        TimeframeConfig(timeframe="1h", limit=100),
        TimeframeConfig(timeframe="4h", limit=100),
        TimeframeConfig(timeframe="1d", limit=100),
    ]


def print_strategy_result(result: dict[str, Any]) -> None:
    print("\n=== STRATEGY RESULT ===")
    print(f"Bias   : {result['bias']}")
    print(f"Signal : {result['signal']}")
    print(f"Reason : {result['reason']}")

    print("\n=== STRATEGY LAYERS ===")
    print("Bias Layer   :", result["timeframe_summary"]["bias_layer"]["reason"])
    print("Setup Layer  :", result["timeframe_summary"]["setup_layer"]["reason"])
    print("Trigger Layer:", result["timeframe_summary"]["trigger_layer"]["reason"])


def print_risk_result(result: dict[str, Any]) -> None:
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


def print_execution_result(result: dict[str, Any]) -> None:
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


def print_ai_payload(payload: dict[str, Any]) -> None:
    print("\n=== AI PAYLOAD ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def print_ai_prompt(prompt: str, max_chars: int = 3000) -> None:
    print("\n=== AI PROMPT ===")
    if len(prompt) <= max_chars:
        print(prompt)
    else:
        print(prompt[:max_chars])
        print("\n... [truncated for preview] ...")


def print_ai_result(result: dict[str, Any]) -> None:
    print("\n=== AI RESULT ===")
    print(f"Source      : {result['source']}")
    print(f"Model       : {result['model']}")
    print(f"Environment : {result['environment']}")
    print(f"Generated At: {result['generated_at']}")

    analysis = result["analysis"]

    print("\n--- MARKET STRUCTURE ---")
    print(analysis["market_structure"])

    print("\n--- RULE ENGINE ASSESSMENT ---")
    print(analysis["rule_engine_assessment"])

    print("\n--- KEY BOTTLENECKS ---")
    for idx, item in enumerate(analysis["key_bottlenecks"], start=1):
        print(f"{idx}. {item}")

    print("\n--- LONG SCENARIO ---")
    print(analysis["long_scenario"])

    print("\n--- SHORT SCENARIO ---")
    print(analysis["short_scenario"])

    print("\n--- FINAL STANCE ---")
    print(f"Stance : {analysis['final_stance']}")
    print(f"Reason : {analysis['stance_reason']}")

    print("\n--- TELEGRAM BRIEFING ---")
    for line in analysis["telegram_briefing"]:
        print(f"- {line}")


def print_log_result(record: dict[str, Any]) -> None:
    print("\n=== LOG RESULT ===")
    print(f"Logged At             : {record['logged_at']}")
    print(f"Symbol                : {record['symbol']}")
    print(f"Rule Signal           : {record['rule_engine']['signal']}")
    print(f"Execution Action      : {record['execution']['action']}")
    print(f"AI Final Stance       : {record['ai']['final_stance']}")
    print(f"Execution / AI Aligned: {record['alignment']['is_aligned']}")


def print_telegram_message_preview(message: str) -> None:
    print("\n=== TELEGRAM MESSAGE PREVIEW ===")
    print(message)


def main() -> None:
    symbol = "BTCUSDT"

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
        symbol=symbol,
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
        symbol=symbol,
        execution_mode="paper",
    )
    execution_result = execution_engine.create_plan(strategy_result, risk_result)

    print_execution_result(execution_result)

    ai_service = AIService(
        config=AIServiceConfig(
            model="gpt-4.1-mini",
            timeout_seconds=30,
            max_retries=2,
            retry_backoff_seconds=1.5,
            environment="paper",
            symbol=symbol,
        )
    )

    ai_output = ai_service.run(
        enriched_data=enriched_data,
        strategy_result=strategy_result,
        risk_result=risk_result,
        execution_result=execution_result,
    )

    print_ai_payload(ai_output["payload"])
    print_ai_prompt(ai_output["prompt"])
    print_ai_result(ai_output["result"])

    logger = TradeAnalysisLogger(
        config=TradeAnalysisLoggerConfig(
            log_dir="logs",
            filename="trade_analysis.jsonl",
            utc_timestamp=True,
        )
    )

    log_record = logger.log(
        symbol=symbol,
        strategy_result=strategy_result,
        risk_result=risk_result,
        execution_result=execution_result,
        ai_result=ai_output["result"],
    )

    print_log_result(log_record)

    telegram_formatter = TelegramFormatter(
        symbol=symbol,
        strategy_result=strategy_result,
        risk_result=risk_result,
        execution_result=execution_result,
        ai_result=ai_output["result"],
    )

    telegram_message = telegram_formatter.format_message()
    print_telegram_message_preview(telegram_message)


if __name__ == "__main__":
    main()