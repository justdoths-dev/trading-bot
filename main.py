from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.services.trading_pipeline import TradingPipeline, TradingPipelineConfig


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


def print_telegram_send_result(result: dict[str, Any]) -> None:
    print("\n=== TELEGRAM SEND RESULT ===")
    print(f"Sent   : {result['sent']}")
    print(f"Reason : {result['reason']}")


def main() -> None:
    pipeline = TradingPipeline(
        config=TradingPipelineConfig(
            symbol="BTCUSDT",
            send_telegram=True,
        )
    )

    result = pipeline.run()

    print_timeframe_preview("RAW DATA", result["raw_data"])
    print_timeframe_preview("ENRICHED DATA", result["enriched_data"])

    print_strategy_result(result["strategy_result"])
    print_risk_result(result["risk_result"])
    print_execution_result(result["execution_result"])

    print_ai_payload(result["ai_output"]["payload"])
    print_ai_prompt(result["ai_output"]["prompt"])
    print_ai_result(result["ai_output"]["result"])

    print_log_result(result["log_record"])
    print_telegram_message_preview(result["telegram_message"])
    print_telegram_send_result(result["telegram_send_result"])


if __name__ == "__main__":
    main()