from __future__ import annotations

import pandas as pd

from src.ai.payload_builder import AIPayloadBuilder


def test_payload_builder_includes_compact_bollinger_snapshot() -> None:
    builder = AIPayloadBuilder(symbol="BTCUSDT")

    enriched_data = {
        "1m": pd.DataFrame(
            [
                {
                    "timestamp": "2026-04-04T00:00:00+00:00",
                    "close": 101.0,
                    "rsi_14": 55.0,
                    "ema_20": 100.5,
                    "ema_50": 99.5,
                    "macd_hist_12_26_9": 0.3,
                    "atr_14": 1.2,
                    "bb_middle_20": 100.0,
                    "bb_std_20": 2.5,
                    "bb_upper_20_2": 105.0,
                    "bb_lower_20_2": 95.0,
                    "bb_bandwidth_20_2": 0.1,
                    "bb_percent_b_20_2": 0.6,
                },
                {
                    "timestamp": "2026-04-04T00:01:00+00:00",
                    "close": 102.0,
                    "rsi_14": 57.0,
                    "ema_20": 100.8,
                    "ema_50": 99.8,
                    "macd_hist_12_26_9": 0.5,
                    "atr_14": 1.3,
                    "bb_middle_20": 100.5,
                    "bb_std_20": 2.75,
                    "bb_upper_20_2": 106.0,
                    "bb_lower_20_2": 95.0,
                    "bb_bandwidth_20_2": 0.109453,
                    "bb_percent_b_20_2": 0.636364,
                },
            ]
        )
    }

    payload = builder.build(
        enriched_data=enriched_data,
        strategy_result={},
        risk_result={},
        execution_result={},
    )

    assert payload["market_summary"]["timeframes"]["1m"]["bollinger_20_2"] == {
        "middle": 100.5,
        "std": 2.75,
        "upper": 106.0,
        "lower": 95.0,
        "bandwidth": 0.109453,
        "percent_b": 0.636364,
    }


def test_payload_builder_keeps_ai_read_only_and_rule_engine_authoritative() -> None:
    builder = AIPayloadBuilder(symbol="BTCUSDT")

    payload = builder.build(
        enriched_data={
            "1m": pd.DataFrame(
                [
                    {
                        "timestamp": "2026-04-04T00:01:00+00:00",
                        "close": 102.0,
                        "rsi_14": 57.0,
                        "ema_20": 100.8,
                        "ema_50": 99.8,
                        "macd_hist_12_26_9": 0.5,
                        "atr_14": 1.3,
                        "bb_middle_20": 100.5,
                        "bb_std_20": 2.75,
                        "bb_upper_20_2": 106.0,
                        "bb_lower_20_2": 95.0,
                        "bb_bandwidth_20_2": 0.109453,
                        "bb_percent_b_20_2": 0.636364,
                    }
                ]
            )
        },
        strategy_result={},
        risk_result={},
        execution_result={},
    )

    decision_policy = payload["decision_policy"]

    assert decision_policy["rule_based_priority"] is True
    assert decision_policy["ai_must_not_override_rule_engine"] is True
    assert decision_policy["ai_must_not_modify_ranking_or_selection"] is True
    assert decision_policy["ai_must_not_trigger_execution"] is True
    assert payload["ai_task"]["role"] == "read_only_higher_level_market_interpreter"
