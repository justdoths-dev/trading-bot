from __future__ import annotations

from typing import Any

import pandas as pd


class AIPayloadBuilder:
    """Build structured payload for AI interpretation."""

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol

    def build(
        self,
        enriched_data: dict[str, pd.DataFrame],
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
    ) -> dict[str, Any]:
        timeframe_summary = self._build_timeframe_summary(enriched_data)
        key_bottlenecks = self._build_key_bottlenecks(strategy_result)

        return {
            "symbol": self.symbol,
            "market_summary": {
                "timeframes": timeframe_summary,
                "key_bottlenecks": key_bottlenecks,
            },
            "rule_based_analysis": {
                "bias": strategy_result.get("bias"),
                "signal": strategy_result.get("signal"),
                "reason": strategy_result.get("reason"),
                "layers": {
                    "bias_layer": strategy_result.get("timeframe_summary", {}).get("bias_layer", {}),
                    "setup_layer": strategy_result.get("timeframe_summary", {}).get("setup_layer", {}),
                    "trigger_layer": strategy_result.get("timeframe_summary", {}).get("trigger_layer", {}),
                },
            },
            "risk_analysis": {
                "execution_allowed": risk_result.get("execution_allowed"),
                "entry_price": risk_result.get("entry_price"),
                "stop_loss": risk_result.get("stop_loss"),
                "take_profit": risk_result.get("take_profit"),
                "risk_per_unit": risk_result.get("risk_per_unit"),
                "reward_per_unit": risk_result.get("reward_per_unit"),
                "risk_reward_ratio": risk_result.get("risk_reward_ratio"),
                "atr_value": risk_result.get("atr_value"),
                "volatility_state": risk_result.get("volatility_state"),
                "reason": risk_result.get("reason"),
            },
            "execution_plan": {
                "action": execution_result.get("action"),
                "execution_mode": execution_result.get("execution_mode"),
                "signal": execution_result.get("signal"),
                "bias": execution_result.get("bias"),
                "execution_allowed": execution_result.get("execution_allowed"),
                "entry_price": execution_result.get("entry_price"),
                "stop_loss": execution_result.get("stop_loss"),
                "take_profit": execution_result.get("take_profit"),
                "reason": execution_result.get("reason"),
            },
            "decision_policy": {
                "rule_based_priority": True,
                "ai_role": "higher_level_interpreter_and_decision_support",
                "ai_must_not_replace_indicator_calculations": True,
            },
            "ai_task": {
                "role": "higher_level_market_interpreter",
                "objectives": [
                    "Summarize current market structure across timeframes.",
                    "Explain whether long, short, or no-trade is most reasonable.",
                    "Explain whether the rule-based engine looks too strict or appropriate.",
                    "Describe what confirmation would be needed for a valid long or short.",
                    "Provide a concise trading briefing suitable for Telegram.",
                ],
            },
        }

    def _build_timeframe_summary(
        self,
        enriched_data: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}

        for timeframe, df in enriched_data.items():
            if df.empty:
                summary[timeframe] = {"status": "empty"}
                continue

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else latest

            close_value = self._safe_float(latest.get("close"))
            ema_20_value = self._safe_float(latest.get("ema_20"))
            ema_50_value = self._safe_float(latest.get("ema_50"))
            macd_hist_value = self._safe_float(latest.get("macd_hist_12_26_9"))
            macd_hist_prev_value = self._safe_float(prev.get("macd_hist_12_26_9"))

            summary[timeframe] = {
                "timestamp": str(latest.get("timestamp")),
                "close": close_value,
                "rsi_14": self._safe_float(latest.get("rsi_14")),
                "ema_20": ema_20_value,
                "ema_50": ema_50_value,
                "macd_hist_12_26_9": macd_hist_value,
                "macd_hist_prev_12_26_9": macd_hist_prev_value,
                "atr_14": self._safe_float(latest.get("atr_14")),
                "ema_bias": self._determine_ema_bias(ema_20_value, ema_50_value),
                "momentum_state": self._determine_momentum_state(
                    macd_hist_value,
                    macd_hist_prev_value,
                ),
                "price_vs_ema20": self._compare_price_to_level(close_value, ema_20_value),
                "price_vs_ema50": self._compare_price_to_level(close_value, ema_50_value),
                "rsi_zone": self._determine_rsi_zone(self._safe_float(latest.get("rsi_14"))),
            }

        return summary

    def _build_key_bottlenecks(self, strategy_result: dict[str, Any]) -> list[str]:
        bottlenecks: list[str] = []

        bias_layer = strategy_result.get("timeframe_summary", {}).get("bias_layer", {})
        setup_layer = strategy_result.get("timeframe_summary", {}).get("setup_layer", {})
        trigger_layer = strategy_result.get("timeframe_summary", {}).get("trigger_layer", {})

        bias = bias_layer.get("bias")
        setup = setup_layer.get("setup")
        trigger = trigger_layer.get("trigger")

        if bias == "neutral_conflict":
            bottlenecks.append("Higher timeframe bias is conflicted between 1d and 4h.")

        if setup == "neutral":
            bottlenecks.append("Mid-timeframe setup is not aligned for either long or short.")

        if trigger == "neutral":
            bottlenecks.append("Lower-timeframe trigger is not aligned for execution timing.")

        setup_details = strategy_result.get("debug", {}).get("setup_details", {})
        trigger_details = strategy_result.get("debug", {}).get("trigger_details", {})

        one_hour = setup_details.get("1h", {})
        fifteen_min = setup_details.get("15m", {})
        five_min = trigger_details.get("5m", {})
        one_min = trigger_details.get("1m", {})

        if one_hour.get("early_recovery_long_check", {}).get("passed"):
            bottlenecks.append("1h shows early long recovery, but not a fully confirmed long setup.")

        if fifteen_min.get("improving_short_check", {}).get("passed"):
            bottlenecks.append("15m momentum is weakening, which blocks stronger long continuation.")

        if fifteen_min.get("improving_long_check", {}).get("passed"):
            bottlenecks.append("15m is improving for long, but not yet fully confirmed.")

        if five_min.get("long_check", {}).get("passed") is False and five_min:
            if five_min.get("long_check", {}).get("checks", {}).get("macd_hist_positive") is False:
                bottlenecks.append("5m long trigger is blocked because MACD histogram remains below zero.")

        if one_min.get("long_check", {}).get("passed") is False and one_min:
            if one_min.get("long_check", {}).get("checks", {}).get("macd_hist_rising") is False:
                bottlenecks.append("1m long trigger is blocked because bullish momentum is not increasing.")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in bottlenecks:
            if item not in seen:
                deduped.append(item)
                seen.add(item)

        return deduped

    def _determine_ema_bias(self, ema_20: float | None, ema_50: float | None) -> str:
        if ema_20 is None or ema_50 is None:
            return "unknown"
        if ema_20 > ema_50:
            return "bullish"
        if ema_20 < ema_50:
            return "bearish"
        return "neutral"

    def _determine_momentum_state(
        self,
        latest_hist: float | None,
        prev_hist: float | None,
    ) -> str:
        if latest_hist is None or prev_hist is None:
            return "unknown"

        if latest_hist > 0 and latest_hist > prev_hist:
            return "strengthening_bullish"
        if latest_hist > 0 and latest_hist < prev_hist:
            return "weakening_bullish"
        if latest_hist < 0 and latest_hist < prev_hist:
            return "strengthening_bearish"
        if latest_hist < 0 and latest_hist > prev_hist:
            return "weakening_bearish"
        return "flat"

    def _compare_price_to_level(self, price: float | None, level: float | None) -> str:
        if price is None or level is None:
            return "unknown"
        if price > level:
            return "above"
        if price < level:
            return "below"
        return "at"

    def _determine_rsi_zone(self, rsi_value: float | None) -> str:
        if rsi_value is None:
            return "unknown"
        if rsi_value >= 70:
            return "overbought"
        if rsi_value <= 30:
            return "oversold"
        if rsi_value > 50:
            return "bullish_zone"
        if rsi_value < 50:
            return "bearish_zone"
        return "neutral_zone"

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None