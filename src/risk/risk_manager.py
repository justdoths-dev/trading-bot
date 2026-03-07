from __future__ import annotations

from typing import Any

import pandas as pd


class RiskManager:
    """ATR-based risk manager v1."""

    required_columns: tuple[str, ...] = ("close", "atr_14")

    def __init__(
        self,
        entry_timeframe: str = "5m",
        atr_column: str = "atr_14",
        stop_atr_multiplier: float = 1.5,
        take_profit_atr_multiplier: float = 2.0,
        min_risk_reward_ratio: float = 1.0,
    ) -> None:
        if stop_atr_multiplier <= 0:
            raise ValueError("stop_atr_multiplier must be greater than 0")
        if take_profit_atr_multiplier <= 0:
            raise ValueError("take_profit_atr_multiplier must be greater than 0")
        if min_risk_reward_ratio <= 0:
            raise ValueError("min_risk_reward_ratio must be greater than 0")

        self.entry_timeframe = entry_timeframe
        self.atr_column = atr_column
        self.stop_atr_multiplier = stop_atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.min_risk_reward_ratio = min_risk_reward_ratio

    def evaluate(
        self,
        strategy_result: dict[str, Any],
        enriched_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        signal = strategy_result.get("signal", "no_signal")

        if signal == "no_signal":
            return {
                "execution_allowed": False,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_per_unit": None,
                "reward_per_unit": None,
                "risk_reward_ratio": None,
                "atr_value": None,
                "volatility_state": "unknown",
                "reason": "Strategy signal is no_signal, so execution is not allowed.",
            }

        if self.entry_timeframe not in enriched_data:
            raise ValueError(
                f"Entry timeframe '{self.entry_timeframe}' is missing from enriched_data."
            )

        df = enriched_data[self.entry_timeframe]
        latest_row = self._latest_row(df, self.entry_timeframe)

        entry_price = float(latest_row["close"])
        atr_value = float(latest_row[self.atr_column])

        volatility_state = self._classify_volatility(latest_row)

        if signal == "long":
            stop_loss = entry_price - (self.stop_atr_multiplier * atr_value)
            take_profit = entry_price + (self.take_profit_atr_multiplier * atr_value)
            risk_per_unit = entry_price - stop_loss
            reward_per_unit = take_profit - entry_price

        elif signal == "short":
            stop_loss = entry_price + (self.stop_atr_multiplier * atr_value)
            take_profit = entry_price - (self.take_profit_atr_multiplier * atr_value)
            risk_per_unit = stop_loss - entry_price
            reward_per_unit = entry_price - take_profit

        else:
            return {
                "execution_allowed": False,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_per_unit": None,
                "reward_per_unit": None,
                "risk_reward_ratio": None,
                "atr_value": atr_value,
                "volatility_state": volatility_state,
                "reason": f"Unsupported strategy signal: {signal}",
            }

        risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

        execution_allowed = risk_reward_ratio >= self.min_risk_reward_ratio

        if execution_allowed:
            reason = (
                f"Signal={signal}, ATR-based levels calculated successfully on "
                f"{self.entry_timeframe}, RR={risk_reward_ratio:.2f}"
            )
        else:
            reason = (
                f"Signal={signal}, but risk/reward ratio {risk_reward_ratio:.2f} "
                f"is below minimum threshold {self.min_risk_reward_ratio:.2f}"
            )

        return {
            "execution_allowed": execution_allowed,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_per_unit": round(risk_per_unit, 2),
            "reward_per_unit": round(reward_per_unit, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "atr_value": round(atr_value, 2),
            "volatility_state": volatility_state,
            "reason": reason,
        }

    def _latest_row(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        if df.empty:
            raise ValueError(f"Timeframe '{timeframe}' DataFrame is empty.")

        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Timeframe '{timeframe}' is missing required columns: {missing}"
            )

        row = df.iloc[-1]

        for col in self.required_columns:
            if pd.isna(row[col]):
                raise ValueError(
                    f"Timeframe '{timeframe}' latest row has NaN in required column '{col}'."
                )

        return row

    def _classify_volatility(self, row: pd.Series) -> str:
        close_price = float(row["close"])
        atr_value = float(row[self.atr_column])

        if close_price <= 0:
            return "unknown"

        atr_ratio = atr_value / close_price

        if atr_ratio < 0.001:
            return "low"
        if atr_ratio < 0.003:
            return "normal"
        return "high"