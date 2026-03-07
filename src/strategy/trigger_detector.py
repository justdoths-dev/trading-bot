from __future__ import annotations

from typing import Any

import pandas as pd


class TriggerDetector:
    """Determine lower-timeframe entry trigger conditions."""

    required_columns: tuple[str, ...] = ("rsi_14", "macd_hist_12_26_9", "atr_14")

    def __init__(
        self,
        rsi_long_threshold: float = 50.0,
        rsi_short_threshold: float = 50.0,
        max_volatility_ratio: float = 0.003,
    ) -> None:
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.max_volatility_ratio = max_volatility_ratio

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_timeframes(enriched_data, required_timeframes=("5m", "1m"))

        five_min_df = enriched_data["5m"]
        one_min_df = enriched_data["1m"]

        five_latest, five_prev = self._latest_two_rows(five_min_df, "5m")
        one_latest, one_prev = self._latest_two_rows(one_min_df, "1m")

        five_long_result = self._evaluate_long_trigger(five_latest, five_prev)
        one_long_result = self._evaluate_long_trigger(one_latest, one_prev)

        five_short_result = self._evaluate_short_trigger(five_latest, five_prev)
        one_short_result = self._evaluate_short_trigger(one_latest, one_prev)

        five_long = five_long_result["passed"]
        one_long = one_long_result["passed"]
        five_short = five_short_result["passed"]
        one_short = one_short_result["passed"]

        if five_long and one_long:
            trigger = "long"
        elif five_short and one_short:
            trigger = "short"
        else:
            trigger = "neutral"

        return {
            "trigger": trigger,
            "reason": (
                f"5m long={five_long} short={five_short}; "
                f"1m long={one_long} short={one_short}"
            ),
            "details": {
                "5m": {
                    "rsi_14": float(five_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(five_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(five_prev["macd_hist_12_26_9"]),
                    "atr_14": float(five_latest["atr_14"]),
                    "long_check": five_long_result,
                    "short_check": five_short_result,
                },
                "1m": {
                    "rsi_14": float(one_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(one_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(one_prev["macd_hist_12_26_9"]),
                    "atr_14": float(one_latest["atr_14"]),
                    "long_check": one_long_result,
                    "short_check": one_short_result,
                },
            },
        }

    def _evaluate_long_trigger(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])
        atr_value = float(latest["atr_14"])
        close_value = float(latest["close"])

        rsi_pass = rsi_value > self.rsi_long_threshold
        hist_positive_pass = hist_latest > 0
        hist_rising_pass = hist_latest > hist_prev
        volatility_pass = self._volatility_ratio(atr_value, close_value) <= self.max_volatility_ratio

        return {
            "passed": rsi_pass and hist_positive_pass and hist_rising_pass and volatility_pass,
            "checks": {
                "rsi_above_threshold": rsi_pass,
                "macd_hist_positive": hist_positive_pass,
                "macd_hist_rising": hist_rising_pass,
                "volatility_ok": volatility_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_threshold": self.rsi_long_threshold,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
                "atr_14": atr_value,
                "close": close_value,
                "atr_ratio": self._volatility_ratio(atr_value, close_value),
                "max_volatility_ratio": self.max_volatility_ratio,
            },
        }

    def _evaluate_short_trigger(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])
        atr_value = float(latest["atr_14"])
        close_value = float(latest["close"])

        rsi_pass = rsi_value < self.rsi_short_threshold
        hist_negative_pass = hist_latest < 0
        hist_falling_pass = hist_latest < hist_prev
        volatility_pass = self._volatility_ratio(atr_value, close_value) <= self.max_volatility_ratio

        return {
            "passed": rsi_pass and hist_negative_pass and hist_falling_pass and volatility_pass,
            "checks": {
                "rsi_below_threshold": rsi_pass,
                "macd_hist_negative": hist_negative_pass,
                "macd_hist_falling": hist_falling_pass,
                "volatility_ok": volatility_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_threshold": self.rsi_short_threshold,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
                "atr_14": atr_value,
                "close": close_value,
                "atr_ratio": self._volatility_ratio(atr_value, close_value),
                "max_volatility_ratio": self.max_volatility_ratio,
            },
        }

    def _volatility_ratio(self, atr_value: float, close_value: float) -> float:
        if close_value <= 0:
            return 0.0
        return atr_value / close_value

    def _validate_timeframes(
        self,
        enriched_data: dict[str, pd.DataFrame],
        required_timeframes: tuple[str, ...],
    ) -> None:
        missing = [tf for tf in required_timeframes if tf not in enriched_data]
        if missing:
            raise ValueError(f"Missing required timeframes for trigger detection: {missing}")

    def _latest_two_rows(self, df: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series]:
        if len(df) < 2:
            raise ValueError(
                f"Timeframe '{timeframe}' must contain at least 2 rows for trigger detection."
            )

        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Timeframe '{timeframe}' is missing required columns: {missing}"
            )

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        for col in self.required_columns:
            if pd.isna(latest[col]) or pd.isna(prev[col]):
                raise ValueError(
                    f"Timeframe '{timeframe}' has NaN in required column '{col}'."
                )

        return latest, prev