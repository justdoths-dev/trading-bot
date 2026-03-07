from __future__ import annotations

from typing import Any

import pandas as pd


class TriggerDetector:
    """Determine lower-timeframe entry trigger conditions."""

    required_columns: tuple[str, ...] = ("rsi_14", "macd_hist_12_26_9", "atr_14")

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_timeframes(enriched_data, required_timeframes=("5m", "1m"))

        five_min_df = enriched_data["5m"]
        one_min_df = enriched_data["1m"]

        five_latest, five_prev = self._latest_two_rows(five_min_df, "5m")
        one_latest, one_prev = self._latest_two_rows(one_min_df, "1m")

        five_long = self._is_long_trigger(five_latest, five_prev)
        one_long = self._is_long_trigger(one_latest, one_prev)

        five_short = self._is_short_trigger(five_latest, five_prev)
        one_short = self._is_short_trigger(one_latest, one_prev)

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
                },
                "1m": {
                    "rsi_14": float(one_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(one_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(one_prev["macd_hist_12_26_9"]),
                    "atr_14": float(one_latest["atr_14"]),
                },
            },
        }

    def _is_long_trigger(self, latest: pd.Series, prev: pd.Series) -> bool:
        return (
            float(latest["rsi_14"]) > 50
            and float(latest["macd_hist_12_26_9"]) > 0
            and float(latest["macd_hist_12_26_9"]) > float(prev["macd_hist_12_26_9"])
        )

    def _is_short_trigger(self, latest: pd.Series, prev: pd.Series) -> bool:
        return (
            float(latest["rsi_14"]) < 50
            and float(latest["macd_hist_12_26_9"]) < 0
            and float(latest["macd_hist_12_26_9"]) < float(prev["macd_hist_12_26_9"])
        )

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