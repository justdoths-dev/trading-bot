from __future__ import annotations

from typing import Any

import pandas as pd


class SetupDetector:
    """Determine whether mid-timeframe conditions support long or short setups."""

    required_columns: tuple[str, ...] = ("rsi_14", "macd_hist_12_26_9")

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_timeframes(enriched_data, required_timeframes=("1h", "15m"))

        one_hour_df = enriched_data["1h"]
        fifteen_min_df = enriched_data["15m"]

        one_hour_latest, one_hour_prev = self._latest_two_rows(one_hour_df, "1h")
        fifteen_latest, fifteen_prev = self._latest_two_rows(fifteen_min_df, "15m")

        one_hour_long = self._is_long_setup(one_hour_latest, one_hour_prev)
        fifteen_long = self._is_long_setup(fifteen_latest, fifteen_prev)

        one_hour_short = self._is_short_setup(one_hour_latest, one_hour_prev)
        fifteen_short = self._is_short_setup(fifteen_latest, fifteen_prev)

        if one_hour_long and fifteen_long:
            setup = "long"
        elif one_hour_short and fifteen_short:
            setup = "short"
        else:
            setup = "neutral"

        return {
            "setup": setup,
            "reason": (
                f"1h long={one_hour_long} short={one_hour_short}; "
                f"15m long={fifteen_long} short={fifteen_short}"
            ),
            "details": {
                "1h": {
                    "rsi_14": float(one_hour_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(one_hour_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(one_hour_prev["macd_hist_12_26_9"]),
                },
                "15m": {
                    "rsi_14": float(fifteen_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(fifteen_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(fifteen_prev["macd_hist_12_26_9"]),
                },
            },
        }

    def _is_long_setup(self, latest: pd.Series, prev: pd.Series) -> bool:
        return (
            float(latest["rsi_14"]) > 50
            and float(latest["macd_hist_12_26_9"]) > float(prev["macd_hist_12_26_9"])
        )

    def _is_short_setup(self, latest: pd.Series, prev: pd.Series) -> bool:
        return (
            float(latest["rsi_14"]) < 50
            and float(latest["macd_hist_12_26_9"]) < float(prev["macd_hist_12_26_9"])
        )

    def _validate_timeframes(
        self,
        enriched_data: dict[str, pd.DataFrame],
        required_timeframes: tuple[str, ...],
    ) -> None:
        missing = [tf for tf in required_timeframes if tf not in enriched_data]
        if missing:
            raise ValueError(f"Missing required timeframes for setup detection: {missing}")

    def _latest_two_rows(self, df: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series]:
        if len(df) < 2:
            raise ValueError(
                f"Timeframe '{timeframe}' must contain at least 2 rows for setup detection."
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