from __future__ import annotations

from typing import Any

import pandas as pd


class BiasDetector:
    """Determine higher-timeframe directional bias."""

    required_columns: tuple[str, ...] = ("ema_20", "ema_50")

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_timeframes(enriched_data, required_timeframes=("1d", "4h"))

        daily_row = self._latest_row(enriched_data["1d"], "1d")
        four_hour_row = self._latest_row(enriched_data["4h"], "4h")

        daily_bias = self._ema_bias(daily_row)
        four_hour_bias = self._ema_bias(four_hour_row)

        if daily_bias == "bullish" and four_hour_bias == "bullish":
            bias = "bullish"
        elif daily_bias == "bearish" and four_hour_bias == "bearish":
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "bias": bias,
            "reason": (
                f"1d ema20/ema50={daily_bias}, "
                f"4h ema20/ema50={four_hour_bias}"
            ),
            "details": {
                "1d": {
                    "ema_20": float(daily_row["ema_20"]),
                    "ema_50": float(daily_row["ema_50"]),
                    "bias": daily_bias,
                },
                "4h": {
                    "ema_20": float(four_hour_row["ema_20"]),
                    "ema_50": float(four_hour_row["ema_50"]),
                    "bias": four_hour_bias,
                },
            },
        }

    def _ema_bias(self, row: pd.Series) -> str:
        ema_20 = float(row["ema_20"])
        ema_50 = float(row["ema_50"])

        if ema_20 > ema_50:
            return "bullish"
        if ema_20 < ema_50:
            return "bearish"
        return "neutral"

    def _validate_timeframes(
        self,
        enriched_data: dict[str, pd.DataFrame],
        required_timeframes: tuple[str, ...],
    ) -> None:
        missing = [tf for tf in required_timeframes if tf not in enriched_data]
        if missing:
            raise ValueError(f"Missing required timeframes for bias detection: {missing}")

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