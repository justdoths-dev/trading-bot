from __future__ import annotations

from typing import Any

import pandas as pd


class SetupDetector:
    """Determine whether mid-timeframe conditions support long or short setups."""

    required_columns: tuple[str, ...] = ("rsi_14", "macd_hist_12_26_9")

    def __init__(
        self,
        rsi_long_threshold: float = 50.0,
        rsi_short_threshold: float = 50.0,
        long_recovery_floor: float = 40.0,
        short_recovery_ceiling: float = 60.0,
        early_long_recovery_floor: float = 30.0,
        early_short_recovery_ceiling: float = 70.0,
    ) -> None:
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold
        self.long_recovery_floor = long_recovery_floor
        self.short_recovery_ceiling = short_recovery_ceiling
        self.early_long_recovery_floor = early_long_recovery_floor
        self.early_short_recovery_ceiling = early_short_recovery_ceiling

    def detect(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate_timeframes(enriched_data, required_timeframes=("1h", "15m"))

        one_hour_df = enriched_data["1h"]
        fifteen_min_df = enriched_data["15m"]

        one_hour_latest, one_hour_prev = self._latest_two_rows(one_hour_df, "1h")
        fifteen_latest, fifteen_prev = self._latest_two_rows(fifteen_min_df, "15m")

        one_hour_long_result = self._evaluate_long_setup(one_hour_latest, one_hour_prev)
        fifteen_long_result = self._evaluate_long_setup(fifteen_latest, fifteen_prev)

        one_hour_short_result = self._evaluate_short_setup(one_hour_latest, one_hour_prev)
        fifteen_short_result = self._evaluate_short_setup(fifteen_latest, fifteen_prev)

        one_hour_improving_long = self._evaluate_improving_long(one_hour_latest, one_hour_prev)
        fifteen_improving_long = self._evaluate_improving_long(fifteen_latest, fifteen_prev)

        one_hour_improving_short = self._evaluate_improving_short(one_hour_latest, one_hour_prev)
        fifteen_improving_short = self._evaluate_improving_short(fifteen_latest, fifteen_prev)

        one_hour_early_long = self._evaluate_early_recovery_long(one_hour_latest, one_hour_prev)
        fifteen_early_long = self._evaluate_early_recovery_long(fifteen_latest, fifteen_prev)

        one_hour_early_short = self._evaluate_early_recovery_short(one_hour_latest, one_hour_prev)
        fifteen_early_short = self._evaluate_early_recovery_short(fifteen_latest, fifteen_prev)

        one_hour_long = one_hour_long_result["passed"]
        fifteen_long = fifteen_long_result["passed"]
        one_hour_short = one_hour_short_result["passed"]
        fifteen_short = fifteen_short_result["passed"]

        one_hour_improving_long_pass = one_hour_improving_long["passed"]
        fifteen_improving_long_pass = fifteen_improving_long["passed"]
        one_hour_improving_short_pass = one_hour_improving_short["passed"]
        fifteen_improving_short_pass = fifteen_improving_short["passed"]

        one_hour_early_long_pass = one_hour_early_long["passed"]
        fifteen_early_long_pass = fifteen_early_long["passed"]
        one_hour_early_short_pass = one_hour_early_short["passed"]
        fifteen_early_short_pass = fifteen_early_short["passed"]

        if one_hour_long and fifteen_long:
            setup = "long"
        elif one_hour_short and fifteen_short:
            setup = "short"
        elif one_hour_improving_long_pass and fifteen_improving_long_pass:
            setup = "improving_long"
        elif one_hour_improving_short_pass and fifteen_improving_short_pass:
            setup = "improving_short"
        elif one_hour_early_long_pass and fifteen_long:
            setup = "early_recovery_long"
        elif one_hour_early_short_pass and fifteen_short:
            setup = "early_recovery_short"
        elif fifteen_long and one_hour_early_long_pass:
            setup = "early_recovery_long"
        elif fifteen_short and one_hour_early_short_pass:
            setup = "early_recovery_short"
        else:
            setup = "neutral"

        return {
            "setup": setup,
            "reason": (
                f"1h long={one_hour_long} short={one_hour_short} "
                f"improving_long={one_hour_improving_long_pass} improving_short={one_hour_improving_short_pass} "
                f"early_long={one_hour_early_long_pass} early_short={one_hour_early_short_pass}; "
                f"15m long={fifteen_long} short={fifteen_short} "
                f"improving_long={fifteen_improving_long_pass} improving_short={fifteen_improving_short_pass} "
                f"early_long={fifteen_early_long_pass} early_short={fifteen_early_short_pass}"
            ),
            "details": {
                "1h": {
                    "rsi_14": float(one_hour_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(one_hour_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(one_hour_prev["macd_hist_12_26_9"]),
                    "long_check": one_hour_long_result,
                    "short_check": one_hour_short_result,
                    "improving_long_check": one_hour_improving_long,
                    "improving_short_check": one_hour_improving_short,
                    "early_recovery_long_check": one_hour_early_long,
                    "early_recovery_short_check": one_hour_early_short,
                },
                "15m": {
                    "rsi_14": float(fifteen_latest["rsi_14"]),
                    "macd_hist_12_26_9": float(fifteen_latest["macd_hist_12_26_9"]),
                    "macd_hist_prev": float(fifteen_prev["macd_hist_12_26_9"]),
                    "long_check": fifteen_long_result,
                    "short_check": fifteen_short_result,
                    "improving_long_check": fifteen_improving_long,
                    "improving_short_check": fifteen_improving_short,
                    "early_recovery_long_check": fifteen_early_long,
                    "early_recovery_short_check": fifteen_early_short,
                },
            },
        }

    def _evaluate_long_setup(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_pass = rsi_value > self.rsi_long_threshold
        hist_rising_pass = hist_latest > hist_prev

        return {
            "passed": rsi_pass and hist_rising_pass,
            "checks": {
                "rsi_above_threshold": rsi_pass,
                "macd_hist_rising": hist_rising_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_threshold": self.rsi_long_threshold,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

    def _evaluate_short_setup(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_pass = rsi_value < self.rsi_short_threshold
        hist_falling_pass = hist_latest < hist_prev

        return {
            "passed": rsi_pass and hist_falling_pass,
            "checks": {
                "rsi_below_threshold": rsi_pass,
                "macd_hist_falling": hist_falling_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_threshold": self.rsi_short_threshold,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

    def _evaluate_improving_long(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_recovery_pass = self.long_recovery_floor <= rsi_value <= self.rsi_long_threshold
        hist_recovery_pass = hist_latest > hist_prev

        return {
            "passed": rsi_recovery_pass and hist_recovery_pass,
            "checks": {
                "rsi_recovery_zone": rsi_recovery_pass,
                "macd_hist_recovering": hist_recovery_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_recovery_floor": self.long_recovery_floor,
                "rsi_threshold": self.rsi_long_threshold,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

    def _evaluate_improving_short(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_recovery_pass = self.rsi_short_threshold <= rsi_value <= self.short_recovery_ceiling
        hist_recovery_pass = hist_latest < hist_prev

        return {
            "passed": rsi_recovery_pass and hist_recovery_pass,
            "checks": {
                "rsi_recovery_zone": rsi_recovery_pass,
                "macd_hist_weakening": hist_recovery_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "rsi_threshold": self.rsi_short_threshold,
                "rsi_recovery_ceiling": self.short_recovery_ceiling,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

    def _evaluate_early_recovery_long(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_floor_pass = self.early_long_recovery_floor <= rsi_value < self.long_recovery_floor
        hist_recovering_pass = hist_latest > hist_prev and hist_latest > 0

        return {
            "passed": rsi_floor_pass and hist_recovering_pass,
            "checks": {
                "rsi_early_recovery_zone": rsi_floor_pass,
                "macd_hist_positive_and_rising": hist_recovering_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "early_long_recovery_floor": self.early_long_recovery_floor,
                "long_recovery_floor": self.long_recovery_floor,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

    def _evaluate_early_recovery_short(self, latest: pd.Series, prev: pd.Series) -> dict[str, Any]:
        rsi_value = float(latest["rsi_14"])
        hist_latest = float(latest["macd_hist_12_26_9"])
        hist_prev = float(prev["macd_hist_12_26_9"])

        rsi_zone_pass = self.short_recovery_ceiling < rsi_value <= self.early_short_recovery_ceiling
        hist_weakening_pass = hist_latest < hist_prev and hist_latest < 0

        return {
            "passed": rsi_zone_pass and hist_weakening_pass,
            "checks": {
                "rsi_early_recovery_zone": rsi_zone_pass,
                "macd_hist_negative_and_falling": hist_weakening_pass,
            },
            "values": {
                "rsi_14": rsi_value,
                "short_recovery_ceiling": self.short_recovery_ceiling,
                "early_short_recovery_ceiling": self.early_short_recovery_ceiling,
                "macd_hist_latest": hist_latest,
                "macd_hist_prev": hist_prev,
            },
        }

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