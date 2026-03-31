"""Base class for modular trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for strategy implementations."""

    strategy_name: str = "base"
    required_timeframes: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Evaluate strategy and return signal payload."""

    def _validate_data(self, data: dict[str, pd.DataFrame]) -> None:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict[str, pandas.DataFrame]")

        for timeframe in self.required_timeframes:
            if timeframe not in data:
                raise ValueError(
                    f"Missing timeframe '{timeframe}' for {self.strategy_name} strategy"
                )

            df = data[timeframe]

            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Timeframe '{timeframe}' must be a pandas DataFrame"
                )

            if df.empty:
                raise ValueError(
                    f"Timeframe '{timeframe}' DataFrame is empty"
                )

            if "close" not in df.columns:
                raise ValueError(
                    f"Timeframe '{timeframe}' missing required 'close' column"
                )

    @staticmethod
    def _momentum_score(df: pd.DataFrame) -> float:
        """Compute momentum score using log returns."""
        if len(df) < 2:
            return 0.0

        prev_close = float(df["close"].iloc[-2])
        last_close = float(df["close"].iloc[-1])

        if prev_close <= 0 or last_close <= 0:
            return 0.0

        return float(np.log(last_close / prev_close))

    @staticmethod
    def _to_signal(score: float, threshold: float = 0.0004) -> str:
        if score > threshold:
            return "long"
        if score < -threshold:
            return "short"
        return "hold"

    @staticmethod
    def _to_confidence(score: float, scale: float = 600.0) -> float:
        confidence = min(abs(score) * scale, 1.0)
        return round(confidence, 4)

    @staticmethod
    def _indicator_score(
        df: pd.DataFrame,
        *,
        trend_weight: float = 1.1,
        momentum_weight: float = 0.9,
        price_weight: float = 0.75,
        rsi_weight: float = 0.55,
    ) -> float:
        """Build a bounded directional score from indicator state."""
        if len(df) < 2:
            return 0.0

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = BaseStrategy._safe_numeric(latest.get("close"))
        ema_20 = BaseStrategy._safe_numeric(latest.get("ema_20"))
        ema_50 = BaseStrategy._safe_numeric(latest.get("ema_50"))
        rsi_14 = BaseStrategy._safe_numeric(latest.get("rsi_14"))
        macd_hist = BaseStrategy._safe_numeric(latest.get("macd_hist_12_26_9"))
        macd_prev = BaseStrategy._safe_numeric(prev.get("macd_hist_12_26_9"))

        score = 0.0

        if ema_20 is not None and ema_50 is not None:
            if ema_20 > ema_50:
                score += trend_weight
            elif ema_20 < ema_50:
                score -= trend_weight

        if close is not None and ema_20 is not None and ema_50 is not None:
            if close > ema_20:
                score += price_weight * 0.45
            elif close < ema_20:
                score -= price_weight * 0.45

            if close > ema_50:
                score += price_weight * 0.55
            elif close < ema_50:
                score -= price_weight * 0.55

        if rsi_14 is not None:
            if rsi_14 >= 57.0:
                score += rsi_weight
            elif rsi_14 <= 43.0:
                score -= rsi_weight

        if macd_hist is not None:
            if macd_hist > 0:
                score += momentum_weight * 0.65
            elif macd_hist < 0:
                score -= momentum_weight * 0.65

        if macd_hist is not None and macd_prev is not None:
            if macd_hist > macd_prev:
                score += momentum_weight * 0.35
            elif macd_hist < macd_prev:
                score -= momentum_weight * 0.35

        return float(max(-1.0, min(score / 3.3, 1.0)))

    @staticmethod
    def _directional_signal(score: float, threshold: float = 0.22) -> str:
        if score >= threshold:
            return "long"
        if score <= -threshold:
            return "short"
        return "hold"

    @staticmethod
    def _bounded_confidence(
        score: float,
        *,
        cap: float = 0.72,
        floor: float = 0.16,
    ) -> float:
        magnitude = abs(score)
        if magnitude < 0.22:
            return round(max(0.0, magnitude * 0.3), 4)

        confidence = floor + (magnitude - 0.22) * 0.7
        return round(min(confidence, cap), 4)

    @staticmethod
    def _score_to_bias(score: float) -> str:
        if score >= 0.22:
            return "bullish"
        if score <= -0.22:
            return "bearish"
        return "neutral"

    @classmethod
    def _build_timeframe_state(cls, df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        score = cls._indicator_score(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        macd_hist = cls._safe_numeric(latest.get("macd_hist_12_26_9"))
        macd_prev = cls._safe_numeric(prev.get("macd_hist_12_26_9"))

        return {
            "timeframe": timeframe,
            "bias": cls._score_to_bias(score),
            "signal": cls._directional_signal(score),
            "score": round(score, 4),
            "close": cls._safe_numeric(latest.get("close")),
            "ema_20": cls._safe_numeric(latest.get("ema_20")),
            "ema_50": cls._safe_numeric(latest.get("ema_50")),
            "rsi_14": cls._safe_numeric(latest.get("rsi_14")),
            "macd_hist_12_26_9": macd_hist,
            "macd_hist_prev_12_26_9": macd_prev,
            "momentum_state": cls._momentum_state(macd_hist, macd_prev),
        }

    @staticmethod
    def _momentum_state(latest: float | None, prev: float | None) -> str:
        if latest is None or prev is None:
            return "unknown"
        if latest > 0 and latest > prev:
            return "strengthening_bullish"
        if latest > 0 and latest < prev:
            return "weakening_bullish"
        if latest < 0 and latest < prev:
            return "strengthening_bearish"
        if latest < 0 and latest > prev:
            return "weakening_bearish"
        return "flat"

    @staticmethod
    def _safe_numeric(value: Any) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
