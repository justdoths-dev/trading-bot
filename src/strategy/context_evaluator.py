"""Detector-oriented market context evaluator for multi-timeframe analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .decision_models import LayerResult


class ContextEvaluator:
    """Classify higher and mid-timeframe market context."""

    required_timeframes: tuple[str, ...] = ("1d", "4h", "1h", "15m")
    required_columns: tuple[str, ...] = (
        "close",
        "ema_20",
        "ema_50",
        "rsi_14",
        "macd_hist_12_26_9",
    )

    def evaluate(self, enriched_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        self._validate(enriched_data)

        timeframe_states = {
            timeframe: self._build_timeframe_state(enriched_data[timeframe], timeframe)
            for timeframe in self.required_timeframes
        }

        higher_biases = (
            timeframe_states["1d"]["bias"],
            timeframe_states["4h"]["bias"],
        )
        mid_biases = (
            timeframe_states["1h"]["bias"],
            timeframe_states["15m"]["bias"],
        )

        higher_alignment = self._alignment(*higher_biases)
        mid_alignment = self._alignment(*mid_biases)
        higher_direction = self._dominant_direction(*higher_biases)
        mid_direction = self._dominant_direction(*mid_biases)
        higher_strength = self._average_strength(timeframe_states, ("1d", "4h"))
        mid_strength = self._average_strength(timeframe_states, ("1h", "15m"))

        context, bias = self._classify_context(
            higher_alignment=higher_alignment,
            mid_alignment=mid_alignment,
            higher_direction=higher_direction,
            mid_direction=mid_direction,
            higher_strength=higher_strength,
            mid_strength=mid_strength,
        )

        confidence = self._context_confidence(timeframe_states, context)
        reason = (
            f"higher={higher_alignment}:{higher_direction}({higher_strength:.3f}), "
            f"mid={mid_alignment}:{mid_direction}({mid_strength:.3f}), context={context}"
        )

        layer = LayerResult(
            name="context",
            state=context,
            bias=bias,
            confidence=confidence,
            reason=reason,
            details={
                "higher_alignment": higher_alignment,
                "higher_direction": higher_direction,
                "higher_strength": round(higher_strength, 4),
                "mid_alignment": mid_alignment,
                "mid_direction": mid_direction,
                "mid_strength": round(mid_strength, 4),
                "timeframes": timeframe_states,
            },
        )
        return {
            "context": context,
            "bias": bias,
            "confidence": confidence,
            "reason": reason,
            "details": layer.details,
            "layer": layer.to_dict(),
        }

    def _classify_context(
        self,
        *,
        higher_alignment: str,
        mid_alignment: str,
        higher_direction: str,
        mid_direction: str,
        higher_strength: float,
        mid_strength: float,
    ) -> tuple[str, str]:
        if higher_alignment == "conflicted":
            return "conflicted", "neutral"

        if higher_direction == "bullish":
            if mid_direction == "bearish":
                return "countertrend_drop", "bullish"
            if mid_direction == "bullish" and mid_alignment == "aligned":
                return "bullish_trend", "bullish"
            if higher_alignment == "aligned" or higher_strength >= 0.33:
                return "weak_bullish_context", "bullish"

        if higher_direction == "bearish":
            if mid_direction == "bullish":
                return "countertrend_bounce", "bearish"
            if mid_direction == "bearish" and mid_alignment == "aligned":
                return "bearish_trend", "bearish"
            if higher_alignment == "aligned" or higher_strength >= 0.33:
                return "weak_bearish_context", "bearish"

        if mid_alignment == "aligned" and mid_direction == "bullish" and mid_strength >= 0.45:
            return "weak_bullish_context", "bullish"
        if mid_alignment == "aligned" and mid_direction == "bearish" and mid_strength >= 0.45:
            return "weak_bearish_context", "bearish"

        return "neutral", "neutral"

    def _context_confidence(
        self,
        timeframe_states: dict[str, dict[str, Any]],
        context: str,
    ) -> float:
        absolute_scores = [
            abs(float(timeframe_states[timeframe]["score"]))
            for timeframe in self.required_timeframes
        ]
        base = sum(absolute_scores) / len(absolute_scores)

        regime_multiplier = {
            "bullish_trend": 0.96,
            "bearish_trend": 0.96,
            "weak_bullish_context": 0.78,
            "weak_bearish_context": 0.78,
            "countertrend_bounce": 0.66,
            "countertrend_drop": 0.66,
            "conflicted": 0.34,
            "neutral": 0.46,
        }.get(context, 0.5)

        return round(min(base * regime_multiplier, 0.86), 4)

    def _build_timeframe_state(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, Any]:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        ema_20 = float(latest["ema_20"])
        ema_50 = float(latest["ema_50"])
        close = float(latest["close"])
        rsi_14 = float(latest["rsi_14"])
        macd_hist = float(latest["macd_hist_12_26_9"])
        macd_prev = float(prev["macd_hist_12_26_9"])

        score = 0.0

        if ema_20 > ema_50:
            score += 1.15
        elif ema_20 < ema_50:
            score -= 1.15

        if close > ema_20:
            score += 0.35
        elif close < ema_20:
            score -= 0.35

        if close > ema_50:
            score += 0.55
        elif close < ema_50:
            score -= 0.55

        if rsi_14 >= 57.0:
            score += 0.45
        elif rsi_14 <= 43.0:
            score -= 0.45

        if macd_hist > 0:
            score += 0.6
        elif macd_hist < 0:
            score -= 0.6

        if macd_hist > macd_prev:
            score += 0.35
        elif macd_hist < macd_prev:
            score -= 0.35

        normalized_score = max(-1.0, min(score / 3.45, 1.0))
        bias = self._score_to_bias(normalized_score)
        momentum_state = self._momentum_state(macd_hist, macd_prev)

        return {
            "timeframe": timeframe,
            "bias": bias,
            "score": round(normalized_score, 4),
            "close": close,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi_14": rsi_14,
            "macd_hist_12_26_9": macd_hist,
            "macd_hist_prev_12_26_9": macd_prev,
            "momentum_state": momentum_state,
            "price_vs_ema20": "above" if close > ema_20 else "below" if close < ema_20 else "at",
            "price_vs_ema50": "above" if close > ema_50 else "below" if close < ema_50 else "at",
        }

    def _average_strength(
        self,
        timeframe_states: dict[str, dict[str, Any]],
        timeframes: tuple[str, ...],
    ) -> float:
        return sum(abs(float(timeframe_states[timeframe]["score"])) for timeframe in timeframes) / len(timeframes)

    def _alignment(self, first: str, second: str) -> str:
        if first == second and first in {"bullish", "bearish"}:
            return "aligned"
        if "neutral" in {first, second}:
            return "mixed"
        return "conflicted"

    def _dominant_direction(self, first: str, second: str) -> str:
        if first == second:
            return first
        if first == "neutral":
            return second
        if second == "neutral":
            return first
        return "neutral"

    def _score_to_bias(self, score: float) -> str:
        if score >= 0.18:
            return "bullish"
        if score <= -0.18:
            return "bearish"
        return "neutral"

    def _momentum_state(self, latest: float, prev: float) -> str:
        if latest > 0 and latest > prev:
            return "strengthening_bullish"
        if latest > 0 and latest < prev:
            return "weakening_bullish"
        if latest < 0 and latest < prev:
            return "strengthening_bearish"
        if latest < 0 and latest > prev:
            return "weakening_bearish"
        return "flat"

    def _validate(self, enriched_data: dict[str, pd.DataFrame]) -> None:
        missing_timeframes = [
            timeframe
            for timeframe in self.required_timeframes
            if timeframe not in enriched_data
        ]
        if missing_timeframes:
            raise ValueError(
                "Missing required timeframes for context evaluation: "
                f"{missing_timeframes}"
            )

        for timeframe in self.required_timeframes:
            df = enriched_data[timeframe]
            if len(df) < 2:
                raise ValueError(
                    f"Timeframe '{timeframe}' must contain at least 2 rows."
                )

            missing_columns = [
                column
                for column in self.required_columns
                if column not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Timeframe '{timeframe}' missing required columns: "
                    f"{missing_columns}"
                )

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            for column in self.required_columns:
                if pd.isna(latest[column]) or pd.isna(prev[column]):
                    raise ValueError(
                        f"Timeframe '{timeframe}' contains NaN in '{column}'."
                    )
