from __future__ import annotations

from typing import Any


class ExecutionEngine:
    """Execution layer v1 for paper trading plans."""

    def __init__(self, symbol: str = "BTCUSDT", execution_mode: str = "paper") -> None:
        if execution_mode not in ("paper", "live"):
            raise ValueError("execution_mode must be either 'paper' or 'live'")

        self.symbol = symbol
        self.execution_mode = execution_mode

    def create_plan(
        self,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
    ) -> dict[str, Any]:
        signal = strategy_result.get("signal", "no_signal")
        bias = strategy_result.get("bias", "unknown")
        strategy_reason = strategy_result.get("reason", "No strategy reason provided.")
        risk_reason = risk_result.get("reason", "No risk reason provided.")

        execution_allowed = bool(risk_result.get("execution_allowed", False))

        if signal in ("no_signal", "watchlist_long", "watchlist_short"):
            return {
                "action": "hold",
                "symbol": self.symbol,
                "execution_mode": self.execution_mode,
                "signal": signal,
                "bias": bias,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "execution_allowed": False,
                "reason": (
                    f"No executable trade plan. "
                    f"Strategy reason: {strategy_reason} | Risk reason: {risk_reason}"
                ),
            }

        if not execution_allowed:
            return {
                "action": "hold",
                "symbol": self.symbol,
                "execution_mode": self.execution_mode,
                "signal": signal,
                "bias": bias,
                "entry_price": risk_result.get("entry_price"),
                "stop_loss": risk_result.get("stop_loss"),
                "take_profit": risk_result.get("take_profit"),
                "execution_allowed": False,
                "reason": (
                    f"Signal exists but execution is blocked by risk rules. "
                    f"Strategy reason: {strategy_reason} | Risk reason: {risk_reason}"
                ),
            }

        if signal == "long":
            action = "buy"
        elif signal == "short":
            action = "sell"
        else:
            action = "hold"

        return {
            "action": action,
            "symbol": self.symbol,
            "execution_mode": self.execution_mode,
            "signal": signal,
            "bias": bias,
            "entry_price": risk_result.get("entry_price"),
            "stop_loss": risk_result.get("stop_loss"),
            "take_profit": risk_result.get("take_profit"),
            "execution_allowed": execution_allowed,
            "reason": (
                f"Executable {signal} plan created. "
                f"Strategy reason: {strategy_reason} | Risk reason: {risk_reason}"
            ),
        }