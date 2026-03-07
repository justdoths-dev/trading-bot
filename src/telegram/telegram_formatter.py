"""Telegram message formatting utilities for manual trade review."""

from __future__ import annotations

from typing import Any


class TelegramFormatter:
    """Build concise Telegram-ready summaries from pipeline outputs."""

    def __init__(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
        max_text_length: int = 280,
    ) -> None:
        self.symbol = symbol
        self.strategy_result = strategy_result or {}
        self.risk_result = risk_result or {}
        self.execution_result = execution_result or {}
        self.ai_result = ai_result or {}
        self.max_text_length = max_text_length

    def format_message(self) -> str:
        """Return a Telegram-ready message string."""
        analysis = self._analysis()

        rule_bias = self._value(self.strategy_result, "bias")
        rule_signal = self._value(self.strategy_result, "signal")
        execution_action = self._value(self.execution_result, "action")
        execution_allowed = self._execution_allowed()

        ai_final_stance = self._value(analysis, "final_stance")
        ai_stance_reason = self._truncate(
            self._value(analysis, "stance_reason"),
            self.max_text_length,
        )

        key_bottlenecks = self._as_list(analysis.get("key_bottlenecks"))
        briefing_lines = self._as_list(analysis.get("telegram_briefing"))

        lines: list[str] = []

        # Header
        lines.append(f"[{self.symbol}]")
        lines.append(f"Signal: {rule_signal}")
        lines.append(f"Bias: {rule_bias}")
        lines.append(f"Action: {execution_action}")
        lines.append(f"Execution Allowed: {execution_allowed}")
        lines.append(f"AI Stance: {ai_final_stance}")
        lines.append("")

        # Optional execution levels
        execution_block = self._build_execution_block()
        if execution_block:
            lines.append("Execution Plan")
            lines.extend(execution_block)
            lines.append("")

        # AI stance reason
        lines.append("Stance Reason")
        lines.append(f"- {ai_stance_reason}")
        lines.append("")

        # Key bottlenecks
        lines.append("Key Bottlenecks")
        lines.extend(self._format_bullets(key_bottlenecks, limit=3))
        lines.append("")

        # Telegram briefing
        lines.append("Briefing")
        lines.extend(self._format_bullets(briefing_lines, limit=4))

        return "\n".join(lines).strip()

    def _analysis(self) -> dict[str, Any]:
        analysis = self.ai_result.get("analysis")
        return analysis if isinstance(analysis, dict) else {}

    def _execution_allowed(self) -> str:
        if "execution_allowed" in self.execution_result:
            return str(self.execution_result.get("execution_allowed"))
        if "execution_allowed" in self.risk_result:
            return str(self.risk_result.get("execution_allowed"))
        return "N/A"

    def _build_execution_block(self) -> list[str]:
        """
        Show price levels only when execution is actually allowed.
        """
        execution_allowed = self.execution_result.get("execution_allowed")
        if execution_allowed is not True:
            return []

        entry_price = self._value(self.execution_result, "entry_price")
        stop_loss = self._value(self.execution_result, "stop_loss")
        take_profit = self._value(self.execution_result, "take_profit")

        lines = [
            f"- Entry: {entry_price}",
            f"- Stop Loss: {stop_loss}",
            f"- Take Profit: {take_profit}",
        ]

        rr_ratio = self.risk_result.get("risk_reward_ratio")
        if rr_ratio not in (None, ""):
            lines.append(f"- RR Ratio: {rr_ratio}")

        return lines

    @staticmethod
    def _value(mapping: dict[str, Any], key: str, default: str = "N/A") -> str:
        value = mapping.get(key, default)
        return default if value in (None, "") else str(value)

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return cleaned
        return []

    def _format_bullets(self, items: list[str], limit: int | None = None) -> list[str]:
        if not items:
            return ["- N/A"]

        selected = items[:limit] if limit is not None else items
        return [f"- {self._truncate(item, self.max_text_length)}" for item in selected]

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."
