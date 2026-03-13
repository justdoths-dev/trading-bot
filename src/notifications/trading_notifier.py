"""Trading alert notifier built on top of the Telegram sender."""

from __future__ import annotations

import logging
import os
from typing import Any

from src.telegram.markdown_utils import escape_markdown
from src.telegram.telegram_formatter import TelegramFormatter
from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)


class TradingNotifier:
    """Send trading alerts to the configured Telegram trading chat."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ) -> None:
        self._bot_token = (bot_token or os.getenv("TELEGRAM_TRADING_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_TRADING_CHAT_ID", "")).strip()
        self._sender = sender

    def build_pipeline_message(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
    ) -> str:
        """Build the canonical trading pipeline Telegram message."""
        formatter = TelegramFormatter(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_result,
        )
        return formatter.format_message()

    def send_pipeline_alert(
        self,
        symbol: str,
        strategy_result: dict[str, Any],
        risk_result: dict[str, Any],
        execution_result: dict[str, Any],
        ai_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Send the canonical trading pipeline Telegram message."""
        message = self.build_pipeline_message(
            symbol=symbol,
            strategy_result=strategy_result,
            risk_result=risk_result,
            execution_result=execution_result,
            ai_result=ai_result,
        )
        return self.send_message(message)

    def send_message(self, message: str) -> dict[str, Any]:
        """Send a prepared trading message through the trading Telegram channel."""
        sender = self._get_sender()

        if sender is None:
            logger.error(
                "Trading notifier configuration missing. Set TELEGRAM_TRADING_BOT_TOKEN and TELEGRAM_TRADING_CHAT_ID."
            )
            return {
                "sent": False,
                "reason": "Trading notifier configuration missing.",
            }

        try:
            response = sender.send_message(message)
            return {
                "sent": True,
                "reason": "Trading alert sent successfully.",
                "response": response,
            }
        except Exception as exc:
            logger.exception("Failed to send trading alert.")
            return {
                "sent": False,
                "reason": f"Trading alert send failed: {exc}",
            }

    def send_trading_alert(
        self,
        symbol: str,
        strategy: str,
        bias: str,
        entry_price: Any,
        stop_loss: Any,
        take_profit: Any,
        reason: str,
        mode: str = "PAPER",
    ) -> bool:
        """Send a manual trading alert message."""
        message = self._format_manual_alert(
            symbol=symbol,
            strategy=strategy,
            bias=bias,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            mode=mode,
        )
        result = self.send_message(message)
        return bool(result.get("sent", False))

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender

    def _format_manual_alert(
        self,
        *,
        symbol: str,
        strategy: str,
        bias: str,
        entry_price: Any,
        stop_loss: Any,
        take_profit: Any,
        reason: str,
        mode: str,
    ) -> str:
        safe_reason = escape_markdown(
            TelegramFormatter._truncate(str(reason).strip(), 500)
        )

        lines = [
            "*Trading Alert*",
            f"*Mode:* {escape_markdown(mode)}",
            f"*Symbol:* {escape_markdown(symbol)}",
            f"*Strategy:* {escape_markdown(strategy)}",
            f"*Bias:* {escape_markdown(bias)}",
            f"*Entry:* {escape_markdown(self._stringify(entry_price))}",
            f"*Stop Loss:* {escape_markdown(self._stringify(stop_loss))}",
            f"*Take Profit:* {escape_markdown(self._stringify(take_profit))}",
            "*Reason:*",
            safe_reason or "N/A",
        ]

        return "\n".join(lines)

    @staticmethod
    def _stringify(value: Any) -> str:
        if value in (None, ""):
            return "N/A"
        return str(value)
