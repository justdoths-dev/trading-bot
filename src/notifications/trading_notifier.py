"""Trading alert notifier built on top of the Telegram sender."""

import logging
import os
from typing import Any

from src.telegram.telegram_sender import TelegramSender
from src.telegram.telegram_formatter import TelegramFormatter
from src.telegram.markdown_utils import escape_markdown

logger = logging.getLogger(__name__)


class TradingNotifier:
    """Send trading alerts to the configured Telegram trading chat."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ):

        self._bot_token = (bot_token or os.getenv("TELEGRAM_TRADING_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_TRADING_CHAT_ID", "")).strip()

        self._sender = sender

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

        message = self._format_message(
            symbol,
            strategy,
            bias,
            entry_price,
            stop_loss,
            take_profit,
            reason,
            mode,
        )

        return self._deliver(message)

    def _deliver(self, message: str) -> bool:

        sender = self._get_sender()

        if sender is None:
            logger.error("Trading notifier configuration missing.")
            return False

        try:
            sender.send_message(message)
            return True
        except Exception:
            logger.exception("Failed to send trading alert.")
            return False

    def _get_sender(self):

        if self._sender:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender

    def _format_message(
        self,
        symbol,
        strategy,
        bias,
        entry_price,
        stop_loss,
        take_profit,
        reason,
        mode,
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
            f"*Entry:* {escape_markdown(str(entry_price))}",
            f"*Stop Loss:* {escape_markdown(str(stop_loss))}",
            f"*Take Profit:* {escape_markdown(str(take_profit))}",
            "*Reason:*",
            safe_reason or "N/A",
        ]

        return "\n".join(lines)
