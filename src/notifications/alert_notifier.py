"""System alert notifier for ops and cron workflows."""

import logging
import os
from datetime import UTC, datetime

from src.telegram.telegram_sender import TelegramSender
from src.telegram.markdown_utils import escape_markdown

logger = logging.getLogger(__name__)


class AlertNotifier:

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ):

        self._bot_token = (bot_token or os.getenv("TELEGRAM_OPS_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_ALERT_CHAT_ID", "")).strip()

        self._sender = sender

    def send_error_alert(self, source: str, message: str, details: str | None = None):

        return self._send_alert("System Error", source, message, details)

    def send_warning_alert(self, source: str, message: str, details: str | None = None):

        return self._send_alert("System Warning", source, message, details)

    def send_cron_alert(self, job_name: str, status: str, details: str | None = None):

        return self._send_alert("Cron Alert", job_name, status, details)

    def _send_alert(self, title, source, message, details):

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"*{escape_markdown(title)}*",
            f"*Source:* {escape_markdown(source)}",
            f"*Time:* {escape_markdown(timestamp)}",
            f"*Message:* {escape_markdown(message)}",
        ]

        if details:
            lines.append("*Details:*")
            lines.append(escape_markdown(details))

        msg = "\n".join(lines)

        sender = self._get_sender()

        if sender is None:
            logger.error("Alert notifier configuration missing.")
            return False

        try:
            sender.send_message(msg)
            return True
        except Exception:
            logger.exception("Failed to send alert.")
            return False

    def _get_sender(self):

        if self._sender:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)

        return self._sender
