"""Research summary notifier for Telegram delivery."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.telegram.telegram_sender import TelegramSender
from src.telegram.telegram_formatter import TelegramFormatter
from src.telegram.markdown_utils import escape_markdown

logger = logging.getLogger(__name__)


class ResearchNotifier:

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
        report_dir: Path | None = None,
    ):

        self._bot_token = (bot_token or os.getenv("TELEGRAM_OPS_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_RESEARCH_CHAT_ID", "")).strip()

        self._sender = sender
        self._report_dir = report_dir or self._default_report_dir()

    def send_latest_summary(self) -> bool:

        summary_md = self._read_text_file(self._report_dir / "summary.md")
        summary_json = self._read_json_file(self._report_dir / "summary.json")

        if summary_md is None and summary_json is None:
            logger.error("Research summary files missing.")
            return False

        message = self._format_message(summary_md, summary_json)

        return self._deliver(message)

    def _deliver(self, message: str):

        sender = self._get_sender()

        if sender is None:
            logger.error("Research notifier configuration missing.")
            return False

        try:
            sender.send_message(message)
            return True
        except Exception:
            logger.exception("Failed to send research summary.")
            return False

    def _get_sender(self):

        if self._sender:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender

    def _format_message(self, markdown: str | None, payload: dict[str, Any] | None):

        lines = ["*Research Summary*"]

        if payload:
            dataset_size = payload.get("dataset_size")

            if dataset_size:
                lines.append(f"*Dataset Size:* {dataset_size}")

        if markdown:

            safe_markdown = escape_markdown(
                TelegramFormatter._truncate(markdown.strip(), 3200)
            )

            lines.append("")
            lines.append("*Highlights:*")
            lines.append(safe_markdown)

        return "\n".join(lines)

    @staticmethod
    def _default_report_dir():

        return Path(__file__).resolve().parents[2] / "logs" / "research_reports" / "latest"

    @staticmethod
    def _read_text_file(path: Path):

        if not path.exists():
            return None

        return path.read_text(encoding="utf-8")

    @staticmethod
    def _read_json_file(path: Path):

        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Invalid research JSON.")
            return None