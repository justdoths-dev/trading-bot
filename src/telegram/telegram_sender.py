"""Telegram sender module for Bot API delivery."""

from __future__ import annotations

from typing import Any

import requests


class TelegramSender:
    """Send messages to a Telegram chat using Bot API."""

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 10) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.timeout_seconds = timeout_seconds

        if not self.bot_token:
            raise ValueError("Telegram bot token is required.")

        if not self.chat_id:
            raise ValueError("Telegram chat id is required.")

        self._endpoint = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def send_message(self, text: str) -> dict[str, Any]:
        """Send a text message and return Telegram API response payload."""

        message = (text or "").strip()

        if not message:
            raise ValueError("Message text cannot be empty.")

        message = self._truncate(message)

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to send Telegram message: {exc}") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Telegram API returned a non-JSON response.") from exc

        if not body.get("ok"):
            description = body.get("description", "Unknown Telegram API error")
            raise RuntimeError(f"Telegram API error: {description}")

        return body

    def _truncate(self, message: str) -> str:
        """Ensure message does not exceed Telegram limits."""
        if len(message) <= self.MAX_MESSAGE_LENGTH:
            return message
        return message[: self.MAX_MESSAGE_LENGTH - 3] + "..."
