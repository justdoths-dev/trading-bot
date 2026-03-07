"""Telegram notification service."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not set.")

if not CHAT_ID:
    logger.warning("TELEGRAM_CHAT_ID not set.")

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None


async def _send_async(text: str) -> None:
    await bot.send_message(chat_id=CHAT_ID, text=text)


def send_message(text: str) -> bool:
    """Send a Telegram message."""

    if not bot or not CHAT_ID:
        logger.error("Telegram configuration missing.")
        return False

    try:
        asyncio.run(_send_async(text))
        return True

    except TelegramError as exc:
        logger.error("Telegram error: %s", exc)
        return False

    except Exception:
        logger.exception("Unexpected error while sending Telegram message.")
        return False
