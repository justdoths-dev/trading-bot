"""Research summary notifier for concise Telegram delivery."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.telegram.markdown_utils import escape_markdown
from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)


class ResearchNotifier:
    """Read the latest research report output and send a concise summary to Telegram."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
        report_dir: Path | None = None,
    ) -> None:
        self._bot_token = (bot_token or os.getenv("TELEGRAM_OPS_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_RESEARCH_CHAT_ID", "")).strip()
        self._sender = sender
        self._report_dir = report_dir or self._default_report_dir()

    def send_latest_summary(self) -> bool:
        """Send the latest research summary to the configured Telegram chat."""
        summary_payload = self._read_json_file(self._report_dir / "summary.json")
        summary_markdown = self._read_text_file(self._report_dir / "summary.md")

        if summary_payload is None and summary_markdown is None:
            logger.error(
                "Latest research summary files are missing from %s.",
                self._report_dir,
            )
            return False

        message = self._format_message(summary_payload, summary_markdown)
        return self._deliver(message)

    def _deliver(self, message: str) -> bool:
        sender = self._get_sender()
        if sender is None:
            logger.error(
                "Research notifier configuration missing. Set TELEGRAM_OPS_BOT_TOKEN and TELEGRAM_RESEARCH_CHAT_ID."
            )
            return False

        try:
            sender.send_message(message)
            return True
        except Exception:
            logger.exception("Failed to send research summary.")
            return False

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender

    def _format_message(
        self,
        payload: dict[str, Any] | None,
        markdown_text: str | None,
    ) -> str:
        if payload is None:
            return self._fallback_markdown_message(markdown_text)

        schema_validation = payload.get("schema_validation", {}) or {}
        dataset_overview = payload.get("dataset_overview", {}) or {}
        top_highlights = payload.get("top_highlights", {}) or {}

        lines: list[str] = ["*Research Summary*"]
        lines.append(
            "- Dataset: "
            f"{self._safe(schema_validation.get('valid_records', 0))} valid analysis records"
        )
        lines.append(
            "- Validation: "
            f"invalid={self._safe(top_highlights.get('invalid_record_count', 0))}, "
            f"warnings={self._safe(schema_validation.get('warning_count', 0))}"
        )
        lines.append(
            "- Strategy Lab Rows: "
            f"{self._safe(top_highlights.get('strategy_lab_dataset_rows', 0))}"
        )

        date_range = dataset_overview.get("date_range", {}) or {}
        start = date_range.get("start")
        end = date_range.get("end")
        if start or end:
            lines.append(
                f"- Date Range: {self._safe(start or 'unknown')} -> {self._safe(end or 'unknown')}"
            )

        lines.append("*Top Highlights*")
        lines.extend(self._highlight_lines(top_highlights))

        return "\n".join(lines)

    def _highlight_lines(self, top_highlights: dict[str, Any]) -> list[str]:
        by_horizon = top_highlights.get("by_horizon", {}) or {}
        lines: list[str] = []

        for horizon in ("15m", "1h", "4h"):
            horizon_data = by_horizon.get(horizon, {}) or {}

            top_symbol = self._safe(horizon_data.get("top_symbol", "n/a"))
            top_strategy = self._safe(horizon_data.get("top_strategy", "n/a"))
            best_alignment = self._safe(
                horizon_data.get("best_alignment_state", "n/a")
            )
            best_ai_execution = self._safe(
                horizon_data.get("best_ai_execution_state", "n/a")
            )

            if (
                top_symbol == "n/a"
                and top_strategy == "n/a"
                and best_alignment == "n/a"
                and best_ai_execution == "n/a"
            ):
                lines.append(f"- {self._safe(horizon)}: no ranking highlights available")
                continue

            candidate_parts = [
                f"symbol={top_symbol}",
                f"strategy={top_strategy}",
                f"align={best_alignment}",
                f"ai={best_ai_execution}",
            ]
            lines.append(f"- {self._safe(horizon)}: " + ", ".join(candidate_parts))

        return lines

    def _fallback_markdown_message(self, markdown_text: str | None) -> str:
        lines = ["*Research Summary*"]

        if markdown_text:
            preview = markdown_text.strip().splitlines()
            preview_line = preview[0] if preview else "summary generated"
            lines.append(f"- {self._safe(preview_line)}")
        else:
            lines.append("- Summary generated.")

        return "\n".join(lines)

    @staticmethod
    def _default_report_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "logs" / "research_reports" / "latest"

    @staticmethod
    def _read_text_file(path: Path) -> str | None:
        if not path.exists():
            logger.warning("Research summary markdown not found: %s", path)
            return None
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            logger.warning("Research summary JSON not found: %s", path)
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.exception("Research summary JSON is invalid: %s", path)
            return None

    @staticmethod
    def _safe(value: Any) -> str:
        return escape_markdown(str(value))
