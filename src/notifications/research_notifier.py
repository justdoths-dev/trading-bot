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
        strategy_lab = payload.get("strategy_lab", {}) or {}

        lines: list[str] = ["*Research Summary*"]

        total_records = dataset_overview.get("total_records", 0)
        valid_records = schema_validation.get("valid_records", total_records)
        invalid_records = schema_validation.get("invalid_records", 0)
        warning_count = schema_validation.get("warning_count", 0)

        lines.append(
            f"- Dataset: {self._safe(total_records)} records"
        )
        lines.append(
            "- Schema: "
            f"valid={self._safe(valid_records)} / "
            f"invalid={self._safe(invalid_records)} / "
            f"warnings={self._safe(warning_count)}"
        )

        symbols_distribution = dataset_overview.get("symbols_distribution", {}) or {}
        if symbols_distribution:
            lines.append(
                f"- Symbols: {self._format_distribution(symbols_distribution, limit=3)}"
            )

        strategy_distribution = dataset_overview.get("selected_strategies_distribution", {}) or {}
        if strategy_distribution:
            lines.append(
                f"- Strategies: {self._format_distribution(strategy_distribution, limit=3)}"
            )

        bias_distribution = dataset_overview.get("bias_distribution", {}) or {}
        if bias_distribution:
            lines.append(
                f"- Bias: {self._format_distribution(bias_distribution, limit=3)}"
            )

        alignment_distribution = dataset_overview.get("alignment_distribution", {}) or {}
        if alignment_distribution:
            lines.append(
                f"- Alignment: {self._format_distribution(alignment_distribution, limit=3)}"
            )

        ai_execution_distribution = dataset_overview.get("ai_execution_distribution", {}) or {}
        if ai_execution_distribution:
            lines.append(
                f"- AI Exec: {self._format_distribution(ai_execution_distribution, limit=3)}"
            )

        horizon_summary = payload.get("horizon_summary", {}) or {}
        lines.append(
            "- Labels: "
            f"15m={self._labeled_count(horizon_summary, '15m')} / "
            f"1h={self._labeled_count(horizon_summary, '1h')} / "
            f"4h={self._labeled_count(horizon_summary, '4h')}"
        )

        dataset_rows = strategy_lab.get("dataset_rows")
        if dataset_rows is not None:
            lines.append(f"- Lab Rows: {self._safe(dataset_rows)}")

        top_strategy_summary = self._top_strategy_summary(strategy_lab)
        if top_strategy_summary:
            lines.append(f"- Top Strategy: {top_strategy_summary}")

        edge_summary = self._edge_summary(strategy_lab)
        if edge_summary:
            lines.append(f"- Edge Findings: {edge_summary}")

        return "\n".join(lines)

    def _fallback_markdown_message(self, markdown_text: str | None) -> str:
        lines = ["*Research Summary*"]
        if markdown_text:
            preview = markdown_text.strip().splitlines()
            preview_line = preview[0] if preview else "summary generated"
            lines.append(f"- {self._safe(preview_line)}")
        else:
            lines.append("- Summary generated.")
        return "\n".join(lines)

    def _format_distribution(self, data: dict[str, Any], limit: int = 3) -> str:
        items = list(data.items())[:limit]
        formatted = []
        for key, value in items:
            formatted.append(f"{self._safe(key)} {self._safe(value)}")
        return ", ".join(formatted) if formatted else "n/a"

    def _labeled_count(self, horizon_summary: dict[str, Any], horizon: str) -> str:
        horizon_data = horizon_summary.get(horizon, {}) or {}
        return self._safe(horizon_data.get("labeled_records", 0))

    def _top_strategy_summary(self, strategy_lab: dict[str, Any]) -> str:
        ranking = strategy_lab.get("ranking", {}) or {}
        parts: list[str] = []

        for horizon in ("15m", "1h", "4h"):
            horizon_rank = ranking.get(horizon, {}) or {}
            by_strategy = horizon_rank.get("by_strategy", {}) or {}
            top_group = self._extract_top_ranked_group(by_strategy)
            if top_group != "n/a":
                parts.append(f"{horizon}:{self._safe(top_group)}")

        return ", ".join(parts) if parts else ""

    def _edge_summary(self, strategy_lab: dict[str, Any]) -> str:
        edge = strategy_lab.get("edge", {}) or {}
        parts: list[str] = []

        for horizon in ("15m", "1h", "4h"):
            horizon_edge = edge.get(horizon, {}) or {}
            total_findings = (
                self._count_edge_findings(horizon_edge.get("by_symbol"))
                + self._count_edge_findings(horizon_edge.get("by_strategy"))
                + self._count_edge_findings(horizon_edge.get("by_alignment_state"))
                + self._count_edge_findings(horizon_edge.get("by_ai_execution_state"))
            )
            parts.append(f"{horizon}:{total_findings}")

        return ", ".join(parts) if parts else ""

    def _extract_top_ranked_group(self, report: Any) -> str:
        if not isinstance(report, dict):
            return "n/a"

        items = report.get("rankings") or report.get("results") or []
        if not isinstance(items, list) or not items:
            return "n/a"

        first = items[0]
        if not isinstance(first, dict):
            return "n/a"

        return str(first.get("group", "n/a"))

    def _count_edge_findings(self, report: Any) -> int:
        if not isinstance(report, dict):
            return 0
        findings = report.get("edge_findings", [])
        if not isinstance(findings, list):
            return 0
        return len(findings)

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