from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.telegram.markdown_utils import escape_markdown
from src.telegram.telegram_sender import TelegramSender

logger = logging.getLogger(__name__)
HORIZON_ORDER = ("15m", "1h", "4h")


class ResearchSummaryNotifier:
    """Send periodic research-only summary messages to the research channel."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        sender: TelegramSender | None = None,
    ) -> None:
        self._bot_token = (bot_token or os.getenv("TELEGRAM_OPS_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.getenv("TELEGRAM_RESEARCH_CHAT_ID", "")).strip()
        self._sender = sender

    def send_message(self, message: str) -> dict[str, Any]:
        sender = self._get_sender()
        if sender is None:
            logger.error(
                "Research summary notifier configuration missing. "
                "Expected TELEGRAM_OPS_BOT_TOKEN and TELEGRAM_RESEARCH_CHAT_ID "
                "for single-bot research-channel routing."
            )
            return {
                "sent": False,
                "reason": (
                    "Research summary notifier configuration missing. "
                    "Expected TELEGRAM_OPS_BOT_TOKEN and TELEGRAM_RESEARCH_CHAT_ID."
                ),
            }

        try:
            response = sender.send_message(message)
            return {
                "sent": True,
                "reason": "Research summary sent successfully.",
                "response": response,
            }
        except Exception as exc:
            logger.exception("Failed to send research summary.")
            return {
                "sent": False,
                "reason": f"Research summary send failed: {exc}",
            }

    def _get_sender(self) -> TelegramSender | None:
        if self._sender is not None:
            return self._sender

        if not self._bot_token or not self._chat_id:
            return None

        self._sender = TelegramSender(self._bot_token, self._chat_id)
        return self._sender


def run_research_summary_notifier(
    *,
    summary_json_path: Path | None = None,
    state_file: Path | None = None,
    trade_analysis_base_path: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    resolved_summary_json_path = summary_json_path or _default_summary_json_path()
    resolved_state_file = state_file or _default_state_file()
    resolved_trade_analysis_base_path = (
        trade_analysis_base_path or _default_trade_analysis_base_path()
    )
    resolved_edge_scores_summary_path = _default_edge_scores_summary_path()
    resolved_edge_score_history_path = _default_edge_score_history_path()
    resolved_score_drift_summary_path = _default_score_drift_summary_path()

    latest_summary = load_json(resolved_summary_json_path)
    edge_scores_summary = load_json(resolved_edge_scores_summary_path)
    score_drift_summary = load_json(resolved_score_drift_summary_path)
    edge_score_history_meta = load_edge_score_history_meta(resolved_edge_score_history_path)
    cumulative_log_meta = load_cumulative_trade_analysis_meta(resolved_trade_analysis_base_path)
    state = load_state(resolved_state_file)

    dataset_growth = _build_dataset_growth(
        latest_summary=latest_summary,
        cumulative_log_meta=cumulative_log_meta,
        state=state,
    )
    message = build_research_summary_message(
        latest_summary=latest_summary,
        edge_scores_summary=edge_scores_summary,
        score_drift_summary=score_drift_summary,
        edge_score_history_meta=edge_score_history_meta,
        cumulative_log_meta=cumulative_log_meta,
        dataset_growth=dataset_growth,
    )
    content_hash = compute_content_hash(message)
    source_stamp = resolve_source_stamp(
        summary_json_path=resolved_summary_json_path,
        latest_summary=latest_summary,
        edge_scores_summary_path=resolved_edge_scores_summary_path,
        edge_scores_summary=edge_scores_summary,
        score_drift_summary_path=resolved_score_drift_summary_path,
        score_drift_summary=score_drift_summary,
        cumulative_log_meta=cumulative_log_meta,
    )

    previous_source_stamp = str(
        state.get("source_stamp", state.get("source_timestamp", ""))
    )

    suppressed = False
    suppression_reason = ""
    if not force:
        if content_hash and content_hash == str(state.get("content_hash", "")):
            suppressed = True
            suppression_reason = "Summary content hash unchanged."
        elif source_stamp and source_stamp == previous_source_stamp:
            suppressed = True
            suppression_reason = "Source change stamp unchanged."

    result = {
        "sent": False,
        "suppressed": suppressed,
        "reason": suppression_reason if suppressed else "",
        "summary_json_path": str(resolved_summary_json_path),
        "trade_analysis_base_path": str(resolved_trade_analysis_base_path),
        "state_file": str(resolved_state_file),
        "content_hash": content_hash,
        "source_stamp": source_stamp,
        "message": message,
        "cumulative_total_records": cumulative_log_meta.get("records"),
        "recent_window_records": dataset_growth.get("recent_window_records"),
    }

    if suppressed:
        return result

    if dry_run:
        result["reason"] = "Dry run only. Message was not sent."
        return result

    notifier = ResearchSummaryNotifier()
    delivery = notifier.send_message(message)

    if delivery.get("sent"):
        save_state(
            resolved_state_file,
            {
                "content_hash": content_hash,
                "source_stamp": source_stamp,
                "cumulative_total_records": dataset_growth.get("current_total_records"),
                "last_sent_at": _utc_now_iso(),
            },
        )

    return {
        **result,
        **delivery,
        "suppressed": False,
    }


def build_research_summary_message(
    *,
    latest_summary: dict[str, Any] | None,
    edge_scores_summary: dict[str, Any] | None,
    score_drift_summary: dict[str, Any] | None,
    edge_score_history_meta: dict[str, Any] | None,
    cumulative_log_meta: dict[str, Any] | None,
    dataset_growth: dict[str, Any],
) -> str:
    generated_at = resolve_display_timestamp(
        latest_summary=latest_summary,
        edge_scores_summary=edge_scores_summary,
        score_drift_summary=score_drift_summary,
    )

    lines = ["*Research Summary*"]
    lines.append(f"Generated: {escape_markdown(generated_at)}")
    lines.append(_dataset_growth_line(dataset_growth, latest_summary, cumulative_log_meta))
    lines.append(_strategy_snapshot_line(latest_summary))
    lines.append(_edge_stability_snapshot_line(edge_scores_summary))
    lines.append(_edge_history_snapshot_line(edge_score_history_meta))
    lines.append(_drift_snapshot_line(score_drift_summary))
    lines.append(
        "Scope: cumulative records reflect all retained trade analysis logs; "
        "strategy, edge, and drift snapshots reflect the latest research outputs\\."
    )
    lines.append("Research-only summary\\. No trading recommendation\\.")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Failed to read JSON file: %s (%s)", path, exc)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON file: %s (%s)", path, exc)
        return None

    return payload if isinstance(payload, dict) else None


def load_state(path: Path) -> dict[str, Any]:
    state = load_json(path)
    return state if isinstance(state, dict) else {}


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_edge_score_history_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None

    valid_records = 0
    last_generated_at = ""

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed edge history JSONL row in %s", path)
                    continue

                if not isinstance(payload, dict):
                    continue

                valid_records += 1
                generated_at = str(payload.get("generated_at", "")).strip()
                if generated_at:
                    last_generated_at = generated_at
    except OSError as exc:
        logger.warning("Failed to read edge score history file: %s (%s)", path, exc)
        return None

    return {
        "records": valid_records,
        "last_generated_at": last_generated_at,
    }


def load_cumulative_trade_analysis_meta(base_path: Path) -> dict[str, Any]:
    paths = _resolve_trade_analysis_paths(base_path)
    valid_records = 0
    malformed_rows = 0
    file_parts: list[str] = []

    for path in paths:
        file_parts.append(_file_stamp_part(path))
        try:
            with _open_text_file(path) as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        malformed_rows += 1
                        continue
                    if isinstance(payload, dict):
                        valid_records += 1
        except OSError as exc:
            logger.warning("Failed to read trade analysis file: %s (%s)", path, exc)

    raw_stamp = "|".join(part for part in file_parts if part)
    files_stamp = (
        hashlib.sha256(raw_stamp.encode("utf-8")).hexdigest() if raw_stamp else ""
    )

    return {
        "records": valid_records,
        "malformed_rows": malformed_rows,
        "files_count": len(paths),
        "files_stamp": files_stamp,
        "paths": [str(path) for path in paths],
    }


def compute_content_hash(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def resolve_source_stamp(
    *,
    summary_json_path: Path,
    latest_summary: dict[str, Any] | None,
    edge_scores_summary_path: Path,
    edge_scores_summary: dict[str, Any] | None,
    score_drift_summary_path: Path,
    score_drift_summary: dict[str, Any] | None,
    cumulative_log_meta: dict[str, Any] | None,
) -> str:
    candidates = [
        _extract_generated_at(latest_summary),
        _extract_generated_at(edge_scores_summary),
        _extract_generated_at(score_drift_summary),
        _file_mtime_token(summary_json_path),
        _file_mtime_token(edge_scores_summary_path),
        _file_mtime_token(score_drift_summary_path),
        str(_safe_int((cumulative_log_meta or {}).get("records"))),
        str((cumulative_log_meta or {}).get("files_stamp", "")),
    ]

    normalized_candidates = [candidate for candidate in candidates if candidate]
    if not normalized_candidates:
        return ""

    raw_stamp = "|".join(normalized_candidates)
    return hashlib.sha256(raw_stamp.encode("utf-8")).hexdigest()


def resolve_display_timestamp(
    *,
    latest_summary: dict[str, Any] | None,
    edge_scores_summary: dict[str, Any] | None,
    score_drift_summary: dict[str, Any] | None,
) -> str:
    for candidate in (
        _extract_generated_at(latest_summary),
        _extract_generated_at(edge_scores_summary),
        _extract_generated_at(score_drift_summary),
    ):
        if candidate:
            return candidate
    return "n/a"


def _dataset_growth_line(
    dataset_growth: dict[str, Any],
    latest_summary: dict[str, Any] | None,
    cumulative_log_meta: dict[str, Any] | None,
) -> str:
    current_total = dataset_growth.get("current_total_records")
    delta = dataset_growth.get("delta")
    recent_window_records = dataset_growth.get("recent_window_records")
    recent_coverage = _safe_dict((latest_summary or {}).get("dataset_overview")).get(
        "label_coverage_any_horizon_pct"
    )
    files_count = (cumulative_log_meta or {}).get("files_count")

    total_text = "n/a" if current_total is None else str(current_total)
    recent_window_text = "n/a" if recent_window_records is None else str(recent_window_records)
    files_text = "n/a" if files_count is None else str(files_count)
    malformed_rows = _safe_int((cumulative_log_meta or {}).get("malformed_rows"))

    if delta is None:
        growth_text = "baseline"
    elif delta >= 0:
        growth_text = f"+{delta}"
    else:
        growth_text = str(delta)

    coverage_text = "n/a" if recent_coverage is None else f"{float(recent_coverage):.2f}%"

    return (
        "- Dataset: "
        f"total records: {escape_markdown(total_text)}; "
        f"growth: {escape_markdown(growth_text)}; "
        f"recent window: {escape_markdown(recent_window_text)}; "
        f"recent coverage: {escape_markdown(coverage_text)}; "
        f"log files: {escape_markdown(files_text)}; "
        f"malformed rows: {escape_markdown(str(malformed_rows))}"
    )


def _strategy_snapshot_line(latest_summary: dict[str, Any] | None) -> str:
    summary = latest_summary or {}
    dataset_overview = _safe_dict(summary.get("dataset_overview"))
    top_highlights = _safe_dict(summary.get("top_highlights"))
    by_horizon = _safe_dict(top_highlights.get("by_horizon"))

    dominant_strategy = _top_distribution_key(
        _safe_dict(dataset_overview.get("selected_strategies_distribution"))
    )
    strategy_lab_rows = _safe_dict(top_highlights).get("strategy_lab_dataset_rows")

    horizon_parts: list[str] = []
    for horizon in HORIZON_ORDER:
        horizon_entry = _safe_dict(by_horizon.get(horizon))
        top_strategy = str(horizon_entry.get("top_strategy", "n/a"))
        if top_strategy != "n/a":
            horizon_parts.append(f"{horizon}: {top_strategy}")

    horizon_text = ", ".join(horizon_parts) if horizon_parts else "no horizon snapshot"
    rows_text = "n/a" if strategy_lab_rows is None else str(strategy_lab_rows)

    return (
        "- Strategy: "
        f"dominant: {escape_markdown(dominant_strategy)}; "
        f"lab rows: {escape_markdown(rows_text)}; "
        f"horizon snapshot: {escape_markdown(horizon_text)}"
    )


def _edge_stability_snapshot_line(edge_scores_summary: dict[str, Any] | None) -> str:
    score_summary = _safe_dict((edge_scores_summary or {}).get("score_summary"))
    parts = [
        _edge_summary_part("strategy", _safe_dict(score_summary.get("top_strategy"))),
        _edge_summary_part("symbol", _safe_dict(score_summary.get("top_symbol"))),
        _edge_summary_part(
            "alignment",
            _safe_dict(score_summary.get("top_alignment_state")),
        ),
    ]
    return "- Edge Stability: " + escape_markdown("; ".join(parts))


def _edge_history_snapshot_line(edge_score_history_meta: dict[str, Any] | None) -> str:
    if not isinstance(edge_score_history_meta, dict):
        return "- Edge History: not available"

    records = edge_score_history_meta.get("records")
    last_generated_at = str(edge_score_history_meta.get("last_generated_at", "n/a"))
    records_text = "n/a" if records is None else str(records)
    return (
        "- Edge History: "
        f"records: {escape_markdown(records_text)}; "
        f"latest snapshot: {escape_markdown(last_generated_at or 'n/a')}"
    )


def _drift_snapshot_line(score_drift_summary: dict[str, Any] | None) -> str:
    drift_summary = _safe_dict((score_drift_summary or {}).get("drift_summary"))
    groups_analyzed = (score_drift_summary or {}).get("groups_analyzed")
    increases = _safe_int(drift_summary.get("increase"))
    decreases = _safe_int(drift_summary.get("decrease"))
    flat = _safe_int(drift_summary.get("flat"))
    groups_text = "n/a" if groups_analyzed is None else str(groups_analyzed)

    return (
        "- Drift: "
        f"groups: {escape_markdown(groups_text)}; "
        f"increase: {escape_markdown(str(increases))}; "
        f"decrease: {escape_markdown(str(decreases))}; "
        f"flat: {escape_markdown(str(flat))}"
    )


def _edge_summary_part(label: str, item: dict[str, Any]) -> str:
    group = str(item.get("group", "n/a"))
    score = item.get("score")
    source = str(item.get("source_preference", "n/a"))
    score_text = "n/a" if score is None else _format_number(float(score))
    return f"{label}: {group} (score: {score_text}, source: {source})"


def _build_dataset_growth(
    *,
    latest_summary: dict[str, Any] | None,
    cumulative_log_meta: dict[str, Any] | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    dataset_overview = _safe_dict((latest_summary or {}).get("dataset_overview"))
    recent_window_records = dataset_overview.get("total_records")
    current_total = (cumulative_log_meta or {}).get("records")
    previous_total = state.get("cumulative_total_records", state.get("dataset_total_records"))

    try:
        current_int = int(current_total) if current_total is not None else None
    except (TypeError, ValueError):
        current_int = None

    try:
        previous_int = int(previous_total) if previous_total is not None else None
    except (TypeError, ValueError):
        previous_int = None

    try:
        recent_window_int = int(recent_window_records) if recent_window_records is not None else None
    except (TypeError, ValueError):
        recent_window_int = None

    delta = None
    if current_int is not None and previous_int is not None:
        delta = current_int - previous_int

    return {
        "current_total_records": current_int,
        "previous_total_records": previous_int,
        "recent_window_records": recent_window_int,
        "delta": delta,
    }


def _extract_generated_at(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""

    generated_at = str(payload.get("generated_at", "")).strip()
    if generated_at:
        return generated_at

    date_range = _safe_dict(_safe_dict(payload.get("dataset_overview")).get("date_range"))
    return str(date_range.get("end", "")).strip()


def _file_mtime_token(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return str(int(path.stat().st_mtime))


def _file_stamp_part(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return path.name
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


def _resolve_trade_analysis_paths(base_path: Path) -> list[Path]:
    directory = base_path.parent
    base_name = base_path.name

    if not directory.exists():
        return []

    matched: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if path.name == base_name or path.name.startswith(f"{base_name}."):
            matched.append(path)

    return sorted(matched, key=lambda item: item.name)


def _open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _top_distribution_key(distribution: dict[str, Any]) -> str:
    if not distribution:
        return "n/a"

    best_key = "n/a"
    best_value = float("-inf")
    for key, value in distribution.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if numeric_value > best_value:
            best_key = str(key)
            best_value = numeric_value
    return best_key


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _default_summary_json_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "latest"
        / "summary.json"
    )


def _default_trade_analysis_base_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _default_edge_scores_summary_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores"
        / "summary.json"
    )


def _default_edge_score_history_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores_history.jsonl"
    )


def _default_score_drift_summary_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "score_drift"
        / "summary.json"
    )


def _default_state_file() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "state"
        / "research_summary_notifier_state.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a periodic research-only Telegram summary from existing research outputs"
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=_default_summary_json_path(),
        help="Path to the latest research summary.json",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=_default_state_file(),
        help="Path to the notifier state file used for suppression",
    )
    parser.add_argument(
        "--trade-analysis-base",
        type=Path,
        default=_default_trade_analysis_base_path(),
        help="Base path for trade analysis JSONL logs; rotated files with the same prefix are included automatically",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the summary message without sending it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send even when the summary content hash or source change stamp is unchanged",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_research_summary_notifier(
        summary_json_path=args.summary_json,
        state_file=args.state_file,
        trade_analysis_base_path=args.trade_analysis_base,
        dry_run=args.dry_run,
        force=args.force,
    )

    if args.dry_run and result.get("message"):
        print(result["message"])
        print()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
