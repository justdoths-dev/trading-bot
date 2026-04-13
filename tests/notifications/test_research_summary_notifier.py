from __future__ import annotations

from src.notifications.research_summary_notifier import (
    _extract_generated_at,
    build_research_summary_message,
)
from src.telegram.markdown_utils import escape_markdown


def _latest_summary(generated_at: str) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "dataset_overview": {
            "total_records": 864,
            "label_coverage_any_horizon_pct": 1.27,
            "selected_strategies_distribution": {"swing": 12, "trend": 5},
        },
        "top_highlights": {
            "strategy_lab_dataset_rows": 17,
            "by_horizon": {
                "15m": {"top_strategy": "swing"},
                "1h": {"top_strategy": "trend"},
                "4h": {"top_strategy": "swing"},
            },
        },
    }


def test_build_research_summary_message_uses_freshness_line_for_mixed_artifact_times() -> None:
    message = build_research_summary_message(
        latest_summary=_latest_summary("2026-04-09T00:25:02.605881+00:00"),
        edge_scores_summary={
            "generated_at": "2026-04-12T10:06:44.042847+00:00",
            "score_summary": {},
        },
        score_drift_summary={
            "generated_at": "2026-04-12T10:06:44.043295+00:00",
            "drift_summary": {},
        },
        edge_score_history_meta={
            "records": 2797,
            "last_generated_at": "2026-04-12T10:06:44.043295+00:00",
        },
        cumulative_log_meta={"records": 7480, "files_count": 2, "malformed_rows": 0},
        dataset_growth={
            "current_total_records": 7480,
            "previous_total_records": 7480,
            "recent_window_records": 864,
            "delta": 0,
        },
    )

    assert "Freshness:" in message
    assert "Generated:" not in message
    assert "latest summary:" in message
    assert "edge scores:" in message
    assert "drift:" in message
    assert escape_markdown("2026-04-09T00:25:02+00:00") in message
    assert escape_markdown("2026-04-12T10:06:44+00:00") in message


def test_build_research_summary_message_keeps_generated_line_when_sources_match() -> None:
    message = build_research_summary_message(
        latest_summary=_latest_summary("2026-04-12T10:06:44+00:00"),
        edge_scores_summary={
            "generated_at": "2026-04-12T10:06:44.042847+00:00",
            "score_summary": {},
        },
        score_drift_summary={
            "generated_at": "2026-04-12T10:06:44.043295+00:00",
            "drift_summary": {},
        },
        edge_score_history_meta=None,
        cumulative_log_meta={"records": 7480, "files_count": 2, "malformed_rows": 0},
        dataset_growth={
            "current_total_records": 7480,
            "previous_total_records": 7470,
            "recent_window_records": 864,
            "delta": 10,
        },
    )

    assert "Generated:" in message
    assert "Freshness:" not in message
    assert escape_markdown("2026-04-12T10:06:44+00:00") in message


def test_extract_generated_at_prefers_top_level_generated_at_over_dataset_range_end() -> None:
    payload = {
        "generated_at": "2026-04-12T10:06:44+00:00",
        "dataset_overview": {
            "date_range": {
                "end": "2026-04-09T00:25:02+00:00",
            }
        },
    }

    assert _extract_generated_at(payload) == "2026-04-12T10:06:44+00:00"


def test_extract_generated_at_falls_back_to_dataset_range_end_for_legacy_payload() -> None:
    payload = {
        "dataset_overview": {
            "date_range": {
                "end": "2026-04-09T00:25:02+00:00",
            }
        }
    }

    assert _extract_generated_at(payload) == "2026-04-09T00:25:02+00:00"


def test_build_research_summary_message_uses_clear_recent_window_labels() -> None:
    message = build_research_summary_message(
        latest_summary=_latest_summary("2026-04-12T10:06:44+00:00"),
        edge_scores_summary={
            "generated_at": "2026-04-12T10:06:44+00:00",
            "score_summary": {},
        },
        score_drift_summary={
            "generated_at": "2026-04-12T10:06:44+00:00",
            "drift_summary": {},
        },
        edge_score_history_meta=None,
        cumulative_log_meta={"records": 7480, "files_count": 2, "malformed_rows": 0},
        dataset_growth={
            "current_total_records": 7480,
            "previous_total_records": 7480,
            "recent_window_records": 864,
            "delta": 0,
        },
    )

    assert "recent coverage:" not in message
    assert "recent window share:" in message
    assert "recent label coverage:" in message
    assert escape_markdown("11.55%") in message
    assert escape_markdown("1.27%") in message
