from __future__ import annotations

from typing import Any

from src.notifications.research_notifier import ResearchNotifier


def _base_payload() -> dict[str, Any]:
    return {
        "schema_validation": {
            "valid_records": 120,
            "warning_count": 2,
        },
        "top_highlights": {
            "invalid_record_count": 1,
            "strategy_lab_dataset_rows": 120,
            "by_horizon": {
                "15m": {
                    "top_symbol": "BTCUSDT",
                    "top_strategy": "swing",
                    "best_alignment_state": "aligned",
                    "best_ai_execution_state": "allowed",
                },
                "1h": {
                    "top_symbol": "BTCUSDT",
                    "top_strategy": "swing",
                    "best_alignment_state": "aligned",
                    "best_ai_execution_state": "allowed",
                },
                "4h": {
                    "top_symbol": "ETHUSDT",
                    "top_strategy": "trend",
                    "best_alignment_state": "mixed",
                    "best_ai_execution_state": "blocked",
                },
            },
        },
        "edge_candidates_preview": {
            "by_horizon": {
                "15m": {
                    "top_strategy": {
                        "group": "swing",
                        "candidate_strength": "moderate",
                        "sample_count": 60,
                        "quality_gate": "passed",
                        "median_future_return_pct": 0.4,
                        "positive_rate_pct": 57.0,
                    },
                    "top_symbol": {
                        "group": "BTCUSDT",
                        "candidate_strength": "weak",
                        "sample_count": 45,
                        "quality_gate": "borderline",
                        "median_future_return_pct": 0.2,
                        "positive_rate_pct": 52.0,
                    },
                    "top_alignment_state": {
                        "group": "aligned",
                        "candidate_strength": "insufficient_data",
                    },
                },
                "1h": {
                    "top_strategy": {
                        "group": "swing",
                        "candidate_strength": "moderate",
                        "sample_count": 62,
                        "quality_gate": "passed",
                        "median_future_return_pct": 0.42,
                        "positive_rate_pct": 58.0,
                    },
                    "top_symbol": {
                        "group": "ETHUSDT",
                        "candidate_strength": "insufficient_data",
                    },
                    "top_alignment_state": {
                        "group": "aligned",
                        "candidate_strength": "insufficient_data",
                    },
                },
                "4h": {
                    "top_strategy": {
                        "group": "trend",
                        "candidate_strength": "weak",
                        "sample_count": 30,
                        "quality_gate": "borderline",
                        "median_future_return_pct": 0.1,
                        "positive_rate_pct": 50.0,
                    },
                    "top_symbol": {
                        "group": "ETHUSDT",
                        "candidate_strength": "insufficient_data",
                    },
                    "top_alignment_state": {
                        "group": "mixed",
                        "candidate_strength": "insufficient_data",
                    },
                },
            }
        },
    }


def test_notifier_reads_edge_stability_preview_safely() -> None:
    notifier = ResearchNotifier(bot_token="token", chat_id="chat")
    payload = _base_payload()
    payload["edge_stability_preview"] = {
        "strategy": {
            "group": "swing",
            "visible_horizons": ["15m", "1h"],
            "stability_label": "multi_horizon_confirmed",
            "stability_score": 2,
        },
        "symbol": {
            "group": "BTCUSDT",
            "visible_horizons": ["15m"],
            "stability_label": "single_horizon_only",
            "stability_score": 1,
        },
        "alignment_state": {
            "group": None,
            "visible_horizons": [],
            "stability_label": "insufficient_data",
            "stability_score": 0,
        },
    }

    message = notifier._format_message(payload, markdown_text=None)

    assert "Stability Notes" in message
    assert "repeats across" in message
    assert "single-horizon only" in message
    assert "buy" not in message.lower()
    assert "sell" not in message.lower()
    assert "recommend" not in message.lower()


def test_notifier_does_not_crash_when_stability_preview_is_missing() -> None:
    notifier = ResearchNotifier(bot_token="token", chat_id="chat")

    message = notifier._format_message(_base_payload(), markdown_text=None)

    assert "Research Summary" in message
    assert "Edge Preview" in message
    assert "Stability Notes" not in message


def test_notifier_keeps_quiet_when_stability_preview_is_insufficient() -> None:
    notifier = ResearchNotifier(bot_token="token", chat_id="chat")
    payload = _base_payload()
    payload["edge_stability_preview"] = {
        "strategy": {
            "group": None,
            "visible_horizons": [],
            "stability_label": "insufficient_data",
            "stability_score": 0,
        },
        "symbol": {
            "group": None,
            "visible_horizons": [],
            "stability_label": "insufficient_data",
            "stability_score": 0,
        },
        "alignment_state": {
            "group": None,
            "visible_horizons": [],
            "stability_label": "insufficient_data",
            "stability_score": 0,
        },
    }

    message = notifier._format_message(payload, markdown_text=None)

    assert "Stability Notes" not in message
    assert "insufficient_data" not in message
