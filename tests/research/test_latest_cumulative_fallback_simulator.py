from __future__ import annotations

import json
from pathlib import Path

from src.research.latest_cumulative_fallback_simulator import (
    render_latest_cumulative_fallback_simulation_markdown,
    run_latest_cumulative_fallback_simulator,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _comparison_summary(
    *,
    latest_15m: str = "insufficient_data",
    cumulative_15m: str = "insufficient_data",
    latest_1h: str = "insufficient_data",
    cumulative_1h: str = "insufficient_data",
    latest_4h: str = "insufficient_data",
    cumulative_4h: str = "insufficient_data",
) -> dict:
    return {
        "edge_candidates_comparison": {
            "15m": {
                "latest_candidate_strength": latest_15m,
                "cumulative_candidate_strength": cumulative_15m,
                "latest_top_strategy_group": "swing",
                "cumulative_top_strategy_group": "swing",
                "latest_top_symbol_group": "BTCUSDT",
                "cumulative_top_symbol_group": "BTCUSDT",
                "latest_top_alignment_state_group": "aligned",
                "cumulative_top_alignment_state_group": "aligned",
            },
            "1h": {
                "latest_candidate_strength": latest_1h,
                "cumulative_candidate_strength": cumulative_1h,
                "latest_top_strategy_group": "trend",
                "cumulative_top_strategy_group": "trend",
                "latest_top_symbol_group": "ETHUSDT",
                "cumulative_top_symbol_group": "ETHUSDT",
                "latest_top_alignment_state_group": "mixed",
                "cumulative_top_alignment_state_group": "mixed",
            },
            "4h": {
                "latest_candidate_strength": latest_4h,
                "cumulative_candidate_strength": cumulative_4h,
                "latest_top_strategy_group": "macro",
                "cumulative_top_strategy_group": "macro",
                "latest_top_symbol_group": "SOLUSDT",
                "cumulative_top_symbol_group": "SOLUSDT",
                "latest_top_alignment_state_group": "aligned",
                "cumulative_top_alignment_state_group": "aligned",
            },
        }
    }


def _candidate(group: str, strength: str) -> dict:
    return {
        "group": group,
        "candidate_strength": strength,
    }


def _summary(
    latest_15m_strength: str,
    latest_1h_strength: str,
    latest_4h_strength: str,
    median_15m: float | None,
    median_1h: float | None,
    median_4h: float | None,
) -> dict:
    def _rank(group: str, median: float | None) -> dict:
        rows = [] if median is None else [{"group": group, "metrics": {"median_future_return_pct": median}}]
        return {"ranked_groups": rows}

    return {
        "edge_candidates_preview": {
            "by_horizon": {
                "15m": {
                    "top_strategy": _candidate("swing", latest_15m_strength),
                    "top_symbol": _candidate("BTCUSDT", latest_15m_strength),
                    "top_alignment_state": _candidate("aligned", latest_15m_strength),
                },
                "1h": {
                    "top_strategy": _candidate("trend", latest_1h_strength),
                    "top_symbol": _candidate("ETHUSDT", latest_1h_strength),
                    "top_alignment_state": _candidate("mixed", latest_1h_strength),
                },
                "4h": {
                    "top_strategy": _candidate("macro", latest_4h_strength),
                    "top_symbol": _candidate("SOLUSDT", latest_4h_strength),
                    "top_alignment_state": _candidate("aligned", latest_4h_strength),
                },
            }
        },
        "strategy_lab": {
            "ranking": {
                "15m": {
                    "by_strategy": _rank("swing", median_15m),
                    "by_symbol": _rank("BTCUSDT", median_15m),
                    "by_alignment_state": _rank("aligned", median_15m),
                },
                "1h": {
                    "by_strategy": _rank("trend", median_1h),
                    "by_symbol": _rank("ETHUSDT", median_1h),
                    "by_alignment_state": _rank("mixed", median_1h),
                },
                "4h": {
                    "by_strategy": _rank("macro", median_4h),
                    "by_symbol": _rank("SOLUSDT", median_4h),
                    "by_alignment_state": _rank("aligned", median_4h),
                },
            }
        },
    }


def _score_drift(
    *,
    swing: str = "flat",
    btc: str = "flat",
    aligned: str = "flat",
    trend: str = "flat",
    eth: str = "flat",
    mixed: str = "flat",
    macro: str = "flat",
    sol: str = "flat",
) -> dict:
    return {
        "score_drift": [
            {"category": "strategy", "group": "swing", "drift_direction": swing},
            {"category": "symbol", "group": "BTCUSDT", "drift_direction": btc},
            {"category": "alignment_state", "group": "aligned", "drift_direction": aligned},
            {"category": "strategy", "group": "trend", "drift_direction": trend},
            {"category": "symbol", "group": "ETHUSDT", "drift_direction": eth},
            {"category": "alignment_state", "group": "mixed", "drift_direction": mixed},
            {"category": "strategy", "group": "macro", "drift_direction": macro},
            {"category": "symbol", "group": "SOLUSDT", "drift_direction": sol},
        ]
    }


def test_all_insufficient_data_results_in_no_recovery(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(comparison, _comparison_summary())
    _write_json(drift, _score_drift())
    _write_json(
        latest,
        _summary(
            "insufficient_data",
            "insufficient_data",
            "insufficient_data",
            None,
            None,
            None,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "insufficient_data",
            "insufficient_data",
            "insufficient_data",
            None,
            None,
            None,
        ),
    )

    summary = run_latest_cumulative_fallback_simulator(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]

    assert summary["recovery_analysis"]["fallback_candidate_possible_count"] == 0
    assert summary["simulated_selection_impact"]["estimated_candidate_count_change"] == 0
    assert summary["final_diagnosis"]["recommendation"] == "not_ready"


def test_fallback_candidate_possible_triggered(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_1h="insufficient_data",
            cumulative_1h="weak",
        ),
    )
    _write_json(drift, _score_drift(trend="flat", eth="increase", mixed="flat"))
    _write_json(
        latest,
        _summary(
            "weak",
            "insufficient_data",
            "insufficient_data",
            0.1,
            0.2,
            None,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "weak",
            "weak",
            "insufficient_data",
            0.1,
            0.2,
            None,
        ),
    )

    summary = run_latest_cumulative_fallback_simulator(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]

    assert summary["recovery_analysis"]["fallback_candidate_possible_count"] == 3
    assert summary["simulated_selection_impact"]["estimated_candidate_count_change"] == 3
    assert (
        summary["simulated_selection_impact"]["estimated_status_shift"][
            "NO_CANDIDATES_AVAILABLE_to_NO_ELIGIBLE_CANDIDATES"
        ]
        == 1
    )


def test_blocked_by_drift_decrease(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_1h="insufficient_data",
            cumulative_1h="weak",
        ),
    )
    _write_json(drift, _score_drift(trend="decrease", eth="decrease", mixed="decrease"))
    _write_json(
        latest,
        _summary(
            "weak",
            "insufficient_data",
            "insufficient_data",
            0.1,
            0.2,
            None,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "weak",
            "weak",
            "insufficient_data",
            0.1,
            0.2,
            None,
        ),
    )

    summary = run_latest_cumulative_fallback_simulator(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]

    assert summary["recovery_analysis"]["blocked_by_drift_decreasing_count"] == 3
    assert summary["recovery_analysis"]["fallback_candidate_possible_count"] == 0


def test_blocked_by_negative_median(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_1h="insufficient_data",
            cumulative_1h="weak",
        ),
    )
    _write_json(drift, _score_drift(trend="flat", eth="flat", mixed="flat"))
    _write_json(
        latest,
        _summary(
            "weak",
            "insufficient_data",
            "insufficient_data",
            0.1,
            -0.2,
            None,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "weak",
            "weak",
            "insufficient_data",
            0.1,
            0.2,
            None,
        ),
    )

    summary = run_latest_cumulative_fallback_simulator(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )["summary"]

    assert summary["recovery_analysis"]["blocked_by_non_positive_median_count"] == 3
    assert summary["recovery_analysis"]["fallback_candidate_possible_count"] == 0


def test_mixed_horizon_behavior(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    drift = tmp_path / "drift.json"
    latest = tmp_path / "latest.json"
    cumulative = tmp_path / "cumulative.json"

    _write_json(
        comparison,
        _comparison_summary(
            latest_15m="weak",
            cumulative_15m="weak",
            latest_1h="insufficient_data",
            cumulative_1h="weak",
            latest_4h="insufficient_data",
            cumulative_4h="weak",
        ),
    )
    _write_json(
        drift,
        _score_drift(
            trend="flat",
            eth="flat",
            mixed="flat",
            macro="decrease",
            sol="decrease",
            aligned="decrease",
        ),
    )
    _write_json(
        latest,
        _summary(
            "weak",
            "insufficient_data",
            "insufficient_data",
            0.1,
            0.2,
            0.2,
        ),
    )
    _write_json(
        cumulative,
        _summary(
            "weak",
            "weak",
            "weak",
            0.1,
            0.2,
            0.2,
        ),
    )

    result = run_latest_cumulative_fallback_simulator(
        comparison, drift, latest, cumulative, tmp_path / "out"
    )
    summary = result["summary"]
    markdown = render_latest_cumulative_fallback_simulation_markdown(summary)

    assert summary["horizon_breakdown"]["1h"]["possible_count"] == 3
    assert summary["horizon_breakdown"]["4h"]["blocked_count"] == 3
    assert summary["final_diagnosis"]["recommendation"] == "promising_but_blocked"
    assert "Final Diagnosis" in markdown