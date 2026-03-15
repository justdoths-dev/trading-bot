from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.research.edge_selection_schema_validator import (
    validate_shadow_output,
    validate_upstream_reports,
)


def test_valid_shadow_output_payload_passes() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "selected",
        "selection_confidence": 0.72,
        "selected_symbol": "BTCUSDT",
        "selected_strategy": "swing",
        "selected_horizon": "4h",
        "selection_score": 4.1,
        "reason_codes": ["SHADOW_SELECTION_ONLY"],
        "ranking": [
            {
                "rank": 1,
                "symbol": "BTCUSDT",
                "strategy": "swing",
                "horizon": "4h",
                "candidate_status": "eligible",
                "selection_score": 4.1,
                "selection_confidence": 0.72,
                "reason_codes": ["SHADOW_SELECTION_ONLY"],
                "selected_candidate_strength": "strong",
                "selected_stability_label": "multi_horizon_confirmed",
                "selected_visible_horizons": ["1h", "4h"],
                "source_preference": "latest",
                "edge_stability_score": 4.0,
                "stability_gate_pass": True,
                "latest_sample_size": 24,
                "cumulative_sample_size": 110,
                "symbol_cumulative_support": 180,
                "strategy_cumulative_support": 141,
                "consecutive_visible_cycles": 3,
                "consecutive_stable_cycles": 2,
                "drift_direction": "flat",
                "score_delta": 0.0,
                "drift_blocked": False,
            }
        ],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is True
    assert result.errors == []


def test_abstain_payload_with_selected_symbol_fails() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "abstain",
        "selection_confidence": None,
        "selected_symbol": "BTCUSDT",
        "selected_strategy": None,
        "selected_horizon": None,
        "selection_score": None,
        "reason_codes": ["ABSTAIN_POLICY_TRIGGERED"],
        "ranking": [],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any("selected_symbol must be None" in error for error in result.errors)


def test_blocked_payload_with_non_null_selection_score_fails() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "blocked",
        "selection_confidence": None,
        "selected_symbol": None,
        "selected_strategy": None,
        "selected_horizon": None,
        "selection_score": 1.0,
        "reason_codes": ["ABSTAIN_POLICY_TRIGGERED"],
        "ranking": [],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any("selection_score must be None" in error for error in result.errors)


def test_invalid_selection_confidence_fails() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "selected",
        "selection_confidence": 1.5,
        "selected_symbol": "BTCUSDT",
        "selected_strategy": "swing",
        "selected_horizon": "1h",
        "selection_score": 3.2,
        "reason_codes": ["SHADOW_SELECTION_ONLY"],
        "ranking": [],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any(
        "selection_confidence must be between 0 and 1" in error
        for error in result.errors
    )


def test_duplicate_rank_values_fail() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "blocked",
        "selection_confidence": None,
        "selected_symbol": None,
        "selected_strategy": None,
        "selected_horizon": None,
        "selection_score": None,
        "reason_codes": ["ABSTAIN_POLICY_TRIGGERED"],
        "ranking": [
            {
                "rank": 1,
                "candidate_status": "eligible",
                "reason_codes": ["SHADOW_SELECTION_ONLY"],
                "selected_candidate_strength": "moderate",
                "selected_stability_label": "single_horizon_only",
                "drift_direction": "flat",
            },
            {
                "rank": 1,
                "candidate_status": "penalized",
                "reason_codes": ["DRIFT_PENALIZED"],
                "selected_candidate_strength": "weak",
                "selected_stability_label": "single_horizon_only",
                "drift_direction": "decrease",
            },
        ],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any("rank must be unique" in error for error in result.errors)


def test_blocked_candidate_with_empty_reason_codes_fails() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "blocked",
        "selection_confidence": None,
        "selected_symbol": None,
        "selected_strategy": None,
        "selected_horizon": None,
        "selection_score": None,
        "reason_codes": ["ABSTAIN_POLICY_TRIGGERED"],
        "ranking": [
            {
                "rank": 1,
                "candidate_status": "blocked",
                "reason_codes": [],
                "selected_candidate_strength": "weak",
                "selected_stability_label": "unstable",
                "drift_direction": "decrease",
            }
        ],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any("must contain at least one entry" in error for error in result.errors)


def test_invalid_selected_visible_horizons_fails() -> None:
    payload = {
        "mode": "shadow",
        "selection_status": "blocked",
        "selection_confidence": None,
        "selected_symbol": None,
        "selected_strategy": None,
        "selected_horizon": None,
        "selection_score": None,
        "reason_codes": ["ABSTAIN_POLICY_TRIGGERED"],
        "ranking": [
            {
                "rank": 1,
                "candidate_status": "penalized",
                "reason_codes": ["DRIFT_PENALIZED"],
                "selected_candidate_strength": "weak",
                "selected_stability_label": "single_horizon_only",
                "selected_visible_horizons": ["1h", "2h"],
                "drift_direction": "decrease",
            }
        ],
    }

    result = validate_shadow_output(payload)

    assert result.is_valid is False
    assert any(
        "selected_visible_horizons must contain only" in error
        for error in result.errors
    )


def test_missing_required_upstream_report_fails(tmp_path: Path) -> None:
    _write_required_reports(tmp_path, skip={Path("comparison") / "summary.json"})

    result = validate_upstream_reports(tmp_path)

    assert result.is_valid is False
    assert any("Missing required upstream report" in error for error in result.errors)


def test_stale_required_upstream_report_fails(tmp_path: Path) -> None:
    _write_required_reports(tmp_path)
    stale_path = tmp_path / "latest" / "summary.json"
    _set_mtime_minutes_ago(stale_path, minutes_ago=180)

    result = validate_upstream_reports(tmp_path, max_age_minutes=90)

    assert result.is_valid is False
    assert any("Required upstream report is stale" in error for error in result.errors)


def test_invalid_max_age_minutes_fails(tmp_path: Path) -> None:
    _write_required_reports(tmp_path)

    result = validate_upstream_reports(tmp_path, max_age_minutes=0)

    assert result.is_valid is False
    assert any("max_age_minutes must be greater than 0" in error for error in result.errors)


def test_missing_optional_edge_scores_history_warns_but_does_not_fail(
    tmp_path: Path,
) -> None:
    _write_required_reports(tmp_path)

    result = validate_upstream_reports(tmp_path)

    assert result.is_valid is True
    assert any(
        "Optional upstream report is missing" in warning for warning in result.warnings
    )


def test_empty_optional_edge_scores_history_warns_but_does_not_fail(
    tmp_path: Path,
) -> None:
    _write_required_reports(tmp_path)
    history_path = tmp_path / "edge_scores_history.jsonl"
    history_path.write_text("", encoding="utf-8")

    result = validate_upstream_reports(tmp_path)

    assert result.is_valid is True
    assert any(
        "Optional upstream history file is empty" in warning
        for warning in result.warnings
    )


def _write_required_reports(base_dir: Path, skip: set[Path] | None = None) -> None:
    skip = skip or set()
    for relative_path in (
        Path("latest") / "summary.json",
        Path("comparison") / "summary.json",
        Path("edge_scores") / "summary.json",
        Path("score_drift") / "summary.json",
    ):
        if relative_path in skip:
            continue
        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"generated_at": datetime.now(UTC).isoformat()}),
            encoding="utf-8",
        )


def _set_mtime_minutes_ago(path: Path, *, minutes_ago: int) -> None:
    timestamp = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).timestamp()
    os.utime(path, (timestamp, timestamp))
