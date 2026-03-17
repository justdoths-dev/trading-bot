from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.edge_selection_schema_validator import validate_shadow_output
from src.research.edge_selection_shadow_writer import read_edge_selection_shadow_outputs
from src.research.shadow_observation_analyzer import run_shadow_observation_analyzer


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "shadow_observation_fixture.jsonl"
)


def _count_map(rows: list[dict], key: str = "value") -> dict[str, int]:
    return {row[key]: row["count"] for row in rows}


def _rate_map(rows: list[dict], key: str = "value") -> dict[str, float]:
    return {row[key]: row["rate"] for row in rows}


def _identity_frequency_map(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (row["symbol"], row["strategy"], row["horizon"]): row
        for row in rows
    }


def _run_analyzer(tmp_path: Path) -> tuple[dict, dict, str]:
    result = run_shadow_observation_analyzer(
        input_path=FIXTURE_PATH,
        output_dir=tmp_path,
    )

    summary_path = Path(result["summary_json"])
    md_path = Path(result["summary_md"])

    assert summary_path.exists() is True
    assert md_path.exists() is True

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    md_text = md_path.read_text(encoding="utf-8")

    return result, summary, md_text


# ---------------------------
# Schema Validation Test
# ---------------------------

def test_shadow_observation_fixture_matches_writer_schema() -> None:
    assert FIXTURE_PATH.exists(), f"Missing fixture file: {FIXTURE_PATH}"

    records = read_edge_selection_shadow_outputs(FIXTURE_PATH)
    assert len(records) == 6

    for record in records:
        validation = validate_shadow_output(record)
        assert validation.is_valid is True, validation.errors


# ---------------------------
# Overall Summary Test
# ---------------------------

def test_shadow_observation_analyzer_overall_summary(tmp_path: Path) -> None:
    result, summary, _ = _run_analyzer(tmp_path)

    assert result["run_count"] == 6
    assert summary["report_type"] == "shadow_observation_summary"

    overall = summary["overall"]
    assert overall["total_runs"] == 6
    assert overall["selected_runs"] == 2
    assert overall["abstain_runs"] == 3
    assert overall["blocked_runs"] == 1

    reason_count_map = _count_map(overall["run_reason_code_frequencies"])
    reason_rate_map = _rate_map(overall["run_reason_code_frequencies"])

    assert reason_count_map["CLEAR_TOP_CANDIDATE"] == 2
    assert reason_rate_map["CLEAR_TOP_CANDIDATE"] == pytest.approx(0.3333, abs=1e-4)
    assert reason_count_map["TOP_CANDIDATES_TIED"] == 1
    assert reason_count_map["NO_ELIGIBLE_CANDIDATES"] == 1
    assert reason_count_map["ALL_CANDIDATES_BLOCKED"] == 1
    assert reason_count_map["UPSTREAM_INPUT_INVALID"] == 1

    top_ranked = overall["top_ranked_candidate_summaries"]
    assert top_ranked["runs_with_top_candidate"] == 5
    assert top_ranked["repeated_identity_count"] == 1
    assert top_ranked["repeated_top_candidate_run_count"] == 2
    assert top_ranked["replacement_count"] == 3
    assert top_ranked["comparable_transitions"] == 4
    assert top_ranked["replacement_rate"] == pytest.approx(0.75, abs=1e-9)

    identity_map = _identity_frequency_map(top_ranked["identity_frequencies"])
    btc_swing_4h = identity_map[("BTCUSDT", "swing", "4h")]
    assert btc_swing_4h["count"] == 2
    assert btc_swing_4h["rate"] == pytest.approx(0.4, abs=1e-9)

    drift_distribution = _count_map(overall["drift_direction_distribution"])
    assert drift_distribution == {
        "increase": 3,
        "flat": 3,
        "decrease": 2,
        "insufficient_history": 2,
    }

    source_distribution = _count_map(overall["source_preference_distribution"])
    assert source_distribution == {
        "latest": 5,
        "cumulative": 3,
        "n/a": 2,
    }


# ---------------------------
# Time Window & Output Test
# ---------------------------

def test_shadow_observation_analyzer_time_windows_and_outputs(tmp_path: Path) -> None:
    _, summary, md_text = _run_analyzer(tmp_path)

    by_day = {row["day"]: row for row in summary["by_day"]}
    assert set(by_day) == {"2026-03-15", "2026-03-16", "2026-03-17"}

    assert by_day["2026-03-15"]["runs"] == 2
    assert by_day["2026-03-15"]["top_candidate_replacement_count"] == 0

    assert by_day["2026-03-16"]["runs"] == 2
    assert by_day["2026-03-16"]["top_candidate_replacement_count"] == 1
    assert by_day["2026-03-16"]["top_candidate_replacement_rate"] == pytest.approx(
        1.0,
        abs=1e-9,
    )

    assert by_day["2026-03-17"]["runs"] == 2
    assert by_day["2026-03-17"]["abstain_runs"] == 1
    assert by_day["2026-03-17"]["blocked_runs"] == 1
    assert by_day["2026-03-17"]["unique_top_ranked_candidates"] == 1

    assert summary["last_24h"]["run_count"] == 3
    assert summary["last_7d"]["run_count"] == 6

    recent_runs = summary["recent_runs"]
    assert len(recent_runs) == 6

    latest_run = max(recent_runs, key=lambda row: row["generated_at"])
    assert latest_run["generated_at"] == "2026-03-17T12:00:00+00:00"
    assert latest_run["selection_status"] == "blocked"
    assert latest_run["top_candidate"]["symbol"] is None

    data_quality = summary["data_quality"]
    assert data_quality["parsed_runs"] == 6
    assert data_quality["malformed_lines"] == 0
    assert data_quality["runs_with_empty_ranking"] == 1
    assert data_quality["runs_with_top_candidate"] == 5

    assert "shadow_observation_summary" in md_text or "Shadow Observation" in md_text
    assert "Replacement" in md_text


# ---------------------------
# Empty Input Test
# ---------------------------

def test_shadow_observation_empty_input(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    result = run_shadow_observation_analyzer(
        input_path=empty_file,
        output_dir=tmp_path,
    )

    summary_path = Path(result["summary_json"])
    md_path = Path(result["summary_md"])

    assert summary_path.exists() is True
    assert md_path.exists() is True

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["run_count"] == 0
    assert summary["report_type"] == "shadow_observation_summary"

    overall = summary["overall"]
    assert overall["total_runs"] == 0
    assert overall["selected_runs"] == 0
    assert overall["abstain_runs"] == 0
    assert overall["blocked_runs"] == 0

    assert summary["by_day"] == []
    assert summary["recent_runs"] == []

    data_quality = summary["data_quality"]
    assert data_quality["parsed_runs"] == 0


# ---------------------------
# Malformed JSON Test
# ---------------------------

def test_shadow_observation_malformed_json_line(tmp_path: Path) -> None:
    malformed_file = tmp_path / "malformed.jsonl"

    malformed_file.write_text(
        (
            '{"generated_at":"2026-03-17T06:15:00+00:00","mode":"shadow","selection_status":"abstain","reason_codes":["ALL_CANDIDATES_BLOCKED"],"candidates_considered":2,"selected_symbol":null,"selected_strategy":null,"selected_horizon":null,"selection_score":null,"selection_confidence":null,"latest_window_record_count":130,"cumulative_record_count":2420,"selection_explanation":"Abstained","ranking":[{"rank":1,"symbol":"XRPUSDT","strategy":"mean_reversion","horizon":"1h","candidate_status":"blocked","reason_codes":["CANDIDATE_STABILITY_UNSTABLE"],"selected_candidate_strength":"moderate","selected_stability_label":"unstable","drift_direction":"insufficient_history","source_preference":"n/a","edge_stability_score":3.1,"selected_visible_horizons":["1h"],"latest_sample_size":19,"cumulative_sample_size":160,"symbol_cumulative_support":220,"strategy_cumulative_support":181,"stability_gate_pass":false,"drift_blocked":false}]}\n'
            '{"generated_at":"2026-03-17T12:00:00+00:00","mode":"shadow","selection_status":"blocked"\n'
        ),
        encoding="utf-8",
    )

    result = run_shadow_observation_analyzer(
        input_path=malformed_file,
        output_dir=tmp_path,
    )

    summary_path = Path(result["summary_json"])
    md_path = Path(result["summary_md"])

    assert summary_path.exists() is True
    assert md_path.exists() is True

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["run_count"] == 1

    overall = summary["overall"]
    assert overall["total_runs"] == 1
    assert overall["abstain_runs"] == 1

    data_quality = summary["data_quality"]
    assert data_quality["parsed_runs"] == 1
    assert data_quality["malformed_lines"] == 1

    