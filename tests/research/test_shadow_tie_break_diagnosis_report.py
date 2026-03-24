from __future__ import annotations

import json
from pathlib import Path

from src.research.shadow_tie_break_diagnosis_report import (
    run_shadow_tie_break_diagnosis_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _candidate(symbol: str, strategy: str, horizon: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_status": "eligible",
        "selection_score": 8.0,
        "aggregate_score": 72.0,
        "sample_count": 50,
        "median_future_return_pct": 0.4,
        "positive_rate_pct": 57.0,
        "robustness_signal_pct": 53.0,
        "selected_candidate_strength": "moderate",
        "selected_stability_label": "single_horizon_only",
        "drift_direction": "flat",
        "edge_stability_score": 3.5,
    }


def _record(
    *,
    generated_at: str,
    selection_status: str,
    reason_codes: list[str],
    ranking: list[dict],
    selected_symbol: str | None = None,
    selected_strategy: str | None = None,
    selected_horizon: str | None = None,
    abstain_diagnosis: dict | None = None,
) -> dict:
    return {
        "generated_at": generated_at,
        "mode": "shadow",
        "selection_status": selection_status,
        "reason_codes": reason_codes,
        "candidates_considered": len(ranking),
        "selected_symbol": selected_symbol,
        "selected_strategy": selected_strategy,
        "selected_horizon": selected_horizon,
        "selection_score": None,
        "selection_confidence": None,
        "selection_explanation": "test",
        "ranking": ranking,
        "abstain_diagnosis": abstain_diagnosis,
    }


def test_shadow_tie_break_diagnosis_report(tmp_path: Path) -> None:
    input_path = tmp_path / "shadow.jsonl"
    output_dir = tmp_path / "out"
    btc = _candidate("BTCUSDT", "swing", "4h")
    eth = _candidate("ETHUSDT", "intraday", "1h")
    _write_jsonl(
        input_path,
        [
            _record(
                generated_at="2026-03-24T00:00:00+00:00",
                selection_status="abstain",
                reason_codes=["TOP_CANDIDATES_TIED"],
                ranking=[btc, eth],
                abstain_diagnosis={
                    "category": "tied_top_candidates",
                    "top_candidate": btc,
                    "compared_candidate": eth,
                },
            ),
            _record(
                generated_at="2026-03-24T01:00:00+00:00",
                selection_status="abstain",
                reason_codes=["TOP_CANDIDATES_TIED"],
                ranking=[btc, eth],
                abstain_diagnosis={
                    "category": "tied_top_candidates",
                    "top_candidate": btc,
                    "compared_candidate": eth,
                },
            ),
            _record(
                generated_at="2026-03-24T02:00:00+00:00",
                selection_status="selected",
                reason_codes=["CLEAR_TOP_CANDIDATE"],
                ranking=[btc],
                selected_symbol="BTCUSDT",
                selected_strategy="swing",
                selected_horizon="4h",
            ),
        ],
    )

    result = run_shadow_tie_break_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        recent_run_limit=10,
    )

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    markdown = Path(result["summary_md"]).read_text(encoding="utf-8")

    assert summary["overall"]["tie_runs"] == 2
    assert summary["overall"]["tie_frequency"] == 0.666667
    assert summary["overall"]["repeated_tie_pair_count"] == 1
    assert summary["overall"]["resolved_within_1_run_count"] == 1
    assert summary["overall"]["resolved_within_3_runs_count"] == 2
    assert summary["overall"]["resolved_within_5_runs_count"] == 2
    assert summary["overall"]["resolved_within_10_runs_count"] == 2
    assert summary["overall"]["resolved_within_1_run_ratio"] == 0.5
    assert summary["overall"]["resolved_within_3_runs_ratio"] == 1.0
    assert summary["overall"]["resolved_within_5_runs_ratio"] == 1.0
    assert summary["overall"]["resolved_within_10_runs_ratio"] == 1.0
    assert summary["overall"]["unresolved_tie_event_count"] == 0
    assert summary["overall"]["tie_signature_collision_count"] == 2
    assert summary["overall"]["time_to_resolution_summary"]["count"] == 2
    assert summary["overall"]["time_to_resolution_summary"]["min"] == 1
    assert summary["overall"]["time_to_resolution_summary"]["max"] == 2
    assert summary["overall"]["time_to_resolution_summary"]["median"] == 1.5
    assert summary["repeated_tie_pairs"][0]["resolved_after_any_tie"] is True
    assert summary["repeated_tie_pairs"][0]["resolved_within_3_runs_count"] == 2
    assert summary["repeated_tie_pairs"][0]["resolved_within_5_runs_count"] == 2
    assert summary["repeated_tie_pairs"][0]["resolved_within_10_runs_count"] == 2
    assert summary["repeated_tie_pairs"][0]["time_to_resolution_summary"]["count"] == 2
    assert summary["repeated_tie_pairs"][0]["time_to_resolution_summary"]["min"] == 1
    assert summary["repeated_tie_pairs"][0]["time_to_resolution_summary"]["max"] == 2
    assert summary["tie_event_resolutions"][0]["time_to_resolution_runs"] == 2
    assert summary["tie_event_resolutions"][1]["time_to_resolution_runs"] == 1
    assert summary["dominant_tie_dimensions"][0]["count"] == 2
    assert "Shadow Tie-Break Diagnosis" in markdown
