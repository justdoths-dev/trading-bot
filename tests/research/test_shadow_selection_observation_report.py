from __future__ import annotations

import json
from pathlib import Path

from src.research.shadow_selection_observation_report import (
    run_shadow_selection_observation_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _candidate(symbol: str, strategy: str, horizon: str, status: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "candidate_status": status,
    }


def _record(
    *,
    generated_at: str,
    selection_status: str,
    selected_symbol: str | None = None,
    selected_strategy: str | None = None,
    selected_horizon: str | None = None,
    selection_score: float | None = None,
    reason_codes: list[str] | None = None,
    ranking: list[dict] | None = None,
    abstain_diagnosis: dict | None = None,
) -> dict:
    return {
        "generated_at": generated_at,
        "mode": "shadow",
        "selection_status": selection_status,
        "reason_codes": reason_codes or [],
        "candidates_considered": len(ranking or []),
        "selected_symbol": selected_symbol,
        "selected_strategy": selected_strategy,
        "selected_horizon": selected_horizon,
        "selection_score": selection_score,
        "selection_confidence": None,
        "selection_explanation": "test",
        "ranking": ranking or [],
        "abstain_diagnosis": abstain_diagnosis,
    }


def test_shadow_selection_observation_report(tmp_path: Path) -> None:
    input_path = tmp_path / "shadow.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        input_path,
        [
            _record(
                generated_at="2026-03-24T00:00:00+00:00",
                selection_status="selected",
                selected_symbol="BTCUSDT",
                selected_strategy="swing",
                selected_horizon="4h",
                selection_score=9.0,
                reason_codes=["CLEAR_TOP_CANDIDATE"],
                ranking=[
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                    _candidate("SOLUSDT", "intraday", "1h", "penalized"),
                ],
            ),
            _record(
                generated_at="2026-03-24T01:00:00+00:00",
                selection_status="selected",
                selected_symbol="BTCUSDT",
                selected_strategy="swing",
                selected_horizon="4h",
                selection_score=9.2,
                reason_codes=["CLEAR_TOP_CANDIDATE"],
                ranking=[
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                ],
            ),
            _record(
                generated_at="2026-03-24T02:00:00+00:00",
                selection_status="abstain",
                reason_codes=["TOP_CANDIDATES_TIED"],
                ranking=[
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                    _candidate("SOLUSDT", "intraday", "1h", "penalized"),
                ],
                abstain_diagnosis={"category": "tied_top_candidates"},
            ),
            _record(
                generated_at="2026-03-24T03:00:00+00:00",
                selection_status="selected",
                selected_symbol="ETHUSDT",
                selected_strategy="intraday",
                selected_horizon="1h",
                selection_score=8.7,
                reason_codes=["CLEAR_TOP_CANDIDATE"],
                ranking=[
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                ],
            ),
        ],
    )

    result = run_shadow_selection_observation_report(
        input_path=input_path,
        output_dir=output_dir,
        recent_run_limit=10,
    )

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    markdown = Path(result["summary_md"]).read_text(encoding="utf-8")

    assert summary["selection_outcomes"]["selected_ratio"] == 0.75
    assert summary["selection_outcomes"]["abstain_ratio"] == 0.25
    assert summary["selection_outcomes"]["tie_frequency"] == 0.25
    assert summary["selected_candidate_repetition"]["unique_selected_candidate_count"] == 2
    assert summary["selected_candidate_repetition"]["longest_repeat_streak"]["streak"] == 2
    assert summary["selection_score_summary"]["count"] == 3
    assert summary["abstain_reason_summary"][0]["reason_code"] == "TOP_CANDIDATES_TIED"
    assert "Recent Shadow Selection Observation Report" in markdown
