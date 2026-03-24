from __future__ import annotations

import json
from pathlib import Path

from src.research.shadow_selection_stability_diagnosis_report import (
    run_shadow_selection_stability_diagnosis_report,
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
    ranking: list[dict] | None = None,
) -> dict:
    return {
        "generated_at": generated_at,
        "mode": "shadow",
        "selection_status": selection_status,
        "reason_codes": [],
        "candidates_considered": len(ranking or []),
        "selected_symbol": selected_symbol,
        "selected_strategy": selected_strategy,
        "selected_horizon": selected_horizon,
        "selection_score": None,
        "selection_confidence": None,
        "selection_explanation": "test",
        "ranking": ranking or [],
    }


def test_shadow_selection_stability_diagnosis_report(tmp_path: Path) -> None:
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
                ranking=[
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                ],
            ),
            _record(
                generated_at="2026-03-24T01:00:00+00:00",
                selection_status="selected",
                selected_symbol="BTCUSDT",
                selected_strategy="swing",
                selected_horizon="4h",
                ranking=[_candidate("BTCUSDT", "swing", "4h", "eligible")],
            ),
            _record(
                generated_at="2026-03-24T02:00:00+00:00",
                selection_status="abstain",
                ranking=[
                    _candidate("BTCUSDT", "swing", "4h", "eligible"),
                    _candidate("ETHUSDT", "intraday", "1h", "eligible"),
                ],
            ),
            _record(
                generated_at="2026-03-24T03:00:00+00:00",
                selection_status="selected",
                selected_symbol="BTCUSDT",
                selected_strategy="swing",
                selected_horizon="4h",
                ranking=[_candidate("BTCUSDT", "swing", "4h", "eligible")],
            ),
            _record(
                generated_at="2026-03-24T04:00:00+00:00",
                selection_status="abstain",
                ranking=[_candidate("ETHUSDT", "intraday", "1h", "eligible")],
            ),
        ],
    )

    result = run_shadow_selection_stability_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        recent_run_limit=10,
    )

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    markdown = Path(result["summary_md"]).read_text(encoding="utf-8")

    btc = summary["candidates"][0]
    assert btc["symbol"] == "BTCUSDT"
    assert btc["selected_recurrence_count"] == 3
    assert btc["eligible_recurrence_count"] == 4
    assert btc["repeat_streak"]["selected"] == 2
    assert btc["convergence_label"] == "weak_convergence"
    assert "Shadow Selection Stability Diagnosis" in markdown
