from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.research import research_analyzer
from src.research.diagnostics import (
    selected_strategy_direct_edge_selection_abstain_path_diagnosis_report as abstain_report,
)
from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_input_mapper import map_edge_selection_input


def test_rule_engine_strategy_rows_materialize_joined_edge_candidates(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(input_path, _rule_engine_strategy_records())

    result = research_analyzer._build_edge_candidate_rows(input_path)

    assert result["row_count"] == 1
    assert result["dropped_row_count"] == 0
    row = result["rows"][0]
    assert row["symbol"] == "BTCUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["selected_candidate_strength"] == "moderate"
    assert row["sample_count"] == 60
    assert row["labeled_count"] == 60
    assert result["empty_reason_summary"]["has_eligible_rows"] is True
    assert result["empty_reason_summary"]["empty_state_category"] == "has_eligible_rows"


def test_joined_edge_candidate_rows_materialize_mapper_seeds_and_candidates(
    tmp_path: Path,
) -> None:
    base_dir = _write_reports_with_joined_edge_rows(tmp_path)

    payload = map_edge_selection_input(base_dir)

    assert payload["ok"] is True
    assert payload["errors"] == []
    assert payload["candidate_seed_count"] == 1
    diagnostics = payload["candidate_seed_diagnostics"]
    assert diagnostics["seed_source"] == "latest.edge_candidate_rows"
    assert diagnostics["joined_candidate_row_count"] == 1
    assert diagnostics["candidate_seed_count"] == 1
    assert diagnostics["horizons_with_seed"] == ["4h"]
    assert diagnostics["fallback_blocked"] is False
    assert diagnostics["fallback_block_reason"] is None
    assert diagnostics["horizon_diagnostics"][-1]["seed_generated"] is True

    assert len(payload["candidates"]) == 1
    candidate = payload["candidates"][0]
    assert candidate["symbol"] == "BTCUSDT"
    assert candidate["strategy"] == "swing"
    assert candidate["horizon"] == "4h"
    assert candidate["seed_origin_type"] == "joined_edge_candidate_row"
    assert candidate["sample_count"] == 60
    assert candidate["labeled_count"] == 60


def test_joined_edge_candidate_rows_move_diagnosis_past_candidate_scarcity(
    tmp_path: Path,
) -> None:
    base_dir = _write_reports_with_joined_edge_rows(tmp_path)

    mapper_payload = map_edge_selection_input(base_dir)
    engine_output = run_edge_selection_engine(mapper_payload)

    assert mapper_payload["candidate_seed_count"] == 1
    assert len(mapper_payload["candidates"]) == 1
    assert engine_output["reason_codes"] != ["NO_CANDIDATES_AVAILABLE"]

    snapshot = {
        "available": True,
        "selection_status": engine_output["selection_status"],
        "reason": engine_output["reason_codes"][0],
        "reason_codes": engine_output["reason_codes"],
        "selection_explanation": engine_output["selection_explanation"],
        "candidate_count": len(mapper_payload["candidates"]),
        "candidate_seed_count": mapper_payload["candidate_seed_count"],
        "candidate_seed_diagnostics": mapper_payload["candidate_seed_diagnostics"],
        "ranking_count": len(engine_output["ranking"]),
        "ranking": engine_output["ranking"],
        "abstain_diagnosis": engine_output.get("abstain_diagnosis"),
    }
    summary = abstain_report.build_abstain_path_summary(
        {
            abstain_report._SNAPSHOT_BASELINE: snapshot,
            abstain_report._SNAPSHOT_PATCH_CLASS_A: snapshot,
            abstain_report._SNAPSHOT_PATCH_CLASS_B: snapshot,
        }
    )

    baseline = summary["snapshots"][abstain_report._SNAPSHOT_BASELINE]
    assert baseline["candidate_count"] == 1
    assert baseline["candidate_seed_count"] == 1
    assert baseline["candidate_presence"] is True
    assert baseline["abstain_path_classification"] == (
        "eligibility_or_acceptance_rejection"
    )
    assert summary["primary_abstain_path_classification"] == (
        "eligibility_or_acceptance_rejection"
    )


def _rule_engine_strategy_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, count, future_return, signal, bias in (
        ("up", 34, 0.42, "long", "bullish"),
        ("down", 26, 0.18, "short", "bearish"),
    ):
        for _ in range(count):
            records.append(
                {
                    "logged_at": "2026-04-01T00:00:00+00:00",
                    "symbol": "BTCUSDT",
                    "rule_engine": {
                        "strategy": "swing",
                        "bias": bias,
                        "signal": signal,
                    },
                    "risk": {
                        "execution_allowed": True,
                        "entry_price": 100.0,
                    },
                    "execution": {
                        "action": signal,
                        "entry_price": 100.0,
                    },
                    "future_label_4h": label,
                    "future_return_4h": future_return,
                }
            )
    return records


def _write_reports_with_joined_edge_rows(tmp_path: Path) -> Path:
    base_dir = tmp_path / "logs" / "research_reports"
    for name in ("latest", "comparison", "edge_scores", "score_drift"):
        (base_dir / name).mkdir(parents=True, exist_ok=True)

    _write_json(
        base_dir / "latest" / "summary.json",
        {
            "generated_at": "2026-03-15T00:00:00+00:00",
            "dataset_overview": {"total_records": 60},
            "edge_candidate_rows": {
                "row_count": 1,
                "rows": [_joined_edge_candidate_row()],
                "diagnostic_row_count": 0,
                "diagnostic_rows": [],
                "empty_reason_summary": {"has_eligible_rows": True},
                "dropped_row_count": 0,
                "dropped_rows": [],
                "identity_horizon_evaluations": [],
            },
        },
    )
    _write_json(
        base_dir / "comparison" / "summary.json",
        {
            "generated_at": "2026-03-15T00:01:00+00:00",
            "dataset_overview_comparison": {
                "latest_total_records": 60,
                "cumulative_total_records": 60,
            },
            "edge_candidates_comparison": {},
        },
    )
    _write_json(
        base_dir / "edge_scores" / "summary.json",
        {
            "generated_at": "2026-03-15T00:02:00+00:00",
            "edge_stability_scores": {
                "symbol": [_score_item("BTCUSDT", score=3.2)],
                "strategy": [_score_item("swing", score=3.0)],
                "alignment_state": [],
            },
        },
    )
    _write_json(
        base_dir / "score_drift" / "summary.json",
        {
            "generated_at": "2026-03-15T00:03:00+00:00",
            "score_drift": [
                {
                    "category": "symbol",
                    "group": "BTCUSDT",
                    "drift_direction": "decrease",
                    "score_delta": -0.4,
                }
            ],
        },
    )
    return base_dir


def _joined_edge_candidate_row() -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "horizon": "4h",
        "selected_candidate_strength": "moderate",
        "selected_stability_label": "single_horizon_only",
        "source_preference": "latest",
        "edge_stability_score": 3.2,
        "drift_direction": None,
        "score_delta": None,
        "selected_visible_horizons": ["4h"],
        "sample_count": 60,
        "labeled_count": 60,
        "coverage_pct": 100.0,
        "median_future_return_pct": 0.42,
        "avg_future_return_pct": 0.316,
        "positive_rate_pct": 56.666667,
        "robustness_signal_pct": 100.0,
        "aggregate_score": 78.0,
        "supporting_major_deficit_count": 0,
        "visibility_reason": "passed_sample_and_quality_gate",
        "chosen_metric_summary": "sample=60, median=0.42, positive_rate=56.666667",
    }


def _score_item(group: str, *, score: float) -> dict[str, Any]:
    return {
        "group": group,
        "score": score,
        "latest_stability_label": "single_horizon_only",
        "cumulative_stability_label": "single_horizon_only",
        "latest_candidate_strength": "moderate",
        "cumulative_candidate_strength": "moderate",
        "source_preference": "latest",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
