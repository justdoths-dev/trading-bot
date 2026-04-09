from __future__ import annotations

import json
from pathlib import Path

from src.research.historical_persistence_diagnosis import (
    build_historical_persistence_diagnosis_report,
    run_historical_persistence_diagnosis,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _diagnostic_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    sample_count: int = 18,
    labeled_count: int | None = None,
    rejection_reason: str = "failed_absolute_minimum_gate",
    rejection_reasons: list[str] | None = None,
    sample_gate: str = "failed",
    quality_gate: str = "passed",
    candidate_strength: str = "moderate",
    diagnostic_category: str = "insufficient_data",
    visibility_reason: str = "failed_absolute_minimum_gate",
    median_future_return_pct: float | None = 0.18,
    positive_rate_pct: float | None = 56.0,
    robustness_signal_pct: float | None = 54.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "diagnostic_category": diagnostic_category,
        "strategy_horizon_compatible": True,
        "rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons
        or ["failed_absolute_minimum_gate", "sample_count_below_absolute_floor"],
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": candidate_strength,
        "classification_reason": None,
        "aggregate_score": None,
        "visibility_reason": visibility_reason,
        "sample_count": sample_count,
        "labeled_count": labeled_count if labeled_count is not None else sample_count,
        "coverage_pct": 100.0,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
    }


def _eligible_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    sample_count: int = 33,
    labeled_count: int | None = None,
    median_future_return_pct: float | None = 0.22,
    positive_rate_pct: float | None = 58.0,
    robustness_signal_pct: float | None = 56.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_candidate_strength": "moderate",
        "sample_count": sample_count,
        "labeled_count": labeled_count if labeled_count is not None else sample_count,
        "coverage_pct": 100.0,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
        "aggregate_score": 0.74,
        "visibility_reason": "passed_sample_and_quality_gate",
    }


def _historical_horizon_eval(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    status: str = "rejected",
    sample_count: int = 18,
    labeled_count: int | None = None,
    rejection_reason: str = "failed_absolute_minimum_gate",
    rejection_reasons: list[str] | None = None,
    sample_gate: str = "failed",
    quality_gate: str = "passed",
    candidate_strength: str = "moderate",
    visibility_reason: str = "failed_absolute_minimum_gate",
    strategy_horizon_compatible: bool = True,
    median_future_return_pct: float | None = 0.18,
    positive_rate_pct: float | None = 56.0,
    signal_match_rate_pct: float | None = 54.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": status,
        "strategy_horizon_compatible": strategy_horizon_compatible,
        "rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons
        or ["failed_absolute_minimum_gate", "sample_count_below_absolute_floor"],
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": candidate_strength,
        "candidate_strength_diagnostics": None,
        "metrics": {
            "sample_count": sample_count,
            "labeled_count": labeled_count if labeled_count is not None else sample_count,
            "coverage_pct": 100.0,
            "median_future_return_pct": median_future_return_pct,
            "positive_rate_pct": positive_rate_pct,
            "signal_match_rate_pct": signal_match_rate_pct,
        },
        "aggregate_score": 0.61 if status == "selected" else None,
        "visibility_reason": visibility_reason,
    }


def _summary_payload(
    diagnostic_rows: list[dict[str, object]] | None,
    *,
    selected_rows: list[dict[str, object]] | None = None,
    identity_horizon_evaluations: list[dict[str, object]] | None = None,
    generated_at: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, object]:
    edge_candidate_rows: dict[str, object] = {
        "row_count": len(selected_rows or []),
        "rows": selected_rows or [],
        "diagnostic_row_count": len(diagnostic_rows or []),
        "empty_reason_summary": {
            "dominant_rejection_reason": "failed_absolute_minimum_gate",
        },
    }
    if diagnostic_rows is not None:
        edge_candidate_rows["diagnostic_rows"] = diagnostic_rows
    if identity_horizon_evaluations is not None:
        edge_candidate_rows["identity_horizon_evaluations"] = identity_horizon_evaluations

    return {
        "generated_at": generated_at,
        "schema_validation": {
            "max_age_hours": 36,
            "max_rows": 2500,
            "valid_records": 40,
        },
        "edge_candidate_rows": edge_candidate_rows,
    }


def _identity_eval_payload(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon_rows: dict[str, dict[str, object]],
    generated_at: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, object]:
    return _summary_payload(
        None,
        identity_horizon_evaluations=[
            {
                "identity_key": f"{symbol}:{strategy}",
                "symbol": symbol,
                "strategy": strategy,
                "horizon_evaluations": horizon_rows,
            }
        ],
        generated_at=generated_at,
    )


def _identity_entry(summary: dict[str, object], identity_key: str) -> dict[str, object]:
    for row in summary["identity_summary"]:
        if row["identity_key"] == identity_key:
            return row
    raise AssertionError(f"Identity {identity_key} not found")


def test_single_appearance_then_disappeared(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [_diagnostic_row(sample_count=9)],
            generated_at="2026-01-01T00:00:00+00:00",
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [],
            generated_at="2026-01-02T00:00:00+00:00",
        ),
    )

    summary = build_historical_persistence_diagnosis_report([first, second])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["total_appearances"] == 1
    assert identity["followup_count"] == 0
    assert identity["final_state"] == "disappeared"
    assert identity["disappearance_type"] == "early_disappearance"
    assert identity["recurrence_strength"] == "single"
    assert identity["persistence_label"] == "noise_like_singleton"
    assert identity["sample_count_growth_pct_reliable"] is False


def test_repeated_appearances_with_sample_growth_and_stable_quality(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "01.json",
        _summary_payload(
            [_diagnostic_row(sample_count=12, median_future_return_pct=0.10, positive_rate_pct=54.0, robustness_signal_pct=53.0)],
            generated_at="2026-01-01T00:00:00+00:00",
        ),
    )
    second = _write_json(
        tmp_path / "02.json",
        _summary_payload(
            [_diagnostic_row(sample_count=18, median_future_return_pct=0.16, positive_rate_pct=56.0, robustness_signal_pct=55.0)],
            generated_at="2026-01-02T00:00:00+00:00",
        ),
    )
    third = _write_json(
        tmp_path / "03.json",
        _summary_payload(
            [_diagnostic_row(sample_count=24, median_future_return_pct=0.21, positive_rate_pct=58.0, robustness_signal_pct=57.0)],
            generated_at="2026-01-03T00:00:00+00:00",
        ),
    )

    summary = build_historical_persistence_diagnosis_report([first, second, third])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["sample_count_first"] == 12
    assert identity["sample_count_last"] == 24
    assert identity["sample_count_growth_abs"] == 12
    assert identity["sample_count_growth_pct"] == 100.0
    assert identity["sample_count_growth_pct_reliable"] is True
    assert identity["consecutive_appearance_max"] == 3
    assert identity["sample_count_non_decreasing_steps"] == 2
    assert identity["sample_count_drop_steps"] == 0
    assert identity["recurrence_strength"] == "moderate"
    assert identity["persistence_label"] == "early_stage_growth_candidate"


def test_repeated_growth_but_weak_quality_stays_unstable(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "01.json",
        _summary_payload(
            [_diagnostic_row(sample_count=12, median_future_return_pct=0.08, positive_rate_pct=52.0, robustness_signal_pct=52.0)],
            generated_at="2026-01-01T00:00:00+00:00",
        ),
    )
    second = _write_json(
        tmp_path / "02.json",
        _summary_payload(
            [_diagnostic_row(sample_count=18, median_future_return_pct=0.05, positive_rate_pct=49.0, robustness_signal_pct=51.0)],
            generated_at="2026-01-02T00:00:00+00:00",
        ),
    )
    third = _write_json(
        tmp_path / "03.json",
        _summary_payload(
            [_diagnostic_row(sample_count=24, median_future_return_pct=-0.02, positive_rate_pct=48.0, robustness_signal_pct=49.0)],
            generated_at="2026-01-03T00:00:00+00:00",
        ),
    )

    summary = build_historical_persistence_diagnosis_report([first, second, third])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["sample_count_growth_abs"] == 12
    assert identity["sample_count_drop_steps"] == 0
    assert identity["persistence_label"] == "unstable_recurrent_candidate"


def test_repeated_but_unstable_candidate(tmp_path: Path) -> None:
    first = _write_json(tmp_path / "01.json", _summary_payload([_diagnostic_row(sample_count=8)]))
    second = _write_json(tmp_path / "02.json", _summary_payload([_diagnostic_row(sample_count=7)]))
    third = _write_json(tmp_path / "03.json", _summary_payload([_diagnostic_row(sample_count=10)]))

    summary = build_historical_persistence_diagnosis_report([first, second, third])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["sample_count_growth_abs"] == 2
    assert identity["sample_count_drop_steps"] == 1
    assert identity["persistence_label"] == "unstable_recurrent_candidate"
    assert identity["state_transition_counts"] == {
        "sample_floor_only_near_miss->sample_floor_only_near_miss": 2
    }


def test_eligible_transition_is_tracked(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "01.json",
        _summary_payload(
            [_diagnostic_row(sample_count=12, median_future_return_pct=0.09, positive_rate_pct=53.0, robustness_signal_pct=52.0)]
        ),
    )
    second = _write_json(
        tmp_path / "02.json",
        _summary_payload(
            [_diagnostic_row(sample_count=18, median_future_return_pct=0.15, positive_rate_pct=56.0, robustness_signal_pct=55.0)]
        ),
    )
    third = _write_json(
        tmp_path / "03.json",
        _summary_payload([], selected_rows=[_eligible_row(sample_count=33)]),
    )

    summary = build_historical_persistence_diagnosis_report([first, second, third])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["ever_eligible"] is True
    assert identity["final_state"] == "eligible"
    assert identity["disappearance_type"] is None
    assert identity["sample_count_growth_abs"] == 21
    assert identity["timeline"][-1]["classification"] == "eligible"


def test_same_snapshot_duplicate_identity_is_deduplicated(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(sample_count=11),
                _diagnostic_row(sample_count=19),
            ]
        ),
    )
    second = _write_json(tmp_path / "second.json", _summary_payload([]))

    summary = build_historical_persistence_diagnosis_report([first, second])
    source = summary["source_summaries"][0]
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert source["source_facts"]["same_snapshot_duplicate_identity_count"] == 1
    assert source["source_facts"]["same_snapshot_dropped_observation_count"] == 1
    assert identity["total_appearances"] == 1
    assert identity["sample_count_first"] == 19


def test_identity_horizon_evaluations_fallback_is_used_safely(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _identity_eval_payload(
            horizon_rows={
                "4h": _historical_horizon_eval(sample_count=18),
            },
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload([], selected_rows=[_eligible_row(sample_count=32)]),
    )

    summary = build_historical_persistence_diagnosis_report([first, second])
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["ever_eligible"] is True
    assert any(
        "diagnostic_rows_synthesized_from_identity_horizon_evaluations" in warning
        for warning in summary["source_summaries"][0]["warnings"]
    )
    assert summary["source_summaries"][0]["extraction_mode"].startswith(
        "identity_horizon_evaluations"
    )


def test_run_writes_expected_output_files(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "snapshot.json",
        _summary_payload([_diagnostic_row(sample_count=15)]),
    )

    result = run_historical_persistence_diagnosis(
        summary_paths=[summary_path],
        write_latest_copy=True,
        output_dir=tmp_path / "out",
    )

    assert result["summary_json"].endswith("historical_persistence_diagnosis_summary.json")
    assert result["identity_summary_json"].endswith(
        "historical_persistence_identity_summary.json"
    )
    assert result["appearance_rows_jsonl"].endswith(
        "historical_persistence_appearance_rows.jsonl"
    )
    assert result["summary_md"].endswith("historical_persistence_diagnosis_summary.md")

    written_summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    written_identity_summary = json.loads(
        Path(result["identity_summary_json"]).read_text(encoding="utf-8")
    )
    appearance_rows = Path(result["appearance_rows_jsonl"]).read_text(
        encoding="utf-8"
    ).strip()

    assert written_summary["metadata"]["report_type"] == "historical_persistence_diagnosis"
    assert written_summary["overall_summary"]["recurrence_strength_counts"]["single"] == 1
    assert written_identity_summary[0]["identity_key"] == "BTCUSDT:swing:4h"
    assert "\"identity_key\": \"BTCUSDT:swing:4h\"" in appearance_rows