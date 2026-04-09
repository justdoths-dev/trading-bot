from __future__ import annotations

import json
from pathlib import Path

from src.research.sub_floor_candidate_validation import (
    build_sub_floor_candidate_validation_markdown,
    build_sub_floor_candidate_validation_report,
    run_sub_floor_candidate_validation,
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
    rejection_reason: str = "failed_absolute_minimum_gate",
    rejection_reasons: list[str] | None = None,
    sample_gate: str = "failed",
    quality_gate: str = "failed",
    candidate_strength: str = "insufficient_data",
    diagnostic_category: str = "insufficient_data",
    visibility_reason: str = "failed_absolute_minimum_gate",
    sample_count: int = 18,
    labeled_count: int = 18,
    median_future_return_pct: float | None = 0.21,
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
        "chosen_metric_summary": "sample=18, median=0.21, positive_rate=56.0",
        "visibility_reason": visibility_reason,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": 100.0,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
    }


def _summary_payload(
    diagnostic_rows: list[dict[str, object]] | None,
    *,
    max_age_hours: int = 36,
) -> dict[str, object]:
    edge_candidate_rows: dict[str, object] = {
        "row_count": 0,
        "diagnostic_row_count": len(diagnostic_rows or []),
        "empty_reason_summary": {
            "dominant_rejection_reason": "failed_absolute_minimum_gate",
        },
    }
    if diagnostic_rows is not None:
        edge_candidate_rows["diagnostic_rows"] = diagnostic_rows

    return {
        "schema_validation": {
            "max_age_hours": max_age_hours,
            "max_rows": 2500,
            "valid_records": 40,
        },
        "edge_candidate_rows": edge_candidate_rows,
    }


def test_classifies_positive_sub_floor_sample_only_near_miss(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "window_36h_summary.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=18,
                    median_future_return_pct=0.21,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ]
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])
    rows = report["classified_candidates"]["sample_floor_only_near_miss"]

    assert len(rows) == 1
    assert rows[0]["computed"]["classification"] == "sample_floor_only_near_miss"
    assert rows[0]["computed"]["sample_count_band"] == "10-19"
    assert rows[0]["computed"]["is_sample_floor_blocked"] is True
    assert rows[0]["computed"]["directional_quality_flags"] == []
    assert rows[0]["computed"]["structural_non_sample_blockers"] == []


def test_classifies_positive_sub_floor_with_directional_weakness(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "window_72h_summary.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=22,
                    median_future_return_pct=0.14,
                    positive_rate_pct=48.0,
                    robustness_signal_pct=47.0,
                )
            ],
            max_age_hours=72,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])
    rows = report["classified_candidates"]["sample_floor_plus_quality_weakness"]

    assert len(rows) == 1
    assert rows[0]["computed"]["classification"] == "sample_floor_plus_quality_weakness"
    assert "positive_rate_not_above_50_pct" in rows[0]["computed"]["directional_quality_flags"]
    assert (
        "signal_match_rate_pct_below_50_pct"
        in rows[0]["computed"]["directional_quality_flags"]
    )


def test_classifies_non_positive_median_rows_conservatively(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "window_144h_summary.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=25,
                    median_future_return_pct=0.0,
                    positive_rate_pct=52.0,
                    robustness_signal_pct=53.0,
                )
            ],
            max_age_hours=144,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])
    rows = report["classified_candidates"]["non_positive_or_flat_edge"]

    assert len(rows) == 1
    assert rows[0]["computed"]["classification"] == "non_positive_or_flat_edge"
    assert rows[0]["computed"]["classification_reasons"] == [
        "median_future_return_pct=0.0 is non-positive"
    ]


def test_aggregates_repeated_identity_across_multiple_windows(tmp_path: Path) -> None:
    path_36h = _write_json(
        tmp_path / "summary_36h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=18,
                    median_future_return_pct=0.21,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=36,
        ),
    )
    path_72h = _write_json(
        tmp_path / "summary_72h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=22,
                    median_future_return_pct=0.16,
                    positive_rate_pct=51.0,
                    robustness_signal_pct=52.0,
                )
            ],
            max_age_hours=72,
        ),
    )

    report = build_sub_floor_candidate_validation_report([path_36h, path_72h])

    assert report["overall_summary"]["repeated_identity_count"] == 1
    repeated = report["repeated_identity_summary"][0]
    assert repeated["identity_key"] == "BTCUSDT:swing:4h"
    assert repeated["distinct_window_count"] == 2
    assert repeated["window_labels"] == ["36h", "72h"]

    ranking = report["near_miss_interest_ranking"][0]
    assert ranking["identity_key"] == "BTCUSDT:swing:4h"
    assert ranking["repeated_identity"] is True


def test_handles_missing_edge_candidate_rows_or_diagnostic_rows_safely(tmp_path: Path) -> None:
    missing_block = _write_json(
        tmp_path / "missing_block.json",
        {"schema_validation": {"max_age_hours": 36}},
    )
    missing_diagnostics = _write_json(
        tmp_path / "missing_diagnostics.json",
        _summary_payload(None, max_age_hours=72),
    )

    report = build_sub_floor_candidate_validation_report([missing_block, missing_diagnostics])

    assert report["overall_summary"]["observation_count"] == 0
    assert report["classified_candidates"]["sample_floor_only_near_miss"] == []
    assert any(
        warning.endswith("edge_candidate_rows_block_missing")
        for warning in report["warnings"]
    )
    assert any(
        warning.endswith("edge_candidate_rows.diagnostic_rows_missing_or_empty")
        for warning in report["warnings"]
    )


def test_json_report_schema_and_markdown_output_are_stable(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "summary_36h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=18,
                    median_future_return_pct=0.21,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=36,
        ),
    )

    result = run_sub_floor_candidate_validation(
        summary_paths=[summary_path],
        output_dir=tmp_path / "out",
    )
    written_json = json.loads(
        (tmp_path / "out" / "sub_floor_candidate_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["summary_json"].endswith("sub_floor_candidate_validation_summary.json")
    assert result["summary_md"].endswith("sub_floor_candidate_validation_summary.md")
    assert written_json["metadata"]["report_type"] == "sub_floor_candidate_validation"
    assert written_json["metadata"]["classification_version"] == "conservative_v2"
    assert set(written_json.keys()) == {
        "metadata",
        "source_summaries",
        "overall_summary",
        "classified_candidates",
        "repeated_identity_summary",
        "near_miss_interest_ranking",
        "sample_band_summary",
        "warnings",
    }

    markdown = build_sub_floor_candidate_validation_markdown(result["summary"])
    assert "## Sample Bands" in markdown
    assert "## Near-Miss Interest Ranking" in markdown
    assert "sample_floor_only_near_miss" in markdown


def test_failed_absolute_minimum_gate_without_explicit_sample_floor_evidence_is_not_sample_floor_blocked(
    tmp_path: Path,
) -> None:
    summary_path = _write_json(
        tmp_path / "summary_36h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    rejection_reason="failed_absolute_minimum_gate",
                    rejection_reasons=["failed_absolute_minimum_gate"],
                    visibility_reason="failed_absolute_minimum_gate",
                    sample_gate="failed",
                    sample_count=18,
                    median_future_return_pct=0.24,
                    positive_rate_pct=58.0,
                    robustness_signal_pct=55.0,
                )
            ],
            max_age_hours=36,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])
    rows = report["classified_candidates"]["non_sample_primary_failure"]

    assert len(rows) == 1
    assert rows[0]["computed"]["classification"] == "non_sample_primary_failure"
    assert rows[0]["computed"]["is_sample_floor_blocked"] is False
    assert "explicit sample floor evidence is absent from rejection_reasons" in rows[0]["computed"]["classification_reasons"]


def test_structural_non_sample_blocker_excludes_row_from_near_miss(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "summary_72h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    rejection_reasons=[
                        "failed_absolute_minimum_gate",
                        "sample_count_below_absolute_floor",
                        "strategy_horizon_incompatible",
                    ],
                    sample_count=22,
                    median_future_return_pct=0.17,
                    positive_rate_pct=59.0,
                    robustness_signal_pct=57.0,
                )
            ],
            max_age_hours=72,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])

    assert report["classified_candidates"]["sample_floor_only_near_miss"] == []
    rows = report["classified_candidates"]["non_sample_primary_failure"]
    assert len(rows) == 1
    assert rows[0]["computed"]["structural_non_sample_blockers"] == [
        "strategy_horizon_incompatible"
    ]


def test_non_positive_rows_are_not_in_near_miss_interest_ranking(tmp_path: Path) -> None:
    path_36h = _write_json(
        tmp_path / "summary_36h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=18,
                    median_future_return_pct=0.0,
                    positive_rate_pct=55.0,
                    robustness_signal_pct=55.0,
                )
            ],
            max_age_hours=36,
        ),
    )
    path_72h = _write_json(
        tmp_path / "summary_72h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=22,
                    median_future_return_pct=-0.01,
                    positive_rate_pct=51.0,
                    robustness_signal_pct=53.0,
                )
            ],
            max_age_hours=72,
        ),
    )

    report = build_sub_floor_candidate_validation_report([path_36h, path_72h])

    assert report["overall_summary"]["repeated_identity_count"] == 1
    assert report["near_miss_interest_ranking"] == []


def test_sample_count_below_30_without_failed_sample_gate_is_not_sample_floor_near_miss(
    tmp_path: Path,
) -> None:
    summary_path = _write_json(
        tmp_path / "summary_144h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_gate="passed",
                    rejection_reason="failed_absolute_minimum_gate",
                    rejection_reasons=["failed_absolute_minimum_gate"],
                    sample_count=25,
                    median_future_return_pct=0.19,
                    positive_rate_pct=57.0,
                    robustness_signal_pct=56.0,
                )
            ],
            max_age_hours=144,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])
    rows = report["classified_candidates"]["non_sample_primary_failure"]

    assert len(rows) == 1
    assert rows[0]["computed"]["classification"] == "non_sample_primary_failure"
    assert rows[0]["computed"]["is_sample_floor_blocked"] is False


def test_window_label_source_uses_actual_metadata_path_and_does_not_warn_for_schema_validation(
    tmp_path: Path,
) -> None:
    summary_path = _write_json(
        tmp_path / "summary_36h.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=18,
                    median_future_return_pct=0.21,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=36,
        ),
    )

    report = build_sub_floor_candidate_validation_report([summary_path])

    source_summary = report["source_summaries"][0]
    assert source_summary["window_label"] == "36h"
    assert source_summary["window_label_source"] == "schema_validation.max_age_hours"

    row = report["classified_candidates"]["sample_floor_only_near_miss"][0]
    assert row["warnings"] == []