from __future__ import annotations

import json
from pathlib import Path

from src.research.candidate_seed_failure_diagnosis_report import (
    build_candidate_seed_failure_diagnosis_summary,
    load_shadow_records,
    render_candidate_seed_failure_diagnosis_markdown,
    run_candidate_seed_failure_diagnosis_report,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row))
            handle.write("\n")


def _seed_diag(
    *,
    total_horizons_evaluated: int = 3,
    horizons_with_seed: list[str] | None = None,
    horizons_without_seed: list[str] | None = None,
    all_insufficient: bool = False,
    horizon_diagnostics: list[dict] | None = None,
) -> dict:
    return {
        "total_horizons_evaluated": total_horizons_evaluated,
        "horizons_with_seed": horizons_with_seed or [],
        "horizons_without_seed": horizons_without_seed or [],
        "all_horizons_insufficient_data": all_insufficient,
        "horizon_diagnostics": horizon_diagnostics or [],
    }


def _record(
    *,
    generated_at: str,
    selection_status: str = "abstain",
    candidate_seed_count: int = 0,
    candidate_seed_diagnostics: dict | None = None,
    abstain_category: str = "no_candidates_available",
    ranking: list[dict] | None = None,
    cumulative_record_count: int = 100,
) -> dict:
    return {
        "generated_at": generated_at,
        "selection_status": selection_status,
        "candidate_seed_count": candidate_seed_count,
        "candidate_seed_diagnostics": candidate_seed_diagnostics or {},
        "abstain_diagnosis": {
            "category": abstain_category,
            "top_candidate": (
                ranking[0]
                if ranking
                else {
                    "selected_candidate_strength": "insufficient_data",
                    "selected_stability_label": "insufficient_data",
                    "selected_visible_horizons": [],
                }
            ),
        },
        "candidates_considered": len(ranking or []),
        "ranking": ranking or [],
        "cumulative_record_count": cumulative_record_count,
    }


def test_normal_dataset_case(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                candidate_seed_count=0,
                candidate_seed_diagnostics=_seed_diag(
                    horizons_without_seed=["15m", "1h", "4h"],
                    all_insufficient=True,
                    horizon_diagnostics=[
                        {
                            "horizon": "15m",
                            "blocker_reasons": [
                                "candidate_strength_insufficient_data",
                                "no_valid_symbol_group",
                            ],
                            "latest_candidate_strength": "insufficient_data",
                            "cumulative_candidate_strength": "insufficient_data",
                        },
                        {
                            "horizon": "1h",
                            "blocker_reasons": ["no_valid_strategy_group"],
                            "latest_candidate_strength": "weak",
                            "cumulative_candidate_strength": "weak",
                        },
                    ],
                ),
            ),
            _record(
                generated_at="2026-03-21T01:00:00+00:00",
                candidate_seed_count=2,
                abstain_category="no_eligible_candidates",
                candidate_seed_diagnostics=_seed_diag(
                    horizons_with_seed=["15m", "1h"],
                    horizons_without_seed=["4h"],
                    horizon_diagnostics=[
                        {
                            "horizon": "15m",
                            "seed_generated": True,
                            "latest_candidate_strength": "weak",
                            "cumulative_candidate_strength": "moderate",
                            "blocker_reasons": [],
                        },
                        {
                            "horizon": "4h",
                            "seed_generated": False,
                            "latest_candidate_strength": "insufficient_data",
                            "cumulative_candidate_strength": "insufficient_data",
                            "blocker_reasons": ["candidate_strength_insufficient_data"],
                        },
                    ],
                ),
                ranking=[
                    {
                        "symbol": "BTCUSDT",
                        "strategy": "swing",
                        "horizon": "1h",
                        "selected_candidate_strength": "moderate",
                        "selected_stability_label": "single_horizon_only",
                        "selected_visible_horizons": ["1h"],
                    }
                ],
            ),
        ],
    )

    result = run_candidate_seed_failure_diagnosis_report(path)
    summary = result["summary"]
    markdown = result["markdown"]

    assert summary["metadata"]["total_records"] == 2
    assert summary["seed_layer"]["seed_zero_ratio"] == 0.5
    assert summary["grouping_layer"]["no_valid_symbol_group_frequency"]["count"] == 1
    assert summary["selection_layer"]["top_candidate_presence_rate"] == 1.0
    assert summary["diagnosis"]["primary_bottleneck"] in {
        "seed_generation_failure",
        "visibility_collapse",
        "strength_insufficient",
    }
    assert "Executive Summary" in markdown
    assert "Final Diagnosis" in markdown


def test_empty_file_case(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    loaded = load_shadow_records(path)
    summary = build_candidate_seed_failure_diagnosis_summary(
        records=loaded["records"],
        input_path=path,
        data_quality=loaded["data_quality"],
    )

    assert loaded["records"] == []
    assert summary["metadata"]["total_records"] == 0
    assert summary["seed_layer"]["seed_zero_ratio"] == 0.0
    assert summary["examples"]["seed_count_zero_cases"] == []


def test_malformed_json_line_case(tmp_path: Path) -> None:
    path = tmp_path / "malformed.jsonl"
    _write_jsonl(
        path,
        [
            '{"selection_status": "abstain"',
            _record(generated_at="2026-03-21T00:00:00+00:00"),
        ],
    )

    loaded = load_shadow_records(path)

    assert loaded["data_quality"]["malformed_lines"] == 1
    assert loaded["data_quality"]["valid_records"] == 1
    assert len(loaded["records"]) == 1


def test_missing_field_case(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    _write_jsonl(
        path,
        [
            {
                "generated_at": "2026-03-21T00:00:00+00:00",
                "selection_status": "abstain",
            }
        ],
    )

    summary = run_candidate_seed_failure_diagnosis_report(path)["summary"]

    assert summary["metadata"]["total_records"] == 1
    assert summary["seed_layer"]["seed_zero_ratio"] == 1.0
    assert summary["selection_layer"]["ranking_depth_distribution"][0]["value"] == 0


def test_mixed_valid_invalid_records_case(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(
        path,
        [
            123,
            _record(generated_at="2026-03-21T00:00:00+00:00"),
            "not-json",
            _record(generated_at="2026-03-21T01:00:00+00:00", selection_status="selected", candidate_seed_count=1),
        ],
    )

    loaded = load_shadow_records(path)
    summary = build_candidate_seed_failure_diagnosis_summary(
        records=loaded["records"],
        input_path=path,
        data_quality=loaded["data_quality"],
    )

    assert loaded["data_quality"]["malformed_lines"] == 2
    assert summary["metadata"]["total_records"] == 2
    statuses = {row["value"]: row["count"] for row in summary["selection_layer"]["selection_status_distribution"]}
    assert statuses["abstain"] == 1
    assert statuses["selected"] == 1


def test_horizon_breakdown_validation(tmp_path: Path) -> None:
    path = tmp_path / "horizon.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                candidate_seed_count=1,
                candidate_seed_diagnostics=_seed_diag(
                    horizons_with_seed=["15m"],
                    horizons_without_seed=["1h", "4h"],
                    horizon_diagnostics=[
                        {"horizon": "1h", "blocker_reasons": ["no_valid_symbol_group"]},
                        {"horizon": "4h", "blocker_reasons": ["no_valid_strategy_group"]},
                    ],
                ),
            )
        ],
    )

    summary = run_candidate_seed_failure_diagnosis_report(path)["summary"]
    with_seed = {row["value"]: row["count"] for row in summary["seed_layer"]["horizon_seed_coverage"]["with_seed"]}
    without_seed = {row["value"]: row["count"] for row in summary["seed_layer"]["horizon_seed_coverage"]["without_seed"]}

    assert with_seed["15m"] == 1
    assert without_seed["1h"] == 1
    assert without_seed["4h"] == 1


def test_blocker_aggregation_validation(tmp_path: Path) -> None:
    path = tmp_path / "blockers.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                candidate_seed_diagnostics=_seed_diag(
                    horizon_diagnostics=[
                        {
                            "horizon": "15m",
                            "blocker_reasons": ["no_valid_symbol_group", "no_valid_strategy_group"],
                            "latest_candidate_strength": "insufficient_data",
                            "cumulative_candidate_strength": "insufficient_data",
                        },
                        {
                            "horizon": "1h",
                            "blocker_reasons": ["no_valid_symbol_group"],
                            "latest_candidate_strength": "weak",
                            "cumulative_candidate_strength": "weak",
                        },
                    ],
                ),
            )
        ],
    )

    summary = run_candidate_seed_failure_diagnosis_report(path)["summary"]
    blocker_counts = {
        row["value"]: row["count"]
        for row in summary["grouping_layer"]["blocker_reason_frequency"]
    }
    grouping_breakdown = summary["grouping_layer"]["grouping_related_blocker_breakdown"]
    fifteen_minute = {row["value"]: row["count"] for row in grouping_breakdown["15m"]}

    assert blocker_counts["no_valid_symbol_group"] == 2
    assert blocker_counts["no_valid_strategy_group"] == 1
    assert fifteen_minute["no_valid_symbol_group"] == 1
    assert fifteen_minute["no_valid_strategy_group"] == 1
