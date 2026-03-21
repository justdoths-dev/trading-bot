from __future__ import annotations

import json
from pathlib import Path

from src.research.strength_blocker_cross_tab_report import (
    build_strength_blocker_cross_tab_summary,
    load_shadow_records,
    render_strength_blocker_cross_tab_markdown,
    run_strength_blocker_cross_tab_report,
)


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row))
            handle.write("\n")


def _record(
    *,
    generated_at: str,
    cumulative_record_count: int = 100,
    horizon_diagnostics: list[dict] | None = None,
) -> dict:
    return {
        "generated_at": generated_at,
        "selection_status": "abstain",
        "candidate_seed_count": 0,
        "cumulative_record_count": cumulative_record_count,
        "candidate_seed_diagnostics": {
            "horizon_diagnostics": horizon_diagnostics or [],
        },
        "abstain_diagnosis": {
            "category": "no_candidates_available",
        },
    }


def test_normal_dataset_case(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["candidate_strength_insufficient_data"],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "weak",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [],
                    },
                ],
            ),
            _record(
                generated_at="2026-03-21T01:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "4h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [
                            "candidate_strength_insufficient_data",
                            "no_valid_symbol_group",
                        ],
                    }
                ],
            ),
        ],
    )

    result = run_strength_blocker_cross_tab_report(path)
    summary = result["summary"]
    markdown = result["markdown"]

    assert summary["metadata"]["total_records"] == 2
    assert summary["metadata"]["total_horizon_rows"] == 3
    assert "Executive Summary" in markdown
    assert "Final Diagnosis" in markdown

    coverage = summary["coverage"]
    assert coverage["grouping_blocked_observation_total"] == 1
    assert coverage["grouping_free_observation_total"] == 2

    strength_status = summary["strength_status"]
    assert strength_status["strength_insufficient_total"] == 2
    assert strength_status["grouping_free_strength_insufficient_total"] == 1
    assert strength_status["grouping_free_ready_total"] == 1


def test_empty_file_case(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    loaded = load_shadow_records(path)
    summary = build_strength_blocker_cross_tab_summary(
        records=loaded["records"],
        input_path=path,
        data_quality=loaded["data_quality"],
    )

    assert loaded["records"] == []
    assert summary["metadata"]["total_records"] == 0
    assert summary["metadata"]["total_horizon_rows"] == 0
    assert summary["diagnosis"]["primary_issue"] == "no_data"


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

    summary = run_strength_blocker_cross_tab_report(path)["summary"]
    assert summary["metadata"]["total_records"] == 1
    assert summary["metadata"]["total_horizon_rows"] == 0
    assert summary["diagnosis"]["primary_issue"] == "no_data"


def test_grouping_dominant_case(tmp_path: Path) -> None:
    path = tmp_path / "grouping_dominant.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["no_valid_symbol_group"],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["no_valid_strategy_group"],
                    },
                    {
                        "horizon": "4h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["no_valid_symbol_or_strategy_group"],
                    },
                ],
            )
        ],
    )

    summary = run_strength_blocker_cross_tab_report(path)["summary"]

    assert summary["coverage"]["grouping_blocked_observation_total"] == 3
    assert summary["coverage"]["grouping_free_observation_total"] == 0
    assert (
        summary["diagnosis"]["primary_issue"]
        == "grouping_blockers_prevent_any_grouping_free_strength_observations"
    )


def test_strength_logic_dominant_case(tmp_path: Path) -> None:
    path = tmp_path / "strength_logic.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["candidate_strength_insufficient_data"],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [],
                    },
                    {
                        "horizon": "4h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [],
                    },
                ],
            )
        ],
    )

    summary = run_strength_blocker_cross_tab_report(path)["summary"]

    assert summary["coverage"]["grouping_free_observation_total"] == 3
    assert summary["strength_status"]["grouping_free_strength_insufficient_total"] == 3
    assert summary["strength_status"]["grouping_free_ready_total"] == 0
    assert (
        summary["diagnosis"]["primary_issue"]
        == "strength_conditions_fail_even_when_grouping_is_not_blocking"
    )


def test_blocker_combo_distribution_case(tmp_path: Path) -> None:
    path = tmp_path / "combos.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [
                            "candidate_strength_insufficient_data",
                            "no_valid_symbol_group",
                        ],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [
                            "candidate_strength_insufficient_data",
                            "no_valid_strategy_group",
                        ],
                    },
                ],
            )
        ],
    )

    summary = run_strength_blocker_cross_tab_report(path)["summary"]
    combos = {
        row["value"]: row["count"]
        for row in summary["blocker_combinations"]["overall_blocker_combo_distribution"]
    }

    assert combos["candidate_strength_insufficient_data + no_valid_symbol_group"] == 1
    assert combos["candidate_strength_insufficient_data + no_valid_strategy_group"] == 1


def test_markdown_render_case(tmp_path: Path) -> None:
    path = tmp_path / "markdown.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "4h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": ["candidate_strength_insufficient_data"],
                    }
                ],
            )
        ],
    )

    summary = run_strength_blocker_cross_tab_report(path)["summary"]
    markdown = render_strength_blocker_cross_tab_markdown(summary)

    assert "Strength Blocker Cross-Tab Report" in markdown
    assert "Overall Blocker Combination Distribution" in markdown