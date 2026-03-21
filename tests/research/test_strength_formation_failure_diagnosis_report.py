from __future__ import annotations

import json
from pathlib import Path

from src.research.strength_formation_failure_diagnosis_report import (
    build_strength_formation_failure_diagnosis_summary,
    load_shadow_records,
    render_strength_formation_failure_diagnosis_markdown,
    run_strength_formation_failure_diagnosis_report,
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
                        "cumulative_candidate_strength": "weak",
                        "blocker_reasons": ["no_valid_symbol_group"],
                    }
                ],
            ),
        ],
    )

    result = run_strength_formation_failure_diagnosis_report(path)
    summary = result["summary"]
    markdown = result["markdown"]

    assert summary["metadata"]["total_records"] == 2
    assert summary["metadata"]["total_horizon_observations"] == 3
    assert "Executive Summary" in markdown
    assert "Strength Overview" in markdown

    latest_dist = {
        row["value"]: row["count"]
        for row in summary["strength_overview"]["latest_candidate_strength_distribution"]
    }
    assert latest_dist["insufficient_data"] == 2
    assert latest_dist["weak"] == 1


def test_empty_file_case(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    loaded = load_shadow_records(path)
    summary = build_strength_formation_failure_diagnosis_summary(
        records=loaded["records"],
        input_path=path,
        data_quality=loaded["data_quality"],
    )

    assert loaded["records"] == []
    assert summary["metadata"]["total_records"] == 0
    assert summary["metadata"]["total_horizon_observations"] == 0
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

    summary = run_strength_formation_failure_diagnosis_report(path)["summary"]
    assert summary["metadata"]["total_records"] == 1
    assert summary["metadata"]["total_horizon_observations"] == 0
    assert summary["diagnosis"]["primary_issue"] == "no_data"


def test_grouping_independent_strength_failure_case(tmp_path: Path) -> None:
    path = tmp_path / "grouping_free.jsonl"
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
                        "blocker_reasons": [],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [],
                    },
                ],
            )
        ],
    )

    summary = run_strength_formation_failure_diagnosis_report(path)["summary"]

    grouping_free_latest = {
        row["value"]: row["count"]
        for row in summary["grouping_independent_strength"][
            "grouping_free_latest_strength_distribution"
        ]
    }
    assert grouping_free_latest["insufficient_data"] == 2
    assert summary["grouping_independent_strength"]["grouping_free_ratio"] == 1.0


def test_transition_distribution_case(tmp_path: Path) -> None:
    path = tmp_path / "transitions.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "latest_candidate_strength": "weak",
                        "cumulative_candidate_strength": "insufficient_data",
                        "blocker_reasons": [],
                    },
                    {
                        "horizon": "1h",
                        "latest_candidate_strength": "insufficient_data",
                        "cumulative_candidate_strength": "weak",
                        "blocker_reasons": [],
                    },
                ],
            )
        ],
    )

    summary = run_strength_formation_failure_diagnosis_report(path)["summary"]
    pair_dist = {
        row["value"]: row["count"]
        for row in summary["transition_analysis"]["latest_vs_cumulative_pair_distribution"]
    }

    assert pair_dist["weak -> insufficient_data"] == 1
    assert pair_dist["insufficient_data -> weak"] == 1
    assert summary["transition_analysis"]["mismatch_rate"] == 1.0


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

    summary = run_strength_formation_failure_diagnosis_report(path)["summary"]
    markdown = render_strength_formation_failure_diagnosis_markdown(summary)

    assert "Strength Formation Failure Diagnosis Report" in markdown
    assert "Final Diagnosis" in markdown