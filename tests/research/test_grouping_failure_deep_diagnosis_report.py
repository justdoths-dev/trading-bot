from __future__ import annotations

import json
from pathlib import Path

from src.research.grouping_failure_deep_diagnosis_report import (
    build_grouping_failure_deep_diagnosis_summary,
    load_shadow_records,
    render_grouping_failure_deep_diagnosis_markdown,
    run_grouping_failure_deep_diagnosis_report,
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
                        "blocker_reasons": ["no_valid_symbol_group"],
                    },
                    {
                        "horizon": "1h",
                        "blocker_reasons": ["no_valid_strategy_group"],
                    },
                    {
                        "horizon": "4h",
                        "blocker_reasons": [],
                    },
                ],
            ),
            _record(
                generated_at="2026-03-21T01:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "blocker_reasons": ["no_valid_symbol_or_strategy_group"],
                    }
                ],
            ),
        ],
    )

    result = run_grouping_failure_deep_diagnosis_report(path)
    summary = result["summary"]
    markdown = result["markdown"]

    assert summary["metadata"]["total_records"] == 2
    assert summary["metadata"]["total_horizon_rows"] == 4
    assert "Executive Summary" in markdown
    assert "Final Diagnosis" in markdown

    coverage = summary["coverage"]
    assert coverage["grouping_blocked_horizon_rows"] == 3
    assert coverage["grouping_free_horizon_rows"] == 1

    reasons = {
        row["value"]: row["count"]
        for row in summary["reason_distribution"]["overall_grouping_reason_distribution"]
    }
    assert reasons["no_valid_symbol_group"] == 1
    assert reasons["no_valid_strategy_group"] == 1
    assert reasons["no_valid_symbol_or_strategy_group"] == 1


def test_empty_file_case(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    loaded = load_shadow_records(path)
    summary = build_grouping_failure_deep_diagnosis_summary(
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

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]
    assert summary["metadata"]["total_records"] == 1
    assert summary["metadata"]["total_horizon_rows"] == 0
    assert summary["diagnosis"]["primary_issue"] == "no_data"


def test_total_grouping_blockade_case(tmp_path: Path) -> None:
    path = tmp_path / "total_blockade.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {"horizon": "15m", "blocker_reasons": ["no_valid_symbol_group"]},
                    {"horizon": "1h", "blocker_reasons": ["no_valid_strategy_group"]},
                    {"horizon": "4h", "blocker_reasons": ["no_valid_symbol_or_strategy_group"]},
                ],
            )
        ],
    )

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]

    assert summary["coverage"]["grouping_blocked_horizon_rows"] == 3
    assert summary["coverage"]["grouping_free_horizon_rows"] == 0
    assert (
        summary["diagnosis"]["primary_issue"]
        == "grouping_blockade_is_total_and_prevents_all_grouping_free_inputs"
    )
    assert summary["record_level_severity"]["records_all_horizons_grouping_blocked"] == 1


def test_symbol_dominant_case(tmp_path: Path) -> None:
    path = tmp_path / "symbol_dominant.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {"horizon": "15m", "blocker_reasons": ["no_valid_symbol_group"]},
                    {"horizon": "1h", "blocker_reasons": ["no_valid_symbol_group"]},
                    {"horizon": "4h", "blocker_reasons": []},
                ],
            )
        ],
    )

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]
    assert summary["reason_distribution"]["symbol_only_count"] == 2
    assert summary["diagnosis"]["dominant_reason"] == "symbol_only"


def test_strategy_dominant_case(tmp_path: Path) -> None:
    path = tmp_path / "strategy_dominant.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {"horizon": "15m", "blocker_reasons": ["no_valid_strategy_group"]},
                    {"horizon": "1h", "blocker_reasons": ["no_valid_strategy_group"]},
                ],
            )
        ],
    )

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]
    assert summary["reason_distribution"]["strategy_only_count"] == 2
    assert summary["diagnosis"]["dominant_reason"] == "strategy_only"


def test_combo_distribution_case(tmp_path: Path) -> None:
    path = tmp_path / "combo.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {
                        "horizon": "15m",
                        "blocker_reasons": ["no_valid_symbol_group", "no_valid_strategy_group"],
                    }
                ],
            )
        ],
    )

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]
    combos = {
        row["value"]: row["count"]
        for row in summary["reason_distribution"]["overall_grouping_combo_distribution"]
    }
    assert combos["no_valid_strategy_group + no_valid_symbol_group"] == 1


def test_markdown_render_case(tmp_path: Path) -> None:
    path = tmp_path / "markdown.jsonl"
    _write_jsonl(
        path,
        [
            _record(
                generated_at="2026-03-21T00:00:00+00:00",
                horizon_diagnostics=[
                    {"horizon": "4h", "blocker_reasons": ["no_valid_symbol_group"]},
                ],
            )
        ],
    )

    summary = run_grouping_failure_deep_diagnosis_report(path)["summary"]
    markdown = render_grouping_failure_deep_diagnosis_markdown(summary)

    assert "Grouping Failure Deep Diagnosis Report" in markdown
    assert "Overall Grouping Reason Distribution" in markdown