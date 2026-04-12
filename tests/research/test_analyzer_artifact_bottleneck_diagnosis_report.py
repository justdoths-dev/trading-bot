from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

from src.research.diagnostics import (
    analyzer_artifact_bottleneck_diagnosis_report as report_module,
)


def _write_summary(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _preview_block(*, horizons: list[str] | None = None) -> dict:
    horizons = horizons or ["15m"]
    return {
        "by_horizon": {
            horizon: {
                "candidate_strength": "moderate",
                "sample_gate": "passed",
                "quality_gate": "passed",
                "visibility_reason": "passed_sample_and_quality_gate",
                "top_symbol": {
                    "group": "BTCUSDT",
                    "candidate_strength": "moderate",
                    "quality_gate": "passed",
                },
                "top_strategy": {
                    "group": "swing",
                    "candidate_strength": "moderate",
                    "quality_gate": "passed",
                },
            }
            for horizon in horizons
        }
    }


def _identity_evaluation(
    *,
    strategy_compatible_horizons: list[str],
    raw_union_horizons: list[str],
    compatibility_union_horizons: list[str],
    nested_strategy_compatible_horizons: list[str] | None = None,
) -> dict:
    compatibility_visibility = {
        "compatibility_filtered_category_union_horizons": (
            compatibility_union_horizons
        ),
        "compatibility_filtered_category_overlap_horizons": (
            compatibility_union_horizons
        ),
    }
    if nested_strategy_compatible_horizons is not None:
        compatibility_visibility["strategy_compatible_horizons"] = (
            nested_strategy_compatible_horizons
        )

    return {
        "identity_key": "BTCUSDT:swing",
        "symbol": "BTCUSDT",
        "strategy": "swing",
        "strategy_compatible_horizons": strategy_compatible_horizons,
        "raw_preview_visibility": {
            "raw_category_union_horizons": raw_union_horizons,
            "raw_category_overlap_horizons": raw_union_horizons,
        },
        "compatibility_filtered_preview_visibility": compatibility_visibility,
        "actual_joined_eligible_horizons": compatibility_union_horizons,
        "horizon_evaluations": {},
    }


def test_missing_summary_file_is_reported_clearly(tmp_path: Path, capsys) -> None:
    missing_path = tmp_path / "missing-summary.json"

    report_module.main(["--summary-path", str(missing_path)])

    captured = json.loads(capsys.readouterr().out)
    report = report_module.build_report(
        summary_path=missing_path.resolve(),
        summary_path_resolution="explicit",
    )

    assert captured["summary_file_exists"] is False
    assert captured["load_error"] == "summary_path_missing"
    assert captured["verdict_category"] == "artifact_missing"
    assert report["artifact_presence"]["load_error"] == "summary_path_missing"
    assert (
        report["final_verdict"]["artifact_sufficient_for_current_snapshot_diagnosis"]
        is False
    )


def test_summary_without_preview_or_joined_blocks_is_flagged(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, {})

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["analyzer_preview"]["preview_block_exists"] is False
    assert report["joined_row_artifact"]["joined_row_block_exists"] is False
    assert report["final_verdict"]["verdict_category"] == (
        "artifact_present_but_expected_blocks_missing"
    )
    assert report["final_verdict"]["expected_missing_blocks"] == [
        "edge_candidates_preview",
        "edge_candidate_rows",
    ]


def test_preview_present_but_joined_row_block_missing(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, {"edge_candidates_preview": _preview_block()})

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["analyzer_preview"]["preview_block_exists"] is True
    assert report["joined_row_artifact"]["joined_row_block_exists"] is False
    assert report["final_verdict"]["verdict_category"] == (
        "preview_present_but_no_joined_row_block"
    )
    assert (
        report["final_verdict"]["artifact_sufficient_for_current_snapshot_diagnosis"]
        is False
    )


def test_joined_row_block_with_empty_reason_summary_surfaces_no_eligible_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(),
            "edge_candidate_rows": {
                "row_count": 0,
                "rows": [],
                "diagnostic_row_count": 2,
                "diagnostic_rows": [
                    {"rejection_reason": "strategy_horizon_incompatible"},
                    {"rejection_reason": "strategy_horizon_incompatible"},
                ],
                "empty_reason_summary": {
                    "has_eligible_rows": False,
                    "diagnostic_row_count": 2,
                    "diagnostic_rejection_reason_counts": {
                        "strategy_horizon_incompatible": 2
                    },
                    "diagnostic_category_counts": {"incompatibility": 2},
                    "dominant_rejection_reason": "strategy_horizon_incompatible",
                    "dominant_diagnostic_category": "incompatibility",
                    "identity_count": 1,
                    "identities_with_eligible_rows": 0,
                    "identities_without_eligible_rows": 1,
                    "identities_blocked_only_by_incompatibility": ["BTCUSDT:swing"],
                    "strategies_without_analyzer_compatible_horizons": ["swing"],
                    "empty_state_category": "only_incompatibility_rejections",
                    "has_only_incompatibility_rejections": True,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 0,
                "dropped_rows": [],
                "identity_horizon_evaluations": [
                    _identity_evaluation(
                        strategy_compatible_horizons=[],
                        raw_union_horizons=["15m"],
                        compatibility_union_horizons=[],
                        nested_strategy_compatible_horizons=[],
                    )
                ],
            },
        },
    )

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["joined_row_artifact"]["row_count"] == 0
    assert report["joined_row_artifact"]["diagnostic_row_count"] == 2
    assert report["joined_row_artifact"]["empty_reason_summary"][
        "empty_state_category"
    ] == "only_incompatibility_rejections"
    assert report["compatibility_diagnostics"]["conservative_signals"] == [
        "raw_preview_visible_but_compatibility_filtered_invisible",
        "all_horizons_incompatible",
    ]
    assert report["final_verdict"]["verdict_category"] == (
        "joined_row_block_present_but_no_eligible_rows"
    )
    assert (
        report["final_verdict"]["artifact_sufficient_for_current_snapshot_diagnosis"]
        is True
    )


def test_joined_rows_and_compatibility_information_are_summarized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(),
            "edge_candidate_rows": {
                "row_count": 2,
                "rows": [
                    {"symbol": "BTCUSDT", "strategy": "swing", "horizon": "15m"},
                    {"symbol": "BTCUSDT", "strategy": "swing", "horizon": "1h"},
                ],
                "diagnostic_row_count": 1,
                "diagnostic_rows": [
                    {"rejection_reason": "failed_absolute_minimum_gate"}
                ],
                "empty_reason_summary": {
                    "has_eligible_rows": True,
                    "diagnostic_row_count": 1,
                    "diagnostic_rejection_reason_counts": {
                        "failed_absolute_minimum_gate": 1
                    },
                    "diagnostic_category_counts": {"insufficient_data": 1},
                    "dominant_rejection_reason": "failed_absolute_minimum_gate",
                    "dominant_diagnostic_category": "insufficient_data",
                    "identity_count": 1,
                    "identities_with_eligible_rows": 1,
                    "identities_without_eligible_rows": 0,
                    "identities_blocked_only_by_incompatibility": [],
                    "strategies_without_analyzer_compatible_horizons": [],
                    "empty_state_category": "has_eligible_rows",
                    "has_only_incompatibility_rejections": False,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 1,
                "dropped_rows": [{"drop_reason": "MISSING_SYMBOL"}],
                "identity_horizon_evaluations": [
                    _identity_evaluation(
                        strategy_compatible_horizons=["15m", "1h"],
                        raw_union_horizons=["15m", "1h"],
                        compatibility_union_horizons=["15m", "1h"],
                        nested_strategy_compatible_horizons=["15m", "1h"],
                    )
                ],
            },
        },
    )

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["joined_row_artifact"]["row_count"] == 2
    assert report["joined_row_artifact"]["dropped_row_count"] == 1
    assert (
        report["compatibility_diagnostics"][
            "identities_with_compatibility_filtered_visibility_count"
        ]
        == 1
    )
    assert report["compatibility_diagnostics"]["conservative_signals"] == [
        "compatibility_filtered_visibility_present"
    ]
    assert report["final_verdict"]["verdict_category"] == (
        "joined_rows_present_with_compatible_candidates"
    )


def test_root_level_strategy_compatible_horizons_are_used_when_nested_field_is_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(horizons=["15m", "1d"]),
            "edge_candidate_rows": {
                "row_count": 0,
                "rows": [],
                "diagnostic_row_count": 1,
                "diagnostic_rows": [
                    {"rejection_reason": "strategy_horizon_incompatible"}
                ],
                "empty_reason_summary": {
                    "has_eligible_rows": False,
                    "diagnostic_row_count": 1,
                    "diagnostic_rejection_reason_counts": {
                        "strategy_horizon_incompatible": 1
                    },
                    "diagnostic_category_counts": {"incompatibility": 1},
                    "dominant_rejection_reason": "strategy_horizon_incompatible",
                    "dominant_diagnostic_category": "incompatibility",
                    "identity_count": 1,
                    "identities_with_eligible_rows": 0,
                    "identities_without_eligible_rows": 1,
                    "identities_blocked_only_by_incompatibility": ["BTCUSDT:swing"],
                    "strategies_without_analyzer_compatible_horizons": ["swing"],
                    "empty_state_category": "only_incompatibility_rejections",
                    "has_only_incompatibility_rejections": True,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 0,
                "dropped_rows": [],
                "identity_horizon_evaluations": [
                    _identity_evaluation(
                        strategy_compatible_horizons=[],
                        raw_union_horizons=["15m", "1d"],
                        compatibility_union_horizons=[],
                        nested_strategy_compatible_horizons=None,
                    )
                ],
            },
        },
    )

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["analyzer_preview"]["preview_horizons_present"] == ["15m", "1d"]
    assert (
        report["compatibility_diagnostics"][
            "identities_with_no_analyzer_compatible_horizons_count"
        ]
        == 1
    )
    assert report["compatibility_diagnostics"]["all_identities_incompatible"] is True


def test_empty_reason_summary_counts_as_compatibility_evidence_without_identity_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(),
            "edge_candidate_rows": {
                "row_count": 0,
                "rows": [],
                "diagnostic_row_count": 1,
                "diagnostic_rows": [
                    {"rejection_reason": "strategy_horizon_incompatible"}
                ],
                "empty_reason_summary": {
                    "has_eligible_rows": False,
                    "diagnostic_row_count": 1,
                    "diagnostic_rejection_reason_counts": {
                        "strategy_horizon_incompatible": 1
                    },
                    "diagnostic_category_counts": {"incompatibility": 1},
                    "dominant_rejection_reason": "strategy_horizon_incompatible",
                    "dominant_diagnostic_category": "incompatibility",
                    "identity_count": 1,
                    "identities_with_eligible_rows": 0,
                    "identities_without_eligible_rows": 1,
                    "identities_blocked_only_by_incompatibility": ["BTCUSDT:swing"],
                    "strategies_without_analyzer_compatible_horizons": ["swing"],
                    "empty_state_category": "only_incompatibility_rejections",
                    "has_only_incompatibility_rejections": True,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 0,
                "dropped_rows": [],
            },
        },
    )

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    assert report["compatibility_diagnostics"]["compatibility_diagnostics_present"] is True
    assert (
        report["compatibility_diagnostics"][
            "identities_blocked_only_by_incompatibility_count"
        ]
        == 1
    )
    assert (
        report["compatibility_diagnostics"][
            "strategies_without_analyzer_compatible_horizons_count"
        ]
        == 1
    )


def test_narrow_verdict_does_not_overclaim_downstream_mapper_or_engine(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(),
            "edge_candidate_rows": {
                "row_count": 0,
                "rows": [],
                "diagnostic_row_count": 0,
                "diagnostic_rows": [],
                "empty_reason_summary": {
                    "has_eligible_rows": False,
                    "diagnostic_row_count": 0,
                    "diagnostic_rejection_reason_counts": {},
                    "diagnostic_category_counts": {},
                    "dominant_rejection_reason": None,
                    "dominant_diagnostic_category": None,
                    "identity_count": 0,
                    "identities_with_eligible_rows": 0,
                    "identities_without_eligible_rows": 0,
                    "identities_blocked_only_by_incompatibility": [],
                    "strategies_without_analyzer_compatible_horizons": [],
                    "empty_state_category": "no_joined_candidates_evaluated",
                    "has_only_incompatibility_rejections": False,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 0,
                "dropped_rows": [],
                "identity_horizon_evaluations": [],
            },
        },
    )

    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )

    verdict = report["final_verdict"]
    assert verdict["scope"] == "analyzer_artifact_only"
    assert verdict["verdict_category"] == "joined_row_block_present_but_no_eligible_rows"
    assert any(
        "does not assess downstream mapper or engine bottlenecks" in note
        for note in verdict["uncertainty_notes"]
    )


def test_write_latest_copy_writes_json_and_markdown_reports(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "summary.json"
    output_dir = tmp_path / "out"
    _write_summary(
        path,
        {
            "edge_candidates_preview": _preview_block(),
            "edge_candidate_rows": {
                "row_count": 1,
                "rows": [{"symbol": "BTCUSDT", "strategy": "swing", "horizon": "15m"}],
                "diagnostic_row_count": 0,
                "diagnostic_rows": [],
                "empty_reason_summary": {
                    "has_eligible_rows": True,
                    "diagnostic_row_count": 0,
                    "diagnostic_rejection_reason_counts": {},
                    "diagnostic_category_counts": {},
                    "dominant_rejection_reason": None,
                    "dominant_diagnostic_category": None,
                    "identity_count": 1,
                    "identities_with_eligible_rows": 1,
                    "identities_without_eligible_rows": 0,
                    "identities_blocked_only_by_incompatibility": [],
                    "strategies_without_analyzer_compatible_horizons": [],
                    "empty_state_category": "has_eligible_rows",
                    "has_only_incompatibility_rejections": False,
                    "has_only_weak_or_insufficient_candidates": False,
                },
                "dropped_row_count": 0,
                "dropped_rows": [],
                "identity_horizon_evaluations": [],
            },
        },
    )

    report_module.main(
        [
            "--summary-path",
            str(path),
            "--write-latest-copy",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = json.loads(capsys.readouterr().out)
    assert Path(captured["written_paths"]["json_report"]).exists()
    assert Path(captured["written_paths"]["markdown_report"]).exists()


def test_wrapper_module_imports_and_runs_correctly() -> None:
    wrapper = importlib.import_module(
        "src.research.analyzer_artifact_bottleneck_diagnosis_report"
    )
    target = importlib.import_module(
        "src.research.diagnostics.analyzer_artifact_bottleneck_diagnosis_report"
    )

    assert wrapper.build_report is target.build_report
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.research.analyzer_artifact_bottleneck_diagnosis_report",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--summary-path" in result.stdout