from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

from src.research.diagnostics import analyzer_mapper_bottleneck_diagnosis_report as report_module


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_analyzer_summary(
    *,
    preview_by_horizon: dict[str, dict] | None = None,
    edge_candidate_rows: dict | None = None,
) -> dict:
    return {
        "edge_candidates_preview": {
            "by_horizon": preview_by_horizon or {},
        },
        "edge_candidate_rows": edge_candidate_rows or {
            "row_count": 0,
            "rows": [],
            "diagnostic_row_count": 0,
            "diagnostic_rows": [],
            "empty_reason_summary": {
                "empty_state_category": "no_joined_candidates_evaluated",
                "diagnostic_rejection_reason_counts": {},
                "diagnostic_category_counts": {},
                "identities_blocked_only_by_incompatibility": [],
                "strategies_without_analyzer_compatible_horizons": [],
                "has_only_incompatibility_rejections": False,
                "has_only_weak_or_insufficient_candidates": False,
            },
            "dropped_row_count": 0,
            "dropped_rows": [],
        },
    }


def _preview_row(
    *,
    sample_gate: str,
    quality_gate: str,
    candidate_strength: str,
    visibility_reason: str,
    top_strategy_group: str = "swing",
    top_symbol_group: str = "BTCUSDT",
    top_alignment_group: str = "aligned",
) -> dict:
    def _slot(group: str, strength: str) -> dict:
        return {
            "group": group,
            "sample_gate": "passed" if strength != "insufficient_data" else "failed",
            "quality_gate": (
                "passed"
                if strength in {"moderate", "strong"}
                else "borderline"
                if strength == "weak"
                else "failed"
            ),
            "candidate_strength": strength,
        }

    return {
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": candidate_strength,
        "visibility_reason": visibility_reason,
        "top_strategy": _slot(top_strategy_group, candidate_strength),
        "top_symbol": _slot(top_symbol_group, candidate_strength),
        "top_alignment_state": _slot(top_alignment_group, candidate_strength),
    }


def test_explicit_trade_analysis_path_is_honored_directly(tmp_path: Path, capsys) -> None:
    explicit_path = tmp_path / "explicit.jsonl"
    default_primary = tmp_path / "default_trade_analysis.jsonl"
    default_fallback = tmp_path / "default_trade_analysis_cumulative.jsonl"
    output_dir = tmp_path / "out"

    _write_jsonl(explicit_path, [{}])
    _write_jsonl(default_primary, [{}])
    _write_jsonl(default_fallback, [{}])

    original_primary = report_module.DEFAULT_PRIMARY_INPUT
    original_fallback = report_module.DEFAULT_FALLBACK_INPUT
    report_module.DEFAULT_PRIMARY_INPUT = default_primary
    report_module.DEFAULT_FALLBACK_INPUT = default_fallback
    try:
        report_module.main(
            [
                "--trade-analysis",
                str(explicit_path),
                "--write-latest-copy",
                "--output-dir",
                str(output_dir),
            ]
        )
    finally:
        report_module.DEFAULT_PRIMARY_INPUT = original_primary
        report_module.DEFAULT_FALLBACK_INPUT = original_fallback

    captured = json.loads(capsys.readouterr().out)
    assert captured["trade_analysis_path"] == str(explicit_path.resolve())
    assert captured["trade_analysis_resolution"] == "explicit"


def test_analyzer_distributions_aggregate_correctly_from_synthetic_rows(tmp_path: Path) -> None:
    path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "latest_summary": _build_analyzer_summary(
                    preview_by_horizon={
                        "15m": _preview_row(
                            sample_gate="passed",
                            quality_gate="passed",
                            candidate_strength="moderate",
                            visibility_reason="passed_sample_and_quality_gate",
                        ),
                        "1h": _preview_row(
                            sample_gate="passed",
                            quality_gate="borderline",
                            candidate_strength="weak",
                            visibility_reason="passed_sample_gate_only",
                            top_symbol_group="ETHUSDT",
                        ),
                        "4h": _preview_row(
                            sample_gate="failed",
                            quality_gate="failed",
                            candidate_strength="insufficient_data",
                            visibility_reason="failed_absolute_minimum_gate",
                            top_strategy_group="insufficient_data",
                            top_symbol_group="insufficient_data",
                            top_alignment_group="insufficient_data",
                        ),
                    }
                )
            },
            {
                "research_reports": {
                    "latest": {
                        "summary": _build_analyzer_summary(
                            preview_by_horizon={
                                "15m": _preview_row(
                                    sample_gate="passed",
                                    quality_gate="passed",
                                    candidate_strength="strong",
                                    visibility_reason="passed_sample_and_quality_gate",
                                )
                            }
                        )
                    }
                }
            },
        ],
    )

    report = report_module.build_report(
        input_path=path.resolve(),
        input_resolution="explicit",
    )

    analyzer = report["analyzer_preview"]
    assert analyzer["rows_with_analyzer_output"] == 2
    assert analyzer["by_horizon"]["15m"]["quality_gate_counts"]["passed"] == 2
    assert analyzer["by_horizon"]["1h"]["quality_gate_counts"]["borderline"] == 1
    assert analyzer["by_horizon"]["4h"]["visibility_reason_counts"][
        "failed_absolute_minimum_gate"
    ] == 1
    assert analyzer["failed_absolute_minimum_visibility_count"] == 1
    assert analyzer["survived_sample_gate_but_weak_or_borderline_count"] == 1
    assert analyzer["passed_quality_gate_count"] == 2
    assert analyzer["visible_group_value_counts"]["top_symbol"]["BTCUSDT"] == 2
    assert analyzer["visible_group_value_counts"]["top_symbol"]["ETHUSDT"] == 1


def test_joined_row_empty_and_fallback_blocked_case_is_surfaced(tmp_path: Path) -> None:
    path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "latest_summary": _build_analyzer_summary(
                    edge_candidate_rows={
                        "row_count": 0,
                        "rows": [],
                        "diagnostic_row_count": 1,
                        "diagnostic_rows": [{"rejection_reason": "strategy_horizon_incompatible"}],
                        "empty_reason_summary": {
                            "empty_state_category": "only_incompatibility_rejections",
                            "diagnostic_rejection_reason_counts": {
                                "strategy_horizon_incompatible": 1
                            },
                            "diagnostic_category_counts": {"incompatibility": 1},
                            "identities_blocked_only_by_incompatibility": ["BTCUSDT:scalping"],
                            "strategies_without_analyzer_compatible_horizons": ["scalping"],
                            "has_only_incompatibility_rejections": True,
                            "has_only_weak_or_insufficient_candidates": False,
                        },
                        "dropped_row_count": 0,
                        "dropped_rows": [],
                    }
                ),
                "edge_selection_mapper_payload": {
                    "candidate_seed_diagnostics": {
                        "seed_source": "latest.edge_candidate_rows",
                        "candidate_seed_count": 0,
                        "fallback_blocked": True,
                        "fallback_block_reason": "JOINED_EDGE_CANDIDATE_ROWS_PRESENT_BUT_EMPTY",
                        "dropped_candidate_row_reasons": {},
                        "horizon_diagnostics": [],
                    }
                },
            }
        ],
    )

    report = report_module.build_report(
        input_path=path.resolve(),
        input_resolution="explicit",
    )

    joined = report["joined_row_formation"]
    mapper = report["mapper_seed_handoff"]
    verdict = report["final_verdict"]
    assert joined["empty_state_category_counts"]["only_incompatibility_rejections"] == 1
    assert joined["has_only_incompatibility_rejections_count"] == 1
    assert mapper["fallback_blocked_count"] == 1
    assert mapper["joined_rows_present_but_empty_count"] == 1
    assert verdict["primary_bottleneck_layer"] == "analyzer"


def test_mapper_dropped_candidate_row_reasons_aggregate_correctly(tmp_path: Path) -> None:
    path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "edge_selection_mapper_payload": {
                    "candidate_seed_diagnostics": {
                        "seed_source": "latest.edge_candidate_rows",
                        "candidate_seed_count": 0,
                        "dropped_candidate_row_reasons": {
                            "MISSING_SYMBOL": 2,
                            "MISSING_STRATEGY": 1,
                        },
                        "horizon_diagnostics": [],
                    }
                }
            },
            {
                "edge_selection_mapper_payload": {
                    "candidate_seed_diagnostics": {
                        "seed_source": "latest.edge_candidate_rows",
                        "candidate_seed_count": 1,
                        "dropped_candidate_row_reasons": {
                            "MISSING_SYMBOL": 1,
                        },
                        "horizon_diagnostics": [
                            {"horizon": "4h", "seed_generated_count": 1}
                        ],
                    }
                }
            },
        ],
    )

    report = report_module.build_report(
        input_path=path.resolve(),
        input_resolution="explicit",
    )

    mapper = report["mapper_seed_handoff"]
    assert mapper["dropped_candidate_row_reasons"]["MISSING_SYMBOL"] == 3
    assert mapper["dropped_candidate_row_reasons"]["MISSING_STRATEGY"] == 1
    assert mapper["candidate_seed_count_total"] == 1
    assert mapper["candidate_seed_count_by_horizon"]["4h"] == 1


def test_engine_abstain_categories_and_candidate_status_aggregates_are_counted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "edge_selection_output": {
                    "selection_status": "abstain",
                    "reason_codes": ["NO_ELIGIBLE_CANDIDATES"],
                    "abstain_diagnosis": {
                        "category": "no_eligible_candidates",
                        "eligible_candidate_count": 0,
                        "penalized_candidate_count": 1,
                        "blocked_candidate_count": 2,
                    },
                    "ranking": [
                        {"candidate_status": "penalized"},
                        {"candidate_status": "blocked"},
                        {"candidate_status": "blocked"},
                    ],
                }
            },
            {
                "edge_selection_output": {
                    "selection_status": "selected",
                    "reason_codes": ["CLEAR_TOP_CANDIDATE"],
                    "ranking": [
                        {"candidate_status": "eligible"},
                        {"candidate_status": "eligible"},
                        {"candidate_status": "penalized"},
                    ],
                }
            },
        ],
    )

    report = report_module.build_report(
        input_path=path.resolve(),
        input_resolution="explicit",
    )

    engine = report["engine_outcome"]
    assert engine["selection_status_counts"]["abstain"] == 1
    assert engine["selection_status_counts"]["selected"] == 1
    assert engine["reason_code_counts"]["NO_ELIGIBLE_CANDIDATES"] == 1
    assert engine["reason_code_counts"]["CLEAR_TOP_CANDIDATE"] == 1
    assert engine["abstain_category_counts"]["no_eligible_candidates"] == 1
    assert engine["aggregate_candidate_status_counts"] == {
        "eligible": 2,
        "penalized": 2,
        "blocked": 2,
    }


def test_mapper_payload_detection_does_not_treat_engine_output_as_mapper_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        path,
        [
            {
                "edge_selection_output": {
                    "selection_status": "abstain",
                    "reason_codes": ["NO_CANDIDATES_AVAILABLE"],
                    "candidate_seed_count": 0,
                    "candidate_seed_diagnostics": {
                        "seed_source": "latest.edge_candidate_rows",
                        "candidate_seed_count": 0,
                        "fallback_blocked": True,
                        "fallback_block_reason": "JOINED_EDGE_CANDIDATE_ROWS_PRESENT_BUT_EMPTY",
                    },
                }
            }
        ],
    )

    report = report_module.build_report(
        input_path=path.resolve(),
        input_resolution="explicit",
    )

    assert report["data_quality"]["rows_with_engine_output"] == 1
    assert report["data_quality"]["rows_with_mapper_payload"] == 0
    assert report["data_quality"]["rows_with_candidate_seed_diagnostics"] == 1


def test_wrapper_module_imports_and_runs_correctly() -> None:
    wrapper = importlib.import_module(
        "src.research.analyzer_mapper_bottleneck_diagnosis_report"
    )
    target = importlib.import_module(
        "src.research.diagnostics.analyzer_mapper_bottleneck_diagnosis_report"
    )

    assert wrapper.build_report is target.build_report
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.research.analyzer_mapper_bottleneck_diagnosis_report",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--trade-analysis" in result.stdout
