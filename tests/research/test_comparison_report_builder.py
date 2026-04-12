from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.research.comparison_report_builder import build_comparison_report


def _stability_visible_horizons(
    stability_label: str,
    stability_group: str | None,
) -> list[str]:
    if not stability_group or stability_label == "insufficient_data":
        return []
    if stability_label == "multi_horizon_confirmed":
        return ["15m", "1h"]
    if stability_label == "single_horizon_only":
        return ["15m"]
    if stability_label == "unstable":
        return ["15m"]
    return []


def _summary_payload(
    *,
    total_records: int,
    coverage: float,
    top_symbol_15m: str,
    top_strategy_15m: str,
    strategy_group_15m: str,
    stability_label: str,
    stability_group: str | None,
    symbol_edges_15m: int,
) -> dict[str, Any]:
    return {
        "dataset_overview": {
            "total_records": total_records,
            "label_coverage_any_horizon_pct": coverage,
        },
        "top_highlights": {
            "by_horizon": {
                "15m": {
                    "top_symbol": top_symbol_15m,
                    "top_strategy": top_strategy_15m,
                    "best_alignment_state": "aligned",
                    "best_ai_execution_state": "allowed",
                },
                "1h": {
                    "top_symbol": "ETHUSDT",
                    "top_strategy": "trend",
                    "best_alignment_state": "mixed",
                    "best_ai_execution_state": "blocked",
                },
                "4h": {
                    "top_symbol": "BTCUSDT",
                    "top_strategy": "swing",
                    "best_alignment_state": "aligned",
                    "best_ai_execution_state": "allowed",
                },
            }
        },
        "edge_candidates_preview": {
            "by_horizon": {
                "15m": {
                    "candidate_strength": "moderate",
                    "top_strategy": {"group": strategy_group_15m},
                    "top_symbol": {"group": top_symbol_15m},
                    "top_alignment_state": {"group": "aligned"},
                },
                "1h": {
                    "candidate_strength": "weak",
                    "top_strategy": {"group": "trend"},
                    "top_symbol": {"group": "ETHUSDT"},
                    "top_alignment_state": {"group": "mixed"},
                },
                "4h": {
                    "candidate_strength": "insufficient_data",
                    "top_strategy": {"group": "n/a"},
                    "top_symbol": {"group": "n/a"},
                    "top_alignment_state": {"group": "n/a"},
                },
            }
        },
        "edge_stability_preview": {
            "strategy": {
                "group": stability_group,
                "visible_horizons": _stability_visible_horizons(
                    stability_label,
                    stability_group,
                ),
                "stability_label": stability_label,
            },
            "symbol": {
                "group": top_symbol_15m,
                "visible_horizons": ["15m"],
                "stability_label": "single_horizon_only",
            },
            "alignment_state": {
                "group": "aligned",
                "visible_horizons": ["15m"],
                "stability_label": "single_horizon_only",
            },
        },
        "strategy_lab": {
            "edge": {
                "15m": {
                    "by_symbol": {"edge_findings": list(range(symbol_edges_15m))},
                    "by_strategy": {"edge_findings": [1]},
                    "by_alignment_state": {"edge_findings": [1, 2]},
                    "by_ai_execution_state": {"edge_findings": []},
                },
                "1h": {
                    "by_symbol": {"edge_findings": []},
                    "by_strategy": {"edge_findings": []},
                    "by_alignment_state": {"edge_findings": []},
                    "by_ai_execution_state": {"edge_findings": []},
                },
                "4h": {
                    "by_symbol": {"edge_findings": []},
                    "by_strategy": {"edge_findings": []},
                    "by_alignment_state": {"edge_findings": []},
                    "by_ai_execution_state": {"edge_findings": []},
                },
            }
        },
    }


def test_build_comparison_report_happy_path(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=200,
                coverage=72.5,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="multi_horizon_confirmed",
                stability_group="swing",
                symbol_edges_15m=3,
            )
        ),
        encoding="utf-8",
    )
    cumulative_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=1200,
                coverage=81.0,
                top_symbol_15m="ETHUSDT",
                top_strategy_15m="trend",
                strategy_group_15m="trend",
                stability_label="single_horizon_only",
                stability_group="trend",
                symbol_edges_15m=1,
            )
        ),
        encoding="utf-8",
    )

    report = build_comparison_report(latest_path, cumulative_path, output_dir)

    assert report["dataset_overview_comparison"]["latest_total_records"] == 200
    assert report["dataset_overview_comparison"]["cumulative_total_records"] == 1200
    assert report["top_highlights_comparison"]["15m"]["latest_top_symbol"] == "BTCUSDT"
    assert (
        report["top_highlights_comparison"]["15m"]["cumulative_top_symbol"]
        == "ETHUSDT"
    )
    assert (
        report["edge_stability_comparison"]["strategy"]["latest_stability_label"]
        == "multi_horizon_confirmed"
    )
    assert (
        report["edge_stability_comparison"]["strategy"]["cumulative_stability_label"]
        == "single_horizon_only"
    )
    assert (
        report["edge_candidates_preview"]["by_horizon"]["15m"]["top_strategy"]["group"]
        == "swing"
    )
    assert (
        report["edge_stability_preview"]["strategy"]["stability_label"]
        == "multi_horizon_confirmed"
    )
    assert "comparison_summary" in report
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()


def test_build_comparison_report_handles_missing_sections_safely(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(json.dumps({}), encoding="utf-8")
    cumulative_path.write_text(
        json.dumps({"dataset_overview": {"total_records": 10}}),
        encoding="utf-8",
    )

    report = build_comparison_report(latest_path, cumulative_path, output_dir)

    assert report["dataset_overview_comparison"]["latest_total_records"] == 0
    assert report["dataset_overview_comparison"]["cumulative_total_records"] == 10
    assert report["top_highlights_comparison"]["15m"]["latest_top_symbol"] == "n/a"
    assert (
        report["edge_candidates_comparison"]["4h"]["latest_candidate_strength"]
        == "insufficient_data"
    )
    assert (
        report["edge_stability_comparison"]["symbol"]["latest_stability_label"]
        == "insufficient_data"
    )
    assert (
        report["strategy_lab_edge_count_comparison"]["1h"]["latest_symbol_edges"] == 0
    )
    assert "comparison_summary" in report
    assert (
        "latest covers 0 records"
        in report["comparison_summary"]["dataset_size_context"]
    )


def test_build_comparison_report_writes_markdown_file(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=20,
                coverage=40.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="single_horizon_only",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )
    cumulative_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=50,
                coverage=60.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="single_horizon_only",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )

    build_comparison_report(latest_path, cumulative_path, output_dir)

    markdown = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Executive Summary" in markdown
    assert "Drift Notes" in markdown


def test_build_comparison_report_writes_json_file(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(json.dumps({}), encoding="utf-8")
    cumulative_path.write_text(json.dumps({}), encoding="utf-8")

    build_comparison_report(latest_path, cumulative_path, output_dir)

    written = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "generated_at" in written
    assert "drift_notes" in written
    assert "comparison_summary" in written


def test_drift_notes_are_conservative_and_non_recommendation_like(
    tmp_path: Path,
) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=300,
                coverage=75.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="multi_horizon_confirmed",
                stability_group="swing",
                symbol_edges_15m=4,
            )
        ),
        encoding="utf-8",
    )
    cumulative_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=1000,
                coverage=80.0,
                top_symbol_15m="ETHUSDT",
                top_strategy_15m="trend",
                strategy_group_15m="trend",
                stability_label="unstable",
                stability_group="trend",
                symbol_edges_15m=1,
            )
        ),
        encoding="utf-8",
    )

    report = build_comparison_report(latest_path, cumulative_path, output_dir)

    joined_notes = " ".join(report["drift_notes"]).lower()
    joined_summary = " ".join(report["comparison_summary"].values()).lower()
    assert "buy" not in joined_notes
    assert "sell" not in joined_notes
    assert "recommended" not in joined_notes
    assert "opportunity" not in joined_notes
    assert "buy" not in joined_summary
    assert "sell" not in joined_summary
    assert "recommended" not in joined_summary
    assert "opportunity" not in joined_summary
    assert any(
        "stability_strengthened" in note for note in report["drift_notes"]
    )


def test_edge_stability_comparison_detects_difference_between_latest_and_cumulative(
    tmp_path: Path,
) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=150,
                coverage=65.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="multi_horizon_confirmed",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )
    cumulative_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=900,
                coverage=82.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="single_horizon_only",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )

    report = build_comparison_report(latest_path, cumulative_path, output_dir)

    comparison = report["edge_stability_comparison"]["strategy"]
    assert comparison["latest_stability_label"] == "multi_horizon_confirmed"
    assert comparison["cumulative_stability_label"] == "single_horizon_only"
    assert comparison["latest_group"] == "swing"
    assert comparison["cumulative_group"] == "swing"


def test_comparison_summary_identifies_divergence_and_alignment(tmp_path: Path) -> None:
    latest_path = tmp_path / "latest.json"
    cumulative_path = tmp_path / "cumulative.json"
    output_dir = tmp_path / "comparison"

    latest_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=120,
                coverage=70.0,
                top_symbol_15m="BTCUSDT",
                top_strategy_15m="swing",
                strategy_group_15m="swing",
                stability_label="single_horizon_only",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )
    cumulative_path.write_text(
        json.dumps(
            _summary_payload(
                total_records=800,
                coverage=78.0,
                top_symbol_15m="ETHUSDT",
                top_strategy_15m="trend",
                strategy_group_15m="trend",
                stability_label="single_horizon_only",
                stability_group="swing",
                symbol_edges_15m=2,
            )
        ),
        encoding="utf-8",
    )

    report = build_comparison_report(latest_path, cumulative_path, output_dir)

    summary = report["comparison_summary"]
    assert "15m" in summary["key_divergence_summary"]
    assert "1h, 4h" in summary["key_alignment_summary"]
