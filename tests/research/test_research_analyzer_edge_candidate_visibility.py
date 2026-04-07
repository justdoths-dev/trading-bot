from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import edge_selection_input_mapper, research_analyzer


def _candidate(group: str, strength: str = "moderate") -> dict[str, Any]:
    return {
        "group": group,
        "sample_count": 60,
        "labeled_count": 60,
        "coverage_pct": 100.0,
        "median_future_return_pct": 0.45,
        "positive_rate_pct": 58.0,
        "signal_match_rate_pct": 57.0,
        "bias_match_rate_pct": 56.0,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": 57.0,
        "sample_gate": "passed",
        "quality_gate": "passed" if strength in {"moderate", "strong"} else "borderline",
        "candidate_strength": strength,
        "visibility_reason": (
            "passed_sample_and_quality_gate"
            if strength in {"moderate", "strong"}
            else "passed_sample_gate_only"
        ),
        "chosen_metric_summary": "sample=60, median=0.45, positive_rate=58.0",
    }


def _insufficient_candidate() -> dict[str, Any]:
    return {
        "group": "insufficient_data",
        "sample_count": 0,
        "labeled_count": 0,
        "coverage_pct": None,
        "median_future_return_pct": None,
        "positive_rate_pct": None,
        "signal_match_rate_pct": None,
        "bias_match_rate_pct": None,
        "robustness_signal": "n/a",
        "robustness_signal_pct": None,
        "sample_gate": "failed",
        "quality_gate": "failed",
        "candidate_strength": "insufficient_data",
        "visibility_reason": "failed_absolute_minimum_gate",
        "chosen_metric_summary": "insufficient_data",
    }


def _edge_candidates_preview(by_horizon: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "minimum_sample_count": 30,
        "strength_thresholds": {
            "scoring_model": research_analyzer.STRENGTH_SCORING_MODEL,
            "raw_score_bands": research_analyzer.STRENGTH_RAW_SCORE_BANDS,
            "hard_floors": {
                "sample_count": 30,
                "labeled_count_gt": 0,
                "median_future_return_pct_gt": 0,
            },
            "emerging_moderate": {
                "sample_count": 40,
                "median_future_return_pct": 0.18,
                "positive_rate_pct": 50.0,
                "robustness_pct": 46.0,
            },
            "moderate": {
                "sample_count": 50,
                "median_future_return_pct": 0.30,
                "positive_rate_pct": 55.0,
                "robustness_pct": 52.0,
            },
            "strong": {
                "sample_count": 80,
                "median_future_return_pct": 0.50,
                "positive_rate_pct": 58.0,
                "robustness_pct": 55.0,
            },
            "classification_thresholds": {
                "moderate_min_aggregate_score": research_analyzer.MODERATE_MIN_AGGREGATE_SCORE,
                "moderate_with_one_supporting_deficit_min_score": research_analyzer.MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE,
                "moderate_with_two_supporting_deficits_min_score": research_analyzer.MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE,
                "moderate_with_three_supporting_deficits_min_score": research_analyzer.MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE,
                "strong_min_aggregate_score": research_analyzer.STRONG_MIN_AGGREGATE_SCORE,
            },
            "recovery_guards": {
                "positive_rate_minimum_floor_pct": research_analyzer.POSITIVE_RATE_MINIMUM_FLOOR_PCT,
                "three_supporting_deficits_min_positive_rate_pct": research_analyzer.THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT,
                "three_supporting_deficits_min_robustness_pct": research_analyzer.THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT,
            },
            "component_weights": research_analyzer.STRENGTH_COMPONENT_WEIGHTS,
        },
        "by_horizon": by_horizon,
    }


def _joined_research_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    future_label_4h: str = "up",
    future_return_4h: float = 0.42,
    bias: str = "bullish",
    rule_signal: str = "long",
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "selected_strategy": strategy,
        "future_label_4h": future_label_4h,
        "future_return_4h": future_return_4h,
        "bias": bias,
        "rule_signal": rule_signal,
    }


def _joined_candidate_dataset() -> list[dict[str, Any]]:
    dataset = [
        _joined_research_row(future_label_4h="up", future_return_4h=0.42)
        for _ in range(34)
    ]
    dataset.extend(
        _joined_research_row(
            future_label_4h="down",
            future_return_4h=0.18,
            bias="bearish",
            rule_signal="short",
        )
        for _ in range(26)
    )
    return dataset


def test_edge_candidate_rows_preserve_existing_four_hour_joined_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        research_analyzer,
        "build_dataset",
        lambda path: _joined_candidate_dataset(),
    )

    result = research_analyzer._build_edge_candidate_rows(Path("ignored.jsonl"))

    assert result["row_count"] == 1
    row = result["rows"][0]
    assert row["symbol"] == "BTCUSDT"
    assert row["strategy"] == "swing"
    assert row["horizon"] == "4h"
    assert row["selected_candidate_strength"] == "moderate"
    assert row["selected_visible_horizons"] == ["4h"]
    assert row["selected_stability_label"] == "single_horizon_only"

    assert result["diagnostic_row_count"] == 2
    assert {
        diagnostic_row["rejection_reason"] for diagnostic_row in result["diagnostic_rows"]
    } == {"strategy_horizon_incompatible", "no_labeled_rows_for_horizon"}

    assert result["empty_reason_summary"]["has_eligible_rows"] is True
    assert result["empty_reason_summary"]["empty_state_category"] == "has_eligible_rows"
    assert result["empty_reason_summary"]["dominant_diagnostic_category"] in {
        "incompatibility",
        "insufficient_data",
    }


def test_edge_candidate_rows_attach_preview_metadata_without_promoting_joined_stability(
    monkeypatch,
) -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_symbol": _candidate("BTCUSDT"),
                "top_strategy": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_symbol": _candidate("BTCUSDT"),
                "top_strategy": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_symbol": _insufficient_candidate(),
                "top_strategy": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )
    stability = research_analyzer._build_edge_stability_preview(preview)

    monkeypatch.setattr(
        research_analyzer,
        "build_dataset",
        lambda path: _joined_candidate_dataset(),
    )

    result = research_analyzer._build_edge_candidate_rows(
        Path("ignored.jsonl"),
        edge_candidates_preview=preview,
        edge_stability_preview=stability,
    )

    row = result["rows"][0]
    assert row["horizon"] == "4h"
    assert row["selected_candidate_strength"] == "moderate"

    assert row["selected_visible_horizons"] == ["4h"]
    assert row["selected_stability_label"] == "single_horizon_only"

    assert row["preview_symbol_visible_horizons"] == ["15m", "1h"]
    assert row["preview_symbol_stability_label"] == "multi_horizon_confirmed"
    assert row["preview_symbol_visibility_reason"] == "repeated_visible_candidate_across_horizons"

    assert row["visibility_reason"].startswith("passed_sample_and_quality_gate")
    assert (
        "joined_selected_but_raw_preview_broader_than_joined" in row["visibility_reason"]
        or "joined_selected_but_compatibility_preview_broader_than_joined"
        in row["visibility_reason"]
    )


def test_edge_candidate_rows_remain_mapper_compatible_after_preview_metadata_enrichment(
    monkeypatch,
) -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_symbol": _candidate("BTCUSDT"),
                "top_strategy": _candidate("swing"),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_symbol": _candidate("BTCUSDT"),
                "top_strategy": _candidate("swing"),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_symbol": _insufficient_candidate(),
                "top_strategy": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )
    stability = research_analyzer._build_edge_stability_preview(preview)

    monkeypatch.setattr(
        research_analyzer,
        "build_dataset",
        lambda path: _joined_candidate_dataset(),
    )

    result = research_analyzer._build_edge_candidate_rows(
        Path("ignored.jsonl"),
        edge_candidates_preview=preview,
        edge_stability_preview=stability,
    )

    seed, drop_reasons = edge_selection_input_mapper._normalize_candidate_row_seed(
        result["rows"][0]
    )

    assert drop_reasons == []
    assert seed is not None
    assert seed["symbol"] == "BTCUSDT"
    assert seed["strategy"] == "swing"
    assert seed["horizon"] == "4h"

    assert seed["selected_visible_horizons"] == ["4h"]
    assert seed["selected_stability_label"] == "single_horizon_only"


def test_edge_candidate_rows_expose_weak_joined_candidates_only_in_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        research_analyzer,
        "build_dataset",
        lambda path: [_joined_research_row()],
    )

    def fake_evaluate(
        *,
        symbol: str,
        strategy: str,
        horizon: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if horizon == "15m":
            return {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "strategy_horizon_compatible": False,
                "status": "rejected",
                "rejection_reason": "strategy_horizon_incompatible",
                "rejection_reasons": ["strategy_horizon_incompatible"],
                "sample_gate": "not_applicable",
                "quality_gate": "not_applicable",
                "candidate_strength": "incompatible",
                "candidate_strength_diagnostics": None,
                "metrics": {},
                "aggregate_score": None,
                "visibility_reason": "strategy_horizon_incompatible",
                "chosen_metric_summary": "strategy_horizon_incompatible",
                "strategy_horizon_compatibility_detail": (
                    research_analyzer._build_strategy_horizon_compatibility_detail(
                        strategy=strategy,
                        horizon=horizon,
                    )
                ),
            }
        if horizon == "1h":
            return {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "strategy_horizon_compatible": True,
                "status": "rejected",
                "rejection_reason": "no_labeled_rows_for_horizon",
                "rejection_reasons": ["no_labeled_rows_for_horizon"],
                "sample_gate": "failed",
                "quality_gate": "failed",
                "candidate_strength": "insufficient_data",
                "candidate_strength_diagnostics": None,
                "metrics": {
                    "sample_count": 18,
                    "labeled_count": 0,
                    "coverage_pct": 0.0,
                    "median_future_return_pct": None,
                },
                "aggregate_score": None,
                "visibility_reason": "no_labeled_rows_for_horizon",
                "chosen_metric_summary": "no_labeled_rows_for_horizon",
            }
        return {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "strategy_horizon_compatible": True,
            "status": "rejected",
            "rejection_reason": "candidate_strength_weak",
            "rejection_reasons": [
                "candidate_strength_weak",
                "aggregate_below_moderate_threshold",
            ],
            "sample_gate": "passed",
            "quality_gate": "failed",
            "candidate_strength": "weak",
            "candidate_strength_diagnostics": {
                "classification_reason": "aggregate_below_moderate_threshold",
            },
            "metrics": {
                "sample_count": 42,
                "labeled_count": 42,
                "coverage_pct": 100.0,
                "median_future_return_pct": 0.16,
                "positive_rate_pct": 51.0,
                "signal_match_rate_pct": 49.0,
            },
            "aggregate_score": 58.0,
            "visibility_reason": "candidate_strength_weak",
            "chosen_metric_summary": "sample=42, median=0.16, positive_rate=51.0",
        }

    monkeypatch.setattr(
        research_analyzer,
        "_evaluate_joined_edge_candidate_horizon",
        fake_evaluate,
    )

    result = research_analyzer._build_edge_candidate_rows(Path("ignored.jsonl"))

    assert result["row_count"] == 0
    assert result["rows"] == []
    assert result["diagnostic_row_count"] == 3
    assert result["empty_reason_summary"]["diagnostic_rejection_reason_counts"] == {
        "candidate_strength_weak": 1,
        "no_labeled_rows_for_horizon": 1,
        "strategy_horizon_incompatible": 1,
    }
    assert result["empty_reason_summary"]["diagnostic_category_counts"] == {
        "quality_rejected": 1,
        "insufficient_data": 1,
        "incompatibility": 1,
    }
    assert result["empty_reason_summary"]["empty_state_category"] == (
        "mixed_rejections_without_eligible_rows"
    )

    weak_rows = [
        row
        for row in result["diagnostic_rows"]
        if row["rejection_reason"] == "candidate_strength_weak"
    ]
    assert len(weak_rows) == 1
    assert weak_rows[0]["candidate_strength"] == "weak"
    assert weak_rows[0]["classification_reason"] == "aggregate_below_moderate_threshold"
    assert weak_rows[0]["aggregate_score"] == 58.0
    assert weak_rows[0]["horizon"] == "4h"
    assert weak_rows[0]["diagnostic_category"] == "quality_rejected"


def test_edge_candidate_rows_make_scalping_incompatibility_explicit(monkeypatch) -> None:
    scalping_rows = [
        _joined_research_row(strategy="scalping", future_label_4h="up", future_return_4h=0.25)
        for _ in range(5)
    ]

    monkeypatch.setattr(
        research_analyzer,
        "build_dataset",
        lambda path: scalping_rows,
    )

    result = research_analyzer._build_edge_candidate_rows(Path("ignored.jsonl"))

    assert result["row_count"] == 0
    assert result["diagnostic_row_count"] == len(research_analyzer.HORIZONS)
    assert {
        row["rejection_reason"] for row in result["diagnostic_rows"]
    } == {"strategy_horizon_incompatible"}
    assert {
        row["diagnostic_category"] for row in result["diagnostic_rows"]
    } == {"incompatibility"}
    assert result["empty_reason_summary"][
        "strategies_without_analyzer_compatible_horizons"
    ] == ["scalping"]
    assert result["empty_reason_summary"][
        "identities_blocked_only_by_incompatibility"
    ] == ["BTCUSDT:scalping"]
    assert result["empty_reason_summary"]["empty_state_category"] == (
        "only_incompatibility_rejections"
    )
    assert result["empty_reason_summary"]["has_only_incompatibility_rejections"] is True
    assert result["empty_reason_summary"]["dominant_diagnostic_category"] == "incompatibility"
    assert "scalping" in result["empty_reason_summary"]["note"]
    assert "15m" in result["empty_reason_summary"]["note"]
    assert "4h" in result["empty_reason_summary"]["note"]

    for row in result["diagnostic_rows"]:
        detail = row["strategy_horizon_compatibility_detail"]
        assert detail is not None
        assert detail["configured_horizons"] == ["1m", "5m"]
        assert detail["analyzer_compatible_horizons"] == []
        assert detail["has_analyzer_compatible_horizon"] is False

    markdown_lines = research_analyzer._markdown_edge_candidate_rows(result)
    markdown_text = "\n".join(markdown_lines)
    assert "No engine-facing eligible joined candidate rows available." in markdown_text
    assert "strategies_without_analyzer_compatible_horizons: ['scalping']" in markdown_text


def test_run_research_analyzer_exposes_diagnostic_rows_in_final_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        research_analyzer,
        "load_jsonl_records",
        lambda input_path: ([{"symbol": "BTCUSDT"}], {"invalid_records": 0, "valid_records": 1}),
    )
    monkeypatch.setattr(
        research_analyzer,
        "calculate_research_metrics",
        lambda records: {"dataset_overview": {"total_records": len(records)}},
    )
    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda input_path: {"dataset_rows": 1, "ranking": {}, "performance": {}, "comparison": {}, "edge": {}, "segment": {}},
    )
    monkeypatch.setattr(
        research_analyzer,
        "_build_edge_candidates_preview",
        lambda strategy_lab: {"by_horizon": {}, "minimum_sample_count": 30, "strength_thresholds": {}},
    )
    monkeypatch.setattr(
        research_analyzer,
        "_build_edge_stability_preview",
        lambda edge_candidates_preview: {},
    )
    monkeypatch.setattr(
        research_analyzer,
        "_build_edge_candidate_rows",
        lambda input_path, edge_candidates_preview=None, edge_stability_preview=None: {
            "row_count": 0,
            "rows": [],
            "diagnostic_row_count": 1,
            "diagnostic_rows": [
                {
                    "symbol": "BTCUSDT",
                    "strategy": "scalping",
                    "horizon": "15m",
                    "diagnostic_category": "incompatibility",
                    "rejection_reason": "strategy_horizon_incompatible",
                }
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
                "identities_blocked_only_by_incompatibility": ["BTCUSDT:scalping"],
                "strategies_without_analyzer_compatible_horizons": ["scalping"],
                "empty_state_category": "only_incompatibility_rejections",
                "has_only_incompatibility_rejections": True,
                "has_only_weak_or_insufficient_candidates": False,
                "note": "No engine-facing eligible joined candidate rows were produced.",
            },
            "dropped_row_count": 0,
            "dropped_rows": [],
            "identity_horizon_evaluations": [],
        },
    )

    written_payloads: list[dict[str, Any]] = []

    def fake_write_summary_files(metrics: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
        written_payloads.append(metrics)
        return output_dir / "summary.json", output_dir / "summary.md"

    monkeypatch.setattr(research_analyzer, "write_summary_files", fake_write_summary_files)

    result = research_analyzer.run_research_analyzer(
        input_path=Path("ignored.jsonl"),
        output_dir=tmp_path,
    )

    assert result["edge_candidate_rows"]["diagnostic_row_count"] == 1
    assert result["edge_candidate_rows"]["diagnostic_rows"][0]["diagnostic_category"] == (
        "incompatibility"
    )
    assert result["edge_candidate_rows"]["empty_reason_summary"]["empty_state_category"] == (
        "only_incompatibility_rejections"
    )
    assert written_payloads
    assert written_payloads[0]["edge_candidate_rows"]["diagnostic_row_count"] == 1