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

    # Joined truth must remain unchanged.
    assert row["selected_visible_horizons"] == ["4h"]
    assert row["selected_stability_label"] == "single_horizon_only"

    # Preview context should be attached as optional metadata only.
    assert row["preview_symbol_visible_horizons"] == ["15m", "1h"]
    assert row["preview_symbol_stability_label"] == "multi_horizon_confirmed"
    assert row["preview_symbol_visibility_reason"] == "repeated_visible_candidate_across_horizons"
    assert "symbol_preview_multi_horizon" in row["visibility_reason"]


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

    # Mapper-facing semantics must remain joined-candidate semantics.
    assert seed["selected_visible_horizons"] == ["4h"]
    assert seed["selected_stability_label"] == "single_horizon_only"
