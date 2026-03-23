from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.research import research_analyzer


def _stub_strategy_lab_metrics(dataset_rows: int = 0) -> dict[str, Any]:
    return {
        "dataset_rows": dataset_rows,
        "performance": {},
        "comparison": {},
        "ranking": {},
        "edge": {},
        "segment": {},
    }


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


def _ranked_group_entry(
    group: str,
    *,
    sample_count: float,
    labeled_count: float | None,
    coverage_pct: float | None,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_pct: float | None = 57.0,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "sample_count": sample_count,
        "median_future_return_pct": median_future_return_pct,
        "signal_match_rate_pct": robustness_pct,
    }
    if labeled_count is not None:
        metrics["labeled_count"] = labeled_count
    if coverage_pct is not None:
        metrics["coverage_pct"] = coverage_pct
    if positive_rate_pct is not None:
        metrics["positive_rate_pct"] = positive_rate_pct

    return {
        "group": group,
        "metrics": metrics,
    }


def _ranked_group(
    group: str,
    *,
    sample_count: float,
    labeled_count: float | None,
    coverage_pct: float | None,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_pct: float | None = 57.0,
) -> dict[str, Any]:
    return {
        "ranked_groups": [
            _ranked_group_entry(
                group,
                sample_count=sample_count,
                labeled_count=labeled_count,
                coverage_pct=coverage_pct,
                median_future_return_pct=median_future_return_pct,
                positive_rate_pct=positive_rate_pct,
                robustness_pct=robustness_pct,
            )
        ]
    }


def test_run_research_analyzer_handles_valid_records(
    monkeypatch,
    tmp_path: Path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    second_record = deepcopy(valid_research_record)
    second_record["symbol"] = "ETHUSDT"
    second_record["selected_strategy"] = "mean_reversion"
    second_record["future_label_15m"] = "down"
    second_record["future_return_15m"] = -0.4

    input_path = write_jsonl([valid_research_record, second_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: _stub_strategy_lab_metrics(dataset_rows=2),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["dataset_overview"]["total_records"] == 2
    assert result["schema_validation"]["valid_records"] == 2
    assert result["schema_validation"]["invalid_records"] == 0
    assert result["strategy_lab"]["dataset_rows"] == 2
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()


def test_run_research_analyzer_skips_invalid_records_without_crashing(
    monkeypatch,
    tmp_path: Path,
    valid_research_record: dict[str, Any],
    write_jsonl,
) -> None:
    invalid_record = deepcopy(valid_research_record)
    invalid_record["risk"] = "invalid"

    input_path = write_jsonl([valid_research_record, invalid_record])
    output_dir = tmp_path / "reports"

    monkeypatch.setattr(
        research_analyzer,
        "_build_strategy_lab_metrics",
        lambda _input_path: (_ for _ in ()).throw(
            AssertionError("strategy lab should be skipped")
        ),
    )

    result = research_analyzer.run_research_analyzer(input_path, output_dir)

    assert result["dataset_overview"]["total_records"] == 1
    assert result["schema_validation"]["total_records"] == 2
    assert result["schema_validation"]["valid_records"] == 1
    assert result["schema_validation"]["invalid_records"] == 1
    assert len(result["schema_validation"]["invalid_examples"]) == 1
    assert result["strategy_lab"]["dataset_rows"] == 0


def test_edge_stability_preview_returns_insufficient_data_without_visible_candidates() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "insufficient_data"
    assert result["strategy"]["stability_score"] == 0


def test_edge_stability_preview_returns_single_horizon_only_for_one_visible_group() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {
                "top_strategy": _candidate("swing"),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "1h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
            "4h": {
                "top_strategy": _insufficient_candidate(),
                "top_symbol": _insufficient_candidate(),
                "top_alignment_state": _insufficient_candidate(),
            },
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "single_horizon_only"
    assert result["strategy"]["stability_score"] == 1


def test_edge_stability_preview_returns_multi_horizon_confirmed_for_repeated_group() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {"top_strategy": _candidate("swing")},
            "1h": {"top_strategy": _candidate("swing")},
            "4h": {"top_strategy": _insufficient_candidate()},
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "multi_horizon_confirmed"
    assert result["strategy"]["stability_score"] == 2


def test_edge_stability_preview_returns_unstable_for_different_visible_groups() -> None:
    preview = _edge_candidates_preview(
        {
            "15m": {"top_strategy": _candidate("alpha")},
            "1h": {"top_strategy": _candidate("beta")},
            "4h": {"top_strategy": _insufficient_candidate()},
        }
    )

    result = research_analyzer._build_edge_stability_preview(preview)

    assert result["strategy"]["stability_label"] == "unstable"
    assert result["strategy"]["stability_score"] == 1


def test_extract_edge_candidate_keeps_weak_visible_when_positive_rate_is_below_fifty() -> None:
    report = _ranked_group(
        "range_mean_revert",
        sample_count=42,
        labeled_count=42,
        coverage_pct=100.0,
        median_future_return_pct=0.12,
        positive_rate_pct=48.0,
    )

    result = research_analyzer._extract_edge_candidate(report)

    assert result["group"] == "range_mean_revert"
    assert result["sample_gate"] == "passed"
    assert result["quality_gate"] == "borderline"
    assert result["candidate_strength"] == "weak"
    assert result["positive_rate_pct"] == 48.0
    assert result["visibility_reason"] == "passed_sample_gate_only"


def test_extract_edge_candidate_keeps_truly_insufficient_candidates_hidden() -> None:
    report = _ranked_group(
        "not_ready",
        sample_count=42,
        labeled_count=0,
        coverage_pct=0.0,
        median_future_return_pct=0.12,
        positive_rate_pct=48.0,
    )

    result = research_analyzer._extract_edge_candidate(report)

    assert result["group"] == "insufficient_data"
    assert result["sample_gate"] == "failed"
    assert result["quality_gate"] == "failed"
    assert result["candidate_strength"] == "insufficient_data"


def test_extract_edge_candidate_still_classifies_moderate_and_strong_correctly() -> None:
    moderate = research_analyzer._extract_edge_candidate(
        _ranked_group(
            "moderate_case",
            sample_count=55,
            labeled_count=55,
            coverage_pct=100.0,
            median_future_return_pct=0.35,
            positive_rate_pct=56.0,
            robustness_pct=53.0,
        )
    )
    strong = research_analyzer._extract_edge_candidate(
        _ranked_group(
            "strong_case",
            sample_count=90,
            labeled_count=90,
            coverage_pct=100.0,
            median_future_return_pct=0.6,
            positive_rate_pct=60.0,
            robustness_pct=57.0,
        )
    )

    assert moderate["candidate_strength"] == "moderate"
    assert moderate["quality_gate"] == "passed"
    assert strong["candidate_strength"] == "strong"
    assert strong["quality_gate"] == "passed"


def test_edge_candidate_preview_markdown_remains_compatible_for_borderline_visibility() -> None:
    strategy_lab = {
        "ranking": {
            "15m": {
                "by_strategy": _ranked_group(
                    "range_mean_revert",
                    sample_count=42,
                    labeled_count=42,
                    coverage_pct=100.0,
                    median_future_return_pct=0.12,
                    positive_rate_pct=48.0,
                ),
            },
            "1h": {},
            "4h": {},
        }
    }

    preview = research_analyzer._build_edge_candidates_preview(strategy_lab)
    markdown_lines = research_analyzer._markdown_edge_candidates_preview(preview)
    markdown = "\n".join(markdown_lines)

    fifteen = preview["by_horizon"]["15m"]
    assert preview["strength_thresholds"]["hard_floors"]["sample_count"] == 30
    assert preview["strength_thresholds"]["hard_floors"]["labeled_count_gt"] == 0
    assert preview["strength_thresholds"]["raw_score_bands"] == research_analyzer.STRENGTH_RAW_SCORE_BANDS
    assert fifteen["top_strategy"]["candidate_strength"] == "weak"
    assert fifteen["top_strategy"]["quality_gate"] == "borderline"
    assert fifteen["sample_gate"] == "passed"
    assert fifteen["quality_gate"] == "borderline"
    assert "sample_gate: passed" in markdown
    assert "quality_gate: borderline" in markdown
    assert "range_mean_revert (weak; sample_gate=passed; quality_gate=borderline;" in markdown


def test_extract_edge_candidate_passes_absolute_minimum_with_coverage_only_support() -> None:
    report = _ranked_group(
        "coverage_only_case",
        sample_count=45,
        labeled_count=0,
        coverage_pct=12.5,
        median_future_return_pct=0.08,
        positive_rate_pct=47.0,
    )

    result = research_analyzer._extract_edge_candidate(report)

    assert result["group"] == "coverage_only_case"
    assert result["sample_gate"] == "passed"
    assert result["quality_gate"] == "borderline"
    assert result["candidate_strength"] == "weak"
    assert result["visibility_reason"] == "passed_sample_gate_only"


def test_extract_edge_candidate_falls_back_to_second_ranked_group_when_first_fails() -> None:
    report = {
        "ranked_groups": [
            _ranked_group_entry(
                "first_blocked",
                sample_count=45,
                labeled_count=0,
                coverage_pct=0.0,
                median_future_return_pct=0.10,
                positive_rate_pct=60.0,
            ),
            _ranked_group_entry(
                "second_viable",
                sample_count=52,
                labeled_count=52,
                coverage_pct=100.0,
                median_future_return_pct=0.11,
                positive_rate_pct=49.0,
            ),
        ]
    }

    result = research_analyzer._extract_edge_candidate(report)

    assert result["group"] == "second_viable"
    assert result["sample_gate"] == "passed"
    assert result["quality_gate"] == "passed"
    assert result["candidate_strength"] == "moderate"
    assert result["visibility_reason"] == "passed_sample_and_quality_gate"


def test_score_candidate_strength_promotes_emerging_moderate_when_only_one_metric_is_subscale() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=52,
        median_future_return_pct=0.35,
        positive_rate_pct=54.0,
        robustness_value=50.0,
    )

    assert diagnostics["final_classification"] == "moderate"
    assert diagnostics["classification_reason"] == "cleared_weighted_moderate_profile"
    assert diagnostics["aggregate_score"] >= research_analyzer.MODERATE_MIN_AGGREGATE_SCORE
    assert "subscale_positive_rate" in diagnostics["soft_penalties"]
    assert diagnostics["major_deficits"] == []


def test_score_candidate_strength_keeps_truly_thin_low_edge_profiles_weak() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=42,
        median_future_return_pct=0.12,
        positive_rate_pct=48.0,
        robustness_value=57.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert "median_return_below_emerging_moderate" in diagnostics["major_deficits"]
    assert "positive_rate_below_emerging_moderate" in diagnostics["major_deficits"]
    assert "positive_rate_below_coinflip" in diagnostics["soft_penalties"]


def test_extract_edge_candidate_returns_strength_diagnostics_for_visible_candidates() -> None:
    result = research_analyzer._extract_edge_candidate(
        _ranked_group(
            "near_moderate_case",
            sample_count=52,
            labeled_count=52,
            coverage_pct=100.0,
            median_future_return_pct=0.35,
            positive_rate_pct=54.0,
            robustness_pct=50.0,
        )
    )

    diagnostics = result["candidate_strength_diagnostics"]

    assert result["candidate_strength"] == "moderate"
    assert diagnostics["scoring_model"] == research_analyzer.STRENGTH_SCORING_MODEL
    assert diagnostics["component_scores"]["positive_rate_pct"]["band"] == "emerging"
    assert diagnostics["component_scores"]["robustness_value"]["band"] == "emerging"
    assert "aggregate_score=" in result["chosen_metric_summary"]


def test_score_candidate_strength_allows_one_supporting_major_deficit_when_score_is_strong_enough() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.35,
        positive_rate_pct=56.0,
        robustness_value=45.0,
    )

    assert (
        diagnostics["aggregate_score"]
        >= research_analyzer.MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE
    )
    assert diagnostics["major_deficits"] == ["robustness_below_emerging_moderate"]
    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "robustness_below_emerging_moderate"
    ]
    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_one_supporting_deficit"
    )


def test_score_candidate_strength_keeps_sample_count_critical_even_with_good_supporting_metrics() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=35,
        median_future_return_pct=0.35,
        positive_rate_pct=57.0,
        robustness_value=53.0,
    )

    assert "sample_count_below_emerging_moderate" in diagnostics["major_deficits"]
    assert diagnostics["major_deficit_breakdown"]["critical"] == [
        "sample_count_below_emerging_moderate"
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "critical_or_unknown_major_deficit_present"


def test_score_candidate_strength_treats_sub_emerging_positive_median_as_supporting_not_critical() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.15,
        positive_rate_pct=57.0,
        robustness_value=53.0,
    )

    assert "median_return_below_emerging_moderate" in diagnostics["major_deficits"]
    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate"
    ]
    assert (
        diagnostics["aggregate_score"]
        >= research_analyzer.MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE
    )
    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_one_supporting_deficit"
    )


def test_score_candidate_strength_uses_emerging_robustness_threshold_consistently() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.35,
        positive_rate_pct=56.0,
        robustness_value=46.0,
    )

    robustness_component = diagnostics["component_scores"]["robustness_value"]

    assert (
        robustness_component["emerging_threshold"]
        == research_analyzer.EDGE_EARLY_MODERATE_ROBUSTNESS_PCT
    )
    assert robustness_component["band"] in {"emerging", "thin"}
    assert "robustness_below_emerging_moderate" not in diagnostics["major_deficits"]


def test_score_metric_component_uses_v5_2_band_values_for_thin_and_below_floor() -> None:
    thin_component = research_analyzer._score_metric_component(
        metric_name="positive_rate_pct",
        value=48.0,
        weight=research_analyzer.STRENGTH_COMPONENT_WEIGHTS["positive_rate_pct"],
        strong_threshold=research_analyzer.EDGE_STRONG_POSITIVE_RATE_PCT,
        moderate_threshold=research_analyzer.EDGE_MODERATE_POSITIVE_RATE_PCT,
        emerging_threshold=research_analyzer.EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
        minimum_threshold=research_analyzer.POSITIVE_RATE_MINIMUM_FLOOR_PCT,
    )
    below_floor_component = research_analyzer._score_metric_component(
        metric_name="positive_rate_pct",
        value=46.0,
        weight=research_analyzer.STRENGTH_COMPONENT_WEIGHTS["positive_rate_pct"],
        strong_threshold=research_analyzer.EDGE_STRONG_POSITIVE_RATE_PCT,
        moderate_threshold=research_analyzer.EDGE_MODERATE_POSITIVE_RATE_PCT,
        emerging_threshold=research_analyzer.EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
        minimum_threshold=research_analyzer.POSITIVE_RATE_MINIMUM_FLOOR_PCT,
    )

    assert thin_component["band"] == "thin"
    assert thin_component["raw_score"] == research_analyzer.STRENGTH_RAW_SCORE_BANDS["thin"]
    assert below_floor_component["band"] == "below_floor"
    assert below_floor_component["raw_score"] == research_analyzer.STRENGTH_RAW_SCORE_BANDS["below_floor"]


def test_score_candidate_strength_allows_two_supporting_deficits_when_sample_is_moderate_and_score_is_high_enough() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=53.0,
    )

    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
    ]
    assert (
        diagnostics["aggregate_score"]
        >= research_analyzer.MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE
    )
    assert diagnostics["final_classification"] == "moderate"
    assert (
        diagnostics["classification_reason"]
        == "cleared_weighted_moderate_profile_with_two_supporting_deficits"
    )


def test_score_candidate_strength_keeps_two_supporting_deficits_weak_when_sample_not_moderate() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=42,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=53.0,
    )

    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "two_supporting_deficits_but_sample_not_moderate"


def test_score_candidate_strength_keeps_two_supporting_deficits_weak_when_positive_rate_below_floor() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=46.0,
        robustness_value=53.0,
    )

    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "two_supporting_deficits_but_positive_rate_below_floor"


def test_score_candidate_strength_three_supporting_deficits_recover_to_moderate_at_v5_3_threshold() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    assert diagnostics["major_deficit_breakdown"]["critical"] == []
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
        "robustness_below_emerging_moderate",
    ]
    assert diagnostics["aggregate_score"] == 59.6
    assert (
        diagnostics["aggregate_score"]
        >= research_analyzer.MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE
    )
    assert diagnostics["final_classification"] == "moderate"
    assert diagnostics["classification_reason"] == "cleared_weighted_moderate_profile_with_three_supporting_deficits"


def test_score_candidate_strength_three_supporting_deficits_with_lower_positive_rate_still_fail_on_aggregate_first() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=48.0,
        robustness_value=45.0,
    )

    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
        "robustness_below_emerging_moderate",
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "three_supporting_deficits_but_positive_rate_too_low"


def test_score_candidate_strength_two_supporting_deficits_blocked_by_positive_rate_floor_remain_weak() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=46.0,
        robustness_value=53.0,
    )

    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "two_supporting_deficits_but_positive_rate_below_floor"


def test_score_candidate_strength_three_supporting_deficits_with_lower_robustness_remain_weak_below_v5_3_threshold() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=44.0,
    )

    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
        "robustness_below_emerging_moderate",
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "three_supporting_deficits_but_aggregate_too_low"


def test_score_candidate_strength_three_supporting_deficits_with_sub_moderate_sample_still_fail_on_aggregate_first() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=45,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    assert "sample_count_below_emerging_moderate" not in diagnostics["major_deficits"]
    assert diagnostics["major_deficit_breakdown"]["supporting"] == [
        "median_return_below_emerging_moderate",
        "positive_rate_below_emerging_moderate",
        "robustness_below_emerging_moderate",
    ]
    assert diagnostics["final_classification"] == "weak"
    assert diagnostics["classification_reason"] == "three_supporting_deficits_but_aggregate_too_low"


def test_candidate_metric_summary_includes_classification_reason() -> None:
    diagnostics = research_analyzer._score_candidate_strength_diagnostics(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_value=45.0,
    )

    summary = research_analyzer._build_candidate_metric_summary(
        sample_count=55,
        median_future_return_pct=0.17,
        positive_rate_pct=49.0,
        robustness_label="signal_match_rate_pct",
        robustness_value=45.0,
        diagnostics=diagnostics,
    )

    assert "classification_reason=" in summary