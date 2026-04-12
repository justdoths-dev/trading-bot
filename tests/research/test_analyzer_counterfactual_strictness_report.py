from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.research.diagnostics import analyzer_counterfactual_strictness_report as report_module


def _write_summary(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _preview_block() -> dict:
    return {
        "by_horizon": {
            horizon: {"candidate_strength": "weak"}
            for horizon in ["15m", "1h", "4h"]
        }
    }


def _diagnostic_category(rejection_reason: str, candidate_strength: str) -> str:
    if rejection_reason == "strategy_horizon_incompatible":
        return "incompatibility"
    if rejection_reason == "failed_absolute_minimum_gate":
        return "insufficient_data"
    if candidate_strength == "weak" or rejection_reason == "candidate_strength_weak":
        return "quality_rejected"
    return "other_rejection"


def _rejected_evaluation(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    rejection_reason: str,
    candidate_strength: str,
    sample_count: int | None,
    labeled_count: int | None,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_signal_pct: float | None,
    classification_reason: str | None = None,
    aggregate_score: float | None = None,
    rejection_reasons: list[str] | None = None,
    strategy_horizon_compatible: bool = True,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "strategy_horizon_compatible": strategy_horizon_compatible,
        "rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons or [rejection_reason],
        "candidate_strength": candidate_strength,
        "candidate_strength_diagnostics": (
            {
                "classification_reason": classification_reason,
                "aggregate_score": aggregate_score,
            }
            if classification_reason is not None or aggregate_score is not None
            else None
        ),
        "aggregate_score": aggregate_score,
        "metrics": {
            "sample_count": sample_count,
            "labeled_count": labeled_count,
            "median_future_return_pct": median_future_return_pct,
            "positive_rate_pct": positive_rate_pct,
            "robustness_signal_pct": robustness_signal_pct,
        },
    }


def _identity(evaluation: dict, strategy_compatible_horizons: list[str]) -> dict:
    return {
        "identity_key": f"{evaluation['symbol']}:{evaluation['strategy']}",
        "symbol": evaluation["symbol"],
        "strategy": evaluation["strategy"],
        "strategy_compatible_horizons": strategy_compatible_horizons,
        "actual_joined_eligible_horizons": [],
        "horizon_evaluations": {evaluation["horizon"]: evaluation},
    }


def _diagnostic_from_evaluation(
    evaluation: dict,
    *,
    analyzer_compatible_horizons: list[str],
) -> dict:
    metrics = evaluation["metrics"]
    diagnostics = evaluation.get("candidate_strength_diagnostics") or {}
    return {
        "symbol": evaluation["symbol"],
        "strategy": evaluation["strategy"],
        "horizon": evaluation["horizon"],
        "status": "rejected",
        "diagnostic_category": _diagnostic_category(
            evaluation["rejection_reason"],
            evaluation["candidate_strength"],
        ),
        "strategy_horizon_compatible": evaluation["strategy_horizon_compatible"],
        "rejection_reason": evaluation["rejection_reason"],
        "rejection_reasons": evaluation["rejection_reasons"],
        "candidate_strength": evaluation["candidate_strength"],
        "classification_reason": diagnostics.get("classification_reason"),
        "aggregate_score": evaluation.get("aggregate_score"),
        "sample_count": metrics.get("sample_count"),
        "labeled_count": metrics.get("labeled_count"),
        "median_future_return_pct": metrics.get("median_future_return_pct"),
        "positive_rate_pct": metrics.get("positive_rate_pct"),
        "robustness_signal_pct": metrics.get("robustness_signal_pct"),
        "analyzer_compatible_horizons": analyzer_compatible_horizons,
    }


def _summary_payload(
    evaluations: list[tuple[dict, list[str]]],
    eligible_rows: list[dict] | None = None,
) -> dict:
    eligible_rows = eligible_rows or []
    diagnostics = [
        _diagnostic_from_evaluation(
            evaluation,
            analyzer_compatible_horizons=compatible,
        )
        for evaluation, compatible in evaluations
    ]
    rejection_counts = Counter(row["rejection_reason"] for row in diagnostics)
    category_counts = Counter(row["diagnostic_category"] for row in diagnostics)
    identities_only_incompatibility = [
        f"{evaluation['symbol']}:{evaluation['strategy']}"
        for evaluation, compatible in evaluations
        if evaluation["rejection_reason"] == "strategy_horizon_incompatible"
        and not compatible
    ]
    strategies_without_compatible = sorted(
        {
            evaluation["strategy"]
            for evaluation, compatible in evaluations
            if evaluation["rejection_reason"] == "strategy_horizon_incompatible"
            and not compatible
        }
    )
    return {
        "edge_candidates_preview": _preview_block(),
        "edge_candidate_rows": {
            "row_count": len(eligible_rows),
            "rows": eligible_rows,
            "diagnostic_row_count": len(diagnostics),
            "diagnostic_rows": diagnostics,
            "empty_reason_summary": {
                "has_eligible_rows": bool(eligible_rows),
                "diagnostic_rejection_reason_counts": dict(rejection_counts),
                "diagnostic_category_counts": dict(category_counts),
                "dominant_rejection_reason": (
                    max(rejection_counts, key=rejection_counts.get)
                    if rejection_counts
                    else None
                ),
                "empty_state_category": (
                    "has_eligible_rows"
                    if eligible_rows
                    else "mixed_rejections_without_eligible_rows"
                ),
                "identities_blocked_only_by_incompatibility": identities_only_incompatibility,
                "strategies_without_analyzer_compatible_horizons": strategies_without_compatible,
            },
            "dropped_row_count": 0,
            "dropped_rows": [],
            "identity_horizon_evaluations": [
                _identity(evaluation, compatible)
                for evaluation, compatible in evaluations
            ],
        },
    }


def _varied_payload() -> dict:
    sample_floor = _rejected_evaluation(
        symbol="BTCUSDT",
        strategy="swing",
        horizon="1h",
        rejection_reason="failed_absolute_minimum_gate",
        rejection_reasons=[
            "failed_absolute_minimum_gate",
            "sample_count_below_absolute_floor",
        ],
        candidate_strength="insufficient_data",
        sample_count=report_module.MIN_EDGE_CANDIDATE_SAMPLE_COUNT - 1,
        labeled_count=report_module.MIN_EDGE_CANDIDATE_SAMPLE_COUNT - 1,
        median_future_return_pct=0.22,
        positive_rate_pct=report_module.POSITIVE_RATE_MINIMUM_FLOOR_PCT + 2.0,
        robustness_signal_pct=report_module.THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT + 2.0,
    )
    sample_moderation = _rejected_evaluation(
        symbol="ETHUSDT",
        strategy="swing",
        horizon="4h",
        rejection_reason="candidate_strength_weak",
        candidate_strength="weak",
        sample_count=report_module.EDGE_MODERATE_SAMPLE_COUNT - 2,
        labeled_count=report_module.EDGE_MODERATE_SAMPLE_COUNT - 2,
        median_future_return_pct=report_module.EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT + 0.05,
        positive_rate_pct=report_module.EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT + 1.0,
        robustness_signal_pct=report_module.EDGE_EARLY_MODERATE_ROBUSTNESS_PCT + 1.0,
        classification_reason="two_supporting_deficits_but_sample_not_moderate",
        aggregate_score=report_module.MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE + 0.5,
        rejection_reasons=[
            "candidate_strength_weak",
            "two_supporting_deficits_but_sample_not_moderate",
        ],
    )
    positive_rate = _rejected_evaluation(
        symbol="XRPUSDT",
        strategy="intraday",
        horizon="15m",
        rejection_reason="candidate_strength_weak",
        candidate_strength="weak",
        sample_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 5,
        labeled_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 5,
        median_future_return_pct=report_module.EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT + 0.08,
        positive_rate_pct=report_module.THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT - 0.8,
        robustness_signal_pct=report_module.THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT + 1.0,
        classification_reason="three_supporting_deficits_but_positive_rate_too_low",
        aggregate_score=report_module.MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE + 0.5,
        rejection_reasons=[
            "candidate_strength_weak",
            "three_supporting_deficits_but_positive_rate_too_low",
        ],
    )
    weak = _rejected_evaluation(
        symbol="SOLUSDT",
        strategy="intraday",
        horizon="1h",
        rejection_reason="candidate_strength_weak",
        candidate_strength="weak",
        sample_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 10,
        labeled_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 10,
        median_future_return_pct=0.05,
        positive_rate_pct=max(report_module.POSITIVE_RATE_MINIMUM_FLOOR_PCT - 5.0, 1.0),
        robustness_signal_pct=max(
            report_module.THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT - 5.0,
            1.0,
        ),
        classification_reason="critical_or_unknown_major_deficit_present",
        aggregate_score=46.0,
        rejection_reasons=[
            "candidate_strength_weak",
            "critical_or_unknown_major_deficit_present",
        ],
    )
    noise = _rejected_evaluation(
        symbol="DOGEUSDT",
        strategy="scalping",
        horizon="15m",
        rejection_reason="strategy_horizon_incompatible",
        candidate_strength="incompatible",
        sample_count=None,
        labeled_count=None,
        median_future_return_pct=None,
        positive_rate_pct=None,
        robustness_signal_pct=None,
        strategy_horizon_compatible=False,
    )
    return _summary_payload(
        [
            (sample_floor, ["1h", "4h"]),
            (sample_moderation, ["1h", "4h"]),
            (positive_rate, ["15m", "1h"]),
            (weak, ["15m", "1h"]),
            (noise, []),
        ]
    )


def test_artifact_with_zero_eligible_rows_reports_expected_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    assert report["analyzer_preview"]["preview_block_exists"] is True
    assert report["joined_row_artifact"]["joined_row_block_exists"] is True
    assert report["joined_row_artifact"]["eligible_joined_row_count"] == 0
    assert report["rejected_row_diagnostics"]["normalized_rejected_row_count"] == 5
    assert report["joined_row_artifact"]["rejection_reason_counts"]["candidate_strength_weak"] == 3


def test_incompatible_strategy_or_horizon_noise_is_separated_correctly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    buckets = report["rejected_row_diagnostics"]["diagnosis_bucket_counts"]
    assert buckets["unsupported_strategy_or_horizon_noise"] == 1
    noise_row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "DOGEUSDT"
    )
    assert noise_row["diagnosis_bucket"] == "unsupported_strategy_or_horizon_noise"
    assert noise_row["scenario_survival_flags"]["baseline_excluding_incompatible_noise"] is False


def test_near_sample_floor_row_is_bucketed_and_rescued(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "BTCUSDT"
    )
    assert row["diagnosis_bucket"] == "near_miss_sample_floor"
    assert row["scenario_survival_flags"]["narrow_sample_floor_relief"] is True
    assert row["scenario_survival_flags"]["combined_narrow_relief"] is True


def test_near_sample_moderation_row_is_bucketed_and_rescued(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "ETHUSDT"
    )
    assert row["diagnosis_bucket"] == "near_miss_sample_moderation"
    assert row["scenario_survival_flags"]["narrow_sample_moderation_relief"] is True


def test_positive_rate_near_threshold_row_is_bucketed_correctly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "XRPUSDT"
    )
    assert row["diagnosis_bucket"] == "near_miss_positive_rate"
    assert row["scenario_survival_flags"]["narrow_positive_rate_near_threshold_relief"] is True


def test_clearly_weak_row_stays_weak_under_narrow_scenarios(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "SOLUSDT"
    )
    assert row["diagnosis_bucket"] == "structurally_weak_or_still_unconvincing"
    assert row["scenario_survival_flags"]["narrow_sample_floor_relief"] is False
    assert row["scenario_survival_flags"]["narrow_sample_moderation_relief"] is False
    assert row["scenario_survival_flags"]["narrow_positive_rate_near_threshold_relief"] is False
    assert row["scenario_survival_flags"]["combined_narrow_relief"] is False


def test_combined_narrow_relief_does_not_explode_beyond_near_miss_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    _write_summary(path, _varied_payload())
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    combined = report["counterfactual_scenarios"]["combined_narrow_relief"]
    assert combined["rescued_rejected_row_count"] == 3
    assert combined["quality_profile"]["bucket_counts"] == {
        "near_miss_sample_floor": 1,
        "near_miss_sample_moderation": 1,
        "near_miss_positive_rate": 1,
    }
    assert report["final_assessment"]["assessment"] == "potential_lossiness_detected"


def test_diagnostic_rows_backfill_missing_identity_fields(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    evaluation = _rejected_evaluation(
        symbol="ADAUSDT",
        strategy="intraday",
        horizon="1h",
        rejection_reason="candidate_strength_weak",
        candidate_strength="weak",
        sample_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 2,
        labeled_count=report_module.EDGE_MODERATE_SAMPLE_COUNT + 2,
        median_future_return_pct=report_module.EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT + 0.02,
        positive_rate_pct=report_module.THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT - 0.5,
        robustness_signal_pct=report_module.THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT + 1.0,
        classification_reason=None,
        aggregate_score=None,
        rejection_reasons=["candidate_strength_weak"],
    )
    payload = _summary_payload([(evaluation, ["15m", "1h"])])
    payload["edge_candidate_rows"]["diagnostic_rows"][0][
        "classification_reason"
    ] = "three_supporting_deficits_but_positive_rate_too_low"
    payload["edge_candidate_rows"]["diagnostic_rows"][0]["aggregate_score"] = (
        report_module.MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE + 0.5
    )

    _write_summary(path, payload)
    report = report_module.build_report(
        summary_path=path.resolve(),
        summary_path_resolution="explicit",
    )
    row = next(
        row for row in report["rejected_row_details"] if row["symbol"] == "ADAUSDT"
    )
    assert row["classification_reason"] == "three_supporting_deficits_but_positive_rate_too_low"
    assert row["aggregate_score"] == (
        report_module.MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE + 0.5
    )


def test_wrapper_module_imports_and_runs_correctly() -> None:
    wrapper = importlib.import_module(
        "src.research.analyzer_counterfactual_strictness_report"
    )
    target = importlib.import_module(
        "src.research.diagnostics.analyzer_counterfactual_strictness_report"
    )
    assert wrapper.build_report is target.build_report
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.research.analyzer_counterfactual_strictness_report",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--summary-path" in result.stdout
