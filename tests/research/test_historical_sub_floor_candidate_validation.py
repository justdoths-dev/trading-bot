from __future__ import annotations

import json
from pathlib import Path

from src.research.historical_sub_floor_candidate_validation import (
    build_historical_sub_floor_candidate_validation_report,
    run_historical_sub_floor_candidate_validation,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _diagnostic_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    rejection_reason: str = "failed_absolute_minimum_gate",
    rejection_reasons: list[str] | None = None,
    sample_gate: str = "failed",
    quality_gate: str = "failed",
    candidate_strength: str = "insufficient_data",
    diagnostic_category: str = "insufficient_data",
    visibility_reason: str = "failed_absolute_minimum_gate",
    sample_count: int = 18,
    labeled_count: int = 18,
    median_future_return_pct: float | None = 0.21,
    positive_rate_pct: float | None = 56.0,
    robustness_signal_pct: float | None = 54.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "diagnostic_category": diagnostic_category,
        "strategy_horizon_compatible": True,
        "rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons
        or ["failed_absolute_minimum_gate", "sample_count_below_absolute_floor"],
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": candidate_strength,
        "classification_reason": None,
        "aggregate_score": None,
        "chosen_metric_summary": "sample=18, median=0.21, positive_rate=56.0",
        "visibility_reason": visibility_reason,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": 100.0,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
    }


def _eligible_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    sample_count: int = 32,
    labeled_count: int = 32,
    median_future_return_pct: float | None = 0.24,
    positive_rate_pct: float | None = 58.0,
    robustness_signal_pct: float | None = 57.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_candidate_strength": "moderate",
        "selected_visible_horizons": [horizon],
        "selected_stability_label": "single_horizon_only",
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": 100.0,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
        "aggregate_score": 0.71,
        "visibility_reason": "passed_sample_and_quality_gate",
        "chosen_metric_summary": "selected_candidate",
    }


def _summary_payload(
    diagnostic_rows: list[dict[str, object]] | None,
    *,
    selected_rows: list[dict[str, object]] | None = None,
    max_age_hours: int = 36,
) -> dict[str, object]:
    edge_candidate_rows: dict[str, object] = {
        "row_count": len(selected_rows or []),
        "rows": selected_rows or [],
        "diagnostic_row_count": len(diagnostic_rows or []),
        "empty_reason_summary": {
            "dominant_rejection_reason": "failed_absolute_minimum_gate",
        },
    }
    if diagnostic_rows is not None:
        edge_candidate_rows["diagnostic_rows"] = diagnostic_rows

    return {
        "schema_validation": {
            "max_age_hours": max_age_hours,
            "max_rows": 2500,
            "valid_records": 40,
        },
        "edge_candidate_rows": edge_candidate_rows,
    }


def _band_entry(summary: dict[str, object], label: str) -> dict[str, object]:
    for row in summary["sample_band_summary"]:
        if row["sample_count_band"] == label:
            return row
    raise AssertionError(f"Band {label} not found")


def _identity_entry(summary: dict[str, object], identity_key: str) -> dict[str, object]:
    for row in summary["identity_summary"]:
        if row["identity_key"] == identity_key:
            return row
    raise AssertionError(f"Identity {identity_key} not found")


def test_sample_band_aggregation_correctness(tmp_path: Path) -> None:
    oldest = _write_json(
        tmp_path / "snapshot_01.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    sample_count=8,
                    labeled_count=8,
                    median_future_return_pct=0.10,
                    positive_rate_pct=55.0,
                    robustness_signal_pct=52.0,
                ),
                _diagnostic_row(
                    symbol="ETHUSDT",
                    sample_count=12,
                    labeled_count=12,
                    median_future_return_pct=-0.01,
                    positive_rate_pct=53.0,
                    robustness_signal_pct=55.0,
                ),
            ],
            max_age_hours=24,
        ),
    )
    middle = _write_json(
        tmp_path / "snapshot_02.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    sample_count=15,
                    labeled_count=15,
                    median_future_return_pct=0.12,
                    positive_rate_pct=48.0,
                    robustness_signal_pct=51.0,
                ),
                _diagnostic_row(
                    symbol="ETHUSDT",
                    sample_count=24,
                    labeled_count=24,
                    median_future_return_pct=0.09,
                    positive_rate_pct=54.0,
                    robustness_signal_pct=54.0,
                ),
            ],
            max_age_hours=48,
        ),
    )
    newest = _write_json(
        tmp_path / "snapshot_03.json",
        _summary_payload(
            [],
            selected_rows=[
                _eligible_row(
                    symbol="BTCUSDT",
                    sample_count=33,
                    labeled_count=33,
                    median_future_return_pct=0.25,
                    positive_rate_pct=60.0,
                    robustness_signal_pct=58.0,
                )
            ],
            max_age_hours=72,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report(
        [oldest, middle, newest]
    )

    band_0_9 = _band_entry(summary, "0-9")
    assert band_0_9["observation_count"] == 1
    assert band_0_9["distinct_identity_count"] == 1
    assert band_0_9["positive_median_ratio"] == 1.0
    assert band_0_9["median_of_medians"] == 0.1
    assert band_0_9["classification_distribution"] == {
        "sample_floor_only_near_miss": 1
    }
    assert band_0_9["later_matured_to_eligible_ratio"] == 1.0

    band_10_19 = _band_entry(summary, "10-19")
    assert band_10_19["observation_count"] == 2
    assert band_10_19["distinct_identity_count"] == 2
    assert band_10_19["positive_median_ratio"] == 0.5
    assert band_10_19["median_of_medians"] == 0.055
    assert band_10_19["classification_distribution"] == {
        "non_positive_or_flat_edge": 1,
        "sample_floor_plus_quality_weakness": 1,
    }
    assert band_10_19["later_matured_to_eligible_ratio"] == 1.0

    band_20_29 = _band_entry(summary, "20-29")
    assert band_20_29["observation_count"] == 1
    assert band_20_29["distinct_identity_count"] == 1
    assert band_20_29["classification_distribution"] == {
        "sample_floor_only_near_miss": 1
    }
    assert band_20_29["later_matured_to_eligible_ratio"] == 0.0

    band_30_plus = _band_entry(summary, "30+")
    assert band_30_plus["observation_count"] == 1
    assert band_30_plus["classification_distribution"] == {"eligible": 1}
    assert band_30_plus["later_matured_to_eligible_ratio"] is None

    assert summary["conservative_conclusion"]["threshold_change_support"] == (
        "insufficient_evidence"
    )


def test_non_positive_or_flat_edge_is_excluded_from_qualifying_cohort(
    tmp_path: Path,
) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="XRPUSDT",
                    sample_count=18,
                    labeled_count=18,
                    median_future_return_pct=0.0,
                    positive_rate_pct=55.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [],
            selected_rows=[_eligible_row(symbol="XRPUSDT", sample_count=32)],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])

    assert summary["overall_summary"]["cohort_identity_count"] == 0
    assert summary["identity_summary"] == []
    assert summary["source_summaries"][0]["classification_distribution"] == {
        "non_positive_or_flat_edge": 1
    }


def test_identity_recurrence_tracking(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=12,
                    labeled_count=12,
                    median_future_return_pct=0.11,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=52.0,
                )
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [
                _diagnostic_row(
                    sample_count=24,
                    labeled_count=24,
                    median_future_return_pct=0.13,
                    positive_rate_pct=54.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=48,
        ),
    )
    third = _write_json(
        tmp_path / "third.json",
        _summary_payload(
            [],
            selected_rows=[_eligible_row(sample_count=34, labeled_count=34)],
            max_age_hours=72,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report(
        [first, second, third]
    )
    identity = _identity_entry(summary, "BTCUSDT:swing:4h")

    assert identity["total_appearances"] == 3
    assert identity["appearances_by_band"] == {"10-19": 1, "20-29": 1, "30+": 1}
    assert identity["classification_distribution"] == {
        "eligible": 1,
        "sample_floor_only_near_miss": 2,
    }
    assert identity["ever_reached_30_plus"] is True
    assert identity["ever_became_eligible"] is True
    assert identity["maturation_outcome"] == "matured_to_eligible"


def test_same_snapshot_duplicate_identity_prefers_edge_candidate_row(
    tmp_path: Path,
) -> None:
    summary_path = _write_json(
        tmp_path / "snapshot.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    sample_count=18,
                    labeled_count=18,
                    median_future_return_pct=0.18,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ],
            selected_rows=[
                _eligible_row(
                    symbol="BTCUSDT",
                    sample_count=32,
                    labeled_count=32,
                )
            ],
            max_age_hours=24,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([summary_path])
    source = summary["source_summaries"][0]

    assert source["observation_count"] == 1
    assert source["classification_distribution"] == {"eligible": 1}
    assert source["source_facts"]["same_snapshot_duplicate_identity_count"] == 1
    assert source["source_facts"]["same_snapshot_cross_kind_duplicate_identity_count"] == 1
    assert source["source_facts"]["same_snapshot_dropped_observation_count"] == 1
    assert source["source_facts"]["same_snapshot_duplicate_resolution_rule"] == (
        "prefer_edge_candidate_rows_rows_over_diagnostic_rows_for_same_snapshot_identity"
    )
    assert (
        "same_snapshot_duplicate_identities_resolved_with_precedence="
        "prefer_edge_candidate_rows_rows_over_diagnostic_rows_for_same_snapshot_identity"
    ) in source["warnings"]


def test_maturation_classification_correctness(tmp_path: Path) -> None:
    oldest = _write_json(
        tmp_path / "oldest.json",
        _summary_payload(
            [
                _diagnostic_row(symbol="BTCUSDT", sample_count=18),
                _diagnostic_row(symbol="ETHUSDT", sample_count=12),
                _diagnostic_row(symbol="SOLUSDT", sample_count=22),
                _diagnostic_row(symbol="XRPUSDT", sample_count=9),
                _diagnostic_row(symbol="ADAUSDT", sample_count=18),
            ],
            max_age_hours=24,
        ),
    )
    newest = _write_json(
        tmp_path / "newest.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="ETHUSDT",
                    sample_count=25,
                    labeled_count=25,
                    median_future_return_pct=0.15,
                    positive_rate_pct=55.0,
                    robustness_signal_pct=55.0,
                ),
                _diagnostic_row(
                    symbol="SOLUSDT",
                    sample_count=19,
                    labeled_count=19,
                    median_future_return_pct=-0.02,
                    positive_rate_pct=54.0,
                    robustness_signal_pct=53.0,
                ),
                _diagnostic_row(
                    symbol="ADAUSDT",
                    sample_count=18,
                    labeled_count=18,
                    rejection_reasons=[
                        "failed_absolute_minimum_gate",
                        "sample_count_below_absolute_floor",
                        "strategy_horizon_incompatible",
                    ],
                    median_future_return_pct=0.16,
                    positive_rate_pct=58.0,
                    robustness_signal_pct=56.0,
                ),
            ],
            selected_rows=[_eligible_row(symbol="BTCUSDT", sample_count=31)],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([oldest, newest])

    assert _identity_entry(summary, "BTCUSDT:swing:4h")["maturation_outcome"] == (
        "matured_to_eligible"
    )
    assert _identity_entry(summary, "ETHUSDT:swing:4h")["maturation_outcome"] == (
        "grew_but_remained_sub_floor"
    )
    assert _identity_entry(summary, "SOLUSDT:swing:4h")["maturation_outcome"] == (
        "degraded_before_maturity"
    )
    assert _identity_entry(summary, "XRPUSDT:swing:4h")["maturation_outcome"] == (
        "disappeared"
    )
    assert _identity_entry(summary, "ADAUSDT:swing:4h")["maturation_outcome"] == (
        "structurally_blocked_later"
    )

    assert summary["maturation_summary"]["outcome_distribution"] == {
        "matured_to_eligible": 1,
        "grew_but_remained_sub_floor": 1,
        "degraded_before_maturity": 1,
        "structurally_blocked_later": 1,
        "disappeared": 1,
    }


def test_structural_blockers_are_excluded_from_near_miss_progression(
    tmp_path: Path,
) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [_diagnostic_row(symbol="ADAUSDT", sample_count=18)],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="ADAUSDT",
                    sample_count=18,
                    rejection_reasons=[
                        "failed_absolute_minimum_gate",
                        "sample_count_below_absolute_floor",
                        "strategy_horizon_incompatible",
                    ],
                    median_future_return_pct=0.14,
                    positive_rate_pct=57.0,
                    robustness_signal_pct=55.0,
                )
            ],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])
    identity = _identity_entry(summary, "ADAUSDT:swing:4h")

    assert identity["maturation_outcome"] == "structurally_blocked_later"
    assert identity["observations"][1]["classification"] == "non_sample_primary_failure"
    assert identity["observations"][1]["structural_non_sample_blockers"] == [
        "strategy_horizon_incompatible"
    ]


def test_false_near_miss_rows_do_not_enter_cohort(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BNBUSDT",
                    rejection_reasons=["failed_absolute_minimum_gate"],
                    visibility_reason="failed_absolute_minimum_gate",
                    sample_gate="failed",
                    sample_count=18,
                    labeled_count=18,
                    median_future_return_pct=0.20,
                    positive_rate_pct=59.0,
                    robustness_signal_pct=56.0,
                )
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [],
            selected_rows=[_eligible_row(symbol="BNBUSDT", sample_count=31)],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])

    assert summary["overall_summary"]["cohort_identity_count"] == 0
    assert summary["identity_summary"] == []
    assert summary["source_summaries"][0]["classification_distribution"] == {
        "non_sample_primary_failure": 1
    }
    assert summary["conservative_conclusion"]["threshold_change_support"] == (
        "insufficient_evidence"
    )


def test_conclusion_remains_conservative_under_weak_evidence(
    tmp_path: Path,
) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(symbol="BTCUSDT", sample_count=18),
                _diagnostic_row(symbol="ETHUSDT", sample_count=18),
                _diagnostic_row(symbol="ADAUSDT", sample_count=18),
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="ETHUSDT",
                    sample_count=14,
                    labeled_count=14,
                    median_future_return_pct=-0.02,
                    positive_rate_pct=53.0,
                    robustness_signal_pct=53.0,
                ),
                _diagnostic_row(
                    symbol="ADAUSDT",
                    sample_count=18,
                    labeled_count=18,
                    rejection_reasons=[
                        "failed_absolute_minimum_gate",
                        "sample_count_below_absolute_floor",
                        "strategy_horizon_incompatible",
                    ],
                    median_future_return_pct=0.12,
                    positive_rate_pct=57.0,
                    robustness_signal_pct=55.0,
                ),
            ],
            selected_rows=[_eligible_row(symbol="BTCUSDT", sample_count=31)],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])
    conclusion = summary["conservative_conclusion"]

    assert conclusion["identities_with_followup_count"] == 3
    assert conclusion["matured_to_eligible_identity_count"] == 1
    assert conclusion["matured_to_eligible_followup_ratio"] == 0.3333
    assert conclusion["degraded_before_maturity_identity_count"] == 1
    assert conclusion["structurally_blocked_later_identity_count"] == 1
    assert conclusion["threshold_change_support"] == "not_supported"
    assert conclusion["minimum_followup_identities_for_signal"] == 3
    assert conclusion["minimum_matured_to_eligible_ratio_for_signal"] == 0.5


def test_positive_median_ratio_uses_observation_count_denominator_when_medians_are_missing(
    tmp_path: Path,
) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    sample_count=15,
                    labeled_count=15,
                    median_future_return_pct=0.11,
                    positive_rate_pct=56.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    sample_count=16,
                    labeled_count=16,
                    median_future_return_pct=None,
                    positive_rate_pct=55.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])
    band_10_19 = _band_entry(summary, "10-19")

    assert band_10_19["observation_count"] == 2
    assert band_10_19["positive_median_ratio"] == 0.5
    assert band_10_19["median_of_medians"] == 0.11


def test_snapshot_classification_does_not_use_future_information(tmp_path: Path) -> None:
    first = _write_json(
        tmp_path / "first.json",
        _summary_payload(
            [
                _diagnostic_row(
                    symbol="DOGEUSDT",
                    sample_count=18,
                    labeled_count=18,
                    median_future_return_pct=0.17,
                    positive_rate_pct=48.0,
                    robustness_signal_pct=54.0,
                )
            ],
            max_age_hours=24,
        ),
    )
    second = _write_json(
        tmp_path / "second.json",
        _summary_payload(
            [],
            selected_rows=[_eligible_row(symbol="DOGEUSDT", sample_count=32)],
            max_age_hours=48,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([first, second])
    identity = _identity_entry(summary, "DOGEUSDT:swing:4h")

    assert identity["observations"][0]["classification"] == (
        "sample_floor_plus_quality_weakness"
    )
    assert identity["observations"][1]["classification"] == "eligible"
    assert identity["maturation_outcome"] == "matured_to_eligible"
    assert summary["source_summaries"][0]["classification_distribution"] == {
        "sample_floor_plus_quality_weakness": 1
    }


def test_report_exposes_unverified_input_order_chronology(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "snapshot.json",
        _summary_payload(
            [_diagnostic_row(sample_count=18)],
            max_age_hours=24,
        ),
    )

    summary = build_historical_sub_floor_candidate_validation_report([summary_path])

    assert summary["metadata"]["chronology_basis"] == "input_order_unverified"
    assert summary["metadata"]["chronology_verified"] is False
    assert any(
        "Chronology cannot be independently verified" in warning
        for warning in summary["warnings"]
    )


def test_report_writes_stable_outputs(tmp_path: Path) -> None:
    summary_path = _write_json(
        tmp_path / "snapshot.json",
        _summary_payload(
            [_diagnostic_row(sample_count=18)],
            max_age_hours=24,
        ),
    )

    result = run_historical_sub_floor_candidate_validation(
        summary_paths=[summary_path],
        output_dir=tmp_path / "out",
    )
    written_json = json.loads(
        (
            tmp_path / "out" / "historical_sub_floor_candidate_validation_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert result["summary_json"].endswith(
        "historical_sub_floor_candidate_validation_summary.json"
    )
    assert result["summary_md"].endswith(
        "historical_sub_floor_candidate_validation_summary.md"
    )
    assert written_json["metadata"]["report_type"] == (
        "historical_sub_floor_candidate_validation"
    )
    assert written_json["metadata"]["classification_version"] == "conservative_v2"
    assert written_json["metadata"]["report_version"] == "v2"
    assert set(written_json.keys()) == {
        "metadata",
        "source_summaries",
        "overall_summary",
        "sample_band_summary",
        "identity_summary",
        "maturation_summary",
        "conservative_conclusion",
        "warnings",
    }