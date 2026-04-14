from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.research.diagnostics import (
    recent_vs_wide_strategy_quality_diagnosis_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _raw_record(
    *,
    logged_at: str,
    symbol: str | None,
    strategy: str | None,
    execution_allowed: bool,
    entry_price: float | None,
    future_label_15m: str | None = None,
    future_return_15m: float | None = None,
    future_label_1h: str | None = None,
    future_return_1h: float | None = None,
    future_label_4h: str | None = None,
    future_return_4h: float | None = None,
) -> dict:
    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "execution": {
            "execution_allowed": execution_allowed,
            "entry_price": entry_price,
            "action": "long",
        },
        "future_label_15m": future_label_15m,
        "future_return_15m": future_return_15m,
        "future_label_1h": future_label_1h,
        "future_return_1h": future_return_1h,
        "future_label_4h": future_label_4h,
        "future_return_4h": future_return_4h,
    }


def _selected_row(
    *,
    symbol: str = "BTCUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
    classification_reason: str = "cleared_weighted_moderate_profile",
    aggregate_score: float = 68.0,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_candidate_strength": "moderate",
        "sample_count": 60,
        "labeled_count": 60,
        "median_future_return_pct": 0.34,
        "positive_rate_pct": 56.0,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": 57.0,
        "aggregate_score": aggregate_score,
        "visibility_reason": "passed_sample_and_quality_gate",
        "horizon_evaluation": {
            "candidate_strength_diagnostics": {
                "classification_reason": classification_reason,
                "aggregate_score": aggregate_score,
            }
        },
    }


def _diagnostic_row(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    rejection_reason: str = "candidate_strength_weak",
    classification_reason: str = "three_supporting_deficits_but_robustness_too_low",
    sample_count: int = 58,
    labeled_count: int = 58,
    median_future_return_pct: float | None = 0.16,
    positive_rate_pct: float | None = 48.5,
    robustness_signal_pct: float | None = 44.0,
    aggregate_score: float | None = 59.1,
    candidate_strength: str = "weak",
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "rejection_reason": rejection_reason,
        "candidate_strength": candidate_strength,
        "classification_reason": classification_reason,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": robustness_signal_pct,
        "aggregate_score": aggregate_score,
        "visibility_reason": rejection_reason,
    }


def _analyzer_result(
    *,
    selected_rows: list[dict] | None = None,
    diagnostic_rows: list[dict] | None = None,
    dominant_rejection_reason: str | None = None,
) -> dict:
    selected_rows = selected_rows or []
    diagnostic_rows = diagnostic_rows or []
    dominant_rejection_reason = dominant_rejection_reason or (
        diagnostic_rows[0]["rejection_reason"] if diagnostic_rows else None
    )
    return {
        "dataset_overview": {
            "date_range": {
                "start": "2026-04-10T00:00:00+00:00",
                "end": "2026-04-14T00:00:00+00:00",
            }
        },
        "edge_candidate_rows": {
            "row_count": len(selected_rows),
            "rows": selected_rows,
            "diagnostic_row_count": len(diagnostic_rows),
            "diagnostic_rows": diagnostic_rows,
            "empty_reason_summary": {
                "dominant_rejection_reason": dominant_rejection_reason,
                "diagnostic_rejection_reason_counts": {
                    dominant_rejection_reason: len(diagnostic_rows)
                }
                if dominant_rejection_reason
                else {},
                "diagnostic_category_counts": {"quality_rejected": len(diagnostic_rows)}
                if diagnostic_rows
                else {},
                "dominant_diagnostic_category": "quality_rejected"
                if diagnostic_rows
                else None,
                "empty_state_category": "has_eligible_rows"
                if selected_rows
                else "mixed_rejections_without_eligible_rows",
            },
        },
    }


def test_build_report_generates_configuration_headlines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                execution_allowed=True,
                entry_price=100.0,
                future_label_4h="up",
                future_return_4h=1.2,
            )
        ],
    )

    def _stub_run_analyzer(
        _input_path: Path,
        _output_dir: Path,
        *,
        latest_window_hours: int,
        latest_max_rows: int,
    ) -> dict:
        if latest_window_hours >= 336:
            return _analyzer_result(selected_rows=[_selected_row()])
        return _analyzer_result(
            diagnostic_rows=[
                _diagnostic_row(
                    symbol="BTCUSDT",
                    strategy="swing",
                    horizon="4h",
                )
            ]
        )

    monkeypatch.setattr(report_module, "run_research_analyzer", _stub_run_analyzer)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[
            report_module.DiagnosisConfiguration(36, 2500),
            report_module.DiagnosisConfiguration(336, 10000),
        ],
    )

    assert len(report["configurations_evaluated"]) == 2
    assert len(report["configuration_headlines"]) == 2
    assert report["configuration_headlines"][0]["edge_candidate_row_count"] == 0
    assert report["configuration_headlines"][1]["edge_candidate_row_count"] == 1


def test_funnel_accounting_tracks_raw_executable_and_labelable_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                execution_allowed=False,
                entry_price=101.0,
                future_label_15m="up",
                future_return_15m=0.5,
                future_label_1h="down",
                future_return_1h=-0.8,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                execution_allowed=True,
                entry_price=202.0,
                future_label_4h="up",
                future_return_4h=1.2,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="XRPUSDT",
                strategy=None,
                execution_allowed=True,
                entry_price=150.0,
            ),
        ],
    )

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(),
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    funnel = report["configuration_summaries"][0]["funnel"]

    assert funnel["raw_input_rows"] == 4
    assert funnel["execution_allowed_rows"] == 2
    assert funnel["positive_entry_rows"] == 3
    assert funnel["executable_positive_entry_rows"] == 2
    assert funnel["research_labelable_dataset_rows"] == 2
    assert funnel["labeled_rows_by_horizon"] == {"15m": 1, "1h": 1, "4h": 1}


def test_selected_row_extraction_preserves_classification_details() -> None:
    rows = report_module.extract_selected_rows(
        {
            "rows": [
                _selected_row(
                    classification_reason="cleared_weighted_moderate_profile_with_two_supporting_deficits",
                    aggregate_score=66.5,
                )
            ]
        }
    )

    assert len(rows) == 1
    assert rows[0]["candidate_strength"] == "moderate"
    assert (
        rows[0]["classification_reason"]
        == "cleared_weighted_moderate_profile_with_two_supporting_deficits"
    )
    assert rows[0]["aggregate_score"] == 66.5


def test_focus_group_breakdown_returns_requested_strategy_horizon_rows() -> None:
    breakdown = report_module.extract_focus_group_breakdown(
        edge_candidate_rows={
            "rows": [_selected_row()],
            "diagnostic_rows": [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    strategy="intraday",
                    horizon="15m",
                ),
                _diagnostic_row(
                    symbol="BTCUSDT",
                    strategy="intraday",
                    horizon="1h",
                ),
                _diagnostic_row(
                    symbol="ETHUSDT",
                    strategy="swing",
                    horizon="4h",
                ),
            ],
        },
        focus_groups=report_module.DEFAULT_FOCUS_GROUPS,
    )

    by_group = {(row["strategy"], row["horizon"]): row for row in breakdown}
    assert len(by_group[("intraday", "15m")]["diagnostic_rows"]) == 1
    assert len(by_group[("intraday", "1h")]["diagnostic_rows"]) == 1
    assert len(by_group[("swing", "4h")]["selected_rows"]) == 1
    assert len(by_group[("swing", "4h")]["diagnostic_rows"]) == 1


def test_final_assessment_flags_quality_weakness_with_wide_window_survivor() -> None:
    widest_focus_groups = [
        {
            "strategy": "intraday",
            "horizon": "15m",
            "selected_rows": [],
            "diagnostic_rows": [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    strategy="intraday",
                    horizon="15m",
                    sample_count=62,
                    labeled_count=62,
                )
            ],
        },
        {
            "strategy": "intraday",
            "horizon": "1h",
            "selected_rows": [],
            "diagnostic_rows": [
                _diagnostic_row(
                    symbol="BTCUSDT",
                    strategy="intraday",
                    horizon="1h",
                    sample_count=56,
                    labeled_count=56,
                )
            ],
        },
        {
            "strategy": "swing",
            "horizon": "4h",
            "selected_rows": [_selected_row()],
            "diagnostic_rows": [
                _diagnostic_row(
                    symbol="ETHUSDT",
                    strategy="swing",
                    horizon="4h",
                    sample_count=58,
                    labeled_count=58,
                )
            ],
        },
    ]

    summaries = [
        {
            "configuration": report_module.DiagnosisConfiguration(36, 2500).to_dict(),
            "funnel": {
                "raw_input_rows": 200,
                "positive_entry_rows": 70,
                "executable_positive_entry_rows": 50,
                "research_labelable_dataset_rows": 40,
                "edge_candidate_row_count": 0,
                "diagnostic_row_count": 4,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
            "focus_group_breakdown": [],
        },
        {
            "configuration": report_module.DiagnosisConfiguration(336, 2500).to_dict(),
            "funnel": {
                "raw_input_rows": 350,
                "positive_entry_rows": 140,
                "executable_positive_entry_rows": 100,
                "research_labelable_dataset_rows": 90,
                "edge_candidate_row_count": 0,
                "diagnostic_row_count": 6,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
            "focus_group_breakdown": [],
        },
        {
            "configuration": report_module.DiagnosisConfiguration(336, 10000).to_dict(),
            "funnel": {
                "raw_input_rows": 450,
                "positive_entry_rows": 160,
                "executable_positive_entry_rows": 100,
                "research_labelable_dataset_rows": 110,
                "edge_candidate_row_count": 1,
                "diagnostic_row_count": 7,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [_selected_row()],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m"),
                _diagnostic_row(
                    symbol="DOGEUSDT",
                    strategy="scalping",
                    horizon="4h",
                    rejection_reason="strategy_horizon_incompatible",
                    classification_reason="strategy_horizon_incompatible",
                    sample_count=0,
                    labeled_count=0,
                    median_future_return_pct=None,
                    positive_rate_pct=None,
                    robustness_signal_pct=None,
                    aggregate_score=None,
                    candidate_strength="incompatible",
                ),
            ],
            "focus_group_breakdown": widest_focus_groups,
        },
    ]

    assessment = report_module.build_final_assessment(summaries)
    factors = assessment["factors"]

    assert factors["recent_window_starvation"]["status"] == "limited"
    assert factors["latest_max_rows_cap_effects"]["status"] == "contributing"
    assert factors["executable_row_scarcity"]["status"] == "contributing"
    assert factors["quality_weakness"]["status"] == "primary"
    assert factors["incompatibility_noise"]["status"] == "noise"
    assert assessment["assessment"] == "quality_weakness_primary"


def test_build_report_does_not_mutate_analyzer_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                execution_allowed=True,
                entry_price=100.0,
                future_label_4h="up",
                future_return_4h=1.2,
            )
        ],
    )

    analyzer_result = _analyzer_result(
        selected_rows=[_selected_row()],
        diagnostic_rows=[
            _diagnostic_row(
                symbol="ETHUSDT",
                strategy="swing",
                horizon="4h",
            )
        ],
    )
    expected = deepcopy(analyzer_result)

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: analyzer_result,
    )

    report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )

    assert analyzer_result == expected
