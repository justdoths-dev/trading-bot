from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.research.diagnostics import (
    executable_entry_scarcity_diagnosis_report as report_module,
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
    action: str,
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
            "action": action,
            "execution_allowed": execution_allowed,
            "entry_price": entry_price,
            "signal": action,
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
                action="long",
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
    assert report["configuration_headlines"][0]["raw_rows_with_known_identity"] == 1
    assert report["configuration_headlines"][0]["edge_candidate_row_count"] == 0
    assert report["configuration_headlines"][1]["edge_candidate_row_count"] == 1


def test_funnel_accounting_tracks_raw_hold_execution_and_entry_rows(
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
                action="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=True,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=False,
                entry_price=101.0,
                future_label_15m="up",
                future_return_15m=0.5,
                future_label_1h="down",
                future_return_1h=-0.8,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=202.0,
                future_label_4h="up",
                future_return_4h=1.2,
            ),
            _raw_record(
                logged_at="2026-04-14T00:20:00+00:00",
                symbol="XRPUSDT",
                strategy=None,
                action="hold",
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
    raw_funnel = report["configuration_summaries"][0]["raw_to_execution_funnel"]
    execution_funnel = report["configuration_summaries"][0]["execution_to_entry_funnel"]

    assert raw_funnel["raw_input_rows"] == 5
    assert raw_funnel["raw_rows_with_known_identity"] == 4
    assert raw_funnel["hold_action_rows"] == 2
    assert raw_funnel["hold_action_known_identity_rows"] == 1
    assert raw_funnel["execution_allowed_rows"] == 3
    assert raw_funnel["execution_allowed_known_identity_rows"] == 2
    assert raw_funnel["positive_entry_rows"] == 3
    assert raw_funnel["positive_entry_known_identity_rows"] == 2
    assert raw_funnel["executable_positive_entry_rows"] == 2
    assert raw_funnel["executable_positive_entry_known_identity_rows"] == 1
    assert raw_funnel["research_labelable_dataset_rows"] == 2
    assert raw_funnel["labeled_rows_by_horizon"] == {"15m": 1, "1h": 1, "4h": 1}

    assert (
        execution_funnel["execution_allowed_missing_positive_entry_known_identity_rows"]
        == 1
    )
    assert execution_funnel["positive_entry_blocked_by_execution_known_identity_rows"] == 1


def test_per_symbol_strategy_breakdown_surfaces_hold_vs_missing_entry_split(
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
                action="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=True,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=False,
                entry_price=101.0,
                future_label_15m="up",
                future_return_15m=0.5,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=202.0,
                future_label_4h="up",
                future_return_4h=1.2,
            ),
        ],
    )

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(
            selected_rows=[_selected_row(symbol="ETHUSDT", strategy="swing", horizon="4h")],
            diagnostic_rows=[
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
        ),
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    rows = report["configuration_summaries"][0]["per_symbol_strategy_counts"]
    by_identity = {(row["symbol"], row["strategy"]): row for row in rows}

    btc_intraday = by_identity[("BTCUSDT", "intraday")]
    assert btc_intraday["raw_count"] == 3
    assert btc_intraday["hold_action_count"] == 1
    assert btc_intraday["non_hold_action_count"] == 2
    assert btc_intraday["execution_allowed_count"] == 1
    assert btc_intraday["positive_entry_count"] == 1
    assert btc_intraday["executable_positive_entry_count"] == 0
    assert btc_intraday["labelable_count"] == 1
    assert btc_intraday["labeled_counts_by_horizon"] == {"15m": 1, "1h": 0, "4h": 0}

    eth_swing = by_identity[("ETHUSDT", "swing")]
    assert eth_swing["surviving_edge_row_count"] == 1
    assert eth_swing["surviving_edge_counts_by_horizon"] == {
        "15m": 0,
        "1h": 0,
        "4h": 1,
    }


def test_strategy_mix_vs_survivor_mix_surfaces_swing_four_hour_concentration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-14T00:0{i}:00+00:00",
            symbol=f"BTCUSDT{i}",
            strategy="intraday",
            action="long",
            execution_allowed=bool(i % 2),
            entry_price=100.0 + i,
            future_label_15m="up",
            future_return_15m=0.4,
        )
        for i in range(5)
    ]
    rows.extend(
        [
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=200.0,
                future_label_4h="up",
                future_return_4h=1.0,
            ),
            _raw_record(
                logged_at="2026-04-14T00:11:00+00:00",
                symbol="SOLUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=210.0,
                future_label_4h="up",
                future_return_4h=1.1,
            ),
        ]
    )
    _write_jsonl(input_path, rows)

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(
            selected_rows=[
                _selected_row(symbol="ETHUSDT", strategy="swing", horizon="4h"),
                _selected_row(symbol="SOLUSDT", strategy="swing", horizon="4h"),
            ],
            diagnostic_rows=[
                _diagnostic_row(symbol="BTCUSDT0", strategy="intraday", horizon="15m")
            ],
        ),
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]
    strategy_mix = summary["strategy_mix"]
    concentration = summary["survivor_concentration_summary"]

    assert strategy_mix["raw_strategy_counts"][0]["value"] == "intraday"
    assert strategy_mix["raw_strategy_counts"][0]["count"] == 5
    assert strategy_mix["surviving_edge_rows_by_strategy_horizon"] == [
        {"strategy": "swing", "horizon": "4h", "count": 2, "share": 1.0}
    ]
    assert concentration["slow_swing_survivor_share"] == 1.0
    assert concentration["raw_intraday_scalping_share"] > 0.5
    assert concentration["survivor_intraday_scalping_share"] == 0.0


def test_final_assessment_flags_executable_entry_scarcity_as_primary() -> None:
    summaries = [
        {
            "configuration": report_module.DiagnosisConfiguration(36, 2500).to_dict(),
            "raw_to_execution_funnel": {
                "raw_input_rows": 220,
                "raw_rows_with_known_identity": 200,
                "hold_action_rows": 120,
                "hold_action_known_identity_rows": 120,
                "non_hold_action_known_identity_rows": 80,
                "execution_allowed_rows": 40,
                "execution_allowed_known_identity_rows": 40,
                "positive_entry_rows": 28,
                "positive_entry_known_identity_rows": 18,
                "executable_positive_entry_rows": 10,
                "executable_positive_entry_known_identity_rows": 10,
                "research_labelable_dataset_rows": 18,
                "edge_candidate_row_count": 0,
                "diagnostic_row_count": 4,
            },
            "execution_to_entry_funnel": {
                "primary_collapse_stage": "positive_entry_scarcity_among_execution_allowed",
                "execution_blocked_non_hold_known_identity_rows": 40,
                "execution_allowed_missing_positive_entry_known_identity_rows": 30,
                "executable_positive_entry_share_of_known_identity": 0.05,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
            "survivor_concentration_summary": {"total_surviving_edge_rows": 0},
        },
        {
            "configuration": report_module.DiagnosisConfiguration(336, 2500).to_dict(),
            "raw_to_execution_funnel": {
                "raw_input_rows": 380,
                "raw_rows_with_known_identity": 350,
                "hold_action_rows": 200,
                "hold_action_known_identity_rows": 200,
                "non_hold_action_known_identity_rows": 150,
                "execution_allowed_rows": 70,
                "execution_allowed_known_identity_rows": 70,
                "positive_entry_rows": 42,
                "positive_entry_known_identity_rows": 28,
                "executable_positive_entry_rows": 18,
                "executable_positive_entry_known_identity_rows": 18,
                "research_labelable_dataset_rows": 28,
                "edge_candidate_row_count": 0,
                "diagnostic_row_count": 6,
            },
            "execution_to_entry_funnel": {
                "primary_collapse_stage": "positive_entry_scarcity_among_execution_allowed",
                "execution_blocked_non_hold_known_identity_rows": 80,
                "execution_allowed_missing_positive_entry_known_identity_rows": 52,
                "executable_positive_entry_share_of_known_identity": 18 / 350,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
            "survivor_concentration_summary": {"total_surviving_edge_rows": 0},
        },
        {
            "configuration": report_module.DiagnosisConfiguration(336, 10000).to_dict(),
            "raw_to_execution_funnel": {
                "raw_input_rows": 520,
                "raw_rows_with_known_identity": 500,
                "hold_action_rows": 280,
                "hold_action_known_identity_rows": 280,
                "non_hold_action_known_identity_rows": 220,
                "execution_allowed_rows": 100,
                "execution_allowed_known_identity_rows": 100,
                "positive_entry_rows": 60,
                "positive_entry_known_identity_rows": 40,
                "executable_positive_entry_rows": 22,
                "executable_positive_entry_known_identity_rows": 22,
                "research_labelable_dataset_rows": 40,
                "edge_candidate_row_count": 1,
                "diagnostic_row_count": 8,
            },
            "execution_to_entry_funnel": {
                "primary_collapse_stage": "positive_entry_scarcity_among_execution_allowed",
                "execution_blocked_non_hold_known_identity_rows": 120,
                "execution_allowed_missing_positive_entry_known_identity_rows": 78,
                "executable_positive_entry_share_of_known_identity": 22 / 500,
            },
            "edge_candidate_outcomes": {"dominant_rejection_reason": "candidate_strength_weak"},
            "selected_survivors": [
                _selected_row(symbol="ETHUSDT", strategy="swing", horizon="4h")
            ],
            "diagnostic_rows": [
                _diagnostic_row(symbol="BTCUSDT", strategy="intraday", horizon="15m")
            ],
            "survivor_concentration_summary": {
                "total_surviving_edge_rows": 1,
                "slow_swing_survivor_share": 1.0,
                "raw_intraday_scalping_share": 0.75,
                "survivor_intraday_scalping_share": 0.0,
                "dominant_survivor_group": {"strategy": "swing", "horizon": "4h"},
            },
        },
    ]

    assessment = report_module.build_final_assessment(summaries)
    factors = assessment["factors"]

    assert factors["hold_row_dominance"]["status"] == "contributing"
    assert factors["execution_gate_scarcity"]["status"] == "contributing"
    assert factors["positive_entry_scarcity_among_execution_allowed"]["status"] == "primary"
    assert factors["recent_window_effect"]["status"] == "contributing"
    assert factors["latest_max_rows_cap_effect"]["status"] == "contributing"
    assert factors["downstream_quality_weakness"]["status"] == "contributing"
    assert factors["slow_swing_survivor_concentration"]["status"] == "concentrated"
    assert assessment["primary_bottleneck"] == "positive_entry_scarcity_among_execution_allowed"
    assert assessment["assessment"] == "positive_entry_scarcity_among_execution_allowed_primary"


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
                action="long",
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
