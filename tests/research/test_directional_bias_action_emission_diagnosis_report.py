from __future__ import annotations

import json
from pathlib import Path

import src.research.directional_bias_action_emission_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    directional_bias_action_emission_diagnosis_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _raw_record(
    *,
    logged_at: str,
    symbol: str | None,
    strategy: str | None,
    bias: str | None,
    action: str | None,
    execution_allowed: bool | None,
    entry_price: float | None,
    execution_signal: str | None = None,
    rule_signal: str | None = None,
    future_label_15m: str | None = None,
    future_return_15m: float | None = None,
    future_label_1h: str | None = None,
    future_return_1h: float | None = None,
    future_label_4h: str | None = None,
    future_return_4h: float | None = None,
) -> dict:
    execution: dict[str, object] = {
        "execution_allowed": execution_allowed,
        "entry_price": entry_price,
    }
    if action is not None:
        execution["action"] = action
        execution["signal"] = execution_signal or action
    elif execution_signal is not None:
        execution["signal"] = execution_signal

    rule_engine: dict[str, object] = {}
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal

    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "execution": execution,
        "rule_engine": rule_engine,
        "future_label_15m": future_label_15m,
        "future_return_15m": future_return_15m,
        "future_label_1h": future_label_1h,
        "future_return_1h": future_return_1h,
        "future_label_4h": future_label_4h,
        "future_return_4h": future_return_4h,
    }


def _selected_row(
    *,
    symbol: str = "ETHUSDT",
    strategy: str = "swing",
    horizon: str = "4h",
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
                "classification_reason": "cleared_weighted_moderate_profile",
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
    candidate_strength: str = "weak",
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "rejection_reason": rejection_reason,
        "candidate_strength": candidate_strength,
        "classification_reason": "three_supporting_deficits_but_robustness_too_low",
        "sample_count": 58,
        "labeled_count": 58,
        "median_future_return_pct": 0.16,
        "positive_rate_pct": 48.5,
        "robustness_signal": "signal_match_rate_pct",
        "robustness_signal_pct": 44.0,
        "aggregate_score": 59.1,
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
            },
        },
    }


def test_build_report_uses_single_effective_input_snapshot_for_report_and_analyzer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    input_path = logs_dir / "trade_analysis.jsonl"
    shard_path = logs_dir / "trade_analysis_btcusdt.jsonl"

    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                action="buy",
                execution_allowed=True,
                entry_price=200.0,
                future_label_4h="up",
                future_return_4h=1.2,
            )
        ],
    )
    _write_jsonl(
        shard_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="scalping",
                bias="bullish",
                action="hold",
                execution_allowed=False,
                entry_price=None,
            )
        ],
    )

    observed: dict[str, object] = {}

    def _fake_run_research_analyzer(input_path_arg: Path, *_args, **_kwargs) -> dict:
        observed["input_path"] = input_path_arg
        observed["rows"] = _read_jsonl(input_path_arg)
        return _analyzer_result(selected_rows=[_selected_row()])

    monkeypatch.setattr(report_module, "run_research_analyzer", _fake_run_research_analyzer)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]

    assert Path(observed["input_path"]).name == (
        "_effective_directional_bias_action_emission_input.jsonl"
    )
    assert len(observed["rows"]) == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True
    assert summary["directional_emission_funnel"]["directional_bias_present_known_identity_rows"] == 2


def test_action_taxonomy_and_directional_emission_funnel_report_exact_counts(
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
                bias="bullish",
                action="buy",
                execution_allowed=True,
                entry_price=101.0,
                future_label_15m="up",
                future_return_15m=0.4,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                action="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bearish",
                action="sell",
                execution_allowed=True,
                entry_price=102.0,
                future_label_1h="down",
                future_return_1h=-0.7,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                action="watchlist_long",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:20:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bearish",
                action=None,
                execution_allowed=False,
                entry_price=None,
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
    summary = report["configuration_summaries"][0]
    taxonomy = summary["action_taxonomy"]
    funnel = summary["directional_emission_funnel"]

    assert taxonomy["known_identity_exact_action_counts"] == {
        "buy": 1,
        "hold": 1,
        "sell": 1,
        "watchlist_long": 1,
        "(missing)": 1,
    }
    assert taxonomy["directional_action_class_counts"] == {
        "buy": 1,
        "hold": 2,
        "sell": 1,
        "unknown": 1,
    }

    assert funnel["directional_bias_present_known_identity_rows"] == 5
    assert funnel["directional_buy_sell_emitted_rows"] == 2
    assert funnel["directional_hold_rows"] == 2
    assert funnel["directional_unknown_action_rows"] == 1
    assert funnel["directional_emitted_labelable_rows"] == 2
    assert funnel["directional_emitted_rows_with_any_future_label"] == 2
    assert funnel["directional_matching_sign_action_rows"] == 2
    assert funnel["directional_opposite_sign_action_rows"] == 0
    assert funnel["primary_collapse_stage"] == "directional_bias_to_action_emission_scarcity"


def test_strategy_specificity_and_bias_sign_breakdown_surface_scalping_zero_emission(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(report_module, "_MIN_STRATEGY_SUPPORT_ROWS", 2)

    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="scalping",
                bias="bullish",
                action="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="scalping",
                bias="bearish",
                action="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                action="buy",
                execution_allowed=True,
                entry_price=201.0,
                future_label_15m="up",
                future_return_15m=0.3,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bearish",
                action="sell",
                execution_allowed=True,
                entry_price=202.0,
                future_label_1h="down",
                future_return_1h=-0.6,
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
    summary = report["configuration_summaries"][0]
    specificity = summary["strategy_specificity"]
    by_strategy = {
        row["strategy"]: row for row in summary["directional_emission_by_strategy"]
    }
    by_sign = {
        (row["strategy"], row["symbol"], row["bias_sign"]): row
        for row in summary["directional_emission_by_strategy_symbol_bias_sign"]
    }

    assert specificity["classification"] == "strategy_specific_zero_emission"
    assert specificity["scalping_zero_emission"] is True
    assert by_strategy["scalping"]["emitted_buy_sell_rows"] == 0
    assert by_strategy["intraday"]["emitted_buy_sell_rows"] == 2
    assert by_sign[("scalping", "BTCUSDT", "bullish")]["emitted_buy_sell_rows"] == 0
    assert by_sign[("scalping", "BTCUSDT", "bearish")]["emitted_buy_sell_rows"] == 0
    assert by_sign[("intraday", "ETHUSDT", "bullish")]["buy_rows"] == 1
    assert by_sign[("intraday", "ETHUSDT", "bearish")]["sell_rows"] == 1


def test_execution_hold_is_not_reclassified_from_upstream_rule_signal(
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
                bias="bullish",
                action="hold",
                execution_allowed=False,
                entry_price=None,
                rule_signal="long",
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
    summary = report["configuration_summaries"][0]
    taxonomy = summary["action_taxonomy"]
    funnel = summary["directional_emission_funnel"]

    assert taxonomy["directional_exact_action_counts"] == {"hold": 1}
    assert taxonomy["directional_action_class_counts"] == {"hold": 1}
    assert funnel["directional_buy_sell_emitted_rows"] == 0
    assert funnel["directional_hold_rows"] == 1
    assert funnel["directional_unknown_action_rows"] == 0


def test_missing_execution_action_stays_unknown_even_with_upstream_rule_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bearish",
                action=None,
                execution_allowed=False,
                entry_price=None,
                rule_signal="short",
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
    summary = report["configuration_summaries"][0]
    taxonomy = summary["action_taxonomy"]
    funnel = summary["directional_emission_funnel"]

    assert taxonomy["directional_exact_action_counts"] == {"(missing)": 1}
    assert taxonomy["directional_action_class_counts"] == {"unknown": 1}
    assert funnel["directional_buy_sell_emitted_rows"] == 0
    assert funnel["directional_hold_rows"] == 0
    assert funnel["directional_unknown_action_rows"] == 1


def test_final_assessment_flags_emission_scarcity_as_primary() -> None:
    summary = {
        "configuration": report_module.DiagnosisConfiguration(336, 10000).to_dict(),
        "directional_emission_funnel": {
            "directional_bias_present_known_identity_rows": 120,
            "directional_buy_sell_emitted_rows": 12,
            "directional_hold_rows": 108,
            "directional_unknown_action_rows": 0,
            "directional_buy_sell_emission_rate": 12 / 120,
            "directional_hold_share": 108 / 120,
            "directional_emitted_labelable_rows": 12,
            "analyzer_diagnostic_row_count": 4,
            "analyzer_selected_row_count": 1,
            "directional_matching_sign_action_rows": 12,
            "directional_opposite_sign_action_rows": 0,
            "primary_collapse_stage": "directional_bias_to_action_emission_scarcity",
        },
        "strategy_specificity": {
            "classification": "strategy_specific_zero_emission",
            "supported_strategy_count": 3,
            "zero_emission_strategies": ["scalping"],
            "scalping_zero_emission": True,
            "scalping_directional_total": 60,
            "scalping_emitted_rows": 0,
            "max_emission_strategy": {
                "strategy": "intraday",
                "buy_sell_emission_rate": 0.16,
            },
            "min_emission_strategy": {
                "strategy": "scalping",
                "buy_sell_emission_rate": 0.0,
            },
            "emission_rate_gap": 0.16,
        },
        "analyzer_candidate_outcomes": {
            "diagnostic_row_count": 4,
            "diagnostic_weak_row_count": 3,
            "selected_row_count": 1,
            "dominant_rejection_reason": "candidate_strength_weak",
        },
        "selected_survivors": [_selected_row()],
        "action_taxonomy": {
            "directional_exact_action_labels": ["buy", "hold", "sell"],
        },
    }

    assessment = report_module.build_final_assessment([summary])
    factors = assessment["factors"]

    assert factors["directional_bias_action_emission_scarcity"]["status"] == "primary"
    assert factors["scalping_zero_emission"]["status"] == "present"
    assert factors["strategy_specificity"]["status"] == "strategy_specific"
    assert factors["directional_sign_mapping_inversion"]["status"] == "not_supported"
    assert factors["downstream_quality_weakness"]["status"] == "contributing"
    assert assessment["primary_bottleneck"] == "directional_bias_action_emission_scarcity"
    assert assessment["assessment"] == "directional_bias_action_emission_scarcity_primary"
    assert assessment["confirmed_observations"]
    assert assessment["evidence_backed_inferences"]
    assert assessment["unresolved_uncertainties"]


def test_main_and_wrapper_follow_existing_report_pattern(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                action="buy",
                execution_allowed=True,
                entry_price=200.0,
                future_label_4h="up",
                future_return_4h=1.2,
            )
        ],
    )

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(
            selected_rows=[_selected_row(symbol="ETHUSDT", strategy="swing", horizon="4h")]
        ),
    )

    report_module.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--config",
            "336/10000",
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert "primary_bottleneck" in captured
    assert wrapper_module.build_report is report_module.build_report
