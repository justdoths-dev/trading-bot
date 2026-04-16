from __future__ import annotations

import json
from pathlib import Path

import src.research.hold_dominance_diagnosis_report as wrapper_module
from src.research.diagnostics import hold_dominance_diagnosis_report as report_module


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
    action: str | None,
    execution_allowed: bool | None,
    entry_price: float | None,
    bias: str | None = None,
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
        "rule_engine": rule_engine,
        "execution": execution,
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
    sample_count: int = 58,
    labeled_count: int = 58,
) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": "rejected",
        "rejection_reason": rejection_reason,
        "candidate_strength": "weak",
        "classification_reason": "three_supporting_deficits_but_robustness_too_low",
        "sample_count": sample_count,
        "labeled_count": labeled_count,
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
                "diagnostic_rejection_reason_counts": {
                    dominant_rejection_reason: len(diagnostic_rows)
                }
                if dominant_rejection_reason
                else {},
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
                action="long",
                execution_allowed=True,
                entry_price=200.0,
                bias="long",
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
                strategy="intraday",
                action="hold",
                execution_allowed=False,
                entry_price=None,
                bias="long",
            )
        ],
    )

    observed: dict[str, object] = {}

    def _fake_run_research_analyzer(input_path_arg: Path, *_args, **_kwargs) -> dict:
        observed["input_path"] = input_path_arg
        observed["rows"] = _read_jsonl(input_path_arg)
        return _analyzer_result(
            selected_rows=[_selected_row(symbol="ETHUSDT", strategy="swing", horizon="4h")]
        )

    monkeypatch.setattr(report_module, "run_research_analyzer", _fake_run_research_analyzer)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]

    assert Path(observed["input_path"]).name == "_effective_hold_diagnosis_input.jsonl"
    assert len(observed["rows"]) == 2
    assert summary["hold_transition_funnel"]["raw_input_rows"] == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True


def test_unknown_actions_are_reported_without_being_forced_into_non_hold(
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
                bias="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                action=None,
                execution_allowed=False,
                entry_price=None,
                bias="long",
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
    funnel = summary["hold_transition_funnel"]
    by_strategy = {row["strategy"]: row for row in summary["hold_ratio_by_strategy"]}

    assert funnel["raw_rows_with_known_identity"] == 2
    assert funnel["unknown_action_known_identity_rows"] == 1
    assert funnel["hold_action_known_identity_rows"] == 1
    assert funnel["non_hold_action_known_identity_rows"] == 0
    assert funnel["unknown_action_share_of_known_identity"] == 0.5

    intraday = by_strategy["intraday"]
    assert intraday["unknown_action_rows"] == 1
    assert intraday["non_hold_rows"] == 0
    assert intraday["unknown_action_ratio"] == 0.5


def test_transition_funnel_bias_and_execution_patterns_are_reported(
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
                action="watchlist_long",
                execution_allowed=False,
                entry_price=None,
                bias="watchlist_long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=False,
                entry_price=101.0,
                bias="long",
                future_label_15m="up",
                future_return_15m=0.4,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                action="hold",
                execution_allowed=False,
                entry_price=None,
                bias="short",
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=202.0,
                bias="long",
                future_label_4h="up",
                future_return_4h=1.1,
            ),
            _raw_record(
                logged_at="2026-04-14T00:20:00+00:00",
                symbol="XRPUSDT",
                strategy=None,
                action="hold",
                execution_allowed=False,
                entry_price=None,
                bias="neutral_conflict",
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
    funnel = summary["hold_transition_funnel"]
    bias_rows = {row["bias"]: row for row in summary["bias_distribution_vs_hold_outcome"]}
    patterns = summary["execution_allowed_false_patterns"]

    assert funnel["raw_input_rows"] == 5
    assert funnel["raw_rows_with_known_identity"] == 4
    assert funnel["unknown_action_known_identity_rows"] == 0
    assert funnel["hold_action_known_identity_rows"] == 2
    assert funnel["non_hold_action_known_identity_rows"] == 2
    assert funnel["directional_bias_present_known_identity_rows"] == 4
    assert funnel["directional_bias_present_hold_rows"] == 2
    assert funnel["non_hold_execution_allowed_false_rows"] == 1
    assert funnel["non_hold_execution_allowed_rows"] == 1
    assert funnel["non_hold_executable_positive_entry_rows"] == 1
    assert funnel["primary_collapse_stage"] == "hold_row_dominance"

    assert bias_rows["long"]["hold_rows"] == 0
    assert bias_rows["long"]["non_hold_rows"] == 2
    assert bias_rows["short"]["hold_rows"] == 1
    assert bias_rows["watchlist_long"]["hold_rows"] == 1

    assert patterns["known_identity_execution_allowed_false_rows"] == 3
    assert patterns["hold_execution_allowed_false_rows"] == 2
    assert patterns["unknown_action_execution_allowed_false_rows"] == 0
    assert patterns["non_hold_execution_allowed_false_rows"] == 1
    assert patterns["by_strategy"][0]["strategy"] == "intraday"


def test_final_assessment_requires_support_and_separates_primary_from_secondary_factors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(24):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"BTCHOLD{index}",
                strategy="intraday",
                action="hold",
                execution_allowed=False,
                entry_price=None,
                bias="long",
            )
        )
    for index in range(8):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T01:{index:02d}:00+00:00",
                symbol=f"BTCBLOCK{index}",
                strategy="intraday",
                action="long",
                execution_allowed=False,
                entry_price=100.0 + index,
                bias="long",
                future_label_15m="up",
                future_return_15m=0.3,
            )
        )
    for index in range(8):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T02:{index:02d}:00+00:00",
                symbol=f"ETHLIVE{index}",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=200.0 + index,
                bias="long",
                future_label_4h="up",
                future_return_4h=1.2,
            )
        )

    _write_jsonl(input_path, rows)

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(
            selected_rows=[
                _selected_row(symbol="ETHLIVE0", strategy="swing", horizon="4h"),
                _selected_row(symbol="ETHLIVE1", strategy="swing", horizon="4h"),
                _selected_row(symbol="ETHLIVE2", strategy="swing", horizon="4h"),
                _selected_row(symbol="ETHLIVE3", strategy="swing", horizon="4h"),
            ],
            diagnostic_rows=[
                _diagnostic_row(symbol="BTCBLOCK0", strategy="intraday", horizon="15m"),
                _diagnostic_row(symbol="BTCBLOCK1", strategy="intraday", horizon="1h"),
            ],
        ),
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    assessment = report["final_assessment"]
    factors = assessment["factors"]

    assert factors["hold_row_dominance"]["status"] == "primary"
    assert factors["downstream_quality_weakness"]["status"] == "contributing"
    assert factors["strategy_mix_mismatch"]["status"] == "contributing"
    assert factors["slow_swing_survivor_concentration"]["status"] == "concentrated"
    assert assessment["primary_bottleneck"] == "hold_row_dominance"
    assert assessment["assessment"] == "hold_row_dominance_primary"
    assert assessment["confirmed_observations"]
    assert assessment["evidence_backed_inferences"]
    assert assessment["unresolved_uncertainties"]
    assert "cannot prove whether hold/no-trade outcomes reflect intended strategy abstention" in assessment[
        "unresolved_uncertainties"
    ][0]


def test_small_sample_stays_at_insufficient_support(
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
                bias="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                action="hold",
                execution_allowed=False,
                entry_price=None,
                bias="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                action="long",
                execution_allowed=False,
                entry_price=101.0,
                bias="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="BNBUSDT",
                strategy="swing",
                action="long",
                execution_allowed=True,
                entry_price=202.0,
                bias="long",
                future_label_4h="up",
                future_return_4h=1.1,
            ),
        ],
    )

    monkeypatch.setattr(
        report_module,
        "run_research_analyzer",
        lambda *_args, **_kwargs: _analyzer_result(
            selected_rows=[_selected_row(symbol="BNBUSDT", strategy="swing", horizon="4h")]
        ),
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    assessment = report["final_assessment"]
    assert assessment["factors"]["hold_row_dominance"]["status"] == "insufficient_support"
    assert assessment["assessment"] == "insufficient_support"


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
                action="long",
                execution_allowed=True,
                entry_price=200.0,
                bias="long",
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
