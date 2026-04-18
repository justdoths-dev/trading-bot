from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_signal_transition_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_signal_transition_diagnosis_report as report_module,
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
    selected_strategy_signal: str | None = "hold",
    selected_strategy_signal_present: bool = True,
    rule_signal: str | None = None,
) -> dict:
    strategy_payloads = {
        "scalping_result": {
            "strategy": "scalping",
            "signal": "hold",
            "confidence": 0.0,
        },
        "intraday_result": {
            "strategy": "intraday",
            "signal": "hold",
            "confidence": 0.0,
        },
        "swing_result": {
            "strategy": "swing",
            "signal": "hold",
            "confidence": 0.0,
        },
    }
    if strategy is not None:
        payload = strategy_payloads[f"{strategy}_result"]
        if selected_strategy_signal_present:
            payload["signal"] = selected_strategy_signal
        else:
            payload.pop("signal", None)

    rule_engine: dict[str, object] = {}
    if bias is not None:
        rule_engine["bias"] = bias
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal

    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "execution": {"action": "hold", "signal": "hold"},
        **strategy_payloads,
    }


def test_build_report_uses_single_effective_input_snapshot_per_configuration(
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
                selected_strategy_signal="long",
                rule_signal="long",
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
                selected_strategy_signal="short",
                rule_signal="hold",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    effective_input_path = Path(summary["effective_input_path"])

    assert effective_input_path.name == (
        "_effective_selected_strategy_rule_signal_transition_input.jsonl"
    )
    assert len(_read_jsonl(effective_input_path)) == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True


def test_transition_matrix_surfaces_preservation_hold_watchlist_no_signal_and_hold_rows(
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
                selected_strategy_signal="long",
                rule_signal="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="short",
            ),
            _raw_record(
                logged_at="2026-04-14T00:02:00+00:00",
                symbol="XRPUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:03:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:04:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="watchlist_long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="watchlist_short",
            ),
            _raw_record(
                logged_at="2026-04-14T00:06:00+00:00",
                symbol="DOGEUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="no_signal",
            ),
            _raw_record(
                logged_at="2026-04-14T00:07:00+00:00",
                symbol="LTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    overview = summary["transition_overview"]
    actionable = summary["actionable_transition_summary"]

    assert overview["selected_strategy_result_signal_counts"] == {
        "long": 4,
        "short": 3,
        "hold": 1,
    }
    assert overview["rule_signal_counts"] == {
        "long": 1,
        "short": 1,
        "hold": 3,
        "watchlist_long": 1,
        "watchlist_short": 1,
        "no_signal": 1,
    }
    assert overview["transition_counts"] == {
        "long->long": 1,
        "long->hold": 1,
        "long->watchlist_long": 1,
        "long->no_signal": 1,
        "short->short": 1,
        "short->hold": 1,
        "short->watchlist_short": 1,
        "hold->hold": 1,
    }
    assert actionable["actionable_selected_strategy_rows"] == 7
    assert (
        actionable["actionable_selected_strategy_rows_with_actionable_rule_signal"]
        == 2
    )
    assert actionable["actionable_selected_strategy_preserved_same_direction_rows"] == 2
    assert actionable["actionable_selected_strategy_collapses_to_hold_rows"] == 2
    assert actionable["actionable_selected_strategy_collapses_to_watchlist_rows_total"] == 2
    assert actionable["actionable_selected_strategy_collapses_to_no_signal_rows"] == 1
    assert actionable["selected_strategy_non_actionable_rows"] == 1


def test_missing_unknown_other_and_unobservable_states_stay_separate(
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
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="unknown",
            ),
            _raw_record(
                logged_at="2026-04-14T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="neutral_conflict",
            ),
            _raw_record(
                logged_at="2026-04-14T00:03:00+00:00",
                symbol="ADAUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal_present=False,
                rule_signal="hold",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    overview = summary["transition_overview"]
    actionable = summary["actionable_transition_summary"]

    assert overview["selected_strategy_result_signal_counts"] == {
        "long": 3,
        "(missing)": 1,
    }
    assert overview["rule_signal_counts"] == {
        "(missing)": 1,
        "unknown": 1,
        "other": 1,
        "hold": 1,
    }
    assert actionable["actionable_selected_strategy_rule_signal_unobservable_rows"] == 1
    assert actionable["actionable_selected_strategy_collapses_to_unknown_rows"] == 1
    assert actionable["actionable_selected_strategy_collapses_to_other_rows"] == 1
    assert actionable["selected_strategy_unobservable_rows"] == 1


def test_support_thresholds_suppress_overclaiming_in_deep_breakdowns(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
            symbol="BTCUSDT",
            strategy="scalping",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
        )
        for index in range(9)
    ]
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    row = report["configuration_summaries"][0]["transition_by_strategy_symbol_bias_sign"][0]

    assert row["actionable_selected_strategy_rows"] == 9
    assert row["support_status"] == "limited_support"
    assert row["downgrade_support_status"] == "limited_support"
    assert row["primary_actionable_transition_path"] == "insufficient_support"
    assert row["primary_actionable_downgrade_path"] == "insufficient_support"


def test_mixed_downgrade_cases_do_not_overclaim_single_primary_path(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(20):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"HOLD{index}",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T01:{index:02d}:00+00:00",
                symbol=f"WATCH{index}",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="watchlist_short",
            )
        )

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]
    actionable = summary["actionable_transition_summary"]
    assessment = report["final_assessment"]

    assert actionable["dominant_observed_actionable_downgrade_path"] == "long->hold"
    assert actionable["primary_actionable_downgrade_path"] == "mixed_or_inconclusive"
    assert assessment["primary_transition_behavior"] == "mixed_or_inconclusive"


def test_reversal_paths_are_tracked_separately(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(15):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"REVLONG{index}",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="short",
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T01:{index:02d}:00+00:00",
                symbol=f"REVSHORT{index}",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="long",
            )
        )

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]
    actionable = summary["actionable_transition_summary"]
    assessment = report["final_assessment"]

    assert actionable["actionable_selected_strategy_rows"] == 30
    assert actionable["actionable_selected_strategy_preserved_opposite_direction_rows"] == 30
    assert actionable["primary_actionable_transition_path"] == "mixed_or_inconclusive"
    assert (
        assessment["primary_transition_behavior"]
        == "actionable_selected_strategy_reverses_direction_at_rule_signal"
    )


def test_composed_selected_strategy_result_is_not_used_as_legacy_source(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    row = _raw_record(
        logged_at="2026-04-14T00:00:00+00:00",
        symbol="BTCUSDT",
        strategy="intraday",
        bias="bullish",
        selected_strategy_signal_present=False,
        rule_signal="hold",
    )
    row.pop("intraday_result", None)
    row["selected_strategy_result"] = {
        "selected_strategy": "intraday",
        "signal": "long",
        "bias": "bullish",
        "confidence": 0.82,
        "reason": "post-composition object",
        "timeframe_summary": {},
        "debug": {},
    }
    _write_jsonl(input_path, [row])

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    overview = summary["transition_overview"]

    assert overview["selected_strategy_result_signal_counts"] == {"(missing)": 1}
    assert summary["actionable_transition_summary"]["selected_strategy_unobservable_rows"] == 1


def test_literal_selected_strategy_result_fallback_is_used_only_when_it_looks_like_legacy_payload(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    row = _raw_record(
        logged_at="2026-04-14T00:00:00+00:00",
        symbol="ETHUSDT",
        strategy="swing",
        bias="bearish",
        selected_strategy_signal_present=False,
        rule_signal="hold",
    )
    row.pop("swing_result", None)
    row["selected_strategy_result"] = {
        "strategy": "swing",
        "signal": "short",
        "confidence": 0.91,
    }
    _write_jsonl(input_path, [row])

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    overview = summary["transition_overview"]
    actionable = summary["actionable_transition_summary"]

    assert overview["selected_strategy_result_signal_counts"] == {"short": 1}
    assert overview["transition_counts"] == {"short->hold": 1}
    assert actionable["actionable_selected_strategy_rows"] == 1
    assert actionable["actionable_selected_strategy_collapses_to_hold_rows"] == 1


def test_final_assessment_strategy_map_excludes_limited_downgrade_support(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(12):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"MAP{index}",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold" if index < 3 else "long",
            )
        )

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )

    strategy_row = report["configuration_summaries"][0]["transition_by_strategy"][0]
    assert strategy_row["support_status"] == "supported"
    assert strategy_row["downgrade_support_status"] == "limited_support"
    assert strategy_row["primary_actionable_downgrade_path"] == "insufficient_support"
    assert report["final_assessment"]["supported_strategy_primary_actionable_downgrade_paths"] == {}


def test_diagnostic_module_entrypoint_and_wrapper_follow_existing_report_pattern(
    tmp_path: Path,
    capsys,
    monkeypatch,
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
                selected_strategy_signal="long",
                rule_signal="long",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_signal_transition_diagnosis_report",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--config",
            "336/10000",
        ],
    )
    runpy.run_path(str(Path(report_module.__file__)), run_name="__main__")
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert "primary_transition_behavior" in captured
    assert wrapper_module.build_report is report_module.build_report