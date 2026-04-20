from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_fully_aligned_hold_residual_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_fully_aligned_hold_residual_diagnosis_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _raw_record(
    *,
    logged_at: str,
    symbol: str | None,
    strategy: str | None,
    bias: str | None,
    selected_strategy_signal: str | None = "hold",
    selected_strategy_signal_present: bool = True,
    rule_signal: str | None = None,
    rule_reason: str | None = None,
    root_reason: str | None = None,
    context_state: str | None = None,
    context_bias: str | None = None,
    setup_state: str | None = None,
    setup_bias: str | None = None,
    trigger_state: str | None = None,
    trigger_bias: str | None = None,
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
            payload["confidence"] = 0.82
        else:
            payload.pop("signal", None)
            payload.pop("confidence", None)

    rule_engine: dict[str, object] = {"confidence": 0.61}
    if bias is not None:
        rule_engine["bias"] = bias
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal
    if rule_reason is not None:
        rule_engine["reason"] = rule_reason

    timeframe_summary: dict[str, object] = {}
    if context_state is not None or context_bias is not None:
        timeframe_summary["context_layer"] = {
            "context": context_state,
            "bias": context_bias,
            "confidence": 0.61,
        }
        timeframe_summary["bias_layer"] = {
            "context": context_state,
            "bias": context_bias,
            "confidence": 0.61,
        }
    if setup_state is not None or setup_bias is not None:
        timeframe_summary["setup_layer"] = {
            "setup": setup_state,
            "bias": setup_bias,
            "confidence": 0.52,
        }
    if trigger_state is not None or trigger_bias is not None:
        timeframe_summary["trigger_layer"] = {
            "trigger": trigger_state,
            "bias": trigger_bias,
            "confidence": 0.53,
        }

    execution_signal = rule_signal or "hold"
    record = {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "execution": {"action": execution_signal, "signal": execution_signal},
        **strategy_payloads,
    }
    if timeframe_summary:
        record["timeframe_summary"] = timeframe_summary
    if root_reason is not None:
        record["reason"] = root_reason
    return record


def _comparison_group_row(report: dict, comparison_group: str) -> dict:
    for row in report["configuration_summaries"][0]["comparison_group_summaries"]:
        if row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(f"comparison_group not found: {comparison_group}")


def _strategy_row(report: dict, strategy: str, comparison_group: str) -> dict:
    for row in report["configuration_summaries"][0]["strategy_fully_aligned_summaries"]:
        if row["strategy"] == strategy and row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(
        f"strategy/comparison_group not found: {strategy}/{comparison_group}"
    )


def test_targets_only_final_fully_aligned_residual_slice(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Risk filter rejected the trade.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Opposing structure invalidated the trade.",
                context_state="bullish_context",
                context_bias="bearish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is not fully ready.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:04:00+00:00",
                symbol="XRPUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Setup is not fully ready.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:05:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["summary"]

    assert summary["actionable_selected_strategy_row_count"] == 5
    assert summary["fully_aligned_row_count"] == 2
    assert summary["preserved_fully_aligned_baseline_row_count"] == 1
    assert summary["collapsed_fully_aligned_row_count"] == 1
    assert summary["other_fully_aligned_rule_outcome_row_count"] == 0

    assert _comparison_group_row(
        report,
        "preserved_fully_aligned_baseline",
    )["row_count"] == 1
    assert _comparison_group_row(report, "collapsed_fully_aligned")["row_count"] == 1


def test_reason_text_is_not_required_for_fully_aligned_classification(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                rule_reason="Completely unrelated phrasing.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    collapsed = _comparison_group_row(report, "collapsed_fully_aligned")

    assert report["summary"]["fully_aligned_row_count"] == 2
    assert collapsed["row_count"] == 1
    assert collapsed["reason_bucket_counts"] == {"insufficient_explanation": 1}


def test_strategy_level_grouping_works_for_fully_aligned_rows(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="scalping",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Risk filter rejected the trade.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="scalping",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert _strategy_row(report, "scalping", "collapsed_fully_aligned")["row_count"] == 2
    assert _strategy_row(
        report,
        "swing",
        "preserved_fully_aligned_baseline",
    )["row_count"] == 1


def test_symbol_level_grouping_respects_minimum_support_threshold(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
        min_symbol_support=2,
    )
    symbol_rows = report["configuration_summaries"][0]["symbol_fully_aligned_summaries"]

    assert [
        (
            row["symbol"],
            row["comparison_group"],
            row["row_count"],
        )
        for row in symbol_rows
    ] == [
        (
            "BTCUSDT",
            "collapsed_fully_aligned",
            2,
        )
    ]


def test_comparison_support_status_is_explicit_and_predictable() -> None:
    assert (
        report_module._comparison_support_status(
            baseline_row_count=29,
            collapsed_row_count=30,
        )
        == "limited_support"
    )
    assert (
        report_module._comparison_support_status(
            baseline_row_count=30,
            collapsed_row_count=30,
        )
        == "supported"
    )


def test_wrapper_entrypoint_matches_existing_pattern(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="neutral",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Risk filter rejected the trade.",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_fully_aligned_hold_residual_diagnosis_report",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--config",
            "336/10000",
            "--min-symbol-support",
            "2",
        ],
    )
    runpy.run_path(str(Path(wrapper_module.__file__)), run_name="__main__")
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert "fully_aligned_row_count" in captured
    assert wrapper_module.build_report is report_module.build_report