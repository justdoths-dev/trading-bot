from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.reason_text_only_setup_trigger_bias_alignment_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    reason_text_only_setup_trigger_bias_alignment_diagnosis_report as report_module,
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
    rule_signal: str | None = None,
    rule_reason: str | None = None,
    setup_state: str | None = None,
    setup_bias: str | None = None,
    trigger_state: str | None = None,
    trigger_bias: str | None = None,
    context_state: str | None = None,
    context_bias: str | None = None,
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
        payload["signal"] = selected_strategy_signal
        payload["confidence"] = 0.82

    rule_engine: dict[str, object] = {
        "signal": rule_signal,
        "reason": rule_reason,
        "bias": bias,
        "confidence": 0.61,
        "strategy": strategy,
    }

    timeframe_summary: dict[str, object] = {
        "context_layer": {
            "context": context_state,
            "bias": context_bias,
            "confidence": 0.61,
        },
        "bias_layer": {
            "context": context_state,
            "bias": context_bias,
            "confidence": 0.61,
        },
        "setup_layer": {
            "setup": setup_state,
            "bias": setup_bias,
            "confidence": 0.5,
        },
        "trigger_layer": {
            "trigger": trigger_state,
            "bias": trigger_bias,
            "confidence": 0.5,
        },
    }

    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "timeframe_summary": timeframe_summary,
        "reason": rule_reason,
        **strategy_payloads,
    }


def test_filters_only_reason_text_only_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at="2026-04-18T00:00:00+00:00",
            symbol="BTCUSDT",
            strategy="intraday",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
            setup_state=None,
            trigger_state=None,
            setup_bias="bullish",
            trigger_bias="bullish",
            context_state="bullish_context",
            context_bias="bullish",
        ),
        _raw_record(
            logged_at="2026-04-18T00:01:00+00:00",
            symbol="ETHUSDT",
            strategy="intraday",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_reason="Broader structure is neutral, so the engine blocks directional trades.",
            setup_state="neutral",
            trigger_state="neutral",
            setup_bias="neutral",
            trigger_bias="neutral",
            context_state="neutral",
            context_bias="neutral",
        ),
    ]

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    targeting = report["configuration_summaries"][0]["reason_text_only_targeting"]

    assert targeting["hold_resolution_target_row_count"] == 2
    assert targeting["reason_text_only_target_row_count"] == 1


def test_detects_bias_support_but_non_directional_state_text(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-18T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
                setup_state=None,
                trigger_state=None,
                setup_bias="bullish",
                trigger_bias="bullish",
                context_state="bullish_context",
                context_bias="bullish",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["bias_alignment_summary"]

    assert summary["reason_text_only_target_row_count"] == 1
    assert summary["dominant_bias_alignment_family"] == "both_bias_support_selected_signal_but_state_text_non_directional"


def test_detects_both_bias_neutral_or_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-18T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
                setup_state=None,
                trigger_state=None,
                setup_bias="neutral",
                trigger_bias="neutral",
                context_state="bullish_context",
                context_bias="bullish",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["bias_alignment_summary"]

    assert summary["dominant_bias_alignment_family"] == "both_bias_neutral_or_missing"


def test_detects_one_or_more_biases_oppose_selected_signal(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-18T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="scalping",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="The engine avoids trades against broader bearish structure.",
                setup_state=None,
                trigger_state=None,
                setup_bias="bearish",
                trigger_bias="bullish",
                context_state="bullish_context",
                context_bias="bullish",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["bias_alignment_summary"]

    assert summary["dominant_bias_alignment_family"] == "one_or_more_biases_oppose_selected_signal"


def test_strategy_breakdown_is_support_threshold_guarded(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-18T00:{index:02d}:00+00:00",
            symbol="BTCUSDT",
            strategy="intraday",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
            setup_state=None,
            trigger_state=None,
            setup_bias="bullish",
            trigger_bias="bullish",
            context_state="bullish_context",
            context_bias="bullish",
        )
        for index in range(9)
    ]
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    row = report["configuration_summaries"][0]["reason_text_only_by_strategy"][0]

    assert row["reason_text_only_target_row_count"] == 9
    assert row["support_status"] == "limited_support"
    assert row["primary_bias_alignment_family"] == "insufficient_support"


def test_entrypoint_and_wrapper_follow_existing_pattern(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-18T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
                setup_state=None,
                trigger_state=None,
                setup_bias="bullish",
                trigger_bias="bullish",
                context_state="bullish_context",
                context_bias="bullish",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reason_text_only_setup_trigger_bias_alignment_diagnosis_report",
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
    assert "primary_bias_alignment_family" in captured
    assert wrapper_module.build_report is report_module.build_report