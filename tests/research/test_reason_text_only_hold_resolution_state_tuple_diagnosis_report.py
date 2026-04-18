from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.reason_text_only_hold_resolution_state_tuple_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    reason_text_only_hold_resolution_state_tuple_diagnosis_report as report_module,
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
    rule_bias: str | None = None,
    rule_reason: str | None = None,
    root_reason: str | None = None,
    context_state: str | None = None,
    context_bias: str | None = None,
    setup_state: str | None = None,
    trigger_state: str | None = None,
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

    rule_engine: dict[str, object] = {
        "confidence": 0.61,
    }
    if strategy is not None:
        rule_engine["strategy"] = strategy
    if rule_bias is not None:
        rule_engine["bias"] = rule_bias
    elif bias is not None:
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
    if setup_state is not None:
        timeframe_summary["setup_layer"] = {
            "setup": setup_state,
            "bias": "bullish" if "long" in setup_state else "bearish"
            if "short" in setup_state
            else "neutral",
            "confidence": 0.5,
        }
    if trigger_state is not None:
        timeframe_summary["trigger_layer"] = {
            "trigger": trigger_state,
            "bias": "bullish" if "long" in trigger_state else "bearish"
            if "short" in trigger_state
            else "neutral",
            "confidence": 0.5,
        }

    record = {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        **strategy_payloads,
    }
    if timeframe_summary:
        record["timeframe_summary"] = timeframe_summary
    if root_reason is not None:
        record["reason"] = root_reason
    return record


def test_filters_only_reason_text_only_rows(tmp_path: Path) -> None:
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
            ),
            _raw_record(
                logged_at="2026-04-18T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Broader structure is neutral, so the engine blocks directional trades.",
                context_state="neutral",
                context_bias="neutral",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    targeting = report["configuration_summaries"][0]["reason_text_only_targeting"]

    assert targeting["hold_resolution_target_row_count"] == 2
    assert targeting["reason_text_only_target_row_count"] == 1
    assert targeting["structured_state_backed_hold_resolution_row_count"] == 1


def test_inventory_is_reused_and_remains_decision_adjacent(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    row = _raw_record(
        logged_at="2026-04-18T00:00:00+00:00",
        symbol="BTCUSDT",
        strategy="intraday",
        bias="bullish",
        selected_strategy_signal="long",
        rule_signal="hold",
        rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
    )
    row["timeframe_summary"] = {
        "15m": {
            "close": 100.0,
            "ema_20": 99.1,
        }
    }
    _write_jsonl(input_path, [row])

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    inventory = report["configuration_summaries"][0]["observable_decision_layer_inventory"]
    field_paths = {row["field_path"] for row in inventory["field_presence_rows"]}

    assert "rule_engine.signal" in field_paths
    assert "selected_strategy_payload.signal" in field_paths
    assert "timeframe_summary.15m.close" not in field_paths
    assert "timeframe_summary.15m.ema_20" not in field_paths


def test_dual_confirmation_absence_family_is_detected(tmp_path: Path) -> None:
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
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["reason_text_only_state_tuple_summary"]

    assert summary["reason_text_only_target_row_count"] == 1
    assert summary["dominant_reason_text_only_state_tuple_family"] == "dual_confirmation_absence"


def test_reason_category_and_family_cross_rows_are_reported(tmp_path: Path) -> None:
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
            ),
            _raw_record(
                logged_at="2026-04-18T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="scalping",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="The engine avoids trades against the broader bearish structure.",
                context_state=None,
                context_bias=None,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    rows = report["configuration_summaries"][0]["state_tuple_family_reason_category_count_rows"]
    labels = {
        (row["state_tuple_family"], row["reason_text_category_label"]): row["count"]
        for row in rows
    }

    assert labels[("dual_confirmation_absence", "confirmation_gap")] == 1
    assert labels[("dual_confirmation_absence", "directional_opposition")] == 1


def test_strategy_breakdown_is_support_threshold_guarded(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-18T00:{index:02d}:00+00:00",
            symbol="BTCUSDT",
            strategy="swing",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
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
    assert row["primary_reason_text_only_state_tuple_family"] == "insufficient_support"


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
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reason_text_only_hold_resolution_state_tuple_diagnosis_report",
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
    assert "primary_reason_text_only_state_tuple_family" in captured
    assert wrapper_module.build_report is report_module.build_report