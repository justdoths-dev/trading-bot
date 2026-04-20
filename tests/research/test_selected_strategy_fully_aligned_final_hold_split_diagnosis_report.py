from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_fully_aligned_final_hold_split_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_fully_aligned_final_hold_split_diagnosis_report as report_module,
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
    selected_strategy_confidence: float | None = 0.82,
    rule_signal: str | None = None,
    rule_reason: str | None = None,
    root_reason: str | None = None,
    rule_engine_confidence: float | None = 0.61,
    context_state: str | None = None,
    context_bias: str | None = None,
    context_confidence: float | None = 0.61,
    bias_confidence: float | None = 0.61,
    setup_state: str | None = None,
    setup_bias: str | None = None,
    setup_confidence: float | None = 0.52,
    trigger_state: str | None = None,
    trigger_bias: str | None = None,
    trigger_confidence: float | None = 0.53,
    execution_signal: str | None = None,
    execution_action: str | None = None,
    execution_allowed: bool | None = None,
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
            if selected_strategy_confidence is None:
                payload.pop("confidence", None)
            else:
                payload["confidence"] = selected_strategy_confidence
        else:
            payload.pop("signal", None)
            payload.pop("confidence", None)

    rule_engine: dict[str, object] = {}
    if rule_engine_confidence is not None:
        rule_engine["confidence"] = rule_engine_confidence
    if bias is not None:
        rule_engine["bias"] = bias
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal
    if rule_reason is not None:
        rule_engine["reason"] = rule_reason

    timeframe_summary: dict[str, object] = {}
    if context_state is not None or context_bias is not None:
        context_layer = {
            "context": context_state,
            "bias": context_bias,
        }
        if context_confidence is not None:
            context_layer["confidence"] = context_confidence
        bias_layer = {
            "context": context_state,
            "bias": context_bias,
        }
        if bias_confidence is not None:
            bias_layer["confidence"] = bias_confidence
        timeframe_summary["context_layer"] = context_layer
        timeframe_summary["bias_layer"] = bias_layer
    if setup_state is not None or setup_bias is not None:
        setup_layer = {
            "setup": setup_state,
            "bias": setup_bias,
        }
        if setup_confidence is not None:
            setup_layer["confidence"] = setup_confidence
        timeframe_summary["setup_layer"] = setup_layer
    if trigger_state is not None or trigger_bias is not None:
        trigger_layer = {
            "trigger": trigger_state,
            "bias": trigger_bias,
        }
        if trigger_confidence is not None:
            trigger_layer["confidence"] = trigger_confidence
        timeframe_summary["trigger_layer"] = trigger_layer

    execution_signal_value = (
        execution_signal if execution_signal is not None else (rule_signal or "hold")
    )
    execution_action_value = (
        execution_action if execution_action is not None else (rule_signal or "hold")
    )
    execution: dict[str, object] = {
        "action": execution_action_value,
        "signal": execution_signal_value,
    }
    if execution_allowed is not None:
        execution["execution_allowed"] = execution_allowed

    record = {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "execution": execution,
        **strategy_payloads,
    }
    if timeframe_summary:
        record["timeframe_summary"] = timeframe_summary
    if root_reason is not None:
        record["reason"] = root_reason
    return record


def _comparison_group_row(report: dict, comparison_group: str) -> dict:
    for row in report["comparison_group_summaries"]:
        if row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(f"comparison_group not found: {comparison_group}")


def _field_row(rows: list[dict], field: str) -> dict:
    for row in rows:
        if row["field"] == field:
            return row
    raise AssertionError(f"field not found: {field}")


def test_targets_only_rows_with_full_alignment_and_rule_bias_alignment(
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
                bias="bullish",
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
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:04:00+00:00",
                symbol="XRPUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:05:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="neutral",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:06:00+00:00",
                symbol="DOGEUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="long",
                rule_signal="hold",
                context_state="bullish_context",
                context_bias="bearish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:07:00+00:00",
                symbol="LTCUSDT",
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
    )

    assert report["summary"]["actionable_selected_strategy_row_count"] == 7
    assert report["summary"]["fully_aligned_row_count"] == 3
    assert report["summary"]["final_rule_bias_aligned_row_count"] == 2
    assert report["summary"]["preserved_final_directional_outcome_row_count"] == 1
    assert report["summary"]["collapsed_final_hold_outcome_row_count"] == 1
    assert report["summary"]["other_rule_bias_aligned_final_outcome_row_count"] == 0


def test_excludes_non_aligned_rule_bias_and_tracks_other_final_outcomes(
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
                bias="bullish",
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
                rule_signal="short",
                execution_signal="short",
                execution_action="short",
                context_state="bullish_context",
                context_bias="bullish",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
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

    assert report["summary"]["fully_aligned_row_count"] == 4
    assert report["summary"]["final_rule_bias_aligned_row_count"] == 3
    assert report["summary"]["preserved_final_directional_outcome_row_count"] == 1
    assert report["summary"]["collapsed_final_hold_outcome_row_count"] == 1
    assert report["summary"]["other_rule_bias_aligned_final_outcome_row_count"] == 1

    assert _comparison_group_row(
        report,
        "preserved_final_directional_outcome",
    )["row_count"] == 1
    assert _comparison_group_row(
        report,
        "collapsed_final_hold_outcome",
    )["row_count"] == 1


def test_numeric_differentiator_and_missing_numeric_fields_are_explicit(
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
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.93,
                rule_signal="long",
                rule_engine_confidence=0.91,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.72,
                bias_confidence=0.71,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.67,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=0.66,
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.91,
                rule_signal="long",
                rule_engine_confidence=0.89,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.71,
                bias_confidence=0.7,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.65,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=0.64,
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.78,
                rule_signal="hold",
                rule_engine_confidence=0.41,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.62,
                bias_confidence=0.61,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=None,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=0.57,
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.76,
                rule_signal="hold",
                rule_engine_confidence=0.39,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.61,
                bias_confidence=0.6,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=None,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=0.55,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    numeric_rows = report["numeric_field_comparison"]["field_comparisons"]
    rule_engine_row = _field_row(numeric_rows, "rule_engine_confidence")
    setup_row = _field_row(numeric_rows, "setup_layer_confidence")
    missingness_row = _field_row(
        report["missingness_comparison"]["field_comparisons"],
        "setup_layer_confidence",
    )

    assert rule_engine_row["comparison_status"] == "higher_on_preserved"
    assert rule_engine_row["preserved_final_directional_outcome"]["median"] == 0.9
    assert rule_engine_row["collapsed_final_hold_outcome"]["median"] == 0.4
    assert (
        report["numeric_field_comparison"]["strongest_numeric_differentiator"]["field"]
        == "rule_engine_confidence"
    )

    assert setup_row["comparison_status"] == "missing_on_collapsed_only"
    assert setup_row["preserved_final_directional_outcome"]["present_row_count"] == 2
    assert setup_row["collapsed_final_hold_outcome"]["present_row_count"] == 0
    assert missingness_row["missingness_pattern"] == "missing_on_collapsed_only"


def test_reason_bucket_difference_is_not_required_for_final_split_classification(
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
                strategy="scalping",
                bias="bullish",
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

    assert report["summary"]["preserved_final_directional_outcome_row_count"] == 1
    assert report["summary"]["collapsed_final_hold_outcome_row_count"] == 1
    assert report["tertiary_reason_comparison"]["comparison_status"] == "non_differentiating"
    assert report["tertiary_reason_comparison"]["reason_bucket_counts"] == {
        "preserved_final_directional_outcome": {"insufficient_explanation": 1},
        "collapsed_final_hold_outcome": {"insufficient_explanation": 1},
    }


def test_execution_and_final_signal_fields_are_surfaced_predictably(
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
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                execution_signal="long",
                execution_action="long",
                execution_allowed=True,
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
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                execution_signal="hold",
                execution_action="hold",
                execution_allowed=False,
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

    outcome_rows = report["final_outcome_field_comparison"]["field_comparisons"]
    rule_signal_row = _field_row(outcome_rows, "rule_signal_state")
    execution_signal_row = _field_row(outcome_rows, "execution_signal")
    execution_action_row = _field_row(outcome_rows, "execution_action")
    execution_allowed_row = _field_row(outcome_rows, "execution_allowed")

    assert rule_signal_row["comparison_status"] == "separates_groups"
    assert rule_signal_row["preserved_final_directional_outcome"]["value_counts"] == {
        "long": 1
    }
    assert rule_signal_row["collapsed_final_hold_outcome"]["value_counts"] == {
        "hold": 1
    }

    assert execution_signal_row["comparison_status"] == "separates_groups"
    assert execution_signal_row["preserved_final_directional_outcome"]["value_counts"] == {
        "long": 1
    }
    assert execution_signal_row["collapsed_final_hold_outcome"]["value_counts"] == {
        "hold": 1
    }

    assert execution_action_row["comparison_status"] == "separates_groups"
    assert execution_action_row["preserved_final_directional_outcome"]["value_counts"] == {
        "long": 1
    }
    assert execution_action_row["collapsed_final_hold_outcome"]["value_counts"] == {
        "hold": 1
    }

    assert execution_allowed_row["comparison_status"] == "separates_groups"
    assert execution_allowed_row["preserved_final_directional_outcome"]["value_counts"] == {
        "true": 1
    }
    assert execution_allowed_row["collapsed_final_hold_outcome"]["value_counts"] == {
        "false": 1
    }


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
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
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
            "selected_strategy_fully_aligned_final_hold_split_diagnosis_report",
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
    assert "final_rule_bias_aligned_row_count" in captured
    assert wrapper_module.build_report is report_module.build_report