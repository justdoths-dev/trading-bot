from __future__ import annotations

import json
from pathlib import Path

import src.research.selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_fully_aligned_rule_engine_confidence_threshold_diagnosis_report as report_module,
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


def _band_row(report: dict, band_label: str) -> dict:
    for row in report["confidence_band_summary"]["bands"]:
        if row["band_label"] == band_label:
            return row
    raise AssertionError(f"band not found: {band_label}")


def _field_row(rows: list[dict], field: str) -> dict:
    for row in rows:
        if row["field"] == field:
            return row
    raise AssertionError(f"field not found: {field}")


def test_targets_only_rows_with_final_rule_bias_aligned_slice(tmp_path: Path) -> None:
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
                rule_engine_confidence=0.85,
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
                rule_engine_confidence=0.22,
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
                rule_engine_confidence=0.18,
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
                rule_engine_confidence=0.24,
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
                rule_engine_confidence=0.24,
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
                rule_engine_confidence=0.24,
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
                rule_engine_confidence=0.24,
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
                rule_engine_confidence=0.24,
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
    assert report["rule_engine_confidence_distribution"][
        "preserved_final_directional_outcome"
    ]["row_count"] == 1
    assert report["rule_engine_confidence_distribution"][
        "collapsed_final_hold_outcome"
    ]["row_count"] == 1


def test_identifies_clean_non_overlapping_split(tmp_path: Path) -> None:
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
                selected_strategy_confidence=0.92,
                rule_signal="long",
                rule_engine_confidence=0.84,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.72,
                bias_confidence=0.72,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=1.0,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.9,
                rule_signal="long",
                rule_engine_confidence=0.78,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.69,
                bias_confidence=0.69,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.95,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.48,
                rule_signal="hold",
                rule_engine_confidence=0.22,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.36,
                bias_confidence=0.36,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.68,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.44,
                rule_signal="hold",
                rule_engine_confidence=0.19,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.33,
                bias_confidence=0.33,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.64,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["threshold_likeness"]["classification"] == "clean_non_overlapping_split"
    assert report["rule_engine_confidence_overlap"]["ranges_overlap"] is False
    assert report["rule_engine_confidence_overlap"]["range_order"] == "collapsed_below_preserved"
    assert report["confidence_band_summary"]["mixed_band_labels"] == []
    assert _band_row(report, "<= 0.25")["collapsed_row_count"] == 2
    assert _band_row(report, "> 0.80")["preserved_row_count"] == 1


def test_identifies_mixed_band_and_surfaces_secondary_summaries_only_there(
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
                selected_strategy_confidence=0.88,
                rule_signal="long",
                rule_engine_confidence=0.83,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.72,
                bias_confidence=0.72,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=1.0,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.82,
                rule_signal="long",
                rule_engine_confidence=0.71,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.67,
                bias_confidence=0.67,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=1.0,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.46,
                rule_signal="hold",
                rule_engine_confidence=0.69,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.35,
                bias_confidence=0.35,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.62,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.41,
                rule_signal="hold",
                rule_engine_confidence=0.24,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.31,
                bias_confidence=0.31,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.61,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["threshold_likeness"]["classification"] == "mostly_separated_with_mixed_band"
    assert report["confidence_band_summary"]["mixed_band_labels"] == ["(0.60, 0.80]"]
    assert _band_row(report, "(0.60, 0.80]")["band_mix_status"] == "mixed"

    mixed_bands = report["mixed_band_secondary_comparison"]["bands"]
    assert len(mixed_bands) == 1
    assert mixed_bands[0]["band_label"] == "(0.60, 0.80]"
    assert {row["field"] for row in mixed_bands[0]["field_comparisons"]} == {
        "setup_layer_confidence",
        "context_layer_confidence",
        "bias_layer_confidence",
        "selected_strategy_confidence",
        "trigger_layer_confidence",
    }
    assert mixed_bands[0]["strongest_secondary_differentiator"]["field"] in {
        "setup_layer_confidence",
        "context_layer_confidence",
        "bias_layer_confidence",
        "selected_strategy_confidence",
    }
    assert (
        _field_row(
            mixed_bands[0]["field_comparisons"],
            "context_layer_confidence",
        )["comparison_status"]
        == "higher_on_preserved"
    )


def test_identifies_broadly_mixed_when_all_populated_bands_are_mixed(
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
                selected_strategy_confidence=0.62,
                rule_signal="long",
                rule_engine_confidence=0.32,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.58,
                bias_confidence=0.58,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.82,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.60,
                rule_signal="long",
                rule_engine_confidence=0.38,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.56,
                bias_confidence=0.56,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.81,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.47,
                rule_signal="hold",
                rule_engine_confidence=0.34,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.44,
                bias_confidence=0.44,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.66,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                selected_strategy_confidence=0.45,
                rule_signal="hold",
                rule_engine_confidence=0.39,
                context_state="bullish_context",
                context_bias="bullish",
                context_confidence=0.42,
                bias_confidence=0.42,
                setup_state="long_confirmed",
                setup_bias="bullish",
                setup_confidence=0.65,
                trigger_state="long_confirmed",
                trigger_bias="bullish",
                trigger_confidence=1.0,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["threshold_likeness"]["classification"] == "broadly_mixed"
    assert report["rule_engine_confidence_overlap"]["ranges_overlap"] is True
    assert report["confidence_band_summary"]["mixed_band_labels"] == ["(0.25, 0.40]"]
    assert report["mixed_band_secondary_comparison"]["mixed_band_count"] == 1
    assert _band_row(report, "(0.25, 0.40]")["band_mix_status"] == "mixed"


def test_handles_missing_rule_engine_confidence_conservatively(
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
                rule_engine_confidence=0.82,
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
                rule_engine_confidence=0.21,
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
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_engine_confidence=None,
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

    assert report["threshold_likeness"]["classification"] == "inconclusive"
    assert report["confidence_band_summary"]["missing_rule_engine_confidence_row_count"] == 1
    assert report["missingness_context"]["explanatory_status"] == "explicit_missingness_requires_caution"
    assert report["missingness_context"]["rule_engine_confidence_missingness"][
        "missingness_pattern"
    ] == "missing_on_collapsed_only"


def test_reason_bucket_difference_is_not_required_for_threshold_classification(
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
                rule_engine_confidence=0.81,
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
                rule_engine_confidence=0.23,
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

    assert report["threshold_likeness"]["classification"] == "clean_non_overlapping_split"
    assert report["tertiary_reason_context"]["comparison_status"] == "non_differentiating"
    assert report["tertiary_reason_context"]["used_for_threshold_classification"] is False


def test_final_outcome_fields_remain_contextual_only(tmp_path: Path) -> None:
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
                rule_engine_confidence=0.79,
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
                rule_engine_confidence=0.24,
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

    outcome_rows = report["final_outcome_context"]["field_comparisons"]
    assert report["final_outcome_context"]["used_for_threshold_classification"] is False
    assert report["threshold_likeness"]["classification_basis"] == (
        "rule_engine_confidence_exact_ranges_and_fixed_bands"
    )
    assert report["threshold_likeness"]["used_for_threshold_classification"] == [
        "rule_engine_confidence"
    ]
    assert _field_row(outcome_rows, "rule_signal_state")["comparison_status"] == "separates_groups"
    assert _field_row(outcome_rows, "execution_signal")["comparison_status"] == "separates_groups"
    assert wrapper_module.build_report is report_module.build_report