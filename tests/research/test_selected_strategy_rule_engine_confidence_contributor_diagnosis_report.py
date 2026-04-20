from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_contributor_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_contributor_diagnosis_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _raw_record(
    *,
    logged_at: str,
    symbol: str,
    strategy: str = "intraday",
    bias: str = "bullish",
    selected_strategy_signal: str = "long",
    selected_strategy_confidence: float | None = 0.82,
    rule_signal: str = "long",
    rule_engine_confidence: float | None = 0.61,
    context_confidence: float | None = 0.61,
    bias_confidence: float | None = 0.61,
    setup_confidence: float | None = 0.52,
    trigger_confidence: float | None = 0.53,
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
    payload = strategy_payloads[f"{strategy}_result"]
    payload["signal"] = selected_strategy_signal
    if selected_strategy_confidence is None:
        payload.pop("confidence", None)
    else:
        payload["confidence"] = selected_strategy_confidence

    rule_engine: dict[str, object] = {
        "bias": bias,
        "signal": rule_signal,
    }
    if rule_engine_confidence is not None:
        rule_engine["confidence"] = rule_engine_confidence

    timeframe_summary: dict[str, object] = {
        "context_layer": {
            "context": "bullish_context",
            "bias": "bullish",
        },
        "bias_layer": {
            "context": "bullish_context",
            "bias": "bullish",
        },
        "setup_layer": {
            "setup": "long_confirmed",
            "bias": "bullish",
        },
        "trigger_layer": {
            "trigger": "long_confirmed",
            "bias": "bullish",
        },
    }
    if context_confidence is not None:
        timeframe_summary["context_layer"]["confidence"] = context_confidence
    if bias_confidence is not None:
        timeframe_summary["bias_layer"]["confidence"] = bias_confidence
    if setup_confidence is not None:
        timeframe_summary["setup_layer"]["confidence"] = setup_confidence
    if trigger_confidence is not None:
        timeframe_summary["trigger_layer"]["confidence"] = trigger_confidence

    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "execution": {"action": rule_signal, "signal": rule_signal},
        "timeframe_summary": timeframe_summary,
        **strategy_payloads,
    }


def _field_row(report: dict, field: str) -> dict:
    for row in report["contributor_comparison"]["field_comparisons"]:
        if row["field"] == field:
            return row
    raise AssertionError(f"field not found: {field}")


def test_supported_comparison_path_emits_contributor_summaries(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []
    for index in range(30):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T00:{index:02d}:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=0.92,
                rule_signal="long",
                rule_engine_confidence=0.86,
                context_confidence=0.74,
                bias_confidence=0.74,
                setup_confidence=0.69,
                trigger_confidence=1.0,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T01:{index:02d}:00+00:00",
                symbol="ETHUSDT",
                selected_strategy_confidence=0.46,
                rule_signal="hold",
                rule_engine_confidence=0.21,
                context_confidence=0.34,
                bias_confidence=0.34,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["summary"]["final_rule_bias_aligned_row_count"] == 60
    assert report["summary"]["preserved_final_directional_outcome_row_count"] == 30
    assert report["summary"]["collapsed_final_hold_outcome_row_count"] == 30

    selected_strategy_row = _field_row(report, "selected_strategy_confidence")
    trigger_row = _field_row(report, "trigger_layer_confidence")

    assert selected_strategy_row["comparison_status"] == "higher_on_preserved"
    assert (
        selected_strategy_row["range_overlap"]["range_order"]
        == "collapsed_below_preserved"
    )
    assert "confidence_band_summary" not in report["rule_engine_confidence_context"]
    assert (
        report["contributor_comparison"]["leading_contributor_differentiator"]["field"]
        == report["contributor_comparison"]["field_comparisons"][0]["field"]
    )
    assert (
        report["contributor_tracking"]["strongest_tracking_contributor"]["field"]
        == "selected_strategy_confidence"
    )
    assert trigger_row["comparison_status"] == "no_clear_separation"
    assert (
        report["contributor_tracking"]["trigger_negative_control_status"]
        == "trigger_remains_negative_control"
    )
    assert report["final_interpretation"]["interpretation_status"] == (
        "persisted_contributors_appear_sufficient"
    )

    strong_families = set(
        report["final_interpretation"]["strong_aligned_non_trigger_contributor_families"]
    )
    assert {"selected_strategy", "context_and_bias"}.issubset(strong_families)


def test_missing_contributor_fields_are_explicit_and_safe(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=0.9,
                rule_signal="long",
                rule_engine_confidence=0.84,
                context_confidence=0.72,
                bias_confidence=0.72,
                setup_confidence=0.66,
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                selected_strategy_confidence=0.88,
                rule_signal="long",
                rule_engine_confidence=0.81,
                context_confidence=0.7,
                bias_confidence=0.7,
                setup_confidence=0.65,
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                selected_strategy_confidence=0.48,
                rule_signal="hold",
                rule_engine_confidence=0.24,
                context_confidence=0.35,
                bias_confidence=0.35,
                setup_confidence=None,
                trigger_confidence=1.0,
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                selected_strategy_confidence=0.45,
                rule_signal="hold",
                rule_engine_confidence=0.2,
                context_confidence=0.32,
                bias_confidence=0.32,
                setup_confidence=None,
                trigger_confidence=1.0,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
        min_symbol_support=1,
    )

    setup_row = _field_row(report, "setup_layer_confidence")
    assert setup_row["comparison_status"] == "missing_on_collapsed_only"
    assert setup_row["collapsed_final_hold_outcome"]["missing_row_count"] == 2
    assert setup_row["group_gap_summary"]["median_gap_share_of_rule_engine_confidence"] is None


def test_unsupported_path_does_not_emit_speculative_interpretation(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=0.9,
                rule_signal="long",
                rule_engine_confidence=0.84,
                context_confidence=0.72,
                bias_confidence=0.72,
                setup_confidence=0.66,
                trigger_confidence=1.0,
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
        min_symbol_support=1,
    )

    assert report["summary"]["preserved_final_directional_outcome_row_count"] == 1
    assert report["summary"]["collapsed_final_hold_outcome_row_count"] == 0
    assert report["final_interpretation"]["interpretation_status"] == (
        "comparison_unsupported"
    )
    assert "hidden aggregate" not in report["final_interpretation"]["explanation"]


def test_wrapper_entrypoint_smoke(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []
    for index in range(30):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T00:{index:02d}:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=0.92,
                rule_signal="long",
                rule_engine_confidence=0.86,
                context_confidence=0.74,
                bias_confidence=0.74,
                setup_confidence=0.69,
                trigger_confidence=1.0,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T01:{index:02d}:00+00:00",
                symbol="ETHUSDT",
                selected_strategy_confidence=0.46,
                rule_signal="hold",
                rule_engine_confidence=0.21,
                context_confidence=0.34,
                bias_confidence=0.34,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
    _write_jsonl(input_path, rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_contributor_diagnosis_report",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--config",
            "336/10000",
            "--min-symbol-support",
            "1",
        ],
    )
    runpy.run_path(str(Path(wrapper_module.__file__)), run_name="__main__")
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert captured["comparison_support_status"] == "supported"
    assert wrapper_module.build_report is report_module.build_report


def test_context_and_bias_family_alone_is_not_sufficient(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []
    for index in range(30):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T02:{index:02d}:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=0.72,
                rule_signal="long",
                rule_engine_confidence=0.86,
                context_confidence=0.74,
                bias_confidence=0.74,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T03:{index:02d}:00+00:00",
                symbol="ETHUSDT",
                selected_strategy_confidence=0.72,
                rule_signal="hold",
                rule_engine_confidence=0.21,
                context_confidence=0.34,
                bias_confidence=0.34,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["contributor_tracking"]["trigger_negative_control_status"] == (
        "trigger_remains_negative_control"
    )
    assert report["final_interpretation"]["strong_aligned_non_trigger_contributors"] == [
        "context_layer_confidence",
        "bias_layer_confidence",
    ]
    assert set(
        report["final_interpretation"]["strong_aligned_non_trigger_contributor_families"]
    ) == {"context_and_bias"}
    assert report["final_interpretation"]["interpretation_status"] == (
        "mixed_persisted_contributor_surface"
    )


def test_overlap_only_second_family_is_not_sufficient(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []
    for index in range(30):
        preserved_selected_strategy_confidence = 0.74 if index % 2 == 0 else 0.71
        collapsed_selected_strategy_confidence = 0.72 if index % 2 == 0 else 0.70

        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T04:{index:02d}:00+00:00",
                symbol="BTCUSDT",
                selected_strategy_confidence=preserved_selected_strategy_confidence,
                rule_signal="long",
                rule_engine_confidence=0.86,
                context_confidence=0.74,
                bias_confidence=0.74,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T05:{index:02d}:00+00:00",
                symbol="ETHUSDT",
                selected_strategy_confidence=collapsed_selected_strategy_confidence,
                rule_signal="hold",
                rule_engine_confidence=0.21,
                context_confidence=0.34,
                bias_confidence=0.34,
                setup_confidence=0.62,
                trigger_confidence=1.0,
            )
        )
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["contributor_tracking"]["trigger_negative_control_status"] == (
        "trigger_remains_negative_control"
    )
    assert set(
        report["final_interpretation"]["strong_aligned_non_trigger_contributor_families"]
    ) == {"context_and_bias"}
    assert set(
        report["final_interpretation"]["aligned_non_trigger_contributor_families"]
    ) == {"context_and_bias", "selected_strategy"}
    assert report["final_interpretation"]["interpretation_status"] == (
        "mixed_persisted_contributor_surface"
    )