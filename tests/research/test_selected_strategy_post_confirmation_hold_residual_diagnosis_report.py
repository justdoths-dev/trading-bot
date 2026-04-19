from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_post_confirmation_hold_residual_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_post_confirmation_hold_residual_diagnosis_report as report_module,
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

    record = {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "execution": {"action": "hold", "signal": "hold"},
        **strategy_payloads,
    }
    if timeframe_summary:
        record["timeframe_summary"] = timeframe_summary
    if root_reason is not None:
        record["reason"] = root_reason
    return record


def _residual_group_row(report: dict, residual_group: str) -> dict:
    for row in report["configuration_summaries"][0]["residual_group_summaries"]:
        if row["residual_group"] == residual_group:
            return row
    raise AssertionError(f"residual_group not found: {residual_group}")


def _strategy_row(report: dict, strategy: str, residual_group: str) -> dict:
    for row in report["configuration_summaries"][0]["strategy_residual_summaries"]:
        if row["strategy"] == strategy and row["residual_group"] == residual_group:
            return row
    raise AssertionError(f"strategy/residual_group not found: {strategy}/{residual_group}")


def test_filters_only_actionable_selected_strategy_rows(tmp_path: Path) -> None:
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
                rule_reason="Confirmation is not ready.",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                rule_reason="Still neutral.",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal_present=False,
                rule_signal="long",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["summary"]["actionable_selected_strategy_row_count"] == 1
    assert report["summary"]["collapsed_to_hold_row_count"] == 1
    assert report["summary"]["residual_target_row_count"] == 1


def test_residual_group_classification_covers_baseline_target_and_reference_groups(
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
                rule_reason="Risk filter rejects the entry.",
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
                rule_reason="Confirmation is not ready.",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is active but setup still missing.",
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
                rule_reason="Opposing trigger invalidated the trade.",
                setup_state="short_confirmed",
                setup_bias="bearish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:05:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Both layers remain neutral.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert _residual_group_row(
        report,
        "preserved_dual_aligned_baseline",
    )["row_count"] == 1
    assert _residual_group_row(report, "collapsed_dual_aligned")["row_count"] == 1
    assert _residual_group_row(report, "collapsed_setup_only")["row_count"] == 1
    assert _residual_group_row(report, "collapsed_trigger_only")["row_count"] == 1
    assert _residual_group_row(report, "collapsed_any_opposition")["row_count"] == 1
    assert _residual_group_row(
        report,
        "collapsed_dual_neutral_or_missing",
    )["row_count"] == 1


def test_unknown_collapsed_activation_pattern_falls_back_to_mixed_group() -> None:
    assert report_module._classify_residual_group(
        comparison_group="collapsed_to_hold",
        activation_pattern="unexpected_pattern",
    ) == "collapsed_mixed_or_inconclusive"


def test_residual_target_row_count_excludes_dual_neutral_reference_rows(
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
                rule_signal="hold",
                rule_reason="Risk filter rejects the entry.",
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
                rule_reason="Confirmation is not ready.",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
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
                rule_reason="Opposing trigger invalidated the trade.",
                setup_state="short_confirmed",
                setup_bias="bearish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:04:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Both layers remain neutral.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["summary"]["collapsed_to_hold_row_count"] == 5
    assert report["summary"]["residual_target_row_count"] == 4
    assert report["summary"]["collapsed_dual_neutral_or_missing_row_count"] == 1


def test_context_relation_classification_is_state_first_and_conservative() -> None:
    assert report_module._context_relation_to_selected(
        context_state="bullish_context",
        context_bias="bullish",
        selected_signal="long",
    ) == "aligned_with_selected"
    assert report_module._context_relation_to_selected(
        context_state="neutral",
        context_bias="bullish",
        selected_signal="long",
    ) == "neutral_or_missing"
    assert report_module._context_relation_to_selected(
        context_state="bearish_context",
        context_bias="bearish",
        selected_signal="long",
    ) == "opposite_to_selected"
    assert report_module._context_relation_to_selected(
        context_state="bullish_conflict_bearish",
        context_bias="bullish",
        selected_signal="long",
    ) == "mixed_or_inconclusive"


def test_reason_text_bucketing_stays_conservative() -> None:
    assert (
        report_module._bucket_reason_text("Confirmation is not ready yet.")
        == "confirmation_not_ready_or_missing"
    )
    assert (
        report_module._bucket_reason_text("Signals conflict with each other.")
        == "conflict_or_disagreement"
    )
    assert (
        report_module._bucket_reason_text("Risk filter rejected the setup.")
        == "risk_or_filter_rejection"
    )
    assert (
        report_module._bucket_reason_text(
            "Broader context is neutral and not supportive."
        )
        == "context_not_supportive"
    )
    assert (
        report_module._bucket_reason_text("Opposing structure invalidated the trade.")
        == "opposition_or_invalidated"
    )
    assert report_module._bucket_reason_text(None) == "insufficient_explanation"


def test_strategy_level_residual_grouping_works(tmp_path: Path) -> None:
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
                rule_signal="hold",
                rule_reason="Risk filter rejects the entry.",
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
                rule_reason="Risk filter rejects the entry.",
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
                rule_signal="hold",
                rule_reason="Opposing trigger invalidated the trade.",
                setup_state="short_confirmed",
                setup_bias="bearish",
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

    assert _strategy_row(report, "scalping", "collapsed_dual_aligned")["row_count"] == 2
    assert _strategy_row(report, "swing", "collapsed_any_opposition")["row_count"] == 1


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
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
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
    symbol_rows = report["configuration_summaries"][0]["symbol_residual_summaries"]

    assert [
        (row["symbol"], row["strategy"], row["residual_group"], row["row_count"])
        for row in symbol_rows
    ] == [("BTCUSDT", "intraday", "collapsed_trigger_only", 2)]


def test_preserved_vs_collapsed_dual_aligned_comparison_is_populated_correctly(
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
                rule_reason="Risk filter rejects the entry.",
                context_state="neutral",
                context_bias="neutral",
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
    comparison = report["preserved_vs_collapsed_dual_aligned_comparison"]

    assert comparison["preserved_dual_aligned_baseline_row_count"] == 1
    assert comparison["collapsed_dual_aligned_row_count"] == 1
    assert comparison["context_relation_counts"][
        "preserved_dual_aligned_baseline"
    ] == {"aligned_with_selected": 1}
    assert comparison["context_relation_counts"]["collapsed_dual_aligned"] == {
        "neutral_or_missing": 1
    }
    assert comparison["reason_bucket_counts"]["collapsed_dual_aligned"] == {
        "risk_or_filter_rejection": 1
    }


def test_final_assessment_is_support_aware_for_small_dual_aligned_comparison(
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
                rule_reason="Risk filter rejects the entry.",
                context_state="neutral",
                context_bias="neutral",
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
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
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

    conclusion = report["final_assessment"]["overall_conclusion"]
    remains_unproven = report["final_assessment"]["remains_unproven"]

    assert "limited support" in conclusion
    assert any("limited support" in item for item in remains_unproven)


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
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_post_confirmation_hold_residual_diagnosis_report",
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
    assert "residual_target_row_count" in captured
    assert wrapper_module.build_report is report_module.build_report