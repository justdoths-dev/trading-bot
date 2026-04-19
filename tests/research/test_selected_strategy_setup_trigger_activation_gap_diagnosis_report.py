from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_setup_trigger_activation_gap_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_setup_trigger_activation_gap_diagnosis_report as report_module,
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


def _overall_group_row(report: dict, comparison_group: str) -> dict:
    rows = report["configuration_summaries"][0]["overall_group_summaries"]
    for row in rows:
        if row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(f"comparison_group not found: {comparison_group}")


def _strategy_group_row(report: dict, strategy: str, comparison_group: str) -> dict:
    rows = report["configuration_summaries"][0]["strategy_group_summaries"]
    for row in rows:
        if row["strategy"] == strategy and row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(
        f"strategy/comparison_group not found: {strategy}/{comparison_group}"
    )


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
                logged_at="2026-04-19T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            )
        ],
    )
    _write_jsonl(
        shard_path,
        [
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Context is bullish but confirmation is not ready.",
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

    assert effective_input_path.exists()
    assert effective_input_path.suffix == ".jsonl"
    assert len(_read_jsonl(effective_input_path)) == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True


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
    assert _overall_group_row(
        report,
        "all_actionable_selected_strategy_rows",
    )["row_count"] == 1


def test_separates_preserved_collapsed_other_actionable_and_other_non_actionable(
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
                setup_state="long_confirmed",
                setup_bias="bullish",
                trigger_state="long_confirmed",
                trigger_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
                rule_reason="Confirmation not ready.",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="XRPUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="short",
                setup_state="short_confirmed",
                setup_bias="bearish",
                trigger_state="short_confirmed",
                trigger_bias="bearish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:03:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="watchlist_long",
                rule_reason="Needs watchlist handling.",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["summary"]

    assert summary["actionable_selected_strategy_row_count"] == 4
    assert summary["preserved_aligned_actionable_row_count"] == 1
    assert summary["collapsed_to_hold_row_count"] == 1
    assert summary["other_actionable_rule_outcome_row_count"] == 1
    assert summary["other_non_actionable_rule_outcome_row_count"] == 1


def test_setup_trigger_and_dual_rates_are_computed_correctly(tmp_path: Path) -> None:
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
                rule_signal="long",
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
                rule_reason="Both layers remain neutral.",
                setup_state="neutral",
                setup_bias="neutral",
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

    preserved = _overall_group_row(report, "preserved_aligned_actionable")
    collapsed = _overall_group_row(report, "collapsed_to_hold")

    assert preserved["setup_activation_rate"] == 1.0
    assert preserved["trigger_activation_rate"] == 0.5
    assert preserved["dual_activation_rate"] == 0.5
    assert preserved["dual_neutral_or_missing_rate"] == 0.0

    assert collapsed["setup_activation_rate"] == 0.0
    assert collapsed["trigger_activation_rate"] == 0.5
    assert collapsed["dual_activation_rate"] == 0.0
    assert collapsed["dual_neutral_or_missing_rate"] == 0.5


def test_strategy_grouping_works(tmp_path: Path) -> None:
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
                rule_reason="Confirmation missing.",
                setup_state="neutral",
                setup_bias="neutral",
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
                rule_reason="Confirmation missing.",
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

    swing_preserved = _strategy_group_row(
        report, "swing", "preserved_aligned_actionable"
    )
    swing_collapsed = _strategy_group_row(report, "swing", "collapsed_to_hold")
    intraday_collapsed = _strategy_group_row(report, "intraday", "collapsed_to_hold")

    assert swing_preserved["row_count"] == 1
    assert swing_preserved["dual_activation_rate"] == 1.0
    assert swing_collapsed["row_count"] == 1
    assert swing_collapsed["dual_neutral_or_missing_rate"] == 1.0
    assert intraday_collapsed["row_count"] == 1


def test_symbol_grouping_respects_minimum_support_threshold(tmp_path: Path) -> None:
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
                rule_reason="Confirmation missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="BTCUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Confirmation missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="neutral",
                trigger_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Confirmation missing.",
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
        min_symbol_support=2,
    )
    symbol_rows = report["configuration_summaries"][0]["symbol_group_summaries"]
    labels = {
        (row["symbol"], row["strategy"], row["comparison_group"]): row["row_count"]
        for row in symbol_rows
    }

    assert labels == {("BTCUSDT", "intraday", "collapsed_to_hold"): 2}


def test_evidence_source_split_is_setup_trigger_centered_and_reported(
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
                rule_signal="hold",
                rule_reason="Confirmation missing.",
            ),
            _raw_record(
                logged_at="2026-04-19T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Confirmation missing.",
                context_state="supports_selected_signal",
                context_bias="bullish",
            ),
            _raw_record(
                logged_at="2026-04-19T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="Confirmation missing.",
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
    collapsed = _overall_group_row(report, "collapsed_to_hold")

    assert collapsed["evidence_source_counts"] == {
        "reason_text_only": 2,
        "structured_state_backed": 1,
    }
    assert report["summary"]["evidence_source_counts"] == {
        "reason_text_only": 2,
        "structured_state_backed": 1,
    }


def test_final_assessment_requires_supported_primary_comparison_for_strong_conclusion(
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
                rule_reason="Confirmation missing.",
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

    conclusion = report["final_assessment"]["overall_conclusion"]
    facts = report["final_assessment"]["facts"]
    inferences = report["final_assessment"]["inferences"]

    assert "limited support" in conclusion
    assert any("limited support" in text for text in inferences)
    assert all("Strategy comparison highlight:" not in text for text in facts)


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
                rule_reason="Confirmation missing.",
                setup_state="neutral",
                setup_bias="neutral",
                trigger_state="neutral",
                trigger_bias="neutral",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_setup_trigger_activation_gap_diagnosis_report",
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
    assert "collapsed_to_hold_row_count" in captured
    assert wrapper_module.build_report is report_module.build_report