from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_dual_aligned_context_hold_residual_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_dual_aligned_context_hold_residual_diagnosis_report as report_module,
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


def _comparison_group_row(report: dict, comparison_group: str) -> dict:
    for row in report["configuration_summaries"][0]["comparison_group_summaries"]:
        if row["comparison_group"] == comparison_group:
            return row
    raise AssertionError(f"comparison_group not found: {comparison_group}")


def _strategy_context_row(
    report: dict,
    strategy: str,
    comparison_group: str,
    context_family: str,
) -> dict:
    for row in report["configuration_summaries"][0]["strategy_context_summaries"]:
        if (
            row["strategy"] == strategy
            and row["comparison_group"] == comparison_group
            and row["context_family"] == context_family
        ):
            return row
    raise AssertionError(
        "strategy/comparison_group/context_family not found: "
        f"{strategy}/{comparison_group}/{context_family}"
    )


def test_filters_only_dual_aligned_rows_from_actionable_selected_strategy_population(
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
                rule_reason="Trigger is active but setup still missing.",
                setup_state="neutral",
                setup_bias="neutral",
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
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )

    assert report["summary"]["actionable_selected_strategy_row_count"] == 2
    assert report["summary"]["dual_aligned_row_count"] == 1
    assert report["summary"]["preserved_dual_aligned_baseline_row_count"] == 1


def test_dual_aligned_comparison_group_split_preserved_collapsed_and_other(
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
                rule_reason="Broader context is neutral and not supportive.",
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
                rule_signal="short",
                rule_reason="Conflicting higher layer flipped the trade.",
                context_state="bearish_context",
                context_bias="bearish",
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

    assert report["summary"]["dual_aligned_row_count"] == 3
    assert report["summary"]["preserved_dual_aligned_baseline_row_count"] == 1
    assert report["summary"]["collapsed_dual_aligned_row_count"] == 1
    assert report["summary"]["other_dual_aligned_rule_outcome_row_count"] == 1


def test_context_state_relation_classification_is_conservative() -> None:
    assert report_module._context_state_relation_to_selected(
        context_state="bullish_context",
        selected_signal="long",
    ) == "aligned_with_selected"
    assert report_module._context_state_relation_to_selected(
        context_state="neutral",
        selected_signal="long",
    ) == "neutral_or_missing"
    assert report_module._context_state_relation_to_selected(
        context_state="bearish_context",
        selected_signal="long",
    ) == "opposite_to_selected"
    assert report_module._context_state_relation_to_selected(
        context_state="bullish_conflict_bearish",
        selected_signal="long",
    ) == "mixed_or_inconclusive"


def test_context_bias_relation_classification_is_conservative() -> None:
    assert report_module._context_bias_relation_to_selected(
        context_bias="bullish",
        selected_signal="long",
    ) == "aligned_with_selected"
    assert report_module._context_bias_relation_to_selected(
        context_bias="neutral",
        selected_signal="long",
    ) == "neutral_or_missing"
    assert report_module._context_bias_relation_to_selected(
        context_bias="bearish",
        selected_signal="long",
    ) == "opposite_to_selected"


def test_combined_context_relation_classification_handles_bias_only_and_conflict() -> None:
    assert report_module._combined_context_relation_to_selected(
        context_state_relation="neutral_or_missing",
        context_bias_relation="aligned_with_selected",
    ) == "aligned_with_selected"
    assert report_module._combined_context_relation_to_selected(
        context_state_relation="neutral_or_missing",
        context_bias_relation="opposite_to_selected",
    ) == "opposite_to_selected"
    assert report_module._combined_context_relation_to_selected(
        context_state_relation="aligned_with_selected",
        context_bias_relation="opposite_to_selected",
    ) == "mixed_or_inconclusive"


def test_context_family_classification_prioritizes_support_neutrality_and_opposition() -> None:
    assert report_module._context_family(
        context_state_relation="aligned_with_selected",
        context_bias_relation="aligned_with_selected",
    ) == "dual_context_aligned"
    assert report_module._context_family(
        context_state_relation="aligned_with_selected",
        context_bias_relation="neutral_or_missing",
    ) == "context_state_only_aligned"
    assert report_module._context_family(
        context_state_relation="neutral_or_missing",
        context_bias_relation="aligned_with_selected",
    ) == "context_bias_only_aligned"
    assert report_module._context_family(
        context_state_relation="neutral_or_missing",
        context_bias_relation="neutral_or_missing",
    ) == "dual_context_neutral_or_missing"
    assert report_module._context_family(
        context_state_relation="aligned_with_selected",
        context_bias_relation="opposite_to_selected",
    ) == "any_context_opposition"
    assert report_module._context_family(
        context_state_relation="mixed_or_inconclusive",
        context_bias_relation="neutral_or_missing",
    ) == "context_mixed_or_inconclusive"


def test_context_family_classification_keeps_mixed_state_above_opposition() -> None:
    assert report_module._context_family(
        context_state_relation="mixed_or_inconclusive",
        context_bias_relation="opposite_to_selected",
    ) == "context_mixed_or_inconclusive"


def test_reason_bucket_population_prefers_rule_reason_then_root_reason(
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
                rule_reason="Opposing structure invalidated the trade.",
                root_reason="Broader context is neutral and not supportive.",
                context_state="neutral",
                context_bias="neutral",
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
                root_reason="Broader context is neutral and not supportive.",
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
    collapsed_row = _comparison_group_row(report, "collapsed_dual_aligned")

    assert collapsed_row["reason_bucket_counts"] == {
        "opposition_or_invalidated": 1,
        "context_not_supportive": 1,
    }
    assert collapsed_row["rule_reason_bucket_counts"] == {
        "opposition_or_invalidated": 1,
        "insufficient_explanation": 1,
    }
    assert collapsed_row["root_reason_bucket_counts"] == {
        "context_not_supportive": 2,
    }
    assert collapsed_row["reason_bucket_source_counts"] == {
        "rule_reason_text": 1,
        "root_reason_text": 1,
    }


def test_strategy_level_context_grouping_works(tmp_path: Path) -> None:
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
                rule_reason="Broader context is neutral and not supportive.",
                context_state="neutral",
                context_bias="neutral",
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
                rule_reason="Broader context is neutral and not supportive.",
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

    assert _strategy_context_row(
        report,
        "scalping",
        "collapsed_dual_aligned",
        "dual_context_neutral_or_missing",
    )["row_count"] == 2
    assert _strategy_context_row(
        report,
        "swing",
        "preserved_dual_aligned_baseline",
        "dual_context_aligned",
    )["row_count"] == 1


def test_symbol_level_context_grouping_respects_minimum_support_threshold(
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
                rule_reason="Opposing structure invalidated the trade.",
                context_state="bearish_context",
                context_bias="bearish",
                setup_state="long_confirmed",
                setup_bias="bullish",
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
                rule_reason="Opposing structure invalidated the trade.",
                context_state="bearish_context",
                context_bias="bearish",
                setup_state="long_confirmed",
                setup_bias="bullish",
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
                rule_reason="Opposing structure invalidated the trade.",
                context_state="bearish_context",
                context_bias="bearish",
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
    symbol_rows = report["configuration_summaries"][0]["symbol_context_summaries"]

    assert [
        (
            row["symbol"],
            row["strategy"],
            row["comparison_group"],
            row["context_family"],
            row["row_count"],
        )
        for row in symbol_rows
    ] == [
        (
            "BTCUSDT",
            "intraday",
            "collapsed_dual_aligned",
            "any_context_opposition",
            2,
        )
    ]


def test_context_family_comparison_is_populated_for_preserved_vs_collapsed(
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
                rule_reason="Broader context is neutral and not supportive.",
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
    comparison = report["context_family_comparison"]

    assert comparison["preserved_dual_aligned_baseline_row_count"] == 1
    assert comparison["collapsed_dual_aligned_row_count"] == 1
    assert comparison["context_family_counts"]["preserved_dual_aligned_baseline"] == {
        "dual_context_aligned": 1
    }
    assert comparison["context_family_counts"]["collapsed_dual_aligned"] == {
        "dual_context_neutral_or_missing": 1
    }
    assert comparison["combined_context_relation_counts"]["collapsed_dual_aligned"] == {
        "neutral_or_missing": 1
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
                rule_reason="Broader context is neutral and not supportive.",
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
                rule_reason="Broader context is neutral and not supportive.",
                context_state="neutral",
                context_bias="neutral",
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
            "selected_strategy_dual_aligned_context_hold_residual_diagnosis_report",
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
    assert "dual_aligned_row_count" in captured
    assert wrapper_module.build_report is report_module.build_report