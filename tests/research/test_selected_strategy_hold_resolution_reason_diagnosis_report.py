from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_hold_resolution_reason_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_hold_resolution_reason_diagnosis_report as report_module,
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
    selected_strategy_result: dict | None = None,
    rule_signal: str | None = None,
    rule_bias: str | None = None,
    rule_reason: str | None = None,
    root_reason: str | None = None,
    execution_action: str | None = "hold",
    execution_signal: str | None = None,
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

    rule_engine: dict[str, object] = {}
    if rule_bias is not None:
        rule_engine["bias"] = rule_bias
    elif bias is not None:
        rule_engine["bias"] = bias
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal
    if rule_reason is not None:
        rule_engine["reason"] = rule_reason

    execution: dict[str, object] = {}
    if execution_action is not None:
        execution["action"] = execution_action
    if execution_signal is not None:
        execution["signal"] = execution_signal
    elif execution_action is not None:
        execution["signal"] = execution_action

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
        "execution": execution,
        **strategy_payloads,
    }
    if timeframe_summary:
        record["timeframe_summary"] = timeframe_summary
    if root_reason is not None:
        record["reason"] = root_reason
    if selected_strategy_result is not None:
        record["selected_strategy_result"] = selected_strategy_result
    return record


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
                rule_signal="hold",
                rule_reason="Broader structure is neutral, so the engine blocks directional trades instead of promoting lower-timeframe signals on their own.",
                context_state="neutral",
                context_bias="neutral",
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
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
                rule_reason="Higher timeframes are conflicted, so the engine blocks directional trades until broader structure improves.",
                context_state="conflicted",
                context_bias="neutral",
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
        "_effective_selected_strategy_hold_resolution_reason_input.jsonl"
    )
    assert len(_read_jsonl(effective_input_path)) == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True


def test_target_row_filtering_excludes_non_target_rows(tmp_path: Path) -> None:
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
                rule_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:02:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:03:00+00:00",
                symbol="BNBUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:04:00+00:00",
                symbol="ADAUSDT",
                strategy="intraday",
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
    targeting = report["configuration_summaries"][0]["hold_resolution_targeting"]

    assert targeting["selected_strategy_actionable_rows"] == 3
    assert targeting["selected_strategy_non_actionable_rows"] == 1
    assert targeting["selected_strategy_unobservable_rows"] == 1
    assert targeting["actionable_selected_strategy_rule_hold_rows"] == 2
    assert targeting["actionable_selected_strategy_rule_non_hold_rows"] == 1
    assert targeting["target_row_count"] == 2


def test_rule_signal_is_sourced_only_from_rule_engine_not_final_action(
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
                execution_action="hold",
                execution_signal="hold",
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal=None,
                execution_action="hold",
                execution_signal="hold",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    targeting = report["configuration_summaries"][0]["hold_resolution_targeting"]

    assert targeting["target_row_count"] == 0
    assert targeting["actionable_selected_strategy_rule_non_hold_rows"] == 1
    assert targeting["actionable_selected_strategy_rule_signal_unobservable_rows"] == 1


def test_rule_signal_uses_raw_rule_engine_signal_even_when_normalized_rule_signal_disagrees(
    tmp_path: Path,
    monkeypatch,
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
                execution_action="hold",
                execution_signal="hold",
            )
        ],
    )

    original_normalize_record = report_module.normalize_record

    def fake_normalize_record(raw_record: dict) -> dict:
        normalized = dict(original_normalize_record(raw_record))
        normalized["rule_signal"] = "hold"
        return normalized

    monkeypatch.setattr(report_module, "normalize_record", fake_normalize_record)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    targeting = report["configuration_summaries"][0]["hold_resolution_targeting"]

    assert targeting["target_row_count"] == 0
    assert targeting["actionable_selected_strategy_rule_non_hold_rows"] == 1


def test_observable_field_inventory_includes_only_fields_actually_present(
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
                rule_signal="hold",
                rule_reason="Higher timeframes are conflicted, so the engine blocks directional trades until broader structure improves.",
                context_state="conflicted",
                context_bias="neutral",
                setup_state="long",
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="swing",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    inventory = report["configuration_summaries"][0][
        "observable_decision_layer_inventory"
    ]
    field_paths = {
        row["field_path"] for row in inventory["field_presence_rows"]
    }

    assert "selected_strategy_payload.signal" in field_paths
    assert "rule_engine.signal" in field_paths
    assert "rule_engine.reason" in field_paths
    assert "timeframe_summary.context_layer.context" in field_paths
    assert "timeframe_summary.setup_layer.setup" in field_paths
    assert "timeframe_summary.trigger_layer.trigger" not in field_paths
    assert "execution.action" not in field_paths


def test_explicit_observable_condition_rows_stay_separate_from_insufficient_explanation(
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
                rule_signal="hold",
                rule_reason="Higher timeframes are conflicted, so the engine blocks directional trades until broader structure improves.",
                context_state="conflicted",
                context_bias="neutral",
            ),
            _raw_record(
                logged_at="2026-04-14T00:01:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["hold_resolution_reason_summary"]

    assert summary["explicit_observable_condition_rows"] == 1
    assert summary["insufficient_persisted_explanation_rows"] == 1
    assert summary["condition_bucket_counts"]["explicit_context_conflict"] == 1
    assert summary["condition_bucket_counts"]["insufficient_persisted_explanation"] == 1


def test_root_reason_text_is_not_hidden_when_rule_reason_text_is_unclassified(
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
                rule_signal="hold",
                rule_reason="engine keeps the position on hold here",
                root_reason="Broader structure remains neutral, so stand aside until the regime changes.",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["hold_resolution_reason_summary"]

    assert summary["explicit_observable_condition_rows"] == 1
    assert summary["condition_bucket_counts"]["explicit_context_neutrality"] == 1


def test_unclassified_reason_text_does_not_create_explicit_bucket(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                rule_reason="engine pauses here for now",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["hold_resolution_reason_summary"]
    categories = {
        row["reason_text_category"]: row["count"]
        for row in summary["reason_text_category_count_rows"]
    }

    assert summary["explicit_observable_condition_rows"] == 0
    assert summary["insufficient_persisted_explanation_rows"] == 1
    assert summary["condition_bucket_counts"]["insufficient_persisted_explanation"] == 1
    assert categories["unclassified"] == 1


def test_state_variant_matching_is_more_robust_than_exact_matches(tmp_path: Path) -> None:
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
                rule_signal="hold",
                context_state="broader_neutral_regime",
                context_bias="neutral",
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]["hold_resolution_reason_summary"]

    assert summary["explicit_observable_condition_rows"] == 1
    assert summary["condition_bucket_counts"]["explicit_context_neutrality"] == 1


def test_bias_sign_breakdown_uses_raw_persisted_bias_fields_not_normalized_bias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
            symbol="BTCUSDT",
            strategy="intraday",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_bias="bullish",
            context_state="neutral",
            context_bias="neutral",
        )
        for index in range(10)
    ]
    _write_jsonl(input_path, rows)

    original_normalize_record = report_module.normalize_record

    def fake_normalize_record(raw_record: dict) -> dict:
        normalized = dict(original_normalize_record(raw_record))
        normalized["bias"] = "bearish"
        return normalized

    monkeypatch.setattr(report_module, "normalize_record", fake_normalize_record)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    row = report["configuration_summaries"][0][
        "hold_resolution_by_strategy_symbol_bias_sign"
    ][0]

    assert row["bias_sign"] == "bullish"


def test_mixed_or_inconclusive_bucket_is_conservative_when_multiple_categories_coexist(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-14T00:{index % 60:02d}:00+00:00",
            symbol=f"BTC{index}",
            strategy="intraday",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            context_state="neutral",
            context_bias="neutral",
            trigger_state="short",
        )
        for index in range(36)
    ]
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]["hold_resolution_reason_summary"]

    assert summary["dominant_observed_condition_bucket"] == "mixed_or_inconclusive"
    assert summary["primary_condition_bucket"] == "mixed_or_inconclusive"


def test_small_slices_are_support_threshold_guarded(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows = [
        _raw_record(
            logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
            symbol="BTCUSDT",
            strategy="scalping",
            bias="bullish",
            selected_strategy_signal="long",
            rule_signal="hold",
            rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
            context_state="bullish_trend",
            context_bias="bullish",
            setup_state="long",
            trigger_state="neutral",
        )
        for index in range(9)
    ]
    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    row = report["configuration_summaries"][0][
        "hold_resolution_by_strategy_symbol_bias_sign"
    ][0]

    assert row["target_row_count"] == 9
    assert row["support_status"] == "limited_support"
    assert row["primary_condition_bucket"] == "insufficient_support"


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
        selected_strategy_result={
            "selected_strategy": "intraday",
            "signal": "long",
            "bias": "bullish",
            "confidence": 0.82,
            "reason": "post-composition object",
            "timeframe_summary": {},
            "debug": {},
        },
    )
    row.pop("intraday_result", None)
    _write_jsonl(input_path, [row])

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    targeting = report["configuration_summaries"][0]["hold_resolution_targeting"]

    assert targeting["selected_strategy_unobservable_rows"] == 1
    assert targeting["target_row_count"] == 0


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
                rule_signal="hold",
                rule_reason="Bullish context exists, but setup and trigger are not both confirmed enough for execution.",
                context_state="bullish_trend",
                context_bias="bullish",
                setup_state="long",
                trigger_state="neutral",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_hold_resolution_reason_diagnosis_report",
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
    assert "primary_observed_hold_resolution_bucket" in captured
    assert wrapper_module.build_report is report_module.build_report