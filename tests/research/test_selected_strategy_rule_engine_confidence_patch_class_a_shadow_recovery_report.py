from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _low_value(delta: float) -> float:
    threshold = float(report_module._LOW_CONFIDENCE_THRESHOLD)
    candidate = round(max(0.0, threshold - delta), 6)
    if candidate >= threshold:
        candidate = round(max(0.0, threshold - 0.01), 6)
    return candidate


def _high_value(delta: float) -> float:
    threshold = float(report_module._LOW_CONFIDENCE_THRESHOLD)
    candidate = round(min(1.0, threshold + delta), 6)
    if candidate <= threshold:
        candidate = round(min(1.0, threshold + 0.01), 6)
    return candidate


def _raw_record(
    *,
    logged_at: str,
    symbol: str,
    strategy: str = "intraday",
    bias: str = "bullish",
    selected_strategy_signal: str = "long",
    selected_strategy_confidence: float | None = 0.60,
    rule_signal: str = "long",
    rule_engine_confidence: float | None = 0.58,
    context_confidence: float | None = 0.60,
    bias_confidence: float | None = 0.60,
    setup_confidence: float | None = 0.82,
    trigger_confidence: float | None = 0.96,
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
            "bias": bias,
        },
        "bias_layer": {
            "context": "bullish_context",
            "bias": bias,
        },
        "setup_layer": {
            "setup": "long_confirmed",
            "bias": bias,
        },
        "trigger_layer": {
            "trigger": "long_confirmed",
            "bias": bias,
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


def _write_rows(path: Path, *, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def _build_report(
    input_path: Path,
    output_dir: Path,
) -> dict:
    return report_module.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )


def _shadow_row(rows: list[dict], symbol: str) -> dict:
    return next(row for row in rows if row["symbol"] == symbol)


def test_eligible_pocket_rows_receive_uplift_and_are_rescued(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []
    for index in range(12):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-23T00:{index:02d}:00+00:00",
                symbol=f"POCKET-{index:02d}",
                rule_signal="hold",
                rule_engine_confidence=0.20 if index % 2 == 0 else 0.24,
                context_confidence=_low_value(0.15),
                bias_confidence=_low_value(0.15),
                selected_strategy_confidence=_low_value(0.12),
                setup_confidence=_high_value(0.09),
                trigger_confidence=_high_value(0.12),
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-23T01:{index:02d}:00+00:00",
                symbol=f"PRESERVED-{index:02d}",
                rule_signal="long",
                rule_engine_confidence=0.56,
                context_confidence=_low_value(0.02),
                bias_confidence=_low_value(0.02),
                selected_strategy_confidence=_low_value(0.03),
                setup_confidence=_high_value(0.10),
                trigger_confidence=_high_value(0.12),
            )
        )
    _write_rows(input_path, rows=rows)

    monkeypatch.setattr(
        report_module,
        "_build_direct_edge_selection_summary",
        lambda **_: {
            "status": report_module._DIRECT_EDGE_SELECTION_AVAILABLE,
            "baseline_edge_selection_count": 0,
            "shadow_edge_selection_count": 1,
            "net_change": 1,
        },
    )

    report = _build_report(input_path, tmp_path / "reports")
    shadow_summary = report["shadow_summary"]

    assert shadow_summary["baseline_pocket_row_count"] == 12
    assert shadow_summary["shadow_pocket_row_count"] == 0
    assert shadow_summary["rescued_from_pocket_row_count"] == 12
    assert shadow_summary["changed_outside_target_class_row_count"] == 0
    assert shadow_summary["rescued_preserved_row_count"] == 0
    assert shadow_summary["rescued_collapsed_row_count"] == 12
    assert shadow_summary["rescued_row_summaries"]["setup_layer_confidence"][
        "row_count"
    ] == 12
    assert report["interpretation_status"] == "patch_class_a_shadow_supported"


def test_ineligible_rows_are_unchanged_and_noncollapsed_rows_are_blocked() -> None:
    raw_records = [
        _raw_record(
            logged_at="2026-04-23T02:00:00+00:00",
            symbol="ELIGIBLE",
            rule_signal="hold",
            rule_engine_confidence=0.24,
            context_confidence=_low_value(0.12),
            bias_confidence=_low_value(0.12),
            selected_strategy_confidence=_low_value(0.10),
            setup_confidence=_high_value(0.08),
            trigger_confidence=_high_value(0.15),
        ),
        _raw_record(
            logged_at="2026-04-23T02:00:30+00:00",
            symbol="PRESERVED-POCKET",
            rule_signal="long",
            rule_engine_confidence=0.24,
            context_confidence=_low_value(0.12),
            bias_confidence=_low_value(0.12),
            selected_strategy_confidence=_low_value(0.10),
            setup_confidence=_high_value(0.08),
            trigger_confidence=_high_value(0.15),
        ),
        _raw_record(
            logged_at="2026-04-23T02:01:00+00:00",
            symbol="LOW-TRIGGER",
            rule_signal="hold",
            rule_engine_confidence=0.24,
            context_confidence=_low_value(0.12),
            bias_confidence=_low_value(0.12),
            selected_strategy_confidence=_low_value(0.10),
            setup_confidence=_high_value(0.08),
            trigger_confidence=_low_value(0.05),
        ),
        _raw_record(
            logged_at="2026-04-23T02:02:00+00:00",
            symbol="WRONG-SIGNATURE",
            rule_signal="hold",
            rule_engine_confidence=0.24,
            context_confidence=_low_value(0.12),
            bias_confidence=_low_value(0.12),
            selected_strategy_confidence=_low_value(0.10),
            setup_confidence=_low_value(0.04),
            trigger_confidence=_high_value(0.15),
        ),
        _raw_record(
            logged_at="2026-04-23T02:03:00+00:00",
            symbol="PRESERVED-OUTSIDE",
            rule_signal="long",
            rule_engine_confidence=0.56,
            context_confidence=_low_value(0.02),
            bias_confidence=_low_value(0.02),
            selected_strategy_confidence=_low_value(0.02),
            setup_confidence=_high_value(0.12),
            trigger_confidence=_high_value(0.15),
        ),
    ]

    comparison_rows = report_module._prepare_comparison_rows(raw_records)
    shadow_rows = report_module._apply_patch_class_a_shadow_candidate(comparison_rows)

    eligible = _shadow_row(shadow_rows, "ELIGIBLE")
    preserved_pocket = _shadow_row(shadow_rows, "PRESERVED-POCKET")
    low_trigger = _shadow_row(shadow_rows, "LOW-TRIGGER")
    wrong_signature = _shadow_row(shadow_rows, "WRONG-SIGNATURE")

    assert eligible["patch_class_a_shadow_eligible"] is True
    assert eligible["shadow_rule_engine_confidence"] > eligible["actual_rule_engine_confidence"]
    assert eligible["rescued_from_pocket"] is True

    assert preserved_pocket["patch_class_a_shadow_eligible"] is False
    assert (
        preserved_pocket["patch_class_a_shadow_block_reason"]
        == "comparison_group_not_collapsed"
    )
    assert (
        preserved_pocket["shadow_rule_engine_confidence"]
        == preserved_pocket["actual_rule_engine_confidence"]
    )

    assert low_trigger["patch_class_a_shadow_eligible"] is False
    assert (
        low_trigger["patch_class_a_shadow_block_reason"]
        == "trigger_below_low_confidence_threshold"
    )
    assert (
        low_trigger["shadow_rule_engine_confidence"]
        == low_trigger["actual_rule_engine_confidence"]
    )

    assert wrong_signature["patch_class_a_shadow_eligible"] is False
    assert wrong_signature["patch_class_a_shadow_block_reason"] == "signature_mismatch"
    assert (
        wrong_signature["shadow_rule_engine_confidence"]
        == wrong_signature["actual_rule_engine_confidence"]
    )


def test_direct_edge_selection_count_unavailable_is_explicit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        rows=[
            _raw_record(
                logged_at="2026-04-23T03:00:00+00:00",
                symbol="POCKET-00",
                rule_signal="hold",
                rule_engine_confidence=0.24,
                context_confidence=_low_value(0.12),
                bias_confidence=_low_value(0.12),
                selected_strategy_confidence=_low_value(0.10),
                setup_confidence=_high_value(0.08),
                trigger_confidence=_high_value(0.15),
            ),
            _raw_record(
                logged_at="2026-04-23T03:01:00+00:00",
                symbol="PRESERVED-00",
                rule_signal="long",
                rule_engine_confidence=0.56,
                context_confidence=_low_value(0.02),
                bias_confidence=_low_value(0.02),
                selected_strategy_confidence=_low_value(0.02),
                setup_confidence=_high_value(0.12),
                trigger_confidence=_high_value(0.15),
            ),
        ],
    )

    monkeypatch.setattr(
        report_module,
        "_build_direct_edge_selection_summary",
        lambda **_: {
            "status": report_module._DIRECT_EDGE_SELECTION_UNAVAILABLE,
            "reason": "snapshot_execution_failed: synthetic-test",
        },
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["direct_edge_selection_summary"]["status"] == (
        report_module._DIRECT_EDGE_SELECTION_UNAVAILABLE
    )
    assert report["interpretation"]["direct_edge_selection_status"] == (
        report_module._DIRECT_EDGE_SELECTION_UNAVAILABLE
    )
    assert report["interpretation_status"] == "patch_class_a_shadow_leaning"


def test_wrapper_import_path_and_entrypoint_smoke(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        rows=[
            _raw_record(
                logged_at="2026-04-23T04:00:00+00:00",
                symbol="POCKET-00",
                rule_signal="hold",
                rule_engine_confidence=0.24,
                context_confidence=_low_value(0.12),
                bias_confidence=_low_value(0.12),
                selected_strategy_confidence=_low_value(0.10),
                setup_confidence=_high_value(0.08),
                trigger_confidence=_high_value(0.15),
            ),
            _raw_record(
                logged_at="2026-04-23T04:01:00+00:00",
                symbol="PRESERVED-00",
                rule_signal="long",
                rule_engine_confidence=0.56,
                context_confidence=_low_value(0.02),
                bias_confidence=_low_value(0.02),
                selected_strategy_confidence=_low_value(0.02),
                setup_confidence=_high_value(0.12),
                trigger_confidence=_high_value(0.15),
            ),
        ],
    )

    monkeypatch.setattr(
        report_module,
        "_build_direct_edge_selection_summary",
        lambda **_: {
            "status": report_module._DIRECT_EDGE_SELECTION_AVAILABLE,
            "baseline_edge_selection_count": 0,
            "shadow_edge_selection_count": 1,
            "net_change": 1,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report",
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
    assert captured["rescued_from_pocket_row_count"] == 1
    assert captured["direct_edge_selection_status"] == (
        report_module._DIRECT_EDGE_SELECTION_AVAILABLE
    )
    assert wrapper_module.build_report is report_module.build_report
