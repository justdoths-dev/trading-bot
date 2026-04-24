from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_a_shadow_recovery_report as patch_class_a_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report as report_module,
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


def _pocket_row(
    *,
    logged_at: str,
    symbol: str,
    rule_engine_confidence: float,
    context_delta: float,
    setup_delta: float,
    selected_strategy_delta: float = 0.10,
    trigger_delta: float = 0.15,
) -> dict:
    return _raw_record(
        logged_at=logged_at,
        symbol=symbol,
        rule_signal="hold",
        rule_engine_confidence=rule_engine_confidence,
        context_confidence=_low_value(context_delta),
        bias_confidence=_low_value(context_delta),
        selected_strategy_confidence=_low_value(selected_strategy_delta),
        setup_confidence=_high_value(setup_delta),
        trigger_confidence=_high_value(trigger_delta),
    )


def _preserved_row(*, logged_at: str, symbol: str) -> dict:
    return _raw_record(
        logged_at=logged_at,
        symbol=symbol,
        rule_signal="long",
        rule_engine_confidence=0.56,
        context_confidence=_low_value(0.02),
        bias_confidence=_low_value(0.02),
        selected_strategy_confidence=_low_value(0.02),
        setup_confidence=_high_value(0.12),
        trigger_confidence=_high_value(0.15),
    )


def _write_rows(path: Path, *, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def _build_report(input_path: Path, output_dir: Path) -> dict:
    return report_module.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )


def _shadow_row(rows: list[dict], symbol_prefix: str) -> dict:
    return next(row for row in rows if str(row["symbol"]).startswith(symbol_prefix))


def _narrower_than_a_rows(repeats: int = 15) -> list[dict]:
    rows: list[dict] = []
    for index in range(repeats):
        rows.extend(
            [
                _pocket_row(
                    logged_at=f"2026-04-24T00:{index:02d}:00+00:00",
                    symbol=f"B1-00-{index:02d}",
                    rule_engine_confidence=0.21,
                    context_delta=0.02,
                    setup_delta=0.08,
                ),
                _pocket_row(
                    logged_at=f"2026-04-24T01:{index:02d}:00+00:00",
                    symbol=f"B1-01-{index:02d}",
                    rule_engine_confidence=0.22,
                    context_delta=0.03,
                    setup_delta=0.09,
                ),
                _pocket_row(
                    logged_at=f"2026-04-24T02:{index:02d}:00+00:00",
                    symbol=f"AONLY-00-{index:02d}",
                    rule_engine_confidence=0.20,
                    context_delta=0.06,
                    setup_delta=0.05,
                ),
                _pocket_row(
                    logged_at=f"2026-04-24T03:{index:02d}:00+00:00",
                    symbol=f"AONLY-01-{index:02d}",
                    rule_engine_confidence=0.22,
                    context_delta=0.07,
                    setup_delta=0.05,
                ),
                _preserved_row(
                    logged_at=f"2026-04-24T04:{index:02d}:00+00:00",
                    symbol=f"PRESERVED-00-{index:02d}",
                ),
                _preserved_row(
                    logged_at=f"2026-04-24T05:{index:02d}:00+00:00",
                    symbol=f"PRESERVED-01-{index:02d}",
                ),
            ]
        )
    return rows


def test_b1_can_rescue_a_narrower_subset_than_a1_and_improve_direct_edge_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(input_path, rows=_narrower_than_a_rows())

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
    comparator_summary = report["patch_class_a_comparator_summary"]

    assert report["summary"]["comparison_support_status"] == "supported"

    assert shadow_summary["baseline_pocket_row_count"] == 60
    assert shadow_summary["shadow_pocket_row_count"] == 30
    assert shadow_summary["rescued_from_pocket_row_count"] == 30
    assert shadow_summary["changed_outside_target_class_row_count"] == 0
    assert shadow_summary["rescued_preserved_row_count"] == 0

    assert comparator_summary["patch_class_a_rescued_from_pocket_row_count"] == 60
    assert comparator_summary["patch_class_b_rescued_from_pocket_row_count"] == 30
    assert comparator_summary["rescued_only_in_patch_class_a_row_count"] == 30
    assert comparator_summary["rescued_only_in_patch_class_b_row_count"] == 0
    assert comparator_summary["scope_relation_status"] == "strict_subset_of_patch_class_a"
    assert comparator_summary["strictly_narrower_than_patch_class_a"] is True

    assert report["interpretation"]["joint_selectivity_status"] == (
        "joint_selectivity_supported_with_direct_edge_improvement"
    )
    assert report["interpretation_status"] == "patch_class_b_shadow_supported"


def test_rows_that_fail_the_joint_gate_remain_unchanged() -> None:
    raw_records = [
        _pocket_row(
            logged_at="2026-04-24T06:00:00+00:00",
            symbol="BLOCKED",
            rule_engine_confidence=0.20,
            context_delta=0.06,
            setup_delta=0.05,
        ),
        _preserved_row(
            logged_at="2026-04-24T06:01:00+00:00",
            symbol="PRESERVED-00",
        ),
    ]

    comparison_rows = patch_class_a_module._prepare_comparison_rows(raw_records)
    patch_class_a_shadow_rows = patch_class_a_module._apply_patch_class_a_shadow_candidate(
        comparison_rows
    )
    shadow_rows = report_module._apply_patch_class_b_shadow_candidate(comparison_rows)

    blocked = _shadow_row(shadow_rows, "BLOCKED")
    blocked_under_a = _shadow_row(patch_class_a_shadow_rows, "BLOCKED")

    assert blocked["patch_class_b_shadow_eligible"] is False
    assert (
        blocked["patch_class_b_shadow_block_reason"]
        == "setup_margin_not_above_context_bias_shortfall"
    )
    assert blocked["shadow_rule_engine_confidence"] == blocked["actual_rule_engine_confidence"]
    assert blocked["rescued_from_pocket"] is False

    assert blocked_under_a["patch_class_a_shadow_eligible"] is True
    assert blocked_under_a["rescued_from_pocket"] is True


def test_direct_edge_selection_can_stay_unchanged_and_report_remains_leaning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(input_path, rows=_narrower_than_a_rows())

    monkeypatch.setattr(
        report_module,
        "_build_direct_edge_selection_summary",
        lambda **_: {
            "status": report_module._DIRECT_EDGE_SELECTION_AVAILABLE,
            "baseline_edge_selection_count": 1,
            "shadow_edge_selection_count": 1,
            "net_change": 0,
        },
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["direct_edge_selection_summary"]["status"] == (
        report_module._DIRECT_EDGE_SELECTION_AVAILABLE
    )
    assert report["direct_edge_selection_summary"]["net_change"] == 0
    assert report["interpretation"]["joint_selectivity_status"] == (
        "joint_selectivity_narrows_rescue_but_direct_edge_unproven"
    )
    assert report["interpretation_status"] == "patch_class_b_shadow_leaning"


def test_preserved_rescue_and_non_target_change_remain_zero() -> None:
    raw_records = [
        _pocket_row(
            logged_at="2026-04-24T07:00:00+00:00",
            symbol="ELIGIBLE",
            rule_engine_confidence=0.21,
            context_delta=0.02,
            setup_delta=0.08,
        ),
        _raw_record(
            logged_at="2026-04-24T07:01:00+00:00",
            symbol="PRESERVED-POCKET",
            rule_signal="long",
            rule_engine_confidence=0.24,
            context_confidence=_low_value(0.02),
            bias_confidence=_low_value(0.02),
            selected_strategy_confidence=_low_value(0.10),
            setup_confidence=_high_value(0.08),
            trigger_confidence=_high_value(0.15),
        ),
        _preserved_row(
            logged_at="2026-04-24T07:02:00+00:00",
            symbol="PRESERVED-OUTSIDE",
        ),
    ]

    comparison_rows = patch_class_a_module._prepare_comparison_rows(raw_records)
    shadow_rows = report_module._apply_patch_class_b_shadow_candidate(comparison_rows)
    shadow_summary = report_module.build_shadow_summary(
        comparison_rows=comparison_rows,
        shadow_rows=shadow_rows,
    )

    assert shadow_summary["rescued_preserved_row_count"] == 0
    assert shadow_summary["rescued_outside_target_class_row_count"] == 0
    assert shadow_summary["changed_outside_target_class_row_count"] == 0
    assert shadow_summary["rescued_collapsed_row_count"] == 1


def test_wrapper_import_path_and_entrypoint_smoke(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(input_path, rows=_narrower_than_a_rows())

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
            "selected_strategy_rule_engine_confidence_patch_class_b_shadow_recovery_report",
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
    assert captured["rescued_from_pocket_row_count"] == 30
    assert captured["scope_relation_status"] == "strict_subset_of_patch_class_a"
    assert captured["direct_edge_selection_status"] == (
        report_module._DIRECT_EDGE_SELECTION_AVAILABLE
    )
    assert wrapper_module.build_report is report_module.build_report