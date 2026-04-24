from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report as report_module,
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


def _write_rows(
    path: Path,
    *,
    pocket_variants: list[dict],
    preserved_outside_variants: list[dict] | None = None,
    collapsed_outside_variants: list[dict] | None = None,
    preserved_background_variants: list[dict] | None = None,
    repeats: int = 36,
) -> None:
    preserved_outside_variants = preserved_outside_variants or []
    collapsed_outside_variants = collapsed_outside_variants or []
    preserved_background_variants = preserved_background_variants or [
        {
            "selected_strategy_confidence": _high_value(0.20),
            "rule_signal": "long",
            "rule_engine_confidence": 0.72,
            "context_confidence": _high_value(0.20),
            "bias_confidence": _high_value(0.20),
            "setup_confidence": _high_value(0.20),
            "trigger_confidence": 0.96,
        }
    ]

    rows: list[dict] = []
    for index in range(repeats):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-22T00:{index:02d}:00+00:00",
                symbol=f"POCKET-{index:02d}",
                rule_signal="hold",
                **pocket_variants[index % len(pocket_variants)],
            )
        )
        for variant_index, variant in enumerate(preserved_outside_variants):
            rows.append(
                _raw_record(
                    logged_at=f"2026-04-22T01:{index:02d}:00+00:00",
                    symbol=f"PRESERVED-SIG-{variant_index}-{index:02d}",
                    rule_signal="long",
                    **variant,
                )
            )
        for variant_index, variant in enumerate(collapsed_outside_variants):
            rows.append(
                _raw_record(
                    logged_at=f"2026-04-22T02:{index:02d}:00+00:00",
                    symbol=f"COLLAPSED-SIG-{variant_index}-{index:02d}",
                    rule_signal="hold",
                    **variant,
                )
            )
        for variant_index, variant in enumerate(preserved_background_variants):
            rows.append(
                _raw_record(
                    logged_at=f"2026-04-22T03:{index:02d}:00+00:00",
                    symbol=f"PRESERVED-BG-{variant_index}-{index:02d}",
                    **variant,
                )
            )
    _write_jsonl(path, rows)


def _build_report(input_path: Path, output_dir: Path) -> dict:
    return report_module.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )


def _field_row(report: dict, field: str) -> dict:
    comparison = report["selected_conditioned_comparison"]
    return next(
        row
        for row in comparison["field_comparisons"]
        if row["field"] == field
    )


def test_signature_conditioned_severity_depth_supported(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        pocket_variants=[
            {
                "selected_strategy_confidence": _low_value(0.16),
                "rule_engine_confidence": 0.19,
                "context_confidence": _low_value(0.18),
                "bias_confidence": _low_value(0.18),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            },
            {
                "selected_strategy_confidence": _low_value(0.10),
                "rule_engine_confidence": 0.22,
                "context_confidence": _low_value(0.12),
                "bias_confidence": _low_value(0.12),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            },
        ],
        preserved_outside_variants=[
            {
                "selected_strategy_confidence": _low_value(0.02),
                "rule_engine_confidence": 0.57,
                "context_confidence": _low_value(0.02),
                "bias_confidence": _low_value(0.02),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    severity = report["severity_depth_assessment"]

    assert severity["severity_depth_status"] == (
        "severity_conditioned_split_supported"
    )
    assert report["interpretation_status"] == (
        "signature_conditioned_severity_depth_supported"
    )
    assert severity["selected_reference_group_label"] == (
        report_module._SIGNATURE_CONDITIONED_PRESERVED_OUTSIDE_GROUP_LABEL
    )
    assert severity["signature_presence_status"] == "held_constant_by_design"
    assert severity["residual_depth_supported"] is True
    assert set(severity["primary_material_left_severity_fields"]) >= {
        report_module._CONTEXT_BIAS_FAMILY_FIELD,
        "selected_strategy_confidence",
        report_module._RESIDUAL_FIELD,
    }
    assert _field_row(report, report_module._CONTEXT_BIAS_FAMILY_FIELD)[
        "left_group_more_severe_by_orientation"
    ] is True
    assert _field_row(report, "selected_strategy_confidence")[
        "left_group_more_severe_by_orientation"
    ] is True


def test_signature_conditioned_similar_depth_stays_inconclusive(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        pocket_variants=[
            {
                "selected_strategy_confidence": _low_value(0.03),
                "rule_engine_confidence": 0.24,
                "context_confidence": _low_value(0.03),
                "bias_confidence": _low_value(0.03),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
        preserved_outside_variants=[
            {
                "selected_strategy_confidence": _low_value(0.02),
                "rule_engine_confidence": 0.28,
                "context_confidence": _low_value(0.02),
                "bias_confidence": _low_value(0.02),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    severity = report["severity_depth_assessment"]

    assert severity["severity_depth_status"] == (
        "severity_conditioned_split_inconclusive"
    )
    assert report["interpretation_status"] == (
        "signature_conditioned_severity_depth_inconclusive"
    )
    assert severity["signature_depth_material_field_count"] == 0
    assert severity["residual_depth_supported"] is False


def test_absent_conditioned_preserved_outside_reference_uses_conservative_fallback(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        pocket_variants=[
            {
                "selected_strategy_confidence": _low_value(0.16),
                "rule_engine_confidence": 0.19,
                "context_confidence": _low_value(0.18),
                "bias_confidence": _low_value(0.18),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
        preserved_outside_variants=[],
        collapsed_outside_variants=[
            {
                "selected_strategy_confidence": _low_value(0.02),
                "rule_engine_confidence": 0.31,
                "context_confidence": _low_value(0.02),
                "bias_confidence": _low_value(0.02),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    reference = report["reference_selection"]
    severity = report["severity_depth_assessment"]

    assert reference["primary_support_status"] == "insufficient_data"
    assert reference["fallback_support_status"] == "supported"
    assert reference["fallback_used"] is True
    assert severity["severity_depth_status"] == (
        "severity_conditioned_split_fallback_leaning"
    )
    assert report["interpretation_status"] == (
        "signature_conditioned_severity_depth_fallback_leaning"
    )
    assert any("fallback" in item for item in report["limitations"])


def test_setup_stays_secondary_and_trigger_remains_negative_control(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        pocket_variants=[
            {
                "selected_strategy_confidence": _low_value(0.16),
                "rule_engine_confidence": 0.19,
                "context_confidence": _low_value(0.18),
                "bias_confidence": _low_value(0.18),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
        preserved_outside_variants=[
            {
                "selected_strategy_confidence": _low_value(0.02),
                "rule_engine_confidence": 0.57,
                "context_confidence": _low_value(0.02),
                "bias_confidence": _low_value(0.02),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["setup_secondary_reading"]["setup_separation_status"] == (
        "setup_stays_secondary_or_flat"
    )
    assert report["severity_depth_assessment"]["setup_stays_secondary"] is True
    assert report["trigger_negative_control_reading"][
        "trigger_negative_control_status"
    ] == "trigger_remains_negative_control"
    assert report["severity_depth_assessment"][
        "trigger_remains_negative_control"
    ] is True


def test_wrapper_import_path_and_entrypoint_smoke(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_rows(
        input_path,
        pocket_variants=[
            {
                "selected_strategy_confidence": _low_value(0.16),
                "rule_engine_confidence": 0.19,
                "context_confidence": _low_value(0.18),
                "bias_confidence": _low_value(0.18),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
        preserved_outside_variants=[
            {
                "selected_strategy_confidence": _low_value(0.02),
                "rule_engine_confidence": 0.57,
                "context_confidence": _low_value(0.02),
                "bias_confidence": _low_value(0.02),
                "setup_confidence": _high_value(0.15),
                "trigger_confidence": 0.96,
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_signature_conditioned_pocket_severity_diagnosis_report",
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
    assert captured["severity_depth_status"] == (
        "severity_conditioned_split_supported"
    )
    assert captured["setup_separation_status"] == "setup_stays_secondary_or_flat"
    assert captured["trigger_negative_control_status"] == (
        "trigger_remains_negative_control"
    )
    assert wrapper_module.build_report is report_module.build_report