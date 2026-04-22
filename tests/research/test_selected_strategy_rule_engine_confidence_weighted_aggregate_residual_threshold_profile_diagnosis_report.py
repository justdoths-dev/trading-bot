from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report as report_module,
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


def _write_balanced_two_group_rows(
    path: Path,
    *,
    preserved_variants: list[dict],
    collapsed_variants: list[dict],
    repeats: int = 30,
    preserved_prefix: str = "BTCUSDT",
    collapsed_prefix: str = "ETHUSDT",
    hour_offset: int = 0,
) -> None:
    rows: list[dict] = []
    for index in range(repeats):
        preserved_kwargs = preserved_variants[index % len(preserved_variants)]
        collapsed_kwargs = collapsed_variants[index % len(collapsed_variants)]
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T{hour_offset:02d}:{index:02d}:00+00:00",
                symbol=f"{preserved_prefix}-{index:02d}",
                **preserved_kwargs,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-19T{hour_offset + 1:02d}:{index:02d}:00+00:00",
                symbol=f"{collapsed_prefix}-{index:02d}",
                **collapsed_kwargs,
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


def test_context_bias_family_mean_surfaces_as_best_threshold_gate_candidate(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.55,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.75,
                "rule_signal": "long",
                "rule_engine_confidence": 0.81,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.45,
                "bias_confidence": 0.45,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.60,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.32,
                "context_confidence": 0.48,
                "bias_confidence": 0.48,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.78,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.54,
                "context_confidence": 0.62,
                "bias_confidence": 0.62,
                "setup_confidence": 0.83,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    summary = report["summary"]
    best_axis = report["best_threshold_axis"]
    best_profile = report["best_threshold_profile"]
    band_profiles = report["top_axis_band_profiles"]

    assert summary["comparison_support_status"] == "supported"
    assert report["interpretation_status"] == "threshold_profile_supported"
    assert best_axis["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert best_axis["profile_strength_status"] == "strong_support"
    assert best_profile["threshold"] == 0.65
    assert best_profile["collapsed_capture_rate"] == 1.0
    assert best_profile["preserved_leakage_rate"] <= 0.05
    assert best_profile["inside_pocket_concentration_gap"] >= 0.9
    assert report["selected_strategy_value_check"][
        "adds_meaningful_value_beyond_best_axis"
    ] is False
    assert band_profiles[0]["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert band_profiles[0]["profile_shape"] in {
        "simple_cutoff",
        "broad_helper_regime",
    }


def test_gate_vs_severity_check_keeps_setup_as_severity_inside_context_gate(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.55,
                "rule_signal": "long",
                "rule_engine_confidence": 0.68,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.62,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.78,
                "context_confidence": 0.76,
                "bias_confidence": 0.76,
                "setup_confidence": 0.72,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.75,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.46,
                "context_confidence": 0.48,
                "bias_confidence": 0.48,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.18,
                "context_confidence": 0.48,
                "bias_confidence": 0.48,
                "setup_confidence": 0.50,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.16,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.45,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    best_axis = report["best_threshold_axis"]
    gate_vs_severity = report["gate_vs_severity_check"]
    conditional_profile = gate_vs_severity["best_conditional_setup_profile"]
    collapsed_median_delta = conditional_profile[
        "collapsed_inside_vs_outside_residual_median_delta"
    ]
    collapsed_pocket_lift = conditional_profile["collapsed_pocket_lift"]

    assert report["interpretation_status"] == "threshold_profile_supported"
    assert best_axis["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert (
        gate_vs_severity["best_gate_axis"]["field"]
        == report_module._CONTEXT_BIAS_FAMILY_FIELD
    )
    assert gate_vs_severity["setup_axis"]["field"] in {
        "setup_layer_confidence",
        "setup_shortfall",
    }
    assert (
        gate_vs_severity["interpretation_status"]
        == "setup_axis_behaves_more_like_severity_inside_gate"
    )
    assert (
        (collapsed_median_delta is not None and collapsed_median_delta <= -0.08)
        or (collapsed_pocket_lift is not None and collapsed_pocket_lift >= 0.20)
    )


def test_false_positive_prevention_keeps_threshold_profile_conservative(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.60,
                "rule_signal": "long",
                "rule_engine_confidence": 0.60,
                "context_confidence": 0.60,
                "bias_confidence": 0.60,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.60,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.60,
                "context_confidence": 0.60,
                "bias_confidence": 0.60,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=4,
    )

    report = _build_report(input_path, tmp_path / "reports")

    best_axis = report["best_threshold_axis"]
    best_profile = report["best_threshold_profile"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["interpretation_status"] == "threshold_profile_inconclusive"
    assert best_axis["profile_strength_status"] != "strong_support"
    assert best_profile["inside_pocket_concentration_gap"] == 0.0
    assert best_profile["collapsed_pocket_lift"] == 0.0


def test_interior_band_is_classified_as_narrow_band() -> None:
    band_rows = [
        {
            "inside_row_count": 10,
            "profile_strength_status": "weak_support",
            "collapsed_capture_rate": 0.25,
            "preserved_leakage_rate": 0.20,
            "inside_pocket_concentration_gap": 0.55,
            "collapsed_pocket_lift": 0.25,
        },
        {
            "inside_row_count": 10,
            "profile_strength_status": "strong_support",
            "collapsed_capture_rate": 0.70,
            "preserved_leakage_rate": 0.05,
            "inside_pocket_concentration_gap": 1.0,
            "collapsed_pocket_lift": 0.60,
        },
        {
            "inside_row_count": 10,
            "profile_strength_status": "weak_support",
            "collapsed_capture_rate": 0.30,
            "preserved_leakage_rate": 0.18,
            "inside_pocket_concentration_gap": 0.55,
            "collapsed_pocket_lift": 0.20,
        },
        {
            "inside_row_count": 10,
            "profile_strength_status": "limited_support",
            "collapsed_capture_rate": 0.10,
            "preserved_leakage_rate": 0.30,
            "inside_pocket_concentration_gap": 0.10,
            "collapsed_pocket_lift": 0.00,
        },
        {
            "inside_row_count": 10,
            "profile_strength_status": "insufficient_data",
            "collapsed_capture_rate": 0.00,
            "preserved_leakage_rate": 0.40,
            "inside_pocket_concentration_gap": 0.00,
            "collapsed_pocket_lift": 0.00,
        },
    ]

    profile_shape, reason = report_module._classify_band_profile(
        band_rows=band_rows,
        extreme_band_side="low",
    )

    assert profile_shape == "narrow_band"
    assert "interior" in reason


def test_insufficient_supported_slice_returns_comparison_unsupported(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.45,
                "bias_confidence": 0.45,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            }
        ],
        repeats=8,
        hour_offset=8,
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "limited_support"
    assert report["interpretation_status"] == "comparison_unsupported"
    assert (
        report["final_assessment"]["interpretation_status"]
        == "comparison_unsupported"
    )


def test_wrapper_import_path_and_entrypoint_smoke(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.55,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.75,
                "rule_signal": "long",
                "rule_engine_confidence": 0.81,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.45,
                "bias_confidence": 0.45,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.60,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.32,
                "context_confidence": 0.48,
                "bias_confidence": 0.48,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.78,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.54,
                "context_confidence": 0.62,
                "bias_confidence": 0.62,
                "setup_confidence": 0.83,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=12,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_threshold_profile_diagnosis_report",
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
    assert captured["baseline_name"] == "weighted_mean_setup_emphasis"
    assert (
        captured["best_threshold_axis"]["field"]
        == report_module._CONTEXT_BIAS_FAMILY_FIELD
    )
    assert captured["interpretation_status"] == "threshold_profile_supported"
    assert wrapper_module.build_report is report_module.build_report