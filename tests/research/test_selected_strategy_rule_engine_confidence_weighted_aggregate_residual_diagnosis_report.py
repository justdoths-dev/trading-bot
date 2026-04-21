from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report as report_module,
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


def test_supported_report_classifies_collapse_heavy_threshold_residual(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.87,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.90,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.85,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.32,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.50,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.35,
                "context_confidence": 0.45,
                "bias_confidence": 0.45,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.40,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    summary = report["summary"]
    residual_comparison = report["residual_class_comparison"]
    classification = report["residual_behavior_classification"]
    negative_pocket = report["residual_sign_distribution"]["sharply_negative_pocket"]

    assert summary["comparison_support_status"] == "supported"
    assert summary["final_rule_bias_aligned_row_count"] == 60
    assert summary["preserved_final_directional_outcome_row_count"] == 30
    assert summary["collapsed_final_hold_outcome_row_count"] == 30
    assert report["baseline_name"] == "weighted_mean_setup_emphasis"
    assert residual_comparison["support_status"] == "supported"
    assert (
        residual_comparison[report_module._COMPARISON_GROUP_PRESERVED]["median"] >= -0.05
    )
    assert (
        residual_comparison[report_module._COMPARISON_GROUP_COLLAPSED]["median"] <= -0.15
    )
    assert negative_pocket["concentration_status"] == "collapsed_concentrated"
    assert (
        classification["strongest_candidate_explanation_class"] == "threshold"
    )
    assert classification["interpretation_status"] == "threshold_explanation_supported"
    assert classification["threshold_supported"] is True
    assert classification["interaction_effect_supported"] is False
    assert report["joint_shortfall_regimes"]["support_status"] == "insufficient_data"
    assert report["joint_shortfall_regimes"]["strongest_joint_regime"] == {}


def test_false_positive_prevention_keeps_residual_mechanism_unclaimed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.48,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.50,
                "trigger_confidence": 1.0,
            }
        ],
        hour_offset=4,
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    classification = report["residual_behavior_classification"]
    residual_comparison = report["residual_class_comparison"]
    negative_pocket = report["residual_sign_distribution"]["sharply_negative_pocket"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert (
        abs(
            residual_comparison[report_module._COMPARISON_GROUP_PRESERVED]["median"]
        )
        <= 0.05
    )
    assert (
        abs(
            residual_comparison[report_module._COMPARISON_GROUP_COLLAPSED]["median"]
        )
        <= 0.05
    )
    assert negative_pocket["concentration_status"] in {"absent", "mixed", "collapsed_leaning"}
    assert classification["strongest_candidate_explanation_class"] == "unclear"
    assert classification["interpretation_status"] == "no_clear_explanation_class_supported"
    assert classification["explanation_class_support_status"] == "unsupported"
    assert "threshold" in classification["unsupported_explanation_classes"]


def test_joint_shortfall_regime_surfaces_as_strongest_interaction_candidate(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.87,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.90,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.10,
                "context_confidence": 0.40,
                "bias_confidence": 0.40,
                "setup_confidence": 0.45,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.60,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.55,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.67,
                "context_confidence": 0.55,
                "bias_confidence": 0.55,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=8,
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    classification = report["residual_behavior_classification"]
    strongest_joint_regime = report["joint_shortfall_regimes"]["strongest_joint_regime"]
    strongest_single_regime = report["residual_low_confidence_regimes"][
        "strongest_single_surface_regime"
    ]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert (
        classification["strongest_candidate_explanation_class"]
        == "interaction_effect"
    )
    assert classification["interpretation_status"] == "interaction_effect_supported"
    assert classification["interaction_effect_supported"] is True
    assert strongest_joint_regime["regime_label"] == "low_setup_and_low_context_bias_family"
    assert strongest_joint_regime["low_surface_count"] == 2
    assert (
        strongest_joint_regime["collapsed_sharply_negative_rate_minus_preserved"]
        > strongest_single_regime["collapsed_sharply_negative_rate_minus_preserved"]
    )


def test_insufficient_supported_slice_returns_conservative_unsupported_interpretation(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.87,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.90,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.32,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.50,
                "trigger_confidence": 1.0,
            }
        ],
        repeats=8,
        hour_offset=12,
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    assert report["summary"]["comparison_support_status"] == "limited_support"
    assert report["residual_behavior_classification"]["support_status"] == "unsupported"
    assert report["residual_behavior_classification"]["interpretation_status"] == (
        "comparison_unsupported"
    )
    assert report["final_assessment"]["interpretation_status"] == "comparison_unsupported"


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
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.87,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.90,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.32,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.50,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.35,
                "context_confidence": 0.45,
                "bias_confidence": 0.45,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.40,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=16,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_diagnosis_report",
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
    assert captured["comparison_support_status"] == "supported"
    assert captured["strongest_candidate_explanation_class"] == "threshold"
    assert wrapper_module.build_report is report_module.build_report
