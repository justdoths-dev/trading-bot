from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_weighted_aggregate_residual_context_band_conditional_setup_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_weighted_aggregate_residual_context_band_conditional_setup_diagnosis_report as report_module,
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
    repeats: int = 36,
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


def test_narrow_context_band_candidate_beats_coarse_single_cutoff(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.78,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.76,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.72,
                "rule_signal": "long",
                "rule_engine_confidence": 0.56,
                "context_confidence": 0.42,
                "bias_confidence": 0.42,
                "setup_confidence": 0.34,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.84,
                "context_confidence": 0.68,
                "bias_confidence": 0.68,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.28,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.48,
                "context_confidence": 0.64,
                "bias_confidence": 0.64,
                "setup_confidence": 0.74,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    best_band = report["best_context_band"]
    comparison = report["context_band_vs_single_cutoff_comparison"]
    context_interpretation = report["context_band_interpretation"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["interpretation_status"] in {
        "context_band_with_conditional_setup_supported",
        "narrow_context_band_supported_setup_inconclusive",
    }
    assert best_band["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert best_band["band_label"] == "(0.50, 0.55]"
    assert best_band["profile_shape"] == "narrow_band"
    assert (
        comparison["comparison_reading"]
        == "narrow_band_better_than_single_cutoff"
    )
    assert comparison["materially_reduces_preserved_leakage"] is True
    assert comparison["keeps_most_collapsed_pocket_membership"] is True
    assert context_interpretation["best_context_band_reading"] == "narrow_band"


def test_broad_context_helper_regime_does_not_overclaim_narrow_band(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.78,
                "context_confidence": 0.74,
                "bias_confidence": 0.74,
                "setup_confidence": 0.78,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.70,
                "bias_confidence": 0.70,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.22,
                "context_confidence": 0.42,
                "bias_confidence": 0.42,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.26,
                "context_confidence": 0.48,
                "bias_confidence": 0.48,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.78,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["best_context_band"]["profile_shape"] in {
        "simple_cutoff",
        "broad_helper_regime",
    }
    assert report["context_band_interpretation"]["best_context_band_reading"] != (
        "narrow_band"
    )
    assert report["context_band_vs_single_cutoff_comparison"][
        "comparison_reading"
    ] == "plain_single_cutoff_or_broad_helper_regime_more_plausible"


def test_conditional_setup_severity_stays_subordinate_to_context_band(
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
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.76,
                "rule_signal": "long",
                "rule_engine_confidence": 0.68,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.78,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.74,
                "rule_signal": "long",
                "rule_engine_confidence": 0.56,
                "context_confidence": 0.44,
                "bias_confidence": 0.44,
                "setup_confidence": 0.34,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.74,
                "rule_signal": "long",
                "rule_engine_confidence": 0.58,
                "context_confidence": 0.44,
                "bias_confidence": 0.44,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.72,
                "bias_confidence": 0.72,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.44,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.00,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.42,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.52,
                "context_confidence": 0.66,
                "bias_confidence": 0.66,
                "setup_confidence": 0.78,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    best_conditional = report["best_conditional_setup_profile"]
    best_threshold = best_conditional["best_conditional_threshold_profile"]
    conditional_profiles = report["conditional_setup_profiles"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["best_context_band"]["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert conditional_profiles["interpretation_status"] == "conditional_setup_supported"
    assert best_conditional["field"] in {"setup_layer_confidence", "setup_shortfall"}
    assert (
        best_conditional["conditional_severity_status"]
        == "sharpens_collapsed_severity_without_replacing_gate"
    )
    assert best_threshold["collapsed_pocket_lift"] >= 0.20 or (
        best_conditional["collapsed_inside_vs_outside_residual_median_delta"]
        is not None
        and best_conditional["collapsed_inside_vs_outside_residual_median_delta"] <= -0.08
    )


def test_false_positive_prevention_keeps_report_conservative(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.70,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.62,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.78,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.64,
                "bias_confidence": 0.64,
                "setup_confidence": 0.66,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.70,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.62,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.78,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.64,
                "bias_confidence": 0.64,
                "setup_confidence": 0.66,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["interpretation_status"] == "context_band_conditional_setup_inconclusive"
    assert report["context_band_interpretation"]["best_context_band_reading"] == (
        "mixed_or_weak"
    )
    assert report["conditional_setup_profiles"]["interpretation_status"] in {
        "conditional_setup_inconclusive",
        "conditional_setup_unavailable",
    }


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
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            }
        ],
        repeats=8,
        hour_offset=8,
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "limited_support"
    assert report["interpretation_status"] == "comparison_unsupported"
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
                "selected_strategy_confidence": 0.78,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.76,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.72,
                "rule_signal": "long",
                "rule_engine_confidence": 0.56,
                "context_confidence": 0.42,
                "bias_confidence": 0.42,
                "setup_confidence": 0.34,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.28,
                "context_confidence": 0.52,
                "bias_confidence": 0.52,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=12,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_weighted_aggregate_residual_context_band_conditional_setup_diagnosis_report",
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
    assert captured["best_context_band"]["field"] == report_module._CONTEXT_BIAS_FAMILY_FIELD
    assert captured["interpretation_status"] in {
        "context_band_with_conditional_setup_supported",
        "narrow_context_band_supported_setup_inconclusive",
    }
    assert wrapper_module.build_report is report_module.build_report