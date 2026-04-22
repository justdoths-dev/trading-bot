from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report as report_module,
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
                logged_at=f"2026-04-21T{hour_offset:02d}:{index:02d}:00+00:00",
                symbol=f"{preserved_prefix}-{index:02d}",
                **preserved_kwargs,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-21T{hour_offset + 1:02d}:{index:02d}:00+00:00",
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


def _build_comparison_rows(
    *,
    preserved_values: list[float],
    collapsed_values: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "comparison_group": report_module._COMPARISON_GROUP_PRESERVED,
            report_module._RULE_ENGINE_CONFIDENCE_FIELD: value,
        }
        for value in preserved_values
    ] + [
        {
            "comparison_group": report_module._COMPARISON_GROUP_COLLAPSED,
            report_module._RULE_ENGINE_CONFIDENCE_FIELD: value,
        }
        for value in collapsed_values
    ]


def test_collapsed_side_low_band_and_bucket_concentration_supported(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.52,
                "context_confidence": 0.74,
                "bias_confidence": 0.74,
                "setup_confidence": 0.78,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.58,
                "context_confidence": 0.76,
                "bias_confidence": 0.76,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.64,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.70,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.88,
                "rule_signal": "long",
                "rule_engine_confidence": 0.76,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.88,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.84,
                "bias_confidence": 0.84,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.90,
                "rule_signal": "long",
                "rule_engine_confidence": 0.88,
                "context_confidence": 0.86,
                "bias_confidence": 0.86,
                "setup_confidence": 0.90,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.10,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.15,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.20,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.80,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    best_low_band = report["low_band_occupancy_profile"]["best_low_band_profile"]
    repeated = report["repeated_value_concentration"]
    bucket_concentration = report["bucket_concentration_profile"]
    dense_region = report["bucket_residual_profile"]["collapsed_dense_low_bucket_region"]
    signature = report["collapsed_side_compression_signature"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert (
        report["interpretation_status"]
        == "collapsed_side_low_band_and_bucket_concentration_supported"
    )
    assert best_low_band["collapsed_capture_rate"] == 1.0
    assert best_low_band["preserved_capture_rate"] == 0.0
    assert best_low_band["profile_strength_status"] == "sharp_low_band"
    assert repeated["concentration_status"] == "collapsed_exact_values_concentrated"
    assert repeated["low_level_pattern_status"] in {
        "collapsed_flooring_or_quantization_candidate",
        "collapsed_low_level_quantization_candidate",
    }
    assert bucket_concentration["concentration_status"] == "collapsed_bucket_concentrated"
    assert dense_region["concentration_status"] == "collapsed_dense_low_bucket_region"
    assert (
        signature["strongest_candidate_pattern"]
        == "collapsed_low_band_with_discrete_bucket_concentration"
    )


def test_false_positive_prevention_keeps_compression_inconclusive(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    shared_variants = [
        {
            "selected_strategy_confidence": 0.82,
            "rule_engine_confidence": 0.30,
            "context_confidence": 0.72,
            "bias_confidence": 0.72,
            "setup_confidence": 0.74,
            "trigger_confidence": 1.0,
        },
        {
            "selected_strategy_confidence": 0.82,
            "rule_engine_confidence": 0.35,
            "context_confidence": 0.74,
            "bias_confidence": 0.74,
            "setup_confidence": 0.76,
            "trigger_confidence": 1.0,
        },
        {
            "selected_strategy_confidence": 0.82,
            "rule_engine_confidence": 0.40,
            "context_confidence": 0.76,
            "bias_confidence": 0.76,
            "setup_confidence": 0.78,
            "trigger_confidence": 1.0,
        },
        {
            "selected_strategy_confidence": 0.82,
            "rule_engine_confidence": 0.45,
            "context_confidence": 0.78,
            "bias_confidence": 0.78,
            "setup_confidence": 0.80,
            "trigger_confidence": 1.0,
        },
    ]
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[{**variant, "rule_signal": "long"} for variant in shared_variants],
        collapsed_variants=[{**variant, "rule_signal": "hold"} for variant in shared_variants],
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["interpretation_status"] == "collapsed_side_compression_inconclusive"
    assert report["low_band_occupancy_profile"]["interpretation_status"] == (
        "mixed_or_weak"
    )
    assert report["repeated_value_concentration"]["concentration_status"] == "mixed"
    assert report["bucket_concentration_profile"]["concentration_status"] == "mixed"


def test_preserved_exact_value_outside_top_3_does_not_look_like_zero_share() -> None:
    preserved_values = (
        [0.40] * 9
        + [0.45] * 9
        + [0.50] * 9
        + [0.55] * 6
        + [0.10] * 3
    )
    collapsed_values = (
        [0.10] * 10
        + [0.12] * 6
        + [0.14] * 4
        + [0.16] * 4
        + [0.18] * 4
        + [0.20] * 4
        + [0.22] * 4
    )

    concentration = report_module.build_repeated_value_concentration(
        comparison_rows=_build_comparison_rows(
            preserved_values=preserved_values,
            collapsed_values=collapsed_values,
        )
    )

    preserved_summary = concentration[report_module._COMPARISON_GROUP_PRESERVED]

    assert concentration["support_status"] == "supported"
    assert preserved_values.count(0.10) / len(preserved_values) > 0.05
    assert all(item["value"] != 0.10 for item in preserved_summary["top_values"])
    assert concentration["low_level_pattern_status"] == "no_low_level_pattern_signal"


def test_near_unique_floats_do_not_become_exact_repeats() -> None:
    preserved_values = [0.45 + (index * 0.01) for index in range(36)]
    collapsed_values = [0.10000001 + (index * 0.000000001) for index in range(36)]

    concentration = report_module.build_repeated_value_concentration(
        comparison_rows=_build_comparison_rows(
            preserved_values=preserved_values,
            collapsed_values=collapsed_values,
        )
    )

    collapsed_summary = concentration[report_module._COMPARISON_GROUP_COLLAPSED]

    assert concentration["support_status"] == "supported"
    assert concentration["concentration_status"] == "mostly_unique"
    assert concentration["low_level_pattern_status"] == "no_low_level_pattern_signal"
    assert collapsed_summary["repeated_value_row_share"] == 0.0
    assert collapsed_summary["most_common_value_share"] == 0.027778
    assert collapsed_summary["top_3_value_share"] == 0.083333


def test_insufficient_supported_slice_returns_comparison_unsupported(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.84,
                "bias_confidence": 0.84,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.10,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.84,
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
        report["final_assessment"]["interpretation_status"] == "comparison_unsupported"
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
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.64,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.82,
                "context_confidence": 0.84,
                "bias_confidence": 0.84,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.10,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.15,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=12,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_collapsed_side_compression_diagnosis_report",
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
    assert captured["best_low_band"]["profile_strength_status"] == "sharp_low_band"
    assert captured["interpretation_status"] in {
        "collapsed_side_low_band_and_bucket_concentration_supported",
        "collapsed_side_low_band_supported",
    }
    assert wrapper_module.build_report is report_module.build_report
