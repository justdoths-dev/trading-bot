from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report as report_module,
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
                logged_at=f"2026-04-22T{hour_offset:02d}:{index:02d}:00+00:00",
                symbol=f"{preserved_prefix}-{index:02d}",
                **preserved_kwargs,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-22T{hour_offset + 1:02d}:{index:02d}:00+00:00",
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


def test_local_origin_preparation_recomputes_helper_fields_without_upstream_contract() -> None:
    threshold = float(report_module._LOW_CONFIDENCE_THRESHOLD)
    clearly_low_value = max(0.0, round(threshold - 0.20, 6))
    clearly_high_value = min(1.0, round(threshold + 0.20, 6))
    if clearly_high_value <= threshold:
        clearly_high_value = min(1.0, round(threshold + 0.05, 6))

    row = {
        report_module._RULE_ENGINE_CONFIDENCE_FIELD: 0.22,
        report_module._BASELINE_NAME: 0.55,
        report_module._RESIDUAL_FIELD: -0.33,
        "context_layer_confidence": clearly_low_value,
        "bias_layer_confidence": clearly_low_value,
        "setup_layer_confidence": clearly_low_value,
        "selected_strategy_confidence": clearly_high_value,
    }

    prepared = report_module._prepare_origin_residual_row(row)

    assert (
        prepared[report_module._CONTEXT_BIAS_FAMILY_FIELD]
        == clearly_low_value
    )
    assert prepared["low_setup_confidence_regime"] is True
    assert prepared["low_context_bias_family_regime"] is True
    assert prepared["low_selected_strategy_confidence_regime"] is False
    assert prepared["setup_shortfall"] == round(threshold - clearly_low_value, 6)
    assert prepared["context_bias_family_shortfall"] == round(
        threshold - clearly_low_value,
        6,
    )
    assert prepared["selected_strategy_shortfall"] == 0.0
    assert prepared["low_confidence_surface_count"] == 2
    assert prepared["exact_low_confidence_fields"] == (
        "setup_layer_confidence",
        report_module._CONTEXT_BIAS_FAMILY_FIELD,
    )


def test_joint_weakness_signature_is_supported_for_compressed_pocket(
    tmp_path: Path,
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
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "long",
                "rule_engine_confidence": 0.80,
                "context_confidence": 0.82,
                "bias_confidence": 0.82,
                "setup_confidence": 0.86,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.74,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.17,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.48,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.58,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.21,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.72,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.60,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.24,
                "context_confidence": 0.72,
                "bias_confidence": 0.72,
                "setup_confidence": 0.58,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    joint = report["joint_weakness_signature"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["pocket_summary"]["zero_collapsed_outside_pocket"] is True
    assert joint["reference_group_label"] == (
        report_module._PRESERVED_OUTSIDE_POCKET_GROUP_LABEL
    )
    assert joint["pocket_complete_row_count"] == report["pocket_summary"]["pocket_row_count"]
    assert joint["signature_status"] == "joint_weakness_signature_supported"
    assert joint["two_or_more_low_surface_profile"]["pocket_rate"] == 1.0
    assert joint["two_or_more_low_surface_profile"]["reference_rate"] == 0.0
    assert joint["dominant_pocket_joint_signature"]["low_surface_count"] == 2
    assert (
        report["contributor_family_by_pocket_membership"][
            "trigger_negative_control_status"
        ]
        == "trigger_remains_negative_control"
    )
    assert report["interpretation_status"] in {
        "compressed_low_band_origin_joint_weakness_supported",
        "compressed_low_band_origin_joint_weakness_with_piecewise_dense_buckets_supported",
    }


def test_dense_bucket_comparison_ignores_preserved_contamination(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.18,
                "context_confidence": 0.84,
                "bias_confidence": 0.84,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.23,
                "context_confidence": 0.84,
                "bias_confidence": 0.84,
                "setup_confidence": 0.88,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
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
                "rule_engine_confidence": 0.18,
                "context_confidence": 0.44,
                "bias_confidence": 0.44,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.44,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.23,
                "context_confidence": 0.76,
                "bias_confidence": 0.76,
                "setup_confidence": 0.36,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    dense_summary = report["dense_bucket_summary"]
    dense = report["dense_bucket_comparison"]

    left_bucket = dense_summary["bucket_summaries"][0]
    right_bucket = dense_summary["bucket_summaries"][1]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["pocket_summary"]["dense_bucket_preserved_contamination_present"] is True
    assert dense_summary["preserved_contamination_present"] is True
    assert dense["preserved_contamination_present"] is True
    assert dense["comparison_population"] == "collapsed_only_dense_bucket_rows"
    assert left_bucket["comparison_population"] == "collapsed_only_origin_rows"
    assert right_bucket["comparison_population"] == "collapsed_only_origin_rows"

    assert dense["left_row_count"] == left_bucket["collapsed_origin_row_count"]
    assert dense["right_row_count"] == right_bucket["collapsed_origin_row_count"]
    assert left_bucket["total_row_count"] > left_bucket["collapsed_origin_row_count"]
    assert right_bucket["total_row_count"] > right_bucket["collapsed_origin_row_count"]

    assert dense["left_preserved_contamination_row_count"] > 0
    assert dense["right_preserved_contamination_row_count"] > 0
    assert dense["piecewise_status"] == "bucket_distinct_piecewise_regimes"
    assert dense["stronger_downward_residual_bucket_label"] == "(0.15, 0.20]"


def test_dense_buckets_can_report_piecewise_origin_split_without_contamination(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.66,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.86,
                "rule_signal": "long",
                "rule_engine_confidence": 0.74,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.84,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.18,
                "context_confidence": 0.44,
                "bias_confidence": 0.44,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.44,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.23,
                "context_confidence": 0.76,
                "bias_confidence": 0.76,
                "setup_confidence": 0.36,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")
    dense = report["dense_bucket_comparison"]

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["dense_bucket_summary"]["preserved_contamination_present"] is False
    assert dense["preserved_contamination_present"] is False
    assert dense["piecewise_status"] == "bucket_distinct_piecewise_regimes"
    assert dense["stronger_downward_residual_bucket_label"] == "(0.15, 0.20]"
    assert dense["left_dominant_exact_low_confidence_signature"][
        "signature_label"
    ] == "setup_layer_confidence + context_bias_family_mean"
    assert dense["right_dominant_exact_low_confidence_signature"][
        "signature_label"
    ] == "setup_layer_confidence + selected_strategy_confidence"
    assert set(dense["material_contributor_shift_fields"]) >= {
        "context_bias_family_mean",
        "selected_strategy_confidence",
    }
    assert report["contributor_family_by_dense_bucket"]["support_status"] == "supported"


def test_absent_dense_bucket_stays_conservative_but_keeps_pocket_comparison_useful(
    tmp_path: Path,
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
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.76,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.22,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.46,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.58,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.24,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.72,
                "trigger_confidence": 1.0,
            },
        ],
    )

    report = _build_report(input_path, tmp_path / "reports")

    assert report["pocket_summary"]["collapsed_rows_outside_pocket_row_count"] == 0
    assert report["pocket_vs_other_comparison"]["support_status"] == "supported"
    assert (
        report["dense_bucket_comparison"]["piecewise_status"]
        == "dense_bucket_comparison_unavailable"
    )
    assert any(
        "No collapsed rows remain outside the fixed low-band pocket" in fact
        for fact in report["interpretation"]["facts"]
    )
    assert report["interpretation_status"] in {
        "compressed_low_band_origin_joint_weakness_supported",
        "compressed_low_band_origin_joint_weakness_leaning",
        "compressed_low_band_origin_single_family_helper_leaning",
        "compressed_low_band_origin_inconclusive",
    }


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
                "rule_engine_confidence": 0.64,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.80,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.22,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.46,
                "trigger_confidence": 1.0,
            }
        ],
        repeats=8,
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
                "selected_strategy_confidence": 0.84,
                "rule_signal": "long",
                "rule_engine_confidence": 0.64,
                "context_confidence": 0.78,
                "bias_confidence": 0.78,
                "setup_confidence": 0.82,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.82,
                "rule_signal": "long",
                "rule_engine_confidence": 0.72,
                "context_confidence": 0.80,
                "bias_confidence": 0.80,
                "setup_confidence": 0.84,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.74,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.17,
                "context_confidence": 0.50,
                "bias_confidence": 0.50,
                "setup_confidence": 0.48,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.58,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.23,
                "context_confidence": 0.54,
                "bias_confidence": 0.54,
                "setup_confidence": 0.72,
                "trigger_confidence": 1.0,
            },
        ],
        hour_offset=12,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_compressed_low_band_origin_diagnosis_report",
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
    assert captured["collapsed_rows_outside_pocket_row_count"] == 0
    assert captured["joint_weakness_status"] in {
        "joint_weakness_signature_supported",
        "joint_weakness_signature_leaning",
        "single_family_cutoff_leaning",
    }
    assert captured["interpretation_status"] in {
        "compressed_low_band_origin_joint_weakness_supported",
        "compressed_low_band_origin_joint_weakness_with_piecewise_dense_buckets_supported",
        "compressed_low_band_origin_joint_weakness_leaning",
        "compressed_low_band_origin_single_family_helper_leaning",
        "compressed_low_band_origin_inconclusive",
    }
    assert wrapper_module.build_report is report_module.build_report