from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import src.research.selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report as report_module,
)
from src.research.diagnostics import (
    selected_strategy_rule_engine_confidence_contributor_diagnosis_report as sibling_module,
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
            "bias": "bullish",
        },
        "bias_layer": {
            "context": "bullish_context",
            "bias": "bullish",
        },
        "setup_layer": {
            "setup": "long_confirmed",
            "bias": "bullish",
        },
        "trigger_layer": {
            "trigger": "long_confirmed",
            "bias": "bullish",
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


def test_build_aggregate_candidate_values_computes_required_candidates() -> None:
    candidate_values = report_module.build_aggregate_candidate_values(
        {
            "setup_layer_confidence": 0.9,
            "context_layer_confidence": 0.6,
            "bias_layer_confidence": 0.4,
            "selected_strategy_confidence": 0.8,
            "trigger_layer_confidence": 0.1,
        }
    )

    assert candidate_values == {
        "context_bias_family_mean": 0.5,
        "min_of_three": 0.5,
        "mean_of_three": 0.733333,
        "weighted_mean_setup_emphasis": 0.775,
        "second_lowest_of_three": 0.8,
    }
    assert set(candidate_values) == set(report_module.AGGREGATE_CANDIDATE_FIELDS)
    assert "trigger_layer_confidence" not in report_module.AGGREGATE_COMPONENT_FIELDS


def test_supported_report_compares_preserved_and_collapsed_rows_and_is_deterministic(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.68,
                "rule_signal": "long",
                "rule_engine_confidence": 0.86,
                "context_confidence": 0.74,
                "bias_confidence": 0.70,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.20,
                "context_confidence": 0.42,
                "bias_confidence": 0.38,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            }
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    summary = report["summary"]
    aggregate_shape = report["aggregate_shape_comparison"]
    best_candidate = aggregate_shape["best_aggregate_candidate"]

    assert summary["comparison_support_status"] == "supported"
    assert summary["final_rule_bias_aligned_row_count"] == 60
    assert summary["preserved_final_directional_outcome_row_count"] == 30
    assert summary["collapsed_final_hold_outcome_row_count"] == 30
    assert aggregate_shape["trigger_excluded_from_aggregate_candidates"] is True
    assert best_candidate["field"] == "context_bias_family_mean"
    assert best_candidate["candidate_rank"] == 1
    assert (
        best_candidate["comparison_to_actual_rule_engine_confidence"][
            "materially_weaker_on_tracked_surfaces"
        ]
        is True
    )
    assert (
        aggregate_shape["best_aggregate_is_sharper_than_any_single_contributor"] is False
    )
    assert (
        report["final_interpretation"]["best_aggregate_candidate"]
        == "context_bias_family_mean"
    )
    assert (
        report["final_interpretation"]["best_aggregate_still_weaker_than_actual_rule_engine_confidence"]
        is True
    )
    assert (
        report["final_interpretation"]["best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance"]
        is False
    )
    assert (
        report["final_interpretation"]["actual_split_mostly_reproducible_by_simple_aggregate"]
        is False
    )
    assert report["final_interpretation"]["interpretation_status"] == (
        "simple_aggregate_does_not_beat_single_contributors"
    )


def test_supported_report_flags_reproducibility_as_unproven_when_best_aggregate_improves_but_remains_weaker(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.2,
                "rule_signal": "long",
                "rule_engine_confidence": 0.70,
                "context_confidence": 0.2,
                "bias_confidence": 0.2,
                "setup_confidence": 0.9,
                "trigger_confidence": 1.0,
            },
            {
                "selected_strategy_confidence": 0.9,
                "rule_signal": "long",
                "rule_engine_confidence": 0.70,
                "context_confidence": 0.9,
                "bias_confidence": 0.9,
                "setup_confidence": 0.2,
                "trigger_confidence": 1.0,
            },
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.4,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.30,
                "context_confidence": 0.4,
                "bias_confidence": 0.4,
                "setup_confidence": 0.4,
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

    aggregate_shape = report["aggregate_shape_comparison"]
    best_candidate = aggregate_shape["best_aggregate_candidate"]

    assert best_candidate["field"] == "weighted_mean_setup_emphasis"
    assert (
        aggregate_shape["best_aggregate_is_sharper_than_any_single_contributor"] is True
    )
    assert (
        best_candidate["comparison_to_actual_rule_engine_confidence"][
            "materially_weaker_on_tracked_surfaces"
        ]
        is True
    )
    assert (
        best_candidate["comparison_to_actual_rule_engine_confidence"][
            "matches_actual_on_tracked_surfaces_within_tolerance"
        ]
        is False
    )
    assert (
        report["final_interpretation"]["actual_split_mostly_reproducible_by_simple_aggregate"]
        is False
    )
    assert report["final_interpretation"]["interpretation_status"] == (
        "aggregate_improves_on_single_contributors_but_remains_weaker_than_actual_rule_engine_confidence"
    )


def test_report_marks_comparison_as_unsupported_when_best_aggregate_lacks_supported_actual_surface_comparison(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_balanced_two_group_rows(
        input_path,
        preserved_variants=[
            {
                "selected_strategy_confidence": 0.68,
                "rule_signal": "long",
                "rule_engine_confidence": 0.86,
                "context_confidence": None,
                "bias_confidence": None,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.20,
                "context_confidence": None,
                "bias_confidence": None,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            }
        ],
        hour_offset=6,
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
        min_symbol_support=1,
    )

    assert report["summary"]["comparison_support_status"] == "supported"
    assert report["aggregate_shape_comparison"]["support_status"] in {
        "supported",
        "limited_support",
    }
    assert report["final_interpretation"]["support_status"] == "unsupported"
    assert report["final_interpretation"]["interpretation_status"] == (
        "comparison_unsupported"
    )
    assert (
        report["final_interpretation"]["actual_split_mostly_reproducible_by_simple_aggregate"]
        is False
    )
    assert (
        report["final_interpretation"]["best_aggregate_matches_actual_on_tracked_surfaces_within_tolerance"]
        is None
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
                "selected_strategy_confidence": 0.68,
                "rule_signal": "long",
                "rule_engine_confidence": 0.86,
                "context_confidence": 0.74,
                "bias_confidence": 0.70,
                "setup_confidence": 0.60,
                "trigger_confidence": 1.0,
            }
        ],
        collapsed_variants=[
            {
                "selected_strategy_confidence": 0.50,
                "rule_signal": "hold",
                "rule_engine_confidence": 0.20,
                "context_confidence": 0.42,
                "bias_confidence": 0.38,
                "setup_confidence": 0.40,
                "trigger_confidence": 1.0,
            }
        ],
        hour_offset=8,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "selected_strategy_rule_engine_confidence_aggregate_shape_diagnosis_report",
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
    assert captured["comparison_support_status"] == "supported"
    assert captured["best_aggregate_candidate"] == "context_bias_family_mean"
    assert wrapper_module.build_report is report_module.build_report


def test_existing_sibling_module_remains_importable() -> None:
    assert sibling_module.REPORT_TYPE == (
        "selected_strategy_rule_engine_confidence_contributor_diagnosis_report"
    )