from __future__ import annotations

import json
from pathlib import Path

import src.research.directional_bias_promotion_path_diagnosis_report as wrapper_module
from src.research.diagnostics import (
    directional_bias_promotion_path_diagnosis_report as report_module,
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
    selected_strategy_signal: str | None = None,
    execution_action: str | None = None,
    execution_signal: str | None = None,
    rule_signal: str | None = None,
    execution_allowed: bool | None = None,
    entry_price: float | None = None,
) -> dict:
    strategy_payloads = {
        "scalping_result": {"strategy": "scalping", "signal": "hold", "confidence": 0.0},
        "intraday_result": {"strategy": "intraday", "signal": "hold", "confidence": 0.0},
        "swing_result": {"strategy": "swing", "signal": "hold", "confidence": 0.0},
    }
    if strategy is not None and selected_strategy_signal is not None:
        strategy_payloads[f"{strategy}_result"]["signal"] = selected_strategy_signal

    execution = {
        "execution_allowed": execution_allowed,
        "entry_price": entry_price,
    }
    if execution_action is not None:
        execution["action"] = execution_action
    if execution_signal is not None:
        execution["signal"] = execution_signal

    rule_engine: dict[str, object] = {}
    if bias is not None:
        rule_engine["bias"] = bias
    if rule_signal is not None:
        rule_engine["signal"] = rule_signal

    return {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": strategy,
        "bias": bias,
        "rule_engine": rule_engine,
        "risk": {
            "execution_allowed": execution_allowed,
            "entry_price": entry_price,
        },
        "execution": execution,
        **strategy_payloads,
    }


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
                rule_signal="long",
                execution_action="buy",
                execution_signal="long",
                execution_allowed=True,
                entry_price=200.0,
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
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
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
        "_effective_directional_bias_promotion_path_input.jsonl"
    )
    assert len(_read_jsonl(effective_input_path)) == 2
    assert summary["source_metadata"]["source_file_count"] == 2
    assert summary["source_metadata"]["effective_input_materialized"] is True
    assert (
        summary["promotion_path_funnel"]["directional_bias_present_known_identity_rows"]
        == 2
    )


def test_report_attributes_no_actionable_and_late_collapse_stages_truthfully(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(
        input_path,
        [
            _raw_record(
                logged_at="2026-04-14T00:00:00+00:00",
                symbol="BTCUSDT",
                strategy="scalping",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:10:00+00:00",
                symbol="SOLUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="short",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=101.0,
            ),
            _raw_record(
                logged_at="2026-04-14T00:15:00+00:00",
                symbol="XRPUSDT",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=True,
                entry_price=202.0,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    funnel = summary["promotion_path_funnel"]
    taxonomy = summary["hold_unknown_proposal_taxonomy"]

    assert funnel["final_buy_sell_emitted_rows"] == 0
    assert funnel["final_hold_rows"] == 4
    assert funnel["final_unknown_rows"] == 0
    assert funnel["hold_unknown_earliest_collapse_stage_counts"] == {
        "selected_strategy_proposal_not_actionable": 1,
        "decision_collapse_after_actionable_strategy_proposal": 1,
        "risk_gate_blocked_after_actionable_rule_signal": 1,
        "execution_layer_hold_after_allowed_rule_signal": 1,
    }
    assert funnel["hold_unknown_collapse_category_counts"] == {
        "no_actionable_upstream_proposal_visible": 1,
        "actionable_upstream_proposal_later_collapses": 3,
    }
    assert funnel["primary_hold_collapse_category"] == (
        "actionable_upstream_proposal_later_collapses"
    )
    assert [
        row["proposal_state"] for row in taxonomy["selected_strategy_proposal_count_rows"]
    ] == ["long", "hold", "short"]
    assert [
        row["proposal_state"] for row in taxonomy["rule_signal_count_rows"]
    ] == ["hold", "long", "short"]


def test_final_emission_remains_execution_layer_only_and_unknown_rows_stay_separate(
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
                execution_signal=None,
                execution_allowed=False,
                entry_price=None,
            ),
            _raw_record(
                logged_at="2026-04-14T00:05:00+00:00",
                symbol="ETHUSDT",
                strategy="intraday",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="short",
                execution_action=None,
                execution_signal=None,
                execution_allowed=None,
                entry_price=None,
            ),
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    funnel = summary["promotion_path_funnel"]

    assert funnel["final_buy_sell_emitted_rows"] == 0
    assert funnel["final_hold_rows"] == 1
    assert funnel["final_unknown_rows"] == 1
    assert funnel["hold_unknown_earliest_collapse_stage_counts"] == {
        "risk_gate_blocked_after_actionable_rule_signal": 1,
        "risk_gate_unobservable_after_actionable_rule_signal": 1,
    }


def test_strategy_specificity_compares_scalping_against_intraday_and_swing(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(10):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"SCALP{index}",
                strategy="scalping",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T01:{index:02d}:00+00:00",
                symbol=f"INTRA{index}",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=100.0 + index,
            )
        )
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T02:{index:02d}:00+00:00",
                symbol=f"SWING{index}",
                strategy="swing",
                bias="bearish",
                selected_strategy_signal="short",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            )
        )

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]
    specificity = summary["strategy_specificity"]

    assert specificity["classification"] == "strategy_specific"
    assert specificity["supported_primary_collapse_categories"] == {
        "scalping": "no_actionable_upstream_proposal_visible",
        "intraday": "actionable_upstream_proposal_later_collapses",
        "swing": "actionable_upstream_proposal_later_collapses",
    }
    assert specificity["supported_primary_collapse_stages"] == {
        "scalping": "selected_strategy_proposal_not_actionable",
        "intraday": "risk_gate_blocked_after_actionable_rule_signal",
        "swing": "decision_collapse_after_actionable_strategy_proposal",
    }
    assert specificity["scalping_primary_collapse_category"] == (
        "no_actionable_upstream_proposal_visible"
    )


def test_final_assessment_keeps_primary_bottleneck_mixed_without_primary_factor(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    rows: list[dict] = []

    for index in range(14):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T00:{index:02d}:00+00:00",
                symbol=f"LATE{index}",
                strategy="intraday",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal="long",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            )
        )

    for index in range(13):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T01:{index:02d}:00+00:00",
                symbol=f"NOACT{index}",
                strategy="scalping",
                bias="bullish",
                selected_strategy_signal="hold",
                rule_signal="hold",
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=False,
                entry_price=None,
            )
        )

    for index in range(13):
        rows.append(
            _raw_record(
                logged_at=f"2026-04-14T02:{index:02d}:00+00:00",
                symbol=f"UNOBS{index}",
                strategy="swing",
                bias="bullish",
                selected_strategy_signal="long",
                rule_signal=None,
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=None,
                entry_price=None,
            )
        )

    _write_jsonl(input_path, rows)

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(336, 10000)],
    )
    summary = report["configuration_summaries"][0]
    funnel = summary["promotion_path_funnel"]
    assessment = report["final_assessment"]

    assert funnel["primary_hold_collapse_category"] == (
        "actionable_upstream_proposal_later_collapses"
    )
    assert assessment["primary_bottleneck"] == "mixed_or_inconclusive"
    assert assessment["primary_hold_path_category"] == "mixed_or_inconclusive"
    assert assessment["dominant_observed_hold_path_category"] == (
        "actionable_upstream_proposal_later_collapses"
    )


def test_missing_rule_signal_after_actionable_selected_strategy_is_unobservable(
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
                rule_signal=None,
                execution_action="hold",
                execution_signal="hold",
                execution_allowed=None,
                entry_price=None,
            )
        ],
    )

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[report_module.DiagnosisConfiguration(36, 2500)],
    )
    summary = report["configuration_summaries"][0]
    funnel = summary["promotion_path_funnel"]

    assert funnel["hold_unknown_earliest_collapse_stage_counts"] == {
        "rule_signal_unobservable_after_actionable_strategy_proposal": 1
    }
    assert funnel["hold_unknown_collapse_category_counts"] == {
        "intermediate_trace_unobservable": 1
    }
    assert funnel["primary_hold_collapse_category"] == "intermediate_trace_unobservable"


def test_main_and_wrapper_follow_existing_report_pattern(
    tmp_path: Path,
    capsys,
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
                rule_signal="long",
                execution_action="buy",
                execution_signal="long",
                execution_allowed=True,
                entry_price=200.0,
            )
        ],
    )

    report_module.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--config",
            "336/10000",
        ]
    )
    captured = json.loads(capsys.readouterr().out)

    assert captured["report_type"] == report_module.REPORT_TYPE
    assert captured["configuration_count"] == 1
    assert "primary_bottleneck" in captured
    assert wrapper_module.build_report is report_module.build_report