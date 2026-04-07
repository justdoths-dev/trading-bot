from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.ai_shadow_comparison_report import (
    build_ai_shadow_comparison_summary,
    resolve_input_path,
    run_ai_shadow_comparison_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _shadow_response(
    *,
    bias: str,
    confidence: str = "medium",
    regime_label: str = "directional_trend",
    recommended_action: str = "hold",
) -> dict[str, object]:
    return {
        "bias": bias,
        "confidence": confidence,
        "regime_label": regime_label,
        "recommended_action": recommended_action,
    }


def _row(
    *,
    symbol: str = "BTCUSDT",
    logged_at: str = "2026-04-07T00:00:00+00:00",
    selected_strategy: str = "momentum_breakout",
    rule_signal: str = "long",
    execution_allowed: bool = True,
    shadow_response: dict[str, object] | None = None,
    shadow_error: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "logged_at": logged_at,
        "symbol": symbol,
        "selected_strategy": selected_strategy,
        "rule_engine": {
            "strategy": selected_strategy,
            "signal": rule_signal,
            "bias": rule_signal,
        },
        "risk": {
            "execution_allowed": execution_allowed,
        },
        "execution": {
            "execution_allowed": execution_allowed,
        },
    }
    if shadow_response is not None or shadow_error is not None:
        row["ai_scaffold_shadow"] = {
            "enabled": True,
            "source": "ai_scaffold_static_mock",
            "annotation_mode": "read_only_shadow",
            "decision_impact": False,
            "response": shadow_response or {},
            "error": shadow_error,
        }
    return row


def test_build_ai_shadow_comparison_summary_with_shadow_rows() -> None:
    summary = build_ai_shadow_comparison_summary(
        records=[
            {
                **_row(
                    rule_signal="long",
                    execution_allowed=True,
                    shadow_response=_shadow_response(
                        bias="long",
                        confidence="high",
                        regime_label="directional_trend",
                        recommended_action="observe_long_setup",
                    ),
                ),
                "_line_number": 1,
            },
            {
                **_row(
                    symbol="ETHUSDT",
                    rule_signal="short",
                    execution_allowed=False,
                    shadow_response=_shadow_response(
                        bias="long",
                        confidence="medium",
                        regime_label="mixed_conditions",
                        recommended_action="observe_long_setup",
                    ),
                ),
                "_line_number": 2,
            },
        ],
        input_path=Path("/tmp/trade_analysis.jsonl"),
        data_quality={"valid_records": 2},
        max_mismatch_examples=3,
    )

    assert summary["metadata"]["total_rows_inspected"] == 2
    assert summary["metadata"]["rows_with_ai_scaffold_shadow"] == 2
    assert summary["metadata"]["rows_with_comparison_available"] == 2
    assert summary["metadata"]["rows_skipped_unavailable"] == 0

    assert summary["bias_alignment_summary"] == {
        "aligned": 1,
        "mismatched": 1,
        "rule_neutral_or_unknown": 0,
        "shadow_neutral_or_unknown": 0,
        "unavailable": 0,
    }
    assert summary["execution_context_summary"] == {
        "allowed_and_shadow_directional": 1,
        "allowed_and_shadow_neutral": 0,
        "blocked_and_shadow_neutral": 0,
        "blocked_but_shadow_directional": 1,
        "unavailable": 0,
    }
    assert summary["recommended_action_summary"] == [
        {
            "recommended_action": "observe_long_setup",
            "count": 2,
            "rate": 1.0,
        }
    ]
    assert summary["confidence_summary"] == [
        {"confidence": "high", "count": 1, "rate": 0.5},
        {"confidence": "medium", "count": 1, "rate": 0.5},
    ]
    assert summary["regime_label_summary"] == [
        {"regime_label": "directional_trend", "count": 1, "rate": 0.5},
        {"regime_label": "mixed_conditions", "count": 1, "rate": 0.5},
    ]
    assert summary["mismatch_summary"]["mismatch_flag_counts"] == [
        {"mismatch_flag": "bias_mismatch", "count": 1, "rate": 0.5},
        {
            "mismatch_flag": "directional_shadow_when_execution_blocked",
            "count": 1,
            "rate": 0.5,
        },
    ]
    assert summary["representative_mismatch_examples"] == [
        {
            "line_number": 2,
            "logged_at": "2026-04-07T00:00:00+00:00",
            "symbol": "ETHUSDT",
            "selected_strategy": "momentum_breakout",
            "rule_bias": "short",
            "shadow_bias": "long",
            "execution_allowed": False,
            "recommended_action": "observe_long_setup",
            "confidence": "medium",
            "regime_label": "mixed_conditions",
            "mismatch_flags": [
                "bias_mismatch",
                "directional_shadow_when_execution_blocked",
            ],
        }
    ]


def test_build_ai_shadow_comparison_summary_handles_missing_shadow_safely() -> None:
    summary = build_ai_shadow_comparison_summary(
        records=[
            {
                **_row(),
                "_line_number": 1,
            },
            {
                **_row(
                    symbol="ETHUSDT",
                    shadow_response=None,
                    shadow_error={"type": "RuntimeError", "message": "forced failure"},
                ),
                "_line_number": 2,
            },
            {
                **_row(
                    symbol="SOLUSDT",
                    shadow_response={},
                    shadow_error=None,
                ),
                "_line_number": 3,
            },
        ],
        input_path=Path("/tmp/trade_analysis.jsonl"),
        data_quality={"valid_records": 3},
        max_mismatch_examples=3,
    )

    assert summary["metadata"]["rows_with_ai_scaffold_shadow"] == 2
    assert summary["metadata"]["rows_with_comparison_available"] == 0
    assert summary["metadata"]["rows_skipped_unavailable"] == 3
    assert summary["comparison_availability"] == {
        "rows_without_ai_scaffold_shadow": 1,
        "rows_with_shadow_error": 1,
        "rows_with_empty_shadow_response": 1,
        "bias_comparison_available": 0,
        "execution_comparison_available": 0,
    }
    assert summary["recommended_action_summary"] == []
    assert summary["confidence_summary"] == []
    assert summary["regime_label_summary"] == []
    assert summary["representative_mismatch_examples"] == []


def test_build_ai_shadow_comparison_summary_is_deterministic_for_bias_classification() -> None:
    summary = build_ai_shadow_comparison_summary(
        records=[
            {
                **_row(
                    symbol="BTCUSDT",
                    rule_signal="neutral",
                    shadow_response=_shadow_response(bias="long"),
                ),
                "_line_number": 10,
            },
            {
                **_row(
                    symbol="ETHUSDT",
                    rule_signal="long",
                    shadow_response=_shadow_response(bias="neutral"),
                ),
                "_line_number": 11,
            },
            {
                **_row(
                    symbol="SOLUSDT",
                    rule_signal="unknown_value",
                    shadow_response=_shadow_response(bias="long"),
                ),
                "_line_number": 12,
            },
        ],
        input_path=Path("/tmp/trade_analysis.jsonl"),
        data_quality={"valid_records": 3},
        max_mismatch_examples=3,
    )

    assert summary["bias_alignment_summary"] == {
        "aligned": 0,
        "mismatched": 0,
        "rule_neutral_or_unknown": 1,
        "shadow_neutral_or_unknown": 1,
        "unavailable": 1,
    }
    assert summary["mismatch_summary"]["rows_with_mismatch_flags"] == 0


def test_resolve_input_path_requires_explicit_input() -> None:
    with pytest.raises(ValueError, match="input_path is required"):
        resolve_input_path(None)


def test_run_ai_shadow_comparison_report_writes_json_and_markdown(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    output_dir = tmp_path / "reports"
    _write_jsonl(
        input_path,
        [
            _row(
                rule_signal="long",
                execution_allowed=True,
                shadow_response=_shadow_response(
                    bias="long",
                    confidence="high",
                    regime_label="directional_trend",
                    recommended_action="observe_long_setup",
                ),
            ),
            _row(
                symbol="ETHUSDT",
                rule_signal="short",
                execution_allowed=False,
                shadow_response=_shadow_response(
                    bias="long",
                    confidence="medium",
                    regime_label="mixed_conditions",
                    recommended_action="observe_long_setup",
                ),
            ),
        ],
    )

    result = run_ai_shadow_comparison_report(
        input_path=input_path,
        output_dir=output_dir,
        max_mismatch_examples=2,
    )

    written_json = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    markdown = Path(result["summary_md"]).read_text(encoding="utf-8")

    assert Path(result["summary_json"]).exists()
    assert Path(result["summary_md"]).exists()
    assert written_json["metadata"]["report_type"] == "ai_shadow_comparison_report"
    assert "bias_alignment_summary" in written_json
    assert "execution_context_summary" in written_json
    assert "representative_mismatch_examples" in written_json
    assert "AI Shadow Comparison Report" in markdown
    assert "Interpretation" in markdown