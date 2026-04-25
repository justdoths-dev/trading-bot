from __future__ import annotations

import json
from pathlib import Path

from src.research import (
    selected_strategy_edge_candidate_support_window_alignment_diagnosis_report as wrapper,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_support_window_alignment_diagnosis_report as report_module,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _record(
    *,
    logged_at: str,
    future_label_4h: str = "up",
    future_return_4h: float = 0.42,
    bias: str = "bullish",
    signal: str = "long",
) -> dict:
    return {
        "logged_at": logged_at,
        "symbol": "BTCUSDT",
        "rule_engine": {
            "strategy": "swing",
            "bias": bias,
            "signal": signal,
        },
        "risk": {
            "execution_allowed": True,
            "entry_price": 100.0,
        },
        "execution": {
            "action": signal,
            "entry_price": 100.0,
        },
        "future_label_4h": future_label_4h,
        "future_return_4h": future_return_4h,
    }


def _recovering_rows() -> list[dict]:
    rows: list[dict] = []
    for _ in range(10):
        rows.append(_record(logged_at="2026-04-12T12:00:00+00:00"))

    for _ in range(30):
        rows.append(_record(logged_at="2026-04-08T00:00:00+00:00"))
    for _ in range(20):
        rows.append(
            _record(
                logged_at="2026-04-08T01:00:00+00:00",
                future_label_4h="down",
                future_return_4h=0.18,
                bias="bearish",
                signal="short",
            )
        )
    return rows


def _absolute_floor_rows() -> list[dict]:
    return [_record(logged_at="2026-04-12T12:00:00+00:00") for _ in range(10)]


def test_wider_support_recovers_edge_rows_mapper_seeds_and_classification(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(input_path, _recovering_rows())

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[
            report_module.SupportWindowConfiguration(36, 2500, "current/latest default"),
            report_module.SupportWindowConfiguration(336, 10000),
        ],
    )

    current, wide = report["configuration_summaries"]

    assert current["source_metadata"]["windowed_record_count"] == 10
    assert current["labelability"]["labelable_basic_count"] == 10
    assert current["labelability"]["labelable_with_future_count"] == 10
    assert current["edge_candidate_rows"]["row_count"] == 0
    assert current["edge_candidate_rows"]["diagnostic_row_count"] == 3
    assert current["edge_candidate_rows"]["dominant_rejection_reason"] == (
        "failed_absolute_minimum_gate"
    )
    assert current["edge_candidate_rows"]["rejection_reason_counts"][
        "failed_absolute_minimum_gate"
    ] == 1
    assert any(
        "sample_count_below_absolute_floor" in row["rejection_reasons"]
        for row in current["edge_candidate_rows"]["top_diagnostic_rows"]
    )

    assert wide["source_metadata"]["windowed_record_count"] == 60
    assert wide["labelability"]["labelable_basic_count"] == 60
    assert wide["labelability"]["labelable_with_future_count"] == 60
    assert wide["edge_candidate_rows"]["row_count"] == 1
    assert wide["candidate_seed_diagnostics"]["candidate_seed_count"] == 1
    assert wide["candidate_seed_diagnostics"]["horizons_with_seed"] == ["4h"]
    assert wide["candidate_seed_diagnostics"]["joined_candidate_row_count"] == 1

    final = report["final_assessment"]
    assert final["classification"] == (
        "latest_support_too_sparse_but_wider_support_recovers"
    )
    assert final["wider_support_produces_edge_candidate_rows"] is True
    assert final["wider_support_produces_mapper_seeds"] is True
    assert final["horizons_with_wider_seed"] == ["4h"]
    assert "support-source alignment" in (
        final["minimal_safe_source_alignment_implication"] or ""
    )


def test_all_windows_fail_absolute_minimum_gate_remains_conservative(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(input_path, _absolute_floor_rows())

    report = report_module.build_report(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        configurations=[
            report_module.SupportWindowConfiguration(36, 2500, "current/latest default"),
            report_module.SupportWindowConfiguration(336, 10000),
        ],
    )

    assert [
        summary["edge_candidate_rows"]["row_count"]
        for summary in report["configuration_summaries"]
    ] == [0, 0]
    assert all(
        summary["classification"]["configuration_classification"]
        == "fails_absolute_minimum_gate"
        for summary in report["configuration_summaries"]
    )
    assert report["final_assessment"]["classification"] == (
        "all_windows_fail_absolute_minimum_gate"
    )
    assert "Gate relaxation or more data" in report["final_assessment"][
        "stop_rule_next_requirement"
    ]


def test_render_markdown_includes_required_window_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "trade_analysis.jsonl"
    _write_jsonl(input_path, _absolute_floor_rows())

    result = (
        report_module.run_selected_strategy_edge_candidate_support_window_alignment_diagnosis_report(
            input_path=input_path,
            output_dir=tmp_path / "reports",
            configurations=[
                report_module.SupportWindowConfiguration(36, 2500, "current/latest default"),
            ],
            write_report_copies=True,
        )
    )

    markdown = result["markdown"]
    assert "raw_record_count" in markdown
    assert "windowed_record_count" in markdown
    assert "labelable_basic_count" in markdown
    assert "edge_candidate_rows.row_count" in markdown
    assert Path(result["written_paths"]["json"]).exists()
    assert Path(result["written_paths"]["markdown"]).exists()


def test_wrapper_smoke_import_parity() -> None:
    assert wrapper.REPORT_TYPE == report_module.REPORT_TYPE
    assert wrapper.DEFAULT_CONFIGURATIONS == report_module.DEFAULT_CONFIGURATIONS
    assert (
        wrapper.run_selected_strategy_edge_candidate_support_window_alignment_diagnosis_report
        is report_module.run_selected_strategy_edge_candidate_support_window_alignment_diagnosis_report
    )
