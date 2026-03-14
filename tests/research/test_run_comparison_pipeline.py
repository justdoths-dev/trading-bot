from __future__ import annotations

from pathlib import Path
from typing import Any

from src.research import run_comparison_pipeline


def test_run_comparison_pipeline_happy_path(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    logs_dir = tmp_path / "logs"
    latest_summary = tmp_path / "reports" / "latest" / "summary.json"
    cumulative_output = tmp_path / "trade_analysis_cumulative.jsonl"
    cumulative_output_dir = tmp_path / "reports" / "cumulative"
    comparison_output_dir = tmp_path / "reports" / "comparison"

    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_cumulative_dataset",
        lambda logs_dir, output_path: calls.setdefault(
            "dataset",
            {
                "logs_dir": logs_dir,
                "output_path": output_path,
                "files_read": ["a", "b"],
                "lines_written": 25,
            },
        ),
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "run_research_analyzer",
        lambda input_path, output_dir: calls.setdefault(
            "analysis",
            {
                "input_path": input_path,
                "output_dir": output_dir,
                "dataset_overview": {"total_records": 25},
                "strategy_lab": {"dataset_rows": 20},
            },
        ),
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_comparison_report",
        lambda latest_summary_path, cumulative_summary_path, output_dir: calls.setdefault(
            "comparison",
            {
                "latest_summary_path": latest_summary_path,
                "cumulative_summary_path": cumulative_summary_path,
                "output_dir": output_dir,
            },
        ),
    )

    result = run_comparison_pipeline.run_comparison_pipeline(
        logs_dir=logs_dir,
        latest_summary=latest_summary,
        cumulative_output=cumulative_output,
        cumulative_output_dir=cumulative_output_dir,
        comparison_output_dir=comparison_output_dir,
    )

    assert calls["dataset"]["logs_dir"] == logs_dir
    assert calls["dataset"]["output_path"] == cumulative_output
    assert calls["analysis"]["input_path"] == cumulative_output
    assert calls["analysis"]["output_dir"] == cumulative_output_dir
    assert calls["comparison"]["latest_summary_path"] == latest_summary
    assert calls["comparison"]["cumulative_summary_path"] == cumulative_output_dir / "summary.json"
    assert calls["comparison"]["output_dir"] == comparison_output_dir
    assert result["cumulative_dataset"]["lines_written"] == 25
    assert result["cumulative_analysis"]["records_analyzed"] == 25
    assert result["cumulative_analysis"]["strategy_lab_dataset_rows"] == 20


def test_run_comparison_pipeline_returns_output_paths_correctly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cumulative_output_dir = tmp_path / "reports" / "cumulative"
    comparison_output_dir = tmp_path / "reports" / "comparison"

    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_cumulative_dataset",
        lambda logs_dir, output_path: {
            "files_read": [],
            "lines_written": 0,
            "output_path": str(output_path),
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "run_research_analyzer",
        lambda input_path, output_dir: {
            "dataset_overview": {"total_records": 0},
            "strategy_lab": {"dataset_rows": 0},
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_comparison_report",
        lambda latest_summary_path, cumulative_summary_path, output_dir: {},
    )

    result = run_comparison_pipeline.run_comparison_pipeline(
        logs_dir=tmp_path / "logs",
        latest_summary=tmp_path / "latest" / "summary.json",
        cumulative_output=tmp_path / "trade_analysis_cumulative.jsonl",
        cumulative_output_dir=cumulative_output_dir,
        comparison_output_dir=comparison_output_dir,
    )

    assert result["cumulative_analysis"]["summary_json"] == str(
        cumulative_output_dir / "summary.json"
    )
    assert result["cumulative_analysis"]["summary_md"] == str(
        cumulative_output_dir / "summary.md"
    )
    assert result["comparison_report"]["summary_json"] == str(
        comparison_output_dir / "summary.json"
    )
    assert result["comparison_report"]["summary_md"] == str(
        comparison_output_dir / "summary.md"
    )


def test_default_paths_resolve_correctly() -> None:
    logs_dir = run_comparison_pipeline._default_logs_dir()

    assert run_comparison_pipeline._default_cumulative_output() == logs_dir / "trade_analysis_cumulative.jsonl"
    assert run_comparison_pipeline._default_latest_summary() == logs_dir / "research_reports" / "latest" / "summary.json"
    assert run_comparison_pipeline._default_cumulative_output_dir() == logs_dir / "research_reports" / "cumulative"
    assert run_comparison_pipeline._default_comparison_output_dir() == logs_dir / "research_reports" / "comparison"


def test_run_comparison_pipeline_preserves_underlying_module_boundaries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    invocation_order: list[str] = []

    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_cumulative_dataset",
        lambda logs_dir, output_path: invocation_order.append("dataset") or {
            "files_read": [],
            "lines_written": 0,
            "output_path": str(output_path),
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "run_research_analyzer",
        lambda input_path, output_dir: invocation_order.append("analysis") or {
            "dataset_overview": {"total_records": 0},
            "strategy_lab": {"dataset_rows": 0},
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_comparison_report",
        lambda latest_summary_path, cumulative_summary_path, output_dir: invocation_order.append("comparison") or {},
    )

    run_comparison_pipeline.run_comparison_pipeline(
        logs_dir=tmp_path / "logs",
        latest_summary=tmp_path / "latest" / "summary.json",
        cumulative_output=tmp_path / "trade_analysis_cumulative.jsonl",
        cumulative_output_dir=tmp_path / "reports" / "cumulative",
        comparison_output_dir=tmp_path / "reports" / "comparison",
    )

    assert invocation_order == ["dataset", "analysis", "comparison"]


def test_run_comparison_pipeline_returns_structured_summary_shape(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_cumulative_dataset",
        lambda logs_dir, output_path: {
            "files_read": ["x"],
            "lines_written": 5,
            "output_path": str(output_path),
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "run_research_analyzer",
        lambda input_path, output_dir: {
            "dataset_overview": {"total_records": 5},
            "strategy_lab": {"dataset_rows": 4},
        },
    )
    monkeypatch.setattr(
        run_comparison_pipeline,
        "build_comparison_report",
        lambda latest_summary_path, cumulative_summary_path, output_dir: {},
    )

    result = run_comparison_pipeline.run_comparison_pipeline(
        logs_dir=tmp_path / "logs",
        latest_summary=tmp_path / "latest" / "summary.json",
        cumulative_output=tmp_path / "trade_analysis_cumulative.jsonl",
        cumulative_output_dir=tmp_path / "reports" / "cumulative",
        comparison_output_dir=tmp_path / "reports" / "comparison",
    )

    assert set(result.keys()) == {
        "cumulative_dataset",
        "cumulative_analysis",
        "comparison_report",
    }
    assert set(result["cumulative_analysis"].keys()) == {
        "records_analyzed",
        "strategy_lab_dataset_rows",
        "summary_json",
        "summary_md",
    }
    assert set(result["comparison_report"].keys()) == {"summary_json", "summary_md"}
