from __future__ import annotations

from pathlib import Path

from src.research.historical_direct_edge_selection_diagnosis import (
    run_comparison_pipeline_step,
)


def test_run_comparison_pipeline_step_writes_snapshot_under_research_reports(
    tmp_path: Path,
) -> None:
    calls: dict[str, Path] = {}

    def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return {"ok": True}

    workspace_root = tmp_path
    result = run_comparison_pipeline_step(fake_pipeline, workspace_root)

    logs_dir = workspace_root / "logs"
    reports_dir = logs_dir / "research_reports"

    assert result == {"ok": True}
    assert calls["logs_dir"] == logs_dir
    assert calls["latest_summary"] == reports_dir / "latest" / "summary.json"
    assert calls["cumulative_output"] == (
        reports_dir / "cumulative" / "trade_analysis_cumulative_snapshot.jsonl"
    )
    assert calls["cumulative_output_dir"] == reports_dir / "cumulative"
    assert calls["comparison_output_dir"] == reports_dir / "comparison"
    assert calls["edge_scores_output_dir"] == reports_dir / "edge_scores"
    assert calls["edge_score_history_output"] == reports_dir / "edge_scores_history.jsonl"
    assert calls["score_drift_output_dir"] == reports_dir / "score_drift"
