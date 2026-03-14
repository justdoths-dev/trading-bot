from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.research.comparison_report_builder import build_comparison_report
from src.research.cumulative_dataset_builder import build_cumulative_dataset
from src.research.edge_score_history_builder import build_edge_score_history
from src.research.edge_stability_score_builder import build_edge_stability_scores
from src.research.research_analyzer import run_research_analyzer
from src.research.score_drift_analyzer import build_score_drift_report


def run_comparison_pipeline(
    logs_dir: Path | None = None,
    latest_summary: Path | None = None,
    cumulative_output: Path | None = None,
    cumulative_output_dir: Path | None = None,
    comparison_output_dir: Path | None = None,
    edge_scores_output_dir: Path | None = None,
    edge_score_history_output: Path | None = None,
    score_drift_output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_logs_dir = logs_dir or _default_logs_dir()
    resolved_latest_summary = latest_summary or _default_latest_summary()
    resolved_cumulative_output = cumulative_output or _default_cumulative_output()
    resolved_cumulative_output_dir = (
        cumulative_output_dir or _default_cumulative_output_dir()
    )
    resolved_comparison_output_dir = (
        comparison_output_dir or _default_comparison_output_dir()
    )
    resolved_edge_scores_output_dir = (
        edge_scores_output_dir or _default_edge_scores_output_dir()
    )
    resolved_edge_score_history_output = (
        edge_score_history_output or _default_edge_score_history_output()
    )
    resolved_score_drift_output_dir = (
        score_drift_output_dir or _default_score_drift_output_dir()
    )

    cumulative_summary_path = resolved_cumulative_output_dir / "summary.json"
    cumulative_summary_md_path = resolved_cumulative_output_dir / "summary.md"

    comparison_summary_path = resolved_comparison_output_dir / "summary.json"
    comparison_summary_md_path = resolved_comparison_output_dir / "summary.md"

    edge_scores_summary_path = resolved_edge_scores_output_dir / "summary.json"
    edge_scores_summary_md_path = resolved_edge_scores_output_dir / "summary.md"

    score_drift_summary_path = resolved_score_drift_output_dir / "summary.json"
    score_drift_summary_md_path = resolved_score_drift_output_dir / "summary.md"

    cumulative_dataset_summary = build_cumulative_dataset(
        logs_dir=resolved_logs_dir,
        output_path=resolved_cumulative_output,
    )

    cumulative_analysis_result = run_research_analyzer(
        input_path=resolved_cumulative_output,
        output_dir=resolved_cumulative_output_dir,
    )

    comparison_report = build_comparison_report(
        latest_summary_path=resolved_latest_summary,
        cumulative_summary_path=cumulative_summary_path,
        output_dir=resolved_comparison_output_dir,
    )

    edge_stability_scores = build_edge_stability_scores(
        input_path=comparison_summary_path,
        output_dir=resolved_edge_scores_output_dir,
    )

    edge_score_history = build_edge_score_history(
        input_path=edge_scores_summary_path,
        output_path=resolved_edge_score_history_output,
    )

    score_drift = build_score_drift_report(
        input_path=resolved_edge_score_history_output,
        output_dir=resolved_score_drift_output_dir,
    )

    return {
        "cumulative_dataset": cumulative_dataset_summary,
        "cumulative_analysis": {
            "records_analyzed": (
                cumulative_analysis_result.get("dataset_overview", {}) or {}
            ).get("total_records", 0),
            "strategy_lab_dataset_rows": (
                cumulative_analysis_result.get("strategy_lab", {}) or {}
            ).get("dataset_rows", 0),
            "summary_json": str(cumulative_summary_path),
            "summary_md": str(cumulative_summary_md_path),
        },
        "comparison_report": {
            **comparison_report,
            "summary_json": str(comparison_summary_path),
            "summary_md": str(comparison_summary_md_path),
        },
        "edge_stability_scores": {
            **edge_stability_scores,
            "summary_json": str(edge_scores_summary_path),
            "summary_md": str(edge_scores_summary_md_path),
        },
        "edge_score_history": edge_score_history,
        "score_drift": {
            **score_drift,
            "summary_json": str(score_drift_summary_path),
            "summary_md": str(score_drift_summary_md_path),
        },
    }


def _default_logs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs"


def _default_cumulative_output() -> Path:
    return _default_logs_dir() / "trade_analysis_cumulative.jsonl"


def _default_latest_summary() -> Path:
    return _default_logs_dir() / "research_reports" / "latest" / "summary.json"


def _default_cumulative_output_dir() -> Path:
    return _default_logs_dir() / "research_reports" / "cumulative"


def _default_comparison_output_dir() -> Path:
    return _default_logs_dir() / "research_reports" / "comparison"


def _default_edge_scores_output_dir() -> Path:
    return _default_logs_dir() / "research_reports" / "edge_scores"


def _default_edge_score_history_output() -> Path:
    return _default_logs_dir() / "research_reports" / "edge_scores_history.jsonl"


def _default_score_drift_output_dir() -> Path:
    return _default_logs_dir() / "research_reports" / "score_drift"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative research analysis, comparison reporting, and "
            "downstream observational score stages"
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=_default_logs_dir(),
        help="Directory containing trade analysis logs and research report outputs",
    )
    parser.add_argument(
        "--latest-summary",
        type=Path,
        default=_default_latest_summary(),
        help="Path to latest research summary.json",
    )
    parser.add_argument(
        "--cumulative-output",
        type=Path,
        default=_default_cumulative_output(),
        help="Path to write cumulative trade analysis JSONL",
    )
    parser.add_argument(
        "--cumulative-output-dir",
        type=Path,
        default=_default_cumulative_output_dir(),
        help="Directory for cumulative research analyzer outputs",
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=_default_comparison_output_dir(),
        help="Directory for comparison report outputs",
    )
    parser.add_argument(
        "--edge-scores-output-dir",
        type=Path,
        default=_default_edge_scores_output_dir(),
        help="Directory for edge stability score outputs",
    )
    parser.add_argument(
        "--edge-score-history-output",
        type=Path,
        default=_default_edge_score_history_output(),
        help="Path to append edge score history JSONL",
    )
    parser.add_argument(
        "--score-drift-output-dir",
        type=Path,
        default=_default_score_drift_output_dir(),
        help="Directory for score drift analysis outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_comparison_pipeline(
        logs_dir=args.logs_dir,
        latest_summary=args.latest_summary,
        cumulative_output=args.cumulative_output,
        cumulative_output_dir=args.cumulative_output_dir,
        comparison_output_dir=args.comparison_output_dir,
        edge_scores_output_dir=args.edge_scores_output_dir,
        edge_score_history_output=args.edge_score_history_output,
        score_drift_output_dir=args.score_drift_output_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
