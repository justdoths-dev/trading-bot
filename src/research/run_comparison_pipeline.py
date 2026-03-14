from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.research.comparison_report_builder import build_comparison_report
from src.research.cumulative_dataset_builder import build_cumulative_dataset
from src.research.research_analyzer import run_research_analyzer


def run_comparison_pipeline(
    logs_dir: Path | None = None,
    latest_summary: Path | None = None,
    cumulative_output: Path | None = None,
    cumulative_output_dir: Path | None = None,
    comparison_output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_logs_dir = logs_dir or _default_logs_dir()
    resolved_cumulative_output = cumulative_output or _default_cumulative_output()
    resolved_latest_summary = latest_summary or _default_latest_summary()
    resolved_cumulative_output_dir = (
        cumulative_output_dir or _default_cumulative_output_dir()
    )
    resolved_comparison_output_dir = (
        comparison_output_dir or _default_comparison_output_dir()
    )

    cumulative_dataset_summary = build_cumulative_dataset(
        logs_dir=resolved_logs_dir,
        output_path=resolved_cumulative_output,
    )

    cumulative_analysis_result = run_research_analyzer(
        input_path=resolved_cumulative_output,
        output_dir=resolved_cumulative_output_dir,
    )

    build_comparison_report(
        latest_summary_path=resolved_latest_summary,
        cumulative_summary_path=resolved_cumulative_output_dir / "summary.json",
        output_dir=resolved_comparison_output_dir,
    )

    return {
        "cumulative_dataset": cumulative_dataset_summary,
        "cumulative_analysis": {
            "records_analyzed": (cumulative_analysis_result.get("dataset_overview", {}) or {}).get(
                "total_records", 0
            ),
            "strategy_lab_dataset_rows": (cumulative_analysis_result.get("strategy_lab", {}) or {}).get(
                "dataset_rows", 0
            ),
            "summary_json": str(resolved_cumulative_output_dir / "summary.json"),
            "summary_md": str(resolved_cumulative_output_dir / "summary.md"),
        },
        "comparison_report": {
            "summary_json": str(resolved_comparison_output_dir / "summary.json"),
            "summary_md": str(resolved_comparison_output_dir / "summary.md"),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cumulative research analysis and latest-vs-cumulative comparison"
    )
    parser.add_argument("--logs-dir", type=Path, default=_default_logs_dir())
    parser.add_argument("--latest-summary", type=Path, default=_default_latest_summary())
    parser.add_argument("--cumulative-output", type=Path, default=_default_cumulative_output())
    parser.add_argument(
        "--cumulative-output-dir",
        type=Path,
        default=_default_cumulative_output_dir(),
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=_default_comparison_output_dir(),
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
