from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.research.experimental_candidate_comparison_utils import (
    build_candidate_summary_comparison,
    load_summary_json,
    write_candidate_summary_comparison,
)

DEFAULT_BASELINE_PATH = Path(
    "logs/research_reports/experiments/candidate_a/latest/summary.json"
)
DEFAULT_EXPERIMENT_PATH = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate/summary.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_c_comparison.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_c_comparison.md"
)


def build_experimental_candidate_ac_comparison_report(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    baseline_path: Path,
    experiment_path: Path,
    baseline_loaded: bool = True,
    experiment_loaded: bool = True,
) -> dict[str, Any]:
    return build_candidate_summary_comparison(
        baseline_summary,
        experiment_summary,
        baseline_name="candidate_a",
        experiment_name="candidate_c",
        baseline_summary_path=baseline_path,
        experiment_summary_path=experiment_path,
        mode="a_vs_c",
        baseline_loaded=baseline_loaded,
        experiment_loaded=experiment_loaded,
    )


def run_experimental_candidate_ac_comparison_report(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    experiment_path: Path = DEFAULT_EXPERIMENT_PATH,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    baseline_summary, baseline_loaded = load_summary_json(baseline_path)
    experiment_summary, experiment_loaded = load_summary_json(experiment_path)

    summary = build_experimental_candidate_ac_comparison_report(
        baseline_summary,
        experiment_summary,
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        baseline_loaded=baseline_loaded,
        experiment_loaded=experiment_loaded,
    )
    outputs = write_candidate_summary_comparison(
        summary,
        json_output_path=json_output_path,
        markdown_output_path=markdown_output_path,
    )

    return {
        "summary": summary,
        **outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Candidate A vs Candidate C experiment-summary comparison report"
    )
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--experiment-summary", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_ac_comparison_report(
        baseline_path=args.baseline_summary,
        experiment_path=args.experiment_summary,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
