from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.research.research_analyzer import run_research_analyzer

DEFAULT_INPUT_PATH = Path(
    "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "logs/research_reports/experiments/candidate_c_asymmetric/c2_moderate"
)


def run_experimental_candidate_c_analyzer(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(
            f"Candidate C analyzer input dataset not found: {input_path}"
        )

    metrics = run_research_analyzer(input_path=input_path, output_dir=output_dir)
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "summary.json"),
        "summary_md": str(output_dir / "summary.md"),
        "dataset_overview": metrics.get("dataset_overview", {}),
        "strategy_lab": metrics.get("strategy_lab", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the research analyzer on the Candidate C2 asymmetric relabel dataset"
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_c_analyzer(
        input_path=args.input_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
