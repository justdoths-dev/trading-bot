from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src.research.cumulative_dataset_builder import discover_log_files
from src.research.experimental_labeling.volatility_thresholds import (
    compute_candidate_b_v1_threshold_metadata,
)

BASE_FILENAME = "trade_analysis.jsonl"
DEFAULT_INPUT_DIR = Path("logs")
DEFAULT_OUTPUT_PATH = (
    DEFAULT_INPUT_DIR / "experiments" / "trade_analysis_relabel_candidate_b_vol_adjusted.jsonl"
)
TARGET_HORIZONS = ("15m", "1h", "4h")
LABELING_METHOD = "candidate_b_volatility_adjusted_v1"


class VolatilityAdjustedRelabelError(ValueError):
    """Raised when the offline Candidate B relabel experiment configuration is invalid."""


def discover_volatility_adjusted_source_files(input_dir: Path) -> list[Path]:
    """Return rotated trade analysis source files ordered from oldest to newest."""
    return discover_log_files(logs_dir=input_dir, base_filename=BASE_FILENAME)


def build_volatility_adjusted_relabel_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """Build the Candidate B volatility-adjusted experimental relabeled dataset."""
    source_files = discover_volatility_adjusted_source_files(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_path(output_path=output_path, source_files=source_files)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "labeling_method": LABELING_METHOD,
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "source_files": [str(path) for path in source_files],
        "source_file_count": len(source_files),
        "total_records_seen": 0,
        "records_written": 0,
        "blank_line_count": 0,
        "invalid_json_line_count": 0,
        "non_object_line_count": 0,
        "fallback_row_count": 0,
        "records_with_numeric_return_15m": 0,
        "records_with_numeric_return_1h": 0,
        "records_with_numeric_return_4h": 0,
        "relabeled_count_15m": 0,
        "relabeled_count_1h": 0,
        "relabeled_count_4h": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for source_file in source_files:
            for raw_line in _iter_source_lines(source_file):
                stripped = raw_line.strip()
                if not stripped:
                    summary["blank_line_count"] += 1
                    continue

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    summary["invalid_json_line_count"] += 1
                    continue

                if not isinstance(payload, dict):
                    summary["non_object_line_count"] += 1
                    continue

                summary["total_records_seen"] += 1
                relabeled = relabel_trade_analysis_row(payload)

                if relabeled["used_fallback_atr_pct"]:
                    summary["fallback_row_count"] += 1

                for horizon in TARGET_HORIZONS:
                    if relabeled["numeric_return_available"][horizon]:
                        summary[f"records_with_numeric_return_{horizon}"] += 1
                    if relabeled["label_rebuilt"][horizon]:
                        summary[f"relabeled_count_{horizon}"] += 1

                handle.write(json.dumps(relabeled["record"], ensure_ascii=False) + "\n")
                summary["records_written"] += 1

    summary["fallback_row_ratio"] = _safe_ratio(
        summary["fallback_row_count"],
        summary["records_written"],
    )

    return summary


def relabel_trade_analysis_row(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild future labels for one row using Candidate B volatility-adjusted thresholds."""
    output = deepcopy(record)
    threshold_metadata = compute_candidate_b_v1_threshold_metadata(output)
    thresholds = threshold_metadata["thresholds"]

    numeric_return_available: dict[str, bool] = {}
    label_rebuilt: dict[str, bool] = {}

    for horizon in TARGET_HORIZONS:
        return_key = f"future_return_{horizon}"
        label_key = f"future_label_{horizon}"
        future_return = _safe_float(output.get(return_key))

        if future_return is None:
            numeric_return_available[horizon] = False
            label_rebuilt[horizon] = False
            continue

        numeric_return_available[horizon] = True
        output[label_key] = _to_label(
            future_return=future_return,
            threshold_pct=float(thresholds[horizon]),
        )
        label_rebuilt[horizon] = True

    output["experimental_labeling"] = {
        "labeling_method": LABELING_METHOD,
        "relabel_timestamp": datetime.now(UTC).isoformat(),
        "atr_pct": threshold_metadata["atr_pct"],
        "used_fallback_atr_pct": threshold_metadata["used_fallback_atr_pct"],
        "thresholds": dict(thresholds),
    }

    return {
        "record": output,
        "numeric_return_available": numeric_return_available,
        "label_rebuilt": label_rebuilt,
        "used_fallback_atr_pct": bool(threshold_metadata["used_fallback_atr_pct"]),
    }


def _safe_float(value: Any) -> float | None:
    """Safely coerce supported scalar values to float."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    return None


def _to_label(*, future_return: float, threshold_pct: float) -> str:
    """Map a numeric future return to up/down/flat using a symmetric threshold."""
    if future_return >= threshold_pct:
        return "up"
    if future_return <= -threshold_pct:
        return "down"
    return "flat"


def _iter_source_lines(path: Path) -> Iterator[str]:
    """Yield text lines from plain or gzip-compressed source files."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield line


def _validate_output_path(*, output_path: Path, source_files: list[Path]) -> None:
    """Ensure the output file does not overlap with any input source file."""
    resolved_output = output_path.resolve()
    resolved_inputs = {path.resolve() for path in source_files}

    if resolved_output in resolved_inputs:
        raise VolatilityAdjustedRelabelError(
            "Output path must not overlap with any input source file: "
            f"{output_path}"
        )


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a rounded ratio, guarding against division by zero."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Candidate B experimental relabeler."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a rotation-aware Candidate B volatility-adjusted relabeled dataset "
            "without modifying production source logs"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing rotated trade analysis logs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the experimental relabeled JSONL dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the Candidate B experimental relabeler."""
    args = parse_args()
    summary = build_volatility_adjusted_relabel_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
