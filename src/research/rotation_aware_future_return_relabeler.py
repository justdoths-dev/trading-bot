from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src.research.cumulative_dataset_builder import discover_log_files

BASE_FILENAME = "trade_analysis.jsonl"
CUMULATIVE_FILENAME = "trade_analysis_cumulative.jsonl"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "logs"
DEFAULT_OUTPUT_PATH = (
    DEFAULT_INPUT_DIR / "experiments" / "trade_analysis_relabel_candidate_a.jsonl"
)
DEFAULT_LABEL_THRESHOLDS_PCT: dict[str, float] = {
    "15m": 0.05,
    "1h": 0.10,
    "4h": 0.15,
}
TARGET_HORIZONS = ("15m", "1h", "4h")


class RelabelExperimentError(ValueError):
    """Raised when the offline relabel experiment configuration is invalid."""


def discover_experiment_source_files(
    input_dir: Path,
    *,
    include_cumulative_source: bool = False,
) -> list[Path]:
    """Return experiment input files ordered from oldest rotation to newest.

    The module name is intentionally explicit: this is a rotation-aware relabeler that
    rebuilds labels from existing future returns only, without fetching prices or
    mutating production logs in place.
    """
    source_files = discover_log_files(logs_dir=input_dir, base_filename=BASE_FILENAME)

    if include_cumulative_source:
        cumulative_path = input_dir / CUMULATIVE_FILENAME
        if cumulative_path.exists() and cumulative_path.is_file():
            source_files = [*source_files, cumulative_path]

    return source_files


def build_rotation_aware_future_return_relabel_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    threshold_15m: float = DEFAULT_LABEL_THRESHOLDS_PCT["15m"],
    threshold_1h: float = DEFAULT_LABEL_THRESHOLDS_PCT["1h"],
    threshold_4h: float = DEFAULT_LABEL_THRESHOLDS_PCT["4h"],
    include_cumulative_source: bool = False,
) -> dict[str, Any]:
    """Build an offline experimental relabeled dataset from rotated trade analysis logs."""
    thresholds_pct = _resolve_thresholds(
        {
            "15m": threshold_15m,
            "1h": threshold_1h,
            "4h": threshold_4h,
        }
    )
    source_files = discover_experiment_source_files(
        input_dir,
        include_cumulative_source=include_cumulative_source,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_path(output_path=output_path, source_files=source_files)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_files": [str(path) for path in source_files],
        "source_file_count": len(source_files),
        "total_records_seen": 0,
        "records_written": 0,
        "blank_line_count": 0,
        "invalid_json_line_count": 0,
        "non_object_line_count": 0,
        "records_with_numeric_return_15m": 0,
        "records_with_numeric_return_1h": 0,
        "records_with_numeric_return_4h": 0,
        "relabeled_count_15m": 0,
        "relabeled_count_1h": 0,
        "relabeled_count_4h": 0,
        "output_path": str(output_path),
        "thresholds_pct": thresholds_pct,
        "include_cumulative_source": include_cumulative_source,
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
                relabeled = _relabel_record(
                    record=payload,
                    thresholds_pct=thresholds_pct,
                )

                for horizon in TARGET_HORIZONS:
                    if relabeled["numeric_return_available"][horizon]:
                        summary[f"records_with_numeric_return_{horizon}"] += 1
                    if relabeled["label_rebuilt"][horizon]:
                        summary[f"relabeled_count_{horizon}"] += 1

                handle.write(json.dumps(relabeled["record"], ensure_ascii=False) + "\n")
                summary["records_written"] += 1

    return summary


def _relabel_record(
    *,
    record: dict[str, Any],
    thresholds_pct: dict[str, float],
) -> dict[str, Any]:
    output = deepcopy(record)
    numeric_return_available: dict[str, bool] = {}
    label_rebuilt: dict[str, bool] = {}

    for horizon in TARGET_HORIZONS:
        return_key = f"future_return_{horizon}"
        label_key = f"future_label_{horizon}"
        numeric_return = _safe_float(output.get(return_key))

        if numeric_return is None:
            numeric_return_available[horizon] = False
            label_rebuilt[horizon] = False
            continue

        numeric_return_available[horizon] = True
        output[label_key] = _to_label(
            future_return=numeric_return,
            threshold_pct=thresholds_pct[horizon],
        )
        label_rebuilt[horizon] = True

    return {
        "record": output,
        "numeric_return_available": numeric_return_available,
        "label_rebuilt": label_rebuilt,
    }


def _safe_float(value: Any) -> float | None:
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


def _resolve_thresholds(provided: dict[str, Any]) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    for horizon in TARGET_HORIZONS:
        raw_value = provided.get(horizon)
        numeric_value = _safe_float(raw_value)
        if numeric_value is None:
            raise RelabelExperimentError(f"Invalid threshold for horizon {horizon}: {raw_value}")
        if numeric_value < 0:
            raise RelabelExperimentError(
                f"Threshold must be non-negative for horizon {horizon}: {numeric_value}"
            )
        thresholds[horizon] = numeric_value

    return thresholds


def _to_label(*, future_return: float, threshold_pct: float) -> str:
    if future_return > threshold_pct:
        return "up"
    if future_return < -threshold_pct:
        return "down"
    return "flat"


def _iter_source_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield line


def _validate_output_path(*, output_path: Path, source_files: list[Path]) -> None:
    resolved_output = output_path.resolve()
    resolved_inputs = {path.resolve() for path in source_files}

    if resolved_output in resolved_inputs:
        raise RelabelExperimentError(
            "Output path must not overlap with any input source file: "
            f"{output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an experimental future-return relabeled dataset from rotated "
            "trade analysis logs without modifying production source logs"
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
    parser.add_argument(
        "--threshold-15m",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["15m"],
        help="Absolute threshold percentage for 15m relabeling.",
    )
    parser.add_argument(
        "--threshold-1h",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["1h"],
        help="Absolute threshold percentage for 1h relabeling.",
    )
    parser.add_argument(
        "--threshold-4h",
        type=float,
        default=DEFAULT_LABEL_THRESHOLDS_PCT["4h"],
        help="Absolute threshold percentage for 4h relabeling.",
    )
    parser.add_argument(
        "--include-cumulative-source",
        action="store_true",
        help=(
            "Also append trade_analysis_cumulative.jsonl as an explicit extra source. "
            "Disabled by default to avoid duplicate-history experiments."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_rotation_aware_future_return_relabel_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        threshold_15m=args.threshold_15m,
        threshold_1h=args.threshold_1h,
        threshold_4h=args.threshold_4h,
        include_cumulative_source=args.include_cumulative_source,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
