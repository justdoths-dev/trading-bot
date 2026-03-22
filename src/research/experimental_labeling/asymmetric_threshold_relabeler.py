from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src.research.cumulative_dataset_builder import discover_log_files
from src.research.experimental_labeling.asymmetric_threshold_config import (
    DEFAULT_VARIANT_NAME,
    TARGET_HORIZONS,
    get_asymmetric_threshold_variant,
)
from src.research.experimental_labeling.asymmetric_thresholds import (
    build_threshold_map,
    compute_asymmetric_threshold_labels,
)

BASE_FILENAME = "trade_analysis.jsonl"
DEFAULT_INPUT_DIR = Path("logs")
LABELING_METHOD = "candidate_c_asymmetric_threshold_v1"


class AsymmetricThresholdRelabelError(ValueError):
    """Raised when the offline Candidate C relabel experiment configuration is invalid."""


def build_default_output_path(variant_name: str = DEFAULT_VARIANT_NAME) -> Path:
    return Path(
        f"logs/experiments/trade_analysis_relabel_candidate_c_{variant_name}.jsonl"
    )


def discover_asymmetric_threshold_source_files(input_dir: Path) -> list[Path]:
    """Return rotated trade analysis source files ordered from oldest to newest."""
    return discover_log_files(logs_dir=input_dir, base_filename=BASE_FILENAME)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _iter_source_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield line


def _validate_output_path(*, output_path: Path, source_files: list[Path]) -> None:
    resolved_output = output_path.resolve()
    resolved_inputs = {path.resolve() for path in source_files}

    if resolved_output in resolved_inputs:
        raise AsymmetricThresholdRelabelError(
            "Output path must not overlap with any input source file: "
            f"{output_path}"
        )


def relabel_trade_analysis_row(
    record: dict[str, Any],
    *,
    variant_name: str = DEFAULT_VARIANT_NAME,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Rebuild future labels for one row using Candidate C asymmetric thresholds."""
    output = deepcopy(record)
    config = get_asymmetric_threshold_variant(variant_name)
    thresholds = build_threshold_map(config)

    computed_labels = compute_asymmetric_threshold_labels(output, config)

    numeric_return_available: dict[str, bool] = {}
    label_rebuilt: dict[str, bool] = {}

    for horizon in TARGET_HORIZONS:
        label_key = f"future_label_{horizon}"
        next_label = computed_labels[horizon]

        if next_label is None:
            numeric_return_available[horizon] = False
            label_rebuilt[horizon] = False
            continue

        numeric_return_available[horizon] = True
        output[label_key] = next_label
        label_rebuilt[horizon] = True

    output["experimental_labeling"] = {
        "labeling_method": LABELING_METHOD,
        "variant": config.variant_name,
        "relabel_timestamp": datetime.now(UTC).isoformat(),
        "thresholds": deepcopy(thresholds),
        "source_path": str(source_path) if source_path is not None else None,
        "numeric_return_available_by_horizon": deepcopy(numeric_return_available),
        "label_rebuilt_by_horizon": deepcopy(label_rebuilt),
    }

    return {
        "record": output,
        "numeric_return_available": numeric_return_available,
        "label_rebuilt": label_rebuilt,
        "variant": config.variant_name,
    }


def build_asymmetric_threshold_relabel_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path | None = None,
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> dict[str, Any]:
    """Build the Candidate C asymmetric-threshold experimental relabeled dataset."""
    config = get_asymmetric_threshold_variant(variant_name)
    resolved_output_path = output_path or build_default_output_path(config.variant_name)
    source_files = discover_asymmetric_threshold_source_files(input_dir)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_path(output_path=resolved_output_path, source_files=source_files)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "labeling_method": LABELING_METHOD,
        "variant": config.variant_name,
        "thresholds": build_threshold_map(config),
        "input_dir": str(input_dir),
        "output_path": str(resolved_output_path),
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
    }

    with resolved_output_path.open("w", encoding="utf-8") as handle:
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
                relabeled = relabel_trade_analysis_row(
                    payload,
                    variant_name=config.variant_name,
                    source_path=source_file,
                )

                for horizon in TARGET_HORIZONS:
                    if relabeled["numeric_return_available"][horizon]:
                        summary[f"records_with_numeric_return_{horizon}"] += 1
                    if relabeled["label_rebuilt"][horizon]:
                        summary[f"relabeled_count_{horizon}"] += 1

                handle.write(json.dumps(relabeled["record"], ensure_ascii=False) + "\n")
                summary["records_written"] += 1

    summary["records_written_ratio"] = _safe_ratio(
        summary["records_written"],
        summary["total_records_seen"],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rotation-aware Candidate C asymmetric-threshold relabeled dataset "
            "without modifying production source logs"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_asymmetric_threshold_relabel_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        variant_name=args.variant,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
