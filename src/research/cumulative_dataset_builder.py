from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

BASE_FILENAME = "trade_analysis.jsonl"


def build_cumulative_dataset(
    logs_dir: Path,
    output_path: Path,
    base_filename: str = BASE_FILENAME,
) -> dict[str, object]:
    """Merge active and rotated trade analysis logs into one cumulative JSONL file."""
    input_paths = discover_log_files(logs_dir=logs_dir, base_filename=base_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_output = output_path.resolve()
    resolved_inputs = {path.resolve() for path in input_paths}
    if resolved_output in resolved_inputs:
        raise ValueError(
            "Output path must not overlap with any input log file: "
            f"{output_path}"
        )

    lines_written = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for input_path in input_paths:
            for line in _iter_log_lines(input_path):
                output_file.write(line)
                lines_written += 1

    return {
        "files_read": [str(path) for path in input_paths],
        "lines_written": lines_written,
        "output_path": str(output_path),
    }


def discover_log_files(
    logs_dir: Path,
    base_filename: str = BASE_FILENAME,
) -> list[Path]:
    """Return available trade analysis log files ordered from oldest rotation to active."""
    candidates = [
        logs_dir / base_filename,
        logs_dir / f"{base_filename}.1",
    ]
    candidates.extend(logs_dir.glob(f"{base_filename}.*.gz"))

    existing_paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path.is_file() and path not in seen:
            existing_paths.append(path)
            seen.add(path)

    return sorted(
        existing_paths,
        key=lambda path: _log_sort_key(path, base_filename),
    )


def _iter_log_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield line


def _log_sort_key(path: Path, base_filename: str) -> tuple[int, int]:
    rotation_number = _extract_rotation_number(path, base_filename)
    if rotation_number is None:
        return (1, 0)
    return (0, -rotation_number)


def _extract_rotation_number(path: Path, base_filename: str) -> int | None:
    name = path.name

    if name == base_filename:
        return None

    if name.startswith(f"{base_filename}."):
        suffix = name[len(base_filename) + 1 :]
        if suffix.endswith(".gz"):
            suffix = suffix[:-3]
        if suffix.isdigit():
            return int(suffix)

    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge rotated trade analysis logs into a cumulative JSONL dataset"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Directory containing trade analysis log files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged cumulative JSONL file",
    )
    parser.add_argument(
        "--base-filename",
        type=str,
        default=BASE_FILENAME,
        help="Base filename to merge, default: trade_analysis.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_cumulative_dataset(
        logs_dir=args.logs_dir,
        output_path=args.output,
        base_filename=args.base_filename,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
