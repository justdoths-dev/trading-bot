from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from src.research.schema_validator import validate_record

DEFAULT_CUMULATIVE_PATH = Path("logs/trade_analysis_cumulative.jsonl")
DEFAULT_CURRENT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_PATH = Path("logs/research_reports/latest/resolved_historical_input.jsonl")
DEFAULT_MANIFEST_PATH = Path(
    "logs/research_reports/latest/resolved_historical_input_manifest.json"
)
DEFAULT_MARKDOWN_PATH = Path(
    "logs/research_reports/latest/resolved_historical_input_manifest.md"
)
DEFAULT_OVERLAP_VALIDATION_REPORT_PATH = Path(
    "logs/research_reports/latest/jsonl_overlap_validation_report.json"
)

FINGERPRINT_METHOD = "sha256(canonical_json_sorted_keys_utf8)"
RESOLUTION_MODE = "cumulative_plus_current_non_overlap"
BUILDER_VERSION = "v1"


@dataclass
class SourceScanStats:
    total_lines: int = 0
    valid_records: int = 0
    invalid_json_lines: int = 0
    blank_lines: int = 0
    invalid_schema_records: int = 0


def canonicalize_record(record: Any) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def fingerprint_record(record: Any) -> tuple[str, str]:
    canonical_json = canonicalize_record(record)
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return fingerprint, canonical_json


def build_resolved_historical_input(
    cumulative_path: Path = DEFAULT_CUMULATIVE_PATH,
    current_path: Path = DEFAULT_CURRENT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
    overlap_validation_report_path: Path = DEFAULT_OVERLAP_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    if not cumulative_path.exists():
        raise FileNotFoundError(f"Cumulative input file does not exist: {cumulative_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current input file does not exist: {current_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    cumulative_stats = SourceScanStats()
    current_stats = SourceScanStats()

    cumulative_fingerprints: set[str] = set()
    current_unique_fingerprints: set[str] = set()
    current_overlap_unique_fingerprints: set[str] = set()
    current_non_overlap_unique_fingerprints: set[str] = set()
    output_unique_fingerprints: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_handle:
        cumulative_valid_records_written = _write_cumulative_records(
            input_path=cumulative_path,
            output_handle=output_handle,
            stats=cumulative_stats,
            cumulative_fingerprints=cumulative_fingerprints,
            output_unique_fingerprints=output_unique_fingerprints,
        )
        (
            current_non_overlap_records_appended,
            current_overlap_record_count,
        ) = _append_current_non_overlap_records(
            input_path=current_path,
            output_handle=output_handle,
            stats=current_stats,
            cumulative_fingerprints=cumulative_fingerprints,
            current_unique_fingerprints=current_unique_fingerprints,
            current_overlap_unique_fingerprints=current_overlap_unique_fingerprints,
            current_non_overlap_unique_fingerprints=current_non_overlap_unique_fingerprints,
            output_unique_fingerprints=output_unique_fingerprints,
        )

    cumulative_unique_record_count = len(cumulative_fingerprints)
    current_unique_record_count = len(current_unique_fingerprints)
    current_overlap_unique_record_count = len(current_overlap_unique_fingerprints)
    current_non_overlap_unique_record_count = len(current_non_overlap_unique_fingerprints)

    cumulative_internal_duplicate_record_count = (
        cumulative_valid_records_written - cumulative_unique_record_count
    )
    current_internal_duplicate_record_count = (
        current_stats.valid_records - current_unique_record_count
    )

    total_output_records = (
        cumulative_valid_records_written + current_non_overlap_records_appended
    )
    total_output_unique_records = len(output_unique_fingerprints)
    output_internal_duplicate_record_count = (
        total_output_records - total_output_unique_records
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "builder_version": BUILDER_VERSION,
        "mode": RESOLUTION_MODE,
        "cumulative_path": str(cumulative_path),
        "current_path": str(current_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "markdown_path": str(markdown_path),
        "overlap_validation_report_path": str(overlap_validation_report_path),
        "fingerprint_method": FINGERPRINT_METHOD,
        "cross_file_overlap_resolution_applied": True,
        "intra_file_dedup_applied": False,
        "cumulative_valid_records_written": cumulative_valid_records_written,
        "cumulative_unique_record_count": cumulative_unique_record_count,
        "cumulative_internal_duplicate_record_count": (
            cumulative_internal_duplicate_record_count
        ),
        "current_valid_record_count": current_stats.valid_records,
        "current_unique_record_count": current_unique_record_count,
        "current_non_overlap_records_appended": current_non_overlap_records_appended,
        "current_non_overlap_unique_record_count": (
            current_non_overlap_unique_record_count
        ),
        "current_overlap_record_count": current_overlap_record_count,
        "current_overlap_unique_record_count": current_overlap_unique_record_count,
        "current_internal_duplicate_record_count": (
            current_internal_duplicate_record_count
        ),
        "invalid_json_lines": (
            cumulative_stats.invalid_json_lines + current_stats.invalid_json_lines
        ),
        "blank_lines": cumulative_stats.blank_lines + current_stats.blank_lines,
        "invalid_schema_records": (
            cumulative_stats.invalid_schema_records
            + current_stats.invalid_schema_records
        ),
        "total_output_records": total_output_records,
        "total_output_unique_records": total_output_unique_records,
        "output_internal_duplicate_record_count": (
            output_internal_duplicate_record_count
        ),
        "decision_summary": (
            "Resolved historical input writes all schema-valid cumulative records first "
            "and then appends only schema-valid current records whose canonical JSON "
            "fingerprints are absent from cumulative. Cross-file overlap is resolved, "
            "but intra-file duplicates are intentionally preserved."
        ),
        "overlap_validation_basis": {
            "policy": RESOLUTION_MODE,
            "current_unique_record_count": current_unique_record_count,
            "current_overlap_unique_record_count": current_overlap_unique_record_count,
            "current_non_overlap_unique_record_count": (
                current_non_overlap_unique_record_count
            ),
            "reason": (
                "Cumulative-only historical reruns would omit current-only valid records, "
                "while including the full current file would reintroduce overlap already "
                "present in cumulative."
            ),
        },
        "source_stats": {
            "cumulative": _stats_to_manifest(cumulative_path, cumulative_stats),
            "current": _stats_to_manifest(current_path, current_stats),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown_summary(manifest) + "\n", encoding="utf-8")

    return manifest


def _write_cumulative_records(
    *,
    input_path: Path,
    output_handle: TextIO,
    stats: SourceScanStats,
    cumulative_fingerprints: set[str],
    output_unique_fingerprints: set[str],
) -> int:
    written = 0

    with input_path.open("r", encoding="utf-8") as input_handle:
        for raw_line in input_handle:
            parsed = _parse_valid_record(raw_line=raw_line, stats=stats)
            if parsed is None:
                continue

            fingerprint, canonical_json = parsed
            cumulative_fingerprints.add(fingerprint)
            output_unique_fingerprints.add(fingerprint)
            output_handle.write(canonical_json)
            output_handle.write("\n")
            written += 1

    return written


def _append_current_non_overlap_records(
    *,
    input_path: Path,
    output_handle: TextIO,
    stats: SourceScanStats,
    cumulative_fingerprints: set[str],
    current_unique_fingerprints: set[str],
    current_overlap_unique_fingerprints: set[str],
    current_non_overlap_unique_fingerprints: set[str],
    output_unique_fingerprints: set[str],
) -> tuple[int, int]:
    appended = 0
    overlap_record_count = 0

    with input_path.open("r", encoding="utf-8") as input_handle:
        for raw_line in input_handle:
            parsed = _parse_valid_record(raw_line=raw_line, stats=stats)
            if parsed is None:
                continue

            fingerprint, canonical_json = parsed
            current_unique_fingerprints.add(fingerprint)

            if fingerprint in cumulative_fingerprints:
                current_overlap_unique_fingerprints.add(fingerprint)
                overlap_record_count += 1
                continue

            current_non_overlap_unique_fingerprints.add(fingerprint)
            output_unique_fingerprints.add(fingerprint)
            output_handle.write(canonical_json)
            output_handle.write("\n")
            appended += 1

    return appended, overlap_record_count


def _parse_valid_record(
    *,
    raw_line: str,
    stats: SourceScanStats,
) -> tuple[str, str] | None:
    stats.total_lines += 1
    stripped = raw_line.strip()

    if not stripped:
        stats.blank_lines += 1
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        stats.invalid_json_lines += 1
        return None

    if not isinstance(parsed, dict):
        stats.invalid_schema_records += 1
        return None

    try:
        validation_result = validate_record(parsed)
    except Exception:
        stats.invalid_schema_records += 1
        return None

    if not isinstance(validation_result, dict) or not validation_result.get("is_valid", False):
        stats.invalid_schema_records += 1
        return None

    stats.valid_records += 1
    return fingerprint_record(parsed)


def _stats_to_manifest(path: Path, stats: SourceScanStats) -> dict[str, Any]:
    return {
        "path": str(path),
        "total_lines": stats.total_lines,
        "valid_records": stats.valid_records,
        "invalid_json_lines": stats.invalid_json_lines,
        "blank_lines": stats.blank_lines,
        "invalid_schema_records": stats.invalid_schema_records,
    }


def build_markdown_summary(manifest: dict[str, Any]) -> str:
    overlap_basis = manifest["overlap_validation_basis"]
    cumulative_stats = manifest["source_stats"]["cumulative"]
    current_stats = manifest["source_stats"]["current"]

    lines = [
        "# Resolved Historical Input Manifest",
        "",
        "## Summary",
        f"- generated_at: {manifest['generated_at']}",
        f"- builder_version: {manifest['builder_version']}",
        f"- mode: {manifest['mode']}",
        f"- fingerprint_method: {manifest['fingerprint_method']}",
        f"- output_path: `{manifest['output_path']}`",
        f"- overlap_validation_report_path: `{manifest['overlap_validation_report_path']}`",
        f"- cross_file_overlap_resolution_applied: {manifest['cross_file_overlap_resolution_applied']}",
        f"- intra_file_dedup_applied: {manifest['intra_file_dedup_applied']}",
        f"- cumulative_valid_records_written: {manifest['cumulative_valid_records_written']}",
        f"- cumulative_unique_record_count: {manifest['cumulative_unique_record_count']}",
        f"- cumulative_internal_duplicate_record_count: {manifest['cumulative_internal_duplicate_record_count']}",
        f"- current_valid_record_count: {manifest['current_valid_record_count']}",
        f"- current_unique_record_count: {manifest['current_unique_record_count']}",
        f"- current_non_overlap_records_appended: {manifest['current_non_overlap_records_appended']}",
        f"- current_non_overlap_unique_record_count: {manifest['current_non_overlap_unique_record_count']}",
        f"- current_overlap_record_count: {manifest['current_overlap_record_count']}",
        f"- current_overlap_unique_record_count: {manifest['current_overlap_unique_record_count']}",
        f"- current_internal_duplicate_record_count: {manifest['current_internal_duplicate_record_count']}",
        f"- invalid_json_lines: {manifest['invalid_json_lines']}",
        f"- blank_lines: {manifest['blank_lines']}",
        f"- invalid_schema_records: {manifest['invalid_schema_records']}",
        f"- total_output_records: {manifest['total_output_records']}",
        f"- total_output_unique_records: {manifest['total_output_unique_records']}",
        f"- output_internal_duplicate_record_count: {manifest['output_internal_duplicate_record_count']}",
        "",
        "## Inputs",
        f"- cumulative_path: `{manifest['cumulative_path']}`",
        f"- current_path: `{manifest['current_path']}`",
        "",
        "## Decision",
        f"- summary: {manifest['decision_summary']}",
        f"- current_unique_record_count: {overlap_basis['current_unique_record_count']}",
        f"- current_overlap_unique_record_count: {overlap_basis['current_overlap_unique_record_count']}",
        f"- current_non_overlap_unique_record_count: {overlap_basis['current_non_overlap_unique_record_count']}",
        f"- reason: {overlap_basis['reason']}",
        "",
        "## Source Stats",
        f"- cumulative_total_lines: {cumulative_stats['total_lines']}",
        f"- cumulative_valid_records: {cumulative_stats['valid_records']}",
        f"- cumulative_invalid_json_lines: {cumulative_stats['invalid_json_lines']}",
        f"- cumulative_blank_lines: {cumulative_stats['blank_lines']}",
        f"- cumulative_invalid_schema_records: {cumulative_stats['invalid_schema_records']}",
        f"- current_total_lines: {current_stats['total_lines']}",
        f"- current_valid_records: {current_stats['valid_records']}",
        f"- current_invalid_json_lines: {current_stats['invalid_json_lines']}",
        f"- current_blank_lines: {current_stats['blank_lines']}",
        f"- current_invalid_schema_records: {current_stats['invalid_schema_records']}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resolved historical JSONL input by writing cumulative records "
            "first and then appending only current records not already present in cumulative."
        )
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=DEFAULT_CUMULATIVE_PATH,
        help="Path to the cumulative trade analysis JSONL file.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT_PATH,
        help="Path to the current trade analysis JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the resolved historical JSONL file.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to write the JSON manifest summary.",
    )
    parser.add_argument(
        "--manifest-md",
        type=Path,
        default=DEFAULT_MARKDOWN_PATH,
        help="Path to write the Markdown summary.",
    )
    parser.add_argument(
        "--overlap-validation-report",
        type=Path,
        default=DEFAULT_OVERLAP_VALIDATION_REPORT_PATH,
        help="Path to the overlap validation report used as policy basis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_resolved_historical_input(
        cumulative_path=args.cumulative,
        current_path=args.current,
        output_path=args.output,
        manifest_path=args.manifest_json,
        markdown_path=args.manifest_md,
        overlap_validation_report_path=args.overlap_validation_report,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
