from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class JsonlRecordFingerprint:
    line_number: int
    fingerprint: str
    canonical_json: str


@dataclass(frozen=True)
class JsonlScanSummary:
    path: str
    total_lines: int
    valid_records: int
    invalid_json_lines: int
    blank_lines: int
    unique_fingerprints: int
    internal_duplicate_records: int


def canonicalize_record(record: Any) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def fingerprint_record(record: Any) -> tuple[str, str]:
    canonical_json = canonicalize_record(record)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest, canonical_json


def iter_jsonl_fingerprints(path: Path) -> Iterable[JsonlRecordFingerprint]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                yield JsonlRecordFingerprint(
                    line_number=line_number,
                    fingerprint="__INVALID_JSON__",
                    canonical_json=stripped,
                )
                continue

            fingerprint, canonical_json = fingerprint_record(record)
            yield JsonlRecordFingerprint(
                line_number=line_number,
                fingerprint=fingerprint,
                canonical_json=canonical_json,
            )


def scan_jsonl(path: Path) -> tuple[JsonlScanSummary, Counter[str], dict[str, list[int]], dict[str, str]]:
    total_lines = 0
    blank_lines = 0
    invalid_json_lines = 0

    fingerprint_counter: Counter[str] = Counter()
    fingerprint_to_lines: dict[str, list[int]] = {}
    fingerprint_to_canonical_json: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            stripped = raw_line.strip()

            if not stripped:
                blank_lines += 1
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            fingerprint, canonical_json = fingerprint_record(record)
            fingerprint_counter[fingerprint] += 1
            fingerprint_to_lines.setdefault(fingerprint, []).append(line_number)
            fingerprint_to_canonical_json.setdefault(fingerprint, canonical_json)

    valid_records = sum(fingerprint_counter.values())
    unique_fingerprints = len(fingerprint_counter)
    internal_duplicate_records = sum(count - 1 for count in fingerprint_counter.values() if count > 1)

    summary = JsonlScanSummary(
        path=str(path),
        total_lines=total_lines,
        valid_records=valid_records,
        invalid_json_lines=invalid_json_lines,
        blank_lines=blank_lines,
        unique_fingerprints=unique_fingerprints,
        internal_duplicate_records=internal_duplicate_records,
    )
    return summary, fingerprint_counter, fingerprint_to_lines, fingerprint_to_canonical_json


def build_overlap_report(
    current_path: Path,
    cumulative_path: Path,
    missing_examples_limit: int = 20,
) -> dict[str, Any]:
    current_summary, current_counter, current_lines, current_json = scan_jsonl(current_path)
    cumulative_summary, cumulative_counter, cumulative_lines, cumulative_json = scan_jsonl(cumulative_path)

    current_set = set(current_counter.keys())
    cumulative_set = set(cumulative_counter.keys())

    overlapping_fingerprints = current_set & cumulative_set
    missing_from_cumulative = sorted(current_set - cumulative_set)
    missing_from_current = sorted(cumulative_set - current_set)

    overlap_count = len(overlapping_fingerprints)
    current_unique_count = len(current_set)
    cumulative_unique_count = len(cumulative_set)

    overlap_ratio_vs_current = (
        overlap_count / current_unique_count if current_unique_count > 0 else 0.0
    )
    overlap_ratio_vs_cumulative = (
        overlap_count / cumulative_unique_count if cumulative_unique_count > 0 else 0.0
    )

    missing_examples: list[dict[str, Any]] = []
    for fingerprint in missing_from_cumulative[:missing_examples_limit]:
        missing_examples.append(
            {
                "fingerprint": fingerprint,
                "current_line_numbers": current_lines.get(fingerprint, []),
                "record_preview": current_json.get(fingerprint, "")[:1000],
            }
        )

    current_duplicate_examples = [
        {
            "fingerprint": fingerprint,
            "count": count,
            "line_numbers": current_lines.get(fingerprint, []),
            "record_preview": current_json.get(fingerprint, "")[:1000],
        }
        for fingerprint, count in current_counter.items()
        if count > 1
    ][:missing_examples_limit]

    cumulative_duplicate_examples = [
        {
            "fingerprint": fingerprint,
            "count": count,
            "line_numbers": cumulative_lines.get(fingerprint, []),
            "record_preview": cumulative_json.get(fingerprint, "")[:1000],
        }
        for fingerprint, count in cumulative_counter.items()
        if count > 1
    ][:missing_examples_limit]

    if len(missing_from_cumulative) == 0:
        historical_input_policy = {
            "decision": "use_cumulative_only",
            "include_current_with_historical": False,
            "reason": "All unique valid records from current are already present in cumulative.",
        }
    else:
        historical_input_policy = {
            "decision": "current_contains_non_overlapping_records",
            "include_current_with_historical": True,
            "reason": (
                "Current contains unique valid records that do not exist in cumulative. "
                "Historical rerun should either include current in addition to cumulative "
                "or cumulative generation consistency should be fixed first."
            ),
            "missing_unique_record_count": len(missing_from_cumulative),
        }

    report: dict[str, Any] = {
        "metadata": {
            "report_type": "jsonl_overlap_validation",
        },
        "inputs": {
            "current_path": str(current_path),
            "cumulative_path": str(cumulative_path),
        },
        "current_summary": current_summary.__dict__,
        "cumulative_summary": cumulative_summary.__dict__,
        "comparison": {
            "current_unique_records": current_unique_count,
            "cumulative_unique_records": cumulative_unique_count,
            "overlap_unique_record_count": overlap_count,
            "overlap_ratio_vs_current": round(overlap_ratio_vs_current, 6),
            "overlap_ratio_vs_cumulative": round(overlap_ratio_vs_cumulative, 6),
            "missing_from_cumulative_unique_record_count": len(missing_from_cumulative),
            "missing_from_current_unique_record_count": len(missing_from_current),
        },
        "duplicates": {
            "current_internal_duplicate_count": current_summary.internal_duplicate_records,
            "cumulative_internal_duplicate_count": cumulative_summary.internal_duplicate_records,
            "current_duplicate_examples": current_duplicate_examples,
            "cumulative_duplicate_examples": cumulative_duplicate_examples,
        },
        "missing_from_cumulative_examples": missing_examples,
        "historical_input_policy": historical_input_policy,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate overlap between current and cumulative trade analysis JSONL files."
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("logs/trade_analysis.jsonl"),
        help="Path to the current JSONL file.",
    )
    parser.add_argument(
        "--cumulative",
        type=Path,
        default=Path("logs/trade_analysis_cumulative.jsonl"),
        help="Path to the cumulative JSONL file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/research_reports/latest/jsonl_overlap_validation_report.json"),
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/research_reports/latest/jsonl_overlap_validation_report.md"),
        help="Path to write the Markdown summary.",
    )
    parser.add_argument(
        "--missing-examples-limit",
        type=int,
        default=20,
        help="Maximum number of missing/duplicate examples to include in report.",
    )
    return parser.parse_args()


def build_markdown_summary(report: dict[str, Any]) -> str:
    current_summary = report["current_summary"]
    cumulative_summary = report["cumulative_summary"]
    comparison = report["comparison"]
    policy = report["historical_input_policy"]

    lines = [
        "# JSONL Overlap Validation Report",
        "",
        "## Inputs",
        f"- current: `{report['inputs']['current_path']}`",
        f"- cumulative: `{report['inputs']['cumulative_path']}`",
        "",
        "## Current Summary",
        f"- total_lines: {current_summary['total_lines']}",
        f"- valid_records: {current_summary['valid_records']}",
        f"- invalid_json_lines: {current_summary['invalid_json_lines']}",
        f"- blank_lines: {current_summary['blank_lines']}",
        f"- unique_fingerprints: {current_summary['unique_fingerprints']}",
        f"- internal_duplicate_records: {current_summary['internal_duplicate_records']}",
        "",
        "## Cumulative Summary",
        f"- total_lines: {cumulative_summary['total_lines']}",
        f"- valid_records: {cumulative_summary['valid_records']}",
        f"- invalid_json_lines: {cumulative_summary['invalid_json_lines']}",
        f"- blank_lines: {cumulative_summary['blank_lines']}",
        f"- unique_fingerprints: {cumulative_summary['unique_fingerprints']}",
        f"- internal_duplicate_records: {cumulative_summary['internal_duplicate_records']}",
        "",
        "## Comparison",
        f"- current_unique_records: {comparison['current_unique_records']}",
        f"- cumulative_unique_records: {comparison['cumulative_unique_records']}",
        f"- overlap_unique_record_count: {comparison['overlap_unique_record_count']}",
        f"- overlap_ratio_vs_current: {comparison['overlap_ratio_vs_current']}",
        f"- overlap_ratio_vs_cumulative: {comparison['overlap_ratio_vs_cumulative']}",
        f"- missing_from_cumulative_unique_record_count: {comparison['missing_from_cumulative_unique_record_count']}",
        f"- missing_from_current_unique_record_count: {comparison['missing_from_current_unique_record_count']}",
        "",
        "## Historical Input Policy",
        f"- decision: {policy['decision']}",
        f"- include_current_with_historical: {policy['include_current_with_historical']}",
        f"- reason: {policy['reason']}",
        "",
        "## Missing From Cumulative Examples",
    ]

    examples = report.get("missing_from_cumulative_examples", [])
    if not examples:
        lines.append("- none")
    else:
        for item in examples:
            lines.append(
                f"- fingerprint={item['fingerprint']} lines={item['current_line_numbers']}"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    report = build_overlap_report(
        current_path=args.current,
        cumulative_path=args.cumulative,
        missing_examples_limit=args.missing_examples_limit,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(
        build_markdown_summary(report),
        encoding="utf-8",
    )

    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(json.dumps(report["historical_input_policy"], ensure_ascii=False, indent=2))
    print(f"json_report={args.output_json}")
    print(f"md_report={args.output_md}")


if __name__ == "__main__":
    main()