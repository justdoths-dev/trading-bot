from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.candidate_quality_gate import (
    apply_candidate_quality_gate,
    coerce_float,
    coerce_int,
    compute_candidate_metrics,
    is_selected_record,
)

REPORT_TYPE = "candidate_quality_gate_relaxation_report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

NEAR_MISS_SAMPLE_COUNT_MIN = 20
NEAR_MISS_SAMPLE_COUNT_MAX = 29
NEAR_MISS_POSITIVE_RATE_MIN = 40.0
NEAR_MISS_POSITIVE_RATE_MAX_EXCLUSIVE = 50.0
NEAR_MISS_MEDIAN_RETURN_MIN = 0.0
NEAR_MISS_MEDIAN_RETURN_MAX = 0.24


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only report for candidate_quality_gate strict drops, "
            "fallback behavior, and near-miss candidates."
        )
    )
    parser.add_argument(
        "--trade-analysis",
        type=Path,
        required=True,
        help="Path to the trade-analysis JSONL file to inspect.",
    )
    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=None,
        help=(
            "Optional comparison summary.json path. When present, drift_notes are "
            "counted into comparison_drift_note_counts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the JSON and Markdown report files should be written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_candidate_quality_gate_relaxation_report(
        trade_analysis_path=args.trade_analysis,
        comparison_summary_path=args.comparison_summary,
    )
    written_paths = write_candidate_quality_gate_relaxation_report(
        report,
        args.output_dir,
    )
    summary = {
        "report_type": REPORT_TYPE,
        "trade_analysis_path": report["inputs"]["trade_analysis_path"],
        "comparison_summary_path": report["inputs"]["comparison_summary_path"],
        "candidate_sets_analyzed": report["summary"]["candidate_sets_analyzed"],
        "input_candidate_count": report["summary"]["input_candidate_count"],
        "strict_kept_count": report["summary"]["strict_kept_count"],
        "strict_dropped_count": report["summary"]["strict_dropped_count"],
        "fallback_applied": report["summary"]["fallback_applied"],
        "fallback_restored_candidate_count": report["summary"][
            "fallback_restored_candidate_count"
        ],
        "written_paths": written_paths,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_candidate_quality_gate_relaxation_report(
    *,
    trade_analysis_path: Path,
    comparison_summary_path: Path | None = None,
) -> dict[str, Any]:
    resolved_trade_analysis_path = _resolve_existing_file(
        trade_analysis_path,
        label="Trade-analysis input",
    )

    loaded_rows = _load_trade_analysis_rows(resolved_trade_analysis_path)
    trade_analysis_rows = loaded_rows["rows"]
    row_diagnostics = loaded_rows["diagnostics"]

    resolved_comparison_summary_path: Path | None = None
    comparison_summary: dict[str, Any] | None = None
    comparison_notes: list[str] = []
    if comparison_summary_path is None:
        comparison_notes.append(
            "No comparison summary was provided; comparison_drift_note_counts is empty. Downstream outcome fields were not available."
        )
    else:
        resolved_comparison_summary_path = _resolve_existing_file(
            comparison_summary_path,
            label="Comparison summary",
        )
        comparison_summary = _load_json_object(
            resolved_comparison_summary_path,
            label="Comparison summary",
        )

    input_candidate_count = 0
    strict_kept_count = 0
    strict_dropped_count = 0
    fallback_applied_row_count = 0
    fallback_restored_candidate_count = 0

    drop_reason_counts: Counter[str] = Counter()
    near_miss_counts: Counter[str] = Counter()
    candidate_row_source_counts: Counter[str] = Counter()
    skipped_candidate_row_reason_counts: Counter[str] = Counter()

    candidate_sets_analyzed = 0

    for row in trade_analysis_rows:
        candidates, source_or_reason = _recover_candidates_from_row(row)
        if candidates is None:
            skipped_candidate_row_reason_counts[source_or_reason] += 1
            continue

        candidate_sets_analyzed += 1
        candidate_row_source_counts[source_or_reason] += 1

        gate_result = _evaluate_candidate_quality_gate(
            candidates,
            resolved_trade_analysis_path,
        )

        input_candidate_count += gate_result["total_candidates"]
        strict_kept_count += gate_result["strict_kept_count"]
        strict_dropped_count += gate_result["strict_dropped_count"]
        fallback_restored_candidate_count += gate_result["fallback_restored_count"]
        if gate_result["fallback_applied"]:
            fallback_applied_row_count += 1

        for dropped in gate_result["strict_dropped_candidates"]:
            reason = dropped.get("reason")
            if isinstance(reason, str) and reason.strip():
                drop_reason_counts[reason.strip()] += 1

        for candidate in candidates:
            metrics = compute_candidate_metrics(candidate, trade_analysis_rows)
            _accumulate_near_miss_counts(near_miss_counts, metrics)

    comparison_drift_note_counts = _build_comparison_drift_note_counts(
        comparison_summary,
        comparison_notes,
    )

    notes = _build_notes(
        row_diagnostics=row_diagnostics,
        candidate_sets_analyzed=candidate_sets_analyzed,
        candidate_row_source_counts=candidate_row_source_counts,
        skipped_candidate_row_reason_counts=skipped_candidate_row_reason_counts,
        comparison_notes=comparison_notes,
    )

    return {
        "generated_at": _utc_now_iso(),
        "report_type": REPORT_TYPE,
        "inputs": {
            "trade_analysis_path": str(resolved_trade_analysis_path),
            "comparison_summary_path": (
                str(resolved_comparison_summary_path)
                if resolved_comparison_summary_path is not None
                else None
            ),
        },
        "data_quality": {
            **row_diagnostics,
            "candidate_sets_analyzed": candidate_sets_analyzed,
            "candidate_row_source_counts": _sorted_counter_dict(
                candidate_row_source_counts
            ),
            "skipped_candidate_row_reason_counts": _sorted_counter_dict(
                skipped_candidate_row_reason_counts
            ),
        },
        "summary": {
            "input_candidate_count": input_candidate_count,
            "strict_kept_count": strict_kept_count,
            "strict_dropped_count": strict_dropped_count,
            "fallback_applied": fallback_applied_row_count > 0,
            "fallback_applied_row_count": fallback_applied_row_count,
            "fallback_restored_candidate_count": fallback_restored_candidate_count,
            "drop_reason_counts": _sorted_counter_dict(drop_reason_counts),
            "near_miss_counts": _sorted_counter_dict(near_miss_counts),
            "comparison_drift_note_counts": comparison_drift_note_counts,
            "candidate_sets_analyzed": candidate_sets_analyzed,
            "notes": notes,
        },
    }


def write_candidate_quality_gate_relaxation_report(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    resolved_output_dir = output_dir.expanduser()
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = Path.cwd() / resolved_output_dir
    resolved_output_dir = resolved_output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / REPORT_JSON_NAME
    md_path = resolved_output_dir / REPORT_MD_NAME

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")

    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def render_markdown(report: dict[str, Any]) -> str:
    inputs = report.get("inputs", {})
    data_quality = report.get("data_quality", {})
    summary = report.get("summary", {})

    lines = [
        "# Candidate Quality Gate Relaxation Report",
        "",
        f"Generated at: {report.get('generated_at')}",
        f"Trade analysis input: {inputs.get('trade_analysis_path')}",
        f"Comparison summary input: {inputs.get('comparison_summary_path') or 'not provided'}",
        "",
        "## Coverage",
        "",
        f"- parsed_row_count: {data_quality.get('parsed_row_count', 0)}",
        f"- selected_record_count: {data_quality.get('selected_record_count', 0)}",
        f"- candidate_sets_analyzed: {data_quality.get('candidate_sets_analyzed', 0)}",
        f"- malformed_line_count: {data_quality.get('malformed_line_count', 0)}",
        f"- non_object_line_count: {data_quality.get('non_object_line_count', 0)}",
        "",
        "## Gate Summary",
        "",
        f"- input_candidate_count: {summary.get('input_candidate_count', 0)}",
        f"- strict_kept_count: {summary.get('strict_kept_count', 0)}",
        f"- strict_dropped_count: {summary.get('strict_dropped_count', 0)}",
        f"- fallback_applied: {summary.get('fallback_applied', False)}",
        f"- fallback_applied_row_count: {summary.get('fallback_applied_row_count', 0)}",
        (
            "- fallback_restored_candidate_count: "
            f"{summary.get('fallback_restored_candidate_count', 0)}"
        ),
        "",
        "## Drop Reasons",
        "",
    ]

    lines.extend(_render_count_mapping(summary.get("drop_reason_counts")))
    lines.extend(["", "## Near-Miss Buckets", ""])
    lines.extend(_render_count_mapping(summary.get("near_miss_counts")))
    lines.extend(["", "## Comparison Drift Notes", ""])
    lines.extend(_render_count_mapping(summary.get("comparison_drift_note_counts")))
    lines.extend(["", "## Notes", ""])

    notes = summary.get("notes")
    if isinstance(notes, list) and notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- none")

    lines.append("")
    return "\n".join(lines)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_existing_file(path: Path, *, label: str) -> Path:
    resolved_path = path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    resolved_path = resolved_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"{label} path must be a file: {resolved_path}")
    return resolved_path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")

    return payload


def _load_trade_analysis_rows(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "total_lines": 0,
        "blank_line_count": 0,
        "malformed_line_count": 0,
        "non_object_line_count": 0,
        "parsed_row_count": 0,
        "selected_record_count": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            diagnostics["total_lines"] += 1
            line = raw_line.strip()
            if not line:
                diagnostics["blank_line_count"] += 1
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["malformed_line_count"] += 1
                continue

            if not isinstance(payload, dict):
                diagnostics["non_object_line_count"] += 1
                continue

            rows.append(payload)

    diagnostics["parsed_row_count"] = len(rows)
    diagnostics["selected_record_count"] = sum(
        1 for row in rows if is_selected_record(row)
    )

    return {
        "rows": rows,
        "diagnostics": diagnostics,
    }


def _recover_candidates_from_row(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str]:
    mapper_payload = row.get("edge_selection_mapper_payload")
    if not isinstance(mapper_payload, dict):
        return None, "missing_mapper_payload"

    gate_block = mapper_payload.get("candidate_quality_gate")
    recovered_from_gate = _recover_candidates_from_gate_block(gate_block)
    if recovered_from_gate is not None:
        return recovered_from_gate, "candidate_quality_gate"

    raw_candidates = mapper_payload.get("candidates")
    if isinstance(raw_candidates, list):
        return _normalize_candidate_list(raw_candidates), "mapper_candidates"

    if isinstance(gate_block, dict):
        return None, "candidate_quality_gate_without_recoverable_candidates"

    return None, "missing_candidate_inputs"


def _recover_candidates_from_gate_block(
    gate_block: Any,
) -> list[dict[str, Any]] | None:
    if not isinstance(gate_block, dict):
        return None

    strict_kept = _normalize_candidate_list(gate_block.get("strict_kept_candidates"))
    strict_dropped = _normalize_dropped_candidate_list(
        gate_block.get("strict_dropped_candidates")
    )

    if strict_kept or strict_dropped:
        return strict_kept + strict_dropped

    fallback_applied = gate_block.get("fallback_applied") is True
    fallback_restored = _normalize_candidate_list(
        gate_block.get("fallback_restored_candidates")
    )
    if fallback_applied and fallback_restored:
        return fallback_restored

    total_candidates = coerce_int(gate_block.get("total_candidates"))
    if total_candidates == 0:
        return []

    return None


def _normalize_candidate_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _normalize_dropped_candidate_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate")
        if isinstance(candidate, dict):
            candidates.append(candidate)
    return candidates


def _evaluate_candidate_quality_gate(
    candidates: list[dict[str, Any]],
    trade_analysis_path: Path,
) -> dict[str, Any]:
    result = apply_candidate_quality_gate(
        candidates,
        trade_analysis_path=trade_analysis_path,
    )

    strict_dropped_candidates = result.get("strict_dropped_candidates")
    if not isinstance(strict_dropped_candidates, list):
        strict_dropped_candidates = []

    return {
        "input_path_used": str(result.get("input_path_used", trade_analysis_path)),
        "total_candidates": coerce_int(result.get("total_candidates")) or 0,
        "strict_kept_candidates": _normalize_candidate_list(
            result.get("strict_kept_candidates")
        ),
        "strict_kept_count": coerce_int(result.get("strict_kept_count")) or 0,
        "strict_dropped_candidates": [
            item for item in strict_dropped_candidates if isinstance(item, dict)
        ],
        "strict_dropped_count": coerce_int(result.get("strict_dropped_count")) or 0,
        "fallback_applied": result.get("fallback_applied") is True,
        "fallback_restored_candidates": _normalize_candidate_list(
            result.get("fallback_restored_candidates")
        ),
        "fallback_restored_count": coerce_int(result.get("fallback_restored_count"))
        or 0,
        "final_kept_candidates": _normalize_candidate_list(
            result.get("final_kept_candidates")
        ),
        "final_kept_count": coerce_int(result.get("final_kept_count")) or 0,
    }


def _accumulate_near_miss_counts(
    near_miss_counts: Counter[str],
    metrics: dict[str, float | int | None],
) -> None:
    sample_count = coerce_int(metrics.get("sample_count"))
    positive_rate_pct = coerce_float(metrics.get("positive_rate_pct"))
    median_return_pct = coerce_float(metrics.get("median_return_pct"))

    if (
        sample_count is not None
        and NEAR_MISS_SAMPLE_COUNT_MIN <= sample_count <= NEAR_MISS_SAMPLE_COUNT_MAX
    ):
        near_miss_counts["sample_count_20_29"] += 1

    if (
        positive_rate_pct is not None
        and NEAR_MISS_POSITIVE_RATE_MIN
        <= positive_rate_pct
        < NEAR_MISS_POSITIVE_RATE_MAX_EXCLUSIVE
    ):
        near_miss_counts["positive_rate_pct_40_00_to_49_99"] += 1

    if (
        median_return_pct is not None
        and NEAR_MISS_MEDIAN_RETURN_MIN
        <= median_return_pct
        <= NEAR_MISS_MEDIAN_RETURN_MAX
    ):
        near_miss_counts["median_return_pct_0_00_to_0_24"] += 1


def _build_comparison_drift_note_counts(
    comparison_summary: dict[str, Any] | None,
    comparison_notes: list[str],
) -> dict[str, int]:
    if comparison_summary is None:
        return {}

    raw_drift_notes = comparison_summary.get("drift_notes")
    if not isinstance(raw_drift_notes, list):
        comparison_notes.append(
            "Comparison summary was provided, but drift_notes is unavailable; comparison_drift_note_counts is empty. Downstream outcome fields were not available."
        )
        return {}

    drift_note_counts: Counter[str] = Counter()
    for item in raw_drift_notes:
        if not isinstance(item, str):
            continue
        note = item.strip()
        if note:
            drift_note_counts[note] += 1

    if not drift_note_counts:
        comparison_notes.append(
            "Comparison summary drift_notes was empty after normalization; comparison_drift_note_counts is empty. Downstream outcome fields were not available."
        )
        return {}

    comparison_notes.append(
        "comparison_drift_note_counts was derived from comparison_summary.drift_notes; downstream outcome fields were not available."
    )
    return _sorted_counter_dict(drift_note_counts)


def _build_notes(
    *,
    row_diagnostics: dict[str, Any],
    candidate_sets_analyzed: int,
    candidate_row_source_counts: Counter[str],
    skipped_candidate_row_reason_counts: Counter[str],
    comparison_notes: list[str],
) -> list[str]:
    notes: list[str] = []

    malformed_line_count = int(row_diagnostics.get("malformed_line_count", 0) or 0)
    non_object_line_count = int(row_diagnostics.get("non_object_line_count", 0) or 0)

    if malformed_line_count > 0 or non_object_line_count > 0:
        notes.append(
            "Malformed or non-object JSONL lines were skipped while loading the trade-analysis input."
        )

    notes.append(f"Candidate sets analyzed: {candidate_sets_analyzed}.")
    notes.append(
        "near_miss_counts are counted per candidate occurrence and are not deduplicated by candidate identity."
    )

    if candidate_row_source_counts:
        source_summary = ", ".join(
            f"{source}={count}"
            for source, count in _sorted_counter_dict(candidate_row_source_counts).items()
        )
        notes.append(f"Candidate extraction sources: {source_summary}.")

    if skipped_candidate_row_reason_counts:
        skipped_summary = ", ".join(
            f"{reason}={count}"
            for reason, count in _sorted_counter_dict(
                skipped_candidate_row_reason_counts
            ).items()
        )
        notes.append(f"Skipped candidate rows: {skipped_summary}.")

    notes.extend(comparison_notes)
    return notes


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        key: count
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def _render_count_mapping(value: Any) -> list[str]:
    if not isinstance(value, dict) or not value:
        return ["- none"]
    return [f"- {key}: {count}" for key, count in value.items()]


if __name__ == "__main__":
    main()
