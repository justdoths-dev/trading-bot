from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_input_mapper import (
    _build_candidate,
    _build_candidate_seeds,
    _build_drift_lookup,
    _build_result,
    _build_score_lookup,
    _candidate_sort_key,
    _dedupe_candidates,
    _extract_cumulative_record_count,
    _extract_latest_record_count,
    _is_valid_candidate,
)

# NOTE:
# This runner currently uses private helper functions from
# src.research.edge_selection_input_mapper for snapshot reconstruction fallback.
# That creates temporary internal coupling. The intended medium-term direction is:
# - expose a public mapper reconstruction API
# - remove direct imports of underscore-prefixed helpers from this runner

DEFAULT_MANIFEST_PATH = Path(
    "logs/research_reports/latest/resolved_historical_input_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("logs/research_reports/historical_rerun")

SUMMARY_JSON_NAME = "historical_edge_rerun_summary.json"
RESULTS_JSONL_NAME = "historical_edge_rerun_results.jsonl"
SUMMARY_MD_NAME = "historical_edge_rerun_summary.md"

EXPECTED_MANIFEST_MODE = "cumulative_plus_current_non_overlap"


@dataclass(frozen=True)
class ResolvedHistoricalManifest:
    manifest_path: Path
    resolved_input_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class ReplayRecord:
    source_line_number: int
    record: dict[str, Any]


@dataclass(frozen=True)
class ReplayAdaptationResult:
    replay_input_mode: str
    mapped_payload: dict[str, Any]


@dataclass
class ReplaySummaryAccumulator:
    total_records_processed: int = 0
    successful_replay_count: int = 0
    selected_count: int = 0
    abstain_count: int = 0
    tie_count: int = 0
    errors_count: int = 0
    selected_candidates: Counter[str] | None = None
    replay_input_mode_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.selected_candidates is None:
            self.selected_candidates = Counter()
        if self.replay_input_mode_counts is None:
            self.replay_input_mode_counts = Counter()


def load_resolved_manifest(manifest_path: Path) -> ResolvedHistoricalManifest:
    resolved_manifest_path = Path(manifest_path)

    try:
        payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Failed to read resolved manifest {resolved_manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Resolved manifest is not valid JSON {resolved_manifest_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Resolved manifest must contain a JSON object: {resolved_manifest_path}")

    mode = payload.get("mode")
    if mode != EXPECTED_MANIFEST_MODE:
        raise ValueError(
            f"Resolved manifest mode must be {EXPECTED_MANIFEST_MODE!r}, got {mode!r}."
        )

    cross_file_overlap_resolution_applied = payload.get(
        "cross_file_overlap_resolution_applied"
    )
    if cross_file_overlap_resolution_applied is not True:
        raise ValueError(
            "Resolved manifest must indicate cross_file_overlap_resolution_applied=true."
        )

    resolved_input_raw = payload.get("output_path")
    if not isinstance(resolved_input_raw, str) or not resolved_input_raw.strip():
        raise ValueError("Resolved manifest must contain a non-empty string output_path.")

    resolved_input_path = _resolve_path_like(
        raw_path=resolved_input_raw,
        base_dir=resolved_manifest_path.parent,
    )
    if not resolved_input_path.exists() or not resolved_input_path.is_file():
        raise ValueError(
            "Resolved input path from manifest does not exist or is not a file: "
            f"{resolved_input_path} "
            f"(raw output_path={resolved_input_raw}, manifest_path={resolved_manifest_path})"
        )

    return ResolvedHistoricalManifest(
        manifest_path=resolved_manifest_path,
        resolved_input_path=resolved_input_path,
        payload=payload,
    )


def iter_resolved_input_records(resolved_input_path: Path) -> Iterator[ReplayRecord]:
    with resolved_input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            content = raw_line.strip()
            if not content:
                continue

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Resolved input JSONL line {line_number} is not valid JSON: {exc}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Resolved input JSONL line {line_number} must contain a JSON object."
                )

            yield ReplayRecord(source_line_number=line_number, record=parsed)


def adapt_record_to_replay_input(
    record: dict[str, Any],
    *,
    allow_snapshot_reconstruction: bool,
) -> ReplayAdaptationResult:
    embedded_payload = _extract_embedded_mapper_payload(record)
    if embedded_payload is not None:
        return ReplayAdaptationResult(
            replay_input_mode="embedded_mapper_payload",
            mapped_payload=embedded_payload,
        )

    if allow_snapshot_reconstruction:
        snapshot_payload = _build_payload_from_embedded_report_snapshots(record)
        if snapshot_payload is not None:
            return ReplayAdaptationResult(
                replay_input_mode="snapshot_reconstruction",
                mapped_payload=snapshot_payload,
            )

    raise ValueError(
        "Source record does not contain an embedded edge-selection mapper payload. "
        "Snapshot reconstruction is unavailable or disabled for this run."
    )


def execute_replay(mapped_payload: dict[str, Any]) -> dict[str, Any]:
    return run_edge_selection_engine(mapped_payload)


def run_historical_edge_rerun(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    allow_snapshot_reconstruction: bool = False,
) -> dict[str, Any]:
    manifest = load_resolved_manifest(manifest_path)
    resolved_run_id = run_id or _build_run_id()
    run_dir = Path(output_root) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / RESULTS_JSONL_NAME
    summary_path = run_dir / SUMMARY_JSON_NAME
    markdown_path = run_dir / SUMMARY_MD_NAME

    accumulator = ReplaySummaryAccumulator()

    with results_path.open("w", encoding="utf-8") as results_handle:
        for replay_record in iter_resolved_input_records(manifest.resolved_input_path):
            detail_row = _process_replay_record(
                replay_record,
                accumulator,
                allow_snapshot_reconstruction=allow_snapshot_reconstruction,
            )
            _write_jsonl_row(results_handle, detail_row)

    summary = build_historical_edge_rerun_summary(
        run_id=resolved_run_id,
        generated_at=datetime.now(UTC).isoformat(),
        manifest=manifest,
        accumulator=accumulator,
        allow_snapshot_reconstruction=allow_snapshot_reconstruction,
    )

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_summary_markdown(summary) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "markdown_path": str(markdown_path),
    }


def build_historical_edge_rerun_summary(
    *,
    run_id: str,
    generated_at: str,
    manifest: ResolvedHistoricalManifest,
    accumulator: ReplaySummaryAccumulator,
    allow_snapshot_reconstruction: bool,
) -> dict[str, Any]:
    successful = accumulator.successful_replay_count
    selected_ratio = round(accumulator.selected_count / successful, 6) if successful else 0.0
    abstain_ratio = round(accumulator.abstain_count / successful, 6) if successful else 0.0

    dominant_candidates = [
        {"candidate": candidate, "count": count}
        for candidate, count in accumulator.selected_candidates.most_common(10)
    ]

    replay_input_modes = [
        {"mode": mode, "count": count}
        for mode, count in accumulator.replay_input_mode_counts.most_common()
    ]

    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "manifest_path": str(manifest.manifest_path),
        "resolved_input_path": str(manifest.resolved_input_path),
        "total_records_processed": accumulator.total_records_processed,
        "successful_replay_count": successful,
        "selected_count": accumulator.selected_count,
        "abstain_count": accumulator.abstain_count,
        "selected_ratio": selected_ratio,
        "abstain_ratio": abstain_ratio,
        "tie_count": accumulator.tie_count,
        "dominant_candidates": dominant_candidates,
        "errors_count": accumulator.errors_count,
        "allow_snapshot_reconstruction": allow_snapshot_reconstruction,
        "replay_input_modes": replay_input_modes,
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    dominant_candidates = summary.get("dominant_candidates")
    dominant_rows = dominant_candidates if isinstance(dominant_candidates, list) else []

    replay_input_modes = summary.get("replay_input_modes")
    mode_rows = replay_input_modes if isinstance(replay_input_modes, list) else []

    lines = [
        "# Historical Edge Rerun Summary",
        "",
        "## Run",
        f"- run_id: {summary.get('run_id', 'n/a')}",
        f"- generated_at: {summary.get('generated_at', 'n/a')}",
        f"- manifest_path: `{summary.get('manifest_path', 'n/a')}`",
        f"- resolved_input_path: `{summary.get('resolved_input_path', 'n/a')}`",
        f"- allow_snapshot_reconstruction: {summary.get('allow_snapshot_reconstruction', False)}",
        "",
        "## Totals",
        f"- total_records_processed: {summary.get('total_records_processed', 0)}",
        f"- successful_replay_count: {summary.get('successful_replay_count', 0)}",
        f"- selected_count: {summary.get('selected_count', 0)}",
        f"- abstain_count: {summary.get('abstain_count', 0)}",
        f"- selected_ratio: {summary.get('selected_ratio', 0.0)}",
        f"- abstain_ratio: {summary.get('abstain_ratio', 0.0)}",
        f"- tie_count: {summary.get('tie_count', 0)}",
        f"- errors_count: {summary.get('errors_count', 0)}",
        "",
        "## Replay Input Modes",
    ]

    if not mode_rows:
        lines.append("- none")
    else:
        for row in mode_rows:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('mode', 'n/a')}: {row.get('count', 0)}")

    lines.extend(
        [
            "",
            "## Dominant Candidates",
        ]
    )

    if not dominant_rows:
        lines.append("- none")
    else:
        for row in dominant_rows:
            if not isinstance(row, dict):
                continue
            lines.append(f"- {row.get('candidate', 'n/a')}: {row.get('count', 0)}")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic historical edge-selection replay from the resolved "
            "historical input manifest without touching production state."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the resolved historical input manifest JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where historical rerun outputs should be written.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed run identifier. Defaults to a UTC timestamp-based id.",
    )
    parser.add_argument(
        "--allow-snapshot-reconstruction",
        action="store_true",
        help=(
            "Allow fallback reconstruction of mapper payloads from embedded research "
            "report snapshots when embedded mapper payloads are not present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_historical_edge_rerun(
        manifest_path=args.manifest,
        output_root=args.output_root,
        run_id=args.run_id,
        allow_snapshot_reconstruction=args.allow_snapshot_reconstruction,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"summary_json={result['summary_path']}")
    print(f"results_jsonl={result['results_path']}")
    print(f"summary_md={result['markdown_path']}")


def _process_replay_record(
    replay_record: ReplayRecord,
    accumulator: ReplaySummaryAccumulator,
    *,
    allow_snapshot_reconstruction: bool,
) -> dict[str, Any]:
    accumulator.total_records_processed += 1
    source_generated_at = _source_generated_at(replay_record.record)

    try:
        adaptation_result = adapt_record_to_replay_input(
            replay_record.record,
            allow_snapshot_reconstruction=allow_snapshot_reconstruction,
        )
        replay_output = execute_replay(adaptation_result.mapped_payload)
    except Exception as exc:
        accumulator.errors_count += 1
        accumulator.replay_input_mode_counts["replay_error"] += 1
        return {
            "source_line_number": replay_record.source_line_number,
            "generated_at": source_generated_at,
            "replay_input_mode": "replay_error",
            "selected_candidate": None,
            "eligible_candidate_count": 0,
            "penalized_candidate_count": 0,
            "abstain_diagnosis": None,
            "error": str(exc),
        }

    accumulator.successful_replay_count += 1
    accumulator.replay_input_mode_counts[adaptation_result.replay_input_mode] += 1

    selection_status = replay_output.get("selection_status")
    eligible_candidate_count = _count_candidates_by_status(replay_output, "eligible")
    penalized_candidate_count = _count_candidates_by_status(replay_output, "penalized")
    abstain_diagnosis = replay_output.get("abstain_diagnosis")
    tie_metadata = _extract_tie_metadata(replay_output)

    selected_candidate = _selected_candidate_identity(replay_output)
    if selection_status == "selected":
        accumulator.selected_count += 1
        if selected_candidate is not None:
            accumulator.selected_candidates[selected_candidate] += 1
    else:
        accumulator.abstain_count += 1

    if tie_metadata is not None:
        accumulator.tie_count += 1

    row = {
        "source_line_number": replay_record.source_line_number,
        "generated_at": source_generated_at,
        "replay_input_mode": adaptation_result.replay_input_mode,
        "selected_candidate": selected_candidate,
        "eligible_candidate_count": eligible_candidate_count,
        "penalized_candidate_count": penalized_candidate_count,
        "abstain_diagnosis": abstain_diagnosis if isinstance(abstain_diagnosis, dict) else None,
    }
    if tie_metadata is not None:
        row["tie_metadata"] = tie_metadata
    return row


def _extract_embedded_mapper_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    candidate_keys = (
        "edge_selection_input",
        "edge_selection_mapper_payload",
        "mapped_edge_selection_input",
        "historical_edge_selection_input",
    )
    for key in candidate_keys:
        payload = record.get(key)
        if _looks_like_mapper_payload(payload):
            return _normalize_mapper_payload(payload)

    research = record.get("research")
    if isinstance(research, dict):
        for key in candidate_keys:
            payload = research.get(key)
            if _looks_like_mapper_payload(payload):
                return _normalize_mapper_payload(payload)

    return None


def _build_payload_from_embedded_report_snapshots(record: dict[str, Any]) -> dict[str, Any] | None:
    snapshots = _extract_report_snapshots(record)
    if snapshots is None:
        return None

    latest_summary = snapshots["latest_summary"]
    comparison_summary = snapshots["comparison_summary"]
    edge_scores_summary = snapshots["edge_scores_summary"]
    score_drift_summary = snapshots["score_drift_summary"]
    history_line_count = snapshots.get("history_line_count")

    latest_window_record_count = _extract_latest_record_count(
        latest_summary,
        comparison_summary,
    )
    cumulative_record_count = _extract_cumulative_record_count(comparison_summary)
    score_lookup = _build_score_lookup(edge_scores_summary)
    drift_lookup = _build_drift_lookup(score_drift_summary)
    seeds, candidate_seed_diagnostics = _build_candidate_seeds(
        latest_summary,
        comparison_summary=comparison_summary,
    )

    candidates = [
        _build_candidate(seed, score_lookup=score_lookup, drift_lookup=drift_lookup)
        for seed in seeds
    ]
    candidates = [candidate for candidate in candidates if _is_valid_candidate(candidate)]
    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=_candidate_sort_key)

    return _build_result(
        ok=True,
        generated_at=_source_generated_at(record) or datetime.now(UTC).isoformat(),
        latest_window_record_count=latest_window_record_count,
        cumulative_record_count=cumulative_record_count,
        candidates=candidates,
        errors=[],
        warnings=[],
        history_line_count=history_line_count,
        candidate_seed_count=len(seeds),
        candidate_seed_diagnostics=candidate_seed_diagnostics,
    )


def _extract_report_snapshots(record: dict[str, Any]) -> dict[str, Any] | None:
    direct = {
        "latest_summary": _coerce_dict(record.get("latest_summary")),
        "comparison_summary": _coerce_dict(record.get("comparison_summary")),
        "edge_scores_summary": _coerce_dict(record.get("edge_scores_summary")),
        "score_drift_summary": _coerce_dict(record.get("score_drift_summary")),
        "history_line_count": _coerce_non_negative_int(record.get("history_line_count")),
    }
    if _has_required_snapshots(direct):
        return direct

    research_reports = record.get("research_reports")
    if isinstance(research_reports, dict):
        latest_nested = _coerce_dict(research_reports.get("latest"))
        comparison_nested = _coerce_dict(research_reports.get("comparison"))
        edge_scores_nested = _coerce_dict(research_reports.get("edge_scores"))
        score_drift_nested = _coerce_dict(research_reports.get("score_drift"))

        nested = {
            "latest_summary": _coerce_dict(
                research_reports.get("latest_summary") or latest_nested.get("summary")
            ),
            "comparison_summary": _coerce_dict(
                research_reports.get("comparison_summary") or comparison_nested.get("summary")
            ),
            "edge_scores_summary": _coerce_dict(
                research_reports.get("edge_scores_summary") or edge_scores_nested.get("summary")
            ),
            "score_drift_summary": _coerce_dict(
                research_reports.get("score_drift_summary") or score_drift_nested.get("summary")
            ),
            "history_line_count": _coerce_non_negative_int(
                research_reports.get("history_line_count")
                or research_reports.get("edge_scores_history_line_count")
            ),
        }
        if _has_required_snapshots(nested):
            return nested

    return None


def _has_required_snapshots(payload: dict[str, Any]) -> bool:
    return all(
        isinstance(payload.get(key), dict) and payload.get(key)
        for key in (
            "latest_summary",
            "comparison_summary",
            "edge_scores_summary",
            "score_drift_summary",
        )
    )


def _looks_like_mapper_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return "candidates" in payload and isinstance(payload.get("candidates"), list)


def _normalize_mapper_payload(payload: dict[str, Any]) -> dict[str, Any]:
    errors_value = payload.get("errors")
    warnings_value = payload.get("warnings")

    normalized_errors = errors_value if isinstance(errors_value, list) else []
    normalized_warnings = warnings_value if isinstance(warnings_value, list) else []

    return {
        "ok": bool(payload.get("ok", True)),
        "generated_at": payload.get("generated_at") or datetime.now(UTC).isoformat(),
        "latest_window_record_count": _coerce_non_negative_int(
            payload.get("latest_window_record_count")
        ),
        "cumulative_record_count": _coerce_non_negative_int(
            payload.get("cumulative_record_count")
        ),
        "candidates": [
            candidate
            for candidate in payload.get("candidates", [])
            if isinstance(candidate, dict)
        ],
        "errors": normalized_errors,
        "warnings": normalized_warnings,
        "history_line_count": _coerce_non_negative_int(payload.get("history_line_count")),
        "candidate_seed_count": _coerce_non_negative_int(payload.get("candidate_seed_count"))
        or 0,
        "candidate_seed_diagnostics": _coerce_dict(payload.get("candidate_seed_diagnostics")),
    }


def _extract_tie_metadata(replay_output: dict[str, Any]) -> dict[str, Any] | None:
    abstain_diagnosis = replay_output.get("abstain_diagnosis")
    if not isinstance(abstain_diagnosis, dict):
        return None
    if abstain_diagnosis.get("category") != "tied_top_candidates":
        return None

    return {
        "category": abstain_diagnosis.get("category"),
        "top_candidate": abstain_diagnosis.get("top_candidate"),
        "compared_candidate": abstain_diagnosis.get("compared_candidate"),
    }


def _selected_candidate_identity(replay_output: dict[str, Any]) -> str | None:
    if replay_output.get("selection_status") != "selected":
        return None

    symbol = replay_output.get("selected_symbol")
    strategy = replay_output.get("selected_strategy")
    horizon = replay_output.get("selected_horizon")
    if not any(value is not None for value in (symbol, strategy, horizon)):
        return None
    return f"{symbol}|{strategy}|{horizon}"


def _count_candidates_by_status(replay_output: dict[str, Any], status: str) -> int:
    ranking = replay_output.get("ranking")
    if not isinstance(ranking, list):
        return 0
    return sum(
        1 for item in ranking if isinstance(item, dict) and item.get("candidate_status") == status
    )


def _source_generated_at(record: dict[str, Any]) -> str | None:
    for key in ("generated_at", "logged_at"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    ai_payload = record.get("ai")
    if isinstance(ai_payload, dict):
        value = ai_payload.get("generated_at")
        if isinstance(value, str) and value.strip():
            return value

    return None


def _write_jsonl_row(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _resolve_path_like(*, raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)

    if candidate.is_absolute():
        return candidate

    cwd_resolved = candidate.resolve()
    if cwd_resolved.exists():
        return cwd_resolved

    return (base_dir / candidate).resolve()


def _build_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


if __name__ == "__main__":
    main()
