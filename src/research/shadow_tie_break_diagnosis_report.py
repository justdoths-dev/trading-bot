from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from src.research.candidate_seed_failure_diagnosis_report import load_shadow_records

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "edge_selection_shadow"
    / "edge_selection_shadow.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "research_reports"
    / "latest"
)
DEFAULT_RECENT_RUN_LIMIT = 100
TIE_REASON_CODE = "TOP_CANDIDATES_TIED"


def run_shadow_tie_break_diagnosis_report(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    recent_run_limit: int = DEFAULT_RECENT_RUN_LIMIT,
) -> dict[str, Any]:
    resolved_input = input_path or DEFAULT_INPUT_PATH
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR
    loaded = load_shadow_records(resolved_input)
    summary = build_shadow_tie_break_diagnosis_summary(
        records=loaded["records"],
        input_path=resolved_input,
        data_quality=loaded["data_quality"],
        recent_run_limit=recent_run_limit,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / "shadow_tie_break_diagnosis_report.json"
    md_path = resolved_output_dir / "shadow_tie_break_diagnosis_report.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_shadow_tie_break_diagnosis_markdown(summary),
        encoding="utf-8",
    )
    return {
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output_dir),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "analyzed_runs": summary["metadata"]["analyzed_runs"],
    }


def build_shadow_tie_break_diagnosis_summary(
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
    recent_run_limit: int,
) -> dict[str, Any]:
    runs = _select_recent_runs(records, recent_run_limit)
    tie_events: list[dict[str, Any]] = []
    selected_by_index: dict[int, tuple[str, str, str]] = {}

    for index, run in enumerate(runs):
        selected_identity = _selected_identity(run)
        if selected_identity is not None:
            selected_by_index[index] = selected_identity

        if not _is_tie_run(run):
            continue

        pair = _extract_tie_pair(run)
        if pair is None:
            continue

        tie_events.append(
            {
                "index": index,
                "pair": pair,
                "signature": _pair_signature(pair),
                "equal_dimensions": _equal_dimensions(pair),
            }
        )

    repeated_pairs = Counter(event["pair"]["pair_key"] for event in tie_events)
    repeated_signatures = Counter(event["signature"] for event in tie_events)
    dominant_dimensions = Counter(
        dimension
        for event in tie_events
        for dimension in event["equal_dimensions"]
    )

    event_resolution_rows: list[dict[str, Any]] = []
    time_to_resolution_runs: list[int] = []

    resolved_within_1 = 0
    resolved_within_3 = 0
    resolved_within_5 = 0
    resolved_within_10 = 0
    unresolved_total = 0

    for event in tie_events:
        resolution = _resolve_event(event, selected_by_index)

        delta = resolution["time_to_resolution_runs"]
        if delta is None:
            unresolved_total += 1
        else:
            time_to_resolution_runs.append(delta)
            if delta <= 1:
                resolved_within_1 += 1
            if delta <= 3:
                resolved_within_3 += 1
            if delta <= 5:
                resolved_within_5 += 1
            if delta <= 10:
                resolved_within_10 += 1

        event_resolution_rows.append(
            {
                "tie_run_index": event["index"],
                "candidates": [
                    _identity_row(event["pair"]["pair_key"][0]),
                    _identity_row(event["pair"]["pair_key"][1]),
                ],
                "resolved_after_tie": resolution["resolved_after_tie"],
                "resolved_winner": (
                    _identity_row(resolution["resolved_winner"])
                    if resolution["resolved_winner"] is not None
                    else None
                ),
                "resolution_selected_run_index": resolution["resolution_selected_run_index"],
                "time_to_resolution_runs": resolution["time_to_resolution_runs"],
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for pair_key, count in sorted(repeated_pairs.items(), key=lambda item: (-item[1], item[0])):
        related = [event for event in tie_events if event["pair"]["pair_key"] == pair_key]
        pair_time_values: list[int] = []
        for event in related:
            resolution = _resolve_event(event, selected_by_index)
            if resolution["time_to_resolution_runs"] is not None:
                pair_time_values.append(resolution["time_to_resolution_runs"])

        pair_rows.append(
            {
                "candidates": [
                    _identity_row(pair_key[0]),
                    _identity_row(pair_key[1]),
                ],
                "count": count,
                "resolved_after_any_tie": len(pair_time_values) > 0,
                "resolved_within_3_runs_count": sum(1 for value in pair_time_values if value <= 3),
                "resolved_within_5_runs_count": sum(1 for value in pair_time_values if value <= 5),
                "resolved_within_10_runs_count": sum(1 for value in pair_time_values if value <= 10),
                "time_to_resolution_summary": _numeric_summary_int(pair_time_values),
            }
        )

    tie_runs = len(tie_events)
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "shadow_tie_break_diagnosis_report",
            "input_path": str(input_path),
            "recent_run_limit": recent_run_limit,
            "analyzed_runs": len(runs),
            "data_quality": data_quality,
        },
        "overall": {
            "tie_runs": tie_runs,
            "tie_frequency": _safe_ratio(tie_runs, len(runs)),
            "resolved_within_1_run_count": resolved_within_1,
            "resolved_within_3_runs_count": resolved_within_3,
            "resolved_within_5_runs_count": resolved_within_5,
            "resolved_within_10_runs_count": resolved_within_10,
            "unresolved_tie_event_count": unresolved_total,
            "resolved_within_1_run_ratio": _safe_ratio(resolved_within_1, tie_runs),
            "resolved_within_3_runs_ratio": _safe_ratio(resolved_within_3, tie_runs),
            "resolved_within_5_runs_ratio": _safe_ratio(resolved_within_5, tie_runs),
            "resolved_within_10_runs_ratio": _safe_ratio(resolved_within_10, tie_runs),
            "repeated_tie_pair_count": sum(1 for count in repeated_pairs.values() if count > 1),
            "tie_signature_collision_count": sum(
                count for count in repeated_signatures.values() if count > 1
            ),
            "same_candidate_tuple_repetition": any(count > 1 for count in repeated_pairs.values()),
            "time_to_resolution_summary": _numeric_summary_int(time_to_resolution_runs),
        },
        "repeated_tie_pairs": pair_rows,
        "tie_event_resolutions": event_resolution_rows,
        "tie_signature_collisions": [
            {"signature": repr(signature), "count": count}
            for signature, count in sorted(
                repeated_signatures.items(),
                key=lambda item: (-item[1], str(item[0])),
            )
            if count > 1
        ],
        "dominant_tie_dimensions": [
            {
                "dimension": dimension,
                "count": count,
                "rate": _safe_ratio(count, tie_runs),
            }
            for dimension, count in sorted(
                dominant_dimensions.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
    }


def render_shadow_tie_break_diagnosis_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall", {})
    lines = [
        "# Shadow Tie-Break Diagnosis",
        "",
        "## Overall",
        "",
        f"- tie_runs: {overall.get('tie_runs', 0)}",
        f"- tie_frequency: {overall.get('tie_frequency', 0.0)}",
        f"- resolved_within_1_run_ratio: {overall.get('resolved_within_1_run_ratio', 0.0)}",
        f"- resolved_within_3_runs_ratio: {overall.get('resolved_within_3_runs_ratio', 0.0)}",
        f"- resolved_within_5_runs_ratio: {overall.get('resolved_within_5_runs_ratio', 0.0)}",
        f"- resolved_within_10_runs_ratio: {overall.get('resolved_within_10_runs_ratio', 0.0)}",
        f"- unresolved_tie_event_count: {overall.get('unresolved_tie_event_count', 0)}",
        f"- time_to_resolution_summary: {overall.get('time_to_resolution_summary', {})}",
        "",
        "## Repeated Tie Pairs",
        "",
    ]
    for row in summary.get("repeated_tie_pairs", []):
        candidates = row.get("candidates", [])
        pair_text = " vs ".join(
            f"{candidate.get('symbol')}/{candidate.get('strategy')}/{candidate.get('horizon')}"
            for candidate in candidates
        )
        lines.append(
            f"- {pair_text}: "
            f"count={row.get('count')}, "
            f"within_3={row.get('resolved_within_3_runs_count')}, "
            f"within_5={row.get('resolved_within_5_runs_count')}, "
            f"within_10={row.get('resolved_within_10_runs_count')}"
        )
    if not summary.get("repeated_tie_pairs"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _select_recent_runs(records: list[dict[str, Any]], recent_run_limit: int) -> list[dict[str, Any]]:
    runs = [_normalize_run(record, index) for index, record in enumerate(records)]
    runs.sort(key=lambda item: (_timestamp_sort_key(item), item["line_number"]))
    if recent_run_limit > 0:
        runs = runs[-recent_run_limit:]
    return runs


def _normalize_run(record: dict[str, Any], index: int) -> dict[str, Any]:
    ranking = record.get("ranking") if isinstance(record.get("ranking"), list) else []
    abstain_diagnosis = (
        record.get("abstain_diagnosis")
        if isinstance(record.get("abstain_diagnosis"), dict)
        else {}
    )
    return {
        "line_number": index + 1,
        "generated_at": _clean_text(record.get("generated_at")),
        "timestamp": _parse_timestamp(record.get("generated_at")),
        "selection_status": _clean_text(record.get("selection_status")) or "unknown",
        "reason_codes": _reason_codes(record),
        "selected_symbol": _clean_text(record.get("selected_symbol")),
        "selected_strategy": _clean_text(record.get("selected_strategy")),
        "selected_horizon": _clean_text(record.get("selected_horizon")),
        "ranking": [item for item in ranking if isinstance(item, dict)],
        "abstain_diagnosis": abstain_diagnosis,
    }


def _extract_tie_pair(run: dict[str, Any]) -> dict[str, Any] | None:
    diagnosis = run.get("abstain_diagnosis") or {}
    first = diagnosis.get("top_candidate") if isinstance(diagnosis.get("top_candidate"), dict) else None
    second = diagnosis.get("compared_candidate") if isinstance(diagnosis.get("compared_candidate"), dict) else None

    if first is None or second is None:
        eligible = [
            item
            for item in run.get("ranking", [])
            if _clean_text(item.get("candidate_status")) == "eligible"
        ]
        ranked_pool = eligible if len(eligible) >= 2 else list(run.get("ranking", []))
        ranked_pool = [item for item in ranked_pool if isinstance(item, dict)]
        ranked_pool.sort(key=_candidate_sort_key, reverse=True)

        if len(ranked_pool) >= 2:
            first = ranked_pool[0]
            second = ranked_pool[1]

    if not isinstance(first, dict) or not isinstance(second, dict):
        return None

    first_identity = _candidate_identity(first)
    second_identity = _candidate_identity(second)
    if first_identity is None or second_identity is None:
        return None

    ordered = tuple(sorted((first_identity, second_identity)))
    return {
        "first": first,
        "second": second,
        "pair_key": ordered,
    }


def _resolve_event(
    event: dict[str, Any],
    selected_by_index: dict[int, tuple[str, str, str]],
) -> dict[str, Any]:
    event_index = event["index"]
    pair_members = set(event["pair"]["pair_key"])
    for index in sorted(selected_by_index):
        if index <= event_index:
            continue
        selected_identity = selected_by_index[index]
        if selected_identity in pair_members:
            return {
                "resolved_after_tie": True,
                "resolved_winner": selected_identity,
                "resolution_selected_run_index": index,
                "time_to_resolution_runs": index - event_index,
            }
    return {
        "resolved_after_tie": False,
        "resolved_winner": None,
        "resolution_selected_run_index": None,
        "time_to_resolution_runs": None,
    }


def _pair_signature(pair: dict[str, Any]) -> tuple[Any, Any]:
    return tuple(sorted((_candidate_signature(pair["first"]), _candidate_signature(pair["second"]))))


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _clean_text(candidate.get("candidate_status")),
        candidate.get("selection_score"),
        candidate.get("aggregate_score"),
        candidate.get("supporting_major_deficit_count"),
        candidate.get("sample_count"),
        candidate.get("median_future_return_pct"),
        candidate.get("positive_rate_pct"),
        candidate.get("robustness_signal_pct"),
        _clean_text(candidate.get("selected_candidate_strength")),
        _clean_text(candidate.get("selected_stability_label")),
        _clean_text(candidate.get("drift_direction")),
        candidate.get("edge_stability_score"),
    )


def _equal_dimensions(pair: dict[str, Any]) -> list[str]:
    first = pair["first"]
    second = pair["second"]
    dimensions = {
        "selection_score": first.get("selection_score") == second.get("selection_score"),
        "aggregate_score": first.get("aggregate_score") == second.get("aggregate_score"),
        "supporting_major_deficit_count": first.get("supporting_major_deficit_count")
        == second.get("supporting_major_deficit_count"),
        "sample_count": first.get("sample_count") == second.get("sample_count"),
        "median_future_return_pct": first.get("median_future_return_pct")
        == second.get("median_future_return_pct"),
        "positive_rate_pct": first.get("positive_rate_pct") == second.get("positive_rate_pct"),
        "robustness_signal_pct": first.get("robustness_signal_pct")
        == second.get("robustness_signal_pct"),
        "selected_candidate_strength": _clean_text(first.get("selected_candidate_strength"))
        == _clean_text(second.get("selected_candidate_strength")),
        "selected_stability_label": _clean_text(first.get("selected_stability_label"))
        == _clean_text(second.get("selected_stability_label")),
        "drift_direction": _clean_text(first.get("drift_direction"))
        == _clean_text(second.get("drift_direction")),
        "edge_stability_score": first.get("edge_stability_score")
        == second.get("edge_stability_score"),
    }
    return [name for name, matched in dimensions.items() if matched]


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _to_float(candidate.get("selection_score")) or 0.0,
        _to_float(candidate.get("aggregate_score")) or 0.0,
        _to_float(candidate.get("edge_stability_score")) or 0.0,
    )


def _numeric_summary_int(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": float(median(ordered)),
    }


def _selected_identity(run: dict[str, Any]) -> tuple[str, str, str] | None:
    if run.get("selection_status") != "selected":
        return None
    symbol = run.get("selected_symbol")
    strategy = run.get("selected_strategy")
    horizon = run.get("selected_horizon")
    if symbol is None or strategy is None or horizon is None:
        return None
    return (symbol, strategy, horizon)


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str] | None:
    symbol = _clean_text(candidate.get("symbol"))
    strategy = _clean_text(candidate.get("strategy"))
    horizon = _clean_text(candidate.get("horizon"))
    if symbol is None or strategy is None or horizon is None:
        return None
    return (symbol, strategy, horizon)


def _identity_row(identity: tuple[str, str, str] | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    symbol, strategy, horizon = identity
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
    }


def _is_tie_run(run: dict[str, Any]) -> bool:
    if TIE_REASON_CODE in run.get("reason_codes", []):
        return True
    diagnosis = run.get("abstain_diagnosis") or {}
    return _clean_text(diagnosis.get("category")) == "tied_top_candidates"


def _reason_codes(record: dict[str, Any]) -> list[str]:
    values = record.get("reason_codes")
    if not isinstance(values, list):
        return []
    return [value for value in (_clean_text(item) for item in values) if value is not None]


def _parse_timestamp(value: Any) -> datetime | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_sort_key(run: dict[str, Any]) -> tuple[int, str]:
    timestamp = run.get("timestamp")
    if isinstance(timestamp, datetime):
        return (1, timestamp.isoformat())
    return (0, run.get("generated_at") or "")


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shadow tie-break diagnosis report."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--recent-run-limit",
        type=int,
        default=DEFAULT_RECENT_RUN_LIMIT,
        help="Maximum number of most-recent shadow runs to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_shadow_tie_break_diagnosis_report(
        input_path=args.input,
        output_dir=args.output_dir,
        recent_run_limit=args.recent_run_limit,
    )


if __name__ == "__main__":
    main()
