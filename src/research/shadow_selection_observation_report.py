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


def run_shadow_selection_observation_report(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    recent_run_limit: int = DEFAULT_RECENT_RUN_LIMIT,
) -> dict[str, Any]:
    resolved_input = input_path or DEFAULT_INPUT_PATH
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR
    loaded = load_shadow_records(resolved_input)
    summary = build_shadow_selection_observation_summary(
        records=loaded["records"],
        input_path=resolved_input,
        data_quality=loaded["data_quality"],
        recent_run_limit=recent_run_limit,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / "shadow_selection_observation_report.json"
    md_path = resolved_output_dir / "shadow_selection_observation_report.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_shadow_selection_observation_markdown(summary),
        encoding="utf-8",
    )
    return {
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output_dir),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "analyzed_runs": summary["metadata"]["analyzed_runs"],
    }


def build_shadow_selection_observation_summary(
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
    recent_run_limit: int,
) -> dict[str, Any]:
    runs = _select_recent_runs(records, recent_run_limit)
    total_runs = len(runs)
    status_counts = Counter(run["selection_status"] for run in runs)
    tie_runs = [run for run in runs if _is_tie_run(run)]

    eligible_count_distribution = Counter(
        len(_ranking_identities(run, "eligible")) for run in runs
    )
    penalized_count_distribution = Counter(
        len(_ranking_identities(run, "penalized")) for run in runs
    )

    selected_identities = [
        identity for identity in (_selected_identity(run) for run in runs) if identity is not None
    ]
    selected_repetition = Counter(selected_identities)
    selection_scores = [
        score for score in (_to_float(run.get("selection_score")) for run in runs) if score is not None
    ]

    abstain_reason_counts = Counter(
        reason
        for run in runs
        if run.get("selection_status") == "abstain"
        for reason in _reason_codes(run)
    )

    longest_repeat_streak = _longest_selected_repeat_streak(runs)
    selected_total = status_counts.get("selected", 0)
    abstain_total = status_counts.get("abstain", 0)
    blocked_total = status_counts.get("blocked", 0)

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "shadow_selection_observation_report",
            "input_path": str(input_path),
            "recent_run_limit": recent_run_limit,
            "analyzed_runs": total_runs,
            "data_quality": data_quality,
        },
        "selection_outcomes": {
            "selected_runs": selected_total,
            "abstain_runs": abstain_total,
            "blocked_runs": blocked_total,
            "selected_ratio": _safe_ratio(selected_total, total_runs),
            "abstain_ratio": _safe_ratio(abstain_total, total_runs),
            "blocked_ratio": _safe_ratio(blocked_total, total_runs),
            "tie_runs": len(tie_runs),
            "tie_frequency": _safe_ratio(len(tie_runs), total_runs),
        },
        "candidate_count_distributions": {
            "eligible_candidate_count_distribution": _distribution_rows(
                eligible_count_distribution,
                total_runs,
            ),
            "penalized_candidate_count_distribution": _distribution_rows(
                penalized_count_distribution,
                total_runs,
            ),
        },
        "selected_candidate_repetition": {
            "unique_selected_candidate_count": len(selected_repetition),
            "selected_candidate_repetition_counts": _identity_rows(
                selected_repetition,
                selected_total,
            ),
            "longest_repeat_streak": longest_repeat_streak,
        },
        "selection_score_summary": _numeric_summary(selection_scores),
        "abstain_reason_summary": _distribution_rows(
            abstain_reason_counts,
            abstain_total,
            key_name="reason_code",
        ),
    }


def render_shadow_selection_observation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Recent Shadow Selection Observation Report",
        "",
        "## Selection Outcomes",
        "",
    ]
    outcomes = summary.get("selection_outcomes", {})
    lines.append(f"- analyzed_runs: {summary.get('metadata', {}).get('analyzed_runs', 0)}")
    lines.append(f"- selected_ratio: {outcomes.get('selected_ratio', 0.0)}")
    lines.append(f"- abstain_ratio: {outcomes.get('abstain_ratio', 0.0)}")
    lines.append(f"- tie_frequency: {outcomes.get('tie_frequency', 0.0)}")
    lines.append("")
    lines.append("## Selected Candidate Repetition")
    lines.append("")
    repetition = summary.get("selected_candidate_repetition", {})
    lines.append(
        f"- unique_selected_candidate_count: {repetition.get('unique_selected_candidate_count', 0)}"
    )
    streak = repetition.get("longest_repeat_streak", {})
    if streak:
        lines.append(
            "- longest_repeat_streak: "
            f"{streak.get('symbol')}/{streak.get('strategy')}/{streak.get('horizon')} "
            f"({streak.get('streak', 0)})"
        )
    else:
        lines.append("- longest_repeat_streak: n/a")
    lines.append("")
    lines.append("## Abstain Reason Summary")
    lines.append("")
    for row in summary.get("abstain_reason_summary", []):
        lines.append(
            f"- {row.get('reason_code')}: {row.get('count')} ({row.get('rate')})"
        )
    if not summary.get("abstain_reason_summary"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _select_recent_runs(records: list[dict[str, Any]], recent_run_limit: int) -> list[dict[str, Any]]:
    normalized = [_normalize_run(record, index) for index, record in enumerate(records)]
    normalized.sort(key=lambda item: (_timestamp_sort_key(item), item["line_number"]))
    if recent_run_limit > 0:
        normalized = normalized[-recent_run_limit:]
    return normalized


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
        "selection_score": record.get("selection_score"),
        "ranking": [item for item in ranking if isinstance(item, dict)],
        "abstain_diagnosis": abstain_diagnosis,
    }


def _ranking_identities(run: dict[str, Any], status: str) -> list[tuple[str, str, str]]:
    identities: list[tuple[str, str, str]] = []
    for item in run.get("ranking", []):
        if _clean_text(item.get("candidate_status")) != status:
            continue
        identity = _candidate_identity(item)
        if identity is not None:
            identities.append(identity)
    return identities


def _selected_identity(run: dict[str, Any]) -> tuple[str, str, str] | None:
    if run.get("selection_status") != "selected":
        return None
    symbol = _clean_text(run.get("selected_symbol"))
    strategy = _clean_text(run.get("selected_strategy"))
    horizon = _clean_text(run.get("selected_horizon"))
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


def _longest_selected_repeat_streak(runs: list[dict[str, Any]]) -> dict[str, Any]:
    best_identity: tuple[str, str, str] | None = None
    best_streak = 0
    current_identity: tuple[str, str, str] | None = None
    current_streak = 0

    for run in runs:
        identity = _selected_identity(run)
        if identity is None:
            current_identity = None
            current_streak = 0
            continue

        if identity == current_identity:
            current_streak += 1
        else:
            current_identity = identity
            current_streak = 1

        if current_streak > best_streak:
            best_identity = identity
            best_streak = current_streak

    if best_identity is None:
        return {}
    symbol, strategy, horizon = best_identity
    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "streak": best_streak,
    }


def _identity_rows(
    counts: Counter[tuple[str, str, str]],
    denominator: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (symbol, strategy, horizon), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        rows.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "count": count,
                "rate": _safe_ratio(count, denominator),
            }
        )
    return rows


def _distribution_rows(
    counts: Counter[Any],
    denominator: int,
    *,
    key_name: str = "value",
) -> list[dict[str, Any]]:
    return [
        {
            key_name: key,
            "count": count,
            "rate": _safe_ratio(count, denominator),
        }
        for key, count in sorted(counts.items(), key=lambda item: (item[0], item[1]))
    ]


def _numeric_summary(values: list[float]) -> dict[str, Any]:
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
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(float(median(ordered)), 6),
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
        description="Build a recent shadow-selection observation report."
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
    run_shadow_selection_observation_report(
        input_path=args.input,
        output_dir=args.output_dir,
        recent_run_limit=args.recent_run_limit,
    )


if __name__ == "__main__":
    main()
