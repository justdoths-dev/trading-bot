from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def run_shadow_selection_stability_diagnosis_report(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    recent_run_limit: int = DEFAULT_RECENT_RUN_LIMIT,
) -> dict[str, Any]:
    resolved_input = input_path or DEFAULT_INPUT_PATH
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR
    loaded = load_shadow_records(resolved_input)
    summary = build_shadow_selection_stability_diagnosis_summary(
        records=loaded["records"],
        input_path=resolved_input,
        data_quality=loaded["data_quality"],
        recent_run_limit=recent_run_limit,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / "shadow_selection_stability_diagnosis_report.json"
    md_path = resolved_output_dir / "shadow_selection_stability_diagnosis_report.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_shadow_selection_stability_diagnosis_markdown(summary),
        encoding="utf-8",
    )
    return {
        "input_path": str(resolved_input),
        "output_dir": str(resolved_output_dir),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "analyzed_runs": summary["metadata"]["analyzed_runs"],
    }


def build_shadow_selection_stability_diagnosis_summary(
    *,
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
    recent_run_limit: int,
) -> dict[str, Any]:
    runs = _select_recent_runs(records, recent_run_limit)
    appearances: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"selected": [], "eligible": []}
    )

    for index, run in enumerate(runs):
        selected_identity = _selected_identity(run)
        if selected_identity is not None:
            appearances[selected_identity]["selected"].append(index)

        for identity in _eligible_identities(run):
            appearances[identity]["eligible"].append(index)

    candidate_rows = [
        _build_candidate_row(identity, series, len(runs))
        for identity, series in sorted(appearances.items())
    ]
    candidate_rows.sort(
        key=lambda row: (
            -row["selected_recurrence_count"],
            -row["eligible_recurrence_count"],
            -row["recent_persistence_window_summary"]["selected"]["last_10"]["count"],
            -row["recent_persistence_window_summary"]["eligible"]["last_10"]["count"],
            row["symbol"],
            row["strategy"],
            row["horizon"],
        )
    )

    label_counts = Counter(row["convergence_label"] for row in candidate_rows)
    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "shadow_selection_stability_diagnosis_report",
            "input_path": str(input_path),
            "recent_run_limit": recent_run_limit,
            "analyzed_runs": len(runs),
            "data_quality": data_quality,
        },
        "overall": {
            "unique_candidate_count": len(candidate_rows),
            "selected_candidate_count": sum(
                1 for row in candidate_rows if row["selected_recurrence_count"] > 0
            ),
            "eligible_candidate_count": sum(
                1 for row in candidate_rows if row["eligible_recurrence_count"] > 0
            ),
            "convergence_label_distribution": [
                {
                    "label": label,
                    "count": count,
                    "rate": _safe_ratio(count, len(candidate_rows)),
                }
                for label, count in sorted(
                    label_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
        "candidates": candidate_rows,
    }


def render_shadow_selection_stability_diagnosis_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Shadow Selection Stability Diagnosis",
        "",
        "## Overall",
        "",
        f"- analyzed_runs: {summary.get('metadata', {}).get('analyzed_runs', 0)}",
        f"- unique_candidate_count: {summary.get('overall', {}).get('unique_candidate_count', 0)}",
        f"- selected_candidate_count: {summary.get('overall', {}).get('selected_candidate_count', 0)}",
        f"- eligible_candidate_count: {summary.get('overall', {}).get('eligible_candidate_count', 0)}",
        "",
        "## Candidate Labels",
        "",
    ]
    for row in summary.get("overall", {}).get("convergence_label_distribution", []):
        lines.append(f"- {row.get('label')}: {row.get('count')} ({row.get('rate')})")
    if not summary.get("overall", {}).get("convergence_label_distribution"):
        lines.append("- none")
    lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    for row in summary.get("candidates", [])[:10]:
        lines.append(
            "- "
            f"{row.get('symbol')}/{row.get('strategy')}/{row.get('horizon')}: "
            f"selected={row.get('selected_recurrence_count')}, "
            f"eligible={row.get('eligible_recurrence_count')}, "
            f"selected_last_10={row.get('recent_persistence_window_summary', {}).get('selected', {}).get('last_10', {}).get('count', 0)}, "
            f"eligible_last_10={row.get('recent_persistence_window_summary', {}).get('eligible', {}).get('last_10', {}).get('count', 0)}, "
            f"label={row.get('convergence_label')}"
        )
    if not summary.get("candidates"):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_candidate_row(
    identity: tuple[str, str, str],
    series: dict[str, list[int]],
    analyzed_runs: int,
) -> dict[str, Any]:
    symbol, strategy, horizon = identity
    selected_positions = sorted(series.get("selected", []))
    eligible_positions = sorted(series.get("eligible", []))

    selected_recent = _recent_window_summary(selected_positions, analyzed_runs)
    eligible_recent = _recent_window_summary(eligible_positions, analyzed_runs)

    selected_streak = _longest_streak(selected_positions)
    eligible_streak = _longest_streak(eligible_positions)
    selected_count = len(selected_positions)
    eligible_count = len(eligible_positions)

    label = _convergence_label(
        analyzed_runs=analyzed_runs,
        selected_count=selected_count,
        eligible_count=eligible_count,
        selected_streak=selected_streak,
        eligible_streak=eligible_streak,
        selected_recent_5=selected_recent["last_5"]["count"],
        selected_recent_10=selected_recent["last_10"]["count"],
        eligible_recent_5=eligible_recent["last_5"]["count"],
        eligible_recent_10=eligible_recent["last_10"]["count"],
    )

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "selected_recurrence_count": selected_count,
        "eligible_recurrence_count": eligible_count,
        "repeat_streak": {
            "selected": selected_streak,
            "eligible": eligible_streak,
        },
        "reappearance_gap": {
            "selected": _gap_summary(selected_positions),
            "eligible": _gap_summary(eligible_positions),
        },
        "recent_persistence_window_summary": {
            "selected": selected_recent,
            "eligible": eligible_recent,
        },
        "convergence_label": label,
    }


def _recent_window_summary(
    positions: list[int],
    analyzed_runs: int,
) -> dict[str, dict[str, int | float]]:
    return {
        "last_5": _window_count(positions, analyzed_runs, 5),
        "last_10": _window_count(positions, analyzed_runs, 10),
        "last_20": _window_count(positions, analyzed_runs, 20),
    }


def _window_count(
    positions: list[int],
    analyzed_runs: int,
    window: int,
) -> dict[str, int | float]:
    if analyzed_runs <= 0:
        return {"count": 0, "ratio": 0.0}
    start = max(0, analyzed_runs - window)
    count = sum(1 for value in positions if value >= start)
    denominator = min(window, analyzed_runs)
    return {"count": count, "ratio": _safe_ratio(count, denominator)}


def _gap_summary(positions: list[int]) -> dict[str, Any]:
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    if not gaps:
        return {"count": 0, "min": None, "max": None, "median": None}
    ordered = sorted(gaps)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "median": float(median(ordered)),
    }


def _convergence_label(
    *,
    analyzed_runs: int,
    selected_count: int,
    eligible_count: int,
    selected_streak: int,
    eligible_streak: int,
    selected_recent_5: int,
    selected_recent_10: int,
    eligible_recent_5: int,
    eligible_recent_10: int,
) -> str:
    if analyzed_runs < 5:
        return "insufficient_recent_history"

    if (
        selected_count >= 4
        and eligible_count >= 4
        and selected_streak >= 2
        and selected_recent_5 >= 2
        and selected_recent_10 >= 3
        and eligible_recent_10 >= 4
    ):
        return "persistent_emergence"

    if (
        selected_count >= 10
        and eligible_count >= 15
        and selected_streak >= 5
        and eligible_streak >= 10
        and selected_recent_10 == 0
        and eligible_recent_10 == 0
    ):
        return "historically_dominant_recently_inactive"

    if (
        eligible_count >= 4
        and (selected_count >= 1 or eligible_streak >= 2)
        and eligible_recent_5 >= 2
        and eligible_recent_10 >= 3
    ):
        return "weak_convergence"

    if (
        eligible_count >= 3
        and selected_count <= 1
        and eligible_recent_5 >= 2
    ):
        return "unstable_rotation"

    if eligible_count <= 2 and selected_count <= 1:
        return "mostly_noise"

    return "weak_convergence"


def _select_recent_runs(
    records: list[dict[str, Any]],
    recent_run_limit: int,
) -> list[dict[str, Any]]:
    runs = [_normalize_run(record, index) for index, record in enumerate(records)]
    runs.sort(key=lambda item: (_timestamp_sort_key(item), item["line_number"]))
    if recent_run_limit > 0:
        runs = runs[-recent_run_limit:]
    return runs


def _normalize_run(record: dict[str, Any], index: int) -> dict[str, Any]:
    ranking = record.get("ranking") if isinstance(record.get("ranking"), list) else []
    return {
        "line_number": index + 1,
        "generated_at": _clean_text(record.get("generated_at")),
        "timestamp": _parse_timestamp(record.get("generated_at")),
        "selection_status": _clean_text(record.get("selection_status")) or "unknown",
        "selected_symbol": _clean_text(record.get("selected_symbol")),
        "selected_strategy": _clean_text(record.get("selected_strategy")),
        "selected_horizon": _clean_text(record.get("selected_horizon")),
        "ranking": [item for item in ranking if isinstance(item, dict)],
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


def _eligible_identities(run: dict[str, Any]) -> set[tuple[str, str, str]]:
    identities: set[tuple[str, str, str]] = set()
    for item in run.get("ranking", []):
        if _clean_text(item.get("candidate_status")) != "eligible":
            continue
        identity = _candidate_identity(item)
        if identity is not None:
            identities.add(identity)
    return identities


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str, str] | None:
    symbol = _clean_text(candidate.get("symbol"))
    strategy = _clean_text(candidate.get("strategy"))
    horizon = _clean_text(candidate.get("horizon"))
    if symbol is None or strategy is None or horizon is None:
        return None
    return (symbol, strategy, horizon)


def _longest_streak(positions: list[int]) -> int:
    if not positions:
        return 0
    best = 1
    current = 1
    for left, right in zip(positions, positions[1:]):
        if right == left + 1:
            current += 1
        else:
            current = 1
        if current > best:
            best = current
    return best


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a shadow-selection stability diagnosis report."
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
    run_shadow_selection_stability_diagnosis_report(
        input_path=args.input,
        output_dir=args.output_dir,
        recent_run_limit=args.recent_run_limit,
    )


if __name__ == "__main__":
    main()
