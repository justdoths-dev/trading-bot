from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HORIZONS = ("15m", "1h", "4h")
CATEGORY_SPECS = (
    ("strategy", "top_strategy", "by_strategy"),
    ("symbol", "top_symbol", "by_symbol"),
    ("alignment_state", "top_alignment_state", "by_alignment_state"),
)

DEFAULT_COMPARISON_SUMMARY_PATH = Path("logs/research_reports/comparison/summary.json")
DEFAULT_SCORE_DRIFT_SUMMARY_PATH = Path("logs/research_reports/score_drift/summary.json")
DEFAULT_LATEST_SUMMARY_PATH = Path("logs/research_reports/latest/summary.json")
DEFAULT_CUMULATIVE_SUMMARY_PATH = Path("logs/research_reports/cumulative/summary.json")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_JSON_NAME = "latest_cumulative_fallback_simulator_summary.json"
DEFAULT_MD_NAME = "latest_cumulative_fallback_simulator_summary.md"


def run_latest_cumulative_fallback_simulator(
    comparison_summary_path: Path | None = None,
    score_drift_summary_path: Path | None = None,
    latest_summary_path: Path | None = None,
    cumulative_summary_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    comparison_path = comparison_summary_path or DEFAULT_COMPARISON_SUMMARY_PATH
    score_drift_path = score_drift_summary_path or DEFAULT_SCORE_DRIFT_SUMMARY_PATH
    latest_path = latest_summary_path or DEFAULT_LATEST_SUMMARY_PATH
    cumulative_path = cumulative_summary_path or DEFAULT_CUMULATIVE_SUMMARY_PATH
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR

    comparison_summary = _load_json_file(comparison_path)
    score_drift_summary = _load_json_file(score_drift_path)
    latest_summary = _load_json_file(latest_path)
    cumulative_summary = _load_json_file(cumulative_path)

    summary = build_latest_cumulative_fallback_simulation_summary(
        comparison_summary=comparison_summary,
        score_drift_summary=score_drift_summary,
        latest_summary=latest_summary,
        cumulative_summary=cumulative_summary,
    )
    markdown = render_latest_cumulative_fallback_simulation_markdown(summary)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / DEFAULT_JSON_NAME
    md_path = resolved_output_dir / DEFAULT_MD_NAME

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "markdown": markdown,
        "summary_json": str(json_path),
        "summary_md": str(md_path),
    }


def build_latest_cumulative_fallback_simulation_summary(
    *,
    comparison_summary: dict[str, Any],
    score_drift_summary: dict[str, Any],
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    rows = _build_pair_rows(
        comparison_summary=comparison_summary,
        score_drift_summary=score_drift_summary,
        latest_summary=latest_summary,
        cumulative_summary=cumulative_summary,
    )

    recovery_analysis = _build_recovery_analysis(rows)
    horizon_breakdown = _build_horizon_breakdown(rows)
    simulated_recovery = _build_simulated_candidate_recovery(rows)
    simulated_impact = _build_simulated_selection_impact(rows)
    final_diagnosis = _build_final_diagnosis(rows, simulated_impact)

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_pairs": len(rows),
        },
        "recovery_analysis": recovery_analysis,
        "horizon_breakdown": horizon_breakdown,
        "simulated_candidate_recovery": simulated_recovery,
        "simulated_selection_impact": simulated_impact,
        "final_diagnosis": final_diagnosis,
    }


def render_latest_cumulative_fallback_simulation_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    recovery = _safe_dict(summary.get("recovery_analysis"))
    breakdown = _safe_dict(summary.get("horizon_breakdown"))
    impact = _safe_dict(summary.get("simulated_selection_impact"))
    final = _safe_dict(summary.get("final_diagnosis"))
    recovered_candidates = (
        _safe_dict(summary.get("simulated_candidate_recovery")).get("recovered_candidates")
        if isinstance(summary.get("simulated_candidate_recovery"), dict)
        else []
    )

    lines = [
        "# Fallback Simulation Report",
        "",
        "## Executive Summary",
        f"- total_pairs: {metadata.get('total_pairs', 0)}",
        f"- fallback_candidate_possible_count: {recovery.get('fallback_candidate_possible_count', 0)}",
        f"- blocked_by_non_positive_median_count: {recovery.get('blocked_by_non_positive_median_count', 0)}",
        f"- blocked_by_drift_decreasing_count: {recovery.get('blocked_by_drift_decreasing_count', 0)}",
        "",
        "## Recovery Analysis",
        f"- fallback_candidate_possible_count: {recovery.get('fallback_candidate_possible_count', 0)}",
        f"- blocked_by_non_positive_median_count: {recovery.get('blocked_by_non_positive_median_count', 0)}",
        f"- blocked_by_drift_decreasing_count: {recovery.get('blocked_by_drift_decreasing_count', 0)}",
        f"- fallback_not_needed_count: {recovery.get('fallback_not_needed_count', 0)}",
        f"- no_candidate_count: {recovery.get('no_candidate_count', 0)}",
        "",
        "## Horizon Breakdown",
    ]

    for horizon in HORIZONS:
        row = _safe_dict(breakdown.get(horizon))
        lines.append(
            f"- {horizon}: possible={row.get('possible_count', 0)}, "
            f"blocked={row.get('blocked_count', 0)}, "
            f"not_needed={row.get('not_needed_count', 0)}, "
            f"no_candidate={row.get('no_candidate_count', 0)}, "
            f"drift={row.get('drift_distribution', {})}"
        )

    lines.extend(
        [
            "",
            "## Simulated Impact",
            f"- estimated_candidate_count_change: {impact.get('estimated_candidate_count_change', 0)}",
            f"- estimated_horizons_with_recovery: {impact.get('estimated_horizons_with_recovery', 0)}",
            "- estimated_status_shift: "
            f"{_safe_dict(impact.get('estimated_status_shift')).get('NO_CANDIDATES_AVAILABLE_to_NO_ELIGIBLE_CANDIDATES', 0)}",
        ]
    )

    if isinstance(recovered_candidates, list) and recovered_candidates:
        for row in recovered_candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- recovered: {row.get('horizon', 'n/a')} / "
                f"{row.get('category', 'n/a')} / "
                f"{row.get('group', 'n/a')} / "
                f"{row.get('reason', 'n/a')}"
            )
    else:
        lines.append("- recovered: none")

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {final.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final.get('secondary_finding', 'unknown')}",
            f"- recommendation: {final.get('recommendation', 'not_ready')}",
            f"- summary: {final.get('summary', 'n/a')}",
        ]
    )

    return "\n".join(lines).strip()


def _build_pair_rows(
    *,
    comparison_summary: dict[str, Any],
    score_drift_summary: dict[str, Any],
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    comparison = _safe_dict(comparison_summary.get("edge_candidates_comparison"))
    drift_lookup = _build_drift_lookup(score_drift_summary)

    rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        comparison_horizon = _safe_dict(comparison.get(horizon))
        latest_preview_horizon = _preview_horizon(latest_summary, horizon)
        cumulative_preview_horizon = _preview_horizon(cumulative_summary, horizon)

        for category, candidate_key, ranking_key in CATEGORY_SPECS:
            latest_candidate = _safe_dict(latest_preview_horizon.get(candidate_key))
            cumulative_candidate = _safe_dict(cumulative_preview_horizon.get(candidate_key))

            latest_strength = _normalize_strength(latest_candidate.get("candidate_strength"))
            cumulative_strength = _normalize_strength(cumulative_candidate.get("candidate_strength"))

            latest_visible = latest_strength != "insufficient_data"
            cumulative_visible = cumulative_strength != "insufficient_data"

            latest_group = _normalize_group(
                latest_candidate.get("group") or comparison_horizon.get(f"latest_top_{category}_group")
            )
            cumulative_group = _normalize_group(
                cumulative_candidate.get("group") or comparison_horizon.get(f"cumulative_top_{category}_group")
            )
            reference_group = cumulative_group if cumulative_group != "n/a" else latest_group

            latest_metrics = _top_rank_metrics(latest_summary, horizon, ranking_key)
            median_latest = _to_float(latest_metrics.get("median_future_return_pct"))

            drift_direction = drift_lookup.get((category, reference_group), "unknown")
            label = _simulate_label(
                latest_visible=latest_visible,
                cumulative_visible=cumulative_visible,
                median_latest=median_latest,
                drift_direction=drift_direction,
            )

            rows.append(
                {
                    "horizon": horizon,
                    "category": category,
                    "group": reference_group,
                    "label": label,
                    "latest_visible": latest_visible,
                    "cumulative_visible": cumulative_visible,
                    "median_latest": median_latest,
                    "drift_direction": drift_direction,
                    "latest_candidate_strength": latest_strength,
                    "cumulative_candidate_strength": cumulative_strength,
                }
            )

    return rows


def _build_recovery_analysis(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "fallback_candidate_possible_count": sum(
            1 for row in rows if row.get("label") == "fallback_candidate_possible"
        ),
        "blocked_by_non_positive_median_count": sum(
            1 for row in rows if row.get("label") == "blocked_by_non_positive_median"
        ),
        "blocked_by_drift_decreasing_count": sum(
            1 for row in rows if row.get("label") == "blocked_by_drift_decreasing"
        ),
        "fallback_not_needed_count": sum(
            1 for row in rows if row.get("label") == "fallback_not_needed"
        ),
        "no_candidate_count": sum(
            1 for row in rows if row.get("label") == "no_candidate"
        ),
    }


def _simulate_label(
    *,
    latest_visible: bool,
    cumulative_visible: bool,
    median_latest: float | None,
    drift_direction: str,
) -> str:
    if latest_visible:
        return "fallback_not_needed"

    if not cumulative_visible:
        return "no_candidate"

    if median_latest is not None and median_latest <= 0:
        return "blocked_by_non_positive_median"

    if drift_direction == "decrease":
        return "blocked_by_drift_decreasing"

    if drift_direction in {"flat", "increase"}:
        return "fallback_candidate_possible"

    return "blocked_by_non_positive_median"


def _build_horizon_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}

    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if row.get("horizon") == horizon]
        drift_counter: Counter[str] = Counter(
            row.get("drift_direction", "unknown")
            for row in horizon_rows
            if row.get("label")
            in {
                "fallback_candidate_possible",
                "blocked_by_drift_decreasing",
            }
        )

        breakdown[horizon] = {
            "possible_count": sum(
                1 for row in horizon_rows if row.get("label") == "fallback_candidate_possible"
            ),
            "blocked_count": sum(
                1
                for row in horizon_rows
                if row.get("label")
                in {"blocked_by_non_positive_median", "blocked_by_drift_decreasing"}
            ),
            "not_needed_count": sum(
                1 for row in horizon_rows if row.get("label") == "fallback_not_needed"
            ),
            "no_candidate_count": sum(
                1 for row in horizon_rows if row.get("label") == "no_candidate"
            ),
            "drift_distribution": dict(drift_counter),
        }

    return breakdown


def _build_simulated_candidate_recovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [
        {
            "horizon": row.get("horizon", "n/a"),
            "category": row.get("category", "n/a"),
            "group": row.get("group", "n/a"),
            "reason": row.get("label", "n/a"),
        }
        for row in rows
        if row.get("label") == "fallback_candidate_possible"
    ]
    return {"recovered_candidates": recovered}


def _build_simulated_selection_impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_count_change = sum(
        1 for row in rows if row.get("label") == "fallback_candidate_possible"
    )

    horizons_with_recovery = {
        row.get("horizon")
        for row in rows
        if row.get("label") == "fallback_candidate_possible"
    }

    # Conservative estimate:
    # if a horizon recovers at least one candidate slot, it may migrate from
    # NO_CANDIDATES_AVAILABLE to NO_ELIGIBLE_CANDIDATES.
    status_shift_count = len(horizons_with_recovery)

    return {
        "estimated_candidate_count_change": candidate_count_change,
        "estimated_horizons_with_recovery": len(horizons_with_recovery),
        "estimated_status_shift": {
            "NO_CANDIDATES_AVAILABLE_to_NO_ELIGIBLE_CANDIDATES": status_shift_count,
        },
    }


def _build_final_diagnosis(rows: list[dict[str, Any]], impact: dict[str, Any]) -> dict[str, Any]:
    possible = sum(1 for row in rows if row.get("label") == "fallback_candidate_possible")
    negative = sum(1 for row in rows if row.get("label") == "blocked_by_non_positive_median")
    decreasing = sum(1 for row in rows if row.get("label") == "blocked_by_drift_decreasing")
    recovered_horizons = int(impact.get("estimated_horizons_with_recovery", 0))

    if possible >= 4 and recovered_horizons >= 2:
        recommendation = "strong_candidate_for_fallback"
    elif possible >= 1:
        recommendation = "promising_but_blocked"
    else:
        recommendation = "not_ready"

    primary = (
        "simulated_recovery_exists_under_non_decreasing_drift"
        if possible > 0
        else "recovery_signal_not_present"
    )
    secondary = (
        "non_positive_median_is_dominant_blocker"
        if negative >= decreasing
        else "drift_decrease_limits_recovery"
    )

    return {
        "primary_finding": primary,
        "secondary_finding": secondary,
        "recommendation": recommendation,
        "summary": (
            f"possible={possible}, blocked_by_non_positive_median={negative}, "
            f"blocked_by_drift_decreasing={decreasing}, "
            f"estimated_horizons_with_recovery={recovered_horizons}."
        ),
    }


def _preview_horizon(summary: dict[str, Any], horizon: str) -> dict[str, Any]:
    edge_candidates_preview = _safe_dict(summary.get("edge_candidates_preview"))
    by_horizon = _safe_dict(edge_candidates_preview.get("by_horizon"))
    return _safe_dict(by_horizon.get(horizon))


def _top_rank_metrics(summary: dict[str, Any], horizon: str, ranking_key: str) -> dict[str, Any]:
    strategy_lab = _safe_dict(summary.get("strategy_lab"))
    ranking = _safe_dict(strategy_lab.get("ranking"))
    horizon_ranking = _safe_dict(ranking.get(horizon))
    report = horizon_ranking.get(ranking_key)

    if not isinstance(report, dict):
        return {}

    ranked_groups = report.get("ranked_groups")
    if not isinstance(ranked_groups, list) or not ranked_groups:
        return {}

    top = ranked_groups[0]
    if not isinstance(top, dict):
        return {}

    return _safe_dict(top.get("metrics"))


def _build_drift_lookup(score_drift_summary: dict[str, Any]) -> dict[tuple[str, str], str]:
    items = score_drift_summary.get("score_drift")
    if not isinstance(items, list):
        return {}

    lookup: dict[tuple[str, str], str] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        category = _normalize_group(item.get("category"))
        group = _normalize_group(item.get("group"))
        direction = _normalize_group(item.get("drift_direction"))

        if category == "n/a" or group == "n/a":
            continue

        lookup[(category, group)] = direction

    return lookup


def _normalize_strength(value: Any) -> str:
    if not isinstance(value, str):
        return "insufficient_data"
    text = value.strip()
    return text or "insufficient_data"


def _normalize_group(value: Any) -> str:
    if not isinstance(value, str):
        return "n/a"
    text = value.strip()
    return text or "n/a"


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest-vs-cumulative fallback simulator")
    parser.add_argument("--comparison-summary", type=Path, default=DEFAULT_COMPARISON_SUMMARY_PATH)
    parser.add_argument("--score-drift-summary", type=Path, default=DEFAULT_SCORE_DRIFT_SUMMARY_PATH)
    parser.add_argument("--latest-summary", type=Path, default=DEFAULT_LATEST_SUMMARY_PATH)
    parser.add_argument("--cumulative-summary", type=Path, default=DEFAULT_CUMULATIVE_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_latest_cumulative_fallback_simulator(
        comparison_summary_path=args.comparison_summary,
        score_drift_summary_path=args.score_drift_summary,
        latest_summary_path=args.latest_summary,
        cumulative_summary_path=args.cumulative_summary,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "summary_json": result["summary_json"],
                "summary_md": result["summary_md"],
                "final_diagnosis": result["summary"].get("final_diagnosis", {}),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()