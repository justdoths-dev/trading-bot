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
DEFAULT_JSON_NAME = "latest_cumulative_fallback_probe_summary.json"
DEFAULT_MD_NAME = "latest_cumulative_fallback_probe_summary.md"

INTERESTING_LABELS = {
    "latest_failed_due_to_non_positive_median_and_drift_not_decreasing",
    "latest_failed_due_to_non_positive_median_and_drift_decreasing",
    "latest_failed_due_to_non_positive_median",
    "latest_failed_for_other_reasons_and_drift_not_decreasing",
    "latest_failed_for_other_reasons_and_drift_decreasing",
    "latest_failed_cumulative_visible",
}


def run_latest_cumulative_fallback_probe(
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

    summary = build_latest_cumulative_fallback_probe_summary(
        comparison_summary=comparison_summary,
        score_drift_summary=score_drift_summary,
        latest_summary=latest_summary,
        cumulative_summary=cumulative_summary,
        input_paths={
            "comparison_summary": str(comparison_path),
            "score_drift_summary": str(score_drift_path),
            "latest_summary": str(latest_path),
            "cumulative_summary": str(cumulative_path),
        },
    )
    markdown = render_latest_cumulative_fallback_probe_markdown(summary)

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


def build_latest_cumulative_fallback_probe_summary(
    *,
    comparison_summary: dict[str, Any],
    score_drift_summary: dict[str, Any],
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
    input_paths: dict[str, str],
) -> dict[str, Any]:
    rows = _build_pair_rows(
        comparison_summary=comparison_summary,
        score_drift_summary=score_drift_summary,
        latest_summary=latest_summary,
        cumulative_summary=cumulative_summary,
    )
    latest_failure_overview = _build_latest_failure_overview(comparison_summary)
    fallback_analysis = _build_fallback_analysis(rows)
    horizon_breakdown = _build_horizon_breakdown(rows)
    examples = _build_examples(rows)
    final_diagnosis = _build_final_diagnosis(
        fallback_analysis=fallback_analysis,
        horizon_breakdown=horizon_breakdown,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_paths": input_paths,
            "total_horizons_evaluated": len(HORIZONS),
            "total_category_pairs_evaluated": len(rows),
        },
        "latest_failure_overview": latest_failure_overview,
        "fallback_eligibility_analysis": fallback_analysis,
        "horizon_breakdown": horizon_breakdown,
        "representative_examples": examples,
        "final_diagnosis": final_diagnosis,
    }


def render_latest_cumulative_fallback_probe_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    fallback = _safe_dict(summary.get("fallback_eligibility_analysis"))
    horizon_breakdown = _safe_dict(summary.get("horizon_breakdown"))
    examples = _safe_dict(summary.get("representative_examples"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines = [
        "# Latest vs Cumulative Fallback Probe",
        "",
        "## Metadata",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- total_horizons_evaluated: {metadata.get('total_horizons_evaluated', 0)}",
        f"- total_category_pairs_evaluated: {metadata.get('total_category_pairs_evaluated', 0)}",
        "",
        "## Latest Failure Overview",
    ]

    overview_rows = summary.get("latest_failure_overview")
    if isinstance(overview_rows, list) and overview_rows:
        for row in overview_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('horizon', 'n/a')}: "
                f"latest={row.get('latest_candidate_strength', 'insufficient_data')} "
                f"(visible={row.get('latest_visible', False)}), "
                f"cumulative={row.get('cumulative_candidate_strength', 'insufficient_data')} "
                f"(visible={row.get('cumulative_visible', False)})"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Fallback Eligibility Analysis",
            f"- latest_failed_cumulative_visible: {fallback.get('latest_failed_cumulative_visible', 0)}",
            f"- latest_failed_due_to_non_positive_median: {fallback.get('latest_failed_due_to_non_positive_median', 0)}",
            "- latest_failed_due_to_non_positive_median_and_drift_not_decreasing: "
            f"{fallback.get('latest_failed_due_to_non_positive_median_and_drift_not_decreasing', 0)}",
            "- latest_failed_due_to_non_positive_median_and_drift_decreasing: "
            f"{fallback.get('latest_failed_due_to_non_positive_median_and_drift_decreasing', 0)}",
            "- latest_failed_for_other_reasons_and_drift_not_decreasing: "
            f"{fallback.get('latest_failed_for_other_reasons_and_drift_not_decreasing', 0)}",
            "- latest_failed_for_other_reasons_and_drift_decreasing: "
            f"{fallback.get('latest_failed_for_other_reasons_and_drift_decreasing', 0)}",
            "",
            "## Horizon Breakdown",
        ]
    )

    if horizon_breakdown:
        for horizon in HORIZONS:
            row = _safe_dict(horizon_breakdown.get(horizon))
            drift = _safe_dict(row.get("drift_direction_distribution"))
            lines.append(
                f"- {horizon}: "
                f"eligible={row.get('fallback_eligible_count', 0)}, "
                f"ineligible={row.get('fallback_ineligible_count', 0)}, "
                f"non_positive_median={row.get('non_positive_median_count', 0)}, "
                f"drift={drift}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Representative Examples"])
    for horizon in HORIZONS:
        lines.append(f"- {horizon}:")
        rows = examples.get(horizon)
        if not isinstance(rows, list) or not rows:
            lines.append("  - none")
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  - "
                f"{row.get('category', 'n/a')} | "
                f"label={row.get('diagnostic_label', 'n/a')} | "
                f"latest_strength={row.get('latest_candidate_strength', 'insufficient_data')} | "
                f"cumulative_strength={row.get('cumulative_candidate_strength', 'insufficient_data')} | "
                f"median={row.get('latest_top_median_future_return_pct', 'n/a')} | "
                f"sample={row.get('latest_top_sample_count', 'n/a')} | "
                f"labeled={row.get('latest_top_labeled_count', 'n/a')} | "
                f"drift={row.get('drift_direction', 'unknown')}"
            )

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}",
            f"- fallback_justification: {final_diagnosis.get('fallback_justification', 'rarely_justified')}",
            f"- summary: {final_diagnosis.get('summary', 'n/a')}",
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
        latest_horizon = _safe_dict(
            _safe_dict(_safe_dict(latest_summary.get("edge_candidates_preview")).get("by_horizon")).get(horizon)
        )
        cumulative_horizon = _safe_dict(
            _safe_dict(_safe_dict(cumulative_summary.get("edge_candidates_preview")).get("by_horizon")).get(horizon)
        )

        for category, candidate_key, ranking_key in CATEGORY_SPECS:
            latest_candidate = _safe_dict(latest_horizon.get(candidate_key))
            cumulative_candidate = _safe_dict(cumulative_horizon.get(candidate_key))

            latest_strength = _normalize_strength(
                latest_candidate.get("candidate_strength"),
                comparison_horizon.get("latest_candidate_strength"),
            )
            cumulative_strength = _normalize_strength(
                cumulative_candidate.get("candidate_strength"),
                comparison_horizon.get("cumulative_candidate_strength"),
            )

            latest_visible = latest_strength != "insufficient_data"
            cumulative_visible = cumulative_strength != "insufficient_data"

            latest_group = _normalize_group(comparison_horizon.get(f"latest_top_{category}_group"))
            cumulative_group = _normalize_group(comparison_horizon.get(f"cumulative_top_{category}_group"))
            reference_group = cumulative_group if cumulative_group != "n/a" else latest_group

            latest_top_metrics = _top_rank_metrics(latest_summary, horizon, ranking_key)
            latest_non_positive_median = _is_non_positive_median(latest_top_metrics)
            drift_direction = drift_lookup.get((category, reference_group), "unknown")

            diagnostic_label = _diagnostic_label(
                latest_visible=latest_visible,
                cumulative_visible=cumulative_visible,
                latest_non_positive_median=latest_non_positive_median,
                drift_direction=drift_direction,
            )

            rows.append(
                {
                    "horizon": horizon,
                    "category": category,
                    "latest_group": latest_group,
                    "cumulative_group": cumulative_group,
                    "latest_candidate_strength": latest_strength,
                    "cumulative_candidate_strength": cumulative_strength,
                    "latest_visible": latest_visible,
                    "cumulative_visible": cumulative_visible,
                    "latest_non_positive_median": latest_non_positive_median,
                    "drift_direction": drift_direction,
                    "diagnostic_label": diagnostic_label,
                    "latest_top_metrics": latest_top_metrics,
                    "latest_top_symbol_group": _normalize_group(comparison_horizon.get("latest_top_symbol_group")),
                    "latest_top_strategy_group": _normalize_group(comparison_horizon.get("latest_top_strategy_group")),
                    "latest_top_alignment_state_group": _normalize_group(
                        comparison_horizon.get("latest_top_alignment_state_group")
                    ),
                    "cumulative_top_symbol_group": _normalize_group(comparison_horizon.get("cumulative_top_symbol_group")),
                    "cumulative_top_strategy_group": _normalize_group(
                        comparison_horizon.get("cumulative_top_strategy_group")
                    ),
                    "cumulative_top_alignment_state_group": _normalize_group(
                        comparison_horizon.get("cumulative_top_alignment_state_group")
                    ),
                }
            )

    return rows


def _build_latest_failure_overview(comparison_summary: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = _safe_dict(comparison_summary.get("edge_candidates_comparison"))
    rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        row = _safe_dict(comparison.get(horizon))
        latest_strength = _normalize_strength(row.get("latest_candidate_strength"))
        cumulative_strength = _normalize_strength(row.get("cumulative_candidate_strength"))
        rows.append(
            {
                "horizon": horizon,
                "latest_candidate_strength": latest_strength,
                "cumulative_candidate_strength": cumulative_strength,
                "latest_visible": latest_strength != "insufficient_data",
                "cumulative_visible": cumulative_strength != "insufficient_data",
            }
        )

    return rows


def _build_fallback_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "latest_failed_cumulative_visible": sum(
            1 for row in rows if (not row["latest_visible"] and row["cumulative_visible"])
        ),
        "latest_failed_due_to_non_positive_median": sum(
            1
            for row in rows
            if (not row["latest_visible"] and row["cumulative_visible"] and row["latest_non_positive_median"])
        ),
        "latest_failed_due_to_non_positive_median_and_drift_not_decreasing": sum(
            1
            for row in rows
            if row["diagnostic_label"] == "latest_failed_due_to_non_positive_median_and_drift_not_decreasing"
        ),
        "latest_failed_due_to_non_positive_median_and_drift_decreasing": sum(
            1
            for row in rows
            if row["diagnostic_label"] == "latest_failed_due_to_non_positive_median_and_drift_decreasing"
        ),
        "latest_failed_for_other_reasons_and_drift_not_decreasing": sum(
            1
            for row in rows
            if row["diagnostic_label"] == "latest_failed_for_other_reasons_and_drift_not_decreasing"
        ),
        "latest_failed_for_other_reasons_and_drift_decreasing": sum(
            1
            for row in rows
            if row["diagnostic_label"] == "latest_failed_for_other_reasons_and_drift_decreasing"
        ),
    }


def _build_horizon_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}

    for horizon in HORIZONS:
        horizon_rows = [row for row in rows if row.get("horizon") == horizon]
        fallback_rows = [row for row in horizon_rows if row.get("diagnostic_label") in INTERESTING_LABELS]
        drift_counter: Counter[str] = Counter(
            str(row.get("drift_direction", "unknown")) for row in fallback_rows
        )

        breakdown[horizon] = {
            "fallback_eligible_count": sum(
                1
                for row in horizon_rows
                if row.get("diagnostic_label")
                in {
                    "latest_failed_due_to_non_positive_median_and_drift_not_decreasing",
                    "latest_failed_for_other_reasons_and_drift_not_decreasing",
                }
            ),
            "fallback_ineligible_count": sum(
                1
                for row in horizon_rows
                if row.get("diagnostic_label")
                in {
                    "latest_failed_due_to_non_positive_median_and_drift_decreasing",
                    "latest_failed_for_other_reasons_and_drift_decreasing",
                    "latest_failed_due_to_non_positive_median",
                    "latest_failed_cumulative_visible",
                }
            ),
            "non_positive_median_count": sum(
                1 for row in horizon_rows if row.get("latest_non_positive_median") is True
            ),
            "drift_direction_distribution": dict(drift_counter),
        }

    return breakdown


def _build_examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {horizon: [] for horizon in HORIZONS}

    for row in rows:
        label = row.get("diagnostic_label")
        if label not in INTERESTING_LABELS:
            continue

        horizon = str(row.get("horizon", "n/a"))
        metrics = _safe_dict(row.get("latest_top_metrics"))

        result.setdefault(horizon, []).append(
            {
                "horizon": horizon,
                "category": row.get("category", "n/a"),
                "latest_top_symbol_group": row.get("latest_top_symbol_group", "n/a"),
                "latest_top_strategy_group": row.get("latest_top_strategy_group", "n/a"),
                "latest_top_alignment_state_group": row.get("latest_top_alignment_state_group", "n/a"),
                "cumulative_top_symbol_group": row.get("cumulative_top_symbol_group", "n/a"),
                "cumulative_top_strategy_group": row.get("cumulative_top_strategy_group", "n/a"),
                "cumulative_top_alignment_state_group": row.get("cumulative_top_alignment_state_group", "n/a"),
                "latest_candidate_strength": row.get("latest_candidate_strength", "insufficient_data"),
                "cumulative_candidate_strength": row.get("cumulative_candidate_strength", "insufficient_data"),
                "drift_direction": row.get("drift_direction", "unknown"),
                "diagnostic_label": label,
                "latest_top_median_future_return_pct": metrics.get("median_future_return_pct"),
                "latest_top_sample_count": metrics.get("sample_count"),
                "latest_top_labeled_count": metrics.get("labeled_count"),
            }
        )

    for horizon in HORIZONS:
        result[horizon] = result.get(horizon, [])[:3]

    return result


def _build_final_diagnosis(
    *,
    fallback_analysis: dict[str, Any],
    horizon_breakdown: dict[str, Any],
) -> dict[str, Any]:
    eligible = int(
        fallback_analysis.get("latest_failed_due_to_non_positive_median_and_drift_not_decreasing", 0)
    ) + int(
        fallback_analysis.get("latest_failed_for_other_reasons_and_drift_not_decreasing", 0)
    )
    ineligible = int(
        fallback_analysis.get("latest_failed_due_to_non_positive_median_and_drift_decreasing", 0)
    ) + int(
        fallback_analysis.get("latest_failed_for_other_reasons_and_drift_decreasing", 0)
    )
    non_positive = int(fallback_analysis.get("latest_failed_due_to_non_positive_median", 0))

    if eligible >= 4:
        justification = "often_structurally_indicated"
    elif eligible >= 1:
        justification = "conditionally_plausible"
    else:
        justification = "rarely_justified"

    primary = (
        "latest_visibility_breaks_while_cumulative_remains_visible"
        if (eligible or ineligible or non_positive) > 0
        else "latest_and_cumulative_are_generally_aligned"
    )

    most_affected = sorted(
        horizon_breakdown.items(),
        key=lambda item: (-_safe_dict(item[1]).get("fallback_eligible_count", 0), item[0]),
    )
    secondary = (
        f"most_affected_horizon={most_affected[0][0]}"
        if most_affected and _safe_dict(most_affected[0][1]).get("fallback_eligible_count", 0) > 0
        else "no_single_horizon_stands_out"
    )

    return {
        "primary_finding": primary,
        "secondary_finding": secondary,
        "fallback_justification": justification,
        "summary": (
            f"Eligible={eligible}, ineligible={ineligible}, "
            f"non_positive_median_cases={non_positive}. "
            f"Overall assessment: {justification}."
        ),
    }


def _diagnostic_label(
    *,
    latest_visible: bool,
    cumulative_visible: bool,
    latest_non_positive_median: bool,
    drift_direction: str,
) -> str:
    if latest_visible or not cumulative_visible:
        return "fallback_not_indicated"

    if latest_non_positive_median and drift_direction in {"increase", "flat"}:
        return "latest_failed_due_to_non_positive_median_and_drift_not_decreasing"
    if latest_non_positive_median and drift_direction == "decrease":
        return "latest_failed_due_to_non_positive_median_and_drift_decreasing"
    if latest_non_positive_median:
        return "latest_failed_due_to_non_positive_median"

    if drift_direction in {"increase", "flat"}:
        return "latest_failed_for_other_reasons_and_drift_not_decreasing"
    if drift_direction == "decrease":
        return "latest_failed_for_other_reasons_and_drift_decreasing"

    return "latest_failed_cumulative_visible"


def _top_rank_metrics(summary: dict[str, Any], horizon: str, ranking_key: str) -> dict[str, Any]:
    ranking = _safe_dict(
        _safe_dict(_safe_dict(summary.get("strategy_lab")).get("ranking")).get(horizon)
    ).get(ranking_key)

    if not isinstance(ranking, dict):
        return {}

    ranked_groups = ranking.get("ranked_groups")
    if not isinstance(ranked_groups, list) or not ranked_groups:
        return {}

    top_row = ranked_groups[0]
    if not isinstance(top_row, dict):
        return {}

    return _safe_dict(top_row.get("metrics"))


def _is_non_positive_median(metrics: dict[str, Any]) -> bool:
    median = _to_float(metrics.get("median_future_return_pct"))
    return median is not None and median <= 0


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


def _normalize_strength(*values: Any) -> str:
    for value in values:
        text = _normalize_group(value)
        if text != "n/a":
            return text
    return "insufficient_data"


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
    parser = argparse.ArgumentParser(
        description="Run latest-vs-cumulative fallback justification probe"
    )
    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=DEFAULT_COMPARISON_SUMMARY_PATH,
    )
    parser.add_argument(
        "--score-drift-summary",
        type=Path,
        default=DEFAULT_SCORE_DRIFT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--latest-summary",
        type=Path,
        default=DEFAULT_LATEST_SUMMARY_PATH,
    )
    parser.add_argument(
        "--cumulative-summary",
        type=Path,
        default=DEFAULT_CUMULATIVE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_latest_cumulative_fallback_probe(
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
