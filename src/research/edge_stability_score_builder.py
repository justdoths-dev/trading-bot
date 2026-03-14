from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CATEGORIES = ("strategy", "symbol", "alignment_state")
HORIZONS = ("15m", "1h", "4h")
CANDIDATE_GROUP_FIELDS = {
    "strategy": ("latest_top_strategy_group", "cumulative_top_strategy_group"),
    "symbol": ("latest_top_symbol_group", "cumulative_top_symbol_group"),
    "alignment_state": (
        "latest_top_alignment_state_group",
        "cumulative_top_alignment_state_group",
    ),
}
STRENGTH_WEIGHTS = {
    "insufficient_data": 0.0,
    "weak": 1.0,
    "moderate": 2.0,
    "strong": 3.0,
}
STABILITY_WEIGHTS = {
    "insufficient_data": 0.0,
    "unstable": 0.5,
    "single_horizon_only": 1.0,
    "multi_horizon_confirmed": 2.0,
}
INVALID_GROUP_VALUES = {None, "", "n/a", "None", "insufficient_data"}


def build_edge_stability_scores(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    comparison_report = _read_comparison_report(input_path)
    scores = {
        category: _build_category_scores(comparison_report, category)
        for category in CATEGORIES
    }

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "edge_stability_scores": scores,
        "score_summary": {
            "top_strategy": _top_item(scores["strategy"]),
            "top_symbol": _top_item(scores["symbol"]),
            "top_alignment_state": _top_item(scores["alignment_state"]),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_build_markdown(summary), encoding="utf-8")

    return summary


def _read_comparison_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _build_category_scores(
    comparison_report: dict[str, Any],
    category: str,
) -> list[dict[str, Any]]:
    edge_candidates = comparison_report.get("edge_candidates_comparison", {}) or {}
    edge_stability = (
        (comparison_report.get("edge_stability_comparison", {}) or {}).get(
            category,
            {},
        )
        or {}
    )

    latest_group_field, cumulative_group_field = CANDIDATE_GROUP_FIELDS[category]
    groups: set[str] = set()

    for horizon in HORIZONS:
        horizon_data = edge_candidates.get(horizon, {}) or {}
        for key in (latest_group_field, cumulative_group_field):
            group = _normalize_group(horizon_data.get(key), category=category)
            if group is not None:
                groups.add(group)

    for key in ("latest_group", "cumulative_group"):
        group = _normalize_group(edge_stability.get(key), category=category)
        if group is not None:
            groups.add(group)

    scored_items = [
        _score_group(
            group=group,
            category=category,
            edge_candidates=edge_candidates,
            edge_stability=edge_stability,
            latest_group_field=latest_group_field,
            cumulative_group_field=cumulative_group_field,
        )
        for group in sorted(groups)
    ]
    scored_items.sort(key=lambda item: (-float(item["score"]), str(item["group"])))
    return scored_items


def _score_group(
    group: str,
    category: str,
    edge_candidates: dict[str, Any],
    edge_stability: dict[str, Any],
    latest_group_field: str,
    cumulative_group_field: str,
) -> dict[str, Any]:
    latest_strength_labels: list[str] = []
    cumulative_strength_labels: list[str] = []

    for horizon in HORIZONS:
        horizon_data = edge_candidates.get(horizon, {}) or {}
        latest_group = _normalize_group(
            horizon_data.get(latest_group_field),
            category=category,
        )
        cumulative_group = _normalize_group(
            horizon_data.get(cumulative_group_field),
            category=category,
        )

        if latest_group == group:
            latest_strength_labels.append(
                str(horizon_data.get("latest_candidate_strength", "insufficient_data"))
            )

        if cumulative_group == group:
            cumulative_strength_labels.append(
                str(
                    horizon_data.get(
                        "cumulative_candidate_strength",
                        "insufficient_data",
                    )
                )
            )

    latest_candidate_strength = _highest_strength(latest_strength_labels)
    cumulative_candidate_strength = _highest_strength(cumulative_strength_labels)

    latest_group_match = (
        _normalize_group(edge_stability.get("latest_group"), category=category) == group
    )
    cumulative_group_match = (
        _normalize_group(edge_stability.get("cumulative_group"), category=category)
        == group
    )

    latest_stability_label = (
        str(edge_stability.get("latest_stability_label", "insufficient_data"))
        if latest_group_match
        else "insufficient_data"
    )
    cumulative_stability_label = (
        str(edge_stability.get("cumulative_stability_label", "insufficient_data"))
        if cumulative_group_match
        else "insufficient_data"
    )

    latest_stability_visible_horizons = (
        _normalize_horizons(edge_stability.get("latest_visible_horizons"))
        if latest_group_match
        else []
    )
    cumulative_stability_visible_horizons = (
        _normalize_horizons(edge_stability.get("cumulative_visible_horizons"))
        if cumulative_group_match
        else []
    )

    latest_horizon_count = len(latest_stability_visible_horizons)
    cumulative_horizon_count = len(cumulative_stability_visible_horizons)

    latest_subscore = _source_subscore(
        candidate_strength=latest_candidate_strength,
        stability_label=latest_stability_label,
        horizon_count=latest_horizon_count,
    )
    cumulative_subscore = _source_subscore(
        candidate_strength=cumulative_candidate_strength,
        stability_label=cumulative_stability_label,
        horizon_count=cumulative_horizon_count,
    )

    source_preference = _select_source_preference(
        latest_candidate_strength=latest_candidate_strength,
        cumulative_candidate_strength=cumulative_candidate_strength,
        latest_stability_label=latest_stability_label,
        cumulative_stability_label=cumulative_stability_label,
        latest_horizon_count=latest_horizon_count,
        cumulative_horizon_count=cumulative_horizon_count,
        latest_subscore=latest_subscore,
        cumulative_subscore=cumulative_subscore,
    )

    if source_preference == "latest":
        selected_strength = latest_candidate_strength
        selected_stability = latest_stability_label
        selected_horizon_count = latest_horizon_count
    else:
        selected_strength = cumulative_candidate_strength
        selected_stability = cumulative_stability_label
        selected_horizon_count = cumulative_horizon_count

    candidate_strength_weight = STRENGTH_WEIGHTS[selected_strength]
    stability_label_weight = STABILITY_WEIGHTS[selected_stability]
    horizon_bonus = _horizon_bonus(selected_horizon_count)
    score = candidate_strength_weight + stability_label_weight + horizon_bonus

    return {
        "group": group,
        "score": score,
        "latest_stability_label": latest_stability_label,
        "cumulative_stability_label": cumulative_stability_label,
        "latest_visible_horizons": latest_stability_visible_horizons,
        "cumulative_visible_horizons": cumulative_stability_visible_horizons,
        "latest_candidate_strength": latest_candidate_strength,
        "cumulative_candidate_strength": cumulative_candidate_strength,
        "source_preference": source_preference,
        "score_components": {
            "candidate_strength_weight": candidate_strength_weight,
            "stability_label_weight": stability_label_weight,
            "horizon_bonus": horizon_bonus,
        },
    }


def _select_source_preference(
    latest_candidate_strength: str,
    cumulative_candidate_strength: str,
    latest_stability_label: str,
    cumulative_stability_label: str,
    latest_horizon_count: int,
    cumulative_horizon_count: int,
    latest_subscore: float,
    cumulative_subscore: float,
) -> str:
    if latest_subscore > cumulative_subscore:
        return "latest"
    if cumulative_subscore > latest_subscore:
        return "cumulative"

    exact_component_tie = (
        latest_candidate_strength == cumulative_candidate_strength
        and latest_stability_label == cumulative_stability_label
        and latest_horizon_count == cumulative_horizon_count
    )

    if exact_component_tie:
        return "cumulative"

    return "latest"


def _source_subscore(
    candidate_strength: str,
    stability_label: str,
    horizon_count: int,
) -> float:
    return (
        STRENGTH_WEIGHTS[candidate_strength]
        + STABILITY_WEIGHTS[stability_label]
        + _horizon_bonus(horizon_count)
    )


def _highest_strength(labels: list[str]) -> str:
    best_label = "insufficient_data"
    best_score = -1.0

    for label in labels:
        score = STRENGTH_WEIGHTS.get(label, 0.0)
        if score > best_score:
            best_label = label
            best_score = score

    return best_label if labels else "insufficient_data"


def _horizon_bonus(horizon_count: int) -> float:
    if horizon_count <= 0:
        return 0.0
    if horizon_count == 1:
        return 0.5
    if horizon_count == 2:
        return 1.0
    return 1.5


def _normalize_group(value: Any, category: str) -> str | None:
    if value in INVALID_GROUP_VALUES:
        return None

    normalized = str(value).strip()
    if normalized in INVALID_GROUP_VALUES:
        return None

    if category == "symbol":
        return normalized.upper()

    return normalized


def _normalize_horizons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        item_str = str(item)
        if item_str in HORIZONS and item_str not in normalized:
            normalized.append(item_str)
    return normalized


def _top_item(scored_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_items:
        return {
            "group": "n/a",
            "score": 0.0,
            "source_preference": "n/a",
        }

    first = scored_items[0]
    return {
        "group": str(first.get("group", "n/a")),
        "score": float(first.get("score", 0.0)),
        "source_preference": str(first.get("source_preference", "n/a")),
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Edge Stability Scores")
    lines.append("")
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append(f"Input path: {summary['input_path']}")
    lines.append("")

    score_summary = summary["score_summary"]
    lines.append("## Score Summary")
    lines.append("")
    lines.append(
        "- top_strategy: "
        f"{score_summary['top_strategy']['group']} "
        f"(score={score_summary['top_strategy']['score']}, "
        f"source={score_summary['top_strategy']['source_preference']})"
    )
    lines.append(
        "- top_symbol: "
        f"{score_summary['top_symbol']['group']} "
        f"(score={score_summary['top_symbol']['score']}, "
        f"source={score_summary['top_symbol']['source_preference']})"
    )
    lines.append(
        "- top_alignment_state: "
        f"{score_summary['top_alignment_state']['group']} "
        f"(score={score_summary['top_alignment_state']['score']}, "
        f"source={score_summary['top_alignment_state']['source_preference']})"
    )
    lines.append("")

    for category in CATEGORIES:
        lines.append(f"## {category}")
        lines.append("")
        scored_items = summary["edge_stability_scores"].get(category, [])
        if not scored_items:
            lines.append("- no scored items available")
            lines.append("")
            continue

        for item in scored_items:
            lines.append(
                f"- {item['group']}: "
                f"score={item['score']}, "
                f"source_preference={item['source_preference']}, "
                f"latest_stability={item['latest_stability_label']}, "
                f"cumulative_stability={item['cumulative_stability_label']}"
            )
        lines.append("")

    return "\n".join(lines)


def _default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "comparison"
        / "summary.json"
    )


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build research-only edge stability scores from comparison report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to comparison summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory for edge stability score outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_edge_stability_scores(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()