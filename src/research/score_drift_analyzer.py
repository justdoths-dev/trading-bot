from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CATEGORIES = ("strategy", "symbol", "alignment_state")
DRIFT_DIRECTIONS = ("increase", "decrease", "flat", "insufficient_history")
DEFAULT_SOURCE_PREFERENCE = "n/a"
DEFAULT_STRENGTH_LABEL = "insufficient_data"
DEFAULT_STABILITY_LABEL = "insufficient_data"
INVALID_GROUP_VALUES = {None, "", "n/a", "None", "insufficient_data"}


def build_score_drift_report(input_path: Path, output_dir: Path) -> dict[str, Any]:
    history_records = _load_history_records(input_path)
    grouped_records = _group_records(history_records)
    generated_at = datetime.now(UTC).isoformat()

    drift_items = [
        _analyze_group(category, group, records)
        for (category, group), records in grouped_records.items()
    ]
    drift_items.sort(key=lambda item: (str(item["category"]), str(item["group"])))

    summary = {
        "generated_at": generated_at,
        "input_path": str(input_path),
        "groups_analyzed": len(drift_items),
        "groups_with_sufficient_history": sum(
            1
            for item in drift_items
            if item["drift_direction"] != "insufficient_history"
        ),
        "drift_summary": _build_drift_counts(drift_items),
        "by_category": _build_category_summary(drift_items),
        "score_drift": drift_items,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_build_markdown(summary), encoding="utf-8")

    return summary


def _load_history_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = _parse_history_line(line)
                if record is None:
                    continue

                normalized = _normalize_history_record(record, line_number)
                if normalized is not None:
                    records.append(normalized)
    except OSError:
        return []

    return records


def _parse_history_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_history_record(
    record: dict[str, Any],
    line_number: int,
) -> dict[str, Any] | None:
    category = _normalize_category(record.get("category"))
    group = _normalize_group(record.get("group"))
    generated_at = _normalize_text(record.get("generated_at"), default="")

    if category is None or group is None or not generated_at:
        return None

    source_preference = _normalize_source_preference(record.get("source_preference"))
    selected_values = _select_source_values(record, source_preference)

    return {
        "generated_at": generated_at,
        "category": category,
        "group": group,
        "score": _coerce_float(record.get("score")),
        "source_preference": source_preference,
        "selected_candidate_strength": selected_values["candidate_strength"],
        "selected_stability_label": selected_values["stability_label"],
        "selected_visible_horizons": selected_values["visible_horizons"],
        "_sort_key": _record_sort_key(generated_at, line_number),
    }


def _group_records(
    history_records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for record in history_records:
        key = (str(record["category"]), str(record["group"]))
        grouped.setdefault(key, []).append(record)

    for records in grouped.values():
        records.sort(key=lambda item: item.get("_sort_key", (False, datetime.min, 0)))

    return grouped


def _analyze_group(
    category: str,
    group: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    latest = records[-1]
    previous = records[-2] if len(records) >= 2 else None

    if previous is None:
        return {
            "category": category,
            "group": group,
            "previous_generated_at": None,
            "latest_generated_at": latest["generated_at"],
            "previous_score": None,
            "latest_score": latest["score"],
            "score_delta": None,
            "drift_direction": "insufficient_history",
            "previous_source_preference": DEFAULT_SOURCE_PREFERENCE,
            "latest_source_preference": latest["source_preference"],
            "previous_selected_candidate_strength": DEFAULT_STRENGTH_LABEL,
            "latest_selected_candidate_strength": latest["selected_candidate_strength"],
            "previous_stability_label": DEFAULT_STABILITY_LABEL,
            "latest_stability_label": latest["selected_stability_label"],
            "stability_transition": (
                f"{DEFAULT_STABILITY_LABEL} -> {latest['selected_stability_label']}"
            ),
            "previous_horizons": [],
            "latest_horizons": latest["selected_visible_horizons"],
            "horizon_count_delta": None,
        }

    score_delta = round(float(latest["score"]) - float(previous["score"]), 6)
    drift_direction = _drift_direction(score_delta)
    previous_horizons = list(previous["selected_visible_horizons"])
    latest_horizons = list(latest["selected_visible_horizons"])

    return {
        "category": category,
        "group": group,
        "previous_generated_at": previous["generated_at"],
        "latest_generated_at": latest["generated_at"],
        "previous_score": previous["score"],
        "latest_score": latest["score"],
        "score_delta": score_delta,
        "drift_direction": drift_direction,
        "previous_source_preference": previous["source_preference"],
        "latest_source_preference": latest["source_preference"],
        "previous_selected_candidate_strength": previous["selected_candidate_strength"],
        "latest_selected_candidate_strength": latest["selected_candidate_strength"],
        "previous_stability_label": previous["selected_stability_label"],
        "latest_stability_label": latest["selected_stability_label"],
        "stability_transition": (
            f"{previous['selected_stability_label']} -> "
            f"{latest['selected_stability_label']}"
        ),
        "previous_horizons": previous_horizons,
        "latest_horizons": latest_horizons,
        "horizon_count_delta": len(latest_horizons) - len(previous_horizons),
    }


def _select_source_values(
    record: dict[str, Any],
    source_preference: str,
) -> dict[str, Any]:
    if source_preference == "latest":
        strength_key = "latest_candidate_strength"
        stability_key = "latest_stability_label"
        horizons_key = "latest_visible_horizons"
    else:
        strength_key = "cumulative_candidate_strength"
        stability_key = "cumulative_stability_label"
        horizons_key = "cumulative_visible_horizons"

    return {
        "candidate_strength": _normalize_text(
            record.get(strength_key),
            default=DEFAULT_STRENGTH_LABEL,
        ),
        "stability_label": _normalize_text(
            record.get(stability_key),
            default=DEFAULT_STABILITY_LABEL,
        ),
        "visible_horizons": _normalize_horizons(record.get(horizons_key)),
    }


def _record_sort_key(generated_at: str, line_number: int) -> tuple[bool, datetime, int]:
    # Invalid timestamps sort before valid ones so valid latest records remain at the end.
    parsed = _parse_generated_at(generated_at)
    if parsed is None:
        return (False, datetime.min.replace(tzinfo=UTC), line_number)
    return (True, parsed, line_number)


def _parse_generated_at(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _build_drift_counts(drift_items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        direction: sum(1 for item in drift_items if item["drift_direction"] == direction)
        for direction in DRIFT_DIRECTIONS
    }


def _build_category_summary(drift_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for category in CATEGORIES:
        category_items = [item for item in drift_items if item["category"] == category]
        summary[category] = {
            "groups_analyzed": len(category_items),
            "groups_with_sufficient_history": sum(
                1
                for item in category_items
                if item["drift_direction"] != "insufficient_history"
            ),
            "drift_summary": {
                direction: sum(
                    1 for item in category_items if item["drift_direction"] == direction
                )
                for direction in DRIFT_DIRECTIONS
            },
            "groups": category_items,
        }

    return summary


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Score Drift Analysis")
    lines.append("")
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append(f"Input path: {summary['input_path']}")
    lines.append(f"Groups analyzed: {summary['groups_analyzed']}")
    lines.append(
        "Groups with sufficient history: "
        f"{summary['groups_with_sufficient_history']}"
    )
    lines.append("")
    lines.append("## Drift Summary")
    lines.append("")

    drift_summary = summary["drift_summary"]
    for direction in DRIFT_DIRECTIONS:
        lines.append(f"- {direction}: {drift_summary.get(direction, 0)}")
    lines.append("")

    by_category = summary["by_category"]
    for category in CATEGORIES:
        category_summary = by_category.get(category, {})
        lines.append(f"## {category}")
        lines.append("")
        lines.append(f"- groups_analyzed: {category_summary.get('groups_analyzed', 0)}")
        lines.append(
            "- groups_with_sufficient_history: "
            f"{category_summary.get('groups_with_sufficient_history', 0)}"
        )

        category_drift_summary = category_summary.get("drift_summary", {})
        for direction in DRIFT_DIRECTIONS:
            lines.append(f"- {direction}: {category_drift_summary.get(direction, 0)}")
        lines.append("")

        category_items = category_summary.get("groups", [])
        if not category_items:
            lines.append("- no observational drift items available")
            lines.append("")
            continue

        for item in category_items:
            if item["drift_direction"] == "insufficient_history":
                lines.append(
                    f"- {item['group']}: insufficient_history, "
                    f"latest_score={item['latest_score']}, "
                    f"latest_stability={item['latest_stability_label']}, "
                    f"latest_source={item['latest_source_preference']}"
                )
                continue

            lines.append(
                f"- {item['group']}: drift={item['drift_direction']}, "
                f"score_delta={item['score_delta']}, "
                f"previous_score={item['previous_score']}, "
                f"latest_score={item['latest_score']}, "
                f"stability_transition={item['stability_transition']}, "
                f"horizon_count_delta={item['horizon_count_delta']}"
            )
        lines.append("")

    return "\n".join(lines)


def _normalize_category(value: Any) -> str | None:
    normalized = _normalize_text(value, default="")
    if normalized in CATEGORIES:
        return normalized
    return None


def _normalize_group(value: Any) -> str | None:
    if value in INVALID_GROUP_VALUES:
        return None

    normalized = str(value).strip()
    if normalized in INVALID_GROUP_VALUES:
        return None

    return normalized


def _normalize_source_preference(value: Any) -> str:
    normalized = _normalize_text(value, default=DEFAULT_SOURCE_PREFERENCE)
    if normalized in {"latest", "cumulative"}:
        return normalized
    return DEFAULT_SOURCE_PREFERENCE


def _normalize_text(value: Any, default: str) -> str:
    if value is None:
        return default

    normalized = str(value).strip()
    return normalized or default


def _normalize_horizons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        item_str = str(item).strip()
        if item_str and item_str not in normalized:
            normalized.append(item_str)

    return normalized


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _drift_direction(score_delta: float) -> str:
    if score_delta > 0:
        return "increase"
    if score_delta < 0:
        return "decrease"
    return "flat"


def _default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores_history.jsonl"
    )


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "score_drift"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a research-only score drift report from edge score history"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to edge score history JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory for score drift outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_score_drift_report(input_path=args.input, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
