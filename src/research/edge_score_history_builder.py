from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

CATEGORIES = ("strategy", "symbol", "alignment_state")
DEFAULT_SOURCE_PREFERENCE = "n/a"
DEFAULT_STRENGTH_LABEL = "insufficient_data"
DEFAULT_STABILITY_LABEL = "insufficient_data"
INVALID_GROUP_VALUES = {None, "", "n/a", "None", "insufficient_data"}


def build_edge_score_history(input_path: Path, output_path: Path) -> dict[str, Any]:
    summary = _load_summary_json(input_path)
    generated_at = _resolve_generated_at(summary)

    records = [
        record
        for category, item in _iter_category_items(summary)
        if (record := _build_history_record(generated_at, category, item)) is not None
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_appended = _append_jsonl_records(output_path, records)

    return {
        "generated_at": generated_at,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_exists": input_path.exists(),
        "records_detected": len(records),
        "records_appended": records_appended,
    }


def _load_summary_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_generated_at(summary: dict[str, Any]) -> str:
    value = summary.get("generated_at")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return datetime.now(UTC).isoformat()


def _iter_category_items(summary: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    scores = summary.get("edge_stability_scores")
    if not isinstance(scores, dict):
        return

    for category in CATEGORIES:
        items = scores.get(category)
        if not isinstance(items, list):
            continue

        for item in items:
            if isinstance(item, dict):
                yield category, item


def _build_history_record(
    generated_at: str,
    category: str,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    group = _normalize_group(item.get("group"))
    if group is None:
        return None

    score = _coerce_float(item.get("score"))
    score_components = item.get("score_components")
    if not isinstance(score_components, dict):
        score_components = {}

    return {
        "generated_at": generated_at,
        "category": category,
        "group": group,
        "score": score,
        "source_preference": _normalize_text(
            item.get("source_preference"),
            default=DEFAULT_SOURCE_PREFERENCE,
        ),
        "latest_candidate_strength": _normalize_text(
            item.get("latest_candidate_strength"),
            default=DEFAULT_STRENGTH_LABEL,
        ),
        "cumulative_candidate_strength": _normalize_text(
            item.get("cumulative_candidate_strength"),
            default=DEFAULT_STRENGTH_LABEL,
        ),
        "latest_stability_label": _normalize_text(
            item.get("latest_stability_label"),
            default=DEFAULT_STABILITY_LABEL,
        ),
        "cumulative_stability_label": _normalize_text(
            item.get("cumulative_stability_label"),
            default=DEFAULT_STABILITY_LABEL,
        ),
        "latest_visible_horizons": _normalize_horizons(
            item.get("latest_visible_horizons"),
        ),
        "cumulative_visible_horizons": _normalize_horizons(
            item.get("cumulative_visible_horizons"),
        ),
        "candidate_strength_weight": _coerce_float(
            score_components.get("candidate_strength_weight"),
        ),
        "stability_label_weight": _coerce_float(
            score_components.get("stability_label_weight"),
        ),
        "horizon_bonus": _coerce_float(score_components.get("horizon_bonus")),
    }


def _append_jsonl_records(path: Path, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    appended = 0
    try:
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                appended += 1
    except OSError:
        return appended

    return appended


def _normalize_group(value: Any) -> str | None:
    if value in INVALID_GROUP_VALUES:
        return None

    normalized = str(value).strip()
    if normalized in INVALID_GROUP_VALUES:
        return None

    return normalized


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


def _default_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores"
        / "summary.json"
    )


def _default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "edge_scores_history.jsonl"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build append-only research history for edge stability score snapshots"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to edge stability score summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Path to append-only edge score history JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_edge_score_history(input_path=args.input, output_path=args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
