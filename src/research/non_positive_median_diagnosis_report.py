from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_JSON_OUTPUT = DEFAULT_INPUT_DIR / "non_positive_median_diagnosis_summary.json"
DEFAULT_MD_OUTPUT = DEFAULT_INPUT_DIR / "non_positive_median_diagnosis_summary.md"

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_CATEGORIES = ("strategy", "symbol", "alignment_state")


@dataclass(frozen=True)
class NormalizedRankingRow:
    source: str
    horizon: str
    category: str
    group: str
    rank: int | None
    median_future_return_pct: float | None
    avg_future_return_pct: float | None
    positive_rate_pct: float | None
    flat_rate_pct: float | None
    labeled_count: int | None
    raw: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _infer_source_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "latest" in stem:
        return "latest"
    if "cumulative" in stem:
        return "cumulative"
    return "unknown"


def _extract_source(raw: dict[str, Any], path: Path) -> str:
    explicit = raw.get("source") or raw.get("window_type") or raw.get("profile_type")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"latest", "cumulative"}:
            return normalized
    return _infer_source_from_path(path)


def _extract_horizon(raw: dict[str, Any], context: dict[str, Any]) -> str | None:
    candidates = (
        raw.get("horizon"),
        raw.get("time_horizon"),
        raw.get("label_horizon"),
        raw.get("future_horizon"),
        context.get("horizon"),
    )
    for value in candidates:
        if isinstance(value, str) and value in TARGET_HORIZONS:
            return value
    return None


def _extract_category(raw: dict[str, Any], context: dict[str, Any]) -> str | None:
    candidates = (
        raw.get("category"),
        raw.get("grouping_category"),
        raw.get("dimension"),
        context.get("category"),
    )
    for value in candidates:
        if isinstance(value, str) and value in TARGET_CATEGORIES:
            return value
    return None


def _extract_group(raw: dict[str, Any]) -> str:
    group = (
        raw.get("group")
        or raw.get("group_name")
        or raw.get("candidate_group")
        or raw.get("bucket")
        or raw.get("name")
        or raw.get("label")
    )
    if group is None:
        return "unknown"
    return str(group)


def _extract_rank(raw: dict[str, Any]) -> int | None:
    return _safe_int(raw.get("rank") or raw.get("position") or raw.get("top_rank"))


def _extract_metric(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in raw:
            value = _safe_float(raw.get(key))
            if value is not None:
                return value
    return None


def _extract_labeled_count(raw: dict[str, Any]) -> int | None:
    for key in ("labeled_count", "label_count", "count", "sample_size", "observations"):
        if key in raw:
            value = _safe_int(raw.get(key))
            if value is not None:
                return value
    return None


def _looks_like_ranking_row(raw: dict[str, Any], context: dict[str, Any]) -> bool:
    horizon = _extract_horizon(raw, context)
    category = _extract_category(raw, context)
    if horizon is None or category is None:
        return False

    metric_keys = {
        "median_future_return_pct",
        "avg_future_return_pct",
        "positive_rate_pct",
        "flat_rate_pct",
        "labeled_count",
        "label_count",
        "sample_size",
        "count",
        "rank",
        "position",
    }
    return any(key in raw for key in metric_keys)


def _walk_rows(node: Any, context: dict[str, Any] | None = None) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    if context is None:
        context = {}

    if isinstance(node, dict):
        next_context = dict(context)

        if isinstance(node.get("horizon"), str) and node["horizon"] in TARGET_HORIZONS:
            next_context["horizon"] = node["horizon"]
        if isinstance(node.get("category"), str) and node["category"] in TARGET_CATEGORIES:
            next_context["category"] = node["category"]
        if isinstance(node.get("grouping_category"), str) and node["grouping_category"] in TARGET_CATEGORIES:
            next_context["category"] = node["grouping_category"]

        if _looks_like_ranking_row(node, next_context):
            yield node, next_context

        for value in node.values():
            yield from _walk_rows(value, next_context)

    elif isinstance(node, list):
        for item in node:
            yield from _walk_rows(item, context)


def _normalize_row(raw: dict[str, Any], context: dict[str, Any], path: Path) -> NormalizedRankingRow | None:
    horizon = _extract_horizon(raw, context)
    category = _extract_category(raw, context)
    if horizon is None or category is None:
        return None

    return NormalizedRankingRow(
        source=_extract_source(raw, path),
        horizon=horizon,
        category=category,
        group=_extract_group(raw),
        rank=_extract_rank(raw),
        median_future_return_pct=_extract_metric(
            raw,
            "median_future_return_pct",
            "median_return_pct",
            "median_pct",
        ),
        avg_future_return_pct=_extract_metric(
            raw,
            "avg_future_return_pct",
            "mean_future_return_pct",
            "avg_return_pct",
            "mean_return_pct",
        ),
        positive_rate_pct=_extract_metric(raw, "positive_rate_pct", "positive_rate"),
        flat_rate_pct=_extract_metric(raw, "flat_rate_pct", "flat_rate"),
        labeled_count=_extract_labeled_count(raw),
        raw=raw,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_normalized_rows(input_dir: Path) -> list[NormalizedRankingRow]:
    rows: list[NormalizedRankingRow] = []
    if not input_dir.exists():
        return rows

    for path in sorted(input_dir.glob("*.json")):
        if path.name in {
            DEFAULT_JSON_OUTPUT.name,
            DEFAULT_MD_OUTPUT.name,
        }:
            continue

        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue

        for raw, context in _walk_rows(payload):
            normalized = _normalize_row(raw, context, path)
            if normalized is not None:
                rows.append(normalized)

    return rows


def _is_non_positive_median(row: NormalizedRankingRow) -> bool:
    return row.median_future_return_pct is not None and row.median_future_return_pct <= 0.0


def _is_positive_median(row: NormalizedRankingRow) -> bool:
    return row.median_future_return_pct is not None and row.median_future_return_pct > 0.0


def _rank_within(row: NormalizedRankingRow, top_n: int) -> bool:
    return row.rank is not None and row.rank <= top_n


def _flat_rate_dominant(row: NormalizedRankingRow) -> bool:
    flat_rate = row.flat_rate_pct
    positive_rate = row.positive_rate_pct

    if flat_rate is None:
        return False
    if positive_rate is None:
        return flat_rate >= 50.0

    negative_rate = max(0.0, 100.0 - positive_rate - (row.flat_rate_pct or 0.0))
    return flat_rate >= positive_rate and flat_rate >= negative_rate


def _has_mean_positive_conflict(row: NormalizedRankingRow) -> bool:
    return _is_non_positive_median(row) and (row.avg_future_return_pct or 0.0) > 0.0


def _has_positive_rate_conflict(row: NormalizedRankingRow, threshold: float = 50.0) -> bool:
    return _is_non_positive_median(row) and (row.positive_rate_pct or 0.0) >= threshold


def _has_sufficient_labels(row: NormalizedRankingRow, min_count: int = 5) -> bool:
    return _is_non_positive_median(row) and (row.labeled_count or 0) >= min_count


def _rows_by_source(rows: Iterable[NormalizedRankingRow], source: str) -> list[NormalizedRankingRow]:
    return [row for row in rows if row.source == source]


def _build_latest_vs_cumulative_pairs(
    rows: Iterable[NormalizedRankingRow],
) -> list[dict[str, Any]]:
    latest_index: dict[tuple[str, str, str, int | None], NormalizedRankingRow] = {}
    cumulative_index: dict[tuple[str, str, str, int | None], NormalizedRankingRow] = {}

    for row in rows:
        key = (row.horizon, row.category, row.group, row.rank)
        if row.source == "latest":
            latest_index[key] = row
        elif row.source == "cumulative":
            cumulative_index[key] = row

    pairs: list[dict[str, Any]] = []
    for key, latest_row in latest_index.items():
        cumulative_row = cumulative_index.get(key)
        if cumulative_row is None:
            continue

        latest_non_positive = _is_non_positive_median(latest_row)
        cumulative_positive = _is_positive_median(cumulative_row)

        pairs.append(
            {
                "horizon": latest_row.horizon,
                "category": latest_row.category,
                "group": latest_row.group,
                "rank": latest_row.rank,
                "latest_median_future_return_pct": latest_row.median_future_return_pct,
                "cumulative_median_future_return_pct": cumulative_row.median_future_return_pct,
                "latest_non_positive": latest_non_positive,
                "cumulative_positive": cumulative_positive,
                "latest_window_noisy_but_cumulative_healthy": latest_non_positive and cumulative_positive,
            }
        )

    return pairs


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _breakdown_by_horizon_and_category(rows: list[NormalizedRankingRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        horizon_rows = [row for row in rows if row.horizon == horizon]
        horizon_non_positive = sum(1 for row in horizon_rows if _is_non_positive_median(row))
        horizon_positive = sum(1 for row in horizon_rows if _is_positive_median(row))

        category_breakdown: dict[str, Any] = {}
        for category in TARGET_CATEGORIES:
            category_rows = [row for row in horizon_rows if row.category == category]
            non_positive_count = sum(1 for row in category_rows if _is_non_positive_median(row))
            positive_count = sum(1 for row in category_rows if _is_positive_median(row))
            category_breakdown[category] = {
                "evaluated_rows": len(category_rows),
                "non_positive_median_count": non_positive_count,
                "positive_median_count": positive_count,
                "non_positive_median_ratio": _ratio(non_positive_count, len(category_rows)),
            }

        result[horizon] = {
            "evaluated_rows": len(horizon_rows),
            "non_positive_median_count": horizon_non_positive,
            "positive_median_count": horizon_positive,
            "non_positive_median_ratio": _ratio(horizon_non_positive, len(horizon_rows)),
            "by_category": category_breakdown,
        }
    return result


def _breakdown_by_category(rows: list[NormalizedRankingRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in TARGET_CATEGORIES:
        category_rows = [row for row in rows if row.category == category]
        non_positive_count = sum(1 for row in category_rows if _is_non_positive_median(row))
        positive_count = sum(1 for row in category_rows if _is_positive_median(row))
        result[category] = {
            "evaluated_rows": len(category_rows),
            "non_positive_median_count": non_positive_count,
            "positive_median_count": positive_count,
            "non_positive_median_ratio": _ratio(non_positive_count, len(category_rows)),
        }
    return result


def _build_rank_scope_breakdown(rows: list[NormalizedRankingRow], top_n: int = 3) -> dict[str, Any]:
    top1_rows = [row for row in rows if row.rank == 1]
    topn_rows = [row for row in rows if _rank_within(row, top_n)]
    beyond_top1_rows = [row for row in topn_rows if row.rank is not None and row.rank > 1]

    top1_non_positive = sum(1 for row in top1_rows if _is_non_positive_median(row))
    topn_non_positive = sum(1 for row in topn_rows if _is_non_positive_median(row))
    beyond_top1_non_positive = sum(1 for row in beyond_top1_rows if _is_non_positive_median(row))

    return {
        "top1": {
            "evaluated_rows": len(top1_rows),
            "non_positive_median_count": top1_non_positive,
            "non_positive_median_ratio": _ratio(top1_non_positive, len(top1_rows)),
        },
        f"top{top_n}": {
            "evaluated_rows": len(topn_rows),
            "non_positive_median_count": topn_non_positive,
            "non_positive_median_ratio": _ratio(topn_non_positive, len(topn_rows)),
        },
        "beyond_top1_within_topn": {
            "evaluated_rows": len(beyond_top1_rows),
            "non_positive_median_count": beyond_top1_non_positive,
            "non_positive_median_ratio": _ratio(beyond_top1_non_positive, len(beyond_top1_rows)),
        },
    }


def _build_metric_interaction_breakdown(rows: list[NormalizedRankingRow]) -> dict[str, Any]:
    non_positive_rows = [row for row in rows if _is_non_positive_median(row)]
    mean_positive_conflict = [row for row in non_positive_rows if _has_mean_positive_conflict(row)]
    positive_rate_conflict = [row for row in non_positive_rows if _has_positive_rate_conflict(row)]
    flat_dominant = [row for row in non_positive_rows if _flat_rate_dominant(row)]
    sufficient_labels = [row for row in non_positive_rows if _has_sufficient_labels(row)]

    return {
        "non_positive_median_rows": len(non_positive_rows),
        "median_le_zero_and_avg_gt_zero_count": len(mean_positive_conflict),
        "median_le_zero_and_avg_gt_zero_ratio": _ratio(len(mean_positive_conflict), len(non_positive_rows)),
        "median_le_zero_and_positive_rate_ge_50_count": len(positive_rate_conflict),
        "median_le_zero_and_positive_rate_ge_50_ratio": _ratio(len(positive_rate_conflict), len(non_positive_rows)),
        "median_le_zero_and_flat_rate_dominant_count": len(flat_dominant),
        "median_le_zero_and_flat_rate_dominant_ratio": _ratio(len(flat_dominant), len(non_positive_rows)),
        "median_le_zero_and_labeled_count_sufficient_count": len(sufficient_labels),
        "median_le_zero_and_labeled_count_sufficient_ratio": _ratio(len(sufficient_labels), len(non_positive_rows)),
    }


def _build_latest_vs_cumulative_summary(rows: list[NormalizedRankingRow]) -> dict[str, Any]:
    pairs = _build_latest_vs_cumulative_pairs(rows)
    noisy_but_healthy = [pair for pair in pairs if pair["latest_window_noisy_but_cumulative_healthy"]]

    horizon_breakdown: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        horizon_pairs = [pair for pair in noisy_but_healthy if pair["horizon"] == horizon]
        horizon_breakdown[horizon] = {
            "latest_non_positive_while_cumulative_positive_count": len(horizon_pairs),
        }

    category_breakdown: dict[str, Any] = {}
    for category in TARGET_CATEGORIES:
        category_pairs = [pair for pair in noisy_but_healthy if pair["category"] == category]
        category_breakdown[category] = {
            "latest_non_positive_while_cumulative_positive_count": len(category_pairs),
        }

    return {
        "pair_count": len(pairs),
        "latest_non_positive_while_cumulative_positive_count": len(noisy_but_healthy),
        "latest_non_positive_while_cumulative_positive_ratio": _ratio(len(noisy_but_healthy), len(pairs)),
        "by_horizon": horizon_breakdown,
        "by_category": category_breakdown,
        "examples": noisy_but_healthy[:10],
    }


def _representative_examples(rows: list[NormalizedRankingRow], limit: int = 10) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _is_non_positive_median(row)]
    candidates.sort(
        key=lambda row: (
            row.rank if row.rank is not None else 9999,
            row.median_future_return_pct if row.median_future_return_pct is not None else 9999.0,
        )
    )

    examples: list[dict[str, Any]] = []
    for row in candidates[:limit]:
        examples.append(
            {
                "source": row.source,
                "horizon": row.horizon,
                "category": row.category,
                "group": row.group,
                "rank": row.rank,
                "median_future_return_pct": row.median_future_return_pct,
                "avg_future_return_pct": row.avg_future_return_pct,
                "positive_rate_pct": row.positive_rate_pct,
                "flat_rate_pct": row.flat_rate_pct,
                "labeled_count": row.labeled_count,
            }
        )
    return examples


def _worst_horizon(horizon_breakdown: dict[str, Any]) -> str | None:
    ordered = sorted(
        TARGET_HORIZONS,
        key=lambda horizon: (
            horizon_breakdown[horizon]["non_positive_median_ratio"],
            horizon_breakdown[horizon]["non_positive_median_count"],
        ),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _worst_category(category_breakdown: dict[str, Any]) -> str | None:
    ordered = sorted(
        TARGET_CATEGORIES,
        key=lambda category: (
            category_breakdown[category]["non_positive_median_ratio"],
            category_breakdown[category]["non_positive_median_count"],
        ),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _build_final_diagnosis(
    overall: dict[str, Any],
    rank_scope: dict[str, Any],
    metric_interactions: dict[str, Any],
    latest_vs_cumulative: dict[str, Any],
    horizon_breakdown: dict[str, Any],
    category_breakdown: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []

    top1_ratio = rank_scope["top1"]["non_positive_median_ratio"]
    top3_ratio = rank_scope["top3"]["non_positive_median_ratio"]
    beyond_top1_ratio = rank_scope["beyond_top1_within_topn"]["non_positive_median_ratio"]

    flat_dominant_ratio = metric_interactions["median_le_zero_and_flat_rate_dominant_ratio"]
    mean_positive_conflict_ratio = metric_interactions["median_le_zero_and_avg_gt_zero_ratio"]
    latest_noise_ratio = latest_vs_cumulative["latest_non_positive_while_cumulative_positive_ratio"]

    if top1_ratio >= 0.5 and beyond_top1_ratio < 0.35:
        labels.append("non_positive_median_is_top_rank_specific")
    if top1_ratio >= 0.5 and top3_ratio >= 0.5:
        labels.append("non_positive_median_is_broad_across_rankings")
    if flat_dominant_ratio >= 0.4:
        labels.append("flat_heavy_distribution_is_primary_median_suppressor")
    if mean_positive_conflict_ratio > 0.0:
        labels.append("mean_positive_but_median_non_positive_conflict_exists")
    if latest_noise_ratio >= 0.3:
        labels.append("latest_window_noise_dominates_median_signal")

    if not labels:
        labels.append("non_positive_median_requires_additional_observation")

    worst_horizon = _worst_horizon(horizon_breakdown)
    worst_category = _worst_category(category_breakdown)

    primary_finding = labels[0]
    secondary_finding = worst_horizon or "unknown_horizon"

    summary = (
        f"evaluated_rows={overall['total_evaluated_rows']}, "
        f"non_positive_median_count={overall['non_positive_median_count']}, "
        f"top1_ratio={top1_ratio}, "
        f"top3_ratio={top3_ratio}, "
        f"worst_horizon={worst_horizon}, "
        f"worst_category={worst_category}."
    )

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "worst_horizon": worst_horizon,
        "worst_category": worst_category,
        "diagnosis_labels": labels,
        "summary": summary,
    }


def build_non_positive_median_diagnosis_report(
    input_dir: Path,
) -> dict[str, Any]:
    rows = load_normalized_rows(input_dir)
    evaluated_rows = [row for row in rows if row.source == "latest"]

    overall = {
        "total_normalized_rows": len(rows),
        "total_evaluated_rows": len(evaluated_rows),
        "non_positive_median_count": sum(1 for row in evaluated_rows if _is_non_positive_median(row)),
        "positive_median_count": sum(1 for row in evaluated_rows if _is_positive_median(row)),
    }
    overall["non_positive_median_ratio"] = _ratio(
        overall["non_positive_median_count"],
        overall["total_evaluated_rows"],
    )

    horizon_breakdown = _breakdown_by_horizon_and_category(evaluated_rows)
    category_breakdown = _breakdown_by_category(evaluated_rows)
    rank_scope = _build_rank_scope_breakdown(evaluated_rows, top_n=3)
    metric_interactions = _build_metric_interaction_breakdown(evaluated_rows)
    latest_vs_cumulative = _build_latest_vs_cumulative_summary(rows)
    representative_examples = _representative_examples(evaluated_rows, limit=10)
    final_diagnosis = _build_final_diagnosis(
        overall=overall,
        rank_scope=rank_scope,
        metric_interactions=metric_interactions,
        latest_vs_cumulative=latest_vs_cumulative,
        horizon_breakdown=horizon_breakdown,
        category_breakdown=category_breakdown,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_dir": str(input_dir),
        },
        "overall_median_blocker_overview": overall,
        "rank_scope_breakdown": rank_scope,
        "horizon_breakdown": horizon_breakdown,
        "category_breakdown": category_breakdown,
        "metric_interaction_breakdown": metric_interactions,
        "latest_vs_cumulative_summary": latest_vs_cumulative,
        "representative_examples": representative_examples,
        "final_diagnosis": final_diagnosis,
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def build_non_positive_median_diagnosis_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall_median_blocker_overview"]
    rank_scope = summary["rank_scope_breakdown"]
    metric_interactions = summary["metric_interaction_breakdown"]
    final_diagnosis = summary["final_diagnosis"]

    lines: list[str] = []
    lines.append("Non-Positive Median Diagnosis")
    lines.append(f"Generated: {summary['metadata']['generated_at']}")
    lines.append("")
    lines.append("Overall Median Blocker Overview")
    lines.append(f"- total_evaluated_rows: {overall['total_evaluated_rows']}")
    lines.append(f"- non_positive_median_count: {overall['non_positive_median_count']}")
    lines.append(f"- positive_median_count: {overall['positive_median_count']}")
    lines.append(f"- non_positive_median_ratio: {overall['non_positive_median_ratio']}")
    lines.append("")
    lines.append("Rank Scope Breakdown")
    lines.append(
        f"- top1: count={rank_scope['top1']['non_positive_median_count']}, "
        f"ratio={rank_scope['top1']['non_positive_median_ratio']}"
    )
    lines.append(
        f"- top3: count={rank_scope['top3']['non_positive_median_count']}, "
        f"ratio={rank_scope['top3']['non_positive_median_ratio']}"
    )
    lines.append(
        f"- beyond_top1_within_top3: count={rank_scope['beyond_top1_within_topn']['non_positive_median_count']}, "
        f"ratio={rank_scope['beyond_top1_within_topn']['non_positive_median_ratio']}"
    )
    lines.append("")
    lines.append("Horizon Breakdown")
    for horizon, payload in summary["horizon_breakdown"].items():
        lines.append(
            f"- {horizon}: non_positive={payload['non_positive_median_count']}, "
            f"positive={payload['positive_median_count']}, "
            f"ratio={payload['non_positive_median_ratio']}"
        )
        for category, category_payload in payload["by_category"].items():
            lines.append(
                f"  - {category}: non_positive={category_payload['non_positive_median_count']}, "
                f"positive={category_payload['positive_median_count']}, "
                f"ratio={category_payload['non_positive_median_ratio']}"
            )
    lines.append("")
    lines.append("Category Breakdown")
    for category, payload in summary["category_breakdown"].items():
        lines.append(
            f"- {category}: non_positive={payload['non_positive_median_count']}, "
            f"positive={payload['positive_median_count']}, "
            f"ratio={payload['non_positive_median_ratio']}"
        )
    lines.append("")
    lines.append("Metric Interaction Breakdown")
    lines.append(
        f"- median <= 0 and avg > 0: {metric_interactions['median_le_zero_and_avg_gt_zero_count']} "
        f"(ratio={metric_interactions['median_le_zero_and_avg_gt_zero_ratio']})"
    )
    lines.append(
        f"- median <= 0 and positive_rate >= 50: {metric_interactions['median_le_zero_and_positive_rate_ge_50_count']} "
        f"(ratio={metric_interactions['median_le_zero_and_positive_rate_ge_50_ratio']})"
    )
    lines.append(
        f"- median <= 0 and flat_rate dominant: {metric_interactions['median_le_zero_and_flat_rate_dominant_count']} "
        f"(ratio={metric_interactions['median_le_zero_and_flat_rate_dominant_ratio']})"
    )
    lines.append(
        f"- median <= 0 and labeled_count sufficient: {metric_interactions['median_le_zero_and_labeled_count_sufficient_count']} "
        f"(ratio={metric_interactions['median_le_zero_and_labeled_count_sufficient_ratio']})"
    )
    lines.append("")
    lines.append("Latest vs Cumulative Summary")
    latest_vs_cumulative = summary["latest_vs_cumulative_summary"]
    lines.append(
        f"- pair_count: {latest_vs_cumulative['pair_count']}"
    )
    lines.append(
        f"- latest_non_positive_while_cumulative_positive_count: "
        f"{latest_vs_cumulative['latest_non_positive_while_cumulative_positive_count']}"
    )
    lines.append(
        f"- latest_non_positive_while_cumulative_positive_ratio: "
        f"{latest_vs_cumulative['latest_non_positive_while_cumulative_positive_ratio']}"
    )
    for horizon, payload in latest_vs_cumulative["by_horizon"].items():
        lines.append(
            f"  - {horizon}: latest_non_positive_while_cumulative_positive_count="
            f"{payload['latest_non_positive_while_cumulative_positive_count']}"
        )
    for category, payload in latest_vs_cumulative["by_category"].items():
        lines.append(
            f"  - {category}: latest_non_positive_while_cumulative_positive_count="
            f"{payload['latest_non_positive_while_cumulative_positive_count']}"
        )
    lines.append("")
    lines.append("Representative Examples")
    if summary["representative_examples"]:
        for example in summary["representative_examples"]:
            lines.append(
                "- "
                f"source={example['source']}, "
                f"horizon={example['horizon']}, "
                f"category={example['category']}, "
                f"group={example['group']}, "
                f"rank={example['rank']}, "
                f"median={_format_pct(example['median_future_return_pct'])}, "
                f"avg={_format_pct(example['avg_future_return_pct'])}, "
                f"positive_rate={_format_pct(example['positive_rate_pct'])}, "
                f"flat_rate={_format_pct(example['flat_rate_pct'])}, "
                f"labeled_count={example['labeled_count']}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Final Diagnosis")
    lines.append(f"- primary_finding: {final_diagnosis['primary_finding']}")
    lines.append(f"- secondary_finding: {final_diagnosis['secondary_finding']}")
    lines.append(f"- worst_horizon: {final_diagnosis['worst_horizon']}")
    lines.append(f"- worst_category: {final_diagnosis['worst_category']}")
    lines.append(f"- diagnosis_labels: {', '.join(final_diagnosis['diagnosis_labels'])}")
    lines.append(f"- summary: {final_diagnosis['summary']}")
    lines.append("")

    return "\n".join(lines)


def write_non_positive_median_diagnosis_report(
    summary: dict[str, Any],
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, str]:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_output_path.write_text(
        build_non_positive_median_diagnosis_markdown(summary),
        encoding="utf-8",
    )

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing research summary JSON files.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Output path for JSON summary.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help="Output path for Markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_non_positive_median_diagnosis_report(args.input_dir)
    outputs = write_non_positive_median_diagnosis_report(
        summary=summary,
        json_output_path=args.json_output,
        markdown_output_path=args.md_output,
    )
    print(
        json.dumps(
            {
                **outputs,
                "final_diagnosis": summary["final_diagnosis"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()