from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT_DIR = Path("logs/research_reports/latest")
DEFAULT_JSON_OUTPUT = DEFAULT_INPUT_DIR / "non_positive_median_diagnosis_summary.json"
DEFAULT_MD_OUTPUT = DEFAULT_INPUT_DIR / "non_positive_median_diagnosis_summary.md"

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_CATEGORIES = ("strategy", "symbol", "alignment_state")


@dataclass(frozen=True)
class MetricRow:
    source: str
    horizon: str
    category: str
    group: str
    rank: int | None
    median_future_return_pct: float
    avg_future_return_pct: float
    positive_rate_pct: float
    flat_rate_pct: float
    labeled_count: int | None
    origin_file: str
    path_hint: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class PairRow:
    horizon: str
    category: str
    group: str
    rank: int | None
    latest_median_future_return_pct: float | None
    cumulative_median_future_return_pct: float | None
    latest_avg_future_return_pct: float | None
    cumulative_avg_future_return_pct: float | None
    latest_positive_rate_pct: float | None
    cumulative_positive_rate_pct: float | None
    latest_flat_rate_pct: float | None
    cumulative_flat_rate_pct: float | None
    latest_labeled_count: int | None
    cumulative_labeled_count: int | None
    origin_file: str
    path_hint: str
    raw: dict[str, Any]


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
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
    if value is None or isinstance(value, bool):
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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _path_hint(parts: list[str]) -> str:
    return " > ".join(parts[-8:])


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _extract_metric(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in raw:
            value = _safe_float(raw.get(key))
            if value is not None:
                return value
    return None


def _extract_labeled_count(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in raw:
            value = _safe_int(raw.get(key))
            if value is not None:
                return value
    return None


def _normalize_group(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _metric_complete(raw: dict[str, Any]) -> bool:
    return (
        _extract_metric(raw, "median_future_return_pct") is not None
        and _extract_metric(raw, "avg_future_return_pct", "mean_future_return_pct") is not None
        and _extract_metric(raw, "positive_rate_pct", "positive_rate") is not None
        and _extract_metric(raw, "flat_rate_pct", "flat_rate") is not None
    )


def _walk_summary_rows(
    node: Any,
    horizon: str | None,
    category: str | None,
    path_parts: list[str],
    out_rows: list[tuple[dict[str, Any], str, str, int | None, list[str]]],
) -> None:
    if isinstance(node, dict):
        if horizon is not None and category is not None and _metric_complete(node):
            out_rows.append((node, horizon, category, None, path_parts))
            return

        for key, value in node.items():
            next_horizon = horizon
            next_category = category

            if key in TARGET_HORIZONS:
                next_horizon = key
            if key in TARGET_CATEGORIES:
                next_category = key

            _walk_summary_rows(
                value,
                next_horizon,
                next_category,
                [*path_parts, key],
                out_rows,
            )

    elif isinstance(node, list):
        for index, item in enumerate(node):
            rank = index + 1 if horizon is not None and category is not None else None
            if isinstance(item, dict) and horizon is not None and category is not None and _metric_complete(item):
                out_rows.append((item, horizon, category, rank, [*path_parts, f"[{index}]"]))
            else:
                _walk_summary_rows(
                    item,
                    horizon,
                    category,
                    [*path_parts, f"[{index}]"],
                    out_rows,
                )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_main_metric_rows(input_dir: Path) -> tuple[list[MetricRow], dict[str, int]]:
    instrumentation = {
        "summary_json_found": 0,
        "summary_candidate_rows_seen": 0,
        "main_rows_count": 0,
        "metric_complete_rows_count": 0,
        "summary_read_error_count": 0,
        "summary_json_decode_error_count": 0,
    }

    summary_path = input_dir / "summary.json"
    rows: list[MetricRow] = []

    if not summary_path.exists():
        return rows, instrumentation

    instrumentation["summary_json_found"] = 1

    try:
        payload = _load_json(summary_path)
    except json.JSONDecodeError:
        instrumentation["summary_json_decode_error_count"] += 1
        return rows, instrumentation
    except OSError:
        instrumentation["summary_read_error_count"] += 1
        return rows, instrumentation

    discovered: list[tuple[dict[str, Any], str, str, int | None, list[str]]] = []
    _walk_summary_rows(payload, None, None, [summary_path.name], discovered)

    instrumentation["summary_candidate_rows_seen"] = len(discovered)

    for raw, horizon, category, rank, path_parts in discovered:
        median = _extract_metric(raw, "median_future_return_pct")
        avg = _extract_metric(raw, "avg_future_return_pct", "mean_future_return_pct")
        positive_rate = _extract_metric(raw, "positive_rate_pct", "positive_rate")
        flat_rate = _extract_metric(raw, "flat_rate_pct", "flat_rate")

        if (
            median is None
            or avg is None
            or positive_rate is None
            or flat_rate is None
        ):
            continue

        instrumentation["metric_complete_rows_count"] += 1

        group = _normalize_group(
            raw.get("group")
            or raw.get("group_name")
            or raw.get("candidate_group")
            or raw.get("name")
            or raw.get("label")
        )

        rows.append(
            MetricRow(
                source="latest",
                horizon=horizon,
                category=category,
                group=group,
                rank=rank,
                median_future_return_pct=median,
                avg_future_return_pct=avg,
                positive_rate_pct=positive_rate,
                flat_rate_pct=flat_rate,
                labeled_count=_extract_labeled_count(
                    raw,
                    "labeled_count",
                    "label_count",
                    "count",
                    "sample_size",
                    "observations",
                ),
                origin_file=summary_path.name,
                path_hint=_path_hint(path_parts),
                raw=raw,
            )
        )

    instrumentation["main_rows_count"] = len(rows)
    return rows, instrumentation


def load_probe_pair_rows(input_dir: Path) -> tuple[list[PairRow], dict[str, int]]:
    instrumentation = {
        "probe_json_found": 0,
        "auxiliary_probe_rows_count": 0,
        "pair_rows_count": 0,
        "probe_read_error_count": 0,
        "probe_json_decode_error_count": 0,
        "probe_rows_missing_horizon_or_category_count": 0,
    }

    probe_path = input_dir / "latest_cumulative_fallback_probe_summary.json"
    rows: list[PairRow] = []

    if not probe_path.exists():
        return rows, instrumentation

    instrumentation["probe_json_found"] = 1

    try:
        payload = _load_json(probe_path)
    except json.JSONDecodeError:
        instrumentation["probe_json_decode_error_count"] += 1
        return rows, instrumentation
    except OSError:
        instrumentation["probe_read_error_count"] += 1
        return rows, instrumentation

    representative_examples = payload.get("representative_examples", {})
    if not isinstance(representative_examples, dict):
        return rows, instrumentation

    for horizon, examples in representative_examples.items():
        if horizon not in TARGET_HORIZONS or not isinstance(examples, list):
            continue

        for index, raw in enumerate(examples):
            if not isinstance(raw, dict):
                continue

            category = raw.get("category")
            if category not in TARGET_CATEGORIES:
                instrumentation["probe_rows_missing_horizon_or_category_count"] += 1
                continue

            rows.append(
                PairRow(
                    horizon=horizon,
                    category=category,
                    group=_normalize_group(
                        raw.get("group")
                        or raw.get("latest_top_group")
                        or raw.get(f"latest_top_{category}_group")
                        or raw.get("latest_group")
                        or raw.get("cumulative_group")
                    ),
                    rank=index + 1,
                    latest_median_future_return_pct=_extract_metric(
                        raw,
                        "latest_top_median_future_return_pct",
                        "latest_median_future_return_pct",
                    ),
                    cumulative_median_future_return_pct=_extract_metric(
                        raw,
                        "cumulative_top_median_future_return_pct",
                        "cumulative_median_future_return_pct",
                    ),
                    latest_avg_future_return_pct=_extract_metric(
                        raw,
                        "latest_top_avg_future_return_pct",
                        "latest_avg_future_return_pct",
                    ),
                    cumulative_avg_future_return_pct=_extract_metric(
                        raw,
                        "cumulative_top_avg_future_return_pct",
                        "cumulative_avg_future_return_pct",
                    ),
                    latest_positive_rate_pct=_extract_metric(
                        raw,
                        "latest_top_positive_rate_pct",
                        "latest_positive_rate_pct",
                    ),
                    cumulative_positive_rate_pct=_extract_metric(
                        raw,
                        "cumulative_top_positive_rate_pct",
                        "cumulative_positive_rate_pct",
                    ),
                    latest_flat_rate_pct=_extract_metric(
                        raw,
                        "latest_top_flat_rate_pct",
                        "latest_flat_rate_pct",
                    ),
                    cumulative_flat_rate_pct=_extract_metric(
                        raw,
                        "cumulative_top_flat_rate_pct",
                        "cumulative_flat_rate_pct",
                    ),
                    latest_labeled_count=_extract_labeled_count(
                        raw,
                        "latest_top_labeled_count",
                        "latest_labeled_count",
                        "latest_labeled_observation_count",
                    ),
                    cumulative_labeled_count=_extract_labeled_count(
                        raw,
                        "cumulative_top_labeled_count",
                        "cumulative_labeled_count",
                        "cumulative_labeled_observation_count",
                    ),
                    origin_file=probe_path.name,
                    path_hint=_path_hint(
                        [probe_path.name, "representative_examples", horizon, f"[{index}]"]
                    ),
                    raw=raw,
                )
            )

    instrumentation["auxiliary_probe_rows_count"] = len(rows)
    instrumentation["pair_rows_count"] = len(rows)
    return rows, instrumentation


def _is_non_positive_median(row: MetricRow) -> bool:
    return row.median_future_return_pct <= 0.0


def _is_positive_median(row: MetricRow) -> bool:
    return row.median_future_return_pct > 0.0


def _rank_within(row: MetricRow, top_n: int) -> bool:
    return row.rank is not None and row.rank <= top_n


def _flat_rate_dominant(row: MetricRow) -> bool:
    negative_rate = max(0.0, 100.0 - row.positive_rate_pct - row.flat_rate_pct)
    return row.flat_rate_pct >= row.positive_rate_pct and row.flat_rate_pct >= negative_rate


def _has_mean_positive_conflict(row: MetricRow) -> bool:
    return _is_non_positive_median(row) and row.avg_future_return_pct > 0.0


def _has_positive_rate_conflict(row: MetricRow, threshold: float = 50.0) -> bool:
    return _is_non_positive_median(row) and row.positive_rate_pct >= threshold


def _has_sufficient_labels(row: MetricRow, min_count: int = 5) -> bool:
    return _is_non_positive_median(row) and (row.labeled_count or 0) >= min_count


def _build_rank_scope_breakdown(rows: list[MetricRow], top_n: int = 3) -> dict[str, Any]:
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


def _breakdown_by_horizon_and_category(rows: list[MetricRow]) -> dict[str, Any]:
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


def _breakdown_by_category(rows: list[MetricRow]) -> dict[str, Any]:
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


def _build_metric_interaction_breakdown(rows: list[MetricRow]) -> dict[str, Any]:
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


def _build_latest_vs_cumulative_summary(pair_rows: list[PairRow]) -> dict[str, Any]:
    noisy_but_healthy: list[dict[str, Any]] = []

    for row in pair_rows:
        latest_non_positive = (
            row.latest_median_future_return_pct is not None
            and row.latest_median_future_return_pct <= 0.0
        )
        cumulative_positive = (
            row.cumulative_median_future_return_pct is not None
            and row.cumulative_median_future_return_pct > 0.0
        )

        if latest_non_positive and cumulative_positive:
            noisy_but_healthy.append(
                {
                    "horizon": row.horizon,
                    "category": row.category,
                    "group": row.group,
                    "rank": row.rank,
                    "latest_median_future_return_pct": row.latest_median_future_return_pct,
                    "cumulative_median_future_return_pct": row.cumulative_median_future_return_pct,
                    "latest_avg_future_return_pct": row.latest_avg_future_return_pct,
                    "cumulative_avg_future_return_pct": row.cumulative_avg_future_return_pct,
                    "latest_positive_rate_pct": row.latest_positive_rate_pct,
                    "cumulative_positive_rate_pct": row.cumulative_positive_rate_pct,
                    "latest_flat_rate_pct": row.latest_flat_rate_pct,
                    "cumulative_flat_rate_pct": row.cumulative_flat_rate_pct,
                    "origin_file": row.origin_file,
                    "path_hint": row.path_hint,
                }
            )

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
        "pair_count": len(pair_rows),
        "latest_non_positive_while_cumulative_positive_count": len(noisy_but_healthy),
        "latest_non_positive_while_cumulative_positive_ratio": _ratio(
            len(noisy_but_healthy),
            len(pair_rows),
        ),
        "by_horizon": horizon_breakdown,
        "by_category": category_breakdown,
        "examples": noisy_but_healthy[:10],
    }


def _representative_examples(rows: list[MetricRow], limit: int = 10) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _is_non_positive_median(row)]
    candidates.sort(
        key=lambda row: (
            row.rank if row.rank is not None else 9999,
            row.median_future_return_pct,
        )
    )

    examples: list[dict[str, Any]] = []
    for row in candidates[:limit]:
        examples.append(
            {
                "source": row.source,
                "origin_file": row.origin_file,
                "path_hint": row.path_hint,
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

    if overall["total_evaluated_rows"] == 0:
        labels.append("main_latest_metric_rows_not_found")
    else:
        top1_ratio = rank_scope["top1"]["non_positive_median_ratio"]
        top3_ratio = rank_scope["top3"]["non_positive_median_ratio"]
        beyond_top1_ratio = rank_scope["beyond_top1_within_topn"]["non_positive_median_ratio"]

        flat_dominant_ratio = metric_interactions["median_le_zero_and_flat_rate_dominant_ratio"]
        mean_positive_conflict_ratio = metric_interactions["median_le_zero_and_avg_gt_zero_ratio"]
        positive_rate_conflict_ratio = metric_interactions["median_le_zero_and_positive_rate_ge_50_ratio"]
        latest_noise_ratio = latest_vs_cumulative["latest_non_positive_while_cumulative_positive_ratio"]

        if top1_ratio >= 0.5 and top3_ratio >= 0.5:
            labels.append("non_positive_median_is_broad_across_rankings")
        elif top1_ratio >= 0.5 and beyond_top1_ratio < 0.35:
            labels.append("non_positive_median_is_top_rank_specific")

        if flat_dominant_ratio >= 0.4:
            labels.append("flat_heavy_distribution_is_primary_median_suppressor")
        if mean_positive_conflict_ratio > 0.0:
            labels.append("mean_positive_but_median_non_positive_conflict_exists")
        if positive_rate_conflict_ratio > 0.0:
            labels.append("positive_rate_strength_conflicts_with_non_positive_median")
        if latest_noise_ratio >= 0.3:
            labels.append("latest_window_noise_dominates_median_signal")

    if not labels:
        labels.append("non_positive_median_requires_additional_observation")

    worst_horizon = _worst_horizon(horizon_breakdown)
    worst_category = _worst_category(category_breakdown)

    summary = (
        f"evaluated_rows={overall['total_evaluated_rows']}, "
        f"non_positive_median_count={overall['non_positive_median_count']}, "
        f"worst_horizon={worst_horizon}, "
        f"worst_category={worst_category}."
    )

    return {
        "primary_finding": labels[0],
        "secondary_finding": worst_horizon or "unknown_horizon",
        "worst_horizon": worst_horizon,
        "worst_category": worst_category,
        "diagnosis_labels": labels,
        "summary": summary,
    }


def build_non_positive_median_diagnosis_report(input_dir: Path) -> dict[str, Any]:
    main_rows, main_stats = load_main_metric_rows(input_dir)
    pair_rows, pair_stats = load_probe_pair_rows(input_dir)

    overall = {
        "total_evaluated_rows": len(main_rows),
        "non_positive_median_count": sum(1 for row in main_rows if _is_non_positive_median(row)),
        "positive_median_count": sum(1 for row in main_rows if _is_positive_median(row)),
    }
    overall["non_positive_median_ratio"] = _ratio(
        overall["non_positive_median_count"],
        overall["total_evaluated_rows"],
    )

    rank_scope = _build_rank_scope_breakdown(main_rows, top_n=3)
    horizon_breakdown = _breakdown_by_horizon_and_category(main_rows)
    category_breakdown = _breakdown_by_category(main_rows)
    metric_interactions = _build_metric_interaction_breakdown(main_rows)
    latest_vs_cumulative = _build_latest_vs_cumulative_summary(pair_rows)
    representative_examples = _representative_examples(main_rows, limit=10)
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
        "source_targeting": {
            "main_diagnosis_source": "summary.json",
            "auxiliary_pair_source": "latest_cumulative_fallback_probe_summary.json",
            "main_rows_count": main_stats["main_rows_count"],
            "metric_complete_rows_count": main_stats["metric_complete_rows_count"],
            "auxiliary_probe_rows_count": pair_stats["auxiliary_probe_rows_count"],
            "pair_rows_count": pair_stats["pair_rows_count"],
        },
        "parser_instrumentation": {
            **main_stats,
            **pair_stats,
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


def build_non_positive_median_diagnosis_markdown(summary: dict[str, Any]) -> str:
    source_targeting = summary["source_targeting"]
    parser_stats = summary["parser_instrumentation"]
    overall = summary["overall_median_blocker_overview"]
    rank_scope = summary["rank_scope_breakdown"]
    metric_interactions = summary["metric_interaction_breakdown"]
    final_diagnosis = summary["final_diagnosis"]
    latest_vs_cumulative = summary["latest_vs_cumulative_summary"]

    lines: list[str] = []
    lines.append("Non-Positive Median Diagnosis")
    lines.append(f"Generated: {summary['metadata']['generated_at']}")
    lines.append("")
    lines.append("Source Targeting")
    lines.append(f"- main_diagnosis_source: {source_targeting['main_diagnosis_source']}")
    lines.append(f"- auxiliary_pair_source: {source_targeting['auxiliary_pair_source']}")
    lines.append(f"- main_rows_count: {source_targeting['main_rows_count']}")
    lines.append(f"- metric_complete_rows_count: {source_targeting['metric_complete_rows_count']}")
    lines.append(f"- auxiliary_probe_rows_count: {source_targeting['auxiliary_probe_rows_count']}")
    lines.append(f"- pair_rows_count: {source_targeting['pair_rows_count']}")
    lines.append("")
    lines.append("Parser Instrumentation")
    for key, value in parser_stats.items():
        lines.append(f"- {key}: {value}")
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
    lines.append(f"- pair_count: {latest_vs_cumulative['pair_count']}")
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
                f"origin_file={example['origin_file']}, "
                f"path_hint={example['path_hint']}, "
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
                "source_targeting": summary["source_targeting"],
                "parser_instrumentation": summary["parser_instrumentation"],
                "final_diagnosis": summary["final_diagnosis"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()