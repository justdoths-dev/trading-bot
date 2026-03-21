from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.non_positive_median_diagnosis_report import (
    DEFAULT_INPUT_DIR,
    MetricRow,
    load_main_metric_rows,
)

DEFAULT_JSON_OUTPUT = DEFAULT_INPUT_DIR / "flat_label_distribution_diagnosis_summary.json"
DEFAULT_MD_OUTPUT = DEFAULT_INPUT_DIR / "flat_label_distribution_diagnosis_summary.md"

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_CATEGORIES = ("strategy", "symbol", "alignment_state")
FLAT_RATE_THRESHOLDS = (40.0, 50.0, 60.0, 70.0)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _is_non_positive_median(row: MetricRow) -> bool:
    return row.median_future_return_pct <= 0.0


def _flat_minus_positive(row: MetricRow) -> float:
    return round(row.flat_rate_pct - row.positive_rate_pct, 4)


def _negative_rate(row: MetricRow) -> float:
    return round(max(0.0, 100.0 - row.positive_rate_pct - row.flat_rate_pct), 4)


def _is_flat_dominant(row: MetricRow) -> bool:
    return row.flat_rate_pct >= row.positive_rate_pct and row.flat_rate_pct >= _negative_rate(row)


def _is_flat_threshold_exceeded(row: MetricRow, threshold: float) -> bool:
    return row.flat_rate_pct >= threshold


def _build_overall_overview(rows: list[MetricRow]) -> dict[str, Any]:
    flat_dominant_rows = [row for row in rows if _is_flat_dominant(row)]
    return {
        "total_evaluated_rows": len(rows),
        "flat_dominant_count": len(flat_dominant_rows),
        "flat_dominant_ratio": _ratio(len(flat_dominant_rows), len(rows)),
        "avg_flat_rate_pct": _avg([row.flat_rate_pct for row in rows]),
        "avg_positive_rate_pct": _avg([row.positive_rate_pct for row in rows]),
        "avg_flat_minus_positive_pct": _avg([_flat_minus_positive(row) for row in rows]),
    }


def _build_threshold_buckets(rows: list[MetricRow]) -> dict[str, Any]:
    buckets: dict[str, Any] = {}
    for threshold in FLAT_RATE_THRESHOLDS:
        count = sum(1 for row in rows if _is_flat_threshold_exceeded(row, threshold))
        key = f"flat_rate_ge_{int(threshold)}"
        buckets[key] = {
            "count": count,
            "ratio": _ratio(count, len(rows)),
        }
    return buckets


def _build_horizon_breakdown(rows: list[MetricRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        horizon_rows = [row for row in rows if row.horizon == horizon]
        flat_dominant_rows = [row for row in horizon_rows if _is_flat_dominant(row)]
        result[horizon] = {
            "evaluated_rows": len(horizon_rows),
            "flat_dominant_count": len(flat_dominant_rows),
            "flat_dominant_ratio": _ratio(len(flat_dominant_rows), len(horizon_rows)),
            "avg_flat_rate_pct": _avg([row.flat_rate_pct for row in horizon_rows]),
            "avg_positive_rate_pct": _avg([row.positive_rate_pct for row in horizon_rows]),
            "avg_flat_minus_positive_pct": _avg([_flat_minus_positive(row) for row in horizon_rows]),
            "threshold_buckets": _build_threshold_buckets(horizon_rows),
        }
    return result


def _build_category_breakdown(rows: list[MetricRow]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category in TARGET_CATEGORIES:
        category_rows = [row for row in rows if row.category == category]
        flat_dominant_rows = [row for row in category_rows if _is_flat_dominant(row)]
        result[category] = {
            "evaluated_rows": len(category_rows),
            "flat_dominant_count": len(flat_dominant_rows),
            "flat_dominant_ratio": _ratio(len(flat_dominant_rows), len(category_rows)),
            "avg_flat_rate_pct": _avg([row.flat_rate_pct for row in category_rows]),
            "avg_positive_rate_pct": _avg([row.positive_rate_pct for row in category_rows]),
            "avg_flat_minus_positive_pct": _avg([_flat_minus_positive(row) for row in category_rows]),
            "threshold_buckets": _build_threshold_buckets(category_rows),
        }
    return result


def _build_interaction_breakdown(rows: list[MetricRow]) -> dict[str, Any]:
    flat_dominant_rows = [row for row in rows if _is_flat_dominant(row)]
    flat_and_non_positive = [row for row in flat_dominant_rows if _is_non_positive_median(row)]
    flat_and_mean_positive = [row for row in flat_dominant_rows if row.avg_future_return_pct > 0.0]
    flat_and_positive_rate_strong = [row for row in flat_dominant_rows if row.positive_rate_pct >= 50.0]

    return {
        "flat_dominant_rows": len(flat_dominant_rows),
        "flat_dominant_and_non_positive_median_count": len(flat_and_non_positive),
        "flat_dominant_and_non_positive_median_ratio": _ratio(len(flat_and_non_positive), len(flat_dominant_rows)),
        "flat_dominant_and_avg_gt_zero_count": len(flat_and_mean_positive),
        "flat_dominant_and_avg_gt_zero_ratio": _ratio(len(flat_and_mean_positive), len(flat_dominant_rows)),
        "flat_dominant_and_positive_rate_ge_50_count": len(flat_and_positive_rate_strong),
        "flat_dominant_and_positive_rate_ge_50_ratio": _ratio(len(flat_and_positive_rate_strong), len(flat_dominant_rows)),
    }


def _representative_examples(rows: list[MetricRow], limit: int = 12) -> list[dict[str, Any]]:
    flat_rows = [row for row in rows if _is_flat_dominant(row)]
    flat_rows.sort(
        key=lambda row: (
            -row.flat_rate_pct,
            row.median_future_return_pct,
        )
    )

    examples: list[dict[str, Any]] = []
    for row in flat_rows[:limit]:
        examples.append(
            {
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
                "flat_minus_positive_pct": _flat_minus_positive(row),
            }
        )
    return examples


def _worst_horizon(horizon_breakdown: dict[str, Any]) -> str | None:
    ordered = sorted(
        TARGET_HORIZONS,
        key=lambda horizon: (
            horizon_breakdown[horizon]["flat_dominant_ratio"],
            horizon_breakdown[horizon]["avg_flat_minus_positive_pct"],
        ),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _worst_category(category_breakdown: dict[str, Any]) -> str | None:
    ordered = sorted(
        TARGET_CATEGORIES,
        key=lambda category: (
            category_breakdown[category]["flat_dominant_ratio"],
            category_breakdown[category]["avg_flat_minus_positive_pct"],
        ),
        reverse=True,
    )
    return ordered[0] if ordered else None


def _build_final_diagnosis(
    overall: dict[str, Any],
    horizon_breakdown: dict[str, Any],
    category_breakdown: dict[str, Any],
    interactions: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []

    if overall["total_evaluated_rows"] == 0:
        labels.append("flat_distribution_source_rows_not_found")
    else:
        if overall["flat_dominant_ratio"] >= 0.6:
            labels.append("flat_distribution_is_broad_across_latest_rows")

        worst_category = _worst_category(category_breakdown)
        if worst_category == "symbol" and category_breakdown["symbol"]["flat_dominant_ratio"] >= 0.8:
            labels.append("symbol_rows_show_strongest_flat_suppression")

        worst_horizon = _worst_horizon(horizon_breakdown)
        if worst_horizon in {"15m", "1h"} and horizon_breakdown[worst_horizon]["flat_dominant_ratio"] >= 0.6:
            labels.append("short_horizon_flat_pressure_is_dominant")

        if interactions["flat_dominant_and_non_positive_median_ratio"] >= 0.6:
            labels.append("flat_dominance_coexists_with_non_positive_median")

        if not labels:
            labels.append("flat_pressure_exists_but_not_primary")

    worst_horizon = _worst_horizon(horizon_breakdown)
    worst_category = _worst_category(category_breakdown)

    return {
        "primary_finding": labels[0],
        "secondary_finding": worst_horizon or "unknown_horizon",
        "worst_horizon": worst_horizon,
        "worst_category": worst_category,
        "diagnosis_labels": labels,
        "summary": (
            f"evaluated_rows={overall['total_evaluated_rows']}, "
            f"flat_dominant_count={overall['flat_dominant_count']}, "
            f"flat_dominant_ratio={overall['flat_dominant_ratio']}, "
            f"worst_horizon={worst_horizon}, "
            f"worst_category={worst_category}."
        ),
    }


def build_flat_label_distribution_diagnosis_report(input_dir: Path) -> dict[str, Any]:
    rows, parser_stats = load_main_metric_rows(input_dir)

    overall = _build_overall_overview(rows)
    horizon_breakdown = _build_horizon_breakdown(rows)
    category_breakdown = _build_category_breakdown(rows)
    interactions = _build_interaction_breakdown(rows)
    examples = _representative_examples(rows)
    final_diagnosis = _build_final_diagnosis(
        overall=overall,
        horizon_breakdown=horizon_breakdown,
        category_breakdown=category_breakdown,
        interactions=interactions,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_dir": str(input_dir),
        },
        "source_targeting": {
            "main_diagnosis_source": "summary.json",
            "main_rows_count": len(rows),
        },
        "parser_instrumentation": parser_stats,
        "overall_flat_distribution_overview": overall,
        "horizon_breakdown": horizon_breakdown,
        "category_breakdown": category_breakdown,
        "interaction_breakdown": interactions,
        "representative_examples": examples,
        "final_diagnosis": final_diagnosis,
    }


def build_flat_label_distribution_diagnosis_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall_flat_distribution_overview"]
    interactions = summary["interaction_breakdown"]
    final_diagnosis = summary["final_diagnosis"]

    lines: list[str] = []
    lines.append("Flat Label Distribution Diagnosis")
    lines.append(f"Generated: {summary['metadata']['generated_at']}")
    lines.append("")
    lines.append("Source Targeting")
    lines.append(f"- main_diagnosis_source: {summary['source_targeting']['main_diagnosis_source']}")
    lines.append(f"- main_rows_count: {summary['source_targeting']['main_rows_count']}")
    lines.append("")
    lines.append("Parser Instrumentation")
    for key, value in summary["parser_instrumentation"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Overall Flat Distribution Overview")
    lines.append(f"- total_evaluated_rows: {overall['total_evaluated_rows']}")
    lines.append(f"- flat_dominant_count: {overall['flat_dominant_count']}")
    lines.append(f"- flat_dominant_ratio: {overall['flat_dominant_ratio']}")
    lines.append(f"- avg_flat_rate_pct: {overall['avg_flat_rate_pct']}")
    lines.append(f"- avg_positive_rate_pct: {overall['avg_positive_rate_pct']}")
    lines.append(f"- avg_flat_minus_positive_pct: {overall['avg_flat_minus_positive_pct']}")
    lines.append("")
    lines.append("Horizon Breakdown")
    for horizon, payload in summary["horizon_breakdown"].items():
        lines.append(
            f"- {horizon}: flat_dominant_count={payload['flat_dominant_count']}, "
            f"ratio={payload['flat_dominant_ratio']}, "
            f"avg_flat_rate_pct={payload['avg_flat_rate_pct']}, "
            f"avg_positive_rate_pct={payload['avg_positive_rate_pct']}, "
            f"avg_flat_minus_positive_pct={payload['avg_flat_minus_positive_pct']}"
        )
        for threshold_key, bucket in payload["threshold_buckets"].items():
            lines.append(f"  - {threshold_key}: count={bucket['count']}, ratio={bucket['ratio']}")
    lines.append("")
    lines.append("Category Breakdown")
    for category, payload in summary["category_breakdown"].items():
        lines.append(
            f"- {category}: flat_dominant_count={payload['flat_dominant_count']}, "
            f"ratio={payload['flat_dominant_ratio']}, "
            f"avg_flat_rate_pct={payload['avg_flat_rate_pct']}, "
            f"avg_positive_rate_pct={payload['avg_positive_rate_pct']}, "
            f"avg_flat_minus_positive_pct={payload['avg_flat_minus_positive_pct']}"
        )
        for threshold_key, bucket in payload["threshold_buckets"].items():
            lines.append(f"  - {threshold_key}: count={bucket['count']}, ratio={bucket['ratio']}")
    lines.append("")
    lines.append("Interaction Breakdown")
    lines.append(
        f"- flat_dominant_and_non_positive_median_count: "
        f"{interactions['flat_dominant_and_non_positive_median_count']} "
        f"(ratio={interactions['flat_dominant_and_non_positive_median_ratio']})"
    )
    lines.append(
        f"- flat_dominant_and_avg_gt_zero_count: "
        f"{interactions['flat_dominant_and_avg_gt_zero_count']} "
        f"(ratio={interactions['flat_dominant_and_avg_gt_zero_ratio']})"
    )
    lines.append(
        f"- flat_dominant_and_positive_rate_ge_50_count: "
        f"{interactions['flat_dominant_and_positive_rate_ge_50_count']} "
        f"(ratio={interactions['flat_dominant_and_positive_rate_ge_50_ratio']})"
    )
    lines.append("")
    lines.append("Representative Examples")
    if summary["representative_examples"]:
        for example in summary["representative_examples"]:
            lines.append(
                "- "
                f"origin_file={example['origin_file']}, "
                f"path_hint={example['path_hint']}, "
                f"horizon={example['horizon']}, "
                f"category={example['category']}, "
                f"group={example['group']}, "
                f"median={_format_pct(example['median_future_return_pct'])}, "
                f"avg={_format_pct(example['avg_future_return_pct'])}, "
                f"positive_rate={_format_pct(example['positive_rate_pct'])}, "
                f"flat_rate={_format_pct(example['flat_rate_pct'])}, "
                f"flat_minus_positive={_format_pct(example['flat_minus_positive_pct'])}"
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


def write_flat_label_distribution_diagnosis_report(
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
        build_flat_label_distribution_diagnosis_markdown(summary),
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
        help="Directory containing latest research summary files.",
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
    summary = build_flat_label_distribution_diagnosis_report(args.input_dir)
    outputs = write_flat_label_distribution_diagnosis_report(
        summary=summary,
        json_output_path=args.json_output,
        markdown_output_path=args.md_output,
    )
    print(
        json.dumps(
            {
                **outputs,
                "source_targeting": summary["source_targeting"],
                "final_diagnosis": summary["final_diagnosis"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()