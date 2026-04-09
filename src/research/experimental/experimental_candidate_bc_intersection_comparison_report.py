from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from src.research.experimental_candidate_comparison_matrix import (
    TARGET_HORIZONS,
    TARGET_LABELS,
    _build_delta_a_to_b,
    _candidate_summary,
    _format_pct,
    _safe_float,
    _safe_text,
    load_jsonl_records,
)
from src.research.experimental_candidate_intersection_utils import (
    build_intersection_datasets,
    filter_candidate_b_records,
    filter_candidate_c_records,
)

DEFAULT_BASELINE_DATASET = Path(
    "logs/experiments/trade_analysis_relabel_candidate_b_vol_adjusted.jsonl"
)
DEFAULT_EXPERIMENT_DATASET = Path(
    "logs/experiments/trade_analysis_relabel_candidate_c_c2_moderate.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_b_vs_c_intersection_comparison.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_b_vs_c_intersection_comparison.md"
)
DEFAULT_CANDIDATE_C_VARIANT = "c2_moderate"
CATEGORY_HIGHLIGHT_LIMIT = 5
MEANINGFUL_COVERAGE_DELTA = -0.03
MEANINGFUL_PURITY_RATE_DELTA = 0.03
MEANINGFUL_PURITY_MEDIAN_DELTA = 0.02


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 6)


def _valid_label(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_LABELS:
        return text
    return None


def _group_value(row: dict[str, Any], category: str) -> str:
    if category == "strategy":
        return _safe_text(row.get("selected_strategy") or row.get("strategy")) or "unknown"
    return _safe_text(row.get("symbol")) or "unknown"


def _build_shared_label_distribution_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    baseline_overview = _safe_dict(baseline_summary.get("dataset_overview"))
    experiment_overview = _safe_dict(experiment_summary.get("dataset_overview"))
    baseline_distribution = _safe_dict(baseline_summary.get("label_distribution_by_horizon"))
    experiment_distribution = _safe_dict(experiment_summary.get("label_distribution_by_horizon"))

    for horizon in TARGET_HORIZONS:
        baseline_labeled = _safe_dict(baseline_overview.get("rows_with_valid_labels_by_horizon")).get(horizon, 0)
        experiment_labeled = _safe_dict(experiment_overview.get("rows_with_valid_labels_by_horizon")).get(horizon, 0)
        horizon_payload: dict[str, Any] = {
            "labeled_records": {
                "baseline": float(baseline_labeled),
                "experiment": float(experiment_labeled),
                "delta": round(float(experiment_labeled) - float(baseline_labeled), 6),
            }
        }

        for label in TARGET_LABELS:
            baseline_label = _safe_dict(_safe_dict(baseline_distribution.get(horizon)).get(label))
            experiment_label = _safe_dict(_safe_dict(experiment_distribution.get(horizon)).get(label))
            baseline_count = int(baseline_label.get("count", 0))
            experiment_count = int(experiment_label.get("count", 0))
            baseline_ratio = _safe_float(baseline_label.get("ratio"))
            experiment_ratio = _safe_float(experiment_label.get("ratio"))
            delta = None
            if baseline_ratio is not None and experiment_ratio is not None:
                delta = round(experiment_ratio - baseline_ratio, 6)
            horizon_payload[label] = {
                "count": {
                    "baseline": float(baseline_count),
                    "experiment": float(experiment_count),
                    "delta": round(float(experiment_count) - float(baseline_count), 6),
                },
                "ratio": {
                    "baseline": baseline_ratio,
                    "experiment": experiment_ratio,
                    "delta": delta,
                },
            }

        result[horizon] = horizon_payload

    return result


def _build_group_metrics(
    records: list[dict[str, Any]],
    *,
    category: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(_group_value(row, category), []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for group_name in sorted(grouped):
        rows = grouped[group_name]
        by_horizon: dict[str, Any] = {}
        for horizon in TARGET_HORIZONS:
            labeled_count = 0
            flat_count = 0
            positive_count = 0
            numeric_returns: list[float] = []

            for row in rows:
                label = _valid_label(row.get(f"future_label_{horizon}"))
                future_return = _safe_float(row.get(f"future_return_{horizon}"))
                if label is not None:
                    labeled_count += 1
                    if label == "flat":
                        flat_count += 1
                if future_return is not None:
                    numeric_returns.append(future_return)
                    if future_return > 0.0:
                        positive_count += 1

            by_horizon[horizon] = {
                "labeled_records": labeled_count,
                "median_future_return_pct": _median_or_none(numeric_returns),
                "positive_rate_pct": _safe_ratio(positive_count, len(numeric_returns)) if numeric_returns else None,
                "flat_rate_pct": _safe_ratio(flat_count, labeled_count) if labeled_count else None,
            }

        result[group_name] = {
            "total_row_count": len(rows),
            "by_horizon": by_horizon,
        }

    return result


def _build_category_level_comparison(
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    *,
    category: str,
) -> dict[str, list[dict[str, Any]]]:
    baseline = _build_group_metrics(baseline_rows, category=category)
    experiment = _build_group_metrics(experiment_rows, category=category)
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in TARGET_HORIZONS:
        rows: list[dict[str, Any]] = []
        for group_name in sorted(set(baseline) | set(experiment)):
            baseline_h = _safe_dict(_safe_dict(baseline.get(group_name)).get("by_horizon")).get(horizon, {})
            experiment_h = _safe_dict(_safe_dict(experiment.get(group_name)).get("by_horizon")).get(horizon, {})
            row = {
                "group": group_name,
                "baseline_total_row_count": _safe_dict(baseline.get(group_name)).get("total_row_count", 0),
                "experiment_total_row_count": _safe_dict(experiment.get(group_name)).get("total_row_count", 0),
                "labeled_records": {
                    "baseline": float(baseline_h.get("labeled_records", 0)),
                    "experiment": float(experiment_h.get("labeled_records", 0)),
                    "delta": round(float(experiment_h.get("labeled_records", 0)) - float(baseline_h.get("labeled_records", 0)), 6),
                },
                "median_future_return_pct": _compare_numeric(
                    _safe_float(baseline_h.get("median_future_return_pct")),
                    _safe_float(experiment_h.get("median_future_return_pct")),
                ),
                "positive_rate_pct": _compare_numeric(
                    _safe_float(baseline_h.get("positive_rate_pct")),
                    _safe_float(experiment_h.get("positive_rate_pct")),
                ),
                "flat_rate_pct": _compare_numeric(
                    _safe_float(baseline_h.get("flat_rate_pct")),
                    _safe_float(experiment_h.get("flat_rate_pct")),
                ),
            }
            impact_score = sum(
                abs(_safe_float(_safe_dict(row[key]).get("delta")) or 0.0)
                for key in ("median_future_return_pct", "positive_rate_pct", "flat_rate_pct")
            ) + abs(_safe_float(_safe_dict(row["labeled_records"]).get("delta")) or 0.0)
            row["impact_score"] = round(impact_score, 6)
            rows.append(row)

        rows.sort(key=lambda item: (-float(item["impact_score"]), str(item["group"])))
        result[horizon] = rows[:CATEGORY_HIGHLIGHT_LIMIT]

    return result


def _compare_numeric(baseline: float | None, experiment: float | None) -> dict[str, Any]:
    delta = None
    if baseline is not None and experiment is not None:
        delta = round(experiment - baseline, 6)
    return {
        "baseline": baseline,
        "experiment": experiment,
        "delta": delta,
    }


def _build_final_diagnosis(delta_on_shared_rows: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    coverage_recovery = 0
    purity_improvement = 0
    purity_loss = 0

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(_safe_dict(delta_on_shared_rows.get("by_horizon")).get(horizon))
        flat_ratio_change = _safe_float(payload.get("flat_ratio_change"))
        up_bucket_positive_rate_change = _safe_float(payload.get("up_bucket_positive_rate_change"))
        up_bucket_median_change = _safe_float(payload.get("up_bucket_median_change"))
        down_bucket_positive_rate_change = _safe_float(payload.get("down_bucket_positive_rate_change"))
        down_bucket_median_change = _safe_float(payload.get("down_bucket_median_change"))

        if flat_ratio_change is not None and flat_ratio_change <= MEANINGFUL_COVERAGE_DELTA:
            coverage_recovery += 1

        horizon_purity_gain = False
        horizon_purity_loss = False

        if up_bucket_positive_rate_change is not None and up_bucket_positive_rate_change >= MEANINGFUL_PURITY_RATE_DELTA:
            horizon_purity_gain = True
        if up_bucket_median_change is not None and up_bucket_median_change >= MEANINGFUL_PURITY_MEDIAN_DELTA:
            horizon_purity_gain = True
        if down_bucket_positive_rate_change is not None and down_bucket_positive_rate_change <= -MEANINGFUL_PURITY_RATE_DELTA:
            horizon_purity_gain = True
        if down_bucket_median_change is not None and down_bucket_median_change <= -MEANINGFUL_PURITY_MEDIAN_DELTA:
            horizon_purity_gain = True

        if up_bucket_positive_rate_change is not None and up_bucket_positive_rate_change <= -MEANINGFUL_PURITY_RATE_DELTA:
            horizon_purity_loss = True
        if up_bucket_median_change is not None and up_bucket_median_change <= -MEANINGFUL_PURITY_MEDIAN_DELTA:
            horizon_purity_loss = True
        if down_bucket_positive_rate_change is not None and down_bucket_positive_rate_change >= MEANINGFUL_PURITY_RATE_DELTA:
            horizon_purity_loss = True
        if down_bucket_median_change is not None and down_bucket_median_change >= MEANINGFUL_PURITY_MEDIAN_DELTA:
            horizon_purity_loss = True

        purity_improvement += int(horizon_purity_gain)
        purity_loss += int(horizon_purity_loss)

        notes.append(
            f"{horizon}: flat_ratio_change={flat_ratio_change if flat_ratio_change is not None else 'n/a'}, up_bucket_positive_rate_change={up_bucket_positive_rate_change if up_bucket_positive_rate_change is not None else 'n/a'}, down_bucket_positive_rate_change={down_bucket_positive_rate_change if down_bucket_positive_rate_change is not None else 'n/a'}."
        )

    if coverage_recovery > 0 and purity_loss == 0:
        primary_finding = "candidate_c_looks_more_seed_friendly_than_candidate_b_on_shared_rows"
        secondary_finding = "coverage_recovery_without_clear_purity_loss"
    elif coverage_recovery > 0 and purity_loss > 0:
        primary_finding = "candidate_c_recovers_coverage_but_gives_back_some_purity_on_shared_rows"
        secondary_finding = "middle_path_tradeoff_is_present"
    elif purity_loss > 0 and coverage_recovery == 0:
        primary_finding = "candidate_c_looks_weaker_than_candidate_b_on_shared_rows"
        secondary_finding = "purity_loss_without_coverage_recovery"
    elif purity_improvement > 0 and coverage_recovery == 0:
        primary_finding = "candidate_c_matches_or_improves_purity_but_not_coverage_on_shared_rows"
        secondary_finding = "shared_rows_do_not_support_seed_recovery_yet"
    else:
        primary_finding = "candidate_c_remains_mixed_vs_candidate_b_on_shared_rows"
        secondary_finding = "coverage_vs_purity_tradeoff_is_still_unresolved"

    notes.append(
        "Flat-share reductions are treated as coverage context only; Candidate C is not considered better on shared rows unless purity holds up well enough for a seed-starved 0-selection system."
    )

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "notes": notes,
    }


def build_experimental_candidate_bc_intersection_comparison_report(
    baseline_records: list[dict[str, Any]],
    experiment_records: list[dict[str, Any]],
    *,
    baseline_dataset_path: Path,
    experiment_dataset_path: Path,
    baseline_instrumentation: dict[str, int] | None = None,
    experiment_instrumentation: dict[str, int] | None = None,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    filtered_baseline_records = filter_candidate_b_records(baseline_records)
    filtered_experiment_records = filter_candidate_c_records(
        experiment_records,
        variant_name=candidate_c_variant,
    )
    baseline_shared_rows, experiment_shared_rows, intersection_overview = build_intersection_datasets(
        filtered_baseline_records,
        filtered_experiment_records,
    )

    baseline_shared_summary = _candidate_summary(
        baseline_shared_rows,
        include_volatility_metadata=True,
    )
    experiment_shared_summary = _candidate_summary(
        experiment_shared_rows,
        include_volatility_metadata=False,
    )
    shared_label_distribution = _build_shared_label_distribution_comparison(
        baseline_shared_summary,
        experiment_shared_summary,
    )
    shared_metric_comparison = {
        horizon: {
            "median_future_return_pct": _compare_numeric(
                _safe_float(_safe_dict(_safe_dict(baseline_shared_summary.get("median_future_return_by_horizon")).get(horizon)).get("overall")),
                _safe_float(_safe_dict(_safe_dict(experiment_shared_summary.get("median_future_return_by_horizon")).get(horizon)).get("overall")),
            ),
            "positive_rate_pct": _compare_numeric(
                _safe_float(_safe_dict(_safe_dict(baseline_shared_summary.get("positive_rate_by_horizon")).get(horizon)).get("ratio"))
                if int(_safe_dict(_safe_dict(baseline_shared_summary.get("positive_rate_by_horizon")).get(horizon)).get("numeric_row_count", 0)) > 0
                else None,
                _safe_float(_safe_dict(_safe_dict(experiment_shared_summary.get("positive_rate_by_horizon")).get(horizon)).get("ratio"))
                if int(_safe_dict(_safe_dict(experiment_shared_summary.get("positive_rate_by_horizon")).get(horizon)).get("numeric_row_count", 0)) > 0
                else None,
            ),
            "negative_rate_pct": _compare_numeric(
                None,
                None,
            ),
            "flat_rate_pct": _compare_numeric(
                _safe_float(_safe_dict(_safe_dict(_safe_dict(shared_label_distribution.get(horizon)).get("flat")).get("ratio")).get("baseline")),
                _safe_float(_safe_dict(_safe_dict(_safe_dict(shared_label_distribution.get(horizon)).get("flat")).get("ratio")).get("experiment")),
            ),
        }
        for horizon in TARGET_HORIZONS
    }
    delta_on_shared_rows = _build_delta_a_to_b(
        baseline_shared_summary,
        experiment_shared_summary,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "experimental_candidate_bc_intersection_comparison_report",
            "baseline_name": "candidate_b",
            "experiment_name": "candidate_c",
            "candidate_c_variant": candidate_c_variant,
        },
        "inputs": {
            "baseline_dataset_path": str(baseline_dataset_path),
            "experiment_dataset_path": str(experiment_dataset_path),
            "baseline_parser_instrumentation": baseline_instrumentation or {},
            "experiment_parser_instrumentation": experiment_instrumentation or {},
        },
        "intersection_overview": intersection_overview,
        "shared_row_label_distribution_by_horizon": shared_label_distribution,
        "shared_row_metric_comparison_by_horizon": shared_metric_comparison,
        "shared_row_category_level_comparison": {
            "strategy": _build_category_level_comparison(
                baseline_shared_rows,
                experiment_shared_rows,
                category="strategy",
            ),
            "symbol": _build_category_level_comparison(
                baseline_shared_rows,
                experiment_shared_rows,
                category="symbol",
            ),
        },
        "delta_on_shared_rows": delta_on_shared_rows,
        "baseline_shared_summary": baseline_shared_summary,
        "experiment_shared_summary": experiment_shared_summary,
        "final_diagnosis": _build_final_diagnosis(delta_on_shared_rows),
    }


def build_experimental_candidate_bc_intersection_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    inputs = _safe_dict(summary.get("inputs"))
    overview = _safe_dict(summary.get("intersection_overview"))
    labels = _safe_dict(summary.get("shared_row_label_distribution_by_horizon"))
    metrics = _safe_dict(summary.get("shared_row_metric_comparison_by_horizon"))
    categories = _safe_dict(summary.get("shared_row_category_level_comparison"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines = [
        "# Candidate B vs Candidate C Intersection Comparison",
        "",
        "## Metadata",
        f"- report_type: {metadata.get('report_type', 'n/a')}",
        f"- baseline_name: {metadata.get('baseline_name', 'n/a')}",
        f"- experiment_name: {metadata.get('experiment_name', 'n/a')}",
        f"- candidate_c_variant: {metadata.get('candidate_c_variant', 'n/a')}",
        "",
        "## Inputs",
        f"- baseline_dataset_path: {inputs.get('baseline_dataset_path', 'n/a')}",
        f"- experiment_dataset_path: {inputs.get('experiment_dataset_path', 'n/a')}",
        "",
        "## Intersection Overview",
        f"- baseline_total_rows: {overview.get('baseline_total_rows', 0)}",
        f"- experiment_total_rows: {overview.get('experiment_total_rows', 0)}",
        f"- shared_row_count: {overview.get('shared_row_count', 0)}",
        f"- baseline_only_row_count: {overview.get('baseline_only_row_count', 0)}",
        f"- experiment_only_row_count: {overview.get('experiment_only_row_count', 0)}",
        f"- shared_ratio_vs_baseline: {_format_pct(_safe_float(overview.get('shared_ratio_vs_baseline')))}",
        f"- shared_ratio_vs_experiment: {_format_pct(_safe_float(overview.get('shared_ratio_vs_experiment')))}",
        "",
        "## Shared-Row Label Distribution By Horizon",
    ]

    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(labels.get(horizon))
        lines.append(
            f"- {horizon}: labeled_records_delta={_safe_dict(payload.get('labeled_records')).get('delta', 'n/a')}, "
            f"up_ratio_delta={_format_pct(_safe_float(_safe_dict(_safe_dict(payload.get('up')).get('ratio')).get('delta')))}, "
            f"down_ratio_delta={_format_pct(_safe_float(_safe_dict(_safe_dict(payload.get('down')).get('ratio')).get('delta')))}, "
            f"flat_ratio_delta={_format_pct(_safe_float(_safe_dict(_safe_dict(payload.get('flat')).get('ratio')).get('delta')))}"
        )

    lines.extend(["", "## Shared-Row Metric Comparison By Horizon"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(metrics.get(horizon))
        lines.append(
            f"- {horizon}: median_delta={_format_pct(_safe_float(_safe_dict(payload.get('median_future_return_pct')).get('delta')))}, "
            f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('positive_rate_pct')).get('delta')))}, "
            f"negative_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('negative_rate_pct')).get('delta')))}, "
            f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('flat_rate_pct')).get('delta')))}"
        )

    for category in ("strategy", "symbol"):
        lines.extend(["", f"## {category.title()} Highlights On Shared Rows"])
        category_payload = _safe_dict(categories.get(category))
        for horizon in TARGET_HORIZONS:
            rows = category_payload.get(horizon, [])
            if not rows:
                lines.append(f"- {horizon}: none")
                continue
            for row in rows:
                row_payload = _safe_dict(row)
                lines.append(
                    f"- {horizon} {row_payload.get('group', 'unknown')}: labeled_records_delta={_safe_dict(row_payload.get('labeled_records')).get('delta', 'n/a')}, "
                    f"median_delta={_format_pct(_safe_float(_safe_dict(row_payload.get('median_future_return_pct')).get('delta')))}, "
                    f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(row_payload.get('positive_rate_pct')).get('delta')))}, "
                    f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(row_payload.get('flat_rate_pct')).get('delta')))}"
                )

    lines.extend([
        "",
        "## Final Diagnosis",
        f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}",
        f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}",
    ])
    for note in final_diagnosis.get("notes", []):
        lines.append(f"- note: {note}")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_bc_intersection_comparison_report(
    baseline_dataset_path: Path = DEFAULT_BASELINE_DATASET,
    experiment_dataset_path: Path = DEFAULT_EXPERIMENT_DATASET,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
    candidate_c_variant: str = DEFAULT_CANDIDATE_C_VARIANT,
) -> dict[str, Any]:
    baseline_records, baseline_instrumentation = load_jsonl_records(baseline_dataset_path)
    experiment_records, experiment_instrumentation = load_jsonl_records(experiment_dataset_path)

    summary = build_experimental_candidate_bc_intersection_comparison_report(
        baseline_records,
        experiment_records,
        baseline_dataset_path=baseline_dataset_path,
        experiment_dataset_path=experiment_dataset_path,
        baseline_instrumentation=baseline_instrumentation,
        experiment_instrumentation=experiment_instrumentation,
        candidate_c_variant=candidate_c_variant,
    )
    markdown = build_experimental_candidate_bc_intersection_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an intersection-aware Candidate B vs Candidate C comparison report"
    )
    parser.add_argument("--baseline-dataset", type=Path, default=DEFAULT_BASELINE_DATASET)
    parser.add_argument("--experiment-dataset", type=Path, default=DEFAULT_EXPERIMENT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--candidate-c-variant", type=str, default=DEFAULT_CANDIDATE_C_VARIANT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_bc_intersection_comparison_report(
        baseline_dataset_path=args.baseline_dataset,
        experiment_dataset_path=args.experiment_dataset,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
        candidate_c_variant=args.candidate_c_variant,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


