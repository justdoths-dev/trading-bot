from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.experimental_relabel_comparison_report import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_EXPERIMENT_PATH,
    DEFAULT_JSON_OUTPUT as DEFAULT_COMPARISON_PATH,
    PRIORITY_SYMBOLS,
    TARGET_HORIZONS,
    _extract_group_sections,
    _extract_horizon_summary,
)

DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/median_recovery_failure_diagnosis_summary.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/median_recovery_failure_diagnosis_summary.md"
)

MEANINGFUL_MEDIAN_RECOVERY_PCT = 0.02
MEANINGFUL_FLAT_REDUCTION_PCT = -5.0
GROUP_EXAMPLE_LIMIT = 6


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def load_summary_json(path: Path) -> dict[str, Any]:
    """Load a summary JSON file safely."""
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _compare_numeric_values(
    baseline: float | int | None,
    experiment: float | int | None,
) -> dict[str, Any]:
    baseline_value = None if baseline is None else float(baseline)
    experiment_value = None if experiment is None else float(experiment)
    delta = None
    if baseline_value is not None and experiment_value is not None:
        delta = round(experiment_value - baseline_value, 6)

    return {
        "baseline": baseline_value,
        "experiment": experiment_value,
        "delta": delta,
    }


def _first_non_none_float(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_non_none_int(*values: Any) -> int | None:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _metric_delta(payload: dict[str, Any], key: str) -> float | None:
    return _safe_float(_safe_dict(payload.get(key)).get("delta"))


def _metric_experiment(payload: dict[str, Any], key: str) -> float | None:
    return _safe_float(_safe_dict(payload.get(key)).get("experiment"))


def _metric_baseline(payload: dict[str, Any], key: str) -> float | None:
    return _safe_float(_safe_dict(payload.get(key)).get("baseline"))


def _build_metric_comparison(
    baseline_metrics: dict[str, Any],
    experiment_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_label_dist = _safe_dict(baseline_metrics.get("label_distribution"))
    experiment_label_dist = _safe_dict(experiment_metrics.get("label_distribution"))

    return {
        "labeled_records": _compare_numeric_values(
            _first_non_none_int(
                baseline_metrics.get("labeled_records"),
                baseline_metrics.get("labeled_count"),
                baseline_metrics.get("sample_count"),
            ),
            _first_non_none_int(
                experiment_metrics.get("labeled_records"),
                experiment_metrics.get("labeled_count"),
                experiment_metrics.get("sample_count"),
            ),
        ),
        "median_future_return_pct": _compare_numeric_values(
            _safe_float(baseline_metrics.get("median_future_return_pct")),
            _safe_float(experiment_metrics.get("median_future_return_pct")),
        ),
        "avg_future_return_pct": _compare_numeric_values(
            _first_non_none_float(
                baseline_metrics.get("avg_future_return_pct"),
                baseline_metrics.get("mean_future_return_pct"),
            ),
            _first_non_none_float(
                experiment_metrics.get("avg_future_return_pct"),
                experiment_metrics.get("mean_future_return_pct"),
            ),
        ),
        "positive_rate_pct": _compare_numeric_values(
            _first_non_none_float(
                baseline_metrics.get("positive_rate_pct"),
                baseline_metrics.get("up_rate_pct"),
            ),
            _first_non_none_float(
                experiment_metrics.get("positive_rate_pct"),
                experiment_metrics.get("up_rate_pct"),
            ),
        ),
        "flat_rate_pct": _compare_numeric_values(
            _safe_float(baseline_metrics.get("flat_rate_pct")),
            _safe_float(experiment_metrics.get("flat_rate_pct")),
        ),
        "negative_rate_pct": _compare_numeric_values(
            _first_non_none_float(
                baseline_metrics.get("negative_rate_pct"),
                baseline_metrics.get("down_rate_pct"),
            ),
            _first_non_none_float(
                experiment_metrics.get("negative_rate_pct"),
                experiment_metrics.get("down_rate_pct"),
            ),
        ),
        "down_label_count": _compare_numeric_values(
            _safe_int(baseline_label_dist.get("down")),
            _safe_int(experiment_label_dist.get("down")),
        ),
        "flat_label_count": _compare_numeric_values(
            _safe_int(baseline_label_dist.get("flat")),
            _safe_int(experiment_label_dist.get("flat")),
        ),
        "up_label_count": _compare_numeric_values(
            _safe_int(baseline_label_dist.get("up")),
            _safe_int(experiment_label_dist.get("up")),
        ),
    }


def _extract_comparison_horizon_rows(
    comparison_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    horizon_comparison = _safe_dict(comparison_summary.get("horizon_comparison"))
    if not horizon_comparison:
        return {}

    return {
        horizon: _safe_dict(horizon_comparison.get(horizon))
        for horizon in TARGET_HORIZONS
    }


def _merge_comparison_payload(
    comparison_payload: dict[str, Any],
    fallback_payload: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for key in (
        "labeled_records",
        "median_future_return_pct",
        "avg_future_return_pct",
        "positive_rate_pct",
        "flat_rate_pct",
        "negative_rate_pct",
        "down_label_count",
        "flat_label_count",
        "up_label_count",
    ):
        primary = _safe_dict(comparison_payload.get(key))
        fallback = _safe_dict(fallback_payload.get(key))
        merged[key] = primary if primary else fallback

    return merged


def _classify_horizon_row(horizon: str, payload: dict[str, Any]) -> dict[str, Any]:
    flat_delta = _metric_delta(payload, "flat_rate_pct")
    median_delta = _metric_delta(payload, "median_future_return_pct")
    avg_delta = _metric_delta(payload, "avg_future_return_pct")
    positive_delta = _metric_delta(payload, "positive_rate_pct")
    negative_delta = _metric_delta(payload, "negative_rate_pct")
    experiment_median = _metric_experiment(payload, "median_future_return_pct")
    experiment_positive_rate = _metric_experiment(payload, "positive_rate_pct")

    diagnosis_labels: list[str] = []

    flat_reduced = flat_delta is not None and flat_delta < 0.0
    meaningful_flat_reduced = (
        flat_delta is not None and flat_delta <= MEANINGFUL_FLAT_REDUCTION_PCT
    )
    meaningful_median_recovery = (
        median_delta is not None and median_delta > MEANINGFUL_MEDIAN_RECOVERY_PCT
    )

    if flat_reduced and not meaningful_median_recovery:
        diagnosis_labels.append("flat_reduction_without_meaningful_median_recovery")

    if avg_delta is not None and avg_delta > 0.0 and not meaningful_median_recovery:
        diagnosis_labels.append("mean_recovery_without_median_recovery")

    if (
        flat_reduced
        and negative_delta is not None
        and negative_delta > 0.0
        and not meaningful_median_recovery
    ):
        diagnosis_labels.append("downside_replacement_is_suppressing_median")

    if (
        meaningful_flat_reduced
        and median_delta is not None
        and 0.0 < median_delta <= MEANINGFUL_MEDIAN_RECOVERY_PCT
    ):
        diagnosis_labels.append("short_horizon_recovery_is_too_shallow")

    if not diagnosis_labels:
        diagnosis_labels.append("median_recovery_failure_source_not_found")

    if "downside_replacement_is_suppressing_median" in diagnosis_labels:
        diagnosis_reason = "downside_replacement_is_suppressing_median"
    elif "mean_recovery_without_median_recovery" in diagnosis_labels:
        diagnosis_reason = "mean_recovery_without_median_recovery"
    elif "short_horizon_recovery_is_too_shallow" in diagnosis_labels:
        diagnosis_reason = "short_horizon_recovery_is_too_shallow"
    elif "flat_reduction_without_meaningful_median_recovery" in diagnosis_labels:
        diagnosis_reason = "flat_reduction_without_meaningful_median_recovery"
    else:
        diagnosis_reason = "median_recovery_failure_source_not_found"

    severity = round(
        (
            max(0.0, -(flat_delta or 0.0))
            + max(0.0, negative_delta or 0.0)
            + max(
                0.0,
                (MEANINGFUL_MEDIAN_RECOVERY_PCT - (median_delta or 0.0)) * 100.0,
            )
            + max(0.0, 50.0 - (experiment_positive_rate or 0.0)) * 0.1
            + max(0.0, -(experiment_median or 0.0)) * 100.0
            + max(0.0, -(positive_delta or 0.0))
        ),
        6,
    )

    return {
        "horizon": horizon,
        "labeled_records": _safe_dict(payload.get("labeled_records")),
        "median_future_return_pct": _safe_dict(payload.get("median_future_return_pct")),
        "avg_future_return_pct": _safe_dict(payload.get("avg_future_return_pct")),
        "positive_rate_pct": _safe_dict(payload.get("positive_rate_pct")),
        "flat_rate_pct": _safe_dict(payload.get("flat_rate_pct")),
        "negative_rate_pct": _safe_dict(payload.get("negative_rate_pct")),
        "down_label_count": _safe_dict(payload.get("down_label_count")),
        "diagnosis_reason": diagnosis_reason,
        "diagnosis_labels": sorted(set(diagnosis_labels)),
        "severity_score": severity,
    }


def _build_horizon_diagnosis_rows(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = _extract_horizon_summary(baseline_summary)
    experiment = _extract_horizon_summary(experiment_summary)
    comparison_rows = _extract_comparison_horizon_rows(comparison_summary)

    rows: list[dict[str, Any]] = []
    for horizon in TARGET_HORIZONS:
        fallback_payload = _build_metric_comparison(
            _safe_dict(baseline.get(horizon)),
            _safe_dict(experiment.get(horizon)),
        )
        merged_payload = _merge_comparison_payload(
            _safe_dict(comparison_rows.get(horizon)),
            fallback_payload,
        )
        rows.append(_classify_horizon_row(horizon, merged_payload))

    rows.sort(key=lambda row: TARGET_HORIZONS.index(str(row["horizon"])))
    return rows


def _group_severity_sort_key(row: dict[str, Any], category: str) -> tuple[Any, ...]:
    group = str(row.get("group", ""))
    horizon = str(row.get("horizon", "15m"))
    return (
        0 if category == "symbol" and group.upper() in PRIORITY_SYMBOLS else 1,
        TARGET_HORIZONS.index(horizon),
        -float(row.get("severity_score", 0.0)),
        group,
    )


def _build_group_examples(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    category: str,
    limit: int,
) -> list[dict[str, Any]]:
    baseline_groups = _extract_group_sections(baseline_summary, category)
    experiment_groups = _extract_group_sections(experiment_summary, category)

    examples: list[dict[str, Any]] = []

    for horizon in TARGET_HORIZONS:
        group_names = sorted(
            set(_safe_dict(baseline_groups.get(horizon)).keys())
            | set(_safe_dict(experiment_groups.get(horizon)).keys())
        )

        for group_name in group_names:
            comparison = _build_metric_comparison(
                _safe_dict(_safe_dict(baseline_groups.get(horizon)).get(group_name)),
                _safe_dict(_safe_dict(experiment_groups.get(horizon)).get(group_name)),
            )

            flat_delta = _metric_delta(comparison, "flat_rate_pct")
            median_delta = _metric_delta(comparison, "median_future_return_pct")
            avg_delta = _metric_delta(comparison, "avg_future_return_pct")
            negative_delta = _metric_delta(comparison, "negative_rate_pct")
            experiment_median = _metric_experiment(comparison, "median_future_return_pct")

            if (
                flat_delta is None
                and median_delta is None
                and avg_delta is None
                and negative_delta is None
            ):
                continue

            diagnostic_labels: list[str] = []
            if flat_delta is not None and flat_delta < 0.0:
                diagnostic_labels.append("flat_reduction_observed")
            if avg_delta is not None and avg_delta > 0.0:
                diagnostic_labels.append("avg_recovery_observed")
            if median_delta is None or median_delta < MEANINGFUL_MEDIAN_RECOVERY_PCT:
                diagnostic_labels.append("median_recovery_remains_weak")
            if negative_delta is not None and negative_delta > 0.0:
                diagnostic_labels.append("downside_share_increased")

            severity_score = round(
                max(0.0, -(flat_delta or 0.0)) * 2.0
                + max(0.0, negative_delta or 0.0)
                + max(
                    0.0,
                    (MEANINGFUL_MEDIAN_RECOVERY_PCT - (median_delta or 0.0)) * 100.0,
                )
                + max(0.0, -(experiment_median or 0.0)) * 100.0,
                6,
            )

            examples.append(
                {
                    "category": category,
                    "group": group_name,
                    "horizon": horizon,
                    "median_future_return_pct": _safe_dict(
                        comparison.get("median_future_return_pct")
                    ),
                    "avg_future_return_pct": _safe_dict(
                        comparison.get("avg_future_return_pct")
                    ),
                    "positive_rate_pct": _safe_dict(comparison.get("positive_rate_pct")),
                    "flat_rate_pct": _safe_dict(comparison.get("flat_rate_pct")),
                    "negative_rate_pct": _safe_dict(comparison.get("negative_rate_pct")),
                    "diagnosis_labels": diagnostic_labels,
                    "severity_score": severity_score,
                }
            )

    examples.sort(key=lambda row: _group_severity_sort_key(row, category))
    return examples[:limit]


def _worst_horizon(horizon_rows: list[dict[str, Any]]) -> str | None:
    if not horizon_rows:
        return None

    ordered = sorted(
        horizon_rows,
        key=lambda row: (
            -float(row.get("severity_score", 0.0)),
            TARGET_HORIZONS.index(str(row.get("horizon", "15m"))),
        ),
    )
    return str(ordered[0]["horizon"])


def _build_final_diagnosis(
    horizon_rows: list[dict[str, Any]],
    symbol_examples: list[dict[str, Any]],
    strategy_examples: list[dict[str, Any]],
    comparison_summary: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []

    meaningful_recovery_horizons = [
        row
        for row in horizon_rows
        if (_metric_delta(row, "median_future_return_pct") or 0.0)
        > MEANINGFUL_MEDIAN_RECOVERY_PCT
    ]
    flat_reduced_without_recovery = [
        row
        for row in horizon_rows
        if "flat_reduction_without_meaningful_median_recovery"
        in _safe_list(row.get("diagnosis_labels"))
    ]
    downside_suppressed = [
        row
        for row in horizon_rows
        if "downside_replacement_is_suppressing_median"
        in _safe_list(row.get("diagnosis_labels"))
    ]
    mean_without_median = [
        row
        for row in horizon_rows
        if "mean_recovery_without_median_recovery"
        in _safe_list(row.get("diagnosis_labels"))
    ]

    if flat_reduced_without_recovery:
        labels.append("flat_reduction_without_meaningful_median_recovery")
    if downside_suppressed:
        labels.append("downside_replacement_is_suppressing_median")
    if mean_without_median:
        labels.append("mean_recovery_without_median_recovery")

    if len(meaningful_recovery_horizons) == 1:
        labels.append("short_horizon_recovery_is_too_shallow")
        labels.append("recovery_is_not_broad_across_horizons")
    elif len(meaningful_recovery_horizons) == 0 and horizon_rows:
        labels.append("recovery_is_not_broad_across_horizons")

    if not labels:
        labels.append("median_recovery_failure_source_not_found")

    worst_horizon = _worst_horizon(horizon_rows)
    comparison_final = _safe_dict(comparison_summary.get("final_diagnosis"))
    secondary_finding = _safe_text(comparison_final.get("primary_finding"))
    if secondary_finding is None:
        secondary_finding = labels[1] if len(labels) > 1 else (worst_horizon or "unknown_horizon")

    return {
        "primary_finding": labels[0],
        "secondary_finding": secondary_finding,
        "worst_horizon": worst_horizon,
        "diagnosis_labels": labels,
        "summary": (
            f"worst_horizon={worst_horizon}, "
            f"flat_reduction_without_recovery_horizons={len(flat_reduced_without_recovery)}, "
            f"meaningful_recovery_horizons={len(meaningful_recovery_horizons)}, "
            f"symbol_examples={len(symbol_examples)}, "
            f"strategy_examples={len(strategy_examples)}."
        ),
    }


def build_median_recovery_failure_diagnosis_summary(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    *,
    baseline_path: Path,
    experiment_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Build the median recovery failure diagnosis payload."""
    horizon_rows = _build_horizon_diagnosis_rows(
        baseline_summary=baseline_summary,
        experiment_summary=experiment_summary,
        comparison_summary=comparison_summary,
    )
    symbol_examples = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="symbol",
        limit=GROUP_EXAMPLE_LIMIT,
    )
    strategy_examples = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="strategy",
        limit=GROUP_EXAMPLE_LIMIT,
    )
    final_diagnosis = _build_final_diagnosis(
        horizon_rows=horizon_rows,
        symbol_examples=symbol_examples,
        strategy_examples=strategy_examples,
        comparison_summary=comparison_summary,
    )

    comparison_final = _safe_dict(comparison_summary.get("final_diagnosis"))

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "baseline_path": str(baseline_path),
            "experiment_path": str(experiment_path),
            "comparison_path": str(comparison_path),
        },
        "source_targeting": {
            "baseline_source": baseline_path.name,
            "experiment_source": experiment_path.name,
            "comparison_source": comparison_path.name,
        },
        "parser_instrumentation": {
            "baseline_loaded": bool(baseline_summary),
            "experiment_loaded": bool(experiment_summary),
            "comparison_loaded": bool(comparison_summary),
            "horizon_rows_count": len(horizon_rows),
            "symbol_examples_count": len(symbol_examples),
            "strategy_examples_count": len(strategy_examples),
        },
        "comparison_context": {
            "primary_finding": comparison_final.get("primary_finding"),
            "diagnosis_labels": _safe_list(comparison_final.get("diagnosis_labels")),
        },
        "horizon_diagnosis_rows": horizon_rows,
        "symbol_level_examples": symbol_examples,
        "strategy_level_examples": strategy_examples,
        "final_diagnosis": final_diagnosis,
    }


def build_median_recovery_failure_diagnosis_markdown(summary: dict[str, Any]) -> str:
    """Render the diagnosis summary as Markdown."""
    source_targeting = _safe_dict(summary.get("source_targeting"))
    parser_instrumentation = _safe_dict(summary.get("parser_instrumentation"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines = [
        "# Median Recovery Failure Diagnosis Report",
        "",
        "## Source Targeting",
        f"- baseline_source: {source_targeting.get('baseline_source', 'n/a')}",
        f"- experiment_source: {source_targeting.get('experiment_source', 'n/a')}",
        f"- comparison_source: {source_targeting.get('comparison_source', 'n/a')}",
        "",
        "## Parser Instrumentation",
    ]
    for key, value in parser_instrumentation.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Horizon Diagnosis"])
    horizon_rows = _safe_list(summary.get("horizon_diagnosis_rows"))
    if not horizon_rows:
        lines.append("- none")
    else:
        for row in horizon_rows:
            row_payload = _safe_dict(row)
            lines.append(
                f"- {row_payload.get('horizon', 'unknown')}: "
                f"median_delta={_format_pct(_metric_delta(row_payload, 'median_future_return_pct'))}, "
                f"avg_delta={_format_pct(_metric_delta(row_payload, 'avg_future_return_pct'))}, "
                f"positive_rate_delta={_format_pct(_metric_delta(row_payload, 'positive_rate_pct'))}, "
                f"flat_rate_delta={_format_pct(_metric_delta(row_payload, 'flat_rate_pct'))}, "
                f"negative_rate_delta={_format_pct(_metric_delta(row_payload, 'negative_rate_pct'))}, "
                f"reason={row_payload.get('diagnosis_reason', 'n/a')}"
            )

    lines.extend(["", "## Symbol-Level Examples"])
    symbol_examples = _safe_list(summary.get("symbol_level_examples"))
    if not symbol_examples:
        lines.append("- none")
    else:
        for row in symbol_examples:
            row_payload = _safe_dict(row)
            lines.append(
                f"- {row_payload.get('group', 'unknown')} {row_payload.get('horizon', 'unknown')}: "
                f"median_delta={_format_pct(_metric_delta(row_payload, 'median_future_return_pct'))}, "
                f"flat_delta={_format_pct(_metric_delta(row_payload, 'flat_rate_pct'))}, "
                f"negative_delta={_format_pct(_metric_delta(row_payload, 'negative_rate_pct'))}, "
                f"labels={', '.join(_safe_list(row_payload.get('diagnosis_labels'))) or 'n/a'}"
            )

    lines.extend(["", "## Strategy-Level Examples"])
    strategy_examples = _safe_list(summary.get("strategy_level_examples"))
    if not strategy_examples:
        lines.append("- none")
    else:
        for row in strategy_examples:
            row_payload = _safe_dict(row)
            lines.append(
                f"- {row_payload.get('group', 'unknown')} {row_payload.get('horizon', 'unknown')}: "
                f"median_delta={_format_pct(_metric_delta(row_payload, 'median_future_return_pct'))}, "
                f"flat_delta={_format_pct(_metric_delta(row_payload, 'flat_rate_pct'))}, "
                f"negative_delta={_format_pct(_metric_delta(row_payload, 'negative_rate_pct'))}, "
                f"labels={', '.join(_safe_list(row_payload.get('diagnosis_labels'))) or 'n/a'}"
            )

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}",
            f"- worst_horizon: {final_diagnosis.get('worst_horizon', 'unknown')}",
            f"- diagnosis_labels: {', '.join(_safe_list(final_diagnosis.get('diagnosis_labels'))) or 'n/a'}",
            f"- summary: {final_diagnosis.get('summary', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)


def run_median_recovery_failure_diagnosis_report(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    experiment_path: Path = DEFAULT_EXPERIMENT_PATH,
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    """Build, render, and persist the diagnosis report."""
    baseline_summary = load_summary_json(baseline_path)
    experiment_summary = load_summary_json(experiment_path)
    comparison_summary = load_summary_json(comparison_path)

    summary = build_median_recovery_failure_diagnosis_summary(
        baseline_summary,
        experiment_summary,
        comparison_summary,
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        comparison_path=comparison_path,
    )
    markdown = build_median_recovery_failure_diagnosis_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "markdown": markdown,
        "json_output_path": json_output_path,
        "markdown_output_path": markdown_output_path,
    }


def main() -> None:
    """CLI entrypoint for the diagnosis report."""
    parser = argparse.ArgumentParser(
        description="Diagnose why flat-label suppression did not recover median returns."
    )
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--experiment-path", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    result = run_median_recovery_failure_diagnosis_report(
        baseline_path=args.baseline_path,
        experiment_path=args.experiment_path,
        comparison_path=args.comparison_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )

    print(
        json.dumps(
            {
                "summary_json": str(result["json_output_path"]),
                "summary_md": str(result["markdown_output_path"]),
                "source_targeting": result["summary"]["source_targeting"],
                "final_diagnosis": result["summary"]["final_diagnosis"],
                "baseline_loaded": result["summary"]["parser_instrumentation"]["baseline_loaded"],
                "experiment_loaded": result["summary"]["parser_instrumentation"]["experiment_loaded"],
                "comparison_loaded": result["summary"]["parser_instrumentation"]["comparison_loaded"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()




