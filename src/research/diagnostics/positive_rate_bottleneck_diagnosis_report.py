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
    SELECTION_GRADE_POSITIVE_RATE_PCT,
    TARGET_HORIZONS,
    _extract_group_sections,
    _extract_horizon_summary,
)

DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/positive_rate_bottleneck_diagnosis_summary.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/positive_rate_bottleneck_diagnosis_summary.md"
)

MEANINGFUL_POSITIVE_RATE_RECOVERY_PCT = 5.0
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


def _extract_positive_rate(metrics: dict[str, Any]) -> float | None:
    return _safe_float(
        metrics.get("positive_rate_pct", metrics.get("up_rate_pct"))
    )


def _extract_flat_rate(metrics: dict[str, Any]) -> float | None:
    return _safe_float(metrics.get("flat_rate_pct"))


def _extract_median(metrics: dict[str, Any]) -> float | None:
    return _safe_float(metrics.get("median_future_return_pct"))


def _metric_delta(payload: dict[str, Any], key: str) -> float | None:
    return _safe_float(_safe_dict(payload.get(key)).get("delta"))


def _metric_experiment(payload: dict[str, Any], key: str) -> float | None:
    return _safe_float(_safe_dict(payload.get(key)).get("experiment"))


def _build_horizon_rows(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = _extract_horizon_summary(baseline_summary)
    experiment = _extract_horizon_summary(experiment_summary)
    rows: list[dict[str, Any]] = []

    for horizon in TARGET_HORIZONS:
        baseline_metrics = _safe_dict(baseline.get(horizon))
        experiment_metrics = _safe_dict(experiment.get(horizon))

        positive_rate = _compare_numeric_values(
            _extract_positive_rate(baseline_metrics),
            _extract_positive_rate(experiment_metrics),
        )
        flat_rate = _compare_numeric_values(
            _extract_flat_rate(baseline_metrics),
            _extract_flat_rate(experiment_metrics),
        )
        median = _compare_numeric_values(
            _extract_median(baseline_metrics),
            _extract_median(experiment_metrics),
        )

        experiment_positive_rate = _safe_float(positive_rate.get("experiment"))
        selection_gap = None
        if experiment_positive_rate is not None:
            selection_gap = round(
                SELECTION_GRADE_POSITIVE_RATE_PCT - experiment_positive_rate,
                6,
            )

        rows.append(
            {
                "horizon": horizon,
                "positive_rate_pct": positive_rate,
                "flat_rate_pct": flat_rate,
                "median_future_return_pct": median,
                "selection_gap_pct": selection_gap,
                "is_selection_grade": (
                    experiment_positive_rate is not None
                    and experiment_positive_rate >= SELECTION_GRADE_POSITIVE_RATE_PCT
                ),
            }
        )

    rows.sort(key=lambda row: TARGET_HORIZONS.index(str(row["horizon"])))
    return rows


def _group_sort_key(
    row: dict[str, Any],
    *,
    category: str,
    closest: bool,
) -> tuple[Any, ...]:
    group = str(row.get("group", ""))
    horizon = str(row.get("horizon", "15m"))
    selection_gap = _safe_float(row.get("selection_gap_pct"))
    positive_rate = _metric_experiment(row, "positive_rate_pct")

    priority_bucket = 0 if category == "symbol" and group.upper() in PRIORITY_SYMBOLS else 1
    horizon_bucket = TARGET_HORIZONS.index(horizon)

    if closest:
        gap_bucket = abs(selection_gap) if selection_gap is not None else float("inf")
        tie_bucket = -(positive_rate or 0.0)
    else:
        gap_bucket = -(selection_gap or 0.0) if selection_gap is not None else float("inf")
        tie_bucket = positive_rate or 0.0

    return (
        priority_bucket,
        horizon_bucket,
        gap_bucket,
        tie_bucket,
        group,
    )


def _build_group_examples(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    category: str,
    limit: int,
    closest: bool,
) -> list[dict[str, Any]]:
    baseline_groups = _extract_group_sections(baseline_summary, category)
    experiment_groups = _extract_group_sections(experiment_summary, category)
    rows: list[dict[str, Any]] = []

    for horizon in TARGET_HORIZONS:
        group_names = sorted(
            set(_safe_dict(baseline_groups.get(horizon)).keys())
            | set(_safe_dict(experiment_groups.get(horizon)).keys())
        )

        for group_name in group_names:
            baseline_metrics = _safe_dict(_safe_dict(baseline_groups.get(horizon)).get(group_name))
            experiment_metrics = _safe_dict(_safe_dict(experiment_groups.get(horizon)).get(group_name))

            positive_rate = _compare_numeric_values(
                _extract_positive_rate(baseline_metrics),
                _extract_positive_rate(experiment_metrics),
            )
            flat_rate = _compare_numeric_values(
                _extract_flat_rate(baseline_metrics),
                _extract_flat_rate(experiment_metrics),
            )
            median = _compare_numeric_values(
                _extract_median(baseline_metrics),
                _extract_median(experiment_metrics),
            )

            experiment_positive_rate = _safe_float(positive_rate.get("experiment"))
            if experiment_positive_rate is None and _safe_float(positive_rate.get("delta")) is None:
                continue

            selection_gap = None
            if experiment_positive_rate is not None:
                selection_gap = round(
                    SELECTION_GRADE_POSITIVE_RATE_PCT - experiment_positive_rate,
                    6,
                )

            rows.append(
                {
                    "category": category,
                    "group": group_name,
                    "horizon": horizon,
                    "positive_rate_pct": positive_rate,
                    "median_future_return_pct": median,
                    "flat_rate_pct": flat_rate,
                    "selection_gap_pct": selection_gap,
                }
            )

    rows.sort(key=lambda row: _group_sort_key(row, category=category, closest=closest))
    return rows[:limit]


def _worst_horizon(horizon_rows: list[dict[str, Any]]) -> str | None:
    if not horizon_rows:
        return None

    ordered = sorted(
        horizon_rows,
        key=lambda row: (
            -(row.get("selection_gap_pct") or 0.0),
            TARGET_HORIZONS.index(str(row.get("horizon", "15m"))),
        ),
    )
    return str(ordered[0]["horizon"])


def _worst_category(
    symbol_blockers: list[dict[str, Any]],
    strategy_blockers: list[dict[str, Any]],
) -> str | None:
    symbol_gap = max(
        (
            _safe_float(row.get("selection_gap_pct"))
            for row in symbol_blockers
            if _safe_float(row.get("selection_gap_pct")) is not None
        ),
        default=None,
    )
    strategy_gap = max(
        (
            _safe_float(row.get("selection_gap_pct"))
            for row in strategy_blockers
            if _safe_float(row.get("selection_gap_pct")) is not None
        ),
        default=None,
    )

    if symbol_gap is None and strategy_gap is None:
        return None
    if strategy_gap is None:
        return "symbol"
    if symbol_gap is None:
        return "strategy"
    return "symbol" if symbol_gap >= strategy_gap else "strategy"


def _build_final_diagnosis(
    horizon_rows: list[dict[str, Any]],
    symbol_closest: list[dict[str, Any]],
    symbol_blockers: list[dict[str, Any]],
    strategy_closest: list[dict[str, Any]],
    strategy_blockers: list[dict[str, Any]],
    comparison_summary: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []

    missing_selection = [
        row for row in horizon_rows if not bool(row.get("is_selection_grade"))
    ]
    recovered_horizons = [
        row
        for row in horizon_rows
        if (_metric_delta(row, "positive_rate_pct") or 0.0)
        >= MEANINGFUL_POSITIVE_RATE_RECOVERY_PCT
    ]

    if missing_selection:
        labels.append("positive_rate_remains_below_selection_grade_across_horizons")

    worst_horizon = _worst_horizon(horizon_rows)
    if worst_horizon == "15m":
        labels.append("short_horizon_positive_rate_is_primary_bottleneck")

    if symbol_blockers:
        labels.append("symbol_level_positive_rate_constraint_detected")
    if strategy_blockers:
        labels.append("strategy_level_positive_rate_constraint_detected")
    if recovered_horizons and missing_selection:
        labels.append("positive_rate_recovery_present_but_insufficient")

    if not labels:
        labels.append("positive_rate_bottleneck_source_not_found")

    comparison_final = _safe_dict(comparison_summary.get("final_diagnosis"))
    secondary_finding = _safe_text(comparison_final.get("primary_finding"))
    if secondary_finding is None:
        secondary_finding = labels[1] if len(labels) > 1 else (worst_horizon or "unknown_horizon")

    worst_category = _worst_category(symbol_blockers, strategy_blockers)

    return {
        "primary_finding": labels[0],
        "secondary_finding": secondary_finding,
        "worst_horizon": worst_horizon,
        "worst_category": worst_category,
        "diagnosis_labels": labels,
        "summary": (
            f"worst_horizon={worst_horizon}, "
            f"worst_category={worst_category}, "
            f"recovered_horizons={len(recovered_horizons)}, "
            f"closest_symbol_examples={len(symbol_closest)}, "
            f"closest_strategy_examples={len(strategy_closest)}."
        ),
    }


def build_positive_rate_bottleneck_diagnosis_summary(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    *,
    baseline_path: Path,
    experiment_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Build the positive-rate bottleneck diagnosis payload."""
    horizon_rows = _build_horizon_rows(baseline_summary, experiment_summary)
    symbol_closest = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="symbol",
        limit=GROUP_EXAMPLE_LIMIT,
        closest=True,
    )
    symbol_blockers = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="symbol",
        limit=GROUP_EXAMPLE_LIMIT,
        closest=False,
    )
    strategy_closest = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="strategy",
        limit=GROUP_EXAMPLE_LIMIT,
        closest=True,
    )
    strategy_blockers = _build_group_examples(
        baseline_summary,
        experiment_summary,
        category="strategy",
        limit=GROUP_EXAMPLE_LIMIT,
        closest=False,
    )

    final_diagnosis = _build_final_diagnosis(
        horizon_rows=horizon_rows,
        symbol_closest=symbol_closest,
        symbol_blockers=symbol_blockers,
        strategy_closest=strategy_closest,
        strategy_blockers=strategy_blockers,
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
            "selection_grade_positive_rate_pct": SELECTION_GRADE_POSITIVE_RATE_PCT,
        },
        "parser_instrumentation": {
            "baseline_loaded": bool(baseline_summary),
            "experiment_loaded": bool(experiment_summary),
            "comparison_loaded": bool(comparison_summary),
            "horizon_rows_count": len(horizon_rows),
            "symbol_closest_count": len(symbol_closest),
            "symbol_blockers_count": len(symbol_blockers),
            "strategy_closest_count": len(strategy_closest),
            "strategy_blockers_count": len(strategy_blockers),
        },
        "comparison_context": {
            "primary_finding": comparison_final.get("primary_finding"),
            "diagnosis_labels": _safe_list(comparison_final.get("diagnosis_labels")),
        },
        "horizon_diagnosis_rows": horizon_rows,
        "symbol_closest_examples": symbol_closest,
        "symbol_blocker_examples": symbol_blockers,
        "strategy_closest_examples": strategy_closest,
        "strategy_blocker_examples": strategy_blockers,
        "final_diagnosis": final_diagnosis,
    }


def build_positive_rate_bottleneck_diagnosis_markdown(summary: dict[str, Any]) -> str:
    """Render the positive-rate bottleneck summary as Markdown."""
    source_targeting = _safe_dict(summary.get("source_targeting"))
    parser_instrumentation = _safe_dict(summary.get("parser_instrumentation"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines = [
        "# Positive Rate Bottleneck Diagnosis Report",
        "",
        "## Source Targeting",
        f"- baseline_source: {source_targeting.get('baseline_source', 'n/a')}",
        f"- experiment_source: {source_targeting.get('experiment_source', 'n/a')}",
        f"- comparison_source: {source_targeting.get('comparison_source', 'n/a')}",
        f"- selection_grade_positive_rate_pct: {source_targeting.get('selection_grade_positive_rate_pct', 'n/a')}",
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
                f"positive_rate_delta={_format_pct(_metric_delta(row_payload, 'positive_rate_pct'))}, "
                f"selection_gap_pct={_format_pct(_safe_float(row_payload.get('selection_gap_pct')))}, "
                f"flat_rate_delta={_format_pct(_metric_delta(row_payload, 'flat_rate_pct'))}, "
                f"median_delta={_format_pct(_metric_delta(row_payload, 'median_future_return_pct'))}"
            )

    def _append_group_section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend(["", f"## {title}"])
        if not rows:
            lines.append("- none")
            return
        for row in rows:
            row_payload = _safe_dict(row)
            lines.append(
                f"- {row_payload.get('group', 'unknown')} {row_payload.get('horizon', 'unknown')}: "
                f"positive_rate={_format_pct(_metric_experiment(row_payload, 'positive_rate_pct'))}, "
                f"selection_gap_pct={_format_pct(_safe_float(row_payload.get('selection_gap_pct')))}"
            )

    _append_group_section(
        "Symbol Closest Examples",
        _safe_list(summary.get("symbol_closest_examples")),
    )
    _append_group_section(
        "Symbol Blocker Examples",
        _safe_list(summary.get("symbol_blocker_examples")),
    )
    _append_group_section(
        "Strategy Closest Examples",
        _safe_list(summary.get("strategy_closest_examples")),
    )
    _append_group_section(
        "Strategy Blocker Examples",
        _safe_list(summary.get("strategy_blocker_examples")),
    )

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}",
            f"- worst_horizon: {final_diagnosis.get('worst_horizon', 'unknown')}",
            f"- worst_category: {final_diagnosis.get('worst_category', 'unknown')}",
            f"- diagnosis_labels: {', '.join(_safe_list(final_diagnosis.get('diagnosis_labels'))) or 'n/a'}",
            f"- summary: {final_diagnosis.get('summary', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)


def run_positive_rate_bottleneck_diagnosis_report(
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

    summary = build_positive_rate_bottleneck_diagnosis_summary(
        baseline_summary,
        experiment_summary,
        comparison_summary,
        baseline_path=baseline_path,
        experiment_path=experiment_path,
        comparison_path=comparison_path,
    )
    markdown = build_positive_rate_bottleneck_diagnosis_markdown(summary)

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
        description="Diagnose why positive rate remains below selection-grade thresholds."
    )
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--experiment-path", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    result = run_positive_rate_bottleneck_diagnosis_report(
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
