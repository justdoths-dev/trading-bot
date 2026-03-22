from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

CANDIDATE_A_DEFAULT_PATH = Path(
    "logs/experiments/trade_analysis_relabel_candidate_a.jsonl"
)
CANDIDATE_B_DEFAULT_PATH = Path(
    "logs/experiments/trade_analysis_relabel_candidate_b_vol_adjusted.jsonl"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_b_comparison.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_comparison/candidate_a_vs_b_comparison.md"
)

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_LABELS = ("up", "down", "flat")
CANDIDATE_B_LABELING_METHOD = "candidate_b_volatility_adjusted_v1"

HIGHLIGHT_GROUP_LIMIT = 5
PRIORITY_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 6)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 6)


def _valid_label(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_LABELS:
        return text
    return None


def _strategy_value(row: dict[str, Any]) -> str:
    return (
        _safe_text(row.get("selected_strategy"))
        or _safe_text(row.get("strategy"))
        or "unknown"
    )


def _symbol_value(row: dict[str, Any]) -> str:
    return _safe_text(row.get("symbol")) or "unknown"


def load_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load dict-like JSONL rows safely and deterministically."""
    instrumentation = {
        "blank_line_count": 0,
        "invalid_json_line_count": 0,
        "non_object_line_count": 0,
    }

    if not path.exists() or not path.is_file():
        return [], instrumentation

    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], instrumentation

    for line in lines:
        stripped = line.strip()
        if not stripped:
            instrumentation["blank_line_count"] += 1
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            instrumentation["invalid_json_line_count"] += 1
            continue
        if not isinstance(payload, dict):
            instrumentation["non_object_line_count"] += 1
            continue
        records.append(payload)

    return records, instrumentation


def _filter_candidate_b_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only Candidate B rows matching the expected experimental labeling method."""
    filtered: list[dict[str, Any]] = []
    for row in records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("labeling_method") == CANDIDATE_B_LABELING_METHOD:
            filtered.append(row)
    return filtered


def _build_dataset_overview(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_future_returns: dict[str, int] = {}
    valid_labels: dict[str, int] = {}

    for horizon in TARGET_HORIZONS:
        numeric_future_returns[horizon] = sum(
            1
            for row in records
            if _safe_float(row.get(f"future_return_{horizon}")) is not None
        )
        valid_labels[horizon] = sum(
            1
            for row in records
            if _valid_label(row.get(f"future_label_{horizon}")) is not None
        )

    return {
        "total_row_count": len(records),
        "rows_with_numeric_future_return_by_horizon": numeric_future_returns,
        "rows_with_valid_labels_by_horizon": valid_labels,
    }


def _build_label_distribution_by_horizon(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        counts = {label: 0 for label in TARGET_LABELS}
        valid_label_count = 0

        for row in records:
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if label is None:
                continue
            counts[label] += 1
            valid_label_count += 1

        result[horizon] = {
            label: {
                "count": counts[label],
                "ratio": _safe_ratio(counts[label], valid_label_count),
            }
            for label in TARGET_LABELS
        }
    return result


def _build_median_future_return_by_horizon(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Overall median future return by horizon.

    Note: this is usually invariant across A/B if both datasets contain the same rows and
    only labels changed. It is retained for completeness, but not treated as a primary
    winner metric.
    """
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        overall_returns: list[float] = []
        for row in records:
            future_return = _safe_float(row.get(f"future_return_{horizon}"))
            if future_return is not None:
                overall_returns.append(future_return)

        result[horizon] = {
            "overall": _median_or_none(overall_returns),
        }
    return result


def _build_label_conditional_median_future_return_by_horizon(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Median future return among rows assigned to each label bucket.

    This is a more meaningful A/B comparison target because relabel experiments change
    bucket assignment, not the underlying future returns themselves.
    """
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        returns_by_label: dict[str, list[float]] = {label: [] for label in TARGET_LABELS}

        for row in records:
            future_return = _safe_float(row.get(f"future_return_{horizon}"))
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if future_return is None or label is None:
                continue
            returns_by_label[label].append(future_return)

        result[horizon] = {
            label: _median_or_none(returns_by_label[label])
            for label in TARGET_LABELS
        }
    return result


def _build_positive_rate_by_horizon(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Overall positive rate by horizon.

    Note: like overall median, this tends to be invariant across relabel candidates when
    only labels change. Retained for completeness and diagnostics.
    """
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        numeric_returns = [
            _safe_float(row.get(f"future_return_{horizon}"))
            for row in records
        ]
        filtered = [value for value in numeric_returns if value is not None]
        positive_count = sum(1 for value in filtered if value > 0.0)

        result[horizon] = {
            "count": positive_count,
            "ratio": _safe_ratio(positive_count, len(filtered)),
            "numeric_row_count": len(filtered),
        }
    return result


def _build_label_conditional_positive_rate_by_horizon(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Positive rate among rows assigned to each label bucket.

    For example:
    - up bucket positive rate should ideally be high
    - down bucket positive rate should ideally be low
    - flat bucket should be mixed / near neutral
    """
    result: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        counts_by_label = {label: 0 for label in TARGET_LABELS}
        positive_by_label = {label: 0 for label in TARGET_LABELS}

        for row in records:
            future_return = _safe_float(row.get(f"future_return_{horizon}"))
            label = _valid_label(row.get(f"future_label_{horizon}"))
            if future_return is None or label is None:
                continue

            counts_by_label[label] += 1
            if future_return > 0.0:
                positive_by_label[label] += 1

        result[horizon] = {
            label: {
                "positive_rate": _safe_ratio(
                    positive_by_label[label],
                    counts_by_label[label],
                ),
                "row_count": counts_by_label[label],
            }
            for label in TARGET_LABELS
        }
    return result


def _build_group_positive_metrics(
    records: list[dict[str, Any]],
    *,
    key_name: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        group = _strategy_value(row) if key_name == "selected_strategy" else _symbol_value(row)
        grouped.setdefault(group, []).append(row)

    result: dict[str, Any] = {}
    for group_name in sorted(grouped):
        rows = grouped[group_name]
        horizons: dict[str, Any] = {}
        for horizon in TARGET_HORIZONS:
            future_returns = [
                _safe_float(row.get(f"future_return_{horizon}"))
                for row in rows
            ]
            numeric_returns = [value for value in future_returns if value is not None]
            positive_count = sum(1 for value in numeric_returns if value > 0.0)
            horizons[horizon] = {
                "positive_rate": _safe_ratio(positive_count, len(numeric_returns)),
                "median_future_return": _median_or_none(numeric_returns),
                "numeric_row_count": len(numeric_returns),
            }

        result[group_name] = {
            "total_row_count": len(rows),
            "by_horizon": horizons,
        }
    return result


def _build_candidate_b_volatility_metadata(records: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_row_count = 0
    threshold_values = {horizon: [] for horizon in TARGET_HORIZONS}
    metadata_row_count = 0

    for row in records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("labeling_method") != CANDIDATE_B_LABELING_METHOD:
            continue

        metadata_row_count += 1
        if metadata.get("used_fallback_atr_pct") is True:
            fallback_row_count += 1

        thresholds = _safe_dict(metadata.get("thresholds"))
        for horizon in TARGET_HORIZONS:
            threshold = _safe_float(thresholds.get(horizon))
            if threshold is not None:
                threshold_values[horizon].append(threshold)

    threshold_statistics = {}
    for horizon in TARGET_HORIZONS:
        values = sorted(threshold_values[horizon])
        threshold_statistics[horizon] = {
            "min": round(min(values), 6) if values else None,
            "max": round(max(values), 6) if values else None,
            "mean": _mean_or_none(values),
            "median": _median_or_none(values),
        }

    return {
        "metadata_row_count": metadata_row_count,
        "fallback_row_count": fallback_row_count,
        "fallback_row_ratio": _safe_ratio(fallback_row_count, metadata_row_count),
        "threshold_statistics_by_horizon": threshold_statistics,
    }


def _candidate_summary(records: list[dict[str, Any]], *, include_volatility_metadata: bool) -> dict[str, Any]:
    summary = {
        "dataset_overview": _build_dataset_overview(records),
        "label_distribution_by_horizon": _build_label_distribution_by_horizon(records),
        "median_future_return_by_horizon": _build_median_future_return_by_horizon(records),
        "label_conditional_median_future_return_by_horizon": _build_label_conditional_median_future_return_by_horizon(records),
        "positive_rate_by_horizon": _build_positive_rate_by_horizon(records),
        "label_conditional_positive_rate_by_horizon": _build_label_conditional_positive_rate_by_horizon(records),
        "positive_rate_by_strategy": _build_group_positive_metrics(
            records,
            key_name="selected_strategy",
        ),
        "positive_rate_by_symbol": _build_group_positive_metrics(
            records,
            key_name="symbol",
        ),
    }
    if include_volatility_metadata:
        summary["volatility_metadata"] = _build_candidate_b_volatility_metadata(records)
    return summary


def _delta_metric(a_value: float | None, b_value: float | None) -> float | None:
    if a_value is None or b_value is None:
        return None
    return round(b_value - a_value, 6)


def _build_delta_a_to_b(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> dict[str, Any]:
    by_horizon: dict[str, Any] = {}

    for horizon in TARGET_HORIZONS:
        a_distribution = _safe_dict(candidate_a["label_distribution_by_horizon"].get(horizon))
        b_distribution = _safe_dict(candidate_b["label_distribution_by_horizon"].get(horizon))

        a_positive = _safe_dict(candidate_a["positive_rate_by_horizon"].get(horizon))
        b_positive = _safe_dict(candidate_b["positive_rate_by_horizon"].get(horizon))

        a_median = _safe_dict(candidate_a["median_future_return_by_horizon"].get(horizon))
        b_median = _safe_dict(candidate_b["median_future_return_by_horizon"].get(horizon))

        a_cond_median = _safe_dict(
            candidate_a["label_conditional_median_future_return_by_horizon"].get(horizon)
        )
        b_cond_median = _safe_dict(
            candidate_b["label_conditional_median_future_return_by_horizon"].get(horizon)
        )

        a_cond_positive = _safe_dict(
            candidate_a["label_conditional_positive_rate_by_horizon"].get(horizon)
        )
        b_cond_positive = _safe_dict(
            candidate_b["label_conditional_positive_rate_by_horizon"].get(horizon)
        )

        by_horizon[horizon] = {
            "flat_ratio_change": _delta_metric(
                _safe_float(_safe_dict(a_distribution.get("flat")).get("ratio")),
                _safe_float(_safe_dict(b_distribution.get("flat")).get("ratio")),
            ),
            "positive_rate_change": _delta_metric(
                _safe_float(a_positive.get("ratio")),
                _safe_float(b_positive.get("ratio")),
            ),
            "overall_median_future_return_change": _delta_metric(
                _safe_float(a_median.get("overall")),
                _safe_float(b_median.get("overall")),
            ),
            "up_ratio_change": _delta_metric(
                _safe_float(_safe_dict(a_distribution.get("up")).get("ratio")),
                _safe_float(_safe_dict(b_distribution.get("up")).get("ratio")),
            ),
            "down_ratio_change": _delta_metric(
                _safe_float(_safe_dict(a_distribution.get("down")).get("ratio")),
                _safe_float(_safe_dict(b_distribution.get("down")).get("ratio")),
            ),
            "up_bucket_median_change": _delta_metric(
                _safe_float(a_cond_median.get("up")),
                _safe_float(b_cond_median.get("up")),
            ),
            "down_bucket_median_change": _delta_metric(
                _safe_float(a_cond_median.get("down")),
                _safe_float(b_cond_median.get("down")),
            ),
            "flat_bucket_median_change": _delta_metric(
                _safe_float(a_cond_median.get("flat")),
                _safe_float(b_cond_median.get("flat")),
            ),
            "up_bucket_positive_rate_change": _delta_metric(
                _safe_float(_safe_dict(a_cond_positive.get("up")).get("positive_rate")),
                _safe_float(_safe_dict(b_cond_positive.get("up")).get("positive_rate")),
            ),
            "down_bucket_positive_rate_change": _delta_metric(
                _safe_float(_safe_dict(a_cond_positive.get("down")).get("positive_rate")),
                _safe_float(_safe_dict(b_cond_positive.get("down")).get("positive_rate")),
            ),
            "flat_bucket_positive_rate_change": _delta_metric(
                _safe_float(_safe_dict(a_cond_positive.get("flat")).get("positive_rate")),
                _safe_float(_safe_dict(b_cond_positive.get("flat")).get("positive_rate")),
            ),
        }

    return {"by_horizon": by_horizon}


def _group_highlights(
    candidate_a_groups: dict[str, Any],
    candidate_b_groups: dict[str, Any],
    *,
    category: str,
) -> list[dict[str, Any]]:
    group_names = sorted(set(candidate_a_groups) | set(candidate_b_groups))
    rows: list[dict[str, Any]] = []

    for group_name in group_names:
        a_group = _safe_dict(candidate_a_groups.get(group_name))
        b_group = _safe_dict(candidate_b_groups.get(group_name))
        score = 0.0
        horizon_changes: dict[str, Any] = {}

        for horizon in TARGET_HORIZONS:
            a_h = _safe_dict(_safe_dict(a_group.get("by_horizon")).get(horizon))
            b_h = _safe_dict(_safe_dict(b_group.get("by_horizon")).get(horizon))

            positive_delta = _delta_metric(
                _safe_float(a_h.get("positive_rate")),
                _safe_float(b_h.get("positive_rate")),
            )
            median_delta = _delta_metric(
                _safe_float(a_h.get("median_future_return")),
                _safe_float(b_h.get("median_future_return")),
            )

            horizon_changes[horizon] = {
                "positive_rate_change": positive_delta,
                "median_future_return_change": median_delta,
            }
            score += abs(positive_delta or 0.0) + abs(median_delta or 0.0)

        rows.append(
            {
                "group": group_name,
                "category": category,
                "candidate_a_total_row_count": _safe_dict(a_group).get("total_row_count", 0),
                "candidate_b_total_row_count": _safe_dict(b_group).get("total_row_count", 0),
                "by_horizon": horizon_changes,
                "impact_score": round(score, 6),
            }
        )

    rows.sort(
        key=lambda row: (
            0 if category == "symbol" and row["group"] in PRIORITY_SYMBOLS else 1,
            -float(row["impact_score"]),
            str(row["group"]),
        )
    )
    return rows[:HIGHLIGHT_GROUP_LIMIT]


def _build_final_summary(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    delta_a_to_b: dict[str, Any],
) -> dict[str, Any]:
    notes: list[str] = []
    improvements = 0
    regressions = 0

    for horizon in TARGET_HORIZONS:
        delta = _safe_dict(delta_a_to_b["by_horizon"].get(horizon))

        flat_change = _safe_float(delta.get("flat_ratio_change"))
        up_bucket_median_change = _safe_float(delta.get("up_bucket_median_change"))
        down_bucket_positive_rate_change = _safe_float(delta.get("down_bucket_positive_rate_change"))

        # Better if flat is reduced, up bucket gets stronger, and down bucket becomes less positive.
        if (
            flat_change is not None
            and flat_change < 0.0
            and (up_bucket_median_change is None or up_bucket_median_change >= 0.0)
            and (down_bucket_positive_rate_change is None or down_bucket_positive_rate_change <= 0.0)
        ):
            improvements += 1
            notes.append(
                f"{horizon}: Candidate B reduced flat share without obvious deterioration in bucket quality."
            )

        if (
            up_bucket_median_change is not None
            and up_bucket_median_change < 0.0
            and down_bucket_positive_rate_change is not None
            and down_bucket_positive_rate_change > 0.0
        ):
            regressions += 1
            notes.append(
                f"{horizon}: Candidate B weakened the up bucket while making the down bucket more positive."
            )

    if improvements > 0 and regressions == 0:
        primary_finding = "candidate_b_looks_structurally_better_than_candidate_a"
        secondary_finding = "bucket_quality_and_distribution_shift_are_directionally_favorable"
    elif regressions > 0 and improvements == 0:
        primary_finding = "candidate_b_looks_structurally_worse_than_candidate_a"
        secondary_finding = "bucket_quality_regression_detected"
    else:
        primary_finding = "candidate_b_results_are_mixed_vs_candidate_a"
        secondary_finding = "comparison_requires_targeted_follow_up"

    if not notes:
        notes.append(
            "No strong structural winner emerged from the current A vs B comparison frame."
        )

    notes.append(
        "Overall positive-rate and overall future-return median are retained for completeness, but they are usually invariant across pure relabel experiments."
    )

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "notes": notes,
    }


def build_experimental_candidate_comparison_matrix(
    candidate_a_records: list[dict[str, Any]],
    candidate_b_records: list[dict[str, Any]],
    *,
    candidate_a_path: Path,
    candidate_b_path: Path,
    candidate_a_instrumentation: dict[str, int] | None = None,
    candidate_b_instrumentation: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a diagnosis-friendly Candidate A vs Candidate B comparison matrix."""
    filtered_candidate_b_records = _filter_candidate_b_records(candidate_b_records)

    candidate_a = _candidate_summary(candidate_a_records, include_volatility_metadata=False)
    candidate_b = _candidate_summary(filtered_candidate_b_records, include_volatility_metadata=True)
    delta_a_to_b = _build_delta_a_to_b(candidate_a, candidate_b)

    strategy_highlights = _group_highlights(
        candidate_a["positive_rate_by_strategy"],
        candidate_b["positive_rate_by_strategy"],
        category="strategy",
    )
    symbol_highlights = _group_highlights(
        candidate_a["positive_rate_by_symbol"],
        candidate_b["positive_rate_by_symbol"],
        category="symbol",
    )
    final_summary = _build_final_summary(candidate_a, candidate_b, delta_a_to_b)

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "comparison_name": "candidate_a_vs_b",
        },
        "inputs": {
            "candidate_a_path": str(candidate_a_path),
            "candidate_b_path": str(candidate_b_path),
            "candidate_a_parser_instrumentation": candidate_a_instrumentation or {},
            "candidate_b_parser_instrumentation": candidate_b_instrumentation or {},
            "candidate_b_filtered_row_count": len(filtered_candidate_b_records),
        },
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "delta_a_to_b": delta_a_to_b,
        "highlights": {
            "strategy_level": strategy_highlights,
            "symbol_level": symbol_highlights,
        },
        "final_summary": final_summary,
    }


def build_experimental_candidate_comparison_markdown(summary: dict[str, Any]) -> str:
    """Render a concise markdown comparison report for Candidate A vs Candidate B."""
    candidate_a = _safe_dict(summary.get("candidate_a"))
    candidate_b = _safe_dict(summary.get("candidate_b"))
    final_summary = _safe_dict(summary.get("final_summary"))
    delta_by_horizon = _safe_dict(_safe_dict(summary.get("delta_a_to_b")).get("by_horizon"))

    lines = [
        "# Candidate A vs Candidate B Comparison",
        "",
        "## Dataset Overview",
        f"- Candidate A total_row_count: {_safe_dict(candidate_a.get('dataset_overview')).get('total_row_count', 0)}",
        f"- Candidate B total_row_count: {_safe_dict(candidate_b.get('dataset_overview')).get('total_row_count', 0)}",
        f"- Candidate B fallback_row_count: {_safe_dict(candidate_b.get('volatility_metadata')).get('fallback_row_count', 0)}",
        f"- Candidate B fallback_row_ratio: {_format_pct(_safe_float(_safe_dict(candidate_b.get('volatility_metadata')).get('fallback_row_ratio')))}",
        "",
        "## Horizon Comparison",
    ]

    for horizon in TARGET_HORIZONS:
        delta = _safe_dict(delta_by_horizon.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"flat_ratio_change={_format_pct(_safe_float(delta.get('flat_ratio_change')))}, "
            f"up_ratio_change={_format_pct(_safe_float(delta.get('up_ratio_change')))}, "
            f"down_ratio_change={_format_pct(_safe_float(delta.get('down_ratio_change')))}, "
            f"up_bucket_median_change={_format_pct(_safe_float(delta.get('up_bucket_median_change')))}, "
            f"down_bucket_positive_rate_change={_format_pct(_safe_float(delta.get('down_bucket_positive_rate_change')))}"
        )

    lines.extend(["", "## Strategy-Level Bottleneck Highlights"])
    for row in _safe_dict(summary.get("highlights")).get("strategy_level", []):
        row_dict = _safe_dict(row)
        lines.append(
            f"- {row_dict.get('group', 'unknown')}: impact_score={_format_pct(_safe_float(row_dict.get('impact_score')))}"
        )

    lines.extend(["", "## Symbol-Level Highlights"])
    for row in _safe_dict(summary.get("highlights")).get("symbol_level", []):
        row_dict = _safe_dict(row)
        lines.append(
            f"- {row_dict.get('group', 'unknown')}: impact_score={_format_pct(_safe_float(row_dict.get('impact_score')))}"
        )

    lines.extend(
        [
            "",
            "## Final Summary",
            f"- primary_finding: {final_summary.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_summary.get('secondary_finding', 'unknown')}",
        ]
    )

    for note in final_summary.get("notes", []):
        lines.append(f"- note: {note}")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_comparison_matrix(
    candidate_a_path: Path = CANDIDATE_A_DEFAULT_PATH,
    candidate_b_path: Path = CANDIDATE_B_DEFAULT_PATH,
    json_output_path: Path = DEFAULT_JSON_OUTPUT,
    markdown_output_path: Path = DEFAULT_MD_OUTPUT,
) -> dict[str, Any]:
    """Build, render, and persist the Candidate A vs Candidate B comparison matrix."""
    candidate_a_records, candidate_a_instrumentation = load_jsonl_records(candidate_a_path)
    candidate_b_records, candidate_b_instrumentation = load_jsonl_records(candidate_b_path)

    summary = build_experimental_candidate_comparison_matrix(
        candidate_a_records,
        candidate_b_records,
        candidate_a_path=candidate_a_path,
        candidate_b_path=candidate_b_path,
        candidate_a_instrumentation=candidate_a_instrumentation,
        candidate_b_instrumentation=candidate_b_instrumentation,
    )
    markdown = build_experimental_candidate_comparison_markdown(summary)

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary": summary,
        "markdown": markdown,
        "json_output_path": markdown_output_path.parent / json_output_path.name if False else json_output_path,
        "markdown_output_path": markdown_output_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Candidate A vs Candidate B comparison module."""
    parser = argparse.ArgumentParser(
        description="Build a research-only comparison matrix for experimental relabel candidates A and B"
    )
    parser.add_argument("--candidate-a-path", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--candidate-b-path", type=Path, default=CANDIDATE_B_DEFAULT_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the Candidate A vs Candidate B comparison module."""
    args = parse_args()
    result = run_experimental_candidate_comparison_matrix(
        candidate_a_path=args.candidate_a_path,
        candidate_b_path=args.candidate_b_path,
        json_output_path=args.json_output,
        markdown_output_path=args.markdown_output,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
