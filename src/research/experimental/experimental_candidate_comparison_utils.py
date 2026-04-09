from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_LABELS = ("up", "down", "flat")
TARGET_CATEGORIES = ("strategy", "symbol")
MEANINGFUL_LABEL_RATIO_DELTA = 0.03
MEANINGFUL_POSITIVE_RATE_DELTA = 2.0
MEANINGFUL_MEDIAN_DELTA = 0.02


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


def _safe_ratio(numerator: int, denominator: int | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}"


def load_summary_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists() or not path.is_file():
        return {}, False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False

    if not isinstance(payload, dict):
        return {}, False

    return payload, True


def _first_non_none_int(*values: Any) -> int | None:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


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


def _normalize_metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    label_distribution = _safe_dict(payload.get("label_distribution"))
    up_count = _safe_int(label_distribution.get("up")) or 0
    down_count = _safe_int(label_distribution.get("down")) or 0
    flat_count = _safe_int(label_distribution.get("flat")) or 0

    labeled_records = _first_non_none_int(
        payload.get("labeled_records"),
        payload.get("labeled_count"),
        payload.get("sample_count"),
    )

    positive_rate = _safe_float(payload.get("positive_rate_pct"))
    if positive_rate is None:
        positive_rate = _safe_float(payload.get("up_rate_pct"))

    negative_rate = _safe_float(payload.get("negative_rate_pct"))
    if negative_rate is None:
        negative_rate = _safe_float(payload.get("down_rate_pct"))

    flat_rate = _safe_float(payload.get("flat_rate_pct"))

    return {
        "labeled_records": labeled_records,
        "label_distribution": {
            "up": up_count,
            "down": down_count,
            "flat": flat_count,
        },
        "median_future_return_pct": _safe_float(payload.get("median_future_return_pct")),
        "positive_rate_pct": positive_rate,
        "negative_rate_pct": negative_rate,
        "flat_rate_pct": flat_rate,
    }


def _extract_horizon_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    direct = _safe_dict(summary.get("horizon_summary"))
    if direct:
        return {
            horizon: _normalize_metric_payload(_safe_dict(direct.get(horizon)))
            for horizon in TARGET_HORIZONS
        }

    strategy_lab = _safe_dict(summary.get("strategy_lab"))
    performance = _safe_dict(strategy_lab.get("performance", summary.get("performance")))
    return {
        horizon: _normalize_metric_payload(_safe_dict(performance.get(horizon)))
        for horizon in TARGET_HORIZONS
    }


def _extract_group_sections(
    summary: dict[str, Any],
    category: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    direct_section = _safe_dict(summary.get(f"by_{category}"))
    if direct_section:
        output: dict[str, dict[str, dict[str, Any]]] = {
            horizon: {} for horizon in TARGET_HORIZONS
        }
        for group_name, payload in sorted(direct_section.items()):
            group_payload = _safe_dict(payload)
            horizon_summary = _safe_dict(group_payload.get("horizon_summary"))
            for horizon in TARGET_HORIZONS:
                normalized = _safe_dict(horizon_summary.get(horizon))
                if normalized:
                    output[horizon][str(group_name)] = _normalize_metric_payload(normalized)
        return output

    strategy_lab = _safe_dict(summary.get("strategy_lab"))
    comparison = _safe_dict(strategy_lab.get("comparison", summary.get("comparison")))
    output = {horizon: {} for horizon in TARGET_HORIZONS}
    bucket_name = f"by_{category}"

    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(comparison.get(horizon))
        bucket = _safe_dict(horizon_payload.get(bucket_name))
        groups = _safe_dict(bucket.get("groups"))
        for group_name, metrics in sorted(groups.items()):
            output[horizon][str(group_name)] = _normalize_metric_payload(
                _safe_dict(metrics)
            )

    return output


def _build_row_count_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_dataset = _safe_dict(baseline_summary.get("dataset_overview"))
    experiment_dataset = _safe_dict(experiment_summary.get("dataset_overview"))
    baseline_validation = _safe_dict(baseline_summary.get("schema_validation"))
    experiment_validation = _safe_dict(experiment_summary.get("schema_validation"))
    baseline_strategy_lab = _safe_dict(baseline_summary.get("strategy_lab"))
    experiment_strategy_lab = _safe_dict(experiment_summary.get("strategy_lab"))

    return {
        "total_records": _compare_numeric_values(
            _safe_int(baseline_dataset.get("total_records")),
            _safe_int(experiment_dataset.get("total_records")),
        ),
        "valid_records": _compare_numeric_values(
            _safe_int(baseline_validation.get("valid_records")),
            _safe_int(experiment_validation.get("valid_records")),
        ),
        "strategy_lab_dataset_rows": _compare_numeric_values(
            _safe_int(baseline_strategy_lab.get("dataset_rows")),
            _safe_int(experiment_strategy_lab.get("dataset_rows")),
        ),
    }


def _build_label_distribution_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = _extract_horizon_summary(baseline_summary)
    experiment = _extract_horizon_summary(experiment_summary)
    result: dict[str, Any] = {}

    for horizon in TARGET_HORIZONS:
        baseline_metrics = _safe_dict(baseline.get(horizon))
        experiment_metrics = _safe_dict(experiment.get(horizon))
        baseline_labeled = _safe_int(baseline_metrics.get("labeled_records"))
        experiment_labeled = _safe_int(experiment_metrics.get("labeled_records"))
        horizon_result: dict[str, Any] = {
            "labeled_records": _compare_numeric_values(
                baseline_labeled,
                experiment_labeled,
            )
        }

        for label in TARGET_LABELS:
            baseline_count = _safe_int(_safe_dict(baseline_metrics.get("label_distribution")).get(label)) or 0
            experiment_count = _safe_int(_safe_dict(experiment_metrics.get("label_distribution")).get(label)) or 0
            baseline_ratio = _safe_ratio(baseline_count, baseline_labeled)
            experiment_ratio = _safe_ratio(experiment_count, experiment_labeled)
            horizon_result[label] = {
                "count": _compare_numeric_values(baseline_count, experiment_count),
                "ratio": _compare_numeric_values(baseline_ratio, experiment_ratio),
            }

        result[horizon] = horizon_result

    return result


def _build_metric_comparison_by_horizon(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = _extract_horizon_summary(baseline_summary)
    experiment = _extract_horizon_summary(experiment_summary)
    result: dict[str, Any] = {}

    for horizon in TARGET_HORIZONS:
        baseline_metrics = _safe_dict(baseline.get(horizon))
        experiment_metrics = _safe_dict(experiment.get(horizon))
        result[horizon] = {
            "median_future_return_pct": _compare_numeric_values(
                _safe_float(baseline_metrics.get("median_future_return_pct")),
                _safe_float(experiment_metrics.get("median_future_return_pct")),
            ),
            "positive_rate_pct": _compare_numeric_values(
                _safe_float(baseline_metrics.get("positive_rate_pct")),
                _safe_float(experiment_metrics.get("positive_rate_pct")),
            ),
            "negative_rate_pct": _compare_numeric_values(
                _safe_float(baseline_metrics.get("negative_rate_pct")),
                _safe_float(experiment_metrics.get("negative_rate_pct")),
            ),
            "flat_rate_pct": _compare_numeric_values(
                _safe_float(baseline_metrics.get("flat_rate_pct")),
                _safe_float(experiment_metrics.get("flat_rate_pct")),
            ),
        }

    return result


def _build_category_level_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    category: str,
) -> dict[str, list[dict[str, Any]]]:
    baseline = _extract_group_sections(baseline_summary, category)
    experiment = _extract_group_sections(experiment_summary, category)
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in TARGET_HORIZONS:
        rows: list[dict[str, Any]] = []
        group_names = sorted(
            set(_safe_dict(baseline.get(horizon)).keys())
            | set(_safe_dict(experiment.get(horizon)).keys())
        )
        for group_name in group_names:
            baseline_metrics = _safe_dict(_safe_dict(baseline.get(horizon)).get(group_name))
            experiment_metrics = _safe_dict(_safe_dict(experiment.get(horizon)).get(group_name))
            rows.append(
                {
                    "group": group_name,
                    "labeled_records": _compare_numeric_values(
                        _safe_int(baseline_metrics.get("labeled_records")),
                        _safe_int(experiment_metrics.get("labeled_records")),
                    ),
                    "median_future_return_pct": _compare_numeric_values(
                        _safe_float(baseline_metrics.get("median_future_return_pct")),
                        _safe_float(experiment_metrics.get("median_future_return_pct")),
                    ),
                    "positive_rate_pct": _compare_numeric_values(
                        _safe_float(baseline_metrics.get("positive_rate_pct")),
                        _safe_float(experiment_metrics.get("positive_rate_pct")),
                    ),
                    "flat_rate_pct": _compare_numeric_values(
                        _safe_float(baseline_metrics.get("flat_rate_pct")),
                        _safe_float(experiment_metrics.get("flat_rate_pct")),
                    ),
                }
            )

        rows.sort(
            key=lambda row: (
                -abs(_safe_float(_safe_dict(row.get("positive_rate_pct")).get("delta")) or 0.0),
                -abs(_safe_float(_safe_dict(row.get("median_future_return_pct")).get("delta")) or 0.0),
                str(row.get("group")),
            )
        )
        result[horizon] = rows[:5]

    return result


def _coverage_and_purity_votes(
    label_distribution: dict[str, Any],
    metric_comparison: dict[str, Any],
) -> tuple[int, int, int, int]:
    coverage_improved = 0
    coverage_lost = 0
    purity_improved = 0
    purity_lost = 0

    for horizon in TARGET_HORIZONS:
        distribution_payload = _safe_dict(label_distribution.get(horizon))
        metric_payload = _safe_dict(metric_comparison.get(horizon))

        flat_ratio_delta = _safe_float(
            _safe_dict(_safe_dict(distribution_payload.get("flat")).get("ratio")).get("delta")
        )
        labeled_delta = _safe_float(
            _safe_dict(distribution_payload.get("labeled_records")).get("delta")
        )
        positive_rate_delta = _safe_float(
            _safe_dict(metric_payload.get("positive_rate_pct")).get("delta")
        )
        median_delta = _safe_float(
            _safe_dict(metric_payload.get("median_future_return_pct")).get("delta")
        )

        if (
            (flat_ratio_delta is not None and flat_ratio_delta <= -MEANINGFUL_LABEL_RATIO_DELTA)
            or (labeled_delta is not None and labeled_delta > 0.0)
        ):
            coverage_improved += 1
        if (
            (flat_ratio_delta is not None and flat_ratio_delta >= MEANINGFUL_LABEL_RATIO_DELTA)
            or (labeled_delta is not None and labeled_delta < 0.0)
        ):
            coverage_lost += 1
        if (
            (positive_rate_delta is not None and positive_rate_delta >= MEANINGFUL_POSITIVE_RATE_DELTA)
            or (median_delta is not None and median_delta >= MEANINGFUL_MEDIAN_DELTA)
        ):
            purity_improved += 1
        if (
            (positive_rate_delta is not None and positive_rate_delta <= -MEANINGFUL_POSITIVE_RATE_DELTA)
            or (median_delta is not None and median_delta <= -MEANINGFUL_MEDIAN_DELTA)
        ):
            purity_lost += 1

    return coverage_improved, coverage_lost, purity_improved, purity_lost


def _build_comparability(
    *,
    baseline_loaded: bool,
    experiment_loaded: bool,
    row_count_comparison: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_loaded or not experiment_loaded:
        return {
            "status": "input_missing",
            "reason": "baseline_or_experiment_summary_not_loaded",
            "directional_only": True,
        }

    total_records_delta = _safe_float(
        _safe_dict(row_count_comparison.get("total_records")).get("delta")
    )
    valid_records_delta = _safe_float(
        _safe_dict(row_count_comparison.get("valid_records")).get("delta")
    )
    dataset_rows_delta = _safe_float(
        _safe_dict(row_count_comparison.get("strategy_lab_dataset_rows")).get("delta")
    )

    mismatched_keys: list[str] = []
    if total_records_delta not in (None, 0.0):
        mismatched_keys.append("total_records")
    if valid_records_delta not in (None, 0.0):
        mismatched_keys.append("valid_records")
    if dataset_rows_delta not in (None, 0.0):
        mismatched_keys.append("strategy_lab_dataset_rows")

    if mismatched_keys:
        return {
            "status": "row_count_mismatch",
            "reason": "summary_row_counts_do_not_match",
            "directional_only": True,
            "mismatched_keys": mismatched_keys,
        }

    return {
        "status": "comparable",
        "reason": "loaded_and_row_counts_match",
        "directional_only": False,
        "mismatched_keys": [],
    }


def _build_final_diagnosis(
    *,
    mode: str,
    baseline_name: str,
    experiment_name: str,
    baseline_loaded: bool,
    experiment_loaded: bool,
    comparability: dict[str, Any],
    label_distribution_by_horizon: dict[str, Any],
    metric_comparison_by_horizon: dict[str, Any],
) -> dict[str, Any]:
    if not baseline_loaded or not experiment_loaded:
        return {
            "primary_finding": "comparison_unavailable_missing_input_summary",
            "secondary_finding": "at_least_one_summary_failed_to_load",
            "notes": [
                f"baseline_loaded={baseline_loaded}, experiment_loaded={experiment_loaded}.",
                "Interpretation is unavailable until both summaries load successfully.",
            ],
        }

    comparability_status = _safe_text(comparability.get("status"))
    mismatched_keys = _safe_list(comparability.get("mismatched_keys"))

    coverage_improved, coverage_lost, purity_improved, purity_lost = _coverage_and_purity_votes(
        label_distribution_by_horizon,
        metric_comparison_by_horizon,
    )

    notes: list[str] = []

    if comparability_status == "row_count_mismatch":
        return {
            "primary_finding": "comparison_is_directional_only_due_to_row_count_mismatch",
            "secondary_finding": "not_apples_to_apples",
            "notes": [
                f"Row-count mismatch detected across: {', '.join(str(key) for key in mismatched_keys) or 'unknown'}.",
                f"{baseline_name} vs {experiment_name} should not be treated as a final selection-readiness comparison until row alignment is confirmed.",
                "Use this report as directional context only.",
            ],
        }

    if mode == "a_vs_c":
        if purity_improved > 0 and coverage_lost == 0:
            primary = "candidate_c_looks_more_balanced_than_candidate_a"
            secondary = "purity_improved_without_clear_coverage_loss"
        elif purity_improved > 0 and coverage_lost > 0:
            primary = "candidate_c_trades_some_coverage_for_purity_vs_candidate_a"
            secondary = "middle_path_signal_is_present_but_not_free"
        elif coverage_lost > 0 and purity_improved == 0:
            primary = "candidate_c_looks_less_seed_friendly_than_candidate_a"
            secondary = "coverage_loss_without_clear_purity_payoff"
        else:
            primary = "candidate_c_remains_mixed_vs_candidate_a"
            secondary = "coverage_vs_purity_tradeoff_is_still_unresolved"

        notes.append(
            f"{experiment_name} should only be considered favorable versus {baseline_name} if any purity gain is meaningful enough to justify seed coverage changes in a 0-selection system."
        )
    else:
        if coverage_improved > 0 and purity_lost == 0:
            primary = "candidate_c_recovers_coverage_without_clear_purity_loss_vs_candidate_b"
            secondary = "candidate_c_looks_more_seed_friendly_than_candidate_b"
        elif coverage_improved > 0 and purity_lost > 0:
            primary = "candidate_c_recovers_coverage_but_gives_back_some_purity_vs_candidate_b"
            secondary = "middle_path_tradeoff_is_present"
        elif purity_lost > 0 and coverage_improved == 0:
            primary = "candidate_c_looks_weaker_than_candidate_b"
            secondary = "purity_loss_without_coverage_recovery"
        else:
            primary = "candidate_c_remains_mixed_vs_candidate_b"
            secondary = "coverage_vs_purity_tradeoff_is_still_unresolved"

        notes.append(
            f"{experiment_name} should only beat {baseline_name} for the current seed-starved system if any coverage recovery does not materially erode the purity signals that made {baseline_name} interesting."
        )

    notes.append(
        "Flat-ratio changes are treated as coverage context, not as a standalone win condition."
    )
    notes.append(
        f"Vote summary: coverage_improved={coverage_improved}, coverage_lost={coverage_lost}, purity_improved={purity_improved}, purity_lost={purity_lost}."
    )

    return {
        "primary_finding": primary,
        "secondary_finding": secondary,
        "notes": notes,
    }


def build_candidate_summary_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    baseline_name: str,
    experiment_name: str,
    baseline_summary_path: Path,
    experiment_summary_path: Path,
    mode: str,
    baseline_loaded: bool = True,
    experiment_loaded: bool = True,
) -> dict[str, Any]:
    row_count_comparison = _build_row_count_comparison(
        baseline_summary,
        experiment_summary,
    )
    label_distribution_by_horizon = _build_label_distribution_comparison(
        baseline_summary,
        experiment_summary,
    )
    metric_comparison_by_horizon = _build_metric_comparison_by_horizon(
        baseline_summary,
        experiment_summary,
    )
    comparability = _build_comparability(
        baseline_loaded=baseline_loaded,
        experiment_loaded=experiment_loaded,
        row_count_comparison=row_count_comparison,
    )
    final_diagnosis = _build_final_diagnosis(
        mode=mode,
        baseline_name=baseline_name,
        experiment_name=experiment_name,
        baseline_loaded=baseline_loaded,
        experiment_loaded=experiment_loaded,
        comparability=comparability,
        label_distribution_by_horizon=label_distribution_by_horizon,
        metric_comparison_by_horizon=metric_comparison_by_horizon,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "experimental_candidate_summary_comparison",
            "baseline_name": baseline_name,
            "experiment_name": experiment_name,
            "baseline_loaded": baseline_loaded,
            "experiment_loaded": experiment_loaded,
        },
        "inputs": {
            "baseline_summary_path": str(baseline_summary_path),
            "experiment_summary_path": str(experiment_summary_path),
        },
        "comparability": comparability,
        "row_count_comparison": row_count_comparison,
        "label_distribution_by_horizon": label_distribution_by_horizon,
        "metric_comparison_by_horizon": metric_comparison_by_horizon,
        "category_level_metrics": {
            category: _build_category_level_comparison(
                baseline_summary,
                experiment_summary,
                category=category,
            )
            for category in TARGET_CATEGORIES
        },
        "final_diagnosis": final_diagnosis,
    }


def build_candidate_summary_comparison_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    inputs = _safe_dict(summary.get("inputs"))
    comparability = _safe_dict(summary.get("comparability"))
    row_counts = _safe_dict(summary.get("row_count_comparison"))
    label_distribution = _safe_dict(summary.get("label_distribution_by_horizon"))
    metrics = _safe_dict(summary.get("metric_comparison_by_horizon"))
    categories = _safe_dict(summary.get("category_level_metrics"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines = [
        f"# {metadata.get('baseline_name', 'baseline')} vs {metadata.get('experiment_name', 'experiment')} Comparison",
        "",
        "## Inputs",
        f"- baseline_summary_path: {inputs.get('baseline_summary_path', 'n/a')}",
        f"- experiment_summary_path: {inputs.get('experiment_summary_path', 'n/a')}",
        f"- baseline_loaded: {metadata.get('baseline_loaded', False)}",
        f"- experiment_loaded: {metadata.get('experiment_loaded', False)}",
        "",
        "## Comparability",
        f"- status: {comparability.get('status', 'unknown')}",
        f"- reason: {comparability.get('reason', 'unknown')}",
        f"- directional_only: {comparability.get('directional_only', True)}",
        "",
        "## Row Counts",
    ]

    mismatched_keys = _safe_list(comparability.get("mismatched_keys"))
    if mismatched_keys:
        lines.append(f"- mismatched_keys: {', '.join(str(key) for key in mismatched_keys)}")

    for key, payload in row_counts.items():
        payload_dict = _safe_dict(payload)
        lines.append(
            f"- {key}: baseline={_format_number(_safe_float(payload_dict.get('baseline')))}, "
            f"experiment={_format_number(_safe_float(payload_dict.get('experiment')))}, "
            f"delta={_format_number(_safe_float(payload_dict.get('delta')))}"
        )

    lines.extend(["", "## Label Distribution By Horizon"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(label_distribution.get(horizon))
        flat_ratio_delta = _safe_float(
            _safe_dict(_safe_dict(payload.get("flat")).get("ratio")).get("delta")
        )
        up_ratio_delta = _safe_float(
            _safe_dict(_safe_dict(payload.get("up")).get("ratio")).get("delta")
        )
        down_ratio_delta = _safe_float(
            _safe_dict(_safe_dict(payload.get("down")).get("ratio")).get("delta")
        )
        labeled_delta = _safe_float(_safe_dict(payload.get("labeled_records")).get("delta"))
        lines.append(
            f"- {horizon}: labeled_records_delta={_format_number(labeled_delta)}, "
            f"up_ratio_delta={_format_pct(up_ratio_delta)}, "
            f"down_ratio_delta={_format_pct(down_ratio_delta)}, "
            f"flat_ratio_delta={_format_pct(flat_ratio_delta)}"
        )

    lines.extend(["", "## Metric Comparison By Horizon"])
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(metrics.get(horizon))
        lines.append(
            f"- {horizon}: median_delta={_format_pct(_safe_float(_safe_dict(payload.get('median_future_return_pct')).get('delta')))}, "
            f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('positive_rate_pct')).get('delta')))}, "
            f"negative_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('negative_rate_pct')).get('delta')))}, "
            f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('flat_rate_pct')).get('delta')))}"
        )

    for category in TARGET_CATEGORIES:
        lines.extend(["", f"## {category.title()} Highlights"])
        category_payload = _safe_dict(categories.get(category))
        for horizon in TARGET_HORIZONS:
            rows = _safe_list(category_payload.get(horizon))
            if not rows:
                lines.append(f"- {horizon}: none")
                continue
            for row in rows:
                row_payload = _safe_dict(row)
                lines.append(
                    f"- {horizon} {row_payload.get('group', 'unknown')}: "
                    f"median_delta={_format_pct(_safe_float(_safe_dict(row_payload.get('median_future_return_pct')).get('delta')))}, "
                    f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(row_payload.get('positive_rate_pct')).get('delta')))}, "
                    f"labeled_records_delta={_format_number(_safe_float(_safe_dict(row_payload.get('labeled_records')).get('delta')))}"
                )

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}",
            f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}",
        ]
    )
    for note in _safe_list(final_diagnosis.get("notes")):
        lines.append(f"- note: {note}")

    return "\n".join(lines).strip() + "\n"


def write_candidate_summary_comparison(
    summary: dict[str, Any],
    *,
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, str]:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.write_text(
        build_candidate_summary_comparison_markdown(summary),
        encoding="utf-8",
    )

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }
