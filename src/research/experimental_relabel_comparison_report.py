from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_BASELINE_PATH = Path("logs/research_reports/cumulative/summary.json")
DEFAULT_EXPERIMENT_PATH = Path(
    "logs/research_reports/experiments/candidate_a/latest/summary.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/experimental_relabel_comparison_summary.json"
)
DEFAULT_MD_OUTPUT = Path(
    "logs/research_reports/experiments/candidate_a/experimental_relabel_comparison_summary.md"
)

TARGET_HORIZONS = ("15m", "1h", "4h")
TARGET_STABILITY_CATEGORIES = ("strategy", "symbol", "alignment_state")
MARKDOWN_GROUP_HIGHLIGHT_LIMIT = 5
PRIORITY_SYMBOLS = ("BTCUSDT", "ETHUSDT")
SELECTION_GRADE_POSITIVE_RATE_PCT = 55.0
MEANINGFUL_FLAT_REDUCTION_PCT = -5.0
MEANINGFUL_MEDIAN_RECOVERY_PCT = 0.02
WEAK_CANDIDATE_STRENGTHS = {"weak", "insufficient_data", "n/a", "unknown", "none", ""}


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
    # Counts may arrive as numeric strings or floats from loosely typed JSON payloads.
    # We intentionally coerce them to integers for comparison reporting.
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


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}"


def _load_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists() or not path.is_file():
        return {}, False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False

    if not isinstance(payload, dict):
        return {}, False

    return payload, True


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


def _compare_text_values(
    baseline: str | None,
    experiment: str | None,
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "experiment": experiment,
        "changed": baseline != experiment,
    }


def _compare_list_values(
    baseline: list[str],
    experiment: list[str],
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "experiment": experiment,
        "changed": baseline != experiment,
    }


def _first_non_none_int(*values: Any) -> int | None:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


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
        "avg_future_return_pct": _safe_float(payload.get("avg_future_return_pct")),
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
                if _safe_dict(horizon_summary.get(horizon)):
                    output[horizon][str(group_name)] = _normalize_metric_payload(
                        _safe_dict(horizon_summary.get(horizon))
                    )
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


def _extract_edge_candidates_preview(
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    preview = _safe_dict(summary.get("edge_candidates_preview"))
    by_horizon = _safe_dict(preview.get("by_horizon", preview))
    return {
        horizon: _safe_dict(by_horizon.get(horizon))
        for horizon in TARGET_HORIZONS
    }


def _extract_edge_stability_preview(
    summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    preview = _safe_dict(summary.get("edge_stability_preview"))
    return {
        category: _safe_dict(preview.get(category))
        for category in TARGET_STABILITY_CATEGORIES
    }


def _build_horizon_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    baseline = _extract_horizon_summary(baseline_summary)
    experiment = _extract_horizon_summary(experiment_summary)
    result: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        baseline_metrics = _safe_dict(baseline.get(horizon))
        experiment_metrics = _safe_dict(experiment.get(horizon))
        result[horizon] = {
            "labeled_records": _compare_numeric_values(
                _safe_int(baseline_metrics.get("labeled_records")),
                _safe_int(experiment_metrics.get("labeled_records")),
            ),
            "label_distribution": {
                label: _compare_numeric_values(
                    _safe_int(_safe_dict(baseline_metrics.get("label_distribution")).get(label)),
                    _safe_int(_safe_dict(experiment_metrics.get("label_distribution")).get(label)),
                )
                for label in ("up", "down", "flat")
            },
            "avg_future_return_pct": _compare_numeric_values(
                _safe_float(baseline_metrics.get("avg_future_return_pct")),
                _safe_float(experiment_metrics.get("avg_future_return_pct")),
            ),
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


def _group_priority(category: str, group: str) -> int:
    if category == "symbol" and str(group).upper() in PRIORITY_SYMBOLS:
        return 0
    return 1


def _build_group_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    category: str,
) -> dict[str, list[dict[str, Any]]]:
    baseline = _extract_group_sections(baseline_summary, category)
    experiment = _extract_group_sections(experiment_summary, category)
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in TARGET_HORIZONS:
        group_names = sorted(
            set(_safe_dict(baseline.get(horizon)).keys())
            | set(_safe_dict(experiment.get(horizon)).keys())
        )
        rows: list[dict[str, Any]] = []
        for group_name in group_names:
            baseline_metrics = _safe_dict(_safe_dict(baseline.get(horizon)).get(group_name))
            experiment_metrics = _safe_dict(_safe_dict(experiment.get(horizon)).get(group_name))
            row = {
                "group": group_name,
                "labeled_records": _compare_numeric_values(
                    _safe_int(baseline_metrics.get("labeled_records")),
                    _safe_int(experiment_metrics.get("labeled_records")),
                ),
                "label_distribution": {
                    label: _compare_numeric_values(
                        _safe_int(_safe_dict(baseline_metrics.get("label_distribution")).get(label)),
                        _safe_int(_safe_dict(experiment_metrics.get("label_distribution")).get(label)),
                    )
                    for label in ("up", "down", "flat")
                },
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
            rows.append(row)

        rows.sort(
            key=lambda item: (
                _group_priority(category, str(item["group"])),
                -abs(
                    _safe_float(
                        _safe_dict(item.get("median_future_return_pct")).get("delta")
                    )
                    or 0.0
                ),
                _safe_float(_safe_dict(item.get("flat_rate_pct")).get("delta")) or 0.0,
                str(item["group"]),
            )
        )
        result[horizon] = rows

    return result


def _build_edge_candidates_preview_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    baseline = _extract_edge_candidates_preview(baseline_summary)
    experiment = _extract_edge_candidates_preview(experiment_summary)
    result: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        baseline_horizon = _safe_dict(baseline.get(horizon))
        experiment_horizon = _safe_dict(experiment.get(horizon))
        horizon_result: dict[str, Any] = {
            "sample_gate": _compare_text_values(
                _safe_text(baseline_horizon.get("sample_gate")),
                _safe_text(experiment_horizon.get("sample_gate")),
            ),
            "quality_gate": _compare_text_values(
                _safe_text(baseline_horizon.get("quality_gate")),
                _safe_text(experiment_horizon.get("quality_gate")),
            ),
            "candidate_strength": _compare_text_values(
                _safe_text(baseline_horizon.get("candidate_strength")),
                _safe_text(experiment_horizon.get("candidate_strength")),
            ),
            "visibility_reason": _compare_text_values(
                _safe_text(baseline_horizon.get("visibility_reason")),
                _safe_text(experiment_horizon.get("visibility_reason")),
            ),
        }

        for candidate_key in ("top_strategy", "top_symbol", "top_alignment_state"):
            baseline_candidate = _safe_dict(baseline_horizon.get(candidate_key))
            experiment_candidate = _safe_dict(experiment_horizon.get(candidate_key))
            horizon_result[candidate_key] = {
                "sample_gate": _compare_text_values(
                    _safe_text(baseline_candidate.get("sample_gate")),
                    _safe_text(experiment_candidate.get("sample_gate")),
                ),
                "quality_gate": _compare_text_values(
                    _safe_text(baseline_candidate.get("quality_gate")),
                    _safe_text(experiment_candidate.get("quality_gate")),
                ),
                "candidate_strength": _compare_text_values(
                    _safe_text(baseline_candidate.get("candidate_strength")),
                    _safe_text(experiment_candidate.get("candidate_strength")),
                ),
                "visibility_reason": _compare_text_values(
                    _safe_text(baseline_candidate.get("visibility_reason")),
                    _safe_text(experiment_candidate.get("visibility_reason")),
                ),
                "group": _compare_text_values(
                    _safe_text(baseline_candidate.get("group")),
                    _safe_text(experiment_candidate.get("group")),
                ),
                "median_future_return_pct": _compare_numeric_values(
                    _safe_float(baseline_candidate.get("median_future_return_pct")),
                    _safe_float(experiment_candidate.get("median_future_return_pct")),
                ),
                "positive_rate_pct": _compare_numeric_values(
                    _safe_float(baseline_candidate.get("positive_rate_pct")),
                    _safe_float(experiment_candidate.get("positive_rate_pct")),
                ),
            }
        result[horizon] = horizon_result

    return result


def _normalize_visible_horizons(value: Any) -> list[str]:
    horizons: list[str] = []
    for item in _safe_list(value):
        text = _safe_text(item)
        if text is not None and text not in horizons:
            horizons.append(text)
    return horizons


def _build_edge_stability_preview_comparison(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    baseline = _extract_edge_stability_preview(baseline_summary)
    experiment = _extract_edge_stability_preview(experiment_summary)
    result: dict[str, dict[str, Any]] = {}

    for category in TARGET_STABILITY_CATEGORIES:
        baseline_category = _safe_dict(baseline.get(category))
        experiment_category = _safe_dict(experiment.get(category))
        result[category] = {
            "group": _compare_text_values(
                _safe_text(baseline_category.get("group")),
                _safe_text(experiment_category.get("group")),
            ),
            "visible_horizons": _compare_list_values(
                _normalize_visible_horizons(baseline_category.get("visible_horizons")),
                _normalize_visible_horizons(experiment_category.get("visible_horizons")),
            ),
            "stability_label": _compare_text_values(
                _safe_text(baseline_category.get("stability_label")),
                _safe_text(experiment_category.get("stability_label")),
            ),
            "stability_score": _compare_numeric_values(
                _safe_float(baseline_category.get("stability_score")),
                _safe_float(experiment_category.get("stability_score")),
            ),
            "visibility_reason": _compare_text_values(
                _safe_text(baseline_category.get("visibility_reason")),
                _safe_text(experiment_category.get("visibility_reason")),
            ),
        }

    return result


def _has_meaningful_group_improvement(
    group_comparison: dict[str, list[dict[str, Any]]],
) -> bool:
    for rows in group_comparison.values():
        for row in rows:
            median_delta = _safe_float(
                _safe_dict(row.get("median_future_return_pct")).get("delta")
            )
            flat_delta = _safe_float(
                _safe_dict(row.get("flat_rate_pct")).get("delta")
            )
            if (
                median_delta is not None
                and median_delta >= MEANINGFUL_MEDIAN_RECOVERY_PCT
            ) or (
                flat_delta is not None
                and flat_delta <= MEANINGFUL_FLAT_REDUCTION_PCT
            ):
                return True
    return False


def _candidate_visibility_preserved(
    stability_comparison: dict[str, dict[str, Any]],
) -> bool:
    for category in TARGET_STABILITY_CATEGORIES:
        payload = _safe_dict(stability_comparison.get(category))
        baseline_visible = _safe_list(
            _safe_dict(payload.get("visible_horizons")).get("baseline")
        )
        experiment_visible = _safe_list(
            _safe_dict(payload.get("visible_horizons")).get("experiment")
        )
        if baseline_visible and not experiment_visible:
            return False
    return True


def _edge_strength_still_weak(
    edge_preview_comparison: dict[str, dict[str, Any]],
) -> bool:
    strengths: list[str] = []
    for horizon_payload in edge_preview_comparison.values():
        experiment_strength = _safe_text(
            _safe_dict(horizon_payload.get("candidate_strength")).get("experiment")
        )
        if experiment_strength is not None:
            strengths.append(experiment_strength.lower())

        for candidate_key in ("top_strategy", "top_symbol", "top_alignment_state"):
            experiment_strength = _safe_text(
                _safe_dict(
                    _safe_dict(horizon_payload.get(candidate_key)).get(
                        "candidate_strength"
                    )
                ).get("experiment")
            )
            if experiment_strength is not None:
                strengths.append(experiment_strength.lower())

    if not strengths:
        return True

    return all(strength in WEAK_CANDIDATE_STRENGTHS for strength in strengths)


def _positive_rate_still_below_selection_grade(
    horizon_comparison: dict[str, dict[str, Any]],
) -> bool:
    available = [
        _safe_float(_safe_dict(payload.get("positive_rate_pct")).get("experiment"))
        for payload in horizon_comparison.values()
    ]
    available = [value for value in available if value is not None]
    if not available:
        return True
    return all(value < SELECTION_GRADE_POSITIVE_RATE_PCT for value in available)


def _build_final_diagnosis(
    horizon_comparison: dict[str, dict[str, Any]],
    by_symbol_comparison: dict[str, list[dict[str, Any]]],
    by_strategy_comparison: dict[str, list[dict[str, Any]]],
    edge_preview_comparison: dict[str, dict[str, Any]],
    stability_comparison: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels: list[str] = []

    flat_reduction_horizons = [
        horizon
        for horizon, payload in horizon_comparison.items()
        if (_safe_float(_safe_dict(payload.get("flat_rate_pct")).get("delta")) or 0.0)
        <= MEANINGFUL_FLAT_REDUCTION_PCT
    ]
    median_recovery_horizons = [
        horizon
        for horizon, payload in horizon_comparison.items()
        if (_safe_float(_safe_dict(payload.get("median_future_return_pct")).get("delta")) or 0.0)
        >= MEANINGFUL_MEDIAN_RECOVERY_PCT
    ]

    if flat_reduction_horizons:
        labels.append("meaningful_flat_suppression_reduction_detected")
    if len(median_recovery_horizons) >= 2:
        labels.append("median_recovery_detected_across_horizons")
    if any(
        horizon in {"15m", "1h"}
        for horizon in set(flat_reduction_horizons + median_recovery_horizons)
    ):
        labels.append("short_horizon_recovery_detected")
    if _candidate_visibility_preserved(stability_comparison):
        labels.append("candidate_visibility_preserved_after_relabel")
    if _edge_strength_still_weak(edge_preview_comparison):
        labels.append("edge_strength_still_weak")
    if _positive_rate_still_below_selection_grade(horizon_comparison):
        labels.append("positive_rate_still_below_selection_grade")
    if _has_meaningful_group_improvement(by_symbol_comparison):
        labels.append("symbol_level_improvement_present")
    if _has_meaningful_group_improvement(by_strategy_comparison):
        labels.append("strategy_level_improvement_present")

    improvement_detected = any(
        label in labels
        for label in (
            "meaningful_flat_suppression_reduction_detected",
            "median_recovery_detected_across_horizons",
            "short_horizon_recovery_detected",
        )
    )
    not_selection_ready = (
        "edge_strength_still_weak" in labels
        or "positive_rate_still_below_selection_grade" in labels
    )
    if improvement_detected and not_selection_ready:
        labels.append("experiment_improved_distribution_but_not_selection_ready")

    if "experiment_improved_distribution_but_not_selection_ready" in labels:
        primary_finding = "experiment_improved_distribution_but_not_selection_ready"
    elif "meaningful_flat_suppression_reduction_detected" in labels:
        primary_finding = "meaningful_flat_suppression_reduction_detected"
    elif "edge_strength_still_weak" in labels:
        primary_finding = "edge_strength_still_weak"
    else:
        primary_finding = "comparison_inconclusive"

    if "median_recovery_detected_across_horizons" in labels:
        secondary_finding = "median_recovery_detected_across_horizons"
    elif "candidate_visibility_preserved_after_relabel" in labels:
        secondary_finding = "candidate_visibility_preserved_after_relabel"
    elif labels:
        secondary_finding = labels[0]
    else:
        secondary_finding = "comparison_inconclusive"

    if not labels:
        labels.append("comparison_inconclusive")

    summary = (
        f"flat_reduction_horizons={flat_reduction_horizons or ['none']}, "
        f"median_recovery_horizons={median_recovery_horizons or ['none']}, "
        f"visibility_preserved={_candidate_visibility_preserved(stability_comparison)}, "
        f"edge_strength_still_weak={_edge_strength_still_weak(edge_preview_comparison)}, "
        f"positive_rate_below_selection_grade={_positive_rate_still_below_selection_grade(horizon_comparison)}."
    )

    return {
        "primary_finding": primary_finding,
        "secondary_finding": secondary_finding,
        "diagnosis_labels": labels,
        "summary": summary,
    }


def build_experimental_relabel_comparison_report(
    baseline_summary: dict[str, Any],
    experiment_summary: dict[str, Any],
    *,
    baseline_path: Path,
    experiment_path: Path,
    baseline_loaded: bool = True,
    experiment_loaded: bool = True,
) -> dict[str, Any]:
    horizon_comparison = _build_horizon_comparison(baseline_summary, experiment_summary)
    by_symbol_comparison = _build_group_comparison(
        baseline_summary,
        experiment_summary,
        category="symbol",
    )
    by_strategy_comparison = _build_group_comparison(
        baseline_summary,
        experiment_summary,
        category="strategy",
    )
    edge_preview_comparison = _build_edge_candidates_preview_comparison(
        baseline_summary,
        experiment_summary,
    )
    stability_comparison = _build_edge_stability_preview_comparison(
        baseline_summary,
        experiment_summary,
    )
    final_diagnosis = _build_final_diagnosis(
        horizon_comparison=horizon_comparison,
        by_symbol_comparison=by_symbol_comparison,
        by_strategy_comparison=by_strategy_comparison,
        edge_preview_comparison=edge_preview_comparison,
        stability_comparison=stability_comparison,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "experimental_relabel_comparison_report",
            "baseline_loaded": baseline_loaded,
            "experiment_loaded": experiment_loaded,
        },
        "source_targeting": {
            "baseline_summary_path": str(baseline_path),
            "experiment_summary_path": str(experiment_path),
        },
        "horizon_comparison": horizon_comparison,
        "by_symbol_comparison": by_symbol_comparison,
        "by_strategy_comparison": by_strategy_comparison,
        "edge_candidates_preview_comparison": edge_preview_comparison,
        "edge_stability_preview_comparison": stability_comparison,
        "final_diagnosis": final_diagnosis,
    }


def _highlight_rows(
    section: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    highlights: dict[str, list[dict[str, Any]]] = {}
    for horizon in TARGET_HORIZONS:
        rows = _safe_list(section.get(horizon))
        highlights[horizon] = rows[:MARKDOWN_GROUP_HIGHLIGHT_LIMIT]
    return highlights


def build_experimental_relabel_comparison_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    source_targeting = _safe_dict(summary.get("source_targeting"))
    horizon_comparison = _safe_dict(summary.get("horizon_comparison"))
    symbol_highlights = _highlight_rows(_safe_dict(summary.get("by_symbol_comparison")))
    strategy_highlights = _highlight_rows(
        _safe_dict(summary.get("by_strategy_comparison"))
    )
    edge_preview = _safe_dict(summary.get("edge_candidates_preview_comparison"))
    stability = _safe_dict(summary.get("edge_stability_preview_comparison"))
    final_diagnosis = _safe_dict(summary.get("final_diagnosis"))

    lines: list[str] = []
    lines.append("Experimental Relabel Comparison Report")
    lines.append(f"Generated: {metadata.get('generated_at', 'unknown')}")
    lines.append("")
    lines.append("Input Paths")
    lines.append(
        f"- baseline_summary_path: {source_targeting.get('baseline_summary_path', 'unknown')}"
    )
    lines.append(
        f"- experiment_summary_path: {source_targeting.get('experiment_summary_path', 'unknown')}"
    )
    lines.append(f"- baseline_loaded: {metadata.get('baseline_loaded', False)}")
    lines.append(f"- experiment_loaded: {metadata.get('experiment_loaded', False)}")
    lines.append("")
    lines.append("High-Level Comparison Summary")
    lines.append(f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}")
    lines.append(
        f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}"
    )
    lines.append(
        f"- diagnosis_labels: {', '.join(_safe_list(final_diagnosis.get('diagnosis_labels'))) or 'none'}"
    )
    lines.append(f"- summary: {final_diagnosis.get('summary', 'n/a')}")
    lines.append("")
    lines.append("Horizon Comparison")
    for horizon in TARGET_HORIZONS:
        payload = _safe_dict(horizon_comparison.get(horizon))
        lines.append(
            f"- {horizon}: labeled_records_delta={_format_number(_safe_float(_safe_dict(payload.get('labeled_records')).get('delta')))}, "
            f"median_delta={_format_pct(_safe_float(_safe_dict(payload.get('median_future_return_pct')).get('delta')))}, "
            f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('positive_rate_pct')).get('delta')))}, "
            f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('flat_rate_pct')).get('delta')))}"
        )
    lines.append("")
    lines.append("By Symbol Highlights")
    for horizon in TARGET_HORIZONS:
        lines.append(f"- {horizon}:")
        rows = symbol_highlights[horizon]
        if not rows:
            lines.append("  - none")
            continue
        for row in rows:
            lines.append(
                "  - "
                f"group={row['group']}, "
                f"median_delta={_format_pct(_safe_float(_safe_dict(row.get('median_future_return_pct')).get('delta')))}, "
                f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(row.get('positive_rate_pct')).get('delta')))}, "
                f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(row.get('flat_rate_pct')).get('delta')))}, "
                f"labeled_records_delta={_format_number(_safe_float(_safe_dict(row.get('labeled_records')).get('delta')))}"
            )
    lines.append("")
    lines.append("By Strategy Highlights")
    for horizon in TARGET_HORIZONS:
        lines.append(f"- {horizon}:")
        rows = strategy_highlights[horizon]
        if not rows:
            lines.append("  - none")
            continue
        for row in rows:
            lines.append(
                "  - "
                f"group={row['group']}, "
                f"median_delta={_format_pct(_safe_float(_safe_dict(row.get('median_future_return_pct')).get('delta')))}, "
                f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(row.get('positive_rate_pct')).get('delta')))}, "
                f"flat_rate_delta={_format_pct(_safe_float(_safe_dict(row.get('flat_rate_pct')).get('delta')))}, "
                f"labeled_records_delta={_format_number(_safe_float(_safe_dict(row.get('labeled_records')).get('delta')))}"
            )
    lines.append("")
    lines.append("Edge Preview Comparison")
    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(edge_preview.get(horizon))
        lines.append(
            f"- {horizon}: sample_gate={_safe_dict(horizon_payload.get('sample_gate')).get('experiment', 'n/a')}, "
            f"quality_gate={_safe_dict(horizon_payload.get('quality_gate')).get('experiment', 'n/a')}, "
            f"candidate_strength={_safe_dict(horizon_payload.get('candidate_strength')).get('experiment', 'n/a')}, "
            f"visibility_reason={_safe_dict(horizon_payload.get('visibility_reason')).get('experiment', 'n/a')}"
        )
        for candidate_key in ("top_strategy", "top_symbol", "top_alignment_state"):
            payload = _safe_dict(horizon_payload.get(candidate_key))
            lines.append(
                "  - "
                f"{candidate_key}: group={_safe_dict(payload.get('group')).get('experiment', 'n/a')}, "
                f"candidate_strength={_safe_dict(payload.get('candidate_strength')).get('experiment', 'n/a')}, "
                f"median_delta={_format_pct(_safe_float(_safe_dict(payload.get('median_future_return_pct')).get('delta')))}, "
                f"positive_rate_delta={_format_pct(_safe_float(_safe_dict(payload.get('positive_rate_pct')).get('delta')))}"
            )
    lines.append("")
    lines.append("Stability Comparison")
    for category in TARGET_STABILITY_CATEGORIES:
        payload = _safe_dict(stability.get(category))
        lines.append(
            f"- {category}: group={_safe_dict(payload.get('group')).get('experiment', 'n/a')}, "
            f"stability_label={_safe_dict(payload.get('stability_label')).get('experiment', 'n/a')}, "
            f"stability_score_delta={_format_number(_safe_float(_safe_dict(payload.get('stability_score')).get('delta')))}, "
            f"visible_horizons={_safe_dict(payload.get('visible_horizons')).get('experiment', [])}"
        )
    lines.append("")
    lines.append("Final Diagnosis")
    lines.append(f"- primary_finding: {final_diagnosis.get('primary_finding', 'unknown')}")
    lines.append(
        f"- secondary_finding: {final_diagnosis.get('secondary_finding', 'unknown')}"
    )
    lines.append(
        f"- diagnosis_labels: {', '.join(_safe_list(final_diagnosis.get('diagnosis_labels'))) or 'none'}"
    )
    lines.append(f"- summary: {final_diagnosis.get('summary', 'n/a')}")
    lines.append("")
    return "\n".join(lines)


def write_experimental_relabel_comparison_report(
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
        build_experimental_relabel_comparison_markdown(summary),
        encoding="utf-8",
    )

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to baseline cumulative summary.json",
    )
    parser.add_argument(
        "--experiment-summary",
        type=Path,
        default=DEFAULT_EXPERIMENT_PATH,
        help="Path to candidate A experimental summary.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Output path for comparison JSON summary",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_OUTPUT,
        help="Output path for comparison markdown summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_summary, baseline_loaded = _load_json(args.baseline_summary)
    experiment_summary, experiment_loaded = _load_json(args.experiment_summary)

    summary = build_experimental_relabel_comparison_report(
        baseline_summary,
        experiment_summary,
        baseline_path=args.baseline_summary,
        experiment_path=args.experiment_summary,
        baseline_loaded=baseline_loaded,
        experiment_loaded=experiment_loaded,
    )
    outputs = write_experimental_relabel_comparison_report(
        summary,
        json_output_path=args.json_output,
        markdown_output_path=args.md_output,
    )
    print(
        json.dumps(
            {
                **outputs,
                "source_targeting": summary["source_targeting"],
                "final_diagnosis": summary["final_diagnosis"],
                "baseline_loaded": summary["metadata"]["baseline_loaded"],
                "experiment_loaded": summary["metadata"]["experiment_loaded"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()



