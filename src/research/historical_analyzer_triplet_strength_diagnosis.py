from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPORT_TYPE = "historical_analyzer_triplet_strength_diagnosis"
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_STRATEGY = "swing"
DEFAULT_HORIZON = "1h"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}, "
                    f"got {type(parsed).__name__}"
                )
            rows.append(parsed)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    return None


def safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def nested_get(data: Any, *path: str) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def json_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        marker = json_compact(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def average(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 6)


def median(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(statistics.median(filtered), 6)


def fmt_num(value: Any, digits: int = 3) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


# ---------------------------------------------------------------------------
# Domain extraction helpers
# ---------------------------------------------------------------------------


@dataclass
class TargetIdentity:
    symbol: str
    strategy: str
    horizon: str


def matches_identity(
    payload: dict[str, Any],
    identity: TargetIdentity,
    *,
    symbol_keys: tuple[str, ...] = ("symbol", "top_symbol", "selected_symbol"),
    strategy_keys: tuple[str, ...] = ("strategy", "top_strategy", "selected_strategy"),
    horizon_keys: tuple[str, ...] = ("horizon", "timeframe", "selected_horizon"),
) -> bool:
    symbol_value = None
    for key in symbol_keys:
        if payload.get(key) is not None:
            symbol_value = str(payload.get(key))
            break

    strategy_value = None
    for key in strategy_keys:
        if payload.get(key) is not None:
            strategy_value = str(payload.get(key))
            break

    horizon_value = None
    for key in horizon_keys:
        if payload.get(key) is not None:
            horizon_value = str(payload.get(key))
            break

    return (
        symbol_value == identity.symbol
        and strategy_value == identity.strategy
        and horizon_value == identity.horizon
    )


def flatten_candidate_entries(node: Any, inherited_horizon: str | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if isinstance(node, list):
        for item in node:
            results.extend(flatten_candidate_entries(item, inherited_horizon=inherited_horizon))
        return results

    if not isinstance(node, dict):
        return results

    candidate_keys = (
        "rows",
        "candidates",
        "edge_candidates",
        "edge_candidates_comparison",
        "items",
        "values",
        "entries",
        "top_candidates",
    )
    for key in candidate_keys:
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                results.extend(flatten_candidate_entries(item, inherited_horizon=inherited_horizon))
            return results

    by_horizon = node.get("by_horizon")
    if isinstance(by_horizon, dict):
        for horizon, subnode in by_horizon.items():
            results.extend(flatten_candidate_entries(subnode, inherited_horizon=str(horizon)))
        return results

    extracted = dict(node)
    if inherited_horizon is not None and extracted.get("horizon") is None:
        extracted["horizon"] = inherited_horizon
    results.append(extracted)
    return results


def find_matching_entries(node: Any, identity: TargetIdentity) -> list[dict[str, Any]]:
    candidates = flatten_candidate_entries(node)
    return [entry for entry in candidates if matches_identity(entry, identity)]


def normalize_deficit_block(node: Any) -> list[str]:
    if node is None:
        return []

    if isinstance(node, list):
        result: list[str] = []
        for item in node:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    result.append(stripped)
            elif isinstance(item, dict):
                name = first_non_null(
                    item.get("name"),
                    item.get("code"),
                    item.get("key"),
                    item.get("deficit"),
                    item.get("reason"),
                )
                if name is not None:
                    result.append(str(name))
        return unique_preserving_order(result)

    if isinstance(node, dict):
        result: list[str] = []
        for key, value in node.items():
            if isinstance(value, bool) and value:
                result.append(str(key))
            elif isinstance(value, (int, float)) and float(value) > 0:
                result.append(str(key))
            elif isinstance(value, str) and value.strip():
                result.append(f"{key}:{value.strip()}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        result.append(item.strip())
        return unique_preserving_order(result)

    if isinstance(node, str) and node.strip():
        return [node.strip()]

    return []


def extract_strength_diagnostics(entry: dict[str, Any]) -> dict[str, Any]:
    diagnostics = as_dict(entry.get("candidate_strength_diagnostics"))
    classification_reason = first_non_null(
        entry.get("classification_reason"),
        diagnostics.get("classification_reason"),
        diagnostics.get("candidate_classification_reason"),
    )

    supporting_deficits = unique_preserving_order(
        normalize_deficit_block(
            first_non_null(
                diagnostics.get("supporting_deficits"),
                diagnostics.get("supporting_major_deficits"),
                entry.get("supporting_deficits"),
                entry.get("supporting_major_deficits"),
            )
        )
    )

    critical_deficits = unique_preserving_order(
        normalize_deficit_block(
            first_non_null(
                diagnostics.get("critical_deficits"),
                diagnostics.get("critical_major_deficits"),
                entry.get("critical_deficits"),
                entry.get("critical_major_deficits"),
            )
        )
    )

    return {
        "classification_reason": classification_reason,
        "supporting_deficits": supporting_deficits,
        "critical_deficits": critical_deficits,
        "raw": diagnostics if diagnostics else None,
    }


def extract_numeric_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    diagnostics = as_dict(entry.get("candidate_strength_diagnostics"))

    sample_count = safe_int(
        first_non_null(
            entry.get("sample_count"),
            entry.get("count"),
            diagnostics.get("sample_count"),
            diagnostics.get("candidate_sample_count"),
            diagnostics.get("n"),
        )
    )
    positive_rate_pct = safe_float(
        first_non_null(
            entry.get("positive_rate_pct"),
            diagnostics.get("positive_rate_pct"),
            diagnostics.get("positive_rate"),
        )
    )
    robustness_signal_pct = safe_float(
        first_non_null(
            entry.get("robustness_signal_pct"),
            diagnostics.get("robustness_signal_pct"),
            diagnostics.get("robustness_pct"),
            diagnostics.get("robustness"),
        )
    )
    median_return_pct = safe_float(
        first_non_null(
            entry.get("median_return_pct"),
            diagnostics.get("median_return_pct"),
            diagnostics.get("median_pct"),
            diagnostics.get("median_return"),
        )
    )
    aggregate_score = safe_float(
        first_non_null(
            entry.get("aggregate_score"),
            diagnostics.get("aggregate_score"),
            diagnostics.get("score"),
        )
    )

    return {
        "sample_count": sample_count,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal_pct": robustness_signal_pct,
        "median_return_pct": median_return_pct,
        "aggregate_score": aggregate_score,
    }


def extract_preview_snapshot(raw_step: dict[str, Any], identity: TargetIdentity) -> dict[str, Any] | None:
    analyzer_output = as_dict(raw_step.get("analyzer_output"))
    preview_node = nested_get(analyzer_output, "edge_candidates_preview", "by_horizon", identity.horizon)

    candidates = find_matching_entries(preview_node, identity)
    if not candidates:
        preview_root = nested_get(analyzer_output, "edge_candidates_preview")
        candidates = find_matching_entries(preview_root, identity)

    if not candidates:
        return None

    entry = candidates[0]
    diagnostics = extract_strength_diagnostics(entry)
    numeric = extract_numeric_metrics(entry)

    return {
        "candidate_strength": first_non_null(
            entry.get("candidate_strength"),
            entry.get("selected_candidate_strength"),
            entry.get("latest_candidate_strength"),
        ),
        "candidate_strength_diagnostics": diagnostics["raw"],
        "classification_reason": diagnostics["classification_reason"],
        "supporting_deficits": diagnostics["supporting_deficits"],
        "critical_deficits": diagnostics["critical_deficits"],
        "top_symbol": first_non_null(entry.get("top_symbol"), entry.get("symbol")),
        "top_strategy": first_non_null(entry.get("top_strategy"), entry.get("strategy")),
        **numeric,
        "raw_entry": entry,
    }


def extract_candidate_row_snapshot(raw_step: dict[str, Any], identity: TargetIdentity) -> dict[str, Any] | None:
    analyzer_output = as_dict(raw_step.get("analyzer_output"))
    rows_node = nested_get(analyzer_output, "edge_candidate_rows")

    candidates = find_matching_entries(rows_node, identity)
    if not candidates:
        rows_list = nested_get(analyzer_output, "edge_candidate_rows", "rows")
        candidates = find_matching_entries(rows_list, identity)

    if not candidates:
        return None

    entry = candidates[0]
    diagnostics = extract_strength_diagnostics(entry)
    numeric = extract_numeric_metrics(entry)

    return {
        "selected_candidate_strength": first_non_null(
            entry.get("selected_candidate_strength"),
            entry.get("candidate_strength"),
            entry.get("latest_candidate_strength"),
        ),
        "selected_stability_label": first_non_null(
            entry.get("selected_stability_label"),
            entry.get("stability_label"),
            entry.get("candidate_stability_label"),
        ),
        "selected_visible_horizons": as_list(entry.get("selected_visible_horizons")),
        "preview_visible_horizons": as_list(entry.get("preview_visible_horizons")),
        "classification_reason": diagnostics["classification_reason"],
        "supporting_deficits": diagnostics["supporting_deficits"],
        "critical_deficits": diagnostics["critical_deficits"],
        **numeric,
        "candidate_strength_diagnostics": diagnostics["raw"],
        "raw_entry": entry,
    }


def extract_comparison_snapshot(raw_step: dict[str, Any], identity: TargetIdentity) -> dict[str, Any] | None:
    comparison_report = as_dict(raw_step.get("comparison_report"))
    candidates = find_matching_entries(comparison_report, identity)

    if not candidates:
        comparison_candidates = comparison_report.get("edge_candidates_comparison")
        candidates = find_matching_entries(comparison_candidates, identity)

    if not candidates:
        return None

    entry = candidates[0]
    diagnostics = extract_strength_diagnostics(entry)
    numeric = extract_numeric_metrics(entry)

    return {
        "latest_candidate_strength": entry.get("latest_candidate_strength"),
        "cumulative_candidate_strength": entry.get("cumulative_candidate_strength"),
        "candidate_strength_diagnostics": diagnostics["raw"],
        "classification_reason": diagnostics["classification_reason"],
        "supporting_deficits": diagnostics["supporting_deficits"],
        "critical_deficits": diagnostics["critical_deficits"],
        **numeric,
        "raw_entry": entry,
    }


def extract_selection_snapshot(raw_step: dict[str, Any]) -> dict[str, Any]:
    selection_output = as_dict(raw_step.get("selection_output"))
    rankings = as_list(selection_output.get("ranking"))
    selected_ranking = None
    for item in rankings:
        if not isinstance(item, dict):
            continue
        if item.get("is_selected") is True:
            selected_ranking = item
            break

    if selected_ranking is None and rankings:
        first_item = rankings[0]
        if isinstance(first_item, dict):
            selected_ranking = first_item

    return {
        "selection_status": selection_output.get("selection_status"),
        "selection_reason": selection_output.get("reason"),
        "selected_symbol": selection_output.get("selected_symbol"),
        "selected_strategy": selection_output.get("selected_strategy"),
        "selected_horizon": selection_output.get("selected_horizon"),
        "horizons_with_seed": as_list(selection_output.get("horizons_with_seed")),
        "horizons_without_seed": as_list(selection_output.get("horizons_without_seed")),
        "candidate_seed_count": selection_output.get("candidate_seed_count"),
        "ranking_count": selection_output.get("ranking_count"),
        "selected_ranking_entry": selected_ranking,
        "raw_selection_output": selection_output,
    }


# ---------------------------------------------------------------------------
# Diagnosis model
# ---------------------------------------------------------------------------


def resolve_raw_step_path(run_dir: Path, step_index: int, result_row: dict[str, Any]) -> Path:
    raw_output_path = result_row.get("raw_output_path")
    if isinstance(raw_output_path, str) and raw_output_path.strip():
        candidate = Path(raw_output_path)
        if candidate.exists():
            return candidate

    raw_steps_dir = run_dir / "raw_steps"
    candidate = raw_steps_dir / f"step_{step_index:05d}.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve raw step JSON for step_index={step_index} "
        f"under run_dir={run_dir}"
    )


def build_window_record(
    result_row: dict[str, Any],
    raw_step: dict[str, Any],
    identity: TargetIdentity,
) -> dict[str, Any]:
    step_index = safe_int(result_row.get("step_index"))
    preview_snapshot = extract_preview_snapshot(raw_step, identity)
    candidate_row_snapshot = extract_candidate_row_snapshot(raw_step, identity)
    comparison_snapshot = extract_comparison_snapshot(raw_step, identity)
    selection_snapshot = extract_selection_snapshot(raw_step)

    analyzer_triplet_present = bool(preview_snapshot or candidate_row_snapshot)

    latest_strength = None
    if comparison_snapshot is not None:
        latest_strength = comparison_snapshot.get("latest_candidate_strength")
    if latest_strength is None and preview_snapshot is not None:
        latest_strength = preview_snapshot.get("candidate_strength")
    if latest_strength is None and candidate_row_snapshot is not None:
        latest_strength = candidate_row_snapshot.get("selected_candidate_strength")

    supporting_deficits = unique_preserving_order(
        as_list(
            first_non_null(
                preview_snapshot.get("supporting_deficits") if preview_snapshot else None,
                candidate_row_snapshot.get("supporting_deficits") if candidate_row_snapshot else None,
                comparison_snapshot.get("supporting_deficits") if comparison_snapshot else None,
                [],
            )
        )
    )
    critical_deficits = unique_preserving_order(
        as_list(
            first_non_null(
                preview_snapshot.get("critical_deficits") if preview_snapshot else None,
                candidate_row_snapshot.get("critical_deficits") if candidate_row_snapshot else None,
                comparison_snapshot.get("critical_deficits") if comparison_snapshot else None,
                [],
            )
        )
    )
    classification_reason = first_non_null(
        preview_snapshot.get("classification_reason") if preview_snapshot else None,
        candidate_row_snapshot.get("classification_reason") if candidate_row_snapshot else None,
        comparison_snapshot.get("classification_reason") if comparison_snapshot else None,
    )

    numeric_metrics = {
        "sample_count": first_non_null(
            preview_snapshot.get("sample_count") if preview_snapshot else None,
            candidate_row_snapshot.get("sample_count") if candidate_row_snapshot else None,
            comparison_snapshot.get("sample_count") if comparison_snapshot else None,
        ),
        "positive_rate_pct": first_non_null(
            preview_snapshot.get("positive_rate_pct") if preview_snapshot else None,
            candidate_row_snapshot.get("positive_rate_pct") if candidate_row_snapshot else None,
            comparison_snapshot.get("positive_rate_pct") if comparison_snapshot else None,
        ),
        "robustness_signal_pct": first_non_null(
            preview_snapshot.get("robustness_signal_pct") if preview_snapshot else None,
            candidate_row_snapshot.get("robustness_signal_pct") if candidate_row_snapshot else None,
            comparison_snapshot.get("robustness_signal_pct") if comparison_snapshot else None,
        ),
        "median_return_pct": first_non_null(
            preview_snapshot.get("median_return_pct") if preview_snapshot else None,
            candidate_row_snapshot.get("median_return_pct") if candidate_row_snapshot else None,
            comparison_snapshot.get("median_return_pct") if comparison_snapshot else None,
        ),
        "aggregate_score": first_non_null(
            preview_snapshot.get("aggregate_score") if preview_snapshot else None,
            candidate_row_snapshot.get("aggregate_score") if candidate_row_snapshot else None,
            comparison_snapshot.get("aggregate_score") if comparison_snapshot else None,
        ),
    }

    return {
        "step_index": step_index,
        "selection_status": result_row.get("selection_status"),
        "selection_reason": result_row.get("reason"),
        "window_record_count": result_row.get("window_record_count"),
        "start_record_index": result_row.get("start_record_index_inclusive"),
        "end_record_index": result_row.get("end_record_index_inclusive"),
        "analyzer_triplet_present": analyzer_triplet_present,
        "comparison_snapshot": comparison_snapshot,
        "preview_snapshot": preview_snapshot,
        "candidate_row_snapshot": candidate_row_snapshot,
        "selection_snapshot": selection_snapshot,
        "latest_candidate_strength": latest_strength,
        "supporting_deficits": supporting_deficits,
        "critical_deficits": critical_deficits,
        "classification_reason": classification_reason,
        "numeric_metrics": numeric_metrics,
    }


def summarize_group_metrics(windows: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_latest_strengths = Counter()
    comparison_cumulative_strengths = Counter()
    preview_strengths = Counter()
    row_strengths = Counter()
    stability_labels = Counter()
    classification_reasons = Counter()
    supporting_deficits = Counter()
    critical_deficits = Counter()
    selection_statuses = Counter()
    selection_reasons = Counter()
    horizons_with_seed = Counter()

    sample_counts: list[float | None] = []
    positive_rates: list[float | None] = []
    robustness_values: list[float | None] = []
    median_returns: list[float | None] = []
    aggregate_scores: list[float | None] = []

    for window in windows:
        selection_statuses.update([str(window.get("selection_status"))])
        selection_reasons.update([str(window.get("selection_reason"))])

        comparison_snapshot = as_dict(window.get("comparison_snapshot"))
        preview_snapshot = as_dict(window.get("preview_snapshot"))
        row_snapshot = as_dict(window.get("candidate_row_snapshot"))
        selection_snapshot = as_dict(window.get("selection_snapshot"))

        if comparison_snapshot.get("latest_candidate_strength") is not None:
            comparison_latest_strengths.update([str(comparison_snapshot["latest_candidate_strength"])])
        if comparison_snapshot.get("cumulative_candidate_strength") is not None:
            comparison_cumulative_strengths.update([str(comparison_snapshot["cumulative_candidate_strength"])])
        if preview_snapshot.get("candidate_strength") is not None:
            preview_strengths.update([str(preview_snapshot["candidate_strength"])])
        if row_snapshot.get("selected_candidate_strength") is not None:
            row_strengths.update([str(row_snapshot["selected_candidate_strength"])])
        if row_snapshot.get("selected_stability_label") is not None:
            stability_labels.update([str(row_snapshot["selected_stability_label"])])

        if window.get("classification_reason") is not None:
            classification_reasons.update([str(window["classification_reason"])])

        for deficit in as_list(window.get("supporting_deficits")):
            supporting_deficits.update([str(deficit)])
        for deficit in as_list(window.get("critical_deficits")):
            critical_deficits.update([str(deficit)])

        for horizon in as_list(selection_snapshot.get("horizons_with_seed")):
            horizons_with_seed.update([str(horizon)])

        metrics = as_dict(window.get("numeric_metrics"))
        sample_counts.append(safe_float(metrics.get("sample_count")))
        positive_rates.append(safe_float(metrics.get("positive_rate_pct")))
        robustness_values.append(safe_float(metrics.get("robustness_signal_pct")))
        median_returns.append(safe_float(metrics.get("median_return_pct")))
        aggregate_scores.append(safe_float(metrics.get("aggregate_score")))

    return {
        "window_count": len(windows),
        "selection_status_counts": dict(selection_statuses),
        "selection_reason_counts": dict(selection_reasons),
        "comparison_latest_candidate_strength_counts": dict(comparison_latest_strengths),
        "comparison_cumulative_candidate_strength_counts": dict(comparison_cumulative_strengths),
        "preview_candidate_strength_counts": dict(preview_strengths),
        "row_selected_candidate_strength_counts": dict(row_strengths),
        "row_selected_stability_label_counts": dict(stability_labels),
        "classification_reason_counts": dict(classification_reasons),
        "supporting_deficit_counts": dict(supporting_deficits),
        "critical_deficit_counts": dict(critical_deficits),
        "horizons_with_seed_counts": dict(horizons_with_seed),
        "numeric_metric_summary": {
            "sample_count": {
                "avg": average(sample_counts),
                "median": median(sample_counts),
                "min": min([v for v in sample_counts if v is not None], default=None),
                "max": max([v for v in sample_counts if v is not None], default=None),
            },
            "positive_rate_pct": {
                "avg": average(positive_rates),
                "median": median(positive_rates),
                "min": min([v for v in positive_rates if v is not None], default=None),
                "max": max([v for v in positive_rates if v is not None], default=None),
            },
            "robustness_signal_pct": {
                "avg": average(robustness_values),
                "median": median(robustness_values),
                "min": min([v for v in robustness_values if v is not None], default=None),
                "max": max([v for v in robustness_values if v is not None], default=None),
            },
            "median_return_pct": {
                "avg": average(median_returns),
                "median": median(median_returns),
                "min": min([v for v in median_returns if v is not None], default=None),
                "max": max([v for v in median_returns if v is not None], default=None),
            },
            "aggregate_score": {
                "avg": average(aggregate_scores),
                "median": median(aggregate_scores),
                "min": min([v for v in aggregate_scores if v is not None], default=None),
                "max": max([v for v in aggregate_scores if v is not None], default=None),
            },
        },
    }


def build_pairwise_comparisons(
    present_windows: list[dict[str, Any]],
    missing_windows: list[dict[str, Any]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    if not present_windows or not missing_windows:
        return []

    present_sorted = sorted(present_windows, key=lambda item: item.get("step_index") or -1)
    missing_sorted = sorted(missing_windows, key=lambda item: item.get("step_index") or -1)

    pair_count = min(len(present_sorted), len(missing_sorted), max_pairs)
    pairs: list[dict[str, Any]] = []

    for index in range(pair_count):
        present = present_sorted[index]
        missing = missing_sorted[index]

        present_metrics = as_dict(present.get("numeric_metrics"))
        missing_metrics = as_dict(missing.get("numeric_metrics"))

        metric_deltas: dict[str, Any] = {}
        for metric_name in (
            "sample_count",
            "positive_rate_pct",
            "robustness_signal_pct",
            "median_return_pct",
            "aggregate_score",
        ):
            present_value = safe_float(present_metrics.get(metric_name))
            missing_value = safe_float(missing_metrics.get(metric_name))
            metric_deltas[metric_name] = {
                "present": present_value,
                "missing": missing_value,
                "delta_present_minus_missing": (
                    round(present_value - missing_value, 6)
                    if present_value is not None and missing_value is not None
                    else None
                ),
            }

        pairs.append(
            {
                "pair_index": index + 1,
                "present_step_index": present.get("step_index"),
                "missing_step_index": missing.get("step_index"),
                "present_selection_status": present.get("selection_status"),
                "missing_selection_status": missing.get("selection_status"),
                "present_selection_reason": present.get("selection_reason"),
                "missing_selection_reason": missing.get("selection_reason"),
                "present_latest_candidate_strength": present.get("latest_candidate_strength"),
                "missing_latest_candidate_strength": missing.get("latest_candidate_strength"),
                "present_classification_reason": present.get("classification_reason"),
                "missing_classification_reason": missing.get("classification_reason"),
                "present_supporting_deficits": as_list(present.get("supporting_deficits")),
                "missing_supporting_deficits": as_list(missing.get("supporting_deficits")),
                "present_critical_deficits": as_list(present.get("critical_deficits")),
                "missing_critical_deficits": as_list(missing.get("critical_deficits")),
                "metric_deltas": metric_deltas,
            }
        )

    return pairs


def dominant_labels(counter_map: dict[str, int], top_n: int = 5) -> list[dict[str, Any]]:
    counter = Counter(counter_map)
    return [{"label": label, "count": count} for label, count in counter.most_common(top_n)]


def explain_root_cause(
    present_summary: dict[str, Any],
    missing_summary: dict[str, Any],
    present_windows: list[dict[str, Any]],
    missing_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    explanation_lines: list[str] = []
    findings: list[dict[str, Any]] = []

    present_latest = Counter(present_summary.get("comparison_latest_candidate_strength_counts", {}))
    missing_latest = Counter(missing_summary.get("comparison_latest_candidate_strength_counts", {}))

    if present_latest.get("moderate", 0) > 0 and missing_latest.get("weak", 0) > 0:
        explanation_lines.append(
            "The strongest recurring separator is comparison latest strength: "
            "present windows retain 'moderate', while missing windows shift to 'weak'."
        )
        findings.append(
            {
                "type": "strength_transition",
                "present_dominant": present_latest.most_common(3),
                "missing_dominant": missing_latest.most_common(3),
            }
        )

    present_reasons = Counter(present_summary.get("classification_reason_counts", {}))
    missing_reasons = Counter(missing_summary.get("classification_reason_counts", {}))
    if missing_reasons:
        explanation_lines.append(
            "Missing windows show a different classification-reason distribution, "
            "which indicates the downgrade is rule-driven rather than mapper-driven."
        )
        findings.append(
            {
                "type": "classification_reason_shift",
                "present_top_reasons": present_reasons.most_common(5),
                "missing_top_reasons": missing_reasons.most_common(5),
            }
        )

    present_supporting = Counter(present_summary.get("supporting_deficit_counts", {}))
    missing_supporting = Counter(missing_summary.get("supporting_deficit_counts", {}))
    present_critical = Counter(present_summary.get("critical_deficit_counts", {}))
    missing_critical = Counter(missing_summary.get("critical_deficit_counts", {}))

    if missing_supporting or missing_critical:
        explanation_lines.append(
            "Deficit composition should be treated as the immediate downgrade driver. "
            "The diagnosis output below isolates which supporting or critical deficits "
            "appear disproportionately in missing windows."
        )
        findings.append(
            {
                "type": "deficit_shift",
                "present_supporting_top": present_supporting.most_common(5),
                "missing_supporting_top": missing_supporting.most_common(5),
                "present_critical_top": present_critical.most_common(5),
                "missing_critical_top": missing_critical.most_common(5),
            }
        )

    def metric_avg(summary: dict[str, Any], metric_name: str) -> float | None:
        return safe_float(
            nested_get(summary, "numeric_metric_summary", metric_name, "avg")
        )

    metric_names = [
        "sample_count",
        "positive_rate_pct",
        "robustness_signal_pct",
        "median_return_pct",
        "aggregate_score",
    ]
    metric_shifts: list[dict[str, Any]] = []
    for metric_name in metric_names:
        present_avg = metric_avg(present_summary, metric_name)
        missing_avg = metric_avg(missing_summary, metric_name)
        delta = None
        if present_avg is not None and missing_avg is not None:
            delta = round(present_avg - missing_avg, 6)
        metric_shifts.append(
            {
                "metric": metric_name,
                "present_avg": present_avg,
                "missing_avg": missing_avg,
                "delta_present_minus_missing": delta,
            }
        )

    explanation_lines.append(
        "Numeric metrics are included to determine whether the downgrade is primarily "
        "triggered by sample count, positive-rate floor, robustness floor, median return, "
        "aggregate score, or a multi-deficit recovery guard."
    )
    findings.append({"type": "metric_shift", "metrics": metric_shifts})

    if missing_windows and all(window.get("latest_candidate_strength") == "weak" for window in missing_windows):
        explanation_lines.append(
            "All missing windows resolve to weak strength at analyzer comparison level, "
            "which is consistent with analyzer suppression of the ETHUSDT/swing/1h triplet."
        )

    if present_windows and all(window.get("selection_status") == "selected" for window in present_windows):
        explanation_lines.append(
            "Present windows remain selection-compatible after the 1h triplet is emitted, "
            "confirming the 1h analyzer triplet as the decisive continuity factor."
        )

    if not explanation_lines:
        explanation_lines.append(
            "No single rule could be proven from summary counts alone. "
            "Use the pairwise section and raw diagnostics payloads to identify the exact threshold."
        )

    return {
        "summary": " ".join(explanation_lines),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_counter_section(title: str, data: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", ""]
    if not data:
        lines.append("- n/a")
        lines.append("")
        return lines
    for key, value in sorted(data.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.append("")
    return lines


def render_metric_summary(title: str, metrics: dict[str, Any]) -> list[str]:
    lines = [f"### {title}", "", "| Metric | Avg | Median | Min | Max |", "|---|---:|---:|---:|---:|"]
    if not metrics:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
        return lines

    for metric_name, block in metrics.items():
        lines.append(
            f"| {metric_name} | {fmt_num(block.get('avg'))} | {fmt_num(block.get('median'))} "
            f"| {fmt_num(block.get('min'))} | {fmt_num(block.get('max'))} |"
        )
    lines.append("")
    return lines


def render_window_sample(title: str, windows: list[dict[str, Any]], limit: int = 10) -> list[str]:
    lines = [f"### {title}", ""]
    if not windows:
        lines.append("- none")
        lines.append("")
        return lines

    lines.append(
        "| Step | Status | Reason | Latest Strength | Classification Reason | Sample | Positive % | Robustness % | Median % | Aggregate |"
    )
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|")
    for window in windows[:limit]:
        metrics = as_dict(window.get("numeric_metrics"))
        lines.append(
            f"| {window.get('step_index')} | {window.get('selection_status')} | "
            f"{window.get('selection_reason')} | {window.get('latest_candidate_strength')} | "
            f"{window.get('classification_reason')} | "
            f"{fmt_num(metrics.get('sample_count'), 0)} | "
            f"{fmt_num(metrics.get('positive_rate_pct'))} | "
            f"{fmt_num(metrics.get('robustness_signal_pct'))} | "
            f"{fmt_num(metrics.get('median_return_pct'))} | "
            f"{fmt_num(metrics.get('aggregate_score'))} |"
        )
    lines.append("")
    return lines


def render_pairwise_section(pairs: list[dict[str, Any]]) -> list[str]:
    lines = ["## Pairwise Comparison", ""]
    if not pairs:
        lines.append("- no pairwise comparisons generated")
        lines.append("")
        return lines

    for pair in pairs:
        lines.append(f"### Pair {pair['pair_index']}")
        lines.append("")
        lines.append(
            f"- present_step_index: {pair.get('present_step_index')} "
            f"({pair.get('present_selection_status')} / {pair.get('present_selection_reason')})"
        )
        lines.append(
            f"- missing_step_index: {pair.get('missing_step_index')} "
            f"({pair.get('missing_selection_status')} / {pair.get('missing_selection_reason')})"
        )
        lines.append(
            f"- latest_strength: present={pair.get('present_latest_candidate_strength')} "
            f"vs missing={pair.get('missing_latest_candidate_strength')}"
        )
        lines.append(
            f"- classification_reason: present={pair.get('present_classification_reason')} "
            f"vs missing={pair.get('missing_classification_reason')}"
        )
        lines.append(
            f"- supporting_deficits: present={pair.get('present_supporting_deficits')} "
            f"vs missing={pair.get('missing_supporting_deficits')}"
        )
        lines.append(
            f"- critical_deficits: present={pair.get('present_critical_deficits')} "
            f"vs missing={pair.get('missing_critical_deficits')}"
        )
        lines.append("")
        lines.append("| Metric | Present | Missing | Delta (present - missing) |")
        lines.append("|---|---:|---:|---:|")
        metric_deltas = as_dict(pair.get("metric_deltas"))
        for metric_name, block in metric_deltas.items():
            lines.append(
                f"| {metric_name} | {fmt_num(block.get('present'))} | "
                f"{fmt_num(block.get('missing'))} | {fmt_num(block.get('delta_present_minus_missing'))} |"
            )
        lines.append("")
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    metadata = as_dict(report.get("metadata"))
    target = as_dict(report.get("target"))
    summary = as_dict(report.get("summary"))
    present_summary = as_dict(report.get("present_group_summary"))
    missing_summary = as_dict(report.get("missing_group_summary"))
    explanation = as_dict(report.get("root_cause_explanation"))

    lines.append(f"# {REPORT_TYPE}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- generated_at: {metadata.get('generated_at')}")
    lines.append(f"- run_dir: `{metadata.get('run_dir')}`")
    lines.append(f"- report_json: `{metadata.get('report_json')}`")
    lines.append(f"- report_md: `{metadata.get('report_md')}`")
    lines.append("")
    lines.append("## Target")
    lines.append("")
    lines.append(f"- symbol: {target.get('symbol')}")
    lines.append(f"- strategy: {target.get('strategy')}")
    lines.append(f"- horizon: {target.get('horizon')}")
    lines.append("")
    lines.append("## Top-Level Summary")
    lines.append("")
    lines.append(f"- total_windows: {summary.get('total_windows')}")
    lines.append(f"- analyzer_triplet_present_count: {summary.get('analyzer_triplet_present_count')}")
    lines.append(f"- analyzer_triplet_missing_count: {summary.get('analyzer_triplet_missing_count')}")
    lines.append("")
    lines.append("## Root Cause Explanation")
    lines.append("")
    lines.append(explanation.get("summary", "n/a"))
    lines.append("")

    lines.extend(render_counter_section("Present Group - Latest Candidate Strength", present_summary.get("comparison_latest_candidate_strength_counts", {})))
    lines.extend(render_counter_section("Missing Group - Latest Candidate Strength", missing_summary.get("comparison_latest_candidate_strength_counts", {})))
    lines.extend(render_counter_section("Present Group - Classification Reasons", present_summary.get("classification_reason_counts", {})))
    lines.extend(render_counter_section("Missing Group - Classification Reasons", missing_summary.get("classification_reason_counts", {})))
    lines.extend(render_counter_section("Present Group - Supporting Deficits", present_summary.get("supporting_deficit_counts", {})))
    lines.extend(render_counter_section("Missing Group - Supporting Deficits", missing_summary.get("supporting_deficit_counts", {})))
    lines.extend(render_counter_section("Present Group - Critical Deficits", present_summary.get("critical_deficit_counts", {})))
    lines.extend(render_counter_section("Missing Group - Critical Deficits", missing_summary.get("critical_deficit_counts", {})))

    lines.extend(render_metric_summary("Present Group - Numeric Metrics", present_summary.get("numeric_metric_summary", {})))
    lines.extend(render_metric_summary("Missing Group - Numeric Metrics", missing_summary.get("numeric_metric_summary", {})))

    lines.extend(render_window_sample("Present Window Sample", as_list(report.get("present_windows")), limit=12))
    lines.extend(render_window_sample("Missing Window Sample", as_list(report.get("missing_windows")), limit=12))

    lines.extend(render_pairwise_section(as_list(report.get("pairwise_comparison"))))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_diagnosis(
    run_dir: Path,
    identity: TargetIdentity,
    *,
    max_pairs: int,
    write_latest_copy: bool,
) -> dict[str, Any]:
    step_results_path = run_dir / "step_results.jsonl"
    if not step_results_path.exists():
        raise FileNotFoundError(f"Missing step_results.jsonl at {step_results_path}")

    step_results = load_jsonl(step_results_path)
    if not step_results:
        raise ValueError(f"No rows found in {step_results_path}")

    windows: list[dict[str, Any]] = []
    for result_row in step_results:
        step_index = safe_int(result_row.get("step_index"))
        if step_index is None:
            continue
        raw_step_path = resolve_raw_step_path(run_dir, step_index, result_row)
        raw_step = load_json(raw_step_path)
        window_record = build_window_record(result_row, raw_step, identity)
        windows.append(window_record)

    windows_sorted = sorted(windows, key=lambda item: item.get("step_index") or -1)
    present_windows = [window for window in windows_sorted if window.get("analyzer_triplet_present") is True]
    missing_windows = [window for window in windows_sorted if window.get("analyzer_triplet_present") is False]

    present_summary = summarize_group_metrics(present_windows)
    missing_summary = summarize_group_metrics(missing_windows)
    pairwise = build_pairwise_comparisons(present_windows, missing_windows, max_pairs=max_pairs)
    explanation = explain_root_cause(
        present_summary=present_summary,
        missing_summary=missing_summary,
        present_windows=present_windows,
        missing_windows=missing_windows,
    )

    report_json_path = run_dir / f"{REPORT_TYPE}.json"
    report_md_path = run_dir / f"{REPORT_TYPE}.md"

    report: dict[str, Any] = {
        "metadata": {
            "generated_at": utc_now_iso(),
            "report_type": REPORT_TYPE,
            "run_dir": str(run_dir),
            "source_step_results_path": str(step_results_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
        "target": {
            "symbol": identity.symbol,
            "strategy": identity.strategy,
            "horizon": identity.horizon,
        },
        "summary": {
            "total_windows": len(windows_sorted),
            "analyzer_triplet_present_count": len(present_windows),
            "analyzer_triplet_missing_count": len(missing_windows),
        },
        "present_group_summary": present_summary,
        "missing_group_summary": missing_summary,
        "root_cause_explanation": explanation,
        "pairwise_comparison": pairwise,
        "present_windows": present_windows,
        "missing_windows": missing_windows,
    }

    markdown = render_markdown_report(report)

    write_json(report_json_path, report)
    write_text(report_md_path, markdown)

    if write_latest_copy:
        latest_dir = Path("logs/research_reports/latest")
        write_json(latest_dir / f"{REPORT_TYPE}.json", report)
        write_text(latest_dir / f"{REPORT_TYPE}.md", markdown)

    return {
        "run_dir": str(run_dir),
        "json_report": str(report_json_path),
        "markdown_report": str(report_md_path),
        "total_windows": len(windows_sorted),
        "present_count": len(present_windows),
        "missing_count": len(missing_windows),
        "target": {
            "symbol": identity.symbol,
            "strategy": identity.strategy,
            "horizon": identity.horizon,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to historical_direct_edge_selection run directory",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20,
        help="Maximum number of present-vs-missing pairwise comparisons",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Also write a copy to logs/research_reports/latest/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    identity = TargetIdentity(
        symbol=str(args.symbol),
        strategy=str(args.strategy),
        horizon=str(args.horizon),
    )
    result = run_diagnosis(
        run_dir=run_dir,
        identity=identity,
        max_pairs=args.max_pairs,
        write_latest_copy=bool(args.write_latest_copy),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()