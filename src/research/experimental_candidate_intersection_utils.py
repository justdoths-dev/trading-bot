from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_B_LABELING_METHOD,
    _safe_float,
    _safe_text,
)
from src.research.experimental_labeling.asymmetric_threshold_config import (
    DEFAULT_VARIANT_NAME,
)
from src.research.experimental_labeling.asymmetric_threshold_relabeler import (
    LABELING_METHOD as CANDIDATE_C_LABELING_METHOD,
)

MATCH_KEY_FIELDS = (
    "logged_at",
    "symbol",
    "selected_strategy",
    "future_return_15m",
    "future_return_1h",
    "future_return_4h",
)
KEY_MISSING = "__missing__"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_key_text(value: Any) -> str:
    text = _safe_text(value)
    return text if text is not None else KEY_MISSING


def _normalize_key_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return KEY_MISSING
    return f"{number:.12g}"


def build_row_match_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Build a deterministic composite key for relabeled-row alignment."""
    return (
        _normalize_key_text(row.get("logged_at")),
        _normalize_key_text(row.get("symbol")),
        _normalize_key_text(row.get("selected_strategy") or row.get("strategy")),
        _normalize_key_number(row.get("future_return_15m")),
        _normalize_key_number(row.get("future_return_1h")),
        _normalize_key_number(row.get("future_return_4h")),
    )


def filter_candidate_b_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("labeling_method") == CANDIDATE_B_LABELING_METHOD:
            filtered.append(row)
    return filtered


def filter_candidate_c_records(
    records: list[dict[str, Any]],
    *,
    variant_name: str = DEFAULT_VARIANT_NAME,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in records:
        metadata = _safe_dict(row.get("experimental_labeling"))
        if metadata.get("labeling_method") != CANDIDATE_C_LABELING_METHOD:
            continue
        if metadata.get("variant") != variant_name:
            continue
        filtered.append(row)
    return filtered


def build_intersection_datasets(
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indexed_baseline: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    indexed_experiment: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in baseline_rows:
        indexed_baseline[build_row_match_key(row)].append(row)
    for row in experiment_rows:
        indexed_experiment[build_row_match_key(row)].append(row)

    baseline_shared_rows: list[dict[str, Any]] = []
    experiment_shared_rows: list[dict[str, Any]] = []
    baseline_only_count = 0
    experiment_only_count = 0

    all_keys = sorted(set(indexed_baseline) | set(indexed_experiment))
    for key in all_keys:
        baseline_bucket = indexed_baseline.get(key, [])
        experiment_bucket = indexed_experiment.get(key, [])
        pair_count = min(len(baseline_bucket), len(experiment_bucket))

        if pair_count:
            baseline_shared_rows.extend(baseline_bucket[:pair_count])
            experiment_shared_rows.extend(experiment_bucket[:pair_count])

        if len(baseline_bucket) > pair_count:
            baseline_only_count += len(baseline_bucket) - pair_count
        if len(experiment_bucket) > pair_count:
            experiment_only_count += len(experiment_bucket) - pair_count

    baseline_total_rows = len(baseline_rows)
    experiment_total_rows = len(experiment_rows)
    shared_row_count = len(baseline_shared_rows)

    intersection_overview = {
        "baseline_total_rows": baseline_total_rows,
        "experiment_total_rows": experiment_total_rows,
        "shared_row_count": shared_row_count,
        "baseline_only_row_count": baseline_only_count,
        "experiment_only_row_count": experiment_only_count,
        "shared_ratio_vs_baseline": round(shared_row_count / baseline_total_rows, 6)
        if baseline_total_rows > 0
        else 0.0,
        "shared_ratio_vs_experiment": round(shared_row_count / experiment_total_rows, 6)
        if experiment_total_rows > 0
        else 0.0,
    }

    return baseline_shared_rows, experiment_shared_rows, intersection_overview
