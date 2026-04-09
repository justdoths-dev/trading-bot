from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

REPORT_TYPE = "sub_floor_candidate_validation"
CLASSIFICATION_VERSION = "conservative_v2"
DEFAULT_JSON_OUTPUT_NAME = "sub_floor_candidate_validation_summary.json"
DEFAULT_MD_OUTPUT_NAME = "sub_floor_candidate_validation_summary.md"

SAMPLE_COUNT_BANDS = (
    ("0-9", 0, 9),
    ("10-19", 10, 19),
    ("20-29", 20, 29),
    ("30+", 30, None),
)

CLASSIFICATION_ORDER = {
    "sample_floor_only_near_miss": 0,
    "sample_floor_plus_quality_weakness": 1,
    "non_positive_or_flat_edge": 2,
    "non_sample_primary_failure": 3,
}

NEAR_MISS_CLASSIFICATIONS = {
    "sample_floor_only_near_miss",
    "sample_floor_plus_quality_weakness",
}

PRIMARY_EDGE_CANDIDATE_PATHS = (
    ("edge_candidate_rows",),
    ("analyzer_output", "edge_candidate_rows"),
    ("summary", "edge_candidate_rows"),
)

WINDOW_LABEL_PATTERN = re.compile(r"(?<!\d)(\d+)h(?![a-z])", re.IGNORECASE)

# Conservative diagnostic-only heuristics.
# These are NOT production thresholds and MUST NOT be interpreted as
# analyzer / mapper / engine gate definitions.
DIAGNOSTIC_POSITIVE_RATE_MIN_PCT = 50.0
DIAGNOSTIC_ROBUSTNESS_MIN_PCT = 50.0

# Structural blockers mean the row is not a meaningful sample-floor near-miss.
# These are different from merely "directionally weak" candidates.
STRUCTURAL_NON_SAMPLE_BLOCKERS = {
    "strategy_horizon_incompatible",
    "no_labeled_rows_for_horizon",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _normalize_symbol(value: Any) -> str:
    text = _safe_text(value)
    return text.upper() if text else "unknown_symbol"


def _normalize_strategy(value: Any) -> str:
    text = _safe_text(value)
    return text.lower() if text else "unknown_strategy"


def _normalize_horizon(value: Any) -> str:
    text = _safe_text(value)
    return text if text else "unknown_horizon"


def _normalize_string_list(value: Any) -> list[str]:
    normalized: list[str] = []
    for item in _safe_list(value):
        text = _safe_text(item)
        if text is not None:
            normalized.append(text)
    return normalized


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        key: counter[key]
        for key, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    }


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 6)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def _window_sort_key(label: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(\d+)h", label)
    if match:
        return (0, int(match.group(1)), label)

    match = re.fullmatch(r"max_rows_(\d+)", label)
    if match:
        return (1, int(match.group(1)), label)

    return (2, 0, label)


def _classification_sort_key(name: str) -> tuple[int, str]:
    return (CLASSIFICATION_ORDER.get(name, 999), name)


def _sample_band_sort_key(label: str) -> int:
    for index, (band, _, _) in enumerate(SAMPLE_COUNT_BANDS):
        if band == label:
            return index
    return len(SAMPLE_COUNT_BANDS)


def _sample_count_band(sample_count: int | None) -> str:
    if sample_count is None:
        return "unknown"

    for label, lower, upper in SAMPLE_COUNT_BANDS:
        if upper is None:
            if sample_count >= lower:
                return label
            continue
        if lower <= sample_count <= upper:
            return label

    return "unknown"


def _get_nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, dict) else None


def _recursive_find_key_path(
    payload: Any,
    target_key: str,
    *,
    max_depth: int = 6,
    current_path: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], Any] | None:
    if len(current_path) > max_depth:
        return None

    if isinstance(payload, dict):
        if target_key in payload:
            return current_path + (target_key,), payload[target_key]
        for key, value in payload.items():
            result = _recursive_find_key_path(
                value,
                target_key,
                max_depth=max_depth,
                current_path=current_path + (str(key),),
            )
            if result is not None:
                return result
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            result = _recursive_find_key_path(
                value,
                target_key,
                max_depth=max_depth,
                current_path=current_path + (f"[{index}]",),
            )
            if result is not None:
                return result

    return None


def _recursive_find_first_scalar_with_path(
    payload: Any,
    candidate_keys: set[str],
    *,
    max_depth: int = 6,
    current_path: tuple[str, ...] = (),
    current_depth: int = 0,
) -> tuple[tuple[str, ...], Any] | None:
    if current_depth > max_depth:
        return None

    if isinstance(payload, dict):
        for key in sorted(candidate_keys):
            if key in payload and not isinstance(payload[key], (dict, list)):
                return current_path + (key,), payload[key]
        for key, value in payload.items():
            found = _recursive_find_first_scalar_with_path(
                value,
                candidate_keys,
                max_depth=max_depth,
                current_path=current_path + (str(key),),
                current_depth=current_depth + 1,
            )
            if found is not None:
                return found
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found = _recursive_find_first_scalar_with_path(
                value,
                candidate_keys,
                max_depth=max_depth,
                current_path=current_path + (f"[{index}]",),
                current_depth=current_depth + 1,
            )
            if found is not None:
                return found

    return None


def _path_to_label(path_parts: tuple[str, ...]) -> str:
    return ".".join(path_parts) if path_parts else "root"


def load_summary_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    if not path.exists():
        return {}, [f"summary_input_missing: {path}"]
    if not path.is_file():
        return {}, [f"summary_input_not_a_file: {path}"]

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, [f"summary_input_invalid_json: {path}"]

    if not isinstance(payload, dict):
        return {}, [f"summary_input_not_an_object: {path}"]

    return payload, warnings


def _extract_edge_candidate_rows_block(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    for keys in PRIMARY_EDGE_CANDIDATE_PATHS:
        block = _get_nested_dict(payload, *keys)
        if block is not None:
            return block, ".".join(keys)

    fallback = _recursive_find_key_path(payload, "edge_candidate_rows")
    if fallback is None:
        return None, None

    path_parts, value = fallback
    if not isinstance(value, dict):
        return None, ".".join(path_parts)

    return value, ".".join(path_parts)


def _infer_window_label(payload: dict[str, Any], path: Path) -> tuple[str, str, dict[str, Any]]:
    max_age_hours_result = _recursive_find_first_scalar_with_path(
        payload,
        {"max_age_hours", "latest_window_hours", "window_hours"},
    )
    max_rows_result = _recursive_find_first_scalar_with_path(
        payload,
        {"max_rows", "latest_max_rows", "window_max_rows"},
    )
    valid_records_result = _recursive_find_first_scalar_with_path(
        payload,
        {"valid_records", "windowed_record_count"},
    )

    max_age_hours = _safe_int(max_age_hours_result[1]) if max_age_hours_result else None
    max_rows = _safe_int(max_rows_result[1]) if max_rows_result else None
    valid_records = _safe_int(valid_records_result[1]) if valid_records_result else None

    max_age_source = _path_to_label(max_age_hours_result[0]) if max_age_hours_result else None
    max_rows_source = _path_to_label(max_rows_result[0]) if max_rows_result else None

    if max_age_hours is not None and max_age_hours > 0:
        return (
            f"{max_age_hours}h",
            max_age_source or "unknown",
            {
                "max_age_hours": max_age_hours,
                "max_rows": max_rows,
                "valid_records": valid_records,
            },
        )

    filename_parts = [path.stem, path.name, path.parent.name]
    for candidate in filename_parts:
        match = WINDOW_LABEL_PATTERN.search(candidate)
        if match:
            return (
                f"{int(match.group(1))}h",
                "file_name_pattern",
                {
                    "max_age_hours": max_age_hours,
                    "max_rows": max_rows,
                    "valid_records": valid_records,
                },
            )

    if max_rows is not None and max_rows > 0:
        return (
            f"max_rows_{max_rows}",
            max_rows_source or "unknown",
            {
                "max_age_hours": max_age_hours,
                "max_rows": max_rows,
                "valid_records": valid_records,
            },
        )

    return (
        path.stem,
        "fallback_path_stem",
        {
            "max_age_hours": max_age_hours,
            "max_rows": max_rows,
            "valid_records": valid_records,
        },
    )


def _sample_floor_blocked(
    rejection_reason: str | None,
    rejection_reasons: list[str],
    sample_gate: str | None,
    visibility_reason: str | None,
    sample_count: int | None,
) -> bool:
    """
    Conservative rule:
    Treat a row as sample-floor blocked only when explicit sample-floor evidence exists.
    We do NOT treat failed_absolute_minimum_gate alone as enough evidence.
    """
    reasons = set(rejection_reasons)
    if rejection_reason:
        reasons.add(rejection_reason)
    if visibility_reason:
        reasons.add(visibility_reason)

    explicit_sample_floor_evidence = "sample_count_below_absolute_floor" in reasons

    if not explicit_sample_floor_evidence:
        return False

    if sample_count is not None and sample_count >= 30:
        return False

    if sample_gate is not None and sample_gate != "failed":
        return False

    return True


def _structural_non_sample_blockers(
    *,
    rejection_reason: str | None,
    rejection_reasons: list[str],
    visibility_reason: str | None,
) -> list[str]:
    reasons = set(rejection_reasons)
    if rejection_reason:
        reasons.add(rejection_reason)
    if visibility_reason:
        reasons.add(visibility_reason)

    blockers = sorted(reasons.intersection(STRUCTURAL_NON_SAMPLE_BLOCKERS))
    return blockers


def _directional_quality_flags(
    *,
    positive_rate_pct: float | None,
    robustness_signal_pct: float | None,
    robustness_signal: str | None,
    candidate_strength: str | None,
    diagnostic_category: str | None,
) -> list[str]:
    flags: list[str] = []

    if positive_rate_pct is not None and positive_rate_pct <= DIAGNOSTIC_POSITIVE_RATE_MIN_PCT:
        flags.append(
            f"positive_rate_not_above_{int(DIAGNOSTIC_POSITIVE_RATE_MIN_PCT)}_pct"
        )

    if (
        robustness_signal_pct is not None
        and robustness_signal_pct < DIAGNOSTIC_ROBUSTNESS_MIN_PCT
    ):
        label = robustness_signal or "robustness_signal"
        flags.append(
            f"{label}_below_{int(DIAGNOSTIC_ROBUSTNESS_MIN_PCT)}_pct"
        )

    if candidate_strength == "weak" or diagnostic_category == "quality_rejected":
        flags.append("analyzer_quality_rejection_present")

    return flags


def _classify_candidate(
    *,
    sample_floor_blocked: bool,
    median_future_return_pct: float | None,
    directional_quality_flags: list[str],
    structural_non_sample_blockers: list[str],
    rejection_reason: str | None,
    rejection_reasons: list[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if structural_non_sample_blockers:
        reasons.append(
            "row contains structural non-sample blockers that invalidate near-miss interpretation"
        )
        reasons.extend(
            f"structural_non_sample_blocker={item}"
            for item in structural_non_sample_blockers
        )
        return "non_sample_primary_failure", reasons

    if not sample_floor_blocked:
        reasons.append(
            "primary rejection is not conservatively attributable to the absolute sample floor"
        )
        if rejection_reason:
            reasons.append(f"primary_rejection_reason={rejection_reason}")
        if "sample_count_below_absolute_floor" not in rejection_reasons:
            reasons.append("explicit sample floor evidence is absent from rejection_reasons")
        return "non_sample_primary_failure", reasons

    if median_future_return_pct is None:
        return (
            "non_positive_or_flat_edge",
            ["median_future_return_pct is missing; no positive directional edge is visible"],
        )

    if median_future_return_pct <= 0:
        return (
            "non_positive_or_flat_edge",
            [f"median_future_return_pct={median_future_return_pct} is non-positive"],
        )

    if directional_quality_flags:
        reasons.append("sample floor failed and at least one directional weakness flag is present")
        reasons.extend(directional_quality_flags)
        return "sample_floor_plus_quality_weakness", reasons

    reasons.append("sample floor failed, but directional edge remains positive with no explicit weakness flag")
    if "sample_count_below_absolute_floor" in rejection_reasons:
        reasons.append("absolute floor miss is explicit in rejection_reasons")
    return "sample_floor_only_near_miss", reasons


def _is_metadata_window_source(window_label_source: str) -> bool:
    source = window_label_source.lower()
    return (
        source.endswith("max_age_hours")
        or source.endswith("latest_window_hours")
        or source.endswith("window_hours")
    )


def _build_observation_warnings(
    *,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_signal_pct: float | None,
    sample_count: int | None,
    window_label_source: str,
) -> list[str]:
    warnings: list[str] = []

    if sample_count is None:
        warnings.append("sample_count_missing")
    if median_future_return_pct is None:
        warnings.append("median_future_return_pct_missing")
    if positive_rate_pct is None:
        warnings.append("positive_rate_pct_missing")
    if robustness_signal_pct is None:
        warnings.append("robustness_signal_pct_missing")
    if not _is_metadata_window_source(window_label_source):
        warnings.append(f"window_label_inferred_from={window_label_source}")

    return warnings


def _normalize_diagnostic_observation(
    row: dict[str, Any],
    *,
    source_path: Path,
    source_name: str,
    window_label: str,
    window_label_source: str,
) -> dict[str, Any]:
    symbol = _normalize_symbol(row.get("symbol"))
    strategy = _normalize_strategy(row.get("strategy"))
    horizon = _normalize_horizon(row.get("horizon"))
    identity_key = f"{symbol}:{strategy}:{horizon}"

    rejection_reason = _safe_text(row.get("rejection_reason"))
    rejection_reasons = _normalize_string_list(row.get("rejection_reasons"))
    sample_gate = _safe_text(row.get("sample_gate"))
    quality_gate = _safe_text(row.get("quality_gate"))
    candidate_strength = _safe_text(row.get("candidate_strength"))
    diagnostic_category = _safe_text(row.get("diagnostic_category"))
    classification_reason = _safe_text(row.get("classification_reason"))
    visibility_reason = _safe_text(row.get("visibility_reason"))
    sample_count = _safe_int(row.get("sample_count"))
    labeled_count = _safe_int(row.get("labeled_count"))
    coverage_pct = _safe_float(row.get("coverage_pct"))
    median_future_return_pct = _safe_float(row.get("median_future_return_pct"))
    positive_rate_pct = _safe_float(row.get("positive_rate_pct"))
    robustness_signal = _safe_text(row.get("robustness_signal"))
    robustness_signal_pct = _safe_float(row.get("robustness_signal_pct"))
    aggregate_score = _safe_float(row.get("aggregate_score"))
    sample_band = _sample_count_band(sample_count)

    sample_floor_blocked = _sample_floor_blocked(
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
        sample_gate=sample_gate,
        visibility_reason=visibility_reason,
        sample_count=sample_count,
    )
    structural_non_sample_blockers = _structural_non_sample_blockers(
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
        visibility_reason=visibility_reason,
    )
    directional_quality_flags = _directional_quality_flags(
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        robustness_signal=robustness_signal,
        candidate_strength=candidate_strength,
        diagnostic_category=diagnostic_category,
    )
    classification, classification_reasons = _classify_candidate(
        sample_floor_blocked=sample_floor_blocked,
        median_future_return_pct=median_future_return_pct,
        directional_quality_flags=directional_quality_flags,
        structural_non_sample_blockers=structural_non_sample_blockers,
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
    )

    warnings = _build_observation_warnings(
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        sample_count=sample_count,
        window_label_source=window_label_source,
    )

    return {
        "source_path": str(source_path),
        "source_name": source_name,
        "window_label": window_label,
        "identity": {
            "identity_key": identity_key,
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
        },
        "facts": {
            "rejection_reason": rejection_reason,
            "rejection_reasons": rejection_reasons,
            "diagnostic_category": diagnostic_category,
            "sample_gate": sample_gate,
            "quality_gate": quality_gate,
            "candidate_strength": candidate_strength,
            "classification_reason": classification_reason,
            "visibility_reason": visibility_reason,
            "sample_count": sample_count,
            "labeled_count": labeled_count,
            "coverage_pct": coverage_pct,
            "median_future_return_pct": median_future_return_pct,
            "positive_rate_pct": positive_rate_pct,
            "robustness_signal": robustness_signal,
            "robustness_signal_pct": robustness_signal_pct,
            "aggregate_score": aggregate_score,
        },
        "computed": {
            "sample_count_band": sample_band,
            "is_sample_floor_blocked": sample_floor_blocked,
            "structural_non_sample_blockers": structural_non_sample_blockers,
            "directional_quality_flags": directional_quality_flags,
            "classification": classification,
            "classification_reasons": classification_reasons,
        },
        "warnings": warnings,
    }


def _source_observation_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    identity = _safe_dict(row.get("identity"))
    computed = _safe_dict(row.get("computed"))
    facts = _safe_dict(row.get("facts"))
    return (
        _classification_sort_key(str(computed.get("classification") or "")),
        _window_sort_key(str(row.get("window_label") or "")),
        str(identity.get("identity_key") or ""),
        -(facts.get("sample_count") or -1),
    )


def _build_source_summary(
    path: Path,
    payload: dict[str, Any],
    payload_warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_name = path.name
    window_label, window_label_source, window_facts = _infer_window_label(payload, path)
    source_warnings = list(payload_warnings)
    block, block_path = _extract_edge_candidate_rows_block(payload)

    if block is None:
        source_warnings.append("edge_candidate_rows_block_missing")
        return (
            {
                "source_path": str(path),
                "source_name": source_name,
                "window_label": window_label,
                "window_label_source": window_label_source,
                "edge_candidate_rows_path": None,
                "source_facts": {
                    **window_facts,
                    "edge_candidate_row_count": 0,
                    "diagnostic_row_count": 0,
                    "dominant_rejection_reason": None,
                },
                "observation_count": 0,
                "classification_counts": {},
                "warnings": source_warnings,
            },
            [],
        )

    diagnostic_rows = [
        item for item in _safe_list(block.get("diagnostic_rows")) if isinstance(item, dict)
    ]
    if not diagnostic_rows:
        source_warnings.append("edge_candidate_rows.diagnostic_rows_missing_or_empty")

    observations = [
        _normalize_diagnostic_observation(
            row,
            source_path=path,
            source_name=source_name,
            window_label=window_label,
            window_label_source=window_label_source,
        )
        for row in diagnostic_rows
    ]
    observations.sort(key=_source_observation_sort_key)

    class_counter: Counter[str] = Counter(
        str(_safe_dict(item.get("computed")).get("classification") or "unknown")
        for item in observations
    )
    empty_reason_summary = _safe_dict(block.get("empty_reason_summary"))

    summary = {
        "source_path": str(path),
        "source_name": source_name,
        "window_label": window_label,
        "window_label_source": window_label_source,
        "edge_candidate_rows_path": block_path,
        "source_facts": {
            **window_facts,
            "edge_candidate_row_count": _safe_int(block.get("row_count")) or 0,
            "diagnostic_row_count": _safe_int(block.get("diagnostic_row_count"))
            or len(diagnostic_rows),
            "dominant_rejection_reason": _safe_text(
                empty_reason_summary.get("dominant_rejection_reason")
            ),
        },
        "observation_count": len(observations),
        "classification_counts": _counter_dict(class_counter),
        "warnings": source_warnings,
    }
    return summary, observations


def _group_observations_by_classification(
    observations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        classification: []
        for classification, _ in sorted(
            CLASSIFICATION_ORDER.items(),
            key=lambda item: item[1],
        )
    }

    for observation in observations:
        classification = str(
            _safe_dict(observation.get("computed")).get("classification")
            or "non_sample_primary_failure"
        )
        grouped.setdefault(classification, []).append(observation)

    for rows in grouped.values():
        rows.sort(key=_source_observation_sort_key)

    return grouped


def _build_sample_band_summary(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        band = str(
            _safe_dict(observation.get("computed")).get("sample_count_band") or "unknown"
        )
        grouped[band].append(observation)

    summaries: list[dict[str, Any]] = []
    for band, rows in grouped.items():
        class_counter: Counter[str] = Counter()
        positive_median_count = 0
        non_positive_or_missing_median_count = 0
        medians: list[float] = []
        positive_rates: list[float] = []
        robustness_values: list[float] = []
        identities: set[str] = set()
        repeated_identities: set[str] = set()
        window_map: dict[str, set[str]] = defaultdict(set)

        for observation in rows:
            computed = _safe_dict(observation.get("computed"))
            facts = _safe_dict(observation.get("facts"))
            identity = _safe_dict(observation.get("identity"))
            identity_key = str(identity.get("identity_key") or "")
            classification = str(computed.get("classification") or "unknown")
            median_value = _safe_float(facts.get("median_future_return_pct"))
            positive_rate = _safe_float(facts.get("positive_rate_pct"))
            robustness_value = _safe_float(facts.get("robustness_signal_pct"))

            class_counter[classification] += 1
            if identity_key:
                identities.add(identity_key)
                window_map[identity_key].add(str(observation.get("window_label") or ""))

            if median_value is not None:
                medians.append(median_value)
                if median_value > 0:
                    positive_median_count += 1
                else:
                    non_positive_or_missing_median_count += 1
            else:
                non_positive_or_missing_median_count += 1

            if positive_rate is not None:
                positive_rates.append(positive_rate)
            if robustness_value is not None:
                robustness_values.append(robustness_value)

        for identity_key, window_labels in window_map.items():
            if len(window_labels) > 1:
                repeated_identities.add(identity_key)

        observation_count = len(rows)
        class_counts = _counter_dict(class_counter)
        summaries.append(
            {
                "sample_count_band": band,
                "observation_count": observation_count,
                "unique_identity_count": len(identities),
                "repeated_identity_count": len(repeated_identities),
                "classification_counts": class_counts,
                "classification_ratios": {
                    key: _ratio(count, observation_count)
                    for key, count in class_counts.items()
                },
                "quality_characteristics": {
                    "positive_median_count": positive_median_count,
                    "positive_median_ratio": _ratio(positive_median_count, observation_count),
                    "non_positive_or_missing_median_count": non_positive_or_missing_median_count,
                    "non_positive_or_missing_median_ratio": _ratio(
                        non_positive_or_missing_median_count,
                        observation_count,
                    ),
                    "median_of_median_future_return_pct": _median_or_none(medians),
                    "median_positive_rate_pct": _median_or_none(positive_rates),
                    "median_robustness_signal_pct": _median_or_none(robustness_values),
                },
            }
        )

    summaries.sort(key=lambda item: _sample_band_sort_key(str(item["sample_count_band"])))
    return summaries


def _identity_summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _classification_sort_key(str(row.get("best_classification") or "")),
        -int(row.get("distinct_window_count") or 0),
        -int(row.get("highest_sample_count") or 0),
        -float(row.get("best_median_future_return_pct") or -999999.0),
        str(row.get("identity_key") or ""),
    )


def _build_repeated_identity_summary(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        identity_key = str(
            _safe_dict(observation.get("identity")).get("identity_key") or ""
        )
        if identity_key:
            grouped[identity_key].append(observation)

    repeated: list[dict[str, Any]] = []
    for identity_key, rows in grouped.items():
        window_labels = sorted(
            {str(row.get("window_label") or "") for row in rows},
            key=_window_sort_key,
        )
        if len(window_labels) <= 1:
            continue

        first_identity = _safe_dict(rows[0].get("identity"))
        class_counter: Counter[str] = Counter()
        best_classification = "non_sample_primary_failure"
        highest_sample_count = 0
        best_median: float | None = None
        observation_rows: list[dict[str, Any]] = []

        for row in sorted(rows, key=_source_observation_sort_key):
            facts = _safe_dict(row.get("facts"))
            computed = _safe_dict(row.get("computed"))
            classification = str(
                computed.get("classification") or "non_sample_primary_failure"
            )
            class_counter[classification] += 1

            if _classification_sort_key(classification) < _classification_sort_key(
                best_classification
            ):
                best_classification = classification

            sample_count = _safe_int(facts.get("sample_count")) or 0
            highest_sample_count = max(highest_sample_count, sample_count)

            median_value = _safe_float(facts.get("median_future_return_pct"))
            if median_value is not None and (
                best_median is None or median_value > best_median
            ):
                best_median = median_value

            observation_rows.append(
                {
                    "window_label": row.get("window_label"),
                    "classification": classification,
                    "sample_count_band": computed.get("sample_count_band"),
                    "sample_count": facts.get("sample_count"),
                    "median_future_return_pct": facts.get("median_future_return_pct"),
                    "positive_rate_pct": facts.get("positive_rate_pct"),
                    "robustness_signal_pct": facts.get("robustness_signal_pct"),
                    "rejection_reason": facts.get("rejection_reason"),
                    "classification_reasons": computed.get("classification_reasons"),
                }
            )

        repeated.append(
            {
                "identity_key": identity_key,
                "symbol": first_identity.get("symbol"),
                "strategy": first_identity.get("strategy"),
                "horizon": first_identity.get("horizon"),
                "observation_count": len(rows),
                "distinct_window_count": len(window_labels),
                "window_labels": window_labels,
                "classification_counts": _counter_dict(class_counter),
                "best_classification": best_classification,
                "highest_sample_count": highest_sample_count,
                "best_median_future_return_pct": best_median,
                "observations": observation_rows,
            }
        )

    repeated.sort(key=_identity_summary_sort_key)
    return repeated


def _build_near_miss_interest_ranking(
    repeated_identities: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        identity_key = str(
            _safe_dict(observation.get("identity")).get("identity_key") or ""
        )
        if identity_key:
            grouped[identity_key].append(observation)

    ranking_rows: list[dict[str, Any]] = []
    repeated_lookup = {
        str(item.get("identity_key") or ""): item for item in repeated_identities
    }

    for identity_key, rows in grouped.items():
        class_counter: Counter[str] = Counter()
        best_classification = "non_sample_primary_failure"
        highest_sample_count = 0
        best_median: float | None = None
        first_identity = _safe_dict(rows[0].get("identity"))
        windows = sorted(
            {str(row.get("window_label") or "") for row in rows},
            key=_window_sort_key,
        )

        for row in rows:
            computed = _safe_dict(row.get("computed"))
            facts = _safe_dict(row.get("facts"))
            classification = str(
                computed.get("classification") or "non_sample_primary_failure"
            )
            class_counter[classification] += 1
            if _classification_sort_key(classification) < _classification_sort_key(
                best_classification
            ):
                best_classification = classification

            sample_count = _safe_int(facts.get("sample_count")) or 0
            highest_sample_count = max(highest_sample_count, sample_count)

            median_value = _safe_float(facts.get("median_future_return_pct"))
            if median_value is not None and (
                best_median is None or median_value > best_median
            ):
                best_median = median_value

        if best_classification not in NEAR_MISS_CLASSIFICATIONS:
            continue

        ranking_rows.append(
            {
                "identity_key": identity_key,
                "symbol": first_identity.get("symbol"),
                "strategy": first_identity.get("strategy"),
                "horizon": first_identity.get("horizon"),
                "best_classification": best_classification,
                "distinct_window_count": len(windows),
                "window_labels": windows,
                "highest_sample_count": highest_sample_count,
                "best_median_future_return_pct": best_median,
                "classification_counts": _counter_dict(class_counter),
                "repeated_identity": identity_key in repeated_lookup,
            }
        )

    ranking_rows.sort(key=_identity_summary_sort_key)

    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index

    return ranking_rows


def build_sub_floor_candidate_validation_report(
    summary_paths: list[Path],
) -> dict[str, Any]:
    source_summaries: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    warnings: list[str] = [
        "Classification is diagnostic-only and does not recommend changing analyzer thresholds."
    ]

    for raw_path in summary_paths:
        path = Path(raw_path)
        payload, payload_warnings = load_summary_json(path)
        source_summary, source_observations = _build_source_summary(
            path,
            payload,
            payload_warnings,
        )
        source_summaries.append(source_summary)
        observations.extend(source_observations)
        warnings.extend(f"{path.name}: {warning}" for warning in source_summary["warnings"])

    observations.sort(key=_source_observation_sort_key)
    classified_candidates = _group_observations_by_classification(observations)
    repeated_identity_summary = _build_repeated_identity_summary(observations)
    near_miss_interest_ranking = _build_near_miss_interest_ranking(
        repeated_identity_summary,
        observations,
    )
    sample_band_summary = _build_sample_band_summary(observations)

    classification_counts = Counter(
        str(_safe_dict(item.get("computed")).get("classification") or "unknown")
        for item in observations
    )
    sample_band_counts = Counter(
        str(_safe_dict(item.get("computed")).get("sample_count_band") or "unknown")
        for item in observations
    )
    window_labels = sorted(
        {str(item.get("window_label") or "") for item in observations},
        key=_window_sort_key,
    )
    unique_identities = {
        str(_safe_dict(item.get("identity")).get("identity_key") or "")
        for item in observations
        if _safe_dict(item.get("identity")).get("identity_key")
    }

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": REPORT_TYPE,
            "classification_version": CLASSIFICATION_VERSION,
            "summary_input_count": len(summary_paths),
            "sample_count_bands": [band for band, _, _ in SAMPLE_COUNT_BANDS],
            "diagnostic_heuristics": {
                "positive_rate_min_pct": DIAGNOSTIC_POSITIVE_RATE_MIN_PCT,
                "robustness_min_pct": DIAGNOSTIC_ROBUSTNESS_MIN_PCT,
                "note": "Diagnostic-only conservative heuristics; not production gates.",
            },
        },
        "source_summaries": source_summaries,
        "overall_summary": {
            "observation_count": len(observations),
            "unique_identity_count": len(unique_identities),
            "window_labels": window_labels,
            "classification_counts": _counter_dict(classification_counts),
            "sample_count_band_counts": _counter_dict(sample_band_counts),
            "repeated_identity_count": len(repeated_identity_summary),
            "near_miss_interest_count": len(near_miss_interest_ranking),
        },
        "classified_candidates": classified_candidates,
        "repeated_identity_summary": repeated_identity_summary,
        "near_miss_interest_ranking": near_miss_interest_ranking,
        "sample_band_summary": sample_band_summary,
        "warnings": sorted(set(warnings)),
    }


def build_sub_floor_candidate_validation_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    overall = _safe_dict(summary.get("overall_summary"))
    classified_candidates = _safe_dict(summary.get("classified_candidates"))
    repeated_identity_summary = _safe_list(summary.get("repeated_identity_summary"))
    near_miss_interest_ranking = _safe_list(summary.get("near_miss_interest_ranking"))
    sample_band_summary = _safe_list(summary.get("sample_band_summary"))
    source_summaries = _safe_list(summary.get("source_summaries"))
    warning_rows = _normalize_string_list(summary.get("warnings"))
    diagnostic_heuristics = _safe_dict(metadata.get("diagnostic_heuristics"))

    lines = [
        "# Sub-Floor Candidate Validation",
        "",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- classification_version: {metadata.get('classification_version', 'n/a')}",
        f"- summary_input_count: {metadata.get('summary_input_count', 0)}",
        f"- observation_count: {overall.get('observation_count', 0)}",
        f"- unique_identity_count: {overall.get('unique_identity_count', 0)}",
        f"- repeated_identity_count: {overall.get('repeated_identity_count', 0)}",
        "",
        "## Diagnostic Heuristics",
        f"- positive_rate_min_pct: {diagnostic_heuristics.get('positive_rate_min_pct', 'n/a')}",
        f"- robustness_min_pct: {diagnostic_heuristics.get('robustness_min_pct', 'n/a')}",
        f"- note: {diagnostic_heuristics.get('note', 'n/a')}",
        "",
        "## Source Inputs",
    ]

    if not source_summaries:
        lines.append("- none")
    else:
        for source in source_summaries:
            source_facts = _safe_dict(source.get("source_facts"))
            lines.append(
                f"- {source.get('window_label', 'n/a')} "
                f"({source.get('source_name', 'n/a')}): "
                f"diagnostic_rows={source_facts.get('diagnostic_row_count', 0)}, "
                f"observations={source.get('observation_count', 0)}, "
                f"dominant_rejection_reason={source_facts.get('dominant_rejection_reason', 'n/a')}, "
                f"window_label_source={source.get('window_label_source', 'n/a')}"
            )

    lines.extend(["", "## Classification Counts"])
    class_counts = _safe_dict(overall.get("classification_counts"))
    if not class_counts:
        lines.append("- none")
    else:
        for classification, count in class_counts.items():
            lines.append(f"- {classification}: {count}")

    for classification in sorted(classified_candidates.keys(), key=_classification_sort_key):
        rows = _safe_list(classified_candidates.get(classification))
        lines.extend(["", f"## {classification}"])
        if not rows:
            lines.append("- none")
            continue

        for row in rows:
            identity = _safe_dict(row.get("identity"))
            facts = _safe_dict(row.get("facts"))
            computed = _safe_dict(row.get("computed"))
            lines.append(
                f"- {identity.get('symbol', 'n/a')} / {identity.get('strategy', 'n/a')} / {identity.get('horizon', 'n/a')} "
                f"@ {row.get('window_label', 'n/a')}: "
                f"sample={facts.get('sample_count', 'n/a')} ({computed.get('sample_count_band', 'n/a')}), "
                f"median={_format_number(_safe_float(facts.get('median_future_return_pct')))}, "
                f"positive_rate={_format_number(_safe_float(facts.get('positive_rate_pct')))}, "
                f"rejection={facts.get('rejection_reason', 'n/a')}, "
                f"structural_blockers={computed.get('structural_non_sample_blockers', [])}"
            )

    lines.extend(["", "## Near-Miss Interest Ranking"])
    if not near_miss_interest_ranking:
        lines.append("- none")
    else:
        for row in near_miss_interest_ranking:
            lines.append(
                f"- rank={row.get('rank', 'n/a')}: "
                f"{row.get('identity_key', 'n/a')} | "
                f"best_classification={row.get('best_classification', 'n/a')} | "
                f"windows={', '.join(_normalize_string_list(row.get('window_labels'))) or 'n/a'} | "
                f"highest_sample_count={row.get('highest_sample_count', 'n/a')} | "
                f"best_median={_format_number(_safe_float(row.get('best_median_future_return_pct')))} | "
                f"repeated_identity={row.get('repeated_identity', False)}"
            )

    lines.extend(["", "## Repeated Identities"])
    if not repeated_identity_summary:
        lines.append("- none")
    else:
        for row in repeated_identity_summary:
            lines.append(
                f"- {row.get('identity_key', 'n/a')}: "
                f"windows={', '.join(_normalize_string_list(row.get('window_labels'))) or 'n/a'}, "
                f"best_classification={row.get('best_classification', 'n/a')}, "
                f"highest_sample_count={row.get('highest_sample_count', 'n/a')}, "
                f"best_median={_format_number(_safe_float(row.get('best_median_future_return_pct')))}"
            )

    lines.extend(["", "## Sample Bands"])
    if not sample_band_summary:
        lines.append("- none")
    else:
        for row in sample_band_summary:
            quality = _safe_dict(row.get("quality_characteristics"))
            lines.append(
                f"- {row.get('sample_count_band', 'n/a')}: "
                f"observations={row.get('observation_count', 0)}, "
                f"classification_counts={row.get('classification_counts', {})}, "
                f"positive_median_ratio={_format_number(_safe_float(quality.get('positive_median_ratio')))}, "
                f"median_of_medians={_format_number(_safe_float(quality.get('median_of_median_future_return_pct')))}"
            )

    lines.extend(["", "## Warnings"])
    if not warning_rows:
        lines.append("- none")
    else:
        for warning in warning_rows:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def write_sub_floor_candidate_validation_report(
    *,
    summary: dict[str, Any],
    markdown: str,
    json_output_path: Path,
    markdown_output_path: Path,
) -> dict[str, str]:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)

    json_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_output_path.write_text(markdown, encoding="utf-8")

    return {
        "summary_json": str(json_output_path),
        "summary_md": str(markdown_output_path),
    }


def run_sub_floor_candidate_validation(
    *,
    summary_paths: list[Path],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    summary = build_sub_floor_candidate_validation_report(summary_paths)
    markdown = build_sub_floor_candidate_validation_markdown(summary)

    result: dict[str, Any] = {
        "summary": summary,
        "markdown": markdown,
    }

    if output_dir is not None:
        outputs = write_sub_floor_candidate_validation_report(
            summary=summary,
            markdown=markdown,
            json_output_path=output_dir / DEFAULT_JSON_OUTPUT_NAME,
            markdown_output_path=output_dir / DEFAULT_MD_OUTPUT_NAME,
        )
        result.update(outputs)

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether recent-window sub-floor edge candidates look like "
            "conservative near-miss signals or thin-sample noise."
        )
    )
    parser.add_argument(
        "summary_paths",
        type=Path,
        nargs="+",
        help="One or more analyzer summary JSON paths to inspect",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory where JSON and Markdown reports will be written",
    )
    parser.add_argument(
        "--stdout-format",
        choices=("json", "markdown"),
        default="json",
        help="Used only when --output-dir is omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_sub_floor_candidate_validation(
        summary_paths=args.summary_paths,
        output_dir=args.output_dir,
    )

    if args.output_dir is not None:
        print(
            json.dumps(
                {
                    "summary_json": result["summary_json"],
                    "summary_md": result["summary_md"],
                    "overall_summary": result["summary"]["overall_summary"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.stdout_format == "markdown":
        print(result["markdown"])
        return

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()