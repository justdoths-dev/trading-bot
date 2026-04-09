from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from src.research import sub_floor_candidate_validation as baseline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "research_reports" / "latest"

REPORT_TYPE = "historical_persistence_diagnosis"
REPORT_TITLE = "Historical Persistence Diagnosis"
REPORT_VERSION = "v1"

DEFAULT_SUMMARY_OUTPUT_NAME = "historical_persistence_diagnosis_summary.json"
DEFAULT_IDENTITY_OUTPUT_NAME = "historical_persistence_identity_summary.json"
DEFAULT_APPEARANCE_OUTPUT_NAME = "historical_persistence_appearance_rows.jsonl"
DEFAULT_MD_OUTPUT_NAME = "historical_persistence_diagnosis_summary.md"

ELIGIBLE_CLASSIFICATION = "eligible"
PERSISTENCE_LABELS = (
    "early_stage_growth_candidate",
    "unstable_recurrent_candidate",
    "noise_like_singleton",
)
SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE = (
    "prefer_eligible_rows_then_highest_sample_count_for_same_snapshot_identity"
)
TIMESTAMP_KEYS = (
    "generated_at",
    "timestamp",
    "logged_at",
    "created_at",
    "run_at",
    "completed_at",
)

MIN_SAMPLE_COUNT_FOR_GROWTH_PCT_RELIABILITY = 10
MIN_POSITIVE_RATE_FOR_EARLY_STAGE = 50.0
MIN_ROBUSTNESS_SIGNAL_FOR_EARLY_STAGE = 50.0


def resolve_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _safe_datetime(value: Any) -> datetime | None:
    text = baseline._safe_text(value)
    if text is None:
        return None

    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _extract_snapshot_timestamp(
    payload: dict[str, Any],
) -> tuple[str | None, datetime | None, str | None]:
    found = baseline._recursive_find_first_scalar_with_path(
        payload,
        set(TIMESTAMP_KEYS),
    )
    if found is None:
        return None, None, None

    path_parts, raw_value = found
    parsed = _safe_datetime(raw_value)
    if parsed is None:
        return baseline._safe_text(raw_value), None, baseline._path_to_label(path_parts)
    return parsed.isoformat(), parsed, baseline._path_to_label(path_parts)


def _median_number(values: list[int | float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 4)


def _ratio_or_none(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return baseline._ratio(count, total)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _normalize_filters(
    *,
    symbol: str | None,
    strategy: str | None,
    horizon: str | None,
) -> dict[str, str | None]:
    return {
        "symbol": baseline._normalize_symbol(symbol) if symbol else None,
        "strategy": baseline._normalize_strategy(strategy) if strategy else None,
        "horizon": baseline._normalize_horizon(horizon) if horizon else None,
    }


def _matches_filters(
    row: dict[str, Any],
    filters: dict[str, str | None],
) -> bool:
    if filters["symbol"] and row.get("symbol") != filters["symbol"]:
        return False
    if filters["strategy"] and row.get("strategy") != filters["strategy"]:
        return False
    if filters["horizon"] and row.get("horizon") != filters["horizon"]:
        return False
    return True


def _candidate_snapshot_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    timestamp = value.get("snapshot_timestamp_parsed")
    return (
        timestamp is None,
        timestamp or datetime.max.replace(tzinfo=UTC),
        str(value.get("path") or ""),
    )


def _looks_like_snapshot_payload(payload: dict[str, Any]) -> bool:
    block, _ = baseline._extract_edge_candidate_rows_block(payload)
    if block is None:
        return False

    if baseline._safe_list(block.get("diagnostic_rows")):
        return True
    if baseline._safe_list(block.get("rows")):
        return True
    if baseline._safe_list(block.get("identity_horizon_evaluations")):
        return True
    return False


def discover_history_summary_paths(
    history_dir: Path,
) -> tuple[list[Path], dict[str, Any], list[str]]:
    resolved_dir = history_dir.expanduser()
    if not resolved_dir.is_absolute():
        resolved_dir = REPO_ROOT / resolved_dir
    resolved_dir = resolved_dir.resolve()

    if not resolved_dir.exists():
        raise FileNotFoundError(f"History directory does not exist: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"History directory is not a directory: {resolved_dir}")

    candidate_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    scanned_json_count = 0
    skipped_non_snapshot_count = 0

    for path in sorted(resolved_dir.rglob("*.json")):
        scanned_json_count += 1
        payload, payload_warnings = baseline.load_summary_json(path)
        if payload_warnings:
            warnings.extend(f"{path.name}: {warning}" for warning in payload_warnings)
            skipped_non_snapshot_count += 1
            continue

        metadata = baseline._safe_dict(payload.get("metadata"))
        if baseline._safe_text(metadata.get("report_type")) == REPORT_TYPE:
            skipped_non_snapshot_count += 1
            continue

        if not _looks_like_snapshot_payload(payload):
            skipped_non_snapshot_count += 1
            continue

        timestamp_text, timestamp_parsed, timestamp_source = _extract_snapshot_timestamp(
            payload
        )
        candidate_rows.append(
            {
                "path": path,
                "snapshot_timestamp": timestamp_text,
                "snapshot_timestamp_parsed": timestamp_parsed,
                "snapshot_timestamp_source": timestamp_source,
            }
        )

    candidate_rows.sort(key=_candidate_snapshot_sort_key)
    if not candidate_rows:
        raise FileNotFoundError(
            "No analyzer-like historical snapshot JSON files were found under "
            f"{resolved_dir}"
        )

    timestamp_count = sum(
        1 for item in candidate_rows if item["snapshot_timestamp_parsed"] is not None
    )
    chronology_verified = timestamp_count == len(candidate_rows)
    chronology_basis = (
        "history_dir_timestamp_sort"
        if chronology_verified
        else "history_dir_timestamp_then_path_unverified"
    )
    if not chronology_verified:
        warnings.append(
            "Some discovered snapshots were missing parseable timestamps; chronology "
            "falls back to path order for those files."
        )

    return (
        [Path(item["path"]) for item in candidate_rows],
        {
            "discovery_mode": "history_dir",
            "history_dir": str(resolved_dir),
            "chronology_basis": chronology_basis,
            "chronology_verified": chronology_verified,
            "scanned_json_count": scanned_json_count,
            "discovered_snapshot_count": len(candidate_rows),
            "skipped_non_snapshot_count": skipped_non_snapshot_count,
        },
        sorted(set(warnings)),
    )


def resolve_history_paths(
    *,
    summary_paths: list[Path] | None,
    history_dir: Path | None,
    limit_snapshots: int | None,
) -> tuple[list[Path], dict[str, Any], list[str]]:
    normalized_summary_paths = [Path(path) for path in (summary_paths or [])]
    if normalized_summary_paths and history_dir is not None:
        raise ValueError("Use either summary_paths or --history-dir, not both.")

    if limit_snapshots is not None and limit_snapshots <= 0:
        raise ValueError("--limit-snapshots must be greater than zero when provided.")

    if history_dir is not None:
        paths, discovery, warnings = discover_history_summary_paths(history_dir)
    elif normalized_summary_paths:
        paths = normalized_summary_paths
        discovery = {
            "discovery_mode": "explicit_summary_paths",
            "history_dir": None,
            "chronology_basis": "input_order_unverified",
            "chronology_verified": False,
            "scanned_json_count": len(paths),
            "discovered_snapshot_count": len(paths),
            "skipped_non_snapshot_count": 0,
        }
        warnings = [
            "Explicit summary_paths are used in the order provided; pass snapshots "
            "oldest-to-newest."
        ]
    else:
        raise ValueError("Provide summary_paths or --history-dir.")

    if limit_snapshots is not None:
        paths = paths[-limit_snapshots:]

    return paths, discovery, warnings


def _select_robustness_signal(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for label, keys in (
        ("signal_match_rate_pct", ("signal_match_rate_pct", "signal_match_rate")),
        ("bias_match_rate_pct", ("bias_match_rate_pct", "bias_match_rate")),
        ("coverage_pct", ("coverage_pct",)),
    ):
        for key in keys:
            value = baseline._safe_float(metrics.get(key))
            if value is not None:
                return label, value
    return None, None


def _normalize_historical_horizon_evaluation_row(
    raw_row: dict[str, Any],
    *,
    default_symbol: str | None = None,
    default_strategy: str | None = None,
    default_horizon: str | None = None,
) -> dict[str, Any]:
    metrics = baseline._safe_dict(raw_row.get("metrics"))
    robustness_signal, robustness_signal_pct = _select_robustness_signal(metrics)

    symbol = baseline._normalize_symbol(raw_row.get("symbol") or default_symbol)
    strategy = baseline._normalize_strategy(raw_row.get("strategy") or default_strategy)
    horizon = baseline._normalize_horizon(raw_row.get("horizon") or default_horizon)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "status": baseline._safe_text(raw_row.get("status")) or "rejected",
        "diagnostic_category": baseline._safe_text(raw_row.get("diagnostic_category")),
        "strategy_horizon_compatible": (
            raw_row.get("strategy_horizon_compatible")
            if isinstance(raw_row.get("strategy_horizon_compatible"), bool)
            else None
        ),
        "rejection_reason": baseline._safe_text(raw_row.get("rejection_reason")),
        "rejection_reasons": baseline._normalize_string_list(
            raw_row.get("rejection_reasons")
        ),
        "sample_gate": baseline._safe_text(raw_row.get("sample_gate")),
        "quality_gate": baseline._safe_text(raw_row.get("quality_gate")),
        "candidate_strength": baseline._safe_text(raw_row.get("candidate_strength")),
        "classification_reason": baseline._safe_text(
            raw_row.get("classification_reason")
        ),
        "aggregate_score": baseline._safe_float(raw_row.get("aggregate_score")),
        "visibility_reason": baseline._safe_text(raw_row.get("visibility_reason")),
        "sample_count": baseline._safe_int(metrics.get("sample_count")),
        "labeled_count": baseline._safe_int(metrics.get("labeled_count")),
        "coverage_pct": baseline._safe_float(metrics.get("coverage_pct")),
        "median_future_return_pct": baseline._safe_float(
            metrics.get("median_future_return_pct")
        ),
        "avg_future_return_pct": baseline._safe_float(metrics.get("avg_future_return_pct")),
        "positive_rate_pct": baseline._safe_float(metrics.get("positive_rate_pct")),
        "robustness_signal": robustness_signal,
        "robustness_signal_pct": robustness_signal_pct,
    }


def _normalize_historical_selected_evaluation_row(
    raw_row: dict[str, Any],
    *,
    default_symbol: str | None = None,
    default_strategy: str | None = None,
    default_horizon: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_historical_horizon_evaluation_row(
        raw_row,
        default_symbol=default_symbol,
        default_strategy=default_strategy,
        default_horizon=default_horizon,
    )
    return {
        "symbol": normalized["symbol"],
        "strategy": normalized["strategy"],
        "horizon": normalized["horizon"],
        "status": normalized["status"],
        "strategy_horizon_compatible": normalized["strategy_horizon_compatible"],
        "selected_candidate_strength": normalized["candidate_strength"],
        "sample_count": normalized["sample_count"],
        "labeled_count": normalized["labeled_count"],
        "coverage_pct": normalized["coverage_pct"],
        "median_future_return_pct": normalized["median_future_return_pct"],
        "avg_future_return_pct": normalized["avg_future_return_pct"],
        "positive_rate_pct": normalized["positive_rate_pct"],
        "robustness_signal": normalized["robustness_signal"],
        "robustness_signal_pct": normalized["robustness_signal_pct"],
        "aggregate_score": normalized["aggregate_score"],
        "visibility_reason": normalized["visibility_reason"],
    }


def _extract_rows_from_identity_horizon_evaluations(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    block = baseline._safe_dict(payload.get("edge_candidate_rows"))
    identity_evaluations = baseline._safe_list(block.get("identity_horizon_evaluations"))

    diagnostic_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for identity_eval in identity_evaluations:
        identity_eval = baseline._safe_dict(identity_eval)
        symbol = baseline._safe_text(identity_eval.get("symbol"))
        strategy = baseline._safe_text(identity_eval.get("strategy"))
        horizon_evaluations = baseline._safe_dict(identity_eval.get("horizon_evaluations"))

        for horizon_name, raw_row in horizon_evaluations.items():
            if not isinstance(raw_row, dict):
                continue

            status = baseline._safe_text(raw_row.get("status")) or "rejected"
            if status == "selected":
                selected_rows.append(
                    _normalize_historical_selected_evaluation_row(
                        raw_row,
                        default_symbol=symbol,
                        default_strategy=strategy,
                        default_horizon=str(horizon_name),
                    )
                )
                continue

            diagnostic_rows.append(
                _normalize_historical_horizon_evaluation_row(
                    raw_row,
                    default_symbol=symbol,
                    default_strategy=strategy,
                    default_horizon=str(horizon_name),
                )
            )

    return diagnostic_rows, selected_rows


def _normalize_diagnostic_appearance_row(
    row: dict[str, Any],
    *,
    snapshot_index: int,
    snapshot_timestamp: str | None,
    snapshot_path: Path,
    source_name: str,
    window_label: str,
    window_label_source: str,
    appearance_source: str,
) -> dict[str, Any]:
    symbol = baseline._normalize_symbol(row.get("symbol"))
    strategy = baseline._normalize_strategy(row.get("strategy"))
    horizon = baseline._normalize_horizon(row.get("horizon"))
    identity_key = f"{symbol}:{strategy}:{horizon}"

    appearance_status = baseline._safe_text(row.get("status")) or "rejected"
    rejection_reason = baseline._safe_text(row.get("rejection_reason"))
    rejection_reasons = baseline._normalize_string_list(row.get("rejection_reasons"))
    sample_gate = baseline._safe_text(row.get("sample_gate"))
    quality_gate = baseline._safe_text(row.get("quality_gate"))
    candidate_strength = baseline._safe_text(row.get("candidate_strength"))
    diagnostic_category = baseline._safe_text(row.get("diagnostic_category"))
    classification_reason = baseline._safe_text(row.get("classification_reason"))
    visibility_reason = baseline._safe_text(row.get("visibility_reason"))
    sample_count = baseline._safe_int(row.get("sample_count"))
    labeled_row_count = baseline._safe_int(row.get("labeled_count"))
    median_future_return_pct = baseline._safe_float(row.get("median_future_return_pct"))
    positive_rate_pct = baseline._safe_float(row.get("positive_rate_pct"))
    robustness_signal = baseline._safe_text(row.get("robustness_signal"))
    robustness_signal_pct = baseline._safe_float(row.get("robustness_signal_pct"))
    edge_score = baseline._safe_float(
        row.get("edge_score") or row.get("aggregate_score")
    )
    horizon_compatible = (
        row.get("strategy_horizon_compatible")
        if isinstance(row.get("strategy_horizon_compatible"), bool)
        else None
    )

    sample_floor_blocked = baseline._sample_floor_blocked(
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
        sample_gate=sample_gate,
        visibility_reason=visibility_reason,
        sample_count=sample_count,
    )
    structural_non_sample_blockers = baseline._structural_non_sample_blockers(
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
        visibility_reason=visibility_reason,
    )
    directional_quality_flags = baseline._directional_quality_flags(
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        robustness_signal=robustness_signal,
        candidate_strength=candidate_strength,
        diagnostic_category=diagnostic_category,
    )
    classification, classification_reasons = baseline._classify_candidate(
        sample_floor_blocked=sample_floor_blocked,
        median_future_return_pct=median_future_return_pct,
        directional_quality_flags=directional_quality_flags,
        structural_non_sample_blockers=structural_non_sample_blockers,
        rejection_reason=rejection_reason,
        rejection_reasons=rejection_reasons,
    )
    warnings = baseline._build_observation_warnings(
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        sample_count=sample_count,
        window_label_source=window_label_source,
    )

    return {
        "snapshot_index": snapshot_index,
        "snapshot_timestamp": snapshot_timestamp,
        "snapshot_path": str(snapshot_path),
        "source_name": source_name,
        "window_label": window_label,
        "window_label_source": window_label_source,
        "appearance_source": appearance_source,
        "snapshot_kind": "diagnostic",
        "identity_key": identity_key,
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "appearance_status": appearance_status,
        "sample_count": sample_count,
        "sample_count_band": baseline._sample_count_band(sample_count),
        "classification": classification,
        "classification_reasons": classification_reasons,
        "eligible": False,
        "primary_rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons,
        "has_labeled_rows": (
            None if labeled_row_count is None else labeled_row_count > 0
        ),
        "labeled_row_count": labeled_row_count,
        "horizon_compatible": horizon_compatible,
        "edge_score": edge_score,
        "win_rate_pct": positive_rate_pct,
        "pnl_pct": median_future_return_pct,
        "diagnostic_category": diagnostic_category,
        "sample_gate": sample_gate,
        "quality_gate": quality_gate,
        "candidate_strength": candidate_strength,
        "classification_reason": classification_reason,
        "visibility_reason": visibility_reason,
        "robustness_signal": robustness_signal,
        "robustness_signal_pct": robustness_signal_pct,
        "structural_non_sample_blockers": structural_non_sample_blockers,
        "directional_quality_flags": directional_quality_flags,
        "warnings": warnings,
    }


def _normalize_selected_appearance_row(
    row: dict[str, Any],
    *,
    snapshot_index: int,
    snapshot_timestamp: str | None,
    snapshot_path: Path,
    source_name: str,
    window_label: str,
    window_label_source: str,
    appearance_source: str,
) -> dict[str, Any]:
    symbol = baseline._normalize_symbol(row.get("symbol"))
    strategy = baseline._normalize_strategy(row.get("strategy"))
    horizon = baseline._normalize_horizon(row.get("horizon"))
    identity_key = f"{symbol}:{strategy}:{horizon}"

    sample_count = baseline._safe_int(row.get("sample_count"))
    labeled_row_count = baseline._safe_int(row.get("labeled_count"))
    median_future_return_pct = baseline._safe_float(
        row.get("pnl_pct")
        or row.get("median_future_return_pct")
        or row.get("avg_future_return_pct")
    )
    positive_rate_pct = baseline._safe_float(
        row.get("win_rate_pct") or row.get("positive_rate_pct")
    )
    robustness_signal = baseline._safe_text(row.get("robustness_signal"))
    robustness_signal_pct = baseline._safe_float(row.get("robustness_signal_pct"))
    edge_score = baseline._safe_float(
        row.get("edge_score") or row.get("aggregate_score")
    )
    appearance_status = baseline._safe_text(row.get("status")) or "selected"
    horizon_compatible = (
        row.get("strategy_horizon_compatible")
        if isinstance(row.get("strategy_horizon_compatible"), bool)
        else True
    )
    warnings = baseline._build_observation_warnings(
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        sample_count=sample_count,
        window_label_source=window_label_source,
    )
    if sample_count is not None and sample_count < 30:
        warnings.append("eligible_row_sample_count_below_30")

    return {
        "snapshot_index": snapshot_index,
        "snapshot_timestamp": snapshot_timestamp,
        "snapshot_path": str(snapshot_path),
        "source_name": source_name,
        "window_label": window_label,
        "window_label_source": window_label_source,
        "appearance_source": appearance_source,
        "snapshot_kind": "eligible",
        "identity_key": identity_key,
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "appearance_status": appearance_status,
        "sample_count": sample_count,
        "sample_count_band": baseline._sample_count_band(sample_count),
        "classification": ELIGIBLE_CLASSIFICATION,
        "classification_reasons": [
            "row is present in edge_candidate_rows.rows for this snapshot"
        ],
        "eligible": True,
        "primary_rejection_reason": None,
        "rejection_reasons": [],
        "has_labeled_rows": (
            None if labeled_row_count is None else labeled_row_count > 0
        ),
        "labeled_row_count": labeled_row_count,
        "horizon_compatible": horizon_compatible,
        "edge_score": edge_score,
        "win_rate_pct": positive_rate_pct,
        "pnl_pct": median_future_return_pct,
        "diagnostic_category": None,
        "sample_gate": "passed",
        "quality_gate": "passed",
        "candidate_strength": baseline._safe_text(
            row.get("selected_candidate_strength") or row.get("candidate_strength")
        ),
        "classification_reason": baseline._safe_text(row.get("classification_reason")),
        "visibility_reason": baseline._safe_text(row.get("visibility_reason"))
        or "passed_sample_and_quality_gate",
        "robustness_signal": robustness_signal,
        "robustness_signal_pct": robustness_signal_pct,
        "structural_non_sample_blockers": [],
        "directional_quality_flags": [],
        "warnings": warnings,
    }


def _appearance_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("snapshot_index") or 0),
        0 if row.get("snapshot_kind") == "eligible" else 1,
        str(row.get("identity_key") or ""),
        -(row.get("sample_count") or -1),
        str(row.get("primary_rejection_reason") or ""),
    )


def _same_snapshot_duplicate_precedence(row: dict[str, Any]) -> tuple[Any, ...]:
    horizon_compatible = row.get("horizon_compatible")
    if horizon_compatible is True:
        compatibility_rank = 0
    elif horizon_compatible is False:
        compatibility_rank = 2
    else:
        compatibility_rank = 1

    return (
        0 if row.get("eligible") else 1,
        compatibility_rank,
        -(row.get("sample_count") or -1),
        -(row.get("labeled_row_count") or -1),
        baseline._classification_sort_key(str(row.get("classification") or "")),
        str(row.get("primary_rejection_reason") or ""),
        str(row.get("appearance_source") or ""),
    )


def _deduplicate_same_snapshot_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_rows: list[dict[str, Any]] = []

    for row in rows:
        identity_key = baseline._safe_text(row.get("identity_key"))
        if identity_key is None:
            unique_rows.append(row)
            continue
        grouped[identity_key].append(row)

    duplicate_identity_count = 0
    cross_kind_duplicate_identity_count = 0
    dropped_observation_count = 0

    for group_rows in grouped.values():
        if len(group_rows) == 1:
            unique_rows.append(group_rows[0])
            continue

        duplicate_identity_count += 1
        if len({str(item.get("snapshot_kind") or "") for item in group_rows}) > 1:
            cross_kind_duplicate_identity_count += 1
        dropped_observation_count += len(group_rows) - 1
        chosen = min(group_rows, key=_same_snapshot_duplicate_precedence)
        unique_rows.append(chosen)

    unique_rows.sort(key=_appearance_sort_key)
    return unique_rows, {
        "same_snapshot_duplicate_identity_count": duplicate_identity_count,
        "same_snapshot_cross_kind_duplicate_identity_count": (
            cross_kind_duplicate_identity_count
        ),
        "same_snapshot_dropped_observation_count": dropped_observation_count,
    }


def _build_snapshot_rows(
    *,
    path: Path,
    payload: dict[str, Any],
    payload_warnings: list[str],
    snapshot_index: int,
    filters: dict[str, str | None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_name = path.name
    snapshot_timestamp, _snapshot_timestamp_parsed, snapshot_timestamp_source = (
        _extract_snapshot_timestamp(payload)
    )
    window_label, window_label_source, window_facts = baseline._infer_window_label(
        payload,
        path,
    )
    block, block_path = baseline._extract_edge_candidate_rows_block(payload)
    source_warnings = list(payload_warnings)

    if block is None:
        source_warnings.append("edge_candidate_rows_block_missing")
        return (
            {
                "snapshot_index": snapshot_index,
                "snapshot_timestamp": snapshot_timestamp,
                "snapshot_timestamp_source": snapshot_timestamp_source,
                "source_path": str(path),
                "source_name": source_name,
                "window_label": window_label,
                "window_label_source": window_label_source,
                "edge_candidate_rows_path": None,
                "extraction_mode": "missing_edge_candidate_rows",
                "source_facts": {
                    **window_facts,
                    "diagnostic_input_count": 0,
                    "selected_input_count": 0,
                    "appearance_row_count_before_filter": 0,
                    "appearance_row_count_after_filter": 0,
                    "filtered_out_count": 0,
                    "same_snapshot_duplicate_resolution_rule": (
                        SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE
                    ),
                },
                "classification_distribution": {},
                "warnings": source_warnings,
            },
            [],
        )

    raw_diagnostic_rows = [
        item
        for item in baseline._safe_list(block.get("diagnostic_rows"))
        if isinstance(item, dict)
    ]
    raw_selected_rows = [
        item for item in baseline._safe_list(block.get("rows")) if isinstance(item, dict)
    ]

    fallback_diagnostic_rows: list[dict[str, Any]] = []
    fallback_selected_rows: list[dict[str, Any]] = []
    diagnostic_source = "diagnostic_rows"
    selected_source = "edge_candidate_rows.rows"

    if not raw_diagnostic_rows or not raw_selected_rows:
        (
            synthesized_diagnostic_rows,
            synthesized_selected_rows,
        ) = _extract_rows_from_identity_horizon_evaluations(payload)
        if not raw_diagnostic_rows and synthesized_diagnostic_rows:
            fallback_diagnostic_rows = synthesized_diagnostic_rows
            diagnostic_source = "identity_horizon_evaluations"
            source_warnings.append(
                "diagnostic_rows_synthesized_from_identity_horizon_evaluations"
            )
        if not raw_selected_rows and synthesized_selected_rows:
            fallback_selected_rows = synthesized_selected_rows
            selected_source = "identity_horizon_evaluations"
            source_warnings.append(
                "eligible_rows_synthesized_from_identity_horizon_evaluations"
            )

    diagnostic_rows = raw_diagnostic_rows or fallback_diagnostic_rows
    selected_rows = raw_selected_rows or fallback_selected_rows

    normalized_rows = [
        _normalize_diagnostic_appearance_row(
            row,
            snapshot_index=snapshot_index,
            snapshot_timestamp=snapshot_timestamp,
            snapshot_path=path,
            source_name=source_name,
            window_label=window_label,
            window_label_source=window_label_source,
            appearance_source=diagnostic_source,
        )
        for row in diagnostic_rows
    ]
    normalized_rows.extend(
        _normalize_selected_appearance_row(
            row,
            snapshot_index=snapshot_index,
            snapshot_timestamp=snapshot_timestamp,
            snapshot_path=path,
            source_name=source_name,
            window_label=window_label,
            window_label_source=window_label_source,
            appearance_source=selected_source,
        )
        for row in selected_rows
    )

    deduped_rows, dedupe_details = _deduplicate_same_snapshot_rows(normalized_rows)
    filtered_rows = [row for row in deduped_rows if _matches_filters(row, filters)]
    filtered_out_count = len(deduped_rows) - len(filtered_rows)

    if dedupe_details["same_snapshot_duplicate_identity_count"] > 0:
        source_warnings.append(
            "same_snapshot_duplicate_identities_resolved_with_precedence="
            f"{SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE}"
        )
    if not diagnostic_rows and not selected_rows:
        source_warnings.append(
            "snapshot_contains_no_diagnostic_rows_or_selected_rows_after_fallback"
        )
    if filtered_out_count > 0:
        source_warnings.append(f"appearance_rows_filtered_out={filtered_out_count}")

    class_counter = Counter(str(row.get("classification") or "unknown") for row in filtered_rows)
    extraction_modes = []
    if diagnostic_rows:
        extraction_modes.append(diagnostic_source)
    if selected_rows:
        extraction_modes.append(selected_source)
    if not extraction_modes:
        extraction_modes.append("empty_snapshot")

    source_summary = {
        "snapshot_index": snapshot_index,
        "snapshot_timestamp": snapshot_timestamp,
        "snapshot_timestamp_source": snapshot_timestamp_source,
        "source_path": str(path),
        "source_name": source_name,
        "window_label": window_label,
        "window_label_source": window_label_source,
        "edge_candidate_rows_path": block_path,
        "extraction_mode": "+".join(extraction_modes),
        "source_facts": {
            **window_facts,
            "diagnostic_input_count": len(diagnostic_rows),
            "selected_input_count": len(selected_rows),
            "appearance_row_count_before_filter": len(deduped_rows),
            "appearance_row_count_after_filter": len(filtered_rows),
            "filtered_out_count": filtered_out_count,
            **dedupe_details,
            "same_snapshot_duplicate_resolution_rule": (
                SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE
            ),
        },
        "classification_distribution": baseline._counter_dict(class_counter),
        "warnings": source_warnings,
    }
    return source_summary, filtered_rows


def _consecutive_appearance_max(snapshot_indices: list[int]) -> int:
    if not snapshot_indices:
        return 0
    best = 1
    current = 1
    for previous, current_index in zip(snapshot_indices, snapshot_indices[1:]):
        if current_index == previous + 1:
            current += 1
        else:
            current = 1
        best = max(best, current)
    return best


def _gaps_between_appearances(snapshot_indices: list[int]) -> list[int]:
    gaps: list[int] = []
    for previous, current in zip(snapshot_indices, snapshot_indices[1:]):
        gap = current - previous - 1
        if gap > 0:
            gaps.append(gap)
    return gaps


def _sample_count_transition_metrics(
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    non_decreasing = 0
    drops = 0
    for previous, current in zip(rows, rows[1:]):
        previous_count = previous.get("sample_count")
        current_count = current.get("sample_count")
        if previous_count is None or current_count is None:
            continue
        if current_count >= previous_count:
            non_decreasing += 1
        else:
            drops += 1
    return non_decreasing, drops


def _final_state(rows: list[dict[str, Any]], total_snapshots: int) -> str:
    if not rows:
        return "unknown"

    last_row = rows[-1]
    last_snapshot_index = int(last_row.get("snapshot_index") or 0)
    if bool(last_row.get("eligible")):
        return "eligible"
    if total_snapshots > 0 and last_snapshot_index < total_snapshots - 1:
        return "disappeared"

    classification = str(last_row.get("classification") or "")
    if classification in baseline.NEAR_MISS_CLASSIFICATIONS:
        return "present_sub_floor"
    if classification == "non_positive_or_flat_edge":
        return "present_non_positive"
    if classification == "non_sample_primary_failure":
        return "present_blocked"
    return "present_unknown"


def _disappearance_type(rows: list[dict[str, Any]], total_snapshots: int) -> str | None:
    final_state = _final_state(rows, total_snapshots)
    if final_state != "disappeared" or total_snapshots <= 1 or not rows:
        return None

    last_snapshot_index = int(rows[-1].get("snapshot_index") or 0)
    progress_ratio = last_snapshot_index / max(1, total_snapshots - 1)

    if progress_ratio <= 0.33:
        return "early_disappearance"
    if progress_ratio <= 0.66:
        return "mid_lifecycle_disappearance"
    return "late_disappearance"


def _build_state_transition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    transitions = Counter(
        f"{previous.get('classification', 'unknown')}->{current.get('classification', 'unknown')}"
        for previous, current in zip(rows, rows[1:])
    )
    return baseline._counter_dict(transitions)


def _recurrence_strength(total_appearances: int) -> str:
    if total_appearances <= 1:
        return "single"
    if total_appearances == 2:
        return "low"
    if total_appearances <= 4:
        return "moderate"
    return "strong"


def _extract_metric_series(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = baseline._safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _metric_degrading(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return values[-1] < values[0]


def _metric_non_negative(values: list[float]) -> bool:
    return all(value >= 0 for value in values)


def _quality_is_too_weak_for_early_stage(rows: list[dict[str, Any]]) -> bool:
    pnl_values = _extract_metric_series(rows, "pnl_pct")
    win_rate_values = _extract_metric_series(rows, "win_rate_pct")
    robustness_values = _extract_metric_series(rows, "robustness_signal_pct")

    if pnl_values and not _metric_non_negative(pnl_values):
        return True
    if win_rate_values and min(win_rate_values) < MIN_POSITIVE_RATE_FOR_EARLY_STAGE:
        return True
    if robustness_values and min(robustness_values) < MIN_ROBUSTNESS_SIGNAL_FOR_EARLY_STAGE:
        return True
    return False


def _quality_is_degrading_for_early_stage(rows: list[dict[str, Any]]) -> bool:
    return any(
        (
            _metric_degrading(_extract_metric_series(rows, "pnl_pct")),
            _metric_degrading(_extract_metric_series(rows, "win_rate_pct")),
            _metric_degrading(_extract_metric_series(rows, "robustness_signal_pct")),
        )
    )


def _build_diagnostic_notes(
    *,
    rows: list[dict[str, Any]],
    sample_count_growth_abs: int | None,
    sample_count_drop_steps: int,
    final_state: str,
    disappearance_type: str | None,
    recurrence_strength: str,
    sample_growth_pct_reliable: bool,
) -> list[str]:
    notes: list[str] = []
    if len(rows) == 1:
        notes.append("single appearance with no historical follow-up")
    else:
        notes.append(f"recurrence strength assessed as {recurrence_strength}")

    if sample_count_growth_abs is not None:
        if sample_count_growth_abs > 0:
            notes.append(f"sample_count increased by {sample_count_growth_abs}")
        elif sample_count_growth_abs < 0:
            notes.append(f"sample_count decreased by {abs(sample_count_growth_abs)}")
        else:
            notes.append("sample_count did not grow across appearances")
    if sample_count_drop_steps > 0:
        notes.append(f"sample_count dropped on {sample_count_drop_steps} step(s)")
    if any(bool(row.get("eligible")) for row in rows):
        notes.append("identity reached eligibility in at least one snapshot")
    if final_state == "disappeared":
        notes.append("identity was absent from the final snapshot in scope")
    if disappearance_type is not None:
        notes.append(f"disappearance classified as {disappearance_type}")
    if not sample_growth_pct_reliable:
        notes.append(
            "sample_count_growth_pct is not reliable because first sample_count is below 10"
        )

    rejection_reason_count = len(
        {
            str(row.get("primary_rejection_reason"))
            for row in rows
            if row.get("primary_rejection_reason")
        }
    )
    if rejection_reason_count > 1:
        notes.append("rejection reasons changed across snapshots")
    return notes


def _persistence_label(
    *,
    rows: list[dict[str, Any]],
    sample_count_growth_abs: int | None,
    sample_count_non_decreasing_steps: int,
    sample_count_drop_steps: int,
    consecutive_appearance_max: int,
    final_state: str,
) -> str:
    if len(rows) <= 1:
        return "noise_like_singleton"

    classifications = {str(row.get("classification") or "") for row in rows}
    has_hard_quality_failure = bool(
        classifications.intersection(
            {"non_positive_or_flat_edge", "non_sample_primary_failure"}
        )
    )
    rejection_reason_count = len(
        {
            str(row.get("primary_rejection_reason"))
            for row in rows
            if row.get("primary_rejection_reason")
        }
    )
    quality_too_weak = _quality_is_too_weak_for_early_stage(rows)
    quality_degrading = _quality_is_degrading_for_early_stage(rows)
    ever_eligible = any(bool(row.get("eligible")) for row in rows)

    if (
        len(rows) >= 3
        and sample_count_growth_abs is not None
        and sample_count_growth_abs >= 10
        and sample_count_drop_steps == 0
        and sample_count_non_decreasing_steps >= len(rows) - 1
        and consecutive_appearance_max >= 2
        and rejection_reason_count <= 1
        and not has_hard_quality_failure
        and not quality_too_weak
        and not quality_degrading
        and (ever_eligible or final_state == "present_sub_floor")
    ):
        return "early_stage_growth_candidate"

    return "unstable_recurrent_candidate"


def _timeline_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_index": row.get("snapshot_index"),
        "snapshot_timestamp": row.get("snapshot_timestamp"),
        "snapshot_path": row.get("snapshot_path"),
        "appearance_status": row.get("appearance_status"),
        "classification": row.get("classification"),
        "eligible": row.get("eligible"),
        "sample_count": row.get("sample_count"),
        "sample_count_band": row.get("sample_count_band"),
        "primary_rejection_reason": row.get("primary_rejection_reason"),
        "horizon_compatible": row.get("horizon_compatible"),
        "edge_score": row.get("edge_score"),
        "win_rate_pct": row.get("win_rate_pct"),
        "pnl_pct": row.get("pnl_pct"),
        "robustness_signal_pct": row.get("robustness_signal_pct"),
    }


def _identity_summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    label_order = {
        "early_stage_growth_candidate": 0,
        "unstable_recurrent_candidate": 1,
        "noise_like_singleton": 2,
    }
    growth_abs = row.get("sample_count_growth_abs")
    sortable_growth = growth_abs if isinstance(growth_abs, int) else -999999
    return (
        label_order.get(str(row.get("persistence_label") or ""), 999),
        -int(row.get("total_appearances") or 0),
        -sortable_growth,
        str(row.get("identity_key") or ""),
    )


def _build_identity_summary(
    rows: list[dict[str, Any]],
    *,
    total_snapshots: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity_key = baseline._safe_text(row.get("identity_key"))
        if identity_key is None:
            continue
        grouped[identity_key].append(row)

    summaries: list[dict[str, Any]] = []
    for identity_key, identity_rows in grouped.items():
        ordered_rows = sorted(identity_rows, key=_appearance_sort_key)
        first_row = ordered_rows[0]
        last_row = ordered_rows[-1]
        snapshot_indices = [int(row.get("snapshot_index") or 0) for row in ordered_rows]

        sample_count_first = first_row.get("sample_count")
        sample_count_last = last_row.get("sample_count")
        sample_counts = [
            int(row.get("sample_count"))
            for row in ordered_rows
            if isinstance(row.get("sample_count"), int)
        ]
        sample_count_growth_abs = None
        if sample_count_first is not None and sample_count_last is not None:
            sample_count_growth_abs = sample_count_last - sample_count_first

        sample_growth_pct_reliable = (
            sample_count_first is not None
            and sample_count_first >= MIN_SAMPLE_COUNT_FOR_GROWTH_PCT_RELIABILITY
        )
        sample_count_growth_pct = None
        if (
            sample_count_growth_abs is not None
            and sample_count_first is not None
            and sample_count_first > 0
            and sample_growth_pct_reliable
        ):
            sample_count_growth_pct = round(
                (sample_count_growth_abs / sample_count_first) * 100.0,
                4,
            )

        sample_count_non_decreasing_steps, sample_count_drop_steps = (
            _sample_count_transition_metrics(ordered_rows)
        )
        consecutive_appearance_max = _consecutive_appearance_max(snapshot_indices)
        final_state = _final_state(ordered_rows, total_snapshots)
        disappearance_type = _disappearance_type(ordered_rows, total_snapshots)

        rejection_reason_counts = Counter(
            str(row.get("primary_rejection_reason"))
            for row in ordered_rows
            if row.get("primary_rejection_reason")
        )
        ever_eligible = any(bool(row.get("eligible")) for row in ordered_rows)
        recurrence_ratio = _ratio_or_none(len(ordered_rows), total_snapshots)
        recurrence_strength = _recurrence_strength(len(ordered_rows))
        diagnostic_notes = _build_diagnostic_notes(
            rows=ordered_rows,
            sample_count_growth_abs=sample_count_growth_abs,
            sample_count_drop_steps=sample_count_drop_steps,
            final_state=final_state,
            disappearance_type=disappearance_type,
            recurrence_strength=recurrence_strength,
            sample_growth_pct_reliable=sample_growth_pct_reliable,
        )
        persistence_label = _persistence_label(
            rows=ordered_rows,
            sample_count_growth_abs=sample_count_growth_abs,
            sample_count_non_decreasing_steps=sample_count_non_decreasing_steps,
            sample_count_drop_steps=sample_count_drop_steps,
            consecutive_appearance_max=consecutive_appearance_max,
            final_state=final_state,
        )

        summaries.append(
            {
                "identity_key": identity_key,
                "symbol": first_row.get("symbol"),
                "strategy": first_row.get("strategy"),
                "horizon": first_row.get("horizon"),
                "first_seen_at": first_row.get("snapshot_timestamp"),
                "last_seen_at": last_row.get("snapshot_timestamp"),
                "first_seen_snapshot_index": first_row.get("snapshot_index"),
                "last_seen_snapshot_index": last_row.get("snapshot_index"),
                "total_appearances": len(ordered_rows),
                "followup_count": max(0, len(ordered_rows) - 1),
                "consecutive_appearance_max": consecutive_appearance_max,
                "gaps_between_appearances": _gaps_between_appearances(snapshot_indices),
                "recurrence_ratio": recurrence_ratio,
                "recurrence_strength": recurrence_strength,
                "sample_count_first": sample_count_first,
                "sample_count_last": sample_count_last,
                "sample_count_max": max(sample_counts) if sample_counts else None,
                "sample_count_min": min(sample_counts) if sample_counts else None,
                "sample_count_growth_abs": sample_count_growth_abs,
                "sample_count_growth_pct": sample_count_growth_pct,
                "sample_count_growth_pct_reliable": sample_growth_pct_reliable,
                "sample_count_non_decreasing_steps": sample_count_non_decreasing_steps,
                "sample_count_drop_steps": sample_count_drop_steps,
                "sample_count_first_band": baseline._sample_count_band(sample_count_first),
                "ever_eligible": ever_eligible,
                "final_state": final_state,
                "disappearance_type": disappearance_type,
                "rejection_reason_counts": baseline._counter_dict(rejection_reason_counts),
                "state_transition_counts": _build_state_transition_counts(ordered_rows),
                "persistence_label": persistence_label,
                "diagnostic_notes": diagnostic_notes,
                "timeline": [_timeline_row(row) for row in ordered_rows],
            }
        )

    summaries.sort(key=_identity_summary_sort_key)
    return summaries


def _build_sample_band_summary(
    identity_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_summary:
        grouped[str(row.get("sample_count_first_band") or "unknown")].append(row)

    labels = [band for band, _, _ in baseline.SAMPLE_COUNT_BANDS]
    observed_unknown = sorted(label for label in grouped.keys() if label not in labels)
    summaries: list[dict[str, Any]] = []

    for label in labels + observed_unknown:
        rows = grouped.get(label, [])
        identity_count = len(rows)
        recurrence_values = [
            float(row["recurrence_ratio"])
            for row in rows
            if row.get("recurrence_ratio") is not None
        ]
        growth_values = [
            int(row["sample_count_growth_abs"])
            for row in rows
            if row.get("sample_count_growth_abs") is not None
        ]
        persistence_label_counts = Counter(
            str(row.get("persistence_label") or "unknown") for row in rows
        )
        summaries.append(
            {
                "sample_count_band": label,
                "identity_count": identity_count,
                "followup_ratio": _ratio_or_none(
                    sum(int(row.get("followup_count") or 0) > 0 for row in rows),
                    identity_count,
                ),
                "recurrence_ratio": _mean_or_none(recurrence_values),
                "growth_ratio": _ratio_or_none(
                    sum((row.get("sample_count_growth_abs") or 0) > 0 for row in rows),
                    identity_count,
                ),
                "eventual_eligibility_ratio": _ratio_or_none(
                    sum(bool(row.get("ever_eligible")) for row in rows),
                    identity_count,
                ),
                "median_appearances": _median_number(
                    [int(row.get("total_appearances") or 0) for row in rows]
                ),
                "median_sample_growth_abs": _median_number(growth_values),
                "persistence_label_counts": baseline._counter_dict(
                    persistence_label_counts
                ),
            }
        )

    summaries.sort(
        key=lambda item: baseline._sample_band_sort_key(str(item["sample_count_band"]))
    )
    return summaries


def _build_grouped_summary(
    identity_summary: list[dict[str, Any]],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_summary:
        key = baseline._safe_text(row.get(field_name)) or f"unknown_{field_name}"
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        recurrence_values = [
            float(row["recurrence_ratio"])
            for row in rows
            if row.get("recurrence_ratio") is not None
        ]
        growth_values = [
            int(row["sample_count_growth_abs"])
            for row in rows
            if row.get("sample_count_growth_abs") is not None
        ]
        persistence_label_counts = Counter(
            str(row.get("persistence_label") or "unknown") for row in rows
        )
        summary_rows.append(
            {
                field_name: key,
                "identity_count": len(rows),
                "appearance_count": sum(
                    int(row.get("total_appearances") or 0) for row in rows
                ),
                "followup_ratio": _ratio_or_none(
                    sum(int(row.get("followup_count") or 0) > 0 for row in rows),
                    len(rows),
                ),
                "recurrence_ratio": _mean_or_none(recurrence_values),
                "growth_ratio": _ratio_or_none(
                    sum((row.get("sample_count_growth_abs") or 0) > 0 for row in rows),
                    len(rows),
                ),
                "eventual_eligibility_ratio": _ratio_or_none(
                    sum(bool(row.get("ever_eligible")) for row in rows),
                    len(rows),
                ),
                "median_sample_growth_abs": _median_number(growth_values),
                "persistence_label_counts": baseline._counter_dict(
                    persistence_label_counts
                ),
            }
        )

    summary_rows.sort(
        key=lambda item: (
            -int(item.get("identity_count") or 0),
            str(item.get(field_name) or ""),
        )
    )
    return summary_rows


def build_historical_persistence_diagnosis_report(
    summary_paths: list[Path],
    *,
    symbol: str | None = None,
    strategy: str | None = None,
    horizon: str | None = None,
    discovery_metadata: dict[str, Any] | None = None,
    discovery_warnings: list[str] | None = None,
) -> dict[str, Any]:
    filters = _normalize_filters(symbol=symbol, strategy=strategy, horizon=horizon)
    discovery_metadata = dict(discovery_metadata or {})
    warnings = list(discovery_warnings or [])
    warnings.extend(
        [
            "Appearance classifications are computed from each snapshot row only; "
            "future snapshots never overwrite earlier classifications.",
            "This diagnosis is offline-only and does not change analyzer, mapper, "
            "engine, or live-trading behavior.",
            "Sample-band summary groups identities by first observed sample_count band.",
            "early_stage_growth_candidate classification is conservative and requires "
            "both sample growth and non-weak, non-degrading quality signals.",
        ]
    )

    source_summaries: list[dict[str, Any]] = []
    appearance_rows: list[dict[str, Any]] = []

    for snapshot_index, raw_path in enumerate(summary_paths):
        path = Path(raw_path)
        payload, payload_warnings = baseline.load_summary_json(path)
        source_summary, source_rows = _build_snapshot_rows(
            path=path,
            payload=payload,
            payload_warnings=payload_warnings,
            snapshot_index=snapshot_index,
            filters=filters,
        )
        source_summaries.append(source_summary)
        appearance_rows.extend(source_rows)
        warnings.extend(
            f"{path.name}: {warning}" for warning in source_summary.get("warnings", [])
        )

    appearance_rows.sort(key=_appearance_sort_key)
    identity_summary = _build_identity_summary(
        appearance_rows,
        total_snapshots=len(source_summaries),
    )

    persistence_label_counts = Counter(
        str(row.get("persistence_label") or "unknown") for row in identity_summary
    )
    growth_values = [
        int(row["sample_count_growth_abs"])
        for row in identity_summary
        if row.get("sample_count_growth_abs") is not None
    ]
    recurrence_strength_counts = Counter(
        str(row.get("recurrence_strength") or "unknown") for row in identity_summary
    )
    disappearance_type_counts = Counter(
        str(row.get("disappearance_type"))
        for row in identity_summary
        if row.get("disappearance_type") is not None
    )

    overall_summary = {
        "total_snapshots": len(source_summaries),
        "total_unique_identities": len(identity_summary),
        "total_appearance_rows": len(appearance_rows),
        "identities_with_single_appearance": sum(
            int(row.get("total_appearances") or 0) == 1 for row in identity_summary
        ),
        "identities_with_followup": sum(
            int(row.get("followup_count") or 0) > 0 for row in identity_summary
        ),
        "identities_with_sample_growth": sum(
            (row.get("sample_count_growth_abs") or 0) > 0 for row in identity_summary
        ),
        "identities_ever_eligible": sum(
            bool(row.get("ever_eligible")) for row in identity_summary
        ),
        "median_appearances_per_identity": _median_number(
            [int(row.get("total_appearances") or 0) for row in identity_summary]
        ),
        "median_sample_growth_abs": _median_number(growth_values),
        "persistence_label_counts": baseline._counter_dict(persistence_label_counts),
        "recurrence_strength_counts": baseline._counter_dict(recurrence_strength_counts),
        "disappearance_type_counts": baseline._counter_dict(disappearance_type_counts),
    }

    if not appearance_rows:
        warnings.append(
            "No appearance rows were available after loading snapshots and applying filters."
        )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": REPORT_TYPE,
            "report_title": REPORT_TITLE,
            "report_version": REPORT_VERSION,
            "classification_version": baseline.CLASSIFICATION_VERSION,
            "summary_input_count": len(summary_paths),
            "chronology_basis": discovery_metadata.get(
                "chronology_basis", "input_order_unverified"
            ),
            "chronology_verified": bool(
                discovery_metadata.get("chronology_verified", False)
            ),
            "discovery_mode": discovery_metadata.get(
                "discovery_mode", "explicit_summary_paths"
            ),
            "history_dir": discovery_metadata.get("history_dir"),
            "scanned_json_count": discovery_metadata.get("scanned_json_count"),
            "discovered_snapshot_count": discovery_metadata.get(
                "discovered_snapshot_count"
            ),
            "skipped_non_snapshot_count": discovery_metadata.get(
                "skipped_non_snapshot_count"
            ),
            "sample_count_bands": [band for band, _, _ in baseline.SAMPLE_COUNT_BANDS],
            "same_snapshot_duplicate_resolution_rule": (
                SAME_SNAPSHOT_DUPLICATE_RESOLUTION_RULE
            ),
            "sample_count_growth_pct_basis": "((last - first) / first) * 100",
            "sample_count_growth_pct_reliability_min_first_sample_count": (
                MIN_SAMPLE_COUNT_FOR_GROWTH_PCT_RELIABILITY
            ),
            "filters": filters,
        },
        "source_summaries": source_summaries,
        "overall_summary": overall_summary,
        "sample_band_summary": _build_sample_band_summary(identity_summary),
        "grouped_summary": {
            "by_symbol": _build_grouped_summary(identity_summary, field_name="symbol"),
            "by_strategy": _build_grouped_summary(
                identity_summary,
                field_name="strategy",
            ),
            "by_horizon": _build_grouped_summary(identity_summary, field_name="horizon"),
        },
        "identity_summary": identity_summary,
        "appearance_rows": appearance_rows,
        "warnings": sorted(set(warnings)),
    }


def build_historical_persistence_diagnosis_markdown(summary: dict[str, Any]) -> str:
    metadata = baseline._safe_dict(summary.get("metadata"))
    overall = baseline._safe_dict(summary.get("overall_summary"))
    sample_band_summary = baseline._safe_list(summary.get("sample_band_summary"))
    grouped_summary = baseline._safe_dict(summary.get("grouped_summary"))
    identity_summary = baseline._safe_list(summary.get("identity_summary"))
    warnings = baseline._normalize_string_list(summary.get("warnings"))

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Run",
        f"- generated_at: {metadata.get('generated_at', 'n/a')}",
        f"- report_version: {metadata.get('report_version', 'n/a')}",
        f"- classification_version: {metadata.get('classification_version', 'n/a')}",
        f"- discovery_mode: {metadata.get('discovery_mode', 'n/a')}",
        f"- chronology_basis: {metadata.get('chronology_basis', 'n/a')}",
        f"- chronology_verified: {metadata.get('chronology_verified', False)}",
        f"- summary_input_count: {metadata.get('summary_input_count', 0)}",
        f"- filters: {metadata.get('filters', {})}",
        "",
        "## Global Summary",
        f"- total_snapshots: {overall.get('total_snapshots', 0)}",
        f"- total_unique_identities: {overall.get('total_unique_identities', 0)}",
        f"- total_appearance_rows: {overall.get('total_appearance_rows', 0)}",
        f"- identities_with_single_appearance: {overall.get('identities_with_single_appearance', 0)}",
        f"- identities_with_followup: {overall.get('identities_with_followup', 0)}",
        f"- identities_with_sample_growth: {overall.get('identities_with_sample_growth', 0)}",
        f"- identities_ever_eligible: {overall.get('identities_ever_eligible', 0)}",
        f"- median_appearances_per_identity: {baseline._format_number(baseline._safe_float(overall.get('median_appearances_per_identity')))}",
        f"- median_sample_growth_abs: {baseline._format_number(baseline._safe_float(overall.get('median_sample_growth_abs')))}",
        f"- persistence_label_counts: {overall.get('persistence_label_counts', {})}",
        f"- recurrence_strength_counts: {overall.get('recurrence_strength_counts', {})}",
        f"- disappearance_type_counts: {overall.get('disappearance_type_counts', {})}",
        "",
        "## Sample Bands",
    ]

    if not sample_band_summary:
        lines.append("- none")
    else:
        for row in sample_band_summary:
            lines.append(
                f"- {row.get('sample_count_band', 'n/a')}: "
                f"identity_count={row.get('identity_count', 0)}, "
                f"followup_ratio={baseline._format_number(baseline._safe_float(row.get('followup_ratio')))}, "
                f"recurrence_ratio={baseline._format_number(baseline._safe_float(row.get('recurrence_ratio')))}, "
                f"growth_ratio={baseline._format_number(baseline._safe_float(row.get('growth_ratio')))}, "
                f"eventual_eligibility_ratio={baseline._format_number(baseline._safe_float(row.get('eventual_eligibility_ratio')))}, "
                f"persistence_label_counts={row.get('persistence_label_counts', {})}"
            )

    for section_name, rows in (
        ("Symbol Groups", baseline._safe_list(grouped_summary.get("by_symbol"))[:12]),
        ("Strategy Groups", baseline._safe_list(grouped_summary.get("by_strategy"))[:12]),
        ("Horizon Groups", baseline._safe_list(grouped_summary.get("by_horizon"))[:12]),
    ):
        lines.extend(["", f"## {section_name}"])
        if not rows:
            lines.append("- none")
            continue
        for row in rows:
            key_name = (
                "symbol"
                if section_name == "Symbol Groups"
                else "strategy"
                if section_name == "Strategy Groups"
                else "horizon"
            )
            lines.append(
                f"- {row.get(key_name, 'n/a')}: "
                f"identity_count={row.get('identity_count', 0)}, "
                f"followup_ratio={baseline._format_number(baseline._safe_float(row.get('followup_ratio')))}, "
                f"growth_ratio={baseline._format_number(baseline._safe_float(row.get('growth_ratio')))}, "
                f"eventual_eligibility_ratio={baseline._format_number(baseline._safe_float(row.get('eventual_eligibility_ratio')))}, "
                f"persistence_label_counts={row.get('persistence_label_counts', {})}"
            )

    lines.extend(["", "## Identity Summary"])
    if not identity_summary:
        lines.append("- none")
    else:
        for row in identity_summary[:25]:
            lines.append(
                f"- {row.get('identity_key', 'n/a')}: "
                f"appearances={row.get('total_appearances', 0)}, "
                f"followup_count={row.get('followup_count', 0)}, "
                f"recurrence_strength={row.get('recurrence_strength', 'n/a')}, "
                f"recurrence_ratio={baseline._format_number(baseline._safe_float(row.get('recurrence_ratio')))}, "
                f"sample_count_growth_abs={row.get('sample_count_growth_abs')}, "
                f"sample_count_growth_pct={row.get('sample_count_growth_pct')}, "
                f"sample_count_growth_pct_reliable={row.get('sample_count_growth_pct_reliable')}, "
                f"ever_eligible={row.get('ever_eligible', False)}, "
                f"final_state={row.get('final_state', 'n/a')}, "
                f"disappearance_type={row.get('disappearance_type', 'n/a')}, "
                f"persistence_label={row.get('persistence_label', 'n/a')}"
            )

    lines.extend(["", "## Warnings"])
    if not warnings:
        lines.append("- none")
    else:
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def write_historical_persistence_diagnosis_report(
    *,
    summary: dict[str, Any],
    markdown: str,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / DEFAULT_SUMMARY_OUTPUT_NAME
    identity_json_path = output_dir / DEFAULT_IDENTITY_OUTPUT_NAME
    appearance_jsonl_path = output_dir / DEFAULT_APPEARANCE_OUTPUT_NAME
    markdown_path = output_dir / DEFAULT_MD_OUTPUT_NAME

    summary_payload = {
        "metadata": summary["metadata"],
        "source_summaries": summary["source_summaries"],
        "overall_summary": summary["overall_summary"],
        "sample_band_summary": summary["sample_band_summary"],
        "grouped_summary": summary["grouped_summary"],
        "warnings": summary["warnings"],
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    identity_json_path.write_text(
        json.dumps(summary["identity_summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with appearance_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in summary["appearance_rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    markdown_path.write_text(markdown, encoding="utf-8")

    return {
        "summary_json": str(summary_json_path),
        "identity_summary_json": str(identity_json_path),
        "appearance_rows_jsonl": str(appearance_jsonl_path),
        "summary_md": str(markdown_path),
    }


def run_historical_persistence_diagnosis(
    *,
    summary_paths: list[Path] | None = None,
    history_dir: Path | None = None,
    limit_snapshots: int | None = None,
    symbol: str | None = None,
    strategy: str | None = None,
    horizon: str | None = None,
    write_latest_copy: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_paths, discovery_metadata, discovery_warnings = resolve_history_paths(
        summary_paths=summary_paths,
        history_dir=history_dir,
        limit_snapshots=limit_snapshots,
    )
    summary = build_historical_persistence_diagnosis_report(
        resolved_paths,
        symbol=symbol,
        strategy=strategy,
        horizon=horizon,
        discovery_metadata=discovery_metadata,
        discovery_warnings=discovery_warnings,
    )
    markdown = build_historical_persistence_diagnosis_markdown(summary)
    result: dict[str, Any] = {
        "summary": summary,
        "markdown": markdown,
        "summary_paths_used": [str(path) for path in resolved_paths],
    }

    if write_latest_copy:
        outputs = write_historical_persistence_diagnosis_report(
            summary=summary,
            markdown=markdown,
            output_dir=resolve_output_dir(output_dir),
        )
        result.update(outputs)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose identity recurrence, sample growth, disappearance, and "
            "eligibility transitions across historical analyzer snapshots."
        )
    )
    parser.add_argument(
        "summary_paths",
        type=Path,
        nargs="*",
        help="Optional explicit snapshot JSON paths ordered oldest-to-newest.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing historical snapshot JSON files. Files "
            "are discovered recursively and ordered chronologically when possible."
        ),
    )
    parser.add_argument(
        "--limit-snapshots",
        type=int,
        default=None,
        help="Keep only the most recent N snapshots after chronology resolution.",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol filter.")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional strategy filter.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        help="Optional horizon filter.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON, JSONL, and Markdown outputs into the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used with --write-latest-copy.",
    )
    parser.add_argument(
        "--stdout-format",
        choices=("json", "markdown"),
        default="json",
        help="Used only for stdout output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_historical_persistence_diagnosis(
        summary_paths=args.summary_paths,
        history_dir=args.history_dir,
        limit_snapshots=args.limit_snapshots,
        symbol=args.symbol,
        strategy=args.strategy,
        horizon=args.horizon,
        write_latest_copy=args.write_latest_copy,
        output_dir=args.output_dir,
    )

    if args.stdout_format == "markdown":
        print(result["markdown"], end="")
        return

    stdout_payload: dict[str, Any] = {
        "summary_paths_used": result["summary_paths_used"],
        "overall_summary": result["summary"]["overall_summary"],
    }
    for key in (
        "summary_json",
        "identity_summary_json",
        "appearance_rows_jsonl",
        "summary_md",
    ):
        if key in result:
            stdout_payload[key] = result[key]
    print(json.dumps(stdout_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()