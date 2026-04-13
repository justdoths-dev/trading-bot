from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.notifications.research_notifier import ResearchNotifier
from src.research.research_metrics import HORIZONS, calculate_research_metrics
from src.research.schema_validator import validate_record
from src.research.strategy_lab.comparison_report import (
    compare_by_ai_execution_state,
    compare_by_alignment_state,
    compare_by_strategy,
    compare_by_symbol,
)
from src.research.strategy_lab.dataset_builder import (
    DEFAULT_LATEST_MAX_ROWS,
    DEFAULT_LATEST_WINDOW_HOURS,
    build_dataset,
    load_jsonl_records_with_metadata,
)
from src.research.strategy_lab.edge_detector import (
    detect_ai_execution_state_edges,
    detect_alignment_state_edges,
    detect_strategy_edges,
    detect_symbol_edges,
)
from src.research.strategy_lab.performance_report import generate_performance_report
from src.research.strategy_lab.ranking_report import (
    rank_by_ai_execution_state,
    rank_by_alignment_state,
    rank_by_strategy,
    rank_by_symbol,
)
from src.research.strategy_lab.segment_report import build_segment_reports
from src.services.cron_health import CronHealthReporter

LOGGER = logging.getLogger(__name__)

MIN_EDGE_CANDIDATE_SAMPLE_COUNT = 30
EDGE_EARLY_MODERATE_SAMPLE_COUNT = 40
EDGE_MODERATE_SAMPLE_COUNT = 50
EDGE_STRONG_SAMPLE_COUNT = 80

EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT = 0.18
EDGE_MODERATE_MEDIAN_RETURN_PCT = 0.30
EDGE_STRONG_MEDIAN_RETURN_PCT = 0.50

EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT = 50.0
EDGE_MODERATE_POSITIVE_RATE_PCT = 55.0
EDGE_STRONG_POSITIVE_RATE_PCT = 58.0

EDGE_EARLY_MODERATE_ROBUSTNESS_PCT = 46.0
EDGE_MODERATE_ROBUSTNESS_PCT = 52.0
EDGE_STRONG_ROBUSTNESS_PCT = 55.0

POSITIVE_RATE_MINIMUM_FLOOR_PCT = 40.0
THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT = 47.0
THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT = 45.0

STRENGTH_COMPONENT_WEIGHTS = {
    "sample_count": 0.30,
    "median_future_return_pct": 0.30,
    "positive_rate_pct": 0.25,
    "robustness_value": 0.15,
}

STRENGTH_SCORING_MODEL = "banded_weighted_v5_2"

STRENGTH_RAW_SCORE_BANDS = {
    "strong": 1.00,
    "moderate": 0.82,
    "emerging": 0.66,
    "thin": 0.50,
    "below_floor": 0.35,
}

MODERATE_MIN_AGGREGATE_SCORE = 62.0
MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE = 66.0
MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE = 54.0
MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE = 59.5
STRONG_MIN_AGGREGATE_SCORE = 85.0

CRITICAL_MAJOR_DEFICITS = {
    "sample_count_below_emerging_moderate",
}
SUPPORTING_MAJOR_DEFICITS = {
    "median_return_below_emerging_moderate",
    "positive_rate_below_emerging_moderate",
    "robustness_below_emerging_moderate",
}

STRATEGY_HORIZON_COMPATIBILITY = {
    "scalping": {"1m", "5m"},
    "intraday": {"5m", "15m", "1h"},
    "swing": {"1h", "4h", "1d"},
}


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _normalize_horizon_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None

    normalized: list[str] = []
    for item in value:
        if item in HORIZONS and item not in normalized:
            normalized.append(item)

    return normalized or None


def _normalize_horizon_strength_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for horizon in HORIZONS:
        strength = value.get(horizon)
        if isinstance(strength, str) and strength.strip():
            normalized[horizon] = strength.strip()
    return normalized


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_materialize_windowed_input(
    input_path: Path,
    *,
    latest_window_hours: int | None,
    latest_max_rows: int | None,
) -> bool:
    """
    Materialize an effective analyzer input whenever the source is the rotation-aware
    live dataset or the caller supplied explicit latest-window overrides.

    This makes downstream modules honor the analyzer CLI window settings even when
    they only accept dataset_path and internally call build_dataset() with defaults.
    """
    if input_path.name == "trade_analysis.jsonl":
        return True

    return (
        latest_window_hours != DEFAULT_LATEST_WINDOW_HOURS
        or latest_max_rows != DEFAULT_LATEST_MAX_ROWS
    )


def _materialize_effective_input(
    *,
    input_path: Path,
    output_dir: Path,
    latest_window_hours: int | None,
    latest_max_rows: int | None,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """
    Build an effective analyzer input file whose contents already reflect the desired
    latest-window policy. Downstream modules can then consume this file via dataset_path
    without silently falling back to dataset_builder defaults.
    """
    raw_records, source_metadata = load_jsonl_records_with_metadata(
        input_path,
        max_age_hours=latest_window_hours,
        max_rows=latest_max_rows,
    )

    if not _should_materialize_windowed_input(
        input_path,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    ):
        return input_path, raw_records, source_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_input_path = output_dir / "_effective_analyzer_input.jsonl"

    with effective_input_path.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    effective_metadata = dict(source_metadata)
    effective_metadata["effective_input_path"] = str(effective_input_path)
    effective_metadata["materialized_effective_input"] = True
    effective_metadata["requested_latest_window_hours"] = latest_window_hours
    effective_metadata["requested_latest_max_rows"] = latest_max_rows

    return effective_input_path, raw_records, effective_metadata


def load_jsonl_records(
    input_path: Path,
    *,
    latest_window_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    latest_max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
    preloaded_raw_records: list[dict[str, Any]] | None = None,
    source_metadata_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSONL records, keeping only schema-valid objects for analysis."""
    if preloaded_raw_records is None or source_metadata_override is None:
        raw_records, source_metadata = load_jsonl_records_with_metadata(
            input_path,
            max_age_hours=latest_window_hours,
            max_rows=latest_max_rows,
        )
    else:
        raw_records = preloaded_raw_records
        source_metadata = source_metadata_override

    records: list[dict[str, Any]] = []
    validation_summary = {
        "input_path": source_metadata.get("input_path", str(input_path)),
        "effective_input_path": source_metadata.get("effective_input_path", str(input_path)),
        "materialized_effective_input": source_metadata.get("materialized_effective_input", False),
        "rotation_aware": source_metadata.get("rotation_aware", False),
        "source_files": source_metadata.get("source_files", []),
        "source_file_count": source_metadata.get("source_file_count", 0),
        "source_row_counts": source_metadata.get("source_row_counts", {}),
        "max_age_hours": source_metadata.get("max_age_hours"),
        "max_rows": source_metadata.get("max_rows"),
        "requested_latest_window_hours": source_metadata.get(
            "requested_latest_window_hours",
            latest_window_hours,
        ),
        "requested_latest_max_rows": source_metadata.get(
            "requested_latest_max_rows",
            latest_max_rows,
        ),
        "raw_record_count": source_metadata.get("raw_record_count", 0),
        "windowed_record_count": source_metadata.get("windowed_record_count", 0),
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "error_count": 0,
        "warning_count": 0,
        "invalid_examples": [],
    }

    for index, parsed in enumerate(raw_records, start=1):
        validation_summary["total_records"] += 1

        result = validate_record(parsed, line_number=index)
        validation_summary["error_count"] += len(result["errors"])
        validation_summary["warning_count"] += len(result["warnings"])

        if result["is_valid"]:
            validation_summary["valid_records"] += 1
            records.append(parsed)
            continue

        validation_summary["invalid_records"] += 1

        if len(validation_summary["invalid_examples"]) < 5:
            validation_summary["invalid_examples"].append(
                {
                    "line_number": index,
                    "errors": result["errors"],
                    "warnings": result["warnings"],
                }
            )

        LOGGER.warning(
            "Skipping invalid research record at synthetic line %s: %s",
            index,
            "; ".join(result["errors"]),
        )

    return records, validation_summary


def write_summary_files(
    metrics: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write summary.json and summary.md into output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    summary_payload = dict(metrics)
    summary_payload["generated_at"] = _resolve_summary_generated_at(metrics)

    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_build_markdown(summary_payload), encoding="utf-8")

    return summary_json_path, summary_md_path


def run_research_analyzer(
    input_path: Path,
    output_dir: Path,
    *,
    latest_window_hours: int | None = DEFAULT_LATEST_WINDOW_HOURS,
    latest_max_rows: int | None = DEFAULT_LATEST_MAX_ROWS,
) -> dict[str, Any]:
    """Run full analyzer flow: load valid records, calculate metrics, and write reports."""
    effective_input_path, preloaded_raw_records, source_metadata = _materialize_effective_input(
        input_path=input_path,
        output_dir=output_dir,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
    )

    records, validation_summary = load_jsonl_records(
        effective_input_path,
        latest_window_hours=latest_window_hours,
        latest_max_rows=latest_max_rows,
        preloaded_raw_records=preloaded_raw_records,
        source_metadata_override=source_metadata,
    )

    base_metrics = calculate_research_metrics(records)

    if records and int(validation_summary.get("invalid_records", 0)) == 0:
        strategy_lab_metrics = _build_strategy_lab_metrics(effective_input_path)
        edge_candidates_preview = _build_edge_candidates_preview(
            strategy_lab=strategy_lab_metrics,
        )
        edge_stability_preview = _build_edge_stability_preview(
            edge_candidates_preview=edge_candidates_preview,
        )
        edge_candidate_rows = _build_edge_candidate_rows(
            effective_input_path,
            edge_candidates_preview=edge_candidates_preview,
            edge_stability_preview=edge_stability_preview,
        )
    else:
        strategy_lab_metrics = _empty_strategy_lab_metrics()
        edge_candidates_preview = _build_edge_candidates_preview(
            strategy_lab=strategy_lab_metrics,
        )
        edge_stability_preview = _build_edge_stability_preview(
            edge_candidates_preview=edge_candidates_preview,
        )
        edge_candidate_rows = _empty_edge_candidate_rows()

    final_metrics = dict(base_metrics)
    final_metrics["schema_validation"] = validation_summary
    final_metrics["strategy_lab"] = strategy_lab_metrics
    final_metrics["edge_candidate_rows"] = edge_candidate_rows
    final_metrics["top_highlights"] = _build_top_highlights(
        schema_validation=validation_summary,
        strategy_lab=strategy_lab_metrics,
    )
    final_metrics["edge_candidates_preview"] = edge_candidates_preview
    final_metrics["edge_stability_preview"] = edge_stability_preview

    write_summary_files(final_metrics, output_dir)
    return final_metrics


def _empty_strategy_lab_metrics() -> dict[str, Any]:
    """Return a safe empty strategy-lab payload for empty or mixed-invalid datasets."""
    return {
        "dataset_rows": 0,
        "performance": {},
        "comparison": {},
        "ranking": {},
        "edge": {},
        "segment": {},
    }


def _empty_edge_candidate_rows() -> dict[str, Any]:
    return {
        "row_count": 0,
        "rows": [],
        "diagnostic_row_count": 0,
        "diagnostic_rows": [],
        "empty_reason_summary": {
            "has_eligible_rows": False,
            "diagnostic_row_count": 0,
            "diagnostic_rejection_reason_counts": {},
            "diagnostic_category_counts": {},
            "dominant_rejection_reason": None,
            "dominant_diagnostic_category": None,
            "identity_count": 0,
            "identities_with_eligible_rows": 0,
            "identities_without_eligible_rows": 0,
            "identities_blocked_only_by_incompatibility": [],
            "strategies_without_analyzer_compatible_horizons": [],
            "empty_state_category": "no_joined_candidates_evaluated",
            "has_only_incompatibility_rejections": False,
            "has_only_weak_or_insufficient_candidates": False,
            "note": "No joined candidates were evaluated.",
        },
        "dropped_row_count": 0,
        "dropped_rows": [],
        "identity_horizon_evaluations": [],
    }


def _build_strategy_lab_metrics(input_path: Path) -> dict[str, Any]:
    """Build full Strategy Research Lab metrics bundle."""
    dataset = build_dataset(path=input_path)

    performance: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    ranking: dict[str, Any] = {}
    edge: dict[str, Any] = {}

    for horizon in HORIZONS:
        performance[horizon] = generate_performance_report(
            horizon=horizon,
            dataset_path=input_path,
        )

        comparison[horizon] = {
            "by_symbol": compare_by_symbol(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_strategy": compare_by_strategy(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_alignment_state": compare_by_alignment_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_ai_execution_state": compare_by_ai_execution_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
        }

        ranking[horizon] = {
            "by_symbol": rank_by_symbol(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_strategy": rank_by_strategy(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_alignment_state": rank_by_alignment_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
            "by_ai_execution_state": rank_by_ai_execution_state(
                horizon=horizon,
                dataset_path=input_path,
            ),
        }

        symbol_rankings = _extract_ranking_items(ranking[horizon]["by_symbol"])
        strategy_rankings = _extract_ranking_items(ranking[horizon]["by_strategy"])
        alignment_rankings = _extract_ranking_items(
            ranking[horizon]["by_alignment_state"]
        )
        ai_execution_rankings = _extract_ranking_items(
            ranking[horizon]["by_ai_execution_state"]
        )

        edge[horizon] = {
            "by_symbol": detect_symbol_edges(
                symbol_rankings,
                horizon=horizon,
            ),
            "by_strategy": detect_strategy_edges(
                strategy_rankings,
                horizon=horizon,
            ),
            "by_alignment_state": detect_alignment_state_edges(
                alignment_rankings,
                horizon=horizon,
            ),
            "by_ai_execution_state": detect_ai_execution_state_edges(
                ai_execution_rankings,
                horizon=horizon,
            ),
        }

    segment = build_segment_reports(
        dataset,
        horizons=tuple(HORIZONS),
        min_samples=10,
    )

    return {
        "dataset_rows": len(dataset),
        "performance": performance,
        "comparison": comparison,
        "ranking": ranking,
        "edge": edge,
        "segment": segment,
    }


def _build_top_highlights(
    schema_validation: dict[str, Any],
    strategy_lab: dict[str, Any],
) -> dict[str, Any]:
    ranking = strategy_lab.get("ranking", {}) or {}
    by_horizon: dict[str, dict[str, str]] = {}

    for horizon in HORIZONS:
        horizon_rank = ranking.get(horizon, {}) or {}
        by_horizon[horizon] = {
            "top_symbol": _extract_top_ranked_group(horizon_rank.get("by_symbol")),
            "top_strategy": _extract_top_ranked_group(horizon_rank.get("by_strategy")),
            "best_alignment_state": _extract_top_ranked_group(
                horizon_rank.get("by_alignment_state")
            ),
            "best_ai_execution_state": _extract_top_ranked_group(
                horizon_rank.get("by_ai_execution_state")
            ),
        }

    return {
        "invalid_record_count": int(schema_validation.get("invalid_records", 0)),
        "strategy_lab_dataset_rows": int(strategy_lab.get("dataset_rows", 0) or 0),
        "by_horizon": by_horizon,
    }


def _extract_ranking_items(report: Any) -> list[dict[str, Any]]:
    """Extract ranking rows from ranking report wrapper."""
    return _extract_ranked_groups(report)


def _extract_ranked_groups(report: Any) -> list[dict[str, Any]]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]

    if not isinstance(report, dict):
        return []

    for key in ("ranked_groups", "rankings", "results"):
        items = report.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    return []


def _build_edge_candidates_preview(strategy_lab: dict[str, Any]) -> dict[str, Any]:
    ranking = strategy_lab.get("ranking", {}) or {}
    by_horizon: dict[str, dict[str, Any]] = {}

    for horizon in HORIZONS:
        horizon_rank = ranking.get(horizon, {}) or {}

        top_strategy = _extract_edge_candidate(horizon_rank.get("by_strategy"))
        top_symbol = _extract_edge_candidate(horizon_rank.get("by_symbol"))
        top_alignment_state = _extract_edge_candidate(
            horizon_rank.get("by_alignment_state")
        )

        visible_strengths = [
            candidate["candidate_strength"]
            for candidate in (top_strategy, top_symbol, top_alignment_state)
            if candidate["candidate_strength"] != "insufficient_data"
        ]

        sample_gate = (
            "passed"
            if any(
                candidate.get("sample_gate") == "passed"
                for candidate in (top_strategy, top_symbol, top_alignment_state)
            )
            else "insufficient_data"
        )

        quality_gate = (
            "passed"
            if any(
                candidate.get("quality_gate") == "passed"
                for candidate in (top_strategy, top_symbol, top_alignment_state)
            )
            else "borderline"
            if visible_strengths
            else "failed"
        )

        by_horizon[horizon] = {
            "top_strategy": top_strategy,
            "top_symbol": top_symbol,
            "top_alignment_state": top_alignment_state,
            "sample_gate": sample_gate,
            "quality_gate": quality_gate,
            "candidate_strength": _max_candidate_strength(visible_strengths),
            "visibility_reason": _horizon_visibility_reason(sample_gate, quality_gate),
        }

    return {
        "minimum_sample_count": MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
        "strength_thresholds": {
            "scoring_model": STRENGTH_SCORING_MODEL,
            "raw_score_bands": STRENGTH_RAW_SCORE_BANDS,
            "hard_floors": {
                "sample_count": MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
                "labeled_count_gt": 0,
                "median_future_return_pct_gt": 0,
            },
            "emerging_moderate": {
                "sample_count": EDGE_EARLY_MODERATE_SAMPLE_COUNT,
                "median_future_return_pct": EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT,
                "positive_rate_pct": EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
                "robustness_pct": EDGE_EARLY_MODERATE_ROBUSTNESS_PCT,
            },
            "moderate": {
                "sample_count": EDGE_MODERATE_SAMPLE_COUNT,
                "median_future_return_pct": EDGE_MODERATE_MEDIAN_RETURN_PCT,
                "positive_rate_pct": EDGE_MODERATE_POSITIVE_RATE_PCT,
                "robustness_pct": EDGE_MODERATE_ROBUSTNESS_PCT,
            },
            "strong": {
                "sample_count": EDGE_STRONG_SAMPLE_COUNT,
                "median_future_return_pct": EDGE_STRONG_MEDIAN_RETURN_PCT,
                "positive_rate_pct": EDGE_STRONG_POSITIVE_RATE_PCT,
                "robustness_pct": EDGE_STRONG_ROBUSTNESS_PCT,
            },
            "classification_thresholds": {
                "moderate_min_aggregate_score": MODERATE_MIN_AGGREGATE_SCORE,
                "moderate_with_one_supporting_deficit_min_score": MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE,
                "moderate_with_two_supporting_deficits_min_score": MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE,
                "moderate_with_three_supporting_deficits_min_score": MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE,
                "strong_min_aggregate_score": STRONG_MIN_AGGREGATE_SCORE,
            },
            "recovery_guards": {
                "positive_rate_minimum_floor_pct": POSITIVE_RATE_MINIMUM_FLOOR_PCT,
                "three_supporting_deficits_min_positive_rate_pct": THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT,
                "three_supporting_deficits_min_robustness_pct": THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT,
            },
            "component_weights": STRENGTH_COMPONENT_WEIGHTS,
        },
        "by_horizon": by_horizon,
    }


def _extract_edge_candidate(report: Any) -> dict[str, Any]:
    ranked_groups = _extract_ranked_groups(report)

    for entry in ranked_groups:
        metrics = entry.get("metrics", {}) or {}
        sample_count = _to_float(metrics.get("sample_count"))
        labeled_count = _to_float(metrics.get("labeled_count"))
        coverage_pct = _to_float(metrics.get("coverage_pct"))
        median_future_return_pct = _to_float(metrics.get("median_future_return_pct"))
        positive_rate_pct = _to_float(
            metrics.get("positive_rate_pct", metrics.get("up_rate_pct"))
        )
        signal_match_rate_pct = _to_float(
            metrics.get("signal_match_rate_pct", metrics.get("signal_match_rate"))
        )
        bias_match_rate_pct = _to_float(
            metrics.get("bias_match_rate_pct", metrics.get("bias_match_rate"))
        )
        robustness_label, robustness_value = _select_robustness_signal(metrics)

        sample_gate_passed = _passes_absolute_minimum_gate(
            sample_count=sample_count,
            labeled_count=labeled_count,
            coverage_pct=coverage_pct,
            median_future_return_pct=median_future_return_pct,
        )

        if not sample_gate_passed:
            continue

        candidate_strength_diagnostics = _score_candidate_strength_diagnostics(
            sample_count=sample_count,
            median_future_return_pct=median_future_return_pct,
            positive_rate_pct=positive_rate_pct,
            robustness_value=robustness_value,
        )
        candidate_strength = str(candidate_strength_diagnostics["final_classification"])
        quality_gate = (
            "passed"
            if candidate_strength in {"moderate", "strong"}
            else "borderline"
        )
        visibility_reason = (
            "passed_sample_and_quality_gate"
            if quality_gate == "passed"
            else "passed_sample_gate_only"
        )

        return {
            "group": str(entry.get("group", "n/a")),
            "sample_count": int(sample_count),
            "labeled_count": int(labeled_count or 0),
            "coverage_pct": coverage_pct,
            "median_future_return_pct": median_future_return_pct,
            "positive_rate_pct": positive_rate_pct,
            "signal_match_rate_pct": signal_match_rate_pct,
            "bias_match_rate_pct": bias_match_rate_pct,
            "robustness_signal": robustness_label,
            "robustness_signal_pct": robustness_value,
            "sample_gate": "passed",
            "quality_gate": quality_gate,
            "candidate_strength": candidate_strength,
            "candidate_strength_diagnostics": candidate_strength_diagnostics,
            "visibility_reason": visibility_reason,
            "chosen_metric_summary": _build_candidate_metric_summary(
                sample_count=int(sample_count),
                median_future_return_pct=median_future_return_pct,
                positive_rate_pct=positive_rate_pct,
                robustness_label=robustness_label,
                robustness_value=robustness_value,
                diagnostics=candidate_strength_diagnostics,
            ),
        }

    return {
        "group": "insufficient_data",
        "sample_count": 0,
        "labeled_count": 0,
        "coverage_pct": None,
        "median_future_return_pct": None,
        "positive_rate_pct": None,
        "signal_match_rate_pct": None,
        "bias_match_rate_pct": None,
        "robustness_signal": "n/a",
        "robustness_signal_pct": None,
        "sample_gate": "failed",
        "quality_gate": "failed",
        "candidate_strength": "insufficient_data",
        "visibility_reason": "failed_absolute_minimum_gate",
        "chosen_metric_summary": "insufficient_data",
    }


def _passes_absolute_minimum_gate(
    *,
    sample_count: float | None,
    labeled_count: float | None,
    coverage_pct: float | None,
    median_future_return_pct: float | None,
) -> bool:
    has_label_support = (
        (labeled_count is not None and labeled_count > 0)
        or (coverage_pct is not None and coverage_pct > 0)
    )
    return (
        sample_count is not None
        and sample_count >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
        and has_label_support
        and median_future_return_pct is not None
        and median_future_return_pct > 0
    )


def _score_metric_component(
    *,
    metric_name: str,
    value: float | None,
    weight: float,
    strong_threshold: float,
    moderate_threshold: float,
    emerging_threshold: float,
    minimum_threshold: float | None = None,
    missing_band: str = "missing",
    missing_score: float = 0.55,
) -> dict[str, Any]:
    if value is None:
        return {
            "metric": metric_name,
            "value": None,
            "weight": weight,
            "band": missing_band,
            "raw_score": missing_score,
            "weighted_score": round(missing_score * weight * 100.0, 6),
            "strong_threshold": strong_threshold,
            "moderate_threshold": moderate_threshold,
            "emerging_threshold": emerging_threshold,
            "minimum_threshold": minimum_threshold,
        }

    if value >= strong_threshold:
        band = "strong"
        raw_score = STRENGTH_RAW_SCORE_BANDS["strong"]
    elif value >= moderate_threshold:
        band = "moderate"
        raw_score = STRENGTH_RAW_SCORE_BANDS["moderate"]
    elif value >= emerging_threshold:
        band = "emerging"
        raw_score = STRENGTH_RAW_SCORE_BANDS["emerging"]
    elif minimum_threshold is None or value >= minimum_threshold:
        band = "thin"
        raw_score = STRENGTH_RAW_SCORE_BANDS["thin"]
    else:
        band = "below_floor"
        raw_score = STRENGTH_RAW_SCORE_BANDS["below_floor"]

    return {
        "metric": metric_name,
        "value": value,
        "weight": weight,
        "band": band,
        "raw_score": raw_score,
        "weighted_score": round(raw_score * weight * 100.0, 6),
        "strong_threshold": strong_threshold,
        "moderate_threshold": moderate_threshold,
        "emerging_threshold": emerging_threshold,
        "minimum_threshold": minimum_threshold,
    }


def _split_major_deficits(
    major_deficits: list[str],
) -> tuple[list[str], list[str], list[str]]:
    critical = [item for item in major_deficits if item in CRITICAL_MAJOR_DEFICITS]
    supporting = [item for item in major_deficits if item in SUPPORTING_MAJOR_DEFICITS]
    other = [
        item
        for item in major_deficits
        if item not in CRITICAL_MAJOR_DEFICITS
        and item not in SUPPORTING_MAJOR_DEFICITS
    ]
    return critical, supporting, other


def _can_classify_as_moderate(
    *,
    aggregate_score: float,
    major_deficits: list[str],
    sample_count: float | None,
    positive_rate_pct: float | None,
    robustness_value: float | None,
) -> tuple[bool, str]:
    critical_major_deficits, supporting_major_deficits, other_major_deficits = (
        _split_major_deficits(major_deficits)
    )

    if critical_major_deficits or other_major_deficits:
        return False, "critical_or_unknown_major_deficit_present"

    if not supporting_major_deficits:
        return (
            aggregate_score >= MODERATE_MIN_AGGREGATE_SCORE,
            "cleared_weighted_moderate_profile"
            if aggregate_score >= MODERATE_MIN_AGGREGATE_SCORE
            else "aggregate_below_moderate_threshold",
        )

    if len(supporting_major_deficits) == 1:
        return (
            aggregate_score >= MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE,
            "cleared_weighted_moderate_profile_with_one_supporting_deficit"
            if aggregate_score >= MODERATE_WITH_ONE_SUPPORTING_DEFICIT_MIN_SCORE
            else "one_supporting_deficit_but_aggregate_too_low",
        )

    if len(supporting_major_deficits) == 2:
        if aggregate_score < MODERATE_WITH_TWO_SUPPORTING_DEFICITS_MIN_SCORE:
            return False, "two_supporting_deficits_but_aggregate_too_low"
        if sample_count is None or sample_count < EDGE_MODERATE_SAMPLE_COUNT:
            return False, "two_supporting_deficits_but_sample_not_moderate"
        if (
            positive_rate_pct is None
            or positive_rate_pct < POSITIVE_RATE_MINIMUM_FLOOR_PCT
        ):
            return False, "two_supporting_deficits_but_positive_rate_below_floor"

        return True, "cleared_weighted_moderate_profile_with_two_supporting_deficits"

    if len(supporting_major_deficits) == 3:
        if aggregate_score < MODERATE_WITH_THREE_SUPPORTING_DEFICITS_MIN_SCORE:
            return False, "three_supporting_deficits_but_aggregate_too_low"
        if sample_count is None or sample_count < EDGE_MODERATE_SAMPLE_COUNT:
            return False, "three_supporting_deficits_but_sample_not_moderate"
        if (
            positive_rate_pct is None
            or positive_rate_pct < THREE_SUPPORTING_DEFICITS_MIN_POSITIVE_RATE_PCT
        ):
            return False, "three_supporting_deficits_but_positive_rate_too_low"
        if (
            robustness_value is None
            or robustness_value < THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT
        ):
            return False, "three_supporting_deficits_but_robustness_too_low"

        return True, "cleared_weighted_moderate_profile_with_three_supporting_deficits"

    if len(supporting_major_deficits) > 3:
        return False, "supporting_deficits_depth_exceeds_recovery_band"

    return False, "supporting_deficits_depth_exceeds_recovery_band"


def _score_candidate_strength_diagnostics(
    *,
    sample_count: float | None,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_value: float | None,
) -> dict[str, Any]:
    hard_blockers: list[str] = []
    soft_penalties: list[str] = []
    major_deficits: list[str] = []

    if sample_count is None or sample_count < MIN_EDGE_CANDIDATE_SAMPLE_COUNT:
        hard_blockers.append("sample_count_below_absolute_floor")
    if median_future_return_pct is None or median_future_return_pct <= 0:
        hard_blockers.append("median_future_return_pct_non_positive")

    sample_component = _score_metric_component(
        metric_name="sample_count",
        value=sample_count,
        weight=STRENGTH_COMPONENT_WEIGHTS["sample_count"],
        strong_threshold=EDGE_STRONG_SAMPLE_COUNT,
        moderate_threshold=EDGE_MODERATE_SAMPLE_COUNT,
        emerging_threshold=EDGE_EARLY_MODERATE_SAMPLE_COUNT,
        minimum_threshold=MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
        missing_band="missing",
        missing_score=0.0,
    )
    median_component = _score_metric_component(
        metric_name="median_future_return_pct",
        value=median_future_return_pct,
        weight=STRENGTH_COMPONENT_WEIGHTS["median_future_return_pct"],
        strong_threshold=EDGE_STRONG_MEDIAN_RETURN_PCT,
        moderate_threshold=EDGE_MODERATE_MEDIAN_RETURN_PCT,
        emerging_threshold=EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT,
        minimum_threshold=0.0,
        missing_band="missing",
        missing_score=0.0,
    )
    positive_component = _score_metric_component(
        metric_name="positive_rate_pct",
        value=positive_rate_pct,
        weight=STRENGTH_COMPONENT_WEIGHTS["positive_rate_pct"],
        strong_threshold=EDGE_STRONG_POSITIVE_RATE_PCT,
        moderate_threshold=EDGE_MODERATE_POSITIVE_RATE_PCT,
        emerging_threshold=EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT,
        minimum_threshold=POSITIVE_RATE_MINIMUM_FLOOR_PCT,
        missing_band="missing_soft",
        missing_score=0.45,
    )
    robustness_component = _score_metric_component(
        metric_name="robustness_value",
        value=robustness_value,
        weight=STRENGTH_COMPONENT_WEIGHTS["robustness_value"],
        strong_threshold=EDGE_STRONG_ROBUSTNESS_PCT,
        moderate_threshold=EDGE_MODERATE_ROBUSTNESS_PCT,
        emerging_threshold=EDGE_EARLY_MODERATE_ROBUSTNESS_PCT,
        minimum_threshold=THREE_SUPPORTING_DEFICITS_MIN_ROBUSTNESS_PCT,
        missing_band="missing_neutral",
        missing_score=0.6,
    )

    if sample_count is not None and sample_count < EDGE_MODERATE_SAMPLE_COUNT:
        soft_penalties.append("thin_sample_count")
    if sample_count is not None and sample_count < EDGE_EARLY_MODERATE_SAMPLE_COUNT:
        major_deficits.append("sample_count_below_emerging_moderate")

    if (
        median_future_return_pct is not None
        and median_future_return_pct < EDGE_MODERATE_MEDIAN_RETURN_PCT
    ):
        soft_penalties.append("subscale_median_return")
    if (
        median_future_return_pct is not None
        and median_future_return_pct < EDGE_EARLY_MODERATE_MEDIAN_RETURN_PCT
    ):
        major_deficits.append("median_return_below_emerging_moderate")

    if positive_rate_pct is None:
        soft_penalties.append("missing_positive_rate")
    else:
        if positive_rate_pct < EDGE_MODERATE_POSITIVE_RATE_PCT:
            soft_penalties.append("subscale_positive_rate")
        if positive_rate_pct < EDGE_EARLY_MODERATE_POSITIVE_RATE_PCT:
            major_deficits.append("positive_rate_below_emerging_moderate")
        if positive_rate_pct < 50.0:
            soft_penalties.append("positive_rate_below_coinflip")

    if robustness_value is None:
        soft_penalties.append("missing_robustness_signal")
    else:
        if robustness_value < EDGE_MODERATE_ROBUSTNESS_PCT:
            soft_penalties.append("subscale_robustness")
        if robustness_value < EDGE_EARLY_MODERATE_ROBUSTNESS_PCT:
            major_deficits.append("robustness_below_emerging_moderate")

    component_scores = {
        "sample_count": sample_component,
        "median_future_return_pct": median_component,
        "positive_rate_pct": positive_component,
        "robustness_value": robustness_component,
    }
    aggregate_score = round(
        sample_component["weighted_score"]
        + median_component["weighted_score"]
        + positive_component["weighted_score"]
        + robustness_component["weighted_score"],
        6,
    )

    critical_major_deficits, supporting_major_deficits, other_major_deficits = (
        _split_major_deficits(major_deficits)
    )

    if hard_blockers:
        final_classification = "insufficient_data"
        classification_reason = "failed_absolute_floor"
    elif (
        aggregate_score >= STRONG_MIN_AGGREGATE_SCORE
        and sample_count is not None
        and sample_count >= EDGE_STRONG_SAMPLE_COUNT
        and median_future_return_pct is not None
        and median_future_return_pct >= EDGE_STRONG_MEDIAN_RETURN_PCT
        and positive_rate_pct is not None
        and positive_rate_pct >= EDGE_MODERATE_POSITIVE_RATE_PCT
        and len(major_deficits) == 0
    ):
        final_classification = "strong"
        classification_reason = "cleared_strong_weighted_profile"
    else:
        can_moderate, moderate_reason = _can_classify_as_moderate(
            aggregate_score=aggregate_score,
            major_deficits=major_deficits,
            sample_count=sample_count,
            positive_rate_pct=positive_rate_pct,
            robustness_value=robustness_value,
        )
        if can_moderate:
            final_classification = "moderate"
            classification_reason = moderate_reason
        else:
            final_classification = "weak"
            classification_reason = moderate_reason

    return {
        "scoring_model": STRENGTH_SCORING_MODEL,
        "component_scores": component_scores,
        "hard_blockers": hard_blockers,
        "soft_penalties": soft_penalties,
        "major_deficits": major_deficits,
        "major_deficit_breakdown": {
            "critical": critical_major_deficits,
            "supporting": supporting_major_deficits,
            "other": other_major_deficits,
        },
        "aggregate_score": aggregate_score,
        "final_classification": final_classification,
        "classification_reason": classification_reason,
    }


def _score_candidate_strength(
    sample_count: float,
    median_future_return_pct: float,
    positive_rate_pct: float | None,
    robustness_value: float | None,
) -> str:
    diagnostics = _score_candidate_strength_diagnostics(
        sample_count=sample_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_value=robustness_value,
    )
    return str(diagnostics["final_classification"])


def _select_robustness_signal(metrics: dict[str, Any]) -> tuple[str, float | None]:
    for output_label, keys in (
        ("signal_match_rate_pct", ("signal_match_rate_pct", "signal_match_rate")),
        ("bias_match_rate_pct", ("bias_match_rate_pct", "bias_match_rate")),
        ("coverage_pct", ("coverage_pct",)),
    ):
        for key in keys:
            value = _to_float(metrics.get(key))
            if value is not None:
                return output_label, value
    return "n/a", None


def _build_candidate_metric_summary(
    sample_count: int,
    median_future_return_pct: float,
    positive_rate_pct: float | None,
    robustness_label: str,
    robustness_value: float | None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    parts = [
        f"sample={sample_count}",
        f"median={median_future_return_pct}",
        f"positive_rate={positive_rate_pct if positive_rate_pct is not None else 'n/a'}",
    ]
    if robustness_label != "n/a" and robustness_value is not None:
        parts.append(f"{robustness_label}={robustness_value}")

    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    aggregate_score = diagnostics.get("aggregate_score")
    if isinstance(aggregate_score, (int, float)):
        parts.append(f"aggregate_score={round(float(aggregate_score), 2)}")

    classification = diagnostics.get("final_classification")
    if isinstance(classification, str) and classification:
        parts.append(f"classification={classification}")

    reason = diagnostics.get("classification_reason")
    if isinstance(reason, str) and reason:
        parts.append(f"classification_reason={reason}")

    penalties = diagnostics.get("soft_penalties")
    if isinstance(penalties, list) and penalties:
        parts.append(f"soft_penalties={'+'.join(str(item) for item in penalties)}")

    deficits = diagnostics.get("major_deficit_breakdown")
    if isinstance(deficits, dict):
        supporting = deficits.get("supporting")
        critical = deficits.get("critical")
        other = deficits.get("other")
        if isinstance(supporting, list) and supporting:
            parts.append(
                f"supporting_major_deficits={'+'.join(str(item) for item in supporting)}"
            )
        if isinstance(critical, list) and critical:
            parts.append(
                f"critical_major_deficits={'+'.join(str(item) for item in critical)}"
            )
        if isinstance(other, list) and other:
            parts.append(
                f"other_major_deficits={'+'.join(str(item) for item in other)}"
            )

    return ", ".join(parts)


def _horizon_visibility_reason(sample_gate: str, quality_gate: str) -> str:
    if sample_gate != "passed":
        return "failed_absolute_minimum_gate"
    if quality_gate == "passed":
        return "passed_sample_and_quality_gate"
    return "passed_sample_gate_only"

def _build_edge_candidate_rows(
    input_path: Path,
    *,
    edge_candidates_preview: dict[str, Any] | None = None,
    edge_stability_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = build_dataset(path=input_path)
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    dropped_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        symbol = _joined_symbol(row)
        strategy = _joined_strategy(row)
        if symbol is None:
            dropped_rows.append(
                {
                    "row_index": index,
                    "symbol": row.get("symbol"),
                    "strategy": row.get("selected_strategy"),
                    "horizon": None,
                    "drop_reason": "MISSING_SYMBOL",
                }
            )
            continue
        if strategy is None:
            dropped_rows.append(
                {
                    "row_index": index,
                    "symbol": row.get("symbol"),
                    "strategy": row.get("selected_strategy"),
                    "horizon": None,
                    "drop_reason": "MISSING_STRATEGY",
                }
            )
            continue
        grouped_rows[(symbol, strategy)].append(row)

    rows: list[dict[str, Any]] = []
    visible_horizons_by_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    identity_horizon_evaluations: list[dict[str, Any]] = []
    preview_visibility = _build_edge_candidate_preview_visibility(
        edge_candidates_preview=edge_candidates_preview,
        edge_stability_preview=edge_stability_preview,
    )

    for (symbol, strategy), identity_rows in sorted(grouped_rows.items()):
        raw_preview_visibility = _build_identity_preview_visibility(
            symbol=symbol,
            strategy=strategy,
            preview_visibility=preview_visibility,
        )
        compatibility_filtered_preview_visibility = (
            _build_compatibility_filtered_preview_visibility(
                strategy=strategy,
                raw_preview_visibility=raw_preview_visibility,
            )
        )

        horizon_evaluations: dict[str, dict[str, Any]] = {}
        for horizon in HORIZONS:
            evaluation = _evaluate_joined_edge_candidate_horizon(
                symbol=symbol,
                strategy=strategy,
                horizon=horizon,
                rows=identity_rows,
            )
            horizon_evaluations[horizon] = evaluation

            if evaluation.get("status") != "selected":
                diagnostic_rows.append(
                    _build_joined_candidate_diagnostic_row_from_evaluation(evaluation)
                )
                continue

            candidate_row = _build_joined_candidate_row_from_evaluation(evaluation)
            rows.append(candidate_row)
            visible_horizons_by_identity[(symbol, strategy)].append(horizon)

        actual_joined_eligible_horizons = sorted(
            visible_horizons_by_identity.get((symbol, strategy), [])
        )
        identity_horizon_evaluations.append(
            {
                "identity_key": f"{symbol}:{strategy}",
                "symbol": symbol,
                "strategy": strategy,
                "strategy_compatible_horizons": _sorted_compatible_horizons(strategy),
                "raw_preview_visibility": raw_preview_visibility,
                "compatibility_filtered_preview_visibility": (
                    compatibility_filtered_preview_visibility
                ),
                "actual_joined_eligible_horizons": actual_joined_eligible_horizons,
                "actual_joined_stability_label": (
                    "multi_horizon_confirmed"
                    if len(actual_joined_eligible_horizons) >= 2
                    else "single_horizon_only"
                ),
                "horizon_evaluations": horizon_evaluations,
            }
        )

    identity_evaluations_by_key = {
        (
            str(item.get("symbol") or ""),
            str(item.get("strategy") or ""),
        ): item
        for item in identity_horizon_evaluations
    }

    for row in rows:
        identity = (str(row["symbol"]), str(row["strategy"]))
        joined_visible_horizons = sorted(visible_horizons_by_identity.get(identity, []))
        identity_evaluation = _coerce_dict(identity_evaluations_by_key.get(identity))

        row["selected_visible_horizons"] = joined_visible_horizons
        row["actual_joined_eligible_horizons"] = joined_visible_horizons
        row["selected_stability_label"] = (
            "multi_horizon_confirmed"
            if len(joined_visible_horizons) >= 2
            else "single_horizon_only"
        )

        raw_preview_visibility = _coerce_dict(
            identity_evaluation.get("raw_preview_visibility")
        )
        compatibility_filtered_preview_visibility = _coerce_dict(
            identity_evaluation.get("compatibility_filtered_preview_visibility")
        )
        symbol_preview = _coerce_dict(raw_preview_visibility.get("symbol"))
        strategy_preview = _coerce_dict(raw_preview_visibility.get("strategy"))
        compatibility_symbol_preview = _coerce_dict(
            compatibility_filtered_preview_visibility.get("symbol")
        )
        compatibility_strategy_preview = _coerce_dict(
            compatibility_filtered_preview_visibility.get("strategy")
        )

        _attach_preview_visibility_metadata(
            row=row,
            symbol_preview=symbol_preview,
            strategy_preview=strategy_preview,
            compatibility_symbol_preview=compatibility_symbol_preview,
            compatibility_strategy_preview=compatibility_strategy_preview,
        )

        row["horizon_evaluation"] = _coerce_dict(
            _coerce_dict(identity_evaluation.get("horizon_evaluations")).get(
                str(row["horizon"])
            )
        )
        row["visibility_diagnostics"] = _build_visibility_diagnostics(
            symbol_preview=symbol_preview,
            strategy_preview=strategy_preview,
            compatibility_symbol_preview=compatibility_symbol_preview,
            compatibility_strategy_preview=compatibility_strategy_preview,
            joined_visible_horizons=joined_visible_horizons,
        )
        row["visibility_reason"] = _resolve_edge_candidate_visibility_reason(
            base_reason=str(
                row.get("visibility_reason") or "passed_sample_and_quality_gate"
            ),
            symbol_preview=symbol_preview,
            strategy_preview=strategy_preview,
            compatibility_symbol_preview=compatibility_symbol_preview,
            compatibility_strategy_preview=compatibility_strategy_preview,
            joined_visible_horizons=joined_visible_horizons,
        )

    rows.sort(
        key=lambda item: (
            str(item.get("horizon") or ""),
            str(item.get("symbol") or ""),
            str(item.get("strategy") or ""),
        )
    )
    diagnostic_rows.sort(
        key=lambda item: (
            str(item.get("horizon") or ""),
            str(item.get("symbol") or ""),
            str(item.get("strategy") or ""),
            str(item.get("rejection_reason") or ""),
        )
    )

    empty_reason_summary = _build_edge_candidate_empty_reason_summary(
        rows=rows,
        diagnostic_rows=diagnostic_rows,
        identity_horizon_evaluations=identity_horizon_evaluations,
    )

    return {
        "row_count": len(rows),
        "rows": rows,
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostic_rows": diagnostic_rows,
        "empty_reason_summary": empty_reason_summary,
        "dropped_row_count": len(dropped_rows),
        "dropped_rows": dropped_rows[:25],
        "identity_horizon_evaluations": identity_horizon_evaluations,
    }


def _build_edge_candidate_preview_visibility(
    *,
    edge_candidates_preview: dict[str, Any] | None,
    edge_stability_preview: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    by_horizon = _coerce_dict(_coerce_dict(edge_candidates_preview).get("by_horizon"))
    stability_root = _coerce_dict(edge_stability_preview)

    return {
        "symbol": _build_category_preview_visibility(
            by_horizon=by_horizon,
            stability_entry=_coerce_dict(stability_root.get("symbol")),
            category="symbol",
            candidate_key="top_symbol",
        ),
        "strategy": _build_category_preview_visibility(
            by_horizon=by_horizon,
            stability_entry=_coerce_dict(stability_root.get("strategy")),
            category="strategy",
            candidate_key="top_strategy",
        ),
    }


def _build_category_preview_visibility(
    *,
    by_horizon: dict[str, Any],
    stability_entry: dict[str, Any],
    category: str,
    candidate_key: str,
) -> dict[str, dict[str, Any]]:
    horizons_by_group: dict[str, set[str]] = defaultdict(set)
    quality_passed_horizons_by_group: dict[str, set[str]] = defaultdict(set)
    horizon_strengths_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    horizon_quality_gates_by_group: dict[str, dict[str, str]] = defaultdict(dict)

    for horizon in HORIZONS:
        horizon_data = _coerce_dict(by_horizon.get(horizon))
        candidate = _coerce_dict(horizon_data.get(candidate_key))
        candidate_strength = str(candidate.get("candidate_strength") or "insufficient_data")
        if candidate_strength == "insufficient_data":
            continue

        group = _normalize_preview_visibility_group(category, candidate.get("group"))
        if group is None:
            continue

        horizons_by_group[group].add(horizon)
        horizon_strengths_by_group[group][horizon] = candidate_strength

        quality_gate = str(candidate.get("quality_gate") or "failed")
        horizon_quality_gates_by_group[group][horizon] = quality_gate
        if quality_gate == "passed":
            quality_passed_horizons_by_group[group].add(horizon)

    stability_group = _normalize_preview_visibility_group(
        category,
        stability_entry.get("group"),
    )

    visibility: dict[str, dict[str, Any]] = {}
    for group, horizons in horizons_by_group.items():
        visible_horizons = sorted(horizons)
        quality_passed_horizons = sorted(quality_passed_horizons_by_group.get(group, set()))
        stability_label = (
            "multi_horizon_confirmed"
            if len(visible_horizons) >= 2
            else "single_horizon_only"
        )
        visibility_reason = (
            "preview_visible_candidate_across_multiple_horizons"
            if len(visible_horizons) >= 2
            else "preview_visible_in_one_horizon_only"
        )

        if group == stability_group:
            stability_label = str(
                stability_entry.get("stability_label") or stability_label
            )
            visibility_reason = str(
                stability_entry.get("visibility_reason") or visibility_reason
            )

        visibility[group] = {
            "visible_horizons": visible_horizons,
            "quality_passed_horizons": quality_passed_horizons,
            "horizon_strengths": dict(horizon_strengths_by_group.get(group, {})),
            "horizon_quality_gates": dict(horizon_quality_gates_by_group.get(group, {})),
            "stability_label": stability_label,
            "visibility_reason": visibility_reason,
        }

    return visibility


def _sorted_compatible_horizons(strategy: str) -> list[str]:
    compatible = STRATEGY_HORIZON_COMPATIBILITY.get(strategy, set())
    return [horizon for horizon in HORIZONS if horizon in compatible]


def _build_strategy_horizon_compatibility_detail(
    *,
    strategy: str,
    horizon: str,
) -> dict[str, Any]:
    configured_horizons = sorted(STRATEGY_HORIZON_COMPATIBILITY.get(strategy, set()))
    analyzer_supported_horizons = list(HORIZONS)
    analyzer_compatible_horizons = [
        item for item in analyzer_supported_horizons if item in configured_horizons
    ]

    if analyzer_compatible_horizons:
        detail = (
            f"Strategy '{strategy}' excludes analyzer horizon '{horizon}'. "
            f"Analyzer evaluates {analyzer_supported_horizons}; "
            f"strategy allows {configured_horizons}; "
            f"overlap is {analyzer_compatible_horizons}."
        )
    else:
        detail = (
            f"Strategy '{strategy}' cannot participate in this analyzer horizon set. "
            f"Analyzer evaluates {analyzer_supported_horizons}; "
            f"strategy allows {configured_horizons}; "
            f"overlap is empty."
        )

    return {
        "configured_horizons": configured_horizons,
        "analyzer_supported_horizons": analyzer_supported_horizons,
        "analyzer_compatible_horizons": analyzer_compatible_horizons,
        "has_analyzer_compatible_horizon": bool(analyzer_compatible_horizons),
        "detail": detail,
    }


def _build_identity_preview_visibility(
    *,
    symbol: str,
    strategy: str,
    preview_visibility: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    symbol_preview = _coerce_dict(preview_visibility.get("symbol", {}).get(symbol))
    strategy_preview = _coerce_dict(preview_visibility.get("strategy", {}).get(strategy))

    symbol_visible_horizons = _normalize_horizon_list(symbol_preview.get("visible_horizons")) or []
    strategy_visible_horizons = _normalize_horizon_list(strategy_preview.get("visible_horizons")) or []

    symbol_quality_passed_horizons = (
        _normalize_horizon_list(symbol_preview.get("quality_passed_horizons")) or []
    )
    strategy_quality_passed_horizons = (
        _normalize_horizon_list(strategy_preview.get("quality_passed_horizons")) or []
    )

    return {
        "symbol": symbol_preview,
        "strategy": strategy_preview,
        "raw_category_union_horizons": _merge_unique_horizons(
            symbol_visible_horizons,
            strategy_visible_horizons,
        ),
        "raw_category_overlap_horizons": _intersect_horizons(
            symbol_visible_horizons,
            strategy_visible_horizons,
        ),
        "raw_category_quality_passed_union_horizons": _merge_unique_horizons(
            symbol_quality_passed_horizons,
            strategy_quality_passed_horizons,
        ),
        "raw_category_quality_passed_overlap_horizons": _intersect_horizons(
            symbol_quality_passed_horizons,
            strategy_quality_passed_horizons,
        ),
    }


def _build_compatibility_filtered_preview_visibility(
    *,
    strategy: str,
    raw_preview_visibility: dict[str, Any],
) -> dict[str, Any]:
    compatible_horizons = _sorted_compatible_horizons(strategy)
    symbol_preview = _coerce_dict(raw_preview_visibility.get("symbol"))
    strategy_preview = _coerce_dict(raw_preview_visibility.get("strategy"))

    filtered_symbol_horizons = _filter_horizons_to_compatible(
        _normalize_horizon_list(symbol_preview.get("visible_horizons")) or [],
        compatible_horizons,
    )
    filtered_strategy_horizons = _filter_horizons_to_compatible(
        _normalize_horizon_list(strategy_preview.get("visible_horizons")) or [],
        compatible_horizons,
    )
    filtered_symbol_quality_passed_horizons = _filter_horizons_to_compatible(
        _normalize_horizon_list(symbol_preview.get("quality_passed_horizons")) or [],
        compatible_horizons,
    )
    filtered_strategy_quality_passed_horizons = _filter_horizons_to_compatible(
        _normalize_horizon_list(strategy_preview.get("quality_passed_horizons")) or [],
        compatible_horizons,
    )

    return {
        "strategy_compatible_horizons": compatible_horizons,
        "symbol": _copy_preview_visibility_entry(
            symbol_preview,
            visible_horizons=filtered_symbol_horizons,
            quality_passed_horizons=filtered_symbol_quality_passed_horizons,
        ),
        "strategy": _copy_preview_visibility_entry(
            strategy_preview,
            visible_horizons=filtered_strategy_horizons,
            quality_passed_horizons=filtered_strategy_quality_passed_horizons,
        ),
        "compatibility_filtered_category_union_horizons": _merge_unique_horizons(
            filtered_symbol_horizons,
            filtered_strategy_horizons,
        ),
        "compatibility_filtered_category_overlap_horizons": _intersect_horizons(
            filtered_symbol_horizons,
            filtered_strategy_horizons,
        ),
        "compatibility_filtered_quality_passed_union_horizons": _merge_unique_horizons(
            filtered_symbol_quality_passed_horizons,
            filtered_strategy_quality_passed_horizons,
        ),
        "compatibility_filtered_quality_passed_overlap_horizons": _intersect_horizons(
            filtered_symbol_quality_passed_horizons,
            filtered_strategy_quality_passed_horizons,
        ),
        "visibility_reason": "filtered_by_strategy_horizon_compatibility",
    }


def _copy_preview_visibility_entry(
    entry: dict[str, Any],
    *,
    visible_horizons: list[str],
    quality_passed_horizons: list[str],
) -> dict[str, Any]:
    horizon_strengths = _normalize_horizon_strength_map(entry.get("horizon_strengths"))
    horizon_quality_gates = _normalize_horizon_strength_map(entry.get("horizon_quality_gates"))

    filtered_horizon_strengths = {
        horizon: strength
        for horizon, strength in horizon_strengths.items()
        if horizon in visible_horizons
    }
    filtered_horizon_quality_gates = {
        horizon: gate
        for horizon, gate in horizon_quality_gates.items()
        if horizon in visible_horizons
    }

    if not entry:
        return {
            "visible_horizons": visible_horizons,
            "quality_passed_horizons": quality_passed_horizons,
            "horizon_strengths": filtered_horizon_strengths,
            "horizon_quality_gates": filtered_horizon_quality_gates,
            "stability_label": None,
            "visibility_reason": None,
        }

    return {
        "visible_horizons": visible_horizons,
        "quality_passed_horizons": quality_passed_horizons,
        "horizon_strengths": filtered_horizon_strengths,
        "horizon_quality_gates": filtered_horizon_quality_gates,
        "stability_label": _normalize_text(entry.get("stability_label")),
        "visibility_reason": _normalize_text(entry.get("visibility_reason")),
    }


def _merge_unique_horizons(*horizon_lists: list[str]) -> list[str]:
    merged: list[str] = []
    for horizon in HORIZONS:
        if any(horizon in items for items in horizon_lists) and horizon not in merged:
            merged.append(horizon)
    return merged


def _intersect_horizons(left: list[str], right: list[str]) -> list[str]:
    return [horizon for horizon in HORIZONS if horizon in left and horizon in right]


def _filter_horizons_to_compatible(
    horizons: list[str],
    compatible_horizons: list[str],
) -> list[str]:
    compatible_set = set(compatible_horizons)
    return [horizon for horizon in horizons if horizon in compatible_set]


def _normalize_preview_visibility_group(category: str, value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"insufficient_data", "n/a", "na", "none", "null", "unknown"}:
        return None

    if category == "symbol":
        return text.upper()
    return lowered


def _attach_preview_visibility_metadata(
    *,
    row: dict[str, Any],
    symbol_preview: dict[str, Any],
    strategy_preview: dict[str, Any],
    compatibility_symbol_preview: dict[str, Any],
    compatibility_strategy_preview: dict[str, Any],
) -> None:
    symbol_visible_horizons = _normalize_horizon_list(
        symbol_preview.get("visible_horizons")
    )
    if symbol_visible_horizons:
        row["preview_symbol_visible_horizons"] = symbol_visible_horizons

    symbol_quality_passed_horizons = _normalize_horizon_list(
        symbol_preview.get("quality_passed_horizons")
    )
    if symbol_quality_passed_horizons is not None:
        row["preview_symbol_quality_passed_horizons"] = symbol_quality_passed_horizons

    symbol_horizon_strengths = _normalize_horizon_strength_map(
        symbol_preview.get("horizon_strengths")
    )
    if symbol_horizon_strengths:
        row["preview_symbol_horizon_strengths"] = symbol_horizon_strengths

    symbol_stability_label = _normalize_text(symbol_preview.get("stability_label"))
    if symbol_stability_label is not None:
        row["preview_symbol_stability_label"] = symbol_stability_label

    symbol_visibility_reason = _normalize_text(symbol_preview.get("visibility_reason"))
    if symbol_visibility_reason is not None:
        row["preview_symbol_visibility_reason"] = symbol_visibility_reason

    strategy_visible_horizons = _normalize_horizon_list(
        strategy_preview.get("visible_horizons")
    )
    if strategy_visible_horizons:
        row["preview_strategy_visible_horizons"] = strategy_visible_horizons

    strategy_quality_passed_horizons = _normalize_horizon_list(
        strategy_preview.get("quality_passed_horizons")
    )
    if strategy_quality_passed_horizons is not None:
        row["preview_strategy_quality_passed_horizons"] = (
            strategy_quality_passed_horizons
        )

    strategy_horizon_strengths = _normalize_horizon_strength_map(
        strategy_preview.get("horizon_strengths")
    )
    if strategy_horizon_strengths:
        row["preview_strategy_horizon_strengths"] = strategy_horizon_strengths

    strategy_stability_label = _normalize_text(strategy_preview.get("stability_label"))
    if strategy_stability_label is not None:
        row["preview_strategy_stability_label"] = strategy_stability_label

    strategy_visibility_reason = _normalize_text(
        strategy_preview.get("visibility_reason")
    )
    if strategy_visibility_reason is not None:
        row["preview_strategy_visibility_reason"] = strategy_visibility_reason

    compatibility_symbol_visible_horizons = _normalize_horizon_list(
        compatibility_symbol_preview.get("visible_horizons")
    )
    if compatibility_symbol_visible_horizons is not None:
        row["compatibility_preview_symbol_visible_horizons"] = (
            compatibility_symbol_visible_horizons
        )

    compatibility_symbol_quality_passed_horizons = _normalize_horizon_list(
        compatibility_symbol_preview.get("quality_passed_horizons")
    )
    if compatibility_symbol_quality_passed_horizons is not None:
        row["compatibility_preview_symbol_quality_passed_horizons"] = (
            compatibility_symbol_quality_passed_horizons
        )

    compatibility_symbol_horizon_strengths = _normalize_horizon_strength_map(
        compatibility_symbol_preview.get("horizon_strengths")
    )
    if compatibility_symbol_horizon_strengths:
        row["compatibility_preview_symbol_horizon_strengths"] = (
            compatibility_symbol_horizon_strengths
        )

    compatibility_symbol_stability_label = _normalize_text(
        compatibility_symbol_preview.get("stability_label")
    )
    if compatibility_symbol_stability_label is not None:
        row["compatibility_preview_symbol_stability_label"] = (
            compatibility_symbol_stability_label
        )

    compatibility_symbol_visibility_reason = _normalize_text(
        compatibility_symbol_preview.get("visibility_reason")
    )
    if compatibility_symbol_visibility_reason is not None:
        row["compatibility_preview_symbol_visibility_reason"] = (
            compatibility_symbol_visibility_reason
        )

    compatibility_strategy_visible_horizons = _normalize_horizon_list(
        compatibility_strategy_preview.get("visible_horizons")
    )
    if compatibility_strategy_visible_horizons is not None:
        row["compatibility_preview_strategy_visible_horizons"] = (
            compatibility_strategy_visible_horizons
        )

    compatibility_strategy_quality_passed_horizons = _normalize_horizon_list(
        compatibility_strategy_preview.get("quality_passed_horizons")
    )
    if compatibility_strategy_quality_passed_horizons is not None:
        row["compatibility_preview_strategy_quality_passed_horizons"] = (
            compatibility_strategy_quality_passed_horizons
        )

    compatibility_strategy_horizon_strengths = _normalize_horizon_strength_map(
        compatibility_strategy_preview.get("horizon_strengths")
    )
    if compatibility_strategy_horizon_strengths:
        row["compatibility_preview_strategy_horizon_strengths"] = (
            compatibility_strategy_horizon_strengths
        )

    compatibility_strategy_stability_label = _normalize_text(
        compatibility_strategy_preview.get("stability_label")
    )
    if compatibility_strategy_stability_label is not None:
        row["compatibility_preview_strategy_stability_label"] = (
            compatibility_strategy_stability_label
        )

    compatibility_strategy_visibility_reason = _normalize_text(
        compatibility_strategy_preview.get("visibility_reason")
    )
    if compatibility_strategy_visibility_reason is not None:
        row["compatibility_preview_strategy_visibility_reason"] = (
            compatibility_strategy_visibility_reason
        )


def _build_visibility_diagnostics(
    *,
    symbol_preview: dict[str, Any],
    strategy_preview: dict[str, Any],
    compatibility_symbol_preview: dict[str, Any],
    compatibility_strategy_preview: dict[str, Any],
    joined_visible_horizons: list[str],
) -> dict[str, Any]:
    raw_symbol_visible_horizons = (
        _normalize_horizon_list(symbol_preview.get("visible_horizons")) or []
    )
    raw_strategy_visible_horizons = (
        _normalize_horizon_list(strategy_preview.get("visible_horizons")) or []
    )
    compatibility_symbol_visible_horizons = (
        _normalize_horizon_list(compatibility_symbol_preview.get("visible_horizons")) or []
    )
    compatibility_strategy_visible_horizons = (
        _normalize_horizon_list(compatibility_strategy_preview.get("visible_horizons")) or []
    )

    raw_union = _merge_unique_horizons(
        raw_symbol_visible_horizons,
        raw_strategy_visible_horizons,
    )
    compatibility_union = _merge_unique_horizons(
        compatibility_symbol_visible_horizons,
        compatibility_strategy_visible_horizons,
    )

    return {
        "raw_symbol_preview_scope": _preview_scope_label(raw_symbol_visible_horizons),
        "raw_strategy_preview_scope": _preview_scope_label(raw_strategy_visible_horizons),
        "compatibility_symbol_preview_scope": _preview_scope_label(
            compatibility_symbol_visible_horizons
        ),
        "compatibility_strategy_preview_scope": _preview_scope_label(
            compatibility_strategy_visible_horizons
        ),
        "raw_preview_union_scope": _preview_scope_label(raw_union),
        "compatibility_preview_union_scope": _preview_scope_label(compatibility_union),
        "actual_joined_scope": _preview_scope_label(joined_visible_horizons),
        "raw_preview_union_horizons": raw_union,
        "compatibility_preview_union_horizons": compatibility_union,
        "actual_joined_horizons": joined_visible_horizons,
    }


def _preview_scope_label(horizons: list[str]) -> str:
    if len(horizons) >= 2:
        return "multi_horizon"
    if len(horizons) == 1:
        return "single_horizon"
    return "empty"


def _resolve_edge_candidate_visibility_reason(
    *,
    base_reason: str,
    symbol_preview: dict[str, Any],
    strategy_preview: dict[str, Any],
    compatibility_symbol_preview: dict[str, Any],
    compatibility_strategy_preview: dict[str, Any],
    joined_visible_horizons: list[str],
) -> str:
    diagnostics = _build_visibility_diagnostics(
        symbol_preview=symbol_preview,
        strategy_preview=strategy_preview,
        compatibility_symbol_preview=compatibility_symbol_preview,
        compatibility_strategy_preview=compatibility_strategy_preview,
        joined_visible_horizons=joined_visible_horizons,
    )

    actual_joined_scope = str(diagnostics.get("actual_joined_scope") or "empty")
    raw_preview_union_scope = str(diagnostics.get("raw_preview_union_scope") or "empty")
    compatibility_preview_union_scope = str(
        diagnostics.get("compatibility_preview_union_scope") or "empty"
    )

    reasons = [base_reason]

    if actual_joined_scope == "multi_horizon":
        reasons.append("joined_selected_multi_horizon")
    elif actual_joined_scope == "single_horizon":
        if compatibility_preview_union_scope == "multi_horizon":
            reasons.append("joined_selected_but_compatibility_preview_broader_than_joined")
        elif raw_preview_union_scope == "multi_horizon":
            reasons.append("joined_selected_but_raw_preview_broader_than_joined")
        else:
            reasons.append("joined_selected_single_horizon")
    else:
        reasons.append("joined_not_selected")

    deduped: list[str] = []
    for reason in reasons:
        if reason and reason not in deduped:
            deduped.append(reason)

    return "+".join(deduped)


def _evaluate_joined_edge_candidate_horizon(
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if horizon not in STRATEGY_HORIZON_COMPATIBILITY.get(strategy, set()):
        compatibility_detail = _build_strategy_horizon_compatibility_detail(
            strategy=strategy,
            horizon=horizon,
        )
        return {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "strategy_horizon_compatible": False,
            "status": "rejected",
            "rejection_reason": "strategy_horizon_incompatible",
            "rejection_reasons": ["strategy_horizon_incompatible"],
            "sample_gate": "not_applicable",
            "quality_gate": "not_applicable",
            "candidate_strength": "incompatible",
            "candidate_strength_diagnostics": None,
            "metrics": {},
            "aggregate_score": None,
            "visibility_reason": "strategy_horizon_incompatible",
            "chosen_metric_summary": "strategy_horizon_incompatible",
            "strategy_horizon_compatibility_detail": compatibility_detail,
        }

    metrics = _build_joined_group_metrics(rows, horizon)
    sample_count = _to_float(metrics.get("sample_count"))
    labeled_count = _to_float(metrics.get("labeled_count"))
    coverage_pct = _to_float(metrics.get("coverage_pct"))
    median_future_return_pct = _to_float(metrics.get("median_future_return_pct"))
    positive_rate_pct = _to_float(metrics.get("positive_rate_pct"))
    robustness_label, robustness_value = _select_robustness_signal(metrics)

    insufficiency_reason = _joined_metrics_insufficient_reason(
        sample_count=sample_count,
        labeled_count=labeled_count,
        median_future_return_pct=median_future_return_pct,
    )
    if insufficiency_reason is not None:
        return {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "strategy_horizon_compatible": True,
            "status": "rejected",
            "rejection_reason": insufficiency_reason,
            "rejection_reasons": [insufficiency_reason],
            "sample_gate": "failed",
            "quality_gate": "failed",
            "candidate_strength": "insufficient_data",
            "candidate_strength_diagnostics": None,
            "metrics": metrics,
            "aggregate_score": None,
            "visibility_reason": insufficiency_reason,
            "chosen_metric_summary": insufficiency_reason,
        }

    absolute_gate_failure_reasons = _absolute_minimum_gate_failure_reasons(
        sample_count=sample_count,
        labeled_count=labeled_count,
        coverage_pct=coverage_pct,
        median_future_return_pct=median_future_return_pct,
    )
    if absolute_gate_failure_reasons:
        return {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "strategy_horizon_compatible": True,
            "status": "rejected",
            "rejection_reason": "failed_absolute_minimum_gate",
            "rejection_reasons": ["failed_absolute_minimum_gate", *absolute_gate_failure_reasons],
            "sample_gate": "failed",
            "quality_gate": "failed",
            "candidate_strength": "insufficient_data",
            "candidate_strength_diagnostics": None,
            "metrics": metrics,
            "aggregate_score": None,
            "visibility_reason": "failed_absolute_minimum_gate",
            "chosen_metric_summary": "failed_absolute_minimum_gate",
        }

    diagnostics = _score_candidate_strength_diagnostics(
        sample_count=sample_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_value=robustness_value,
    )
    candidate_strength = str(diagnostics["final_classification"])
    aggregate_score = _to_float(diagnostics.get("aggregate_score"))

    if candidate_strength in {"moderate", "strong"}:
        return {
            "symbol": symbol,
            "strategy": strategy,
            "horizon": horizon,
            "strategy_horizon_compatible": True,
            "status": "selected",
            "rejection_reason": None,
            "rejection_reasons": [],
            "sample_gate": "passed",
            "quality_gate": "passed",
            "candidate_strength": candidate_strength,
            "candidate_strength_diagnostics": diagnostics,
            "metrics": metrics,
            "aggregate_score": aggregate_score,
            "visibility_reason": "passed_sample_and_quality_gate",
            "chosen_metric_summary": _build_candidate_metric_summary(
                sample_count=int(sample_count or 0),
                median_future_return_pct=median_future_return_pct or 0.0,
                positive_rate_pct=positive_rate_pct,
                robustness_label=robustness_label,
                robustness_value=robustness_value,
                diagnostics=diagnostics,
            ),
        }

    classification_reason = str(
        diagnostics.get("classification_reason") or "candidate_strength_weak"
    )
    rejection_reasons = ["candidate_strength_weak"]
    if classification_reason not in rejection_reasons:
        rejection_reasons.append(classification_reason)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "strategy_horizon_compatible": True,
        "status": "rejected",
        "rejection_reason": "candidate_strength_weak",
        "rejection_reasons": rejection_reasons,
        "sample_gate": "passed",
        "quality_gate": "failed",
        "candidate_strength": candidate_strength,
        "candidate_strength_diagnostics": diagnostics,
        "metrics": metrics,
        "aggregate_score": aggregate_score,
        "visibility_reason": "candidate_strength_weak",
        "chosen_metric_summary": _build_candidate_metric_summary(
            sample_count=int(sample_count or 0),
            median_future_return_pct=median_future_return_pct or 0.0,
            positive_rate_pct=positive_rate_pct,
            robustness_label=robustness_label,
            robustness_value=robustness_value,
            diagnostics=diagnostics,
        ),
    }


def _joined_metrics_insufficient_reason(
    *,
    sample_count: float | None,
    labeled_count: float | None,
    median_future_return_pct: float | None,
) -> str | None:
    if sample_count is None or sample_count <= 0:
        return "sample_count_zero"
    if labeled_count is None or labeled_count <= 0:
        return "no_labeled_rows_for_horizon"
    if median_future_return_pct is None:
        return "missing_median_future_return"
    return None


def _absolute_minimum_gate_failure_reasons(
    *,
    sample_count: float | None,
    labeled_count: float | None,
    coverage_pct: float | None,
    median_future_return_pct: float | None,
) -> list[str]:
    reasons: list[str] = []

    if sample_count is None or sample_count < MIN_EDGE_CANDIDATE_SAMPLE_COUNT:
        reasons.append("sample_count_below_absolute_floor")

    has_label_support = (
        (labeled_count is not None and labeled_count > 0)
        or (coverage_pct is not None and coverage_pct > 0)
    )
    if not has_label_support:
        reasons.append("no_label_support_for_absolute_minimum_gate")

    if median_future_return_pct is None:
        reasons.append("missing_median_future_return")
    elif median_future_return_pct <= 0:
        reasons.append("median_future_return_non_positive")

    return reasons


def _build_joined_candidate_row_from_evaluation(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    metrics = _coerce_dict(evaluation.get("metrics"))
    diagnostics = _coerce_dict(evaluation.get("candidate_strength_diagnostics"))
    sample_count = _to_float(metrics.get("sample_count"))
    labeled_count = _to_float(metrics.get("labeled_count"))
    coverage_pct = _to_float(metrics.get("coverage_pct"))
    median_future_return_pct = _to_float(metrics.get("median_future_return_pct"))
    positive_rate_pct = _to_float(metrics.get("positive_rate_pct"))
    robustness_label, robustness_value = _select_robustness_signal(metrics)

    major_deficit_breakdown = diagnostics.get("major_deficit_breakdown")
    supporting_major_deficit_count = (
        len(major_deficit_breakdown.get("supporting", []))
        if isinstance(major_deficit_breakdown, dict)
        and isinstance(major_deficit_breakdown.get("supporting"), list)
        else 0
    )

    return {
        "symbol": evaluation.get("symbol"),
        "strategy": evaluation.get("strategy"),
        "horizon": evaluation.get("horizon"),
        "selected_candidate_strength": evaluation.get("candidate_strength"),
        "selected_stability_label": None,
        "source_preference": None,
        "edge_stability_score": None,
        "drift_direction": None,
        "score_delta": None,
        "selected_visible_horizons": [str(evaluation.get("horizon"))],
        "sample_count": int(sample_count or 0),
        "labeled_count": int(labeled_count or 0),
        "coverage_pct": coverage_pct,
        "median_future_return_pct": median_future_return_pct,
        "avg_future_return_pct": _to_float(metrics.get("avg_future_return_pct")),
        "positive_rate_pct": positive_rate_pct,
        "up_rate_pct": _to_float(metrics.get("up_rate_pct")),
        "down_rate_pct": _to_float(metrics.get("down_rate_pct")),
        "flat_rate_pct": _to_float(metrics.get("flat_rate_pct")),
        "robustness_signal": robustness_label,
        "robustness_signal_pct": robustness_value,
        "aggregate_score": _to_float(evaluation.get("aggregate_score")),
        "supporting_major_deficit_count": supporting_major_deficit_count,
        "visibility_reason": str(
            evaluation.get("visibility_reason") or "passed_sample_and_quality_gate"
        ),
        "chosen_metric_summary": str(
            evaluation.get("chosen_metric_summary") or "passed_sample_and_quality_gate"
        ),
    }


def _diagnostic_category_from_evaluation(evaluation: dict[str, Any]) -> str:
    rejection_reason = str(evaluation.get("rejection_reason") or "")
    candidate_strength = str(evaluation.get("candidate_strength") or "")

    if rejection_reason == "strategy_horizon_incompatible":
        return "incompatibility"
    if rejection_reason in {
        "sample_count_zero",
        "no_labeled_rows_for_horizon",
        "missing_median_future_return",
        "failed_absolute_minimum_gate",
    }:
        return "insufficient_data"
    if candidate_strength == "weak" or rejection_reason == "candidate_strength_weak":
        return "quality_rejected"
    return "other_rejection"


def _build_joined_candidate_diagnostic_row_from_evaluation(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    metrics = _coerce_dict(evaluation.get("metrics"))
    diagnostics = _coerce_dict(evaluation.get("candidate_strength_diagnostics"))
    compatibility_detail = _coerce_dict(
        evaluation.get("strategy_horizon_compatibility_detail")
    )
    sample_count = _to_float(metrics.get("sample_count"))
    labeled_count = _to_float(metrics.get("labeled_count"))
    coverage_pct = _to_float(metrics.get("coverage_pct"))
    median_future_return_pct = _to_float(metrics.get("median_future_return_pct"))
    positive_rate_pct = _to_float(metrics.get("positive_rate_pct"))
    robustness_label, robustness_value = _select_robustness_signal(metrics)
    strategy = str(evaluation.get("strategy") or "")

    return {
        "symbol": evaluation.get("symbol"),
        "strategy": evaluation.get("strategy"),
        "horizon": evaluation.get("horizon"),
        "status": evaluation.get("status"),
        "diagnostic_category": _diagnostic_category_from_evaluation(evaluation),
        "strategy_horizon_compatible": bool(
            evaluation.get("strategy_horizon_compatible")
        ),
        "rejection_reason": evaluation.get("rejection_reason"),
        "rejection_reasons": evaluation.get("rejection_reasons", []),
        "sample_gate": evaluation.get("sample_gate"),
        "quality_gate": evaluation.get("quality_gate"),
        "candidate_strength": evaluation.get("candidate_strength"),
        "classification_reason": diagnostics.get("classification_reason"),
        "aggregate_score": _to_float(evaluation.get("aggregate_score")),
        "chosen_metric_summary": evaluation.get("chosen_metric_summary"),
        "visibility_reason": evaluation.get("visibility_reason"),
        "sample_count": int(sample_count or 0),
        "labeled_count": int(labeled_count or 0),
        "coverage_pct": coverage_pct,
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": robustness_label,
        "robustness_signal_pct": robustness_value,
        "strategy_configured_horizons": sorted(
            STRATEGY_HORIZON_COMPATIBILITY.get(strategy, set())
        ),
        "analyzer_supported_horizons": list(HORIZONS),
        "analyzer_compatible_horizons": _sorted_compatible_horizons(strategy),
        "strategy_horizon_compatibility_detail": compatibility_detail or None,
    }


def _build_edge_candidate_empty_reason_summary(
    *,
    rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    identity_horizon_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection_reason_counts: dict[str, int] = {}
    diagnostic_category_counts: dict[str, int] = {}
    strategies_without_analyzer_compatible_horizons: set[str] = set()
    identities_blocked_only_by_incompatibility: list[str] = []

    for row in diagnostic_rows:
        reason = str(row.get("rejection_reason") or "unknown")
        rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1

        diagnostic_category = str(row.get("diagnostic_category") or "other_rejection")
        diagnostic_category_counts[diagnostic_category] = (
            diagnostic_category_counts.get(diagnostic_category, 0) + 1
        )

        if reason != "strategy_horizon_incompatible":
            continue

        strategy = _normalize_text(row.get("strategy"))
        analyzer_compatible_horizons = row.get("analyzer_compatible_horizons")
        if (
            strategy is not None
            and isinstance(analyzer_compatible_horizons, list)
            and not analyzer_compatible_horizons
        ):
            strategies_without_analyzer_compatible_horizons.add(strategy)

    identities_with_eligible_rows = 0
    identities_without_eligible_rows = 0

    for entry in identity_horizon_evaluations:
        actual_joined_eligible_horizons = (
            _normalize_horizon_list(entry.get("actual_joined_eligible_horizons")) or []
        )
        if actual_joined_eligible_horizons:
            identities_with_eligible_rows += 1
            continue

        identities_without_eligible_rows += 1
        horizon_evaluations = _coerce_dict(entry.get("horizon_evaluations"))
        rejection_reasons = [
            _normalize_text(
                _coerce_dict(horizon_evaluations.get(horizon)).get("rejection_reason")
            )
            for horizon in HORIZONS
        ]
        filtered_rejection_reasons = [reason for reason in rejection_reasons if reason]
        if filtered_rejection_reasons and all(
            reason == "strategy_horizon_incompatible"
            for reason in filtered_rejection_reasons
        ):
            identity_key = _normalize_text(entry.get("identity_key"))
            if identity_key is not None:
                identities_blocked_only_by_incompatibility.append(identity_key)

    dominant_rejection_reason = None
    if rejection_reason_counts:
        dominant_rejection_reason = min(
            rejection_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    dominant_diagnostic_category = None
    if diagnostic_category_counts:
        dominant_diagnostic_category = min(
            diagnostic_category_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    has_only_incompatibility_rejections = bool(diagnostic_rows) and all(
        str(row.get("rejection_reason") or "") == "strategy_horizon_incompatible"
        for row in diagnostic_rows
    )
    has_only_weak_or_insufficient_candidates = bool(diagnostic_rows) and all(
        str(row.get("diagnostic_category") or "")
        in {"quality_rejected", "insufficient_data"}
        for row in diagnostic_rows
    )

    if rows:
        empty_state_category = "has_eligible_rows"
        note = (
            "Engine-facing eligible rows remain strict: only joined moderate/strong "
            "selections are emitted in rows. Rejected joined candidates are exposed "
            "separately in diagnostic_rows."
        )
    elif diagnostic_rows:
        if has_only_incompatibility_rejections:
            empty_state_category = "only_incompatibility_rejections"
        elif has_only_weak_or_insufficient_candidates:
            empty_state_category = "only_weak_or_insufficient_candidates"
        else:
            empty_state_category = "mixed_rejections_without_eligible_rows"

        note_parts = [
            "No engine-facing eligible joined candidate rows were produced.",
            f"Joined candidates were evaluated, but all were rejected; dominant rejection_reason is {dominant_rejection_reason or 'unknown'}.",
        ]
        if strategies_without_analyzer_compatible_horizons:
            note_parts.append(
                "Strategies with no overlap against analyzer horizons "
                f"{list(HORIZONS)} cannot participate in this analyzer path: "
                f"{sorted(strategies_without_analyzer_compatible_horizons)}."
            )
        note = " ".join(note_parts)
    else:
        empty_state_category = "no_joined_candidates_evaluated"
        note = "No joined candidates were evaluated."

    return {
        "has_eligible_rows": bool(rows),
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostic_rejection_reason_counts": rejection_reason_counts,
        "diagnostic_category_counts": diagnostic_category_counts,
        "dominant_rejection_reason": dominant_rejection_reason,
        "dominant_diagnostic_category": dominant_diagnostic_category,
        "identity_count": len(identity_horizon_evaluations),
        "identities_with_eligible_rows": identities_with_eligible_rows,
        "identities_without_eligible_rows": identities_without_eligible_rows,
        "identities_blocked_only_by_incompatibility": (
            identities_blocked_only_by_incompatibility
        ),
        "strategies_without_analyzer_compatible_horizons": sorted(
            strategies_without_analyzer_compatible_horizons
        ),
        "empty_state_category": empty_state_category,
        "has_only_incompatibility_rejections": has_only_incompatibility_rejections,
        "has_only_weak_or_insufficient_candidates": (
            has_only_weak_or_insufficient_candidates
        ),
        "note": note,
    }


def _build_joined_group_metrics(
    rows: list[dict[str, Any]],
    horizon: str,
) -> dict[str, Any]:
    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"
    sample_count = len(rows)

    label_counts = {"up": 0, "down": 0, "flat": 0}
    returns: list[float] = []
    bias_known = 0
    bias_match = 0
    signal_known = 0
    signal_match = 0

    for row in rows:
        label = _joined_normalize_label(row.get(label_key))
        if label not in {"up", "down", "flat"}:
            continue

        label_counts[label] += 1

        future_return = _to_float(row.get(return_key))
        if future_return is not None:
            returns.append(future_return)

        bias_direction = _joined_bias_to_direction(row.get("bias"))
        if bias_direction in {"up", "down", "flat"}:
            bias_known += 1
            if bias_direction == label:
                bias_match += 1

        signal_direction = _joined_signal_to_direction(_joined_extract_signal(row))
        if signal_direction in {"up", "down", "flat"}:
            signal_known += 1
            if signal_direction == label:
                signal_match += 1

    labeled_count = sum(label_counts.values())
    positive_rate_pct = None
    if labeled_count > 0:
        positive_rate_pct = max(
            _joined_pct(label_counts["up"], labeled_count) or 0.0,
            _joined_pct(label_counts["down"], labeled_count) or 0.0,
        )

    return {
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": _joined_pct(labeled_count, sample_count),
        "up_rate_pct": _joined_pct(label_counts["up"], labeled_count),
        "down_rate_pct": _joined_pct(label_counts["down"], labeled_count),
        "flat_rate_pct": _joined_pct(label_counts["flat"], labeled_count),
        "positive_rate_pct": positive_rate_pct,
        "bias_match_rate_pct": _joined_pct(bias_match, bias_known),
        "signal_match_rate_pct": _joined_pct(signal_match, signal_known),
        "avg_future_return_pct": round(sum(returns) / len(returns), 6) if returns else None,
        "median_future_return_pct": _median_or_none(returns),
    }


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return round(float(ordered[middle]), 6)
    return round(float((ordered[middle - 1] + ordered[middle]) / 2.0), 6)


def _joined_symbol(row: dict[str, Any]) -> str | None:
    value = row.get("symbol")
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _joined_strategy(row: dict[str, Any]) -> str | None:
    value = row.get("selected_strategy")
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text not in STRATEGY_HORIZON_COMPATIBILITY:
        return None
    return text


def _joined_extract_signal(row: dict[str, Any]) -> Any:
    for key in ("rule_signal", "execution_signal", "execution_action"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _joined_normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"up", "down", "flat"}:
        return text
    return "unknown"


def _joined_bias_to_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("bullish", "long", "buy", "up"):
        return "up"
    if text in ("bearish", "short", "sell", "down"):
        return "down"
    if text in ("neutral", "hold", "flat", "no_trade"):
        return "flat"
    return "unknown"


def _joined_signal_to_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("long", "buy", "up"):
        return "up"
    if text in ("short", "sell", "down"):
        return "down"
    if text in ("hold", "neutral", "flat", "no_trade"):
        return "flat"
    return "unknown"


def _joined_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100.0, 2)


def _build_edge_stability_preview(
    edge_candidates_preview: dict[str, Any],
) -> dict[str, Any]:
    by_horizon = edge_candidates_preview.get("by_horizon", {}) or {}

    return {
        "strategy": _build_stability_entry(by_horizon, "top_strategy"),
        "symbol": _build_stability_entry(by_horizon, "top_symbol"),
        "alignment_state": _build_stability_entry(by_horizon, "top_alignment_state"),
    }


def _build_stability_entry(
    by_horizon: dict[str, Any],
    candidate_key: str,
) -> dict[str, Any]:
    visible_candidates: dict[str, list[str]] = {}

    for horizon in HORIZONS:
        horizon_data = by_horizon.get(horizon, {}) or {}
        candidate = horizon_data.get(candidate_key)
        if not isinstance(candidate, dict):
            continue
        if candidate.get("candidate_strength") == "insufficient_data":
            continue

        group = str(candidate.get("group", "insufficient_data"))
        if group == "insufficient_data":
            continue

        visible_candidates.setdefault(group, []).append(horizon)

    if not visible_candidates:
        return {
            "group": None,
            "visible_horizons": [],
            "stability_label": "insufficient_data",
            "stability_score": 0,
            "visibility_reason": "no_visible_candidates",
        }

    if len(visible_candidates) == 1:
        group, horizons = next(iter(visible_candidates.items()))
        horizons = list(horizons)

        if len(horizons) >= 2:
            return {
                "group": group,
                "visible_horizons": horizons,
                "stability_label": "multi_horizon_confirmed",
                "stability_score": 2,
                "visibility_reason": "repeated_visible_candidate_across_horizons",
            }

        return {
            "group": group,
            "visible_horizons": horizons,
            "stability_label": "single_horizon_only",
            "stability_score": 1,
            "visibility_reason": "visible_in_one_horizon_only",
        }

    ranked_candidates = sorted(
        visible_candidates.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    top_group, top_horizons = ranked_candidates[0]
    top_horizons = list(top_horizons)

    second_group, second_horizons = ranked_candidates[1]
    second_horizons = list(second_horizons)

    if len(top_horizons) >= 2 and len(top_horizons) > len(second_horizons):
        return {
            "group": top_group,
            "visible_horizons": top_horizons,
            "stability_label": "multi_horizon_confirmed",
            "stability_score": 2,
            "visibility_reason": "dominant_visible_candidate_across_multiple_horizons",
        }

    return {
        "group": top_group,
        "visible_horizons": top_horizons,
        "stability_label": "unstable",
        "stability_score": 1,
        "visibility_reason": "multiple_visible_candidates_without_convergence",
    }


def _max_candidate_strength(strengths: list[str]) -> str:
    if not strengths:
        return "insufficient_data"
    if "strong" in strengths:
        return "strong"
    if "moderate" in strengths:
        return "moderate"
    if "weak" in strengths:
        return "weak"
    return "insufficient_data"


def _resolve_summary_generated_at(metrics: dict[str, Any]) -> str:
    generated_at = str(metrics.get("generated_at", "")).strip()
    if generated_at:
        return generated_at
    return datetime.now(UTC).isoformat()


def _build_markdown(metrics: dict[str, Any]) -> str:
    overview = metrics.get("dataset_overview", {}) or {}
    top_highlights = metrics.get("top_highlights", {}) or {}
    edge_candidates_preview = metrics.get("edge_candidates_preview", {}) or {}
    edge_candidate_rows = metrics.get("edge_candidate_rows", {}) or {}
    edge_stability_preview = metrics.get("edge_stability_preview", {}) or {}
    horizons = metrics.get("horizon_summary", {}) or {}
    by_symbol = metrics.get("by_symbol", {}) or {}
    by_strategy = metrics.get("by_strategy", {}) or {}
    strategy_lab = metrics.get("strategy_lab", {}) or {}
    schema_validation = metrics.get("schema_validation", {}) or {}

    lines: list[str] = []
    lines.append("# Research Summary")
    lines.append("")
    lines.append(f"Generated at: {_resolve_summary_generated_at(metrics)}")
    lines.append("")

    lines.extend(_markdown_top_highlights(top_highlights))
    lines.extend(_markdown_edge_candidates_preview(edge_candidates_preview))
    lines.extend(_markdown_edge_candidate_rows(edge_candidate_rows))
    lines.extend(_markdown_identity_horizon_evaluations(edge_candidate_rows))
    lines.extend(_markdown_edge_stability_preview(edge_stability_preview))

    lines.append("## Schema Validation")
    lines.append("")
    lines.append(f"- input_path: {schema_validation.get('input_path', 'unknown')}")
    lines.append(
        f"- effective_input_path: {schema_validation.get('effective_input_path', 'unknown')}"
    )
    lines.append(
        f"- materialized_effective_input: {schema_validation.get('materialized_effective_input', False)}"
    )
    lines.append(f"- rotation_aware: {schema_validation.get('rotation_aware', False)}")
    lines.append(f"- source_file_count: {schema_validation.get('source_file_count', 0)}")
    lines.append(
        f"- requested_latest_window_hours: {schema_validation.get('requested_latest_window_hours', 'n/a')}"
    )
    lines.append(
        f"- requested_latest_max_rows: {schema_validation.get('requested_latest_max_rows', 'n/a')}"
    )
    lines.append(f"- max_age_hours: {schema_validation.get('max_age_hours', 'n/a')}")
    lines.append(f"- max_rows: {schema_validation.get('max_rows', 'n/a')}")
    lines.append(f"- raw_record_count: {schema_validation.get('raw_record_count', 0)}")
    lines.append(
        f"- windowed_record_count: {schema_validation.get('windowed_record_count', 0)}"
    )
    lines.append(f"- total_records: {schema_validation.get('total_records', 0)}")
    lines.append(f"- valid_records: {schema_validation.get('valid_records', 0)}")
    lines.append(f"- invalid_records: {schema_validation.get('invalid_records', 0)}")
    lines.append(f"- error_count: {schema_validation.get('error_count', 0)}")
    lines.append(f"- warning_count: {schema_validation.get('warning_count', 0)}")
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- total_records: {overview.get('total_records', 0)}")
    lines.append(
        f"- records_with_any_future_label: {overview.get('records_with_any_future_label', 0)}"
    )
    lines.append(
        "- label_coverage_any_horizon_pct: "
        f"{_fmt_metric(overview.get('label_coverage_any_horizon_pct'))}"
    )

    date_range = overview.get("date_range", {}) or {}
    lines.append(f"- date_range.start: {date_range.get('start', 'unknown')}")
    lines.append(f"- date_range.end: {date_range.get('end', 'unknown')}")
    lines.append("")

    lines.extend(
        _markdown_distribution(
            "symbols_distribution",
            overview.get("symbols_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "selected_strategies_distribution",
            overview.get("selected_strategies_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "bias_distribution",
            overview.get("bias_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "ai_execution_distribution",
            overview.get("ai_execution_distribution", {}),
        )
    )
    lines.extend(
        _markdown_distribution(
            "alignment_distribution",
            overview.get("alignment_distribution", {}),
        )
    )

    lines.append("## Horizon Summary")
    lines.append("")
    for horizon in HORIZONS:
        lines.extend(
            _markdown_horizon_block(
                horizon,
                horizons.get(horizon, {}) or {},
                heading_level=3,
            )
        )

    lines.append("## By Symbol")
    lines.append("")
    if by_symbol:
        for symbol, data in by_symbol.items():
            lines.append(f"### {symbol}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                "- records_with_any_future_label: "
                f"{data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                "- label_coverage_any_horizon_pct: "
                f"{_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No symbol groups available.")
        lines.append("")

    lines.append("## By Strategy")
    lines.append("")
    if by_strategy:
        for strategy, data in by_strategy.items():
            lines.append(f"### {strategy}")
            lines.append(f"- total_records: {data.get('total_records', 0)}")
            lines.append(
                "- records_with_any_future_label: "
                f"{data.get('records_with_any_future_label', 0)}"
            )
            lines.append(
                "- label_coverage_any_horizon_pct: "
                f"{_fmt_metric(data.get('label_coverage_any_horizon_pct'))}"
            )
            lines.append("")

            group_horizons = data.get("horizon_summary", {}) or {}
            for horizon in HORIZONS:
                lines.extend(
                    _markdown_horizon_block(
                        horizon,
                        group_horizons.get(horizon, {}) or {},
                        heading_level=4,
                    )
                )
    else:
        lines.append("No strategy groups available.")
        lines.append("")

    lines.extend(_markdown_strategy_lab_block(strategy_lab))

    return "\n".join(lines).rstrip() + "\n"


def _markdown_top_highlights(top_highlights: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    by_horizon = top_highlights.get("by_horizon", {}) or {}

    lines.append("## Top Highlights")
    lines.append("")
    lines.append(
        f"- invalid_record_count: {top_highlights.get('invalid_record_count', 0)}"
    )
    lines.append(
        "- strategy_lab_dataset_rows: "
        f"{top_highlights.get('strategy_lab_dataset_rows', 0)}"
    )
    lines.append("")

    for horizon in HORIZONS:
        horizon_data = by_horizon.get(horizon, {}) or {}
        lines.append(f"### {horizon}")
        lines.append(f"- top_symbol: {horizon_data.get('top_symbol', 'n/a')}")
        lines.append(f"- top_strategy: {horizon_data.get('top_strategy', 'n/a')}")
        lines.append(
            f"- best_alignment_state: {horizon_data.get('best_alignment_state', 'n/a')}"
        )
        lines.append(
            "- best_ai_execution_state: "
            f"{horizon_data.get('best_ai_execution_state', 'n/a')}"
        )
        lines.append("")

    return lines


def _markdown_edge_candidates_preview(
    edge_candidates_preview: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    by_horizon = edge_candidates_preview.get("by_horizon", {}) or {}

    lines.append("## Edge Candidate Preview")
    lines.append("")
    lines.append(
        "- minimum_sample_count: "
        f"{edge_candidates_preview.get('minimum_sample_count', MIN_EDGE_CANDIDATE_SAMPLE_COUNT)}"
    )
    lines.append("")

    for horizon in HORIZONS:
        horizon_data = by_horizon.get(horizon, {}) or {}
        lines.append(f"### {horizon}")
        lines.append(
            f"- sample_gate: {horizon_data.get('sample_gate', 'insufficient_data')}"
        )
        lines.append(f"- quality_gate: {horizon_data.get('quality_gate', 'failed')}")
        lines.append(
            f"- candidate_strength: {horizon_data.get('candidate_strength', 'insufficient_data')}"
        )
        lines.append(
            f"- visibility_reason: {horizon_data.get('visibility_reason', 'failed_absolute_minimum_gate')}"
        )
        lines.append(
            f"- top_strategy: {_format_edge_candidate_markdown(horizon_data.get('top_strategy'))}"
        )
        lines.append(
            f"- top_symbol: {_format_edge_candidate_markdown(horizon_data.get('top_symbol'))}"
        )
        lines.append(
            "- top_alignment_state: "
            f"{_format_edge_candidate_markdown(horizon_data.get('top_alignment_state'))}"
        )
        lines.append("")

    return lines


def _markdown_edge_candidate_rows(edge_candidate_rows: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    rows = (
        edge_candidate_rows.get("rows", [])
        if isinstance(edge_candidate_rows.get("rows"), list)
        else []
    )
    diagnostic_rows = (
        edge_candidate_rows.get("diagnostic_rows", [])
        if isinstance(edge_candidate_rows.get("diagnostic_rows"), list)
        else []
    )
    empty_reason_summary = _coerce_dict(edge_candidate_rows.get("empty_reason_summary"))

    lines.append("## Edge Candidate Rows")
    lines.append("")
    lines.append(f"- row_count: {edge_candidate_rows.get('row_count', 0)}")
    lines.append(
        f"- diagnostic_row_count: {edge_candidate_rows.get('diagnostic_row_count', 0)}"
    )
    lines.append(f"- dropped_row_count: {edge_candidate_rows.get('dropped_row_count', 0)}")
    lines.append(
        "- diagnostic_rejection_reason_counts: "
        f"{empty_reason_summary.get('diagnostic_rejection_reason_counts', {})}"
    )
    lines.append(
        "- diagnostic_category_counts: "
        f"{empty_reason_summary.get('diagnostic_category_counts', {})}"
    )
    lines.append(
        f"- dominant_rejection_reason: {empty_reason_summary.get('dominant_rejection_reason', 'n/a')}"
    )
    lines.append(
        f"- dominant_diagnostic_category: {empty_reason_summary.get('dominant_diagnostic_category', 'n/a')}"
    )
    lines.append(
        f"- empty_state_category: {empty_reason_summary.get('empty_state_category', 'n/a')}"
    )
    lines.append(
        "- strategies_without_analyzer_compatible_horizons: "
        f"{empty_reason_summary.get('strategies_without_analyzer_compatible_horizons', [])}"
    )
    lines.append(f"- note: {empty_reason_summary.get('note', 'n/a')}")
    lines.append("")

    if not rows:
        lines.append("No engine-facing eligible joined candidate rows available.")
        if diagnostic_rows:
            lines.append("")
            lines.append("Diagnostic rejected joined candidates:")
            for row in diagnostic_rows[:12]:
                lines.append(
                    "- "
                    f"{row.get('symbol', 'n/a')} / {row.get('strategy', 'n/a')} / {row.get('horizon', 'n/a')} "
                    f"(category={row.get('diagnostic_category', 'n/a')}; "
                    f"reason={row.get('rejection_reason', 'unknown')}; "
                    f"candidate_strength={row.get('candidate_strength', 'n/a')}; "
                    f"sample_gate={row.get('sample_gate', 'n/a')}; "
                    f"quality_gate={row.get('quality_gate', 'n/a')}; "
                    f"analyzer_compatible_horizons={row.get('analyzer_compatible_horizons', [])})"
                )
        lines.append("")
        return lines

    for row in rows[:12]:
        lines.append(
            "- "
            f"{row.get('symbol', 'n/a')} / {row.get('strategy', 'n/a')} / {row.get('horizon', 'n/a')} "
            f"(strength={row.get('selected_candidate_strength', 'n/a')}; "
            f"actual_joined_eligible_horizons={row.get('actual_joined_eligible_horizons', [])}; "
            f"raw_symbol_preview={row.get('preview_symbol_visible_horizons', [])}; "
            f"raw_strategy_preview={row.get('preview_strategy_visible_horizons', [])}; "
            f"compatibility_symbol_preview={row.get('compatibility_preview_symbol_visible_horizons', [])}; "
            f"compatibility_strategy_preview={row.get('compatibility_preview_strategy_visible_horizons', [])}; "
            f"visibility_reason={row.get('visibility_reason', 'n/a')})"
        )
    if diagnostic_rows:
        lines.append("")
        lines.append("Rejected joined candidate diagnostics:")
        for row in diagnostic_rows[:12]:
            lines.append(
                "- "
                f"{row.get('symbol', 'n/a')} / {row.get('strategy', 'n/a')} / {row.get('horizon', 'n/a')} "
                f"(category={row.get('diagnostic_category', 'n/a')}; "
                f"reason={row.get('rejection_reason', 'unknown')}; "
                f"candidate_strength={row.get('candidate_strength', 'n/a')}; "
                f"classification_reason={row.get('classification_reason', 'n/a')}; "
                f"analyzer_compatible_horizons={row.get('analyzer_compatible_horizons', [])})"
            )
    lines.append("")
    return lines


def _markdown_identity_horizon_evaluations(
    edge_candidate_rows: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    identity_horizon_evaluations = (
        edge_candidate_rows.get("identity_horizon_evaluations", [])
        if isinstance(edge_candidate_rows.get("identity_horizon_evaluations"), list)
        else []
    )

    lines.append("## Identity Horizon Diagnostics")
    lines.append("")

    if not identity_horizon_evaluations:
        lines.append("No identity horizon diagnostics available.")
        lines.append("")
        return lines

    for entry in identity_horizon_evaluations[:12]:
        raw_preview_visibility = _coerce_dict(entry.get("raw_preview_visibility"))
        compatibility_filtered_preview_visibility = _coerce_dict(
            entry.get("compatibility_filtered_preview_visibility")
        )
        horizon_evaluations = _coerce_dict(entry.get("horizon_evaluations"))

        lines.append(
            f"### {entry.get('symbol', 'n/a')} / {entry.get('strategy', 'n/a')}"
        )
        lines.append(
            "- raw_preview_visibility: "
            f"symbol={_format_preview_visibility_entry(raw_preview_visibility.get('symbol'))}; "
            f"strategy={_format_preview_visibility_entry(raw_preview_visibility.get('strategy'))}; "
            f"union={raw_preview_visibility.get('raw_category_union_horizons', [])}; "
            f"overlap={raw_preview_visibility.get('raw_category_overlap_horizons', [])}; "
            f"quality_passed_overlap={raw_preview_visibility.get('raw_category_quality_passed_overlap_horizons', [])}"
        )
        lines.append(
            "- compatibility_filtered_preview_visibility: "
            f"symbol={_format_preview_visibility_entry(compatibility_filtered_preview_visibility.get('symbol'))}; "
            f"strategy={_format_preview_visibility_entry(compatibility_filtered_preview_visibility.get('strategy'))}; "
            f"union={compatibility_filtered_preview_visibility.get('compatibility_filtered_category_union_horizons', [])}; "
            f"overlap={compatibility_filtered_preview_visibility.get('compatibility_filtered_category_overlap_horizons', [])}; "
            f"quality_passed_overlap={compatibility_filtered_preview_visibility.get('compatibility_filtered_quality_passed_overlap_horizons', [])}; "
            f"compatible_horizons={compatibility_filtered_preview_visibility.get('strategy_compatible_horizons', [])}"
        )
        lines.append(
            "- actual_joined_eligible_horizons: "
            f"{entry.get('actual_joined_eligible_horizons', [])} "
            f"({entry.get('actual_joined_stability_label', 'single_horizon_only')})"
        )
        lines.append("- horizon_evaluations:")
        for horizon in HORIZONS:
            evaluation = _coerce_dict(horizon_evaluations.get(horizon))
            lines.append(
                f"  - {horizon}: status={evaluation.get('status', 'rejected')}, "
                f"reason={evaluation.get('rejection_reason') or 'selected'}, "
                f"reasons={evaluation.get('rejection_reasons', [])}, "
                f"candidate_strength={evaluation.get('candidate_strength', 'n/a')}, "
                f"sample_gate={evaluation.get('sample_gate', 'n/a')}, "
                f"quality_gate={evaluation.get('quality_gate', 'n/a')}"
            )
        lines.append("")

    return lines


def _format_preview_visibility_entry(entry: Any) -> str:
    if not isinstance(entry, dict):
        return "none"
    return (
        f"visible_horizons={entry.get('visible_horizons', [])}, "
        f"quality_passed_horizons={entry.get('quality_passed_horizons', [])}, "
        f"horizon_strengths={entry.get('horizon_strengths', {})}, "
        f"stability_label={entry.get('stability_label')}, "
        f"visibility_reason={entry.get('visibility_reason')}"
    )


def _format_edge_candidate_markdown(candidate: Any) -> str:
    if not isinstance(candidate, dict):
        return "insufficient_data"
    if candidate.get("candidate_strength") == "insufficient_data":
        return (
            f"insufficient_data ({candidate.get('visibility_reason', 'failed_absolute_minimum_gate')})"
        )
    return (
        f"{candidate.get('group', 'n/a')} "
        f"({candidate.get('candidate_strength', 'insufficient_data')}; "
        f"sample_gate={candidate.get('sample_gate', 'failed')}; "
        f"quality_gate={candidate.get('quality_gate', 'failed')}; "
        f"{candidate.get('chosen_metric_summary', 'n/a')})"
    )


def _markdown_edge_stability_preview(
    edge_stability_preview: dict[str, Any],
) -> list[str]:
    lines: list[str] = []

    lines.append("## Edge Stability Preview")
    lines.append("")

    if not edge_stability_preview:
        lines.append("No stability preview available.")
        lines.append("")
        return lines

    for label, entry in (
        ("strategy", edge_stability_preview.get("strategy", {})),
        ("symbol", edge_stability_preview.get("symbol", {})),
        ("alignment_state", edge_stability_preview.get("alignment_state", {})),
    ):
        lines.append(f"### {label}")
        lines.append(f"- group: {entry.get('group', 'insufficient_data')}")
        lines.append(
            f"- visible_horizons: {', '.join(entry.get('visible_horizons', [])) or 'none'}"
        )
        lines.append(
            f"- stability_label: {entry.get('stability_label', 'insufficient_data')}"
        )
        lines.append(f"- stability_score: {entry.get('stability_score', 0)}")
        lines.append(
            f"- visibility_reason: {entry.get('visibility_reason', 'no_visible_candidates')}"
        )
        lines.append("")

    return lines


def _markdown_strategy_lab_block(strategy_lab: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    lines.append("## Strategy Research Lab")
    lines.append("")
    lines.append(f"- dataset_rows: {strategy_lab.get('dataset_rows', 0)}")
    lines.append("")

    performance = strategy_lab.get("performance", {}) or {}
    ranking = strategy_lab.get("ranking", {}) or {}
    edge = strategy_lab.get("edge", {}) or {}
    segment = strategy_lab.get("segment", {}) or {}

    lines.append("### Performance")
    lines.append("")
    if performance:
        for horizon in HORIZONS:
            report = performance.get(horizon, {}) or {}
            lines.append(f"#### {horizon}")
            lines.append(f"- sample_count: {report.get('sample_count', 0)}")
            lines.append(f"- labeled_count: {report.get('labeled_count', 0)}")
            lines.append(f"- coverage_pct: {_fmt_metric(report.get('coverage_pct'))}")
            lines.append(
                f"- signal_match_rate: {_fmt_metric(report.get('signal_match_rate'))}"
            )
            lines.append(
                f"- bias_match_rate: {_fmt_metric(report.get('bias_match_rate'))}"
            )
            lines.append(
                "- avg_future_return_pct: "
                f"{_fmt_metric(report.get('avg_future_return_pct'))}"
            )
            lines.append(
                "- median_future_return_pct: "
                f"{_fmt_metric(report.get('median_future_return_pct'))}"
            )
            lines.append("")
    else:
        lines.append("No performance report available.")
        lines.append("")

    lines.append("### Ranking Highlights")
    lines.append("")
    if ranking:
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_rank = ranking.get(horizon, {}) or {}

            top_symbol = _extract_top_ranked_group(horizon_rank.get("by_symbol"))
            top_strategy = _extract_top_ranked_group(horizon_rank.get("by_strategy"))
            top_alignment = _extract_top_ranked_group(
                horizon_rank.get("by_alignment_state")
            )
            top_ai_execution = _extract_top_ranked_group(
                horizon_rank.get("by_ai_execution_state")
            )

            lines.append(f"- top_symbol: {top_symbol}")
            lines.append(f"- top_strategy: {top_strategy}")
            lines.append(f"- top_alignment_state: {top_alignment}")
            lines.append(f"- top_ai_execution_state: {top_ai_execution}")
            lines.append("")
    else:
        lines.append("No ranking report available.")
        lines.append("")

    lines.append("### Edge Highlights")
    lines.append("")
    if edge:
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_edge = edge.get(horizon, {}) or {}

            lines.append(
                f"- symbol_edges: {_count_edge_findings(horizon_edge.get('by_symbol'))}"
            )
            lines.append(
                f"- strategy_edges: {_count_edge_findings(horizon_edge.get('by_strategy'))}"
            )
            lines.append(
                "- alignment_state_edges: "
                f"{_count_edge_findings(horizon_edge.get('by_alignment_state'))}"
            )
            lines.append(
                "- ai_execution_state_edges: "
                f"{_count_edge_findings(horizon_edge.get('by_ai_execution_state'))}"
            )
            lines.append("")
    else:
        lines.append("No edge report available.")
        lines.append("")

    lines.append("### Segment Highlights")
    lines.append("")
    if segment:
        segment_reports = segment.get("reports", {}) or {}
        for horizon in HORIZONS:
            lines.append(f"#### {horizon}")
            horizon_segments = segment_reports.get(horizon, {}) or {}

            hour_count = (
                (horizon_segments.get("hour_of_day", {}) or {}).get(
                    "qualified_segments", 0
                )
            )
            day_count = (
                (horizon_segments.get("day_of_week", {}) or {}).get(
                    "qualified_segments", 0
                )
            )
            week_part_count = (
                (horizon_segments.get("week_part", {}) or {}).get(
                    "qualified_segments", 0
                )
            )

            lines.append(f"- hour_of_day qualified_segments: {hour_count}")
            lines.append(f"- day_of_week qualified_segments: {day_count}")
            lines.append(f"- week_part qualified_segments: {week_part_count}")
            lines.append("")
    else:
        lines.append("No segment report available.")
        lines.append("")

    return lines


def _extract_top_ranked_group(report: Any) -> str:
    items = _extract_ranked_groups(report)
    if not items:
        return "n/a"

    first = items[0]
    return str(first.get("group", "n/a"))


def _count_edge_findings(report: Any) -> int:
    if not isinstance(report, dict):
        return 0

    findings = report.get("edge_findings", [])
    if not isinstance(findings, list):
        return 0

    return len(findings)


def _markdown_horizon_block(
    horizon: str,
    horizon_data: dict[str, Any],
    heading_level: int,
) -> list[str]:
    heading = "#" * heading_level
    lines: list[str] = []

    lines.append(f"{heading} {horizon}")
    lines.append("")
    lines.append(f"- labeled_records: {horizon_data.get('labeled_records', 0)}")

    label_dist = horizon_data.get("label_distribution", {}) or {}
    lines.append(
        "- label_distribution: "
        f"up={label_dist.get('up', 0)}, "
        f"down={label_dist.get('down', 0)}, "
        f"flat={label_dist.get('flat', 0)}"
    )

    lines.append(
        f"- avg_future_return_pct: {_fmt_metric(horizon_data.get('avg_future_return_pct'))}"
    )
    lines.append(
        "- median_future_return_pct: "
        f"{_fmt_metric(horizon_data.get('median_future_return_pct'))}"
    )
    lines.append(
        f"- positive_rate_pct: {_fmt_metric(horizon_data.get('positive_rate_pct'))}"
    )
    lines.append(
        f"- negative_rate_pct: {_fmt_metric(horizon_data.get('negative_rate_pct'))}"
    )
    lines.append(f"- flat_rate_pct: {_fmt_metric(horizon_data.get('flat_rate_pct'))}")

    bias_vs_label = horizon_data.get("bias_vs_label", {}) or {}
    lines.append(
        "- bias_vs_label: "
        f"match={bias_vs_label.get('match', 0)}, "
        f"mismatch={bias_vs_label.get('mismatch', 0)}, "
        f"unknown={bias_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(bias_vs_label.get('match_rate_pct'))}"
    )

    signal_vs_label = horizon_data.get("signal_vs_label", {}) or {}
    lines.append(
        "- signal_vs_label: "
        f"match={signal_vs_label.get('match', 0)}, "
        f"mismatch={signal_vs_label.get('mismatch', 0)}, "
        f"unknown={signal_vs_label.get('unknown', 0)}, "
        f"match_rate_pct={_fmt_metric(signal_vs_label.get('match_rate_pct'))}"
    )

    lines.append("")
    return lines


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"

    return str(value)


def _markdown_distribution(title: str, values: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    lines.append(f"### {title}")
    lines.append("")

    if not values:
        lines.append("No data.")
        lines.append("")
        return lines

    for key, count in values.items():
        lines.append(f"- {key}: {count}")

    lines.append("")
    return lines


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "research_reports" / "latest"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze trade_analysis.jsonl research metrics"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to write summary.json and summary.md",
    )
    parser.add_argument(
        "--latest-window-hours",
        type=int,
        default=DEFAULT_LATEST_WINDOW_HOURS,
        help="Recent window in hours for rotation-aware latest analysis",
    )
    parser.add_argument(
        "--latest-max-rows",
        type=int,
        default=DEFAULT_LATEST_MAX_ROWS,
        help="Maximum number of rows to keep for rotation-aware latest analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reporter = CronHealthReporter("research_analyzer")

    try:
        metrics = run_research_analyzer(
            input_path=args.input,
            output_dir=args.output_dir,
            latest_window_hours=args.latest_window_hours,
            latest_max_rows=args.latest_max_rows,
        )

        print(
            f"Records analyzed: {metrics.get('dataset_overview', {}).get('total_records', 0)}"
        )
        print(
            f"Strategy lab dataset rows: {metrics.get('strategy_lab', {}).get('dataset_rows', 0)}"
        )
        print(f"Summary JSON: {(args.output_dir / 'summary.json').resolve()}")
        print(f"Summary MD: {(args.output_dir / 'summary.md').resolve()}")

        try:
            notifier = ResearchNotifier()
            notifier.send_latest_summary()
        except Exception:
            LOGGER.exception("ResearchNotifier failed to send summary.")

        reporter.success(
            {
                "records_analyzed": metrics.get("dataset_overview", {}).get(
                    "total_records", 0
                ),
                "valid_records": metrics.get("schema_validation", {}).get(
                    "valid_records", 0
                ),
                "invalid_records": metrics.get("schema_validation", {}).get(
                    "invalid_records", 0
                ),
                "strategy_lab_dataset_rows": metrics.get("strategy_lab", {}).get(
                    "dataset_rows", 0
                ),
            }
        )

    except Exception as exc:
        reporter.failure(
            error=exc,
            message="Research analyzer failed",
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
