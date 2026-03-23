from __future__ import annotations

import argparse
import json
import logging
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

POSITIVE_RATE_MINIMUM_FLOOR_PCT = 48.0

STRENGTH_COMPONENT_WEIGHTS = {
    "sample_count": 0.30,
    "median_future_return_pct": 0.30,
    "positive_rate_pct": 0.25,
    "robustness_value": 0.15,
}

STRENGTH_SCORING_MODEL = "banded_weighted_v5_1"

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
STRONG_MIN_AGGREGATE_SCORE = 85.0

CRITICAL_MAJOR_DEFICITS = {
    "sample_count_below_emerging_moderate",
}
SUPPORTING_MAJOR_DEFICITS = {
    "median_return_below_emerging_moderate",
    "positive_rate_below_emerging_moderate",
    "robustness_below_emerging_moderate",
}


def load_jsonl_records(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSONL records, keeping only schema-valid objects for analysis."""
    raw_records, source_metadata = load_jsonl_records_with_metadata(input_path)

    records: list[dict[str, Any]] = []
    validation_summary = {
        "input_path": str(input_path),
        "rotation_aware": source_metadata.get("rotation_aware", False),
        "source_files": source_metadata.get("source_files", []),
        "source_file_count": source_metadata.get("source_file_count", 0),
        "source_row_counts": source_metadata.get("source_row_counts", {}),
        "max_age_hours": source_metadata.get("max_age_hours"),
        "max_rows": source_metadata.get("max_rows"),
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

    summary_json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_build_markdown(metrics), encoding="utf-8")

    return summary_json_path, summary_md_path


def run_research_analyzer(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run full analyzer flow: load valid records, calculate metrics, and write reports."""
    records, validation_summary = load_jsonl_records(input_path)

    base_metrics = calculate_research_metrics(records)

    if records and int(validation_summary.get("invalid_records", 0)) == 0:
        strategy_lab_metrics = _build_strategy_lab_metrics(input_path)
    else:
        strategy_lab_metrics = _empty_strategy_lab_metrics()

    final_metrics = dict(base_metrics)
    final_metrics["schema_validation"] = validation_summary
    final_metrics["strategy_lab"] = strategy_lab_metrics
    final_metrics["top_highlights"] = _build_top_highlights(
        schema_validation=validation_summary,
        strategy_lab=strategy_lab_metrics,
    )
    final_metrics["edge_candidates_preview"] = _build_edge_candidates_preview(
        strategy_lab=strategy_lab_metrics,
    )
    final_metrics["edge_stability_preview"] = _build_edge_stability_preview(
        edge_candidates_preview=final_metrics["edge_candidates_preview"],
    )

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
                "strong_min_aggregate_score": STRONG_MIN_AGGREGATE_SCORE,
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

    if len(supporting_major_deficits) >= 3:
        return False, "three_or_more_supporting_deficits_present"

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
        minimum_threshold=EDGE_EARLY_MODERATE_ROBUSTNESS_PCT,
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


def _build_markdown(metrics: dict[str, Any]) -> str:
    overview = metrics.get("dataset_overview", {}) or {}
    top_highlights = metrics.get("top_highlights", {}) or {}
    edge_candidates_preview = metrics.get("edge_candidates_preview", {}) or {}
    edge_stability_preview = metrics.get("edge_stability_preview", {}) or {}
    horizons = metrics.get("horizon_summary", {}) or {}
    by_symbol = metrics.get("by_symbol", {}) or {}
    by_strategy = metrics.get("by_strategy", {}) or {}
    strategy_lab = metrics.get("strategy_lab", {}) or {}
    schema_validation = metrics.get("schema_validation", {}) or {}

    lines: list[str] = []
    lines.append("# Research Summary")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
    lines.append("")

    lines.extend(_markdown_top_highlights(top_highlights))
    lines.extend(_markdown_edge_candidates_preview(edge_candidates_preview))
    lines.extend(_markdown_edge_stability_preview(edge_stability_preview))

    lines.append("## Schema Validation")
    lines.append("")
    lines.append(f"- input_path: {schema_validation.get('input_path', 'unknown')}")
    lines.append(f"- rotation_aware: {schema_validation.get('rotation_aware', False)}")
    lines.append(f"- source_file_count: {schema_validation.get('source_file_count', 0)}")
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


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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