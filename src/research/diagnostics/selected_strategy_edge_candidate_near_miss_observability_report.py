from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.research_analyzer import (
    MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
    run_research_analyzer,
)
from src.research.strategy_lab.dataset_builder import (
    DEFAULT_LATEST_MAX_ROWS,
    DEFAULT_LATEST_WINDOW_HOURS,
    build_dataset,
    load_jsonl_records_with_metadata,
)

REPORT_TYPE = "selected_strategy_edge_candidate_near_miss_observability_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Near-Miss Observability Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

DIAGNOSTIC_USABLE_SAMPLE_COUNT = MIN_EDGE_CANDIDATE_SAMPLE_COUNT
DIAGNOSTIC_NON_TRIVIAL_POSITIVE_RATE_PCT = 40.0
DIAGNOSTIC_PAPER_REVIEW_MAX_DEFICIT_COUNT = 2

HARD_INCOMPATIBILITY_REASONS = {"strategy_horizon_incompatible"}
SAMPLE_LIMITED_REASONS = {
    "failed_absolute_minimum_gate",
    "sample_count_below_absolute_floor",
    "sample_count_zero",
    "no_labeled_rows_for_horizon",
    "missing_median_future_return",
    "no_label_support_for_absolute_minimum_gate",
}
NEGATIVE_RETURN_REASONS = {
    "median_future_return_non_positive",
    "median_future_return_pct_non_positive",
}
QUALITY_REASON_KEYWORDS = {
    "aggregate": "aggregate_too_low",
    "positive_rate": "positive_rate_too_low",
    "robustness": "robustness_too_low",
}
NEAR_MISS_CLASSES = {
    "quality_weak_near_miss",
    "aggregate_deficit_near_miss",
    "robustness_deficit_near_miss",
    "positive_rate_deficit_near_miss",
    "mixed_near_miss",
}


@dataclass(frozen=True)
class SupportWindowConfiguration:
    latest_window_hours: int
    latest_max_rows: int
    label: str | None = None

    @property
    def display_name(self) -> str:
        if isinstance(self.label, str) and self.label.strip():
            return self.label.strip()
        return f"{self.latest_window_hours}h/{self.latest_max_rows}"

    @property
    def slug(self) -> str:
        base = f"{self.latest_window_hours}h_{self.latest_max_rows}"
        if self.label:
            label = "".join(
                char.lower() if char.isalnum() else "_"
                for char in self.label.strip()
            ).strip("_")
            if label:
                return f"{label}_{base}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "latest_window_hours": self.latest_window_hours,
            "latest_max_rows": self.latest_max_rows,
            "slug": self.slug,
        }


DEFAULT_CONFIGURATIONS: tuple[SupportWindowConfiguration, ...] = (
    SupportWindowConfiguration(
        DEFAULT_LATEST_WINDOW_HOURS,
        DEFAULT_LATEST_MAX_ROWS,
        "36h/2500",
    ),
    SupportWindowConfiguration(72, 2500),
    SupportWindowConfiguration(144, 5000),
    SupportWindowConfiguration(336, 10000),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only near-miss observability report from rejected "
            "edge candidate diagnostic rows."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Support window in WINDOW_HOURS/MAX_ROWS form. Repeatable.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_selected_strategy_edge_candidate_near_miss_observability_report(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        configurations=parse_configuration_values(args.config),
        write_report_copies=args.write_latest_copy,
    )
    final = _safe_dict(result["report"].get("final_assessment"))
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "classification": final.get("classification"),
                "configuration_count": len(
                    result["report"].get("configuration_summaries", [])
                ),
                "paper_only_review_candidates_present": final.get(
                    "paper_only_review_candidates_present"
                ),
                "quality_near_misses_present": final.get(
                    "quality_near_misses_present"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    if not values:
        return list(DEFAULT_CONFIGURATIONS)

    parsed: list[SupportWindowConfiguration] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "/" not in item:
            raise ValueError(
                f"Invalid configuration '{value}'. Expected WINDOW_HOURS/MAX_ROWS."
            )
        hours_raw, rows_raw = item.split("/", 1)
        parsed.append(
            SupportWindowConfiguration(
                latest_window_hours=int(hours_raw),
                latest_max_rows=int(rows_raw),
            )
        )

    return parsed or list(DEFAULT_CONFIGURATIONS)


def run_selected_strategy_edge_candidate_near_miss_observability_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[SupportWindowConfiguration] | None = None,
    write_report_copies: bool = False,
) -> dict[str, Any]:
    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )

    written_paths: dict[str, str] = {}
    if write_report_copies:
        written_paths = write_report_files(report, output_dir)

    return {
        "input_path": report["inputs"]["input_path"],
        "output_dir": report["inputs"]["output_dir"],
        "written_paths": written_paths,
        "report": report,
        "markdown": render_markdown(report),
    }


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[SupportWindowConfiguration] | None = None,
) -> dict[str, Any]:
    resolved_input = resolve_path(input_path)
    resolved_output = resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)

    summaries: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        analyzer_output_dir = (
            resolved_output / f"_{REPORT_TYPE}" / "analyzer_runs" / configuration.slug
        )
        raw_records, source_metadata = load_jsonl_records_with_metadata(
            path=resolved_input,
            max_age_hours=configuration.latest_window_hours,
            max_rows=configuration.latest_max_rows,
        )
        labelable_dataset = build_dataset(
            path=resolved_input,
            max_age_hours=configuration.latest_window_hours,
            max_rows=configuration.latest_max_rows,
        )
        analyzer_metrics = run_research_analyzer(
            resolved_input,
            analyzer_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
        )

        summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                analyzer_output_dir=analyzer_output_dir,
                analyzer_metrics=analyzer_metrics,
                source_metadata=source_metadata,
                raw_record_count=len(raw_records),
                labelable_count=len(labelable_dataset),
            )
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
        },
        "diagnostic_only": True,
        "diagnostic_ranking_note": (
            "near_miss_rank_score is report-ordering only and is not used by "
            "candidate quality, mapper, engine, or execution logic."
        ),
        "diagnostic_thresholds": diagnostic_thresholds(),
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Rejected rows come from edge_candidate_rows.diagnostic_rows.",
            "No gate threshold, mapper, engine, execution, or production latest-window behavior is changed.",
            "Observational buckets are conservative and do not promote diagnostic rows into production candidates.",
        ],
    }


def diagnostic_thresholds() -> dict[str, Any]:
    return {
        "usable_sample_count": DIAGNOSTIC_USABLE_SAMPLE_COUNT,
        "non_trivial_positive_rate_pct": DIAGNOSTIC_NON_TRIVIAL_POSITIVE_RATE_PCT,
        "paper_review_max_deficit_count": DIAGNOSTIC_PAPER_REVIEW_MAX_DEFICIT_COUNT,
        "hard_incompatibility_reasons": sorted(HARD_INCOMPATIBILITY_REASONS),
        "sample_limited_reasons": sorted(SAMPLE_LIMITED_REASONS),
        "negative_return_reasons": sorted(NEGATIVE_RETURN_REASONS),
    }


def build_configuration_summary(
    *,
    configuration: SupportWindowConfiguration,
    input_path: Path | None = None,
    analyzer_output_dir: Path | None = None,
    analyzer_metrics: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
    raw_record_count: int | None = None,
    labelable_count: int | None = None,
) -> dict[str, Any]:
    edge_candidate_rows = _safe_dict(analyzer_metrics.get("edge_candidate_rows"))
    diagnostic_rows = [
        row for row in _safe_list(edge_candidate_rows.get("diagnostic_rows"))
        if isinstance(row, dict)
    ]
    near_miss_rows = [build_near_miss_row(row) for row in diagnostic_rows]
    near_miss_rows.sort(key=_near_miss_sort_key)

    class_counts = Counter(
        str(row["near_miss_classification"]) for row in near_miss_rows
    )
    bucket_counts = Counter(
        str(row["suggested_next_policy_bucket"]) for row in near_miss_rows
    )
    top_rows = near_miss_rows[:10]
    best_row = top_rows[0] if top_rows else None
    paper_ready = [
        row
        for row in near_miss_rows
        if row["suggested_next_policy_bucket"] == "paper_only_candidate_review"
    ]

    source = _safe_dict(source_metadata)
    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path) if input_path is not None else None,
        "analyzer_output_dir": (
            str(analyzer_output_dir) if analyzer_output_dir is not None else None
        ),
        "source_metadata": {
            "raw_record_count": _safe_int(
                source.get("raw_record_count", raw_record_count)
            ),
            "windowed_record_count": _safe_int(
                source.get("windowed_record_count", raw_record_count)
            ),
            "max_age_hours": source.get(
                "max_age_hours", configuration.latest_window_hours
            ),
            "max_rows": source.get("max_rows", configuration.latest_max_rows),
        },
        "labelable_count": _safe_int(labelable_count),
        "edge_candidate_rows": {
            "row_count": _safe_int(edge_candidate_rows.get("row_count")),
            "diagnostic_row_count": len(diagnostic_rows),
            "total_diagnostic_rows": len(diagnostic_rows),
            "empty_reason_summary": _safe_dict(
                edge_candidate_rows.get("empty_reason_summary")
            ),
        },
        "near_miss_count_by_class": dict(sorted(class_counts.items())),
        "suggested_next_policy_bucket_counts": dict(sorted(bucket_counts.items())),
        "top_near_miss_rows": top_rows,
        "best_near_miss_row": best_row,
        "paper_only_review_candidate_count": len(paper_ready),
        "any_row_suitable_for_paper_only_review": bool(paper_ready),
        "window_suggestion": classify_window_suggestion(near_miss_rows),
    }


def build_near_miss_row(row: dict[str, Any]) -> dict[str, Any]:
    rejection_reasons = normalize_string_list(row.get("rejection_reasons"))
    rejection_reason = _safe_text(row.get("rejection_reason"))
    if rejection_reason and rejection_reason not in rejection_reasons:
        rejection_reasons.insert(0, rejection_reason)

    candidate_strength = (
        _safe_text(row.get("candidate_strength"))
        or _safe_text(row.get("selected_candidate_strength"))
        or "unknown"
    )
    sample_count = _safe_int(row.get("sample_count"))
    labeled_count = _safe_int(row.get("labeled_count"))
    median_future_return_pct = _safe_float(row.get("median_future_return_pct"))
    positive_rate_pct = _safe_float(row.get("positive_rate_pct"))
    robustness_signal_pct = _safe_float(row.get("robustness_signal_pct"))
    aggregate_score = _safe_float(row.get("aggregate_score"))
    deficit_labels = extract_deficit_labels(row, rejection_reasons)
    classification = classify_near_miss_row(
        diagnostic_category=_safe_text(row.get("diagnostic_category")),
        rejection_reasons=rejection_reasons,
        candidate_strength=candidate_strength,
        sample_count=sample_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        aggregate_score=aggregate_score,
        deficit_labels=deficit_labels,
    )
    bucket = suggested_policy_bucket(
        classification=classification,
        sample_count=sample_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        aggregate_score=aggregate_score,
        deficit_count=len(deficit_labels),
    )
    score = near_miss_rank_score(
        classification=classification,
        rejection_reasons=rejection_reasons,
        candidate_strength=candidate_strength,
        sample_count=sample_count,
        labeled_count=labeled_count,
        median_future_return_pct=median_future_return_pct,
        positive_rate_pct=positive_rate_pct,
        robustness_signal_pct=robustness_signal_pct,
        aggregate_score=aggregate_score,
    )

    return {
        "symbol": row.get("symbol"),
        "strategy": row.get("strategy"),
        "horizon": row.get("horizon"),
        "diagnostic_category": row.get("diagnostic_category"),
        "rejection_reason": rejection_reason,
        "rejection_reasons": rejection_reasons,
        "near_miss_classification": classification,
        "near_miss_rank_score": score,
        "candidate_strength": candidate_strength,
        "sample_count": sample_count,
        "labeled_count": labeled_count,
        "coverage_pct": _safe_float(row.get("coverage_pct")),
        "median_future_return_pct": median_future_return_pct,
        "positive_rate_pct": positive_rate_pct,
        "robustness_signal": row.get("robustness_signal"),
        "robustness_signal_pct": robustness_signal_pct,
        "aggregate_score": aggregate_score,
        "supporting_major_deficit_count": _safe_optional_int(
            row.get("supporting_major_deficit_count")
        ),
        "diagnostic_deficit_labels": deficit_labels,
        "diagnostic_deficit_count": len(deficit_labels),
        "chosen_metric_summary": row.get("chosen_metric_summary"),
        "visibility_reason": visibility_reason(
            classification=classification,
            rejection_reasons=rejection_reasons,
            sample_count=sample_count,
            median_future_return_pct=median_future_return_pct,
            aggregate_score=aggregate_score,
        ),
        "source_visibility_reason": row.get("visibility_reason"),
        "suggested_next_policy_bucket": bucket,
    }


def classify_near_miss_row(
    *,
    diagnostic_category: str | None,
    rejection_reasons: Sequence[str],
    candidate_strength: str,
    sample_count: int,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    aggregate_score: float | None,
    deficit_labels: Sequence[str],
) -> str:
    reasons = set(rejection_reasons)
    if reasons & HARD_INCOMPATIBILITY_REASONS or diagnostic_category == "incompatibility":
        return "hard_blocked_incompatible"

    if reasons & NEGATIVE_RETURN_REASONS or (
        median_future_return_pct is not None and median_future_return_pct <= 0
    ):
        return "negative_return_blocked"

    if reasons & SAMPLE_LIMITED_REASONS or sample_count < DIAGNOSTIC_USABLE_SAMPLE_COUNT:
        return "insufficient_sample"

    usable_quality_metrics = (
        median_future_return_pct is not None
        and median_future_return_pct > 0
        and sample_count >= DIAGNOSTIC_USABLE_SAMPLE_COUNT
    )
    if not usable_quality_metrics:
        return "not_a_near_miss"

    has_candidate_strength_weak = (
        "candidate_strength_weak" in reasons or candidate_strength == "weak"
    )
    deficit_set = set(deficit_labels)

    if len(deficit_set) > 1 and has_candidate_strength_weak:
        return "mixed_near_miss"
    if "aggregate_too_low" in deficit_set:
        return "aggregate_deficit_near_miss"
    if "robustness_too_low" in deficit_set:
        return "robustness_deficit_near_miss"
    if "positive_rate_too_low" in deficit_set:
        return "positive_rate_deficit_near_miss"
    if has_candidate_strength_weak and aggregate_score is not None:
        return "quality_weak_near_miss"

    if positive_rate_pct is None or aggregate_score is None:
        return "not_a_near_miss"
    return "mixed_near_miss" if has_candidate_strength_weak else "not_a_near_miss"


def suggested_policy_bucket(
    *,
    classification: str,
    sample_count: int,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    aggregate_score: float | None,
    deficit_count: int,
) -> str:
    if classification == "hard_blocked_incompatible":
        return "hard_block"
    if classification == "negative_return_blocked":
        return "hard_block"
    if classification == "insufficient_sample":
        return "collect_more_data"
    if classification not in NEAR_MISS_CLASSES:
        return "hard_block"

    has_paper_review_shape = (
        median_future_return_pct is not None
        and median_future_return_pct > 0
        and sample_count >= DIAGNOSTIC_USABLE_SAMPLE_COUNT
        and positive_rate_pct is not None
        and positive_rate_pct >= DIAGNOSTIC_NON_TRIVIAL_POSITIVE_RATE_PCT
        and aggregate_score is not None
        and deficit_count <= DIAGNOSTIC_PAPER_REVIEW_MAX_DEFICIT_COUNT
    )
    if has_paper_review_shape:
        return "paper_only_candidate_review"
    if deficit_count <= DIAGNOSTIC_PAPER_REVIEW_MAX_DEFICIT_COUNT:
        return "human_review_candidate_review"
    return "observe_near_miss"


def extract_deficit_labels(
    row: dict[str, Any],
    rejection_reasons: Sequence[str],
) -> list[str]:
    labels: list[str] = []
    for reason in rejection_reasons:
        normalized = str(reason)
        for keyword, label in QUALITY_REASON_KEYWORDS.items():
            if keyword in normalized and label not in labels:
                labels.append(label)

    classification_reason = _safe_text(row.get("classification_reason"))
    if classification_reason:
        for keyword, label in QUALITY_REASON_KEYWORDS.items():
            if keyword in classification_reason and label not in labels:
                labels.append(label)

    explicit_count = _safe_optional_int(row.get("supporting_major_deficit_count"))
    if explicit_count is not None and explicit_count > len(labels):
        for index in range(explicit_count - len(labels)):
            labels.append(f"supporting_deficit_{index + 1}")

    return labels


def near_miss_rank_score(
    *,
    classification: str,
    rejection_reasons: Sequence[str],
    candidate_strength: str,
    sample_count: int,
    labeled_count: int,
    median_future_return_pct: float | None,
    positive_rate_pct: float | None,
    robustness_signal_pct: float | None,
    aggregate_score: float | None,
) -> float:
    score = 0.0
    if median_future_return_pct is not None and median_future_return_pct > 0:
        score += 100.0
        score += min(median_future_return_pct, 5.0) * 20.0
    elif median_future_return_pct is not None:
        score -= 50.0

    score += min(max(sample_count, 0), 250) * 0.12
    score += min(max(labeled_count, 0), 250) * 0.06
    if positive_rate_pct is not None:
        score += max(min(positive_rate_pct, 100.0), 0.0) * 0.35
    if robustness_signal_pct is not None:
        score += max(min(robustness_signal_pct, 100.0), 0.0) * 0.2
    if aggregate_score is not None:
        score += max(min(aggregate_score, 100.0), 0.0) * 0.55

    score -= len(list(rejection_reasons)) * 6.0
    if classification == "hard_blocked_incompatible":
        score -= 90.0
    elif classification == "negative_return_blocked":
        score -= 70.0
    elif classification == "insufficient_sample":
        score -= 25.0

    if candidate_strength == "weak":
        score += 10.0
    elif candidate_strength == "insufficient_data":
        score -= 10.0
    elif candidate_strength == "incompatible":
        score -= 30.0

    return round(score, 6)


def classify_window_suggestion(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "hard_no_candidate"
    buckets = Counter(str(row.get("suggested_next_policy_bucket")) for row in rows)
    classes = Counter(str(row.get("near_miss_classification")) for row in rows)
    if buckets.get("paper_only_candidate_review", 0) > 0:
        return "policy_split_review"
    if any(classes.get(name, 0) > 0 for name in NEAR_MISS_CLASSES):
        return "policy_split_review"
    if classes.get("insufficient_sample", 0) > 0 and sum(classes.values()) == classes.get(
        "insufficient_sample", 0
    ):
        return "collect_more_data"
    if classes.get("insufficient_sample", 0) > 0:
        return "collect_more_data"
    return "hard_no_candidate"


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "classification": "mixed_or_inconclusive",
            "summary": "No support-window configurations were evaluated.",
            "paper_only_review_candidates_present": False,
            "quality_near_misses_present": False,
        }

    class_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    total_diagnostic_rows = 0
    for summary in summaries:
        edge_rows = _safe_dict(summary.get("edge_candidate_rows"))
        total_diagnostic_rows += _safe_int(edge_rows.get("total_diagnostic_rows"))
        class_counts.update(
            {
                str(key): _safe_int(value)
                for key, value in _safe_dict(
                    summary.get("near_miss_count_by_class")
                ).items()
            }
        )
        bucket_counts.update(
            {
                str(key): _safe_int(value)
                for key, value in _safe_dict(
                    summary.get("suggested_next_policy_bucket_counts")
                ).items()
            }
        )
    window_suggestions = Counter(
        str(summary.get("window_suggestion")) for summary in summaries
    )

    paper_present = bucket_counts.get("paper_only_candidate_review", 0) > 0
    quality_present = any(class_counts.get(name, 0) > 0 for name in NEAR_MISS_CLASSES)
    sample_only = (
        bool(class_counts)
        and class_counts.get("insufficient_sample", 0) == sum(class_counts.values())
    )
    hard_only = bool(class_counts) and all(
        key in {"hard_blocked_incompatible", "negative_return_blocked", "not_a_near_miss"}
        for key in class_counts
    )

    if total_diagnostic_rows == 0:
        classification = "no_near_miss_candidates"
        summary = "No rejected diagnostic rows were available for near-miss observation."
    elif paper_present:
        classification = "paper_only_review_candidates_present"
        summary = (
            "At least one rejected diagnostic row has positive return, usable sample "
            "support, non-trivial positive rate, aggregate support, and only limited deficits."
        )
    elif quality_present:
        classification = "quality_near_misses_present_policy_split_review"
        summary = (
            "Quality-limited near-miss rows exist, but this report does not change "
            "candidate policy or production selection behavior."
        )
    elif sample_only:
        classification = "mostly_sample_limited_collect_more_data"
        summary = "Observed rejected rows are sample-limited; more data is required."
    elif hard_only:
        classification = "hard_blocked_no_policy_change_recommended"
        summary = "Observed rejected rows remain hard-blocked or non-near-miss."
    else:
        classification = "mixed_or_inconclusive"
        summary = "Rejected diagnostic rows contain mixed blockers without a clean conclusion."

    return {
        "classification": classification,
        "summary": summary,
        "paper_only_review_candidates_present": paper_present,
        "quality_near_misses_present": quality_present,
        "near_miss_class_counts": dict(sorted(class_counts.items())),
        "suggested_next_policy_bucket_counts": dict(sorted(bucket_counts.items())),
        "window_suggestion_counts": dict(sorted(window_suggestions.items())),
        "stop_rule_next_requirement": (
            "Stop here. A later stage must define an explicit candidate policy split contract."
            if quality_present or paper_present
            else "Stop here. More data is required before a policy split is justified."
        ),
    }


def visibility_reason(
    *,
    classification: str,
    rejection_reasons: Sequence[str],
    sample_count: int,
    median_future_return_pct: float | None,
    aggregate_score: float | None,
) -> str:
    if classification == "hard_blocked_incompatible":
        return "strategy-horizon incompatibility is a hard structural blocker."
    if classification == "negative_return_blocked":
        return "Median future return is non-positive; this is not paper-only ready."
    if classification == "insufficient_sample":
        return (
            "Sample support is below the diagnostic usable floor "
            f"({sample_count} < {DIAGNOSTIC_USABLE_SAMPLE_COUNT})."
        )
    if classification in NEAR_MISS_CLASSES:
        return (
            "Rejected row has positive median return and usable sample support, "
            f"but remains blocked by {list(rejection_reasons)} with aggregate_score={aggregate_score}."
        )
    if median_future_return_pct is None:
        return "Median future return is missing."
    return "Rejected row does not have enough usable evidence for near-miss treatment."


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output = resolve_path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output / REPORT_JSON_NAME
    md_path = resolved_output / REPORT_MD_NAME
    _write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown(report: dict[str, Any]) -> str:
    final = _safe_dict(report.get("final_assessment"))
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Final Assessment",
        f"- classification: {final.get('classification')}",
        f"- summary: {final.get('summary')}",
        f"- paper_only_review_candidates_present: {final.get('paper_only_review_candidates_present')}",
        f"- quality_near_misses_present: {final.get('quality_near_misses_present')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Diagnostic Thresholds",
        f"- {report.get('diagnostic_thresholds')}",
        "",
        "## Configuration Summaries",
    ]

    for summary in _safe_list(report.get("configuration_summaries")):
        item = _safe_dict(summary)
        configuration = _safe_dict(item.get("configuration"))
        edge_rows = _safe_dict(item.get("edge_candidate_rows"))
        best = _safe_dict(item.get("best_near_miss_row"))
        lines.extend(
            [
                "",
                f"### {configuration.get('display_name')}",
                f"- total_diagnostic_rows: {edge_rows.get('total_diagnostic_rows')}",
                f"- row_count: {edge_rows.get('row_count')}",
                f"- near_miss_count_by_class: {item.get('near_miss_count_by_class')}",
                f"- any_row_suitable_for_paper_only_review: {item.get('any_row_suitable_for_paper_only_review')}",
                f"- window_suggestion: {item.get('window_suggestion')}",
                f"- best_near_miss_row: {best}",
            ]
        )

    return "\n".join(lines) + "\n"


def _near_miss_sort_key(row: dict[str, Any]) -> tuple[float, str, str, str]:
    return (
        -float(row.get("near_miss_rank_score") or 0.0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _safe_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        if value is None:
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None
