from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.experimental_candidate_c_analyzer_runner import (
    DEFAULT_INPUT_PATH as DEFAULT_CANDIDATE_C_DATASET,
)
from src.research.experimental_candidate_c_analyzer_runner import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from src.research.experimental_candidate_c_topn_eligibility_failure_report import (
    build_restored_topn_candidate_inputs,
)
from src.research.experimental_candidate_c_topn_preservation_report import (
    DEFAULT_TOP_N_STRATEGIES,
    DEFAULT_TOP_N_SYMBOLS,
    _extract_ranked_groups,
    _normalize_strategy,
    _normalize_symbol,
    _safe_dict,
    _safe_float,
    _safe_list,
    _safe_text,
    _write_json,
)
from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    load_jsonl_records,
)
from src.research.experimental_candidate_intersection_utils import filter_candidate_c_records
from src.research.research_analyzer import (
    _passes_absolute_minimum_gate,
    _score_candidate_strength_diagnostics,
    _select_robustness_signal,
    run_research_analyzer,
)

DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_weak_strength_diagnosis_report.json"
DEFAULT_MD_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_weak_strength_diagnosis_report.md"
DEFAULT_ANALYZER_TRACE_DIR = DEFAULT_OUTPUT_DIR / "topn_weak_strength_trace"
REPRESENTATIVE_LIMIT = 5

# Must stay aligned with src/research/research_analyzer.py (banded_weighted_v3)
CRITICAL_MAJOR_DEFICIT_LABELS = {
    "sample_count_below_emerging_moderate",
    "median_return_below_emerging_moderate",
}
SUPPORTING_MAJOR_DEFICIT_LABELS = {
    "positive_rate_below_emerging_moderate",
    "robustness_below_emerging_moderate",
}

METRIC_KEYWORDS = {
    "sample": "sample_count",
    "median": "median_future_return_pct",
    "positive": "positive_rate_pct",
    "robustness": "robustness_value",
}


def _sorted_counter(counter: Counter[str], *, key_name: str) -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _visible_ranked_groups_with_diagnostics(
    summary: dict[str, Any],
    *,
    bucket: str,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    ranking = _safe_dict(_safe_dict(summary.get("strategy_lab")).get("ranking"))
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in ("15m", "1h", "4h"):
        visible_rows: list[dict[str, Any]] = []
        raw_rows = _extract_ranked_groups(_safe_dict(ranking.get(horizon)).get(bucket))
        for entry in raw_rows:
            metrics = _safe_dict(entry.get("metrics"))
            sample_count = _safe_float(metrics.get("sample_count"))
            labeled_count = _safe_float(metrics.get("labeled_count"))
            coverage_pct = _safe_float(metrics.get("coverage_pct"))
            median_future_return_pct = _safe_float(metrics.get("median_future_return_pct"))
            positive_rate_pct = _safe_float(
                metrics.get("positive_rate_pct", metrics.get("up_rate_pct"))
            )
            robustness_signal, robustness_value = _select_robustness_signal(metrics)

            sample_gate_passed = _passes_absolute_minimum_gate(
                sample_count=sample_count,
                labeled_count=labeled_count,
                coverage_pct=coverage_pct,
                median_future_return_pct=median_future_return_pct,
            )
            if not sample_gate_passed:
                continue

            diagnostics = _score_candidate_strength_diagnostics(
                sample_count=sample_count,
                median_future_return_pct=median_future_return_pct,
                positive_rate_pct=positive_rate_pct,
                robustness_value=robustness_value,
            )
            candidate_strength = _safe_text(diagnostics.get("final_classification")) or "weak"
            quality_gate = (
                "passed" if candidate_strength in {"moderate", "strong"} else "borderline"
            )

            visible_rows.append(
                {
                    "group": _safe_text(entry.get("group")) or "n/a",
                    "rank": entry.get("rank"),
                    "score": _safe_float(entry.get("score")) or 0.0,
                    "sample_count": int(sample_count or 0),
                    "labeled_count": labeled_count,
                    "coverage_pct": coverage_pct,
                    "median_future_return_pct": median_future_return_pct,
                    "positive_rate_pct": positive_rate_pct,
                    "robustness_signal": robustness_signal,
                    "robustness_value": robustness_value,
                    "candidate_strength": candidate_strength,
                    "quality_gate": quality_gate,
                    "sample_gate": "passed",
                    "candidate_strength_diagnostics": diagnostics,
                }
            )

        result[horizon] = visible_rows[: max(1, int(top_n))]

    return result


def _index_groups_by_horizon_and_name(
    groups: dict[str, list[dict[str, Any]]],
    *,
    category: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for horizon, rows in groups.items():
        for row in rows:
            group = _safe_text(row.get("group"))
            normalized_group = (
                _normalize_symbol(group) if category == "symbol" else _normalize_strategy(group)
            )
            if normalized_group is None:
                continue
            index[(horizon, normalized_group)] = row
    return index


def _major_deficit_breakdown(diagnostics: dict[str, Any]) -> dict[str, list[str]]:
    breakdown = _safe_dict(diagnostics.get("major_deficit_breakdown"))
    critical = [
        _safe_text(item)
        for item in _safe_list(breakdown.get("critical"))
        if _safe_text(item)
    ]
    supporting = [
        _safe_text(item)
        for item in _safe_list(breakdown.get("supporting"))
        if _safe_text(item)
    ]
    other = [
        _safe_text(item)
        for item in _safe_list(breakdown.get("other"))
        if _safe_text(item)
    ]

    if critical or supporting or other:
        supporting_extended = list(supporting)
        supporting_extended.extend(other)
        return {
            "critical": critical,
            "supporting": supporting_extended,
        }

    major_deficits = [
        _safe_text(item)
        for item in _safe_list(diagnostics.get("major_deficits"))
        if _safe_text(item)
    ]
    critical = [item for item in major_deficits if item in CRITICAL_MAJOR_DEFICIT_LABELS]
    supporting = [item for item in major_deficits if item in SUPPORTING_MAJOR_DEFICIT_LABELS]
    remainder = [
        item
        for item in major_deficits
        if item not in (CRITICAL_MAJOR_DEFICIT_LABELS | SUPPORTING_MAJOR_DEFICIT_LABELS)
    ]
    supporting.extend(remainder)
    return {
        "critical": critical,
        "supporting": supporting,
    }


def _metric_name_from_label(label: str) -> str:
    for keyword, metric_name in METRIC_KEYWORDS.items():
        if keyword in label:
            return metric_name
    return "unmapped_metric"


def _aggregate_score_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(value)
        for value in (
            _safe_float(_safe_dict(item.get("candidate_strength_diagnostics")).get("aggregate_score"))
            for item in candidates
        )
        if value is not None
    ]
    if not scores:
        return {
            "available_count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "bucket_counts": [],
        }

    bucket_counts = Counter()
    for score in scores:
        if score < 50.0:
            bucket_counts["below_50"] += 1
        elif score < 62.0:
            bucket_counts["50_to_61_99"] += 1
        elif score < 75.0:
            bucket_counts["62_to_74_99"] += 1
        elif score < 85.0:
            bucket_counts["75_to_84_99"] += 1
        else:
            bucket_counts["85_plus"] += 1

    return {
        "available_count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "avg": round(sum(scores) / len(scores), 6),
        "bucket_counts": _sorted_counter(bucket_counts, key_name="score_bucket"),
    }


def _recommended_next_change(
    *,
    weak_count: int,
    total_count: int,
    dominant_reason: str | None,
    dominant_metric: str | None,
    critical_major_deficit_total: int,
    hard_blocker_total: int,
) -> str:
    if weak_count == 0:
        return "none_required"
    if hard_blocker_total > 0:
        return "candidate strength formula redesign"
    if dominant_reason == "cleared_weighted_moderate_profile":
        return "weak-to-penalized engine rule redesign"
    if (
        dominant_metric in {"positive_rate_pct", "median_future_return_pct", "sample_count"}
        and critical_major_deficit_total == 0
    ):
        return "candidate strength threshold adjustment"
    if total_count > 0 and weak_count == total_count:
        return "candidate strength formula redesign"
    return "combined scoring + engine redesign"


def build_restored_topn_strength_inputs(
    *,
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    experiment_summary: dict[str, Any],
    top_n_symbols: int,
    top_n_strategies: int,
) -> dict[str, Any]:
    restored_inputs = build_restored_topn_candidate_inputs(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=experiment_summary,
        top_n_symbols=top_n_symbols,
        top_n_strategies=top_n_strategies,
    )
    restored_candidates = _safe_list(restored_inputs.get("restored_candidates"))

    topn_symbols = _visible_ranked_groups_with_diagnostics(
        experiment_summary,
        bucket="by_symbol",
        top_n=top_n_symbols,
    )
    topn_strategies = _visible_ranked_groups_with_diagnostics(
        experiment_summary,
        bucket="by_strategy",
        top_n=top_n_strategies,
    )
    symbol_index = _index_groups_by_horizon_and_name(topn_symbols, category="symbol")
    strategy_index = _index_groups_by_horizon_and_name(topn_strategies, category="strategy")

    candidate_strength_details: list[dict[str, Any]] = []
    for candidate in restored_candidates:
        horizon = _safe_text(candidate.get("horizon")) or "unknown"
        symbol = _normalize_symbol(candidate.get("symbol")) or "unknown"
        strategy = _normalize_strategy(candidate.get("strategy")) or "unknown"
        symbol_row = _safe_dict(symbol_index.get((horizon, symbol)))
        strategy_row = _safe_dict(strategy_index.get((horizon, strategy)))

        support_category = "symbol"
        support_row = symbol_row
        if float(strategy_row.get("score") or 0.0) > float(symbol_row.get("score") or 0.0):
            support_category = "strategy"
            support_row = strategy_row

        support_row = _safe_dict(support_row)
        diagnostics = _safe_dict(support_row.get("candidate_strength_diagnostics"))
        major_deficit_breakdown = _major_deficit_breakdown(diagnostics)

        candidate_strength_details.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "support_category": support_category,
                "support_group": support_row.get("group"),
                "sample_count": support_row.get("sample_count"),
                "median_future_return_pct": support_row.get("median_future_return_pct"),
                "positive_rate_pct": support_row.get("positive_rate_pct"),
                "robustness_signal": support_row.get("robustness_signal"),
                "robustness_value": support_row.get("robustness_value"),
                "candidate_strength": support_row.get("candidate_strength")
                or candidate.get("selected_candidate_strength"),
                "quality_gate": support_row.get("quality_gate"),
                "score": support_row.get("score"),
                "edge_stability_score": candidate.get("edge_stability_score"),
                "candidate_strength_diagnostics": diagnostics,
                "final_classification": _safe_text(diagnostics.get("final_classification"))
                or support_row.get("candidate_strength")
                or candidate.get("selected_candidate_strength"),
                "classification_reason": _safe_text(diagnostics.get("classification_reason"))
                or "unknown",
                "aggregate_score": _safe_float(diagnostics.get("aggregate_score")),
                "hard_blockers": [
                    _safe_text(item)
                    for item in _safe_list(diagnostics.get("hard_blockers"))
                    if _safe_text(item)
                ],
                "soft_penalties": [
                    _safe_text(item)
                    for item in _safe_list(diagnostics.get("soft_penalties"))
                    if _safe_text(item)
                ],
                "major_deficits": [
                    _safe_text(item)
                    for item in _safe_list(diagnostics.get("major_deficits"))
                    if _safe_text(item)
                ],
                "major_deficit_breakdown": major_deficit_breakdown,
                "component_scores": _safe_dict(diagnostics.get("component_scores")),
            }
        )

    visibility_context = {}
    topn_summary = _safe_dict(restored_inputs.get("topn_summary"))
    if topn_summary:
        visibility_context = {
            "baseline_top1_visible_count": _safe_dict(
                topn_summary.get("baseline_top1_summary")
            ).get("visible_c2_exclusive_identity_count", 0),
            "topn_visible_count": _safe_dict(
                topn_summary.get("experimental_topn_summary")
            ).get("visible_c2_exclusive_identity_count", 0),
            "newly_visible_identity_count": _safe_dict(
                topn_summary.get("coverage_gain_summary")
            ).get("newly_visible_identity_count", 0),
        }

    return {
        "restored_inputs": restored_inputs,
        "candidate_strength_details": candidate_strength_details,
        "visibility_context": visibility_context,
    }


def build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
    *,
    candidate_strength_details: list[dict[str, Any]],
    experimental_config: dict[str, Any] | None = None,
    visibility_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    experimental_config = _safe_dict(experimental_config)
    visibility_context = _safe_dict(visibility_context)

    classification_counts = Counter(
        _safe_text(item.get("final_classification"))
        or _safe_text(item.get("candidate_strength"))
        or "unknown"
        for item in candidate_strength_details
    )
    weak_candidates = [
        item
        for item in candidate_strength_details
        if (_safe_text(item.get("final_classification")) or "unknown") == "weak"
    ]

    classification_reason_counts: Counter[str] = Counter()
    major_deficit_counts: Counter[str] = Counter()
    critical_major_deficit_counts: Counter[str] = Counter()
    supporting_major_deficit_counts: Counter[str] = Counter()
    soft_penalty_counts: Counter[str] = Counter()
    hard_blocker_counts: Counter[str] = Counter()
    component_band_counters: dict[str, Counter[str]] = {}
    metric_deficit_counts: Counter[str] = Counter()

    for candidate in candidate_strength_details:
        diagnostics = _safe_dict(candidate.get("candidate_strength_diagnostics"))
        reason = (
            _safe_text(diagnostics.get("classification_reason"))
            or _safe_text(candidate.get("classification_reason"))
            or "unknown"
        )
        classification_reason_counts[reason] += 1

        for label in candidate.get("hard_blockers") or []:
            hard_blocker_counts[str(label)] += 1
            metric_deficit_counts[_metric_name_from_label(str(label))] += 1

        for label in candidate.get("soft_penalties") or []:
            soft_penalty_counts[str(label)] += 1
            metric_deficit_counts[_metric_name_from_label(str(label))] += 1

        for label in candidate.get("major_deficits") or []:
            major_deficit_counts[str(label)] += 1
            metric_deficit_counts[_metric_name_from_label(str(label))] += 1

        breakdown = _safe_dict(candidate.get("major_deficit_breakdown"))
        for label in breakdown.get("critical", []):
            critical_major_deficit_counts[str(label)] += 1
        for label in breakdown.get("supporting", []):
            supporting_major_deficit_counts[str(label)] += 1

        component_scores = _safe_dict(candidate.get("component_scores"))
        for metric_name, component in component_scores.items():
            band = _safe_text(_safe_dict(component).get("band")) or "unknown"
            component_band_counters.setdefault(metric_name, Counter())[band] += 1

    dominant_reason = None
    if classification_reason_counts:
        dominant_reason = sorted(
            classification_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    dominant_metric = None
    if metric_deficit_counts:
        dominant_metric = sorted(
            metric_deficit_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    weak_count = classification_counts.get("weak", 0)
    moderate_count = classification_counts.get("moderate", 0)
    strong_count = classification_counts.get("strong", 0)
    non_weak_count = moderate_count + strong_count

    recommended_next_change = _recommended_next_change(
        weak_count=weak_count,
        total_count=len(candidate_strength_details),
        dominant_reason=dominant_reason,
        dominant_metric=dominant_metric,
        critical_major_deficit_total=sum(critical_major_deficit_counts.values()),
        hard_blocker_total=sum(hard_blocker_counts.values()),
    )

    representative_weak_candidates = sorted(
        weak_candidates,
        key=lambda item: (
            len(item.get("hard_blockers") or []),
            len(item.get("major_deficits") or []),
            -(item.get("aggregate_score") or 0.0),
            str(item.get("symbol")),
            str(item.get("strategy")),
            str(item.get("horizon")),
        ),
        reverse=True,
    )[:REPRESENTATIVE_LIMIT]

    component_band_counts = {
        metric_name: _sorted_counter(counter, key_name="band")
        for metric_name, counter in sorted(component_band_counters.items(), key=lambda item: item[0])
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "experimental_candidate_c_topn_weak_strength_diagnosis_report",
        "architecture_note": {
            "summary": (
                "This report isolates restored top-N candidates and explains their strength "
                "classification using analyzer-provided candidate_strength_diagnostics instead "
                "of legacy inferred weak-driver labels."
            )
        },
        "experimental_config": experimental_config,
        "visibility_context": visibility_context,
        "restored_candidate_count": len(candidate_strength_details),
        "weak_count": weak_count,
        "moderate_count": moderate_count,
        "strong_count": strong_count,
        "non_weak_count": non_weak_count,
        "classification_reason_counts": _sorted_counter(
            classification_reason_counts,
            key_name="classification_reason",
        ),
        "hard_blocker_counts": _sorted_counter(
            hard_blocker_counts,
            key_name="hard_blocker",
        ),
        "major_deficit_counts": _sorted_counter(
            major_deficit_counts,
            key_name="major_deficit",
        ),
        "critical_major_deficit_counts": _sorted_counter(
            critical_major_deficit_counts,
            key_name="major_deficit",
        ),
        "supporting_major_deficit_counts": _sorted_counter(
            supporting_major_deficit_counts,
            key_name="major_deficit",
        ),
        "soft_penalty_counts": _sorted_counter(
            soft_penalty_counts,
            key_name="soft_penalty",
        ),
        "component_band_counts": component_band_counts,
        "aggregate_score_summary": _aggregate_score_summary(candidate_strength_details),
        "representative_weak_candidates": representative_weak_candidates,
        "root_assessment": {
            "dominant_classification_reason": dominant_reason,
            "dominant_metric_pattern": dominant_metric,
            "hard_blocker_count": sum(hard_blocker_counts.values()),
            "critical_major_deficit_count": sum(critical_major_deficit_counts.values()),
            "supporting_major_deficit_count": sum(supporting_major_deficit_counts.values()),
            "recommended_next_change": recommended_next_change,
            "summary": (
                f"The analyzer now shows restored candidates through diagnostics-first strength scoring; "
                f"the dominant classification reason is {dominant_reason or 'n/a'} and the most common "
                f"metric pattern is {dominant_metric or 'n/a'}, so the next recommended change is "
                f"{recommended_next_change}."
            ),
        },
        "restored_candidate_strength_details": candidate_strength_details,
    }


def render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown(
    summary: dict[str, Any],
) -> str:
    root = _safe_dict(summary.get("root_assessment"))
    config = _safe_dict(summary.get("experimental_config"))
    visibility = _safe_dict(summary.get("visibility_context"))
    reason_counts = _safe_list(summary.get("classification_reason_counts"))
    hard_blocker_counts = _safe_list(summary.get("hard_blocker_counts"))
    critical_major_deficit_counts = _safe_list(summary.get("critical_major_deficit_counts"))
    supporting_major_deficit_counts = _safe_list(summary.get("supporting_major_deficit_counts"))

    lines = [
        "# Candidate C2 Top-N Weak Strength Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Config: top_n_symbols={config.get('top_n_symbols', 'n/a')}, top_n_strategies={config.get('top_n_strategies', 'n/a')}",
        f"- Newly visible identities: {visibility.get('newly_visible_identity_count', 0)}",
        f"- Restored candidate count: {summary.get('restored_candidate_count', 0)}",
        f"- Weak count: {summary.get('weak_count', 0)}",
        f"- Moderate count: {summary.get('moderate_count', 0)}",
        f"- Strong count: {summary.get('strong_count', 0)}",
        f"- Dominant classification reason: {root.get('dominant_classification_reason', 'n/a')}",
        f"- Dominant metric pattern: {root.get('dominant_metric_pattern', 'n/a')}",
        f"- Recommended next change: {root.get('recommended_next_change', 'n/a')}",
        f"- Root assessment: {root.get('summary', 'n/a')}",
        "",
        "## Classification Reasons",
    ]

    if reason_counts:
        for item in reason_counts[:5]:
            lines.append(
                f"- {item.get('classification_reason')}: {item.get('count', 0)}"
            )
    else:
        lines.append("- No classification reasons recorded.")

    lines.extend(["", "## Hard Blockers"])
    if hard_blocker_counts:
        for item in hard_blocker_counts[:5]:
            lines.append(f"- {item.get('hard_blocker')}: {item.get('count', 0)}")
    else:
        lines.append("- No hard blockers recorded.")

    lines.extend(["", "## Critical Major Deficits"])
    if critical_major_deficit_counts:
        for item in critical_major_deficit_counts[:5]:
            lines.append(f"- {item.get('major_deficit')}: {item.get('count', 0)}")
    else:
        lines.append("- No critical major deficits recorded.")

    lines.extend(["", "## Supporting Major Deficits"])
    if supporting_major_deficit_counts:
        for item in supporting_major_deficit_counts[:5]:
            lines.append(f"- {item.get('major_deficit')}: {item.get('count', 0)}")
    else:
        lines.append("- No supporting major deficits recorded.")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_c_topn_weak_strength_diagnosis_report(
    *,
    baseline_dataset_path: Path = CANDIDATE_A_DEFAULT_PATH,
    experiment_dataset_path: Path = DEFAULT_CANDIDATE_C_DATASET,
    output_json_path: Path = DEFAULT_JSON_OUTPUT,
    output_md_path: Path = DEFAULT_MD_OUTPUT,
    top_n_symbols: int = DEFAULT_TOP_N_SYMBOLS,
    top_n_strategies: int = DEFAULT_TOP_N_STRATEGIES,
    analyzer_trace_dir: Path = DEFAULT_ANALYZER_TRACE_DIR,
) -> dict[str, Any]:
    baseline_rows, _ = load_jsonl_records(baseline_dataset_path)
    experiment_loaded_rows, _ = load_jsonl_records(experiment_dataset_path)
    experiment_rows = filter_candidate_c_records(experiment_loaded_rows)
    experiment_summary = run_research_analyzer(
        input_path=experiment_dataset_path,
        output_dir=analyzer_trace_dir,
    )

    strength_inputs = build_restored_topn_strength_inputs(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=experiment_summary,
        top_n_symbols=top_n_symbols,
        top_n_strategies=top_n_strategies,
    )
    summary = build_experimental_candidate_c_topn_weak_strength_diagnosis_summary(
        candidate_strength_details=_safe_list(strength_inputs.get("candidate_strength_details")),
        experimental_config={
            "top_n_symbols": top_n_symbols,
            "top_n_strategies": top_n_strategies,
        },
        visibility_context=_safe_dict(strength_inputs.get("visibility_context")),
    )
    markdown = render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown(summary)

    _write_json(output_json_path, summary)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown, encoding="utf-8")

    return {
        "json_output_path": str(output_json_path),
        "md_output_path": str(output_md_path),
        "summary": summary,
        "markdown": markdown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental Candidate C2 restored top-N weak-strength diagnosis report."
    )
    parser.add_argument("--baseline-dataset", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--experiment-dataset", type=Path, default=DEFAULT_CANDIDATE_C_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--top-n-symbols", type=int, default=DEFAULT_TOP_N_SYMBOLS)
    parser.add_argument("--top-n-strategies", type=int, default=DEFAULT_TOP_N_STRATEGIES)
    parser.add_argument("--analyzer-trace-dir", type=Path, default=DEFAULT_ANALYZER_TRACE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_c_topn_weak_strength_diagnosis_report(
        baseline_dataset_path=args.baseline_dataset,
        experiment_dataset_path=args.experiment_dataset,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        top_n_symbols=args.top_n_symbols,
        top_n_strategies=args.top_n_strategies,
        analyzer_trace_dir=args.analyzer_trace_dir,
    )
    print(
        json.dumps(
            {
                "json_output_path": result["json_output_path"],
                "md_output_path": result["md_output_path"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
