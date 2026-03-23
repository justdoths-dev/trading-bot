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
    _evaluate_ranked_group,
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
    EDGE_MODERATE_MEDIAN_RETURN_PCT,
    EDGE_MODERATE_POSITIVE_RATE_PCT,
    EDGE_MODERATE_ROBUSTNESS_PCT,
    EDGE_MODERATE_SAMPLE_COUNT,
    _select_robustness_signal,
    run_research_analyzer,
)

DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_weak_strength_diagnosis_report.json"
DEFAULT_MD_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_weak_strength_diagnosis_report.md"
DEFAULT_ANALYZER_TRACE_DIR = DEFAULT_OUTPUT_DIR / "topn_weak_strength_trace"

TARGET_HORIZONS = ("15m", "1h", "4h")
REPRESENTATIVE_LIMIT = 5


def _sorted_counter(counter: Counter[str], *, key_name: str) -> list[dict[str, Any]]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _sorted_grouped_breakdown(
    grouped_counters: dict[str, Counter[str]],
    grouped_candidate_counts: Counter[str],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for value, counter in sorted(grouped_counters.items(), key=lambda item: item[0]):
        items.append(
            {
                field_name: value,
                "candidate_count": grouped_candidate_counts.get(value, 0),
                "weak_driver_counts": _sorted_counter(counter, key_name="weak_driver"),
            }
        )
    items.sort(key=lambda item: (-item["candidate_count"], item[field_name]))
    return items


def _visible_ranked_groups_with_metrics(
    summary: dict[str, Any],
    *,
    bucket: str,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    ranking = _safe_dict(_safe_dict(summary.get("strategy_lab")).get("ranking"))
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in TARGET_HORIZONS:
        visible_rows: list[dict[str, Any]] = []
        raw_rows = _extract_ranked_groups(_safe_dict(ranking.get(horizon)).get(bucket))
        for entry in raw_rows:
            evaluated = _evaluate_ranked_group(entry)
            if evaluated.get("sample_gate") != "passed":
                continue

            metrics = _safe_dict(entry.get("metrics"))
            robustness_signal, robustness_value = _select_robustness_signal(metrics)

            visible_rows.append(
                {
                    **evaluated,
                    "median_future_return_pct": _safe_float(
                        metrics.get("median_future_return_pct")
                    ),
                    "positive_rate_pct": _safe_float(
                        metrics.get("positive_rate_pct", metrics.get("up_rate_pct"))
                    ),
                    "robustness_signal": robustness_signal,
                    "robustness_value": robustness_value,
                    "coverage_pct": _safe_float(metrics.get("coverage_pct")),
                    "labeled_count": _safe_float(metrics.get("labeled_count")),
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


def _weak_drivers_from_metrics(candidate: dict[str, Any]) -> list[str]:
    drivers: list[str] = []

    sample_count = _safe_float(candidate.get("sample_count"))
    median_future_return_pct = _safe_float(candidate.get("median_future_return_pct"))
    positive_rate_pct = _safe_float(candidate.get("positive_rate_pct"))
    robustness_value = _safe_float(candidate.get("robustness_value"))

    if sample_count is None or sample_count < EDGE_MODERATE_SAMPLE_COUNT:
        drivers.append("low_sample_count")
    if (
        median_future_return_pct is None
        or median_future_return_pct < EDGE_MODERATE_MEDIAN_RETURN_PCT
    ):
        drivers.append("low_median_return")
    if positive_rate_pct is None or positive_rate_pct < EDGE_MODERATE_POSITIVE_RATE_PCT:
        drivers.append("low_positive_rate")
    if robustness_value is not None and robustness_value < EDGE_MODERATE_ROBUSTNESS_PCT:
        drivers.append("low_robustness_value")

    return drivers


def _combination_label(drivers: list[str]) -> str:
    return "+".join(sorted(drivers)) if drivers else "meets_moderate_thresholds"


def _distribution_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def _metric_summary(
        values: list[float],
        threshold: float,
        *,
        missing_count: int = 0,
    ) -> dict[str, Any]:
        if not values:
            return {
                "available_count": 0,
                "missing_count": missing_count,
                "below_moderate_count": 0,
                "at_or_above_moderate_count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "moderate_threshold": threshold,
            }

        avg = round(sum(values) / len(values), 6)
        below = sum(1 for value in values if value < threshold)
        return {
            "available_count": len(values),
            "missing_count": missing_count,
            "below_moderate_count": below,
            "at_or_above_moderate_count": len(values) - below,
            "min": min(values),
            "max": max(values),
            "avg": avg,
            "moderate_threshold": threshold,
        }

    sample_values = [
        float(value)
        for value in (_safe_float(item.get("sample_count")) for item in candidates)
        if value is not None
    ]
    median_values = [
        float(value)
        for value in (_safe_float(item.get("median_future_return_pct")) for item in candidates)
        if value is not None
    ]
    positive_values = [
        float(value)
        for value in (_safe_float(item.get("positive_rate_pct")) for item in candidates)
        if value is not None
    ]
    robustness_values = [
        float(value)
        for value in (_safe_float(item.get("robustness_value")) for item in candidates)
        if value is not None
    ]

    positive_missing = sum(
        1 for item in candidates if _safe_float(item.get("positive_rate_pct")) is None
    )
    robustness_missing = sum(
        1 for item in candidates if _safe_float(item.get("robustness_value")) is None
    )

    return {
        "sample_count": _metric_summary(sample_values, EDGE_MODERATE_SAMPLE_COUNT),
        "median_future_return_pct": _metric_summary(
            median_values,
            EDGE_MODERATE_MEDIAN_RETURN_PCT,
        ),
        "positive_rate_pct": _metric_summary(
            positive_values,
            EDGE_MODERATE_POSITIVE_RATE_PCT,
            missing_count=positive_missing,
        ),
        "robustness_value": _metric_summary(
            robustness_values,
            EDGE_MODERATE_ROBUSTNESS_PCT,
            missing_count=robustness_missing,
        ),
    }


def _recommended_next_change(
    *,
    dominant_driver: str | None,
    weak_count: int,
    total_count: int,
    single_metric_dominates: bool,
    has_multi_driver_combinations: bool,
) -> str:
    if weak_count == 0:
        return "none_required"

    if single_metric_dominates:
        if dominant_driver in {
            "low_positive_rate",
            "low_median_return",
            "low_sample_count",
        }:
            return "candidate strength threshold adjustment"
        if dominant_driver == "low_robustness_value":
            return "candidate strength formula redesign"

    if total_count > 0 and weak_count == total_count and has_multi_driver_combinations:
        return "combined scoring + engine redesign"

    return "candidate strength formula redesign"


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

    topn_symbols = _visible_ranked_groups_with_metrics(
        experiment_summary,
        bucket="by_symbol",
        top_n=top_n_symbols,
    )
    topn_strategies = _visible_ranked_groups_with_metrics(
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

        detail = {
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
            "source_candidate_strength": candidate.get("selected_candidate_strength"),
        }
        detail["weak_drivers"] = _weak_drivers_from_metrics(detail)
        detail["weak_driver_combination"] = _combination_label(detail["weak_drivers"])
        candidate_strength_details.append(detail)

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

    weak_candidates = [
        candidate
        for candidate in candidate_strength_details
        if _safe_text(candidate.get("candidate_strength")) == "weak"
    ]
    non_weak_candidates = [
        candidate
        for candidate in candidate_strength_details
        if _safe_text(candidate.get("candidate_strength")) != "weak"
    ]

    weak_driver_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    by_horizon: dict[str, Counter[str]] = {}
    by_symbol: dict[str, Counter[str]] = {}
    by_strategy: dict[str, Counter[str]] = {}
    horizon_candidate_counts: Counter[str] = Counter()
    symbol_candidate_counts: Counter[str] = Counter()
    strategy_candidate_counts: Counter[str] = Counter()

    for candidate in weak_candidates:
        horizon = _safe_text(candidate.get("horizon")) or "unknown"
        symbol = _safe_text(candidate.get("symbol")) or "unknown"
        strategy = _safe_text(candidate.get("strategy")) or "unknown"

        horizon_candidate_counts[horizon] += 1
        symbol_candidate_counts[symbol] += 1
        strategy_candidate_counts[strategy] += 1

        drivers = list(candidate.get("weak_drivers") or [])
        if not drivers:
            drivers = ["unclassified_weak_driver"]

        for driver in drivers:
            weak_driver_counts[driver] += 1
            by_horizon.setdefault(horizon, Counter())[driver] += 1
            by_symbol.setdefault(symbol, Counter())[driver] += 1
            by_strategy.setdefault(strategy, Counter())[driver] += 1

        combination_counts[_combination_label(drivers)] += 1

    dominant_driver = None
    if weak_driver_counts:
        dominant_driver = sorted(
            weak_driver_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    all_weak = len(weak_candidates) == len(candidate_strength_details) and len(candidate_strength_details) > 0
    single_driver_only = all(
        len(candidate.get("weak_drivers") or []) <= 1 for candidate in weak_candidates
    )
    dominant_driver_count = weak_driver_counts.get(dominant_driver or "", 0)
    has_multi_driver_combinations = any("+" in label for label in combination_counts)

    recommended_change = _recommended_next_change(
        dominant_driver=dominant_driver,
        weak_count=len(weak_candidates),
        total_count=len(candidate_strength_details),
        single_metric_dominates=dominant_driver_count == len(weak_candidates) and single_driver_only,
        has_multi_driver_combinations=has_multi_driver_combinations,
    )

    representative_weak_candidates = sorted(
        weak_candidates,
        key=lambda item: (
            -len(item.get("weak_drivers") or []),
            str(item.get("symbol")),
            str(item.get("strategy")),
            str(item.get("horizon")),
        ),
    )[:REPRESENTATIVE_LIMIT]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "experimental_candidate_c_topn_weak_strength_diagnosis_report",
        "architecture_note": {
            "summary": (
                "This report isolates top-N restored candidates and traces the exact scoring inputs "
                "that keep their candidate_strength at weak under the current analyzer thresholds."
            )
        },
        "experimental_config": experimental_config,
        "visibility_context": visibility_context,
        "restored_candidate_count": len(candidate_strength_details),
        "weak_count": len(weak_candidates),
        "non_weak_count": len(non_weak_candidates),
        "weak_driver_counts": {
            "individual": _sorted_counter(weak_driver_counts, key_name="weak_driver"),
            "combinations": _sorted_counter(
                combination_counts,
                key_name="weak_driver_combination",
            ),
        },
        "weak_driver_breakdown_by_horizon": _sorted_grouped_breakdown(
            by_horizon,
            horizon_candidate_counts,
            field_name="horizon",
        ),
        "weak_driver_breakdown_by_symbol": _sorted_grouped_breakdown(
            by_symbol,
            symbol_candidate_counts,
            field_name="symbol",
        ),
        "weak_driver_breakdown_by_strategy": _sorted_grouped_breakdown(
            by_strategy,
            strategy_candidate_counts,
            field_name="strategy",
        ),
        "representative_weak_candidates": representative_weak_candidates,
        "scoring_input_distribution_summary": _distribution_summary(candidate_strength_details),
        "root_assessment": {
            "dominant_weak_driver": dominant_driver,
            "all_restored_candidates_are_weak": all_weak,
            "single_metric_dominates": dominant_driver_count == len(weak_candidates)
            and single_driver_only,
            "recommended_next_change": recommended_change,
            "summary": (
                f"The dominant driver of weak classification is {dominant_driver or 'n/a'}; "
                f"the next recommended change is {recommended_change}."
            ),
        },
        "restored_candidate_strength_details": candidate_strength_details,
    }


def render_experimental_candidate_c_topn_weak_strength_diagnosis_markdown(
    summary: dict[str, Any],
) -> str:
    root = _safe_dict(summary.get("root_assessment"))
    weak_driver_counts = _safe_dict(summary.get("weak_driver_counts"))
    individual = _safe_list(weak_driver_counts.get("individual"))
    config = _safe_dict(summary.get("experimental_config"))
    visibility_context = _safe_dict(summary.get("visibility_context"))

    lines = [
        "# Candidate C2 Top-N Weak Strength Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Config: top_n_symbols={config.get('top_n_symbols', 'n/a')}, top_n_strategies={config.get('top_n_strategies', 'n/a')}",
        f"- Baseline top-1 visible identities: {visibility_context.get('baseline_top1_visible_count', 0)}",
        f"- Top-N visible identities: {visibility_context.get('topn_visible_count', 0)}",
        f"- Newly visible identities: {visibility_context.get('newly_visible_identity_count', 0)}",
        f"- Restored candidate count: {summary.get('restored_candidate_count', 0)}",
        f"- Weak count: {summary.get('weak_count', 0)}",
        f"- Non-weak count: {summary.get('non_weak_count', 0)}",
        f"- Dominant weak driver: {root.get('dominant_weak_driver', 'n/a')}",
        f"- Recommended next change: {root.get('recommended_next_change', 'n/a')}",
        f"- Root assessment: {root.get('summary', 'n/a')}",
        "",
        "## Weak Driver Counts",
    ]

    if individual:
        for item in individual[:5]:
            lines.append(f"- {item.get('weak_driver')}: {item.get('count', 0)}")
    else:
        lines.append("- No weak drivers recorded.")

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
        candidate_strength_details=_safe_list(
            strength_inputs.get("candidate_strength_details")
        ),
        experimental_config={
            "top_n_symbols": max(1, int(top_n_symbols)),
            "top_n_strategies": max(1, int(top_n_strategies)),
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
