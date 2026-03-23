from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.experimental_candidate_c_analyzer_runner import (
    DEFAULT_INPUT_PATH as DEFAULT_CANDIDATE_C_DATASET,
)
from src.research.experimental_candidate_c_analyzer_runner import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from src.research.experimental_candidate_comparison_matrix import (
    CANDIDATE_A_DEFAULT_PATH,
    load_jsonl_records,
)
from src.research.experimental_candidate_intersection_utils import (
    build_intersection_datasets,
    filter_candidate_c_records,
)
from src.research.experimental_candidate_c_topn_preservation_report import (
    DEFAULT_TOP_N_STRATEGIES,
    DEFAULT_TOP_N_SYMBOLS,
    _build_engine_payload,
    _build_exclusive_identities,
    _build_exclusive_rows,
    _build_experimental_candidates,
    _build_visibility,
    _identity_key,
    _normalize_horizon,
    _normalize_strategy,
    _normalize_symbol,
    _safe_dict,
    _safe_list,
    _safe_text,
    _visible_ranked_groups,
    _write_json,
    build_experimental_candidate_c_topn_preservation_summary,
)
from src.research.research_analyzer import run_research_analyzer

DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_eligibility_failure_report.json"
DEFAULT_MD_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_eligibility_failure_report.md"
DEFAULT_ANALYZER_TRACE_DIR = DEFAULT_OUTPUT_DIR / "topn_eligibility_failure_trace"
REPRESENTATIVE_LIMIT = 5

STRUCTURAL_READINESS_REASONS = {
    "CANDIDATE_STABILITY_SINGLE_HORIZON_ONLY",
}
QUALITY_REASONS = {
    "CANDIDATE_STRENGTH_WEAK",
    "CANDIDATE_EDGE_STABILITY_SCORE_LOW",
    "CANDIDATE_DRIFT_DECREASING",
}
SUPPORT_STABILITY_REASONS = {
    "CANDIDATE_STRENGTH_INSUFFICIENT_DATA",
    "CANDIDATE_STABILITY_INSUFFICIENT_DATA",
    "CANDIDATE_STABILITY_UNSTABLE",
    "CANDIDATE_LATEST_SAMPLE_TOO_LOW",
    "CANDIDATE_CUMULATIVE_SAMPLE_TOO_LOW",
    "CANDIDATE_SYMBOL_SUPPORT_TOO_LOW",
    "CANDIDATE_STRATEGY_SUPPORT_TOO_LOW",
}
EXPLICIT_GATE_REASONS = {
    "CANDIDATE_IDENTITY_INCOMPLETE",
    *SUPPORT_STABILITY_REASONS,
}


def _candidate_key(candidate: dict[str, Any]) -> str:
    symbol = _normalize_symbol(candidate.get("symbol")) or "unknown"
    strategy = _normalize_strategy(candidate.get("strategy")) or "unknown"
    horizon = _normalize_horizon(candidate.get("horizon")) or "unknown"
    return _identity_key(symbol, strategy, horizon)


def _non_empty_reason_codes(candidate: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in (_safe_text(value) for value in _safe_list(candidate.get("reason_codes"))):
        if not reason or reason in seen:
            continue
        seen.add(reason)
        ordered.append(reason)
    return ordered


def _status_counts(candidates: list[dict[str, Any]]) -> tuple[int, int, int]:
    blocked = 0
    penalized = 0
    eligible = 0
    for candidate in candidates:
        status = _safe_text(candidate.get("candidate_status")) or "unknown"
        if status == "blocked":
            blocked += 1
        elif status == "penalized":
            penalized += 1
        elif status == "eligible":
            eligible += 1
    return blocked, penalized, eligible


def _sorted_counter(counter: Counter[str], *, value_label: str) -> list[dict[str, Any]]:
    return [
        {value_label: key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _sorted_breakdown(
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
                "blocker_counts": _sorted_counter(counter, value_label="reason_code"),
            }
        )
    items.sort(key=lambda item: (-item["candidate_count"], item[field_name]))
    return items


def _failure_family_for_reason(reason_code: str | None) -> str:
    if reason_code in STRUCTURAL_READINESS_REASONS:
        return "structural_readiness_design"
    if reason_code in QUALITY_REASONS:
        return "weak_candidate_quality_or_low_confidence"
    if reason_code in SUPPORT_STABILITY_REASONS:
        return "insufficient_stability_or_support"
    if reason_code in EXPLICIT_GATE_REASONS:
        return "explicit_engine_gate_blocker"
    return "unknown_failure_family"


def _recommended_next_change(dominant_reason: str | None, dominant_family: str) -> str:
    if dominant_reason in SUPPORT_STABILITY_REASONS or dominant_reason in STRUCTURAL_READINESS_REASONS:
        return "stability/support handling redesign"
    if dominant_reason in QUALITY_REASONS or dominant_family == "weak_candidate_quality_or_low_confidence":
        return "readiness scoring redesign"
    if dominant_family == "explicit_engine_gate_blocker":
        return "eligibility gate redesign"
    return "existence-layer separation"


def build_restored_topn_candidate_inputs(
    *,
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    experiment_summary: dict[str, Any],
    top_n_symbols: int,
    top_n_strategies: int,
) -> dict[str, Any]:
    top_n_symbols = max(1, int(top_n_symbols))
    top_n_strategies = max(1, int(top_n_strategies))

    topn_summary = build_experimental_candidate_c_topn_preservation_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=experiment_summary,
        top_n_symbols=top_n_symbols,
        top_n_strategies=top_n_strategies,
    )

    _, experiment_shared_rows, _ = build_intersection_datasets(baseline_rows, experiment_rows)
    exclusive_rows = _build_exclusive_rows(experiment_rows, experiment_shared_rows)
    identities = _build_exclusive_identities(exclusive_rows)

    top1_symbols = _visible_ranked_groups(experiment_summary, bucket="by_symbol", top_n=1)
    top1_strategies = _visible_ranked_groups(experiment_summary, bucket="by_strategy", top_n=1)
    topn_symbols = _visible_ranked_groups(experiment_summary, bucket="by_symbol", top_n=top_n_symbols)
    topn_strategies = _visible_ranked_groups(
        experiment_summary,
        bucket="by_strategy",
        top_n=top_n_strategies,
    )

    top1_visibility = _build_visibility(
        identities,
        preserved_symbols=top1_symbols,
        preserved_strategies=top1_strategies,
    )
    topn_visibility = _build_visibility(
        identities,
        preserved_symbols=topn_symbols,
        preserved_strategies=topn_strategies,
    )

    top1_keys = {item["identity_key"] for item in top1_visibility["visible_identities"]}
    restored_visible_identities = [
        item
        for item in topn_visibility["visible_identities"]
        if item["identity_key"] not in top1_keys
    ]
    restored_keys = {item["identity_key"] for item in restored_visible_identities}

    topn_candidates = _build_experimental_candidates(
        visible_identities=topn_visibility["visible_identities"],
        preserved_symbols=topn_symbols,
        preserved_strategies=topn_strategies,
    )
    restored_candidates = [
        candidate for candidate in topn_candidates if _candidate_key(candidate) in restored_keys
    ]

    topn_shadow_output = run_edge_selection_engine(_build_engine_payload(topn_candidates))
    restored_ranking = [
        _safe_dict(item)
        for item in _safe_list(topn_shadow_output.get("ranking"))
        if _candidate_key(_safe_dict(item)) in restored_keys
    ]

    return {
        "topn_summary": topn_summary,
        "restored_visible_identities": restored_visible_identities,
        "restored_candidates": restored_candidates,
        "restored_ranking": restored_ranking,
    }


def build_experimental_candidate_c_topn_eligibility_failure_summary(
    *,
    restored_candidates: list[dict[str, Any]],
    restored_ranking: list[dict[str, Any]],
    visibility_context: dict[str, Any] | None = None,
    experimental_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    visibility_context = _safe_dict(visibility_context)
    experimental_config = _safe_dict(experimental_config)

    ranking_by_key = {_candidate_key(candidate): candidate for candidate in restored_ranking}
    candidate_details: list[dict[str, Any]] = []

    blocker_counts: Counter[str] = Counter()
    by_horizon: dict[str, Counter[str]] = {}
    by_symbol: dict[str, Counter[str]] = {}
    by_strategy: dict[str, Counter[str]] = {}
    by_strength: dict[str, Counter[str]] = {}
    by_stability: dict[str, Counter[str]] = {}
    horizon_candidate_counts: Counter[str] = Counter()
    symbol_candidate_counts: Counter[str] = Counter()
    strategy_candidate_counts: Counter[str] = Counter()
    strength_candidate_counts: Counter[str] = Counter()
    stability_candidate_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    primary_reason_counts: Counter[str] = Counter()

    for candidate in restored_candidates:
        key = _candidate_key(candidate)
        ranked = _safe_dict(ranking_by_key.get(key))
        merged = dict(candidate)
        merged.update(ranked)

        reason_codes = _non_empty_reason_codes(merged)
        candidate_status = _safe_text(merged.get("candidate_status")) or "unknown"
        primary_reason = reason_codes[0] if reason_codes else None
        primary_family = _failure_family_for_reason(primary_reason)

        if candidate_status != "eligible":
            horizon = _safe_text(merged.get("horizon")) or "unknown"
            symbol = _safe_text(merged.get("symbol")) or "unknown"
            strategy = _safe_text(merged.get("strategy")) or "unknown"
            strength = _safe_text(merged.get("selected_candidate_strength")) or "unknown"
            stability = _safe_text(merged.get("selected_stability_label")) or "unknown"

            horizon_candidate_counts[horizon] += 1
            symbol_candidate_counts[symbol] += 1
            strategy_candidate_counts[strategy] += 1
            strength_candidate_counts[strength] += 1
            stability_candidate_counts[stability] += 1

            for reason in reason_codes:
                blocker_counts[reason] += 1
                by_horizon.setdefault(horizon, Counter())[reason] += 1
                by_symbol.setdefault(symbol, Counter())[reason] += 1
                by_strategy.setdefault(strategy, Counter())[reason] += 1
                by_strength.setdefault(strength, Counter())[reason] += 1
                by_stability.setdefault(stability, Counter())[reason] += 1

            if primary_reason is not None:
                primary_reason_counts[primary_reason] += 1
            family_counts[primary_family] += 1

        candidate_details.append(
            {
                "symbol": merged.get("symbol"),
                "strategy": merged.get("strategy"),
                "horizon": merged.get("horizon"),
                "candidate_status": candidate_status,
                "reason_codes": reason_codes,
                "advisory_reason_codes": [
                    code
                    for code in (
                        _safe_text(value)
                        for value in _safe_list(merged.get("advisory_reason_codes"))
                    )
                    if code
                ],
                "candidate_strength": _safe_text(merged.get("selected_candidate_strength"))
                or "unknown",
                "stability_label": _safe_text(merged.get("selected_stability_label"))
                or "unknown",
                "edge_stability_score": merged.get("edge_stability_score"),
                "latest_sample_size": merged.get("latest_sample_size"),
                "cumulative_sample_size": merged.get("cumulative_sample_size"),
                "symbol_cumulative_support": merged.get("symbol_cumulative_support"),
                "strategy_cumulative_support": merged.get("strategy_cumulative_support"),
                "selection_score": merged.get("selection_score"),
                "selection_confidence": merged.get("selection_confidence"),
                "source_preference": merged.get("source_preference"),
                "selected_visible_horizons": _safe_list(merged.get("selected_visible_horizons")),
                "gate_diagnostics": _safe_dict(merged.get("gate_diagnostics")),
                "primary_failure_family": primary_family,
            }
        )

    blocked_count, penalized_count, eligibility_passed_count = _status_counts(candidate_details)
    blocker_summary = _sorted_counter(blocker_counts, value_label="reason_code")
    primary_reason_summary = _sorted_counter(primary_reason_counts, value_label="reason_code")

    single_most_common_reason = blocker_summary[0]["reason_code"] if blocker_summary else None
    dominant_failure_family = _failure_family_for_reason(single_most_common_reason)
    recommended_change = _recommended_next_change(single_most_common_reason, dominant_failure_family)

    representative_blocked_candidates = sorted(
        [item for item in candidate_details if item["candidate_status"] != "eligible"],
        key=lambda item: (
            0 if item["candidate_status"] == "blocked" else 1,
            -len(item["reason_codes"]),
            str(item["symbol"]),
            str(item["strategy"]),
            str(item["horizon"]),
        ),
    )[:REPRESENTATIVE_LIMIT]

    baseline_visible = int(visibility_context.get("baseline_top1_visible_count") or 0)
    topn_visible = int(visibility_context.get("topn_visible_count") or 0)
    newly_visible = int(visibility_context.get("newly_visible_identity_count") or 0)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "experimental_candidate_c_topn_eligibility_failure_report",
        "architecture_note": {
            "summary": (
                "This report isolates candidates that became visible only after top-N preservation "
                "and diagnoses why those restored candidates still fail eligibility or readiness."
            )
        },
        "experimental_config": experimental_config,
        "visibility_context": visibility_context,
        "restored_candidate_count": len(restored_candidates),
        "eligibility_passed_count": eligibility_passed_count,
        "blocked_count": blocked_count,
        "penalized_count": penalized_count,
        "blocker_counts": blocker_summary,
        "primary_failure_reason_counts": primary_reason_summary,
        "blocker_breakdown_by_horizon": _sorted_breakdown(
            by_horizon,
            horizon_candidate_counts,
            field_name="horizon",
        ),
        "blocker_breakdown_by_symbol": _sorted_breakdown(
            by_symbol,
            symbol_candidate_counts,
            field_name="symbol",
        ),
        "blocker_breakdown_by_strategy": _sorted_breakdown(
            by_strategy,
            strategy_candidate_counts,
            field_name="strategy",
        ),
        "blocker_breakdown_by_candidate_strength": _sorted_breakdown(
            by_strength,
            strength_candidate_counts,
            field_name="candidate_strength",
        ),
        "blocker_breakdown_by_stability_label": _sorted_breakdown(
            by_stability,
            stability_candidate_counts,
            field_name="stability_label",
        ),
        "representative_blocked_candidates": representative_blocked_candidates,
        "root_assessment": {
            "visibility_restoration_confirmed": topn_visible > baseline_visible,
            "newly_visible_identity_count": newly_visible,
            "restored_candidates_failed_readiness": len(restored_candidates) > eligibility_passed_count,
            "single_most_common_failure_reason": single_most_common_reason,
            "dominant_failure_family": dominant_failure_family,
            "failure_family_counts": _sorted_counter(
                family_counts,
                value_label="failure_family",
            ),
            "recommended_next_change": recommended_change,
            "summary": (
                f"Visibility was restored for {newly_visible} additional identities, but restored candidates still failed readiness; "
                f"the most common failure reason was {single_most_common_reason or 'n/a'}, so the next recommended change is {recommended_change}."
            ),
        },
        "restored_candidate_details": candidate_details,
    }


def render_experimental_candidate_c_topn_eligibility_failure_markdown(
    summary: dict[str, Any],
) -> str:
    root = _safe_dict(summary.get("root_assessment"))
    blocker_counts = _safe_list(summary.get("blocker_counts"))
    representatives = _safe_list(summary.get("representative_blocked_candidates"))
    config = _safe_dict(summary.get("experimental_config"))
    visibility_context = _safe_dict(summary.get("visibility_context"))

    lines = [
        "# Candidate C2 Top-N Eligibility Failure Report",
        "",
        "## Executive Summary",
        f"- Config: top_n_symbols={config.get('top_n_symbols', 'n/a')}, top_n_strategies={config.get('top_n_strategies', 'n/a')}",
        f"- Baseline top-1 visible identities: {visibility_context.get('baseline_top1_visible_count', 0)}",
        f"- Top-N visible identities: {visibility_context.get('topn_visible_count', 0)}",
        f"- Newly visible identities: {visibility_context.get('newly_visible_identity_count', 0)}",
        f"- Restored candidate count: {summary.get('restored_candidate_count', 0)}",
        f"- Eligibility passed count: {summary.get('eligibility_passed_count', 0)}",
        f"- Blocked count: {summary.get('blocked_count', 0)}",
        f"- Penalized count: {summary.get('penalized_count', 0)}",
        f"- Most common failure reason: {root.get('single_most_common_failure_reason', 'n/a')}",
        f"- Dominant failure family: {root.get('dominant_failure_family', 'n/a')}",
        f"- Recommended next change: {root.get('recommended_next_change', 'n/a')}",
        f"- Root assessment: {root.get('summary', 'n/a')}",
        "",
        "## Top Blockers",
    ]

    if blocker_counts:
        for item in blocker_counts[:5]:
            lines.append(f"- {item.get('reason_code')}: {item.get('count', 0)}")
    else:
        lines.append("- No blocker reasons recorded.")

    lines.extend(["", "## Representative Candidates"])
    if representatives:
        for item in representatives:
            lines.append(
                f"- {item.get('symbol')} | {item.get('strategy')} | {item.get('horizon')} | "
                f"status={item.get('candidate_status')} | reasons={', '.join(item.get('reason_codes', [])) or 'n/a'}"
            )
    else:
        lines.append("- No restored candidates were blocked or penalized.")

    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_c_topn_eligibility_failure_report(
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
    restored_inputs = build_restored_topn_candidate_inputs(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=experiment_summary,
        top_n_symbols=top_n_symbols,
        top_n_strategies=top_n_strategies,
    )
    topn_summary = _safe_dict(restored_inputs.get("topn_summary"))
    visibility_context = {
        "baseline_top1_visible_count": _safe_dict(
            topn_summary.get("baseline_top1_summary")
        ).get("visible_c2_exclusive_identity_count", 0),
        "topn_visible_count": _safe_dict(topn_summary.get("experimental_topn_summary")).get(
            "visible_c2_exclusive_identity_count",
            0,
        ),
        "newly_visible_identity_count": _safe_dict(
            topn_summary.get("coverage_gain_summary")
        ).get("newly_visible_identity_count", 0),
    }
    experimental_config = {
        "top_n_symbols": max(1, int(top_n_symbols)),
        "top_n_strategies": max(1, int(top_n_strategies)),
    }
    summary = build_experimental_candidate_c_topn_eligibility_failure_summary(
        restored_candidates=_safe_list(restored_inputs.get("restored_candidates")),
        restored_ranking=_safe_list(restored_inputs.get("restored_ranking")),
        visibility_context=visibility_context,
        experimental_config=experimental_config,
    )
    markdown = render_experimental_candidate_c_topn_eligibility_failure_markdown(summary)

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
        description="Run experimental Candidate C2 restored top-N eligibility failure report."
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
    result = run_experimental_candidate_c_topn_eligibility_failure_report(
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
