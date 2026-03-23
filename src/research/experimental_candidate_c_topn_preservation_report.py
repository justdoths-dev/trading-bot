from __future__ import annotations

import argparse
import json
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
    build_row_match_key,
    filter_candidate_c_records,
)
from src.research.research_analyzer import (
    _passes_absolute_minimum_gate,
    _score_candidate_strength,
    _select_robustness_signal,
    run_research_analyzer,
)

DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_preservation_report.json"
DEFAULT_MD_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_c_topn_preservation_report.md"
DEFAULT_ANALYZER_TRACE_DIR = DEFAULT_OUTPUT_DIR / "topn_preservation_trace"

DEFAULT_TOP_N_SYMBOLS = 2
DEFAULT_TOP_N_STRATEGIES = 2

TARGET_HORIZONS = ("15m", "1h", "4h")
TOP_GROUP_LIMIT = 5


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
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _normalize_symbol(value: Any) -> str | None:
    text = _safe_text(value)
    return text.upper() if text is not None else None


def _normalize_strategy(value: Any) -> str | None:
    return _safe_text(value)


def _normalize_horizon(value: Any) -> str | None:
    text = _safe_text(value)
    if text in TARGET_HORIZONS:
        return text
    return None


def _identity_key(symbol: str, strategy: str, horizon: str) -> str:
    return f"{symbol}|{strategy}|{horizon}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def _evaluate_ranked_group(entry: dict[str, Any] | None) -> dict[str, Any]:
    item = entry if isinstance(entry, dict) else {}
    metrics = _safe_dict(item.get("metrics"))

    sample_count = _safe_float(metrics.get("sample_count"))
    labeled_count = _safe_float(metrics.get("labeled_count"))
    coverage_pct = _safe_float(metrics.get("coverage_pct"))
    median_future_return_pct = _safe_float(metrics.get("median_future_return_pct"))
    positive_rate_pct = _safe_float(
        metrics.get("positive_rate_pct", metrics.get("up_rate_pct"))
    )
    _, robustness_value = _select_robustness_signal(metrics)

    sample_gate_passed = _passes_absolute_minimum_gate(
        sample_count=sample_count,
        labeled_count=labeled_count,
        coverage_pct=coverage_pct,
        median_future_return_pct=median_future_return_pct,
    )

    candidate_strength = (
        _score_candidate_strength(
            sample_count=sample_count or 0.0,
            median_future_return_pct=median_future_return_pct or 0.0,
            positive_rate_pct=positive_rate_pct,
            robustness_value=robustness_value,
        )
        if sample_gate_passed
        else "insufficient_data"
    )
    quality_gate = (
        "passed"
        if candidate_strength in {"moderate", "strong"}
        else "borderline"
        if sample_gate_passed
        else "failed"
    )

    return {
        "group": _safe_text(item.get("group")) or "n/a",
        "rank": _safe_int(item.get("rank")) or 0,
        "score": _safe_float(item.get("score")) or 0.0,
        "sample_count": _safe_int(sample_count) or 0,
        "candidate_strength": candidate_strength,
        "quality_gate": quality_gate,
        "sample_gate": "passed" if sample_gate_passed else "failed",
    }


def _visible_ranked_groups(
    summary: dict[str, Any],
    *,
    bucket: str,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    ranking = _safe_dict(_safe_dict(summary.get("strategy_lab")).get("ranking"))
    result: dict[str, list[dict[str, Any]]] = {}

    for horizon in TARGET_HORIZONS:
        rows = [
            _evaluate_ranked_group(entry)
            for entry in _extract_ranked_groups(_safe_dict(ranking.get(horizon)).get(bucket))
        ]
        visible = [row for row in rows if row["sample_gate"] == "passed"]
        result[horizon] = visible[:top_n]

    return result


def _build_exclusive_rows(
    source_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_counts: dict[tuple[str, ...], int] = {}
    shared_counts: dict[tuple[str, ...], int] = {}
    emitted_counts: dict[tuple[str, ...], int] = {}

    for row in source_rows:
        key = build_row_match_key(row)
        source_counts[key] = source_counts.get(key, 0) + 1

    for row in shared_rows:
        key = build_row_match_key(row)
        shared_counts[key] = shared_counts.get(key, 0) + 1

    exclusive_rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = build_row_match_key(row)
        allowed = source_counts.get(key, 0) - shared_counts.get(key, 0)
        emitted = emitted_counts.get(key, 0)
        if emitted >= allowed:
            continue
        exclusive_rows.append(row)
        emitted_counts[key] = emitted + 1

    return exclusive_rows


def _build_exclusive_identities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("selected_strategy") or row.get("strategy"))
        if symbol is None or strategy is None:
            continue

        for horizon in TARGET_HORIZONS:
            label = _safe_text(row.get(f"future_label_{horizon}"))
            if label not in {"up", "down"}:
                continue

            key = (symbol, strategy, horizon)
            entry = grouped.setdefault(
                key,
                {
                    "identity_key": _identity_key(symbol, strategy, horizon),
                    "symbol": symbol,
                    "strategy": strategy,
                    "horizon": horizon,
                    "directional_row_count": 0,
                },
            )
            entry["directional_row_count"] += 1

    identities = list(grouped.values())
    identities.sort(key=lambda item: (-int(item["directional_row_count"]), item["identity_key"]))
    return identities


def _build_visibility(
    identities: list[dict[str, Any]],
    *,
    preserved_symbols: dict[str, list[dict[str, Any]]],
    preserved_strategies: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    visible: list[dict[str, Any]] = []
    invisible: list[dict[str, Any]] = []

    for identity in identities:
        horizon = str(identity["horizon"])
        symbol_groups = {
            _normalize_symbol(item.get("group"))
            for item in preserved_symbols.get(horizon, [])
            if _normalize_symbol(item.get("group")) is not None
        }
        strategy_groups = {
            _normalize_strategy(item.get("group"))
            for item in preserved_strategies.get(horizon, [])
            if _normalize_strategy(item.get("group")) is not None
        }

        if identity["symbol"] in symbol_groups and identity["strategy"] in strategy_groups:
            visible.append(identity)
        else:
            invisible.append(identity)

    return {
        "visible_identities": visible,
        "invisible_identities": invisible,
        "visible_count": len(visible),
        "invisible_count": len(invisible),
        "visible_ratio": _safe_ratio(len(visible), len(identities)),
    }


def _group_horizon_counts(
    groups: dict[str, list[dict[str, Any]]],
    *,
    category: str,
) -> dict[str, list[str]]:
    counts: dict[str, list[str]] = {}

    for horizon, rows in groups.items():
        for row in rows:
            group = _safe_text(row.get("group"))
            if group is None:
                continue

            normalized_group = (
                _normalize_symbol(group) if category == "symbol" else _normalize_strategy(group)
            )
            if normalized_group is None:
                continue

            counts.setdefault(normalized_group, [])
            if horizon not in counts[normalized_group]:
                counts[normalized_group].append(horizon)

    return counts


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


def _build_experimental_candidates(
    *,
    visible_identities: list[dict[str, Any]],
    preserved_symbols: dict[str, list[dict[str, Any]]],
    preserved_strategies: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    symbol_horizons = _group_horizon_counts(preserved_symbols, category="symbol")
    strategy_horizons = _group_horizon_counts(preserved_strategies, category="strategy")
    symbol_index = _index_groups_by_horizon_and_name(preserved_symbols, category="symbol")
    strategy_index = _index_groups_by_horizon_and_name(preserved_strategies, category="strategy")

    candidates: list[dict[str, Any]] = []

    for identity in visible_identities:
        symbol = str(identity["symbol"])
        strategy = str(identity["strategy"])
        horizon = str(identity["horizon"])

        symbol_row = _safe_dict(symbol_index.get((horizon, symbol)))
        strategy_row = _safe_dict(strategy_index.get((horizon, strategy)))
        if not symbol_row or not strategy_row:
            continue

        symbol_score = float(symbol_row.get("score") or 0.0)
        strategy_score = float(strategy_row.get("score") or 0.0)

        if symbol_score >= strategy_score:
            support_row = symbol_row
            support_category = "symbol"
            support_horizons = symbol_horizons.get(symbol, [])
        else:
            support_row = strategy_row
            support_category = "strategy"
            support_horizons = strategy_horizons.get(strategy, [])

        stability_label = (
            "multi_horizon_confirmed" if len(support_horizons) >= 2 else "single_horizon_only"
        )

        candidates.append(
            {
                "symbol": symbol,
                "strategy": strategy,
                "horizon": horizon,
                "selected_candidate_strength": support_row.get("candidate_strength")
                or "insufficient_data",
                "selected_stability_label": stability_label,
                "edge_stability_score": float(support_row.get("score") or 0.0),
                "latest_sample_size": int(support_row.get("sample_count") or 0),
                "selected_visible_horizons": support_horizons,
                "drift_direction": "flat",
                "source_preference": support_category,
            }
        )

    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate["symbol"]),
            str(candidate["strategy"]),
            str(candidate["horizon"]),
        )
        existing = deduped.get(key)
        if existing is None or float(candidate["edge_stability_score"]) > float(
            existing["edge_stability_score"]
        ):
            deduped[key] = candidate

    return list(deduped.values())


def _build_engine_payload(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ok": True,
        "candidate_seed_count": len(candidates),
        "candidate_seed_diagnostics": {
            "mode": "experimental_topn_preservation",
        },
        "candidates": candidates,
    }


def _build_summary_from_candidates(
    *,
    label: str,
    candidates: list[dict[str, Any]],
    visibility: dict[str, Any],
    shadow_output: dict[str, Any],
) -> dict[str, Any]:
    ranking = [_safe_dict(item) for item in _safe_list(shadow_output.get("ranking"))]
    eligible = [item for item in ranking if _safe_text(item.get("candidate_status")) == "eligible"]
    selected = _safe_text(shadow_output.get("selection_status")) == "selected"

    emitted_keys = {
        _identity_key(
            _normalize_symbol(item.get("symbol")) or "unknown",
            _normalize_strategy(item.get("strategy")) or "unknown",
            _normalize_horizon(item.get("horizon")) or "unknown",
        )
        for item in candidates
    }
    visible_keys = {item["identity_key"] for item in visibility["visible_identities"]}
    emitted_visible_count = len(emitted_keys & visible_keys)

    eligible_visible_keys = {
        _identity_key(
            _normalize_symbol(item.get("symbol")) or "unknown",
            _normalize_strategy(item.get("strategy")) or "unknown",
            _normalize_horizon(item.get("horizon")) or "unknown",
        )
        for item in eligible
    }

    return {
        "label": label,
        "visible_c2_exclusive_identity_count": visibility["visible_count"],
        "invisible_c2_exclusive_identity_count": visibility["invisible_count"],
        "mapper_emitted_candidate_count": len(candidates),
        "mapper_emitted_visible_identity_count": emitted_visible_count,
        "eligibility_passed_count": len(eligible),
        "eligibility_passed_visible_identity_count": len(eligible_visible_keys & visible_keys),
        "final_selection_count": 1 if selected else 0,
        "selection_status": _safe_text(shadow_output.get("selection_status")) or "unknown",
    }


def _build_per_horizon_preserved_groups(
    *,
    top1_symbols: dict[str, list[dict[str, Any]]],
    top1_strategies: dict[str, list[dict[str, Any]]],
    topn_symbols: dict[str, list[dict[str, Any]]],
    topn_strategies: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for horizon in TARGET_HORIZONS:
        top1_symbol_groups = [
            _safe_text(item.get("group"))
            for item in top1_symbols.get(horizon, [])
            if _safe_text(item.get("group"))
        ]
        top1_strategy_groups = [
            _safe_text(item.get("group"))
            for item in top1_strategies.get(horizon, [])
            if _safe_text(item.get("group"))
        ]
        topn_symbol_groups = [
            _safe_text(item.get("group"))
            for item in topn_symbols.get(horizon, [])
            if _safe_text(item.get("group"))
        ]
        topn_strategy_groups = [
            _safe_text(item.get("group"))
            for item in topn_strategies.get(horizon, [])
            if _safe_text(item.get("group"))
        ]

        result[horizon] = {
            "top1_symbol_groups": top1_symbol_groups,
            "top1_strategy_groups": top1_strategy_groups,
            "topn_symbol_groups": topn_symbol_groups,
            "topn_strategy_groups": topn_strategy_groups,
            "top1_identity_combinations": [
                f"{symbol}|{strategy}|{horizon}"
                for symbol in top1_symbol_groups
                for strategy in top1_strategy_groups
            ],
            "topn_identity_combinations": [
                f"{symbol}|{strategy}|{horizon}"
                for symbol in topn_symbol_groups
                for strategy in topn_strategy_groups
            ],
            "top1_ranked_symbol_groups": top1_symbols.get(horizon, [])[:TOP_GROUP_LIMIT],
            "top1_ranked_strategy_groups": top1_strategies.get(horizon, [])[:TOP_GROUP_LIMIT],
            "topn_ranked_symbol_groups": topn_symbols.get(horizon, [])[:TOP_GROUP_LIMIT],
            "topn_ranked_strategy_groups": topn_strategies.get(horizon, [])[:TOP_GROUP_LIMIT],
        }

    return result


def build_experimental_candidate_c_topn_preservation_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    experiment_summary: dict[str, Any],
    top_n_symbols: int,
    top_n_strategies: int,
) -> dict[str, Any]:
    top_n_symbols = max(1, int(top_n_symbols))
    top_n_strategies = max(1, int(top_n_strategies))

    _, experiment_shared_rows, _ = build_intersection_datasets(baseline_rows, experiment_rows)
    exclusive_rows = _build_exclusive_rows(experiment_rows, experiment_shared_rows)
    identities = _build_exclusive_identities(exclusive_rows)

    top1_symbols = _visible_ranked_groups(experiment_summary, bucket="by_symbol", top_n=1)
    top1_strategies = _visible_ranked_groups(experiment_summary, bucket="by_strategy", top_n=1)
    topn_symbols = _visible_ranked_groups(
        experiment_summary,
        bucket="by_symbol",
        top_n=top_n_symbols,
    )
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

    top1_candidates = _build_experimental_candidates(
        visible_identities=top1_visibility["visible_identities"],
        preserved_symbols=top1_symbols,
        preserved_strategies=top1_strategies,
    )
    topn_candidates = _build_experimental_candidates(
        visible_identities=topn_visibility["visible_identities"],
        preserved_symbols=topn_symbols,
        preserved_strategies=topn_strategies,
    )

    top1_shadow_output = run_edge_selection_engine(_build_engine_payload(top1_candidates))
    topn_shadow_output = run_edge_selection_engine(_build_engine_payload(topn_candidates))

    baseline_top1_summary = _build_summary_from_candidates(
        label="top1",
        candidates=top1_candidates,
        visibility=top1_visibility,
        shadow_output=top1_shadow_output,
    )
    experimental_topn_summary = _build_summary_from_candidates(
        label="topn",
        candidates=topn_candidates,
        visibility=topn_visibility,
        shadow_output=topn_shadow_output,
    )

    top1_visible_keys = {item["identity_key"] for item in top1_visibility["visible_identities"]}
    topn_visible_keys = {item["identity_key"] for item in topn_visibility["visible_identities"]}
    newly_visible = sorted(topn_visible_keys - top1_visible_keys)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "experimental_candidate_c_topn_preservation_report",
        "architecture_note": {
            "summary": (
                "This experimental report compares winner-only top-1 preservation against "
                "top-N preservation for symbol and strategy groups per horizon, then measures "
                "how much C2-exclusive identity visibility and downstream survival improve."
            )
        },
        "baseline_top1_summary": baseline_top1_summary,
        "experimental_topn_summary": experimental_topn_summary,
        "coverage_gain_summary": {
            "total_c2_exclusive_identity_count": len(identities),
            "top1_invisible_count": top1_visibility["invisible_count"],
            "top1_visible_count": top1_visibility["visible_count"],
            "topn_visible_count": topn_visibility["visible_count"],
            "newly_visible_identity_count": len(newly_visible),
            "newly_visible_identity_keys": newly_visible,
            "visibility_gain_ratio": _safe_ratio(len(newly_visible), len(identities)),
        },
        "per_horizon_preserved_groups": _build_per_horizon_preserved_groups(
            top1_symbols=top1_symbols,
            top1_strategies=top1_strategies,
            topn_symbols=topn_symbols,
            topn_strategies=topn_strategies,
        ),
        "c2_exclusive_identity_visibility_top1": {
            "visible_count": top1_visibility["visible_count"],
            "invisible_count": top1_visibility["invisible_count"],
            "visible_identity_keys": [
                item["identity_key"] for item in top1_visibility["visible_identities"]
            ],
        },
        "c2_exclusive_identity_visibility_topn": {
            "visible_count": topn_visibility["visible_count"],
            "invisible_count": topn_visibility["invisible_count"],
            "visible_identity_keys": [
                item["identity_key"] for item in topn_visibility["visible_identities"]
            ],
        },
        "mapper_emission_delta": {
            "baseline_top1": baseline_top1_summary["mapper_emitted_visible_identity_count"],
            "experimental_topn": experimental_topn_summary["mapper_emitted_visible_identity_count"],
            "delta": (
                experimental_topn_summary["mapper_emitted_visible_identity_count"]
                - baseline_top1_summary["mapper_emitted_visible_identity_count"]
            ),
        },
        "eligibility_delta": {
            "baseline_top1": baseline_top1_summary["eligibility_passed_visible_identity_count"],
            "experimental_topn": experimental_topn_summary[
                "eligibility_passed_visible_identity_count"
            ],
            "delta": (
                experimental_topn_summary["eligibility_passed_visible_identity_count"]
                - baseline_top1_summary["eligibility_passed_visible_identity_count"]
            ),
        },
        "final_selection_delta": {
            "baseline_top1": baseline_top1_summary["final_selection_count"],
            "experimental_topn": experimental_topn_summary["final_selection_count"],
            "delta": (
                experimental_topn_summary["final_selection_count"]
                - baseline_top1_summary["final_selection_count"]
            ),
        },
        "root_assessment": {
            "topn_meaningfully_restores_identity_visibility": (
                topn_visibility["visible_count"] > top1_visibility["visible_count"]
            ),
            "comparison_collapse_confirmed_dominant_bottleneck": len(newly_visible) > 0,
            "eligibility_next_dominant_blocker_after_restoration": (
                experimental_topn_summary["visible_c2_exclusive_identity_count"]
                > experimental_topn_summary["eligibility_passed_visible_identity_count"]
            ),
            "summary": (
                "Top-N preservation restores identity visibility, confirming comparison "
                "collapse as the dominant structural bottleneck; readiness remains the next "
                "blocker where restored candidates still fail eligibility."
                if len(newly_visible) > 0
                else (
                    "Top-N preservation does not materially restore visibility, so "
                    "comparison collapse is not yet alleviated by this experimental setting."
                )
            ),
        },
        "experimental_config": {
            "top_n_symbols": top_n_symbols,
            "top_n_strategies": top_n_strategies,
        },
    }


def render_experimental_candidate_c_topn_preservation_markdown(
    summary: dict[str, Any],
) -> str:
    baseline = _safe_dict(summary.get("baseline_top1_summary"))
    experimental = _safe_dict(summary.get("experimental_topn_summary"))
    coverage = _safe_dict(summary.get("coverage_gain_summary"))
    root = _safe_dict(summary.get("root_assessment"))
    config = _safe_dict(summary.get("experimental_config"))

    lines = [
        "# Candidate C2 Top-N Preservation Report",
        "",
        "## Executive Summary",
        f"- Config: top_n_symbols={config.get('top_n_symbols', 'n/a')}, "
        f"top_n_strategies={config.get('top_n_strategies', 'n/a')}",
        f"- Baseline top-1 visible identities: {baseline.get('visible_c2_exclusive_identity_count', 0)}",
        f"- Experimental top-N visible identities: {experimental.get('visible_c2_exclusive_identity_count', 0)}",
        f"- Newly visible identities: {coverage.get('newly_visible_identity_count', 0)}",
        f"- Top-1 invisible identities: {coverage.get('top1_invisible_count', 0)}",
        f"- Baseline mapper-emitted visible identities: {baseline.get('mapper_emitted_visible_identity_count', 0)}",
        f"- Experimental mapper-emitted visible identities: {experimental.get('mapper_emitted_visible_identity_count', 0)}",
        f"- Baseline eligibility-passed visible identities: {baseline.get('eligibility_passed_visible_identity_count', 0)}",
        f"- Experimental eligibility-passed visible identities: {experimental.get('eligibility_passed_visible_identity_count', 0)}",
        f"- Root assessment: {root.get('summary', 'n/a')}",
        "",
        "## Deltas",
        f"- Mapper emission delta: {_safe_dict(summary.get('mapper_emission_delta')).get('delta', 0)}",
        f"- Eligibility delta: {_safe_dict(summary.get('eligibility_delta')).get('delta', 0)}",
        f"- Final selection delta: {_safe_dict(summary.get('final_selection_delta')).get('delta', 0)}",
    ]
    return "\n".join(lines).strip() + "\n"


def run_experimental_candidate_c_topn_preservation_report(
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

    summary = build_experimental_candidate_c_topn_preservation_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=experiment_summary,
        top_n_symbols=top_n_symbols,
        top_n_strategies=top_n_strategies,
    )
    markdown = render_experimental_candidate_c_topn_preservation_markdown(summary)

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
        description="Run experimental Candidate C2 top-N preservation coverage report."
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
    result = run_experimental_candidate_c_topn_preservation_report(
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
