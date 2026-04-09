from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.comparison_report_builder import build_comparison_report
from src.research.edge_selection_engine import run_edge_selection_engine
from src.research.edge_selection_input_mapper import map_edge_selection_input
from src.research.edge_stability_score_builder import build_edge_stability_scores
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
from src.research.score_drift_analyzer import build_score_drift_report

DEFAULT_JSON_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "candidate_c_mapper_exclusion_diagnosis_summary.json"
)
DEFAULT_MD_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "candidate_c_mapper_exclusion_diagnosis_summary.md"
)
DEFAULT_TRACE_DIR = DEFAULT_OUTPUT_DIR / "edge_selection_trace"

TARGET_HORIZONS = ("15m", "1h", "4h")
TOP_GROUP_LIMIT = 5
REPRESENTATIVE_IDENTITY_LIMIT = 12

DROP_STAGE_ORDER = (
    "absent_from_ranked_groups",
    "absent_from_preview_symbol",
    "absent_from_preview_strategy",
    "removed_by_comparison_collapse",
    "removed_by_mapper_seed_contract",
    "emitted_but_failed_eligibility",
    "ranked_but_not_selected",
    "selected",
)

COMPARISON_COLLAPSE_STAGES = {
    "absent_from_preview_symbol",
    "absent_from_preview_strategy",
    "removed_by_comparison_collapse",
}
MAPPER_CONTRACT_STAGES = {"removed_by_mapper_seed_contract"}
ELIGIBILITY_STAGES = {"emitted_but_failed_eligibility", "ranked_but_not_selected"}


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


def _build_directional_identities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "directional_label_distribution": {"up": 0, "down": 0},
                },
            )
            entry["directional_row_count"] += 1
            entry["directional_label_distribution"][label] += 1

    identities = list(grouped.values())
    identities.sort(
        key=lambda item: (
            -int(item["directional_row_count"]),
            str(item["symbol"]),
            str(item["strategy"]),
            str(item["horizon"]),
        )
    )
    return identities


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
        "rank": _safe_int(item.get("rank")),
        "score": _safe_float(item.get("score")),
        "sample_count": _safe_int(sample_count),
        "candidate_strength": candidate_strength,
        "quality_gate": quality_gate,
    }


def _build_rank_stage(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ranking = _safe_dict(_safe_dict(summary.get("strategy_lab")).get("ranking"))
    stage: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(ranking.get(horizon))
        symbol_groups = [
            _evaluate_ranked_group(entry)
            for entry in _extract_ranked_groups(horizon_payload.get("by_symbol"))
        ]
        strategy_groups = [
            _evaluate_ranked_group(entry)
            for entry in _extract_ranked_groups(horizon_payload.get("by_strategy"))
        ]
        stage[horizon] = {
            "ranked_symbol_groups": symbol_groups[:TOP_GROUP_LIMIT],
            "ranked_strategy_groups": strategy_groups[:TOP_GROUP_LIMIT],
            "all_ranked_symbol_groups": [
                str(_normalize_symbol(item.get("group")) or item.get("group"))
                for item in symbol_groups
                if _safe_text(item.get("group")) is not None
            ],
            "all_ranked_strategy_groups": [
                str(_normalize_strategy(item.get("group")) or item.get("group"))
                for item in strategy_groups
                if _safe_text(item.get("group")) is not None
            ],
        }

    return stage


def _build_preview_stage(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_horizon = _safe_dict(_safe_dict(summary.get("edge_candidates_preview")).get("by_horizon"))
    stage: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        horizon_payload = _safe_dict(by_horizon.get(horizon))
        stage[horizon] = {
            "preview_selected_symbol_group": _safe_dict(horizon_payload.get("top_symbol")),
            "preview_selected_strategy_group": _safe_dict(horizon_payload.get("top_strategy")),
        }

    return stage


def _build_comparison_stage(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comparison = _safe_dict(summary.get("edge_candidates_comparison"))
    stage: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        row = _safe_dict(comparison.get(horizon))
        stage[horizon] = {
            "comparison_preserved_symbol_group": _normalize_symbol(
                row.get("latest_top_symbol_group")
            ),
            "comparison_preserved_strategy_group": _normalize_strategy(
                row.get("latest_top_strategy_group")
            ),
            "latest_candidate_strength": _safe_text(row.get("latest_candidate_strength"))
            or "insufficient_data",
        }

    return stage


def _extract_horizon_diagnostics(
    mapped_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    seed_diagnostics = _safe_dict(mapped_payload.get("candidate_seed_diagnostics"))
    raw = seed_diagnostics.get("horizon_diagnostics")

    if isinstance(raw, list):
        items = [_safe_dict(item) for item in raw]
    elif isinstance(raw, dict):
        items = []
        for horizon, payload in raw.items():
            row = _safe_dict(payload)
            if _safe_text(row.get("horizon")) is None:
                row = {"horizon": horizon, **row}
            items.append(row)
    else:
        items = []

    output: dict[str, dict[str, Any]] = {}
    for item in items:
        horizon = _normalize_horizon(item.get("horizon"))
        if horizon is None:
            continue
        output[horizon] = item
    return output


def _build_mapper_stage(mapped_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapper_candidates = [_safe_dict(item) for item in _safe_list(mapped_payload.get("candidates"))]
    horizon_diagnostics = _extract_horizon_diagnostics(mapped_payload)

    stage: dict[str, dict[str, Any]] = {}
    for horizon in TARGET_HORIZONS:
        emitted = None
        for item in mapper_candidates:
            if _normalize_horizon(item.get("horizon")) == horizon:
                emitted = item
                break

        diag = _safe_dict(horizon_diagnostics.get(horizon))
        stage[horizon] = {
            "mapper_seed_inputs_used": {
                "horizon": horizon,
                "seed_generated": bool(diag.get("seed_generated")),
                "latest_top_symbol_group": _normalize_symbol(diag.get("latest_top_symbol_group")),
                "latest_top_strategy_group": _normalize_strategy(
                    diag.get("latest_top_strategy_group")
                ),
                "cumulative_top_symbol_group": _normalize_symbol(
                    diag.get("cumulative_top_symbol_group")
                ),
                "cumulative_top_strategy_group": _normalize_strategy(
                    diag.get("cumulative_top_strategy_group")
                ),
                "blocker_reasons": _safe_list(diag.get("blocker_reasons")),
            },
            "mapper_seed_candidate_emitted": emitted or {},
        }

    return stage


def _build_stage_traces(
    experiment_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    mapped_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rank_stage = _build_rank_stage(experiment_summary)
    preview_stage = _build_preview_stage(experiment_summary)
    comparison_stage = _build_comparison_stage(comparison_summary)
    mapper_stage = _build_mapper_stage(mapped_payload)

    traces: dict[str, dict[str, Any]] = {}
    for horizon in TARGET_HORIZONS:
        traces[horizon] = {
            "horizon": horizon,
            "ranked_symbol_groups": _safe_list(rank_stage.get(horizon, {}).get("ranked_symbol_groups")),
            "ranked_strategy_groups": _safe_list(
                rank_stage.get(horizon, {}).get("ranked_strategy_groups")
            ),
            "all_ranked_symbol_groups": _safe_list(
                rank_stage.get(horizon, {}).get("all_ranked_symbol_groups")
            ),
            "all_ranked_strategy_groups": _safe_list(
                rank_stage.get(horizon, {}).get("all_ranked_strategy_groups")
            ),
            "preview_selected_symbol_group": _safe_dict(
                preview_stage.get(horizon, {}).get("preview_selected_symbol_group")
            ),
            "preview_selected_strategy_group": _safe_dict(
                preview_stage.get(horizon, {}).get("preview_selected_strategy_group")
            ),
            "comparison_preserved_symbol_group": comparison_stage.get(horizon, {}).get(
                "comparison_preserved_symbol_group"
            ),
            "comparison_preserved_strategy_group": comparison_stage.get(horizon, {}).get(
                "comparison_preserved_strategy_group"
            ),
            "mapper_seed_inputs_used": _safe_dict(
                mapper_stage.get(horizon, {}).get("mapper_seed_inputs_used")
            ),
            "mapper_seed_candidate_emitted": _safe_dict(
                mapper_stage.get(horizon, {}).get("mapper_seed_candidate_emitted")
            ),
        }

    return traces


def _build_shadow_lookup(
    shadow_output: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in _safe_list(shadow_output.get("ranking")):
        row = _safe_dict(item)
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("strategy"))
        horizon = _normalize_horizon(row.get("horizon"))
        if symbol is None or strategy is None or horizon is None:
            continue
        lookup[(symbol, strategy, horizon)] = row
    return lookup


def _render_architecture_note() -> dict[str, Any]:
    return {
        "path_trace": [
            {
                "stage": "relabel_output",
                "module": "experimental candidate C2 dataset",
                "note": (
                    "Directional identities start as symbol + strategy + horizon "
                    "combinations from exclusive relabeled rows."
                ),
            },
            {
                "stage": "ranked_groups",
                "module": "research_analyzer strategy_lab ranking",
                "note": "Symbol and strategy groups are ranked independently per horizon.",
            },
            {
                "stage": "preview_selection",
                "module": "research_analyzer edge_candidates_preview",
                "note": (
                    "Only one preview-visible symbol group and one preview-visible "
                    "strategy group survive per horizon."
                ),
            },
            {
                "stage": "comparison_preservation",
                "module": "comparison_report_builder edge_candidates_comparison",
                "note": (
                    "Preview winners are copied into comparison-preserved symbol and "
                    "strategy groups."
                ),
            },
            {
                "stage": "mapper_seed_generation",
                "module": "edge_selection_input_mapper",
                "note": (
                    "Mapper seed generation converts the preserved groups into at most "
                    "one synthetic candidate identity per horizon."
                ),
            },
            {
                "stage": "eligibility_and_selection",
                "module": "edge_selection_engine",
                "note": "Only mapper-emitted candidates can fail eligibility or remain unselected.",
            },
        ]
    }


def _classify_identity(
    identity: dict[str, Any],
    *,
    stage_traces: dict[str, dict[str, Any]],
    shadow_lookup: dict[tuple[str, str, str], dict[str, Any]],
    shadow_output: dict[str, Any],
) -> dict[str, Any]:
    symbol = str(identity["symbol"])
    strategy = str(identity["strategy"])
    horizon = str(identity["horizon"])
    trace = _safe_dict(stage_traces.get(horizon))

    ranked_symbols = {
        str(item)
        for item in _safe_list(trace.get("all_ranked_symbol_groups"))
        if _safe_text(item) is not None
    }
    ranked_strategies = {
        str(item)
        for item in _safe_list(trace.get("all_ranked_strategy_groups"))
        if _safe_text(item) is not None
    }

    preview_symbol = _normalize_symbol(
        _safe_dict(trace.get("preview_selected_symbol_group")).get("group")
    )
    preview_strategy = _normalize_strategy(
        _safe_dict(trace.get("preview_selected_strategy_group")).get("group")
    )
    comparison_symbol = _normalize_symbol(trace.get("comparison_preserved_symbol_group"))
    comparison_strategy = _normalize_strategy(trace.get("comparison_preserved_strategy_group"))

    mapper_emitted = _safe_dict(trace.get("mapper_seed_candidate_emitted"))
    mapper_identity_match = (
        _normalize_symbol(mapper_emitted.get("symbol")) == symbol
        and _normalize_strategy(mapper_emitted.get("strategy")) == strategy
        and _normalize_horizon(mapper_emitted.get("horizon")) == horizon
    )

    ranking_item = shadow_lookup.get((symbol, strategy, horizon), {})
    candidate_status = _safe_text(_safe_dict(ranking_item).get("candidate_status"))
    selected = (
        _normalize_symbol(shadow_output.get("selected_symbol")) == symbol
        and _normalize_strategy(shadow_output.get("selected_strategy")) == strategy
        and _normalize_horizon(shadow_output.get("selected_horizon")) == horizon
    )

    preview_symbol_match = preview_symbol == symbol
    preview_strategy_match = preview_strategy == strategy

    if symbol not in ranked_symbols or strategy not in ranked_strategies:
        earliest_drop_stage = "absent_from_ranked_groups"
    elif not preview_symbol_match:
        earliest_drop_stage = "absent_from_preview_symbol"
    elif not preview_strategy_match:
        earliest_drop_stage = "absent_from_preview_strategy"
    elif comparison_symbol != symbol or comparison_strategy != strategy:
        earliest_drop_stage = "removed_by_comparison_collapse"
    elif not mapper_identity_match:
        earliest_drop_stage = "removed_by_mapper_seed_contract"
    elif candidate_status != "eligible":
        earliest_drop_stage = "emitted_but_failed_eligibility"
    elif not selected:
        earliest_drop_stage = "ranked_but_not_selected"
    else:
        earliest_drop_stage = "selected"

    return {
        **identity,
        "preview_symbol_match": preview_symbol_match,
        "preview_strategy_match": preview_strategy_match,
        "preview_constructed": preview_symbol_match and preview_strategy_match,
        "earliest_drop_stage": earliest_drop_stage,
        "appears_in_mapper": mapper_identity_match,
        "eligibility_passed": candidate_status == "eligible",
        "reaches_final_selection": selected,
        "stage_trace": {
            "preview_symbol_group": preview_symbol,
            "preview_strategy_group": preview_strategy,
            "comparison_symbol_group": comparison_symbol,
            "comparison_strategy_group": comparison_strategy,
            "mapper_emitted_symbol": _normalize_symbol(mapper_emitted.get("symbol")),
            "mapper_emitted_strategy": _normalize_strategy(mapper_emitted.get("strategy")),
        },
        "ranking_candidate": _safe_dict(ranking_item),
    }


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common()]


def _build_breakdown(
    identities: list[dict[str, Any]],
    *,
    key_name: str,
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for item in identities:
        key = str(item.get(key_name) or "unknown")
        stage = str(item.get("earliest_drop_stage") or "unknown")
        buckets[key][stage] += 1
    return {key: _counter_rows(counter) for key, counter in sorted(buckets.items())}


def _build_earliest_drop_stage_counts(
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter(
        str(item.get("earliest_drop_stage") or "unknown") for item in identities
    )
    ordered: list[dict[str, Any]] = []
    for stage in DROP_STAGE_ORDER:
        if counts.get(stage):
            ordered.append({"value": stage, "count": counts[stage]})
    return ordered


def _build_representative_dropped_identities(
    identities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dropped = [item for item in identities if item.get("earliest_drop_stage") != "selected"]
    dropped.sort(
        key=lambda item: (
            DROP_STAGE_ORDER.index(item["earliest_drop_stage"])
            if item["earliest_drop_stage"] in DROP_STAGE_ORDER
            else len(DROP_STAGE_ORDER),
            -int(item.get("directional_row_count", 0)),
            str(item.get("symbol")),
            str(item.get("strategy")),
            str(item.get("horizon")),
        )
    )

    rows: list[dict[str, Any]] = []
    for item in dropped[:REPRESENTATIVE_IDENTITY_LIMIT]:
        rows.append(
            {
                "identity_key": item["identity_key"],
                "symbol": item["symbol"],
                "strategy": item["strategy"],
                "horizon": item["horizon"],
                "directional_row_count": item["directional_row_count"],
                "earliest_drop_stage": item["earliest_drop_stage"],
                "stage_trace": item.get("stage_trace", {}),
            }
        )
    return rows


def _build_per_horizon_stage_trace_summary(
    stage_traces: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for horizon in TARGET_HORIZONS:
        trace = _safe_dict(stage_traces.get(horizon))
        mapper_candidate = _safe_dict(trace.get("mapper_seed_candidate_emitted"))
        summary[horizon] = {
            "ranked_symbol_group_count": len(_safe_list(trace.get("all_ranked_symbol_groups"))),
            "ranked_strategy_group_count": len(
                _safe_list(trace.get("all_ranked_strategy_groups"))
            ),
            "preview_selected_symbol_group": _normalize_symbol(
                _safe_dict(trace.get("preview_selected_symbol_group")).get("group")
            ),
            "preview_selected_strategy_group": _normalize_strategy(
                _safe_dict(trace.get("preview_selected_strategy_group")).get("group")
            ),
            "comparison_preserved_symbol_group": _normalize_symbol(
                trace.get("comparison_preserved_symbol_group")
            ),
            "comparison_preserved_strategy_group": _normalize_strategy(
                trace.get("comparison_preserved_strategy_group")
            ),
            "mapper_seed_generated": bool(
                _safe_dict(trace.get("mapper_seed_inputs_used")).get("seed_generated")
            ),
            "mapper_seed_candidate_emitted": {
                "symbol": _normalize_symbol(mapper_candidate.get("symbol")),
                "strategy": _normalize_strategy(mapper_candidate.get("strategy")),
                "horizon": _normalize_horizon(mapper_candidate.get("horizon")),
            },
        }

    return summary


def _build_root_bottleneck_assessment(
    identities: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_counts = Counter(
        str(item.get("earliest_drop_stage") or "unknown") for item in identities
    )
    comparison_collapse_count = sum(
        stage_counts.get(stage, 0) for stage in COMPARISON_COLLAPSE_STAGES
    )
    mapper_seed_contract_count = sum(
        stage_counts.get(stage, 0) for stage in MAPPER_CONTRACT_STAGES
    )
    eligibility_bottleneck_count = sum(
        stage_counts.get(stage, 0) for stage in ELIGIBILITY_STAGES
    )

    dominant_bottleneck = "comparison_collapse_bottleneck"
    dominant_count = comparison_collapse_count

    if mapper_seed_contract_count > dominant_count:
        dominant_bottleneck = "mapper_seed_contract_bottleneck"
        dominant_count = mapper_seed_contract_count

    if eligibility_bottleneck_count > dominant_count:
        dominant_bottleneck = "eligibility_bottleneck"

    return {
        "comparison_collapse_bottleneck_count": comparison_collapse_count,
        "mapper_seed_contract_bottleneck_count": mapper_seed_contract_count,
        "eligibility_bottleneck_count": eligibility_bottleneck_count,
        "dominant_bottleneck": dominant_bottleneck,
        "summary": (
            "The dominant bottleneck is comparison collapse: identities disappear while "
            "symbol and strategy groups are reduced to one winner per horizon."
            if dominant_bottleneck == "comparison_collapse_bottleneck"
            else (
                "The dominant bottleneck is mapper seed contract: identities survive "
                "comparison but disappear when only the synthetic preserved pair is emitted."
            )
            if dominant_bottleneck == "mapper_seed_contract_bottleneck"
            else (
                "The dominant bottleneck is eligibility: identities are emitted into the "
                "mapper but fail engine readiness or final selection."
            )
        ),
    }


def _write_stage_trace_artifacts(
    *,
    stage_trace_dir: Path,
    stage_traces: dict[str, dict[str, Any]],
) -> None:
    stage_trace_dir.mkdir(parents=True, exist_ok=True)

    for horizon in TARGET_HORIZONS:
        _write_json(
            stage_trace_dir / f"{horizon}_stage_trace.json",
            _safe_dict(stage_traces.get(horizon)),
        )

    _write_json(
        stage_trace_dir / "stage_trace_summary.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "horizons": stage_traces,
        },
    )


def build_experimental_candidate_c_mapper_exclusion_diagnosis_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    experiment_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    mapped_payload: dict[str, Any],
    shadow_output: dict[str, Any],
    trace_dir: Path,
    stage_traces: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _, experiment_shared_rows, intersection_overview = build_intersection_datasets(
        baseline_rows,
        experiment_rows,
    )
    exclusive_rows = _build_exclusive_rows(experiment_rows, experiment_shared_rows)
    identities = _build_directional_identities(exclusive_rows)
    resolved_stage_traces = stage_traces or _build_stage_traces(
        experiment_summary,
        comparison_summary,
        mapped_payload,
    )
    shadow_lookup = _build_shadow_lookup(shadow_output)

    diagnostics = [
        _classify_identity(
            identity,
            stage_traces=resolved_stage_traces,
            shadow_lookup=shadow_lookup,
            shadow_output=shadow_output,
        )
        for identity in identities
    ]

    mapper_emitted_count = sum(1 for item in diagnostics if item["appears_in_mapper"])
    preview_constructed_count = sum(1 for item in diagnostics if item["preview_constructed"])
    eligibility_passed_count = sum(1 for item in diagnostics if item["eligibility_passed"])
    final_selection_count = sum(1 for item in diagnostics if item["reaches_final_selection"])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "experimental_candidate_c_mapper_exclusion_diagnosis_report",
        "architecture_note": _render_architecture_note(),
        "trace_dir": str(trace_dir),
        "stage_trace_dir": str(trace_dir / "stage_traces"),
        "total_exclusive_rows": len(exclusive_rows),
        "total_exclusive_identities": len(diagnostics),
        "mapper_emitted_count": mapper_emitted_count,
        "preview_constructed_count": preview_constructed_count,
        "eligibility_passed_count": eligibility_passed_count,
        "final_selection_count": final_selection_count,
        "earliest_drop_stage_counts": _build_earliest_drop_stage_counts(diagnostics),
        "per_horizon_stage_trace_summary": _build_per_horizon_stage_trace_summary(
            resolved_stage_traces
        ),
        "representative_dropped_identities": _build_representative_dropped_identities(
            diagnostics
        ),
        "rejection_breakdown_by_symbol": _build_breakdown(diagnostics, key_name="symbol"),
        "rejection_breakdown_by_strategy": _build_breakdown(
            diagnostics,
            key_name="strategy",
        ),
        "rejection_breakdown_by_horizon": _build_breakdown(
            diagnostics,
            key_name="horizon",
        ),
        "root_bottleneck_assessment": _build_root_bottleneck_assessment(diagnostics),
        "intersection_overview": intersection_overview,
        "stage_traces": resolved_stage_traces,
        "identity_diagnostics": diagnostics,
    }


def render_experimental_candidate_c_mapper_exclusion_diagnosis_markdown(
    summary: dict[str, Any],
) -> str:
    root = _safe_dict(summary.get("root_bottleneck_assessment"))
    lines = [
        "# Candidate C2 Mapper Exclusion Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Total C2-exclusive rows: {summary.get('total_exclusive_rows', 0)}",
        f"- Total C2-exclusive directional identities: {summary.get('total_exclusive_identities', 0)}",
        f"- Preview constructed count: {summary.get('preview_constructed_count', 0)}",
        f"- Mapper emitted count: {summary.get('mapper_emitted_count', 0)}",
        f"- Eligibility passed count: {summary.get('eligibility_passed_count', 0)}",
        f"- Final selection count: {summary.get('final_selection_count', 0)}",
        f"- Dominant bottleneck: {root.get('dominant_bottleneck', 'unknown')}",
        f"- Summary: {root.get('summary', 'n/a')}",
        "",
        "## Earliest Drop Stage Counts",
    ]

    for row in _safe_list(summary.get("earliest_drop_stage_counts")):
        item = _safe_dict(row)
        lines.append(f"- {item.get('value', 'unknown')}: {item.get('count', 0)}")

    lines.extend(["", "## Per-Horizon Stage Trace Summary"])
    per_horizon = _safe_dict(summary.get("per_horizon_stage_trace_summary"))
    for horizon in TARGET_HORIZONS:
        row = _safe_dict(per_horizon.get(horizon))
        lines.append(
            f"- {horizon}: "
            f"ranked_symbol_groups={row.get('ranked_symbol_group_count', 0)}, "
            f"ranked_strategy_groups={row.get('ranked_strategy_group_count', 0)}, "
            f"preview_symbol={row.get('preview_selected_symbol_group') or 'n/a'}, "
            f"preview_strategy={row.get('preview_selected_strategy_group') or 'n/a'}, "
            f"comparison_symbol={row.get('comparison_preserved_symbol_group') or 'n/a'}, "
            f"comparison_strategy={row.get('comparison_preserved_strategy_group') or 'n/a'}, "
            f"seed_generated={row.get('mapper_seed_generated', False)}, "
            f"emitted_symbol={_safe_dict(row.get('mapper_seed_candidate_emitted')).get('symbol') or 'n/a'}, "
            f"emitted_strategy={_safe_dict(row.get('mapper_seed_candidate_emitted')).get('strategy') or 'n/a'}"
        )

    lines.extend(
        [
            "",
            "## Root Bottleneck Assessment",
            f"- Comparison collapse bottleneck count: {root.get('comparison_collapse_bottleneck_count', 0)}",
            f"- Mapper seed contract bottleneck count: {root.get('mapper_seed_contract_bottleneck_count', 0)}",
            f"- Eligibility bottleneck count: {root.get('eligibility_bottleneck_count', 0)}",
            f"- Dominant bottleneck: {root.get('dominant_bottleneck', 'unknown')}",
            f"- Summary: {root.get('summary', 'n/a')}",
            "",
            "## Expected Outputs",
            f"- JSON summary: {DEFAULT_JSON_OUTPUT}",
            f"- Markdown summary: {DEFAULT_MD_OUTPUT}",
            f"- Stage trace dir: {summary.get('stage_trace_dir', 'n/a')}",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def _prepare_edge_selection_trace(
    *,
    baseline_dataset_path: Path,
    experiment_dataset_path: Path,
    trace_dir: Path,
) -> dict[str, Any]:
    cumulative_dir = trace_dir / "cumulative"
    latest_dir = trace_dir / "latest"
    comparison_dir = trace_dir / "comparison"
    edge_scores_dir = trace_dir / "edge_scores"
    score_drift_dir = trace_dir / "score_drift"

    baseline_summary = run_research_analyzer(
        input_path=baseline_dataset_path,
        output_dir=cumulative_dir,
    )
    experiment_summary = run_research_analyzer(
        input_path=experiment_dataset_path,
        output_dir=latest_dir,
    )

    comparison_summary = build_comparison_report(
        latest_summary_path=latest_dir / "summary.json",
        cumulative_summary_path=cumulative_dir / "summary.json",
        output_dir=comparison_dir,
    )
    edge_scores_summary = build_edge_stability_scores(
        input_path=comparison_dir / "summary.json",
        output_dir=edge_scores_dir,
    )
    build_score_drift_report(
        input_path=trace_dir / "edge_scores_history.jsonl",
        output_dir=score_drift_dir,
    )

    mapped_payload = map_edge_selection_input(trace_dir)
    shadow_output = run_edge_selection_engine(mapped_payload)
    stage_traces = _build_stage_traces(
        experiment_summary,
        comparison_summary,
        mapped_payload,
    )

    _write_json(trace_dir / "mapper_payload.json", mapped_payload)
    _write_json(trace_dir / "shadow_output.json", shadow_output)
    _write_stage_trace_artifacts(
        stage_trace_dir=trace_dir / "stage_traces",
        stage_traces=stage_traces,
    )

    return {
        "baseline_summary": baseline_summary,
        "experiment_summary": experiment_summary,
        "comparison_summary": comparison_summary,
        "edge_scores_summary": edge_scores_summary,
        "mapped_payload": mapped_payload,
        "shadow_output": shadow_output,
        "stage_traces": stage_traces,
    }


def run_experimental_candidate_c_mapper_exclusion_diagnosis_report(
    *,
    baseline_dataset_path: Path = CANDIDATE_A_DEFAULT_PATH,
    experiment_dataset_path: Path = DEFAULT_CANDIDATE_C_DATASET,
    output_json_path: Path = DEFAULT_JSON_OUTPUT,
    output_md_path: Path = DEFAULT_MD_OUTPUT,
    trace_dir: Path = DEFAULT_TRACE_DIR,
) -> dict[str, Any]:
    baseline_rows, _ = load_jsonl_records(baseline_dataset_path)
    experiment_loaded_rows, _ = load_jsonl_records(experiment_dataset_path)
    experiment_rows = filter_candidate_c_records(experiment_loaded_rows)

    trace = _prepare_edge_selection_trace(
        baseline_dataset_path=baseline_dataset_path,
        experiment_dataset_path=experiment_dataset_path,
        trace_dir=trace_dir,
    )
    summary = build_experimental_candidate_c_mapper_exclusion_diagnosis_summary(
        baseline_rows=baseline_rows,
        experiment_rows=experiment_rows,
        experiment_summary=trace["experiment_summary"],
        comparison_summary=trace["comparison_summary"],
        mapped_payload=trace["mapped_payload"],
        shadow_output=trace["shadow_output"],
        trace_dir=trace_dir,
        stage_traces=trace["stage_traces"],
    )
    markdown = render_experimental_candidate_c_mapper_exclusion_diagnosis_markdown(summary)

    _write_json(output_json_path, summary)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown, encoding="utf-8")

    return {
        "baseline_dataset_path": str(baseline_dataset_path),
        "experiment_dataset_path": str(experiment_dataset_path),
        "trace_dir": str(trace_dir),
        "json_output_path": str(output_json_path),
        "md_output_path": str(output_md_path),
        "summary": summary,
        "markdown": markdown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why C2-exclusive directional identities never reach mapper output."
    )
    parser.add_argument("--baseline-dataset", type=Path, default=CANDIDATE_A_DEFAULT_PATH)
    parser.add_argument("--experiment-dataset", type=Path, default=DEFAULT_CANDIDATE_C_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experimental_candidate_c_mapper_exclusion_diagnosis_report(
        baseline_dataset_path=args.baseline_dataset,
        experiment_dataset_path=args.experiment_dataset,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
        trace_dir=args.trace_dir,
    )
    print(
        json.dumps(
            {
                "json_output_path": result["json_output_path"],
                "md_output_path": result["md_output_path"],
                "trace_dir": result["trace_dir"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
