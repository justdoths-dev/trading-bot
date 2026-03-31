from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from src.research.research_analyzer import (
    HORIZONS,
    _build_edge_candidate_rows,
    _build_edge_candidates_preview,
    _build_edge_stability_preview,
    _build_strategy_lab_metrics,
)
from src.research.strategy_lab.dataset_builder import build_dataset


def _default_input_path() -> Path:
    return Path(__file__).resolve().parents[2] / "logs" / "trade_analysis.jsonl"


def _default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "logs"
        / "research_reports"
        / "latest"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose recent-window visibility by explicitly separating "
            "target-only recovery signals from full-context joined-candidate survival."
        )
    )
    parser.add_argument("--input", type=Path, default=_default_input_path())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--windows", type=int, nargs="+", default=[100, 150, 200, 250])
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["BTCUSDT:swing", "ETHUSDT:swing"],
    )
    parser.add_argument(
        "--anchor-horizon",
        choices=list(HORIZONS),
        default="4h",
    )
    parser.add_argument("--keep-temp-files", action="store_true")
    return parser.parse_args()


def _normalize_symbol(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().upper()
        return text or None
    return None


def _normalize_strategy(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().lower()
        return text or None
    return None


def _normalize_label(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"up", "down", "flat"}:
            return text
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100.0, 2)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 6)


def _avg_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]

    if isinstance(value, set):
        return [_json_safe_value(item) for item in sorted(value, key=str)]

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe_value(v) for k, v in row.items()}


def _parse_targets(raw_targets: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []

    for item in raw_targets:
        if ":" not in item:
            continue

        symbol_raw, strategy_raw = item.split(":", 1)
        symbol = _normalize_symbol(symbol_raw)
        strategy = _normalize_strategy(strategy_raw)
        if symbol and strategy:
            parsed.append((symbol, strategy))

    deduped: list[tuple[str, str]] = []
    for item in parsed:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _filter_recent_labeled_row_indices(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    horizon: str,
    window: int,
) -> list[int]:
    label_key = f"future_label_{horizon}"
    labeled = [
        index
        for index, row in indexed_rows
        if _normalize_label(row.get(label_key)) is not None
    ]
    return labeled[-window:] if window > 0 else labeled


def _extract_logged_at(row: dict[str, Any]) -> str | None:
    value = row.get("logged_at")
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _filter_identity_rows(
    dataset: list[dict[str, Any]],
    *,
    symbol: str,
    strategy: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in dataset
        if _normalize_symbol(row.get("symbol")) == symbol
        and _normalize_strategy(row.get("selected_strategy")) == strategy
    ]


def _build_full_context_recent_subset(
    dataset: list[dict[str, Any]],
    *,
    targets: list[tuple[str, str]],
    anchor_horizon: str,
    labeled_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build a recent-window dataset while preserving full-category context.

    Strategy:
    - For each target identity, find recent labeled rows for the anchor horizon.
    - Collect their original row indices from the full dataset.
    - Use the earliest selected row index as the cutoff.
    - Keep ALL rows from cutoff_index to the dataset tail.

    This produces a strict full-context subset suitable for analyzer/joined survival
    diagnosis without relying on logged_at preservation.
    """
    selected_anchor_indices: list[int] = []

    for symbol, strategy in targets:
        indexed_rows = [
            (index, row)
            for index, row in enumerate(dataset)
            if _normalize_symbol(row.get("symbol")) == symbol
            and _normalize_strategy(row.get("selected_strategy")) == strategy
        ]
        recent_indices = _filter_recent_labeled_row_indices(
            indexed_rows=indexed_rows,
            horizon=anchor_horizon,
            window=labeled_window,
        )
        selected_anchor_indices.extend(recent_indices)

    if not selected_anchor_indices:
        return [], {
            "subset_mode": "empty",
            "cutoff_row_index": None,
            "cutoff_logged_at": None,
            "selected_anchor_row_count": 0,
            "full_context_row_count": 0,
        }

    cutoff_row_index = min(selected_anchor_indices)
    subset = dataset[cutoff_row_index:]

    cutoff_logged_at = None
    if 0 <= cutoff_row_index < len(dataset):
        cutoff_logged_at = _extract_logged_at(dataset[cutoff_row_index])

    return subset, {
        "subset_mode": "full_context_row_index_cutoff",
        "cutoff_row_index": cutoff_row_index,
        "cutoff_logged_at": cutoff_logged_at,
        "selected_anchor_row_count": len(selected_anchor_indices),
        "full_context_row_count": len(subset),
    }


def _write_subset(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            safe_row = _json_safe_row(row)
            f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")


def _build_metrics_from_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    metric_mode: str,
) -> dict[str, Any]:
    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"

    label_counter: Counter[str] = Counter()
    returns: list[float] = []

    for row in rows:
        label = _normalize_label(row.get(label_key))
        if label is None:
            continue

        label_counter[label] += 1

        future_return = _to_float(row.get(return_key))
        if future_return is not None:
            returns.append(future_return)

    labeled_count = sum(label_counter.values())
    up_count = label_counter.get("up", 0)
    down_count = label_counter.get("down", 0)
    flat_count = label_counter.get("flat", 0)

    directional_dominance_pct = None
    if labeled_count > 0:
        directional_dominance_pct = max(
            _pct(up_count, labeled_count) or 0.0,
            _pct(down_count, labeled_count) or 0.0,
        )

    return {
        "metric_mode": metric_mode,
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "row_count": len(rows),
        "subset_labeled_count": labeled_count,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "up_rate_pct": _pct(up_count, labeled_count),
        "down_rate_pct": _pct(down_count, labeled_count),
        "flat_rate_pct": _pct(flat_count, labeled_count),
        "directional_dominance_pct": directional_dominance_pct,
        "avg_future_return_pct": _avg_or_none(returns),
        "median_future_return_pct": _median_or_none(returns),
    }


def _build_target_only_recent_metrics(
    dataset: list[dict[str, Any]],
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    labeled_window: int,
) -> dict[str, Any]:
    target_rows = _filter_identity_rows(dataset, symbol=symbol, strategy=strategy)
    label_key = f"future_label_{horizon}"

    labeled_target_rows = [
        row
        for row in target_rows
        if _normalize_label(row.get(label_key)) is not None
    ]
    recent_rows = labeled_target_rows[-labeled_window:] if labeled_window > 0 else labeled_target_rows

    metrics = _build_metrics_from_rows(
        recent_rows,
        symbol=symbol,
        strategy=strategy,
        horizon=horizon,
        metric_mode="target_only_recent_labeled_rows",
    )
    metrics["target_row_count_total"] = len(target_rows)
    metrics["available_labeled_total"] = len(labeled_target_rows)
    return metrics


def _build_subset_context_target_metrics(
    subset: list[dict[str, Any]],
    *,
    symbol: str,
    strategy: str,
    horizon: str,
) -> dict[str, Any]:
    subset_target_rows = _filter_identity_rows(subset, symbol=symbol, strategy=strategy)
    metrics = _build_metrics_from_rows(
        subset_target_rows,
        symbol=symbol,
        strategy=strategy,
        horizon=horizon,
        metric_mode="full_context_subset_target_rows",
    )
    metrics["subset_target_row_count_total"] = len(subset_target_rows)
    return metrics


def _summarize(rows_block: dict[str, Any]) -> dict[str, Any]:
    rows = rows_block.get("rows", []) if isinstance(rows_block.get("rows"), list) else []

    symbol_counter: Counter[str] = Counter()
    strategy_counter: Counter[str] = Counter()
    horizon_counter: Counter[str] = Counter()
    stability_counter: Counter[str] = Counter()

    for row in rows:
        symbol_counter[str(row.get("symbol", "n/a"))] += 1
        strategy_counter[str(row.get("strategy", "n/a"))] += 1
        horizon_counter[str(row.get("horizon", "n/a"))] += 1
        stability_counter[str(row.get("selected_stability_label", "n/a"))] += 1

    return {
        "row_count": rows_block.get("row_count", 0),
        "rows": rows,
        "symbol_counter": dict(symbol_counter),
        "strategy_counter": dict(strategy_counter),
        "horizon_counter": dict(horizon_counter),
        "stability_counter": dict(stability_counter),
        "identity_horizon_evaluations": rows_block.get(
            "identity_horizon_evaluations", []
        ),
    }


def _extract_preview_summary(
    edge_candidates_preview: dict[str, Any],
    edge_stability_preview: dict[str, Any],
) -> dict[str, Any]:
    by_horizon = edge_candidates_preview.get("by_horizon", {}) or {}

    raw_preview_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_data = by_horizon.get(horizon, {}) or {}
        raw_preview_by_horizon[horizon] = {
            "top_symbol": horizon_data.get("top_symbol"),
            "top_strategy": horizon_data.get("top_strategy"),
            "top_alignment_state": horizon_data.get("top_alignment_state"),
            "sample_gate": horizon_data.get("sample_gate"),
            "quality_gate": horizon_data.get("quality_gate"),
            "candidate_strength": horizon_data.get("candidate_strength"),
            "visibility_reason": horizon_data.get("visibility_reason"),
        }

    return {
        "raw_preview_by_horizon": raw_preview_by_horizon,
        "stability_preview": edge_stability_preview,
    }


def _scope_label(horizons: list[str]) -> str:
    if len(horizons) >= 2:
        return "multi_horizon"
    if len(horizons) == 1:
        return "single_horizon"
    return "empty"


def _extract_identity_entry(
    joined_candidate_rows: dict[str, Any],
    identity_key: str,
) -> dict[str, Any] | None:
    entries = joined_candidate_rows.get("identity_horizon_evaluations", [])
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if str(entry.get("identity_key")) == identity_key:
            return entry
    return None


def _build_recovery_vs_survival_summary(
    *,
    identity_key: str,
    identity_entry: dict[str, Any] | None,
    target_only_metrics: dict[str, Any],
    subset_context_metrics: dict[str, Any],
) -> dict[str, Any]:
    joined_horizons = []
    joined_stability_label = None
    rejection_reasons: dict[str, str] = {}

    if identity_entry:
        joined_horizons = identity_entry.get("actual_joined_eligible_horizons", []) or []
        joined_stability_label = identity_entry.get("actual_joined_stability_label")
        horizon_evaluations = identity_entry.get("horizon_evaluations", {}) or {}
        for horizon in HORIZONS:
            evaluation = horizon_evaluations.get(horizon, {}) or {}
            if evaluation.get("status") == "selected":
                rejection_reasons[horizon] = "selected"
            else:
                rejection_reasons[horizon] = str(
                    evaluation.get("rejection_reason") or "unknown"
                )
    else:
        joined_stability_label = "identity_not_present_in_evaluations"
        rejection_reasons = {h: "identity_not_present_in_evaluations" for h in HORIZONS}

    horizon_summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        target_metric = target_only_metrics.get(horizon, {}) or {}
        subset_metric = subset_context_metrics.get(horizon, {}) or {}

        target_labeled = target_metric.get("subset_labeled_count")
        subset_labeled = subset_metric.get("subset_labeled_count")
        target_median = target_metric.get("median_future_return_pct")
        subset_median = subset_metric.get("median_future_return_pct")
        target_dom = target_metric.get("directional_dominance_pct")
        subset_dom = subset_metric.get("directional_dominance_pct")

        local_recovery_signal = (
            (target_labeled or 0) > 0 and (
                target_median is not None or target_dom is not None
            )
        )
        subset_signal_present = (
            (subset_labeled or 0) > 0 and (
                subset_median is not None or subset_dom is not None
            )
        )
        survives_joined = horizon in joined_horizons

        if survives_joined:
            diagnosis = "local_recovery_and_joined_survival"
        elif local_recovery_signal and subset_signal_present:
            diagnosis = "local_recovery_present_but_joined_survival_failed"
        elif local_recovery_signal and not subset_signal_present:
            diagnosis = "target_only_recovery_present_but_subset_context_thin_or_empty"
        else:
            diagnosis = "no_meaningful_local_recovery_signal"

        horizon_summary[horizon] = {
            "target_only_recent_metrics": target_metric,
            "subset_context_target_metrics": subset_metric,
            "survives_joined": survives_joined,
            "joined_status_or_rejection_reason": rejection_reasons.get(horizon),
            "diagnosis": diagnosis,
        }

    return {
        "identity_key": identity_key,
        "actual_joined_eligible_horizons": joined_horizons,
        "actual_joined_stability_label": joined_stability_label,
        "by_horizon": horizon_summary,
    }


def _build_interpretation(entry: dict[str, Any]) -> list[str]:
    raw_preview_visibility = entry.get("raw_preview_visibility", {}) or {}
    compatibility_filtered_preview_visibility = (
        entry.get("compatibility_filtered_preview_visibility", {}) or {}
    )

    raw_union = raw_preview_visibility.get("raw_category_union_horizons", []) or []
    raw_overlap = raw_preview_visibility.get("raw_category_overlap_horizons", []) or []
    compat_union = (
        compatibility_filtered_preview_visibility.get(
            "compatibility_filtered_category_union_horizons", []
        )
        or []
    )
    compat_overlap = (
        compatibility_filtered_preview_visibility.get(
            "compatibility_filtered_category_overlap_horizons", []
        )
        or []
    )
    joined = entry.get("actual_joined_eligible_horizons", []) or []

    horizon_evaluations = entry.get("horizon_evaluations", {}) or {}
    rejection_counter: Counter[str] = Counter()
    for horizon in HORIZONS:
        evaluation = horizon_evaluations.get(horizon, {}) or {}
        if evaluation.get("status") == "selected":
            continue
        reason = str(evaluation.get("rejection_reason") or "unknown")
        rejection_counter[reason] += 1

    dominant_rejections = ", ".join(
        f"{reason}={count}" for reason, count in rejection_counter.most_common()
    ) or "none"

    lines = [
        f"- raw_preview_union_horizons: {raw_union} ({_scope_label(raw_union)})",
        f"- raw_preview_overlap_horizons: {raw_overlap} ({_scope_label(raw_overlap)})",
        f"- compatibility_filtered_union_horizons: {compat_union} ({_scope_label(compat_union)})",
        f"- compatibility_filtered_overlap_horizons: {compat_overlap} ({_scope_label(compat_overlap)})",
        f"- actual_joined_eligible_horizons: {joined} ({_scope_label(joined)})",
        f"- dominant_rejection_reasons: {dominant_rejections}",
    ]

    if len(raw_union) > len(compat_union):
        lines.append(
            "- interpretation_note: raw preview breadth shrank after strategy-horizon compatibility filtering."
        )

    if len(compat_union) > len(joined):
        lines.append(
            "- interpretation_note: compatibility-filtered preview still exceeded actual joined eligibility because some compatible horizons failed candidate-quality requirements."
        )

    if len(joined) == 1:
        lines.append(
            "- interpretation_note: actual joined visibility collapsed to a single surviving horizon."
        )
    elif len(joined) == 0:
        lines.append(
            "- interpretation_note: no joined candidate survived in this recent window."
        )

    return lines


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Recent Window Candidate Visibility Diagnosis v2")
    lines.append("")
    lines.append(
        "This report explicitly separates target-only recovery signals from full-context joined-candidate survival."
    )
    lines.append("")
    lines.append(f"- generated_at: {payload.get('generated_at')}")
    lines.append(f"- input_path: {payload.get('input_path')}")
    lines.append(f"- dataset_rows: {payload.get('dataset_rows')}")
    lines.append(f"- anchor_horizon: {payload.get('anchor_horizon')}")
    lines.append(f"- targets: {payload.get('targets')}")
    lines.append("")

    for snapshot in payload.get("windows", []):
        joined = snapshot.get("joined_candidate_rows", {}) or {}
        preview = snapshot.get("preview_summary", {}) or {}
        raw_preview_by_horizon = preview.get("raw_preview_by_horizon", {}) or {}
        target_only_metrics = snapshot.get("target_only_recent_metrics", {}) or {}
        subset_context_metrics = snapshot.get("subset_context_target_metrics", {}) or {}
        recovery_vs_survival = snapshot.get("recovery_vs_survival", {}) or {}
        identity_horizon_evaluations = (
            joined.get("identity_horizon_evaluations", [])
            if isinstance(joined.get("identity_horizon_evaluations"), list)
            else []
        )
        subset_metadata = snapshot.get("subset_metadata", {}) or {}

        lines.append(f"## Window {snapshot.get('labeled_window')}")
        lines.append("")
        lines.append(f"- subset_dataset_rows: {snapshot.get('subset_dataset_rows', 0)}")
        lines.append(f"- subset_mode: {subset_metadata.get('subset_mode')}")
        lines.append(f"- cutoff_row_index: {subset_metadata.get('cutoff_row_index')}")
        lines.append(f"- cutoff_logged_at: {subset_metadata.get('cutoff_logged_at')}")
        lines.append(
            f"- selected_anchor_row_count: {subset_metadata.get('selected_anchor_row_count')}"
        )
        lines.append(
            f"- full_context_row_count: {subset_metadata.get('full_context_row_count')}"
        )
        lines.append(f"- row_count: {joined.get('row_count', 0)}")
        lines.append(f"- symbol_counter: {joined.get('symbol_counter', {})}")
        lines.append(f"- strategy_counter: {joined.get('strategy_counter', {})}")
        lines.append(f"- horizon_counter: {joined.get('horizon_counter', {})}")
        lines.append(f"- stability_counter: {joined.get('stability_counter', {})}")
        lines.append("")

        lines.append("### Joined Candidate Rows")
        lines.append("")
        rows = joined.get("rows", [])
        if rows:
            for row in rows[:20]:
                lines.append(
                    "- "
                    f"{row.get('symbol')} / {row.get('strategy')} / {row.get('horizon')} "
                    f"(strength={row.get('selected_candidate_strength')}, "
                    f"actual_joined_eligible_horizons={row.get('actual_joined_eligible_horizons')}, "
                    f"raw_preview_symbol={row.get('preview_symbol_visible_horizons')}, "
                    f"raw_preview_strategy={row.get('preview_strategy_visible_horizons')}, "
                    f"compatibility_preview_symbol={row.get('compatibility_preview_symbol_visible_horizons')}, "
                    f"compatibility_preview_strategy={row.get('compatibility_preview_strategy_visible_horizons')}, "
                    f"aggregate_score={row.get('aggregate_score')}, "
                    f"visibility_reason={row.get('visibility_reason')})"
                )
        else:
            lines.append("No joined candidate rows.")
        lines.append("")

        lines.append("### Raw Preview By Horizon")
        lines.append("")
        for horizon in HORIZONS:
            horizon_preview = raw_preview_by_horizon.get(horizon, {}) or {}
            lines.append(f"#### {horizon}")
            lines.append(f"- sample_gate: {horizon_preview.get('sample_gate')}")
            lines.append(f"- quality_gate: {horizon_preview.get('quality_gate')}")
            lines.append(f"- candidate_strength: {horizon_preview.get('candidate_strength')}")
            lines.append(f"- visibility_reason: {horizon_preview.get('visibility_reason')}")
            lines.append(f"- top_symbol: {horizon_preview.get('top_symbol')}")
            lines.append(f"- top_strategy: {horizon_preview.get('top_strategy')}")
            lines.append(f"- top_alignment_state: {horizon_preview.get('top_alignment_state')}")
            lines.append("")

        lines.append("### Stability Preview")
        lines.append("")
        stability_preview = preview.get("stability_preview", {}) or {}
        for label in ("symbol", "strategy", "alignment_state"):
            entry = stability_preview.get(label, {}) or {}
            lines.append(f"#### {label}")
            lines.append(f"- group: {entry.get('group')}")
            lines.append(f"- visible_horizons: {entry.get('visible_horizons')}")
            lines.append(f"- stability_label: {entry.get('stability_label')}")
            lines.append(f"- stability_score: {entry.get('stability_score')}")
            lines.append(f"- visibility_reason: {entry.get('visibility_reason')}")
            lines.append("")

        lines.append("### Identity Visibility Diagnostics")
        lines.append("")
        if identity_horizon_evaluations:
            for entry in identity_horizon_evaluations:
                raw_preview_visibility = entry.get("raw_preview_visibility", {}) or {}
                compatibility_filtered_preview_visibility = (
                    entry.get("compatibility_filtered_preview_visibility", {}) or {}
                )
                horizon_evaluations = entry.get("horizon_evaluations", {}) or {}

                lines.append(f"#### {entry.get('identity_key')}")
                lines.append("**Interpretation Summary**")
                lines.extend(_build_interpretation(entry))
                lines.append("")

                lines.append("**Visibility Layers**")
                lines.append(
                    "- raw_preview_visibility: "
                    f"symbol={raw_preview_visibility.get('symbol')}, "
                    f"strategy={raw_preview_visibility.get('strategy')}, "
                    f"union={raw_preview_visibility.get('raw_category_union_horizons')}, "
                    f"overlap={raw_preview_visibility.get('raw_category_overlap_horizons')}, "
                    f"quality_passed_overlap={raw_preview_visibility.get('raw_category_quality_passed_overlap_horizons')}"
                )
                lines.append(
                    "- compatibility_filtered_preview_visibility: "
                    f"symbol={compatibility_filtered_preview_visibility.get('symbol')}, "
                    f"strategy={compatibility_filtered_preview_visibility.get('strategy')}, "
                    f"union={compatibility_filtered_preview_visibility.get('compatibility_filtered_category_union_horizons')}, "
                    f"overlap={compatibility_filtered_preview_visibility.get('compatibility_filtered_category_overlap_horizons')}, "
                    f"quality_passed_overlap={compatibility_filtered_preview_visibility.get('compatibility_filtered_quality_passed_overlap_horizons')}, "
                    f"compatible_horizons={compatibility_filtered_preview_visibility.get('strategy_compatible_horizons')}"
                )
                lines.append(
                    "- actual_joined_eligible_horizons: "
                    f"{entry.get('actual_joined_eligible_horizons')} "
                    f"({entry.get('actual_joined_stability_label')})"
                )
                lines.append("")

                lines.append("**Horizon Decisions**")
                for horizon in HORIZONS:
                    evaluation = horizon_evaluations.get(horizon, {}) or {}
                    lines.append(
                        "- "
                        f"{horizon}: status={evaluation.get('status')}, "
                        f"reason={evaluation.get('rejection_reason') or 'selected'}, "
                        f"reasons={evaluation.get('rejection_reasons')}, "
                        f"candidate_strength={evaluation.get('candidate_strength')}, "
                        f"sample_gate={evaluation.get('sample_gate')}, "
                        f"quality_gate={evaluation.get('quality_gate')}, "
                        f"aggregate_score={evaluation.get('aggregate_score')}"
                    )
                lines.append("")
        else:
            lines.append("No identity visibility diagnostics available.")
            lines.append("")

        lines.append("### Recovery vs Survival")
        lines.append("")
        for identity_key, identity_summary in recovery_vs_survival.items():
            lines.append(f"#### {identity_key}")
            lines.append(
                f"- actual_joined_eligible_horizons: {identity_summary.get('actual_joined_eligible_horizons')}"
            )
            lines.append(
                f"- actual_joined_stability_label: {identity_summary.get('actual_joined_stability_label')}"
            )
            lines.append("")

            by_horizon = identity_summary.get("by_horizon", {}) or {}
            for horizon in HORIZONS:
                horizon_summary = by_horizon.get(horizon, {}) or {}
                target_metric = horizon_summary.get("target_only_recent_metrics", {}) or {}
                subset_metric = horizon_summary.get("subset_context_target_metrics", {}) or {}

                lines.append(f"- {horizon}:")
                lines.append(
                    f"  - diagnosis: {horizon_summary.get('diagnosis')}"
                )
                lines.append(
                    f"  - survives_joined: {horizon_summary.get('survives_joined')}"
                )
                lines.append(
                    f"  - joined_status_or_rejection_reason: {horizon_summary.get('joined_status_or_rejection_reason')}"
                )
                lines.append(
                    "  - target_only_recent_metrics: "
                    f"labeled_count={target_metric.get('subset_labeled_count')}, "
                    f"dominance_pct={target_metric.get('directional_dominance_pct')}, "
                    f"avg_return={target_metric.get('avg_future_return_pct')}, "
                    f"median_return={target_metric.get('median_future_return_pct')}"
                )
                lines.append(
                    "  - subset_context_target_metrics: "
                    f"labeled_count={subset_metric.get('subset_labeled_count')}, "
                    f"dominance_pct={subset_metric.get('directional_dominance_pct')}, "
                    f"avg_return={subset_metric.get('avg_future_return_pct')}, "
                    f"median_return={subset_metric.get('median_future_return_pct')}"
                )
            lines.append("")

        lines.append("### Target-Only Recent Metrics")
        lines.append("")
        for identity, metrics_by_horizon in target_only_metrics.items():
            lines.append(f"#### {identity}")
            for horizon in HORIZONS:
                metric = metrics_by_horizon.get(horizon, {}) or {}
                lines.append(
                    "- "
                    f"{horizon}: "
                    f"labeled_count={metric.get('subset_labeled_count')}, "
                    f"up_rate_pct={metric.get('up_rate_pct')}, "
                    f"down_rate_pct={metric.get('down_rate_pct')}, "
                    f"flat_rate_pct={metric.get('flat_rate_pct')}, "
                    f"directional_dominance_pct={metric.get('directional_dominance_pct')}, "
                    f"avg_future_return_pct={metric.get('avg_future_return_pct')}, "
                    f"median_future_return_pct={metric.get('median_future_return_pct')}"
                )
            lines.append("")

        lines.append("### Subset-Context Target Metrics")
        lines.append("")
        for identity, metrics_by_horizon in subset_context_metrics.items():
            lines.append(f"#### {identity}")
            for horizon in HORIZONS:
                metric = metrics_by_horizon.get(horizon, {}) or {}
                lines.append(
                    "- "
                    f"{horizon}: "
                    f"labeled_count={metric.get('subset_labeled_count')}, "
                    f"up_rate_pct={metric.get('up_rate_pct')}, "
                    f"down_rate_pct={metric.get('down_rate_pct')}, "
                    f"flat_rate_pct={metric.get('flat_rate_pct')}, "
                    f"directional_dominance_pct={metric.get('directional_dominance_pct')}, "
                    f"avg_future_return_pct={metric.get('avg_future_return_pct')}, "
                    f"median_future_return_pct={metric.get('median_future_return_pct')}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    dataset = build_dataset(path=args.input)
    targets = _parse_targets(args.targets)

    windows_payload: list[dict[str, Any]] = []

    for window in args.windows:
        subset, subset_metadata = _build_full_context_recent_subset(
            dataset,
            targets=targets,
            anchor_horizon=args.anchor_horizon,
            labeled_window=window,
        )

        tmp = args.output_dir / f"_tmp_recent_window_{args.anchor_horizon}_{window}.jsonl"
        _write_subset(subset, tmp)

        try:
            strategy_lab = _build_strategy_lab_metrics(tmp)
            preview = _build_edge_candidates_preview(strategy_lab)
            stability = _build_edge_stability_preview(preview)
            rows = _build_edge_candidate_rows(
                tmp,
                edge_candidates_preview=preview,
                edge_stability_preview=stability,
            )
        finally:
            if not args.keep_temp_files and tmp.exists():
                tmp.unlink()

        joined_summary = _summarize(rows)

        target_only_recent_metrics: dict[str, Any] = {}
        subset_context_target_metrics: dict[str, Any] = {}
        recovery_vs_survival: dict[str, Any] = {}

        for symbol, strategy in targets:
            identity_key = f"{symbol}:{strategy}"

            target_only_recent_metrics[identity_key] = {}
            subset_context_target_metrics[identity_key] = {}

            for horizon in HORIZONS:
                target_only_recent_metrics[identity_key][horizon] = _build_target_only_recent_metrics(
                    dataset,
                    symbol=symbol,
                    strategy=strategy,
                    horizon=horizon,
                    labeled_window=window,
                )
                subset_context_target_metrics[identity_key][horizon] = _build_subset_context_target_metrics(
                    subset,
                    symbol=symbol,
                    strategy=strategy,
                    horizon=horizon,
                )

            identity_entry = _extract_identity_entry(joined_summary, identity_key)
            recovery_vs_survival[identity_key] = _build_recovery_vs_survival_summary(
                identity_key=identity_key,
                identity_entry=identity_entry,
                target_only_metrics=target_only_recent_metrics[identity_key],
                subset_context_metrics=subset_context_target_metrics[identity_key],
            )

        windows_payload.append(
            {
                "labeled_window": window,
                "subset_dataset_rows": len(subset),
                "subset_metadata": subset_metadata,
                "joined_candidate_rows": joined_summary,
                "preview_summary": _extract_preview_summary(preview, stability),
                "target_only_recent_metrics": target_only_recent_metrics,
                "subset_context_target_metrics": subset_context_target_metrics,
                "recovery_vs_survival": recovery_vs_survival,
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(args.input),
        "dataset_rows": len(dataset),
        "targets": [f"{symbol}:{strategy}" for symbol, strategy in targets],
        "anchor_horizon": args.anchor_horizon,
        "windows": windows_payload,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "recent_window_candidate_visibility_diagnosis_v2.json"
    md_path = args.output_dir / "recent_window_candidate_visibility_diagnosis_v2.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    print(f"Dataset rows: {len(dataset)}")
    print(f"Diagnosis JSON: {json_path.resolve()}")
    print(f"Diagnosis MD: {md_path.resolve()}")


if __name__ == "__main__":
    main()