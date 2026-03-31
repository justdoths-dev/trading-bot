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
            "Diagnose recent-window candidate visibility using full-context datasets "
            "cut at target-anchor recency boundaries."
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


def _filter_recent_labeled_rows(
    rows: list[dict[str, Any]],
    horizon: str,
    window: int,
) -> list[dict[str, Any]]:
    label_key = f"future_label_{horizon}"
    labeled = [row for row in rows if _normalize_label(row.get(label_key)) is not None]
    return labeled[-window:] if window > 0 else labeled


def _extract_logged_at(row: dict[str, Any]) -> str:
    value = row.get("logged_at")
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return ""


def _build_full_context_recent_subset(
    dataset: list[dict[str, Any]],
    *,
    targets: list[tuple[str, str]],
    anchor_horizon: str,
    labeled_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build a recent-window dataset while preserving full-category context.

    Logic:
    1. For each target identity, collect recent labeled rows on anchor_horizon.
    2. Find the earliest logged_at among those selected anchor rows.
    3. Keep ALL dataset rows whose logged_at is >= that cutoff.
    4. If logged_at is missing everywhere or cutoff cannot be determined, fall back
       to target-row-only subset and record fallback metadata.
    """
    selected_anchor_rows: list[dict[str, Any]] = []

    for symbol, strategy in targets:
        identity_rows = [
            row
            for row in dataset
            if _normalize_symbol(row.get("symbol")) == symbol
            and _normalize_strategy(row.get("selected_strategy")) == strategy
        ]
        recent_rows = _filter_recent_labeled_rows(
            identity_rows,
            horizon=anchor_horizon,
            window=labeled_window,
        )
        selected_anchor_rows.extend(recent_rows)

    if not selected_anchor_rows:
        return [], {
            "subset_mode": "empty",
            "cutoff_logged_at": None,
            "selected_anchor_row_count": 0,
            "full_context_row_count": 0,
        }

    anchor_logged_ats = [
        _extract_logged_at(row)
        for row in selected_anchor_rows
        if _extract_logged_at(row)
    ]

    if not anchor_logged_ats:
        target_ids = {id(row) for row in selected_anchor_rows}
        fallback_subset = [row for row in dataset if id(row) in target_ids]
        return fallback_subset, {
            "subset_mode": "target_row_only_fallback",
            "cutoff_logged_at": None,
            "selected_anchor_row_count": len(selected_anchor_rows),
            "full_context_row_count": len(fallback_subset),
        }

    cutoff_logged_at = min(anchor_logged_ats)
    full_context_subset = [
        row for row in dataset
        if _extract_logged_at(row) and _extract_logged_at(row) >= cutoff_logged_at
    ]

    if not full_context_subset:
        target_ids = {id(row) for row in selected_anchor_rows}
        fallback_subset = [row for row in dataset if id(row) in target_ids]
        return fallback_subset, {
            "subset_mode": "target_row_only_fallback_after_empty_cut",
            "cutoff_logged_at": cutoff_logged_at,
            "selected_anchor_row_count": len(selected_anchor_rows),
            "full_context_row_count": len(fallback_subset),
        }

    return full_context_subset, {
        "subset_mode": "full_context_cutoff",
        "cutoff_logged_at": cutoff_logged_at,
        "selected_anchor_row_count": len(selected_anchor_rows),
        "full_context_row_count": len(full_context_subset),
    }


def _write_subset(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            safe_row = _json_safe_row(row)
            f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")


def _build_target_recent_metrics(
    dataset: list[dict[str, Any]],
    *,
    symbol: str,
    strategy: str,
    horizon: str,
    labeled_window: int,
) -> dict[str, Any]:
    target_rows = [
        row
        for row in dataset
        if _normalize_symbol(row.get("symbol")) == symbol
        and _normalize_strategy(row.get("selected_strategy")) == strategy
    ]

    label_key = f"future_label_{horizon}"
    return_key = f"future_return_{horizon}"

    recent_labeled_rows = _filter_recent_labeled_rows(
        target_rows,
        horizon=horizon,
        window=labeled_window,
    )

    label_counter: Counter[str] = Counter()
    returns: list[float] = []

    for row in recent_labeled_rows:
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

    positive_rate_pct = None
    if labeled_count > 0:
        positive_rate_pct = max(
            _pct(up_count, labeled_count) or 0.0,
            _pct(down_count, labeled_count) or 0.0,
        )

    return {
        "symbol": symbol,
        "strategy": strategy,
        "horizon": horizon,
        "target_row_count": len(target_rows),
        "available_labeled_total": len(
            [row for row in target_rows if _normalize_label(row.get(label_key)) is not None]
        ),
        "subset_labeled_count": labeled_count,
        "up_rate_pct": _pct(up_count, labeled_count),
        "down_rate_pct": _pct(down_count, labeled_count),
        "flat_rate_pct": _pct(flat_count, labeled_count),
        "positive_rate_pct": positive_rate_pct,
        "avg_future_return_pct": (
            round(sum(returns) / len(returns), 6) if returns else None
        ),
        "median_future_return_pct": _median_or_none(returns),
    }


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
    lines.append("# Recent Window Candidate Visibility Diagnosis")
    lines.append("")
    lines.append(f"- generated_at: {payload.get('generated_at')}")
    lines.append(f"- input_path: {payload.get('input_path')}")
    lines.append(f"- dataset_rows: {payload.get('dataset_rows')}")
    lines.append(f"- anchor_horizon: {payload.get('anchor_horizon')}")
    lines.append("")

    for snapshot in payload.get("windows", []):
        joined = snapshot.get("joined_candidate_rows", {}) or {}
        preview = snapshot.get("preview_summary", {}) or {}
        raw_preview_by_horizon = preview.get("raw_preview_by_horizon", {}) or {}
        target_recent_metrics = snapshot.get("target_recent_metrics", {}) or {}
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
        lines.append(
            "Preview is category-level, while joined candidate rows are identity-level and filtered by strategy-horizon compatibility plus candidate quality gates."
        )
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

        lines.append("### Target Recent Metrics")
        lines.append("")
        for identity, metrics_by_horizon in target_recent_metrics.items():
            lines.append(f"#### {identity}")
            for horizon in HORIZONS:
                metric = metrics_by_horizon.get(horizon, {}) or {}
                lines.append(
                    "- "
                    f"{horizon}: "
                    f"subset_labeled_count={metric.get('subset_labeled_count')}, "
                    f"up_rate_pct={metric.get('up_rate_pct')}, "
                    f"down_rate_pct={metric.get('down_rate_pct')}, "
                    f"flat_rate_pct={metric.get('flat_rate_pct')}, "
                    f"positive_rate_pct={metric.get('positive_rate_pct')}, "
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

        target_recent_metrics: dict[str, Any] = {}
        for symbol, strategy in targets:
            identity_key = f"{symbol}:{strategy}"
            target_recent_metrics[identity_key] = {}
            for horizon in HORIZONS:
                target_recent_metrics[identity_key][horizon] = _build_target_recent_metrics(
                    dataset,
                    symbol=symbol,
                    strategy=strategy,
                    horizon=horizon,
                    labeled_window=window,
                )

        windows_payload.append(
            {
                "labeled_window": window,
                "subset_dataset_rows": len(subset),
                "subset_metadata": subset_metadata,
                "joined_candidate_rows": _summarize(rows),
                "preview_summary": _extract_preview_summary(preview, stability),
                "target_recent_metrics": target_recent_metrics,
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