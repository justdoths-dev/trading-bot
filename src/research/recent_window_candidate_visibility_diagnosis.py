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
        description="Diagnose recent-window candidate visibility using true subset datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Path to trade_analysis JSONL input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to write diagnosis outputs",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[100, 150, 200, 250],
        help="Recent labeled windows to inspect",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["BTCUSDT:swing", "ETHUSDT:swing"],
        help="Target identities in SYMBOL:strategy format",
    )
    parser.add_argument(
        "--anchor-horizon",
        choices=list(HORIZONS),
        default="4h",
        help="Horizon used to construct recent labeled subset windows",
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep generated temporary subset JSONL files",
    )
    return parser.parse_args()


def _normalize_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return text or None


def _normalize_strategy(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
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
    targets: list[tuple[str, str]] = []
    for item in raw_targets:
        if ":" not in item:
            continue
        symbol, strategy = item.split(":", 1)
        symbol_norm = _normalize_symbol(symbol)
        strategy_norm = _normalize_strategy(strategy)
        if symbol_norm and strategy_norm:
            targets.append((symbol_norm, strategy_norm))

    deduped: list[tuple[str, str]] = []
    for item in targets:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _filter_recent_labeled_rows(
    rows: list[dict[str, Any]],
    horizon: str,
    labeled_window: int,
) -> list[dict[str, Any]]:
    label_key = f"future_label_{horizon}"
    labeled = [row for row in rows if _normalize_label(row.get(label_key)) is not None]
    if labeled_window <= 0:
        return labeled
    return labeled[-labeled_window:]


def _select_recent_subset_dataset(
    dataset: list[dict[str, Any]],
    *,
    targets: list[tuple[str, str]],
    anchor_horizon: str,
    labeled_window: int,
) -> list[dict[str, Any]]:
    target_row_ids: set[int] = set()

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
            labeled_window=labeled_window,
        )
        for row in recent_rows:
            target_row_ids.add(id(row))

    subset = [row for row in dataset if id(row) in target_row_ids]

    if not subset:
        return []

    return subset


def _write_subset_jsonl(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    labeled_window: int,
    anchor_horizon: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"_recent_window_subset_{anchor_horizon}_{labeled_window}.jsonl"
    )
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            safe_row = _json_safe_row(row)
            f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")
    return path


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
        labeled_window=labeled_window,
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


def _summarize_rows(rows_block: dict[str, Any]) -> dict[str, Any]:
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
        "dropped_row_count": rows_block.get("dropped_row_count", 0),
        "symbol_counter": dict(symbol_counter),
        "strategy_counter": dict(strategy_counter),
        "horizon_counter": dict(horizon_counter),
        "stability_counter": dict(stability_counter),
        "rows": rows,
    }


def _extract_preview_summary(
    edge_candidates_preview: dict[str, Any],
    edge_stability_preview: dict[str, Any],
) -> dict[str, Any]:
    by_horizon = edge_candidates_preview.get("by_horizon", {}) or {}

    preview_by_horizon: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_data = by_horizon.get(horizon, {}) or {}
        preview_by_horizon[horizon] = {
            "top_symbol": horizon_data.get("top_symbol"),
            "top_strategy": horizon_data.get("top_strategy"),
            "top_alignment_state": horizon_data.get("top_alignment_state"),
            "sample_gate": horizon_data.get("sample_gate"),
            "quality_gate": horizon_data.get("quality_gate"),
            "candidate_strength": horizon_data.get("candidate_strength"),
            "visibility_reason": horizon_data.get("visibility_reason"),
        }

    return {
        "by_horizon": preview_by_horizon,
        "stability_preview": edge_stability_preview,
    }


def _build_window_snapshot(
    full_dataset: list[dict[str, Any]],
    *,
    output_dir: Path,
    labeled_window: int,
    targets: list[tuple[str, str]],
    anchor_horizon: str,
    keep_temp_files: bool,
) -> dict[str, Any]:
    subset_dataset = _select_recent_subset_dataset(
        full_dataset,
        targets=targets,
        anchor_horizon=anchor_horizon,
        labeled_window=labeled_window,
    )

    subset_path = _write_subset_jsonl(
        subset_dataset,
        output_dir=output_dir,
        labeled_window=labeled_window,
        anchor_horizon=anchor_horizon,
    )

    try:
        strategy_lab_metrics = _build_strategy_lab_metrics(subset_path)
        edge_candidates_preview = _build_edge_candidates_preview(
            strategy_lab=strategy_lab_metrics,
        )
        edge_stability_preview = _build_edge_stability_preview(
            edge_candidates_preview=edge_candidates_preview,
        )
        edge_candidate_rows = _build_edge_candidate_rows(
            subset_path,
            edge_candidates_preview=edge_candidates_preview,
            edge_stability_preview=edge_stability_preview,
        )
    finally:
        if not keep_temp_files and subset_path.exists():
            subset_path.unlink()

    target_metrics: dict[str, Any] = {}
    for symbol, strategy in targets:
        identity_key = f"{symbol}:{strategy}"
        target_metrics[identity_key] = {}
        for horizon in HORIZONS:
            target_metrics[identity_key][horizon] = _build_target_recent_metrics(
                full_dataset,
                symbol=symbol,
                strategy=strategy,
                horizon=horizon,
                labeled_window=labeled_window,
            )

    return {
        "labeled_window": labeled_window,
        "anchor_horizon": anchor_horizon,
        "subset_dataset_rows": len(subset_dataset),
        "joined_candidate_rows": _summarize_rows(edge_candidate_rows),
        "preview_summary": _extract_preview_summary(
            edge_candidates_preview=edge_candidates_preview,
            edge_stability_preview=edge_stability_preview,
        ),
        "target_recent_metrics": target_metrics,
    }


def _write_outputs(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recent_window_candidate_visibility_diagnosis.json"
    md_path = output_dir / "recent_window_candidate_visibility_diagnosis.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_build_markdown(payload), encoding="utf-8")
    return json_path, md_path


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


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
        window = snapshot.get("labeled_window")
        joined = snapshot.get("joined_candidate_rows", {}) or {}
        preview = snapshot.get("preview_summary", {}) or {}
        target_recent_metrics = snapshot.get("target_recent_metrics", {}) or {}

        lines.append(f"## Recent labeled window = {window}")
        lines.append("")
        lines.append(f"- subset_dataset_rows: {snapshot.get('subset_dataset_rows', 0)}")
        lines.append(f"- row_count: {joined.get('row_count', 0)}")
        lines.append(f"- dropped_row_count: {joined.get('dropped_row_count', 0)}")
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
                    f"stability={row.get('selected_stability_label')}, "
                    f"selected_visible_horizons={row.get('selected_visible_horizons')}, "
                    f"preview_symbol_visible_horizons={row.get('preview_symbol_visible_horizons')}, "
                    f"preview_strategy_visible_horizons={row.get('preview_strategy_visible_horizons')}, "
                    f"aggregate_score={row.get('aggregate_score')}, "
                    f"visibility_reason={row.get('visibility_reason')})"
                )
        else:
            lines.append("No joined candidate rows.")
        lines.append("")

        lines.append("### Preview By Horizon")
        lines.append("")
        for horizon in HORIZONS:
            horizon_preview = (preview.get("by_horizon", {}) or {}).get(horizon, {}) or {}
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
                    f"up_rate_pct={_fmt(metric.get('up_rate_pct'))}, "
                    f"down_rate_pct={_fmt(metric.get('down_rate_pct'))}, "
                    f"flat_rate_pct={_fmt(metric.get('flat_rate_pct'))}, "
                    f"positive_rate_pct={_fmt(metric.get('positive_rate_pct'))}, "
                    f"avg_future_return_pct={_fmt(metric.get('avg_future_return_pct'))}, "
                    f"median_future_return_pct={_fmt(metric.get('median_future_return_pct'))}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    full_dataset = build_dataset(path=args.input)
    targets = _parse_targets(args.targets)

    windows_payload: list[dict[str, Any]] = []
    for labeled_window in args.windows:
        windows_payload.append(
            _build_window_snapshot(
                full_dataset,
                output_dir=args.output_dir,
                labeled_window=labeled_window,
                targets=targets,
                anchor_horizon=args.anchor_horizon,
                keep_temp_files=args.keep_temp_files,
            )
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_path": str(args.input),
        "dataset_rows": len(full_dataset),
        "targets": [f"{symbol}:{strategy}" for symbol, strategy in targets],
        "anchor_horizon": args.anchor_horizon,
        "windows": windows_payload,
    }

    json_path, md_path = _write_outputs(args.output_dir, payload)

    print(f"Dataset rows: {len(full_dataset)}")
    print(f"Diagnosis JSON: {json_path.resolve()}")
    print(f"Diagnosis MD: {md_path.resolve()}")


if __name__ == "__main__":
    main()