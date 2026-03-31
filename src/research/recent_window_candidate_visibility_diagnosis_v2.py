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
        return value.strip().upper()
    return None


def _normalize_strategy(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip().lower()
    return None


def _normalize_label(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"up", "down", "flat"}:
            return text
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except:
        return None


def _pct(n, d):
    return round((n / d) * 100.0, 2) if d > 0 else None


def _median_or_none(values):
    if not values:
        return None
    return round(float(median(values)), 6)


def _parse_targets(raw):
    result = []
    for item in raw:
        if ":" in item:
            s, st = item.split(":", 1)
            s = _normalize_symbol(s)
            st = _normalize_strategy(st)
            if s and st:
                result.append((s, st))
    return list(dict.fromkeys(result))


def _filter_recent_labeled_rows(rows, horizon, window):
    key = f"future_label_{horizon}"
    labeled = [r for r in rows if _normalize_label(r.get(key))]
    return labeled[-window:] if window > 0 else labeled


def _select_subset(dataset, targets, horizon, window):
    ids = set()
    for symbol, strategy in targets:
        rows = [
            r
            for r in dataset
            if _normalize_symbol(r.get("symbol")) == symbol
            and _normalize_strategy(r.get("selected_strategy")) == strategy
        ]
        subset = _filter_recent_labeled_rows(rows, horizon, window)
        for r in subset:
            ids.add(id(r))
    return [r for r in dataset if id(r) in ids]


def _write_subset(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summarize(rows_block):
    rows = rows_block.get("rows", [])
    return {
        "row_count": rows_block.get("row_count"),
        "rows": rows,
        "identity_horizon_evaluations": rows_block.get(
            "identity_horizon_evaluations", []
        ),
    }


def _build_interpretation(entry):
    raw_union = entry["raw_preview_visibility"].get(
        "raw_category_union_horizons", []
    )
    compat_union = entry["compatibility_filtered_preview_visibility"].get(
        "compatibility_filtered_category_union_horizons", []
    )
    joined = entry.get("actual_joined_eligible_horizons", [])

    return (
        f"Raw preview: {raw_union} → "
        f"Compatibility filtered: {compat_union} → "
        f"Actual joined: {joined}"
    )


def _build_markdown(payload):
    lines = []
    lines.append("# Recent Window Candidate Visibility Diagnosis\n")

    for snapshot in payload["windows"]:
        lines.append(f"## Window {snapshot['labeled_window']}\n")

        joined = snapshot["joined_candidate_rows"]
        evals = joined["identity_horizon_evaluations"]

        for entry in evals:
            lines.append(f"### {entry['identity_key']}\n")

            # Interpretation summary
            lines.append("**Interpretation**")
            lines.append(_build_interpretation(entry) + "\n")

            # Horizon decisions
            lines.append("**Horizon Decisions**")
            for h, ev in entry["horizon_evaluations"].items():
                lines.append(
                    f"- {h}: status={ev['status']} | reason={ev['rejection_reason']}"
                )
            lines.append("")

    return "\n".join(lines)


def main():
    args = _parse_args()
    dataset = build_dataset(path=args.input)
    targets = _parse_targets(args.targets)

    results = []

    for w in args.windows:
        subset = _select_subset(dataset, targets, args.anchor_horizon, w)

        tmp = args.output_dir / f"_tmp_{w}.jsonl"
        _write_subset(subset, tmp)

        strategy_lab = _build_strategy_lab_metrics(tmp)
        preview = _build_edge_candidates_preview(strategy_lab)
        stability = _build_edge_stability_preview(preview)
        rows = _build_edge_candidate_rows(
            tmp,
            edge_candidates_preview=preview,
            edge_stability_preview=stability,
        )

        if not args.keep_temp_files:
            tmp.unlink(missing_ok=True)

        results.append(
            {
                "labeled_window": w,
                "joined_candidate_rows": _summarize(rows),
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "windows": results,
    }

    json_path = args.output_dir / "recent_window_candidate_visibility_diagnosis.json"
    md_path = args.output_dir / "recent_window_candidate_visibility_diagnosis.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    print("done")


if __name__ == "__main__":
    main()