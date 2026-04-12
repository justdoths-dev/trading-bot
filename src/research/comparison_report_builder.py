from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HORIZONS = ("15m", "1h", "4h")
STABILITY_KEYS = ("strategy", "symbol", "alignment_state")
STRENGTH_RANK = {
    "insufficient_data": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}
STABILITY_RANK = {
    "insufficient_data": 0,
    "unstable": 1,
    "single_horizon_only": 2,
    "multi_horizon_confirmed": 3,
}


def build_comparison_report(
    latest_summary_path: Path,
    cumulative_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    latest_summary = _read_summary_file(latest_summary_path)
    cumulative_summary = _read_summary_file(cumulative_summary_path)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "latest_input": str(latest_summary_path),
        "cumulative_input": str(cumulative_summary_path),
        "edge_candidates_preview": _safe_dict(
            latest_summary.get("edge_candidates_preview")
        ),
        "edge_stability_preview": _safe_dict(
            latest_summary.get("edge_stability_preview")
        ),
        "dataset_overview_comparison": _build_dataset_overview_comparison(
            latest_summary,
            cumulative_summary,
        ),
        "top_highlights_comparison": _build_top_highlights_comparison(
            latest_summary,
            cumulative_summary,
        ),
        "edge_candidates_comparison": _build_edge_candidates_comparison(
            latest_summary,
            cumulative_summary,
        ),
        "edge_stability_comparison": _build_edge_stability_comparison(
            latest_summary,
            cumulative_summary,
        ),
        "strategy_lab_edge_count_comparison": _build_strategy_lab_edge_count_comparison(
            latest_summary,
            cumulative_summary,
        ),
    }
    report["drift_notes"] = _build_drift_notes(report)
    report["comparison_summary"] = _build_comparison_summary(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    return report


def _read_summary_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_dataset_overview_comparison(
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    latest = latest_summary.get("dataset_overview", {}) or {}
    cumulative = cumulative_summary.get("dataset_overview", {}) or {}

    return {
        "latest_total_records": latest.get("total_records", 0),
        "cumulative_total_records": cumulative.get("total_records", 0),
        "latest_label_coverage_any_horizon_pct": latest.get(
            "label_coverage_any_horizon_pct", "n/a"
        ),
        "cumulative_label_coverage_any_horizon_pct": cumulative.get(
            "label_coverage_any_horizon_pct", "n/a"
        ),
    }


def _build_top_highlights_comparison(
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    latest = (latest_summary.get("top_highlights", {}) or {}).get("by_horizon", {}) or {}
    cumulative = (cumulative_summary.get("top_highlights", {}) or {}).get("by_horizon", {}) or {}

    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        latest_horizon = latest.get(horizon, {}) or {}
        cumulative_horizon = cumulative.get(horizon, {}) or {}
        result[horizon] = {
            "latest_top_symbol": latest_horizon.get("top_symbol", "n/a"),
            "cumulative_top_symbol": cumulative_horizon.get("top_symbol", "n/a"),
            "latest_top_strategy": latest_horizon.get("top_strategy", "n/a"),
            "cumulative_top_strategy": cumulative_horizon.get("top_strategy", "n/a"),
            "latest_best_alignment_state": latest_horizon.get(
                "best_alignment_state", "n/a"
            ),
            "cumulative_best_alignment_state": cumulative_horizon.get(
                "best_alignment_state", "n/a"
            ),
            "latest_best_ai_execution_state": latest_horizon.get(
                "best_ai_execution_state", "n/a"
            ),
            "cumulative_best_ai_execution_state": cumulative_horizon.get(
                "best_ai_execution_state", "n/a"
            ),
        }

    return result


def _build_edge_candidates_comparison(
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    latest = (latest_summary.get("edge_candidates_preview", {}) or {}).get("by_horizon", {}) or {}
    cumulative = (cumulative_summary.get("edge_candidates_preview", {}) or {}).get("by_horizon", {}) or {}

    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        latest_horizon = latest.get(horizon, {}) or {}
        cumulative_horizon = cumulative.get(horizon, {}) or {}
        result[horizon] = {
            "latest_candidate_strength": latest_horizon.get(
                "candidate_strength", "insufficient_data"
            ),
            "cumulative_candidate_strength": cumulative_horizon.get(
                "candidate_strength", "insufficient_data"
            ),
            "latest_top_strategy_group": _group_from_candidate(
                latest_horizon.get("top_strategy")
            ),
            "cumulative_top_strategy_group": _group_from_candidate(
                cumulative_horizon.get("top_strategy")
            ),
            "latest_top_symbol_group": _group_from_candidate(
                latest_horizon.get("top_symbol")
            ),
            "cumulative_top_symbol_group": _group_from_candidate(
                cumulative_horizon.get("top_symbol")
            ),
            "latest_top_alignment_state_group": _group_from_candidate(
                latest_horizon.get("top_alignment_state")
            ),
            "cumulative_top_alignment_state_group": _group_from_candidate(
                cumulative_horizon.get("top_alignment_state")
            ),
        }

    return result


def _build_edge_stability_comparison(
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    latest = latest_summary.get("edge_stability_preview", {}) or {}
    cumulative = cumulative_summary.get("edge_stability_preview", {}) or {}

    result: dict[str, Any] = {}
    for key in STABILITY_KEYS:
        latest_entry = latest.get(key, {}) or {}
        cumulative_entry = cumulative.get(key, {}) or {}
        result[key] = {
            "latest_stability_label": latest_entry.get(
                "stability_label", "insufficient_data"
            ),
            "cumulative_stability_label": cumulative_entry.get(
                "stability_label", "insufficient_data"
            ),
            "latest_group": latest_entry.get("group"),
            "cumulative_group": cumulative_entry.get("group"),
            "latest_visible_horizons": latest_entry.get("visible_horizons", []),
            "cumulative_visible_horizons": cumulative_entry.get("visible_horizons", []),
        }

    return result


def _build_strategy_lab_edge_count_comparison(
    latest_summary: dict[str, Any],
    cumulative_summary: dict[str, Any],
) -> dict[str, Any]:
    latest = ((latest_summary.get("strategy_lab", {}) or {}).get("edge", {})) or {}
    cumulative = ((cumulative_summary.get("strategy_lab", {}) or {}).get("edge", {})) or {}

    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        latest_horizon = latest.get(horizon, {}) or {}
        cumulative_horizon = cumulative.get(horizon, {}) or {}
        result[horizon] = {
            "latest_symbol_edges": _edge_count(latest_horizon.get("by_symbol")),
            "cumulative_symbol_edges": _edge_count(cumulative_horizon.get("by_symbol")),
            "latest_strategy_edges": _edge_count(latest_horizon.get("by_strategy")),
            "cumulative_strategy_edges": _edge_count(cumulative_horizon.get("by_strategy")),
            "latest_alignment_state_edges": _edge_count(latest_horizon.get("by_alignment_state")),
            "cumulative_alignment_state_edges": _edge_count(cumulative_horizon.get("by_alignment_state")),
            "latest_ai_execution_state_edges": _edge_count(latest_horizon.get("by_ai_execution_state")),
            "cumulative_ai_execution_state_edges": _edge_count(cumulative_horizon.get("by_ai_execution_state")),
        }

    return result


def _build_drift_notes(report: dict[str, Any]) -> list[str]:
    notes: list[str] = []

    for horizon, comparison in report["top_highlights_comparison"].items():
        if comparison["latest_top_strategy"] != comparison["cumulative_top_strategy"]:
            notes.append(f"{horizon}: top_strategy_changed")
        if comparison["latest_top_symbol"] != comparison["cumulative_top_symbol"]:
            notes.append(f"{horizon}: top_symbol_changed")

    for key, comparison in report["edge_stability_comparison"].items():
        latest_rank = STABILITY_RANK.get(comparison["latest_stability_label"], 0)
        cumulative_rank = STABILITY_RANK.get(comparison["cumulative_stability_label"], 0)
        if latest_rank > cumulative_rank:
            notes.append(f"{key}: stability_strengthened")
        elif latest_rank < cumulative_rank:
            notes.append(f"{key}: stability_weakened")

    for horizon, comparison in report["edge_candidates_comparison"].items():
        latest_rank = STRENGTH_RANK.get(comparison["latest_candidate_strength"], 0)
        cumulative_rank = STRENGTH_RANK.get(comparison["cumulative_candidate_strength"], 0)
        if latest_rank > cumulative_rank:
            notes.append(f"{horizon}: candidate_visibility_increased")
        elif latest_rank < cumulative_rank:
            notes.append(f"{horizon}: candidate_visibility_decreased")

    if not notes:
        return ["no_material_change"]

    return notes


def _build_comparison_summary(report: dict[str, Any]) -> dict[str, str]:
    dataset = report["dataset_overview_comparison"]
    top_highlights = report["top_highlights_comparison"]
    edge_candidates = report["edge_candidates_comparison"]
    edge_stability = report["edge_stability_comparison"]

    dataset_size_context = (
        f"latest covers {dataset['latest_total_records']} records versus {dataset['cumulative_total_records']} in cumulative baseline"
    )
    coverage_context = (
        f"label coverage is {dataset['latest_label_coverage_any_horizon_pct']} for latest versus {dataset['cumulative_label_coverage_any_horizon_pct']} for cumulative"
    )

    aligned_horizons: list[str] = []
    divergent_horizons: list[str] = []
    for horizon in HORIZONS:
        comparison = top_highlights[horizon]
        if (
            comparison["latest_top_symbol"] == comparison["cumulative_top_symbol"]
            and comparison["latest_top_strategy"] == comparison["cumulative_top_strategy"]
            and comparison["latest_best_alignment_state"] == comparison["cumulative_best_alignment_state"]
            and comparison["latest_best_ai_execution_state"] == comparison["cumulative_best_ai_execution_state"]
        ):
            aligned_horizons.append(horizon)
        else:
            divergent_horizons.append(horizon)

    key_alignment_summary = (
        f"{', '.join(aligned_horizons)} remain aligned with cumulative baseline"
        if aligned_horizons
        else "no horizon remains fully aligned with cumulative baseline"
    )
    key_divergence_summary = (
        f"latest diverges from cumulative in {', '.join(divergent_horizons)}"
        if divergent_horizons
        else "no material divergence detected across tracked horizons"
    )

    visible_horizons = [
        horizon
        for horizon in HORIZONS
        if edge_candidates[horizon]["latest_candidate_strength"] != "insufficient_data"
        or edge_candidates[horizon]["cumulative_candidate_strength"] != "insufficient_data"
    ]
    candidate_summary = (
        f"candidate visibility remains present in {', '.join(visible_horizons)}"
        if visible_horizons
        else "no candidate visibility detected in either latest or cumulative view"
    )

    stability_parts: list[str] = []
    for key in STABILITY_KEYS:
        comparison = edge_stability[key]
        latest_rank = STABILITY_RANK.get(comparison["latest_stability_label"], 0)
        cumulative_rank = STABILITY_RANK.get(comparison["cumulative_stability_label"], 0)
        if latest_rank > cumulative_rank:
            stability_parts.append(f"{key} stability strengthened")
        elif latest_rank < cumulative_rank:
            stability_parts.append(f"{key} stability weakened")

    if stability_parts:
        stability_summary = "; ".join(stability_parts)
    elif any(
        edge_stability[key]["latest_stability_label"] == "multi_horizon_confirmed"
        for key in STABILITY_KEYS
    ):
        stability_summary = "multi-horizon confirmed stability remains present in at least one tracked category"
    else:
        stability_summary = "no multi-horizon confirmed candidate detected"

    return {
        "dataset_size_context": dataset_size_context,
        "coverage_context": coverage_context,
        "key_alignment_summary": key_alignment_summary,
        "key_divergence_summary": key_divergence_summary,
        "candidate_summary": candidate_summary,
        "stability_summary": stability_summary,
    }


def _group_from_candidate(candidate: Any) -> str:
    if not isinstance(candidate, dict):
        return "n/a"
    group = candidate.get("group")
    if group in (None, ""):
        return "n/a"
    return str(group)


def _edge_count(report: Any) -> int:
    if not isinstance(report, dict):
        return 0
    findings = report.get("edge_findings", [])
    if not isinstance(findings, list):
        return 0
    return len(findings)


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Comparison Overview")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append(f"Latest input: {report['latest_input']}")
    lines.append(f"Cumulative input: {report['cumulative_input']}")
    lines.append("")

    summary = report["comparison_summary"]
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- {summary['dataset_size_context']}")
    lines.append(f"- {summary['coverage_context']}")
    lines.append(f"- {summary['key_alignment_summary']}")
    lines.append(f"- {summary['key_divergence_summary']}")
    lines.append(f"- {summary['candidate_summary']}")
    lines.append(f"- {summary['stability_summary']}")
    lines.append("")

    dataset = report["dataset_overview_comparison"]
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- latest_total_records: {dataset['latest_total_records']}")
    lines.append(f"- cumulative_total_records: {dataset['cumulative_total_records']}")
    lines.append(f"- latest_label_coverage_any_horizon_pct: {dataset['latest_label_coverage_any_horizon_pct']}")
    lines.append(f"- cumulative_label_coverage_any_horizon_pct: {dataset['cumulative_label_coverage_any_horizon_pct']}")
    lines.append("")

    lines.append("## Top Highlights by Horizon")
    lines.append("")
    for horizon in HORIZONS:
        comparison = report["top_highlights_comparison"][horizon]
        lines.append(f"### {horizon}")
        lines.append(f"- top_symbol: latest={comparison['latest_top_symbol']} | cumulative={comparison['cumulative_top_symbol']}")
        lines.append(f"- top_strategy: latest={comparison['latest_top_strategy']} | cumulative={comparison['cumulative_top_strategy']}")
        lines.append(f"- best_alignment_state: latest={comparison['latest_best_alignment_state']} | cumulative={comparison['cumulative_best_alignment_state']}")
        lines.append(f"- best_ai_execution_state: latest={comparison['latest_best_ai_execution_state']} | cumulative={comparison['cumulative_best_ai_execution_state']}")
        lines.append("")

    lines.append("## Edge Candidate Preview Comparison")
    lines.append("")
    for horizon in HORIZONS:
        comparison = report["edge_candidates_comparison"][horizon]
        lines.append(f"### {horizon}")
        lines.append(f"- candidate_strength: latest={comparison['latest_candidate_strength']} | cumulative={comparison['cumulative_candidate_strength']}")
        lines.append(f"- top_strategy.group: latest={comparison['latest_top_strategy_group']} | cumulative={comparison['cumulative_top_strategy_group']}")
        lines.append(f"- top_symbol.group: latest={comparison['latest_top_symbol_group']} | cumulative={comparison['cumulative_top_symbol_group']}")
        lines.append(f"- top_alignment_state.group: latest={comparison['latest_top_alignment_state_group']} | cumulative={comparison['cumulative_top_alignment_state_group']}")
        lines.append("")

    lines.append("## Edge Stability Preview Comparison")
    lines.append("")
    for key in STABILITY_KEYS:
        comparison = report["edge_stability_comparison"][key]
        lines.append(f"### {key}")
        lines.append(f"- stability_label: latest={comparison['latest_stability_label']} | cumulative={comparison['cumulative_stability_label']}")
        lines.append(f"- group: latest={comparison['latest_group']} | cumulative={comparison['cumulative_group']}")
        lines.append(f"- visible_horizons: latest={comparison['latest_visible_horizons']} | cumulative={comparison['cumulative_visible_horizons']}")
        lines.append("")

    lines.append("## Strategy Lab Edge Count Comparison")
    lines.append("")
    for horizon in HORIZONS:
        comparison = report["strategy_lab_edge_count_comparison"][horizon]
        lines.append(f"### {horizon}")
        lines.append(f"- symbol_edges: latest={comparison['latest_symbol_edges']} | cumulative={comparison['cumulative_symbol_edges']}")
        lines.append(f"- strategy_edges: latest={comparison['latest_strategy_edges']} | cumulative={comparison['cumulative_strategy_edges']}")
        lines.append(f"- alignment_state_edges: latest={comparison['latest_alignment_state_edges']} | cumulative={comparison['cumulative_alignment_state_edges']}")
        lines.append(f"- ai_execution_state_edges: latest={comparison['latest_ai_execution_state_edges']} | cumulative={comparison['cumulative_ai_execution_state_edges']}")
        lines.append("")

    lines.append("## Drift Notes")
    lines.append("")
    for note in report["drift_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a latest-vs-cumulative research comparison report")
    parser.add_argument("--latest", type=Path, required=True, help="Path to latest summary.json")
    parser.add_argument("--cumulative", type=Path, required=True, help="Path to cumulative summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison summary outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_comparison_report(
        latest_summary_path=args.latest,
        cumulative_summary_path=args.cumulative,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
