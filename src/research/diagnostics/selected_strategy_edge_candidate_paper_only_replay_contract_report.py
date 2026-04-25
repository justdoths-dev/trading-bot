from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_near_miss_observability_report as near_miss_report,
)
from src.research.diagnostics import (
    selected_strategy_edge_candidate_policy_split_report as policy_split_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_paper_only_replay_contract_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Paper-Only Replay Contract Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

REPLAY_CONTRACT_VERSION = "paper_only_replay_v1"
REPLAY_MODE = "observation_only"
PAPER_ONLY_POLICY_CLASS = "paper_only_candidate"

DEFAULT_INPUT_PATH = policy_split_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = policy_split_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = policy_split_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = policy_split_report.SupportWindowConfiguration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only paper replay contract from paper-only policy "
            "split candidates. This does not create trades, orders, fills, or PnL."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Support window in WINDOW_HOURS/MAX_ROWS form. Repeatable.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_selected_strategy_edge_candidate_paper_only_replay_contract_report(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        configurations=parse_configuration_values(args.config),
        write_report_copies=args.write_latest_copy,
    )
    final = _safe_dict(result["report"].get("final_assessment"))
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "recommended_next_stage": final.get("recommended_next_stage"),
                "paper_replay_candidates_present": final.get(
                    "paper_replay_candidates_present"
                ),
                "production_candidates_present": final.get(
                    "production_candidates_present"
                ),
                "human_review_candidates_present": final.get(
                    "human_review_candidates_present"
                ),
                "replay_contract_supported": final.get("replay_contract_supported"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return policy_split_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return policy_split_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_only_replay_contract_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[SupportWindowConfiguration] | None = None,
    write_report_copies: bool = False,
) -> dict[str, Any]:
    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )

    written_paths: dict[str, str] = {}
    if write_report_copies:
        written_paths = write_report_files(report, output_dir)

    return {
        "input_path": report["inputs"]["input_path"],
        "output_dir": report["inputs"]["output_dir"],
        "written_paths": written_paths,
        "report": report,
        "markdown": render_markdown(report),
    }


def build_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[SupportWindowConfiguration] | None = None,
) -> dict[str, Any]:
    source_policy_report = policy_split_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(source_policy_report.get("configuration_summaries"))
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_policy_split_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_policy_report.get("inputs")),
        "diagnostic_only": True,
        "replay_contract": {
            "replay_contract_version": REPLAY_CONTRACT_VERSION,
            "replay_mode": REPLAY_MODE,
            "source_report_type": policy_split_report.REPORT_TYPE,
            "source_policy_class": PAPER_ONLY_POLICY_CLASS,
            "paper_replay_rows_are_trades": False,
            "paper_replay_rows_are_orders": False,
            "paper_replay_rows_are_fills": False,
            "paper_replay_rows_enter_live_mapper_or_engine": False,
            "synthetic_prices_or_pnl_created": False,
        },
        "source_policy_split_final_assessment": _safe_dict(
            source_policy_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_policy_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Paper-only replay candidates are observation rows, not trades.",
            "No order, fill, entry price, exit price, or realized PnL is synthesized.",
            "Paper-only replay rows must not enter the existing live mapper or engine path.",
            "Production candidate quality gates, mapper, engine, execution gate, and latest-window defaults are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_policy_split_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_policy_split_summary)
    configuration = _safe_dict(source_summary.get("configuration"))
    policy_rows = [
        row
        for row in _safe_list(source_summary.get("policy_rows"))
        if isinstance(row, dict)
    ]
    paper_replay_rows = [
        build_paper_replay_row(policy_row=row, configuration=configuration)
        for row in policy_rows
        if is_paper_replay_source_row(row)
    ]
    paper_replay_rows = deduplicate_paper_replay_candidate_ids(paper_replay_rows)
    paper_replay_rows.sort(key=_paper_replay_sort_key)

    summary = {
        "configuration": configuration,
        "source_policy_split_summary": source_summary,
        "paper_replay_candidate_count": len(paper_replay_rows),
        "human_review_candidate_count": _safe_int(
            source_summary.get("human_review_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("eligible_production_candidate_count")
        ),
        "hard_block_count": _safe_int(source_summary.get("hard_block_count")),
        "paper_replay_rows": paper_replay_rows,
        "best_paper_replay_candidate": _first(paper_replay_rows),
    }
    summary["replay_safety_invariants"] = build_replay_safety_invariants(
        summary, policy_rows
    )
    return summary


def is_paper_replay_source_row(row: dict[str, Any]) -> bool:
    return (
        row.get("policy_class") == PAPER_ONLY_POLICY_CLASS
        and row.get("source_row_type")
        == policy_split_report.SOURCE_REJECTED_DIAGNOSTIC_NEAR_MISS_ROW
    )


def build_paper_replay_row(
    *,
    policy_row: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    source_policy_class = str(policy_row.get("policy_class") or "")
    source_row_type = str(policy_row.get("source_row_type") or "")
    safety_notes = near_miss_report.normalize_string_list(policy_row.get("safety_notes"))
    safety_notes.extend(
        [
            "Paper-only replay row is an observation contract, not a trade.",
            "No order execution, synthetic fill, synthetic price, or PnL claim is permitted.",
            "The row must not be fed into the live mapper or engine path.",
        ]
    )

    base_id = build_paper_replay_candidate_id(
        configuration=configuration,
        policy_row=policy_row,
    )

    return {
        "paper_replay_candidate_base_id": base_id,
        "paper_replay_candidate_id": base_id,
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "replay_mode": REPLAY_MODE,
        "source_policy_class": source_policy_class,
        "source_row_type": source_row_type,
        "symbol": policy_row.get("symbol"),
        "strategy": policy_row.get("strategy"),
        "horizon": policy_row.get("horizon"),
        "production_live_selection_allowed": False,
        "mapper_live_path_allowed": False,
        "engine_live_path_allowed": False,
        "paper_replay_allowed": source_policy_class == PAPER_ONLY_POLICY_CLASS,
        "human_review_allowed": policy_row.get("human_review_allowed") is True,
        "no_order_execution": True,
        "no_synthetic_fill": True,
        "no_pnl_claim": True,
        "sample_count": _safe_int(policy_row.get("sample_count")),
        "labeled_count": _safe_int(policy_row.get("labeled_count")),
        "median_future_return_pct": _safe_float(
            policy_row.get("median_future_return_pct")
        ),
        "positive_rate_pct": _safe_float(policy_row.get("positive_rate_pct")),
        "robustness_signal": policy_row.get("robustness_signal"),
        "robustness_signal_pct": _safe_float(
            policy_row.get("robustness_signal_pct")
        ),
        "aggregate_score": _safe_float(policy_row.get("aggregate_score")),
        "source_rejection_reason": policy_row.get("source_rejection_reason"),
        "source_rejection_reasons": near_miss_report.normalize_string_list(
            policy_row.get("source_rejection_reasons")
        ),
        "near_miss_classification": policy_row.get("near_miss_classification"),
        "suggested_next_policy_bucket": policy_row.get("suggested_next_policy_bucket"),
        "replay_observation_reason": (
            "Track the non-live paper-only near-miss candidate for future policy "
            "review without creating trades, orders, fills, or PnL."
        ),
        "safety_notes": safety_notes,
    }


def build_paper_replay_candidate_id(
    *,
    configuration: dict[str, Any],
    policy_row: dict[str, Any],
) -> str:
    configuration_slug = str(
        configuration.get("slug") or configuration.get("display_name") or ""
    )
    return ":".join(
        [
            REPLAY_CONTRACT_VERSION,
            _stable_id_part(configuration_slug),
            _stable_id_part(policy_row.get("symbol")),
            _stable_id_part(policy_row.get("strategy")),
            _stable_id_part(policy_row.get("horizon")),
            _stable_id_part(policy_row.get("policy_class")),
        ]
    )


def deduplicate_paper_replay_candidate_ids(
    paper_replay_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen_counts: Counter[str] = Counter()
    deduplicated_rows: list[dict[str, Any]] = []

    for row in paper_replay_rows:
        item = dict(row)
        base_id = str(
            item.get("paper_replay_candidate_base_id")
            or item.get("paper_replay_candidate_id")
            or ""
        )
        seen_counts[base_id] += 1
        occurrence = seen_counts[base_id]

        item["paper_replay_candidate_base_id"] = base_id
        if occurrence == 1:
            item["paper_replay_candidate_id"] = base_id
        else:
            item["paper_replay_candidate_id"] = f"{base_id}:dup_{occurrence}"
        deduplicated_rows.append(item)

    return deduplicated_rows


def build_replay_safety_invariants(
    summary: dict[str, Any],
    source_policy_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paper_rows = [
        row
        for row in _safe_list(summary.get("paper_replay_rows"))
        if isinstance(row, dict)
    ]
    source_rows = [
        row for row in _safe_list(source_policy_rows or []) if isinstance(row, dict)
    ]
    paper_candidate_ids = [row.get("paper_replay_candidate_id") for row in paper_rows]
    paper_ids = set(paper_candidate_ids)

    def source_ids_for(policy_class: str) -> set[str]:
        return {
            build_paper_replay_candidate_id(
                configuration=_safe_dict(summary.get("configuration")),
                policy_row=row,
            )
            for row in source_rows
            if row.get("policy_class") == policy_class
        }

    return {
        "all_paper_replay_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False for row in paper_rows
        ),
        "all_paper_replay_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in paper_rows
        ),
        "all_paper_replay_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in paper_rows
        ),
        "all_paper_replay_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in paper_rows
        ),
        "all_paper_replay_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in paper_rows
        ),
        "all_paper_replay_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in paper_rows
        ),
        "no_hard_block_rows_in_paper_replay_rows": paper_ids.isdisjoint(
            source_ids_for("hard_block")
        ),
        "no_collect_more_data_rows_in_paper_replay_rows": paper_ids.isdisjoint(
            source_ids_for("collect_more_data")
        ),
        "no_human_review_candidate_rows_in_paper_replay_rows": paper_ids.isdisjoint(
            source_ids_for("human_review_candidate")
        ),
        "paper_replay_rows_sourced_only_from_paper_only_candidate_policy": all(
            row.get("source_policy_class") == PAPER_ONLY_POLICY_CLASS
            for row in paper_rows
        ),
        "paper_replay_candidate_ids_are_unique": len(paper_candidate_ids)
        == len(paper_ids),
        "paper_replay_candidate_count": len(paper_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    replay_invariants = []
    for summary in configuration_summaries:
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )
        counts["human_review_candidate"] += _safe_int(
            summary.get("human_review_candidate_count")
        )
        counts["hard_block"] += _safe_int(summary.get("hard_block_count"))
        replay_invariants.append(_safe_dict(summary.get("replay_safety_invariants")))

    paper_present = counts["paper_replay_candidate"] > 0
    production_present = counts["production_candidate"] > 0
    human_present = counts["human_review_candidate"] > 0

    if paper_present:
        recommended_next_stage = "design_paper_replay_observation_journal"
    elif not any([paper_present, production_present, human_present]):
        recommended_next_stage = "collect_more_data"
    elif production_present and not any([paper_present, human_present]):
        recommended_next_stage = "no_paper_replay_contract_needed"
    else:
        recommended_next_stage = "mixed_or_inconclusive"

    invariant_keys = [
        "all_paper_replay_rows_production_live_selection_disallowed",
        "all_paper_replay_rows_mapper_live_path_disallowed",
        "all_paper_replay_rows_engine_live_path_disallowed",
        "all_paper_replay_rows_no_order_execution",
        "all_paper_replay_rows_no_synthetic_fill",
        "all_paper_replay_rows_no_pnl_claim",
        "no_hard_block_rows_in_paper_replay_rows",
        "no_collect_more_data_rows_in_paper_replay_rows",
        "no_human_review_candidate_rows_in_paper_replay_rows",
        "paper_replay_rows_sourced_only_from_paper_only_candidate_policy",
        "paper_replay_candidate_ids_are_unique",
    ]

    return {
        "paper_replay_candidates_present": paper_present,
        "production_candidates_present": production_present,
        "human_review_candidates_present": human_present,
        "replay_contract_supported": paper_present,
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "replay_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in replay_invariants)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_paper_replay_observation_journal":
        return "Stop here. The next stage should design a paper replay observation journal."
    if recommended_next_stage == "collect_more_data":
        return "Stop here. More data is required before a paper replay contract is useful."
    if recommended_next_stage == "no_paper_replay_contract_needed":
        return "Stop here. No paper replay contract is needed for the current policy split."
    return "Stop here. This report does not implement paper trading execution."


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output = resolve_path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output / REPORT_JSON_NAME
    md_path = resolved_output / REPORT_MD_NAME
    _write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def render_markdown(report: dict[str, Any]) -> str:
    final = _safe_dict(report.get("final_assessment"))
    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Final Assessment",
        f"- paper_replay_candidates_present: {final.get('paper_replay_candidates_present')}",
        f"- production_candidates_present: {final.get('production_candidates_present')}",
        f"- human_review_candidates_present: {final.get('human_review_candidates_present')}",
        f"- replay_contract_supported: {final.get('replay_contract_supported')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract",
        "- Paper-only replay candidates are observation rows, not trades, orders, fills, or live engine candidates.",
        "- No synthetic entry price, exit price, fill, order, or realized PnL is generated.",
        "- Paper-only replay rows must not enter the live mapper or engine path.",
        "- Candidate quality gate, mapper, engine, execution gate, and production thresholds are unchanged.",
        "",
        "## Configuration Summaries",
    ]

    for summary in _safe_list(report.get("configuration_summaries")):
        item = _safe_dict(summary)
        configuration = _safe_dict(item.get("configuration"))
        lines.extend(
            [
                "",
                f"### {configuration.get('display_name')}",
                f"- paper_replay_candidate_count: {item.get('paper_replay_candidate_count')}",
                f"- human_review_candidate_count: {item.get('human_review_candidate_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- hard_block_count: {item.get('hard_block_count')}",
                f"- best_paper_replay_candidate: {item.get('best_paper_replay_candidate')}",
                f"- replay_safety_invariants: {item.get('replay_safety_invariants')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _paper_replay_sort_key(row: dict[str, Any]) -> tuple[float, str, str, str, str]:
    return (
        -float(row.get("aggregate_score") or 0.0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("paper_replay_candidate_id") or ""),
    )


def _stable_id_part(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    return (
        "".join(char if char.isalnum() else "_" for char in text).strip("_")
        or "unknown"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _first(rows: Any) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    first = rows[0]
    return first if isinstance(first, dict) else None


def _safe_dict(value: Any) -> dict[str, Any]:
    return policy_split_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return policy_split_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return policy_split_report._safe_int(value)


def _safe_float(value: Any) -> float | None:
    return policy_split_report._safe_float(value)


if __name__ == "__main__":
    main()
