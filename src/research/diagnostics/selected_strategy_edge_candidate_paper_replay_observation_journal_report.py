from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_only_replay_contract_report as replay_contract_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_paper_replay_observation_journal_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Paper Replay Observation Journal Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

JOURNAL_CONTRACT_VERSION = "paper_replay_observation_journal_v1"
OBSERVATION_STATUS = "open_observation"
OBSERVATION_LIFECYCLE_STATE = "created_from_report_only_contract"
PAPER_ONLY_POLICY_CLASS = replay_contract_report.PAPER_ONLY_POLICY_CLASS

DEFAULT_INPUT_PATH = replay_contract_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = replay_contract_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = replay_contract_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = replay_contract_report.SupportWindowConfiguration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only paper replay observation journal from the "
            "paper-only replay contract. This does not create trades, orders, "
            "fills, prices, or PnL."
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
    result = run_selected_strategy_edge_candidate_paper_replay_observation_journal_report(
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
                "journal_entries_present": final.get("journal_entries_present"),
                "journal_contract_supported": final.get(
                    "journal_contract_supported"
                ),
                "journal_entry_count": final.get("journal_entry_count"),
                "paper_replay_candidate_count": final.get(
                    "paper_replay_candidate_count"
                ),
                "production_candidate_count": final.get(
                    "production_candidate_count"
                ),
                "human_review_candidate_count": final.get(
                    "human_review_candidate_count"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return replay_contract_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return replay_contract_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_replay_observation_journal_report(
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
    source_replay_report = replay_contract_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(source_replay_report.get("configuration_summaries"))
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_replay_contract_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_replay_report.get("inputs")),
        "diagnostic_only": True,
        "journal_contract": {
            "journal_contract_version": JOURNAL_CONTRACT_VERSION,
            "observation_status": OBSERVATION_STATUS,
            "observation_lifecycle_state": OBSERVATION_LIFECYCLE_STATE,
            "source_report_type": replay_contract_report.REPORT_TYPE,
            "source_policy_class": PAPER_ONLY_POLICY_CLASS,
            "journal_rows_are_trades": False,
            "journal_rows_are_orders": False,
            "journal_rows_are_fills": False,
            "journal_rows_enter_live_mapper_or_engine": False,
            "synthetic_prices_or_pnl_created": False,
        },
        "source_paper_only_replay_contract_final_assessment": _safe_dict(
            source_replay_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_replay_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Paper replay observation journal rows are report-only records, not trades.",
            "Only source paper replay rows with source_policy_class paper_only_candidate become journal rows.",
            "No order, fill, entry price, exit price, realized PnL, or unrealized PnL is synthesized.",
            "Journal rows must not enter the live mapper or engine path.",
            "Existing paper-only replay contract semantics and production candidate quality gates are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_replay_contract_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_replay_contract_summary)
    configuration = _safe_dict(source_summary.get("configuration"))
    paper_replay_rows = [
        row
        for row in _safe_list(source_summary.get("paper_replay_rows"))
        if isinstance(row, dict)
    ]
    journal_rows = [
        build_journal_row(paper_replay_row=row, configuration=configuration)
        for row in paper_replay_rows
        if is_journal_source_row(row)
    ]
    journal_rows.sort(key=_journal_sort_key)

    summary = {
        "configuration": configuration,
        "source_replay_contract_summary": source_summary,
        "journal_entry_count": len(journal_rows),
        "paper_replay_candidate_count": _safe_int(
            source_summary.get("paper_replay_candidate_count")
        ),
        "human_review_candidate_count": _safe_int(
            source_summary.get("human_review_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("production_candidate_count")
        ),
        "journal_rows": journal_rows,
        "best_journal_row": _first(journal_rows),
    }
    summary["journal_safety_invariants"] = build_journal_safety_invariants(summary)
    return summary


def is_journal_source_row(row: dict[str, Any]) -> bool:
    return row.get("source_policy_class") == PAPER_ONLY_POLICY_CLASS


def build_journal_row(
    *,
    paper_replay_row: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    paper_replay_candidate_id = str(paper_replay_row.get("paper_replay_candidate_id") or "")
    safety_notes = replay_contract_report.near_miss_report.normalize_string_list(
        paper_replay_row.get("safety_notes")
    )
    safety_notes.extend(
        [
            "Observation journal row is report-only and is not a trade.",
            "No order, fill, synthetic price, realized PnL, or unrealized PnL is permitted.",
            "The row must not be fed into the live mapper or engine path.",
        ]
    )

    return {
        "journal_entry_id": build_journal_entry_id(
            paper_replay_candidate_id=paper_replay_candidate_id
        ),
        "journal_contract_version": JOURNAL_CONTRACT_VERSION,
        "replay_contract_version": paper_replay_row.get("replay_contract_version"),
        "replay_mode": paper_replay_row.get("replay_mode"),
        "source_report_type": replay_contract_report.REPORT_TYPE,
        "source_policy_class": paper_replay_row.get("source_policy_class"),
        "source_configuration": configuration,
        "paper_replay_candidate_id": paper_replay_candidate_id,
        "paper_replay_candidate_base_id": paper_replay_row.get(
            "paper_replay_candidate_base_id"
        ),
        "symbol": paper_replay_row.get("symbol"),
        "strategy": paper_replay_row.get("strategy"),
        "horizon": paper_replay_row.get("horizon"),
        "observation_status": OBSERVATION_STATUS,
        "observation_lifecycle_state": OBSERVATION_LIFECYCLE_STATE,
        "production_live_selection_allowed": False,
        "mapper_live_path_allowed": False,
        "engine_live_path_allowed": False,
        "no_order_execution": True,
        "no_synthetic_fill": True,
        "no_pnl_claim": True,
        "order_id": None,
        "fill_id": None,
        "entry_price": None,
        "exit_price": None,
        "realized_pnl": None,
        "unrealized_pnl": None,
        "sample_count": _safe_int(paper_replay_row.get("sample_count")),
        "labeled_count": _safe_int(paper_replay_row.get("labeled_count")),
        "median_future_return_pct": _safe_float(
            paper_replay_row.get("median_future_return_pct")
        ),
        "positive_rate_pct": _safe_float(paper_replay_row.get("positive_rate_pct")),
        "robustness_signal": paper_replay_row.get("robustness_signal"),
        "robustness_signal_pct": _safe_float(
            paper_replay_row.get("robustness_signal_pct")
        ),
        "aggregate_score": _safe_float(paper_replay_row.get("aggregate_score")),
        "source_rejection_reason": paper_replay_row.get("source_rejection_reason"),
        "source_rejection_reasons": replay_contract_report.near_miss_report.normalize_string_list(
            paper_replay_row.get("source_rejection_reasons")
        ),
        "near_miss_classification": paper_replay_row.get("near_miss_classification"),
        "suggested_next_policy_bucket": paper_replay_row.get(
            "suggested_next_policy_bucket"
        ),
        "journal_observation_reason": (
            "Open a report-only observation journal entry for this paper replay "
            "candidate without creating orders, fills, prices, or PnL."
        ),
        "safety_notes": safety_notes,
    }


def build_journal_entry_id(*, paper_replay_candidate_id: str) -> str:
    return f"{JOURNAL_CONTRACT_VERSION}:{paper_replay_candidate_id}"


def build_journal_safety_invariants(summary: dict[str, Any]) -> dict[str, Any]:
    journal_rows = [
        row for row in _safe_list(summary.get("journal_rows")) if isinstance(row, dict)
    ]
    journal_entry_ids = [row.get("journal_entry_id") for row in journal_rows]

    return {
        "all_journal_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False
            for row in journal_rows
        ),
        "all_journal_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in journal_rows
        ),
        "all_journal_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in journal_rows
        ),
        "all_journal_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in journal_rows
        ),
        "all_journal_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in journal_rows
        ),
        "all_journal_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in journal_rows
        ),
        "no_order_or_fill_identifiers_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in journal_rows
        ),
        "no_price_or_pnl_fields_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in journal_rows
        ),
        "journal_rows_sourced_only_from_paper_replay_candidates": all(
            row.get("source_policy_class") == PAPER_ONLY_POLICY_CLASS
            and row.get("paper_replay_candidate_id")
            for row in journal_rows
        ),
        "journal_entry_ids_are_unique": len(journal_entry_ids)
        == len(set(journal_entry_ids)),
        "journal_entry_count": len(journal_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    journal_invariants = []
    for summary in configuration_summaries:
        counts["journal_entry"] += _safe_int(summary.get("journal_entry_count"))
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )
        counts["human_review_candidate"] += _safe_int(
            summary.get("human_review_candidate_count")
        )
        journal_invariants.append(_safe_dict(summary.get("journal_safety_invariants")))

    journal_present = counts["journal_entry"] > 0
    production_present = counts["production_candidate"] > 0
    human_present = counts["human_review_candidate"] > 0

    if journal_present:
        recommended_next_stage = "design_paper_replay_outcome_tracking_contract"
    elif not production_present:
        recommended_next_stage = "collect_more_data"
    else:
        recommended_next_stage = "mixed_or_inconclusive"

    invariant_keys = [
        "all_journal_rows_production_live_selection_disallowed",
        "all_journal_rows_mapper_live_path_disallowed",
        "all_journal_rows_engine_live_path_disallowed",
        "all_journal_rows_no_order_execution",
        "all_journal_rows_no_synthetic_fill",
        "all_journal_rows_no_pnl_claim",
        "no_order_or_fill_identifiers_present",
        "no_price_or_pnl_fields_present",
        "journal_rows_sourced_only_from_paper_replay_candidates",
        "journal_entry_ids_are_unique",
    ]

    return {
        "journal_entries_present": journal_present,
        "journal_entry_count": counts["journal_entry"],
        "paper_replay_candidate_count": counts["paper_replay_candidate"],
        "production_candidate_count": counts["production_candidate"],
        "human_review_candidate_count": counts["human_review_candidate"],
        "production_candidates_present": production_present,
        "human_review_candidates_present": human_present,
        "journal_contract_supported": journal_present,
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "journal_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in journal_invariants)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_paper_replay_outcome_tracking_contract":
        return "Stop here. The next stage should design a paper replay outcome tracking contract."
    if recommended_next_stage == "collect_more_data":
        return "Stop here. More data is required before outcome tracking is useful."
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
        f"- journal_entries_present: {final.get('journal_entries_present')}",
        f"- journal_entry_count: {final.get('journal_entry_count')}",
        f"- paper_replay_candidate_count: {final.get('paper_replay_candidate_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- human_review_candidate_count: {final.get('human_review_candidate_count')}",
        f"- journal_contract_supported: {final.get('journal_contract_supported')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- journal_contract_version: {JOURNAL_CONTRACT_VERSION}",
        f"- source_report_type: {replay_contract_report.REPORT_TYPE}",
        f"- observation_status: {OBSERVATION_STATUS}",
        f"- observation_lifecycle_state: {OBSERVATION_LIFECYCLE_STATE}",
        "- Journal rows are report-only observation records, not trades, orders, fills, or live engine candidates.",
        "- Entry price, exit price, realized PnL, and unrealized PnL remain explicit null fields.",
        "- Candidate quality gate, mapper, engine, execution gate, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- journal_safety_invariant_summary: {final.get('journal_safety_invariant_summary')}",
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
                f"- journal_entry_count: {item.get('journal_entry_count')}",
                f"- paper_replay_candidate_count: {item.get('paper_replay_candidate_count')}",
                f"- human_review_candidate_count: {item.get('human_review_candidate_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- journal_safety_invariants: {item.get('journal_safety_invariants')}",
                f"- best_journal_row: {item.get('best_journal_row')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _journal_sort_key(row: dict[str, Any]) -> tuple[float, str, str, str, str]:
    return (
        -float(row.get("aggregate_score") or 0.0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("journal_entry_id") or ""),
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
    return replay_contract_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return replay_contract_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return replay_contract_report._safe_int(value)


def _safe_float(value: Any) -> float | None:
    return replay_contract_report._safe_float(value)


if __name__ == "__main__":
    main()
