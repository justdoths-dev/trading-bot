from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_observation_journal_report as journal_report,
)

REPORT_TYPE = "selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Paper Replay Outcome Tracking Contract Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

OUTCOME_TRACKING_CONTRACT_VERSION = "paper_replay_outcome_tracking_v1"
TRACKING_MODE = "label_only_observation"

DEFAULT_INPUT_PATH = journal_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = journal_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = journal_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = journal_report.SupportWindowConfiguration

FUTURE_FIELD_BY_HORIZON = {
    "15m": ("future_label_15m", "future_return_15m"),
    "1h": ("future_label_1h", "future_return_1h"),
    "4h": ("future_label_4h", "future_return_4h"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only paper replay outcome tracking contract from "
            "paper replay observation journal rows. This does not create "
            "trades, orders, fills, prices, or PnL."
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
    result = (
        run_selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report(
            input_path=resolve_path(args.input),
            output_dir=resolve_path(args.output_dir),
            configurations=parse_configuration_values(args.config),
            write_report_copies=args.write_latest_copy,
        )
    )
    final = _safe_dict(result["report"].get("final_assessment"))
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "recommended_next_stage": final.get("recommended_next_stage"),
                "outcome_tracking_rows_present": final.get(
                    "outcome_tracking_rows_present"
                ),
                "outcome_tracking_contract_supported": final.get(
                    "outcome_tracking_contract_supported"
                ),
                "outcome_tracking_row_count": final.get(
                    "outcome_tracking_row_count"
                ),
                "outcome_observation_available_count": final.get(
                    "outcome_observation_available_count"
                ),
                "outcome_observation_unavailable_count": final.get(
                    "outcome_observation_unavailable_count"
                ),
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
    return journal_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return journal_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report(
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
    source_journal_report = journal_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(source_journal_report.get("configuration_summaries"))
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_journal_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_journal_report.get("inputs")),
        "diagnostic_only": True,
        "outcome_tracking_contract": {
            "outcome_tracking_contract_version": OUTCOME_TRACKING_CONTRACT_VERSION,
            "tracking_mode": TRACKING_MODE,
            "source_report_type": journal_report.REPORT_TYPE,
            "source_journal_contract_version": journal_report.JOURNAL_CONTRACT_VERSION,
            "outcome_tracking_rows_are_trades": False,
            "outcome_tracking_rows_are_orders": False,
            "outcome_tracking_rows_are_fills": False,
            "outcome_tracking_rows_enter_live_mapper_or_engine": False,
            "synthetic_prices_or_pnl_created": False,
            "external_market_data_join_performed": False,
        },
        "source_journal_final_assessment": _safe_dict(
            source_journal_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_journal_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Outcome tracking rows are report-only records derived from paper replay observation journal rows.",
            "Future labels and returns are copied only when already present on the journal row for the matching horizon.",
            "No return, price, fill, order, realized PnL, or unrealized PnL is synthesized.",
            "Outcome tracking rows must not enter the live mapper or engine path.",
            "Existing candidate quality gates, mapper, engine, execution gate, and production thresholds are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_journal_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_journal_summary)
    configuration = _safe_dict(source_summary.get("configuration"))
    journal_rows = [
        row
        for row in _safe_list(source_summary.get("journal_rows"))
        if isinstance(row, dict)
    ]
    outcome_rows = [
        build_outcome_tracking_row(journal_row=row)
        for row in journal_rows
        if is_outcome_tracking_source_row(row)
    ]
    outcome_rows.sort(key=_outcome_tracking_sort_key)

    summary = {
        "configuration": configuration,
        "source_journal_summary": source_summary,
        "outcome_tracking_row_count": len(outcome_rows),
        "paper_replay_candidate_count": _safe_int(
            source_summary.get("paper_replay_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("production_candidate_count")
        ),
        "human_review_candidate_count": _safe_int(
            source_summary.get("human_review_candidate_count")
        ),
        "outcome_rows": outcome_rows,
        "best_outcome_tracking_row": _first(outcome_rows),
        "outcome_availability_summary": build_outcome_availability_summary(
            outcome_rows
        ),
    }
    summary["outcome_tracking_safety_invariants"] = (
        build_outcome_tracking_safety_invariants(summary)
    )
    return summary


def is_outcome_tracking_source_row(row: dict[str, Any]) -> bool:
    return (
        row.get("journal_contract_version") == journal_report.JOURNAL_CONTRACT_VERSION
        and row.get("journal_entry_id")
        and row.get("paper_replay_candidate_id")
        and row.get("source_policy_class") == journal_report.PAPER_ONLY_POLICY_CLASS
    )


def build_outcome_tracking_row(*, journal_row: dict[str, Any]) -> dict[str, Any]:
    journal_entry_id = str(journal_row.get("journal_entry_id") or "")
    future_label, future_return, tracked_horizon = extract_outcome_fields(journal_row)
    label_available = future_label is not None
    return_available = future_return is not None
    observation_available = label_available and return_available
    safety_notes = journal_report.replay_contract_report.near_miss_report.normalize_string_list(
        journal_row.get("safety_notes")
    )
    safety_notes.extend(
        [
            "Outcome tracking row is report-only and is not a trade.",
            "No future return, price, fill, order, realized PnL, or unrealized PnL is synthesized.",
            "The row must not be fed into the live mapper or engine path.",
        ]
    )

    return {
        "outcome_tracking_id": build_outcome_tracking_id(
            journal_entry_id=journal_entry_id
        ),
        "outcome_tracking_contract_version": OUTCOME_TRACKING_CONTRACT_VERSION,
        "journal_entry_id": journal_entry_id,
        "paper_replay_candidate_id": journal_row.get("paper_replay_candidate_id"),
        "source_report_type": journal_report.REPORT_TYPE,
        "source_policy_class": journal_row.get("source_policy_class"),
        "symbol": journal_row.get("symbol"),
        "strategy": journal_row.get("strategy"),
        "horizon": journal_row.get("horizon"),
        "observation_status": journal_row.get("observation_status"),
        "observation_lifecycle_state": journal_row.get(
            "observation_lifecycle_state"
        ),
        "tracking_mode": TRACKING_MODE,
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
        "tracked_future_label": future_label,
        "tracked_future_return_pct": future_return,
        "tracked_horizon": tracked_horizon,
        "outcome_label_available": label_available,
        "outcome_return_available": return_available,
        "outcome_observation_available": observation_available,
        "source_rejection_reason": journal_row.get("source_rejection_reason"),
        "source_rejection_reasons": journal_report.replay_contract_report.near_miss_report.normalize_string_list(
            journal_row.get("source_rejection_reasons")
        ),
        "near_miss_classification": journal_row.get("near_miss_classification"),
        "suggested_next_policy_bucket": journal_row.get(
            "suggested_next_policy_bucket"
        ),
        "safety_notes": safety_notes,
    }


def extract_outcome_fields(
    journal_row: dict[str, Any],
) -> tuple[str | None, float | None, str | None]:
    horizon = _safe_text(journal_row.get("horizon"))
    field_names = FUTURE_FIELD_BY_HORIZON.get(horizon or "")
    if field_names is None:
        return None, None, horizon

    label_field, return_field = field_names
    return (
        _safe_text(journal_row.get(label_field)),
        _safe_float(journal_row.get(return_field)),
        horizon,
    )


def build_outcome_tracking_id(*, journal_entry_id: str) -> str:
    return f"{OUTCOME_TRACKING_CONTRACT_VERSION}:{journal_entry_id}"


def build_outcome_availability_summary(
    outcome_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in outcome_rows if isinstance(row, dict)]
    available_count = sum(
        1 for row in rows if row.get("outcome_observation_available") is True
    )
    label_available_count = sum(
        1 for row in rows if row.get("outcome_label_available") is True
    )
    return_available_count = sum(
        1 for row in rows if row.get("outcome_return_available") is True
    )
    unavailable_count = len(rows) - available_count

    return {
        "outcome_tracking_row_count": len(rows),
        "outcome_observation_available_count": available_count,
        "outcome_observation_unavailable_count": unavailable_count,
        "outcome_label_available_count": label_available_count,
        "outcome_label_unavailable_count": len(rows) - label_available_count,
        "outcome_return_available_count": return_available_count,
        "outcome_return_unavailable_count": len(rows) - return_available_count,
    }


def build_outcome_tracking_safety_invariants(
    summary: dict[str, Any],
) -> dict[str, Any]:
    outcome_rows = [
        row for row in _safe_list(summary.get("outcome_rows")) if isinstance(row, dict)
    ]
    outcome_tracking_ids = [row.get("outcome_tracking_id") for row in outcome_rows]

    return {
        "all_outcome_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False
            for row in outcome_rows
        ),
        "all_outcome_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in outcome_rows
        ),
        "all_outcome_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in outcome_rows
        ),
        "all_outcome_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in outcome_rows
        ),
        "all_outcome_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in outcome_rows
        ),
        "all_outcome_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in outcome_rows
        ),
        "no_order_or_fill_identifiers_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in outcome_rows
        ),
        "no_price_or_pnl_fields_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in outcome_rows
        ),
        "outcome_rows_sourced_only_from_paper_replay_observation_journal_rows": all(
            row.get("source_report_type") == journal_report.REPORT_TYPE
            and row.get("source_policy_class") == journal_report.PAPER_ONLY_POLICY_CLASS
            and row.get("journal_entry_id")
            and row.get("paper_replay_candidate_id")
            for row in outcome_rows
        ),
        "outcome_tracking_ids_are_unique": len(outcome_tracking_ids)
        == len(set(outcome_tracking_ids)),
        "outcome_tracking_row_count": len(outcome_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invariant_summaries = []
    for summary in configuration_summaries:
        availability = _safe_dict(summary.get("outcome_availability_summary"))
        counts["outcome_tracking_row"] += _safe_int(
            summary.get("outcome_tracking_row_count")
        )
        counts["outcome_observation_available"] += _safe_int(
            availability.get("outcome_observation_available_count")
        )
        counts["outcome_observation_unavailable"] += _safe_int(
            availability.get("outcome_observation_unavailable_count")
        )
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )
        counts["human_review_candidate"] += _safe_int(
            summary.get("human_review_candidate_count")
        )
        invariant_summaries.append(
            _safe_dict(summary.get("outcome_tracking_safety_invariants"))
        )

    rows_present = counts["outcome_tracking_row"] > 0
    available_present = counts["outcome_observation_available"] > 0

    if rows_present and available_present:
        recommended_next_stage = "design_paper_replay_result_evaluation_report"
    elif rows_present:
        recommended_next_stage = "design_outcome_attachment_source_contract"
    else:
        recommended_next_stage = "collect_more_data"

    invariant_keys = [
        "all_outcome_rows_production_live_selection_disallowed",
        "all_outcome_rows_mapper_live_path_disallowed",
        "all_outcome_rows_engine_live_path_disallowed",
        "all_outcome_rows_no_order_execution",
        "all_outcome_rows_no_synthetic_fill",
        "all_outcome_rows_no_pnl_claim",
        "no_order_or_fill_identifiers_present",
        "no_price_or_pnl_fields_present",
        "outcome_rows_sourced_only_from_paper_replay_observation_journal_rows",
        "outcome_tracking_ids_are_unique",
    ]

    return {
        "outcome_tracking_rows_present": rows_present,
        "outcome_tracking_row_count": counts["outcome_tracking_row"],
        "outcome_observation_available_count": counts[
            "outcome_observation_available"
        ],
        "outcome_observation_unavailable_count": counts[
            "outcome_observation_unavailable"
        ],
        "paper_replay_candidate_count": counts["paper_replay_candidate"],
        "production_candidate_count": counts["production_candidate"],
        "human_review_candidate_count": counts["human_review_candidate"],
        "outcome_tracking_contract_supported": rows_present,
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "outcome_tracking_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in invariant_summaries)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_outcome_attachment_source_contract":
        return "Stop here. The next stage should define a source contract for attaching future labels and returns."
    if recommended_next_stage == "design_paper_replay_result_evaluation_report":
        return "Stop here. The next stage should design a report-only paper replay result evaluation report."
    if recommended_next_stage == "collect_more_data":
        return "Stop here. More paper replay observation journal rows are required before outcome tracking is useful."
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
        f"- outcome_tracking_rows_present: {final.get('outcome_tracking_rows_present')}",
        f"- outcome_tracking_row_count: {final.get('outcome_tracking_row_count')}",
        f"- outcome_observation_available_count: {final.get('outcome_observation_available_count')}",
        f"- outcome_observation_unavailable_count: {final.get('outcome_observation_unavailable_count')}",
        f"- paper_replay_candidate_count: {final.get('paper_replay_candidate_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- human_review_candidate_count: {final.get('human_review_candidate_count')}",
        f"- outcome_tracking_contract_supported: {final.get('outcome_tracking_contract_supported')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- outcome_tracking_contract_version: {OUTCOME_TRACKING_CONTRACT_VERSION}",
        f"- source_report_type: {journal_report.REPORT_TYPE}",
        f"- tracking_mode: {TRACKING_MODE}",
        "- Outcome rows are report-only observation records, not trades, orders, fills, or live engine candidates.",
        "- Future labels and returns are copied only when already present on the source journal row for the matching horizon.",
        "- Entry price, exit price, realized PnL, and unrealized PnL remain explicit null fields.",
        "- Candidate quality gate, mapper, engine, execution gate, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- outcome_tracking_safety_invariant_summary: {final.get('outcome_tracking_safety_invariant_summary')}",
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
                f"- outcome_tracking_row_count: {item.get('outcome_tracking_row_count')}",
                f"- paper_replay_candidate_count: {item.get('paper_replay_candidate_count')}",
                f"- human_review_candidate_count: {item.get('human_review_candidate_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- outcome_availability_summary: {item.get('outcome_availability_summary')}",
                f"- outcome_tracking_safety_invariants: {item.get('outcome_tracking_safety_invariants')}",
                f"- best_outcome_tracking_row: {item.get('best_outcome_tracking_row')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _outcome_tracking_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("journal_entry_id") or ""),
        str(row.get("outcome_tracking_id") or ""),
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
    return journal_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return journal_report._safe_list(value)


def _safe_text(value: Any) -> str | None:
    return journal_report.replay_contract_report.near_miss_report._safe_text(value)


def _safe_int(value: Any) -> int:
    return journal_report._safe_int(value)


def _safe_float(value: Any) -> float | None:
    return journal_report._safe_float(value)


if __name__ == "__main__":
    main()
