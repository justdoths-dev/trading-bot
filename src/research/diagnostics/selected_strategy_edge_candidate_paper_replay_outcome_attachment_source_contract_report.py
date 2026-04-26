from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_tracking_contract_report as outcome_tracking_report,
)

REPORT_TYPE = (
    "selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report"
)
REPORT_TITLE = (
    "Selected Strategy Edge Candidate Paper Replay Outcome Attachment Source "
    "Contract Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION = "paper_replay_outcome_attachment_source_v1"
ATTACHMENT_MODE = "attachment_source_contract_only"

EXACT_ATTACHMENT_STATUS = "exact_horizon_future_label_return_available"
AGGREGATE_ATTACHMENT_STATUS = "aggregate_diagnostic_metrics_only"
UNAVAILABLE_ATTACHMENT_STATUS = "outcome_source_unavailable"

DEFAULT_INPUT_PATH = outcome_tracking_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = outcome_tracking_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = outcome_tracking_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = outcome_tracking_report.SupportWindowConfiguration

AGGREGATE_METRIC_FIELDS = (
    "sample_count",
    "labeled_count",
    "median_future_return_pct",
    "positive_rate_pct",
    "robustness_signal_pct",
    "aggregate_score",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only paper replay outcome attachment source contract "
            "from paper replay outcome tracking rows. This reports whether exact "
            "future labels/returns or aggregate diagnostic metrics are available; "
            "it does not create trades, orders, fills, prices, or PnL."
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
    result = run_selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report(
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
                "attachment_source_rows_present": final.get(
                    "attachment_source_rows_present"
                ),
                "attachment_source_row_count": final.get(
                    "attachment_source_row_count"
                ),
                "exact_outcome_attachment_available_count": final.get(
                    "exact_outcome_attachment_available_count"
                ),
                "aggregate_metric_attachment_available_count": final.get(
                    "aggregate_metric_attachment_available_count"
                ),
                "aggregate_metric_only_count": final.get(
                    "aggregate_metric_only_count"
                ),
                "outcome_source_unavailable_count": final.get(
                    "outcome_source_unavailable_count"
                ),
                "paper_replay_candidate_count": final.get(
                    "paper_replay_candidate_count"
                ),
                "production_candidate_count": final.get(
                    "production_candidate_count"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return outcome_tracking_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return outcome_tracking_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report(
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
    source_outcome_tracking_report = outcome_tracking_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(
            source_outcome_tracking_report.get("configuration_summaries")
        )
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_outcome_tracking_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_outcome_tracking_report.get("inputs")),
        "diagnostic_only": True,
        "outcome_attachment_source_contract": {
            "outcome_attachment_source_contract_version": (
                OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION
            ),
            "attachment_mode": ATTACHMENT_MODE,
            "source_report_type": outcome_tracking_report.REPORT_TYPE,
            "source_outcome_tracking_contract_version": (
                outcome_tracking_report.OUTCOME_TRACKING_CONTRACT_VERSION
            ),
            "attachment_source_rows_are_trades": False,
            "attachment_source_rows_are_orders": False,
            "attachment_source_rows_are_fills": False,
            "attachment_source_rows_enter_live_mapper_or_engine": False,
            "synthetic_prices_or_pnl_created": False,
            "synthetic_future_labels_or_returns_created": False,
        },
        "source_outcome_tracking_final_assessment": _safe_dict(
            source_outcome_tracking_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_outcome_tracking_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Attachment source rows are report-only records derived from outcome tracking rows.",
            "Exact future labels and returns are copied only when matching horizon source fields already exist.",
            "Aggregate diagnostic metrics remain aggregate-only evidence and are not renamed as exact outcomes, trade results, fill results, or PnL.",
            "No order, fill, entry price, exit price, realized PnL, unrealized PnL, future label, or future return is synthesized.",
            "Existing candidate quality gates, mapper, engine, execution gate, and production live selection behavior are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_outcome_tracking_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_outcome_tracking_summary)
    configuration = _safe_dict(source_summary.get("configuration"))
    outcome_rows = [
        row
        for row in _safe_list(source_summary.get("outcome_rows"))
        if isinstance(row, dict)
    ]
    journal_rows_by_source = build_journal_row_index(source_summary)
    attachment_rows = [
        build_attachment_source_row(
            outcome_tracking_row=row,
            source_journal_row=find_source_journal_row(row, journal_rows_by_source),
        )
        for row in outcome_rows
    ]
    attachment_rows = deduplicate_attachment_source_ids(attachment_rows)
    attachment_rows.sort(key=_attachment_source_sort_key)

    summary = {
        "configuration": configuration,
        "source_outcome_tracking_summary": source_summary,
        "attachment_source_row_count": len(attachment_rows),
        "paper_replay_candidate_count": _safe_int(
            source_summary.get("paper_replay_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("production_candidate_count")
        ),
        "attachment_source_rows": attachment_rows,
        "best_attachment_source_row": _first(attachment_rows),
        "attachment_source_availability_summary": (
            build_attachment_source_availability_summary(attachment_rows)
        ),
    }
    summary["attachment_source_safety_invariants"] = (
        build_attachment_source_safety_invariants(summary)
    )
    return summary


def build_journal_row_index(
    source_outcome_tracking_summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    source_journal_summary = _safe_dict(
        source_outcome_tracking_summary.get("source_journal_summary")
    )
    journal_rows = [
        row
        for row in _safe_list(source_journal_summary.get("journal_rows"))
        if isinstance(row, dict)
    ]
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in journal_rows:
        journal_entry_id = str(row.get("journal_entry_id") or "")
        paper_replay_candidate_id = str(row.get("paper_replay_candidate_id") or "")
        if journal_entry_id or paper_replay_candidate_id:
            index[(journal_entry_id, paper_replay_candidate_id)] = row
    return index


def find_source_journal_row(
    outcome_tracking_row: dict[str, Any],
    journal_rows_by_source: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    journal_entry_id = str(outcome_tracking_row.get("journal_entry_id") or "")
    paper_replay_candidate_id = str(
        outcome_tracking_row.get("paper_replay_candidate_id") or ""
    )
    return journal_rows_by_source.get((journal_entry_id, paper_replay_candidate_id), {})


def build_attachment_source_row(
    *,
    outcome_tracking_row: dict[str, Any],
    source_journal_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_journal = source_journal_row or {}
    horizon = _safe_text(outcome_tracking_row.get("horizon"))
    source_payload = merge_source_payload(
        outcome_tracking_row=outcome_tracking_row,
        source_journal_row=source_journal,
    )
    (
        exact_available,
        exact_label,
        exact_return,
        exact_label_field,
        exact_return_field,
    ) = extract_exact_source_outcome(source_payload=source_payload, horizon=horizon)
    aggregate_available = (
        not exact_available
        and aggregate_diagnostic_metrics_available(source_payload=source_payload)
    )

    if exact_available:
        attachment_source_status = EXACT_ATTACHMENT_STATUS
        safety_note = (
            "Exact horizon future label and return are copied from existing "
            "source fields."
        )
    elif aggregate_available:
        attachment_source_status = AGGREGATE_ATTACHMENT_STATUS
        safety_note = (
            "Aggregate diagnostic metrics are attached as aggregate-only "
            "evidence, not exact outcomes or PnL."
        )
    else:
        attachment_source_status = UNAVAILABLE_ATTACHMENT_STATUS
        safety_note = (
            "No exact horizon future outcome or aggregate diagnostic metric "
            "source is available."
        )

    safety_notes = outcome_tracking_report.journal_report.replay_contract_report.near_miss_report.normalize_string_list(
        source_payload.get("safety_notes")
    )
    safety_notes.extend(
        [
            safety_note,
            "Attachment source row is report-only and is not a trade.",
            "The row must not be fed into the live mapper or engine path.",
            "No order, fill, price, PnL, future label, or future return is synthesized.",
        ]
    )

    source_outcome_tracking_id = str(
        outcome_tracking_row.get("outcome_tracking_id") or ""
    )
    return {
        "outcome_attachment_source_id": build_outcome_attachment_source_id(
            source_outcome_tracking_id=source_outcome_tracking_id
        ),
        "outcome_attachment_source_contract_version": (
            OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION
        ),
        "attachment_mode": ATTACHMENT_MODE,
        "source_outcome_tracking_id": source_outcome_tracking_id,
        "source_journal_entry_id": outcome_tracking_row.get("journal_entry_id"),
        "source_paper_replay_candidate_id": outcome_tracking_row.get(
            "paper_replay_candidate_id"
        ),
        "symbol": outcome_tracking_row.get("symbol"),
        "strategy": outcome_tracking_row.get("strategy"),
        "horizon": horizon,
        "source_policy_class": outcome_tracking_row.get("source_policy_class"),
        "attachment_source_status": attachment_source_status,
        "exact_outcome_attachment_available": exact_available,
        "exact_attached_future_label": exact_label if exact_available else None,
        "exact_attached_future_return_pct": exact_return if exact_available else None,
        "aggregate_metric_attachment_available": aggregate_available,
        "aggregate_sample_count": (
            _safe_int(source_payload.get("sample_count"))
            if aggregate_available
            else None
        ),
        "aggregate_labeled_count": (
            _safe_int(source_payload.get("labeled_count"))
            if aggregate_available
            else None
        ),
        "aggregate_median_future_return_pct": (
            _safe_float(source_payload.get("median_future_return_pct"))
            if aggregate_available
            else None
        ),
        "aggregate_positive_rate_pct": (
            _safe_float(source_payload.get("positive_rate_pct"))
            if aggregate_available
            else None
        ),
        "aggregate_robustness_signal": (
            source_payload.get("robustness_signal") if aggregate_available else None
        ),
        "aggregate_robustness_signal_pct": (
            _safe_float(source_payload.get("robustness_signal_pct"))
            if aggregate_available
            else None
        ),
        "aggregate_score": (
            _safe_float(source_payload.get("aggregate_score"))
            if aggregate_available
            else None
        ),
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
        "safety_notes": safety_notes,
        "exact_source_label_field": exact_label_field,
        "exact_source_return_field": exact_return_field,
        "exact_source_fields_present": exact_available,
    }


def merge_source_payload(
    *,
    outcome_tracking_row: dict[str, Any],
    source_journal_row: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(source_journal_row)
    payload.update(outcome_tracking_row)
    for key, value in source_journal_row.items():
        if key not in payload or payload.get(key) is None:
            payload[key] = value
    return payload


def extract_exact_source_outcome(
    *,
    source_payload: dict[str, Any],
    horizon: str | None,
) -> tuple[bool, str | None, float | None, str | None, str | None]:
    field_names = outcome_tracking_report.FUTURE_FIELD_BY_HORIZON.get(horizon or "")
    if field_names is None:
        return False, None, None, None, None

    label_field, return_field = field_names
    label_present = _has_value(source_payload.get(label_field))
    return_present = _has_value(source_payload.get(return_field))
    if not label_present or not return_present:
        return False, None, None, label_field, return_field

    label = _safe_text(source_payload.get(label_field))
    future_return = _safe_float(source_payload.get(return_field))
    if label is None or future_return is None:
        return False, None, None, label_field, return_field
    return True, label, future_return, label_field, return_field


def aggregate_diagnostic_metrics_available(*, source_payload: dict[str, Any]) -> bool:
    return all(_has_value(source_payload.get(field)) for field in AGGREGATE_METRIC_FIELDS)


def build_outcome_attachment_source_id(*, source_outcome_tracking_id: str) -> str:
    return (
        f"{OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION}:"
        f"{source_outcome_tracking_id}"
    )


def deduplicate_attachment_source_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = str(item.get("outcome_attachment_source_id") or "")
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["outcome_attachment_source_id"] = f"{base_id}:dup_{seen[base_id]}"
        deduplicated.append(item)
    return deduplicated


def build_attachment_source_availability_summary(
    attachment_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in attachment_rows if isinstance(row, dict)]
    exact_available_count = sum(
        1 for row in rows if row.get("exact_outcome_attachment_available") is True
    )
    aggregate_available_count = sum(
        1 for row in rows if row.get("aggregate_metric_attachment_available") is True
    )
    aggregate_metric_only_count = sum(
        1
        for row in rows
        if row.get("aggregate_metric_attachment_available") is True
        and row.get("exact_outcome_attachment_available") is False
    )
    unavailable_count = sum(
        1
        for row in rows
        if row.get("attachment_source_status") == UNAVAILABLE_ATTACHMENT_STATUS
    )

    return {
        "attachment_source_row_count": len(rows),
        "exact_outcome_attachment_available_count": exact_available_count,
        "exact_outcome_attachment_unavailable_count": (
            len(rows) - exact_available_count
        ),
        "aggregate_metric_attachment_available_count": aggregate_available_count,
        "aggregate_metric_only_count": aggregate_metric_only_count,
        "outcome_source_unavailable_count": unavailable_count,
    }


def build_attachment_source_safety_invariants(
    summary: dict[str, Any],
) -> dict[str, Any]:
    attachment_rows = [
        row
        for row in _safe_list(summary.get("attachment_source_rows"))
        if isinstance(row, dict)
    ]
    attachment_source_ids = [
        row.get("outcome_attachment_source_id") for row in attachment_rows
    ]

    return {
        "all_attachment_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False
            for row in attachment_rows
        ),
        "all_attachment_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in attachment_rows
        ),
        "all_attachment_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in attachment_rows
        ),
        "all_attachment_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in attachment_rows
        ),
        "all_attachment_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in attachment_rows
        ),
        "all_attachment_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in attachment_rows
        ),
        "no_order_or_fill_identifiers_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in attachment_rows
        ),
        "no_price_or_pnl_fields_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in attachment_rows
        ),
        "exact_outcomes_not_synthesized": all(
            (
                row.get("exact_outcome_attachment_available") is False
                and row.get("exact_attached_future_label") is None
                and row.get("exact_attached_future_return_pct") is None
            )
            or (
                row.get("exact_outcome_attachment_available") is True
                and row.get("exact_source_fields_present") is True
                and row.get("exact_attached_future_label") is not None
                and row.get("exact_attached_future_return_pct") is not None
            )
            for row in attachment_rows
        ),
        "aggregate_metrics_not_marked_as_exact_outcomes": all(
            (
                row.get("attachment_source_status") != AGGREGATE_ATTACHMENT_STATUS
            )
            or (
                row.get("aggregate_metric_attachment_available") is True
                and row.get("exact_outcome_attachment_available") is False
                and row.get("exact_attached_future_label") is None
                and row.get("exact_attached_future_return_pct") is None
            )
            for row in attachment_rows
        ),
        "attachment_source_ids_are_unique": len(attachment_source_ids)
        == len(set(attachment_source_ids)),
        "attachment_source_row_count": len(attachment_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invariant_summaries = []
    for summary in configuration_summaries:
        availability = _safe_dict(
            summary.get("attachment_source_availability_summary")
        )
        counts["attachment_source_row"] += _safe_int(
            summary.get("attachment_source_row_count")
        )
        counts["exact_outcome_attachment_available"] += _safe_int(
            availability.get("exact_outcome_attachment_available_count")
        )
        counts["aggregate_metric_attachment_available"] += _safe_int(
            availability.get("aggregate_metric_attachment_available_count")
        )
        counts["aggregate_metric_only"] += _safe_int(
            availability.get("aggregate_metric_only_count")
        )
        counts["outcome_source_unavailable"] += _safe_int(
            availability.get("outcome_source_unavailable_count")
        )
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )
        invariant_summaries.append(
            _safe_dict(summary.get("attachment_source_safety_invariants"))
        )

    exact_available_count = counts["exact_outcome_attachment_available"]
    aggregate_available_count = counts["aggregate_metric_attachment_available"]
    if exact_available_count > 0:
        recommended_next_stage = "design_paper_replay_result_evaluation_report"
    elif aggregate_available_count > 0:
        recommended_next_stage = (
            "design_aggregate_metric_paper_replay_evaluation_report"
        )
    else:
        recommended_next_stage = "collect_more_data_or_attach_source_identity"

    invariant_keys = [
        "all_attachment_rows_production_live_selection_disallowed",
        "all_attachment_rows_mapper_live_path_disallowed",
        "all_attachment_rows_engine_live_path_disallowed",
        "all_attachment_rows_no_order_execution",
        "all_attachment_rows_no_synthetic_fill",
        "all_attachment_rows_no_pnl_claim",
        "no_order_or_fill_identifiers_present",
        "no_price_or_pnl_fields_present",
        "exact_outcomes_not_synthesized",
        "aggregate_metrics_not_marked_as_exact_outcomes",
        "attachment_source_ids_are_unique",
    ]

    rows_present = counts["attachment_source_row"] > 0
    return {
        "attachment_source_rows_present": rows_present,
        "attachment_source_row_count": counts["attachment_source_row"],
        "exact_outcome_attachment_available_count": exact_available_count,
        "exact_outcome_attachment_unavailable_count": (
            counts["attachment_source_row"] - exact_available_count
        ),
        "aggregate_metric_attachment_available_count": aggregate_available_count,
        "aggregate_metric_only_count": counts["aggregate_metric_only"],
        "outcome_source_unavailable_count": counts["outcome_source_unavailable"],
        "paper_replay_candidate_count": counts["paper_replay_candidate"],
        "production_candidate_count": counts["production_candidate"],
        "outcome_attachment_source_contract_supported": rows_present,
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "attachment_source_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in invariant_summaries)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_paper_replay_result_evaluation_report":
        return "Stop here. The next stage should design a report-only paper replay result evaluation report."
    if (
        recommended_next_stage
        == "design_aggregate_metric_paper_replay_evaluation_report"
    ):
        return "Stop here. The next stage should evaluate aggregate diagnostic metrics as aggregate-only evidence."
    if recommended_next_stage == "collect_more_data_or_attach_source_identity":
        return "Stop here. More source identity or future outcome data is required before evaluation."
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
        f"- attachment_source_rows_present: {final.get('attachment_source_rows_present')}",
        f"- attachment_source_row_count: {final.get('attachment_source_row_count')}",
        f"- exact_outcome_attachment_available_count: {final.get('exact_outcome_attachment_available_count')}",
        f"- exact_outcome_attachment_unavailable_count: {final.get('exact_outcome_attachment_unavailable_count')}",
        f"- aggregate_metric_attachment_available_count: {final.get('aggregate_metric_attachment_available_count')}",
        f"- aggregate_metric_only_count: {final.get('aggregate_metric_only_count')}",
        f"- outcome_source_unavailable_count: {final.get('outcome_source_unavailable_count')}",
        f"- paper_replay_candidate_count: {final.get('paper_replay_candidate_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- outcome_attachment_source_contract_version: {OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION}",
        f"- source_report_type: {outcome_tracking_report.REPORT_TYPE}",
        f"- attachment_mode: {ATTACHMENT_MODE}",
        "- Attachment source rows are report-only records, not trades, orders, fills, or live engine candidates.",
        "- Exact future labels and returns are copied only from existing matching horizon source fields.",
        "- Aggregate diagnostic metrics remain aggregate-only evidence and are not exact outcomes, trade results, fill results, or PnL.",
        "- Entry price, exit price, realized PnL, and unrealized PnL remain explicit null fields.",
        "- Candidate quality gate, mapper, engine, execution gate, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- attachment_source_safety_invariant_summary: {final.get('attachment_source_safety_invariant_summary')}",
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
                f"- attachment_source_row_count: {item.get('attachment_source_row_count')}",
                f"- paper_replay_candidate_count: {item.get('paper_replay_candidate_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- attachment_source_availability_summary: {item.get('attachment_source_availability_summary')}",
                f"- attachment_source_safety_invariants: {item.get('attachment_source_safety_invariants')}",
                f"- best_attachment_source_row: {item.get('best_attachment_source_row')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _attachment_source_sort_key(
    row: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("source_outcome_tracking_id") or ""),
        str(row.get("outcome_attachment_source_id") or ""),
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


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _safe_dict(value: Any) -> dict[str, Any]:
    return outcome_tracking_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return outcome_tracking_report._safe_list(value)


def _safe_text(value: Any) -> str | None:
    return outcome_tracking_report._safe_text(value)


def _safe_int(value: Any) -> int:
    return outcome_tracking_report._safe_int(value)


def _safe_float(value: Any) -> float | None:
    return outcome_tracking_report._safe_float(value)


if __name__ == "__main__":
    main()
