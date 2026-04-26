from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_paper_replay_outcome_attachment_source_contract_report as attachment_source_report,
)

REPORT_TYPE = (
    "selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report"
)
REPORT_TITLE = (
    "Selected Strategy Edge Candidate Aggregate Metric Paper Replay Evaluation "
    "Report"
)
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

EVALUATION_CONTRACT_VERSION = "aggregate_metric_paper_replay_evaluation_v1"
EVALUATION_MODE = "aggregate_metric_only_evaluation"

PROMISING_BUCKET = "aggregate_metric_promising_observation"
WATCHLIST_BUCKET = "aggregate_metric_watchlist_observation"
WEAK_BUCKET = "aggregate_metric_weak_observation"
UNAVAILABLE_BUCKET = "aggregate_metric_unavailable"

_BUCKET_SORT_ORDER = {
    PROMISING_BUCKET: 0,
    WATCHLIST_BUCKET: 1,
    WEAK_BUCKET: 2,
    UNAVAILABLE_BUCKET: 3,
}

REQUIRED_AGGREGATE_METRIC_FIELDS = (
    "aggregate_sample_count",
    "aggregate_labeled_count",
    "aggregate_median_future_return_pct",
    "aggregate_positive_rate_pct",
    "aggregate_robustness_signal_pct",
    "aggregate_score",
)

PCT_FIELDS = {
    "aggregate_positive_rate_pct",
    "aggregate_robustness_signal_pct",
}
COUNT_FIELDS = {
    "aggregate_sample_count",
    "aggregate_labeled_count",
}

DEFAULT_INPUT_PATH = attachment_source_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = attachment_source_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = attachment_source_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = attachment_source_report.SupportWindowConfiguration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only aggregate metric paper replay evaluation from "
            "paper replay outcome attachment source rows. Aggregate diagnostic "
            "metrics remain aggregate-only observations and are not treated as "
            "exact outcomes, trades, fills, prices, or PnL."
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
    result = run_selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report(
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
                "aggregate_evaluation_rows_present": final.get(
                    "aggregate_evaluation_rows_present"
                ),
                "aggregate_evaluation_row_count": final.get(
                    "aggregate_evaluation_row_count"
                ),
                "promising_observation_count": final.get(
                    "promising_observation_count"
                ),
                "watchlist_observation_count": final.get(
                    "watchlist_observation_count"
                ),
                "weak_observation_count": final.get("weak_observation_count"),
                "unavailable_count": final.get("unavailable_count"),
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
    return attachment_source_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return attachment_source_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_aggregate_metric_paper_replay_evaluation_report(
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
    source_attachment_report = attachment_source_report.build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
    )
    source_summaries = [
        summary
        for summary in _safe_list(
            source_attachment_report.get("configuration_summaries")
        )
        if isinstance(summary, dict)
    ]
    summaries = [
        build_configuration_summary(source_attachment_summary=summary)
        for summary in source_summaries
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": _safe_dict(source_attachment_report.get("inputs")),
        "diagnostic_only": True,
        "aggregate_metric_paper_replay_evaluation_contract": {
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "evaluation_mode": EVALUATION_MODE,
            "source_report_type": attachment_source_report.REPORT_TYPE,
            "source_outcome_attachment_source_contract_version": (
                attachment_source_report.OUTCOME_ATTACHMENT_SOURCE_CONTRACT_VERSION
            ),
            "aggregate_metrics_are_exact_future_outcomes": False,
            "aggregate_metrics_are_trade_results": False,
            "aggregate_metrics_are_fill_results": False,
            "aggregate_metrics_are_realized_or_unrealized_pnl": False,
            "evaluation_rows_enter_live_mapper_or_engine": False,
            "synthetic_prices_orders_fills_or_pnl_created": False,
            "synthetic_future_labels_or_returns_created": False,
        },
        "source_outcome_attachment_source_final_assessment": _safe_dict(
            source_attachment_report.get("final_assessment")
        ),
        "configurations_evaluated": _safe_list(
            source_attachment_report.get("configurations_evaluated")
        ),
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Evaluation rows are report-only observations derived from aggregate diagnostic metric attachment source rows.",
            "Aggregate diagnostic metrics are evaluated only as aggregate-only evidence when exact outcome fields are unavailable.",
            "Aggregate metrics are not exact future outcomes, trade results, fill results, realized PnL, or unrealized PnL.",
            "No synthetic future label, future return, price, order, fill, or PnL is created.",
            "Existing candidate quality gates, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        ],
    }


def build_configuration_summary(
    *,
    source_attachment_summary: dict[str, Any],
) -> dict[str, Any]:
    source_summary = dict(source_attachment_summary)
    configuration = _safe_dict(source_summary.get("configuration"))
    attachment_rows = [
        row
        for row in _safe_list(source_summary.get("attachment_source_rows"))
        if isinstance(row, dict) and is_aggregate_metric_evaluation_source_row(row)
    ]
    evaluation_rows = [
        build_aggregate_metric_evaluation_row(attachment_source_row=row)
        for row in attachment_rows
    ]
    evaluation_rows = deduplicate_aggregate_metric_evaluation_ids(evaluation_rows)
    evaluation_rows.sort(key=_aggregate_metric_evaluation_sort_key)

    summary = {
        "configuration": configuration,
        "source_attachment_summary": source_summary,
        "aggregate_evaluation_row_count": len(evaluation_rows),
        "paper_replay_candidate_count": _safe_int(
            source_summary.get("paper_replay_candidate_count")
        ),
        "production_candidate_count": _safe_int(
            source_summary.get("production_candidate_count")
        ),
        "aggregate_evaluation_rows": evaluation_rows,
        "best_aggregate_metric_evaluation_row": _first(evaluation_rows),
        "aggregate_metric_evaluation_bucket_summary": (
            build_aggregate_metric_evaluation_bucket_summary(evaluation_rows)
        ),
    }
    summary["aggregate_metric_evaluation_safety_invariants"] = (
        build_aggregate_metric_evaluation_safety_invariants(summary)
    )
    return summary


def is_aggregate_metric_evaluation_source_row(row: dict[str, Any]) -> bool:
    return (
        row.get("attachment_source_status")
        == attachment_source_report.AGGREGATE_ATTACHMENT_STATUS
        and row.get("aggregate_metric_attachment_available") is True
    )


def build_aggregate_metric_evaluation_row(
    *,
    attachment_source_row: dict[str, Any],
) -> dict[str, Any]:
    source_id = str(attachment_source_row.get("outcome_attachment_source_id") or "")
    bucket, reason = classify_aggregate_metric_bucket(attachment_source_row)
    validation_errors = aggregate_metric_validation_errors(attachment_source_row)
    safety_notes = attachment_source_report.outcome_tracking_report.journal_report.replay_contract_report.near_miss_report.normalize_string_list(
        attachment_source_row.get("safety_notes")
    )
    safety_notes.extend(
        [
            "Aggregate metric evaluation row is report-only and is not a trade.",
            "Aggregate metrics remain aggregate-only evidence and are not exact outcomes, trade results, fill results, or PnL.",
            "No future label, future return, price, order, fill, realized PnL, or unrealized PnL is synthesized.",
            "The row must not be fed into the live mapper, engine, or production selection path.",
        ]
    )

    return {
        "aggregate_metric_evaluation_id": build_aggregate_metric_evaluation_id(
            source_outcome_attachment_source_id=source_id
        ),
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_mode": EVALUATION_MODE,
        "source_outcome_attachment_source_id": source_id,
        "source_outcome_tracking_id": attachment_source_row.get(
            "source_outcome_tracking_id"
        ),
        "source_journal_entry_id": attachment_source_row.get(
            "source_journal_entry_id"
        ),
        "source_paper_replay_candidate_id": attachment_source_row.get(
            "source_paper_replay_candidate_id"
        ),
        "symbol": attachment_source_row.get("symbol"),
        "strategy": attachment_source_row.get("strategy"),
        "horizon": attachment_source_row.get("horizon"),
        "source_policy_class": attachment_source_row.get("source_policy_class"),
        "attachment_source_status": attachment_source_row.get(
            "attachment_source_status"
        ),
        "aggregate_sample_count": _safe_optional_int(
            attachment_source_row.get("aggregate_sample_count")
        ),
        "aggregate_labeled_count": _safe_optional_int(
            attachment_source_row.get("aggregate_labeled_count")
        ),
        "aggregate_median_future_return_pct": _safe_float(
            attachment_source_row.get("aggregate_median_future_return_pct")
        ),
        "aggregate_positive_rate_pct": _safe_float(
            attachment_source_row.get("aggregate_positive_rate_pct")
        ),
        "aggregate_robustness_signal": attachment_source_row.get(
            "aggregate_robustness_signal"
        ),
        "aggregate_robustness_signal_pct": _safe_float(
            attachment_source_row.get("aggregate_robustness_signal_pct")
        ),
        "aggregate_score": _safe_float(attachment_source_row.get("aggregate_score")),
        "aggregate_metric_validation_errors": validation_errors,
        "evaluation_bucket": bucket,
        "evaluation_reason": reason,
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
        "exact_outcome_used": False,
        "aggregate_metric_only": True,
        "safety_notes": safety_notes,
    }


def classify_aggregate_metric_bucket(row: dict[str, Any]) -> tuple[str, str]:
    validation_errors = aggregate_metric_validation_errors(row)
    if validation_errors:
        return (
            UNAVAILABLE_BUCKET,
            "Required aggregate metric fields are missing or invalid: "
            + ", ".join(validation_errors),
        )

    sample_count = _safe_optional_int(row.get("aggregate_sample_count"))
    median_return = _safe_float(row.get("aggregate_median_future_return_pct"))
    positive_rate = _safe_float(row.get("aggregate_positive_rate_pct"))
    aggregate_score = _safe_float(row.get("aggregate_score"))

    if sample_count is None or median_return is None:
        return (
            UNAVAILABLE_BUCKET,
            "Required aggregate metric fields are unavailable after validation.",
        )
    if positive_rate is None or aggregate_score is None:
        return (
            UNAVAILABLE_BUCKET,
            "Required aggregate metric fields are unavailable after validation.",
        )

    if (
        sample_count >= 30
        and median_return > 0
        and positive_rate >= 45
        and aggregate_score >= 60
    ):
        return (
            PROMISING_BUCKET,
            "Aggregate metrics satisfy the conservative promising observation thresholds.",
        )
    if (
        sample_count >= 30
        and median_return > 0
        and positive_rate >= 40
        and aggregate_score >= 50
    ):
        return (
            WATCHLIST_BUCKET,
            "Aggregate metrics satisfy the conservative watchlist observation thresholds.",
        )
    return (
        WEAK_BUCKET,
        "Aggregate metrics are available but do not satisfy promising or watchlist thresholds.",
    )


def aggregate_metric_validation_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for field in REQUIRED_AGGREGATE_METRIC_FIELDS:
        if not _has_value(row.get(field)):
            errors.append(f"{field}:missing")

    for field in COUNT_FIELDS:
        value = row.get(field)
        if not _has_value(value):
            continue
        parsed = _safe_optional_int(value)
        if parsed is None:
            errors.append(f"{field}:invalid_integer")
        elif parsed < 0:
            errors.append(f"{field}:negative")

    sample_count = _safe_optional_int(row.get("aggregate_sample_count"))
    labeled_count = _safe_optional_int(row.get("aggregate_labeled_count"))
    if (
        sample_count is not None
        and labeled_count is not None
        and labeled_count > sample_count
    ):
        errors.append("aggregate_labeled_count:greater_than_sample_count")

    for field in (
        "aggregate_median_future_return_pct",
        "aggregate_positive_rate_pct",
        "aggregate_robustness_signal_pct",
        "aggregate_score",
    ):
        value = row.get(field)
        if not _has_value(value):
            continue
        parsed_float = _safe_float(value)
        if parsed_float is None:
            errors.append(f"{field}:invalid_number")

    for field in PCT_FIELDS:
        value = row.get(field)
        if not _has_value(value):
            continue
        parsed_pct = _safe_float(value)
        if parsed_pct is not None and not 0.0 <= parsed_pct <= 100.0:
            errors.append(f"{field}:outside_0_100")

    aggregate_score = _safe_float(row.get("aggregate_score"))
    if aggregate_score is not None and not 0.0 <= aggregate_score <= 100.0:
        errors.append("aggregate_score:outside_0_100")

    return errors


def build_aggregate_metric_evaluation_id(
    *,
    source_outcome_attachment_source_id: str,
) -> str:
    return (
        f"{EVALUATION_CONTRACT_VERSION}:"
        f"{source_outcome_attachment_source_id}"
    )


def deduplicate_aggregate_metric_evaluation_ids(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    deduplicated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        base_id = str(item.get("aggregate_metric_evaluation_id") or "")
        seen[base_id] += 1
        if seen[base_id] > 1:
            item["aggregate_metric_evaluation_id"] = f"{base_id}:dup_{seen[base_id]}"
        deduplicated.append(item)
    return deduplicated


def build_aggregate_metric_evaluation_bucket_summary(
    evaluation_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in evaluation_rows if isinstance(row, dict)]
    buckets = Counter(str(row.get("evaluation_bucket") or "") for row in rows)
    return {
        "aggregate_evaluation_row_count": len(rows),
        "promising_observation_count": buckets[PROMISING_BUCKET],
        "watchlist_observation_count": buckets[WATCHLIST_BUCKET],
        "weak_observation_count": buckets[WEAK_BUCKET],
        "unavailable_count": buckets[UNAVAILABLE_BUCKET],
        "bucket_counts": dict(buckets),
    }


def build_aggregate_metric_evaluation_safety_invariants(
    summary: dict[str, Any],
) -> dict[str, Any]:
    evaluation_rows = [
        row
        for row in _safe_list(summary.get("aggregate_evaluation_rows"))
        if isinstance(row, dict)
    ]
    evaluation_ids = [
        row.get("aggregate_metric_evaluation_id") for row in evaluation_rows
    ]

    return {
        "all_evaluation_rows_production_live_selection_disallowed": all(
            row.get("production_live_selection_allowed") is False
            for row in evaluation_rows
        ),
        "all_evaluation_rows_mapper_live_path_disallowed": all(
            row.get("mapper_live_path_allowed") is False for row in evaluation_rows
        ),
        "all_evaluation_rows_engine_live_path_disallowed": all(
            row.get("engine_live_path_allowed") is False for row in evaluation_rows
        ),
        "all_evaluation_rows_no_order_execution": all(
            row.get("no_order_execution") is True for row in evaluation_rows
        ),
        "all_evaluation_rows_no_synthetic_fill": all(
            row.get("no_synthetic_fill") is True for row in evaluation_rows
        ),
        "all_evaluation_rows_no_pnl_claim": all(
            row.get("no_pnl_claim") is True for row in evaluation_rows
        ),
        "no_order_or_fill_identifiers_present": all(
            row.get("order_id") is None and row.get("fill_id") is None
            for row in evaluation_rows
        ),
        "no_price_or_pnl_fields_present": all(
            row.get("entry_price") is None
            and row.get("exit_price") is None
            and row.get("realized_pnl") is None
            and row.get("unrealized_pnl") is None
            for row in evaluation_rows
        ),
        "exact_outcomes_not_used": all(
            row.get("exact_outcome_used") is False for row in evaluation_rows
        ),
        "aggregate_metrics_remain_aggregate_only": all(
            row.get("aggregate_metric_only") is True
            and row.get("attachment_source_status")
            == attachment_source_report.AGGREGATE_ATTACHMENT_STATUS
            for row in evaluation_rows
        ),
        "aggregate_metric_evaluation_ids_are_unique": len(evaluation_ids)
        == len(set(evaluation_ids)),
        "aggregate_evaluation_row_count": len(evaluation_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    invariant_summaries = []
    for summary in configuration_summaries:
        bucket_summary = _safe_dict(
            summary.get("aggregate_metric_evaluation_bucket_summary")
        )
        counts["aggregate_evaluation_row"] += _safe_int(
            summary.get("aggregate_evaluation_row_count")
        )
        counts["promising_observation"] += _safe_int(
            bucket_summary.get("promising_observation_count")
        )
        counts["watchlist_observation"] += _safe_int(
            bucket_summary.get("watchlist_observation_count")
        )
        counts["weak_observation"] += _safe_int(
            bucket_summary.get("weak_observation_count")
        )
        counts["unavailable"] += _safe_int(bucket_summary.get("unavailable_count"))
        counts["paper_replay_candidate"] += _safe_int(
            summary.get("paper_replay_candidate_count")
        )
        counts["production_candidate"] += _safe_int(
            summary.get("production_candidate_count")
        )
        invariant_summaries.append(
            _safe_dict(summary.get("aggregate_metric_evaluation_safety_invariants"))
        )

    if counts["promising_observation"] > 0:
        recommended_next_stage = "design_paper_replay_watchlist_report"
    elif counts["watchlist_observation"] > 0:
        recommended_next_stage = "continue_aggregate_observation"
    else:
        recommended_next_stage = "collect_more_data"

    invariant_keys = [
        "all_evaluation_rows_production_live_selection_disallowed",
        "all_evaluation_rows_mapper_live_path_disallowed",
        "all_evaluation_rows_engine_live_path_disallowed",
        "all_evaluation_rows_no_order_execution",
        "all_evaluation_rows_no_synthetic_fill",
        "all_evaluation_rows_no_pnl_claim",
        "no_order_or_fill_identifiers_present",
        "no_price_or_pnl_fields_present",
        "exact_outcomes_not_used",
        "aggregate_metrics_remain_aggregate_only",
        "aggregate_metric_evaluation_ids_are_unique",
    ]

    rows_present = counts["aggregate_evaluation_row"] > 0
    return {
        "aggregate_evaluation_rows_present": rows_present,
        "aggregate_evaluation_row_count": counts["aggregate_evaluation_row"],
        "promising_observation_count": counts["promising_observation"],
        "watchlist_observation_count": counts["watchlist_observation"],
        "weak_observation_count": counts["weak_observation"],
        "unavailable_count": counts["unavailable"],
        "paper_replay_candidate_count": counts["paper_replay_candidate"],
        "production_candidate_count": counts["production_candidate"],
        "recommended_next_stage": recommended_next_stage,
        "candidate_counts": dict(counts),
        "aggregate_metric_evaluation_safety_invariant_summary": {
            key: all(invariants.get(key) is True for invariants in invariant_summaries)
            for key in invariant_keys
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_paper_replay_watchlist_report":
        return "Stop here. The next stage should design a report-only paper replay watchlist report."
    if recommended_next_stage == "continue_aggregate_observation":
        return "Stop here. Continue collecting aggregate-only observations before any live selection design."
    if recommended_next_stage == "collect_more_data":
        return "Stop here. More aggregate diagnostic metric evidence is required before watchlist design."
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
        f"- aggregate_evaluation_rows_present: {final.get('aggregate_evaluation_rows_present')}",
        f"- aggregate_evaluation_row_count: {final.get('aggregate_evaluation_row_count')}",
        f"- promising_observation_count: {final.get('promising_observation_count')}",
        f"- watchlist_observation_count: {final.get('watchlist_observation_count')}",
        f"- weak_observation_count: {final.get('weak_observation_count')}",
        f"- unavailable_count: {final.get('unavailable_count')}",
        f"- production_candidate_count: {final.get('production_candidate_count')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract Summary",
        f"- evaluation_contract_version: {EVALUATION_CONTRACT_VERSION}",
        f"- source_report_type: {attachment_source_report.REPORT_TYPE}",
        f"- evaluation_mode: {EVALUATION_MODE}",
        "- Evaluation rows are report-only aggregate metric observations, not trades, orders, fills, or live engine candidates.",
        "- Aggregate diagnostic metrics remain aggregate-only evidence and are not exact future outcomes, trade results, fill results, or PnL.",
        "- Entry price, exit price, realized PnL, and unrealized PnL remain explicit null fields.",
        "- Candidate quality gate, mapper, engine, execution gate, runtime trading logic, and production thresholds are unchanged.",
        "",
        "## Safety Invariants",
        f"- aggregate_metric_evaluation_safety_invariant_summary: {final.get('aggregate_metric_evaluation_safety_invariant_summary')}",
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
                f"- aggregate_evaluation_row_count: {item.get('aggregate_evaluation_row_count')}",
                f"- paper_replay_candidate_count: {item.get('paper_replay_candidate_count')}",
                f"- production_candidate_count: {item.get('production_candidate_count')}",
                f"- aggregate_metric_evaluation_bucket_summary: {item.get('aggregate_metric_evaluation_bucket_summary')}",
                f"- aggregate_metric_evaluation_safety_invariants: {item.get('aggregate_metric_evaluation_safety_invariants')}",
                f"- best_aggregate_metric_evaluation_row: {item.get('best_aggregate_metric_evaluation_row')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _aggregate_metric_evaluation_sort_key(
    row: dict[str, Any],
) -> tuple[int, float, int, str, str, str, str, str]:
    return (
        _BUCKET_SORT_ORDER.get(str(row.get("evaluation_bucket") or ""), 9),
        -float(row.get("aggregate_score") or 0.0),
        -int(row.get("aggregate_sample_count") or 0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("source_outcome_tracking_id") or ""),
        str(row.get("aggregate_metric_evaluation_id") or ""),
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
    return attachment_source_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return attachment_source_report._safe_list(value)


def _safe_int(value: Any) -> int:
    return attachment_source_report._safe_int(value)


def _safe_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not _has_value(value):
        return None

    parsed = _safe_float(value)
    if parsed is None:
        return None

    if not parsed.is_integer():
        return None

    integer = int(parsed)
    if integer < 0:
        return None

    return integer


def _safe_float(value: Any) -> float | None:
    parsed = attachment_source_report._safe_float(value)
    if parsed is None:
        return None
    if not isfinite(parsed):
        return None
    return parsed


if __name__ == "__main__":
    main()
