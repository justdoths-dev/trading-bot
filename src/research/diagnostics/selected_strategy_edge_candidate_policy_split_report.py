from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.diagnostics import (
    selected_strategy_edge_candidate_near_miss_observability_report as near_miss_report,
)
from src.research.research_analyzer import run_research_analyzer
from src.research.strategy_lab.dataset_builder import (
    build_dataset,
    load_jsonl_records_with_metadata,
)

REPORT_TYPE = "selected_strategy_edge_candidate_policy_split_report"
REPORT_TITLE = "Selected Strategy Edge Candidate Policy Split Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = near_miss_report.DEFAULT_INPUT_PATH
DEFAULT_OUTPUT_DIR = near_miss_report.DEFAULT_OUTPUT_DIR
DEFAULT_CONFIGURATIONS = near_miss_report.DEFAULT_CONFIGURATIONS
SupportWindowConfiguration = near_miss_report.SupportWindowConfiguration

POLICY_CLASSES = (
    "production_eligible_candidate",
    "paper_only_candidate",
    "human_review_candidate",
    "observe_near_miss_candidate",
    "collect_more_data",
    "hard_block",
)
LIVE_ALLOWED_POLICY_CLASSES = {"production_eligible_candidate"}
NON_LIVE_POLICY_CLASSES = set(POLICY_CLASSES) - LIVE_ALLOWED_POLICY_CLASSES

SOURCE_ELIGIBLE_EDGE_CANDIDATE_ROW = "eligible_edge_candidate_row"
SOURCE_REJECTED_DIAGNOSTIC_NEAR_MISS_ROW = "rejected_diagnostic_near_miss_row"

BUCKET_TO_POLICY_CLASS = {
    "paper_only_candidate_review": "paper_only_candidate",
    "human_review_candidate_review": "human_review_candidate",
    "observe_near_miss": "observe_near_miss_candidate",
    "collect_more_data": "collect_more_data",
    "hard_block": "hard_block",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a report-only policy split contract from eligible edge "
            "candidate rows and rejected near-miss diagnostic rows."
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
    result = run_selected_strategy_edge_candidate_policy_split_report(
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
                "production_candidates_present": final.get(
                    "production_candidates_present"
                ),
                "paper_only_candidates_present": final.get(
                    "paper_only_candidates_present"
                ),
                "human_review_candidates_present": final.get(
                    "human_review_candidates_present"
                ),
                "policy_split_supported": final.get("policy_split_supported"),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    return near_miss_report.resolve_path(path)


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    return near_miss_report.parse_configuration_values(values)


def run_selected_strategy_edge_candidate_policy_split_report(
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
    resolved_input = resolve_path(input_path)
    resolved_output = resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)

    summaries: list[dict[str, Any]] = []
    for configuration in effective_configurations:
        analyzer_output_dir = (
            resolved_output / f"_{REPORT_TYPE}" / "analyzer_runs" / configuration.slug
        )
        raw_records, source_metadata = load_jsonl_records_with_metadata(
            path=resolved_input,
            max_age_hours=configuration.latest_window_hours,
            max_rows=configuration.latest_max_rows,
        )
        labelable_dataset = build_dataset(
            path=resolved_input,
            max_age_hours=configuration.latest_window_hours,
            max_rows=configuration.latest_max_rows,
        )
        analyzer_metrics = run_research_analyzer(
            resolved_input,
            analyzer_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
        )

        summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                analyzer_output_dir=analyzer_output_dir,
                analyzer_metrics=analyzer_metrics,
                source_metadata=source_metadata,
                raw_record_count=len(raw_records),
                labelable_count=len(labelable_dataset),
            )
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
        },
        "diagnostic_only": True,
        "report_contract": {
            "live_selection_source": "edge_candidate_rows.rows",
            "diagnostic_policy_source": "edge_candidate_rows.diagnostic_rows",
            "non_live_policy_classes": sorted(NON_LIVE_POLICY_CLASSES),
            "production_gate_changes": False,
            "mapper_engine_execution_changes": False,
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_summaries": summaries,
        "final_assessment": build_final_assessment(summaries),
        "assumptions": [
            "Production-live selection remains limited to existing eligible rows from edge_candidate_rows.rows.",
            "Rows derived from edge_candidate_rows.diagnostic_rows are report-only and non-live.",
            "No production threshold, candidate quality gate, mapper, engine, execution gate, or live trading behavior is changed.",
        ],
    }


def build_configuration_summary(
    *,
    configuration: SupportWindowConfiguration,
    input_path: Path | None = None,
    analyzer_output_dir: Path | None = None,
    analyzer_metrics: dict[str, Any],
    source_metadata: dict[str, Any] | None = None,
    raw_record_count: int | None = None,
    labelable_count: int | None = None,
) -> dict[str, Any]:
    edge_candidate_rows = _safe_dict(analyzer_metrics.get("edge_candidate_rows"))
    eligible_rows = [
        row for row in _safe_list(edge_candidate_rows.get("rows")) if isinstance(row, dict)
    ]
    diagnostic_rows = [
        row
        for row in _safe_list(edge_candidate_rows.get("diagnostic_rows"))
        if isinstance(row, dict)
    ]

    production_rows = [
        build_production_policy_row(row) for row in eligible_rows
    ]
    diagnostic_policy_rows = [
        build_diagnostic_policy_row(row) for row in diagnostic_rows
    ]
    policy_rows = production_rows + diagnostic_policy_rows
    policy_rows.sort(key=_policy_sort_key)

    policy_class_counts = Counter(str(row["policy_class"]) for row in policy_rows)
    top_by_class = top_policy_rows_by_class(policy_rows)
    source = _safe_dict(source_metadata)

    summary = {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path) if input_path is not None else None,
        "analyzer_output_dir": (
            str(analyzer_output_dir) if analyzer_output_dir is not None else None
        ),
        "source_metadata": {
            "raw_record_count": _safe_int(
                source.get("raw_record_count", raw_record_count)
            ),
            "windowed_record_count": _safe_int(
                source.get("windowed_record_count", raw_record_count)
            ),
            "max_age_hours": source.get(
                "max_age_hours", configuration.latest_window_hours
            ),
            "max_rows": source.get("max_rows", configuration.latest_max_rows),
        },
        "labelable_count": _safe_int(labelable_count),
        "eligible_production_candidate_count": policy_class_counts.get(
            "production_eligible_candidate", 0
        ),
        "paper_only_candidate_count": policy_class_counts.get(
            "paper_only_candidate", 0
        ),
        "human_review_candidate_count": policy_class_counts.get(
            "human_review_candidate", 0
        ),
        "observe_near_miss_candidate_count": policy_class_counts.get(
            "observe_near_miss_candidate", 0
        ),
        "collect_more_data_count": policy_class_counts.get("collect_more_data", 0),
        "hard_block_count": policy_class_counts.get("hard_block", 0),
        "policy_class_counts": {
            policy_class: policy_class_counts.get(policy_class, 0)
            for policy_class in POLICY_CLASSES
        },
        "top_policy_rows_by_class": top_by_class,
        "best_paper_only_candidate": _first(top_by_class.get("paper_only_candidate")),
        "best_human_review_candidate": _first(
            top_by_class.get("human_review_candidate")
        ),
        "policy_rows": policy_rows,
    }
    summary["safety_invariants"] = build_safety_invariants(summary)
    return summary


def build_production_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_class": "production_eligible_candidate",
        "source_row_type": SOURCE_ELIGIBLE_EDGE_CANDIDATE_ROW,
        "symbol": row.get("symbol"),
        "strategy": row.get("strategy"),
        "horizon": row.get("horizon"),
        "production_live_selection_allowed": True,
        "paper_replay_allowed": True,
        "human_review_allowed": True,
        "observability_only": False,
        "source_diagnostic_category": None,
        "source_rejection_reason": None,
        "source_rejection_reasons": [],
        "near_miss_classification": None,
        "suggested_next_policy_bucket": None,
        "candidate_strength": _candidate_strength(row),
        "sample_count": _safe_int(row.get("sample_count")),
        "labeled_count": _safe_int(row.get("labeled_count")),
        "median_future_return_pct": _safe_float(row.get("median_future_return_pct")),
        "positive_rate_pct": _safe_float(row.get("positive_rate_pct")),
        "robustness_signal": row.get("robustness_signal"),
        "robustness_signal_pct": _safe_float(row.get("robustness_signal_pct")),
        "aggregate_score": _safe_float(row.get("aggregate_score")),
        "policy_reason": (
            "Existing eligible edge candidate row; this is the only source allowed "
            "to remain production-live eligible."
        ),
        "safety_notes": [
            "Origin is edge_candidate_rows.rows.",
            "This report does not change mapper, engine, execution gate, or production thresholds.",
        ],
    }


def build_diagnostic_policy_row(row: dict[str, Any]) -> dict[str, Any]:
    near_miss_row = (
        dict(row)
        if "suggested_next_policy_bucket" in row or "near_miss_classification" in row
        else near_miss_report.build_near_miss_row(row)
    )
    suggested_bucket = near_miss_row.get("suggested_next_policy_bucket")
    policy_class = BUCKET_TO_POLICY_CLASS.get(str(suggested_bucket), "hard_block")
    unknown_bucket = str(suggested_bucket) not in BUCKET_TO_POLICY_CLASS
    policy_class, safety_override_note = enforce_diagnostic_policy_safety(
        policy_class, near_miss_row
    )
    permissions = policy_permissions(policy_class)

    safety_notes = [
        "Origin is edge_candidate_rows.diagnostic_rows.",
        "Diagnostic-derived rows are explicitly non-live and must not be fed into the live mapper or engine.",
    ]
    if unknown_bucket:
        safety_notes.append("Unknown suggested policy bucket was conservatively hard-blocked.")
    if safety_override_note:
        safety_notes.append(safety_override_note)

    return {
        "policy_class": policy_class,
        "source_row_type": SOURCE_REJECTED_DIAGNOSTIC_NEAR_MISS_ROW,
        "symbol": near_miss_row.get("symbol"),
        "strategy": near_miss_row.get("strategy"),
        "horizon": near_miss_row.get("horizon"),
        "production_live_selection_allowed": False,
        "paper_replay_allowed": permissions["paper_replay_allowed"],
        "human_review_allowed": permissions["human_review_allowed"],
        "observability_only": permissions["observability_only"],
        "source_diagnostic_category": near_miss_row.get("diagnostic_category"),
        "source_rejection_reason": near_miss_row.get("rejection_reason"),
        "source_rejection_reasons": near_miss_report.normalize_string_list(
            near_miss_row.get("rejection_reasons")
        ),
        "near_miss_classification": near_miss_row.get("near_miss_classification"),
        "suggested_next_policy_bucket": suggested_bucket,
        "candidate_strength": _candidate_strength(near_miss_row),
        "sample_count": _safe_int(near_miss_row.get("sample_count")),
        "labeled_count": _safe_int(near_miss_row.get("labeled_count")),
        "median_future_return_pct": _safe_float(
            near_miss_row.get("median_future_return_pct")
        ),
        "positive_rate_pct": _safe_float(near_miss_row.get("positive_rate_pct")),
        "robustness_signal": near_miss_row.get("robustness_signal"),
        "robustness_signal_pct": _safe_float(
            near_miss_row.get("robustness_signal_pct")
        ),
        "aggregate_score": _safe_float(near_miss_row.get("aggregate_score")),
        "policy_reason": policy_reason(policy_class, unknown_bucket=unknown_bucket),
        "safety_notes": safety_notes,
    }


def enforce_diagnostic_policy_safety(
    policy_class: str,
    near_miss_row: dict[str, Any],
) -> tuple[str, str | None]:
    rejection_reasons = set(
        near_miss_report.normalize_string_list(near_miss_row.get("rejection_reasons"))
    )
    diagnostic_category = _safe_text(near_miss_row.get("diagnostic_category"))
    near_miss_classification = _safe_text(
        near_miss_row.get("near_miss_classification")
    )
    median_future_return_pct = _safe_float(
        near_miss_row.get("median_future_return_pct")
    )

    structurally_incompatible = (
        diagnostic_category == "incompatibility"
        or "strategy_horizon_incompatible" in rejection_reasons
        or near_miss_classification == "hard_blocked_incompatible"
    )
    if structurally_incompatible:
        return _override_policy_class(
            policy_class,
            "hard_block",
            "Safety override: structural incompatibility forces hard_block.",
        )

    non_positive_return = (
        near_miss_classification == "negative_return_blocked"
        or "median_future_return_non_positive" in rejection_reasons
        or "median_future_return_pct_non_positive" in rejection_reasons
        or (
            median_future_return_pct is not None
            and median_future_return_pct <= 0
        )
    )
    if non_positive_return:
        return _override_policy_class(
            policy_class,
            "hard_block",
            "Safety override: negative or non-positive median return forces hard_block.",
        )

    sample_limited_reasons = {
        "sample_count_below_absolute_floor",
        "sample_count_zero",
        "no_labeled_rows_for_horizon",
        "missing_median_future_return",
        "no_label_support_for_absolute_minimum_gate",
    }
    sample_limited = (
        near_miss_classification == "insufficient_sample"
        or bool(rejection_reasons & sample_limited_reasons)
    )
    if sample_limited:
        return _override_policy_class(
            policy_class,
            "collect_more_data",
            "Safety override: sample-limited diagnostic row forces collect_more_data.",
        )

    return policy_class, None


def _override_policy_class(
    current_policy_class: str,
    enforced_policy_class: str,
    note: str,
) -> tuple[str, str | None]:
    if current_policy_class == enforced_policy_class:
        return current_policy_class, None
    return enforced_policy_class, note


def policy_permissions(policy_class: str) -> dict[str, bool]:
    if policy_class == "paper_only_candidate":
        return {
            "paper_replay_allowed": True,
            "human_review_allowed": True,
            "observability_only": False,
        }
    if policy_class == "human_review_candidate":
        return {
            "paper_replay_allowed": False,
            "human_review_allowed": True,
            "observability_only": False,
        }
    if policy_class == "observe_near_miss_candidate":
        return {
            "paper_replay_allowed": False,
            "human_review_allowed": False,
            "observability_only": True,
        }
    return {
        "paper_replay_allowed": False,
        "human_review_allowed": False,
        "observability_only": False,
    }


def policy_reason(policy_class: str, *, unknown_bucket: bool = False) -> str:
    if unknown_bucket:
        return "Unknown diagnostic bucket; conservative non-live hard block applied."
    if policy_class == "paper_only_candidate":
        return "Near-miss diagnostic row mapped to paper-only candidate review; non-live by contract."
    if policy_class == "human_review_candidate":
        return "Near-miss diagnostic row mapped to human review; non-live by contract."
    if policy_class == "observe_near_miss_candidate":
        return "Near-miss diagnostic row remains observability-only; non-live by contract."
    if policy_class == "collect_more_data":
        return "Diagnostic row is sample-limited; collect more data before policy escalation."
    return "Diagnostic row is structurally blocked or unsafe for candidate treatment."


def top_policy_rows_by_class(
    policy_rows: Sequence[dict[str, Any]],
    *,
    limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped[str(row.get("policy_class"))].append(dict(row))

    return {
        policy_class: sorted(grouped.get(policy_class, []), key=_policy_sort_key)[
            :limit
        ]
        for policy_class in POLICY_CLASSES
    }


def build_safety_invariants(summary: dict[str, Any]) -> dict[str, Any]:
    policy_rows = [
        row for row in _safe_list(summary.get("policy_rows")) if isinstance(row, dict)
    ]
    live_allowed_rows = [
        row for row in policy_rows if row.get("production_live_selection_allowed") is True
    ]
    diagnostic_live_allowed_rows = [
        row
        for row in live_allowed_rows
        if row.get("source_row_type") == SOURCE_REJECTED_DIAGNOSTIC_NEAR_MISS_ROW
    ]
    live_from_non_production_class = [
        row
        for row in live_allowed_rows
        if row.get("policy_class") != "production_eligible_candidate"
    ]

    return {
        "production_live_selection_source_limited_to_edge_candidate_rows": all(
            row.get("source_row_type") == SOURCE_ELIGIBLE_EDGE_CANDIDATE_ROW
            for row in live_allowed_rows
        ),
        "diagnostic_rows_live_selection_allowed": len(diagnostic_live_allowed_rows) == 0,
        "non_production_policy_classes_live_selection_allowed": (
            len(live_from_non_production_class) == 0
        ),
        "paper_only_rows_non_live": all(
            row.get("production_live_selection_allowed") is False
            for row in policy_rows
            if row.get("policy_class") == "paper_only_candidate"
        ),
        "human_review_rows_non_live": all(
            row.get("production_live_selection_allowed") is False
            for row in policy_rows
            if row.get("policy_class") == "human_review_candidate"
        ),
        "hard_block_rows_non_live": all(
            row.get("production_live_selection_allowed") is False
            for row in policy_rows
            if row.get("policy_class") == "hard_block"
        ),
        "live_allowed_row_count": len(live_allowed_rows),
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    for summary in configuration_summaries:
        class_counts.update(
            {
                str(key): _safe_int(value)
                for key, value in _safe_dict(summary.get("policy_class_counts")).items()
            }
        )

    production_present = class_counts.get("production_eligible_candidate", 0) > 0
    paper_present = class_counts.get("paper_only_candidate", 0) > 0
    human_present = class_counts.get("human_review_candidate", 0) > 0
    collect_present = class_counts.get("collect_more_data", 0) > 0
    observable_present = class_counts.get("observe_near_miss_candidate", 0) > 0

    if paper_present:
        recommended_next_stage = "design_paper_only_replay_contract"
    elif human_present:
        recommended_next_stage = "design_human_review_contract"
    elif collect_present and not production_present:
        recommended_next_stage = "collect_more_data"
    elif production_present and not any(
        [paper_present, human_present, collect_present, observable_present]
    ):
        recommended_next_stage = "no_policy_split_needed"
    else:
        recommended_next_stage = "mixed_or_inconclusive"

    return {
        "production_candidates_present": production_present,
        "paper_only_candidates_present": paper_present,
        "human_review_candidates_present": human_present,
        "policy_split_supported": paper_present or human_present or observable_present,
        "recommended_next_stage": recommended_next_stage,
        "policy_class_counts": {
            policy_class: class_counts.get(policy_class, 0)
            for policy_class in POLICY_CLASSES
        },
        "safety_invariant_summary": {
            "only_production_eligible_candidates_are_live_allowed": all(
                _safe_dict(summary.get("safety_invariants")).get(
                    "production_live_selection_source_limited_to_edge_candidate_rows"
                )
                is True
                and _safe_dict(summary.get("safety_invariants")).get(
                    "non_production_policy_classes_live_selection_allowed"
                )
                is True
                for summary in configuration_summaries
            ),
            "near_miss_derived_buckets_are_non_live": all(
                _safe_dict(summary.get("safety_invariants")).get(
                    "diagnostic_rows_live_selection_allowed"
                )
                is True
                for summary in configuration_summaries
            ),
        },
        "stop_rule_next_requirement": stop_rule_next_requirement(
            recommended_next_stage
        ),
    }


def stop_rule_next_requirement(recommended_next_stage: str) -> str:
    if recommended_next_stage == "design_paper_only_replay_contract":
        return "Stop here. The next stage should design a paper-only replay contract."
    if recommended_next_stage == "design_human_review_contract":
        return "Stop here. The next stage should design a human-review contract."
    if recommended_next_stage == "collect_more_data":
        return "Stop here. More data is required before a policy split can advance."
    return "Stop here. This report does not implement paper replay or live execution."


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
        f"- production_candidates_present: {final.get('production_candidates_present')}",
        f"- paper_only_candidates_present: {final.get('paper_only_candidates_present')}",
        f"- human_review_candidates_present: {final.get('human_review_candidates_present')}",
        f"- policy_split_supported: {final.get('policy_split_supported')}",
        f"- recommended_next_stage: {final.get('recommended_next_stage')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Contract",
        "- Only production_eligible_candidate rows from edge_candidate_rows.rows may be production-live allowed.",
        "- Diagnostic-derived policy rows are non-live report artifacts.",
        "- No mapper, engine, execution gate, candidate quality gate, or threshold behavior is changed.",
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
                f"- eligible_production_candidate_count: {item.get('eligible_production_candidate_count')}",
                f"- paper_only_candidate_count: {item.get('paper_only_candidate_count')}",
                f"- human_review_candidate_count: {item.get('human_review_candidate_count')}",
                f"- observe_near_miss_candidate_count: {item.get('observe_near_miss_candidate_count')}",
                f"- collect_more_data_count: {item.get('collect_more_data_count')}",
                f"- hard_block_count: {item.get('hard_block_count')}",
                f"- policy_class_counts: {item.get('policy_class_counts')}",
                f"- best_paper_only_candidate: {item.get('best_paper_only_candidate')}",
                f"- best_human_review_candidate: {item.get('best_human_review_candidate')}",
                f"- safety_invariants: {item.get('safety_invariants')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _policy_sort_key(row: dict[str, Any]) -> tuple[int, float, str, str, str]:
    class_rank = {
        "production_eligible_candidate": 0,
        "paper_only_candidate": 1,
        "human_review_candidate": 2,
        "observe_near_miss_candidate": 3,
        "collect_more_data": 4,
        "hard_block": 5,
    }.get(str(row.get("policy_class")), 9)
    return (
        class_rank,
        -float(row.get("aggregate_score") or 0.0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
    )


def _candidate_strength(row: dict[str, Any]) -> str | None:
    return (
        _safe_text(row.get("candidate_strength"))
        or _safe_text(row.get("selected_candidate_strength"))
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
    return near_miss_report._safe_dict(value)


def _safe_list(value: Any) -> list[Any]:
    return near_miss_report._safe_list(value)


def _safe_text(value: Any) -> str | None:
    return near_miss_report._safe_text(value)


def _safe_int(value: Any) -> int:
    return near_miss_report._safe_int(value)


def _safe_float(value: Any) -> float | None:
    return near_miss_report._safe_float(value)


if __name__ == "__main__":
    main()
