from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from src.research.edge_selection_input_mapper import map_edge_selection_input
from src.research.future_return_labeling_common import has_future_fields_for_horizon
from src.research.research_analyzer import HORIZONS, run_research_analyzer
from src.research.strategy_lab.dataset_builder import (
    DEFAULT_LATEST_MAX_ROWS,
    DEFAULT_LATEST_WINDOW_HOURS,
    build_dataset,
    load_jsonl_records_with_metadata,
)

REPORT_TYPE = (
    "selected_strategy_edge_candidate_support_window_alignment_diagnosis_report"
)
REPORT_TITLE = "Selected Strategy Edge Candidate Support Window Alignment Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")


@dataclass(frozen=True)
class SupportWindowConfiguration:
    latest_window_hours: int
    latest_max_rows: int
    label: str | None = None

    @property
    def display_name(self) -> str:
        if isinstance(self.label, str) and self.label.strip():
            return self.label.strip()
        return f"{self.latest_window_hours}h/{self.latest_max_rows}"

    @property
    def slug(self) -> str:
        base = f"{self.latest_window_hours}h_{self.latest_max_rows}"
        if self.label:
            label = "".join(
                char.lower() if char.isalnum() else "_"
                for char in self.label.strip()
            ).strip("_")
            if label:
                return f"{label}_{base}"
        return base

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "latest_window_hours": self.latest_window_hours,
            "latest_max_rows": self.latest_max_rows,
            "slug": self.slug,
        }


DEFAULT_CONFIGURATIONS: tuple[SupportWindowConfiguration, ...] = (
    SupportWindowConfiguration(
        DEFAULT_LATEST_WINDOW_HOURS,
        DEFAULT_LATEST_MAX_ROWS,
        "current/latest default",
    ),
    SupportWindowConfiguration(72, 2500),
    SupportWindowConfiguration(144, 5000),
    SupportWindowConfiguration(336, 10000),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only support-window alignment report for selected "
            "strategy edge candidate materialization."
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
    result = run_selected_strategy_edge_candidate_support_window_alignment_diagnosis_report(
        input_path=resolve_path(args.input),
        output_dir=resolve_path(args.output_dir),
        configurations=parse_configuration_values(args.config),
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    final = _safe_dict(report.get("final_assessment"))
    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "classification": final.get("classification"),
                "configuration_count": len(report.get("configuration_summaries", [])),
                "wider_support_produces_edge_candidate_rows": final.get(
                    "wider_support_produces_edge_candidate_rows"
                ),
                "wider_support_produces_mapper_seeds": final.get(
                    "wider_support_produces_mapper_seeds"
                ),
                "written_paths": result["written_paths"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def parse_configuration_values(
    values: Sequence[str] | None,
) -> list[SupportWindowConfiguration]:
    if not values:
        return list(DEFAULT_CONFIGURATIONS)

    parsed: list[SupportWindowConfiguration] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "/" not in item:
            raise ValueError(
                f"Invalid configuration '{value}'. Expected WINDOW_HOURS/MAX_ROWS."
            )
        hours_raw, rows_raw = item.split("/", 1)
        parsed.append(
            SupportWindowConfiguration(
                latest_window_hours=int(hours_raw),
                latest_max_rows=int(rows_raw),
            )
        )

    return parsed or list(DEFAULT_CONFIGURATIONS)


def run_selected_strategy_edge_candidate_support_window_alignment_diagnosis_report(
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
        mapper_report_dir = (
            resolved_output / f"_{REPORT_TYPE}" / "mapper_runs" / configuration.slug
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
        mapper_payload = run_mapper_diagnosis(
            mapper_report_dir=mapper_report_dir,
            analyzer_metrics=analyzer_metrics,
            source_metadata=source_metadata,
            labelable_count=len(labelable_dataset),
        )

        summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                analyzer_output_dir=analyzer_output_dir,
                mapper_report_dir=mapper_report_dir,
                raw_records=raw_records,
                source_metadata=source_metadata,
                labelable_dataset=labelable_dataset,
                analyzer_metrics=analyzer_metrics,
                mapper_payload=mapper_payload,
            )
        )

    final_assessment = build_final_assessment(summaries)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_summaries": summaries,
        "final_assessment": final_assessment,
        "assumptions": [
            "Diagnosis-only: analyzer and mapper gates are reused without threshold or runtime behavior changes.",
            "Mapper runs consume temporary report bundles under the requested output directory.",
            "Support-window comparisons use the same latest-window and max_rows parameters passed to the analyzer.",
        ],
    }


def build_configuration_summary(
    *,
    configuration: SupportWindowConfiguration,
    input_path: Path,
    analyzer_output_dir: Path,
    mapper_report_dir: Path,
    raw_records: Sequence[dict[str, Any]],
    source_metadata: dict[str, Any],
    labelable_dataset: Sequence[dict[str, Any]],
    analyzer_metrics: dict[str, Any],
    mapper_payload: dict[str, Any],
) -> dict[str, Any]:
    edge_candidate_rows = _safe_dict(analyzer_metrics.get("edge_candidate_rows"))
    empty_reason_summary = _safe_dict(edge_candidate_rows.get("empty_reason_summary"))
    diagnostic_rows = _safe_list(edge_candidate_rows.get("diagnostic_rows"))
    mapper_diagnostics = _safe_dict(mapper_payload.get("candidate_seed_diagnostics"))

    rejection_counts = _safe_dict(
        empty_reason_summary.get("diagnostic_rejection_reason_counts")
    )
    dominant_rejection_reason = empty_reason_summary.get("dominant_rejection_reason")

    row_count = _safe_int(edge_candidate_rows.get("row_count"))
    mapper_seed_count = _safe_int(mapper_payload.get("candidate_seed_count"))

    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "analyzer_output_dir": str(analyzer_output_dir),
        "analyzer_summary_path": str(analyzer_output_dir / "summary.json"),
        "mapper_report_dir": str(mapper_report_dir),
        "materialized_effective_input_path": _safe_dict(
            analyzer_metrics.get("schema_validation")
        ).get("effective_input_path"),
        "source_metadata": {
            "input_path": source_metadata.get("input_path", str(input_path)),
            "rotation_aware": bool(source_metadata.get("rotation_aware", False)),
            "source_files": _safe_list(source_metadata.get("source_files")),
            "source_file_count": _safe_int(source_metadata.get("source_file_count")),
            "raw_record_count": _safe_int(
                source_metadata.get("raw_record_count", len(raw_records))
            ),
            "windowed_record_count": _safe_int(
                source_metadata.get("windowed_record_count", len(raw_records))
            ),
            "max_age_hours": source_metadata.get("max_age_hours"),
            "max_rows": source_metadata.get("max_rows"),
        },
        "labelability": {
            "labelable_basic_count": len(labelable_dataset),
            "labelable_with_future_count": sum(
                1
                for row in labelable_dataset
                if any(has_future_fields_for_horizon(row, horizon) for horizon in HORIZONS)
            ),
            "labelable_with_future_by_horizon": {
                horizon: sum(
                    1
                    for row in labelable_dataset
                    if has_future_fields_for_horizon(row, horizon)
                )
                for horizon in HORIZONS
            },
        },
        "edge_candidate_rows": {
            "row_count": row_count,
            "diagnostic_row_count": _safe_int(
                edge_candidate_rows.get("diagnostic_row_count")
            ),
            "empty_reason_summary": empty_reason_summary,
            "rejection_reason_counts": rejection_counts,
            "dominant_rejection_reason": dominant_rejection_reason,
            "top_diagnostic_rows": summarize_diagnostic_rows(diagnostic_rows),
            "top_rows": summarize_candidate_rows(_safe_list(edge_candidate_rows.get("rows"))),
        },
        "candidate_seed_diagnostics": {
            "mapper_ok": bool(mapper_payload.get("ok", False)),
            "candidate_seed_count": mapper_seed_count,
            "candidate_count": len(_safe_list(mapper_payload.get("candidates"))),
            "horizons_with_seed": _safe_list(
                mapper_diagnostics.get("horizons_with_seed")
            ),
            "horizons_without_seed": _safe_list(
                mapper_diagnostics.get("horizons_without_seed")
            ),
            "seed_source": mapper_diagnostics.get("seed_source"),
            "joined_candidate_row_count": _safe_int(
                mapper_diagnostics.get("joined_candidate_row_count")
            ),
            "fallback_blocked": bool(mapper_diagnostics.get("fallback_blocked", False)),
            "fallback_block_reason": mapper_diagnostics.get("fallback_block_reason"),
            "horizon_diagnostics": _safe_list(
                mapper_diagnostics.get("horizon_diagnostics")
            ),
            "errors": _safe_list(mapper_payload.get("errors")),
            "warnings": _safe_list(mapper_payload.get("warnings")),
        },
        "classification": classify_configuration_details(
            row_count=row_count,
            mapper_seed_count=mapper_seed_count,
            dominant_rejection_reason=dominant_rejection_reason,
            rejection_reason_counts=rejection_counts,
            empty_reason_summary=empty_reason_summary,
            diagnostic_rows=diagnostic_rows,
        ),
    }


def run_mapper_diagnosis(
    *,
    mapper_report_dir: Path,
    analyzer_metrics: dict[str, Any],
    source_metadata: dict[str, Any],
    labelable_count: int,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    latest_summary = dict(analyzer_metrics)
    latest_summary["generated_at"] = generated_at

    latest_overview = _safe_dict(latest_summary.get("dataset_overview"))
    latest_overview["total_records"] = labelable_count
    latest_summary["dataset_overview"] = latest_overview

    comparison_summary = {
        "generated_at": generated_at,
        "dataset_overview_comparison": {
            "latest_total_records": _safe_int(
                source_metadata.get("windowed_record_count", labelable_count)
            ),
            "cumulative_total_records": _safe_int(
                source_metadata.get("raw_record_count", labelable_count)
            ),
        },
        "edge_candidates_comparison": {},
    }
    edge_scores_summary = {
        "generated_at": generated_at,
        "edge_stability_scores": {"symbol": [], "strategy": [], "alignment_state": []},
    }
    score_drift_summary = {"generated_at": generated_at, "score_drift": []}

    _write_json(mapper_report_dir / "latest" / "summary.json", latest_summary)
    _write_json(mapper_report_dir / "comparison" / "summary.json", comparison_summary)
    _write_json(mapper_report_dir / "edge_scores" / "summary.json", edge_scores_summary)
    _write_json(mapper_report_dir / "score_drift" / "summary.json", score_drift_summary)

    return map_edge_selection_input(mapper_report_dir)


def classify_configuration(
    *,
    row_count: int,
    mapper_seed_count: int,
    dominant_rejection_reason: Any,
    rejection_reason_counts: dict[str, Any],
    empty_reason_summary: dict[str, Any],
    diagnostic_rows: Sequence[Any],
) -> str:
    if row_count > 0 and mapper_seed_count > 0:
        return "eligible_rows_and_mapper_seeds"
    if row_count > 0:
        return "eligible_rows_without_mapper_seeds"

    reason = _normalize_text(dominant_rejection_reason)
    if reason == "failed_absolute_minimum_gate":
        return "fails_absolute_minimum_gate"
    if reason == "strategy_horizon_incompatible":
        return "fails_strategy_horizon_compatibility"
    if _all_diagnostic_reasons_include(
        diagnostic_rows,
        "median_future_return_non_positive",
    ):
        return "fails_negative_median_return"
    if _safe_bool(empty_reason_summary.get("has_only_incompatibility_rejections")):
        return "fails_strategy_horizon_compatibility"
    if _safe_bool(empty_reason_summary.get("has_only_weak_or_insufficient_candidates")):
        if "failed_absolute_minimum_gate" in rejection_reason_counts:
            return "fails_absolute_minimum_gate"
        return "fails_quality_or_negative_return"
    return "mixed_or_inconclusive"


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "classification": "mixed_or_inconclusive",
            "summary": "No support-window configurations were evaluated.",
            "wider_support_produces_edge_candidate_rows": False,
            "wider_support_produces_mapper_seeds": False,
            "horizons_with_wider_seed": [],
            "minimal_safe_source_alignment_implication": None,
        }

    current = summaries[0]
    wider = summaries[1:]
    current_rows = _safe_int(
        _safe_dict(current.get("edge_candidate_rows")).get("row_count")
    )
    current_seed_count = _safe_int(
        _safe_dict(current.get("candidate_seed_diagnostics")).get(
            "candidate_seed_count"
        )
    )
    wider_with_rows = [
        item
        for item in wider
        if _safe_int(_safe_dict(item.get("edge_candidate_rows")).get("row_count")) > 0
    ]
    wider_with_seeds = [
        item
        for item in wider
        if _safe_int(
            _safe_dict(item.get("candidate_seed_diagnostics")).get(
                "candidate_seed_count"
            )
        )
        > 0
    ]

    horizons_with_seed = sorted(
        {
            str(horizon)
            for item in wider_with_seeds
            for horizon in _safe_list(
                _safe_dict(item.get("candidate_seed_diagnostics")).get(
                    "horizons_with_seed"
                )
            )
        }
    )

    if current_rows == 0 and current_seed_count == 0 and wider_with_rows and wider_with_seeds:
        classification = "latest_support_too_sparse_but_wider_support_recovers"
        summary = (
            "Current/latest support produces no eligible joined edge candidate rows, "
            "while a wider support window produces mapper seeds."
        )
        implication = (
            "Next recovery should design support-source alignment so diagnosis-proven "
            "wider support can feed edge candidate materialization without relaxing gates."
        )
    elif _all_configurations_match(summaries, "fails_absolute_minimum_gate"):
        classification = "all_windows_fail_absolute_minimum_gate"
        summary = (
            "Every evaluated support window produced zero eligible rows and remained "
            "blocked by the absolute minimum sample gate."
        )
        implication = None
    elif _all_configurations_match(
        summaries,
        "fails_strategy_horizon_compatibility",
    ):
        classification = "all_windows_fail_strategy_horizon_compatibility"
        summary = (
            "Every evaluated support window produced zero eligible rows because the "
            "evaluated strategies were incompatible with analyzer horizons."
        )
        implication = None
    elif _all_configurations_match(summaries, "fails_quality_or_negative_return") or (
        summaries
        and all(
            _safe_dict(item.get("classification")).get("configuration_classification")
            in {"fails_quality_or_negative_return", "fails_negative_median_return"}
            for item in summaries
        )
    ):
        classification = "all_windows_fail_quality_or_negative_return"
        summary = (
            "Every evaluated support window produced zero eligible rows after sample "
            "support was available, due to quality or non-positive median-return rejection."
        )
        implication = None
    else:
        classification = "mixed_or_inconclusive"
        summary = "No clean support-source conclusion is available from these windows."
        implication = None

    if classification != "latest_support_too_sparse_but_wider_support_recovers":
        next_requirement = (
            "Gate relaxation or more data would be required before candidate recovery, "
            "but this diagnosis artifact does not implement either."
        )
    else:
        next_requirement = (
            "Stop here and design support-source alignment recovery in a separate stage."
        )

    return {
        "classification": classification,
        "summary": summary,
        "current_configuration": _safe_dict(current.get("configuration")),
        "current_edge_candidate_row_count": current_rows,
        "current_candidate_seed_count": current_seed_count,
        "wider_support_produces_edge_candidate_rows": bool(wider_with_rows),
        "wider_support_produces_mapper_seeds": bool(wider_with_seeds),
        "wider_configurations_with_rows": [
            _safe_dict(item.get("configuration")) for item in wider_with_rows
        ],
        "wider_configurations_with_seeds": [
            _safe_dict(item.get("configuration")) for item in wider_with_seeds
        ],
        "horizons_with_wider_seed": horizons_with_seed,
        "minimal_safe_source_alignment_implication": implication,
        "stop_rule_next_requirement": next_requirement,
    }


def _all_configurations_match(
    summaries: Sequence[dict[str, Any]],
    classification: str,
) -> bool:
    return bool(summaries) and all(
        _safe_dict(item.get("classification")).get("configuration_classification")
        == classification
        for item in summaries
    )


def summarize_diagnostic_rows(rows: Sequence[Any], limit: int = 10) -> list[dict[str, Any]]:
    summarized = [
        _summarize_edge_row(row, diagnostic=True)
        for row in rows
        if isinstance(row, dict)
    ]
    summarized.sort(
        key=lambda item: (
            str(item.get("rejection_reason") or ""),
            str(item.get("horizon") or ""),
            str(item.get("symbol") or ""),
            str(item.get("strategy") or ""),
        )
    )
    return summarized[:limit]


def summarize_candidate_rows(rows: Sequence[Any], limit: int = 10) -> list[dict[str, Any]]:
    summarized = [
        _summarize_edge_row(row, diagnostic=False)
        for row in rows
        if isinstance(row, dict)
    ]
    summarized.sort(
        key=lambda item: (
            str(item.get("horizon") or ""),
            str(item.get("symbol") or ""),
            str(item.get("strategy") or ""),
        )
    )
    return summarized[:limit]


def _summarize_edge_row(row: dict[str, Any], *, diagnostic: bool) -> dict[str, Any]:
    payload = {
        "symbol": row.get("symbol"),
        "strategy": row.get("strategy"),
        "horizon": row.get("horizon"),
        "sample_count": row.get("sample_count"),
        "labeled_count": row.get("labeled_count"),
        "median_future_return_pct": row.get("median_future_return_pct"),
        "positive_rate_pct": row.get("positive_rate_pct"),
        "candidate_strength": row.get("candidate_strength")
        or row.get("selected_candidate_strength"),
        "aggregate_score": row.get("aggregate_score"),
    }
    if diagnostic:
        payload["diagnostic_category"] = row.get("diagnostic_category")
        payload["rejection_reason"] = row.get("rejection_reason")
        payload["rejection_reasons"] = _safe_list(row.get("rejection_reasons"))
    else:
        payload["visibility_reason"] = row.get("visibility_reason")
        payload["selected_visible_horizons"] = _safe_list(
            row.get("selected_visible_horizons")
        )
    return payload


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
        f"- classification: {final.get('classification')}",
        f"- summary: {final.get('summary')}",
        f"- wider_support_produces_edge_candidate_rows: {final.get('wider_support_produces_edge_candidate_rows')}",
        f"- wider_support_produces_mapper_seeds: {final.get('wider_support_produces_mapper_seeds')}",
        f"- horizons_with_wider_seed: {final.get('horizons_with_wider_seed')}",
        f"- stop_rule_next_requirement: {final.get('stop_rule_next_requirement')}",
        "",
        "## Configuration Summaries",
    ]

    for summary in _safe_list(report.get("configuration_summaries")):
        item = _safe_dict(summary)
        configuration = _safe_dict(item.get("configuration"))
        source = _safe_dict(item.get("source_metadata"))
        labelability = _safe_dict(item.get("labelability"))
        edge_rows = _safe_dict(item.get("edge_candidate_rows"))
        seeds = _safe_dict(item.get("candidate_seed_diagnostics"))
        classification = _safe_dict(item.get("classification"))
        lines.extend(
            [
                "",
                f"### {configuration.get('display_name')}",
                f"- raw_record_count: {source.get('raw_record_count')}",
                f"- windowed_record_count: {source.get('windowed_record_count')}",
                f"- labelable_basic_count: {labelability.get('labelable_basic_count')}",
                f"- labelable_with_future_count: {labelability.get('labelable_with_future_count')}",
                f"- edge_candidate_rows.row_count: {edge_rows.get('row_count')}",
                f"- diagnostic_row_count: {edge_rows.get('diagnostic_row_count')}",
                f"- dominant_rejection_reason: {edge_rows.get('dominant_rejection_reason')}",
                f"- rejection_reason_counts: {edge_rows.get('rejection_reason_counts')}",
                f"- mapper_candidate_seed_count: {seeds.get('candidate_seed_count')}",
                f"- horizons_with_seed: {seeds.get('horizons_with_seed')}",
                f"- classification: {classification.get('configuration_classification')}",
                f"- classification_reason: {classification.get('classification_reason')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _all_diagnostic_reasons_include(
    rows: Sequence[Any],
    reason: str,
) -> bool:
    filtered = [row for row in rows if isinstance(row, dict)]
    if not filtered:
        return False
    for row in filtered:
        reasons = _safe_list(row.get("rejection_reasons"))
        if reason not in reasons:
            return False
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _counter_total(counter: dict[str, Any], keys: set[str]) -> int:
    total = 0
    for key, value in counter.items():
        if key in keys:
            total += _safe_int(value)
    return total


def _dominant_explicit_rejection_detail(
    rejection_reason_counts: dict[str, Any],
    diagnostic_rows: Sequence[Any],
) -> str:
    if rejection_reason_counts:
        reason = min(
            ((str(key), _safe_int(value)) for key, value in rejection_reason_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )[0]
        return reason

    expanded = Counter()
    for row in diagnostic_rows:
        if not isinstance(row, dict):
            continue
        for reason in _safe_list(row.get("rejection_reasons")):
            if isinstance(reason, str) and reason.strip():
                expanded[reason.strip()] += 1
    if not expanded:
        return "none"
    return min(expanded.items(), key=lambda item: (-item[1], item[0]))[0]


def _negative_median_rejection_count(diagnostic_rows: Sequence[Any]) -> int:
    return sum(
        1
        for row in diagnostic_rows
        if isinstance(row, dict)
        and "median_future_return_non_positive" in _safe_list(row.get("rejection_reasons"))
    )


def _sample_floor_rejection_count(diagnostic_rows: Sequence[Any]) -> int:
    return sum(
        1
        for row in diagnostic_rows
        if isinstance(row, dict)
        and "sample_count_below_absolute_floor" in _safe_list(row.get("rejection_reasons"))
    )


def _classification_reason(
    *,
    row_count: int,
    mapper_seed_count: int,
    rejection_reason_counts: dict[str, Any],
    dominant_rejection_reason: Any,
    diagnostic_rows: Sequence[Any],
) -> str:
    if row_count > 0 and mapper_seed_count > 0:
        return "Eligible joined edge candidate rows produced mapper seeds."
    if row_count > 0:
        return "Eligible joined edge candidate rows appeared, but mapper did not produce seeds."

    dominant = _normalize_text(dominant_rejection_reason)
    sample_floor = _sample_floor_rejection_count(diagnostic_rows)
    negative_median = _negative_median_rejection_count(diagnostic_rows)
    incompatibility = _counter_total(
        rejection_reason_counts,
        {"strategy_horizon_incompatible"},
    )

    if dominant == "failed_absolute_minimum_gate" and sample_floor:
        return "No eligible rows; dominant blocker is failed_absolute_minimum_gate with sample_count_below_absolute_floor."
    if negative_median:
        return "No eligible rows; diagnostics include median_future_return_non_positive."
    if incompatibility and incompatibility == sum(_safe_int(v) for v in rejection_reason_counts.values()):
        return "No eligible rows; diagnostics are strategy-horizon incompatibility only."
    return (
        "No eligible rows; dominant explicit rejection detail is "
        f"{_dominant_explicit_rejection_detail(rejection_reason_counts, diagnostic_rows)}."
    )


def classify_configuration_details(
    *,
    row_count: int,
    mapper_seed_count: int,
    dominant_rejection_reason: Any,
    rejection_reason_counts: dict[str, Any],
    empty_reason_summary: dict[str, Any],
    diagnostic_rows: Sequence[Any],
) -> dict[str, Any]:
    classification = classify_configuration(
        row_count=row_count,
        mapper_seed_count=mapper_seed_count,
        dominant_rejection_reason=dominant_rejection_reason,
        rejection_reason_counts=rejection_reason_counts,
        empty_reason_summary=empty_reason_summary,
        diagnostic_rows=diagnostic_rows,
    )
    return {
        "configuration_classification": classification,
        "classification_reason": _classification_reason(
            row_count=row_count,
            mapper_seed_count=mapper_seed_count,
            rejection_reason_counts=rejection_reason_counts,
            dominant_rejection_reason=dominant_rejection_reason,
            diagnostic_rows=diagnostic_rows,
        ),
    }
