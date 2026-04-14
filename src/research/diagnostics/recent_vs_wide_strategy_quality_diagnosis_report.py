from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.future_return_labeling_common import has_future_fields_for_horizon
from src.research.research_analyzer import (
    HORIZONS,
    MIN_EDGE_CANDIDATE_SAMPLE_COUNT,
    run_research_analyzer,
)
from src.research.strategy_lab.dataset_builder import (
    build_dataset,
    load_jsonl_records_with_metadata,
    normalize_record,
)

REPORT_TYPE = "recent_vs_wide_strategy_quality_diagnosis_report"
REPORT_TITLE = "Recent vs Wide Strategy Quality Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

DEFAULT_FOCUS_GROUPS: tuple[tuple[str, str], ...] = (
    ("intraday", "15m"),
    ("intraday", "1h"),
    ("swing", "4h"),
)

_CANDIDATE_STRENGTH_ORDER = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "insufficient_data": 0,
    "incompatible": -1,
    None: -1,
}


@dataclass(frozen=True)
class DiagnosisConfiguration:
    latest_window_hours: int
    latest_max_rows: int
    label: str | None = None

    @property
    def display_name(self) -> str:
        if isinstance(self.label, str) and self.label.strip():
            return self.label.strip()
        return f"{self.latest_window_hours}h / {self.latest_max_rows}"

    @property
    def slug(self) -> str:
        return f"{self.latest_window_hours}h_{self.latest_max_rows}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "latest_window_hours": self.latest_window_hours,
            "latest_max_rows": self.latest_max_rows,
            "display_name": self.display_name,
            "slug": self.slug,
        }


DEFAULT_CONFIGURATIONS: tuple[DiagnosisConfiguration, ...] = (
    DiagnosisConfiguration(36, 2500),
    DiagnosisConfiguration(72, 2500),
    DiagnosisConfiguration(144, 2500),
    DiagnosisConfiguration(144, 5000),
    DiagnosisConfiguration(336, 2500),
    DiagnosisConfiguration(336, 10000),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnosis-only recent-vs-wide strategy quality report by "
            "running the existing analyzer across multiple latest-window settings."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Window/max_rows pair in the form WINDOW_HOURS/MAX_ROWS. Repeatable.",
    )
    parser.add_argument(
        "--focus-group",
        action="append",
        default=None,
        help="Strategy/horizon pair in the form STRATEGY:HORIZON. Repeatable.",
    )
    parser.add_argument(
        "--write-latest-copy",
        action="store_true",
        help="Write JSON and Markdown report copies into the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    configurations = parse_configuration_values(args.config)
    focus_groups = parse_focus_group_values(args.focus_group)

    result = run_recent_vs_wide_strategy_quality_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        focus_groups=focus_groups,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    final_assessment = _safe_dict(report.get("final_assessment"))
    widest = _safe_dict(final_assessment.get("widest_configuration"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["inputs"]["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": widest.get("display_name"),
                "widest_selected_survivor_count": len(
                    _safe_list(final_assessment.get("widest_selected_survivors"))
                ),
                "factor_statuses": {
                    key: _safe_dict(value).get("status")
                    for key, value in _safe_dict(final_assessment.get("factors")).items()
                },
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
) -> list[DiagnosisConfiguration]:
    if not values:
        return list(DEFAULT_CONFIGURATIONS)

    parsed: list[DiagnosisConfiguration] = []
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
            DiagnosisConfiguration(
                latest_window_hours=int(hours_raw),
                latest_max_rows=int(rows_raw),
            )
        )

    if not parsed:
        return list(DEFAULT_CONFIGURATIONS)
    return parsed


def parse_focus_group_values(
    values: Sequence[str] | None,
) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_FOCUS_GROUPS)

    parsed: list[tuple[str, str]] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid focus group '{value}'. Expected STRATEGY:HORIZON."
            )
        strategy_raw, horizon_raw = item.split(":", 1)
        strategy = _normalize_strategy(strategy_raw)
        horizon = _normalize_horizon(horizon_raw)
        if strategy is None or horizon is None:
            raise ValueError(f"Invalid focus group '{value}'.")
        parsed.append((strategy, horizon))

    deduped: list[tuple[str, str]] = []
    for item in parsed:
        if item not in deduped:
            deduped.append(item)
    return deduped or list(DEFAULT_FOCUS_GROUPS)


def run_recent_vs_wide_strategy_quality_diagnosis_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
    focus_groups: Sequence[tuple[str, str]] | None = None,
    write_report_copies: bool = False,
) -> dict[str, Any]:
    report = build_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        focus_groups=focus_groups,
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
    configurations: Sequence[DiagnosisConfiguration] | None = None,
    focus_groups: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    resolved_input = resolve_path(input_path)
    resolved_output = resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)
    effective_focus_groups = list(focus_groups or DEFAULT_FOCUS_GROUPS)

    configuration_summaries: list[dict[str, Any]] = []
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
        configuration_summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                analyzer_output_dir=analyzer_output_dir,
                raw_records=raw_records,
                source_metadata=source_metadata,
                labelable_dataset=labelable_dataset,
                analyzer_metrics=analyzer_metrics,
                focus_groups=effective_focus_groups,
            )
        )

    final_assessment = build_final_assessment(configuration_summaries)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": REPORT_TYPE,
        "report_title": REPORT_TITLE,
        "inputs": {
            "input_path": str(resolved_input),
            "output_dir": str(resolved_output),
            "focus_groups": [
                {"strategy": strategy, "horizon": horizon}
                for strategy, horizon in effective_focus_groups
            ],
        },
        "configurations_evaluated": [
            configuration.to_dict() for configuration in effective_configurations
        ],
        "configuration_headlines": [
            _safe_dict(summary.get("headline")) for summary in configuration_summaries
        ],
        "configuration_summaries": configuration_summaries,
        "final_assessment": final_assessment,
        "assumptions": [
            "This report is diagnosis-only and reuses the existing analyzer without modifying thresholds or runtime behavior.",
            "Window and max-row comparisons assume the input path follows the same latest-window semantics as the analyzer path.",
            "Raw/executable/labelable funnel counts are derived from the same windowed input records and build_dataset filtering used by the analyzer stack.",
        ],
    }


def build_configuration_summary(
    *,
    configuration: DiagnosisConfiguration,
    input_path: Path,
    analyzer_output_dir: Path,
    raw_records: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    labelable_dataset: list[dict[str, Any]],
    analyzer_metrics: dict[str, Any],
    focus_groups: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    normalized_raw_rows = [
        normalize_record(record)
        for record in raw_records
        if isinstance(record, dict)
    ]
    edge_candidate_rows = _safe_dict(analyzer_metrics.get("edge_candidate_rows"))
    empty_reason_summary = _safe_dict(edge_candidate_rows.get("empty_reason_summary"))

    selected_rows = extract_selected_rows(edge_candidate_rows)
    diagnostic_rows = extract_diagnostic_rows(edge_candidate_rows)
    focus_group_breakdown = extract_focus_group_breakdown(
        edge_candidate_rows=edge_candidate_rows,
        focus_groups=focus_groups,
    )
    survivor_vs_near_miss = build_survivor_vs_near_miss_breakdown(
        focus_group_breakdown=focus_group_breakdown,
    )
    per_symbol_strategy_counts = build_identity_counts(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        selected_rows=selected_rows,
        diagnostic_rows=diagnostic_rows,
    )
    funnel = build_funnel_summary(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        edge_candidate_rows=edge_candidate_rows,
    )
    raw_strategy_counts = _counter_rows(
        Counter(
            strategy
            for strategy in (
                _normalize_strategy(row.get("selected_strategy"))
                for row in normalized_raw_rows
            )
            if strategy is not None
        )
    )

    date_range = _safe_dict(
        _safe_dict(analyzer_metrics.get("dataset_overview")).get("date_range")
    )

    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "analyzer_output_dir": str(analyzer_output_dir),
        "analyzer_summary_path": str(analyzer_output_dir / "summary.json"),
        "source_metadata": {
            "input_path": source_metadata.get("input_path", str(input_path)),
            "rotation_aware": bool(source_metadata.get("rotation_aware", False)),
            "source_files": _safe_list(source_metadata.get("source_files")),
            "source_file_count": int(source_metadata.get("source_file_count", 0) or 0),
            "raw_record_count": int(
                source_metadata.get("raw_record_count", len(raw_records)) or 0
            ),
            "windowed_record_count": int(
                source_metadata.get("windowed_record_count", len(raw_records)) or 0
            ),
            "max_age_hours": source_metadata.get("max_age_hours"),
            "max_rows": source_metadata.get("max_rows"),
        },
        "headline": {
            "display_name": configuration.display_name,
            "latest_window_hours": configuration.latest_window_hours,
            "latest_max_rows": configuration.latest_max_rows,
            "date_range_start": date_range.get("start"),
            "date_range_end": date_range.get("end"),
            "raw_input_rows": funnel["raw_input_rows"],
            "executable_positive_entry_rows": funnel["executable_positive_entry_rows"],
            "research_labelable_dataset_rows": funnel["research_labelable_dataset_rows"],
            "labeled_rows_by_horizon": funnel["labeled_rows_by_horizon"],
            "edge_candidate_row_count": funnel["edge_candidate_row_count"],
            "diagnostic_row_count": funnel["diagnostic_row_count"],
            "dominant_rejection_reason": funnel["dominant_rejection_reason"],
            "selected_survivor_count": len(selected_rows),
        },
        "funnel": funnel,
        "raw_strategy_counts": raw_strategy_counts,
        "per_symbol_strategy_counts": per_symbol_strategy_counts,
        "edge_candidate_outcomes": {
            "selected_row_count": len(selected_rows),
            "diagnostic_row_count": len(diagnostic_rows),
            "diagnostic_rejection_reason_counts": _safe_dict(
                empty_reason_summary.get("diagnostic_rejection_reason_counts")
            ),
            "diagnostic_category_counts": _safe_dict(
                empty_reason_summary.get("diagnostic_category_counts")
            ),
            "dominant_rejection_reason": empty_reason_summary.get("dominant_rejection_reason"),
            "dominant_diagnostic_category": empty_reason_summary.get(
                "dominant_diagnostic_category"
            ),
            "empty_state_category": empty_reason_summary.get("empty_state_category"),
        },
        "selected_survivors": selected_rows,
        "diagnostic_rows": diagnostic_rows,
        "focus_group_breakdown": focus_group_breakdown,
        "survivor_vs_near_miss": survivor_vs_near_miss,
    }


def build_funnel_summary(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    edge_candidate_rows: dict[str, Any],
) -> dict[str, Any]:
    raw_input_rows = len(normalized_raw_rows)
    execution_allowed_rows = sum(
        1 for row in normalized_raw_rows if row.get("execution_allowed") is True
    )
    positive_entry_rows = sum(
        1 for row in normalized_raw_rows if _has_positive_entry(row.get("entry_price"))
    )
    executable_positive_entry_rows = sum(
        1
        for row in normalized_raw_rows
        if row.get("execution_allowed") is True
        and _has_positive_entry(row.get("entry_price"))
    )

    labeled_rows_by_horizon = {
        horizon: sum(
            1 for row in labelable_dataset if has_future_fields_for_horizon(row, horizon)
        )
        for horizon in HORIZONS
    }
    empty_reason_summary = _safe_dict(edge_candidate_rows.get("empty_reason_summary"))

    return {
        "raw_input_rows": raw_input_rows,
        "raw_rows_with_known_identity": sum(
            1
            for row in normalized_raw_rows
            if _normalize_symbol(row.get("symbol")) is not None
            and _normalize_strategy(row.get("selected_strategy")) is not None
        ),
        "execution_allowed_rows": execution_allowed_rows,
        "positive_entry_rows": positive_entry_rows,
        "executable_positive_entry_rows": executable_positive_entry_rows,
        "research_labelable_dataset_rows": len(labelable_dataset),
        "rows_with_any_future_label": sum(
            1
            for row in labelable_dataset
            if any(has_future_fields_for_horizon(row, horizon) for horizon in HORIZONS)
        ),
        "labeled_rows_by_horizon": labeled_rows_by_horizon,
        "edge_candidate_row_count": int(edge_candidate_rows.get("row_count", 0) or 0),
        "diagnostic_row_count": int(
            edge_candidate_rows.get("diagnostic_row_count", 0) or 0
        ),
        "dominant_rejection_reason": empty_reason_summary.get("dominant_rejection_reason"),
    }


def build_identity_counts(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, Any]] = {}

    def _ensure(symbol: str, strategy: str) -> dict[str, Any]:
        key = (symbol, strategy)
        if key not in counts:
            counts[key] = {
                "symbol": symbol,
                "strategy": strategy,
                "raw_count": 0,
                "execution_allowed_count": 0,
                "positive_entry_count": 0,
                "executable_positive_entry_count": 0,
                "labelable_count": 0,
                "labeled_counts_by_horizon": {horizon: 0 for horizon in HORIZONS},
                "selected_row_count": 0,
                "selected_horizons": [],
                "diagnostic_row_count": 0,
                "_diagnostic_rejection_reason_counts": Counter(),
            }
        return counts[key]

    for row in normalized_raw_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("selected_strategy"))
        if symbol is None or strategy is None:
            continue
        entry = _ensure(symbol, strategy)
        entry["raw_count"] += 1
        if row.get("execution_allowed") is True:
            entry["execution_allowed_count"] += 1
        if _has_positive_entry(row.get("entry_price")):
            entry["positive_entry_count"] += 1
            if row.get("execution_allowed") is True:
                entry["executable_positive_entry_count"] += 1

    for row in labelable_dataset:
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("selected_strategy"))
        if symbol is None or strategy is None:
            continue
        entry = _ensure(symbol, strategy)
        entry["labelable_count"] += 1
        for horizon in HORIZONS:
            if has_future_fields_for_horizon(row, horizon):
                entry["labeled_counts_by_horizon"][horizon] += 1

    for row in selected_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("strategy"))
        horizon = _normalize_horizon(row.get("horizon"))
        if symbol is None or strategy is None:
            continue
        entry = _ensure(symbol, strategy)
        entry["selected_row_count"] += 1
        if horizon is not None and horizon not in entry["selected_horizons"]:
            entry["selected_horizons"].append(horizon)

    for row in diagnostic_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        strategy = _normalize_strategy(row.get("strategy"))
        rejection_reason = _normalize_text(row.get("rejection_reason"))
        if symbol is None or strategy is None:
            continue
        entry = _ensure(symbol, strategy)
        entry["diagnostic_row_count"] += 1
        if rejection_reason is not None:
            entry["_diagnostic_rejection_reason_counts"][rejection_reason] += 1

    rows: list[dict[str, Any]] = []
    for item in counts.values():
        rejection_counts = Counter(item.pop("_diagnostic_rejection_reason_counts"))
        rows.append(
            {
                **item,
                "selected_horizons": sorted(
                    item.get("selected_horizons", []),
                    key=_horizon_sort_key,
                ),
                "diagnostic_rejection_reason_counts": dict(rejection_counts),
                "dominant_diagnostic_rejection_reason": _dominant_counter_key(
                    rejection_counts
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("labelable_count", 0)),
            -int(item.get("positive_entry_count", 0)),
            -int(item.get("raw_count", 0)),
            str(item.get("symbol", "")),
            str(item.get("strategy", "")),
        )
    )
    return rows


def extract_selected_rows(edge_candidate_rows: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _safe_list(edge_candidate_rows.get("rows"))
    extracted = [
        _extract_candidate_row_summary(row, selected=True)
        for row in rows
        if isinstance(row, dict)
    ]
    extracted.sort(key=_candidate_row_sort_key)
    return extracted


def extract_diagnostic_rows(edge_candidate_rows: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _safe_list(edge_candidate_rows.get("diagnostic_rows"))
    extracted = [
        _extract_candidate_row_summary(row, selected=False)
        for row in rows
        if isinstance(row, dict)
    ]
    extracted.sort(key=_candidate_row_sort_key)
    return extracted


def extract_focus_group_breakdown(
    *,
    edge_candidate_rows: dict[str, Any],
    focus_groups: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected_rows = extract_selected_rows(edge_candidate_rows)
    diagnostic_rows = extract_diagnostic_rows(edge_candidate_rows)

    breakdown: list[dict[str, Any]] = []
    for strategy, horizon in focus_groups:
        selected = [
            row
            for row in selected_rows
            if row.get("strategy") == strategy and row.get("horizon") == horizon
        ]
        diagnostics = [
            row
            for row in diagnostic_rows
            if row.get("strategy") == strategy and row.get("horizon") == horizon
        ]
        breakdown.append(
            {
                "strategy": strategy,
                "horizon": horizon,
                "selected_rows": selected,
                "diagnostic_rows": diagnostics,
            }
        )

    return breakdown


def build_survivor_vs_near_miss_breakdown(
    *,
    focus_group_breakdown: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []

    for focus_group in focus_group_breakdown:
        selected_rows = _safe_list(focus_group.get("selected_rows"))
        diagnostic_rows = _safe_list(focus_group.get("diagnostic_rows"))
        if not selected_rows or not diagnostic_rows:
            continue

        survivor = _safe_dict(selected_rows[0])
        near_miss = _safe_dict(diagnostic_rows[0])
        comparisons.append(
            {
                "strategy": focus_group.get("strategy"),
                "horizon": focus_group.get("horizon"),
                "survivor": survivor,
                "near_miss": near_miss,
                "sample_count_gap": _gap(
                    survivor.get("sample_count"), near_miss.get("sample_count")
                ),
                "labeled_count_gap": _gap(
                    survivor.get("labeled_count"), near_miss.get("labeled_count")
                ),
                "median_future_return_pct_gap": _gap(
                    survivor.get("median_future_return_pct"),
                    near_miss.get("median_future_return_pct"),
                ),
                "positive_rate_pct_gap": _gap(
                    survivor.get("positive_rate_pct"),
                    near_miss.get("positive_rate_pct"),
                ),
                "robustness_signal_pct_gap": _gap(
                    survivor.get("robustness_signal_pct"),
                    near_miss.get("robustness_signal_pct"),
                ),
                "aggregate_score_gap": _gap(
                    survivor.get("aggregate_score"), near_miss.get("aggregate_score")
                ),
            }
        )

    return comparisons


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "factors": {},
            "widest_configuration": None,
            "widest_selected_survivors": [],
            "overall_conclusion": "No configurations were evaluated.",
        }

    baseline = min(
        summaries,
        key=lambda item: (
            int(_safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0),
            int(_safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0),
        ),
    )
    widest = max(
        summaries,
        key=lambda item: (
            int(_safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0),
            int(_safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0),
        ),
    )
    widest_same_cap = _select_widest_same_cap_configuration(summaries, baseline)
    max_row_pair = _select_max_row_pair(summaries)

    factors = {
        "recent_window_starvation": _assess_recent_window_starvation(
            baseline=baseline,
            widest_same_cap=widest_same_cap,
        ),
        "latest_max_rows_cap_effects": _assess_latest_max_rows_cap_effects(
            max_row_pair=max_row_pair,
        ),
        "executable_row_scarcity": _assess_executable_row_scarcity(widest),
        "quality_weakness": _assess_quality_weakness(widest),
        "incompatibility_noise": _assess_incompatibility_noise(widest),
    }

    return {
        "assessment": _overall_assessment_label(factors),
        "factors": factors,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "widest_selected_survivors": _safe_list(widest.get("selected_survivors")),
        "overall_conclusion": _build_overall_conclusion(
            factors=factors,
            widest=widest,
        ),
    }


def _assess_recent_window_starvation(
    *,
    baseline: dict[str, Any],
    widest_same_cap: dict[str, Any] | None,
) -> dict[str, Any]:
    if widest_same_cap is None:
        return {
            "status": "not_supported",
            "evidence": ["No same-max-rows window comparison was available."],
        }

    baseline_funnel = _safe_dict(baseline.get("funnel"))
    comparison_funnel = _safe_dict(widest_same_cap.get("funnel"))

    baseline_labelable = int(
        baseline_funnel.get("research_labelable_dataset_rows", 0) or 0
    )
    comparison_labelable = int(
        comparison_funnel.get("research_labelable_dataset_rows", 0) or 0
    )
    baseline_selected = int(baseline_funnel.get("edge_candidate_row_count", 0) or 0)
    comparison_selected = int(
        comparison_funnel.get("edge_candidate_row_count", 0) or 0
    )
    baseline_raw = int(baseline_funnel.get("raw_input_rows", 0) or 0)
    comparison_raw = int(comparison_funnel.get("raw_input_rows", 0) or 0)

    evidence = [
        (
            f"Latest-window raw rows moved from {baseline_raw} to {comparison_raw} "
            f"when hours widened at the same row cap."
        ),
        (
            f"Research-labelable rows moved from {baseline_labelable} to "
            f"{comparison_labelable} under the same comparison."
        ),
        (
            f"Selected edge rows moved from {baseline_selected} to "
            f"{comparison_selected} under the same comparison."
        ),
    ]

    if baseline_labelable == 0 and comparison_labelable > 0:
        status = "primary"
    elif comparison_selected > baseline_selected:
        status = "contributing"
    elif comparison_labelable > baseline_labelable or comparison_raw > baseline_raw:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_latest_max_rows_cap_effects(
    *,
    max_row_pair: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    if max_row_pair is None:
        return {
            "status": "not_supported",
            "evidence": ["No same-window max-rows comparison was available."],
        }

    lower, higher = max_row_pair
    low_funnel = _safe_dict(lower.get("funnel"))
    high_funnel = _safe_dict(higher.get("funnel"))

    low_raw = int(low_funnel.get("raw_input_rows", 0) or 0)
    high_raw = int(high_funnel.get("raw_input_rows", 0) or 0)
    low_labelable = int(low_funnel.get("research_labelable_dataset_rows", 0) or 0)
    high_labelable = int(high_funnel.get("research_labelable_dataset_rows", 0) or 0)
    low_selected = int(low_funnel.get("edge_candidate_row_count", 0) or 0)
    high_selected = int(high_funnel.get("edge_candidate_row_count", 0) or 0)

    evidence = [
        (
            f"At {lower['configuration']['latest_window_hours']}h, raw rows moved from "
            f"{low_raw} to {high_raw} when latest_max_rows increased."
        ),
        (
            f"At the same window, research-labelable rows moved from {low_labelable} "
            f"to {high_labelable}."
        ),
        f"Selected edge rows moved from {low_selected} to {high_selected}.",
    ]

    if high_selected > low_selected:
        status = "contributing"
    elif high_labelable > low_labelable or high_raw > low_raw:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_executable_row_scarcity(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("funnel"))
    raw_rows = int(funnel.get("raw_input_rows", 0) or 0)
    positive_entry_rows = int(funnel.get("positive_entry_rows", 0) or 0)
    executable_positive_entry_rows = int(
        funnel.get("executable_positive_entry_rows", 0) or 0
    )
    labelable_rows = int(funnel.get("research_labelable_dataset_rows", 0) or 0)

    positive_entry_ratio = _safe_ratio(positive_entry_rows, raw_rows)
    executable_positive_entry_ratio = _safe_ratio(executable_positive_entry_rows, raw_rows)
    labelable_ratio = _safe_ratio(labelable_rows, raw_rows)

    evidence = [
        (
            f"Widest configuration kept {raw_rows} raw rows, but only "
            f"{positive_entry_rows} had a positive entry_price."
        ),
        (
            f"Only {executable_positive_entry_rows} rows were both execution_allowed "
            f"and positive-entry."
        ),
        (
            f"The research-labelable dataset retained {labelable_rows} rows "
            f"({labelable_ratio:.2%} of raw rows)."
        ),
    ]

    if labelable_ratio <= 0.15 or executable_positive_entry_ratio <= 0.10:
        status = "primary"
    elif labelable_ratio <= 0.35 or executable_positive_entry_ratio <= 0.20:
        status = "contributing"
    elif positive_entry_ratio < 0.60:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_quality_weakness(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("funnel"))
    edge_candidate_outcomes = _safe_dict(widest.get("edge_candidate_outcomes"))
    focus_groups = _safe_list(widest.get("focus_group_breakdown"))
    quality_relevant_focus_rejections = 0

    for focus_group in focus_groups:
        for row in _safe_list(_safe_dict(focus_group).get("diagnostic_rows")):
            row_dict = _safe_dict(row)
            sample_count = int(row_dict.get("sample_count", 0) or 0)
            labeled_count = int(row_dict.get("labeled_count", 0) or 0)
            rejection_reason = row_dict.get("rejection_reason")
            candidate_strength = row_dict.get("candidate_strength")
            if (
                sample_count >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
                and labeled_count > 0
                and (
                    rejection_reason == "candidate_strength_weak"
                    or candidate_strength == "weak"
                )
            ):
                quality_relevant_focus_rejections += 1

    dominant_rejection_reason = edge_candidate_outcomes.get("dominant_rejection_reason")
    selected_rows = int(funnel.get("edge_candidate_row_count", 0) or 0)
    diagnostic_rows = int(funnel.get("diagnostic_row_count", 0) or 0)

    evidence = [
        (
            f"Widest configuration produced {selected_rows} selected edge rows and "
            f"{diagnostic_rows} diagnostic rows."
        ),
        (
            "Quality-relevant focus-group rejections with adequate sample and label "
            f"support: {quality_relevant_focus_rejections}."
        ),
        f"Dominant rejection reason was {dominant_rejection_reason or 'none'}.",
    ]

    if quality_relevant_focus_rejections > 0 and selected_rows <= 1:
        status = "primary"
    elif dominant_rejection_reason == "candidate_strength_weak":
        status = "contributing"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_incompatibility_noise(widest: dict[str, Any]) -> dict[str, Any]:
    diagnostic_rows = _safe_list(widest.get("diagnostic_rows"))
    incompatible_rows = [
        row
        for row in diagnostic_rows
        if _safe_dict(row).get("rejection_reason") == "strategy_horizon_incompatible"
    ]
    evidence = [
        (
            f"Incompatible strategy/horizon diagnostic rows: {len(incompatible_rows)} "
            f"out of {len(diagnostic_rows)} total diagnostics."
        )
    ]

    if not diagnostic_rows:
        status = "not_supported"
    elif incompatible_rows and len(incompatible_rows) < len(diagnostic_rows):
        status = "noise"
    elif incompatible_rows:
        status = "contributing"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _overall_assessment_label(factors: dict[str, Any]) -> str:
    statuses = {
        name: _safe_dict(payload).get("status") for name, payload in factors.items()
    }
    if statuses.get("quality_weakness") == "primary":
        return "quality_weakness_primary"
    if statuses.get("executable_row_scarcity") == "primary":
        return "executable_row_scarcity_primary"
    if statuses.get("recent_window_starvation") == "primary":
        return "recent_window_starvation_primary"
    if any(status in {"contributing", "limited"} for status in statuses.values()):
        return "mixed_contributing_factors"
    return "not_supported_or_inconclusive"


def _build_overall_conclusion(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    factor_statuses = {
        name: _safe_dict(payload).get("status") for name, payload in factors.items()
    }
    widest_configuration = _safe_dict(widest.get("configuration")).get(
        "display_name", "widest configuration"
    )
    widest_survivors = _safe_list(widest.get("selected_survivors"))
    widest_survivor_labels = [
        f"{row.get('symbol')} / {row.get('strategy')} / {row.get('horizon')}"
        for row in widest_survivors
    ]

    parts: list[str] = []
    if factor_statuses.get("quality_weakness") == "primary":
        parts.append(
            "Wide-window evidence still rejects most compatible rows as weak, so "
            "strategy-quality weakness is the primary bottleneck."
        )
    elif factor_statuses.get("executable_row_scarcity") == "primary":
        parts.append(
            "The main collapse occurs before labeling and edge scoring, so executable "
            "positive-entry scarcity is the primary bottleneck."
        )
    elif factor_statuses.get("recent_window_starvation") == "primary":
        parts.append(
            "Latest-window starvation remains the dominant explanation because wider "
            "same-cap windows materially restore labelable support and survivors."
        )
    else:
        parts.append(
            "The evidence is mixed, but no single upstream starvation effect fully "
            "explains the missing edge candidates."
        )

    if factor_statuses.get("recent_window_starvation") in {"contributing", "limited"}:
        parts.append(
            "Wider windows recover more raw and labelable rows, but that alone does "
            "not broadly restore eligible candidates."
        )
    if factor_statuses.get("latest_max_rows_cap_effects") in {"contributing", "limited"}:
        parts.append(
            "Higher latest_max_rows helps at the margin, which means the cap matters, "
            "but it is not sufficient on its own."
        )
    if factor_statuses.get("incompatibility_noise") == "noise":
        parts.append(
            "Strategy/horizon incompatibility adds diagnostic noise, but it does not "
            "explain the remaining compatible near-miss rows."
        )
    if widest_survivor_labels:
        parts.append(
            f"Under {widest_configuration}, surviving selections were limited to: "
            f"{', '.join(widest_survivor_labels)}."
        )

    return " ".join(parts)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {REPORT_TITLE}", ""]

    lines.append("## Configurations Evaluated")
    lines.append("")
    for configuration in _safe_list(report.get("configurations_evaluated")):
        config = _safe_dict(configuration)
        lines.append(
            f"- {config.get('display_name')}: "
            f"latest_window_hours={config.get('latest_window_hours')}, "
            f"latest_max_rows={config.get('latest_max_rows')}"
        )
    lines.append("")

    lines.append("## Per-Configuration Headline Summary")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        headline = _safe_dict(_safe_dict(summary).get("headline"))
        lines.append(f"### {headline.get('display_name', 'n/a')}")
        lines.append(f"- raw_input_rows: {headline.get('raw_input_rows', 0)}")
        lines.append(
            "- executable_positive_entry_rows: "
            f"{headline.get('executable_positive_entry_rows', 0)}"
        )
        lines.append(
            "- research_labelable_dataset_rows: "
            f"{headline.get('research_labelable_dataset_rows', 0)}"
        )
        lines.append(
            "- labeled_rows_by_horizon: "
            f"{_format_labeled_counts(headline.get('labeled_rows_by_horizon'))}"
        )
        lines.append(
            f"- edge_candidate_row_count: {headline.get('edge_candidate_row_count', 0)}"
        )
        lines.append(
            f"- diagnostic_row_count: {headline.get('diagnostic_row_count', 0)}"
        )
        lines.append(
            "- dominant_rejection_reason: "
            f"{headline.get('dominant_rejection_reason', 'none')}"
        )
        lines.append("")

    lines.append("## Raw-to-Executable-to-Labelable Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(f"- raw_input_rows: {funnel.get('raw_input_rows', 0)}")
        lines.append(
            f"- execution_allowed_rows: {funnel.get('execution_allowed_rows', 0)}"
        )
        lines.append(f"- positive_entry_rows: {funnel.get('positive_entry_rows', 0)}")
        lines.append(
            "- executable_positive_entry_rows: "
            f"{funnel.get('executable_positive_entry_rows', 0)}"
        )
        lines.append(
            "- research_labelable_dataset_rows: "
            f"{funnel.get('research_labelable_dataset_rows', 0)}"
        )
        lines.append(
            "- labeled_rows_by_horizon: "
            f"{_format_labeled_counts(funnel.get('labeled_rows_by_horizon'))}"
        )
        lines.append(
            f"- edge_candidate_row_count: {funnel.get('edge_candidate_row_count', 0)}"
        )
        lines.append(
            f"- diagnostic_row_count: {funnel.get('diagnostic_row_count', 0)}"
        )
        lines.append(
            "- dominant_rejection_reason: "
            f"{funnel.get('dominant_rejection_reason', 'none')}"
        )
        lines.append("")

    lines.append("## Per Symbol/Strategy Counts")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        counts = _safe_list(_safe_dict(summary).get("per_symbol_strategy_counts"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        if not counts:
            lines.append("No symbol/strategy counts available.")
            lines.append("")
            continue
        for row in counts[:15]:
            item = _safe_dict(row)
            labeled_counts = _safe_dict(item.get("labeled_counts_by_horizon"))
            lines.append(
                "- "
                f"{item.get('symbol', 'n/a')} / {item.get('strategy', 'n/a')}: "
                f"raw={item.get('raw_count', 0)}, "
                f"execution_allowed={item.get('execution_allowed_count', 0)}, "
                f"positive_entry={item.get('positive_entry_count', 0)}, "
                f"executable_positive_entry={item.get('executable_positive_entry_count', 0)}, "
                f"labelable={item.get('labelable_count', 0)}, "
                f"labeled(15m/1h/4h)="
                f"{labeled_counts.get('15m', 0)}/{labeled_counts.get('1h', 0)}/{labeled_counts.get('4h', 0)}, "
                f"selected={item.get('selected_row_count', 0)}, "
                f"diagnostic={item.get('diagnostic_row_count', 0)}, "
                f"dominant_rejection_reason={item.get('dominant_diagnostic_rejection_reason', 'none')}"
            )
        lines.append("")

    lines.append("## Selected Survivors")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        survivors = _safe_list(_safe_dict(summary).get("selected_survivors"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        if not survivors:
            lines.append("No selected survivors.")
            lines.append("")
            continue
        for row in survivors:
            lines.append(f"- {_format_candidate_row(_safe_dict(row))}")
        lines.append("")

    lines.append("## Near-Miss Diagnostic Rows")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        for focus_group in _safe_list(_safe_dict(summary).get("focus_group_breakdown")):
            focus = _safe_dict(focus_group)
            lines.append(
                f"- {focus.get('strategy', 'n/a')} / {focus.get('horizon', 'n/a')}"
            )
            diagnostic_rows = _safe_list(focus.get("diagnostic_rows"))
            selected_rows = _safe_list(focus.get("selected_rows"))
            if selected_rows:
                for row in selected_rows[:3]:
                    lines.append(f"  selected: {_format_candidate_row(_safe_dict(row))}")
            if diagnostic_rows:
                for row in diagnostic_rows[:3]:
                    lines.append(f"  diagnostic: {_format_candidate_row(_safe_dict(row))}")
            if not selected_rows and not diagnostic_rows:
                lines.append("  no rows")
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    for factor_name, payload in _safe_dict(final_assessment.get("factors")).items():
        factor = _safe_dict(payload)
        evidence = _safe_list(factor.get("evidence"))
        lines.append(f"- {factor_name}: {factor.get('status', 'unknown')}")
        for item in evidence:
            lines.append(f"  evidence: {item}")
    lines.append(
        f"- overall_conclusion: {final_assessment.get('overall_conclusion', 'n/a')}"
    )
    lines.append("")

    return "\n".join(lines)


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    resolved_output_dir = resolve_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = resolved_output_dir / REPORT_JSON_NAME
    md_path = resolved_output_dir / REPORT_MD_NAME

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def _extract_candidate_row_summary(
    row: dict[str, Any],
    *,
    selected: bool,
) -> dict[str, Any]:
    row_dict = _safe_dict(row)
    horizon_evaluation = _safe_dict(row_dict.get("horizon_evaluation"))
    diagnostics = _safe_dict(
        row_dict.get("candidate_strength_diagnostics")
        or horizon_evaluation.get("candidate_strength_diagnostics")
    )

    aggregate_score = row_dict.get("aggregate_score")
    if aggregate_score is None:
        aggregate_score = diagnostics.get("aggregate_score")

    candidate_strength = row_dict.get("selected_candidate_strength")
    if candidate_strength is None:
        candidate_strength = row_dict.get("candidate_strength")
    if candidate_strength is None:
        candidate_strength = diagnostics.get("final_classification")

    return {
        "symbol": _normalize_symbol(row_dict.get("symbol")),
        "strategy": _normalize_strategy(
            row_dict.get("strategy") or row_dict.get("selected_strategy")
        ),
        "horizon": _normalize_horizon(row_dict.get("horizon")),
        "status": "selected" if selected else row_dict.get("status", "rejected"),
        "sample_count": int(row_dict.get("sample_count", 0) or 0),
        "labeled_count": int(row_dict.get("labeled_count", 0) or 0),
        "median_future_return_pct": _to_float(row_dict.get("median_future_return_pct")),
        "positive_rate_pct": _to_float(row_dict.get("positive_rate_pct")),
        "robustness_signal": _normalize_text(row_dict.get("robustness_signal")),
        "robustness_signal_pct": _to_float(row_dict.get("robustness_signal_pct")),
        "aggregate_score": _to_float(aggregate_score),
        "candidate_strength": _normalize_text(candidate_strength),
        "classification_reason": _normalize_text(
            row_dict.get("classification_reason") or diagnostics.get("classification_reason")
        ),
        "rejection_reason": _normalize_text(row_dict.get("rejection_reason")),
        "visibility_reason": _normalize_text(row_dict.get("visibility_reason")),
        "chosen_metric_summary": _normalize_text(row_dict.get("chosen_metric_summary")),
    }


def _select_widest_same_cap_configuration(
    summaries: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any] | None:
    baseline_config = _safe_dict(baseline.get("configuration"))
    baseline_max_rows = int(baseline_config.get("latest_max_rows", 0) or 0)
    matching = [
        summary
        for summary in summaries
        if int(
            _safe_dict(summary.get("configuration")).get("latest_max_rows", 0) or 0
        )
        == baseline_max_rows
    ]
    if len(matching) < 2:
        return None
    return max(
        matching,
        key=lambda item: int(
            _safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0
        ),
    )


def _select_max_row_pair(
    summaries: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        window_hours = int(
            _safe_dict(summary.get("configuration")).get("latest_window_hours", 0) or 0
        )
        by_window[window_hours].append(summary)

    candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for same_window_summaries in by_window.values():
        if len(same_window_summaries) < 2:
            continue
        ordered = sorted(
            same_window_summaries,
            key=lambda item: int(
                _safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0
            ),
        )
        candidate_pairs.append((ordered[0], ordered[-1]))

    if not candidate_pairs:
        return None

    return max(
        candidate_pairs,
        key=lambda pair: (
            int(
                _safe_dict(pair[1].get("configuration")).get(
                    "latest_window_hours", 0
                )
                or 0
            ),
            int(
                _safe_dict(pair[1].get("configuration")).get("latest_max_rows", 0)
                or 0
            ),
        ),
    )


def _candidate_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_CANDIDATE_STRENGTH_ORDER.get(row.get("candidate_strength"), -1),
        -(_to_float(row.get("aggregate_score")) or -1.0),
        -int(row.get("sample_count", 0) or 0),
        -int(row.get("labeled_count", 0) or 0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("rejection_reason") or ""),
    )


def _format_candidate_row(row: dict[str, Any]) -> str:
    return (
        f"{row.get('symbol', 'n/a')} / {row.get('strategy', 'n/a')} / "
        f"{row.get('horizon', 'n/a')} "
        f"(sample_count={row.get('sample_count', 0)}, "
        f"labeled_count={row.get('labeled_count', 0)}, "
        f"median_future_return_pct={row.get('median_future_return_pct')}, "
        f"positive_rate_pct={row.get('positive_rate_pct')}, "
        f"robustness_signal_pct={row.get('robustness_signal_pct')}, "
        f"aggregate_score={row.get('aggregate_score')}, "
        f"candidate_strength={row.get('candidate_strength')}, "
        f"classification_reason={row.get('classification_reason')}, "
        f"rejection_reason={row.get('rejection_reason')})"
    )


def _format_labeled_counts(value: Any) -> str:
    counts = _safe_dict(value)
    return ", ".join(f"{horizon}={counts.get(horizon, 0)}" for horizon in HORIZONS)


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _dominant_counter_key(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return min(counter.items(), key=lambda item: (-item[1], item[0]))[0]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _gap(left: Any, right: Any) -> float | None:
    left_value = _to_float(left)
    right_value = _to_float(right)
    if left_value is None or right_value is None:
        return None
    return round(left_value - right_value, 6)


def _has_positive_entry(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is not None and numeric > 0


def _horizon_sort_key(value: str) -> int:
    if value in HORIZONS:
        return HORIZONS.index(value)
    return len(HORIZONS)


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


def _normalize_horizon(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in HORIZONS:
        return text
    return None


def _normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in {"up", "down", "flat"}:
        return text
    return None


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


__all__ = [
    "DEFAULT_CONFIGURATIONS",
    "DEFAULT_FOCUS_GROUPS",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DiagnosisConfiguration",
    "REPORT_JSON_NAME",
    "REPORT_MD_NAME",
    "REPORT_TITLE",
    "REPORT_TYPE",
    "build_configuration_summary",
    "build_final_assessment",
    "build_identity_counts",
    "build_report",
    "extract_diagnostic_rows",
    "extract_focus_group_breakdown",
    "extract_selected_rows",
    "main",
    "parse_args",
    "parse_configuration_values",
    "parse_focus_group_values",
    "render_markdown",
    "run_recent_vs_wide_strategy_quality_diagnosis_report",
    "write_report_files",
]
