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

REPORT_TYPE = "executable_entry_scarcity_diagnosis_report"
REPORT_TITLE = "Executable Entry Scarcity Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_HOLD_ACTION_VALUES = {"hold", "neutral", "flat", "no_trade", "no-trade"}
_FAST_STRATEGIES = {"scalping", "intraday"}
_SLOW_SURVIVOR_TARGETS = {("swing", "4h"), ("swing", "1d")}
_CANDIDATE_STRENGTH_ORDER = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "insufficient_data": 0,
    "incompatible": -1,
    None: -1,
}
_PRIMARY_STAGE_ORDER = {
    "positive_entry_scarcity_among_execution_allowed": 0,
    "execution_gate_scarcity": 1,
    "hold_row_dominance": 2,
    "no_upstream_collapse_detected": 3,
}
_UPSTREAM_BOTTLENECK_NAMES = (
    "positive_entry_scarcity_among_execution_allowed",
    "execution_gate_scarcity",
    "hold_row_dominance",
)


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
            "Build a diagnosis-only executable-entry scarcity report across multiple "
            "latest-window configurations using the existing dataset builder and analyzer."
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

    result = run_executable_entry_scarcity_diagnosis_report(
        input_path=input_path,
        output_dir=output_dir,
        configurations=configurations,
        write_report_copies=args.write_latest_copy,
    )
    report = result["report"]
    final_assessment = _safe_dict(report.get("final_assessment"))

    print(
        json.dumps(
            {
                "report_type": REPORT_TYPE,
                "input_path": report["inputs"]["input_path"],
                "configuration_count": len(report.get("configurations_evaluated", [])),
                "widest_configuration": _safe_dict(
                    final_assessment.get("widest_configuration")
                ).get("display_name"),
                "primary_bottleneck": final_assessment.get("primary_bottleneck"),
                "upstream_collapse_stage": final_assessment.get("upstream_collapse_stage"),
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

    return parsed or list(DEFAULT_CONFIGURATIONS)


def run_executable_entry_scarcity_diagnosis_report(
    *,
    input_path: Path,
    output_dir: Path,
    configurations: Sequence[DiagnosisConfiguration] | None = None,
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
    configurations: Sequence[DiagnosisConfiguration] | None = None,
) -> dict[str, Any]:
    resolved_input = resolve_path(input_path)
    resolved_output = resolve_path(output_dir)
    effective_configurations = list(configurations or DEFAULT_CONFIGURATIONS)

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
            "This report is diagnosis-only and reuses the existing analyzer and dataset-builder semantics without changing thresholds or runtime behavior.",
            "Research-labelable rows are counted using build_dataset(), so positive entry_price is required but execution_allowed is not.",
            "Executable positive-entry rows are counted separately to expose upstream scarcity before downstream edge-quality filtering.",
        ],
    }


def build_configuration_summary(
    *,
    configuration: DiagnosisConfiguration,
    input_path: Path,
    analyzer_output_dir: Path,
    raw_records: Sequence[dict[str, Any]],
    source_metadata: dict[str, Any],
    labelable_dataset: Sequence[dict[str, Any]],
    analyzer_metrics: dict[str, Any],
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
    raw_to_execution_funnel = build_raw_to_execution_funnel(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        edge_candidate_rows=edge_candidate_rows,
    )
    execution_to_entry_funnel = build_execution_to_entry_funnel(raw_to_execution_funnel)
    per_symbol_strategy_counts = build_symbol_strategy_execution_counts(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        selected_rows=selected_rows,
        diagnostic_rows=diagnostic_rows,
    )
    strategy_mix = build_strategy_mix_summary(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        selected_rows=selected_rows,
    )
    survivor_concentration_summary = build_survivor_concentration_summary(
        strategy_mix=strategy_mix,
        selected_rows=selected_rows,
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
            "raw_input_rows": raw_to_execution_funnel["raw_input_rows"],
            "raw_rows_with_known_identity": raw_to_execution_funnel[
                "raw_rows_with_known_identity"
            ],
            "hold_action_rows": raw_to_execution_funnel["hold_action_rows"],
            "execution_allowed_rows": raw_to_execution_funnel["execution_allowed_rows"],
            "positive_entry_rows": raw_to_execution_funnel["positive_entry_rows"],
            "executable_positive_entry_rows": raw_to_execution_funnel[
                "executable_positive_entry_rows"
            ],
            "research_labelable_dataset_rows": raw_to_execution_funnel[
                "research_labelable_dataset_rows"
            ],
            "labeled_rows_by_horizon": raw_to_execution_funnel["labeled_rows_by_horizon"],
            "edge_candidate_row_count": raw_to_execution_funnel["edge_candidate_row_count"],
            "diagnostic_row_count": raw_to_execution_funnel["diagnostic_row_count"],
            "primary_collapse_stage": execution_to_entry_funnel["primary_collapse_stage"],
            "dominant_rejection_reason": empty_reason_summary.get("dominant_rejection_reason"),
        },
        "raw_to_execution_funnel": raw_to_execution_funnel,
        "execution_to_entry_funnel": execution_to_entry_funnel,
        "per_symbol_strategy_counts": per_symbol_strategy_counts,
        "strategy_mix": strategy_mix,
        "survivor_concentration_summary": survivor_concentration_summary,
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
    }


def build_raw_to_execution_funnel(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    edge_candidate_rows: dict[str, Any],
) -> dict[str, Any]:
    known_identity_rows = [
        row
        for row in normalized_raw_rows
        if _normalize_symbol(row.get("symbol")) is not None
        and _normalize_strategy(row.get("selected_strategy")) is not None
    ]
    hold_action_rows = [row for row in normalized_raw_rows if _is_hold_action(row)]
    hold_action_known_identity_rows = [
        row for row in known_identity_rows if _is_hold_action(row)
    ]
    execution_allowed_rows = [
        row for row in normalized_raw_rows if row.get("execution_allowed") is True
    ]
    execution_allowed_known_identity_rows = [
        row for row in known_identity_rows if row.get("execution_allowed") is True
    ]
    positive_entry_rows = [
        row for row in normalized_raw_rows if _has_positive_entry(row.get("entry_price"))
    ]
    positive_entry_known_identity_rows = [
        row for row in known_identity_rows if _has_positive_entry(row.get("entry_price"))
    ]
    executable_positive_entry_rows = [
        row
        for row in normalized_raw_rows
        if row.get("execution_allowed") is True and _has_positive_entry(row.get("entry_price"))
    ]
    executable_positive_entry_known_identity_rows = [
        row
        for row in known_identity_rows
        if row.get("execution_allowed") is True and _has_positive_entry(row.get("entry_price"))
    ]
    empty_reason_summary = _safe_dict(edge_candidate_rows.get("empty_reason_summary"))

    labeled_rows_by_horizon = {
        horizon: sum(
            1 for row in labelable_dataset if has_future_fields_for_horizon(row, horizon)
        )
        for horizon in HORIZONS
    }

    return {
        "raw_input_rows": len(normalized_raw_rows),
        "raw_rows_with_known_identity": len(known_identity_rows),
        "raw_rows_without_known_identity": len(normalized_raw_rows) - len(known_identity_rows),
        "hold_action_rows": len(hold_action_rows),
        "non_hold_action_rows": len(normalized_raw_rows) - len(hold_action_rows),
        "hold_action_known_identity_rows": len(hold_action_known_identity_rows),
        "non_hold_action_known_identity_rows": (
            len(known_identity_rows) - len(hold_action_known_identity_rows)
        ),
        "execution_allowed_rows": len(execution_allowed_rows),
        "execution_allowed_known_identity_rows": len(execution_allowed_known_identity_rows),
        "positive_entry_rows": len(positive_entry_rows),
        "positive_entry_known_identity_rows": len(positive_entry_known_identity_rows),
        "executable_positive_entry_rows": len(executable_positive_entry_rows),
        "executable_positive_entry_known_identity_rows": len(
            executable_positive_entry_known_identity_rows
        ),
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


def build_execution_to_entry_funnel(raw_funnel: dict[str, Any]) -> dict[str, Any]:
    raw_known = int(raw_funnel.get("raw_rows_with_known_identity", 0) or 0)
    hold_known = int(raw_funnel.get("hold_action_known_identity_rows", 0) or 0)
    non_hold_known = int(raw_funnel.get("non_hold_action_known_identity_rows", 0) or 0)
    execution_allowed_known = int(
        raw_funnel.get("execution_allowed_known_identity_rows", 0) or 0
    )
    positive_entry_known = int(raw_funnel.get("positive_entry_known_identity_rows", 0) or 0)
    executable_positive_known = int(
        raw_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    labelable_rows = int(raw_funnel.get("research_labelable_dataset_rows", 0) or 0)

    execution_blocked_non_hold = max(non_hold_known - execution_allowed_known, 0)
    execution_allowed_missing_positive_entry = max(
        execution_allowed_known - executable_positive_known,
        0,
    )
    positive_entry_blocked_by_execution = max(
        positive_entry_known - executable_positive_known,
        0,
    )

    stage_losses = {
        "hold_row_dominance": hold_known,
        "execution_gate_scarcity": execution_blocked_non_hold,
        "positive_entry_scarcity_among_execution_allowed": (
            execution_allowed_missing_positive_entry
        ),
    }
    primary_collapse_stage = _primary_stage(stage_losses)

    return {
        "hold_action_share_of_known_identity": _safe_ratio(hold_known, raw_known),
        "execution_allowed_share_of_non_hold_known_identity": _safe_ratio(
            execution_allowed_known,
            non_hold_known,
        ),
        "positive_entry_share_of_known_identity": _safe_ratio(
            positive_entry_known,
            raw_known,
        ),
        "executable_positive_entry_share_of_known_identity": _safe_ratio(
            executable_positive_known,
            raw_known,
        ),
        "executable_positive_entry_share_of_execution_allowed_known_identity": (
            _safe_ratio(executable_positive_known, execution_allowed_known)
        ),
        "labelable_share_of_known_identity": _safe_ratio(labelable_rows, raw_known),
        "execution_blocked_non_hold_known_identity_rows": execution_blocked_non_hold,
        "execution_allowed_missing_positive_entry_known_identity_rows": (
            execution_allowed_missing_positive_entry
        ),
        "positive_entry_blocked_by_execution_known_identity_rows": (
            positive_entry_blocked_by_execution
        ),
        "labelable_without_execution_allowed_rows": max(
            labelable_rows - executable_positive_known,
            0,
        ),
        "primary_collapse_stage": primary_collapse_stage,
        "stage_loss_counts": stage_losses,
    }


def build_symbol_strategy_execution_counts(
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
                "hold_action_count": 0,
                "non_hold_action_count": 0,
                "execution_allowed_count": 0,
                "positive_entry_count": 0,
                "executable_positive_entry_count": 0,
                "labelable_count": 0,
                "labeled_counts_by_horizon": {horizon: 0 for horizon in HORIZONS},
                "surviving_edge_row_count": 0,
                "surviving_edge_counts_by_horizon": {horizon: 0 for horizon in HORIZONS},
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
        if _is_hold_action(row):
            entry["hold_action_count"] += 1
        else:
            entry["non_hold_action_count"] += 1
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
        entry["surviving_edge_row_count"] += 1
        if horizon is not None:
            entry["surviving_edge_counts_by_horizon"][horizon] += 1

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
        execution_allowed_count = int(item.get("execution_allowed_count", 0) or 0)
        executable_positive_entry_count = int(
            item.get("executable_positive_entry_count", 0) or 0
        )
        rows.append(
            {
                **item,
                "execution_allowed_missing_positive_entry_count": max(
                    execution_allowed_count - executable_positive_entry_count,
                    0,
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
            -int(item.get("executable_positive_entry_count", 0)),
            -int(item.get("raw_count", 0)),
            str(item.get("symbol", "")),
            str(item.get("strategy", "")),
        )
    )
    return rows


def build_strategy_mix_summary(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_counter: Counter[str] = Counter()
    hold_counter: Counter[str] = Counter()
    execution_allowed_counter: Counter[str] = Counter()
    positive_entry_counter: Counter[str] = Counter()
    executable_positive_entry_counter: Counter[str] = Counter()
    labelable_counter: Counter[str] = Counter()
    labeled_strategy_horizon_counter: Counter[tuple[str, str]] = Counter()
    survivor_strategy_counter: Counter[str] = Counter()
    survivor_strategy_horizon_counter: Counter[tuple[str, str]] = Counter()

    for row in normalized_raw_rows:
        strategy = _normalize_strategy(row.get("selected_strategy"))
        if strategy is None:
            continue
        raw_counter[strategy] += 1
        if _is_hold_action(row):
            hold_counter[strategy] += 1
        if row.get("execution_allowed") is True:
            execution_allowed_counter[strategy] += 1
        if _has_positive_entry(row.get("entry_price")):
            positive_entry_counter[strategy] += 1
            if row.get("execution_allowed") is True:
                executable_positive_entry_counter[strategy] += 1

    for row in labelable_dataset:
        strategy = _normalize_strategy(row.get("selected_strategy"))
        if strategy is None:
            continue
        labelable_counter[strategy] += 1
        for horizon in HORIZONS:
            if has_future_fields_for_horizon(row, horizon):
                labeled_strategy_horizon_counter[(strategy, horizon)] += 1

    for row in selected_rows:
        strategy = _normalize_strategy(row.get("strategy"))
        horizon = _normalize_horizon(row.get("horizon"))
        if strategy is None:
            continue
        survivor_strategy_counter[strategy] += 1
        if horizon is not None:
            survivor_strategy_horizon_counter[(strategy, horizon)] += 1

    all_strategies = sorted(
        set(raw_counter)
        | set(execution_allowed_counter)
        | set(executable_positive_entry_counter)
        | set(labelable_counter)
        | set(survivor_strategy_counter)
    )
    strategy_share_shift_rows: list[dict[str, Any]] = []
    for strategy in all_strategies:
        raw_count = raw_counter[strategy]
        executable_positive_entry_count = executable_positive_entry_counter[strategy]
        labelable_count = labelable_counter[strategy]
        surviving_edge_count = survivor_strategy_counter[strategy]

        raw_share = _safe_ratio(raw_count, sum(raw_counter.values()))
        executable_positive_share = _safe_ratio(
            executable_positive_entry_count,
            sum(executable_positive_entry_counter.values()),
        )
        labelable_share = _safe_ratio(labelable_count, sum(labelable_counter.values()))
        surviving_edge_share = _safe_ratio(
            surviving_edge_count,
            sum(survivor_strategy_counter.values()),
        )
        strategy_share_shift_rows.append(
            {
                "strategy": strategy,
                "raw_count": raw_count,
                "raw_share": raw_share,
                "executable_positive_entry_count": executable_positive_entry_count,
                "executable_positive_entry_share": executable_positive_share,
                "labelable_count": labelable_count,
                "labelable_share": labelable_share,
                "surviving_edge_count": surviving_edge_count,
                "surviving_edge_share": surviving_edge_share,
                "survivor_minus_raw_share_delta": round(
                    surviving_edge_share - raw_share,
                    6,
                ),
            }
        )

    strategy_share_shift_rows.sort(
        key=lambda item: (
            -(_to_float(item.get("surviving_edge_share"), default=0.0) or 0.0),
            -(_to_float(item.get("survivor_minus_raw_share_delta"), default=0.0) or 0.0),
            str(item.get("strategy", "")),
        )
    )

    return {
        "raw_strategy_counts": _counter_rows(raw_counter),
        "hold_action_counts_by_strategy": _counter_rows(hold_counter),
        "execution_allowed_counts_by_strategy": _counter_rows(execution_allowed_counter),
        "positive_entry_counts_by_strategy": _counter_rows(positive_entry_counter),
        "executable_positive_entry_counts_by_strategy": _counter_rows(
            executable_positive_entry_counter
        ),
        "labelable_counts_by_strategy": _counter_rows(labelable_counter),
        "labeled_counts_by_strategy_horizon": _strategy_horizon_counter_rows(
            labeled_strategy_horizon_counter
        ),
        "surviving_edge_rows_by_strategy": _counter_rows(survivor_strategy_counter),
        "surviving_edge_rows_by_strategy_horizon": _strategy_horizon_counter_rows(
            survivor_strategy_horizon_counter
        ),
        "strategy_share_shift": strategy_share_shift_rows,
    }


def build_survivor_concentration_summary(
    *,
    strategy_mix: dict[str, Any],
    selected_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    survivor_strategy_rows = _safe_list(strategy_mix.get("surviving_edge_rows_by_strategy"))
    raw_strategy_rows = _safe_list(strategy_mix.get("raw_strategy_counts"))
    survivor_strategy_horizon_rows = _safe_list(
        strategy_mix.get("surviving_edge_rows_by_strategy_horizon")
    )

    raw_total = sum(int(_safe_dict(row).get("count", 0) or 0) for row in raw_strategy_rows)
    survivor_total = len(selected_rows)

    raw_fast_count = sum(
        int(_safe_dict(row).get("count", 0) or 0)
        for row in raw_strategy_rows
        if _safe_dict(row).get("value") in _FAST_STRATEGIES
    )
    survivor_fast_count = sum(
        int(_safe_dict(row).get("count", 0) or 0)
        for row in survivor_strategy_rows
        if _safe_dict(row).get("value") in _FAST_STRATEGIES
    )
    slow_swing_survivor_count = sum(
        1
        for row in selected_rows
        if (
            _normalize_strategy(_safe_dict(row).get("strategy")),
            _normalize_horizon(_safe_dict(row).get("horizon")),
        )
        in _SLOW_SURVIVOR_TARGETS
    )

    dominant_survivor_group = None
    if survivor_strategy_horizon_rows:
        top_row = _safe_dict(survivor_strategy_horizon_rows[0])
        dominant_survivor_group = {
            "strategy": top_row.get("strategy"),
            "horizon": top_row.get("horizon"),
            "count": int(top_row.get("count", 0) or 0),
            "share": _to_float(top_row.get("share"), default=0.0),
        }

    return {
        "total_surviving_edge_rows": survivor_total,
        "survivors_by_strategy": survivor_strategy_rows,
        "survivors_by_strategy_horizon": survivor_strategy_horizon_rows,
        "dominant_survivor_group": dominant_survivor_group,
        "slow_swing_survivor_count": slow_swing_survivor_count,
        "slow_swing_survivor_share": _safe_ratio(slow_swing_survivor_count, survivor_total),
        "raw_intraday_scalping_share": _safe_ratio(raw_fast_count, raw_total),
        "survivor_intraday_scalping_share": _safe_ratio(
            survivor_fast_count,
            survivor_total,
        ),
    }


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


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "primary_bottleneck": "none",
            "upstream_collapse_stage": "no_upstream_collapse_detected",
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
        "hold_row_dominance": _assess_hold_row_dominance(widest),
        "execution_gate_scarcity": _assess_execution_gate_scarcity(widest),
        "positive_entry_scarcity_among_execution_allowed": (
            _assess_positive_entry_scarcity(widest)
        ),
        "recent_window_effect": _assess_recent_window_effect(
            baseline=baseline,
            widest_same_cap=widest_same_cap,
        ),
        "latest_max_rows_cap_effect": _assess_latest_max_rows_cap_effect(
            max_row_pair=max_row_pair
        ),
        "downstream_quality_weakness": _assess_downstream_quality_weakness(widest),
        "slow_swing_survivor_concentration": _assess_slow_swing_survivor_concentration(
            widest
        ),
    }

    primary_bottleneck = _primary_bottleneck_label(factors=factors, widest=widest)

    return {
        "assessment": _overall_assessment_label(
            primary_bottleneck=primary_bottleneck,
            factors=factors,
            widest=widest,
        ),
        "primary_bottleneck": primary_bottleneck,
        "upstream_collapse_stage": _safe_dict(
            widest.get("execution_to_entry_funnel")
        ).get("primary_collapse_stage", "no_upstream_collapse_detected"),
        "factors": factors,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "widest_selected_survivors": _safe_list(widest.get("selected_survivors")),
        "overall_conclusion": _build_overall_conclusion(
            factors=factors,
            widest=widest,
            primary_bottleneck=primary_bottleneck,
        ),
    }


def _assess_hold_row_dominance(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
    execution_funnel = _safe_dict(widest.get("execution_to_entry_funnel"))

    raw_known = int(funnel.get("raw_rows_with_known_identity", 0) or 0)
    hold_known = int(funnel.get("hold_action_known_identity_rows", 0) or 0)
    hold_share = _safe_ratio(hold_known, raw_known)

    evidence = [
        (
            f"Known-identity raw rows: {raw_known}; hold/no-trade known-identity rows: "
            f"{hold_known} ({hold_share:.2%})."
        ),
        (
            "Primary collapse stage at widest configuration: "
            f"{execution_funnel.get('primary_collapse_stage', 'unknown')}."
        ),
    ]

    if raw_known <= 0:
        status = "not_supported"
    elif hold_share >= 0.75:
        status = "primary"
    elif hold_share >= 0.50:
        status = "contributing"
    elif hold_share >= 0.35:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_execution_gate_scarcity(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
    execution_funnel = _safe_dict(widest.get("execution_to_entry_funnel"))

    non_hold_known = int(funnel.get("non_hold_action_known_identity_rows", 0) or 0)
    execution_allowed_known = int(
        funnel.get("execution_allowed_known_identity_rows", 0) or 0
    )
    blocked = int(
        execution_funnel.get("execution_blocked_non_hold_known_identity_rows", 0) or 0
    )
    allowed_share = _safe_ratio(execution_allowed_known, non_hold_known)

    evidence = [
        (
            f"Known-identity non-hold rows: {non_hold_known}; execution_allowed known-identity "
            f"rows: {execution_allowed_known} ({allowed_share:.2%})."
        ),
        f"Non-hold rows blocked before execution eligibility: {blocked}.",
    ]

    if non_hold_known <= 0:
        status = "not_supported"
    elif allowed_share <= 0.30:
        status = "primary"
    elif allowed_share <= 0.55:
        status = "contributing"
    elif allowed_share <= 0.75:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_positive_entry_scarcity(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
    execution_funnel = _safe_dict(widest.get("execution_to_entry_funnel"))

    execution_allowed_known = int(
        funnel.get("execution_allowed_known_identity_rows", 0) or 0
    )
    executable_positive_known = int(
        funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    missing_positive = int(
        execution_funnel.get(
            "execution_allowed_missing_positive_entry_known_identity_rows",
            0,
        )
        or 0
    )
    executable_share = _safe_ratio(executable_positive_known, execution_allowed_known)

    evidence = [
        (
            "Execution-allowed known-identity rows: "
            f"{execution_allowed_known}; executable positive-entry known-identity rows: "
            f"{executable_positive_known} ({executable_share:.2%})."
        ),
        (
            "Execution-allowed known-identity rows missing a positive entry_price: "
            f"{missing_positive}."
        ),
    ]

    if execution_allowed_known <= 0:
        status = "not_supported"
    elif executable_share <= 0.25:
        status = "primary"
    elif executable_share <= 0.45:
        status = "contributing"
    elif executable_share <= 0.65:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_recent_window_effect(
    *,
    baseline: dict[str, Any],
    widest_same_cap: dict[str, Any] | None,
) -> dict[str, Any]:
    if widest_same_cap is None:
        return {
            "status": "not_supported",
            "evidence": ["No same-max-rows recent-vs-wide comparison was available."],
        }

    baseline_funnel = _safe_dict(baseline.get("raw_to_execution_funnel"))
    comparison_funnel = _safe_dict(widest_same_cap.get("raw_to_execution_funnel"))

    baseline_exec = int(
        baseline_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    comparison_exec = int(
        comparison_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
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

    evidence = [
        (
            f"Same-cap executable positive-entry known-identity rows moved from {baseline_exec} "
            f"to {comparison_exec} when widening the latest window."
        ),
        (
            f"Research-labelable dataset rows moved from {baseline_labelable} to "
            f"{comparison_labelable} under the same comparison."
        ),
        f"Selected edge rows moved from {baseline_selected} to {comparison_selected}.",
    ]

    if baseline_exec == 0 and comparison_exec > 0 and comparison_selected > baseline_selected:
        status = "primary"
    elif (
        comparison_exec > baseline_exec
        or comparison_labelable > baseline_labelable
        or comparison_selected > baseline_selected
    ):
        status = "contributing"
    elif comparison_funnel.get("raw_input_rows", 0) != baseline_funnel.get("raw_input_rows", 0):
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_latest_max_rows_cap_effect(
    *,
    max_row_pair: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    if max_row_pair is None:
        return {
            "status": "not_supported",
            "evidence": ["No same-window latest_max_rows comparison was available."],
        }

    lower, higher = max_row_pair
    low_funnel = _safe_dict(lower.get("raw_to_execution_funnel"))
    high_funnel = _safe_dict(higher.get("raw_to_execution_funnel"))

    low_exec = int(
        low_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    high_exec = int(
        high_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    low_labelable = int(low_funnel.get("research_labelable_dataset_rows", 0) or 0)
    high_labelable = int(high_funnel.get("research_labelable_dataset_rows", 0) or 0)
    low_selected = int(low_funnel.get("edge_candidate_row_count", 0) or 0)
    high_selected = int(high_funnel.get("edge_candidate_row_count", 0) or 0)

    evidence = [
        (
            f"At {lower['configuration']['latest_window_hours']}h, executable positive-entry "
            f"known-identity rows moved from {low_exec} to {high_exec} when latest_max_rows "
            "increased."
        ),
        (
            f"At the same window, research-labelable rows moved from {low_labelable} to "
            f"{high_labelable}."
        ),
        f"Selected edge rows moved from {low_selected} to {high_selected}.",
    ]

    if high_selected > low_selected and high_exec > low_exec:
        status = "contributing"
    elif high_exec > low_exec or high_labelable > low_labelable:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_downstream_quality_weakness(widest: dict[str, Any]) -> dict[str, Any]:
    raw_funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
    edge_candidate_outcomes = _safe_dict(widest.get("edge_candidate_outcomes"))
    diagnostic_rows = _safe_list(widest.get("diagnostic_rows"))

    quality_relevant_rejections = 0
    for row in diagnostic_rows:
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
            quality_relevant_rejections += 1

    executable_positive_known = int(
        raw_funnel.get("executable_positive_entry_known_identity_rows", 0) or 0
    )
    labelable_rows = int(raw_funnel.get("research_labelable_dataset_rows", 0) or 0)
    selected_rows = int(raw_funnel.get("edge_candidate_row_count", 0) or 0)
    dominant_rejection_reason = edge_candidate_outcomes.get("dominant_rejection_reason")

    evidence = [
        (
            f"Widest configuration had {executable_positive_known} executable positive-entry "
            f"known-identity rows and {labelable_rows} research-labelable rows."
        ),
        (
            "Quality-relevant rejected diagnostic rows with sufficient sample support: "
            f"{quality_relevant_rejections}."
        ),
        (
            f"Selected edge rows: {selected_rows}; dominant rejection reason: "
            f"{dominant_rejection_reason or 'none'}."
        ),
    ]

    if (
        quality_relevant_rejections > 0
        and labelable_rows >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
        and executable_positive_known >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
        and selected_rows <= 1
    ):
        status = "primary"
    elif quality_relevant_rejections > 0 or dominant_rejection_reason == "candidate_strength_weak":
        status = "contributing"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_slow_swing_survivor_concentration(widest: dict[str, Any]) -> dict[str, Any]:
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    total_survivors = int(concentration.get("total_surviving_edge_rows", 0) or 0)
    slow_share = _to_float(concentration.get("slow_swing_survivor_share"), default=0.0)
    raw_fast_share = _to_float(concentration.get("raw_intraday_scalping_share"), default=0.0)
    survivor_fast_share = _to_float(
        concentration.get("survivor_intraday_scalping_share"),
        default=0.0,
    )
    dominant_group = _safe_dict(concentration.get("dominant_survivor_group"))

    evidence = [
        f"Total surviving edge rows at widest configuration: {total_survivors}.",
        (
            f"Slow swing survivor share: {slow_share:.2%}; raw intraday/scalping share: "
            f"{raw_fast_share:.2%}; survivor intraday/scalping share: {survivor_fast_share:.2%}."
        ),
        (
            "Dominant survivor group: "
            f"{dominant_group.get('strategy', 'n/a')} / {dominant_group.get('horizon', 'n/a')}."
        ),
    ]

    if total_survivors <= 0:
        status = "not_supported"
    elif slow_share >= 0.75 and raw_fast_share > survivor_fast_share:
        status = "concentrated"
    elif slow_share > 0.50 or raw_fast_share > survivor_fast_share:
        status = "present"
    else:
        status = "mixed"

    return {"status": status, "evidence": evidence}


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
            "- raw_rows_with_known_identity: "
            f"{headline.get('raw_rows_with_known_identity', 0)}"
        )
        lines.append(f"- hold_action_rows: {headline.get('hold_action_rows', 0)}")
        lines.append(
            f"- execution_allowed_rows: {headline.get('execution_allowed_rows', 0)}"
        )
        lines.append(f"- positive_entry_rows: {headline.get('positive_entry_rows', 0)}")
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
            "- primary_collapse_stage: "
            f"{headline.get('primary_collapse_stage', 'n/a')}"
        )
        lines.append(
            "- dominant_rejection_reason: "
            f"{headline.get('dominant_rejection_reason', 'none')}"
        )
        lines.append("")

    lines.append("## Raw-to-Execution Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("raw_to_execution_funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(f"- raw_input_rows: {funnel.get('raw_input_rows', 0)}")
        lines.append(
            f"- raw_rows_with_known_identity: {funnel.get('raw_rows_with_known_identity', 0)}"
        )
        lines.append(f"- hold_action_rows: {funnel.get('hold_action_rows', 0)}")
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
        lines.append("")

    lines.append("## Execution-to-Entry Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("execution_to_entry_funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- hold_action_share_of_known_identity: "
            f"{_format_ratio(funnel.get('hold_action_share_of_known_identity'))}"
        )
        lines.append(
            "- execution_allowed_share_of_non_hold_known_identity: "
            f"{_format_ratio(funnel.get('execution_allowed_share_of_non_hold_known_identity'))}"
        )
        lines.append(
            "- executable_positive_entry_share_of_execution_allowed_known_identity: "
            f"{_format_ratio(funnel.get('executable_positive_entry_share_of_execution_allowed_known_identity'))}"
        )
        lines.append(
            "- execution_blocked_non_hold_known_identity_rows: "
            f"{funnel.get('execution_blocked_non_hold_known_identity_rows', 0)}"
        )
        lines.append(
            "- execution_allowed_missing_positive_entry_known_identity_rows: "
            f"{funnel.get('execution_allowed_missing_positive_entry_known_identity_rows', 0)}"
        )
        lines.append(
            "- positive_entry_blocked_by_execution_known_identity_rows: "
            f"{funnel.get('positive_entry_blocked_by_execution_known_identity_rows', 0)}"
        )
        lines.append(
            "- primary_collapse_stage: "
            f"{funnel.get('primary_collapse_stage', 'n/a')}"
        )
        lines.append("")

    lines.append("## Per Symbol/Strategy Execution Scarcity Counts")
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
            surviving_counts = _safe_dict(item.get("surviving_edge_counts_by_horizon"))
            lines.append(
                "- "
                f"{item.get('symbol', 'n/a')} / {item.get('strategy', 'n/a')}: "
                f"raw={item.get('raw_count', 0)}, "
                f"hold={item.get('hold_action_count', 0)}, "
                f"non_hold={item.get('non_hold_action_count', 0)}, "
                f"execution_allowed={item.get('execution_allowed_count', 0)}, "
                f"positive_entry={item.get('positive_entry_count', 0)}, "
                f"executable_positive_entry={item.get('executable_positive_entry_count', 0)}, "
                f"labelable={item.get('labelable_count', 0)}, "
                f"labeled(15m/1h/4h)="
                f"{labeled_counts.get('15m', 0)}/{labeled_counts.get('1h', 0)}/{labeled_counts.get('4h', 0)}, "
                f"survivors(15m/1h/4h)="
                f"{surviving_counts.get('15m', 0)}/{surviving_counts.get('1h', 0)}/{surviving_counts.get('4h', 0)}, "
                f"diagnostic={item.get('diagnostic_row_count', 0)}"
            )
        lines.append("")

    lines.append("## Strategy Mix vs Surviving Edge Mix")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        strategy_mix = _safe_dict(_safe_dict(summary).get("strategy_mix"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- raw_strategy_counts: "
            f"{_format_counter_rows(strategy_mix.get('raw_strategy_counts'))}"
        )
        lines.append(
            "- execution_allowed_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('execution_allowed_counts_by_strategy'))}"
        )
        lines.append(
            "- executable_positive_entry_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('executable_positive_entry_counts_by_strategy'))}"
        )
        lines.append(
            "- labelable_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('labelable_counts_by_strategy'))}"
        )
        lines.append(
            "- surviving_edge_rows_by_strategy_horizon: "
            f"{_format_strategy_horizon_rows(strategy_mix.get('surviving_edge_rows_by_strategy_horizon'))}"
        )
        lines.append(
            "- strategy_share_shift: "
            f"{_format_strategy_share_shift(strategy_mix.get('strategy_share_shift'))}"
        )
        lines.append("")

    lines.append("## Survivor Concentration Summary")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        concentration = _safe_dict(
            _safe_dict(summary).get("survivor_concentration_summary")
        )
        dominant = _safe_dict(concentration.get("dominant_survivor_group"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- total_surviving_edge_rows: "
            f"{concentration.get('total_surviving_edge_rows', 0)}"
        )
        lines.append(
            "- slow_swing_survivor_share: "
            f"{_format_ratio(concentration.get('slow_swing_survivor_share'))}"
        )
        lines.append(
            "- raw_intraday_scalping_share: "
            f"{_format_ratio(concentration.get('raw_intraday_scalping_share'))}"
        )
        lines.append(
            "- survivor_intraday_scalping_share: "
            f"{_format_ratio(concentration.get('survivor_intraday_scalping_share'))}"
        )
        lines.append(
            "- dominant_survivor_group: "
            f"{dominant.get('strategy', 'n/a')} / {dominant.get('horizon', 'n/a')}"
        )
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append(
        f"- primary_bottleneck: {final_assessment.get('primary_bottleneck', 'n/a')}"
    )
    lines.append(
        f"- upstream_collapse_stage: {final_assessment.get('upstream_collapse_stage', 'n/a')}"
    )
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
            row_dict.get("classification_reason")
            or diagnostics.get("classification_reason")
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
                    "latest_window_hours",
                    0,
                )
                or 0
            ),
            int(
                _safe_dict(pair[1].get("configuration")).get("latest_max_rows", 0)
                or 0
            ),
        ),
    )


def _primary_bottleneck_label(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    for name in _UPSTREAM_BOTTLENECK_NAMES:
        if _safe_dict(factors.get(name)).get("status") == "primary":
            return name

    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "primary":
        return "downstream_quality_weakness"

    widest_execution = _safe_dict(widest.get("execution_to_entry_funnel"))
    stage = widest_execution.get("primary_collapse_stage")
    executable_share = _to_float(
        widest_execution.get("executable_positive_entry_share_of_known_identity"),
        default=0.0,
    )

    if stage in _UPSTREAM_BOTTLENECK_NAMES:
        stage_status = _safe_dict(factors.get(stage)).get("status")
        if stage_status in {"contributing", "limited"} and (executable_share or 0.0) <= 0.25:
            return stage

    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "contributing":
        return "downstream_quality_weakness"

    return "mixed_or_inconclusive"


def _overall_assessment_label(
    *,
    primary_bottleneck: str,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    if primary_bottleneck in _UPSTREAM_BOTTLENECK_NAMES:
        return f"{primary_bottleneck}_primary"
    if primary_bottleneck == "downstream_quality_weakness":
        return "downstream_quality_weakness_primary"

    statuses = {
        name: _safe_dict(payload).get("status") for name, payload in factors.items()
    }
    if any(
        status in {"contributing", "limited", "present", "concentrated"}
        for status in statuses.values()
    ):
        widest_funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
        if int(widest_funnel.get("edge_candidate_row_count", 0) or 0) <= 1:
            return "mixed_contributing_factors"
    return "not_supported_or_inconclusive"


def _build_overall_conclusion(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
    primary_bottleneck: str,
) -> str:
    widest_config = _safe_dict(widest.get("configuration")).get(
        "display_name",
        "widest configuration",
    )
    widest_funnel = _safe_dict(widest.get("raw_to_execution_funnel"))
    widest_execution = _safe_dict(widest.get("execution_to_entry_funnel"))
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))

    parts: list[str] = []
    if primary_bottleneck == "hold_row_dominance":
        parts.append(
            "The main bottleneck is that most known-identity rows remain hold/no-trade rows before execution eligibility."
        )
    elif primary_bottleneck == "execution_gate_scarcity":
        parts.append(
            "The main bottleneck is that many non-hold rows fail execution gating before they become executable."
        )
    elif primary_bottleneck == "positive_entry_scarcity_among_execution_allowed":
        parts.append(
            "The main bottleneck is that many execution-allowed rows still lack a positive entry_price, so executable positive-entry rows remain scarce."
        )
    elif primary_bottleneck == "downstream_quality_weakness":
        parts.append(
            "The widest window restores enough executable input, but most compatible rows still fail downstream quality selection."
        )
    else:
        parts.append(
            "The evidence is mixed and no single stage fully explains the latest edge recovery gap."
        )

    parts.append(
        f"At {widest_config}, the primary collapse stage was "
        f"{widest_execution.get('primary_collapse_stage', 'unknown')}."
    )
    parts.append(
        "Known-identity executable positive-entry rows were "
        f"{widest_funnel.get('executable_positive_entry_known_identity_rows', 0)} "
        f"out of {widest_funnel.get('raw_rows_with_known_identity', 0)} known-identity raw rows."
    )

    if _safe_dict(factors.get("recent_window_effect")).get("status") in {
        "primary",
        "contributing",
        "limited",
    }:
        parts.append(
            "Wider recent windows recover additional executable and labelable rows, so recency matters."
        )
    if _safe_dict(factors.get("latest_max_rows_cap_effect")).get("status") in {
        "contributing",
        "limited",
    }:
        parts.append(
            "Higher latest_max_rows helps at the margin, but it does not resolve the main scarcity on its own."
        )
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") in {
        "primary",
        "contributing",
    }:
        parts.append(
            "Downstream quality weakness remains visible after the upstream funnel, but it is not the first collapse stage unless explicitly flagged above."
        )
    if _safe_dict(factors.get("slow_swing_survivor_concentration")).get("status") in {
        "concentrated",
        "present",
    }:
        dominant = _safe_dict(concentration.get("dominant_survivor_group"))
        parts.append(
            "Survivors are concentrated away from the raw activity mix, with the dominant surviving group at "
            f"{dominant.get('strategy', 'n/a')} / {dominant.get('horizon', 'n/a')}."
        )

    return " ".join(parts)


def _candidate_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -_CANDIDATE_STRENGTH_ORDER.get(row.get("candidate_strength"), -1),
        -(_to_float(row.get("aggregate_score"), default=-1.0) or -1.0),
        -int(row.get("sample_count", 0) or 0),
        -int(row.get("labeled_count", 0) or 0),
        str(row.get("symbol") or ""),
        str(row.get("strategy") or ""),
        str(row.get("horizon") or ""),
        str(row.get("rejection_reason") or ""),
    )


def _strategy_horizon_counter_rows(
    counter: Counter[tuple[str, str]],
) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "strategy": strategy,
            "horizon": horizon,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for (strategy, horizon), count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0][0], _horizon_sort_key(item[0][1])),
        )
    ]


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "value": value,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _format_labeled_counts(value: Any) -> str:
    counts = _safe_dict(value)
    return ", ".join(f"{horizon}={counts.get(horizon, 0)}" for horizon in HORIZONS)


def _format_ratio(value: Any) -> str:
    ratio = _to_float(value)
    if ratio is None:
        return "n/a"
    return f"{ratio:.2%}"


def _format_counter_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get('value', 'n/a')}={_safe_dict(row).get('count', 0)}"
        for row in rows
    )


def _format_strategy_horizon_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get('strategy', 'n/a')}/{_safe_dict(row).get('horizon', 'n/a')}="
        f"{_safe_dict(row).get('count', 0)}"
        for row in rows
    )


def _format_strategy_share_shift(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    top_rows = rows[:5]
    return ", ".join(
        f"{_safe_dict(row).get('strategy', 'n/a')}: "
        f"raw={_format_ratio(_safe_dict(row).get('raw_share'))}, "
        f"survivor={_format_ratio(_safe_dict(row).get('surviving_edge_share'))}"
        for row in top_rows
    )


def _primary_stage(stage_losses: dict[str, int]) -> str:
    candidates = [(name, max(int(count or 0), 0)) for name, count in stage_losses.items()]
    top_name, top_count = min(
        candidates,
        key=lambda item: (-item[1], _PRIMARY_STAGE_ORDER.get(item[0], 99)),
    )
    if top_count <= 0:
        return "no_upstream_collapse_detected"
    return top_name


def _dominant_counter_key(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return min(counter.items(), key=lambda item: (-item[1], item[0]))[0]


def _is_hold_action(row: dict[str, Any]) -> bool:
    action = _normalize_action(
        row.get("execution_action")
        or row.get("execution_signal")
        or row.get("rule_signal")
    )
    return action in _HOLD_ACTION_VALUES


def _normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _has_positive_entry(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is not None and numeric > 0


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


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


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _to_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DiagnosisConfiguration",
    "REPORT_JSON_NAME",
    "REPORT_MD_NAME",
    "REPORT_TITLE",
    "REPORT_TYPE",
    "build_configuration_summary",
    "build_execution_to_entry_funnel",
    "build_final_assessment",
    "build_raw_to_execution_funnel",
    "build_report",
    "build_strategy_mix_summary",
    "build_survivor_concentration_summary",
    "build_symbol_strategy_execution_counts",
    "extract_diagnostic_rows",
    "extract_selected_rows",
    "main",
    "parse_args",
    "parse_configuration_values",
    "render_markdown",
    "run_executable_entry_scarcity_diagnosis_report",
    "write_report_files",
]