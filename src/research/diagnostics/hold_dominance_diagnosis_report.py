from __future__ import annotations
import argparse
import json
from collections import Counter
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

REPORT_TYPE = "hold_dominance_diagnosis_report"
REPORT_TITLE = "Hold Dominance Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_EFFECTIVE_INPUT_FILENAME = "_effective_hold_diagnosis_input.jsonl"

_ACTION_CLASS_HOLD = "hold"
_ACTION_CLASS_NON_HOLD = "non_hold"
_ACTION_CLASS_UNKNOWN = "unknown"

_HOLD_LIKE_ACTION_VALUES = {
    "hold",
    "neutral",
    "flat",
    "no_trade",
    "no-trade",
    "no_signal",
    "watchlist",
    "watchlist_long",
    "watchlist_short",
}
_NON_HOLD_ACTION_VALUES = {
    "long",
    "short",
    "buy",
    "sell",
}
_NON_DIRECTIONAL_BIAS_VALUES = {
    "hold",
    "neutral",
    "flat",
    "no_trade",
    "no-trade",
    "no_signal",
    "watchlist",
    "neutral_conflict",
    "conflict",
    "mixed",
    "unknown",
}
_DIRECTIONAL_BIAS_VALUES = {
    "long",
    "short",
    "watchlist_long",
    "watchlist_short",
    "bullish",
    "bearish",
}

_FAST_STRATEGIES = {"scalping", "intraday"}
_SLOW_SURVIVOR_TARGETS = {("swing", "4h")}

_CANDIDATE_STRENGTH_ORDER = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "insufficient_data": 0,
    "incompatible": -1,
    None: -1,
}
_PRIMARY_STAGE_ORDER = {
    "hold_row_dominance": 0,
    "execution_gate_scarcity": 1,
}

_MIN_PRIMARY_SUPPORT_ROWS = MIN_EDGE_CANDIDATE_SAMPLE_COUNT
_MIN_DIRECTIONAL_BIAS_SUPPORT_ROWS = 10
_MIN_SURVIVOR_SUPPORT_ROWS = 3


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
            "Build a diagnosis-only hold-dominance report across multiple latest-window "
            "configurations using a single effective input snapshot per configuration."
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

    result = run_hold_dominance_diagnosis_report(
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


def run_hold_dominance_diagnosis_report(
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
        (
            effective_input_path,
            raw_records,
            source_metadata,
        ) = _materialize_configuration_input(
            input_path=resolved_input,
            analyzer_output_dir=analyzer_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
        )
        labelable_dataset = build_dataset(
            path=effective_input_path,
            rotation_aware=False,
        )
        analyzer_metrics = run_research_analyzer(
            effective_input_path,
            analyzer_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
        )
        configuration_summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                effective_input_path=effective_input_path,
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
            "This report is diagnosis-only and reuses a single effective input snapshot per configuration so the funnel, labelable dataset, and analyzer all observe the same rows.",
            "Hold/no-trade rows are inferred from normalized execution/rule action values; rows without a recognized action class are reported separately as unknown rather than forced into non-hold.",
            "Current fields support evidence about where rows collapse before execution eligibility, but they do not prove whether those holds came from intended abstention, conservative strategy logic, or conservative pre-entry gating.",
        ],
    }


def _materialize_configuration_input(
    *,
    input_path: Path,
    analyzer_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    raw_records, source_metadata = load_jsonl_records_with_metadata(
        path=input_path,
        max_age_hours=latest_window_hours,
        max_rows=latest_max_rows,
    )

    analyzer_output_dir.mkdir(parents=True, exist_ok=True)
    effective_input_path = analyzer_output_dir / _EFFECTIVE_INPUT_FILENAME
    with effective_input_path.open("w", encoding="utf-8") as handle:
        for record in raw_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        **_safe_dict(source_metadata),
        "effective_input_path": str(effective_input_path),
        "effective_input_record_count": len(raw_records),
        "effective_input_materialized": True,
        "effective_input_window_hours": latest_window_hours,
        "effective_input_max_rows": latest_max_rows,
    }
    return effective_input_path, raw_records, metadata


def build_configuration_summary(
    *,
    configuration: DiagnosisConfiguration,
    input_path: Path,
    effective_input_path: Path,
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
    hold_transition_funnel = build_hold_transition_funnel(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        edge_candidate_rows=edge_candidate_rows,
    )
    hold_ratio_by_strategy = build_hold_breakdown_rows(
        normalized_raw_rows=normalized_raw_rows,
        group_fields=("strategy",),
    )
    hold_ratio_by_symbol = build_hold_breakdown_rows(
        normalized_raw_rows=normalized_raw_rows,
        group_fields=("symbol",),
    )
    hold_ratio_by_strategy_symbol = build_hold_breakdown_rows(
        normalized_raw_rows=normalized_raw_rows,
        group_fields=("strategy", "symbol"),
    )
    non_hold_ratio_by_strategy = build_non_hold_ratio_by_strategy(
        hold_ratio_by_strategy=hold_ratio_by_strategy
    )
    bias_distribution_vs_hold_outcome = build_bias_distribution_vs_hold_outcome(
        normalized_raw_rows=normalized_raw_rows
    )
    execution_allowed_false_patterns = build_execution_allowed_false_patterns(
        hold_transition_funnel=hold_transition_funnel,
        hold_ratio_by_strategy=hold_ratio_by_strategy,
        hold_ratio_by_strategy_symbol=hold_ratio_by_strategy_symbol,
    )
    strategy_mix = build_strategy_mix_summary(
        normalized_raw_rows=normalized_raw_rows,
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
        "effective_input_path": str(effective_input_path),
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
            "effective_input_path": source_metadata.get(
                "effective_input_path",
                str(effective_input_path),
            ),
            "effective_input_record_count": int(
                source_metadata.get("effective_input_record_count", len(raw_records)) or 0
            ),
            "effective_input_materialized": bool(
                source_metadata.get("effective_input_materialized", True)
            ),
        },
        "headline": {
            "display_name": configuration.display_name,
            "latest_window_hours": configuration.latest_window_hours,
            "latest_max_rows": configuration.latest_max_rows,
            "date_range_start": date_range.get("start"),
            "date_range_end": date_range.get("end"),
            "raw_input_rows": hold_transition_funnel["raw_input_rows"],
            "raw_rows_with_known_identity": hold_transition_funnel[
                "raw_rows_with_known_identity"
            ],
            "unknown_action_known_identity_rows": hold_transition_funnel[
                "unknown_action_known_identity_rows"
            ],
            "unknown_action_share_of_known_identity": hold_transition_funnel[
                "unknown_action_share_of_known_identity"
            ],
            "hold_action_known_identity_rows": hold_transition_funnel[
                "hold_action_known_identity_rows"
            ],
            "non_hold_action_known_identity_rows": hold_transition_funnel[
                "non_hold_action_known_identity_rows"
            ],
            "hold_share_of_known_identity": hold_transition_funnel[
                "hold_share_of_known_identity"
            ],
            "hold_share_of_classified_actions": hold_transition_funnel[
                "hold_share_of_classified_actions"
            ],
            "non_hold_execution_allowed_false_rows": hold_transition_funnel[
                "non_hold_execution_allowed_false_rows"
            ],
            "non_hold_execution_allowed_rows": hold_transition_funnel[
                "non_hold_execution_allowed_rows"
            ],
            "non_hold_executable_positive_entry_rows": hold_transition_funnel[
                "non_hold_executable_positive_entry_rows"
            ],
            "directional_bias_present_hold_rows": hold_transition_funnel[
                "directional_bias_present_hold_rows"
            ],
            "research_labelable_dataset_rows": hold_transition_funnel[
                "research_labelable_dataset_rows"
            ],
            "labeled_rows_by_horizon": hold_transition_funnel["labeled_rows_by_horizon"],
            "edge_candidate_row_count": hold_transition_funnel["edge_candidate_row_count"],
            "diagnostic_row_count": hold_transition_funnel["diagnostic_row_count"],
            "primary_collapse_stage": hold_transition_funnel["primary_collapse_stage"],
            "dominant_rejection_reason": empty_reason_summary.get(
                "dominant_rejection_reason"
            ),
        },
        "hold_transition_funnel": hold_transition_funnel,
        "hold_ratio_by_strategy": hold_ratio_by_strategy,
        "hold_ratio_by_symbol": hold_ratio_by_symbol,
        "hold_ratio_by_strategy_symbol": hold_ratio_by_strategy_symbol,
        "non_hold_ratio_by_strategy": non_hold_ratio_by_strategy,
        "bias_distribution_vs_hold_outcome": bias_distribution_vs_hold_outcome,
        "execution_allowed_false_patterns": execution_allowed_false_patterns,
        "strategy_mix": strategy_mix,
        "survivor_concentration_summary": survivor_concentration_summary,
        "edge_candidate_outcomes": {
            "selected_row_count": len(selected_rows),
            "diagnostic_row_count": len(diagnostic_rows),
            "diagnostic_rejection_reason_counts": _safe_dict(
                empty_reason_summary.get("diagnostic_rejection_reason_counts")
            ),
            "dominant_rejection_reason": empty_reason_summary.get(
                "dominant_rejection_reason"
            ),
        },
        "selected_survivors": selected_rows,
        "diagnostic_rows": diagnostic_rows,
    }


def build_hold_transition_funnel(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    edge_candidate_rows: dict[str, Any],
) -> dict[str, Any]:
    known_identity_rows = [
        row for row in normalized_raw_rows if _has_known_identity(row)
    ]
    hold_action_known_identity_rows = [
        row for row in known_identity_rows if _action_class(row) == _ACTION_CLASS_HOLD
    ]
    non_hold_action_known_identity_rows = [
        row for row in known_identity_rows if _action_class(row) == _ACTION_CLASS_NON_HOLD
    ]
    unknown_action_known_identity_rows = [
        row for row in known_identity_rows if _action_class(row) == _ACTION_CLASS_UNKNOWN
    ]
    classified_action_known_identity_rows = (
        len(hold_action_known_identity_rows) + len(non_hold_action_known_identity_rows)
    )
    bias_present_known_identity_rows = [
        row for row in known_identity_rows if _has_bias(row)
    ]
    directional_bias_present_known_identity_rows = [
        row for row in known_identity_rows if _has_directional_bias(row)
    ]
    directional_bias_present_hold_rows = [
        row
        for row in directional_bias_present_known_identity_rows
        if _action_class(row) == _ACTION_CLASS_HOLD
    ]
    directional_bias_present_unknown_rows = [
        row
        for row in directional_bias_present_known_identity_rows
        if _action_class(row) == _ACTION_CLASS_UNKNOWN
    ]
    execution_allowed_known_identity_rows = [
        row for row in known_identity_rows if row.get("execution_allowed") is True
    ]
    execution_allowed_false_known_identity_rows = [
        row for row in known_identity_rows if row.get("execution_allowed") is False
    ]
    hold_execution_allowed_false_rows = [
        row
        for row in hold_action_known_identity_rows
        if row.get("execution_allowed") is False
    ]
    unknown_action_execution_allowed_false_rows = [
        row
        for row in unknown_action_known_identity_rows
        if row.get("execution_allowed") is False
    ]
    non_hold_execution_allowed_rows = [
        row
        for row in non_hold_action_known_identity_rows
        if row.get("execution_allowed") is True
    ]
    non_hold_execution_allowed_false_rows = [
        row
        for row in non_hold_action_known_identity_rows
        if row.get("execution_allowed") is False
    ]
    positive_entry_known_identity_rows = [
        row for row in known_identity_rows if _has_positive_entry(row.get("entry_price"))
    ]
    non_hold_executable_positive_entry_rows = [
        row
        for row in non_hold_execution_allowed_rows
        if _has_positive_entry(row.get("entry_price"))
    ]

    labelable_known_identity_rows = [
        row for row in labelable_dataset if _has_known_identity(row)
    ]
    labeled_rows_by_horizon = {
        horizon: sum(
            1
            for row in labelable_known_identity_rows
            if has_future_fields_for_horizon(row, horizon)
        )
        for horizon in HORIZONS
    }

    stage_losses = {
        "hold_row_dominance": len(hold_action_known_identity_rows),
        "execution_gate_scarcity": len(non_hold_execution_allowed_false_rows),
    }

    return {
        "raw_input_rows": len(normalized_raw_rows),
        "raw_rows_with_known_identity": len(known_identity_rows),
        "raw_rows_without_known_identity": len(normalized_raw_rows) - len(known_identity_rows),
        "classified_action_known_identity_rows": classified_action_known_identity_rows,
        "unknown_action_known_identity_rows": len(unknown_action_known_identity_rows),
        "unknown_action_share_of_known_identity": _safe_ratio(
            len(unknown_action_known_identity_rows),
            len(known_identity_rows),
        ),
        "bias_present_known_identity_rows": len(bias_present_known_identity_rows),
        "directional_bias_present_known_identity_rows": len(
            directional_bias_present_known_identity_rows
        ),
        "directional_bias_present_hold_rows": len(directional_bias_present_hold_rows),
        "directional_bias_present_unknown_rows": len(
            directional_bias_present_unknown_rows
        ),
        "hold_action_known_identity_rows": len(hold_action_known_identity_rows),
        "non_hold_action_known_identity_rows": len(non_hold_action_known_identity_rows),
        "execution_allowed_known_identity_rows": len(execution_allowed_known_identity_rows),
        "execution_allowed_false_known_identity_rows": len(
            execution_allowed_false_known_identity_rows
        ),
        "hold_execution_allowed_false_rows": len(hold_execution_allowed_false_rows),
        "unknown_action_execution_allowed_false_rows": len(
            unknown_action_execution_allowed_false_rows
        ),
        "non_hold_execution_allowed_rows": len(non_hold_execution_allowed_rows),
        "non_hold_execution_allowed_false_rows": len(
            non_hold_execution_allowed_false_rows
        ),
        "positive_entry_known_identity_rows": len(positive_entry_known_identity_rows),
        "non_hold_executable_positive_entry_rows": len(
            non_hold_executable_positive_entry_rows
        ),
        "research_labelable_dataset_rows": len(labelable_dataset),
        "research_labelable_known_identity_rows": len(labelable_known_identity_rows),
        "rows_with_any_future_label": sum(
            1
            for row in labelable_known_identity_rows
            if any(has_future_fields_for_horizon(row, horizon) for horizon in HORIZONS)
        ),
        "labeled_rows_by_horizon": labeled_rows_by_horizon,
        "edge_candidate_row_count": int(edge_candidate_rows.get("row_count", 0) or 0),
        "diagnostic_row_count": int(
            edge_candidate_rows.get("diagnostic_row_count", 0) or 0
        ),
        "hold_share_of_known_identity": _safe_ratio(
            len(hold_action_known_identity_rows),
            len(known_identity_rows),
        ),
        "hold_share_of_classified_actions": _safe_ratio(
            len(hold_action_known_identity_rows),
            classified_action_known_identity_rows,
        ),
        "non_hold_share_of_known_identity": _safe_ratio(
            len(non_hold_action_known_identity_rows),
            len(known_identity_rows),
        ),
        "directional_bias_present_hold_share": _safe_ratio(
            len(directional_bias_present_hold_rows),
            len(directional_bias_present_known_identity_rows),
        ),
        "non_hold_execution_allowed_share": _safe_ratio(
            len(non_hold_execution_allowed_rows),
            len(non_hold_action_known_identity_rows),
        ),
        "non_hold_execution_allowed_false_share": _safe_ratio(
            len(non_hold_execution_allowed_false_rows),
            len(non_hold_action_known_identity_rows),
        ),
        "non_hold_executable_positive_entry_share": _safe_ratio(
            len(non_hold_executable_positive_entry_rows),
            len(non_hold_execution_allowed_rows),
        ),
        "primary_collapse_stage": _primary_stage(stage_losses),
        "stage_loss_counts": stage_losses,
    }


def build_hold_breakdown_rows(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    counts: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in normalized_raw_rows:
        if not _has_known_identity(row):
            continue
        key_values = tuple(_group_value(row, field) for field in group_fields)
        if any(value is None for value in key_values):
            continue

        entry = counts.setdefault(
            key_values,
            {
                **{field: key_values[index] for index, field in enumerate(group_fields)},
                "total_rows": 0,
                "classified_action_rows": 0,
                "unknown_action_rows": 0,
                "hold_rows": 0,
                "non_hold_rows": 0,
                "bias_present_rows": 0,
                "directional_bias_present_rows": 0,
                "directional_bias_hold_rows": 0,
                "directional_bias_unknown_rows": 0,
                "execution_allowed_false_rows": 0,
                "non_hold_execution_allowed_false_rows": 0,
                "non_hold_execution_allowed_rows": 0,
                "positive_entry_rows": 0,
                "non_hold_executable_positive_entry_rows": 0,
            },
        )

        action_class = _action_class(row)
        is_hold = action_class == _ACTION_CLASS_HOLD
        is_non_hold = action_class == _ACTION_CLASS_NON_HOLD
        is_unknown = action_class == _ACTION_CLASS_UNKNOWN
        execution_allowed = row.get("execution_allowed") is True
        execution_blocked = row.get("execution_allowed") is False
        has_positive_entry = _has_positive_entry(row.get("entry_price"))
        has_bias = _has_bias(row)
        has_directional_bias = _has_directional_bias(row)

        entry["total_rows"] += 1
        if not is_unknown:
            entry["classified_action_rows"] += 1
        else:
            entry["unknown_action_rows"] += 1
        if is_hold:
            entry["hold_rows"] += 1
        if is_non_hold:
            entry["non_hold_rows"] += 1
        if has_bias:
            entry["bias_present_rows"] += 1
        if has_directional_bias:
            entry["directional_bias_present_rows"] += 1
            if is_hold:
                entry["directional_bias_hold_rows"] += 1
            if is_unknown:
                entry["directional_bias_unknown_rows"] += 1
        if execution_blocked:
            entry["execution_allowed_false_rows"] += 1
        if is_non_hold and execution_blocked:
            entry["non_hold_execution_allowed_false_rows"] += 1
        if is_non_hold and execution_allowed:
            entry["non_hold_execution_allowed_rows"] += 1
        if has_positive_entry:
            entry["positive_entry_rows"] += 1
        if is_non_hold and execution_allowed and has_positive_entry:
            entry["non_hold_executable_positive_entry_rows"] += 1

    rows: list[dict[str, Any]] = []
    for entry in counts.values():
        total_rows = int(entry.get("total_rows", 0) or 0)
        classified_rows = int(entry.get("classified_action_rows", 0) or 0)
        non_hold_rows = int(entry.get("non_hold_rows", 0) or 0)
        directional_bias_rows = int(entry.get("directional_bias_present_rows", 0) or 0)
        non_hold_execution_allowed_rows = int(
            entry.get("non_hold_execution_allowed_rows", 0) or 0
        )
        rows.append(
            {
                **entry,
                "hold_ratio": _safe_ratio(int(entry.get("hold_rows", 0) or 0), total_rows),
                "non_hold_ratio": _safe_ratio(non_hold_rows, total_rows),
                "unknown_action_ratio": _safe_ratio(
                    int(entry.get("unknown_action_rows", 0) or 0),
                    total_rows,
                ),
                "hold_share_of_classified_actions": _safe_ratio(
                    int(entry.get("hold_rows", 0) or 0),
                    classified_rows,
                ),
                "execution_allowed_false_ratio": _safe_ratio(
                    int(entry.get("execution_allowed_false_rows", 0) or 0),
                    total_rows,
                ),
                "non_hold_execution_allowed_false_share_of_non_hold": _safe_ratio(
                    int(entry.get("non_hold_execution_allowed_false_rows", 0) or 0),
                    non_hold_rows,
                ),
                "directional_bias_present_hold_share": _safe_ratio(
                    int(entry.get("directional_bias_hold_rows", 0) or 0),
                    directional_bias_rows,
                ),
                "non_hold_executable_positive_entry_share_of_execution_allowed": _safe_ratio(
                    int(entry.get("non_hold_executable_positive_entry_rows", 0) or 0),
                    non_hold_execution_allowed_rows,
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("total_rows", 0) or 0),
            -int(item.get("hold_rows", 0) or 0),
            tuple(str(item.get(field, "")) for field in group_fields),
        )
    )
    return rows


def build_non_hold_ratio_by_strategy(
    *,
    hold_ratio_by_strategy: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in hold_ratio_by_strategy:
        row_dict = _safe_dict(row)
        rows.append(
            {
                "strategy": row_dict.get("strategy"),
                "total_rows": int(row_dict.get("total_rows", 0) or 0),
                "non_hold_rows": int(row_dict.get("non_hold_rows", 0) or 0),
                "unknown_action_rows": int(row_dict.get("unknown_action_rows", 0) or 0),
                "non_hold_ratio": _to_float(row_dict.get("non_hold_ratio"), default=0.0),
                "non_hold_execution_allowed_rows": int(
                    row_dict.get("non_hold_execution_allowed_rows", 0) or 0
                ),
                "non_hold_execution_allowed_false_rows": int(
                    row_dict.get("non_hold_execution_allowed_false_rows", 0) or 0
                ),
            }
        )
    return rows


def build_bias_distribution_vs_hold_outcome(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}

    for row in normalized_raw_rows:
        if not _has_known_identity(row):
            continue
        bias = _normalize_bias(row.get("bias"))
        if bias is None:
            continue
        entry = counts.setdefault(
            bias,
            {
                "bias": bias,
                "total_rows": 0,
                "hold_rows": 0,
                "non_hold_rows": 0,
                "unknown_action_rows": 0,
                "execution_allowed_false_rows": 0,
            },
        )
        entry["total_rows"] += 1
        action_class = _action_class(row)
        if action_class == _ACTION_CLASS_HOLD:
            entry["hold_rows"] += 1
        elif action_class == _ACTION_CLASS_NON_HOLD:
            entry["non_hold_rows"] += 1
        else:
            entry["unknown_action_rows"] += 1
        if row.get("execution_allowed") is False:
            entry["execution_allowed_false_rows"] += 1

    rows: list[dict[str, Any]] = []
    for entry in counts.values():
        total_rows = int(entry.get("total_rows", 0) or 0)
        rows.append(
            {
                **entry,
                "hold_ratio": _safe_ratio(int(entry.get("hold_rows", 0) or 0), total_rows),
                "non_hold_ratio": _safe_ratio(
                    int(entry.get("non_hold_rows", 0) or 0),
                    total_rows,
                ),
                "unknown_action_ratio": _safe_ratio(
                    int(entry.get("unknown_action_rows", 0) or 0),
                    total_rows,
                ),
                "execution_allowed_false_ratio": _safe_ratio(
                    int(entry.get("execution_allowed_false_rows", 0) or 0),
                    total_rows,
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("total_rows", 0) or 0),
            str(item.get("bias", "")),
        )
    )
    return rows


def build_execution_allowed_false_patterns(
    *,
    hold_transition_funnel: dict[str, Any],
    hold_ratio_by_strategy: Sequence[dict[str, Any]],
    hold_ratio_by_strategy_symbol: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_strategy = [
        {
            "strategy": row.get("strategy"),
            "execution_allowed_false_rows": int(row.get("execution_allowed_false_rows", 0) or 0),
            "execution_allowed_false_ratio": _to_float(
                row.get("execution_allowed_false_ratio"),
                default=0.0,
            ),
            "non_hold_execution_allowed_false_rows": int(
                row.get("non_hold_execution_allowed_false_rows", 0) or 0
            ),
            "non_hold_execution_allowed_false_share_of_non_hold": _to_float(
                row.get("non_hold_execution_allowed_false_share_of_non_hold"),
                default=0.0,
            ),
            "unknown_action_rows": int(row.get("unknown_action_rows", 0) or 0),
        }
        for row in hold_ratio_by_strategy
        if int(row.get("execution_allowed_false_rows", 0) or 0) > 0
    ]
    by_strategy_symbol = [
        {
            "strategy": row.get("strategy"),
            "symbol": row.get("symbol"),
            "execution_allowed_false_rows": int(row.get("execution_allowed_false_rows", 0) or 0),
            "non_hold_execution_allowed_false_rows": int(
                row.get("non_hold_execution_allowed_false_rows", 0) or 0
            ),
            "non_hold_execution_allowed_false_share_of_non_hold": _to_float(
                row.get("non_hold_execution_allowed_false_share_of_non_hold"),
                default=0.0,
            ),
            "unknown_action_rows": int(row.get("unknown_action_rows", 0) or 0),
        }
        for row in hold_ratio_by_strategy_symbol
        if int(row.get("execution_allowed_false_rows", 0) or 0) > 0
    ]

    return {
        "known_identity_execution_allowed_false_rows": int(
            hold_transition_funnel.get("execution_allowed_false_known_identity_rows", 0) or 0
        ),
        "hold_execution_allowed_false_rows": int(
            hold_transition_funnel.get("hold_execution_allowed_false_rows", 0) or 0
        ),
        "unknown_action_execution_allowed_false_rows": int(
            hold_transition_funnel.get("unknown_action_execution_allowed_false_rows", 0) or 0
        ),
        "non_hold_execution_allowed_false_rows": int(
            hold_transition_funnel.get("non_hold_execution_allowed_false_rows", 0) or 0
        ),
        "by_strategy": by_strategy,
        "by_strategy_symbol": by_strategy_symbol[:20],
    }


def build_strategy_mix_summary(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_counter: Counter[str] = Counter()
    hold_counter: Counter[str] = Counter()
    non_hold_counter: Counter[str] = Counter()
    unknown_counter: Counter[str] = Counter()
    survivor_strategy_counter: Counter[str] = Counter()
    survivor_strategy_horizon_counter: Counter[tuple[str, str]] = Counter()

    for row in normalized_raw_rows:
        if not _has_known_identity(row):
            continue
        strategy = _normalize_strategy(row.get("selected_strategy"))
        if strategy is None:
            continue
        raw_counter[strategy] += 1
        action_class = _action_class(row)
        if action_class == _ACTION_CLASS_HOLD:
            hold_counter[strategy] += 1
        elif action_class == _ACTION_CLASS_NON_HOLD:
            non_hold_counter[strategy] += 1
        else:
            unknown_counter[strategy] += 1

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
        | set(non_hold_counter)
        | set(unknown_counter)
        | set(survivor_strategy_counter)
    )
    strategy_share_shift = []
    raw_total = sum(raw_counter.values())
    survivor_total = sum(survivor_strategy_counter.values())

    for strategy in all_strategies:
        raw_count = raw_counter[strategy]
        hold_count = hold_counter[strategy]
        non_hold_count = non_hold_counter[strategy]
        unknown_action_count = unknown_counter[strategy]
        surviving_edge_count = survivor_strategy_counter[strategy]
        raw_share = _safe_ratio(raw_count, raw_total)
        surviving_edge_share = _safe_ratio(surviving_edge_count, survivor_total)
        strategy_share_shift.append(
            {
                "strategy": strategy,
                "raw_count": raw_count,
                "raw_share": raw_share,
                "hold_count": hold_count,
                "non_hold_count": non_hold_count,
                "unknown_action_count": unknown_action_count,
                "surviving_edge_count": surviving_edge_count,
                "surviving_edge_share": surviving_edge_share,
                "survivor_minus_raw_share_delta": round(
                    surviving_edge_share - raw_share,
                    6,
                ),
            }
        )

    strategy_share_shift.sort(
        key=lambda item: (
            -(_to_float(item.get("surviving_edge_share"), default=0.0) or 0.0),
            -(_to_float(item.get("survivor_minus_raw_share_delta"), default=0.0) or 0.0),
            str(item.get("strategy", "")),
        )
    )

    return {
        "raw_strategy_counts": _counter_rows(raw_counter),
        "hold_action_counts_by_strategy": _counter_rows(hold_counter),
        "non_hold_action_counts_by_strategy": _counter_rows(non_hold_counter),
        "unknown_action_counts_by_strategy": _counter_rows(unknown_counter),
        "surviving_edge_rows_by_strategy": _counter_rows(survivor_strategy_counter),
        "surviving_edge_rows_by_strategy_horizon": _strategy_horizon_counter_rows(
            survivor_strategy_horizon_counter
        ),
        "strategy_share_shift": strategy_share_shift,
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
            "factors": {},
            "widest_configuration": None,
            "widest_selected_survivors": [],
            "confirmed_observations": [],
            "evidence_backed_inferences": [],
            "unresolved_uncertainties": [],
            "overall_conclusion": "No configurations were evaluated.",
        }

    widest = max(
        summaries,
        key=lambda item: (
            int(_safe_dict(item.get("configuration")).get("latest_window_hours", 0) or 0),
            int(_safe_dict(item.get("configuration")).get("latest_max_rows", 0) or 0),
        ),
    )

    factors = {
        "hold_row_dominance": _assess_hold_row_dominance(widest),
        "execution_gate_scarcity": _assess_execution_gate_scarcity(widest),
        "directional_bias_present_hold_overlap": _assess_directional_bias_hold_overlap(
            widest
        ),
        "downstream_quality_weakness": _assess_downstream_quality_weakness(widest),
        "strategy_mix_mismatch": _assess_strategy_mix_mismatch(widest),
        "slow_swing_survivor_concentration": _assess_slow_swing_survivor_concentration(
            widest
        ),
    }

    primary_bottleneck = _primary_bottleneck_label(factors=factors, widest=widest)
    confirmed_observations = _build_confirmed_observations(widest)
    evidence_backed_inferences = _build_evidence_backed_inferences(
        factors=factors,
        widest=widest,
        primary_bottleneck=primary_bottleneck,
    )
    unresolved_uncertainties = _build_unresolved_uncertainties(widest)

    return {
        "assessment": _overall_assessment_label(
            primary_bottleneck=primary_bottleneck,
            factors=factors,
        ),
        "primary_bottleneck": primary_bottleneck,
        "factors": factors,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "widest_selected_survivors": _safe_list(widest.get("selected_survivors")),
        "confirmed_observations": confirmed_observations,
        "evidence_backed_inferences": evidence_backed_inferences,
        "unresolved_uncertainties": unresolved_uncertainties,
        "overall_conclusion": _build_overall_conclusion(
            factors=factors,
            widest=widest,
            primary_bottleneck=primary_bottleneck,
        ),
    }


def _assess_hold_row_dominance(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))

    raw_known = int(funnel.get("raw_rows_with_known_identity", 0) or 0)
    hold_known = int(funnel.get("hold_action_known_identity_rows", 0) or 0)
    directional_bias_present_rows = int(
        funnel.get("directional_bias_present_known_identity_rows", 0) or 0
    )
    directional_bias_hold_rows = int(
        funnel.get("directional_bias_present_hold_rows", 0) or 0
    )
    hold_share = _to_float(funnel.get("hold_share_of_known_identity"), default=0.0) or 0.0
    hold_share_of_classified_actions = _to_float(
        funnel.get("hold_share_of_classified_actions"),
        default=0.0,
    ) or 0.0
    unknown_share = _to_float(
        funnel.get("unknown_action_share_of_known_identity"),
        default=0.0,
    ) or 0.0

    evidence = [
        (
            f"Known-identity rows: {raw_known}; hold/no-trade known-identity rows: "
            f"{hold_known} ({hold_share:.2%})."
        ),
        (
            "Known-identity rows with unknown action classification: "
            f"{int(funnel.get('unknown_action_known_identity_rows', 0) or 0)} "
            f"({unknown_share:.2%})."
        ),
        (
            "Primary collapse stage at widest configuration: "
            f"{funnel.get('primary_collapse_stage', 'unknown')}."
        ),
        (
            "Directional-bias-present rows that still ended hold/no-trade: "
            f"{directional_bias_hold_rows} of {directional_bias_present_rows} "
            f"({_safe_ratio(directional_bias_hold_rows, directional_bias_present_rows):.2%})."
        ),
    ]

    if raw_known < _MIN_PRIMARY_SUPPORT_ROWS:
        status = "insufficient_support"
    elif (
        funnel.get("primary_collapse_stage") == "hold_row_dominance"
        and hold_share >= 0.50
        and hold_share_of_classified_actions >= 0.50
    ) or hold_share >= 0.70:
        status = "primary"
    elif hold_share >= 0.50:
        status = "contributing"
    elif hold_share >= 0.35:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_execution_gate_scarcity(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))

    raw_known = int(funnel.get("raw_rows_with_known_identity", 0) or 0)
    non_hold_rows = int(funnel.get("non_hold_action_known_identity_rows", 0) or 0)
    blocked_rows = int(funnel.get("non_hold_execution_allowed_false_rows", 0) or 0)
    blocked_share = _to_float(
        funnel.get("non_hold_execution_allowed_false_share"),
        default=0.0,
    ) or 0.0

    evidence = [
        (
            f"Known-identity non-hold rows: {non_hold_rows}; execution_allowed=False "
            f"non-hold rows: {blocked_rows} ({blocked_share:.2%})."
        ),
        (
            f"Execution-allowed non-hold rows: "
            f"{int(funnel.get('non_hold_execution_allowed_rows', 0) or 0)}."
        ),
    ]

    if raw_known < _MIN_PRIMARY_SUPPORT_ROWS or non_hold_rows <= 0:
        status = "insufficient_support" if raw_known < _MIN_PRIMARY_SUPPORT_ROWS else "not_supported"
    elif (
        funnel.get("primary_collapse_stage") == "execution_gate_scarcity"
        and blocked_share >= 0.50
    ):
        status = "primary"
    elif blocked_share >= 0.35:
        status = "contributing"
    elif blocked_share >= 0.20:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_directional_bias_hold_overlap(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))

    directional_rows = int(
        funnel.get("directional_bias_present_known_identity_rows", 0) or 0
    )
    directional_hold_rows = int(
        funnel.get("directional_bias_present_hold_rows", 0) or 0
    )
    directional_unknown_rows = int(
        funnel.get("directional_bias_present_unknown_rows", 0) or 0
    )
    directional_hold_share = _to_float(
        funnel.get("directional_bias_present_hold_share"),
        default=0.0,
    ) or 0.0

    evidence = [
        (
            "Directional-bias-present known-identity rows: "
            f"{directional_rows}; hold/no-trade outcomes among them: "
            f"{directional_hold_rows} ({directional_hold_share:.2%}); "
            f"unknown-action outcomes among them: {directional_unknown_rows}."
        )
    ]

    if directional_rows < _MIN_DIRECTIONAL_BIAS_SUPPORT_ROWS:
        status = "insufficient_support"
    elif directional_hold_share >= 0.50:
        status = "present"
    elif directional_hold_share >= 0.25:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_downstream_quality_weakness(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))
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

    executable_positive_rows = int(
        funnel.get("non_hold_executable_positive_entry_rows", 0) or 0
    )
    labelable_rows = int(funnel.get("research_labelable_dataset_rows", 0) or 0)
    selected_rows = int(edge_candidate_outcomes.get("selected_row_count", 0) or 0)
    dominant_rejection_reason = edge_candidate_outcomes.get("dominant_rejection_reason")

    evidence = [
        (
            f"Non-hold executable positive-entry rows: {executable_positive_rows}; "
            f"research-labelable rows: {labelable_rows}."
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
        and executable_positive_rows >= MIN_EDGE_CANDIDATE_SAMPLE_COUNT
        and selected_rows <= 1
    ):
        status = "primary"
    elif quality_relevant_rejections > 0 or dominant_rejection_reason == "candidate_strength_weak":
        status = "contributing"
    elif labelable_rows < _MIN_PRIMARY_SUPPORT_ROWS and executable_positive_rows <= 0:
        status = "insufficient_support"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_strategy_mix_mismatch(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    total_survivors = int(concentration.get("total_surviving_edge_rows", 0) or 0)
    raw_known = int(funnel.get("raw_rows_with_known_identity", 0) or 0)
    raw_fast_share = _to_float(
        concentration.get("raw_intraday_scalping_share"),
        default=0.0,
    ) or 0.0
    survivor_fast_share = _to_float(
        concentration.get("survivor_intraday_scalping_share"),
        default=0.0,
    ) or 0.0
    slow_swing_share = _to_float(
        concentration.get("slow_swing_survivor_share"),
        default=0.0,
    ) or 0.0

    evidence = [
        (
            f"Raw intraday/scalping share: {raw_fast_share:.2%}; survivor intraday/scalping "
            f"share: {survivor_fast_share:.2%}; slow swing survivor share: "
            f"{slow_swing_share:.2%}."
        )
    ]

    if raw_known < _MIN_PRIMARY_SUPPORT_ROWS or total_survivors < _MIN_SURVIVOR_SUPPORT_ROWS:
        status = "insufficient_support"
    elif (
        raw_fast_share >= 0.50
        and (raw_fast_share - survivor_fast_share) >= 0.30
        and slow_swing_share >= 0.50
    ):
        status = "contributing"
    elif raw_fast_share > survivor_fast_share:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_slow_swing_survivor_concentration(widest: dict[str, Any]) -> dict[str, Any]:
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    total_survivors = int(concentration.get("total_surviving_edge_rows", 0) or 0)
    slow_share = _to_float(concentration.get("slow_swing_survivor_share"), default=0.0)
    raw_fast_share = _to_float(
        concentration.get("raw_intraday_scalping_share"),
        default=0.0,
    )
    survivor_fast_share = _to_float(
        concentration.get("survivor_intraday_scalping_share"),
        default=0.0,
    )
    dominant_group = _safe_dict(concentration.get("dominant_survivor_group"))

    evidence = [
        f"Total surviving edge rows at widest configuration: {total_survivors}.",
        (
            f"Slow swing survivor share: {slow_share:.2%}; raw intraday/scalping share: "
            f"{raw_fast_share:.2%}; survivor intraday/scalping share: "
            f"{survivor_fast_share:.2%}."
        ),
        (
            "Dominant survivor group: "
            f"{dominant_group.get('strategy', 'n/a')} / {dominant_group.get('horizon', 'n/a')}."
        ),
    ]

    if total_survivors < _MIN_SURVIVOR_SUPPORT_ROWS:
        status = "insufficient_support"
    elif slow_share >= 0.75 and (raw_fast_share or 0.0) > (survivor_fast_share or 0.0):
        status = "concentrated"
    elif slow_share > 0.50 or (raw_fast_share or 0.0) > (survivor_fast_share or 0.0):
        status = "present"
    else:
        status = "mixed"

    return {"status": status, "evidence": evidence}


def _primary_bottleneck_label(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    if _safe_dict(factors.get("hold_row_dominance")).get("status") == "primary":
        return "hold_row_dominance"
    if _safe_dict(factors.get("execution_gate_scarcity")).get("status") == "primary":
        return "execution_gate_scarcity"
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "primary":
        return "downstream_quality_weakness"

    widest_funnel = _safe_dict(widest.get("hold_transition_funnel"))
    primary_stage = widest_funnel.get("primary_collapse_stage")

    if (
        primary_stage == "hold_row_dominance"
        and _safe_dict(factors.get("hold_row_dominance")).get("status")
        in {"contributing", "limited"}
    ):
        return "hold_row_dominance"
    if (
        primary_stage == "execution_gate_scarcity"
        and _safe_dict(factors.get("execution_gate_scarcity")).get("status")
        in {"contributing", "limited"}
    ):
        return "execution_gate_scarcity"
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "contributing":
        return "downstream_quality_weakness"

    return "mixed_or_inconclusive"


def _overall_assessment_label(
    *,
    primary_bottleneck: str,
    factors: dict[str, Any],
) -> str:
    if primary_bottleneck == "hold_row_dominance" and _safe_dict(
        factors.get("hold_row_dominance")
    ).get("status") == "primary":
        return "hold_row_dominance_primary"
    if primary_bottleneck == "execution_gate_scarcity" and _safe_dict(
        factors.get("execution_gate_scarcity")
    ).get("status") == "primary":
        return "execution_gate_scarcity_primary"
    if primary_bottleneck == "downstream_quality_weakness" and _safe_dict(
        factors.get("downstream_quality_weakness")
    ).get("status") == "primary":
        return "downstream_quality_weakness_primary"

    statuses = {
        name: _safe_dict(payload).get("status") for name, payload in factors.items()
    }
    if any(
        status in {"contributing", "limited", "present", "concentrated"}
        for status in statuses.values()
    ):
        return "mixed_contributing_factors"
    if any(status == "insufficient_support" for status in statuses.values()):
        return "insufficient_support"
    return "not_supported_or_inconclusive"


def _build_confirmed_observations(widest: dict[str, Any]) -> list[str]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))
    edge_candidate_outcomes = _safe_dict(widest.get("edge_candidate_outcomes"))
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    dominant_group = _safe_dict(concentration.get("dominant_survivor_group"))

    observations = [
        (
            f"Known-identity rows: {int(funnel.get('raw_rows_with_known_identity', 0) or 0)}; "
            f"unknown-action rows: {int(funnel.get('unknown_action_known_identity_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('unknown_action_share_of_known_identity'))}); "
            f"hold/no-trade rows: {int(funnel.get('hold_action_known_identity_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('hold_share_of_known_identity'))})."
        ),
        (
            "Directional-bias-present rows that still ended hold/no-trade: "
            f"{int(funnel.get('directional_bias_present_hold_rows', 0) or 0)} of "
            f"{int(funnel.get('directional_bias_present_known_identity_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('directional_bias_present_hold_share'))})."
        ),
        (
            "Non-hold rows blocked by execution_allowed=False before eligibility: "
            f"{int(funnel.get('non_hold_execution_allowed_false_rows', 0) or 0)} of "
            f"{int(funnel.get('non_hold_action_known_identity_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('non_hold_execution_allowed_false_share'))})."
        ),
        (
            f"Selected edge rows at the widest configuration: "
            f"{int(edge_candidate_outcomes.get('selected_row_count', 0) or 0)}; "
            f"dominant survivor group: {dominant_group.get('strategy', 'n/a')} / "
            f"{dominant_group.get('horizon', 'n/a')}."
        ),
    ]
    return observations


def _build_evidence_backed_inferences(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
    primary_bottleneck: str,
) -> list[str]:
    inferences: list[str] = []
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    dominant_group = _safe_dict(concentration.get("dominant_survivor_group"))
    unknown_share = _to_float(
        _safe_dict(widest.get("hold_transition_funnel")).get(
            "unknown_action_share_of_known_identity"
        ),
        default=0.0,
    ) or 0.0

    if primary_bottleneck == "hold_row_dominance":
        inferences.append(
            "The first major collapse happens upstream because known-identity rows are lost to hold/no-trade before execution eligibility becomes broadly available."
        )
    elif primary_bottleneck == "execution_gate_scarcity":
        inferences.append(
            "The data suggests non-hold rows still face meaningful pre-entry gating pressure after they escape hold/no-trade status."
        )
    elif primary_bottleneck == "downstream_quality_weakness":
        inferences.append(
            "The widest configuration restores enough upstream candidates that downstream quality rejection becomes the dominant remaining loss."
        )

    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") in {
        "primary",
        "contributing",
    } and primary_bottleneck != "downstream_quality_weakness":
        inferences.append(
            "Downstream quality weakness is still visible after the upstream collapse, but the evidence places it after the main hold-dominance bottleneck rather than before it."
        )

    if unknown_share > 0.0:
        inferences.append(
            "Some known-identity rows do not expose a recognized action class, so the hold vs non-hold split should be read alongside the unknown-action share rather than as a forced binary partition."
        )

    if _safe_dict(factors.get("strategy_mix_mismatch")).get("status") in {
        "contributing",
        "limited",
    } or _safe_dict(factors.get("slow_swing_survivor_concentration")).get("status") in {
        "concentrated",
        "present",
    }:
        inferences.append(
            "The raw activity mix and survivor mix diverge: intraday/scalping rows dominate the upstream flow, while durable survivors concentrate in "
            f"{dominant_group.get('strategy', 'n/a')} / {dominant_group.get('horizon', 'n/a')}."
        )

    return inferences


def _build_unresolved_uncertainties(widest: dict[str, Any]) -> list[str]:
    funnel = _safe_dict(widest.get("hold_transition_funnel"))
    directional_rows = int(
        funnel.get("directional_bias_present_known_identity_rows", 0) or 0
    )

    uncertainties = [
        "The current schema does not expose a per-row hold reason code, so this report cannot prove whether hold/no-trade outcomes reflect intended strategy abstention under current market conditions or overly conservative strategy logic.",
        "The available fields do not isolate whether a row became hold/no-trade inside strategy selection logic or inside a pre-entry gating path before execution eligibility.",
    ]
    if directional_rows > 0:
        uncertainties.append(
            "Bias-bearing rows that still end hold/no-trade are observable, but the schema does not explain why those rows were withheld."
        )
    else:
        uncertainties.append(
            "The widest configuration did not contain directional-bias-present rows, so bias-bearing hold overlap cannot be assessed there."
        )
    if int(funnel.get("unknown_action_known_identity_rows", 0) or 0) > 0:
        uncertainties.append(
            "Rows with unknown action classification are reported separately, but the current schema does not reveal whether they should ultimately be interpreted as abstention, watchlist state, or incomplete execution metadata."
        )
    return uncertainties


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
    funnel = _safe_dict(widest.get("hold_transition_funnel"))
    concentration = _safe_dict(widest.get("survivor_concentration_summary"))
    dominant = _safe_dict(concentration.get("dominant_survivor_group"))

    parts: list[str] = []
    if primary_bottleneck == "hold_row_dominance":
        parts.append(
            "The primary confirmed bottleneck is upstream hold/no-trade dominance among known-identity rows."
        )
    elif primary_bottleneck == "execution_gate_scarcity":
        parts.append(
            "The primary confirmed bottleneck is execution gating pressure after rows escape hold/no-trade status."
        )
    elif primary_bottleneck == "downstream_quality_weakness":
        parts.append(
            "The primary confirmed bottleneck is downstream quality weakness after upstream eligibility is restored."
        )
    else:
        parts.append(
            "The evidence is mixed and no single stage is confirmed as the only bottleneck."
        )

    parts.append(
        f"At {widest_config}, known-identity rows with unknown action classification were "
        f"{int(funnel.get('unknown_action_known_identity_rows', 0) or 0)} of "
        f"{int(funnel.get('raw_rows_with_known_identity', 0) or 0)} "
        f"({_format_ratio(funnel.get('unknown_action_share_of_known_identity'))})."
    )
    parts.append(
        f"At {widest_config}, hold/no-trade known-identity rows were "
        f"{int(funnel.get('hold_action_known_identity_rows', 0) or 0)} of "
        f"{int(funnel.get('raw_rows_with_known_identity', 0) or 0)} "
        f"({_format_ratio(funnel.get('hold_share_of_known_identity'))})."
    )
    parts.append(
        "Non-hold rows blocked by execution_allowed=False were "
        f"{int(funnel.get('non_hold_execution_allowed_false_rows', 0) or 0)} of "
        f"{int(funnel.get('non_hold_action_known_identity_rows', 0) or 0)} "
        f"({_format_ratio(funnel.get('non_hold_execution_allowed_false_share'))})."
    )

    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") in {
        "primary",
        "contributing",
    }:
        parts.append(
            "Downstream quality weakness remains a real secondary pressure after the upstream funnel."
        )
    if _safe_dict(factors.get("slow_swing_survivor_concentration")).get("status") in {
        "concentrated",
        "present",
    }:
        parts.append(
            "Survivors remain concentrated away from the dominant raw mix, with the leading survivor group at "
            f"{dominant.get('strategy', 'n/a')} / {dominant.get('horizon', 'n/a')}."
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
            "- raw_rows_with_known_identity: "
            f"{headline.get('raw_rows_with_known_identity', 0)}"
        )
        lines.append(
            "- unknown_action_known_identity_rows: "
            f"{headline.get('unknown_action_known_identity_rows', 0)} "
            f"({_format_ratio(headline.get('unknown_action_share_of_known_identity'))})"
        )
        lines.append(
            "- hold_action_known_identity_rows: "
            f"{headline.get('hold_action_known_identity_rows', 0)}"
        )
        lines.append(
            "- non_hold_action_known_identity_rows: "
            f"{headline.get('non_hold_action_known_identity_rows', 0)}"
        )
        lines.append(
            "- hold_share_of_known_identity: "
            f"{_format_ratio(headline.get('hold_share_of_known_identity'))}"
        )
        lines.append(
            "- hold_share_of_classified_actions: "
            f"{_format_ratio(headline.get('hold_share_of_classified_actions'))}"
        )
        lines.append(
            "- directional_bias_present_hold_rows: "
            f"{headline.get('directional_bias_present_hold_rows', 0)}"
        )
        lines.append(
            "- non_hold_execution_allowed_false_rows: "
            f"{headline.get('non_hold_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            "- non_hold_execution_allowed_rows: "
            f"{headline.get('non_hold_execution_allowed_rows', 0)}"
        )
        lines.append(
            "- non_hold_executable_positive_entry_rows: "
            f"{headline.get('non_hold_executable_positive_entry_rows', 0)}"
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

    lines.append("## Hold Transition Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("hold_transition_funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            f"- raw_rows_with_known_identity: {funnel.get('raw_rows_with_known_identity', 0)}"
        )
        lines.append(
            f"- unknown_action_known_identity_rows: {funnel.get('unknown_action_known_identity_rows', 0)}"
        )
        lines.append(
            f"- hold_action_known_identity_rows: {funnel.get('hold_action_known_identity_rows', 0)}"
        )
        lines.append(
            f"- non_hold_action_known_identity_rows: {funnel.get('non_hold_action_known_identity_rows', 0)}"
        )
        lines.append(
            "- directional_bias_present_known_identity_rows: "
            f"{funnel.get('directional_bias_present_known_identity_rows', 0)}"
        )
        lines.append(
            "- directional_bias_present_hold_rows: "
            f"{funnel.get('directional_bias_present_hold_rows', 0)}"
        )
        lines.append(
            "- non_hold_execution_allowed_rows: "
            f"{funnel.get('non_hold_execution_allowed_rows', 0)}"
        )
        lines.append(
            "- non_hold_execution_allowed_false_rows: "
            f"{funnel.get('non_hold_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            "- non_hold_executable_positive_entry_rows: "
            f"{funnel.get('non_hold_executable_positive_entry_rows', 0)}"
        )
        lines.append(
            "- unknown_action_share_of_known_identity: "
            f"{_format_ratio(funnel.get('unknown_action_share_of_known_identity'))}"
        )
        lines.append(
            "- hold_share_of_known_identity: "
            f"{_format_ratio(funnel.get('hold_share_of_known_identity'))}"
        )
        lines.append(
            "- hold_share_of_classified_actions: "
            f"{_format_ratio(funnel.get('hold_share_of_classified_actions'))}"
        )
        lines.append(
            "- non_hold_execution_allowed_false_share: "
            f"{_format_ratio(funnel.get('non_hold_execution_allowed_false_share'))}"
        )
        lines.append(
            "- non_hold_executable_positive_entry_share: "
            f"{_format_ratio(funnel.get('non_hold_executable_positive_entry_share'))}"
        )
        lines.append(
            "- primary_collapse_stage: "
            f"{funnel.get('primary_collapse_stage', 'n/a')}"
        )
        lines.append("")

    for key, title in (
        ("hold_ratio_by_strategy", "Hold Ratio By Strategy"),
        ("hold_ratio_by_symbol", "Hold Ratio By Symbol"),
        ("hold_ratio_by_strategy_symbol", "Hold Ratio By Strategy and Symbol"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for summary in _safe_list(report.get("configuration_summaries")):
            config = _safe_dict(_safe_dict(summary).get("configuration"))
            rows = _safe_list(_safe_dict(summary).get(key))
            lines.append(f"### {config.get('display_name', 'n/a')}")
            if not rows:
                lines.append("No grouped hold-ratio rows available.")
                lines.append("")
                continue
            for row in rows[:15]:
                lines.append(f"- {_format_hold_breakdown_row(row, key=key)}")
            lines.append("")

    lines.append("## Non-Hold Ratio By Strategy")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        rows = _safe_list(_safe_dict(summary).get("non_hold_ratio_by_strategy"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        if not rows:
            lines.append("No non-hold strategy rows available.")
            lines.append("")
            continue
        for row in rows[:15]:
            item = _safe_dict(row)
            lines.append(
                "- "
                f"{item.get('strategy', 'n/a')}: "
                f"non_hold={item.get('non_hold_rows', 0)}/{item.get('total_rows', 0)} "
                f"({_format_ratio(item.get('non_hold_ratio'))}), "
                f"unknown_action={item.get('unknown_action_rows', 0)}, "
                f"execution_allowed={item.get('non_hold_execution_allowed_rows', 0)}, "
                f"execution_allowed_false={item.get('non_hold_execution_allowed_false_rows', 0)}"
            )
        lines.append("")

    lines.append("## Bias Distribution vs Final Hold Outcome")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        rows = _safe_list(_safe_dict(summary).get("bias_distribution_vs_hold_outcome"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        if not rows:
            lines.append("No bias-bearing rows available.")
            lines.append("")
            continue
        for row in rows[:15]:
            item = _safe_dict(row)
            lines.append(
                "- "
                f"{item.get('bias', 'n/a')}: "
                f"hold={item.get('hold_rows', 0)}/{item.get('total_rows', 0)} "
                f"({_format_ratio(item.get('hold_ratio'))}), "
                f"non_hold={item.get('non_hold_rows', 0)}/{item.get('total_rows', 0)} "
                f"({_format_ratio(item.get('non_hold_ratio'))}), "
                f"unknown_action={item.get('unknown_action_rows', 0)}/{item.get('total_rows', 0)} "
                f"({_format_ratio(item.get('unknown_action_ratio'))}), "
                f"execution_allowed_false={item.get('execution_allowed_false_rows', 0)}"
            )
        lines.append("")

    lines.append("## execution_allowed=False Patterns")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        patterns = _safe_dict(_safe_dict(summary).get("execution_allowed_false_patterns"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- known_identity_execution_allowed_false_rows: "
            f"{patterns.get('known_identity_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            f"- hold_execution_allowed_false_rows: {patterns.get('hold_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            "- unknown_action_execution_allowed_false_rows: "
            f"{patterns.get('unknown_action_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            "- non_hold_execution_allowed_false_rows: "
            f"{patterns.get('non_hold_execution_allowed_false_rows', 0)}"
        )
        lines.append(
            "- by_strategy: "
            f"{_format_execution_allowed_false_rows(patterns.get('by_strategy'))}"
        )
        lines.append(
            "- by_strategy_symbol: "
            f"{_format_execution_allowed_false_rows(patterns.get('by_strategy_symbol'))}"
        )
        lines.append("")

    lines.append("## Strategy Mix and Survivor Concentration")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        strategy_mix = _safe_dict(_safe_dict(summary).get("strategy_mix"))
        concentration = _safe_dict(
            _safe_dict(summary).get("survivor_concentration_summary")
        )
        dominant = _safe_dict(concentration.get("dominant_survivor_group"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- raw_strategy_counts: "
            f"{_format_counter_rows(strategy_mix.get('raw_strategy_counts'))}"
        )
        lines.append(
            "- hold_action_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('hold_action_counts_by_strategy'))}"
        )
        lines.append(
            "- non_hold_action_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('non_hold_action_counts_by_strategy'))}"
        )
        lines.append(
            "- unknown_action_counts_by_strategy: "
            f"{_format_counter_rows(strategy_mix.get('unknown_action_counts_by_strategy'))}"
        )
        lines.append(
            "- surviving_edge_rows_by_strategy_horizon: "
            f"{_format_strategy_horizon_rows(strategy_mix.get('surviving_edge_rows_by_strategy_horizon'))}"
        )
        lines.append(
            "- strategy_share_shift: "
            f"{_format_strategy_share_shift(strategy_mix.get('strategy_share_shift'))}"
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
            "- slow_swing_survivor_share: "
            f"{_format_ratio(concentration.get('slow_swing_survivor_share'))}"
        )
        lines.append(
            "- dominant_survivor_group: "
            f"{dominant.get('strategy', 'n/a')} / {dominant.get('horizon', 'n/a')}"
        )
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append(f"- assessment: {final_assessment.get('assessment', 'n/a')}")
    lines.append(
        f"- primary_bottleneck: {final_assessment.get('primary_bottleneck', 'n/a')}"
    )
    for factor_name, payload in _safe_dict(final_assessment.get("factors")).items():
        factor = _safe_dict(payload)
        lines.append(f"- {factor_name}: {factor.get('status', 'unknown')}")
        for item in _safe_list(factor.get("evidence")):
            lines.append(f"  evidence: {item}")
    for item in _safe_list(final_assessment.get("confirmed_observations")):
        lines.append(f"- confirmed_observation: {item}")
    for item in _safe_list(final_assessment.get("evidence_backed_inferences")):
        lines.append(f"- evidence_backed_inference: {item}")
    for item in _safe_list(final_assessment.get("unresolved_uncertainties")):
        lines.append(f"- unresolved_uncertainty: {item}")
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


def _format_hold_breakdown_row(row: dict[str, Any], *, key: str) -> str:
    item = _safe_dict(row)
    if key == "hold_ratio_by_strategy":
        label = f"{item.get('strategy', 'n/a')}"
    elif key == "hold_ratio_by_symbol":
        label = f"{item.get('symbol', 'n/a')}"
    else:
        label = f"{item.get('strategy', 'n/a')} / {item.get('symbol', 'n/a')}"
    return (
        f"{label}: hold={item.get('hold_rows', 0)}/{item.get('total_rows', 0)} "
        f"({_format_ratio(item.get('hold_ratio'))}), "
        f"non_hold={item.get('non_hold_rows', 0)}/{item.get('total_rows', 0)} "
        f"({_format_ratio(item.get('non_hold_ratio'))}), "
        f"unknown_action={item.get('unknown_action_rows', 0)}/{item.get('total_rows', 0)} "
        f"({_format_ratio(item.get('unknown_action_ratio'))}), "
        f"execution_allowed_false={item.get('execution_allowed_false_rows', 0)}, "
        f"non_hold_execution_allowed_false={item.get('non_hold_execution_allowed_false_rows', 0)}"
    )


def _format_execution_allowed_false_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    formatted: list[str] = []
    for row in rows[:8]:
        item = _safe_dict(row)
        prefix = (
            f"{item.get('strategy', 'n/a')} / {item.get('symbol', 'n/a')}"
            if item.get("symbol") is not None
            else f"{item.get('strategy', 'n/a')}"
        )
        formatted.append(
            f"{prefix}: blocked={item.get('execution_allowed_false_rows', 0)}, "
            f"non_hold_blocked={item.get('non_hold_execution_allowed_false_rows', 0)} "
            f"({_format_ratio(item.get('non_hold_execution_allowed_false_share_of_non_hold'))}), "
            f"unknown_action={item.get('unknown_action_rows', 0)}"
        )
    return ", ".join(formatted)


def _group_value(row: dict[str, Any], field: str) -> str | None:
    if field == "strategy":
        return _normalize_strategy(row.get("selected_strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    raise ValueError(f"Unsupported group field: {field}")


def _primary_stage(stage_losses: dict[str, int]) -> str:
    candidates = [(name, max(int(count or 0), 0)) for name, count in stage_losses.items()]
    top_name, top_count = min(
        candidates,
        key=lambda item: (-item[1], _PRIMARY_STAGE_ORDER.get(item[0], 99)),
    )
    if top_count <= 0:
        return "no_upstream_collapse_detected"
    return top_name


def _has_known_identity(row: dict[str, Any]) -> bool:
    return (
        _normalize_symbol(row.get("symbol")) is not None
        and _normalize_strategy(row.get("selected_strategy")) is not None
    )


def _action_class(row: dict[str, Any]) -> str:
    action = _normalize_action(
        row.get("execution_action")
        or row.get("execution_signal")
        or row.get("rule_signal")
    )
    if action is None:
        return _ACTION_CLASS_UNKNOWN
    if action in _HOLD_LIKE_ACTION_VALUES:
        return _ACTION_CLASS_HOLD
    if action in _NON_HOLD_ACTION_VALUES:
        return _ACTION_CLASS_NON_HOLD
    return _ACTION_CLASS_UNKNOWN


def _has_bias(row: dict[str, Any]) -> bool:
    return _normalize_bias(row.get("bias")) is not None


def _has_directional_bias(row: dict[str, Any]) -> bool:
    bias = _normalize_bias(row.get("bias"))
    if bias is None or bias in _NON_DIRECTIONAL_BIAS_VALUES:
        return False
    return bias in _DIRECTIONAL_BIAS_VALUES


def _normalize_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _normalize_bias(value: Any) -> str | None:
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
    "build_bias_distribution_vs_hold_outcome",
    "build_configuration_summary",
    "build_execution_allowed_false_patterns",
    "build_final_assessment",
    "build_hold_breakdown_rows",
    "build_hold_transition_funnel",
    "build_non_hold_ratio_by_strategy",
    "build_report",
    "build_strategy_mix_summary",
    "build_survivor_concentration_summary",
    "extract_diagnostic_rows",
    "extract_selected_rows",
    "main",
    "parse_args",
    "parse_configuration_values",
    "render_markdown",
    "run_hold_dominance_diagnosis_report",
    "write_report_files",
]

