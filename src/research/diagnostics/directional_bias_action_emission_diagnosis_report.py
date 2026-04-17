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

REPORT_TYPE = "directional_bias_action_emission_diagnosis_report"
REPORT_TITLE = "Directional Bias Action Emission Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_EFFECTIVE_INPUT_FILENAME = "_effective_directional_bias_action_emission_input.jsonl"
_MISSING_ACTION_LABEL = "(missing)"

_ACTION_CLASS_HOLD = "hold"
_ACTION_CLASS_BUY = "buy"
_ACTION_CLASS_SELL = "sell"
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
_BUY_LIKE_ACTION_VALUES = {"buy", "long"}
_SELL_LIKE_ACTION_VALUES = {"sell", "short"}

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
_BULLISH_BIAS_VALUES = {"long", "bullish", "watchlist_long", "buy"}
_BEARISH_BIAS_VALUES = {"short", "bearish", "watchlist_short", "sell"}

_PRIMARY_STAGE_ORDER = {
    "directional_bias_to_action_emission_scarcity": 0,
    "post_emission_labelability_scarcity": 1,
}
_ACTION_CLASS_ORDER = {
    _ACTION_CLASS_HOLD: 0,
    _ACTION_CLASS_BUY: 1,
    _ACTION_CLASS_SELL: 2,
    _ACTION_CLASS_UNKNOWN: 3,
}
_CANDIDATE_STRENGTH_ORDER = {
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "insufficient_data": 0,
    "incompatible": -1,
    None: -1,
}

_MIN_PRIMARY_SUPPORT_ROWS = MIN_EDGE_CANDIDATE_SAMPLE_COUNT
_MIN_DIRECTIONAL_SUPPORT_ROWS = 10
_MIN_STRATEGY_SUPPORT_ROWS = MIN_EDGE_CANDIDATE_SAMPLE_COUNT
_MIN_SIGN_SUPPORT_ROWS = 10


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
            "Build a diagnosis-only directional-bias-to-action emission report across "
            "multiple latest-window configurations using a single effective input "
            "snapshot per configuration."
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

    result = run_directional_bias_action_emission_diagnosis_report(
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


def run_directional_bias_action_emission_diagnosis_report(
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
            "This report is diagnosis-only and reuses a single effective input snapshot per configuration so the report, labelable dataset, and analyzer all observe the same rows.",
            "Action taxonomy is reported in two layers: exact normalized action labels and diagnosis-level action classes (hold, buy, sell, unknown).",
            "Raw directional rows and analyzer candidate rows come from the same effective input snapshot, but the current schema does not expose raw-row-to-candidate lineage, so analyzer stages are reported as candidate-row counts rather than exact raw-row survivals.",
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
    action_taxonomy = build_action_taxonomy_summary(normalized_raw_rows)
    directional_emission_funnel = build_directional_emission_funnel(
        normalized_raw_rows=normalized_raw_rows,
        labelable_dataset=labelable_dataset,
        edge_candidate_rows=edge_candidate_rows,
    )
    directional_emission_by_strategy = build_directional_emission_breakdown(
        normalized_raw_rows=normalized_raw_rows,
        group_fields=("strategy",),
    )
    directional_emission_by_strategy_symbol = build_directional_emission_breakdown(
        normalized_raw_rows=normalized_raw_rows,
        group_fields=("strategy", "symbol"),
    )
    directional_emission_by_strategy_symbol_bias_sign = (
        build_directional_emission_breakdown(
            normalized_raw_rows=normalized_raw_rows,
            group_fields=("strategy", "symbol", "bias_sign"),
        )
    )
    strategy_specificity = build_strategy_specificity_summary(
        directional_emission_by_strategy=directional_emission_by_strategy
    )
    analyzer_candidate_outcomes = build_analyzer_candidate_outcomes(
        selected_rows=selected_rows,
        diagnostic_rows=diagnostic_rows,
        empty_reason_summary=empty_reason_summary,
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
            "raw_input_rows": directional_emission_funnel["raw_input_rows"],
            "raw_rows_with_known_identity": directional_emission_funnel[
                "raw_rows_with_known_identity"
            ],
            "directional_bias_present_known_identity_rows": directional_emission_funnel[
                "directional_bias_present_known_identity_rows"
            ],
            "directional_buy_sell_emitted_rows": directional_emission_funnel[
                "directional_buy_sell_emitted_rows"
            ],
            "directional_buy_sell_emission_rate": directional_emission_funnel[
                "directional_buy_sell_emission_rate"
            ],
            "directional_emitted_labelable_rows": directional_emission_funnel[
                "directional_emitted_labelable_rows"
            ],
            "directional_emitted_rows_with_any_future_label": directional_emission_funnel[
                "directional_emitted_rows_with_any_future_label"
            ],
            "analyzer_diagnostic_row_count": directional_emission_funnel[
                "analyzer_diagnostic_row_count"
            ],
            "analyzer_selected_row_count": directional_emission_funnel[
                "analyzer_selected_row_count"
            ],
            "primary_collapse_stage": directional_emission_funnel[
                "primary_collapse_stage"
            ],
            "scalping_zero_emission": strategy_specificity["scalping_zero_emission"],
            "dominant_rejection_reason": empty_reason_summary.get(
                "dominant_rejection_reason"
            ),
        },
        "action_taxonomy": action_taxonomy,
        "directional_emission_funnel": directional_emission_funnel,
        "directional_emission_by_strategy": directional_emission_by_strategy,
        "directional_emission_by_strategy_symbol": directional_emission_by_strategy_symbol,
        "directional_emission_by_strategy_symbol_bias_sign": (
            directional_emission_by_strategy_symbol_bias_sign
        ),
        "strategy_specificity": strategy_specificity,
        "analyzer_candidate_outcomes": analyzer_candidate_outcomes,
        "selected_survivors": selected_rows,
        "diagnostic_rows": diagnostic_rows,
    }


def build_action_taxonomy_summary(
    normalized_raw_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    known_identity_rows = [row for row in normalized_raw_rows if _has_known_identity(row)]
    directional_rows = [row for row in known_identity_rows if _has_directional_bias(row)]

    known_exact_counter: Counter[str] = Counter()
    known_class_counter: Counter[str] = Counter()
    directional_exact_counter: Counter[str] = Counter()
    directional_class_counter: Counter[str] = Counter()

    for row in known_identity_rows:
        label = _action_label(row) or _MISSING_ACTION_LABEL
        known_exact_counter[label] += 1
        known_class_counter[_action_class(row)] += 1

    for row in directional_rows:
        label = _action_label(row) or _MISSING_ACTION_LABEL
        directional_exact_counter[label] += 1
        directional_class_counter[_action_class(row)] += 1

    return {
        "known_identity_row_count": len(known_identity_rows),
        "directional_row_count": len(directional_rows),
        "known_identity_exact_action_counts": dict(known_exact_counter),
        "known_identity_exact_action_count_rows": _action_label_counter_rows(
            known_exact_counter
        ),
        "known_identity_action_class_counts": dict(known_class_counter),
        "known_identity_action_class_count_rows": _action_class_counter_rows(
            known_class_counter
        ),
        "directional_exact_action_counts": dict(directional_exact_counter),
        "directional_exact_action_count_rows": _action_label_counter_rows(
            directional_exact_counter
        ),
        "directional_action_class_counts": dict(directional_class_counter),
        "directional_action_class_count_rows": _action_class_counter_rows(
            directional_class_counter
        ),
        "known_identity_exact_action_labels": sorted(known_exact_counter),
        "directional_exact_action_labels": sorted(directional_exact_counter),
    }


def build_directional_emission_funnel(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    labelable_dataset: Sequence[dict[str, Any]],
    edge_candidate_rows: dict[str, Any],
) -> dict[str, Any]:
    known_identity_rows = [row for row in normalized_raw_rows if _has_known_identity(row)]
    directional_rows = [row for row in known_identity_rows if _has_directional_bias(row)]
    buy_rows = [row for row in directional_rows if _action_class(row) == _ACTION_CLASS_BUY]
    sell_rows = [row for row in directional_rows if _action_class(row) == _ACTION_CLASS_SELL]
    hold_rows = [row for row in directional_rows if _action_class(row) == _ACTION_CLASS_HOLD]
    unknown_rows = [
        row for row in directional_rows if _action_class(row) == _ACTION_CLASS_UNKNOWN
    ]
    emitted_rows = buy_rows + sell_rows

    labelable_membership = _dataset_membership_counter(labelable_dataset)
    directional_labelable_rows = _rows_present_in_dataset(
        directional_rows,
        labelable_membership,
    )
    emitted_labelable_rows = _rows_present_in_dataset(
        emitted_rows,
        labelable_membership,
    )

    matching_sign_action_rows = 0
    opposite_sign_action_rows = 0
    for row in emitted_rows:
        bias_sign = _bias_sign(row)
        action_class = _action_class(row)
        if bias_sign == "bullish" and action_class == _ACTION_CLASS_BUY:
            matching_sign_action_rows += 1
        elif bias_sign == "bearish" and action_class == _ACTION_CLASS_SELL:
            matching_sign_action_rows += 1
        else:
            opposite_sign_action_rows += 1

    emitted_labeled_rows_by_horizon = {
        horizon: sum(
            1 for row in emitted_labelable_rows if has_future_fields_for_horizon(row, horizon)
        )
        for horizon in HORIZONS
    }
    directional_labeled_rows_by_horizon = {
        horizon: sum(
            1
            for row in directional_labelable_rows
            if has_future_fields_for_horizon(row, horizon)
        )
        for horizon in HORIZONS
    }

    stage_losses = {
        "directional_bias_to_action_emission_scarcity": max(
            len(directional_rows) - len(emitted_rows),
            0,
        ),
        "post_emission_labelability_scarcity": max(
            len(emitted_rows) - len(emitted_labelable_rows),
            0,
        ),
    }

    return {
        "raw_input_rows": len(normalized_raw_rows),
        "raw_rows_with_known_identity": len(known_identity_rows),
        "directional_bias_present_known_identity_rows": len(directional_rows),
        "directional_buy_emitted_rows": len(buy_rows),
        "directional_sell_emitted_rows": len(sell_rows),
        "directional_buy_sell_emitted_rows": len(emitted_rows),
        "directional_hold_rows": len(hold_rows),
        "directional_unknown_action_rows": len(unknown_rows),
        "directional_buy_sell_emission_rate": _safe_ratio(
            len(emitted_rows),
            len(directional_rows),
        ),
        "directional_hold_share": _safe_ratio(len(hold_rows), len(directional_rows)),
        "directional_unknown_action_share": _safe_ratio(
            len(unknown_rows),
            len(directional_rows),
        ),
        "directional_labelable_rows_any_action": len(directional_labelable_rows),
        "directional_labelable_rows_with_any_future_label": sum(
            1
            for row in directional_labelable_rows
            if any(has_future_fields_for_horizon(row, horizon) for horizon in HORIZONS)
        ),
        "directional_labeled_rows_by_horizon": directional_labeled_rows_by_horizon,
        "directional_emitted_labelable_rows": len(emitted_labelable_rows),
        "directional_emitted_labelable_share_of_directional": _safe_ratio(
            len(emitted_labelable_rows),
            len(directional_rows),
        ),
        "directional_emitted_labelable_share_of_emitted": _safe_ratio(
            len(emitted_labelable_rows),
            len(emitted_rows),
        ),
        "directional_emitted_rows_with_any_future_label": sum(
            1
            for row in emitted_labelable_rows
            if any(has_future_fields_for_horizon(row, horizon) for horizon in HORIZONS)
        ),
        "directional_emitted_labeled_rows_by_horizon": emitted_labeled_rows_by_horizon,
        "directional_matching_sign_action_rows": matching_sign_action_rows,
        "directional_opposite_sign_action_rows": opposite_sign_action_rows,
        "directional_opposite_sign_action_share_of_emitted": _safe_ratio(
            opposite_sign_action_rows,
            len(emitted_rows),
        ),
        "analyzer_diagnostic_row_count": int(
            edge_candidate_rows.get("diagnostic_row_count", 0) or 0
        ),
        "analyzer_selected_row_count": int(edge_candidate_rows.get("row_count", 0) or 0),
        "primary_collapse_stage": _primary_stage(stage_losses),
        "stage_losses": stage_losses,
        "analyzer_stage_lineage_note": (
            "Analyzer diagnostic/selected stages are candidate-row counts from the same "
            "effective input snapshot; the current schema does not expose raw-row-to-"
            "candidate lineage, so later stages are not exact raw-row survival counts."
        ),
    }


def build_directional_emission_breakdown(
    *,
    normalized_raw_rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    directional_rows = [
        row
        for row in normalized_raw_rows
        if _has_known_identity(row) and _has_directional_bias(row)
    ]
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}

    for row in directional_rows:
        group_values = [_group_value(row, field) for field in group_fields]
        if any(value is None for value in group_values):
            continue

        key = tuple(str(value) for value in group_values)
        if key not in grouped:
            entry = {
                "directional_total": 0,
                "hold_rows": 0,
                "buy_rows": 0,
                "sell_rows": 0,
                "unknown_action_rows": 0,
                "matching_sign_action_rows": 0,
                "opposite_sign_action_rows": 0,
                "exact_action_counts": Counter(),
            }
            for field, value in zip(group_fields, group_values, strict=True):
                entry[field] = value
            grouped[key] = entry

        entry = grouped[key]
        entry["directional_total"] += 1
        action_class = _action_class(row)
        exact_action = _action_label(row) or _MISSING_ACTION_LABEL
        _safe_counter(entry["exact_action_counts"])[exact_action] += 1

        if action_class == _ACTION_CLASS_HOLD:
            entry["hold_rows"] += 1
        elif action_class == _ACTION_CLASS_BUY:
            entry["buy_rows"] += 1
        elif action_class == _ACTION_CLASS_SELL:
            entry["sell_rows"] += 1
        else:
            entry["unknown_action_rows"] += 1

        bias_sign = _bias_sign(row)
        if action_class in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL}:
            if bias_sign == "bullish" and action_class == _ACTION_CLASS_BUY:
                entry["matching_sign_action_rows"] += 1
            elif bias_sign == "bearish" and action_class == _ACTION_CLASS_SELL:
                entry["matching_sign_action_rows"] += 1
            else:
                entry["opposite_sign_action_rows"] += 1

    rows: list[dict[str, Any]] = []
    for entry in grouped.values():
        directional_total = int(entry.get("directional_total", 0) or 0)
        emitted_buy_sell_rows = int(entry.get("buy_rows", 0) or 0) + int(
            entry.get("sell_rows", 0) or 0
        )
        exact_action_counts = _safe_counter(entry["exact_action_counts"])
        rows.append(
            {
                **{field: entry.get(field) for field in group_fields},
                "directional_total": directional_total,
                "hold_rows": int(entry.get("hold_rows", 0) or 0),
                "buy_rows": int(entry.get("buy_rows", 0) or 0),
                "sell_rows": int(entry.get("sell_rows", 0) or 0),
                "unknown_action_rows": int(entry.get("unknown_action_rows", 0) or 0),
                "emitted_buy_sell_rows": emitted_buy_sell_rows,
                "buy_sell_emission_rate": _safe_ratio(
                    emitted_buy_sell_rows,
                    directional_total,
                ),
                "hold_share": _safe_ratio(
                    int(entry.get("hold_rows", 0) or 0),
                    directional_total,
                ),
                "unknown_action_share": _safe_ratio(
                    int(entry.get("unknown_action_rows", 0) or 0),
                    directional_total,
                ),
                "matching_sign_action_rows": int(
                    entry.get("matching_sign_action_rows", 0) or 0
                ),
                "opposite_sign_action_rows": int(
                    entry.get("opposite_sign_action_rows", 0) or 0
                ),
                "exact_action_counts": dict(exact_action_counts),
                "exact_action_count_rows": _action_label_counter_rows(exact_action_counts),
                "zero_emission": emitted_buy_sell_rows == 0,
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("directional_total", 0) or 0),
            str(item.get("strategy") or ""),
            str(item.get("symbol") or ""),
            str(item.get("bias_sign") or ""),
        )
    )
    return rows


def build_strategy_specificity_summary(
    *,
    directional_emission_by_strategy: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    strategy_rows = [_safe_dict(row) for row in directional_emission_by_strategy]
    supported_rows = [
        row
        for row in strategy_rows
        if int(row.get("directional_total", 0) or 0) >= _MIN_STRATEGY_SUPPORT_ROWS
    ]
    zero_emission_strategies = [
        str(row.get("strategy"))
        for row in supported_rows
        if int(row.get("emitted_buy_sell_rows", 0) or 0) == 0
    ]
    nonzero_emission_strategies = [
        str(row.get("strategy"))
        for row in supported_rows
        if int(row.get("emitted_buy_sell_rows", 0) or 0) > 0
    ]

    max_row = None
    min_row = None
    if supported_rows:
        max_row = max(
            supported_rows,
            key=lambda item: (
                _to_float(item.get("buy_sell_emission_rate"), default=0.0) or 0.0,
                str(item.get("strategy") or ""),
            ),
        )
        min_row = min(
            supported_rows,
            key=lambda item: (
                _to_float(item.get("buy_sell_emission_rate"), default=0.0) or 0.0,
                str(item.get("strategy") or ""),
            ),
        )

    max_rate = _to_float(_safe_dict(max_row).get("buy_sell_emission_rate"), default=0.0)
    min_rate = _to_float(_safe_dict(min_row).get("buy_sell_emission_rate"), default=0.0)
    rate_gap = round((max_rate or 0.0) - (min_rate or 0.0), 6)

    scalping_row = next(
        (row for row in strategy_rows if row.get("strategy") == "scalping"),
        None,
    )
    scalping_directional_total = int(
        _safe_dict(scalping_row).get("directional_total", 0) or 0
    )
    scalping_emitted_rows = int(
        _safe_dict(scalping_row).get("emitted_buy_sell_rows", 0) or 0
    )
    scalping_zero_emission = (
        scalping_directional_total >= _MIN_STRATEGY_SUPPORT_ROWS
        and scalping_emitted_rows == 0
    )

    if len(supported_rows) < 2:
        classification = "insufficient_support"
    elif scalping_zero_emission and nonzero_emission_strategies:
        classification = "strategy_specific_zero_emission"
    elif rate_gap >= 0.05:
        classification = "strategy_specific_gap"
    else:
        classification = "mixed_or_uniform"

    return {
        "supported_strategy_count": len(supported_rows),
        "supported_strategies": [str(row.get("strategy")) for row in supported_rows],
        "zero_emission_strategies": zero_emission_strategies,
        "nonzero_emission_strategies": nonzero_emission_strategies,
        "max_emission_strategy": _emission_summary_row(max_row),
        "min_emission_strategy": _emission_summary_row(min_row),
        "emission_rate_gap": rate_gap,
        "classification": classification,
        "scalping_zero_emission": scalping_zero_emission,
        "scalping_directional_total": scalping_directional_total,
        "scalping_emitted_rows": scalping_emitted_rows,
    }


def build_analyzer_candidate_outcomes(
    *,
    selected_rows: Sequence[dict[str, Any]],
    diagnostic_rows: Sequence[dict[str, Any]],
    empty_reason_summary: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_rejection_reason_counts: Counter[str] = Counter()
    selected_strategy_horizon_counter: Counter[tuple[str, str]] = Counter()
    diagnostic_strategy_horizon_counter: Counter[tuple[str, str]] = Counter()
    diagnostic_weak_row_count = 0

    for row in selected_rows:
        strategy = _normalize_strategy(row.get("strategy"))
        horizon = _normalize_horizon(row.get("horizon"))
        if strategy is not None and horizon is not None:
            selected_strategy_horizon_counter[(strategy, horizon)] += 1

    for row in diagnostic_rows:
        row_dict = _safe_dict(row)
        reason = _normalize_text(row_dict.get("rejection_reason")) or "unknown"
        diagnostic_rejection_reason_counts[reason] += 1

        strategy = _normalize_strategy(row_dict.get("strategy"))
        horizon = _normalize_horizon(row_dict.get("horizon"))
        if strategy is not None and horizon is not None:
            diagnostic_strategy_horizon_counter[(strategy, horizon)] += 1

        if reason == "candidate_strength_weak" or row_dict.get("candidate_strength") == "weak":
            diagnostic_weak_row_count += 1

    return {
        "selected_row_count": len(selected_rows),
        "diagnostic_row_count": len(diagnostic_rows),
        "diagnostic_weak_row_count": diagnostic_weak_row_count,
        "diagnostic_rejection_reason_counts": dict(diagnostic_rejection_reason_counts),
        "dominant_rejection_reason": empty_reason_summary.get("dominant_rejection_reason"),
        "selected_rows_by_strategy_horizon": _strategy_horizon_counter_rows(
            selected_strategy_horizon_counter
        ),
        "diagnostic_rows_by_strategy_horizon": _strategy_horizon_counter_rows(
            diagnostic_strategy_horizon_counter
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
        "directional_bias_action_emission_scarcity": _assess_directional_emission_scarcity(
            widest
        ),
        "scalping_zero_emission": _assess_scalping_zero_emission(widest),
        "strategy_specificity": _assess_strategy_specificity(widest),
        "directional_sign_mapping_inversion": _assess_sign_mapping_inversion(widest),
        "downstream_quality_weakness": _assess_downstream_quality_weakness(widest),
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


def _assess_directional_emission_scarcity(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("directional_emission_funnel"))
    directional_rows = int(
        funnel.get("directional_bias_present_known_identity_rows", 0) or 0
    )
    emitted_rows = int(funnel.get("directional_buy_sell_emitted_rows", 0) or 0)
    hold_rows = int(funnel.get("directional_hold_rows", 0) or 0)
    unknown_rows = int(funnel.get("directional_unknown_action_rows", 0) or 0)
    emission_rate = _to_float(
        funnel.get("directional_buy_sell_emission_rate"),
        default=0.0,
    ) or 0.0
    hold_share = _to_float(funnel.get("directional_hold_share"), default=0.0) or 0.0

    evidence = [
        (
            "Directional-bias-present known-identity rows: "
            f"{directional_rows}; emitted buy/sell rows: {emitted_rows} "
            f"({emission_rate:.2%})."
        ),
        (
            f"Directional hold rows: {hold_rows} ({hold_share:.2%}); "
            f"directional unknown-action rows: {unknown_rows}."
        ),
        (
            "Primary raw funnel collapse stage at widest configuration: "
            f"{funnel.get('primary_collapse_stage', 'unknown')}."
        ),
    ]

    if directional_rows < _MIN_PRIMARY_SUPPORT_ROWS:
        status = "insufficient_support"
    elif (
        funnel.get("primary_collapse_stage") == "directional_bias_to_action_emission_scarcity"
        and emission_rate <= 0.10
        and hold_share >= 0.70
    ):
        status = "primary"
    elif emission_rate <= 0.15:
        status = "contributing"
    elif emission_rate <= 0.25:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_scalping_zero_emission(widest: dict[str, Any]) -> dict[str, Any]:
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))
    directional_rows = int(strategy_specificity.get("scalping_directional_total", 0) or 0)
    emitted_rows = int(strategy_specificity.get("scalping_emitted_rows", 0) or 0)
    evidence = [
        (
            "Scalping directional rows at widest configuration: "
            f"{directional_rows}; emitted buy/sell rows: {emitted_rows}."
        ),
        (
            "Supported zero-emission strategies: "
            f"{strategy_specificity.get('zero_emission_strategies', [])}."
        ),
    ]

    if directional_rows < _MIN_STRATEGY_SUPPORT_ROWS:
        status = "insufficient_support"
    elif emitted_rows == 0:
        status = "present"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_strategy_specificity(widest: dict[str, Any]) -> dict[str, Any]:
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))
    max_row = _safe_dict(strategy_specificity.get("max_emission_strategy"))
    min_row = _safe_dict(strategy_specificity.get("min_emission_strategy"))
    evidence = [
        (
            f"Supported strategy count: {int(strategy_specificity.get('supported_strategy_count', 0) or 0)}; "
            f"emission-rate gap: {_format_ratio(strategy_specificity.get('emission_rate_gap'))}."
        ),
        (
            "Lowest supported strategy: "
            f"{min_row.get('strategy', 'n/a')} "
            f"({_format_ratio(min_row.get('buy_sell_emission_rate'))}); "
            "highest supported strategy: "
            f"{max_row.get('strategy', 'n/a')} "
            f"({_format_ratio(max_row.get('buy_sell_emission_rate'))})."
        ),
    ]

    classification = strategy_specificity.get("classification")
    if classification == "insufficient_support":
        status = "insufficient_support"
    elif classification == "strategy_specific_zero_emission":
        status = "strategy_specific"
    elif classification == "strategy_specific_gap":
        status = "present"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_sign_mapping_inversion(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("directional_emission_funnel"))
    emitted_rows = int(funnel.get("directional_buy_sell_emitted_rows", 0) or 0)
    opposite_rows = int(funnel.get("directional_opposite_sign_action_rows", 0) or 0)
    matching_rows = int(funnel.get("directional_matching_sign_action_rows", 0) or 0)
    evidence = [
        (
            f"Directional emitted rows: {emitted_rows}; sign-matching emitted rows: "
            f"{matching_rows}; opposite-sign emitted rows: {opposite_rows}."
        )
    ]

    if emitted_rows < _MIN_SIGN_SUPPORT_ROWS:
        status = "insufficient_support"
    elif opposite_rows > 0:
        status = "present"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_downstream_quality_weakness(widest: dict[str, Any]) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("directional_emission_funnel"))
    outcomes = _safe_dict(widest.get("analyzer_candidate_outcomes"))
    labelable_rows = int(funnel.get("directional_emitted_labelable_rows", 0) or 0)
    diagnostic_rows = int(outcomes.get("diagnostic_row_count", 0) or 0)
    diagnostic_weak_rows = int(outcomes.get("diagnostic_weak_row_count", 0) or 0)
    selected_rows = int(outcomes.get("selected_row_count", 0) or 0)
    dominant_reason = outcomes.get("dominant_rejection_reason")

    evidence = [
        (
            f"Directional emitted labelable rows: {labelable_rows}; "
            f"diagnostic candidate rows: {diagnostic_rows}; "
            f"diagnostic weak rows: {diagnostic_weak_rows}; "
            f"selected rows: {selected_rows}."
        ),
        (
            f"Dominant analyzer rejection reason: {dominant_reason or 'none'}."
        ),
    ]

    if (
        diagnostic_weak_rows > 0
        and labelable_rows >= _MIN_PRIMARY_SUPPORT_ROWS
        and selected_rows <= 1
        and (_to_float(funnel.get("directional_buy_sell_emission_rate"), default=0.0) or 0.0)
        >= 0.20
    ):
        status = "primary"
    elif diagnostic_weak_rows > 0 or dominant_reason == "candidate_strength_weak":
        status = "contributing"
    elif labelable_rows < _MIN_DIRECTIONAL_SUPPORT_ROWS:
        status = "insufficient_support"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _primary_bottleneck_label(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    if _safe_dict(factors.get("directional_bias_action_emission_scarcity")).get(
        "status"
    ) == "primary":
        return "directional_bias_action_emission_scarcity"
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "primary":
        return "downstream_quality_weakness"

    primary_stage = _safe_dict(widest.get("directional_emission_funnel")).get(
        "primary_collapse_stage"
    )
    if (
        primary_stage == "directional_bias_to_action_emission_scarcity"
        and _safe_dict(factors.get("directional_bias_action_emission_scarcity")).get(
            "status"
        )
        in {"contributing", "limited"}
    ):
        return "directional_bias_action_emission_scarcity"
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") == "contributing":
        return "downstream_quality_weakness"

    return "mixed_or_inconclusive"


def _overall_assessment_label(
    *,
    primary_bottleneck: str,
    factors: dict[str, Any],
) -> str:
    if primary_bottleneck == "directional_bias_action_emission_scarcity" and _safe_dict(
        factors.get("directional_bias_action_emission_scarcity")
    ).get("status") == "primary":
        return "directional_bias_action_emission_scarcity_primary"
    if primary_bottleneck == "downstream_quality_weakness" and _safe_dict(
        factors.get("downstream_quality_weakness")
    ).get("status") == "primary":
        return "downstream_quality_weakness_primary"

    statuses = {
        name: _safe_dict(payload).get("status") for name, payload in factors.items()
    }
    if any(
        status in {"contributing", "limited", "present", "strategy_specific"}
        for status in statuses.values()
    ):
        return "mixed_contributing_factors"
    if any(status == "insufficient_support" for status in statuses.values()):
        return "insufficient_support"
    return "not_supported_or_inconclusive"


def _build_confirmed_observations(widest: dict[str, Any]) -> list[str]:
    funnel = _safe_dict(widest.get("directional_emission_funnel"))
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))
    outcomes = _safe_dict(widest.get("analyzer_candidate_outcomes"))

    return [
        (
            "Directional-bias-present rows: "
            f"{int(funnel.get('directional_bias_present_known_identity_rows', 0) or 0)}; "
            f"buy/sell emitted rows: {int(funnel.get('directional_buy_sell_emitted_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('directional_buy_sell_emission_rate'))})."
        ),
        (
            "Directional hold rows: "
            f"{int(funnel.get('directional_hold_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('directional_hold_share'))}); "
            f"directional unknown-action rows: {int(funnel.get('directional_unknown_action_rows', 0) or 0)}."
        ),
        (
            "Directional emitted labelable rows: "
            f"{int(funnel.get('directional_emitted_labelable_rows', 0) or 0)}; "
            f"diagnostic candidate rows: {int(outcomes.get('diagnostic_row_count', 0) or 0)}; "
            f"selected candidate rows: {int(outcomes.get('selected_row_count', 0) or 0)}."
        ),
        (
            "Scalping directional rows: "
            f"{int(strategy_specificity.get('scalping_directional_total', 0) or 0)}; "
            f"scalping emitted buy/sell rows: {int(strategy_specificity.get('scalping_emitted_rows', 0) or 0)}."
        ),
        (
            "Opposite-sign directional emissions: "
            f"{int(funnel.get('directional_opposite_sign_action_rows', 0) or 0)} of "
            f"{int(funnel.get('directional_buy_sell_emitted_rows', 0) or 0)} emitted directional rows."
        ),
    ]


def _build_evidence_backed_inferences(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
    primary_bottleneck: str,
) -> list[str]:
    inferences: list[str] = []
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))
    min_row = _safe_dict(strategy_specificity.get("min_emission_strategy"))
    max_row = _safe_dict(strategy_specificity.get("max_emission_strategy"))

    if primary_bottleneck == "directional_bias_action_emission_scarcity":
        inferences.append(
            "The first confirmed collapse happens before labelability: most directional-bias rows remain hold instead of being promoted into buy/sell emission."
        )
    elif primary_bottleneck == "downstream_quality_weakness":
        inferences.append(
            "The widest configuration restores enough emitted labelable support that downstream quality rejection becomes the dominant remaining loss."
        )

    if _safe_dict(factors.get("strategy_specificity")).get("status") in {
        "strategy_specific",
        "present",
    }:
        inferences.append(
            "The emission bottleneck does not look uniform: "
            f"{min_row.get('strategy', 'n/a')} emits materially less often than "
            f"{max_row.get('strategy', 'n/a')}."
        )

    if _safe_dict(factors.get("scalping_zero_emission")).get("status") == "present":
        inferences.append(
            "Scalping appears to be a true zero-emission path in the inspected dataset rather than just a lower-emission variant."
        )

    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") in {
        "primary",
        "contributing",
    } and primary_bottleneck != "downstream_quality_weakness":
        inferences.append(
            "Downstream quality weakness remains a real secondary bottleneck after the few emitted, labelable rows reach the analyzer."
        )

    if _safe_dict(factors.get("directional_sign_mapping_inversion")).get("status") == "not_supported":
        inferences.append(
            "The observed emission path is sign-consistent: bullish bias maps only to buy-like emission and bearish bias maps only to sell-like emission when emission happens."
        )

    return inferences


def _build_unresolved_uncertainties(widest: dict[str, Any]) -> list[str]:
    action_taxonomy = _safe_dict(widest.get("action_taxonomy"))

    uncertainties = [
        "The current schema does not expose a per-row reason code for why a directional bias stayed hold, so this report cannot prove whether the bottleneck reflects intended conservatism, a dead confirmation path, or another strategy-composition constraint.",
        "The report can count raw directional rows and analyzer candidate rows from the same effective input snapshot, but it cannot trace each raw emitted row into a specific candidate-row outcome because raw-row-to-candidate lineage is not exposed.",
        "The current data shows where scarcity occurs, but it does not isolate whether the failed promotion happens inside strategy logic, decision composition, or another upstream confirmation layer.",
    ]
    exact_labels = _safe_list(action_taxonomy.get("directional_exact_action_labels"))
    if exact_labels:
        uncertainties.append(
            f"Exact directional action labels were observed as: {exact_labels}; the report surfaces them, but the schema alone does not explain which labels were intentionally abstentive versus merely intermediate."
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
    funnel = _safe_dict(widest.get("directional_emission_funnel"))
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))

    parts: list[str] = []
    if primary_bottleneck == "directional_bias_action_emission_scarcity":
        parts.append(
            "The primary confirmed bottleneck is directional-bias-to-action emission scarcity."
        )
    elif primary_bottleneck == "downstream_quality_weakness":
        parts.append(
            "The primary confirmed bottleneck is downstream quality weakness after directional emission becomes available."
        )
    else:
        parts.append(
            "The evidence is mixed and no single stage is confirmed as the only bottleneck."
        )

    parts.append(
        f"At {widest_config}, directional-bias-present rows were "
        f"{int(funnel.get('directional_bias_present_known_identity_rows', 0) or 0)} and "
        f"buy/sell emitted rows were {int(funnel.get('directional_buy_sell_emitted_rows', 0) or 0)} "
        f"({_format_ratio(funnel.get('directional_buy_sell_emission_rate'))})."
    )
    parts.append(
        f"Directional emitted labelable rows were {int(funnel.get('directional_emitted_labelable_rows', 0) or 0)}, "
        f"while analyzer diagnostic/selected candidate rows were "
        f"{int(funnel.get('analyzer_diagnostic_row_count', 0) or 0)}/"
        f"{int(funnel.get('analyzer_selected_row_count', 0) or 0)}."
    )

    if _safe_dict(factors.get("strategy_specificity")).get("status") in {
        "strategy_specific",
        "present",
    }:
        parts.append(
            "The bottleneck is materially strategy-specific rather than uniform, with supported zero-emission strategies at "
            f"{strategy_specificity.get('zero_emission_strategies', [])}."
        )
    if _safe_dict(factors.get("directional_sign_mapping_inversion")).get("status") == "not_supported":
        parts.append(
            "Observed directional emission is sign-consistent, so directional inversion is not supported by the inspected snapshot."
        )
    if _safe_dict(factors.get("downstream_quality_weakness")).get("status") in {
        "primary",
        "contributing",
    }:
        parts.append(
            "Downstream quality weakness remains a secondary pressure after the sparse emitted rows reach analyzer selection."
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
            "- directional_bias_present_known_identity_rows: "
            f"{headline.get('directional_bias_present_known_identity_rows', 0)}"
        )
        lines.append(
            "- directional_buy_sell_emitted_rows: "
            f"{headline.get('directional_buy_sell_emitted_rows', 0)} "
            f"({_format_ratio(headline.get('directional_buy_sell_emission_rate'))})"
        )
        lines.append(
            "- directional_emitted_labelable_rows: "
            f"{headline.get('directional_emitted_labelable_rows', 0)}"
        )
        lines.append(
            "- directional_emitted_rows_with_any_future_label: "
            f"{headline.get('directional_emitted_rows_with_any_future_label', 0)}"
        )
        lines.append(
            "- analyzer_diagnostic_row_count: "
            f"{headline.get('analyzer_diagnostic_row_count', 0)}"
        )
        lines.append(
            "- analyzer_selected_row_count: "
            f"{headline.get('analyzer_selected_row_count', 0)}"
        )
        lines.append(
            "- primary_collapse_stage: "
            f"{headline.get('primary_collapse_stage', 'n/a')}"
        )
        lines.append(
            f"- scalping_zero_emission: {headline.get('scalping_zero_emission', False)}"
        )
        lines.append("")

    lines.append("## Exact Action Taxonomy")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        taxonomy = _safe_dict(_safe_dict(summary).get("action_taxonomy"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- known_identity_exact_action_counts: "
            f"{_format_action_label_rows(taxonomy.get('known_identity_exact_action_count_rows'))}"
        )
        lines.append(
            "- known_identity_action_class_counts: "
            f"{_format_action_class_rows(taxonomy.get('known_identity_action_class_count_rows'))}"
        )
        lines.append(
            "- directional_exact_action_counts: "
            f"{_format_action_label_rows(taxonomy.get('directional_exact_action_count_rows'))}"
        )
        lines.append(
            "- directional_action_class_counts: "
            f"{_format_action_class_rows(taxonomy.get('directional_action_class_count_rows'))}"
        )
        lines.append("")

    lines.append("## Directional Bias Action Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("directional_emission_funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- directional_bias_present_known_identity_rows: "
            f"{funnel.get('directional_bias_present_known_identity_rows', 0)}"
        )
        lines.append(
            "- directional_buy_emitted_rows: "
            f"{funnel.get('directional_buy_emitted_rows', 0)}"
        )
        lines.append(
            "- directional_sell_emitted_rows: "
            f"{funnel.get('directional_sell_emitted_rows', 0)}"
        )
        lines.append(
            "- directional_buy_sell_emitted_rows: "
            f"{funnel.get('directional_buy_sell_emitted_rows', 0)} "
            f"({_format_ratio(funnel.get('directional_buy_sell_emission_rate'))})"
        )
        lines.append(
            "- directional_hold_rows: "
            f"{funnel.get('directional_hold_rows', 0)} "
            f"({_format_ratio(funnel.get('directional_hold_share'))})"
        )
        lines.append(
            "- directional_unknown_action_rows: "
            f"{funnel.get('directional_unknown_action_rows', 0)} "
            f"({_format_ratio(funnel.get('directional_unknown_action_share'))})"
        )
        lines.append(
            "- directional_emitted_labelable_rows: "
            f"{funnel.get('directional_emitted_labelable_rows', 0)} "
            f"({_format_ratio(funnel.get('directional_emitted_labelable_share_of_emitted'))} of emitted)"
        )
        lines.append(
            "- directional_emitted_rows_with_any_future_label: "
            f"{funnel.get('directional_emitted_rows_with_any_future_label', 0)}"
        )
        lines.append(
            "- directional_emitted_labeled_rows_by_horizon: "
            f"{_format_labeled_counts(funnel.get('directional_emitted_labeled_rows_by_horizon'))}"
        )
        lines.append(
            "- analyzer_diagnostic_row_count: "
            f"{funnel.get('analyzer_diagnostic_row_count', 0)}"
        )
        lines.append(
            "- analyzer_selected_row_count: "
            f"{funnel.get('analyzer_selected_row_count', 0)}"
        )
        lines.append(
            "- directional_matching_sign_action_rows: "
            f"{funnel.get('directional_matching_sign_action_rows', 0)}"
        )
        lines.append(
            "- directional_opposite_sign_action_rows: "
            f"{funnel.get('directional_opposite_sign_action_rows', 0)}"
        )
        lines.append(
            "- primary_collapse_stage: "
            f"{funnel.get('primary_collapse_stage', 'n/a')}"
        )
        lines.append(
            f"- analyzer_stage_lineage_note: {funnel.get('analyzer_stage_lineage_note', 'n/a')}"
        )
        lines.append("")

    for key, title in (
        ("directional_emission_by_strategy", "Directional Emission By Strategy"),
        (
            "directional_emission_by_strategy_symbol",
            "Directional Emission By Strategy and Symbol",
        ),
        (
            "directional_emission_by_strategy_symbol_bias_sign",
            "Directional Emission By Strategy, Symbol, and Bias Sign",
        ),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for summary in _safe_list(report.get("configuration_summaries")):
            config = _safe_dict(_safe_dict(summary).get("configuration"))
            rows = _safe_list(_safe_dict(summary).get(key))
            lines.append(f"### {config.get('display_name', 'n/a')}")
            if not rows:
                lines.append("No directional emission rows available.")
                lines.append("")
                continue
            for row in rows[:20]:
                item = _safe_dict(row)
                label_parts = [
                    str(item.get("strategy"))
                    if item.get("strategy") is not None
                    else None,
                    str(item.get("symbol"))
                    if item.get("symbol") is not None
                    else None,
                    str(item.get("bias_sign"))
                    if item.get("bias_sign") is not None
                    else None,
                ]
                label = " / ".join(part for part in label_parts if part)
                lines.append(
                    "- "
                    f"{label}: directional_total={item.get('directional_total', 0)}, "
                    f"hold={item.get('hold_rows', 0)}, "
                    f"buy={item.get('buy_rows', 0)}, "
                    f"sell={item.get('sell_rows', 0)}, "
                    f"unknown_action={item.get('unknown_action_rows', 0)}, "
                    f"emission={_format_ratio(item.get('buy_sell_emission_rate'))}, "
                    f"exact_actions={_format_action_label_rows(item.get('exact_action_count_rows'))}"
                )
            lines.append("")

    lines.append("## Strategy Specificity")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        specificity = _safe_dict(_safe_dict(summary).get("strategy_specificity"))
        max_row = _safe_dict(specificity.get("max_emission_strategy"))
        min_row = _safe_dict(specificity.get("min_emission_strategy"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            f"- classification: {specificity.get('classification', 'n/a')}"
        )
        lines.append(
            f"- supported_strategies: {specificity.get('supported_strategies', [])}"
        )
        lines.append(
            f"- zero_emission_strategies: {specificity.get('zero_emission_strategies', [])}"
        )
        lines.append(
            "- max_emission_strategy: "
            f"{max_row.get('strategy', 'n/a')} "
            f"({_format_ratio(max_row.get('buy_sell_emission_rate'))})"
        )
        lines.append(
            "- min_emission_strategy: "
            f"{min_row.get('strategy', 'n/a')} "
            f"({_format_ratio(min_row.get('buy_sell_emission_rate'))})"
        )
        lines.append(
            "- emission_rate_gap: "
            f"{_format_ratio(specificity.get('emission_rate_gap'))}"
        )
        lines.append("")

    lines.append("## Analyzer Candidate Outcomes")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        outcomes = _safe_dict(_safe_dict(summary).get("analyzer_candidate_outcomes"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(f"- selected_row_count: {outcomes.get('selected_row_count', 0)}")
        lines.append(f"- diagnostic_row_count: {outcomes.get('diagnostic_row_count', 0)}")
        lines.append(
            f"- diagnostic_weak_row_count: {outcomes.get('diagnostic_weak_row_count', 0)}"
        )
        lines.append(
            "- diagnostic_rejection_reason_counts: "
            f"{outcomes.get('diagnostic_rejection_reason_counts', {})}"
        )
        lines.append(
            "- selected_rows_by_strategy_horizon: "
            f"{_format_strategy_horizon_rows(outcomes.get('selected_rows_by_strategy_horizon'))}"
        )
        lines.append(
            "- diagnostic_rows_by_strategy_horizon: "
            f"{_format_strategy_horizon_rows(outcomes.get('diagnostic_rows_by_strategy_horizon'))}"
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


def _dataset_membership_counter(rows: Sequence[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        if isinstance(row, dict):
            counter[_row_fingerprint(row)] += 1
    return counter


def _rows_present_in_dataset(
    rows: Sequence[dict[str, Any]],
    membership_counter: Counter[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    remaining = Counter(membership_counter)
    matched: list[dict[str, Any]] = []
    for row in rows:
        fingerprint = _row_fingerprint(row)
        if remaining[fingerprint] <= 0:
            continue
        remaining[fingerprint] -= 1
        matched.append(row)
    return matched


def _row_fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _datetime_key(row.get("logged_at")),
        _normalize_symbol(row.get("symbol")),
        _normalize_strategy(row.get("selected_strategy") or row.get("strategy")),
        _normalize_bias(row.get("bias")),
        _action_label(row),
        _to_float(row.get("entry_price")),
        _normalize_text(row.get("future_label_15m")),
        _to_float(row.get("future_return_15m")),
        _normalize_text(row.get("future_label_1h")),
        _to_float(row.get("future_return_1h")),
        _normalize_text(row.get("future_label_4h")),
        _to_float(row.get("future_return_4h")),
    )


def _datetime_key(value: Any) -> str | None:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return None
    return _normalize_text(value)


def _action_label_counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "action_label": label,
            "count": count,
            "share": _safe_ratio(count, total),
            "action_class": _action_class_from_label(
                label if label != _MISSING_ACTION_LABEL else None
            ),
        }
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _action_class_counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "action_class": label,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for label, count in sorted(
            counter.items(),
            key=lambda item: (_ACTION_CLASS_ORDER.get(item[0], 99), -item[1], item[0]),
        )
    ]


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


def _emission_summary_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    row_dict = _safe_dict(row)
    if not row_dict:
        return None
    return {
        "strategy": row_dict.get("strategy"),
        "directional_total": int(row_dict.get("directional_total", 0) or 0),
        "emitted_buy_sell_rows": int(row_dict.get("emitted_buy_sell_rows", 0) or 0),
        "buy_sell_emission_rate": _to_float(
            row_dict.get("buy_sell_emission_rate"),
            default=0.0,
        ),
    }


def _format_labeled_counts(value: Any) -> str:
    counts = _safe_dict(value)
    return ", ".join(f"{horizon}={counts.get(horizon, 0)}" for horizon in HORIZONS)


def _format_ratio(value: Any) -> str:
    ratio = _to_float(value)
    if ratio is None:
        return "n/a"
    return f"{ratio:.2%}"


def _format_action_label_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get('action_label', 'n/a')}={_safe_dict(row).get('count', 0)}"
        for row in rows
    )


def _format_action_class_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get('action_class', 'n/a')}={_safe_dict(row).get('count', 0)}"
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


def _group_value(row: dict[str, Any], field: str) -> str | None:
    if field == "strategy":
        return _normalize_strategy(row.get("selected_strategy") or row.get("strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    if field == "bias_sign":
        return _bias_sign(row)
    raise ValueError(f"Unsupported group field: {field}")


def _primary_stage(stage_losses: dict[str, int]) -> str:
    candidates = [(name, max(int(count or 0), 0)) for name, count in stage_losses.items()]
    top_name, top_count = min(
        candidates,
        key=lambda item: (-item[1], _PRIMARY_STAGE_ORDER.get(item[0], 99)),
    )
    if top_count <= 0:
        return "no_raw_funnel_collapse_detected"
    return top_name


def _has_known_identity(row: dict[str, Any]) -> bool:
    return (
        _normalize_symbol(row.get("symbol")) is not None
        and _normalize_strategy(row.get("selected_strategy") or row.get("strategy"))
        is not None
    )


def _has_directional_bias(row: dict[str, Any]) -> bool:
    return _bias_sign(row) is not None


def _bias_sign(row: dict[str, Any]) -> str | None:
    bias = _normalize_bias(row.get("bias"))
    if bias is None or bias in _NON_DIRECTIONAL_BIAS_VALUES:
        return None
    if bias in _BULLISH_BIAS_VALUES:
        return "bullish"
    if bias in _BEARISH_BIAS_VALUES:
        return "bearish"
    return None


def _action_label(row: dict[str, Any]) -> str | None:
    return _normalize_action(
        row.get("execution_action")
        or row.get("execution_signal")
    )


def _action_class(row: dict[str, Any]) -> str:
    return _action_class_from_label(_action_label(row))


def _action_class_from_label(value: str | None) -> str:
    if value is None:
        return _ACTION_CLASS_UNKNOWN
    if value in _HOLD_LIKE_ACTION_VALUES:
        return _ACTION_CLASS_HOLD
    if value in _BUY_LIKE_ACTION_VALUES:
        return _ACTION_CLASS_BUY
    if value in _SELL_LIKE_ACTION_VALUES:
        return _ACTION_CLASS_SELL
    return _ACTION_CLASS_UNKNOWN


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _horizon_sort_key(value: str) -> int:
    if value in HORIZONS:
        return HORIZONS.index(value)
    return len(HORIZONS)


def _safe_counter(value: Any) -> Counter[str]:
    if isinstance(value, Counter):
        return value
    return Counter()


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
    "build_action_taxonomy_summary",
    "build_analyzer_candidate_outcomes",
    "build_configuration_summary",
    "build_directional_emission_breakdown",
    "build_directional_emission_funnel",
    "build_final_assessment",
    "build_report",
    "build_strategy_specificity_summary",
    "extract_diagnostic_rows",
    "extract_selected_rows",
    "main",
    "parse_args",
    "parse_configuration_values",
    "render_markdown",
    "run_directional_bias_action_emission_diagnosis_report",
    "write_report_files",
]
