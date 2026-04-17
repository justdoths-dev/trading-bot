from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.research.strategy_lab.dataset_builder import (
    load_jsonl_records_with_metadata,
    normalize_record,
)

REPORT_TYPE = "directional_bias_promotion_path_diagnosis_report"
REPORT_TITLE = "Directional Bias Promotion Path Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_EFFECTIVE_INPUT_FILENAME = "_effective_directional_bias_promotion_path_input.jsonl"
_MISSING_LABEL = "(missing)"

_ACTION_CLASS_HOLD = "hold"
_ACTION_CLASS_BUY = "buy"
_ACTION_CLASS_SELL = "sell"
_ACTION_CLASS_UNKNOWN = "unknown"

_HOLD_LIKE_VALUES = {
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
_BUY_LIKE_VALUES = {"buy", "long"}
_SELL_LIKE_VALUES = {"sell", "short"}

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

_CATEGORY_NO_ACTIONABLE = "no_actionable_upstream_proposal_visible"
_CATEGORY_LATE_COLLAPSE = "actionable_upstream_proposal_later_collapses"
_CATEGORY_UNOBSERVABLE = "intermediate_trace_unobservable"

_DETAILED_STAGE_SELECTED_NOT_ACTIONABLE = "selected_strategy_proposal_not_actionable"
_DETAILED_STAGE_RULE_NOT_ACTIONABLE = "rule_signal_not_actionable"
_DETAILED_STAGE_DECISION_COLLAPSE = (
    "decision_collapse_after_actionable_strategy_proposal"
)
_DETAILED_STAGE_RISK_BLOCKED = "risk_gate_blocked_after_actionable_rule_signal"
_DETAILED_STAGE_EXECUTION_COLLAPSE = (
    "execution_layer_hold_after_allowed_rule_signal"
)
_DETAILED_STAGE_RULE_UNOBSERVABLE = (
    "rule_signal_unobservable_after_actionable_strategy_proposal"
)
_DETAILED_STAGE_RISK_UNOBSERVABLE = (
    "risk_gate_unobservable_after_actionable_rule_signal"
)
_DETAILED_STAGE_PROMOTION_UNOBSERVABLE = "promotion_path_unobservable"

_DETAILED_STAGE_ORDER = {
    _DETAILED_STAGE_SELECTED_NOT_ACTIONABLE: 0,
    _DETAILED_STAGE_RULE_NOT_ACTIONABLE: 1,
    _DETAILED_STAGE_DECISION_COLLAPSE: 2,
    _DETAILED_STAGE_RISK_BLOCKED: 3,
    _DETAILED_STAGE_EXECUTION_COLLAPSE: 4,
    _DETAILED_STAGE_RULE_UNOBSERVABLE: 5,
    _DETAILED_STAGE_RISK_UNOBSERVABLE: 6,
    _DETAILED_STAGE_PROMOTION_UNOBSERVABLE: 7,
}
_CATEGORY_ORDER = {
    _CATEGORY_NO_ACTIONABLE: 0,
    _CATEGORY_LATE_COLLAPSE: 1,
    _CATEGORY_UNOBSERVABLE: 2,
}
_DETAILED_STAGE_TO_CATEGORY = {
    _DETAILED_STAGE_SELECTED_NOT_ACTIONABLE: _CATEGORY_NO_ACTIONABLE,
    _DETAILED_STAGE_RULE_NOT_ACTIONABLE: _CATEGORY_NO_ACTIONABLE,
    _DETAILED_STAGE_DECISION_COLLAPSE: _CATEGORY_LATE_COLLAPSE,
    _DETAILED_STAGE_RISK_BLOCKED: _CATEGORY_LATE_COLLAPSE,
    _DETAILED_STAGE_EXECUTION_COLLAPSE: _CATEGORY_LATE_COLLAPSE,
    _DETAILED_STAGE_RULE_UNOBSERVABLE: _CATEGORY_UNOBSERVABLE,
    _DETAILED_STAGE_RISK_UNOBSERVABLE: _CATEGORY_UNOBSERVABLE,
    _DETAILED_STAGE_PROMOTION_UNOBSERVABLE: _CATEGORY_UNOBSERVABLE,
}

_MIN_PRIMARY_SUPPORT_ROWS = 30
_MIN_DIRECTIONAL_SUPPORT_ROWS = 10
_MIN_STRATEGY_SPECIFIC_SUPPORT_ROWS = 10


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
            "Build a diagnosis-only directional-bias promotion-path report across "
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

    result = run_directional_bias_promotion_path_diagnosis_report(
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
                "dominant_observed_hold_path_category": final_assessment.get(
                    "dominant_observed_hold_path_category"
                ),
                "primary_hold_path_category": final_assessment.get(
                    "primary_hold_path_category"
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

    return parsed or list(DEFAULT_CONFIGURATIONS)


def run_directional_bias_promotion_path_diagnosis_report(
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
        run_output_dir = resolved_output / f"_{REPORT_TYPE}" / configuration.slug
        (
            effective_input_path,
            raw_records,
            source_metadata,
        ) = _materialize_configuration_input(
            input_path=resolved_input,
            run_output_dir=run_output_dir,
            latest_window_hours=configuration.latest_window_hours,
            latest_max_rows=configuration.latest_max_rows,
        )
        configuration_summaries.append(
            build_configuration_summary(
                configuration=configuration,
                input_path=resolved_input,
                effective_input_path=effective_input_path,
                run_output_dir=run_output_dir,
                raw_records=raw_records,
                source_metadata=source_metadata,
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
            "This report is diagnosis-only and reuses one materialized effective input snapshot per configuration so every raw-stage count comes from the same observed rows.",
            "Final emitted action uses execution-layer fields only: execution.action first, then execution.signal if action is absent. rule_engine.signal is upstream trace only and never reclassifies final emission.",
            "Earliest hold-collapse attribution uses only persisted stages that are directly observable in the current schema: selected_strategy_result signal, rule_engine signal, risk execution_allowed, and execution-layer action/signal.",
            "The raw schema also carries timeframe_summary context/setup/trigger layers, but this report does not over-interpret them as a persisted proposal stage because they do not expose a stable execution-facing proposal label.",
        ],
    }


def _materialize_configuration_input(
    *,
    input_path: Path,
    run_output_dir: Path,
    latest_window_hours: int,
    latest_max_rows: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    raw_records, source_metadata = load_jsonl_records_with_metadata(
        path=input_path,
        max_age_hours=latest_window_hours,
        max_rows=latest_max_rows,
    )

    run_output_dir.mkdir(parents=True, exist_ok=True)
    effective_input_path = run_output_dir / _EFFECTIVE_INPUT_FILENAME
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
    run_output_dir: Path,
    raw_records: Sequence[dict[str, Any]],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    stage_rows = [
        _build_stage_row(raw_record)
        for raw_record in raw_records
        if isinstance(raw_record, dict)
    ]
    promotion_path_funnel = build_promotion_path_funnel(stage_rows)
    hold_unknown_proposal_taxonomy = build_hold_unknown_proposal_taxonomy(stage_rows)
    promotion_path_by_strategy = build_promotion_path_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy",),
    )
    promotion_path_by_strategy_symbol = build_promotion_path_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy", "symbol"),
    )
    promotion_path_by_strategy_symbol_bias_sign = build_promotion_path_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy", "symbol", "bias_sign"),
    )
    strategy_specificity = build_strategy_specificity_summary(
        promotion_path_by_strategy=promotion_path_by_strategy
    )

    return {
        "configuration": configuration.to_dict(),
        "input_path": str(input_path),
        "effective_input_path": str(effective_input_path),
        "run_output_dir": str(run_output_dir),
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
                source_metadata.get("effective_input_record_count", len(raw_records))
                or 0
            ),
            "effective_input_materialized": bool(
                source_metadata.get("effective_input_materialized", True)
            ),
        },
        "headline": {
            "display_name": configuration.display_name,
            "latest_window_hours": configuration.latest_window_hours,
            "latest_max_rows": configuration.latest_max_rows,
            "raw_input_rows": promotion_path_funnel["raw_input_rows"],
            "raw_rows_with_known_identity": promotion_path_funnel[
                "raw_rows_with_known_identity"
            ],
            "directional_bias_present_known_identity_rows": promotion_path_funnel[
                "directional_bias_present_known_identity_rows"
            ],
            "final_buy_sell_emitted_rows": promotion_path_funnel[
                "final_buy_sell_emitted_rows"
            ],
            "final_buy_sell_emission_rate": promotion_path_funnel[
                "final_buy_sell_emission_rate"
            ],
            "final_hold_rows": promotion_path_funnel["final_hold_rows"],
            "final_unknown_rows": promotion_path_funnel["final_unknown_rows"],
            "final_hold_or_unknown_rows": promotion_path_funnel[
                "final_hold_or_unknown_rows"
            ],
            "primary_hold_collapse_stage": promotion_path_funnel[
                "primary_hold_collapse_stage"
            ],
            "primary_hold_collapse_category": promotion_path_funnel[
                "primary_hold_collapse_category"
            ],
            "scalping_primary_collapse_category": strategy_specificity.get(
                "scalping_primary_collapse_category"
            ),
            "strategy_specificity_classification": strategy_specificity.get(
                "classification"
            ),
        },
        "promotion_path_funnel": promotion_path_funnel,
        "hold_unknown_proposal_taxonomy": hold_unknown_proposal_taxonomy,
        "promotion_path_by_strategy": promotion_path_by_strategy,
        "promotion_path_by_strategy_symbol": promotion_path_by_strategy_symbol,
        "promotion_path_by_strategy_symbol_bias_sign": (
            promotion_path_by_strategy_symbol_bias_sign
        ),
        "strategy_specificity": strategy_specificity,
    }


def build_promotion_path_funnel(
    stage_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    known_identity_rows = [row for row in stage_rows if _has_known_identity(row)]
    directional_rows = [row for row in known_identity_rows if _has_directional_bias(row)]
    emitted_rows = [
        row
        for row in directional_rows
        if row.get("final_action_class") in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL}
    ]
    hold_rows = [
        row
        for row in directional_rows
        if row.get("final_action_class") == _ACTION_CLASS_HOLD
    ]
    unknown_rows = [
        row
        for row in directional_rows
        if row.get("final_action_class") == _ACTION_CLASS_UNKNOWN
    ]
    hold_unknown_rows = hold_rows + unknown_rows

    stage_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    selected_strategy_counter: Counter[str] = Counter()
    rule_signal_counter: Counter[str] = Counter()
    actionable_selected_strategy_rows = 0
    actionable_rule_signal_rows = 0
    risk_blocked_rows = 0

    for row in hold_unknown_rows:
        stage = str(
            row.get("earliest_collapse_stage") or _DETAILED_STAGE_PROMOTION_UNOBSERVABLE
        )
        category = str(row.get("collapse_category") or _CATEGORY_UNOBSERVABLE)
        stage_counter[stage] += 1
        category_counter[category] += 1
        selected_strategy_counter[
            row.get("selected_strategy_proposal_label") or _MISSING_LABEL
        ] += 1
        rule_signal_counter[row.get("rule_signal_label") or _MISSING_LABEL] += 1

        if bool(row.get("selected_strategy_proposal_actionable")):
            actionable_selected_strategy_rows += 1
        if bool(row.get("rule_signal_actionable")):
            actionable_rule_signal_rows += 1
        if row.get("earliest_collapse_stage") == _DETAILED_STAGE_RISK_BLOCKED:
            risk_blocked_rows += 1

    return {
        "raw_input_rows": len(stage_rows),
        "raw_rows_with_known_identity": len(known_identity_rows),
        "directional_bias_present_known_identity_rows": len(directional_rows),
        "final_buy_rows": sum(
            1
            for row in directional_rows
            if row.get("final_action_class") == _ACTION_CLASS_BUY
        ),
        "final_sell_rows": sum(
            1
            for row in directional_rows
            if row.get("final_action_class") == _ACTION_CLASS_SELL
        ),
        "final_buy_sell_emitted_rows": len(emitted_rows),
        "final_hold_rows": len(hold_rows),
        "final_unknown_rows": len(unknown_rows),
        "final_hold_or_unknown_rows": len(hold_unknown_rows),
        "final_buy_sell_emission_rate": _safe_ratio(len(emitted_rows), len(directional_rows)),
        "final_hold_share": _safe_ratio(len(hold_rows), len(directional_rows)),
        "final_unknown_share": _safe_ratio(len(unknown_rows), len(directional_rows)),
        "final_hold_or_unknown_share": _safe_ratio(
            len(hold_unknown_rows),
            len(directional_rows),
        ),
        "hold_unknown_rows_with_actionable_selected_strategy_proposal": (
            actionable_selected_strategy_rows
        ),
        "hold_unknown_rows_with_actionable_rule_signal": actionable_rule_signal_rows,
        "hold_unknown_rows_blocked_at_risk_gate": risk_blocked_rows,
        "hold_unknown_earliest_collapse_stage_counts": dict(stage_counter),
        "hold_unknown_earliest_collapse_stage_count_rows": _counter_rows(
            stage_counter,
            key_name="stage",
            order_map=_DETAILED_STAGE_ORDER,
        ),
        "hold_unknown_collapse_category_counts": dict(category_counter),
        "hold_unknown_collapse_category_count_rows": _counter_rows(
            category_counter,
            key_name="category",
            order_map=_CATEGORY_ORDER,
        ),
        "selected_strategy_proposal_counts_for_hold_unknown": dict(selected_strategy_counter),
        "selected_strategy_proposal_count_rows_for_hold_unknown": _counter_rows(
            selected_strategy_counter,
            key_name="proposal_state",
        ),
        "rule_signal_counts_for_hold_unknown": dict(rule_signal_counter),
        "rule_signal_count_rows_for_hold_unknown": _counter_rows(
            rule_signal_counter,
            key_name="proposal_state",
        ),
        "primary_hold_collapse_stage": _primary_counter_key(
            stage_counter,
            order_map=_DETAILED_STAGE_ORDER,
            empty="no_hold_unknown_rows",
        ),
        "primary_hold_collapse_category": _primary_counter_key(
            category_counter,
            order_map=_CATEGORY_ORDER,
            empty="no_hold_unknown_rows",
        ),
        "observable_stage_model": [
            {
                "stage": "selected_strategy_proposal",
                "fields": [
                    "selected_strategy",
                    "scalping_result.signal",
                    "intraday_result.signal",
                    "swing_result.signal",
                ],
                "used_for_primary_attribution": True,
            },
            {
                "stage": "rule_signal",
                "fields": ["rule_engine.signal"],
                "used_for_primary_attribution": True,
            },
            {
                "stage": "risk_gate",
                "fields": [
                    "risk.execution_allowed",
                    "risk.entry_price",
                    "execution.execution_allowed",
                    "execution.entry_price",
                ],
                "used_for_primary_attribution": True,
            },
            {
                "stage": "final_execution_action",
                "fields": ["execution.action", "execution.signal"],
                "used_for_primary_attribution": True,
            },
            {
                "stage": "decision_confirmation_layers",
                "fields": [
                    "timeframe_summary.context_layer",
                    "timeframe_summary.setup_layer",
                    "timeframe_summary.trigger_layer",
                ],
                "used_for_primary_attribution": False,
            },
        ],
    }


def build_hold_unknown_proposal_taxonomy(
    stage_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    directional_hold_unknown_rows = [
        row
        for row in stage_rows
        if _has_known_identity(row)
        and _has_directional_bias(row)
        and row.get("final_action_class") in {_ACTION_CLASS_HOLD, _ACTION_CLASS_UNKNOWN}
    ]

    selected_strategy_actionable_counter: Counter[str] = Counter()
    rule_actionable_counter: Counter[str] = Counter()

    for row in directional_hold_unknown_rows:
        selected_class = row.get("selected_strategy_proposal_class")
        rule_class = row.get("rule_signal_class")
        selected_strategy_actionable_counter[
            "actionable"
            if selected_class in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL}
            else "not_actionable_or_missing"
        ] += 1
        rule_actionable_counter[
            "actionable"
            if rule_class in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL}
            else "not_actionable_or_missing"
        ] += 1

    return {
        "directional_hold_unknown_row_count": len(directional_hold_unknown_rows),
        "selected_strategy_proposal_count_rows": _counter_rows(
            Counter(
                row.get("selected_strategy_proposal_label") or _MISSING_LABEL
                for row in directional_hold_unknown_rows
            ),
            key_name="proposal_state",
        ),
        "rule_signal_count_rows": _counter_rows(
            Counter(
                row.get("rule_signal_label") or _MISSING_LABEL
                for row in directional_hold_unknown_rows
            ),
            key_name="proposal_state",
        ),
        "selected_strategy_actionability_count_rows": _counter_rows(
            selected_strategy_actionable_counter,
            key_name="actionability",
        ),
        "rule_signal_actionability_count_rows": _counter_rows(
            rule_actionable_counter,
            key_name="actionability",
        ),
    }


def build_promotion_path_breakdown(
    *,
    stage_rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    directional_rows = [
        row
        for row in stage_rows
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
                "final_buy_rows": 0,
                "final_sell_rows": 0,
                "final_hold_rows": 0,
                "final_unknown_rows": 0,
                "selected_strategy_proposal_counts": Counter(),
                "rule_signal_counts": Counter(),
                "earliest_collapse_stage_counts": Counter(),
                "collapse_category_counts": Counter(),
                "hold_unknown_actionable_selected_strategy_rows": 0,
                "hold_unknown_actionable_rule_signal_rows": 0,
            }
            for field, value in zip(group_fields, group_values, strict=True):
                entry[field] = value
            grouped[key] = entry

        entry = grouped[key]
        entry["directional_total"] += 1
        final_action_class = row.get("final_action_class")
        if final_action_class == _ACTION_CLASS_BUY:
            entry["final_buy_rows"] += 1
        elif final_action_class == _ACTION_CLASS_SELL:
            entry["final_sell_rows"] += 1
        elif final_action_class == _ACTION_CLASS_HOLD:
            entry["final_hold_rows"] += 1
        else:
            entry["final_unknown_rows"] += 1

        if final_action_class in {_ACTION_CLASS_HOLD, _ACTION_CLASS_UNKNOWN}:
            _safe_counter(entry["selected_strategy_proposal_counts"])[
                row.get("selected_strategy_proposal_label") or _MISSING_LABEL
            ] += 1
            _safe_counter(entry["rule_signal_counts"])[
                row.get("rule_signal_label") or _MISSING_LABEL
            ] += 1
            _safe_counter(entry["earliest_collapse_stage_counts"])[
                row.get("earliest_collapse_stage") or _DETAILED_STAGE_PROMOTION_UNOBSERVABLE
            ] += 1
            _safe_counter(entry["collapse_category_counts"])[
                row.get("collapse_category") or _CATEGORY_UNOBSERVABLE
            ] += 1

            if bool(row.get("selected_strategy_proposal_actionable")):
                entry["hold_unknown_actionable_selected_strategy_rows"] += 1
            if bool(row.get("rule_signal_actionable")):
                entry["hold_unknown_actionable_rule_signal_rows"] += 1

    rows: list[dict[str, Any]] = []
    for entry in grouped.values():
        directional_total = int(entry.get("directional_total", 0) or 0)
        final_buy_rows = int(entry.get("final_buy_rows", 0) or 0)
        final_sell_rows = int(entry.get("final_sell_rows", 0) or 0)
        final_hold_rows = int(entry.get("final_hold_rows", 0) or 0)
        final_unknown_rows = int(entry.get("final_unknown_rows", 0) or 0)
        final_emitted_rows = final_buy_rows + final_sell_rows
        hold_unknown_rows = final_hold_rows + final_unknown_rows
        selected_strategy_counts = _safe_counter(entry["selected_strategy_proposal_counts"])
        rule_signal_counts = _safe_counter(entry["rule_signal_counts"])
        stage_counts = _safe_counter(entry["earliest_collapse_stage_counts"])
        category_counts = _safe_counter(entry["collapse_category_counts"])

        rows.append(
            {
                **{field: entry.get(field) for field in group_fields},
                "directional_total": directional_total,
                "final_buy_rows": final_buy_rows,
                "final_sell_rows": final_sell_rows,
                "final_buy_sell_emitted_rows": final_emitted_rows,
                "final_buy_sell_emission_rate": _safe_ratio(
                    final_emitted_rows,
                    directional_total,
                ),
                "final_hold_rows": final_hold_rows,
                "final_unknown_rows": final_unknown_rows,
                "final_hold_or_unknown_rows": hold_unknown_rows,
                "final_hold_or_unknown_share": _safe_ratio(
                    hold_unknown_rows,
                    directional_total,
                ),
                "hold_unknown_actionable_selected_strategy_rows": int(
                    entry.get("hold_unknown_actionable_selected_strategy_rows", 0) or 0
                ),
                "hold_unknown_actionable_rule_signal_rows": int(
                    entry.get("hold_unknown_actionable_rule_signal_rows", 0) or 0
                ),
                "selected_strategy_proposal_counts": dict(selected_strategy_counts),
                "selected_strategy_proposal_count_rows": _counter_rows(
                    selected_strategy_counts,
                    key_name="proposal_state",
                ),
                "rule_signal_counts": dict(rule_signal_counts),
                "rule_signal_count_rows": _counter_rows(
                    rule_signal_counts,
                    key_name="proposal_state",
                ),
                "earliest_collapse_stage_counts": dict(stage_counts),
                "earliest_collapse_stage_count_rows": _counter_rows(
                    stage_counts,
                    key_name="stage",
                    order_map=_DETAILED_STAGE_ORDER,
                ),
                "primary_earliest_collapse_stage": _primary_counter_key(
                    stage_counts,
                    order_map=_DETAILED_STAGE_ORDER,
                    empty="no_hold_unknown_rows",
                ),
                "collapse_category_counts": dict(category_counts),
                "collapse_category_count_rows": _counter_rows(
                    category_counts,
                    key_name="category",
                    order_map=_CATEGORY_ORDER,
                ),
                "primary_collapse_category": _primary_counter_key(
                    category_counts,
                    order_map=_CATEGORY_ORDER,
                    empty="no_hold_unknown_rows",
                ),
                "support_status": (
                    "supported"
                    if hold_unknown_rows >= _MIN_DIRECTIONAL_SUPPORT_ROWS
                    else "limited_support"
                ),
                "zero_emission": final_emitted_rows == 0,
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
    promotion_path_by_strategy: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    strategy_rows = [_safe_dict(row) for row in promotion_path_by_strategy]
    supported_rows = [
        row
        for row in strategy_rows
        if int(row.get("final_hold_or_unknown_rows", 0) or 0)
        >= _MIN_STRATEGY_SPECIFIC_SUPPORT_ROWS
    ]

    supported_category_map = {
        str(row.get("strategy")): str(row.get("primary_collapse_category"))
        for row in supported_rows
    }
    supported_stage_map = {
        str(row.get("strategy")): str(row.get("primary_earliest_collapse_stage"))
        for row in supported_rows
    }
    distinct_supported_categories = {
        category
        for category in supported_category_map.values()
        if category not in {"", "no_hold_unknown_rows"}
    }

    scalping_row = next(
        (row for row in strategy_rows if row.get("strategy") == "scalping"),
        None,
    )
    scalping_directional_total = int(
        _safe_dict(scalping_row).get("directional_total", 0) or 0
    )
    scalping_emitted_rows = int(
        _safe_dict(scalping_row).get("final_buy_sell_emitted_rows", 0) or 0
    )
    scalping_hold_unknown_rows = int(
        _safe_dict(scalping_row).get("final_hold_or_unknown_rows", 0) or 0
    )

    if len(supported_rows) < 2:
        classification = "insufficient_support"
    elif len(distinct_supported_categories) > 1:
        classification = "strategy_specific"
    else:
        classification = "consistent_or_mixed"

    return {
        "classification": classification,
        "supported_strategy_count": len(supported_rows),
        "supported_strategies": [str(row.get("strategy")) for row in supported_rows],
        "supported_primary_collapse_categories": supported_category_map,
        "supported_primary_collapse_stages": supported_stage_map,
        "scalping_directional_total": scalping_directional_total,
        "scalping_emitted_rows": scalping_emitted_rows,
        "scalping_hold_unknown_rows": scalping_hold_unknown_rows,
        "scalping_primary_collapse_category": _safe_dict(scalping_row).get(
            "primary_collapse_category"
        ),
        "strategy_rows": supported_rows,
    }


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "primary_bottleneck": "none",
            "primary_hold_path_category": "none",
            "dominant_observed_hold_path_category": "none",
            "factors": {},
            "widest_configuration": None,
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
        _CATEGORY_NO_ACTIONABLE: _assess_category(widest, _CATEGORY_NO_ACTIONABLE),
        _CATEGORY_LATE_COLLAPSE: _assess_category(widest, _CATEGORY_LATE_COLLAPSE),
        _CATEGORY_UNOBSERVABLE: _assess_category(widest, _CATEGORY_UNOBSERVABLE),
        "strategy_specificity": _assess_strategy_specificity(widest),
    }
    dominant_observed_hold_path_category = _dominant_observed_hold_path_category(widest)
    primary_bottleneck = _primary_bottleneck_label(factors=factors, widest=widest)
    confirmed_observations = _build_confirmed_observations(widest)
    evidence_backed_inferences = _build_evidence_backed_inferences(
        widest=widest,
        factors=factors,
        primary_bottleneck=primary_bottleneck,
    )
    unresolved_uncertainties = _build_unresolved_uncertainties(widest)

    return {
        "assessment": _overall_assessment_label(
            primary_bottleneck=primary_bottleneck,
            factors=factors,
        ),
        "primary_bottleneck": primary_bottleneck,
        "primary_hold_path_category": primary_bottleneck,
        "dominant_observed_hold_path_category": dominant_observed_hold_path_category,
        "factors": factors,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "confirmed_observations": confirmed_observations,
        "evidence_backed_inferences": evidence_backed_inferences,
        "unresolved_uncertainties": unresolved_uncertainties,
        "overall_conclusion": _build_overall_conclusion(
            widest=widest,
            factors=factors,
            primary_bottleneck=primary_bottleneck,
        ),
    }


def _assess_category(
    widest: dict[str, Any],
    category: str,
) -> dict[str, Any]:
    funnel = _safe_dict(widest.get("promotion_path_funnel"))
    counts = _safe_dict(funnel.get("hold_unknown_collapse_category_counts"))
    hold_unknown_rows = int(funnel.get("final_hold_or_unknown_rows", 0) or 0)
    category_rows = int(counts.get(category, 0) or 0)
    category_share = _safe_ratio(category_rows, hold_unknown_rows)

    evidence = [
        (
            f"Directional final hold/unknown rows: {hold_unknown_rows}; "
            f"{category}: {category_rows} ({category_share:.2%})."
        ),
        (
            "Primary detailed hold-collapse stage: "
            f"{funnel.get('primary_hold_collapse_stage', 'unknown')}."
        ),
    ]

    if hold_unknown_rows < _MIN_PRIMARY_SUPPORT_ROWS:
        status = "insufficient_support"
    elif category_share >= 0.60:
        status = "primary"
    elif category_share >= 0.35:
        status = "contributing"
    elif category_rows > 0:
        status = "limited"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _assess_strategy_specificity(widest: dict[str, Any]) -> dict[str, Any]:
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))
    evidence = [
        (
            f"Supported strategy count: {int(strategy_specificity.get('supported_strategy_count', 0) or 0)}; "
            f"supported primary categories: {strategy_specificity.get('supported_primary_collapse_categories', {})}."
        ),
        (
            "Supported primary stages: "
            f"{strategy_specificity.get('supported_primary_collapse_stages', {})}."
        ),
    ]

    classification = strategy_specificity.get("classification")
    if classification == "insufficient_support":
        status = "insufficient_support"
    elif classification == "strategy_specific":
        status = "strategy_specific"
    else:
        status = "not_supported"

    return {"status": status, "evidence": evidence}


def _primary_bottleneck_label(
    *,
    factors: dict[str, Any],
    widest: dict[str, Any],
) -> str:
    for category in (
        _CATEGORY_NO_ACTIONABLE,
        _CATEGORY_LATE_COLLAPSE,
        _CATEGORY_UNOBSERVABLE,
    ):
        if _safe_dict(factors.get(category)).get("status") == "primary":
            return category

    return "mixed_or_inconclusive"


def _dominant_observed_hold_path_category(widest: dict[str, Any]) -> str:
    funnel = _safe_dict(widest.get("promotion_path_funnel"))
    category = funnel.get("primary_hold_collapse_category")
    if category is None:
        return "none"
    return str(category)


def _overall_assessment_label(
    *,
    primary_bottleneck: str,
    factors: dict[str, Any],
) -> str:
    if primary_bottleneck == _CATEGORY_NO_ACTIONABLE and _safe_dict(
        factors.get(_CATEGORY_NO_ACTIONABLE)
    ).get("status") == "primary":
        return "no_actionable_upstream_proposal_visible_primary"
    if primary_bottleneck == _CATEGORY_LATE_COLLAPSE and _safe_dict(
        factors.get(_CATEGORY_LATE_COLLAPSE)
    ).get("status") == "primary":
        return "actionable_upstream_proposal_later_collapses_primary"
    if primary_bottleneck == _CATEGORY_UNOBSERVABLE and _safe_dict(
        factors.get(_CATEGORY_UNOBSERVABLE)
    ).get("status") == "primary":
        return "intermediate_trace_unobservable_primary"

    if any(
        _safe_dict(factors.get(category)).get("status")
        in {"contributing", "limited", "strategy_specific"}
        for category in (
            _CATEGORY_NO_ACTIONABLE,
            _CATEGORY_LATE_COLLAPSE,
            _CATEGORY_UNOBSERVABLE,
            "strategy_specificity",
        )
    ):
        return "mixed_contributing_factors"
    if any(
        _safe_dict(factors.get(category)).get("status") == "insufficient_support"
        for category in (
            _CATEGORY_NO_ACTIONABLE,
            _CATEGORY_LATE_COLLAPSE,
            _CATEGORY_UNOBSERVABLE,
            "strategy_specificity",
        )
    ):
        return "insufficient_support"
    return "not_supported_or_inconclusive"


def _build_confirmed_observations(widest: dict[str, Any]) -> list[str]:
    funnel = _safe_dict(widest.get("promotion_path_funnel"))
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))

    return [
        (
            "Directional-bias-present rows: "
            f"{int(funnel.get('directional_bias_present_known_identity_rows', 0) or 0)}; "
            f"final buy/sell rows: {int(funnel.get('final_buy_sell_emitted_rows', 0) or 0)} "
            f"({_format_ratio(funnel.get('final_buy_sell_emission_rate'))})."
        ),
        (
            "Directional final hold rows: "
            f"{int(funnel.get('final_hold_rows', 0) or 0)}; "
            f"directional final unknown rows: {int(funnel.get('final_unknown_rows', 0) or 0)}."
        ),
        (
            "Directional final hold/unknown rows by collapse category: "
            f"{funnel.get('hold_unknown_collapse_category_counts', {})}."
        ),
        (
            "Directional final hold/unknown rows by earliest observable stage: "
            f"{funnel.get('hold_unknown_earliest_collapse_stage_counts', {})}."
        ),
        (
            "Supported per-strategy primary collapse categories: "
            f"{strategy_specificity.get('supported_primary_collapse_categories', {})}."
        ),
    ]


def _build_evidence_backed_inferences(
    *,
    widest: dict[str, Any],
    factors: dict[str, Any],
    primary_bottleneck: str,
) -> list[str]:
    inferences: list[str] = []
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))

    if primary_bottleneck == _CATEGORY_NO_ACTIONABLE:
        inferences.append(
            "Most directional rows that finish as hold/unknown do not show an actionable upstream proposal in the observable schema before execution."
        )
    elif primary_bottleneck == _CATEGORY_LATE_COLLAPSE:
        inferences.append(
            "An actionable upstream proposal is often visible, but it later collapses at the rule-selection, risk, or execution layer before final buy/sell emission."
        )
    elif primary_bottleneck == _CATEGORY_UNOBSERVABLE:
        inferences.append(
            "The current schema leaves too much of the promotion path unobserved to prove where most directional rows lose actionability."
        )

    if _safe_dict(factors.get("strategy_specificity")).get("status") == "strategy_specific":
        inferences.append(
            "The promotion-path explanation is materially strategy-specific rather than uniform across scalping, intraday, and swing."
        )

    scalping_category = strategy_specificity.get("scalping_primary_collapse_category")
    if isinstance(scalping_category, str) and scalping_category:
        inferences.append(
            f"Scalping's supported primary hold-path category is {scalping_category}."
        )

    return inferences


def _build_unresolved_uncertainties(widest: dict[str, Any]) -> list[str]:
    return [
        "This report can trace selected-strategy proposal, rule signal, risk execution_allowed, and execution-layer action, but it cannot prove the internal reason a strategy or decision layer chose hold.",
        "The schema exposes timeframe_summary context/setup/trigger layers, yet those layers do not persist a stable execution-facing proposal label, so this report does not use them as definitive collapse stages.",
        "If a row has an actionable strategy proposal but missing downstream rule or risk fields, the report classifies that as unobservable rather than guessing whether the collapse happened in decision composition, risk evaluation, or execution assembly.",
    ]


def _build_overall_conclusion(
    *,
    widest: dict[str, Any],
    factors: dict[str, Any],
    primary_bottleneck: str,
) -> str:
    widest_config = _safe_dict(widest.get("configuration")).get(
        "display_name",
        "widest configuration",
    )
    funnel = _safe_dict(widest.get("promotion_path_funnel"))
    strategy_specificity = _safe_dict(widest.get("strategy_specificity"))

    parts = [
        f"At {widest_config}, directional-bias-present rows were {int(funnel.get('directional_bias_present_known_identity_rows', 0) or 0)} and final buy/sell rows were {int(funnel.get('final_buy_sell_emitted_rows', 0) or 0)} ({_format_ratio(funnel.get('final_buy_sell_emission_rate'))}).",
        f"Directional final hold/unknown rows were {int(funnel.get('final_hold_or_unknown_rows', 0) or 0)} with dominant observed hold-path category {funnel.get('primary_hold_collapse_category', 'unknown')} and dominant observed detailed stage {funnel.get('primary_hold_collapse_stage', 'unknown')}.",
    ]

    if primary_bottleneck == _CATEGORY_NO_ACTIONABLE:
        parts.insert(
            0,
            "The strongest supported explanation is that most directional rows never show an actionable upstream proposal in the current observable schema.",
        )
    elif primary_bottleneck == _CATEGORY_LATE_COLLAPSE:
        parts.insert(
            0,
            "The strongest supported explanation is that actionable upstream proposals are visible but later collapse before final execution-layer buy/sell emission.",
        )
    elif primary_bottleneck == _CATEGORY_UNOBSERVABLE:
        parts.insert(
            0,
            "The strongest supported explanation is that the current observable schema stops short of proving where most directional rows lose actionability.",
        )
    else:
        parts.insert(
            0,
            "No single hold-path explanation is fully dominant in the inspected configuration, so the outcome remains mixed or inconclusive.",
        )

    if _safe_dict(factors.get("strategy_specificity")).get("status") == "strategy_specific":
        parts.append(
            "Supported strategies do not share one uniform primary hold-path category: "
            f"{strategy_specificity.get('supported_primary_collapse_categories', {})}."
        )

    return " ".join(parts)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {REPORT_TITLE}", ""]

    lines.append("## Configurations Evaluated")
    lines.append("")
    for configuration in _safe_list(report.get("configurations_evaluated")):
        config = _safe_dict(configuration)
        lines.append(
            f"- {config.get('display_name')}: latest_window_hours={config.get('latest_window_hours')}, latest_max_rows={config.get('latest_max_rows')}"
        )
    lines.append("")

    lines.append("## Per-Configuration Headline Summary")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        headline = _safe_dict(_safe_dict(summary).get("headline"))
        lines.append(f"### {headline.get('display_name', 'n/a')}")
        lines.append(f"- raw_input_rows: {headline.get('raw_input_rows', 0)}")
        lines.append(
            f"- directional_bias_present_known_identity_rows: {headline.get('directional_bias_present_known_identity_rows', 0)}"
        )
        lines.append(
            f"- final_buy_sell_emitted_rows: {headline.get('final_buy_sell_emitted_rows', 0)} ({_format_ratio(headline.get('final_buy_sell_emission_rate'))})"
        )
        lines.append(f"- final_hold_rows: {headline.get('final_hold_rows', 0)}")
        lines.append(f"- final_unknown_rows: {headline.get('final_unknown_rows', 0)}")
        lines.append(
            f"- final_hold_or_unknown_rows: {headline.get('final_hold_or_unknown_rows', 0)}"
        )
        lines.append(
            f"- primary_hold_collapse_stage: {headline.get('primary_hold_collapse_stage', 'n/a')}"
        )
        lines.append(
            f"- primary_hold_collapse_category: {headline.get('primary_hold_collapse_category', 'n/a')}"
        )
        lines.append(
            f"- strategy_specificity_classification: {headline.get('strategy_specificity_classification', 'n/a')}"
        )
        lines.append("")

    lines.append("## Promotion Path Funnel")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        funnel = _safe_dict(_safe_dict(summary).get("promotion_path_funnel"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            f"- final_buy_sell_emitted_rows: {funnel.get('final_buy_sell_emitted_rows', 0)} ({_format_ratio(funnel.get('final_buy_sell_emission_rate'))})"
        )
        lines.append(f"- final_hold_rows: {funnel.get('final_hold_rows', 0)}")
        lines.append(f"- final_unknown_rows: {funnel.get('final_unknown_rows', 0)}")
        lines.append(
            "- hold_unknown_collapse_category_counts: "
            f"{_format_counter_rows(funnel.get('hold_unknown_collapse_category_count_rows'), 'category')}"
        )
        lines.append(
            "- hold_unknown_earliest_collapse_stage_counts: "
            f"{_format_counter_rows(funnel.get('hold_unknown_earliest_collapse_stage_count_rows'), 'stage')}"
        )
        lines.append(
            "- selected_strategy_proposal_counts_for_hold_unknown: "
            f"{_format_counter_rows(funnel.get('selected_strategy_proposal_count_rows_for_hold_unknown'), 'proposal_state')}"
        )
        lines.append(
            "- rule_signal_counts_for_hold_unknown: "
            f"{_format_counter_rows(funnel.get('rule_signal_count_rows_for_hold_unknown'), 'proposal_state')}"
        )
        lines.append("")

    lines.append("## Hold Unknown Proposal Taxonomy")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        taxonomy = _safe_dict(_safe_dict(summary).get("hold_unknown_proposal_taxonomy"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(
            "- selected_strategy_proposal_count_rows: "
            f"{_format_counter_rows(taxonomy.get('selected_strategy_proposal_count_rows'), 'proposal_state')}"
        )
        lines.append(
            "- rule_signal_count_rows: "
            f"{_format_counter_rows(taxonomy.get('rule_signal_count_rows'), 'proposal_state')}"
        )
        lines.append(
            "- selected_strategy_actionability_count_rows: "
            f"{_format_counter_rows(taxonomy.get('selected_strategy_actionability_count_rows'), 'actionability')}"
        )
        lines.append(
            "- rule_signal_actionability_count_rows: "
            f"{_format_counter_rows(taxonomy.get('rule_signal_actionability_count_rows'), 'actionability')}"
        )
        lines.append("")

    for key, title in (
        ("promotion_path_by_strategy", "Promotion Path By Strategy"),
        ("promotion_path_by_strategy_symbol", "Promotion Path By Strategy and Symbol"),
        (
            "promotion_path_by_strategy_symbol_bias_sign",
            "Promotion Path By Strategy, Symbol, and Bias Sign",
        ),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for summary in _safe_list(report.get("configuration_summaries")):
            config = _safe_dict(_safe_dict(summary).get("configuration"))
            rows = _safe_list(_safe_dict(summary).get(key))
            lines.append(f"### {config.get('display_name', 'n/a')}")
            if not rows:
                lines.append("No directional rows available.")
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
                    f"emitted={item.get('final_buy_sell_emitted_rows', 0)}, "
                    f"hold={item.get('final_hold_rows', 0)}, "
                    f"unknown={item.get('final_unknown_rows', 0)}, "
                    f"primary_category={item.get('primary_collapse_category', 'n/a')}, "
                    f"primary_stage={item.get('primary_earliest_collapse_stage', 'n/a')}, "
                    f"support_status={item.get('support_status', 'n/a')}"
                )
            lines.append("")

    lines.append("## Strategy Specificity")
    lines.append("")
    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        specificity = _safe_dict(_safe_dict(summary).get("strategy_specificity"))
        lines.append(f"### {config.get('display_name', 'n/a')}")
        lines.append(f"- classification: {specificity.get('classification', 'n/a')}")
        lines.append(
            f"- supported_primary_collapse_categories: {specificity.get('supported_primary_collapse_categories', {})}"
        )
        lines.append(
            f"- supported_primary_collapse_stages: {specificity.get('supported_primary_collapse_stages', {})}"
        )
        lines.append(
            f"- scalping_primary_collapse_category: {specificity.get('scalping_primary_collapse_category', 'n/a')}"
        )
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append(f"- assessment: {final_assessment.get('assessment', 'n/a')}")
    lines.append(
        f"- primary_bottleneck: {final_assessment.get('primary_bottleneck', 'n/a')}"
    )
    lines.append(
        "- dominant_observed_hold_path_category: "
        f"{final_assessment.get('dominant_observed_hold_path_category', 'n/a')}"
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


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_record(raw_record)
    strategy = _normalize_strategy(normalized.get("selected_strategy"))
    selected_strategy_payload = _as_dict(
        raw_record.get(f"{strategy}_result") if strategy is not None else None
    )

    selected_strategy_proposal_label = _normalize_signal(
        selected_strategy_payload.get("signal")
    )
    rule_signal_label = _normalize_signal(normalized.get("rule_signal"))
    final_action_label = _normalize_signal(
        normalized.get("execution_action") or normalized.get("execution_signal")
    )

    selected_strategy_proposal_class = _action_class_from_label(
        selected_strategy_proposal_label
    )
    rule_signal_class = _action_class_from_label(rule_signal_label)
    final_action_class = _action_class_from_label(final_action_label)
    earliest_collapse_stage, collapse_category = _earliest_collapse_stage(
        selected_strategy_proposal_label=selected_strategy_proposal_label,
        selected_strategy_proposal_class=selected_strategy_proposal_class,
        rule_signal_label=rule_signal_label,
        rule_signal_class=rule_signal_class,
        execution_allowed=normalized.get("execution_allowed"),
        final_action_class=final_action_class,
    )

    return {
        **normalized,
        "strategy": strategy,
        "bias_sign": _bias_sign(normalized),
        "selected_strategy_proposal_label": selected_strategy_proposal_label,
        "selected_strategy_proposal_class": selected_strategy_proposal_class,
        "selected_strategy_proposal_actionable": selected_strategy_proposal_class
        in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL},
        "rule_signal_label": rule_signal_label,
        "rule_signal_class": rule_signal_class,
        "rule_signal_actionable": rule_signal_class
        in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL},
        "final_action_label": final_action_label,
        "final_action_class": final_action_class,
        "earliest_collapse_stage": earliest_collapse_stage,
        "collapse_category": collapse_category,
    }


def _earliest_collapse_stage(
    *,
    selected_strategy_proposal_label: str | None,
    selected_strategy_proposal_class: str,
    rule_signal_label: str | None,
    rule_signal_class: str,
    execution_allowed: Any,
    final_action_class: str,
) -> tuple[str | None, str | None]:
    if final_action_class not in {_ACTION_CLASS_HOLD, _ACTION_CLASS_UNKNOWN}:
        return None, None

    selected_strategy_actionable = selected_strategy_proposal_class in {
        _ACTION_CLASS_BUY,
        _ACTION_CLASS_SELL,
    }
    rule_actionable = rule_signal_class in {_ACTION_CLASS_BUY, _ACTION_CLASS_SELL}
    selected_strategy_observable = selected_strategy_proposal_label is not None
    rule_observable = rule_signal_label is not None

    if selected_strategy_actionable:
        if rule_observable and not rule_actionable:
            return _DETAILED_STAGE_DECISION_COLLAPSE, _CATEGORY_LATE_COLLAPSE
        if not rule_observable:
            return _DETAILED_STAGE_RULE_UNOBSERVABLE, _CATEGORY_UNOBSERVABLE

    if rule_actionable:
        if execution_allowed is False:
            return _DETAILED_STAGE_RISK_BLOCKED, _CATEGORY_LATE_COLLAPSE
        if execution_allowed is True:
            return _DETAILED_STAGE_EXECUTION_COLLAPSE, _CATEGORY_LATE_COLLAPSE
        return _DETAILED_STAGE_RISK_UNOBSERVABLE, _CATEGORY_UNOBSERVABLE

    if selected_strategy_observable and not selected_strategy_actionable:
        return _DETAILED_STAGE_SELECTED_NOT_ACTIONABLE, _CATEGORY_NO_ACTIONABLE
    if rule_observable and not rule_actionable:
        return _DETAILED_STAGE_RULE_NOT_ACTIONABLE, _CATEGORY_NO_ACTIONABLE
    return _DETAILED_STAGE_PROMOTION_UNOBSERVABLE, _CATEGORY_UNOBSERVABLE


def _group_value(row: dict[str, Any], field: str) -> str | None:
    if field == "strategy":
        return _normalize_strategy(row.get("strategy") or row.get("selected_strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    if field == "bias_sign":
        return _bias_sign(row)
    raise ValueError(f"Unsupported group field: {field}")


def _has_known_identity(row: dict[str, Any]) -> bool:
    return (
        _normalize_symbol(row.get("symbol")) is not None
        and _normalize_strategy(row.get("strategy") or row.get("selected_strategy"))
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


def _action_class_from_label(value: str | None) -> str:
    if value is None:
        return _ACTION_CLASS_UNKNOWN
    if value in _HOLD_LIKE_VALUES:
        return _ACTION_CLASS_HOLD
    if value in _BUY_LIKE_VALUES:
        return _ACTION_CLASS_BUY
    if value in _SELL_LIKE_VALUES:
        return _ACTION_CLASS_SELL
    return _ACTION_CLASS_UNKNOWN


def _normalize_signal(value: Any) -> str | None:
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


def _format_ratio(value: Any) -> str:
    ratio = _to_float(value)
    if ratio is None:
        return "n/a"
    return f"{ratio:.2%}"


def _format_counter_rows(value: Any, key_name: str) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get(key_name, 'n/a')}={_safe_dict(row).get('count', 0)}"
        for row in rows
    )


def _counter_rows(
    counter: Counter[str],
    *,
    key_name: str,
    order_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            key_name: label,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for label, count in sorted(
            counter.items(),
            key=lambda item: (
                order_map.get(item[0], 99) if isinstance(order_map, dict) else 99,
                -item[1],
                item[0],
            ),
        )
    ]


def _primary_counter_key(
    counter: Counter[str],
    *,
    order_map: dict[str, int] | None = None,
    empty: str,
) -> str:
    if not counter:
        return empty
    return min(
        counter.items(),
        key=lambda item: (
            -item[1],
            order_map.get(item[0], 99) if isinstance(order_map, dict) else 99,
            item[0],
        ),
    )[0]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _safe_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_counter(value: Any) -> Counter[str]:
    if isinstance(value, Counter):
        return value
    return Counter()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _to_float(value: Any, *, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
