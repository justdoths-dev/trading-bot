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

REPORT_TYPE = "selected_strategy_hold_resolution_reason_diagnosis_report"
REPORT_TITLE = "Selected Strategy Hold Resolution Reason Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_EFFECTIVE_INPUT_FILENAME = (
    "_effective_selected_strategy_hold_resolution_reason_input.jsonl"
)
_MISSING_LABEL = "(missing)"

_ACTIONABLE_SIGNAL_STATES = {"long", "short"}
_LONG_SIGNAL_VALUES = {"long", "buy"}
_SHORT_SIGNAL_VALUES = {"short", "sell"}
_HOLD_SIGNAL_VALUES = {"hold", "neutral", "flat", "no_trade", "no-trade"}
_WATCHLIST_LONG_SIGNAL_VALUES = {"watchlist_long"}
_WATCHLIST_SHORT_SIGNAL_VALUES = {"watchlist_short"}
_WATCHLIST_SIGNAL_VALUES = {"watchlist"}
_NO_SIGNAL_VALUES = {"no_signal", "no-signal"}
_UNKNOWN_SIGNAL_VALUES = {"unknown"}

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
_BULLISH_BIAS_VALUES = {"long", "bullish", "buy", "watchlist_long"}
_BEARISH_BIAS_VALUES = {"short", "bearish", "sell", "watchlist_short"}

_BUCKET_EXPLICIT_CONTEXT_CONFLICT = "explicit_context_conflict"
_BUCKET_EXPLICIT_CONTEXT_NEUTRALITY = "explicit_context_neutrality"
_BUCKET_EXPLICIT_CONFIRMATION_GAP = "explicit_confirmation_gap"
_BUCKET_EXPLICIT_DIRECTIONAL_OPPOSITION = "explicit_directional_opposition"
_BUCKET_MIXED_OR_INCONCLUSIVE = "mixed_or_inconclusive"
_BUCKET_INSUFFICIENT_PERSISTED_EXPLANATION = "insufficient_persisted_explanation"

_BUCKET_ORDER = {
    _BUCKET_EXPLICIT_CONTEXT_CONFLICT: 0,
    _BUCKET_EXPLICIT_CONTEXT_NEUTRALITY: 1,
    _BUCKET_EXPLICIT_CONFIRMATION_GAP: 2,
    _BUCKET_EXPLICIT_DIRECTIONAL_OPPOSITION: 3,
    _BUCKET_MIXED_OR_INCONCLUSIVE: 4,
    _BUCKET_INSUFFICIENT_PERSISTED_EXPLANATION: 5,
}
_NON_PRIMARY_BUCKET_VALUES = {
    "",
    "none",
    "insufficient_support",
    "mixed_or_inconclusive",
    "no_target_rows",
}
_SIGNAL_STATE_ORDER = {
    "long": 0,
    "short": 1,
    "hold": 2,
    "watchlist_long": 3,
    "watchlist_short": 4,
    "watchlist": 5,
    "no_signal": 6,
    "unknown": 7,
    "other": 8,
    _MISSING_LABEL: 9,
}

_COMPOSED_DECISION_ONLY_KEYS = {"selected_strategy", "timeframe_summary", "debug"}

_REASON_CONFLICT_MARKERS = (
    "conflict",
    "conflicted",
    "conflicting",
    "contradict",
    "contrary",
    "disagree",
    "disagreement",
    "misalign",
    "mixed",
)
_REASON_NEUTRALITY_MARKERS = (
    "neutral",
    "sideways",
    "flat",
    "range-bound",
    "range bound",
    "stand aside",
    "stay flat",
    "remain flat",
    "remains flat",
    "remain neutral",
    "remains neutral",
    "stays neutral",
)
_REASON_CONFIRMATION_MARKERS = (
    "not both confirmed",
    "not confirmed",
    "not enough confirmation",
    "insufficient confirmation",
    "needs confirmation",
    "requires confirmation",
    "await",
    "waiting for confirmation",
    "not strong enough",
    "not ready",
    "alignment is not ready",
    "only trend-following",
    "only a fully aligned",
    "fully aligned",
)
_REASON_DIRECTIONAL_OPPOSITION_MARKERS = (
    "opposed",
    "opposing",
    "against",
    "countertrend",
    "counter-trend",
    "opposite",
    "opposes",
)
_STATE_CONFLICT_MARKERS = (
    "conflict",
    "conflicted",
    "conflicting",
    "mixed",
    "misalign",
    "disagree",
)
_STATE_NEUTRALITY_MARKERS = (
    "neutral",
    "flat",
    "sideways",
    "range",
    "hold",
    "no_trade",
    "no-trade",
)
_STRONG_CONFIRMATION_MARKERS = (
    "confirmed",
    "fully_confirmed",
    "fully-confirmed",
    "ready",
    "triggered",
    "active",
    "valid",
    "aligned",
)
_WEAK_CONFIRMATION_MARKERS = (
    "weak",
    "early",
    "partial",
    "candidate",
    "pending",
    "await",
    "not_ready",
    "not-ready",
    "not ready",
    "unconfirmed",
)

_PRIMARY_FACTOR_THRESHOLD = 0.60

_MIN_PRIMARY_TARGET_SUPPORT_ROWS = 30
_MIN_BREAKDOWN_TARGET_SUPPORT_ROWS = 10
_MIN_BIAS_SIGN_BREAKDOWN_TARGET_SUPPORT_ROWS = 10

_ALLOWED_INVENTORY_PREFIXES = (
    "selected_strategy_payload",
    "rule_engine",
    "timeframe_summary.context_layer",
    "timeframe_summary.bias_layer",
    "timeframe_summary.setup_layer",
    "timeframe_summary.trigger_layer",
    "reason",
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
        return f"{self.latest_window_hours}_{self.latest_max_rows}"

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
            "Build a diagnosis-only report for rows where an actionable "
            "selected_strategy_result.signal collapses to rule_engine.signal=hold "
            "using a single effective input snapshot per configuration."
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

    result = run_selected_strategy_hold_resolution_reason_diagnosis_report(
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
                "primary_observed_hold_resolution_bucket": final_assessment.get(
                    "primary_observed_hold_resolution_bucket"
                ),
                "dominant_observed_condition_bucket": final_assessment.get(
                    "dominant_observed_condition_bucket"
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


def run_selected_strategy_hold_resolution_reason_diagnosis_report(
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
        effective_input_path, raw_records, source_metadata = (
            _materialize_configuration_input(
                input_path=resolved_input,
                run_output_dir=run_output_dir,
                latest_window_hours=configuration.latest_window_hours,
                latest_max_rows=configuration.latest_max_rows,
            )
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
        "final_assessment": build_final_assessment(configuration_summaries),
        "assumptions": [
            "This report is diagnosis-only and reuses one materialized effective input snapshot per configuration so every targeting, inventory, and bucket count comes from the same observed rows.",
            "selected_strategy_result.signal is sourced only from the legacy payload addressed by selected_strategy (for example swing_result / intraday_result / scalping_result). A literal selected_strategy_result object is used only when it looks like the same legacy payload rather than a composed post-decision object.",
            "rule_signal is sourced only from the raw persisted rule_engine.signal field. Final execution action, execution.signal, and normalized fallback fields are never used as proxies for the upstream rule-signal hold transition.",
            "Observable decision-layer inventory is intentionally narrowed to selected_strategy payload fields, rule_engine fields, and timeframe_summary decision-adjacent layers (context_layer, bias_layer, setup_layer, trigger_layer). Raw per-timeframe indicator fan-out is not treated as hold-resolution inventory.",
            "Explicit observable condition buckets are structured-state-backed only. Classified reason text is tracked separately as auxiliary evidence and does not by itself promote a row out of insufficient_persisted_explanation.",
            "Condition buckets remain evidence-backed and conservative: if multiple structured categories co-occur materially the row is treated as mixed_or_inconclusive, and rows without enough structured evidence remain insufficient_persisted_explanation.",
            "Support thresholds gate primary-bucket claims so small slices stay descriptive instead of over-interpreted.",
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
    hold_resolution_targeting = build_hold_resolution_targeting(stage_rows)
    target_rows = [
        row
        for row in stage_rows
        if row.get("selected_strategy_hold_resolution_target_row") is True
    ]
    observable_decision_layer_inventory = build_observable_decision_layer_inventory(
        target_rows
    )
    hold_resolution_reason_summary = build_hold_resolution_reason_summary(
        target_rows=target_rows,
        support_threshold=_MIN_PRIMARY_TARGET_SUPPORT_ROWS,
    )
    hold_resolution_by_strategy = build_hold_resolution_breakdown(
        target_rows=target_rows,
        group_fields=("strategy",),
        support_threshold=_MIN_BREAKDOWN_TARGET_SUPPORT_ROWS,
    )
    hold_resolution_by_strategy_symbol = build_hold_resolution_breakdown(
        target_rows=target_rows,
        group_fields=("strategy", "symbol"),
        support_threshold=_MIN_BREAKDOWN_TARGET_SUPPORT_ROWS,
    )
    hold_resolution_by_strategy_symbol_bias_sign = build_hold_resolution_breakdown(
        target_rows=target_rows,
        group_fields=("strategy", "symbol", "bias_sign"),
        support_threshold=_MIN_BIAS_SIGN_BREAKDOWN_TARGET_SUPPORT_ROWS,
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
            "transition_row_count": hold_resolution_targeting["transition_row_count"],
            "selected_strategy_actionable_rows": hold_resolution_targeting[
                "selected_strategy_actionable_rows"
            ],
            "target_row_count": hold_resolution_targeting["target_row_count"],
            "explicit_observable_condition_rows": hold_resolution_reason_summary[
                "explicit_observable_condition_rows"
            ],
            "reason_text_only_classified_rows": hold_resolution_reason_summary[
                "reason_text_only_classified_rows"
            ],
            "insufficient_persisted_explanation_rows": (
                hold_resolution_reason_summary[
                    "insufficient_persisted_explanation_rows"
                ]
            ),
            "dominant_observed_condition_bucket": hold_resolution_reason_summary[
                "dominant_observed_condition_bucket"
            ],
            "primary_condition_bucket": hold_resolution_reason_summary[
                "primary_condition_bucket"
            ],
        },
        "hold_resolution_targeting": hold_resolution_targeting,
        "observable_decision_layer_inventory": observable_decision_layer_inventory,
        "hold_resolution_reason_summary": hold_resolution_reason_summary,
        "hold_resolution_by_strategy": hold_resolution_by_strategy,
        "hold_resolution_by_strategy_symbol": hold_resolution_by_strategy_symbol,
        "hold_resolution_by_strategy_symbol_bias_sign": (
            hold_resolution_by_strategy_symbol_bias_sign
        ),
        "confirmed_observations": _build_configuration_confirmed_observations(
            configuration=configuration,
            hold_resolution_targeting=hold_resolution_targeting,
            observable_decision_layer_inventory=observable_decision_layer_inventory,
            hold_resolution_reason_summary=hold_resolution_reason_summary,
            hold_resolution_by_strategy=hold_resolution_by_strategy,
        ),
        "evidence_backed_inferences": _build_configuration_evidence_backed_inferences(
            hold_resolution_reason_summary=hold_resolution_reason_summary,
            hold_resolution_by_strategy=hold_resolution_by_strategy,
        ),
        "unresolved_uncertainties": _build_configuration_unresolved_uncertainties(
            observable_decision_layer_inventory=observable_decision_layer_inventory,
            hold_resolution_reason_summary=hold_resolution_reason_summary,
        ),
    }


def build_hold_resolution_targeting(
    stage_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rule_state_counter: Counter[str] = Counter()
    selected_strategy_actionable_rows = 0
    selected_strategy_non_actionable_rows = 0
    selected_strategy_unobservable_rows = 0
    actionable_selected_strategy_rule_hold_rows = 0
    actionable_selected_strategy_rule_non_hold_rows = 0
    actionable_selected_strategy_rule_signal_unobservable_rows = 0
    rows_with_known_identity = 0

    for row in stage_rows:
        if _has_known_identity(row):
            rows_with_known_identity += 1

        selected_state = row.get("selected_strategy_result_signal_state")
        rule_state = row.get("rule_signal_state")

        if selected_state not in _ACTIONABLE_SIGNAL_STATES:
            if selected_state is None:
                selected_strategy_unobservable_rows += 1
            else:
                selected_strategy_non_actionable_rows += 1
            continue

        selected_strategy_actionable_rows += 1
        rule_state_counter[_display_signal_state(rule_state)] += 1

        if rule_state == "hold":
            actionable_selected_strategy_rule_hold_rows += 1
        elif rule_state is None:
            actionable_selected_strategy_rule_signal_unobservable_rows += 1
        else:
            actionable_selected_strategy_rule_non_hold_rows += 1

    return {
        "transition_row_count": len(stage_rows),
        "rows_with_known_identity": rows_with_known_identity,
        "selected_strategy_actionable_rows": selected_strategy_actionable_rows,
        "selected_strategy_non_actionable_rows": selected_strategy_non_actionable_rows,
        "selected_strategy_unobservable_rows": selected_strategy_unobservable_rows,
        "actionable_selected_strategy_rule_hold_rows": (
            actionable_selected_strategy_rule_hold_rows
        ),
        "actionable_selected_strategy_rule_non_hold_rows": (
            actionable_selected_strategy_rule_non_hold_rows
        ),
        "actionable_selected_strategy_rule_signal_unobservable_rows": (
            actionable_selected_strategy_rule_signal_unobservable_rows
        ),
        "rule_signal_count_rows_on_actionable_selected_strategy": _counter_rows(
            rule_state_counter,
            key_name="signal_state",
            order_map=_SIGNAL_STATE_ORDER,
        ),
        "target_row_count": actionable_selected_strategy_rule_hold_rows,
    }


def build_observable_decision_layer_inventory(
    target_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    object_counter: Counter[str] = Counter()
    field_counter: Counter[str] = Counter()
    selected_strategy_payload_source_counter: Counter[str] = Counter()
    reason_text_source_counter: Counter[str] = Counter()
    evidence_source_counter: Counter[str] = Counter()
    bias_sign_source_counter: Counter[str] = Counter()
    context_state_counter: Counter[str] = Counter()
    setup_state_counter: Counter[str] = Counter()
    trigger_state_counter: Counter[str] = Counter()
    rule_bias_counter: Counter[str] = Counter()

    rows_with_rule_reason_text = 0
    rows_with_root_reason_text = 0
    rows_with_context_state = 0
    rows_with_setup_state = 0
    rows_with_trigger_state = 0
    rows_with_context_bias = 0
    rows_with_rule_bias = 0

    for row in target_rows:
        payload_source = row.get("selected_strategy_payload_source")
        if isinstance(payload_source, str) and payload_source.strip():
            selected_strategy_payload_source_counter[payload_source] += 1

        reason_text_sources: list[str] = []
        if row.get("rule_reason_text") is not None:
            rows_with_rule_reason_text += 1
            reason_text_sources.append("rule_reason_text")
        if row.get("root_reason_text") is not None:
            rows_with_root_reason_text += 1
            reason_text_sources.append("root_reason_text")
        if reason_text_sources:
            reason_text_source_counter[" + ".join(reason_text_sources)] += 1

        evidence_source = row.get("hold_resolution_evidence_source")
        if isinstance(evidence_source, str) and evidence_source.strip():
            evidence_source_counter[evidence_source] += 1

        if row.get("context_state") is not None:
            rows_with_context_state += 1
            context_state_counter[str(row["context_state"])] += 1
        if row.get("setup_state") is not None:
            rows_with_setup_state += 1
            setup_state_counter[str(row["setup_state"])] += 1
        if row.get("trigger_state") is not None:
            rows_with_trigger_state += 1
            trigger_state_counter[str(row["trigger_state"])] += 1
        if row.get("context_bias") is not None:
            rows_with_context_bias += 1
        if row.get("rule_bias") is not None:
            rows_with_rule_bias += 1
            rule_bias_counter[str(row["rule_bias"])] += 1
        if isinstance(row.get("bias_sign_source"), str):
            bias_sign_source_counter[str(row["bias_sign_source"])] += 1

        _collect_field_presence(
            row.get("selected_strategy_payload"),
            prefix="selected_strategy_payload",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("rule_engine_payload"),
            prefix="rule_engine",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("context_layer_payload"),
            prefix="timeframe_summary.context_layer",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("bias_layer_payload"),
            prefix="timeframe_summary.bias_layer",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("setup_layer_payload"),
            prefix="timeframe_summary.setup_layer",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("trigger_layer_payload"),
            prefix="timeframe_summary.trigger_layer",
            object_counter=object_counter,
            field_counter=field_counter,
        )
        _collect_field_presence(
            row.get("root_reason_text"),
            prefix="reason",
            object_counter=object_counter,
            field_counter=field_counter,
        )

    object_counter = _filter_inventory_counter(
        object_counter,
        allowed_prefixes=_ALLOWED_INVENTORY_PREFIXES,
    )
    field_counter = _filter_inventory_counter(
        field_counter,
        allowed_prefixes=_ALLOWED_INVENTORY_PREFIXES,
    )

    target_row_count = len(target_rows)
    return {
        "target_row_count": target_row_count,
        "selected_strategy_payload_source_count_rows": _counter_rows(
            selected_strategy_payload_source_counter,
            key_name="payload_source",
        ),
        "reason_text_source_count_rows": _counter_rows(
            reason_text_source_counter,
            key_name="reason_text_source",
            total=target_row_count,
            include_share=True,
        ),
        "evidence_source_count_rows": _counter_rows(
            evidence_source_counter,
            key_name="evidence_source",
            total=target_row_count,
            include_share=True,
        ),
        "bias_sign_source_count_rows": _counter_rows(
            bias_sign_source_counter,
            key_name="bias_sign_source",
            total=target_row_count,
            include_share=True,
        ),
        "object_presence_rows": _counter_rows(
            object_counter,
            key_name="object_path",
            total=target_row_count,
            include_share=True,
        ),
        "field_presence_rows": _counter_rows(
            field_counter,
            key_name="field_path",
            total=target_row_count,
            include_share=True,
        ),
        "rows_with_rule_reason_text": rows_with_rule_reason_text,
        "rows_with_root_reason_text": rows_with_root_reason_text,
        "rows_with_context_state": rows_with_context_state,
        "rows_with_setup_state": rows_with_setup_state,
        "rows_with_trigger_state": rows_with_trigger_state,
        "rows_with_context_bias": rows_with_context_bias,
        "rows_with_rule_bias": rows_with_rule_bias,
        "context_state_count_rows": _counter_rows(
            context_state_counter,
            key_name="context_state",
            total=target_row_count,
            include_share=True,
        ),
        "setup_state_count_rows": _counter_rows(
            setup_state_counter,
            key_name="setup_state",
            total=target_row_count,
            include_share=True,
        ),
        "trigger_state_count_rows": _counter_rows(
            trigger_state_counter,
            key_name="trigger_state",
            total=target_row_count,
            include_share=True,
        ),
        "rule_bias_count_rows": _counter_rows(
            rule_bias_counter,
            key_name="bias",
            total=target_row_count,
            include_share=True,
        ),
    }


def build_hold_resolution_reason_summary(
    *,
    target_rows: Sequence[dict[str, Any]],
    support_threshold: int,
) -> dict[str, Any]:
    bucket_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    reason_category_counter: Counter[str] = Counter()
    evidence_source_counter: Counter[str] = Counter()

    explicit_observable_condition_rows = 0
    structured_state_only_rows = 0
    structured_state_and_reason_text_rows = 0
    reason_text_only_classified_rows = 0
    no_classified_observable_evidence_rows = 0
    insufficient_persisted_explanation_rows = 0

    for row in target_rows:
        bucket = str(
            row.get("hold_resolution_condition_bucket")
            or _BUCKET_INSUFFICIENT_PERSISTED_EXPLANATION
        )
        bucket_counter[bucket] += 1

        has_structured = bool(row.get("hold_resolution_structured_observable_condition"))
        has_reason = bool(row.get("hold_resolution_reason_text_observable_condition"))
        evidence_source = str(
            row.get("hold_resolution_evidence_source")
            or "no_classified_observable_evidence"
        )
        evidence_source_counter[evidence_source] += 1

        if has_structured:
            explicit_observable_condition_rows += 1
            if has_reason:
                structured_state_and_reason_text_rows += 1
            else:
                structured_state_only_rows += 1
        elif has_reason:
            reason_text_only_classified_rows += 1
            insufficient_persisted_explanation_rows += 1
        else:
            no_classified_observable_evidence_rows += 1
            insufficient_persisted_explanation_rows += 1

        for flag in _safe_list(row.get("hold_resolution_observable_flags")):
            if isinstance(flag, str) and flag.strip():
                flag_counter[flag] += 1
        for category in _safe_list(row.get("hold_resolution_reason_text_categories")):
            if isinstance(category, str) and category.strip():
                reason_category_counter[category] += 1

    target_row_count = len(target_rows)
    return {
        "target_row_count": target_row_count,
        "explicit_observable_condition_rows": explicit_observable_condition_rows,
        "structured_state_backed_rows": explicit_observable_condition_rows,
        "structured_state_only_rows": structured_state_only_rows,
        "structured_state_and_reason_text_rows": structured_state_and_reason_text_rows,
        "reason_text_only_classified_rows": reason_text_only_classified_rows,
        "no_classified_observable_evidence_rows": no_classified_observable_evidence_rows,
        "insufficient_persisted_explanation_rows": (
            insufficient_persisted_explanation_rows
        ),
        "explicit_observable_condition_rate": _safe_ratio(
            explicit_observable_condition_rows,
            target_row_count,
        ),
        "reason_text_only_classified_rate": _safe_ratio(
            reason_text_only_classified_rows,
            target_row_count,
        ),
        "insufficient_persisted_explanation_rate": _safe_ratio(
            insufficient_persisted_explanation_rows,
            target_row_count,
        ),
        "condition_bucket_counts": dict(bucket_counter),
        "condition_bucket_count_rows": _counter_rows(
            bucket_counter,
            key_name="condition_bucket",
            total=target_row_count,
            include_share=True,
            order_map=_BUCKET_ORDER,
        ),
        "evidence_source_count_rows": _counter_rows(
            evidence_source_counter,
            key_name="evidence_source",
            total=target_row_count,
            include_share=True,
        ),
        "observable_condition_flag_count_rows": _counter_rows(
            flag_counter,
            key_name="condition_flag",
            total=target_row_count,
            include_share=True,
        ),
        "reason_text_category_count_rows": _counter_rows(
            reason_category_counter,
            key_name="reason_text_category",
            total=target_row_count,
            include_share=True,
        ),
        "dominant_observed_condition_bucket": _dominant_bucket(
            bucket_counter,
            empty="none",
        ),
        "primary_condition_bucket": _primary_bucket(
            bucket_counter,
            support_threshold=support_threshold,
            empty="no_target_rows",
        ),
        "support_status": (
            "supported"
            if target_row_count >= support_threshold
            else "limited_support"
        ),
    }


def build_hold_resolution_breakdown(
    *,
    target_rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    support_threshold: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in target_rows:
        group_values = [_group_value(row, field) for field in group_fields]
        if any(value is None for value in group_values):
            continue
        grouped.setdefault(tuple(str(value) for value in group_values), []).append(row)

    rows: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        summary = build_hold_resolution_reason_summary(
            target_rows=grouped_rows,
            support_threshold=support_threshold,
        )
        rows.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "target_row_count": summary["target_row_count"],
                "explicit_observable_condition_rows": summary[
                    "explicit_observable_condition_rows"
                ],
                "reason_text_only_classified_rows": summary[
                    "reason_text_only_classified_rows"
                ],
                "insufficient_persisted_explanation_rows": summary[
                    "insufficient_persisted_explanation_rows"
                ],
                "condition_bucket_count_rows": summary["condition_bucket_count_rows"],
                "dominant_observed_condition_bucket": summary[
                    "dominant_observed_condition_bucket"
                ],
                "primary_condition_bucket": summary["primary_condition_bucket"],
                "support_status": summary["support_status"],
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("target_row_count", 0) or 0),
            str(item.get("strategy") or ""),
            str(item.get("symbol") or ""),
            str(item.get("bias_sign") or ""),
        )
    )
    return rows


def build_final_assessment(
    configuration_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = list(configuration_summaries)
    if not summaries:
        return {
            "assessment": "no_configurations_evaluated",
            "primary_observed_hold_resolution_bucket": "none",
            "dominant_observed_condition_bucket": "none",
            "widest_configuration": None,
            "supported_strategy_primary_condition_buckets": {},
            "supported_strategy_condition_buckets": {},
            "strategy_level_consistency": "unknown",
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
    reason_summary = _safe_dict(widest.get("hold_resolution_reason_summary"))
    inventory = _safe_dict(widest.get("observable_decision_layer_inventory"))
    strategy_bucket_map = _supported_strategy_condition_buckets(
        widest.get("hold_resolution_by_strategy")
    )

    primary_bucket = str(
        reason_summary.get("primary_condition_bucket") or "no_target_rows"
    )
    dominant_bucket = str(
        reason_summary.get("dominant_observed_condition_bucket") or "none"
    )
    target_row_count = int(reason_summary.get("target_row_count", 0) or 0)

    return {
        "assessment": _overall_assessment_label(
            primary_bucket=primary_bucket,
            target_row_count=target_row_count,
        ),
        "primary_observed_hold_resolution_bucket": primary_bucket,
        "dominant_observed_condition_bucket": dominant_bucket,
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "supported_strategy_primary_condition_buckets": (
            _supported_strategy_primary_condition_buckets(
                widest.get("hold_resolution_by_strategy")
            )
        ),
        "supported_strategy_condition_buckets": strategy_bucket_map,
        "strategy_level_consistency": _strategy_level_consistency(strategy_bucket_map),
        "confirmed_observations": _build_final_confirmed_observations(
            widest=widest,
            reason_summary=reason_summary,
            inventory=inventory,
            strategy_bucket_map=strategy_bucket_map,
        ),
        "evidence_backed_inferences": _build_final_evidence_backed_inferences(
            reason_summary=reason_summary,
            strategy_bucket_map=strategy_bucket_map,
        ),
        "unresolved_uncertainties": _build_final_unresolved_uncertainties(
            inventory=inventory,
            reason_summary=reason_summary,
        ),
        "overall_conclusion": _build_overall_conclusion(
            widest=widest,
            reason_summary=reason_summary,
            strategy_bucket_map=strategy_bucket_map,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('report_title', REPORT_TITLE)}",
        "",
        f"- Generated at: {report.get('generated_at', _MISSING_LABEL)}",
        f"- Input path: {_safe_dict(report.get('inputs')).get('input_path', _MISSING_LABEL)}",
        f"- Output dir: {_safe_dict(report.get('inputs')).get('output_dir', _MISSING_LABEL)}",
        "",
        "## Final Assessment",
    ]

    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.extend(
        [
            f"- Assessment: {final_assessment.get('assessment', 'none')}",
            (
                "- Primary observed hold-resolution bucket: "
                f"{final_assessment.get('primary_observed_hold_resolution_bucket', 'none')}"
            ),
            (
                "- Dominant observed condition bucket: "
                f"{final_assessment.get('dominant_observed_condition_bucket', 'none')}"
            ),
            (
                "- Strategy-level consistency: "
                f"{final_assessment.get('strategy_level_consistency', 'unknown')}"
            ),
            "",
            "### Confirmed Observations",
        ]
    )
    for item in _safe_list(final_assessment.get("confirmed_observations")):
        lines.append(f"- {item}")

    lines.extend(["", "### Evidence-Backed Inferences"])
    for item in _safe_list(final_assessment.get("evidence_backed_inferences")):
        lines.append(f"- {item}")

    lines.extend(["", "### Unresolved Uncertainties"])
    for item in _safe_list(final_assessment.get("unresolved_uncertainties")):
        lines.append(f"- {item}")

    for summary in _safe_list(report.get("configuration_summaries")):
        summary_dict = _safe_dict(summary)
        configuration = _safe_dict(summary_dict.get("configuration"))
        reason_summary = _safe_dict(summary_dict.get("hold_resolution_reason_summary"))
        lines.extend(
            [
                "",
                f"## Configuration: {configuration.get('display_name', _MISSING_LABEL)}",
                f"- Target rows: {reason_summary.get('target_row_count', 0)}",
                (
                    "- Dominant observed condition bucket: "
                    f"{reason_summary.get('dominant_observed_condition_bucket', 'none')}"
                ),
                (
                    "- Primary condition bucket: "
                    f"{reason_summary.get('primary_condition_bucket', 'none')}"
                ),
                (
                    "- Explicit observable rows: "
                    f"{reason_summary.get('explicit_observable_condition_rows', 0)}"
                ),
                (
                    "- Reason-text-only classified rows: "
                    f"{reason_summary.get('reason_text_only_classified_rows', 0)}"
                ),
                (
                    "- Condition bucket rows: "
                    f"{_format_counter_rows(reason_summary.get('condition_bucket_count_rows'), 'condition_bucket')}"
                ),
                (
                    "- Strategy breakdown: "
                    f"{_format_breakdown_rows(summary_dict.get('hold_resolution_by_strategy'))}"
                ),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def write_report_files(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    md_path = output_dir / REPORT_MD_NAME
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }


def _build_stage_row(raw_record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_record(raw_record)
    strategy = _normalize_strategy(
        raw_record.get("selected_strategy") or normalized.get("selected_strategy")
    )
    (
        selected_strategy_payload,
        selected_strategy_payload_source,
    ) = _selected_strategy_result_payload(
        raw_record=raw_record,
        strategy=strategy,
    )
    rule_engine_payload = _as_dict(raw_record.get("rule_engine"))
    timeframe_summary_payload = _as_dict(raw_record.get("timeframe_summary"))
    context_layer = _as_dict(timeframe_summary_payload.get("context_layer"))
    bias_layer = _as_dict(timeframe_summary_payload.get("bias_layer"))
    setup_layer = _as_dict(timeframe_summary_payload.get("setup_layer"))
    trigger_layer = _as_dict(timeframe_summary_payload.get("trigger_layer"))

    selected_label = _normalize_signal(selected_strategy_payload.get("signal"))
    rule_label = _normalize_signal(rule_engine_payload.get("signal"))
    selected_state = _surface_signal_state(selected_label)
    rule_state = _surface_signal_state(rule_label)

    raw_bias = _normalize_bias(raw_record.get("bias"))
    context_bias = _normalize_bias(
        context_layer.get("bias") or bias_layer.get("bias")
    )
    rule_bias = _normalize_bias(rule_engine_payload.get("bias"))
    bias_sign, bias_sign_source = _derive_bias_sign(
        rule_bias=rule_bias,
        context_bias=context_bias,
        raw_bias=raw_bias,
    )

    row = {
        **normalized,
        "strategy": strategy,
        "symbol": _normalize_symbol(raw_record.get("symbol") or normalized.get("symbol")),
        "top_level_bias": raw_bias,
        "bias": raw_bias,
        "bias_sign": bias_sign,
        "bias_sign_source": bias_sign_source,
        "selected_strategy_payload": selected_strategy_payload,
        "selected_strategy_payload_source": selected_strategy_payload_source,
        "selected_strategy_result_signal_label": selected_label,
        "selected_strategy_result_signal_state": selected_state,
        "rule_engine_payload": rule_engine_payload,
        "rule_signal_label": rule_label,
        "rule_signal_state": rule_state,
        "root_reason_text": _clean_text(raw_record.get("reason")),
        "rule_reason_text": _clean_text(rule_engine_payload.get("reason")),
        "timeframe_summary_payload": timeframe_summary_payload,
        "context_layer_payload": context_layer,
        "bias_layer_payload": bias_layer,
        "setup_layer_payload": setup_layer,
        "trigger_layer_payload": trigger_layer,
        "context_state": _normalize_text(
            context_layer.get("context") or bias_layer.get("context")
        ),
        "context_bias": context_bias,
        "setup_state": _normalize_text(setup_layer.get("setup")),
        "trigger_state": _normalize_text(trigger_layer.get("trigger")),
        "rule_bias": rule_bias,
    }
    row.update(_build_hold_resolution_condition_details(row))
    return row


def _selected_strategy_result_payload(
    *,
    raw_record: dict[str, Any],
    strategy: str | None,
) -> tuple[dict[str, Any], str | None]:
    if strategy is not None:
        strategy_payload = _as_dict(raw_record.get(f"{strategy}_result"))
        if strategy_payload:
            return strategy_payload, f"{strategy}_result"

    explicit_selected_payload = _as_dict(raw_record.get("selected_strategy_result"))
    if not explicit_selected_payload:
        return {}, None

    safe_payload = _safe_explicit_selected_strategy_result_payload(
        payload=explicit_selected_payload,
        strategy=strategy,
    )
    if safe_payload:
        return safe_payload, "selected_strategy_result"
    return {}, None


def _safe_explicit_selected_strategy_result_payload(
    *,
    payload: dict[str, Any],
    strategy: str | None,
) -> dict[str, Any]:
    if strategy is None:
        return {}

    payload_strategy = _normalize_strategy(payload.get("strategy"))
    payload_selected_strategy = _normalize_strategy(payload.get("selected_strategy"))

    if payload_selected_strategy is not None:
        return {}

    if payload_strategy is None:
        if any(key in payload for key in _COMPOSED_DECISION_ONLY_KEYS):
            return {}
        return {}

    if payload_strategy != strategy:
        return {}

    return payload


def _build_hold_resolution_condition_details(
    row: dict[str, Any],
) -> dict[str, Any]:
    selected_state = row.get("selected_strategy_result_signal_state")
    rule_state = row.get("rule_signal_state")
    target_row = selected_state in _ACTIONABLE_SIGNAL_STATES and rule_state == "hold"
    if not target_row:
        return {
            "selected_strategy_hold_resolution_target_row": False,
            "hold_resolution_condition_bucket": None,
            "hold_resolution_explicit_observable_condition": False,
            "hold_resolution_structured_observable_condition": False,
            "hold_resolution_reason_text_observable_condition": False,
            "hold_resolution_evidence_source": "not_target_row",
            "hold_resolution_observable_flags": [],
            "hold_resolution_reason_text_categories": [],
            "hold_resolution_structured_condition_categories": [],
            "hold_resolution_condition_categories": [],
        }

    observable_flags: list[str] = []
    reason_text_categories: set[str] = set()
    structured_category_set: set[str] = set()

    rule_reason_text = row.get("rule_reason_text")
    root_reason_text = row.get("root_reason_text")

    if rule_reason_text is not None:
        observable_flags.append("rule_reason_text_present")
    if root_reason_text is not None:
        observable_flags.append("root_reason_text_present")

    context_state = _normalize_text(row.get("context_state"))
    setup_state = _normalize_text(row.get("setup_state"))
    trigger_state = _normalize_text(row.get("trigger_state"))
    context_bias = _normalize_bias(row.get("context_bias"))
    rule_bias = _normalize_bias(row.get("rule_bias"))

    if context_state is not None:
        observable_flags.append("context_state_present")
    if setup_state is not None:
        observable_flags.append("setup_state_present")
    if trigger_state is not None:
        observable_flags.append("trigger_state_present")
    if context_bias is not None:
        observable_flags.append("context_bias_present")
    if rule_bias is not None:
        observable_flags.append("rule_bias_present")

    if _text_contains_any(context_state, _STATE_CONFLICT_MARKERS) or rule_bias == "neutral_conflict":
        structured_category_set.add(_BUCKET_EXPLICIT_CONTEXT_CONFLICT)
        if _text_contains_any(context_state, _STATE_CONFLICT_MARKERS):
            observable_flags.append("context_state_conflicted")
        if rule_bias == "neutral_conflict":
            observable_flags.append("rule_bias_neutral_conflict")

    neutral_state_detected = False
    if _text_contains_any(context_state, _STATE_NEUTRALITY_MARKERS):
        neutral_state_detected = True
        observable_flags.append("context_state_neutral_like")
    elif context_state is None and rule_bias == "neutral":
        neutral_state_detected = True
        observable_flags.append("rule_bias_neutral")
    if neutral_state_detected:
        structured_category_set.add(_BUCKET_EXPLICIT_CONTEXT_NEUTRALITY)

    context_direction = _direction_from_bias(context_bias)
    if context_direction is not None and context_direction != selected_state:
        structured_category_set.add(_BUCKET_EXPLICIT_DIRECTIONAL_OPPOSITION)
        observable_flags.append("context_bias_opposes_selected_signal")

    alignment_rows: list[tuple[str, str]] = []
    for layer_name, layer_state in (("setup", setup_state), ("trigger", trigger_state)):
        alignment = _layer_alignment(layer_state, str(selected_state))
        alignment_rows.append((layer_name, alignment))
        if alignment == "confirmed":
            observable_flags.append(f"{layer_name}_state_confirms_selected_signal")
        elif alignment == "same_direction_but_not_fully_confirmed":
            observable_flags.append(
                f"{layer_name}_state_same_direction_not_fully_confirmed"
            )
        elif alignment in {"neutral_or_missing_direction", "missing"}:
            observable_flags.append(f"{layer_name}_state_neutral_or_missing_direction")
        elif alignment == "opposed":
            observable_flags.append(f"{layer_name}_state_opposes_selected_signal")

    same_direction_confirmed = sum(
        1 for _, alignment in alignment_rows if alignment == "confirmed"
    )
    same_direction_weak = sum(
        1
        for _, alignment in alignment_rows
        if alignment == "same_direction_but_not_fully_confirmed"
    )
    neutral_or_missing = sum(
        1
        for _, alignment in alignment_rows
        if alignment in {"neutral_or_missing_direction", "missing"}
    )
    opposed = sum(1 for _, alignment in alignment_rows if alignment == "opposed")

    if opposed > 0:
        structured_category_set.add(_BUCKET_EXPLICIT_DIRECTIONAL_OPPOSITION)

    if same_direction_weak > 0:
        structured_category_set.add(_BUCKET_EXPLICIT_CONFIRMATION_GAP)
    elif same_direction_confirmed > 0 and neutral_or_missing > 0:
        structured_category_set.add(_BUCKET_EXPLICIT_CONFIRMATION_GAP)

    reason_category_set: set[str] = set()
    for source_name, reason_text in (
        ("rule_reason_text", rule_reason_text),
        ("root_reason_text", root_reason_text),
    ):
        if reason_text is None:
            continue
        reason_matches = _classify_reason_text(reason_text)
        if reason_matches:
            for category_name in sorted(reason_matches):
                reason_text_categories.add(category_name)
                observable_flags.append(f"{source_name}_mentions_{category_name}")
            reason_category_set.update(reason_matches)
        else:
            reason_text_categories.add("unclassified")
            observable_flags.append(f"{source_name}_unclassified")

    has_structured = bool(structured_category_set)
    has_reason = bool(reason_category_set)

    ordered_structured_categories = sorted(
        structured_category_set,
        key=lambda value: _BUCKET_ORDER[value],
    )
    if not ordered_structured_categories:
        bucket = _BUCKET_INSUFFICIENT_PERSISTED_EXPLANATION
    elif len(ordered_structured_categories) > 1:
        bucket = _BUCKET_MIXED_OR_INCONCLUSIVE
    else:
        bucket = ordered_structured_categories[0]

    evidence_source = _hold_resolution_evidence_source(
        has_structured=has_structured,
        has_reason=has_reason,
    )

    return {
        "selected_strategy_hold_resolution_target_row": True,
        "hold_resolution_condition_bucket": bucket,
        "hold_resolution_explicit_observable_condition": has_structured,
        "hold_resolution_structured_observable_condition": has_structured,
        "hold_resolution_reason_text_observable_condition": has_reason,
        "hold_resolution_evidence_source": evidence_source,
        "hold_resolution_observable_flags": sorted(set(observable_flags)),
        "hold_resolution_reason_text_categories": sorted(reason_text_categories),
        "hold_resolution_structured_condition_categories": ordered_structured_categories,
        "hold_resolution_condition_categories": ordered_structured_categories,
    }


def _build_configuration_confirmed_observations(
    *,
    configuration: DiagnosisConfiguration,
    hold_resolution_targeting: dict[str, Any],
    observable_decision_layer_inventory: dict[str, Any],
    hold_resolution_reason_summary: dict[str, Any],
    hold_resolution_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    target_rows = int(hold_resolution_reason_summary.get("target_row_count", 0) or 0)
    explicit_rows = int(
        hold_resolution_reason_summary.get("explicit_observable_condition_rows", 0)
        or 0
    )
    reason_text_only_rows = int(
        hold_resolution_reason_summary.get("reason_text_only_classified_rows", 0) or 0
    )
    insufficient_rows = int(
        hold_resolution_reason_summary.get("insufficient_persisted_explanation_rows", 0)
        or 0
    )
    dominant_bucket = str(
        hold_resolution_reason_summary.get("dominant_observed_condition_bucket") or "none"
    )
    observations = [
        (
            f"At {configuration.display_name}, {target_rows} rows met the target "
            "condition: selected_strategy_result.signal was actionable while "
            "rule_engine.signal was hold."
        ),
        (
            f"{explicit_rows} target rows carried structured-state-backed observable "
            f"condition buckets, while {insufficient_rows} rows remained "
            "insufficient_persisted_explanation."
        ),
    ]

    if reason_text_only_rows > 0:
        observations.append(
            f"{reason_text_only_rows} target rows had classified reason text without enough structured-state support to promote them out of insufficient_persisted_explanation."
        )

    top_fields = [
        _safe_dict(item).get("field_path")
        for item in _safe_list(observable_decision_layer_inventory.get("field_presence_rows"))
        [:5]
    ]
    visible_fields = [str(item) for item in top_fields if isinstance(item, str)]
    if visible_fields:
        observations.append(
            "Observed decision-layer inventory on target rows emphasized "
            + ", ".join(visible_fields)
            + "."
        )

    if dominant_bucket != "none":
        observations.append(
            "The dominant structured hold-resolution bucket on the target rows was "
            f"{dominant_bucket}."
        )

    strategy_bucket_map = _supported_strategy_condition_buckets(hold_resolution_by_strategy)
    if strategy_bucket_map:
        rendered = ", ".join(
            f"{strategy}={bucket}"
            for strategy, bucket in sorted(strategy_bucket_map.items())
        )
        observations.append(
            "Supported strategy slices resolved to: " + rendered + "."
        )

    actionable_rows = int(
        hold_resolution_targeting.get("selected_strategy_actionable_rows", 0) or 0
    )
    if actionable_rows > 0:
        observations.append(
            f"The targeting summary observed {actionable_rows} actionable selected-strategy "
            f"rows, of which {target_rows} collapsed specifically to rule_engine.signal=hold."
        )

    return observations


def _build_configuration_evidence_backed_inferences(
    *,
    hold_resolution_reason_summary: dict[str, Any],
    hold_resolution_by_strategy: Sequence[dict[str, Any]],
) -> list[str]:
    target_rows = int(hold_resolution_reason_summary.get("target_row_count", 0) or 0)
    if target_rows <= 0:
        return [
            "No target rows were observed, so no evidence-backed hold-resolution inference is available for this configuration."
        ]

    primary_bucket = str(
        hold_resolution_reason_summary.get("primary_condition_bucket") or "no_target_rows"
    )
    explicit_rate = _to_float(
        hold_resolution_reason_summary.get("explicit_observable_condition_rate"),
        default=0.0,
    )
    reason_text_only_rows = int(
        hold_resolution_reason_summary.get("reason_text_only_classified_rows", 0) or 0
    )
    strategy_bucket_map = _supported_strategy_condition_buckets(hold_resolution_by_strategy)

    inferences = [
        "Because the target rows are already restricted to actionable selected-strategy proposals that ended at rule_engine.signal=hold, the observed buckets remain evidence about decision-layer hold resolution rather than proposal scarcity or execution-layer action mapping.",
    ]

    if primary_bucket not in _NON_PRIMARY_BUCKET_VALUES:
        inferences.append(
            f"The structured persisted evidence is more consistent with {primary_bucket} dominating the observed hold collapses than with a single downstream execution explanation."
        )
    if explicit_rate >= _PRIMARY_FACTOR_THRESHOLD:
        inferences.append(
            "A majority of target rows retained structured-state-backed evidence, which means the report is not relying only on mirrored reason text for its dominant bucket."
        )
    if reason_text_only_rows > 0:
        inferences.append(
            "Some target rows still depended on reason-text-only evidence and therefore remained insufficient_persisted_explanation, which keeps the interpretation conservative."
        )
    if _strategy_level_consistency(strategy_bucket_map) == "split":
        inferences.append(
            "Strategy-level supported slices do not fully agree on one bucket, so the dominant overall pattern should not be overgeneralized as a universal single-mechanism explanation."
        )

    return inferences


def _build_configuration_unresolved_uncertainties(
    *,
    observable_decision_layer_inventory: dict[str, Any],
    hold_resolution_reason_summary: dict[str, Any],
) -> list[str]:
    insufficient_rows = int(
        hold_resolution_reason_summary.get("insufficient_persisted_explanation_rows", 0)
        or 0
    )
    target_rows = int(hold_resolution_reason_summary.get("target_row_count", 0) or 0)
    uncertainties = [
        "The report cannot prove the exact internal rule, threshold, or conflict-resolution branch that turned the selected-strategy proposal into rule_engine.signal=hold.",
        "The persisted evidence does not by itself distinguish intended conservatism from unintended neutralization.",
    ]

    if insufficient_rows > 0:
        uncertainties.append(
            f"{insufficient_rows} of {target_rows} target rows lacked enough structured-state evidence to attribute the hold outcome to one explicit observable bucket."
        )

    if int(observable_decision_layer_inventory.get("rows_with_rule_reason_text", 0) or 0) == 0:
        uncertainties.append(
            "No explicit rule_engine.reason text was observed on the target rows in this configuration, so explanation depends more heavily on neighboring state fields when available."
        )

    return uncertainties


def _build_final_confirmed_observations(
    *,
    widest: dict[str, Any],
    reason_summary: dict[str, Any],
    inventory: dict[str, Any],
    strategy_bucket_map: dict[str, str],
) -> list[str]:
    configuration = _safe_dict(widest.get("configuration"))
    target_rows = int(reason_summary.get("target_row_count", 0) or 0)
    explicit_rows = int(reason_summary.get("explicit_observable_condition_rows", 0) or 0)
    reason_text_only_rows = int(
        reason_summary.get("reason_text_only_classified_rows", 0) or 0
    )
    insufficient_rows = int(
        reason_summary.get("insufficient_persisted_explanation_rows", 0) or 0
    )
    dominant_bucket = str(reason_summary.get("dominant_observed_condition_bucket") or "none")
    field_rows = _safe_list(inventory.get("field_presence_rows"))
    top_fields = [str(_safe_dict(item).get("field_path")) for item in field_rows[:5]]
    visible_fields = [item for item in top_fields if item and item != "None"]

    observations = [
        (
            f"The widest supported configuration {configuration.get('display_name', _MISSING_LABEL)} "
            f"contained {target_rows} target rows where actionable selected-strategy proposals "
            "collapsed to rule_engine.signal=hold."
        ),
        (
            f"{explicit_rows} of those target rows exposed structured-state-backed observable "
            f"condition buckets, while {insufficient_rows} rows remained insufficient_persisted_explanation."
        ),
    ]
    if reason_text_only_rows > 0:
        observations.append(
            f"{reason_text_only_rows} target rows had classified reason text without enough structured-state support to count as explicit observable hold-resolution evidence."
        )
    if dominant_bucket != "none":
        observations.append(
            f"The dominant structured condition bucket at the widest configuration was {dominant_bucket}."
        )
    if visible_fields:
        observations.append(
            "The most frequently observed decision-adjacent persisted field paths on target rows were "
            + ", ".join(visible_fields)
            + "."
        )
    if strategy_bucket_map:
        rendered = ", ".join(
            f"{strategy}={bucket}" for strategy, bucket in sorted(strategy_bucket_map.items())
        )
        observations.append(
            "Supported strategy-level bucket outcomes were " + rendered + "."
        )
    return observations


def _build_final_evidence_backed_inferences(
    *,
    reason_summary: dict[str, Any],
    strategy_bucket_map: dict[str, str],
) -> list[str]:
    target_rows = int(reason_summary.get("target_row_count", 0) or 0)
    if target_rows <= 0:
        return [
            "No target rows were available at the widest configuration, so no evidence-backed inference can be drawn from this report."
        ]

    primary_bucket = str(reason_summary.get("primary_condition_bucket") or "no_target_rows")
    explicit_rate = _to_float(
        reason_summary.get("explicit_observable_condition_rate"),
        default=0.0,
    )
    reason_text_only_rows = int(
        reason_summary.get("reason_text_only_classified_rows", 0) or 0
    )

    inferences = [
        "The observed hold collapses remain an upstream decision-layer phenomenon because this report never substitutes final execution action for rule_engine.signal.",
    ]
    if primary_bucket not in _NON_PRIMARY_BUCKET_VALUES:
        inferences.append(
            f"The widest-configuration evidence is more consistent with {primary_bucket} dominating the structured hold-resolution path than with purely downstream execution or reporting artifacts."
        )
    if explicit_rate >= _PRIMARY_FACTOR_THRESHOLD:
        inferences.append(
            "A majority of target rows still carrying structured-state-backed evidence suggests the dominant bucket is not just a byproduct of reason-text templating."
        )
    if reason_text_only_rows > 0:
        inferences.append(
            "Reason-text-only rows remain in insufficient_persisted_explanation, which keeps the report from overclaiming complete observability."
        )
    if _strategy_level_consistency(strategy_bucket_map) == "split":
        inferences.append(
            "Strategy-level supported slices split across buckets, so the overall dominant bucket should be read as a leading pattern rather than a universal single cause."
        )
    return inferences


def _build_final_unresolved_uncertainties(
    *,
    inventory: dict[str, Any],
    reason_summary: dict[str, Any],
) -> list[str]:
    insufficient_rows = int(
        reason_summary.get("insufficient_persisted_explanation_rows", 0) or 0
    )
    target_rows = int(reason_summary.get("target_row_count", 0) or 0)
    uncertainties = [
        "The report does not prove whether the observed hold-resolution bucket reflects intended conservatism, confirmation thresholds, neutrality normalization, or conflict resolution semantics inside the live decision path.",
        "The report does not prove whether one common internal mechanism or several strategy-specific mechanisms produced the observed hold collapses.",
    ]
    if insufficient_rows > 0:
        uncertainties.append(
            f"{insufficient_rows} of {target_rows} target rows still lack enough structured-state evidence for a bucket stronger than insufficient_persisted_explanation."
        )
    if int(inventory.get("rows_with_rule_reason_text", 0) or 0) == 0:
        uncertainties.append(
            "Without explicit rule_engine.reason text on the observed target rows, the persisted evidence is limited to neighboring state fields when available."
        )
    return uncertainties


def _build_overall_conclusion(
    *,
    widest: dict[str, Any],
    reason_summary: dict[str, Any],
    strategy_bucket_map: dict[str, str],
) -> str:
    configuration = _safe_dict(widest.get("configuration"))
    primary_bucket = str(reason_summary.get("primary_condition_bucket") or "no_target_rows")
    target_rows = int(reason_summary.get("target_row_count", 0) or 0)
    insufficient_rows = int(
        reason_summary.get("insufficient_persisted_explanation_rows", 0) or 0
    )
    strategy_consistency = _strategy_level_consistency(strategy_bucket_map)

    if target_rows <= 0:
        return (
            "No target rows were observed, so this artifact cannot diagnose hold-resolution "
            "conditions for actionable selected-strategy proposals at the evaluated widest configuration."
        )

    if primary_bucket in _NON_PRIMARY_BUCKET_VALUES:
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            "the structured persisted evidence for actionable selected-strategy -> hold collapses remained "
            f"{primary_bucket}, and {insufficient_rows} rows still lacked enough structured-state explanation."
        )

    if strategy_consistency == "split" and strategy_bucket_map:
        rendered = ", ".join(
            f"{strategy}={bucket}" for strategy, bucket in sorted(strategy_bucket_map.items())
        )
        return (
            f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
            "actionable selected-strategy -> rule_signal=hold collapses were most consistently "
            f"associated with {primary_bucket} overall, but supported strategy slices remained split "
            f"({rendered}), while {insufficient_rows} rows still lacked enough structured-state explanation "
            "to prove a single universal internal mechanism."
        )

    return (
        f"At the widest configuration {configuration.get('display_name', _MISSING_LABEL)}, "
        "actionable selected-strategy -> rule_signal=hold collapses were most consistently "
        f"associated with {primary_bucket} on structured persisted decision-layer fields, while "
        f"{insufficient_rows} rows still lacked enough structured-state explanation to prove a single exact internal mechanism."
    )


def _dominant_bucket(counter: Counter[str], *, empty: str) -> str:
    if not counter:
        return empty
    return min(
        counter.items(),
        key=lambda item: (
            -item[1],
            _BUCKET_ORDER.get(item[0], 99),
            item[0],
        ),
    )[0]


def _primary_bucket(
    counter: Counter[str],
    *,
    support_threshold: int,
    empty: str,
) -> str:
    total = sum(counter.values())
    if total == 0:
        return empty
    if total < support_threshold:
        return "insufficient_support"
    dominant_count = max(counter.values())
    dominant = [key for key, value in counter.items() if value == dominant_count]
    if len(dominant) > 1 or _safe_ratio(dominant_count, total) < _PRIMARY_FACTOR_THRESHOLD:
        return _BUCKET_MIXED_OR_INCONCLUSIVE
    dominant.sort(key=lambda value: (_BUCKET_ORDER.get(value, 99), value))
    return dominant[0]


def _overall_assessment_label(primary_bucket: str, target_row_count: int) -> str:
    if target_row_count < _MIN_PRIMARY_TARGET_SUPPORT_ROWS:
        return "insufficient_support"
    if primary_bucket in _NON_PRIMARY_BUCKET_VALUES:
        return primary_bucket
    return f"{primary_bucket}_primary"


def _supported_strategy_primary_condition_buckets(rows: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _safe_list(rows):
        row_dict = _safe_dict(row)
        strategy = _normalize_strategy(row_dict.get("strategy"))
        if strategy is None:
            continue
        if row_dict.get("support_status") != "supported":
            continue

        primary_bucket = str(row_dict.get("primary_condition_bucket") or "").strip()
        if primary_bucket in _NON_PRIMARY_BUCKET_VALUES:
            continue

        result[strategy] = primary_bucket
    return result


def _supported_strategy_condition_buckets(rows: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _safe_list(rows):
        row_dict = _safe_dict(row)
        strategy = _normalize_strategy(row_dict.get("strategy"))
        if strategy is None:
            continue
        if row_dict.get("support_status") != "supported":
            continue
        bucket = str(row_dict.get("primary_condition_bucket") or "").strip()
        if not bucket:
            continue
        result[strategy] = bucket
    return result


def _strategy_level_consistency(strategy_bucket_map: dict[str, str]) -> str:
    if not strategy_bucket_map:
        return "unknown"
    unique_buckets = {value for value in strategy_bucket_map.values() if value}
    if len(unique_buckets) <= 1:
        return "aligned"
    return "split"


def _group_value(row: dict[str, Any], field: str) -> str | None:
    if field == "strategy":
        return _normalize_strategy(row.get("strategy") or row.get("selected_strategy"))
    if field == "symbol":
        return _normalize_symbol(row.get("symbol"))
    if field == "bias_sign":
        value = row.get("bias_sign")
        return str(value) if isinstance(value, str) and value.strip() else None
    raise ValueError(f"Unsupported group field: {field}")


def _has_known_identity(row: dict[str, Any]) -> bool:
    return (
        _normalize_symbol(row.get("symbol")) is not None
        and _normalize_strategy(row.get("strategy") or row.get("selected_strategy"))
        is not None
    )


def _derive_bias_sign(
    *,
    rule_bias: str | None,
    context_bias: str | None,
    raw_bias: str | None,
) -> tuple[str | None, str | None]:
    for source_name, value in (
        ("rule_engine.bias", rule_bias),
        ("timeframe_summary.context_bias", context_bias),
        ("top_level.bias", raw_bias),
    ):
        direction = _direction_from_bias(value)
        if direction == "long":
            return "bullish", source_name
        if direction == "short":
            return "bearish", source_name
    return None, None


def _surface_signal_state(value: str | None) -> str | None:
    if value is None:
        return None
    if value in _LONG_SIGNAL_VALUES:
        return "long"
    if value in _SHORT_SIGNAL_VALUES:
        return "short"
    if value in _HOLD_SIGNAL_VALUES:
        return "hold"
    if value in _WATCHLIST_LONG_SIGNAL_VALUES:
        return "watchlist_long"
    if value in _WATCHLIST_SHORT_SIGNAL_VALUES:
        return "watchlist_short"
    if value in _WATCHLIST_SIGNAL_VALUES:
        return "watchlist"
    if value in _NO_SIGNAL_VALUES:
        return "no_signal"
    if value in _UNKNOWN_SIGNAL_VALUES:
        return "unknown"
    return "other"


def _display_signal_state(value: str | None) -> str:
    return value or _MISSING_LABEL


def _direction_from_bias(value: str | None) -> str | None:
    bias = _normalize_bias(value)
    if bias is None:
        return None
    if bias in _BULLISH_BIAS_VALUES:
        return "long"
    if bias in _BEARISH_BIAS_VALUES:
        return "short"
    return None


def _layer_alignment(value: str | None, selected_signal: str) -> str:
    if value is None:
        return "missing"
    text = _normalize_text(value)
    if text is None:
        return "missing"
    direction = _direction_from_text(text)
    if direction is None:
        return "neutral_or_missing_direction"
    if direction != selected_signal:
        return "opposed"
    if _text_contains_any(text, _WEAK_CONFIRMATION_MARKERS):
        return "same_direction_but_not_fully_confirmed"
    if text == selected_signal or _text_contains_any(text, _STRONG_CONFIRMATION_MARKERS):
        return "confirmed"
    return "same_direction_but_not_fully_confirmed"


def _direction_from_text(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    if "long" in text or "bullish" in text or text == "buy":
        return "long"
    if "short" in text or "bearish" in text or text == "sell":
        return "short"
    return None


def _classify_reason_text(text: str) -> set[str]:
    normalized = _normalize_text(text)
    if normalized is None:
        return set()
    categories: set[str] = set()
    if _text_contains_any(normalized, _REASON_CONFLICT_MARKERS):
        categories.add("conflict")
    if _text_contains_any(normalized, _REASON_NEUTRALITY_MARKERS):
        categories.add("neutrality")
    if _text_contains_any(normalized, _REASON_CONFIRMATION_MARKERS):
        categories.add("confirmation_gap")
    if _text_contains_any(normalized, _REASON_DIRECTIONAL_OPPOSITION_MARKERS):
        categories.add("directional_opposition")
    return categories


def _hold_resolution_evidence_source(
    *,
    has_structured: bool,
    has_reason: bool,
) -> str:
    if has_structured and has_reason:
        return "structured_state_and_reason_text"
    if has_structured:
        return "structured_state_only"
    if has_reason:
        return "reason_text_only"
    return "no_classified_observable_evidence"


def _text_contains_any(text: str | None, markers: Sequence[str]) -> bool:
    if text is None:
        return False
    return any(marker in text for marker in markers)


def _collect_field_presence(
    value: Any,
    *,
    prefix: str,
    object_counter: Counter[str],
    field_counter: Counter[str],
) -> None:
    if isinstance(value, dict):
        if not value:
            return
        object_counter[prefix] += 1
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            if isinstance(child, dict):
                _collect_field_presence(
                    child,
                    prefix=child_prefix,
                    object_counter=object_counter,
                    field_counter=field_counter,
                )
                continue
            if isinstance(child, list):
                if child:
                    field_counter[child_prefix] += 1
                continue
            if _has_meaningful_value(child):
                field_counter[child_prefix] += 1
        return

    if _has_meaningful_value(value):
        field_counter[prefix] += 1


def _filter_inventory_counter(
    counter: Counter[str],
    *,
    allowed_prefixes: Sequence[str],
) -> Counter[str]:
    filtered: Counter[str] = Counter()
    for key, value in counter.items():
        key_text = str(key)
        if any(
            key_text == prefix or key_text.startswith(f"{prefix}.")
            for prefix in allowed_prefixes
        ):
            filtered[key_text] = value
    return filtered


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


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


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _format_counter_rows(value: Any, key_name: str) -> str:
    parts: list[str] = []
    for item in _safe_list(value):
        row = _safe_dict(item)
        label = row.get(key_name)
        count = row.get("count")
        share = row.get("share")
        if share is None:
            parts.append(f"{label}={count}")
        else:
            parts.append(f"{label}={count} ({share:.2%})")
    return ", ".join(parts) if parts else "none"


def _format_breakdown_rows(value: Any) -> str:
    parts: list[str] = []
    for item in _safe_list(value)[:5]:
        row = _safe_dict(item)
        strategy = row.get("strategy")
        symbol = row.get("symbol")
        bias_sign = row.get("bias_sign")
        bucket = row.get("primary_condition_bucket")
        count = row.get("target_row_count")
        label_parts = [str(strategy)]
        if symbol is not None:
            label_parts.append(str(symbol))
        if bias_sign is not None:
            label_parts.append(str(bias_sign))
        parts.append(f"{'/'.join(label_parts)}={bucket} ({count})")
    return ", ".join(parts) if parts else "none"


def _counter_rows(
    counter: Counter[str],
    *,
    key_name: str,
    total: int | None = None,
    include_share: bool = False,
    order_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        {
            key_name: key,
            "count": value,
            **(
                {"share": _safe_ratio(value, total)}
                if include_share and isinstance(total, int) and total > 0
                else {}
            ),
        }
        for key, value in counter.items()
    ]
    rows.sort(
        key=lambda item: (
            -int(item.get("count", 0) or 0),
            (
                (order_map or {}).get(str(item.get(key_name)), 99)
                if order_map is not None
                else 0
            ),
            str(item.get(key_name) or ""),
        )
    )
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _to_float(value: Any, *, default: float) -> float:
    try:
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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


if __name__ == "__main__":
    main()