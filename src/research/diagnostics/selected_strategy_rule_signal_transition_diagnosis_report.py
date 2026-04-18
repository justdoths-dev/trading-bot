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

REPORT_TYPE = "selected_strategy_rule_signal_transition_diagnosis_report"
REPORT_TITLE = "Selected Strategy Rule Signal Transition Diagnosis Report"
REPORT_JSON_NAME = f"{REPORT_TYPE}.json"
REPORT_MD_NAME = f"{REPORT_TYPE}.md"

DEFAULT_INPUT_PATH = Path("logs/trade_analysis.jsonl")
DEFAULT_OUTPUT_DIR = Path("logs/research_reports/latest")

_EFFECTIVE_INPUT_FILENAME = (
    "_effective_selected_strategy_rule_signal_transition_input.jsonl"
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
_WATCHLIST_SIGNAL_STATES = {"watchlist_long", "watchlist_short", "watchlist"}

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

_PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION = (
    "actionable_selected_strategy_preserved_same_direction"
)
_PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD = (
    "actionable_selected_strategy_collapses_to_hold"
)
_PRIMARY_BEHAVIOR_COLLAPSES_TO_WATCHLIST = (
    "actionable_selected_strategy_collapses_to_watchlist"
)
_PRIMARY_BEHAVIOR_COLLAPSES_TO_NO_SIGNAL = (
    "actionable_selected_strategy_collapses_to_no_signal"
)
_PRIMARY_BEHAVIOR_REVERSES_DIRECTION = (
    "actionable_selected_strategy_reverses_direction_at_rule_signal"
)
_PRIMARY_BEHAVIOR_RULE_UNOBSERVABLE = (
    "rule_signal_unobservable_after_actionable_selected_strategy"
)
_PRIMARY_BEHAVIOR_COLLAPSES_TO_UNKNOWN_OR_OTHER = (
    "actionable_selected_strategy_collapses_to_unknown_or_other"
)

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
_PRIMARY_BEHAVIOR_ORDER = {
    _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD: 0,
    _PRIMARY_BEHAVIOR_COLLAPSES_TO_WATCHLIST: 1,
    _PRIMARY_BEHAVIOR_COLLAPSES_TO_NO_SIGNAL: 2,
    _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION: 3,
    _PRIMARY_BEHAVIOR_REVERSES_DIRECTION: 4,
    _PRIMARY_BEHAVIOR_RULE_UNOBSERVABLE: 5,
    _PRIMARY_BEHAVIOR_COLLAPSES_TO_UNKNOWN_OR_OTHER: 6,
}
_NON_PRIMARY_PATH_VALUES = {
    "",
    "none",
    "insufficient_support",
    "mixed_or_inconclusive",
    "no_actionable_downgrade_rows",
}
_COMPOSED_DECISION_ONLY_KEYS = {"selected_strategy", "timeframe_summary", "debug"}

_PRIMARY_FACTOR_THRESHOLD = 0.60
_CONTRIBUTING_FACTOR_THRESHOLD = 0.35

_MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS = 30
_MIN_PRIMARY_DOWNGRADE_SUPPORT_ROWS = 10
_MIN_BREAKDOWN_ACTIONABLE_SUPPORT_ROWS = 10
_MIN_BREAKDOWN_DOWNGRADE_SUPPORT_ROWS = 10
_MIN_BIAS_SIGN_BREAKDOWN_ACTIONABLE_SUPPORT_ROWS = 10
_MIN_BIAS_SIGN_BREAKDOWN_DOWNGRADE_SUPPORT_ROWS = 10


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
            "Build a diagnosis-only selected-strategy-to-rule-signal transition "
            "report across multiple latest-window configurations using a single "
            "effective input snapshot per configuration."
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

    result = run_selected_strategy_rule_signal_transition_diagnosis_report(
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
                "primary_transition_behavior": final_assessment.get(
                    "primary_transition_behavior"
                ),
                "dominant_observed_actionable_downgrade_path": (
                    final_assessment.get("dominant_observed_actionable_downgrade_path")
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


def run_selected_strategy_rule_signal_transition_diagnosis_report(
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
            "This report is diagnosis-only and reuses one materialized effective input snapshot per configuration so every transition count comes from the same observed rows.",
            "selected_strategy_result.signal is sourced only from the legacy payload addressed by selected_strategy (for example swing_result / intraday_result / scalping_result). As a conservative fallback, a literal selected_strategy_result object is used only when it looks like a legacy strategy payload rather than a composed post-decision object.",
            "rule_signal is sourced only from rule_engine.signal and remains an upstream trace, not a proxy for final execution action.",
            "If final emitted action is referenced for context, it remains execution-layer only: execution.action first, then execution.signal if action is absent.",
            "hold, watchlist_long, watchlist_short, watchlist, no_signal, unknown, missing, and other remain separate surfaced states so the report does not over-collapse mixed or unobservable cases.",
            "Support thresholds gate deep-breakdown interpretation and exact-path claims so small slices stay descriptive instead of over-interpreted.",
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
    transition_overview = build_transition_overview(stage_rows)
    actionable_transition_summary = build_actionable_transition_summary(
        stage_rows=stage_rows,
        actionable_support_threshold=_MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS,
        downgrade_support_threshold=_MIN_PRIMARY_DOWNGRADE_SUPPORT_ROWS,
    )
    transition_by_strategy = build_transition_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy",),
        actionable_support_threshold=_MIN_BREAKDOWN_ACTIONABLE_SUPPORT_ROWS,
        downgrade_support_threshold=_MIN_BREAKDOWN_DOWNGRADE_SUPPORT_ROWS,
    )
    transition_by_strategy_symbol = build_transition_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy", "symbol"),
        actionable_support_threshold=_MIN_BREAKDOWN_ACTIONABLE_SUPPORT_ROWS,
        downgrade_support_threshold=_MIN_BREAKDOWN_DOWNGRADE_SUPPORT_ROWS,
    )
    transition_by_strategy_symbol_bias_sign = build_transition_breakdown(
        stage_rows=stage_rows,
        group_fields=("strategy", "symbol", "bias_sign"),
        actionable_support_threshold=_MIN_BIAS_SIGN_BREAKDOWN_ACTIONABLE_SUPPORT_ROWS,
        downgrade_support_threshold=_MIN_BIAS_SIGN_BREAKDOWN_DOWNGRADE_SUPPORT_ROWS,
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
            "raw_input_rows": transition_overview["transition_row_count"],
            "rows_with_known_identity": transition_overview["rows_with_known_identity"],
            "actionable_selected_strategy_rows": actionable_transition_summary[
                "actionable_selected_strategy_rows"
            ],
            "actionable_selected_strategy_rows_with_actionable_rule_signal": (
                actionable_transition_summary[
                    "actionable_selected_strategy_rows_with_actionable_rule_signal"
                ]
            ),
            "actionable_selected_strategy_preserved_same_direction_rows": (
                actionable_transition_summary[
                    "actionable_selected_strategy_preserved_same_direction_rows"
                ]
            ),
            "actionable_selected_strategy_collapses_to_hold_rows": (
                actionable_transition_summary[
                    "actionable_selected_strategy_collapses_to_hold_rows"
                ]
            ),
            "actionable_selected_strategy_collapses_to_watchlist_rows": (
                actionable_transition_summary[
                    "actionable_selected_strategy_collapses_to_watchlist_rows_total"
                ]
            ),
            "actionable_selected_strategy_collapses_to_no_signal_rows": (
                actionable_transition_summary[
                    "actionable_selected_strategy_collapses_to_no_signal_rows"
                ]
            ),
            "actionable_selected_strategy_rule_signal_unobservable_rows": (
                actionable_transition_summary[
                    "actionable_selected_strategy_rule_signal_unobservable_rows"
                ]
            ),
            "dominant_observed_actionable_transition_path": (
                actionable_transition_summary[
                    "dominant_observed_actionable_transition_path"
                ]
            ),
            "dominant_observed_actionable_downgrade_path": (
                actionable_transition_summary[
                    "dominant_observed_actionable_downgrade_path"
                ]
            ),
            "primary_actionable_downgrade_path": actionable_transition_summary[
                "primary_actionable_downgrade_path"
            ],
        },
        "support": {
            "transition_row_count": transition_overview["transition_row_count"],
            "rows_with_selected_strategy_result_signal_observable": (
                transition_overview[
                    "rows_with_selected_strategy_result_signal_observable"
                ]
            ),
            "rows_with_rule_signal_observable": transition_overview[
                "rows_with_rule_signal_observable"
            ],
            "actionable_support_status": actionable_transition_summary[
                "support_status"
            ],
            "downgrade_support_status": actionable_transition_summary[
                "downgrade_support_status"
            ],
            "primary_actionable_support_threshold_rows": (
                _MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS
            ),
            "primary_downgrade_support_threshold_rows": (
                _MIN_PRIMARY_DOWNGRADE_SUPPORT_ROWS
            ),
        },
        "transition_overview": transition_overview,
        "actionable_transition_summary": actionable_transition_summary,
        "transition_by_strategy": transition_by_strategy,
        "transition_by_strategy_symbol": transition_by_strategy_symbol,
        "transition_by_strategy_symbol_bias_sign": (
            transition_by_strategy_symbol_bias_sign
        ),
        "observations": _build_configuration_observations(
            configuration=configuration,
            transition_overview=transition_overview,
            actionable_transition_summary=actionable_transition_summary,
        ),
        "confirmed_interpretations": _build_configuration_confirmed_interpretations(
            configuration=configuration,
            actionable_transition_summary=actionable_transition_summary,
        ),
        "unconfirmed_interpretations": _build_configuration_unconfirmed_interpretations(
            actionable_transition_summary=actionable_transition_summary
        ),
    }


def build_transition_overview(stage_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_counter: Counter[str] = Counter()
    rule_counter: Counter[str] = Counter()
    transition_counter: Counter[tuple[str, str]] = Counter()
    selected_other_counter: Counter[str] = Counter()
    rule_other_counter: Counter[str] = Counter()
    rows_with_known_identity = 0
    rows_with_selected_strategy_result_signal_observable = 0
    rows_with_rule_signal_observable = 0
    rows_with_observable_transition = 0

    for row in stage_rows:
        if _has_known_identity(row):
            rows_with_known_identity += 1

        selected_state = row.get("selected_strategy_result_signal_state")
        rule_state = row.get("rule_signal_state")
        selected_label = _display_signal_state(selected_state)
        rule_label = _display_signal_state(rule_state)

        selected_counter[selected_label] += 1
        rule_counter[rule_label] += 1
        transition_counter[(selected_label, rule_label)] += 1

        if selected_state is not None:
            rows_with_selected_strategy_result_signal_observable += 1
        if rule_state is not None:
            rows_with_rule_signal_observable += 1
        if selected_state is not None and rule_state is not None:
            rows_with_observable_transition += 1

        if selected_state == "other":
            selected_other_counter[
                row.get("selected_strategy_result_signal_label") or _MISSING_LABEL
            ] += 1
        if rule_state == "other":
            rule_other_counter[row.get("rule_signal_label") or _MISSING_LABEL] += 1

    return {
        "transition_row_count": len(stage_rows),
        "rows_with_known_identity": rows_with_known_identity,
        "rows_with_selected_strategy_result_signal_observable": (
            rows_with_selected_strategy_result_signal_observable
        ),
        "rows_with_rule_signal_observable": rows_with_rule_signal_observable,
        "rows_with_observable_transition": rows_with_observable_transition,
        "selected_strategy_result_signal_counts": dict(selected_counter),
        "selected_strategy_result_signal_count_rows": _counter_rows(
            selected_counter,
            key_name="signal_state",
            order_map=_SIGNAL_STATE_ORDER,
        ),
        "rule_signal_counts": dict(rule_counter),
        "rule_signal_count_rows": _counter_rows(
            rule_counter,
            key_name="signal_state",
            order_map=_SIGNAL_STATE_ORDER,
        ),
        "transition_counts": _transition_counter_dict(transition_counter),
        "transition_count_rows": _transition_counter_rows(transition_counter),
        "selected_strategy_result_other_signal_count_rows": _counter_rows(
            selected_other_counter,
            key_name="signal_label",
        ),
        "rule_signal_other_signal_count_rows": _counter_rows(
            rule_other_counter,
            key_name="signal_label",
        ),
        "dominant_observed_transition_path": _dominant_observed_transition_path(
            transition_counter,
            empty="none",
        ),
    }


def build_actionable_transition_summary(
    *,
    stage_rows: Sequence[dict[str, Any]],
    actionable_support_threshold: int,
    downgrade_support_threshold: int,
) -> dict[str, Any]:
    actionable_transition_counter: Counter[tuple[str, str]] = Counter()
    actionable_downgrade_counter: Counter[tuple[str, str]] = Counter()
    actionable_rule_state_counter: Counter[str] = Counter()

    actionable_selected_strategy_rows = 0
    selected_strategy_non_actionable_rows = 0
    selected_strategy_unobservable_rows = 0
    actionable_selected_strategy_rows_with_actionable_rule_signal = 0
    actionable_selected_strategy_preserved_same_direction_rows = 0
    actionable_selected_strategy_preserved_opposite_direction_rows = 0
    actionable_selected_strategy_collapses_to_hold_rows = 0
    actionable_selected_strategy_collapses_to_watchlist_long_rows = 0
    actionable_selected_strategy_collapses_to_watchlist_short_rows = 0
    actionable_selected_strategy_collapses_to_watchlist_rows = 0
    actionable_selected_strategy_collapses_to_no_signal_rows = 0
    actionable_selected_strategy_collapses_to_unknown_rows = 0
    actionable_selected_strategy_collapses_to_other_rows = 0
    actionable_selected_strategy_rule_signal_unobservable_rows = 0

    for row in stage_rows:
        selected_state = row.get("selected_strategy_result_signal_state")
        rule_state = row.get("rule_signal_state")

        if selected_state not in _ACTIONABLE_SIGNAL_STATES:
            if selected_state is None:
                selected_strategy_unobservable_rows += 1
            else:
                selected_strategy_non_actionable_rows += 1
            continue

        actionable_selected_strategy_rows += 1
        selected_label = _display_signal_state(selected_state)
        rule_label = _display_signal_state(rule_state)
        actionable_transition_counter[(selected_label, rule_label)] += 1
        actionable_rule_state_counter[rule_label] += 1

        if rule_state in _ACTIONABLE_SIGNAL_STATES:
            actionable_selected_strategy_rows_with_actionable_rule_signal += 1
            if rule_state == selected_state:
                actionable_selected_strategy_preserved_same_direction_rows += 1
            else:
                actionable_selected_strategy_preserved_opposite_direction_rows += 1
            continue

        if rule_state is None:
            actionable_selected_strategy_rule_signal_unobservable_rows += 1
            continue

        actionable_downgrade_counter[(selected_label, rule_label)] += 1
        if rule_state == "hold":
            actionable_selected_strategy_collapses_to_hold_rows += 1
        elif rule_state == "watchlist_long":
            actionable_selected_strategy_collapses_to_watchlist_long_rows += 1
        elif rule_state == "watchlist_short":
            actionable_selected_strategy_collapses_to_watchlist_short_rows += 1
        elif rule_state == "watchlist":
            actionable_selected_strategy_collapses_to_watchlist_rows += 1
        elif rule_state == "no_signal":
            actionable_selected_strategy_collapses_to_no_signal_rows += 1
        elif rule_state == "unknown":
            actionable_selected_strategy_collapses_to_unknown_rows += 1
        else:
            actionable_selected_strategy_collapses_to_other_rows += 1

    watchlist_total = (
        actionable_selected_strategy_collapses_to_watchlist_long_rows
        + actionable_selected_strategy_collapses_to_watchlist_short_rows
        + actionable_selected_strategy_collapses_to_watchlist_rows
    )
    unknown_or_other_total = (
        actionable_selected_strategy_collapses_to_unknown_rows
        + actionable_selected_strategy_collapses_to_other_rows
    )
    observable_non_actionable_total = (
        actionable_selected_strategy_collapses_to_hold_rows
        + watchlist_total
        + actionable_selected_strategy_collapses_to_no_signal_rows
        + unknown_or_other_total
    )
    support_status = (
        "supported"
        if actionable_selected_strategy_rows >= actionable_support_threshold
        else "limited_support"
    )
    downgrade_support_status = (
        "supported"
        if observable_non_actionable_total >= downgrade_support_threshold
        else "limited_support"
    )
    factor_counts = {
        _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION: (
            actionable_selected_strategy_preserved_same_direction_rows
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD: (
            actionable_selected_strategy_collapses_to_hold_rows
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_WATCHLIST: watchlist_total,
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_NO_SIGNAL: (
            actionable_selected_strategy_collapses_to_no_signal_rows
        ),
        _PRIMARY_BEHAVIOR_REVERSES_DIRECTION: (
            actionable_selected_strategy_preserved_opposite_direction_rows
        ),
        _PRIMARY_BEHAVIOR_RULE_UNOBSERVABLE: (
            actionable_selected_strategy_rule_signal_unobservable_rows
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_UNKNOWN_OR_OTHER: unknown_or_other_total,
    }

    return {
        "actionable_selected_strategy_rows": actionable_selected_strategy_rows,
        "selected_strategy_non_actionable_rows": selected_strategy_non_actionable_rows,
        "selected_strategy_unobservable_rows": selected_strategy_unobservable_rows,
        "actionable_selected_strategy_rows_with_actionable_rule_signal": (
            actionable_selected_strategy_rows_with_actionable_rule_signal
        ),
        "actionable_selected_strategy_actionable_rule_signal_rate": _safe_ratio(
            actionable_selected_strategy_rows_with_actionable_rule_signal,
            actionable_selected_strategy_rows,
        ),
        "actionable_selected_strategy_preserved_same_direction_rows": (
            actionable_selected_strategy_preserved_same_direction_rows
        ),
        "actionable_selected_strategy_preserved_same_direction_rate": _safe_ratio(
            actionable_selected_strategy_preserved_same_direction_rows,
            actionable_selected_strategy_rows,
        ),
        "actionable_selected_strategy_preserved_opposite_direction_rows": (
            actionable_selected_strategy_preserved_opposite_direction_rows
        ),
        "actionable_selected_strategy_collapses_to_hold_rows": (
            actionable_selected_strategy_collapses_to_hold_rows
        ),
        "actionable_selected_strategy_collapses_to_watchlist_long_rows": (
            actionable_selected_strategy_collapses_to_watchlist_long_rows
        ),
        "actionable_selected_strategy_collapses_to_watchlist_short_rows": (
            actionable_selected_strategy_collapses_to_watchlist_short_rows
        ),
        "actionable_selected_strategy_collapses_to_watchlist_rows": (
            actionable_selected_strategy_collapses_to_watchlist_rows
        ),
        "actionable_selected_strategy_collapses_to_watchlist_rows_total": (
            watchlist_total
        ),
        "actionable_selected_strategy_collapses_to_no_signal_rows": (
            actionable_selected_strategy_collapses_to_no_signal_rows
        ),
        "actionable_selected_strategy_collapses_to_unknown_rows": (
            actionable_selected_strategy_collapses_to_unknown_rows
        ),
        "actionable_selected_strategy_collapses_to_other_rows": (
            actionable_selected_strategy_collapses_to_other_rows
        ),
        "actionable_selected_strategy_collapses_to_unknown_or_other_rows": (
            unknown_or_other_total
        ),
        "actionable_selected_strategy_rule_signal_unobservable_rows": (
            actionable_selected_strategy_rule_signal_unobservable_rows
        ),
        "actionable_rule_signal_state_count_rows": _counter_rows(
            actionable_rule_state_counter,
            key_name="signal_state",
            order_map=_SIGNAL_STATE_ORDER,
        ),
        "actionable_transition_counts": _transition_counter_dict(
            actionable_transition_counter
        ),
        "actionable_transition_path_count_rows": _transition_counter_rows(
            actionable_transition_counter
        ),
        "actionable_downgrade_counts": _transition_counter_dict(
            actionable_downgrade_counter
        ),
        "actionable_downgrade_path_count_rows": _transition_counter_rows(
            actionable_downgrade_counter
        ),
        "dominant_observed_actionable_transition_path": (
            _dominant_observed_transition_path(
                actionable_transition_counter,
                empty="none",
            )
        ),
        "primary_actionable_transition_path": _primary_transition_path(
            actionable_transition_counter,
            support_threshold=actionable_support_threshold,
            empty="no_actionable_selected_strategy_rows",
        ),
        "dominant_observed_actionable_downgrade_path": (
            _dominant_observed_transition_path(
                actionable_downgrade_counter,
                empty="none",
            )
        ),
        "primary_actionable_downgrade_path": _primary_transition_path(
            actionable_downgrade_counter,
            support_threshold=downgrade_support_threshold,
            empty="no_actionable_downgrade_rows",
        ),
        "primary_behavior": _primary_behavior(
            factor_counts,
            actionable_selected_strategy_rows,
            support_threshold=actionable_support_threshold,
        ),
        "support_status": support_status,
        "downgrade_support_status": downgrade_support_status,
    }


def build_transition_breakdown(
    *,
    stage_rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
    actionable_support_threshold: int,
    downgrade_support_threshold: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in stage_rows:
        group_values = [_group_value(row, field) for field in group_fields]
        if any(value is None for value in group_values):
            continue
        grouped.setdefault(tuple(str(value) for value in group_values), []).append(row)

    rows: list[dict[str, Any]] = []
    for key, grouped_rows in grouped.items():
        overview = build_transition_overview(grouped_rows)
        actionable = build_actionable_transition_summary(
            stage_rows=grouped_rows,
            actionable_support_threshold=actionable_support_threshold,
            downgrade_support_threshold=downgrade_support_threshold,
        )
        rows.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                "transition_row_count": overview["transition_row_count"],
                "transition_count_rows": overview["transition_count_rows"],
                "actionable_selected_strategy_rows": actionable[
                    "actionable_selected_strategy_rows"
                ],
                "actionable_selected_strategy_rows_with_actionable_rule_signal": (
                    actionable[
                        "actionable_selected_strategy_rows_with_actionable_rule_signal"
                    ]
                ),
                "actionable_selected_strategy_collapses_to_hold_rows": actionable[
                    "actionable_selected_strategy_collapses_to_hold_rows"
                ],
                "actionable_selected_strategy_collapses_to_watchlist_rows": (
                    actionable[
                        "actionable_selected_strategy_collapses_to_watchlist_rows_total"
                    ]
                ),
                "actionable_selected_strategy_collapses_to_no_signal_rows": (
                    actionable[
                        "actionable_selected_strategy_collapses_to_no_signal_rows"
                    ]
                ),
                "actionable_selected_strategy_rule_signal_unobservable_rows": (
                    actionable[
                        "actionable_selected_strategy_rule_signal_unobservable_rows"
                    ]
                ),
                "dominant_observed_actionable_transition_path": actionable[
                    "dominant_observed_actionable_transition_path"
                ],
                "primary_actionable_transition_path": actionable[
                    "primary_actionable_transition_path"
                ],
                "dominant_observed_actionable_downgrade_path": actionable[
                    "dominant_observed_actionable_downgrade_path"
                ],
                "primary_actionable_downgrade_path": actionable[
                    "primary_actionable_downgrade_path"
                ],
                "primary_behavior": actionable["primary_behavior"],
                "support_status": actionable["support_status"],
                "downgrade_support_status": actionable["downgrade_support_status"],
            }
        )

    rows.sort(
        key=lambda item: (
            -int(item.get("transition_row_count", 0) or 0),
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
            "primary_transition_behavior": "none",
            "dominant_observed_actionable_downgrade_path": "none",
            "widest_configuration": None,
            "factors": {},
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
    actionable = _safe_dict(widest.get("actionable_transition_summary"))
    actionable_rows = int(actionable.get("actionable_selected_strategy_rows", 0) or 0)

    factors = _build_factor_assessment(actionable, actionable_rows)
    primary_behavior = _primary_behavior_from_factors(factors, actionable_rows)
    strategy_map = _supported_strategy_primary_actionable_downgrade_paths(
        widest.get("transition_by_strategy")
    )
    observations = _build_confirmed_observations(widest, actionable_rows)

    return {
        "assessment": _overall_assessment_label(primary_behavior, actionable_rows),
        "primary_transition_behavior": primary_behavior,
        "dominant_observed_actionable_transition_path": actionable.get(
            "dominant_observed_actionable_transition_path",
            "none",
        ),
        "dominant_observed_actionable_downgrade_path": actionable.get(
            "dominant_observed_actionable_downgrade_path",
            "none",
        ),
        "primary_actionable_transition_path": actionable.get(
            "primary_actionable_transition_path",
            "none",
        ),
        "primary_actionable_downgrade_path": actionable.get(
            "primary_actionable_downgrade_path",
            "none",
        ),
        "widest_configuration": _safe_dict(widest.get("configuration")),
        "supported_strategy_primary_actionable_downgrade_paths": strategy_map,
        "factors": factors,
        "confirmed_observations": observations,
        "evidence_backed_inferences": _build_evidence_backed_inferences(
            primary_behavior,
            actionable,
            strategy_map,
        ),
        "unresolved_uncertainties": [
            "This report measures selected_strategy_result.signal -> rule_engine.signal directly, but it does not prove the internal rule or condition that produced a given downgrade.",
            "This report does not prove whether an observed downgrade is intended conservatism, conflict resolution, confirmation gating, neutrality normalization, or another internal mechanism.",
            "If rule_signal is missing after an actionable selected-strategy row, the report preserves that as unobservable rather than guessing where the collapse occurred.",
            "The selected_strategy_result fallback stays intentionally conservative because this code path cannot prove every raw selected_strategy_result object is a legacy strategy payload rather than a composed decision payload.",
        ],
        "overall_conclusion": _build_overall_conclusion(
            widest,
            primary_behavior,
            strategy_map,
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {REPORT_TITLE}", ""]
    lines.append("## Configurations")
    lines.append("")
    for configuration in _safe_list(report.get("configurations_evaluated")):
        config = _safe_dict(configuration)
        lines.append(
            f"- {config.get('display_name')}: latest_window_hours={config.get('latest_window_hours')}, latest_max_rows={config.get('latest_max_rows')}"
        )
    lines.append("")

    for summary in _safe_list(report.get("configuration_summaries")):
        config = _safe_dict(_safe_dict(summary).get("configuration"))
        headline = _safe_dict(_safe_dict(summary).get("headline"))
        overview = _safe_dict(_safe_dict(summary).get("transition_overview"))
        actionable = _safe_dict(_safe_dict(summary).get("actionable_transition_summary"))
        lines.append(f"## {config.get('display_name', 'n/a')}")
        lines.append("")
        lines.append(f"- raw_input_rows: {headline.get('raw_input_rows', 0)}")
        lines.append(
            "- selected_strategy_result_signal_counts: "
            f"{_format_counter_rows(overview.get('selected_strategy_result_signal_count_rows'), 'signal_state')}"
        )
        lines.append(
            "- rule_signal_counts: "
            f"{_format_counter_rows(overview.get('rule_signal_count_rows'), 'signal_state')}"
        )
        lines.append(
            "- transition_counts: "
            f"{_format_transition_rows(overview.get('transition_count_rows'))}"
        )
        lines.append(
            "- actionable_transition_paths: "
            f"{_format_transition_rows(actionable.get('actionable_transition_path_count_rows'))}"
        )
        lines.append(
            "- actionable_downgrade_paths: "
            f"{_format_transition_rows(actionable.get('actionable_downgrade_path_count_rows'))}"
        )
        lines.append(
            "- primary_actionable_downgrade_path: "
            f"{actionable.get('primary_actionable_downgrade_path', 'n/a')}"
        )
        lines.append("")

    lines.append("## Final Assessment")
    lines.append("")
    final_assessment = _safe_dict(report.get("final_assessment"))
    lines.append(f"- assessment: {final_assessment.get('assessment', 'n/a')}")
    lines.append(
        "- primary_transition_behavior: "
        f"{final_assessment.get('primary_transition_behavior', 'n/a')}"
    )
    lines.append(
        "- dominant_observed_actionable_downgrade_path: "
        f"{final_assessment.get('dominant_observed_actionable_downgrade_path', 'n/a')}"
    )
    for factor_name, payload in _safe_dict(final_assessment.get("factors")).items():
        lines.append(f"- {factor_name}: {_safe_dict(payload).get('status', 'unknown')}")
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
    selected_strategy_result_payload = _selected_strategy_result_payload(
        raw_record=raw_record,
        strategy=strategy,
    )
    selected_label = _normalize_signal(selected_strategy_result_payload.get("signal"))
    rule_label = _normalize_signal(normalized.get("rule_signal"))
    selected_state = _surface_signal_state(selected_label)
    rule_state = _surface_signal_state(rule_label)
    return {
        **normalized,
        "strategy": strategy,
        "symbol": _normalize_symbol(normalized.get("symbol")),
        "bias_sign": _bias_sign(normalized),
        "selected_strategy_result_signal_label": selected_label,
        "selected_strategy_result_signal_state": selected_state,
        "rule_signal_label": rule_label,
        "rule_signal_state": rule_state,
    }


def _selected_strategy_result_payload(
    *,
    raw_record: dict[str, Any],
    strategy: str | None,
) -> dict[str, Any]:
    if strategy is not None:
        strategy_payload = _as_dict(raw_record.get(f"{strategy}_result"))
        if strategy_payload:
            return strategy_payload

    explicit_selected_payload = _as_dict(raw_record.get("selected_strategy_result"))
    if not explicit_selected_payload:
        return {}

    return _safe_explicit_selected_strategy_result_payload(
        payload=explicit_selected_payload,
        strategy=strategy,
    )


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


def _build_factor_assessment(
    actionable: dict[str, Any],
    actionable_rows: int,
) -> dict[str, Any]:
    counts = {
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD: int(
            actionable.get("actionable_selected_strategy_collapses_to_hold_rows", 0) or 0
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_WATCHLIST: int(
            actionable.get(
                "actionable_selected_strategy_collapses_to_watchlist_rows_total",
                0,
            )
            or 0
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_NO_SIGNAL: int(
            actionable.get("actionable_selected_strategy_collapses_to_no_signal_rows", 0)
            or 0
        ),
        _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION: int(
            actionable.get(
                "actionable_selected_strategy_preserved_same_direction_rows",
                0,
            )
            or 0
        ),
        _PRIMARY_BEHAVIOR_REVERSES_DIRECTION: int(
            actionable.get(
                "actionable_selected_strategy_preserved_opposite_direction_rows",
                0,
            )
            or 0
        ),
        _PRIMARY_BEHAVIOR_RULE_UNOBSERVABLE: int(
            actionable.get(
                "actionable_selected_strategy_rule_signal_unobservable_rows",
                0,
            )
            or 0
        ),
        _PRIMARY_BEHAVIOR_COLLAPSES_TO_UNKNOWN_OR_OTHER: int(
            actionable.get(
                "actionable_selected_strategy_collapses_to_unknown_or_other_rows",
                0,
            )
            or 0
        ),
    }
    result: dict[str, Any] = {}
    for label, rows in counts.items():
        share = _safe_ratio(rows, actionable_rows)
        if actionable_rows < _MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS:
            status = "insufficient_support"
        elif share >= _PRIMARY_FACTOR_THRESHOLD:
            status = "primary"
        elif share >= _CONTRIBUTING_FACTOR_THRESHOLD:
            status = "contributing"
        elif rows > 0:
            status = "limited"
        else:
            status = "not_supported"
        result[label] = {
            "status": status,
            "evidence": [
                f"Actionable selected-strategy rows: {actionable_rows}; {label}: {rows} ({share:.2%})."
            ],
        }
    return result


def _build_configuration_observations(
    *,
    configuration: DiagnosisConfiguration,
    transition_overview: dict[str, Any],
    actionable_transition_summary: dict[str, Any],
) -> list[str]:
    return [
        (
            f"{configuration.display_name}: transition rows={transition_overview.get('transition_row_count', 0)}, "
            "selected_strategy_result.signal observable rows="
            f"{transition_overview.get('rows_with_selected_strategy_result_signal_observable', 0)}, "
            f"rule_signal observable rows={transition_overview.get('rows_with_rule_signal_observable', 0)}."
        ),
        (
            "Actionable selected-strategy rows="
            f"{actionable_transition_summary.get('actionable_selected_strategy_rows', 0)}; "
            "actionable rule-signal rows from those rows="
            f"{actionable_transition_summary.get('actionable_selected_strategy_rows_with_actionable_rule_signal', 0)}."
        ),
        (
            "Dominant observed actionable transition path="
            f"{actionable_transition_summary.get('dominant_observed_actionable_transition_path', 'none')}; "
            "dominant observed actionable downgrade path="
            f"{actionable_transition_summary.get('dominant_observed_actionable_downgrade_path', 'none')}."
        ),
    ]


def _build_configuration_confirmed_interpretations(
    *,
    configuration: DiagnosisConfiguration,
    actionable_transition_summary: dict[str, Any],
) -> list[str]:
    items = [
        (
            f"{configuration.display_name} directly measures selected_strategy_result.signal -> rule_engine.signal "
            f"for {int(actionable_transition_summary.get('actionable_selected_strategy_rows', 0) or 0)} actionable selected-strategy rows."
        )
    ]
    if (
        actionable_transition_summary.get(
            "actionable_selected_strategy_collapses_to_hold_rows",
            0,
        )
        or 0
    ) > 0:
        items.append(
            "Actionable selected-strategy rows that turn into rule_signal=hold are observed directly."
        )
    if (
        actionable_transition_summary.get(
            "actionable_selected_strategy_rows_with_actionable_rule_signal",
            0,
        )
        or 0
    ) > 0:
        items.append(
            "Actionable selected-strategy rows that remain actionable at rule_signal are also observed directly."
        )
    return items


def _build_configuration_unconfirmed_interpretations(
    *,
    actionable_transition_summary: dict[str, Any],
) -> list[str]:
    items = [
        "This report does not prove whether a long/short -> hold transition comes from decision composition, confirmation logic, signal normalization, neutrality resolution, or conflict resolution.",
        "This report does not prove whether observed collapses are intended conservatism or an unintended bottleneck.",
        "This report uses a conservative selected_strategy_result fallback and therefore does not prove that every omitted literal selected_strategy_result object would have been a valid legacy strategy payload.",
    ]
    if actionable_transition_summary.get("primary_actionable_downgrade_path") in {
        "insufficient_support",
        "mixed_or_inconclusive",
    }:
        items.append(
            "Exact dominant downgrade-path claims remain intentionally suppressed when support is thin or the downgrade mix stays split."
        )
    return items


def _build_confirmed_observations(
    widest: dict[str, Any],
    actionable_rows: int,
) -> list[str]:
    actionable = _safe_dict(widest.get("actionable_transition_summary"))
    overview = _safe_dict(widest.get("transition_overview"))
    return [
        (
            "Transition rows: "
            f"{int(overview.get('transition_row_count', 0) or 0)}; "
            "selected_strategy_result.signal observable rows: "
            f"{int(overview.get('rows_with_selected_strategy_result_signal_observable', 0) or 0)}; "
            f"rule_signal observable rows: {int(overview.get('rows_with_rule_signal_observable', 0) or 0)}."
        ),
        (
            f"Actionable selected-strategy rows: {actionable_rows}; "
            "actionable rule-signal rows from those rows: "
            f"{int(actionable.get('actionable_selected_strategy_rows_with_actionable_rule_signal', 0) or 0)}."
        ),
        (
            "Dominant observed actionable transition path: "
            f"{actionable.get('dominant_observed_actionable_transition_path', 'none')}; "
            "dominant observed actionable downgrade path: "
            f"{actionable.get('dominant_observed_actionable_downgrade_path', 'none')}."
        ),
    ]


def _build_evidence_backed_inferences(
    primary_behavior: str,
    actionable: dict[str, Any],
    strategy_map: dict[str, str],
) -> list[str]:
    items: list[str] = []
    if primary_behavior == _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD:
        items.append(
            "In the widest supported configuration, actionable selected-strategy proposals usually become rule_signal=hold more often than any other measured transition behavior."
        )
    elif primary_behavior == _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION:
        items.append(
            "In the widest supported configuration, actionable selected-strategy proposals usually remain actionable with the same direction at rule_signal."
        )
    elif primary_behavior == _PRIMARY_BEHAVIOR_REVERSES_DIRECTION:
        items.append(
            "In the widest supported configuration, actionable selected-strategy proposals most often remain actionable but reverse direction at rule_signal."
        )
    elif primary_behavior == "insufficient_support":
        items.append(
            "Current actionable support is insufficient to classify one primary transition behavior."
        )
    else:
        items.append(
            "No single exact transition behavior is dominant enough to claim more than a mixed or conservative interpretation."
        )

    if strategy_map:
        items.append(f"Supported per-strategy primary downgrade paths are {strategy_map}.")
    if actionable.get("primary_actionable_downgrade_path") in {
        "insufficient_support",
        "mixed_or_inconclusive",
    }:
        items.append(
            "Exact downgrade-path dominance stays intentionally conservative because the observable downgrade mix is thin or split."
        )
    return items


def _build_overall_conclusion(
    widest: dict[str, Any],
    primary_behavior: str,
    strategy_map: dict[str, str],
) -> str:
    configuration = _safe_dict(widest.get("configuration"))
    actionable = _safe_dict(widest.get("actionable_transition_summary"))
    prefix = (
        "The strongest supported transition interpretation is that actionable selected-strategy proposals usually collapse into rule_signal=hold."
        if primary_behavior == _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD
        else (
            "The strongest supported transition interpretation is that actionable selected-strategy proposals usually remain actionable with the same direction at rule_signal."
            if primary_behavior == _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION
            else (
                "The strongest supported transition interpretation is that actionable selected-strategy proposals usually remain actionable but reverse direction at rule_signal."
                if primary_behavior == _PRIMARY_BEHAVIOR_REVERSES_DIRECTION
                else (
                    "Support is insufficient to classify one primary selected-strategy-to-rule-signal transition behavior."
                    if primary_behavior == "insufficient_support"
                    else "No single selected-strategy-to-rule-signal transition behavior is fully dominant, so the outcome remains mixed or conservative by design."
                )
            )
        )
    )
    suffix = (
        " Supported per-strategy primary actionable downgrade paths were "
        f"{strategy_map}."
        if strategy_map
        else ""
    )
    return (
        f"{prefix} At {configuration.get('display_name', 'the widest configuration')}, "
        "dominant observed actionable transition path was "
        f"{actionable.get('dominant_observed_actionable_transition_path', 'none')} and "
        "dominant observed actionable downgrade path was "
        f"{actionable.get('dominant_observed_actionable_downgrade_path', 'none')}."
        f"{suffix}"
    )


def _primary_behavior_from_factors(
    factors: dict[str, Any],
    actionable_rows: int,
) -> str:
    if actionable_rows < _MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS:
        return "insufficient_support"
    candidates = [
        name
        for name, payload in factors.items()
        if _safe_dict(payload).get("status") == "primary"
    ]
    if len(candidates) == 1:
        return candidates[0]
    return "mixed_or_inconclusive"


def _overall_assessment_label(primary_behavior: str, actionable_rows: int) -> str:
    if actionable_rows < _MIN_PRIMARY_ACTIONABLE_SUPPORT_ROWS:
        return "insufficient_support"
    if primary_behavior == _PRIMARY_BEHAVIOR_COLLAPSES_TO_HOLD:
        return "actionable_selected_strategy_collapses_to_hold_primary"
    if primary_behavior == _PRIMARY_BEHAVIOR_PRESERVED_SAME_DIRECTION:
        return "actionable_selected_strategy_preserved_same_direction_primary"
    if primary_behavior == _PRIMARY_BEHAVIOR_REVERSES_DIRECTION:
        return "actionable_selected_strategy_reverses_direction_primary"
    return "mixed_or_inconclusive"


def _primary_behavior(
    counts: dict[str, int],
    actionable_rows: int,
    *,
    support_threshold: int,
) -> str:
    if actionable_rows < support_threshold:
        return "insufficient_support"
    positive_counts = {key: value for key, value in counts.items() if value > 0}
    if not positive_counts:
        return "no_actionable_selected_strategy_rows"
    dominant_count = max(positive_counts.values())
    dominant = [
        key for key, value in positive_counts.items() if value == dominant_count
    ]
    if len(dominant) > 1 or _safe_ratio(dominant_count, actionable_rows) < _PRIMARY_FACTOR_THRESHOLD:
        return "mixed_or_inconclusive"
    dominant.sort(key=lambda value: (_PRIMARY_BEHAVIOR_ORDER.get(value, 99), value))
    return dominant[0]


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


def _bias_sign(row: dict[str, Any]) -> str | None:
    bias = _normalize_bias(row.get("bias"))
    if bias is None or bias in _NON_DIRECTIONAL_BIAS_VALUES:
        return None
    if bias in _BULLISH_BIAS_VALUES:
        return "bullish"
    if bias in _BEARISH_BIAS_VALUES:
        return "bearish"
    return None


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
    return value if value is not None else _MISSING_LABEL


def _transition_counter_dict(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {
        _format_transition_path(selected_state, rule_state): count
        for (selected_state, rule_state), count in sorted(
            counter.items(),
            key=lambda item: (
                _SIGNAL_STATE_ORDER.get(item[0][0], 99),
                _SIGNAL_STATE_ORDER.get(item[0][1], 99),
                item[0][0],
                item[0][1],
            ),
        )
    }


def _transition_counter_rows(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "selected_strategy_result_signal_state": selected_state,
            "rule_signal_state": rule_state,
            "transition_path": _format_transition_path(selected_state, rule_state),
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for (selected_state, rule_state), count in sorted(
            counter.items(),
            key=lambda item: (
                -item[1],
                _SIGNAL_STATE_ORDER.get(item[0][0], 99),
                _SIGNAL_STATE_ORDER.get(item[0][1], 99),
                item[0][0],
                item[0][1],
            ),
        )
    ]


def _format_transition_path(selected_state: str, rule_state: str) -> str:
    return f"{selected_state}->{rule_state}"


def _dominant_observed_transition_path(
    counter: Counter[tuple[str, str]],
    *,
    empty: str,
) -> str:
    if not counter:
        return empty
    return _format_transition_path(
        *min(
            counter.items(),
            key=lambda item: (
                -item[1],
                _SIGNAL_STATE_ORDER.get(item[0][0], 99),
                _SIGNAL_STATE_ORDER.get(item[0][1], 99),
                item[0][0],
                item[0][1],
            ),
        )[0]
    )


def _primary_transition_path(
    counter: Counter[tuple[str, str]],
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
        return "mixed_or_inconclusive"
    dominant.sort(
        key=lambda item: (
            _SIGNAL_STATE_ORDER.get(item[0], 99),
            _SIGNAL_STATE_ORDER.get(item[1], 99),
            item[0],
            item[1],
        )
    )
    return _format_transition_path(*dominant[0])


def _supported_strategy_primary_actionable_downgrade_paths(
    rows: Any,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _safe_list(rows):
        row_dict = _safe_dict(row)
        strategy = _normalize_strategy(row_dict.get("strategy"))
        if strategy is None:
            continue
        if row_dict.get("support_status") != "supported":
            continue
        if row_dict.get("downgrade_support_status") != "supported":
            continue

        primary_path = str(row_dict.get("primary_actionable_downgrade_path") or "").strip()
        if primary_path in _NON_PRIMARY_PATH_VALUES:
            continue

        result[strategy] = primary_path
    return result


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


def _format_counter_rows(value: Any, key_name: str) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get(key_name, 'n/a')}={_safe_dict(row).get('count', 0)}"
        for row in rows
    )


def _format_transition_rows(value: Any) -> str:
    rows = _safe_list(value)
    if not rows:
        return "none"
    return ", ".join(
        f"{_safe_dict(row).get('transition_path', 'n/a')}={_safe_dict(row).get('count', 0)}"
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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


if __name__ == "__main__":
    main()