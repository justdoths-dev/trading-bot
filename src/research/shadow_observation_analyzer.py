from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "edge_selection_shadow"
    / "edge_selection_shadow.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "research_reports"
    / "latest"
)

VALID_SELECTION_STATUSES = ("selected", "abstain", "blocked")
VALID_CANDIDATE_STATUSES = ("eligible", "penalized", "blocked")
VALID_HORIZONS = ("15m", "1h", "4h")

RECENT_RUN_LIMIT = 10
MAX_MALFORMED_LINE_NUMBERS = 20
MAX_FREQUENCY_ROWS = 20


def run_shadow_observation_analyzer(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_input_path = input_path or DEFAULT_INPUT_PATH
    resolved_output_dir = output_dir or DEFAULT_OUTPUT_DIR

    loaded = _load_shadow_runs(resolved_input_path)
    summary = build_shadow_observation_summary(
        runs=loaded["runs"],
        input_path=resolved_input_path,
        data_quality=loaded["data_quality"],
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_dir / "shadow_observation_summary.json"
    md_path = resolved_output_dir / "shadow_observation_summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    return {
        "input_path": str(resolved_input_path),
        "output_dir": str(resolved_output_dir),
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "run_count": len(loaded["runs"]),
    }


def build_shadow_observation_summary(
    runs: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    latest_timestamp = _latest_timestamp(runs)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "report_type": "shadow_observation_summary",
        "input_path": str(input_path),
        "overall": _build_overall_summary(runs),
        "by_day": _build_by_day_summary(runs),
        "recent_runs": _build_recent_runs_summary(runs, RECENT_RUN_LIMIT),
        "last_24h": _build_window_summary(
            runs,
            latest_timestamp,
            timedelta(hours=24),
            "last_24h",
        ),
        "last_7d": _build_window_summary(
            runs,
            latest_timestamp,
            timedelta(days=7),
            "last_7d",
        ),
        "data_quality": _finalize_data_quality(data_quality, runs, latest_timestamp),
        "unavailable_metrics": _build_unavailable_metrics(runs, latest_timestamp),
    }


def _load_shadow_runs(path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    malformed_line_numbers: list[int] = []

    data_quality = {
        "input_exists": path.exists(),
        "input_is_file": path.is_file() if path.exists() else False,
        "total_lines": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "malformed_line_numbers": malformed_line_numbers,
        "non_object_lines": 0,
        "non_list_ranking_runs": 0,
        "ranking_items_non_object": 0,
        "invalid_timestamp_runs": 0,
        "invalid_selection_status_runs": 0,
        "invalid_candidate_status_items": 0,
        "invalid_horizon_values": 0,
    }

    if not path.exists():
        return {"runs": runs, "data_quality": data_quality}

    if not path.is_file():
        data_quality["malformed_lines"] = 1
        malformed_line_numbers.append(0)
        return {"runs": runs, "data_quality": data_quality}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            data_quality["total_lines"] += 1
            content = line.strip()

            if not content:
                data_quality["blank_lines"] += 1
                continue

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                data_quality["malformed_lines"] += 1
                _append_limited(
                    malformed_line_numbers,
                    line_number,
                    MAX_MALFORMED_LINE_NUMBERS,
                )
                continue

            if not isinstance(payload, dict):
                data_quality["malformed_lines"] += 1
                data_quality["non_object_lines"] += 1
                _append_limited(
                    malformed_line_numbers,
                    line_number,
                    MAX_MALFORMED_LINE_NUMBERS,
                )
                continue

            runs.append(
                _normalize_run(
                    payload=payload,
                    line_number=line_number,
                    data_quality=data_quality,
                )
            )

    return {"runs": runs, "data_quality": data_quality}


def _normalize_run(
    payload: dict[str, Any],
    line_number: int,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    generated_at = _clean_text(payload.get("generated_at"))
    timestamp = _parse_timestamp(generated_at)
    if generated_at is not None and timestamp is None:
        data_quality["invalid_timestamp_runs"] += 1

    selection_status = _normalize_selection_status(payload.get("selection_status"))
    if payload.get("selection_status") is not None and selection_status is None:
        data_quality["invalid_selection_status_runs"] += 1

    selected_horizon = _normalize_horizon(payload.get("selected_horizon"))
    if payload.get("selected_horizon") is not None and selected_horizon is None:
        data_quality["invalid_horizon_values"] += 1

    ranking_value = payload.get("ranking")
    if not isinstance(ranking_value, list):
        ranking_value = []
        data_quality["non_list_ranking_runs"] += 1

    ranking_items: list[dict[str, Any]] = []
    for item in ranking_value:
        if not isinstance(item, dict):
            data_quality["ranking_items_non_object"] += 1
            continue
        ranking_items.append(_normalize_ranking_item(item, data_quality))

    top_candidate = ranking_items[0] if ranking_items else None

    return {
        "line_number": line_number,
        "generated_at": generated_at,
        "timestamp": timestamp,
        "day": timestamp.date().isoformat() if timestamp is not None else None,
        "selection_status": selection_status,
        "reason_codes": _normalize_string_list(payload.get("reason_codes")),
        "candidates_considered": _to_non_negative_int(payload.get("candidates_considered")),
        "ranking_depth": len(ranking_items),
        "selected_symbol": _clean_text(payload.get("selected_symbol")),
        "selected_strategy": _clean_text(payload.get("selected_strategy")),
        "selected_horizon": selected_horizon,
        "selection_score": _to_float(payload.get("selection_score")),
        "selection_confidence": _to_float(payload.get("selection_confidence")),
        "selection_explanation": _clean_text(payload.get("selection_explanation")),
        "latest_window_record_count": _to_non_negative_int(
            payload.get("latest_window_record_count")
        ),
        "cumulative_record_count": _to_non_negative_int(
            payload.get("cumulative_record_count")
        ),
        "ranking_items": ranking_items,
        "top_candidate": top_candidate,
    }


def _normalize_ranking_item(
    item: dict[str, Any],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    horizon = _normalize_horizon(item.get("horizon"))
    if item.get("horizon") is not None and horizon is None:
        data_quality["invalid_horizon_values"] += 1

    candidate_status = _normalize_candidate_status(item.get("candidate_status"))
    if item.get("candidate_status") is not None and candidate_status is None:
        data_quality["invalid_candidate_status_items"] += 1

    visible_horizons = _normalize_horizon_list(item.get("selected_visible_horizons"))
    raw_visible_horizons = item.get("selected_visible_horizons")
    if isinstance(raw_visible_horizons, list):
        invalid_visible_horizon_count = sum(
            1 for value in raw_visible_horizons if _normalize_horizon(value) is None
        )
        data_quality["invalid_horizon_values"] += invalid_visible_horizon_count

    return {
        "rank": _to_positive_int(item.get("rank")),
        "symbol": _clean_text(item.get("symbol")),
        "strategy": _clean_text(item.get("strategy")),
        "horizon": horizon,
        "candidate_status": candidate_status,
        "selection_score": _to_float(item.get("selection_score")),
        "selection_confidence": _to_float(item.get("selection_confidence")),
        "reason_codes": _normalize_string_list(item.get("reason_codes")),
        "latest_sample_size": _to_non_negative_int(item.get("latest_sample_size")),
        "cumulative_sample_size": _to_non_negative_int(item.get("cumulative_sample_size")),
        "symbol_cumulative_support": _to_non_negative_int(
            item.get("symbol_cumulative_support")
        ),
        "strategy_cumulative_support": _to_non_negative_int(
            item.get("strategy_cumulative_support")
        ),
        "selected_candidate_strength": _clean_text(item.get("selected_candidate_strength")),
        "selected_stability_label": _clean_text(item.get("selected_stability_label")),
        "selected_visible_horizons": visible_horizons,
        "source_preference": _clean_text(item.get("source_preference")),
        "edge_stability_score": _to_float(item.get("edge_stability_score")),
        "stability_gate_pass": _to_bool(item.get("stability_gate_pass")),
        "consecutive_visible_cycles": _to_non_negative_int(
            item.get("consecutive_visible_cycles")
        ),
        "consecutive_stable_cycles": _to_non_negative_int(
            item.get("consecutive_stable_cycles")
        ),
        "drift_direction": _clean_text(item.get("drift_direction")),
        "score_delta": _to_float(item.get("score_delta")),
        "drift_blocked": _to_bool(item.get("drift_blocked")),
    }


def _build_overall_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(runs)
    selection_status_counter = Counter(
        run["selection_status"]
        for run in runs
        if run.get("selection_status") is not None
    )
    run_reason_code_counter = Counter()
    ranking_reason_code_counter = Counter()
    candidate_status_counter = Counter()
    candidate_strength_counter = Counter()
    stability_label_counter = Counter()
    drift_direction_counter = Counter()
    selected_symbol_counter = Counter()
    selected_strategy_counter = Counter()
    selected_horizon_counter = Counter()
    source_preference_counter = Counter()
    visible_horizon_counter = Counter()

    candidates_considered_values: list[float] = []
    ranking_depth_values: list[float] = []
    selection_score_values: list[float] = []
    selection_confidence_values: list[float] = []
    latest_window_record_count_values: list[float] = []
    cumulative_record_count_values: list[float] = []
    support_values: dict[str, list[float]] = defaultdict(list)

    drift_blocked_true = 0
    drift_blocked_known = 0
    total_ranking_items = 0

    for run in runs:
        run_reason_code_counter.update(run.get("reason_codes") or [])

        _append_numeric_if_present(
            candidates_considered_values,
            run.get("candidates_considered"),
        )
        ranking_depth_values.append(float(run.get("ranking_depth", 0) or 0))
        _append_numeric_if_present(selection_score_values, run.get("selection_score"))
        _append_numeric_if_present(
            selection_confidence_values,
            run.get("selection_confidence"),
        )
        _append_numeric_if_present(
            latest_window_record_count_values,
            run.get("latest_window_record_count"),
        )
        _append_numeric_if_present(
            cumulative_record_count_values,
            run.get("cumulative_record_count"),
        )

        selected_symbol = run.get("selected_symbol")
        if selected_symbol is not None:
            selected_symbol_counter[selected_symbol] += 1

        selected_strategy = run.get("selected_strategy")
        if selected_strategy is not None:
            selected_strategy_counter[selected_strategy] += 1

        selected_horizon = run.get("selected_horizon")
        if selected_horizon is not None:
            selected_horizon_counter[selected_horizon] += 1

        for item in run.get("ranking_items") or []:
            total_ranking_items += 1
            ranking_reason_code_counter.update(item.get("reason_codes") or [])

            candidate_status = item.get("candidate_status")
            if candidate_status is not None:
                candidate_status_counter[candidate_status] += 1

            candidate_strength = item.get("selected_candidate_strength")
            if candidate_strength is not None:
                candidate_strength_counter[candidate_strength] += 1

            stability_label = item.get("selected_stability_label")
            if stability_label is not None:
                stability_label_counter[stability_label] += 1

            drift_direction = item.get("drift_direction")
            if drift_direction is not None:
                drift_direction_counter[drift_direction] += 1

            source_preference = item.get("source_preference")
            if source_preference is not None:
                source_preference_counter[source_preference] += 1

            for horizon in item.get("selected_visible_horizons") or []:
                visible_horizon_counter[horizon] += 1

            drift_blocked = item.get("drift_blocked")
            if drift_blocked is not None:
                drift_blocked_known += 1
                if drift_blocked:
                    drift_blocked_true += 1

            _append_numeric_if_present(
                support_values["latest_sample_size"],
                item.get("latest_sample_size"),
            )
            _append_numeric_if_present(
                support_values["cumulative_sample_size"],
                item.get("cumulative_sample_size"),
            )
            _append_numeric_if_present(
                support_values["symbol_cumulative_support"],
                item.get("symbol_cumulative_support"),
            )
            _append_numeric_if_present(
                support_values["strategy_cumulative_support"],
                item.get("strategy_cumulative_support"),
            )
            _append_numeric_if_present(
                support_values["edge_stability_score"],
                item.get("edge_stability_score"),
            )
            _append_numeric_if_present(
                support_values["score_delta"],
                item.get("score_delta"),
            )
            _append_numeric_if_present(
                support_values["consecutive_visible_cycles"],
                item.get("consecutive_visible_cycles"),
            )
            _append_numeric_if_present(
                support_values["consecutive_stable_cycles"],
                item.get("consecutive_stable_cycles"),
            )

    selected_runs = selection_status_counter.get("selected", 0)
    abstain_runs = selection_status_counter.get("abstain", 0)
    blocked_runs = selection_status_counter.get("blocked", 0)

    return {
        "total_runs": total_runs,
        "selection_status": _build_count_rate_map(selection_status_counter, total_runs),
        "selected_runs": selected_runs,
        "abstain_runs": abstain_runs,
        "blocked_runs": blocked_runs,
        "selected_rate": _ratio(selected_runs, total_runs),
        "abstain_rate": _ratio(abstain_runs, total_runs),
        "blocked_rate": _ratio(blocked_runs, total_runs),
        "average_candidates_considered": _average(candidates_considered_values),
        "average_ranking_depth": _average(ranking_depth_values),
        "run_reason_code_frequencies": _build_frequency_rows(
            run_reason_code_counter,
            total_runs,
        ),
        "ranking_reason_code_frequencies": _build_frequency_rows(
            ranking_reason_code_counter,
            total_ranking_items,
        ),
        "candidate_status_frequencies": _build_frequency_rows(
            candidate_status_counter,
            total_ranking_items,
        ),
        "candidate_strength_distribution": _build_frequency_rows(
            candidate_strength_counter,
            total_ranking_items,
        ),
        "stability_label_distribution": _build_frequency_rows(
            stability_label_counter,
            total_ranking_items,
        ),
        "drift_direction_distribution": _build_frequency_rows(
            drift_direction_counter,
            total_ranking_items,
        ),
        "source_preference_distribution": _build_frequency_rows(
            source_preference_counter,
            total_ranking_items,
        ),
        "visible_horizon_distribution": _build_frequency_rows(
            visible_horizon_counter,
            total_ranking_items,
        ),
        "drift_blocked_rate": _ratio(drift_blocked_true, drift_blocked_known),
        "selected_symbol_frequency": _build_frequency_rows(
            selected_symbol_counter,
            selected_runs,
        ),
        "selected_strategy_frequency": _build_frequency_rows(
            selected_strategy_counter,
            selected_runs,
        ),
        "selected_horizon_frequency": _build_frequency_rows(
            selected_horizon_counter,
            selected_runs,
        ),
        "secondary_metrics": {
            "selection_score": _build_number_summary(selection_score_values),
            "selection_confidence": _build_number_summary(selection_confidence_values),
            "latest_window_record_count": _build_number_summary(
                latest_window_record_count_values
            ),
            "cumulative_record_count": _build_number_summary(
                cumulative_record_count_values
            ),
        },
        "support_metrics": _build_support_metrics(support_values),
        "top_ranked_candidate_summaries": _build_top_candidate_summary(runs),
        "selection_status_streak_summaries": _build_status_streak_summary(runs),
        "consecutive_abstain_streaks": _build_consecutive_abstain_streaks(runs),
    }


def _build_support_metrics(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        key: _build_number_summary(numbers)
        for key, numbers in sorted(values.items(), key=lambda item: item[0])
    }


def _build_top_candidate_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    identity_counter = Counter()
    reason_counter = Counter()
    status_counter = Counter()
    strength_counter = Counter()
    stability_counter = Counter()
    drift_counter = Counter()
    source_preference_counter = Counter()
    visible_horizon_counter = Counter()

    top_identities: list[tuple[str, str, str] | None] = []

    for run in runs:
        top_candidate = run.get("top_candidate")
        identity = _candidate_identity(top_candidate)
        top_identities.append(identity)

        if identity is not None:
            identity_counter[identity] += 1

        if not isinstance(top_candidate, dict):
            continue

        reason_counter.update(top_candidate.get("reason_codes") or [])

        candidate_status = top_candidate.get("candidate_status")
        if candidate_status is not None:
            status_counter[candidate_status] += 1

        candidate_strength = top_candidate.get("selected_candidate_strength")
        if candidate_strength is not None:
            strength_counter[candidate_strength] += 1

        stability_label = top_candidate.get("selected_stability_label")
        if stability_label is not None:
            stability_counter[stability_label] += 1

        drift_direction = top_candidate.get("drift_direction")
        if drift_direction is not None:
            drift_counter[drift_direction] += 1

        source_preference = top_candidate.get("source_preference")
        if source_preference is not None:
            source_preference_counter[source_preference] += 1

        for horizon in top_candidate.get("selected_visible_horizons") or []:
            visible_horizon_counter[horizon] += 1

    runs_with_top_candidate = sum(1 for identity in top_identities if identity is not None)
    comparable_transitions = 0
    replacement_count = 0
    previous_identity: tuple[str, str, str] | None = None

    for identity in top_identities:
        if previous_identity is not None and identity is not None:
            comparable_transitions += 1
            if identity != previous_identity:
                replacement_count += 1
        previous_identity = identity

    streaks = _build_identity_streaks(top_identities, runs)
    identity_frequency_rows: list[dict[str, Any]] = []

    for identity, count in sorted(
        identity_counter.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    )[:MAX_FREQUENCY_ROWS]:
        identity_frequency_rows.append(
            {
                "symbol": identity[0],
                "strategy": identity[1],
                "horizon": identity[2],
                "count": count,
                "rate": _ratio(count, runs_with_top_candidate),
            }
        )

    return {
        "runs_with_top_candidate": runs_with_top_candidate,
        "identity_frequencies": identity_frequency_rows,
        "repeated_identity_count": sum(
            1 for count in identity_counter.values() if count > 1
        ),
        "repeated_top_candidate_run_count": sum(
            count for count in identity_counter.values() if count > 1
        ),
        "replacement_count": replacement_count,
        "comparable_transitions": comparable_transitions,
        "replacement_rate": _ratio(replacement_count, comparable_transitions),
        "reason_code_frequencies": _build_frequency_rows(reason_counter, runs_with_top_candidate),
        "candidate_status_frequencies": _build_frequency_rows(
            status_counter,
            runs_with_top_candidate,
        ),
        "candidate_strength_distribution": _build_frequency_rows(
            strength_counter,
            runs_with_top_candidate,
        ),
        "stability_label_distribution": _build_frequency_rows(
            stability_counter,
            runs_with_top_candidate,
        ),
        "drift_direction_distribution": _build_frequency_rows(
            drift_counter,
            runs_with_top_candidate,
        ),
        "source_preference_distribution": _build_frequency_rows(
            source_preference_counter,
            runs_with_top_candidate,
        ),
        "visible_horizon_distribution": _build_frequency_rows(
            visible_horizon_counter,
            runs_with_top_candidate,
        ),
        "consecutive_appearance_summary": {
            "streak_count": len(streaks),
            "max_streak": max((row["length"] for row in streaks), default=0),
            "average_streak": _average([float(row["length"]) for row in streaks]),
            "identity_streaks": _summarize_identity_streaks(streaks),
        },
    }


def _build_status_streak_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    streaks = _build_status_streaks(runs)
    grouped: dict[str, list[int]] = defaultdict(list)

    for streak in streaks:
        grouped[streak["selection_status"]].append(streak["length"])

    rows: list[dict[str, Any]] = []
    for selection_status in VALID_SELECTION_STATUSES:
        lengths = grouped.get(selection_status, [])
        rows.append(
            {
                "selection_status": selection_status,
                "streak_count": len(lengths),
                "max_streak": max(lengths, default=0),
                "average_streak": _average([float(length) for length in lengths]),
            }
        )

    return rows


def _build_consecutive_abstain_streaks(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    streaks = [
        streak
        for streak in _build_status_streaks(runs)
        if streak["selection_status"] == "abstain"
    ]
    return sorted(
        streaks,
        key=lambda item: (-item["length"], item["start_run_index"] or 0),
    )[:MAX_FREQUENCY_ROWS]


def _build_status_streaks(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    streaks: list[dict[str, Any]] = []
    current_status: str | None = None
    current_start_index: int | None = None
    current_start_line: int | None = None
    current_start_time: str | None = None
    current_length = 0

    for run_index, run in enumerate(runs, start=1):
        status = run.get("selection_status")
        if status not in VALID_SELECTION_STATUSES:
            if current_status is not None:
                previous_run = runs[run_index - 2]
                streaks.append(
                    {
                        "selection_status": current_status,
                        "length": current_length,
                        "start_run_index": current_start_index,
                        "end_run_index": run_index - 1,
                        "start_line_number": current_start_line,
                        "end_line_number": previous_run["line_number"],
                        "start_generated_at": current_start_time,
                        "end_generated_at": previous_run.get("generated_at"),
                    }
                )
            current_status = None
            current_start_index = None
            current_start_line = None
            current_start_time = None
            current_length = 0
            continue

        if status == current_status:
            current_length += 1
            continue

        if current_status is not None:
            previous_run = runs[run_index - 2]
            streaks.append(
                {
                    "selection_status": current_status,
                    "length": current_length,
                    "start_run_index": current_start_index,
                    "end_run_index": run_index - 1,
                    "start_line_number": current_start_line,
                    "end_line_number": previous_run["line_number"],
                    "start_generated_at": current_start_time,
                    "end_generated_at": previous_run.get("generated_at"),
                }
            )

        current_status = status
        current_start_index = run_index
        current_start_line = run["line_number"]
        current_start_time = run.get("generated_at")
        current_length = 1

    if current_status is not None and runs:
        last_run = runs[-1]
        streaks.append(
            {
                "selection_status": current_status,
                "length": current_length,
                "start_run_index": current_start_index,
                "end_run_index": len(runs),
                "start_line_number": current_start_line,
                "end_line_number": last_run["line_number"],
                "start_generated_at": current_start_time,
                "end_generated_at": last_run.get("generated_at"),
            }
        )

    return streaks


def _build_identity_streaks(
    top_identities: list[tuple[str, str, str] | None],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    streaks: list[dict[str, Any]] = []
    current_identity: tuple[str, str, str] | None = None
    current_start_index: int | None = None
    current_start_line: int | None = None
    current_start_time: str | None = None
    current_length = 0

    for run_index, identity in enumerate(top_identities, start=1):
        if identity is None:
            if current_identity is not None:
                previous_run = runs[run_index - 2]
                streaks.append(
                    _build_identity_streak_row(
                        current_identity,
                        current_length,
                        current_start_index,
                        run_index - 1,
                        current_start_line,
                        previous_run["line_number"],
                        current_start_time,
                        previous_run.get("generated_at"),
                    )
                )
            current_identity = None
            current_start_index = None
            current_start_line = None
            current_start_time = None
            current_length = 0
            continue

        if identity == current_identity:
            current_length += 1
            continue

        if current_identity is not None:
            previous_run = runs[run_index - 2]
            streaks.append(
                _build_identity_streak_row(
                    current_identity,
                    current_length,
                    current_start_index,
                    run_index - 1,
                    current_start_line,
                    previous_run["line_number"],
                    current_start_time,
                    previous_run.get("generated_at"),
                )
            )

        current_identity = identity
        current_start_index = run_index
        current_start_line = runs[run_index - 1]["line_number"]
        current_start_time = runs[run_index - 1].get("generated_at")
        current_length = 1

    if current_identity is not None and runs:
        last_run = runs[-1]
        streaks.append(
            _build_identity_streak_row(
                current_identity,
                current_length,
                current_start_index,
                len(runs),
                current_start_line,
                last_run["line_number"],
                current_start_time,
                last_run.get("generated_at"),
            )
        )

    return streaks


def _build_by_day_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        day = run.get("day")
        if day is not None:
            grouped_runs[day].append(run)

    rows: list[dict[str, Any]] = []
    for day in sorted(grouped_runs):
        day_runs = grouped_runs[day]
        selection_status_counter = Counter(
            run["selection_status"]
            for run in day_runs
            if run.get("selection_status") is not None
        )
        top_identities = [_candidate_identity(run.get("top_candidate")) for run in day_runs]
        comparable_transitions = 0
        replacement_count = 0
        previous_identity: tuple[str, str, str] | None = None

        for identity in top_identities:
            if previous_identity is not None and identity is not None:
                comparable_transitions += 1
                if identity != previous_identity:
                    replacement_count += 1
            previous_identity = identity

        rows.append(
            {
                "day": day,
                "runs": len(day_runs),
                "selection_status": _build_count_rate_map(
                    selection_status_counter,
                    len(day_runs),
                ),
                "abstain_runs": selection_status_counter.get("abstain", 0),
                "selected_runs": selection_status_counter.get("selected", 0),
                "blocked_runs": selection_status_counter.get("blocked", 0),
                "average_candidates_considered": _average(
                    [
                        float(run["candidates_considered"])
                        for run in day_runs
                        if run.get("candidates_considered") is not None
                    ]
                ),
                "average_ranking_depth": _average(
                    [float(run.get("ranking_depth", 0) or 0) for run in day_runs]
                ),
                "top_candidate_replacement_count": replacement_count,
                "top_candidate_replacement_rate": _ratio(
                    replacement_count,
                    comparable_transitions,
                ),
                "unique_top_ranked_candidates": len(
                    {identity for identity in top_identities if identity is not None}
                ),
            }
        )

    return rows


def _build_recent_runs_summary(
    runs: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run in runs[-limit:]:
        top_candidate = run.get("top_candidate") or {}
        rows.append(
            {
                "line_number": run["line_number"],
                "generated_at": run.get("generated_at"),
                "selection_status": run.get("selection_status"),
                "reason_codes": run.get("reason_codes") or [],
                "candidates_considered": run.get("candidates_considered"),
                "ranking_depth": run.get("ranking_depth"),
                "selected_symbol": run.get("selected_symbol"),
                "selected_strategy": run.get("selected_strategy"),
                "selected_horizon": run.get("selected_horizon"),
                "top_candidate": {
                    "symbol": top_candidate.get("symbol"),
                    "strategy": top_candidate.get("strategy"),
                    "horizon": top_candidate.get("horizon"),
                    "candidate_status": top_candidate.get("candidate_status"),
                    "selected_candidate_strength": top_candidate.get(
                        "selected_candidate_strength"
                    ),
                    "selected_stability_label": top_candidate.get(
                        "selected_stability_label"
                    ),
                    "source_preference": top_candidate.get("source_preference"),
                    "selected_visible_horizons": top_candidate.get("selected_visible_horizons")
                    or [],
                    "drift_direction": top_candidate.get("drift_direction"),
                    "drift_blocked": top_candidate.get("drift_blocked"),
                    "reason_codes": top_candidate.get("reason_codes") or [],
                },
            }
        )

    return rows


def _build_window_summary(
    runs: list[dict[str, Any]],
    latest_timestamp: datetime | None,
    window: timedelta,
    window_name: str,
) -> dict[str, Any] | None:
    if latest_timestamp is None:
        return None

    threshold = latest_timestamp - window
    window_runs = [
        run
        for run in runs
        if isinstance(run.get("timestamp"), datetime) and run["timestamp"] >= threshold
    ]

    return {
        "window": window_name,
        "reference_timestamp": latest_timestamp.isoformat(),
        "run_count": len(window_runs),
        "summary": _build_overall_summary(window_runs),
    }


def _finalize_data_quality(
    data_quality: dict[str, Any],
    runs: list[dict[str, Any]],
    latest_timestamp: datetime | None,
) -> dict[str, Any]:
    runs_with_timestamp = sum(1 for run in runs if run.get("timestamp") is not None)
    runs_without_timestamp = len(runs) - runs_with_timestamp
    runs_missing_selection_status = sum(
        1 for run in runs if run.get("selection_status") is None
    )
    runs_with_empty_ranking = sum(
        1 for run in runs if (run.get("ranking_depth") or 0) == 0
    )
    runs_with_top_candidate = sum(
        1 for run in runs if run.get("top_candidate") is not None
    )

    return {
        **data_quality,
        "parsed_runs": len(runs),
        "runs_with_timestamp": runs_with_timestamp,
        "runs_without_timestamp": runs_without_timestamp,
        "timestamp_coverage_rate": _ratio(runs_with_timestamp, len(runs)),
        "runs_missing_selection_status": runs_missing_selection_status,
        "runs_with_empty_ranking": runs_with_empty_ranking,
        "runs_with_top_candidate": runs_with_top_candidate,
        "latest_observed_timestamp": latest_timestamp.isoformat()
        if latest_timestamp is not None
        else None,
    }


def _build_unavailable_metrics(
    runs: list[dict[str, Any]],
    latest_timestamp: datetime | None,
) -> list[dict[str, Any]]:
    unavailable = [
        {
            "metric": "realized_outcome_quality",
            "reason": (
                "Shadow logs do not include post-selection truth labels, realized returns, "
                "or PnL."
            ),
        },
        {
            "metric": "win_rate",
            "reason": "Shadow logs do not include realized trade outcomes.",
        },
        {
            "metric": "abstain_correctness",
            "reason": (
                "Shadow logs do not include later truth labels for abstained runs."
            ),
        },
        {
            "metric": "missed_opportunity_rate",
            "reason": (
                "Shadow logs do not include a ground-truth set of all later-valid "
                "opportunities."
            ),
        },
    ]

    if latest_timestamp is None:
        unavailable.extend(
            [
                {
                    "metric": "last_24h",
                    "reason": "No parseable generated_at timestamps were available.",
                },
                {
                    "metric": "last_7d",
                    "reason": "No parseable generated_at timestamps were available.",
                },
                {
                    "metric": "by_day",
                    "reason": "No parseable generated_at timestamps were available.",
                },
            ]
        )
    else:
        runs_without_timestamp = sum(1 for run in runs if run.get("timestamp") is None)
        if runs_without_timestamp > 0:
            unavailable.append(
                {
                    "metric": "full_timestamp_coverage",
                    "reason": (
                        "Some runs are missing parseable generated_at values, so time-based "
                        "sections use the timestamped subset only."
                    ),
                }
            )

    return unavailable


def _build_count_rate_map(counter: Counter[str], total: int) -> dict[str, dict[str, Any]]:
    return {
        status: {
            "count": counter.get(status, 0),
            "rate": _ratio(counter.get(status, 0), total),
        }
        for status in VALID_SELECTION_STATUSES
    }


def _build_frequency_rows(counter: Counter[Any], total: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        rows.append(
            {
                "value": key,
                "count": count,
                "rate": _ratio(count, total),
            }
        )
    return rows[:MAX_FREQUENCY_ROWS]


def _build_number_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "present_count": 0,
            "average": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "present_count": len(values),
        "average": _average(values),
        "median": round(float(median(values)), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _summarize_identity_streaks(streaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for streak in streaks:
        identity = (streak["symbol"], streak["strategy"], streak["horizon"])
        grouped[identity].append(streak)

    rows: list[dict[str, Any]] = []
    for identity, identity_streaks in sorted(
        grouped.items(),
        key=lambda item: (
            -max(streak["length"] for streak in item[1]),
            item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        lengths = [float(streak["length"]) for streak in identity_streaks]
        rows.append(
            {
                "symbol": identity[0],
                "strategy": identity[1],
                "horizon": identity[2],
                "appearance_count": int(sum(lengths)),
                "streak_count": len(identity_streaks),
                "max_streak": int(max(lengths, default=0)),
                "average_streak": _average(lengths),
            }
        )

    return rows[:MAX_FREQUENCY_ROWS]


def _build_identity_streak_row(
    identity: tuple[str, str, str],
    length: int,
    start_run_index: int | None,
    end_run_index: int | None,
    start_line_number: int | None,
    end_line_number: int | None,
    start_generated_at: str | None,
    end_generated_at: str | None,
) -> dict[str, Any]:
    return {
        "symbol": identity[0],
        "strategy": identity[1],
        "horizon": identity[2],
        "length": length,
        "start_run_index": start_run_index,
        "end_run_index": end_run_index,
        "start_line_number": start_line_number,
        "end_line_number": end_line_number,
        "start_generated_at": start_generated_at,
        "end_generated_at": end_generated_at,
    }


def _candidate_identity(candidate: dict[str, Any] | None) -> tuple[str, str, str] | None:
    if not isinstance(candidate, dict):
        return None

    symbol = candidate.get("symbol")
    strategy = candidate.get("strategy")
    horizon = candidate.get("horizon")

    if symbol is None or strategy is None or horizon is None:
        return None

    return (symbol, strategy, horizon)


def _latest_timestamp(runs: list[dict[str, Any]]) -> datetime | None:
    timestamps = [
        run["timestamp"]
        for run in runs
        if isinstance(run.get("timestamp"), datetime)
    ]
    if not timestamps:
        return None
    return max(timestamps)


def _append_numeric_if_present(values: list[float], value: Any) -> None:
    numeric_value = _to_float(value)
    if numeric_value is not None:
        values.append(numeric_value)


def _append_limited(values: list[int], value: int, limit: int) -> None:
    if len(values) < limit:
        values.append(value)


def _normalize_selection_status(value: Any) -> str | None:
    text = _clean_text(value)
    if text in VALID_SELECTION_STATUSES:
        return text
    return None


def _normalize_candidate_status(value: Any) -> str | None:
    text = _clean_text(value)
    if text in VALID_CANDIDATE_STATUSES:
        return text
    return None


def _normalize_horizon(value: Any) -> str | None:
    text = _clean_text(value)
    if text in VALID_HORIZONS:
        return text
    return None


def _normalize_horizon_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    horizons: list[str] = []
    for item in value:
        horizon = _normalize_horizon(item)
        if horizon is not None and horizon not in horizons:
            horizons.append(horizon)
    return horizons


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text is not None:
            normalized.append(text)
    return normalized


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_non_negative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _to_positive_int(value: Any) -> int | None:
    parsed = _to_non_negative_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    text = _clean_text(value)
    if text is None:
        return None

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _render_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall", {}) or {}
    top_ranked = overall.get("top_ranked_candidate_summaries", {}) or {}
    data_quality = summary.get("data_quality", {}) or {}
    last_24h = summary.get("last_24h")
    last_7d = summary.get("last_7d")

    lines = [
        "# Shadow Observation Summary",
        "",
        f"- Generated at: {summary.get('generated_at') or 'n/a'}",
        f"- Input path: `{summary.get('input_path') or 'n/a'}`",
        f"- Total runs: {overall.get('total_runs', 0)}",
        f"- Parsed runs: {data_quality.get('parsed_runs', 0)}",
        f"- Malformed lines skipped: {data_quality.get('malformed_lines', 0)}",
        "",
        "## Overall",
        "",
        f"- Selected runs: {overall.get('selected_runs', 0)} ({_format_pct(overall.get('selected_rate'))})",
        f"- Abstain runs: {overall.get('abstain_runs', 0)} ({_format_pct(overall.get('abstain_rate'))})",
        f"- Blocked runs: {overall.get('blocked_runs', 0)} ({_format_pct(overall.get('blocked_rate'))})",
        f"- Average candidates considered: {_format_number(overall.get('average_candidates_considered'))}",
        f"- Average ranking depth: {_format_number(overall.get('average_ranking_depth'))}",
        f"- Drift blocked rate: {_format_pct(overall.get('drift_blocked_rate'))}",
        "",
        "## Recent Windows",
        "",
    ]

    if isinstance(last_24h, dict):
        window_summary = (last_24h.get("summary") or {})
        lines.extend(
            [
                f"- Last 24h run count: {last_24h.get('run_count', 0)}",
                f"  - Selected rate: {_format_pct(window_summary.get('selected_rate'))}",
                f"  - Abstain rate: {_format_pct(window_summary.get('abstain_rate'))}",
            ]
        )
    else:
        lines.append("- Last 24h: n/a")

    if isinstance(last_7d, dict):
        window_summary = (last_7d.get("summary") or {})
        lines.extend(
            [
                f"- Last 7d run count: {last_7d.get('run_count', 0)}",
                f"  - Selected rate: {_format_pct(window_summary.get('selected_rate'))}",
                f"  - Abstain rate: {_format_pct(window_summary.get('abstain_rate'))}",
            ]
        )
    else:
        lines.append("- Last 7d: n/a")

    lines.extend(
        [
            "",
            "## Selection Status",
            "",
            "| Status | Count | Rate |",
            "| --- | ---: | ---: |",
        ]
    )

    selection_status = overall.get("selection_status", {}) or {}
    for status in VALID_SELECTION_STATUSES:
        row = selection_status.get(status, {}) or {}
        lines.append(
            f"| {status} | {row.get('count', 0)} | {_format_pct(row.get('rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Run Reason Codes",
            "",
            "| Reason Code | Count | Rate |",
            "| --- | ---: | ---: |",
        ]
    )

    for row in (overall.get("run_reason_code_frequencies", []) or [])[:10]:
        lines.append(
            f"| {row.get('value') or 'n/a'} | {row.get('count', 0)} | {_format_pct(row.get('rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Top-Ranked Candidate Behavior",
            "",
            f"- Runs with top candidate: {top_ranked.get('runs_with_top_candidate', 0)}",
            f"- Replacement count: {top_ranked.get('replacement_count', 0)}",
            f"- Replacement rate: {_format_pct(top_ranked.get('replacement_rate'))}",
            "",
            "| Symbol | Strategy | Horizon | Count | Rate |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )

    for row in (top_ranked.get("identity_frequencies", []) or [])[:10]:
        lines.append(
            "| {symbol} | {strategy} | {horizon} | {count} | {rate} |".format(
                symbol=row.get("symbol") or "n/a",
                strategy=row.get("strategy") or "n/a",
                horizon=row.get("horizon") or "n/a",
                count=row.get("count", 0),
                rate=_format_pct(row.get("rate")),
            )
        )

    lines.extend(
        [
            "",
            "## Consecutive Abstain Streaks",
            "",
            "| Length | Start Run | End Run | Start Time | End Time |",
            "| ---: | ---: | ---: | --- | --- |",
        ]
    )

    for row in (overall.get("consecutive_abstain_streaks", []) or [])[:10]:
        lines.append(
            "| {length} | {start_run} | {end_run} | {start_time} | {end_time} |".format(
                length=row.get("length", 0),
                start_run=row.get("start_run_index") or "n/a",
                end_run=row.get("end_run_index") or "n/a",
                start_time=row.get("start_generated_at") or "n/a",
                end_time=row.get("end_generated_at") or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## By Day",
            "",
            "| Day | Runs | Selected | Abstain | Blocked | Avg Candidates | Avg Depth |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in (summary.get("by_day", []) or [])[-10:]:
        status_row = row.get("selection_status", {}) or {}
        lines.append(
            "| {day} | {runs} | {selected} | {abstain} | {blocked} | {avg_candidates} | {avg_depth} |".format(
                day=row.get("day") or "n/a",
                runs=row.get("runs", 0),
                selected=(status_row.get("selected", {}) or {}).get("count", 0),
                abstain=(status_row.get("abstain", {}) or {}).get("count", 0),
                blocked=(status_row.get("blocked", {}) or {}).get("count", 0),
                avg_candidates=_format_number(row.get("average_candidates_considered")),
                avg_depth=_format_number(row.get("average_ranking_depth")),
            )
        )

    lines.extend(
        [
            "",
            "## Recent Runs",
            "",
            "| Line | Generated At | Status | Top Symbol | Top Strategy | Top Horizon | Top Status |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in summary.get("recent_runs", []) or []:
        top_candidate = row.get("top_candidate", {}) or {}
        lines.append(
            "| {line} | {generated_at} | {status} | {symbol} | {strategy} | {horizon} | {candidate_status} |".format(
                line=row.get("line_number", 0),
                generated_at=row.get("generated_at") or "n/a",
                status=row.get("selection_status") or "n/a",
                symbol=top_candidate.get("symbol") or "n/a",
                strategy=top_candidate.get("strategy") or "n/a",
                horizon=top_candidate.get("horizon") or "n/a",
                candidate_status=top_candidate.get("candidate_status") or "n/a",
            )
        )

    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            f"- Input exists: {data_quality.get('input_exists')}",
            f"- Input is file: {data_quality.get('input_is_file')}",
            f"- Total lines: {data_quality.get('total_lines', 0)}",
            f"- Blank lines: {data_quality.get('blank_lines', 0)}",
            f"- Malformed lines: {data_quality.get('malformed_lines', 0)}",
            f"- Invalid timestamp runs: {data_quality.get('invalid_timestamp_runs', 0)}",
            f"- Invalid selection status runs: {data_quality.get('invalid_selection_status_runs', 0)}",
            f"- Invalid candidate status items: {data_quality.get('invalid_candidate_status_items', 0)}",
            f"- Invalid horizon values: {data_quality.get('invalid_horizon_values', 0)}",
            f"- Runs without timestamp: {data_quality.get('runs_without_timestamp', 0)}",
            f"- Runs missing selection status: {data_quality.get('runs_missing_selection_status', 0)}",
            "",
            "## Unavailable Metrics",
            "",
        ]
    )

    for item in summary.get("unavailable_metrics", []) or []:
        lines.append(
            f"- {item.get('metric') or 'unknown'}: {item.get('reason') or 'n/a'}"
        )

    lines.append("")
    return "\n".join(lines)


def _format_pct(value: Any) -> str:
    numeric_value = _to_float(value)
    if numeric_value is None:
        return "n/a"
    return f"{round(numeric_value * 100, 2)}%"


def _format_number(value: Any) -> str:
    numeric_value = _to_float(value)
    if numeric_value is None:
        return "n/a"
    return str(round(numeric_value, 4))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze append-only edge selection shadow JSONL logs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to logs/edge_selection_shadow/edge_selection_shadow.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for shadow observation summary outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_shadow_observation_analyzer(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "input_path": result["input_path"],
                "output_dir": result["output_dir"],
                "summary_json": result["summary_json"],
                "summary_md": result["summary_md"],
                "run_count": result["run_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
