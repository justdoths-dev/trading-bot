from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "edge_selection_shadow"
    / "edge_selection_shadow.jsonl"
)

VALID_HORIZONS = ("15m", "1h", "4h")
MAX_EXAMPLES_PER_CATEGORY = 3

GROUPING_REASONS = {
    "no_valid_symbol_group",
    "no_valid_strategy_group",
    "no_valid_symbol_or_strategy_group",
}


def run_grouping_failure_deep_diagnosis_report(
    input_path: Path | None = None,
) -> dict[str, Any]:
    resolved = input_path or DEFAULT_INPUT_PATH
    loaded = load_shadow_records(resolved)
    summary = build_grouping_failure_deep_diagnosis_summary(
        records=loaded["records"],
        input_path=resolved,
        data_quality=loaded["data_quality"],
    )
    return {
        "input_path": str(resolved),
        "summary": summary,
        "markdown": render_grouping_failure_deep_diagnosis_markdown(summary),
    }


def load_shadow_records(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    data_quality = {
        "input_exists": path.exists(),
        "input_is_file": path.is_file() if path.exists() else False,
        "total_lines": 0,
        "valid_records": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "non_object_lines": 0,
        "malformed_line_numbers": [],
    }

    if not path.exists() or not path.is_file():
        return {"records": records, "data_quality": data_quality}

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
                data_quality["malformed_line_numbers"].append(line_number)
                continue

            if not isinstance(payload, dict):
                data_quality["malformed_lines"] += 1
                data_quality["non_object_lines"] += 1
                data_quality["malformed_line_numbers"].append(line_number)
                continue

            records.append(payload)
            data_quality["valid_records"] += 1

    return {"records": records, "data_quality": data_quality}


def build_grouping_failure_deep_diagnosis_summary(
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    total_horizon_rows = 0
    grouping_blocked_horizon_rows = 0
    grouping_free_horizon_rows = 0

    grouping_reason_dist: Counter[str] = Counter()
    grouping_combo_dist: Counter[str] = Counter()

    per_horizon_reason_dist: dict[str, Counter[str]] = defaultdict(Counter)
    per_horizon_combo_dist: dict[str, Counter[str]] = defaultdict(Counter)

    per_horizon_total: Counter[str] = Counter()
    per_horizon_blocked: Counter[str] = Counter()
    per_horizon_free: Counter[str] = Counter()

    symbol_only_count = 0
    strategy_only_count = 0
    symbol_or_strategy_combined_count = 0
    mixed_grouping_block_count = 0

    total_records = len(records)
    records_with_any_grouping_block = 0
    records_all_horizons_grouping_blocked = 0
    records_partial_grouping_blocked = 0
    records_no_grouping_block = 0

    examples: dict[str, list[dict[str, Any]]] = {
        "symbol_only_examples": [],
        "strategy_only_examples": [],
        "combined_grouping_examples": [],
        "all_horizons_blocked_examples": [],
        "partial_block_examples": [],
    }

    for record in records:
        horizon_rows = _extract_horizon_rows(record)

        record_grouping_blocked_horizons: list[str] = []
        record_grouping_free_horizons: list[str] = []

        for row in horizon_rows:
            horizon = row["horizon"]
            grouping_reasons = row["grouping_reasons"]
            has_grouping_block = bool(grouping_reasons)

            total_horizon_rows += 1
            per_horizon_total[horizon] += 1

            combo_key = _combo_key(grouping_reasons)
            grouping_combo_dist[combo_key] += 1
            per_horizon_combo_dist[horizon][combo_key] += 1

            if has_grouping_block:
                grouping_blocked_horizon_rows += 1
                per_horizon_blocked[horizon] += 1
                record_grouping_blocked_horizons.append(horizon)

                for reason in grouping_reasons:
                    grouping_reason_dist[reason] += 1
                    per_horizon_reason_dist[horizon][reason] += 1

                category = _grouping_category(grouping_reasons)
                example = _build_example(record, row)

                if category == "symbol_only":
                    symbol_only_count += 1
                    _push_example(examples["symbol_only_examples"], example)
                elif category == "strategy_only":
                    strategy_only_count += 1
                    _push_example(examples["strategy_only_examples"], example)
                elif category == "symbol_or_strategy_combined":
                    symbol_or_strategy_combined_count += 1
                    _push_example(examples["combined_grouping_examples"], example)
                else:
                    mixed_grouping_block_count += 1
                    _push_example(examples["combined_grouping_examples"], example)
            else:
                grouping_free_horizon_rows += 1
                per_horizon_free[horizon] += 1
                record_grouping_free_horizons.append(horizon)

        if record_grouping_blocked_horizons:
            records_with_any_grouping_block += 1

            if len(record_grouping_free_horizons) == 0 and len(record_grouping_blocked_horizons) > 0:
                records_all_horizons_grouping_blocked += 1
                _push_example(
                    examples["all_horizons_blocked_examples"],
                    _build_record_example(
                        record,
                        blocked_horizons=record_grouping_blocked_horizons,
                        free_horizons=record_grouping_free_horizons,
                    ),
                )
            else:
                records_partial_grouping_blocked += 1
                _push_example(
                    examples["partial_block_examples"],
                    _build_record_example(
                        record,
                        blocked_horizons=record_grouping_blocked_horizons,
                        free_horizons=record_grouping_free_horizons,
                    ),
                )
        else:
            records_no_grouping_block += 1

    diagnosis = _build_diagnosis(
        total_records=total_records,
        total_horizon_rows=total_horizon_rows,
        grouping_blocked_horizon_rows=grouping_blocked_horizon_rows,
        grouping_free_horizon_rows=grouping_free_horizon_rows,
        symbol_only_count=symbol_only_count,
        strategy_only_count=strategy_only_count,
        symbol_or_strategy_combined_count=symbol_or_strategy_combined_count,
        mixed_grouping_block_count=mixed_grouping_block_count,
        records_with_any_grouping_block=records_with_any_grouping_block,
        records_all_horizons_grouping_blocked=records_all_horizons_grouping_blocked,
        records_partial_grouping_blocked=records_partial_grouping_blocked,
        per_horizon_blocked=per_horizon_blocked,
        per_horizon_total=per_horizon_total,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "grouping_failure_deep_diagnosis_report",
            "input_path": str(input_path),
            "total_records": total_records,
            "total_horizon_rows": total_horizon_rows,
            "data_quality": data_quality,
        },
        "coverage": {
            "grouping_blocked_horizon_rows": grouping_blocked_horizon_rows,
            "grouping_free_horizon_rows": grouping_free_horizon_rows,
            "grouping_blocked_ratio": _safe_ratio(grouping_blocked_horizon_rows, total_horizon_rows),
            "grouping_free_ratio": _safe_ratio(grouping_free_horizon_rows, total_horizon_rows),
        },
        "reason_distribution": {
            "overall_grouping_reason_distribution": _counter_rows(grouping_reason_dist),
            "overall_grouping_combo_distribution": _counter_rows(grouping_combo_dist),
            "symbol_only_count": symbol_only_count,
            "strategy_only_count": strategy_only_count,
            "symbol_or_strategy_combined_count": symbol_or_strategy_combined_count,
            "mixed_grouping_block_count": mixed_grouping_block_count,
        },
        "per_horizon": {
            "total": _counter_rows(per_horizon_total),
            "blocked": _counter_rows(per_horizon_blocked),
            "free": _counter_rows(per_horizon_free),
            "blocked_rate_by_horizon": _rate_rows(
                numerator=per_horizon_blocked,
                denominator=per_horizon_total,
            ),
            "reason_distribution": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_reason_dist.items())
            },
            "combo_distribution": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_combo_dist.items())
            },
        },
        "record_level_severity": {
            "records_with_any_grouping_block": records_with_any_grouping_block,
            "records_with_any_grouping_block_ratio": _safe_ratio(
                records_with_any_grouping_block,
                total_records,
            ),
            "records_all_horizons_grouping_blocked": records_all_horizons_grouping_blocked,
            "records_all_horizons_grouping_blocked_ratio": _safe_ratio(
                records_all_horizons_grouping_blocked,
                total_records,
            ),
            "records_partial_grouping_blocked": records_partial_grouping_blocked,
            "records_partial_grouping_blocked_ratio": _safe_ratio(
                records_partial_grouping_blocked,
                total_records,
            ),
            "records_no_grouping_block": records_no_grouping_block,
            "records_no_grouping_block_ratio": _safe_ratio(
                records_no_grouping_block,
                total_records,
            ),
        },
        "diagnosis": diagnosis,
        "examples": examples,
    }


def render_grouping_failure_deep_diagnosis_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    coverage = _safe_dict(summary.get("coverage"))
    reason_distribution = _safe_dict(summary.get("reason_distribution"))
    per_horizon = _safe_dict(summary.get("per_horizon"))
    severity = _safe_dict(summary.get("record_level_severity"))
    diagnosis = _safe_dict(summary.get("diagnosis"))

    lines = [
        "# Grouping Failure Deep Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Total records: {metadata.get('total_records', 0)}",
        f"- Total horizon rows: {metadata.get('total_horizon_rows', 0)}",
        f"- Grouping blocked ratio: {_fmt(coverage.get('grouping_blocked_ratio'))}",
        f"- Grouping free ratio: {_fmt(coverage.get('grouping_free_ratio'))}",
        f"- Records all-horizons blocked ratio: {_fmt(severity.get('records_all_horizons_grouping_blocked_ratio'))}",
        f"- Primary diagnosis: {diagnosis.get('primary_issue', 'unknown')}",
        f"- Secondary diagnosis: {diagnosis.get('secondary_issue', 'unknown')}",
        "",
        "## Overall Grouping Reason Distribution",
        _md_dist(reason_distribution.get("overall_grouping_reason_distribution")),
        "",
        "## Overall Grouping Combo Distribution",
        _md_dist(reason_distribution.get("overall_grouping_combo_distribution")),
        "",
        "## Per-Horizon Blocked Rate",
        _md_dist(per_horizon.get("blocked_rate_by_horizon")),
        "",
        "## Per-Horizon Reason Distribution",
        _md_nested_dist(per_horizon.get("reason_distribution")),
        "",
        "## Per-Horizon Combo Distribution",
        _md_nested_dist(per_horizon.get("combo_distribution")),
        "",
        "## Record-Level Severity",
        _md_key_values(
            [
                ("records_with_any_grouping_block", severity.get("records_with_any_grouping_block")),
                (
                    "records_all_horizons_grouping_blocked",
                    severity.get("records_all_horizons_grouping_blocked"),
                ),
                (
                    "records_partial_grouping_blocked",
                    severity.get("records_partial_grouping_blocked"),
                ),
                ("records_no_grouping_block", severity.get("records_no_grouping_block")),
            ]
        ),
        "",
        "## Final Diagnosis",
        f"- {diagnosis.get('summary', 'n/a')}",
    ]
    return "\n".join(lines)


def _extract_horizon_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    seed_diag = record.get("candidate_seed_diagnostics")
    if not isinstance(seed_diag, dict):
        return []

    horizon_diagnostics = seed_diag.get("horizon_diagnostics")
    if not isinstance(horizon_diagnostics, list):
        return []

    result: list[dict[str, Any]] = []

    for item in horizon_diagnostics:
        if not isinstance(item, dict):
            continue

        horizon = _text(item.get("horizon"))
        if horizon not in VALID_HORIZONS:
            continue

        blocker_reasons = _reason_list(item.get("blocker_reasons"))
        grouping_reasons = [reason for reason in blocker_reasons if reason in GROUPING_REASONS]

        result.append(
            {
                "horizon": horizon,
                "grouping_reasons": grouping_reasons,
                "all_blocker_reasons": blocker_reasons,
            }
        )

    return result


def _grouping_category(grouping_reasons: list[str]) -> str:
    reason_set = set(grouping_reasons)

    if reason_set == {"no_valid_symbol_group"}:
        return "symbol_only"
    if reason_set == {"no_valid_strategy_group"}:
        return "strategy_only"
    if reason_set == {"no_valid_symbol_or_strategy_group"}:
        return "symbol_or_strategy_combined"
    return "mixed"


def _combo_key(reasons: list[str]) -> str:
    if not reasons:
        return "none"
    return " + ".join(sorted(set(reasons)))


def _build_diagnosis(
    *,
    total_records: int,
    total_horizon_rows: int,
    grouping_blocked_horizon_rows: int,
    grouping_free_horizon_rows: int,
    symbol_only_count: int,
    strategy_only_count: int,
    symbol_or_strategy_combined_count: int,
    mixed_grouping_block_count: int,
    records_with_any_grouping_block: int,
    records_all_horizons_grouping_blocked: int,
    records_partial_grouping_blocked: int,
    per_horizon_blocked: Counter[str],
    per_horizon_total: Counter[str],
) -> dict[str, Any]:
    if total_horizon_rows == 0:
        return {
            "primary_issue": "no_data",
            "secondary_issue": "no_data",
            "summary": "No horizon-level grouping rows were found.",
            "dominant_blocked_horizons": [],
        }

    grouping_blocked_ratio = _safe_ratio(grouping_blocked_horizon_rows, total_horizon_rows)
    grouping_free_ratio = _safe_ratio(grouping_free_horizon_rows, total_horizon_rows)
    all_horizons_blocked_record_ratio = _safe_ratio(
        records_all_horizons_grouping_blocked,
        total_records,
    )
    partial_block_record_ratio = _safe_ratio(
        records_partial_grouping_blocked,
        total_records,
    )

    dominant_reason = max(
        {
            "symbol_only": symbol_only_count,
            "strategy_only": strategy_only_count,
            "symbol_or_strategy_combined": symbol_or_strategy_combined_count,
            "mixed": mixed_grouping_block_count,
        }.items(),
        key=lambda item: (item[1], item[0]),
        default=("unknown", 0),
    )[0]

    blocked_rates = {
        horizon: _safe_ratio(per_horizon_blocked.get(horizon, 0), per_horizon_total.get(horizon, 0))
        for horizon in VALID_HORIZONS
    }
    max_rate = max(blocked_rates.values(), default=0.0)
    dominant_blocked_horizons = [
        horizon for horizon, rate in blocked_rates.items() if rate == max_rate and rate > 0
    ]

    primary_issue = "mixed_grouping_failure"
    secondary_issue = "collect_blocker_distribution_details"

    if grouping_free_horizon_rows == 0:
        primary_issue = "grouping_blockade_is_total_and_prevents_all_grouping_free_inputs"
        secondary_issue = "strength_cannot_be_tested_independently_until_grouping_free_rows_exist"
    elif all_horizons_blocked_record_ratio >= 0.8:
        primary_issue = "most_records_are_grouping_blocked_across_all_horizons"
        secondary_issue = "grouping_failure_is_systemic_not_partial"
    elif dominant_reason == "symbol_or_strategy_combined":
        primary_issue = "combined_symbol_or_strategy_group_validation_is_dominant_blocker"
        secondary_issue = "grouping_logic_fails_before_symbol_and_strategy_can_be_separated"
    elif dominant_reason == "symbol_only":
        primary_issue = "symbol_group_validation_is_dominant_blocker"
        secondary_issue = "symbol_grouping_granularity_or_density_is_primary_suspect"
    elif dominant_reason == "strategy_only":
        primary_issue = "strategy_group_validation_is_dominant_blocker"
        secondary_issue = "strategy_grouping_granularity_or_density_is_primary_suspect"
    elif partial_block_record_ratio >= 0.5:
        primary_issue = "grouping_failure_is_partial_but_widespread"
        secondary_issue = "horizon_specific_or_combo_specific_grouping_rules_need_review"

    return {
        "primary_issue": primary_issue,
        "secondary_issue": secondary_issue,
        "grouping_blocked_ratio": grouping_blocked_ratio,
        "grouping_free_ratio": grouping_free_ratio,
        "all_horizons_blocked_record_ratio": all_horizons_blocked_record_ratio,
        "partial_block_record_ratio": partial_block_record_ratio,
        "dominant_reason": dominant_reason,
        "dominant_blocked_horizons": dominant_blocked_horizons,
        "blocked_rate_by_horizon": blocked_rates,
        "summary": (
            f"Primary issue={primary_issue}; secondary issue={secondary_issue}; "
            f"grouping_blocked_ratio={grouping_blocked_ratio:.2%}; "
            f"grouping_free_ratio={grouping_free_ratio:.2%}; "
            f"all_horizons_blocked_record_ratio={all_horizons_blocked_record_ratio:.2%}; "
            f"dominant_reason={dominant_reason}; "
            f"dominant_blocked_horizons={dominant_blocked_horizons}."
        ),
    }


def _build_example(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": _text(record.get("generated_at")) or "n/a",
        "cumulative_record_count": _int(record.get("cumulative_record_count")),
        "selection_status": _text(record.get("selection_status")) or "unknown",
        "horizon": row["horizon"],
        "grouping_reasons": row["grouping_reasons"],
        "all_blocker_reasons": row["all_blocker_reasons"],
    }


def _build_record_example(
    record: dict[str, Any],
    *,
    blocked_horizons: list[str],
    free_horizons: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": _text(record.get("generated_at")) or "n/a",
        "cumulative_record_count": _int(record.get("cumulative_record_count")),
        "selection_status": _text(record.get("selection_status")) or "unknown",
        "blocked_horizons": blocked_horizons,
        "free_horizons": free_horizons,
    }


def _push_example(target: list[dict[str, Any]], example: dict[str, Any]) -> None:
    target.append(example)
    target.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
    del target[MAX_EXAMPLES_PER_CATEGORY:]


def _reason_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _text(item)
        if text:
            result.append(text)
    return result


def _counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "value": value,
            "count": count,
            "ratio": _safe_ratio(count, total),
        }
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def _rate_rows(
    *,
    numerator: Counter[str],
    denominator: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in VALID_HORIZONS:
        denom = denominator.get(horizon, 0)
        num = numerator.get(horizon, 0)
        rows.append(
            {
                "value": horizon,
                "count": num,
                "denominator": denom,
                "ratio": _safe_ratio(num, denom),
            }
        )
    return rows


def _md_dist(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "- none"

    lines: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            count = row.get("count", 0)
            ratio = _fmt(row.get("ratio"))
            denominator = row.get("denominator")
            if denominator is None:
                lines.append(f"- {row.get('value', 'n/a')}: {count} ({ratio})")
            else:
                lines.append(
                    f"- {row.get('value', 'n/a')}: {count}/{denominator} ({ratio})"
                )
    return "\n".join(lines)


def _md_nested_dist(value: Any) -> str:
    nested = value if isinstance(value, dict) else {}
    if not nested:
        return "- none"

    lines: list[str] = []
    for label, rows in sorted(nested.items()):
        lines.append(f"- {label}:")
        if not isinstance(rows, list) or not rows:
            lines.append("  - none")
            continue
        for row in rows:
            if isinstance(row, dict):
                lines.append(
                    f"  - {row.get('value', 'n/a')}: {row.get('count', 0)} ({_fmt(row.get('ratio'))})"
                )
    return "\n".join(lines)


def _md_key_values(rows: list[tuple[str, Any]]) -> str:
    if not rows:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in rows)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator > 0 else 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def _default_output_paths(input_path: Path) -> tuple[Path, Path]:
    output_dir = input_path.parent
    json_path = output_dir / "grouping_failure_deep_diagnosis_summary.json"
    markdown_path = output_dir / "grouping_failure_deep_diagnosis_summary.md"
    return json_path, markdown_path


def main() -> None:
    result = run_grouping_failure_deep_diagnosis_report()
    summary = result["summary"]
    markdown = result["markdown"]

    input_path = Path(result["input_path"])
    json_path, markdown_path = _default_output_paths(input_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "json_summary_path": str(json_path),
                "markdown_summary_path": str(markdown_path),
                "total_records": summary.get("metadata", {}).get("total_records", 0),
                "total_horizon_rows": summary.get("metadata", {}).get("total_horizon_rows", 0),
                "primary_issue": summary.get("diagnosis", {}).get("primary_issue", "unknown"),
                "secondary_issue": summary.get("diagnosis", {}).get("secondary_issue", "unknown"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()