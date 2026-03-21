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

PRIMARY_STRENGTH_REASON = "candidate_strength_insufficient_data"

STRENGTH_ORDER = {
    "insufficient_data": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


def run_strength_blocker_cross_tab_report(
    input_path: Path | None = None,
) -> dict[str, Any]:
    resolved = input_path or DEFAULT_INPUT_PATH
    loaded = load_shadow_records(resolved)
    summary = build_strength_blocker_cross_tab_summary(
        records=loaded["records"],
        input_path=resolved,
        data_quality=loaded["data_quality"],
    )
    return {
        "input_path": str(resolved),
        "summary": summary,
        "markdown": render_strength_blocker_cross_tab_markdown(summary),
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


def build_strength_blocker_cross_tab_summary(
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    total_horizon_rows = 0

    grouping_blocked_total = 0
    grouping_free_total = 0

    strength_insufficient_total = 0
    grouping_free_strength_insufficient_total = 0
    grouping_blocked_strength_insufficient_total = 0

    ready_total = 0
    grouping_free_ready_total = 0

    blocker_combo_dist: Counter[str] = Counter()
    per_horizon_blocker_combo_dist: dict[str, Counter[str]] = defaultdict(Counter)

    per_horizon_total: Counter[str] = Counter()
    per_horizon_grouping_blocked: Counter[str] = Counter()
    per_horizon_grouping_free: Counter[str] = Counter()
    per_horizon_strength_insufficient: Counter[str] = Counter()
    per_horizon_ready: Counter[str] = Counter()
    per_horizon_grouping_free_strength_insufficient: Counter[str] = Counter()
    per_horizon_grouping_free_ready: Counter[str] = Counter()

    examples: dict[str, list[dict[str, Any]]] = {
        "grouping_blocked_strength_failed_examples": [],
        "grouping_free_strength_failed_examples": [],
        "grouping_free_strength_ready_examples": [],
        "strength_reason_without_grouping_examples": [],
    }

    for record in records:
        for row in _extract_horizon_rows(record):
            total_horizon_rows += 1
            horizon = row["horizon"]
            latest_strength = row["latest_strength"]
            cumulative_strength = row["cumulative_strength"]
            blocker_reasons = row["blocker_reasons"]

            has_grouping_blocker = any(reason in GROUPING_REASONS for reason in blocker_reasons)
            has_strength_reason = PRIMARY_STRENGTH_REASON in blocker_reasons
            is_strength_insufficient = (
                latest_strength == "insufficient_data"
                and cumulative_strength == "insufficient_data"
            )
            is_ready = (
                _strength_rank(latest_strength) >= 1
                or _strength_rank(cumulative_strength) >= 1
            )

            combo_key = _combo_key(blocker_reasons)
            blocker_combo_dist[combo_key] += 1
            per_horizon_blocker_combo_dist[horizon][combo_key] += 1

            per_horizon_total[horizon] += 1

            if has_grouping_blocker:
                grouping_blocked_total += 1
                per_horizon_grouping_blocked[horizon] += 1
            else:
                grouping_free_total += 1
                per_horizon_grouping_free[horizon] += 1

            if is_strength_insufficient:
                strength_insufficient_total += 1
                per_horizon_strength_insufficient[horizon] += 1

            if is_ready:
                ready_total += 1
                per_horizon_ready[horizon] += 1

            if not has_grouping_blocker and is_strength_insufficient:
                grouping_free_strength_insufficient_total += 1
                per_horizon_grouping_free_strength_insufficient[horizon] += 1
                _push_example(
                    examples["grouping_free_strength_failed_examples"],
                    _build_example(record, row),
                )

            if has_grouping_blocker and is_strength_insufficient:
                grouping_blocked_strength_insufficient_total += 1
                _push_example(
                    examples["grouping_blocked_strength_failed_examples"],
                    _build_example(record, row),
                )

            if not has_grouping_blocker and is_ready:
                grouping_free_ready_total += 1
                per_horizon_grouping_free_ready[horizon] += 1
                _push_example(
                    examples["grouping_free_strength_ready_examples"],
                    _build_example(record, row),
                )

            if not has_grouping_blocker and has_strength_reason:
                _push_example(
                    examples["strength_reason_without_grouping_examples"],
                    _build_example(record, row),
                )

    diagnosis = _build_diagnosis(
        total_horizon_rows=total_horizon_rows,
        grouping_blocked_total=grouping_blocked_total,
        grouping_free_total=grouping_free_total,
        strength_insufficient_total=strength_insufficient_total,
        grouping_free_strength_insufficient_total=grouping_free_strength_insufficient_total,
        grouping_blocked_strength_insufficient_total=grouping_blocked_strength_insufficient_total,
        ready_total=ready_total,
        grouping_free_ready_total=grouping_free_ready_total,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "strength_blocker_cross_tab_report",
            "input_path": str(input_path),
            "total_records": len(records),
            "total_horizon_rows": total_horizon_rows,
            "data_quality": data_quality,
        },
        "coverage": {
            "grouping_blocked_observation_total": grouping_blocked_total,
            "grouping_free_observation_total": grouping_free_total,
            "grouping_blocked_ratio": _safe_ratio(grouping_blocked_total, total_horizon_rows),
            "grouping_free_ratio": _safe_ratio(grouping_free_total, total_horizon_rows),
        },
        "strength_status": {
            "strength_insufficient_total": strength_insufficient_total,
            "strength_insufficient_ratio": _safe_ratio(
                strength_insufficient_total,
                total_horizon_rows,
            ),
            "ready_total": ready_total,
            "ready_ratio": _safe_ratio(ready_total, total_horizon_rows),
            "grouping_free_strength_insufficient_total": grouping_free_strength_insufficient_total,
            "grouping_free_strength_insufficient_ratio": _safe_ratio(
                grouping_free_strength_insufficient_total,
                grouping_free_total,
            ),
            "grouping_blocked_strength_insufficient_total": grouping_blocked_strength_insufficient_total,
            "grouping_blocked_strength_insufficient_ratio": _safe_ratio(
                grouping_blocked_strength_insufficient_total,
                grouping_blocked_total,
            ),
            "grouping_free_ready_total": grouping_free_ready_total,
            "grouping_free_ready_ratio": _safe_ratio(
                grouping_free_ready_total,
                grouping_free_total,
            ),
        },
        "blocker_combinations": {
            "overall_blocker_combo_distribution": _counter_rows(blocker_combo_dist),
            "per_horizon_blocker_combo_distribution": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_blocker_combo_dist.items())
            },
        },
        "per_horizon": {
            "total": _counter_rows(per_horizon_total),
            "grouping_blocked": _counter_rows(per_horizon_grouping_blocked),
            "grouping_free": _counter_rows(per_horizon_grouping_free),
            "strength_insufficient": _counter_rows(per_horizon_strength_insufficient),
            "ready": _counter_rows(per_horizon_ready),
            "grouping_free_strength_insufficient": _counter_rows(
                per_horizon_grouping_free_strength_insufficient
            ),
            "grouping_free_ready": _counter_rows(per_horizon_grouping_free_ready),
            "grouping_free_strength_insufficient_rate_by_horizon": _rate_rows(
                numerator=per_horizon_grouping_free_strength_insufficient,
                denominator=per_horizon_grouping_free,
            ),
            "grouping_free_ready_rate_by_horizon": _rate_rows(
                numerator=per_horizon_grouping_free_ready,
                denominator=per_horizon_grouping_free,
            ),
        },
        "diagnosis": diagnosis,
        "examples": examples,
    }


def render_strength_blocker_cross_tab_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    coverage = _safe_dict(summary.get("coverage"))
    strength_status = _safe_dict(summary.get("strength_status"))
    blocker_combinations = _safe_dict(summary.get("blocker_combinations"))
    per_horizon = _safe_dict(summary.get("per_horizon"))
    diagnosis = _safe_dict(summary.get("diagnosis"))

    lines = [
        "# Strength Blocker Cross-Tab Report",
        "",
        "## Executive Summary",
        f"- Total records: {metadata.get('total_records', 0)}",
        f"- Total horizon rows: {metadata.get('total_horizon_rows', 0)}",
        f"- Grouping blocked ratio: {_fmt(coverage.get('grouping_blocked_ratio'))}",
        f"- Grouping free ratio: {_fmt(coverage.get('grouping_free_ratio'))}",
        f"- Strength insufficient ratio: {_fmt(strength_status.get('strength_insufficient_ratio'))}",
        f"- Grouping-free strength insufficient ratio: {_fmt(strength_status.get('grouping_free_strength_insufficient_ratio'))}",
        f"- Grouping-free ready ratio: {_fmt(strength_status.get('grouping_free_ready_ratio'))}",
        f"- Primary diagnosis: {diagnosis.get('primary_issue', 'unknown')}",
        f"- Secondary diagnosis: {diagnosis.get('secondary_issue', 'unknown')}",
        "",
        "## Coverage",
        _md_key_values(
            [
                ("grouping_blocked_observation_total", coverage.get("grouping_blocked_observation_total")),
                ("grouping_free_observation_total", coverage.get("grouping_free_observation_total")),
            ]
        ),
        "",
        "## Strength Status",
        _md_key_values(
            [
                ("strength_insufficient_total", strength_status.get("strength_insufficient_total")),
                ("ready_total", strength_status.get("ready_total")),
                (
                    "grouping_free_strength_insufficient_total",
                    strength_status.get("grouping_free_strength_insufficient_total"),
                ),
                ("grouping_free_ready_total", strength_status.get("grouping_free_ready_total")),
            ]
        ),
        "",
        "## Overall Blocker Combination Distribution",
        _md_dist(blocker_combinations.get("overall_blocker_combo_distribution")),
        "",
        "## Per-Horizon Blocker Combination Distribution",
        _md_nested_dist(blocker_combinations.get("per_horizon_blocker_combo_distribution")),
        "",
        "## Per-Horizon Grouping-Free Strength Insufficient Rate",
        _md_dist(per_horizon.get("grouping_free_strength_insufficient_rate_by_horizon")),
        "",
        "## Per-Horizon Grouping-Free Ready Rate",
        _md_dist(per_horizon.get("grouping_free_ready_rate_by_horizon")),
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

        result.append(
            {
                "horizon": horizon,
                "latest_strength": _normalize_strength(item.get("latest_candidate_strength")),
                "cumulative_strength": _normalize_strength(item.get("cumulative_candidate_strength")),
                "blocker_reasons": _reason_list(item.get("blocker_reasons")),
            }
        )

    return result


def _combo_key(blocker_reasons: list[str]) -> str:
    if not blocker_reasons:
        return "none"

    unique_reasons = sorted(set(blocker_reasons))
    return " + ".join(unique_reasons)


def _build_diagnosis(
    *,
    total_horizon_rows: int,
    grouping_blocked_total: int,
    grouping_free_total: int,
    strength_insufficient_total: int,
    grouping_free_strength_insufficient_total: int,
    grouping_blocked_strength_insufficient_total: int,
    ready_total: int,
    grouping_free_ready_total: int,
) -> dict[str, Any]:
    if total_horizon_rows == 0:
        return {
            "primary_issue": "no_data",
            "secondary_issue": "no_data",
            "summary": "No horizon-level rows were found.",
        }

    grouping_blocked_ratio = _safe_ratio(grouping_blocked_total, total_horizon_rows)
    grouping_free_ratio = _safe_ratio(grouping_free_total, total_horizon_rows)
    strength_insufficient_ratio = _safe_ratio(strength_insufficient_total, total_horizon_rows)
    grouping_free_strength_insufficient_ratio = _safe_ratio(
        grouping_free_strength_insufficient_total,
        grouping_free_total,
    )
    grouping_blocked_strength_insufficient_ratio = _safe_ratio(
        grouping_blocked_strength_insufficient_total,
        grouping_blocked_total,
    )
    grouping_free_ready_ratio = _safe_ratio(grouping_free_ready_total, grouping_free_total)
    ready_ratio = _safe_ratio(ready_total, total_horizon_rows)

    primary_issue = "mixed_or_unclear"
    secondary_issue = "collect_more_probe_data"

    if grouping_free_total == 0:
        primary_issue = "grouping_blockers_prevent_any_grouping_free_strength_observations"
        secondary_issue = "cannot_test_strength_logic_independently_until_grouping_free_rows_exist"
    elif grouping_free_strength_insufficient_ratio >= 0.9 and grouping_free_ready_ratio <= 0.1:
        primary_issue = "strength_conditions_fail_even_when_grouping_is_not_blocking"
        secondary_issue = "strength_logic_is_more_likely_than_grouping_to_be_primary_bottleneck"
    elif grouping_blocked_ratio >= 0.8 and grouping_free_ratio <= 0.2:
        primary_issue = "grouping_blockers_dominate_strength_pipeline_inputs"
        secondary_issue = "grouping_logic_likely_prevents_strength_from_forming"
    elif (
        grouping_blocked_strength_insufficient_ratio >= 0.9
        and grouping_free_strength_insufficient_ratio <= 0.25
        and grouping_free_ready_ratio >= 0.5
    ):
        primary_issue = "grouping_is_primary_driver_of_strength_failure"
        secondary_issue = "strength_can_form_when_grouping_blockers_are_absent"
    elif strength_insufficient_ratio >= 0.9 and ready_ratio <= 0.1:
        primary_issue = "strength_pipeline_is_globally_starved"
        secondary_issue = "need_upstream_density_or_condition_probe"

    return {
        "primary_issue": primary_issue,
        "secondary_issue": secondary_issue,
        "grouping_blocked_ratio": grouping_blocked_ratio,
        "grouping_free_ratio": grouping_free_ratio,
        "strength_insufficient_ratio": strength_insufficient_ratio,
        "grouping_free_strength_insufficient_ratio": grouping_free_strength_insufficient_ratio,
        "grouping_blocked_strength_insufficient_ratio": grouping_blocked_strength_insufficient_ratio,
        "grouping_free_ready_ratio": grouping_free_ready_ratio,
        "ready_ratio": ready_ratio,
        "summary": (
            f"Primary issue={primary_issue}; secondary issue={secondary_issue}; "
            f"grouping_blocked_ratio={grouping_blocked_ratio:.2%}; "
            f"grouping_free_ratio={grouping_free_ratio:.2%}; "
            f"strength_insufficient_ratio={strength_insufficient_ratio:.2%}; "
            f"grouping_free_strength_insufficient_ratio={grouping_free_strength_insufficient_ratio:.2%}; "
            f"grouping_free_ready_ratio={grouping_free_ready_ratio:.2%}."
        ),
    }


def _build_example(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": _text(record.get("generated_at")) or "n/a",
        "cumulative_record_count": _int(record.get("cumulative_record_count")),
        "selection_status": _text(record.get("selection_status")) or "unknown",
        "horizon": row["horizon"],
        "latest_strength": row["latest_strength"],
        "cumulative_strength": row["cumulative_strength"],
        "blocker_reasons": row["blocker_reasons"],
    }


def _push_example(target: list[dict[str, Any]], example: dict[str, Any]) -> None:
    target.append(example)
    target.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
    del target[MAX_EXAMPLES_PER_CATEGORY:]


def _normalize_strength(value: Any) -> str:
    text = _text(value)
    if text in STRENGTH_ORDER:
        return text
    return "insufficient_data"


def _strength_rank(value: str) -> int:
    return STRENGTH_ORDER.get(value, 0)


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
    json_path = output_dir / "strength_blocker_cross_tab_summary.json"
    markdown_path = output_dir / "strength_blocker_cross_tab_summary.md"
    return json_path, markdown_path


def main() -> None:
    result = run_strength_blocker_cross_tab_report()
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