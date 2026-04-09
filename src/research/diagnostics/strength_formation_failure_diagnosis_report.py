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
STRENGTH_ORDER = {
    "insufficient_data": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
}


def run_strength_formation_failure_diagnosis_report(
    input_path: Path | None = None,
) -> dict[str, Any]:
    resolved = input_path or DEFAULT_INPUT_PATH
    loaded = load_shadow_records(resolved)
    summary = build_strength_formation_failure_diagnosis_summary(
        records=loaded["records"],
        input_path=resolved,
        data_quality=loaded["data_quality"],
    )
    return {
        "input_path": str(resolved),
        "summary": summary,
        "markdown": render_strength_formation_failure_diagnosis_markdown(summary),
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


def build_strength_formation_failure_diagnosis_summary(
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    latest_strength_dist: Counter[str] = Counter()
    cumulative_strength_dist: Counter[str] = Counter()

    per_horizon_latest: dict[str, Counter[str]] = defaultdict(Counter)
    per_horizon_cumulative: dict[str, Counter[str]] = defaultdict(Counter)

    latest_vs_cumulative_pairs: Counter[str] = Counter()
    per_horizon_pair_dist: dict[str, Counter[str]] = defaultdict(Counter)

    horizon_observation_count: Counter[str] = Counter()
    horizon_grouping_free_observation_count: Counter[str] = Counter()

    grouping_free_latest_strength_dist: Counter[str] = Counter()
    grouping_free_cumulative_strength_dist: Counter[str] = Counter()

    grouping_blocked_count = 0
    grouping_free_count = 0
    all_insufficient_pair_count = 0
    any_promotable_count = 0
    mismatch_count = 0

    examples: dict[str, list[dict[str, Any]]] = {
        "all_insufficient_examples": [],
        "latest_weak_cumulative_insufficient_examples": [],
        "latest_insufficient_cumulative_weak_examples": [],
        "mismatch_examples": [],
        "grouping_free_but_strength_failed_examples": [],
    }

    for record in records:
        for horizon_row in _extract_horizon_rows(record):
            horizon = horizon_row["horizon"]
            latest_strength = horizon_row["latest_strength"]
            cumulative_strength = horizon_row["cumulative_strength"]
            blocker_reasons = horizon_row["blocker_reasons"]
            has_grouping_failure = any(reason in GROUPING_REASONS for reason in blocker_reasons)

            horizon_observation_count[horizon] += 1
            latest_strength_dist[latest_strength] += 1
            cumulative_strength_dist[cumulative_strength] += 1
            per_horizon_latest[horizon][latest_strength] += 1
            per_horizon_cumulative[horizon][cumulative_strength] += 1

            pair_key = f"{latest_strength} -> {cumulative_strength}"
            latest_vs_cumulative_pairs[pair_key] += 1
            per_horizon_pair_dist[horizon][pair_key] += 1

            if has_grouping_failure:
                grouping_blocked_count += 1
            else:
                grouping_free_count += 1
                horizon_grouping_free_observation_count[horizon] += 1
                grouping_free_latest_strength_dist[latest_strength] += 1
                grouping_free_cumulative_strength_dist[cumulative_strength] += 1

            if latest_strength == "insufficient_data" and cumulative_strength == "insufficient_data":
                all_insufficient_pair_count += 1
                _push_example(
                    examples["all_insufficient_examples"],
                    _build_horizon_example(record, horizon_row),
                )

            if latest_strength == "weak" and cumulative_strength == "insufficient_data":
                _push_example(
                    examples["latest_weak_cumulative_insufficient_examples"],
                    _build_horizon_example(record, horizon_row),
                )

            if latest_strength == "insufficient_data" and cumulative_strength == "weak":
                _push_example(
                    examples["latest_insufficient_cumulative_weak_examples"],
                    _build_horizon_example(record, horizon_row),
                )

            if latest_strength != cumulative_strength:
                mismatch_count += 1
                _push_example(
                    examples["mismatch_examples"],
                    _build_horizon_example(record, horizon_row),
                )

            if _strength_rank(latest_strength) >= 1 or _strength_rank(cumulative_strength) >= 1:
                any_promotable_count += 1

            if (
                not has_grouping_failure
                and latest_strength == "insufficient_data"
                and cumulative_strength == "insufficient_data"
            ):
                _push_example(
                    examples["grouping_free_but_strength_failed_examples"],
                    _build_horizon_example(record, horizon_row),
                )

    total_observations = sum(horizon_observation_count.values())

    diagnosis = _build_strength_diagnosis(
        total_observations=total_observations,
        all_insufficient_pair_count=all_insufficient_pair_count,
        mismatch_count=mismatch_count,
        grouping_free_count=grouping_free_count,
        grouping_free_latest_strength_dist=grouping_free_latest_strength_dist,
        grouping_free_cumulative_strength_dist=grouping_free_cumulative_strength_dist,
        horizon_observation_count=horizon_observation_count,
        per_horizon_latest=per_horizon_latest,
        per_horizon_cumulative=per_horizon_cumulative,
        any_promotable_count=any_promotable_count,
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "strength_formation_failure_diagnosis_report",
            "input_path": str(input_path),
            "total_records": len(records),
            "total_horizon_observations": total_observations,
            "data_quality": data_quality,
        },
        "strength_overview": {
            "latest_candidate_strength_distribution": _counter_rows(latest_strength_dist),
            "cumulative_candidate_strength_distribution": _counter_rows(cumulative_strength_dist),
            "latest_insufficient_data_ratio": _safe_ratio(
                latest_strength_dist.get("insufficient_data", 0),
                total_observations,
            ),
            "cumulative_insufficient_data_ratio": _safe_ratio(
                cumulative_strength_dist.get("insufficient_data", 0),
                total_observations,
            ),
            "promotable_strength_ratio": _safe_ratio(any_promotable_count, total_observations),
        },
        "per_horizon_strength": {
            "latest": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_latest.items())
            },
            "cumulative": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_cumulative.items())
            },
            "observation_count": _counter_rows(horizon_observation_count),
        },
        "transition_analysis": {
            "latest_vs_cumulative_pair_distribution": _counter_rows(latest_vs_cumulative_pairs),
            "mismatch_rate": _safe_ratio(mismatch_count, total_observations),
            "all_insufficient_pair_ratio": _safe_ratio(
                all_insufficient_pair_count,
                total_observations,
            ),
            "per_horizon_pair_distribution": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(per_horizon_pair_dist.items())
            },
        },
        "grouping_independent_strength": {
            "grouping_blocked_ratio": _safe_ratio(grouping_blocked_count, total_observations),
            "grouping_free_ratio": _safe_ratio(grouping_free_count, total_observations),
            "grouping_free_latest_strength_distribution": _counter_rows(
                grouping_free_latest_strength_dist
            ),
            "grouping_free_cumulative_strength_distribution": _counter_rows(
                grouping_free_cumulative_strength_dist
            ),
            "grouping_free_observation_count_by_horizon": _counter_rows(
                horizon_grouping_free_observation_count
            ),
        },
        "diagnosis": diagnosis,
        "examples": examples,
    }


def render_strength_formation_failure_diagnosis_markdown(summary: dict[str, Any]) -> str:
    metadata = _safe_dict(summary.get("metadata"))
    overview = _safe_dict(summary.get("strength_overview"))
    per_horizon = _safe_dict(summary.get("per_horizon_strength"))
    transition = _safe_dict(summary.get("transition_analysis"))
    grouping_independent = _safe_dict(summary.get("grouping_independent_strength"))
    diagnosis = _safe_dict(summary.get("diagnosis"))

    lines = [
        "# Strength Formation Failure Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Total records: {metadata.get('total_records', 0)}",
        f"- Total horizon observations: {metadata.get('total_horizon_observations', 0)}",
        f"- Latest insufficient_data ratio: {_fmt(overview.get('latest_insufficient_data_ratio'))}",
        f"- Cumulative insufficient_data ratio: {_fmt(overview.get('cumulative_insufficient_data_ratio'))}",
        f"- Promotable strength ratio: {_fmt(overview.get('promotable_strength_ratio'))}",
        f"- Primary diagnosis: {diagnosis.get('primary_issue', 'unknown')}",
        f"- Secondary diagnosis: {diagnosis.get('secondary_issue', 'unknown')}",
        "",
        "## Strength Overview",
        _md_dist("Latest strength distribution", overview.get("latest_candidate_strength_distribution")),
        _md_dist("Cumulative strength distribution", overview.get("cumulative_candidate_strength_distribution")),
        "",
        "## Per-Horizon Latest Strength",
        _md_nested_dist(per_horizon.get("latest")),
        "",
        "## Per-Horizon Cumulative Strength",
        _md_nested_dist(per_horizon.get("cumulative")),
        "",
        "## Transition Analysis",
        _md_dist("Latest -> cumulative pair distribution", transition.get("latest_vs_cumulative_pair_distribution")),
        f"- Mismatch rate: {_fmt(transition.get('mismatch_rate'))}",
        f"- All insufficient pair ratio: {_fmt(transition.get('all_insufficient_pair_ratio'))}",
        "",
        "## Grouping-Independent Strength",
        f"- Grouping blocked ratio: {_fmt(grouping_independent.get('grouping_blocked_ratio'))}",
        f"- Grouping free ratio: {_fmt(grouping_independent.get('grouping_free_ratio'))}",
        _md_dist(
            "Grouping-free latest strength distribution",
            grouping_independent.get("grouping_free_latest_strength_distribution"),
        ),
        _md_dist(
            "Grouping-free cumulative strength distribution",
            grouping_independent.get("grouping_free_cumulative_strength_distribution"),
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

        latest_strength = _normalize_strength(item.get("latest_candidate_strength"))
        cumulative_strength = _normalize_strength(item.get("cumulative_candidate_strength"))
        blocker_reasons = _reason_list(item.get("blocker_reasons"))

        result.append(
            {
                "horizon": horizon,
                "latest_strength": latest_strength,
                "cumulative_strength": cumulative_strength,
                "blocker_reasons": blocker_reasons,
            }
        )

    return result


def _build_strength_diagnosis(
    *,
    total_observations: int,
    all_insufficient_pair_count: int,
    mismatch_count: int,
    grouping_free_count: int,
    grouping_free_latest_strength_dist: Counter[str],
    grouping_free_cumulative_strength_dist: Counter[str],
    horizon_observation_count: Counter[str],
    per_horizon_latest: dict[str, Counter[str]],
    per_horizon_cumulative: dict[str, Counter[str]],
    any_promotable_count: int,
) -> dict[str, Any]:
    if total_observations == 0:
        return {
            "primary_issue": "no_data",
            "secondary_issue": "no_data",
            "summary": "No horizon-level strength observations were found.",
        }

    latest_insufficient_ratio = _safe_ratio(
        sum(
            counter.get("insufficient_data", 0)
            for counter in per_horizon_latest.values()
        ),
        total_observations,
    )
    cumulative_insufficient_ratio = _safe_ratio(
        sum(
            counter.get("insufficient_data", 0)
            for counter in per_horizon_cumulative.values()
        ),
        total_observations,
    )
    all_insufficient_pair_ratio = _safe_ratio(all_insufficient_pair_count, total_observations)
    mismatch_rate = _safe_ratio(mismatch_count, total_observations)
    promotable_ratio = _safe_ratio(any_promotable_count, total_observations)

    grouping_free_latest_insufficient_ratio = _safe_ratio(
        grouping_free_latest_strength_dist.get("insufficient_data", 0),
        grouping_free_count,
    )
    grouping_free_cumulative_insufficient_ratio = _safe_ratio(
        grouping_free_cumulative_strength_dist.get("insufficient_data", 0),
        grouping_free_count,
    )

    horizon_insufficient_latest = {
        horizon: _safe_ratio(counter.get("insufficient_data", 0), horizon_observation_count.get(horizon, 0))
        for horizon, counter in per_horizon_latest.items()
    }
    weakest_horizon = max(
        horizon_insufficient_latest.items(),
        key=lambda item: (item[1], item[0]),
        default=("unknown", 0.0),
    )[0]

    primary_issue = "strength_conditions_misaligned_with_data_density"
    secondary_issue = "visibility_collapse_downstream_of_strength_failure"

    if grouping_free_count > 0 and (
        grouping_free_latest_insufficient_ratio >= 0.9
        and grouping_free_cumulative_insufficient_ratio >= 0.9
    ):
        primary_issue = "strength_failure_persists_even_without_grouping_blockers"
    elif cumulative_insufficient_ratio >= 0.9 and latest_insufficient_ratio >= 0.9:
        primary_issue = "latest_and_cumulative_strength_both_stuck_at_insufficient_data"
    elif promotable_ratio >= 0.25 and mismatch_rate >= 0.25:
        primary_issue = "strength_progress_exists_but_fails_to_stabilize_cumulatively"

    if weakest_horizon in VALID_HORIZONS:
        secondary_issue = f"horizon_{weakest_horizon}_shows_the_heaviest_strength_failure"

    return {
        "primary_issue": primary_issue,
        "secondary_issue": secondary_issue,
        "latest_insufficient_data_ratio": latest_insufficient_ratio,
        "cumulative_insufficient_data_ratio": cumulative_insufficient_ratio,
        "grouping_free_latest_insufficient_ratio": grouping_free_latest_insufficient_ratio,
        "grouping_free_cumulative_insufficient_ratio": grouping_free_cumulative_insufficient_ratio,
        "all_insufficient_pair_ratio": all_insufficient_pair_ratio,
        "mismatch_rate": mismatch_rate,
        "promotable_ratio": promotable_ratio,
        "weakest_horizon": weakest_horizon,
        "summary": (
            f"Primary issue={primary_issue}; secondary issue={secondary_issue}; "
            f"latest_insufficient={latest_insufficient_ratio:.2%}; "
            f"cumulative_insufficient={cumulative_insufficient_ratio:.2%}; "
            f"grouping_free_latest_insufficient={grouping_free_latest_insufficient_ratio:.2%}; "
            f"grouping_free_cumulative_insufficient={grouping_free_cumulative_insufficient_ratio:.2%}."
        ),
    }


def _build_horizon_example(record: dict[str, Any], horizon_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": _text(record.get("generated_at")) or "n/a",
        "cumulative_record_count": _int(record.get("cumulative_record_count")),
        "selection_status": _text(record.get("selection_status")) or "unknown",
        "horizon": horizon_row["horizon"],
        "latest_strength": horizon_row["latest_strength"],
        "cumulative_strength": horizon_row["cumulative_strength"],
        "blocker_reasons": horizon_row["blocker_reasons"],
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


def _md_dist(title: str, rows: Any) -> str:
    lines = [f"- {title}:"]
    if not isinstance(rows, list) or not rows:
        lines.append("  - none")
        return "\n".join(lines)

    for row in rows:
        if isinstance(row, dict):
            lines.append(
                f"  - {row.get('value', 'n/a')}: {row.get('count', 0)} ({_fmt(row.get('ratio'))})"
            )
    return "\n".join(lines)


def _md_nested_dist(value: Any) -> str:
    nested = value if isinstance(value, dict) else {}
    lines: list[str] = []
    if not nested:
        return "- none"

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
    json_path = output_dir / "strength_formation_failure_diagnosis_summary.json"
    markdown_path = output_dir / "strength_formation_failure_diagnosis_summary.md"
    return json_path, markdown_path


def main() -> None:
    result = run_strength_formation_failure_diagnosis_report()
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
                "total_horizon_observations": summary.get("metadata", {}).get(
                    "total_horizon_observations",
                    0,
                ),
                "primary_issue": summary.get("diagnosis", {}).get("primary_issue", "unknown"),
                "secondary_issue": summary.get("diagnosis", {}).get("secondary_issue", "unknown"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()