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

VISIBILITY_COLLAPSE_LABELS = {
    "insufficient_data",
    "single_horizon_only",
    "unstable",
}


def run_candidate_seed_failure_diagnosis_report(
    input_path: Path | None = None,
) -> dict[str, Any]:
    resolved = input_path or DEFAULT_INPUT_PATH
    loaded = load_shadow_records(resolved)
    summary = build_candidate_seed_failure_diagnosis_summary(
        records=loaded["records"],
        input_path=resolved,
        data_quality=loaded["data_quality"],
    )
    return {
        "input_path": str(resolved),
        "summary": summary,
        "markdown": render_candidate_seed_failure_diagnosis_markdown(summary),
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


def build_candidate_seed_failure_diagnosis_summary(
    records: list[dict[str, Any]],
    input_path: Path,
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    seed_count_dist: Counter[int] = Counter()
    with_seed: Counter[str] = Counter()
    without_seed: Counter[str] = Counter()

    blocker_freq: Counter[str] = Counter()
    grouping_breakdown: dict[str, Counter[str]] = defaultdict(Counter)

    strength_dist: Counter[str] = Counter()
    visible_count_dist: Counter[int] = Counter()
    stability_dist: Counter[str] = Counter()

    selection_status_dist: Counter[str] = Counter()
    abstain_cat_dist: Counter[str] = Counter()
    ranking_depth_dist: Counter[int] = Counter()

    seed_zero = 0
    all_insufficient = 0
    top_candidate_present = 0

    grouping_failure_runs = 0
    insufficient_strength_runs = 0
    weak_runs = 0
    confirmed_runs = 0

    no_symbol = 0
    no_strategy = 0

    examples: dict[str, list[dict[str, Any]]] = {
        "seed_count_zero_cases": [],
        "grouping_failure_cases": [],
        "strength_insufficient_cases": [],
        "visibility_collapse_cases": [],
        "seed_present_but_abstain_cases": [],
    }

    for record in records:
        row = _normalize_record(record)

        seed_count_dist[row["candidate_seed_count"]] += 1
        selection_status_dist[row["selection_status"]] += 1
        ranking_depth_dist[row["ranking_depth"]] += 1

        if row["candidate_seed_count"] == 0:
            seed_zero += 1

        if row["all_horizons_insufficient_data"]:
            all_insufficient += 1

        if row["top_candidate_present"]:
            top_candidate_present += 1

        for horizon in row["horizons_with_seed"]:
            with_seed[horizon] += 1

        for horizon in row["horizons_without_seed"]:
            without_seed[horizon] += 1

        for reason in row["blocker_reasons"]:
            blocker_freq[reason] += 1

        for horizon, reasons in row["grouping_breakdown"].items():
            for reason in reasons:
                grouping_breakdown[horizon][reason] += 1

        if row["has_grouping_failure"]:
            grouping_failure_runs += 1

        if "no_valid_symbol_group" in row["blocker_reasons"]:
            no_symbol += 1

        if "no_valid_strategy_group" in row["blocker_reasons"]:
            no_strategy += 1

        strength_dist[row["candidate_strength"]] += 1
        if row["candidate_strength"] == "insufficient_data":
            insufficient_strength_runs += 1
        elif row["candidate_strength"] == "weak":
            weak_runs += 1
        else:
            confirmed_runs += 1

        visible_count_dist[row["visible_horizons_count"]] += 1
        stability_dist[row["stability_label"]] += 1

        if row["selection_status"] == "abstain":
            abstain_cat_dist[row["abstain_category"]] += 1

        example = _build_example(row)

        if row["candidate_seed_count"] == 0:
            _push_example(examples["seed_count_zero_cases"], example)

        if row["has_grouping_failure"]:
            _push_example(examples["grouping_failure_cases"], example)

        if row["candidate_strength"] in {"insufficient_data", "weak"}:
            _push_example(examples["strength_insufficient_cases"], example)

        if row["stability_label"] in VISIBILITY_COLLAPSE_LABELS:
            _push_example(examples["visibility_collapse_cases"], example)

        if row["candidate_seed_count"] > 0 and row["selection_status"] == "abstain":
            _push_example(examples["seed_present_but_abstain_cases"], example)

    total = len(records)

    visibility_collapse_runs = sum(
        count
        for label, count in stability_dist.items()
        if label in VISIBILITY_COLLAPSE_LABELS
    )

    diagnosis = _build_diagnosis(
        total_records=total,
        seed_zero_runs=seed_zero,
        grouping_failure_runs=grouping_failure_runs,
        insufficient_strength_runs=insufficient_strength_runs,
        visibility_collapse_runs=visibility_collapse_runs,
        seed_present_runs=total - seed_zero,
        healthy_abstain_runs=(
            abstain_cat_dist.get("NO_ELIGIBLE_CANDIDATES", 0)
            + abstain_cat_dist.get("TOP_CANDIDATES_TIED", 0)
        ),
        no_candidate_abstain_runs=abstain_cat_dist.get("NO_CANDIDATES_AVAILABLE", 0),
    )

    return {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "report_type": "candidate_seed_failure_diagnosis_report",
            "input_path": str(input_path),
            "total_records": total,
            "data_quality": data_quality,
        },
        "seed_layer": {
            "candidate_seed_count_distribution": _counter_rows(seed_count_dist),
            "seed_zero_ratio": _safe_ratio(seed_zero, total),
            "horizon_seed_coverage": {
                "with_seed": _horizon_rows(with_seed, total),
                "without_seed": _horizon_rows(without_seed, total),
            },
            "all_horizons_insufficient_data_ratio": _safe_ratio(all_insufficient, total),
        },
        "grouping_layer": {
            "blocker_reason_frequency": _counter_rows(blocker_freq),
            "no_valid_symbol_group_frequency": {
                "count": no_symbol,
                "ratio": _safe_ratio(no_symbol, total),
            },
            "no_valid_strategy_group_frequency": {
                "count": no_strategy,
                "ratio": _safe_ratio(no_strategy, total),
            },
            "grouping_failure_ratio": _safe_ratio(grouping_failure_runs, total),
            "grouping_related_blocker_breakdown": {
                horizon: _counter_rows(counter)
                for horizon, counter in sorted(grouping_breakdown.items())
            },
        },
        "strength_layer": {
            "candidate_strength_distribution": _counter_rows(strength_dist),
            "insufficient_data_ratio": _safe_ratio(insufficient_strength_runs, total),
            "weak_vs_confirmed_ratio": {
                "weak": _safe_ratio(weak_runs, total),
                "confirmed": _safe_ratio(confirmed_runs, total),
            },
        },
        "visibility_layer": {
            "visible_horizons_count_distribution": _counter_rows(visible_count_dist),
            "stability_label_distribution": _counter_rows(stability_dist),
        },
        "selection_layer": {
            "selection_status_distribution": _counter_rows(selection_status_dist),
            "abstain_category_ratio": {
                "NO_CANDIDATES_AVAILABLE": _safe_ratio(
                    abstain_cat_dist.get("NO_CANDIDATES_AVAILABLE", 0),
                    total,
                ),
                "NO_ELIGIBLE_CANDIDATES": _safe_ratio(
                    abstain_cat_dist.get("NO_ELIGIBLE_CANDIDATES", 0),
                    total,
                ),
                "TOP_CANDIDATES_TIED": _safe_ratio(
                    abstain_cat_dist.get("TOP_CANDIDATES_TIED", 0),
                    total,
                ),
            },
            "ranking_depth_distribution": _counter_rows(ranking_depth_dist),
            "top_candidate_presence_rate": _safe_ratio(top_candidate_present, total),
        },
        "diagnosis": diagnosis,
        "examples": examples,
    }


def render_candidate_seed_failure_diagnosis_markdown(summary: dict[str, Any]) -> str:
    metadata = summary.get("metadata", {})
    seed = summary.get("seed_layer", {})
    grouping = summary.get("grouping_layer", {})
    strength = summary.get("strength_layer", {})
    visibility = summary.get("visibility_layer", {})
    selection = summary.get("selection_layer", {})
    diagnosis = summary.get("diagnosis", {})

    lines = [
        "# Candidate Seed Failure Diagnosis Report",
        "",
        "## Executive Summary",
        f"- Total records: {metadata.get('total_records', 0)}",
        f"- Seed zero ratio: {_fmt(seed.get('seed_zero_ratio'))}",
        f"- Primary bottleneck: {diagnosis.get('primary_bottleneck', 'unknown')}",
        f"- Secondary bottleneck: {diagnosis.get('secondary_bottleneck', 'unknown')}",
        f"- Assessment: {diagnosis.get('starvation_vs_healthy_abstain_assessment', 'unknown')}",
        "",
        "## Seed Layer",
        _md_dist(seed.get("candidate_seed_count_distribution")),
        "",
        "## Grouping Layer",
        _md_dist(grouping.get("blocker_reason_frequency")),
        "",
        "## Strength Layer",
        _md_dist(strength.get("candidate_strength_distribution")),
        "",
        "## Visibility Layer",
        _md_dist(visibility.get("stability_label_distribution")),
        "",
        "## Selection Layer",
        _md_dist(selection.get("selection_status_distribution")),
        "",
        "## Final Diagnosis",
        f"- {diagnosis.get('summary', 'n/a')}",
    ]

    return "\n".join(lines)


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    selection_status = _text(record.get("selection_status")) or "unknown"

    ranking = record.get("ranking") if isinstance(record.get("ranking"), list) else []
    ranking_items = [item for item in ranking if isinstance(item, dict)]

    abstain = record.get("abstain_diagnosis")
    abstain = abstain if isinstance(abstain, dict) else {}

    seed_diag = _seed_diag(record, abstain)
    top_candidate = _top_candidate(record, abstain, ranking_items)

    blocker_reasons = _blocker_reasons(seed_diag)
    grouping_breakdown = _grouping_breakdown(seed_diag)
    candidate_strength = _strength(seed_diag, top_candidate)
    visible_horizons = _visible_horizons(seed_diag, top_candidate)
    stability_label = _stability_label(seed_diag, top_candidate, visible_horizons)

    return {
        "generated_at": _text(record.get("generated_at")) or "n/a",
        "cumulative_record_count": _int(record.get("cumulative_record_count")),
        "selection_status": selection_status,
        "candidate_seed_count": _int(record.get("candidate_seed_count")),
        "ranking_depth": len(ranking_items),
        "top_candidate_present": bool(top_candidate),
        "abstain_category": _abstain_category(abstain.get("category")),
        "all_horizons_insufficient_data": (
            seed_diag.get("all_horizons_insufficient_data") is True
            or record.get("all_horizons_insufficient_data") is True
        ),
        "horizons_with_seed": _horizons(
            _first(seed_diag.get("horizons_with_seed"), record.get("horizons_with_seed"))
        ),
        "horizons_without_seed": _horizons(
            _first(seed_diag.get("horizons_without_seed"), record.get("horizons_without_seed"))
        ),
        "blocker_reasons": blocker_reasons,
        "grouping_breakdown": grouping_breakdown,
        "candidate_strength": candidate_strength,
        "visible_horizons_count": len(visible_horizons),
        "stability_label": stability_label,
        "has_grouping_failure": any(
            reason in GROUPING_REASONS for reason in blocker_reasons
        ),
    }


def _seed_diag(record: dict[str, Any], abstain: dict[str, Any]) -> dict[str, Any]:
    for payload in (record, abstain):
        value = payload.get("candidate_seed_diagnostics")
        if isinstance(value, dict):
            return value
    return {}


def _top_candidate(
    record: dict[str, Any],
    abstain: dict[str, Any],
    ranking_items: list[dict[str, Any]],
) -> dict[str, Any]:
    value = abstain.get("top_candidate")
    if isinstance(value, dict) and value:
        return value

    if ranking_items:
        return ranking_items[0]

    if any(
        _text(record.get(key))
        for key in ("selected_symbol", "selected_strategy", "selected_horizon")
    ):
        return {
            "selected_candidate_strength": _text(record.get("selected_candidate_strength")),
            "selected_stability_label": _text(record.get("selected_stability_label")),
            "selected_visible_horizons": record.get("selected_visible_horizons"),
        }

    return {}


def _blocker_reasons(seed_diag: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    horizon_diagnostics = seed_diag.get("horizon_diagnostics")
    if not isinstance(horizon_diagnostics, list):
        return reasons

    for item in horizon_diagnostics:
        if not isinstance(item, dict):
            continue
        raw_reasons = item.get("blocker_reasons")
        if not isinstance(raw_reasons, list):
            continue
        for reason in raw_reasons:
            text = _text(reason)
            if text:
                reasons.append(text)

    return reasons


def _grouping_breakdown(seed_diag: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    horizon_diagnostics = seed_diag.get("horizon_diagnostics")
    if not isinstance(horizon_diagnostics, list):
        return result

    for item in horizon_diagnostics:
        if not isinstance(item, dict):
            continue

        horizon = _text(item.get("horizon"))
        if not horizon:
            continue

        raw_reasons = item.get("blocker_reasons")
        if not isinstance(raw_reasons, list):
            continue

        reasons = [
            reason
            for reason in (_text(value) for value in raw_reasons)
            if reason in GROUPING_REASONS
        ]
        if reasons:
            result[horizon] = reasons

    return result


def _strength(seed_diag: dict[str, Any], top_candidate: dict[str, Any]) -> str:
    values: list[str] = []

    horizon_diagnostics = seed_diag.get("horizon_diagnostics")
    if isinstance(horizon_diagnostics, list):
        for item in horizon_diagnostics:
            if not isinstance(item, dict):
                continue
            for key in ("latest_candidate_strength", "cumulative_candidate_strength"):
                value = _text(item.get(key))
                if value:
                    values.append(value)

    if isinstance(top_candidate, dict):
        for key in ("selected_candidate_strength", "candidate_strength"):
            value = _text(top_candidate.get(key))
            if value:
                values.append(value)

    if not values:
        return "insufficient_data"

    priority = {"strong": 4, "moderate": 3, "weak": 2, "insufficient_data": 1}
    worst = "strong"

    for value in values:
        if priority.get(value, 0) < priority.get(worst, 0):
            worst = value

    return worst


def _visible_horizons(seed_diag: dict[str, Any], top_candidate: dict[str, Any]) -> list[str]:
    visible: list[str] = []

    horizon_diagnostics = seed_diag.get("horizon_diagnostics")
    if isinstance(horizon_diagnostics, list):
        for item in horizon_diagnostics:
            if not isinstance(item, dict):
                continue

            horizon = _text(item.get("horizon"))
            if not horizon:
                continue

            strength = _text(item.get("latest_candidate_strength"))
            if strength and strength != "insufficient_data" and horizon not in visible:
                visible.append(horizon)

    if visible:
        return visible

    if isinstance(top_candidate, dict):
        return _horizons(
            _first(
                top_candidate.get("selected_visible_horizons"),
                top_candidate.get("visible_horizons"),
            )
        )

    return []


def _stability_label(
    seed_diag: dict[str, Any],
    top_candidate: dict[str, Any],
    visible_horizons: list[str],
) -> str:
    if isinstance(top_candidate, dict):
        label = _text(top_candidate.get("selected_stability_label")) or _text(
            top_candidate.get("stability_label")
        )
        if label:
            return label

    if seed_diag.get("all_horizons_insufficient_data") is True:
        return "insufficient_data"

    count = len(visible_horizons)
    if count == 0:
        return "insufficient_data"
    if count == 1:
        return "single_horizon_only"
    if count >= 2:
        return "multi_horizon_confirmed"

    return "insufficient_data"


def _abstain_category(value: Any) -> str:
    text = (_text(value) or "n/a").replace("-", "_").replace(" ", "_").upper()
    return {"TIED_TOP_CANDIDATES": "TOP_CANDIDATES_TIED"}.get(text, text)


def _build_diagnosis(
    *,
    total_records: int,
    seed_zero_runs: int,
    grouping_failure_runs: int,
    insufficient_strength_runs: int,
    visibility_collapse_runs: int,
    seed_present_runs: int,
    healthy_abstain_runs: int,
    no_candidate_abstain_runs: int,
) -> dict[str, Any]:
    if total_records == 0:
        return {
            "primary_bottleneck": "unknown",
            "secondary_bottleneck": "unknown",
            "starvation_vs_healthy_abstain_assessment": "no_data",
            "summary": "No data available.",
            "bottleneck_scores": {},
        }

    seed_score = _safe_ratio(seed_zero_runs, total_records)
    grouping_score = _safe_ratio(grouping_failure_runs, total_records)
    strength_score = _safe_ratio(insufficient_strength_runs, total_records)
    visibility_score = _safe_ratio(visibility_collapse_runs, total_records)
    selection_score = min(
        _safe_ratio(seed_present_runs, total_records),
        _safe_ratio(healthy_abstain_runs, total_records),
    )

    weighted_scores = [
        ("seed_generation_failure", round(seed_score * 1.5, 4)),
        ("grouping_failure", round(grouping_score * 1.2, 4)),
        ("strength_insufficient", round(strength_score, 4)),
        ("visibility_collapse", round(visibility_score, 4)),
        ("selection_layer_issue", round(selection_score, 4)),
    ]
    weighted_scores.sort(key=lambda item: (-item[1], item[0]))

    primary = weighted_scores[0][0]
    secondary = weighted_scores[1][0] if len(weighted_scores) > 1 else primary

    if _safe_ratio(seed_zero_runs + no_candidate_abstain_runs, total_records) >= 0.5:
        assessment = "structural_starvation"
    elif (
        _safe_ratio(seed_present_runs, total_records) >= 0.5
        and _safe_ratio(healthy_abstain_runs, total_records) >= 0.25
    ):
        assessment = "healthy_abstain_bias"
    else:
        assessment = "mixed_or_unclear"

    return {
        "primary_bottleneck": primary,
        "secondary_bottleneck": secondary,
        "starvation_vs_healthy_abstain_assessment": assessment,
        "summary": (
            f"Primary bottleneck appears to be {primary}; "
            f"secondary pressure appears at {secondary}. "
            f"Assessment={assessment}."
        ),
        "bottleneck_scores": {name: score for name, score in weighted_scores},
    }


def _build_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": row["generated_at"],
        "cumulative_record_count": row["cumulative_record_count"],
        "selection_status": row["selection_status"],
        "seed_count": row["candidate_seed_count"],
        "blocker_reasons": row["blocker_reasons"],
        "horizons_with_seed": row["horizons_with_seed"],
        "horizons_without_seed": row["horizons_without_seed"],
    }


def _push_example(target: list[dict[str, Any]], example: dict[str, Any]) -> None:
    target.append(example)
    target.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
    del target[MAX_EXAMPLES_PER_CATEGORY:]


def _counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {"value": value, "count": count, "ratio": _safe_ratio(count, total)}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    ]


def _horizon_rows(counter: Counter[str], total_records: int) -> list[dict[str, Any]]:
    return [
        {
            "value": horizon,
            "count": counter.get(horizon, 0),
            "ratio": _safe_ratio(counter.get(horizon, 0), total_records),
        }
        for horizon in VALID_HORIZONS
    ]


def _md_dist(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "- none"

    lines: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            lines.append(
                f"- {row.get('value', 'n/a')}: "
                f"{row.get('count', 0)} ({_fmt(row.get('ratio'))})"
            )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "0.00%"


def _horizons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        text = _text(item)
        if text and text in VALID_HORIZONS and text not in result:
            result.append(text)
    return result


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


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None

def _default_output_paths(input_path: Path) -> tuple[Path, Path]:
    output_dir = input_path.parent
    json_path = output_dir / "candidate_seed_failure_diagnosis_summary.json"
    markdown_path = output_dir / "candidate_seed_failure_diagnosis_summary.md"
    return json_path, markdown_path


def main() -> None:
    result = run_candidate_seed_failure_diagnosis_report()
    summary = result["summary"]
    markdown = result["markdown"]

    input_path = Path(result["input_path"])
    json_path, markdown_path = _default_output_paths(input_path)

    # ✅ 핵심 수정
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path.write_text(markdown, encoding="utf-8")

    print(json.dumps({
        "input_path": str(input_path),
        "json_summary_path": str(json_path),
        "markdown_summary_path": str(markdown_path),
        "total_records": summary.get("metadata", {}).get("total_records", 0),
        "primary_bottleneck": summary.get("diagnosis", {}).get("primary_bottleneck", "unknown"),
        "secondary_bottleneck": summary.get("diagnosis", {}).get("secondary_bottleneck", "unknown"),
        "assessment": summary.get("diagnosis", {}).get("starvation_vs_healthy_abstain_assessment", "unknown"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()